
from __future__ import annotations

import os
import pickle
import re
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple
import textwrap

# Workaround for some Streamlit/protobuf version combinations on Windows.
# If you prefer the faster C++ implementation, pin `protobuf<=3.20.*` instead.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from minisom import MiniSom
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from embeddings_db import EmbeddingRecord, EmbeddingStore, make_embedding_key

# Compatibility shims for NumPy 2.x with older libs (e.g., Streamlit 0.82)
# that still reference deprecated aliases like np.object.
with warnings.catch_warnings():
	warnings.simplefilter("ignore", FutureWarning)
	if not hasattr(np, "object"):
		np.object = object  # type: ignore[attr-defined]
	if not hasattr(np, "bool"):
		np.bool = bool  # type: ignore[attr-defined]
	if not hasattr(np, "int"):
		np.int = int  # type: ignore[attr-defined]
	if not hasattr(np, "float"):
		np.float = float  # type: ignore[attr-defined]
	if not hasattr(np, "str"):
		np.str = str  # type: ignore[attr-defined]


DATA_CSV_DEFAULT = "ag_news_prompts.csv"
ARTIFACT_DIR_DEFAULT = "artifacts"
EMBEDDINGS_DB_FILENAME = "embeddings.sqlite"


@dataclass(frozen=True)
class SomParams:
	som_x: int = 5
	som_y: int = 10
	learning_rate: float = 0.1
	topology: str = "rectangular"
	num_iterations: int = 10_000
	sigma: float = 2.0
	random_seed: int = 42
	neighborhood_function: str = "gaussian"


@dataclass
class SomBundle:
	df: pd.DataFrame
	embs: np.ndarray
	som: MiniSom
	params: SomParams
	embed_model_name: str


def _safe_mkdir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(x, axis=1, keepdims=True)
	norms = np.where(norms == 0, 1.0, norms)
	return x / norms


@lru_cache(maxsize=4)
def _load_embedding_model(model_name: str) -> SentenceTransformer:
	return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _load_ollama_client(base_url: str) -> OpenAI:
	return OpenAI(api_key="ollama", base_url=base_url)


@lru_cache(maxsize=4)
def _load_df(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	required = {"id", "prompt_name", "input", "target"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"CSV missing required columns: {sorted(missing)}")
	return df


def _encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
	embs = model.encode(texts, batch_size=32, show_progress_bar=False)
	embs = np.asarray(embs, dtype=np.float32)
	return _normalize_rows(embs)


def _embeddings_db_path(artifact_dir: str) -> str:
	return os.path.join(artifact_dir, EMBEDDINGS_DB_FILENAME)


def _encode_texts_cached(
	*,
	model: SentenceTransformer,
	model_name: str,
	texts: list[str],
	db_path: str,
	dataset_tag: Optional[str] = None,
	prompt_ids: Optional[list[int]] = None,
) -> np.ndarray:
	"""Encode texts, reusing/storing embeddings in a SQLite DB."""
	store = EmbeddingStore(db_path)
	keys = [make_embedding_key(model_name, t) for t in texts]
	found = store.get_many(keys)

	missing_idx = [i for i, k in enumerate(keys) if k not in found]
	computed_by_key: dict[str, np.ndarray] = {}
	if missing_idx:
		missing_texts = [texts[i] for i in missing_idx]
		missing_embs = _encode_texts(model, missing_texts)
		records: list[EmbeddingRecord] = []
		for local_i, i in enumerate(missing_idx):
			key = keys[i]
			emb = np.asarray(missing_embs[local_i], dtype=np.float32)
			computed_by_key[key] = emb
			records.append(
				EmbeddingRecord(
					key=key,
					model_name=model_name,
					text=texts[i],
					dim=int(emb.shape[0]),
					normalized=True,
					embedding=emb,
					dataset_tag=dataset_tag,
					prompt_id=(prompt_ids[i] if prompt_ids is not None else None),
				)
			)
		store.upsert_many(records)

	out = []
	for k in keys:
		emb = found.get(k)
		if emb is None:
			emb = computed_by_key[k]
		out.append(np.asarray(emb, dtype=np.float32))

	return _normalize_rows(np.vstack(out))


def _backfill_embeddings_db_from_bundle(*, bundle: SomBundle, csv_path: str, db_path: str) -> None:
	"""Best-effort: populate the embeddings DB using an already loaded bundle.

	This avoids recomputing embeddings when a SOM artifact exists but the DB is empty.
	"""
	try:
		texts = bundle.df["input"].astype(str).tolist()
		if not texts:
			return
		store = EmbeddingStore(db_path)
		key0 = make_embedding_key(bundle.embed_model_name, texts[0])
		if key0 in store.get_many([key0]):
			return

		dataset_tag = os.path.basename(csv_path)
		prompt_ids = bundle.df["id"].astype(int).tolist() if "id" in bundle.df.columns else None
		records: list[EmbeddingRecord] = []
		for i, text in enumerate(texts):
			emb = np.asarray(bundle.embs[i], dtype=np.float32)
			records.append(
				EmbeddingRecord(
					key=make_embedding_key(bundle.embed_model_name, text),
					model_name=bundle.embed_model_name,
					text=text,
					dim=int(emb.shape[0]),
					normalized=True,
					embedding=emb,
					dataset_tag=dataset_tag,
					prompt_id=(prompt_ids[i] if prompt_ids is not None else None),
				)
			)
			if len(records) >= 500:
				store.upsert_many(records)
				records.clear()
		if records:
			store.upsert_many(records)
	except Exception:
		# Non-fatal: DB backfill is an optimization.
		return


def _som_artifact_path(artifact_dir: str, embed_model_name: str, params: SomParams) -> str:
	safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", embed_model_name)
	return os.path.join(
		artifact_dir,
		f"som_{safe_model}_{params.som_x}x{params.som_y}_it{params.num_iterations}_sig{params.sigma}_lr{params.learning_rate}.pkl",
	)


def _save_bundle(path: str, bundle: SomBundle) -> None:
	payload = {
		"embed_model_name": bundle.embed_model_name,
		"params": {
			"som_x": bundle.params.som_x,
			"som_y": bundle.params.som_y,
			"learning_rate": bundle.params.learning_rate,
			"topology": bundle.params.topology,
			"num_iterations": bundle.params.num_iterations,
			"sigma": bundle.params.sigma,
			"random_seed": bundle.params.random_seed,
			"neighborhood_function": bundle.params.neighborhood_function,
		},
		"df": bundle.df,
		"embs": bundle.embs,
		"som_weights": bundle.som.get_weights(),
	}
	with open(path, "wb") as f:
		pickle.dump(payload, f)


def _load_bundle(path: str) -> SomBundle:
	with open(path, "rb") as f:
		payload = pickle.load(f)

	raw_params = payload["params"]
	if isinstance(raw_params, dict):
		params = SomParams(**raw_params)
	else:
		# Backward-compatibility if an older artifact stored the dataclass.
		params = raw_params
	df: pd.DataFrame = payload["df"]
	embs: np.ndarray = payload["embs"]
	som_weights: np.ndarray = payload["som_weights"]
	embed_model_name: str = payload["embed_model_name"]

	som = MiniSom(
		params.som_x,
		params.som_y,
		int(embs.shape[1]),
		sigma=float(params.sigma),
		learning_rate=float(params.learning_rate),
		neighborhood_function=params.neighborhood_function,
		topology=params.topology,
		random_seed=int(params.random_seed),
	)
	som._weights = som_weights  # MiniSom doesn't expose a public setter

	return SomBundle(df=df, embs=embs, som=som, params=params, embed_model_name=embed_model_name)


def build_or_load_som_bundle(
	*,
	csv_path: str,
	artifact_dir: str,
	embed_model_name: str,
	params: SomParams,
) -> SomBundle:
	_safe_mkdir(artifact_dir)
	artifact_path = _som_artifact_path(artifact_dir, embed_model_name, params)
	db_path = _embeddings_db_path(artifact_dir)

	if os.path.exists(artifact_path):
		try:
			bundle = _load_bundle(artifact_path)
			_backfill_embeddings_db_from_bundle(bundle=bundle, csv_path=csv_path, db_path=db_path)
			return bundle
		except Exception:
			# If artifact is corrupted or incompatible with current code, rebuild.
			try:
				os.remove(artifact_path)
			except OSError:
				pass

	df = _load_df(csv_path).reset_index(drop=True)
	model = _load_embedding_model(embed_model_name)
	texts = df["input"].astype(str).tolist()
	prompt_ids = df["id"].astype(int).tolist()
	dataset_tag = os.path.basename(csv_path)
	embs = _encode_texts_cached(
		model=model,
		model_name=embed_model_name,
		texts=texts,
		db_path=db_path,
		dataset_tag=dataset_tag,
		prompt_ids=prompt_ids,
	)

	som = MiniSom(
		params.som_x,
		params.som_y,
		embs.shape[1],
		sigma=params.sigma,
		learning_rate=params.learning_rate,
		neighborhood_function=params.neighborhood_function,
		topology=params.topology,
		random_seed=params.random_seed,
	)
	som.random_weights_init(embs)
	som.train_random(embs, params.num_iterations)

	bmus = [som.winner(v) for v in embs]
	df["som_x"] = [b[0] for b in bmus]
	df["som_y"] = [b[1] for b in bmus]

	bundle = SomBundle(df=df, embs=embs, som=som, params=params, embed_model_name=embed_model_name)
	_save_bundle(artifact_path, bundle)
	return bundle


@lru_cache(maxsize=2)
def _get_bundle_cached(csv_path: str, artifact_dir: str, embed_model_name: str, params: SomParams) -> SomBundle:
	return build_or_load_som_bundle(
		csv_path=csv_path,
		artifact_dir=artifact_dir,
		embed_model_name=embed_model_name,
		params=params,
	)


def _recommend_for_embedding(
	*,
	prompt_emb: np.ndarray,
	bundle: SomBundle,
	k_pos: int,
	k_neg: int,
	radius_pos: int,
	radius_neg: int,
) -> tuple[Tuple[int, int], pd.DataFrame, pd.DataFrame]:
	"""Return (bmu, positives_df, negatives_df)."""
	df = bundle.df
	embs = bundle.embs
	som = bundle.som
	params = bundle.params

	bmu_x, bmu_y = som.winner(prompt_emb)

	x0 = max(0, bmu_x - radius_pos)
	x1 = min(params.som_x - 1, bmu_x + radius_pos)
	y0 = max(0, bmu_y - radius_pos)
	y1 = min(params.som_y - 1, bmu_y + radius_pos)
	mask_pos = (df["som_x"].between(x0, x1)) & (df["som_y"].between(y0, y1))
	pos_idx = np.flatnonzero(mask_pos.values)

	positives = pd.DataFrame()
	if len(pos_idx) > 0 and k_pos > 0:
		sims = np.dot(embs[pos_idx], prompt_emb)
		topk_local = np.argsort(sims)[-min(k_pos, len(pos_idx)) :][::-1]
		topk_idx = pos_idx[topk_local]
		positives = df.iloc[topk_idx].copy()
		positives["similarity"] = sims[topk_local]
		positives = positives[["id", "prompt_name", "input", "target", "similarity", "som_x", "som_y"]]

	manhattan = np.abs(df["som_x"].values - bmu_x) + np.abs(df["som_y"].values - bmu_y)
	neg_idx = np.flatnonzero(manhattan >= radius_neg)

	negatives = pd.DataFrame()
	if len(neg_idx) > 0 and k_neg > 0:
		sims = np.dot(embs[neg_idx], prompt_emb)
		bottomk_local = np.argsort(sims)[: min(k_neg, len(neg_idx))]
		bottomk_idx = neg_idx[bottomk_local]
		negatives = df.iloc[bottomk_idx].copy()
		negatives["similarity"] = sims[bottomk_local]
		negatives = negatives[["id", "prompt_name", "input", "target", "similarity", "som_x", "som_y"]]

	return (int(bmu_x), int(bmu_y)), positives, negatives


def _strip_think_blocks(text: str) -> str:
	text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
	return text.strip()


def _normalize_for_compare(text: str) -> str:
	"""Normalize text for 'did it change?' comparisons."""
	return re.sub(r"\s+", " ", (text or "").strip())


def _create_personalized_prompt(
	*,
	client: OpenAI,
	model_name: str,
	original_prompt: str,
	positives: pd.DataFrame,
	negatives: Optional[pd.DataFrame] = None,
) -> str:
	similar_prompts = ""
	if positives is not None and not positives.empty:
		similar_prompts += "\n--- SIMILAR PROMPTS (good examples to draw inspiration from) ---\n"
		for i, (_, row) in enumerate(positives.iterrows(), 1):
			similar_prompts += f"\nExample {i}:\n"
			similar_prompts += f"Input: {str(row['input'])[:300]}...\n"
			similar_prompts += f"Expected response: {row['target']}\n"

	contrasting_prompts = ""
	if negatives is not None and not negatives.empty:
		contrasting_prompts += "\n--- CONTRASTING PROMPTS (different examples to avoid confusion) ---\n"
		for i, (_, row) in enumerate(negatives.iterrows(), 1):
			contrasting_prompts += f"\nExample {i}:\n"
			contrasting_prompts += f"Input: {str(row['input'])[:300]}...\n"
			contrasting_prompts += f"Expected response: {row['target']}\n"

	meta_prompt = f"""You are a prompt engineering specialist. Your goal is to improve the original prompt by combining it with insights from the similar examples provided.

=== ORIGINAL PROMPT ===
{original_prompt}

=== REFERENCE EXAMPLES ===
{similar_prompts}
{contrasting_prompts}

=== TASK ===
Create an IMPROVED version of the original prompt that:
1. Maintains the same task/objective as the original
2. Incorporates successful patterns from similar examples
3. Is clearer and more specific
4. Uses consistent formatting

Return ONLY the improved prompt, without additional explanations."""

	def _call(meta: str, *, temperature: float) -> str:
		response = client.chat.completions.create(
			model=model_name,
			messages=[
				{
					"role": "system",
					"content": "You are an expert prompt engineer. Output only the improved prompt text.",
				},
				{"role": "user", "content": meta},
			],
			max_tokens=2000,
			temperature=float(temperature),
		)
		raw = response.choices[0].message.content or ""
		return _strip_think_blocks(raw)

	# 1) First attempt: standard improvement.
	content = _call(meta_prompt, temperature=0.5)

	# 2) Retry if model returned empty after stripping (common when it outputs only <think>).
	if not content:
		retry_prompt = (
			meta_prompt
			+ "\n\nIMPORTANT: Do not include <think> tags. Do not output an empty response. Output the improved prompt text only."
		)
		content = _call(retry_prompt, temperature=0.6)

	# 3) Retry if unchanged (force a rewrite with a structured template).
	if _normalize_for_compare(content) == _normalize_for_compare(original_prompt):
		force_rewrite = (
			meta_prompt
			+ "\n\nIMPORTANT: Rewrite the prompt so it is measurably different in wording and structure (do NOT return the same text), "
			  "while keeping the exact same objective. Use a clear template with short sections like: Context, Task, Output format, Constraints."
		)
		content = _call(force_rewrite, temperature=0.7)

	# Final fallback: if still empty, keep the original.
	if not content:
		return original_prompt
	return content


def _call_llm_chat(*, client: OpenAI, model_name: str, user_content: str, max_tokens: int = 800) -> str:
	user_content = user_content.strip()
	if not user_content:
		return ""

	response = client.chat.completions.create(
		model=model_name,
		messages=[{"role": "user", "content": user_content}],
		max_tokens=int(max_tokens),
		temperature=0.0,
	)
	content = response.choices[0].message.content or ""
	return _strip_think_blocks(content)



def _wrap_hover(text: str, width: int = 55) -> str:
	"""Wrap long text into multiple HTML lines for Plotly hover."""
	return "<br>".join(textwrap.wrap(str(text), width=width))


def _build_som_figure(
	*,
	bundle: SomBundle,
	user_bmu: Optional[Tuple[int, int]] = None,
	positives: Optional[pd.DataFrame] = None,
	negatives: Optional[pd.DataFrame] = None,
	seed: int = 0,
) -> go.Figure:
	df = bundle.df
	params = bundle.params

	u = bundle.som.distance_map().T

	# Explicit coordinates for cell centers (helps Plotly + older Streamlit render consistently)
	heat_x = np.arange(params.som_x) + 0.5
	heat_y = np.arange(params.som_y) + 0.5

	rng = np.random.default_rng(seed)
	jitter_x = rng.uniform(-0.15, 0.15, len(df))
	jitter_y = rng.uniform(-0.15, 0.15, len(df))
	x = df["som_x"].to_numpy(dtype=float) + 0.5 + jitter_x
	y = df["som_y"].to_numpy(dtype=float) + 0.5 + jitter_y

	hover = (
		"<b>id=</b>" + df["id"].astype(str)
		+ "  <b>type=</b>" + df["prompt_name"].astype(str)
		+ "  <b>som=</b>(" + df["som_x"].astype(str) + "," + df["som_y"].astype(str) + ")"
		+ "<br><b>input:</b><br>" + df["input"].astype(str).str.slice(0, 200).apply(lambda t: _wrap_hover(t, 55))
		+ "<br><b>target=</b>" + df["target"].astype(str)
	)

	fig = go.Figure()
	fig.add_trace(
		go.Heatmap(
			z=u,
			x=heat_x,
			y=heat_y,
			colorscale="rdbu",
			opacity=0.95,
			showscale=True,
			colorbar={"title": "U-Matrix"},
		)
	)
	fig.add_trace(
		go.Scatter(
			x=x,
			y=y,
			mode="markers",
			marker={
				"size": 9,
				"opacity": 0.85,
				# Use a neutral color to avoid confusion with the RdBu heatmap (red/blue).
				"color": "rgba(200, 200, 200, 0.90)",
				"line": {"width": 1, "color": "rgba(0,0,0,0.6)"},
			},
			hovertext=hover,
			hoverinfo="text",
			name="Prompts",
		)
	)

	if positives is not None and not positives.empty:
		fig.add_trace(
			go.Scatter(
				x=positives["som_x"].to_numpy(dtype=float) + 0.5,
				y=positives["som_y"].to_numpy(dtype=float) + 0.5,
				mode="markers",
				# Green is distinct from RdBu and reads as "positive".
				# Note: for *-open symbols Plotly typically uses `marker.color` for the outline.
				marker={
					"size": 14,
					"symbol": "circle-open",
					"color": "rgba(0, 200, 0, 1.0)",
					"line": {"width": 3, "color": "rgba(0, 200, 0, 1.0)"},
				},
				name="Similar",
				hovertext=(
					"similarity=" + positives["similarity"].round(3).astype(str)
					+ "<br>id=" + positives["id"].astype(str)
					+ "<br>type=" + positives["prompt_name"].astype(str)
				),
				hoverinfo="text",
			)
		)

	if negatives is not None and not negatives.empty:
		fig.add_trace(
			go.Scatter(
				x=negatives["som_x"].to_numpy(dtype=float) + 0.5,
				y=negatives["som_y"].to_numpy(dtype=float) + 0.5,
				mode="markers",
				# Purple is distinct from RdBu and avoids the heatmap's red/blue hues.
				# Note: for *-open symbols Plotly typically uses `marker.color` for the outline.
				marker={
					"size": 14,
					"symbol": "diamond-open",
					"color": "rgba(160, 0, 255, 1.0)",
					"line": {"width": 3, "color": "rgba(160, 0, 255, 1.0)"},
				},
				name="Different",
				hovertext=(
					"similarity=" + negatives["similarity"].round(3).astype(str)
					+ "<br>id=" + negatives["id"].astype(str)
					+ "<br>type=" + negatives["prompt_name"].astype(str)
				),
				hoverinfo="text",
			)
		)

	if user_bmu is not None:
		fig.add_trace(
			go.Scatter(
				x=[user_bmu[0] + 0.5],
				y=[user_bmu[1] + 0.5],
				mode="markers",
				# Gold stands out without overlapping the heatmap's red/blue palette.
				marker={"size": 18, "symbol": "x", "color": "rgba(255, 215, 0, 1.0)"},
				name="Your BMU",
				hovertext=f"BMU=({user_bmu[0]},{user_bmu[1]})",
				hoverinfo="text",
			)
		)

	fig.update_layout(
		height=620,
		margin={"l": 10, "r": 10, "t": 30, "b": 10},
		legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},

		xaxis={
			"title": "SOM X",
			"range": [0, params.som_x],
			"tickmode": "linear",
			"tick0": 0,
			"dtick": 1,
			"constrain": "domain",
		},
		yaxis={
			"title": "SOM Y",
			"range": [params.som_y, 0],
			"tickmode": "linear",
			"tick0": 0,
			"dtick": 1,
			"scaleanchor": "x",
		},
	)
	return fig


def main() -> None:
	st.set_page_config(page_title="SOPM Interface", layout="wide")
	st.title("Self-Organizing Prompt Maps for Lightweight Prompt Adaptation")

	# Session state wiring: keeps text areas/buttons connected across reruns.
	st.session_state.setdefault("user_prompt_text", "")
	st.session_state.setdefault("improved_prompt_text", "")
	st.session_state.setdefault("llm_test_prompt_text", "")
	st.session_state.setdefault("llm_response_text", "")

	with st.sidebar:
		st.header("Config")
		csv_path = st.text_input("Dataset CSV", value=DATA_CSV_DEFAULT)
		artifact_dir = st.text_input("Artifacts dir", value=ARTIFACT_DIR_DEFAULT)
		embed_model_name = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
		ollama_base_url = st.text_input("Ollama base_url", value="http://localhost:11434/v1")
		llm_model_name = st.text_input("LLM model (Ollama)", value="qwen3")

		st.subheader("Recommendations")
		k_pos = st.number_input("k similar", min_value=0, max_value=20, value=5, step=1)
		k_neg = st.number_input("k different", min_value=0, max_value=20, value=3, step=1)
		radius_pos = st.number_input("similar radius", min_value=0, max_value=10, value=3, step=1)
		radius_neg = st.number_input("different radius (Manhattan)", min_value=0, max_value=30, value=5, step=1)

		st.subheader("SOM")
		som_x = st.number_input("som_x", min_value=2, max_value=30, value=5, step=1)
		som_y = st.number_input("som_y", min_value=2, max_value=30, value=10, step=1)
		num_iterations = st.number_input("iterations", min_value=100, max_value=200_000, value=10_000, step=100)
		sigma = st.number_input("sigma", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
		learning_rate = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
		random_seed = st.number_input("random_seed", min_value=0, max_value=10_000, value=42, step=1)

		params = SomParams(
			som_x=int(som_x),
			som_y=int(som_y),
			num_iterations=int(num_iterations),
			sigma=float(sigma),
			learning_rate=float(learning_rate),
			random_seed=int(random_seed),
		)

	col_left, col_right = st.columns([1.05, 1])

	with col_left:
		st.subheader("1) Your prompt")
		user_prompt = st.text_area("Paste or type your prompt here", key="user_prompt_text", height=220)
		run_button = st.button("Improve prompt")

		st.subheader("2) Result")

		last_error = ""
		improved_prompt = ""
		bmu = None
		pos_df = pd.DataFrame()
		neg_df = pd.DataFrame()

		if run_button:
			if not user_prompt.strip():
				last_error = "Empty prompt. Enter some text to continue."
			else:
				try:
					with st.spinner("Loading/training SOM…"):
						t0 = time.time()
						bundle = _get_bundle_cached(csv_path, artifact_dir, embed_model_name, params)
						st.write(f"Dataset: {len(bundle.df)} prompts")
						st.write(f"Time (SOM bundle): {time.time() - t0:.2f}s")

					with st.spinner("Generating embedding and recommendations…"):
						model = _load_embedding_model(embed_model_name)
						db_path = _embeddings_db_path(artifact_dir)
						prompt_emb = _encode_texts_cached(
							model=model,
							model_name=embed_model_name,
							texts=[user_prompt],
							db_path=db_path,
						)[0]
						bmu, pos_df, neg_df = _recommend_for_embedding(
							prompt_emb=prompt_emb,
							bundle=bundle,
							k_pos=int(k_pos),
							k_neg=int(k_neg),
							radius_pos=int(radius_pos),
							radius_neg=int(radius_neg),
						)

					with st.spinner("Calling LLM (Ollama) to improve the prompt…"):
						client = _load_ollama_client(ollama_base_url)
						improved_prompt = _create_personalized_prompt(
							client=client,
							model_name=llm_model_name,
							original_prompt=user_prompt,
							positives=pos_df,
							negatives=neg_df,
						)
						st.session_state["improved_prompt_text"] = improved_prompt

				except Exception as e:
					last_error = str(e)

		if last_error:
			st.error(last_error)

		st.text_area("Improved prompt", key="improved_prompt_text", height=260)

		st.subheader("3) Similar prompts (positives)")
		if isinstance(pos_df, pd.DataFrame) and not pos_df.empty:
			st.dataframe(pos_df)
		else:
			st.write("No similar prompts found (try increasing the radius).")

		st.subheader("4) Different prompts (negatives)")
		if isinstance(neg_df, pd.DataFrame) and not neg_df.empty:
			st.dataframe(neg_df)
		else:
			st.write("No different prompts found (try adjusting the radius).")

		st.subheader("5) Run prompt on the LLM (Ollama)")
		if not st.session_state.get("llm_test_prompt_text"):
			# Seed with improved prompt if available, else the original.
			seed_text = st.session_state.get("improved_prompt_text") or st.session_state.get("user_prompt_text")
			st.session_state["llm_test_prompt_text"] = seed_text
		test_prompt = st.text_area("Prompt to send to the LLM", key="llm_test_prompt_text", height=180)
		max_tokens = st.number_input("max_tokens", min_value=16, max_value=4096, value=800, step=16)
		run_llm = st.button("Run on Ollama")
		if run_llm:
			try:
				with st.spinner("Calling Ollama…"):
					client = _load_ollama_client(ollama_base_url)
					reply = _call_llm_chat(
						client=client,
						model_name=llm_model_name,
						user_content=test_prompt,
						max_tokens=int(max_tokens),
					)
					st.session_state["llm_response_text"] = reply
			except Exception as e:
				st.error(f"Error calling Ollama: {e}")

		st.text_area("LLM response", key="llm_response_text", height=180)
		if bmu is not None:
			st.caption(f"Your prompt's BMU on the SOM: ({bmu[0]}, {bmu[1]})")

	with col_right:
		st.markdown(
			"""
			<style>
			:root {
				--umatrix-card-bg: var(--secondary-background-color, var(--secondaryBackgroundColor, rgba(255, 255, 255, 0.98)));
				--umatrix-card-fg: var(--text-color, var(--textColor, rgba(49, 51, 63, 0.98)));
				--umatrix-card-border: rgba(49, 51, 63, 0.18);
				--umatrix-card-shadow: rgba(0, 0, 0, 0.10);
				--umatrix-icon-border: rgba(49, 51, 63, 0.25);
			}
			@media (prefers-color-scheme: dark) {
				:root {
					--umatrix-card-bg: var(--secondary-background-color, var(--secondaryBackgroundColor, rgba(38, 39, 48, 0.98)));
					--umatrix-card-fg: var(--text-color, var(--textColor, rgba(250, 250, 250, 0.98)));
					--umatrix-card-border: rgba(250, 250, 250, 0.14);
					--umatrix-card-shadow: rgba(0, 0, 0, 0.35);
					--umatrix-icon-border: rgba(250, 250, 250, 0.25);
				}
			}

			.umatrix-title-row { display: flex; align-items: center; gap: 0.5rem; margin: 0 0 0.25rem 0; }
			.umatrix-title-row h3 { margin: 0; }
			.umatrix-info { position: relative; display: inline-block; cursor: help; user-select: none; }
			.umatrix-info-icon { font-size: 0.95rem; padding: 0.1rem 0.35rem; border-radius: 999px; border: 1px solid var(--umatrix-icon-border); }
			.umatrix-card {
				visibility: hidden;
				opacity: 0;
				position: absolute;
				top: 1.55rem;
				left: -0.25rem;
				z-index: 9999;
				width: min(560px, 20vw);
				background: var(--umatrix-card-bg);
				color: var(--umatrix-card-fg);
				border: 1px solid var(--umatrix-card-border);
				border-radius: 0.6rem;
				padding: 0.75rem 0.9rem;
				box-shadow: 0 6px 18px var(--umatrix-card-shadow);
				transition: opacity 120ms ease-in-out;
			}
			.umatrix-info:hover .umatrix-card { visibility: visible; opacity: 1; }
			.umatrix-card p { margin: 0.35rem 0 0 0; }
			</style>

			<div class="umatrix-title-row">
			  <h3>SOM U-Matrix Map</h3>
			  <div class="umatrix-info" aria-label="Information about the U-Matrix">
				<span class="umatrix-info-icon">&#9432;</span>
				<div class="umatrix-card">
				  <div><b>What is the U-Matrix?</b></div>
				  <p>
					The U-Matrix (Unified Distance Matrix) shows the <b>average distance</b> between each SOM neuron's weights and its neighbors.
					It helps visualize the clustering structure learned by the map.
				  </p>
				  <p>
					In general: <b>higher values</b> (lighter regions) indicate <b>boundaries</b> between clusters, and <b>lower values</b> (darker regions)
					suggest <b>more homogeneous areas</b> where examples are more similar.
				  </p>
				</div>
			  </div>
			</div>
			""",
			unsafe_allow_html=True,
		)
		try:
			bundle = build_or_load_som_bundle(
				csv_path=csv_path,
				artifact_dir=artifact_dir,
				embed_model_name=embed_model_name,
				params=params,
			)
			fig = _build_som_figure(bundle=bundle, user_bmu=bmu, positives=pos_df, negatives=neg_df)
			st.plotly_chart(fig, use_container_width=True)

			st.subheader("Browse prompts on the map")
			counts = (
				bundle.df.groupby(["som_x", "som_y"], as_index=False)
				.size()
				.rename(columns={"size": "n_prompts"})
				.sort_values(["som_x", "som_y"], ascending=[True, True])
			)
			options = [f"({int(r.som_x)},{int(r.som_y)}) — {int(r.n_prompts)} prompts" for r in counts.itertuples(index=False)]
			option_to_node = {
				opt: (int(r.som_x), int(r.som_y))
				for opt, r in zip(options, counts.itertuples(index=False))
			}
			default_opt = options[0] if options else None
			if default_opt is not None:
				selected = st.selectbox("Select a neuron (x,y)", options=options, index=0)
				node = option_to_node[selected]
				subset = bundle.df[(bundle.df["som_x"] == node[0]) & (bundle.df["som_y"] == node[1])]
				st.dataframe(subset[["id", "prompt_name", "input", "target", "som_x", "som_y"]])
		except Exception as e:
			st.error(f"Error loading/plotting SOM: {e}")


if __name__ == "__main__":
	main()

