# SOPM Interface

A Streamlit application that uses **Self-Organizing Maps (SOM)** to organize a prompt dataset and recommend similar/contrasting examples for a given user prompt, then leverages a local **Ollama** LLM to produce an improved version of that prompt.

## How it works

1. The app trains (or loads from cache) a SOM over sentence embeddings of a prompt dataset.
2. When you submit a prompt, it is embedded and mapped to the SOM grid (Best Matching Unit — BMU).
3. Prompts in the neighborhood of the BMU are retrieved as **similar examples**; prompts far away are retrieved as **contrasting examples**.
4. Both sets are forwarded to an Ollama LLM together with the original prompt, which returns an improved version.

```
User prompt → Embedding → SOM → Similar / Contrasting prompts → LLM → Improved prompt
```

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| [Ollama](https://ollama.com/) | any recent release |

Pull the LLM model you intend to use before starting:

```bash
ollama pull qwen3
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/samhenrique/sopm-interface.git
cd sopm-interface

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running

### PowerShell (Windows)

```powershell
.\run_sopmInterface.ps1
```

### Any platform

```bash
streamlit run sopmInterface.py
```

The app opens at `http://localhost:8501`.

## Configuration

All parameters are available in the **sidebar**:

| Parameter | Description | Default |
|---|---|---|
| Dataset CSV | Path to the prompt dataset | `ag_news_prompts.csv` |
| Artifacts dir | Where SOM and embeddings cache are stored | `artifacts/` |
| Embedding model | Sentence-Transformers model name | `all-MiniLM-L6-v2` |
| Ollama base_url | Ollama API endpoint | `http://localhost:11434/v1` |
| LLM model | Model name in Ollama | `qwen3` |
| k similar | Number of similar prompts to retrieve | `5` |
| k different | Number of contrasting prompts to retrieve | `3` |
| similar radius | Grid radius for similar prompts | `3` |
| different radius | Manhattan distance threshold for contrasting prompts | `5` |
| som_x / som_y | SOM grid dimensions | `5 × 10` |
| iterations | SOM training iterations | `10 000` |
| sigma / learning_rate | SOM training hyperparameters | `2.0 / 0.1` |

## Dataset format

The CSV must contain the following columns:

| Column | Description |
|---|---|
| `id` | Unique integer identifier |
| `prompt_name` | Prompt category / type label |
| `input` | The prompt text |
| `target` | Expected LLM response |

## Project structure

```
sopm-interface/
├── sopmInterface.py       # Main Streamlit application
├── embeddings_db.py       # SQLite-backed embedding cache
├── run_sopmInterface.ps1  # PowerShell launch script
├── ag_news_prompts.csv    # Default prompt dataset
├── requirements.txt       # Pinned Python dependencies
└── artifacts/             # Auto-created at runtime (SOM + embeddings cache)
```
