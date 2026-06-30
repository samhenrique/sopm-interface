$ErrorActionPreference = 'Stop'

# Ensure we run with the project's venv Python
$python = Join-Path $PSScriptRoot 'sopm\Scripts\python.exe'

& $python -m streamlit run (Join-Path $PSScriptRoot 'sopmInterface.py') --server.port 8501
