@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)

echo Installing/updating backend dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install -r "..\AI chatbot\requirements-assistant.txt"

if not defined AQUASCAN_ALLOW_EMBEDDING_DOWNLOAD set AQUASCAN_ALLOW_EMBEDDING_DOWNLOAD=0
if not defined AQUASCAN_ASSISTANT_THREADS set AQUASCAN_ASSISTANT_THREADS=6
if not defined AQUASCAN_ASSISTANT_MAX_TOKENS set AQUASCAN_ASSISTANT_MAX_TOKENS=256
if not defined AQUASCAN_ASSISTANT_TIMEOUT_SECONDS set AQUASCAN_ASSISTANT_TIMEOUT_SECONDS=180
if not defined AQUASCAN_RETRIEVAL_K set AQUASCAN_RETRIEVAL_K=4

if not defined AQUASCAN_ASSISTANT_GPU_LAYERS (
  ".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, r'..\AI chatbot'); import aquaculture_assistant.llm.local_llm; import llama_cpp; info=llama_cpp.llama_cpp.llama_print_system_info().decode(); raise SystemExit(0 if 'CUDA' in info else 1)"
  if errorlevel 1 (
    set AQUASCAN_ASSISTANT_GPU_LAYERS=0
  ) else (
    set AQUASCAN_ASSISTANT_GPU_LAYERS=-1
  )
)

echo.
echo Starting AquaScan backend on http://127.0.0.1:8000
echo   - Health:     http://127.0.0.1:8000/health
echo   - Model info: http://127.0.0.1:8000/model/info
echo   - Assistant:  http://127.0.0.1:8000/assistant/health
echo   - API docs:   http://127.0.0.1:8000/docs
echo   - LLM GPU layers: %AQUASCAN_ASSISTANT_GPU_LAYERS%
echo   - LLM max tokens: %AQUASCAN_ASSISTANT_MAX_TOKENS%
echo.
".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
