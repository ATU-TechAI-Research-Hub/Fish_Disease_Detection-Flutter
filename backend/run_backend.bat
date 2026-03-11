@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)

echo Installing backend dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Starting FastAPI backend on http://127.0.0.1:8000
".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
