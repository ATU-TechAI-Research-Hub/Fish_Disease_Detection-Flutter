# Freshwater Fish Disease Aquaculture in South Asia

Flutter + FastAPI application for freshwater fish disease detection.

## Project Overview

- **Frontend (Flutter):** Capture/upload a fish image and display the predicted disease details.
- **Backend (FastAPI):** Load the exported ONNX classifier and run real image inference.
- **Training Pipeline:** Prepare the dataset, train the classifier on the local GPU, export ONNX, and reuse `assets/diseases.json` for disease metadata.
- **Data Source:** `assets/diseases.json` with 7 disease classes plus the local fish image dataset.

## Train The Model

```bash
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m train.prepare_dataset
.venv\Scripts\python.exe -m train.train_classifier --epochs 6
```

Optional, for CUDA-enabled GPU training on Windows:

```bash
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

Training artifacts are written to:

- `backend/train/artifacts/`
- `backend/app/ml/`

## Run Backend (FastAPI)

```bash
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs:

- Base URL: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Run Frontend (Flutter)

```bash
flutter pub get
flutter run
```

## Notes

- Flutter app calls `POST /predict` using multipart upload (`file` field).
- Default API URL in app is `http://10.0.2.2:8000` on Android and `http://127.0.0.1:8000` on desktop.
- The backend returns the predicted disease metadata, confidence, and runtime source.
- ONNX Runtime uses GPU inference when the required CUDA/cuDNN runtime DLLs are available; otherwise it falls back to CPU automatically.
- The live Flutter API test is opt-in: run `RUN_LIVE_API_TEST=1 flutter test test/api_prediction_service_test.dart` only when the backend is running and the local dataset is present.
