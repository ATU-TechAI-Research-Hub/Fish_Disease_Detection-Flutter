# FastAPI Backend

This backend serves the real fish disease classifier for the Flutter app.
It uses:

- a cleaned manifest generated from the dataset folders
- a PyTorch training pipeline
- an exported ONNX model
- ONNX Runtime inference inside FastAPI

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional, for GPU training with CUDA-enabled PyTorch on Windows:

```bash
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

## Prepare The Dataset

This scans `Freshwater_Fish_Disease_Aquaculture_in_south_asia/Train` and `Test`,
detects duplicate filename stems, reconciles `Train.csv`, and creates clean
train/validation/test manifests in `backend/train/artifacts/`.

```bash
.venv\Scripts\python.exe -m train.prepare_dataset
```

## Train And Export The Model

This trains the classifier, evaluates it, and exports:

- `backend/app/ml/fish_disease_classifier.onnx`
- `backend/app/ml/class_map.json`
- training metrics in `backend/train/artifacts/`

```bash
.venv\Scripts\python.exe -m train.train_classifier --epochs 6
```

## Run The API

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or on Windows, just run:

```bash
run_backend.bat
```

## Smoke Test The API

```bash
.venv\Scripts\python.exe tests\smoke_test.py --image-path "path\to\fish.jpg"
```

## Runtime Note

- FastAPI prefers `onnxruntime-gpu` when the required CUDA 12 and cuDNN 9 runtime DLLs are available on the machine.
- If those DLLs are missing, the backend now falls back cleanly to `onnxruntime-cpu` and still serves real predictions.

## Endpoints

- `GET /` - base info page
- `GET /health` - health check
- `GET /diseases` - list all disease classes
- `POST /predict` - upload image (`file`) and get a real classifier prediction
