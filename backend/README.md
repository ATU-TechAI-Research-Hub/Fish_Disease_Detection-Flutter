# AquaScan Backend (FastAPI)

This is the inference / training backend for AquaScan, implementing the custom
CNN architecture from Tamut et al., *Aquac. J.* 2025, 5(1), 6 (doi: 10.3390/aquacj5010006).

## Highlights

| Component | Tech |
|-----------|------|
| Web server | FastAPI + Uvicorn |
| ML inference (primary) | TensorFlow / Keras (`.h5`) |
| ML inference (fallback) | ONNX Runtime (offline-friendly) |
| Training | Keras (`Adam`, `categorical_crossentropy`, `EarlyStopping`, `ReduceLROnPlateau`) |
| Image input | 150×150 RGB, normalised by /255 (matches the paper) |

The backend uses `model/model.h5` (when present) as the primary inference
artifact. If it cannot be found, it transparently falls back to the legacy ONNX
model in `backend/app/ml/fish_disease_classifier.onnx`. Both backends share the
same preprocessing pipeline (`app/core/preprocessing.py`), so predictions are
consistent across both modes.

## Project layout

```
backend/
├── app/
│   ├── core/
│   │   ├── preprocessing.py       # Paper-exact preprocessing pipeline (150x150, /255)
│   │   ├── labels.py              # Loads model/labels.json (typed dataclasses)
│   │   └── model_loader.py        # KerasH5Model + OnnxModel (unified interface)
│   ├── ml/
│   │   └── fish_disease_classifier.onnx  # ONNX fallback model (offline)
│   ├── services/
│   │   └── prediction_service.py  # High-level prediction logic + tiers
│   ├── main.py                    # FastAPI app + routes
│   └── models.py                  # Pydantic request/response models
├── train/
│   ├── audit_dataset.py           # Duplicate/leakage, corruption and balance audit
│   ├── data.py                    # Inference-aligned loader + deterministic split
│   ├── train.py                   # MobileNetV2 transfer learning or paper CNN
│   └── evaluate.py                # Accuracy, calibration and robust imbalance metrics
├── tests/
│   ├── smoke_test.py              # End-to-end HTTP smoke test
│   └── predict_cli.py             # Direct (no-API) CLI prediction
├── outputs/                       # Training & evaluation reports (generated)
├── requirements.txt
└── run_backend.bat
```

## Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train `model.h5`

The trainer reads images directly from
`Freshwater_Fish_Disease_Aquaculture_in_south_asia/Train/<class>/...` and writes
the result to `model/model.h5`.

The default is now ImageNet-pretrained MobileNetV2 with staged fine-tuning.
The original architecture remains available with `--architecture paper_cnn`.
See [ACCURACY_RESEARCH.md](ACCURACY_RESEARCH.md) for the evidence, limitations,
leakage audit, controlled comparison protocol and model promotion criteria.

```powershell
.venv\Scripts\python.exe -m train.audit_dataset
.venv\Scripts\python.exe -m train.train
.venv\Scripts\python.exe -m train.train --architecture paper_cnn
```

Training also writes:

- `backend/outputs/training_history.json` — per-epoch loss & accuracy
- `backend/outputs/training_summary.json` — config + final metrics
- `backend/outputs/validation_split.json` — exact reproducible file split

## Evaluate the model

```powershell
.venv\Scripts\python.exe -m train.evaluate
```

This walks the `Test/` folder, computes predictions, and writes to
`backend/outputs/`:

- `evaluation_summary.json` — top-1/top-3 and balanced accuracy, loss,
  macro/weighted P/R/F1, MCC, Cohen's kappa, Brier score and ECE
- `classification_report.json` — per-class precision / recall / F1
- `confusion_matrix.csv` + `confusion_matrix.json`

## Run the API

```powershell
run_backend.bat
```

or manually:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Smoke-test the API

```powershell
.venv\Scripts\python.exe -m tests.smoke_test
```

## No-API CLI prediction

```powershell
.venv\Scripts\python.exe -m tests.predict_cli --image-path "..\Freshwater_Fish_Disease_Aquaculture_in_south_asia\Test\Bacterial Red disease\Bacterial Red disease (1).jpg"
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Liveness + model status |
| `GET` | `/model/info` | Detailed model status (backend, path, device, num_classes) |
| `GET` | `/diseases` | All 7 disease records (cause / symptoms / treatment / prevention) |
| `POST` | `/predict` | Multipart upload `file` → `PredictionResponse` |
| `GET` | `/docs` | Swagger UI |
