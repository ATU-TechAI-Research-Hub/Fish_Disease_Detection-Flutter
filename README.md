# AquaScan — AI-Powered Freshwater Fish Disease Detection

A complete Flutter + FastAPI application that identifies freshwater fish
diseases from a single photo, implementing the CNN architecture from
[Tamut et al., *Aquaculture Journal* 2025, 5(1), 6](https://doi.org/10.3390/aquacj5010006).

The app runs **fully offline** once installed: the model, labels, dataset
metadata, and disease information are bundled locally and inference is executed
against a local Keras `model.h5` (primary) with an ONNX fallback. No external
APIs are contacted for predictions.

---

## At a glance

| Layer | Tech |
|-------|------|
| Frontend | Flutter (Material 3, Dart 3.3+) |
| Backend | FastAPI + Uvicorn (Python 3.11+) |
| Model (primary) | Keras `.h5` (TensorFlow 2.x) |
| Model (fallback) | ONNX Runtime |
| Architecture | Paper-exact CNN — 3 Conv2D blocks → Dense(256) → Softmax (~2.2M params) |
| Input | 150×150 RGB, pixel values divided by 255 |
| Loss / optimizer | `categorical_crossentropy` / Adam |
| Callbacks | `EarlyStopping`, `ReduceLROnPlateau` |
| Dataset | [Kaggle — Freshwater Fish Disease, South Asia](https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia) — 2,444 images, 7 classes |

---

## Disease classes (matches the paper)

| # | Class | Type |
|---|-------|------|
| 1 | Bacterial Red Disease | Bacterial |
| 2 | Bacterial Aeromoniasis | Bacterial |
| 3 | Bacterial Gill Disease | Bacterial |
| 4 | Fungal Saprolegniasis | Fungal |
| 5 | Healthy Fish | Healthy |
| 6 | Parasitic Disease | Parasitic |
| 7 | Viral – White Tail Disease | Viral |

The Flutter UI also surfaces a synthetic *"No Fish Detected"* class when the
prediction is too uncertain (low max-confidence or high entropy).

---

## Features

- Camera & gallery upload (Android / iOS) and gallery upload on desktop / web
- Image preview before prediction
- Result card with:
  - detected disease name
  - confidence percentage
  - **High / Medium / Low confidence tier**
  - cause, symptoms, treatment, prevention
  - recommended next step (tier-aware)
  - low-quality / unrecognisable image warning
  - top-3 predictions with progress bars
- Loading state with animated header
- Connection / "Offline mode" indicator that polls `/health`
- Scan history with re-open + clear-all
- Disease library — read about each class without scanning
- Accessibility:
  - `Semantics` labels on result tier badges and status pills
  - Tooltips on connection-status icons
  - Material 3 contrast / responsive layouts
- Offline-friendly:
  - Local Keras `.h5` model (~8.4 MB)
  - ONNX fallback (~8.8 MB) bundled with the repo
  - No network calls for predictions
  - Cleartext HTTP allowed only on local network

---

## Project structure

```
Aquaculture/
├── model/
│   ├── labels.json                  # Class index → disease (canonical source)
│   └── model.h5                     # ← place / generated Keras model here
├── assets/
│   └── diseases.json                # Per-disease metadata (cause/symptoms/treatment/prevention)
├── lib/                             # Flutter app
│   ├── main.dart
│   ├── models/
│   │   ├── disease_model.dart
│   │   └── prediction_result_model.dart
│   ├── screens/
│   │   ├── app_shell.dart
│   │   ├── home_screen.dart
│   │   ├── result_screen.dart
│   │   ├── disease_library_screen.dart
│   │   └── scan_history_screen.dart
│   ├── services/
│   │   ├── api_prediction_service.dart
│   │   ├── backend_status_service.dart
│   │   └── scan_history_service.dart
│   ├── theme/
│   │   └── app_theme.dart
│   └── widgets/
│       ├── backend_status_banner.dart
│       ├── bubble_background.dart
│       ├── confidence_ring.dart
│       └── wave_clipper.dart
├── backend/
│   ├── app/
│   │   ├── core/                    # Shared inference primitives
│   │   │   ├── preprocessing.py     # Paper preprocessing (150×150, /255)
│   │   │   ├── labels.py            # Label loader
│   │   │   └── model_loader.py      # KerasH5Model + OnnxModel (unified)
│   │   ├── ml/
│   │   │   └── fish_disease_classifier.onnx   # Offline ONNX fallback
│   │   ├── services/
│   │   │   └── prediction_service.py
│   │   ├── main.py                  # FastAPI app
│   │   └── models.py                # Pydantic response models
│   ├── train/
│   │   ├── train.py                 # Paper-exact Keras trainer → model/model.h5
│   │   └── evaluate.py              # Confusion matrix + per-class P/R/F1
│   ├── tests/
│   │   ├── smoke_test.py            # End-to-end HTTP test
│   │   └── predict_cli.py           # No-API CLI prediction
│   ├── outputs/                     # Training & evaluation reports (generated)
│   ├── requirements.txt
│   ├── run_backend.bat
│   └── README.md
├── Freshwater_Fish_Disease_Aquaculture_in_south_asia/   # Kaggle dataset (Train/, Test/)
├── pubspec.yaml
└── README.md
```

---

## Quick start

### 1. Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
run_backend.bat                                    # or: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify in a browser:

- http://127.0.0.1:8000/        → service info
- http://127.0.0.1:8000/health  → `model_ready: true` once the model is loaded
- http://127.0.0.1:8000/docs    → Swagger UI

### 2. Train `model.h5` (optional but recommended)

If `model/model.h5` does not exist yet, run:

```powershell
cd backend
.venv\Scripts\python.exe -m train.train
```

This saves a new model to `model/model.h5` and writes training stats to
`backend/outputs/`. The backend will pick the new file up automatically on its
next restart. Until then it transparently uses the bundled ONNX fallback.

### 3. Evaluate the model

```powershell
cd backend
.venv\Scripts\python.exe -m train.evaluate
```

Generates `backend/outputs/evaluation_summary.json`, `confusion_matrix.csv`,
and a per-class classification report.

### 4. Flutter app

```bash
flutter pub get
flutter run                # or: flutter run -d windows / chrome / macos
```

For physical Android/iOS devices, set `lanIp` in
`lib/services/api_prediction_service.dart` to your computer's LAN IP
(`ipconfig` on Windows, `ipconfig getifaddr en0` on macOS).

---

## How `model.h5` is loaded

`backend/app/main.py` resolves the model in this order:

1. `model/model.h5`                    — project-level, primary
2. `backend/app/ml/model.h5`           — secondary (e.g. when packaging)
3. `backend/app/ml/fish_disease_classifier.onnx` — ONNX fallback

The first artifact that loads successfully becomes the active backend.

You can override the default with an environment variable:

```powershell
# Default: prefer Keras .h5 with ONNX fallback
$env:AQUASCAN_MODEL_PREFERENCE = "h5"   # equivalent to no override

# Force the ONNX backend (skip .h5 entirely)
$env:AQUASCAN_MODEL_PREFERENCE = "onnx"
```

Whatever backend is selected, the same preprocessing pipeline runs in front of
it (see `backend/app/core/preprocessing.py`), so inference is identical:

1. EXIF-aware orientation fix.
2. Convert to RGB.
3. Resize to 150 × 150 using LANCZOS.
4. Normalize to `[0, 1]` (divide by 255).
5. Add batch dimension → `(1, 150, 150, 3)`.

If the model cannot be loaded, `/health` reports `model_ready: false` and the
Flutter UI shows a yellow "Model not loaded" pill instead of letting users try
to scan.

---

## Offline usage

Everything required to run AquaScan is shipped or generated locally:

| Asset | Path |
|-------|------|
| Keras model | `model/model.h5` |
| ONNX fallback | `backend/app/ml/fish_disease_classifier.onnx` |
| Label mapping | `model/labels.json` |
| Disease descriptions | `assets/diseases.json` |
| Backend code | `backend/...` |
| Flutter app | `lib/...` |

After the first `pip install` and `flutter pub get`, the app does **not** need
internet access. The only outbound HTTP request is to `127.0.0.1:8000` (the
local FastAPI server).

The Flutter UI itself includes:

- A persistent **Offline mode / AI online / Model not loaded** banner.
- A compact connectivity pill in the wave header.
- Scan buttons that refuse to start if the backend is unreachable.

---

## Online usage

The app does not require online services to operate. If a network is present:

- The Flutter app reaches the backend over HTTP on either `127.0.0.1`,
  `10.0.2.2` (Android emulator), or the configured `lanIp`.
- The backend itself only performs local file I/O — no telemetry is sent.

If hosting the backend remotely is desired, expose port `8000` behind a reverse
proxy (e.g. Nginx) and update `lanIp` in
`lib/services/api_prediction_service.dart` to the public URL.

---

## API reference

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Service info, links to docs |
| `GET` | `/health` | Returns `model_ready`, backend type, device |
| `GET` | `/model/info` | Full `ModelStatus` (paths, num_classes, image_size) |
| `GET` | `/diseases` | List of disease metadata records |
| `POST` | `/predict` | Multipart upload `file` → `PredictionResponse` |

**`PredictionResponse`** payload:

```json
{
  "prediction": { "id": 1, "name": "Bacterial Red Disease", "type": "Bacterial", "cause": "...", "symptoms": "...", "treatment": "...", "prevention": "..." },
  "confidence": 0.9521,
  "confidence_tier": "high",
  "source": "keras-h5",
  "filename": "fish.jpg",
  "inference_ms": 18.4,
  "top_predictions": [
    {"disease_id": 1, "disease_name": "Bacterial Red Disease", "confidence": 0.9521},
    {"disease_id": 6, "disease_name": "Parasitic Disease",     "confidence": 0.0287},
    {"disease_id": 2, "disease_name": "Bacterial Aeromoniasis", "confidence": 0.0103}
  ],
  "warning": null,
  "recommendation": "Recommended action: Isolate infected fish, improve aeration, ..."
}
```

---

## Testing prediction without the Flutter app

End-to-end HTTP smoke test (backend must be running):

```powershell
cd backend
.venv\Scripts\python.exe -m tests.smoke_test
```

Direct (no HTTP) CLI prediction:

```powershell
.venv\Scripts\python.exe -m tests.predict_cli --image-path "..\Freshwater_Fish_Disease_Aquaculture_in_south_asia\Test\Bacterial Red disease\Bacterial Red disease (1).jpg"
```

---

## Train / fine-tune the model

The trainer follows the paper as faithfully as the Kaggle dataset allows:

| Setting | Value |
|---------|-------|
| Input | 150×150×3 RGB, /255 |
| Architecture | `Conv2D(128, 5×5) → MaxPool → BN → Dropout(0.25)` × Conv2D(64, 3×3) → MaxPool → BN → Dropout(0.25) → Conv2D(32, 3×3) → MaxPool → BN → Dropout(0.25) → Flatten → Dense(256, relu) → Dropout(0.5) → Dense(K, softmax) |
| Optimizer | `Adam` (LR 1e-3 by default) |
| Loss | `categorical_crossentropy` |
| Callbacks | `EarlyStopping(patience=12)`, `ReduceLROnPlateau(factor=0.5)` |
| Class weights | Inverse-frequency (for imbalanced classes) |
| Augmentation | Rotation, shifts, horizontal flip, zoom, brightness |

Run with custom hyperparameters:

```powershell
cd backend
.venv\Scripts\python.exe -m train.train --epochs 60 --batch-size 32 --learning-rate 0.001
```

Outputs (in `backend/outputs/`):

- `training_history.json` — per-epoch metrics
- `training_summary.json` — config + best val accuracy
- `model/model.h5` — the trained Keras model (in the project root)

---

## Evaluating accuracy / precision / recall / F1

```powershell
cd backend
.venv\Scripts\python.exe -m train.evaluate
```

Writes to `backend/outputs/`:

- `evaluation_summary.json` — accuracy, loss, top-3 accuracy, macro / weighted averages
- `classification_report.json` — per-class precision / recall / F1 / support
- `confusion_matrix.csv` — 7×7 confusion matrix
- `confusion_matrix.json` — same data as JSON

These map directly to the metrics reported in the research paper (accuracy,
precision, recall, F1, confusion matrix, test loss).

---

## Troubleshooting

**`/health` shows `model_ready: false`.**
No `.h5` or `.onnx` model could be loaded. Either drop a trained `model.h5`
into `model/` or run `python -m train.train` to create one.

**Flutter app shows "Offline mode".**
Backend is not running or not reachable. Start it with `run_backend.bat` or
adjust `lanIp` in `lib/services/api_prediction_service.dart`.

**400 Bad Request on `/predict`.**
The image is empty, larger than 15 MB, or is in an unsupported format. JPEG,
PNG, WebP, GIF, and BMP are accepted.

**TensorFlow installation fails.**
TensorFlow 2.x officially supports Python 3.9-3.13. The bundled `.venv` was
created with Python 3.13 and TensorFlow 2.21 — if you hit issues, recreate the
venv with `py -3.13 -m venv .venv`.

**PyTorch / ONNX prerequisites missing.**
The legacy ONNX fallback only requires `onnxruntime`; PyTorch is no longer
needed for inference.

---

## Citation

```
Tamut, J., Mangang, Y.A., & Chingakham, C. (2025).
"Image Classification of Freshwater Fish Diseases in South Asian Aquaculture
Using Convolutional Neural Network."
Aquaculture Journal, 5(1), 6. https://doi.org/10.3390/aquacj5010006
```

---

## License

Developed for academic research at Atlantic Technological University (ATU).
