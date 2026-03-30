# AI-Powered Fish Disease Detection

Flutter + FastAPI application for freshwater fish disease detection, implementing the custom CNN architecture from [Tamut et al., *Aquac. J.* 2025, 5, 6](https://doi.org/10.3390/aquacj5010006).

## Project Overview

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Flutter (Dart) | Mobile app — capture/upload fish images, display results |
| **Backend** | FastAPI (Python) | REST API — receives images, runs AI inference, returns predictions |
| **ML Model** | Custom CNN → ONNX | 3 Conv2D blocks, 2.2M params, 7-class classifier (8.4 MB) |
| **Training** | PyTorch + CUDA | GPU-accelerated training on 2,444 fish images |
| **Dataset** | Kaggle (South Asian freshwater fish) | 7 disease classes, original Train/Test split preserved |

**Achieved: 95.4% test accuracy**

---

## Prerequisites

Install these before starting:

| Tool | Version | Download |
|------|---------|----------|
| **Python** | 3.11+ | https://www.python.org/downloads/ |
| **Flutter SDK** | 3.3+ | https://docs.flutter.dev/get-started/install |
| **Git** | any | https://git-scm.com/downloads |
| **Android Studio** or **VS Code** | latest | For Flutter device/emulator |
| **NVIDIA GPU drivers** | latest *(optional)* | For CUDA-accelerated training |

Verify installations:

```bash
python --version
flutter doctor
git --version
```

---

## Project Structure

```
Aquaculture/
├── assets/
│   └── diseases.json                  # Disease metadata (7 classes)
├── lib/                               # Flutter frontend
│   ├── main.dart                      # App entry point
│   ├── models/
│   │   ├── disease_model.dart         # Disease data model
│   │   └── prediction_result_model.dart
│   ├── screens/
│   │   ├── app_shell.dart             # Bottom navigation shell
│   │   ├── home_screen.dart           # Home — scan buttons
│   │   ├── result_screen.dart         # Prediction result display
│   │   ├── disease_library_screen.dart # Disease encyclopedia
│   │   └── scan_history_screen.dart   # Past scan history
│   ├── services/
│   │   ├── api_prediction_service.dart # HTTP client to backend
│   │   └── scan_history_service.dart   # In-memory scan history
│   ├── theme/
│   │   └── app_theme.dart             # App-wide theme & colors
│   └── widgets/
│       ├── confidence_ring.dart       # Animated confidence gauge
│       ├── gradient_card.dart         # Gradient action card
│       └── shimmer_loading.dart       # Shimmer placeholder
├── backend/
│   ├── requirements.txt               # Python dependencies
│   ├── run_backend.bat                # One-click Windows launcher
│   ├── app/
│   │   ├── main.py                    # FastAPI app + routes
│   │   ├── models.py                  # Pydantic response models
│   │   ├── ml/
│   │   │   ├── fish_disease_classifier.onnx  # Trained model
│   │   │   └── class_map.json                # Class index → disease mapping
│   │   └── services/
│   │       └── prediction_service.py  # ONNX inference logic
│   ├── train/
│   │   ├── prepare_dataset.py         # Dataset scanning & splitting
│   │   ├── train_classifier.py        # PyTorch training pipeline
│   │   └── export_model.py            # ONNX export utility
│   └── tests/
│       └── smoke_test.py              # Backend API smoke test
├── Freshwater_Fish_Disease_Aquaculture_in_south_asia/  # Dataset (not in repo)
│   ├── Train/                         # 1,747 training images
│   │   ├── Bacterial Red disease/
│   │   ├── Bacterial diseases - Aeromoniasis/
│   │   ├── Bacterial gill disease/
│   │   ├── Fungal diseases Saprolegniasis/
│   │   ├── Healthy Fish/
│   │   ├── Parasitic diseases/
│   │   └── Viral diseases White tail disease/
│   └── Test/                          # 697 test images
│       └── (same 7 folders)
├── pubspec.yaml                       # Flutter dependencies
└── README.md                          # This file
```

---

## Complete Step-by-Step Setup

### Step 1 — Clone the Repository

```bash
git clone git@github.com:ATU-TechAI-Research-Hub/Fish_Disease_Detection-Flutter.git
cd Fish_Disease_Detection-Flutter
```

---

### Step 2 — Download the Dataset

Download the **Freshwater Fish Disease Aquaculture in South Asia** dataset from Kaggle:

https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia

Place the extracted folder so the structure looks like this:

```
Aquaculture/
└── Freshwater_Fish_Disease_Aquaculture_in_south_asia/
    ├── Train/
    │   ├── Bacterial Red disease/
    │   │   ├── Bacterial Red disease (1).jpg
    │   │   ├── Bacterial Red disease (2).jpg
    │   │   └── ...
    │   ├── Bacterial diseases - Aeromoniasis/
    │   ├── Bacterial gill disease/
    │   ├── Fungal diseases Saprolegniasis/
    │   ├── Healthy Fish/
    │   ├── Parasitic diseases/
    │   └── Viral diseases White tail disease/
    └── Test/
        └── (same 7 subfolders)
```

The dataset contains **2,444 total images** across 7 classes.

> **Note:** If you only want to run the app (not retrain), you can skip this step — the pre-trained ONNX model is already included in `backend/app/ml/`.

---

### Step 3 — Set Up the Python Backend

Open a terminal and navigate to the `backend` folder:

```bash
cd backend
```

**3a. Create a Python virtual environment:**

```bash
python -m venv .venv
```

**3b. Activate the virtual environment:**

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

macOS / Linux:
```bash
source .venv/bin/activate
```

**3c. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

This installs: FastAPI, uvicorn, onnxruntime-gpu, PyTorch, torchvision, Pillow, pandas, scikit-learn, numpy, tqdm.

**3d. (Optional) Install CUDA-enabled PyTorch for GPU training:**

If you have an NVIDIA GPU and want faster training:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

---

### Step 4 — Prepare the Dataset (Only If Retraining)

> **Skip this step** if you are using the pre-trained ONNX model already in `backend/app/ml/`.

From the `backend` folder, run:

```bash
python -m train.prepare_dataset
```

On Windows without activating the venv, use the full path:

```powershell
& ".venv\Scripts\python.exe" -m train.prepare_dataset
```

**What this does:**
1. Scans `Freshwater_Fish_Disease_Aquaculture_in_south_asia/Train/` and `Test/` folders
2. Maps each subfolder to one of the 7 disease classes
3. Splits the data: **1,747 training** / **348 validation** / **349 test** images
4. Generates artifacts in `backend/train/artifacts/`:
   - `class_map.json` — class index to disease ID mapping
   - `train_split.csv`, `val_split.csv`, `test_split.csv` — image manifests
   - `dataset_summary.json` — dataset statistics

**Expected output:**

```
Scanning Train folder... found 1747 images
Scanning Test folder... found 697 images
Total: 2444 images across 7 classes
Artifacts saved to backend\train\artifacts\
```

---

### Step 5 — Train the CNN Model (Only If Retraining)

> **Skip this step** if you are using the pre-trained ONNX model already in `backend/app/ml/`.

From the `backend` folder, run:

```bash
python -m train.train_classifier --epochs 50
```

On Windows without activating the venv:

```powershell
& ".venv\Scripts\python.exe" -m train.train_classifier --epochs 50
```

**What this does:**
1. Loads train/val/test splits from Step 4
2. Builds the custom CNN model (PaperCNN architecture):
   - 3 Conv2D blocks (128 → 64 → 32 filters) with BatchNorm, MaxPool, Dropout
   - Dense(256) → Dense(7, Softmax)
   - 2,201,639 total parameters
3. Trains for up to 50 epochs with:
   - Adam optimizer (lr=0.001)
   - CrossEntropy loss + L1 regularization
   - ReduceLROnPlateau scheduler
   - Early stopping (patience=10) on validation loss
   - Data augmentation: horizontal flip, rotation ±15°, color jitter
4. Evaluates on the test set
5. Exports the trained model

**Output files:**
| File | Location | Description |
|------|----------|-------------|
| `fish_disease_classifier.onnx` | `backend/app/ml/` | Trained ONNX model (~8.4 MB) |
| `class_map.json` | `backend/app/ml/` | Class mapping for inference |
| `metrics.json` | `backend/train/artifacts/` | Training metrics & accuracy |

**Expected training output:**

```
Using device: cuda (NVIDIA GeForce GTX 1660 Ti)
Model: PaperCNN | Parameters: 2,201,639

Epoch  1/50 ▸ train_loss=1.42  train_acc=48.2%  val_loss=0.89  val_acc=67.5%
Epoch  2/50 ▸ train_loss=0.81  train_acc=70.1%  val_loss=0.52  val_acc=82.3%
...
Epoch 50/50 ▸ train_loss=0.08  train_acc=98.1%  val_loss=0.15  val_acc=95.7%

Test Accuracy: 95.4%
Model exported to backend\app\ml\fish_disease_classifier.onnx (8.40 MB)
```

---

### Step 6 — Start the FastAPI Backend Server

From the `backend` folder, run:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On Windows without activating the venv:

```powershell
& ".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Or use the one-click launcher:**

```cmd
run_backend.bat
```

**Expected output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to stop)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Verify the backend is running — open these URLs in your browser:**

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8000 | Base info page |
| http://127.0.0.1:8000/health | Health check — shows `model_ready: true` |
| http://127.0.0.1:8000/docs | Swagger UI — interactive API documentation |
| http://127.0.0.1:8000/diseases | List all 7 disease classes as JSON |

**Health check response should look like:**

```json
{
  "status": "ok",
  "model_ready": true,
  "source": "onnxruntime-cpu"
}
```

> **Keep this terminal open.** The backend must be running while you use the Flutter app.

---

### Step 7 — Smoke Test the Backend (Optional)

Open a **second terminal** (keep the backend running in the first). From the `backend` folder:

```bash
python tests/smoke_test.py --image-path "path\to\any\fish\image.jpg"
```

Example with a dataset image:

```powershell
& ".venv\Scripts\python.exe" tests\smoke_test.py --image-path "..\Freshwater_Fish_Disease_Aquaculture_in_south_asia\Test\Bacterial Red disease\Bacterial Red disease (1).jpg"
```

**Expected output:**

```json
Health: {
  "status": "ok",
  "model_ready": true,
  "source": "onnxruntime-cpu"
}
Predict: {
  "prediction": {
    "id": 1,
    "name": "Bacterial Red Disease",
    "type": "Bacterial",
    "cause": "Usually caused by poor water quality...",
    "symptoms": "Red patches on the body...",
    "treatment": "Isolate infected fish...",
    "prevention": "Maintain clean water..."
  },
  "confidence": 0.9823,
  "source": "onnxruntime-cpu",
  "filename": "Bacterial Red disease (1).jpg"
}
```

---

### Step 8 — Set Up the Flutter Frontend

Open a **new terminal** at the project root (`Aquaculture/` folder):

**8a. Install Flutter dependencies:**

```bash
flutter pub get
```

**8b. Check your Flutter environment:**

```bash
flutter doctor
```

Make sure at least one target platform shows a green checkmark (Android, Chrome, Windows, etc.).

---

### Step 9 — Run the Flutter App

Make sure the backend server is still running (Step 6), then:

**Option A — Run on an Android emulator:**

```bash
flutter run
```

Select the Android emulator when prompted. The app connects to the backend at `http://10.0.2.2:8000` (Android emulator's alias for the host machine's localhost).

**Option B — Run on Chrome (web):**

```bash
flutter run -d chrome
```

The app connects to `http://127.0.0.1:8000`.

**Option C — Run on Windows desktop:**

```bash
flutter run -d windows
```

The app connects to `http://127.0.0.1:8000`.

**Option D — Run on a physical Android device:**

1. Enable USB debugging on your phone
2. Connect via USB
3. Run `flutter run`
4. **Important:** Your phone and computer must be on the same Wi-Fi network. Update the backend URL in `lib/services/api_prediction_service.dart` to your computer's local IP (e.g., `http://192.168.1.100:8000`).

---

### Step 10 — Use the App

Once the app launches, you will see three tabs:

**Home tab:**
1. Tap **"Take a Photo"** to capture a live fish image with the camera
2. Or tap **"Upload from Gallery"** to select an existing image
3. The app sends the image to the backend for AI analysis
4. Results appear with the disease name, confidence gauge, and treatment details

**Diseases tab:**
- Browse all 7 detectable fish disease classes
- Tap any disease to see its cause, symptoms, treatment, and prevention

**History tab:**
- View all past scan results from the current session
- Tap any entry to see the full result again
- Use "Clear" to reset the history

---

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HOW IT WORKS                                  │
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  Camera/  │───▸│  Flutter  │───▸│  FastAPI  │───▸│  ONNX Model  │  │
│  │  Gallery  │    │  Mobile   │    │  Backend  │    │  (CNN 8.4MB) │  │
│  └──────────┘    │   App     │    │  Server   │    └──────┬───────┘  │
│                  └─────┬─────┘    └─────┬─────┘           │          │
│                        │                │                  │          │
│                        │   HTTP POST    │    Inference     │          │
│                        │   /predict     │    (150x150      │          │
│                        │   multipart    │     → softmax)   │          │
│                        │                │                  │          │
│                  ┌─────▼─────┐    ┌─────▼─────┐    ┌──────▼───────┐  │
│                  │  Display   │◂──│   JSON     │◂──│  Prediction  │  │
│                  │  Results   │   │  Response  │   │  + Confidence│  │
│                  └───────────┘    └───────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Detailed pipeline:**

1. **User** takes a photo or selects an image from the gallery
2. **Flutter app** sends the image as a multipart `POST /predict` request to the backend
3. **FastAPI** receives the image bytes
4. **Prediction service** preprocesses the image:
   - Resize to 150 × 150 pixels
   - Convert to RGB float array
   - Normalize pixel values to [0, 1] (divide by 255)
   - Transpose to CHW format (channels first)
5. **ONNX Runtime** loads the CNN model and runs inference
6. **Softmax** converts logits to class probabilities
7. **Class map** translates the predicted class index to a disease ID
8. **Disease metadata** from `diseases.json` provides name, cause, symptoms, treatment, prevention
9. **JSON response** is returned with the disease details + confidence score + runtime source
10. **Flutter app** displays the result: disease name, animated confidence ring, info cards

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Base info and available endpoints |
| `GET` | `/health` | Health check — model status and runtime |
| `GET` | `/diseases` | List all 7 disease classes |
| `POST` | `/predict` | Upload an image (`file` field) → get prediction |
| `GET` | `/docs` | Swagger UI (interactive API docs) |
| `GET` | `/redoc` | ReDoc (alternative API docs) |

**`POST /predict` request:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@path/to/fish_image.jpg"
```

**Response format:**

```json
{
  "prediction": {
    "id": 1,
    "name": "Bacterial Red Disease",
    "type": "Bacterial",
    "cause": "Usually caused by poor water quality...",
    "symptoms": "Red patches on the body...",
    "treatment": "Isolate infected fish...",
    "prevention": "Maintain clean water..."
  },
  "confidence": 0.9823,
  "source": "onnxruntime-cpu",
  "filename": "fish_image.jpg"
}
```

---

## Disease Classes (7)

| # | Class | Type |
|---|-------|------|
| 1 | Bacterial Red Disease | Bacterial |
| 2 | Bacterial Aeromoniasis | Bacterial |
| 3 | Bacterial Gill Disease | Bacterial |
| 4 | Fungal Saprolegniasis | Fungal |
| 5 | Healthy Fish | Healthy |
| 6 | Parasitic Disease | Parasitic |
| 7 | White Tail Disease | Viral |

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Custom CNN (Tamut et al., 2025) |
| Input size | 150 × 150 × 3 (RGB) |
| Conv blocks | 3 (128 → 64 → 32 filters) |
| Classifier | Dense(256) → Dense(7) |
| Parameters | 2,201,639 |
| ONNX size | ~8.4 MB |
| Test accuracy | 95.4% |
| Training GPU | NVIDIA GTX 1660 Ti |
| Framework | PyTorch → ONNX export |
| Inference | ONNX Runtime (CPU or CUDA) |

---

## Troubleshooting

### Backend won't start

- Make sure you activated the virtual environment or use the full path to `.venv\Scripts\python.exe`
- Check that port 8000 is not already in use: `netstat -ano | findstr :8000`
- Verify the ONNX model exists: `dir backend\app\ml\fish_disease_classifier.onnx`

### `model_ready: false` at /health

- The ONNX model or `class_map.json` is missing from `backend/app/ml/`
- Either retrain (Steps 4–5) or ensure the files were cloned properly

### Flutter app shows "Connection Failed"

- Make sure the backend is running (Step 6) before launching the Flutter app
- On Android emulator, the app uses `http://10.0.2.2:8000` — this only works if the backend is on the same machine
- On physical device, update the base URL in `lib/services/api_prediction_service.dart` to your computer's local IP

### CUDA / GPU not detected during training

- Install CUDA-enabled PyTorch: `pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`
- Training falls back to CPU automatically if CUDA is unavailable

### PowerShell `&&` operator error

PowerShell versions before 7.0 don't support `&&`. Use `;` instead, or run commands one at a time.

---

## Quick Start (TL;DR)

If the ONNX model is already in `backend/app/ml/`, you only need two terminals:

**Terminal 1 — Backend:**

```bash
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Flutter:**

```bash
flutter pub get
flutter run
```

---

## References

1. Tamut, J., Mangang, Y.A., & Chingakham, C. (2025). "Image Classification of Freshwater Fish Diseases in South Asian Aquaculture Using Convolutional Neural Network." *Aquaculture Journal*, 5(1), 6. https://doi.org/10.3390/aquacj5010006

---

## License

This project was developed for academic research at Arkansas Tech University.
