# AquaScan — AI-Powered Fish Disease Detection

Flutter + FastAPI mobile application for freshwater fish disease detection, implementing the custom CNN architecture from [Tamut et al., *Aquac. J.* 2025, 5, 6](https://doi.org/10.3390/aquacj5010006).

## Project Overview

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Flutter (Dart) | Mobile app — capture/upload fish images, display results |
| **Backend** | FastAPI (Python) | REST API — receives images, runs AI inference, returns predictions |
| **ML Model** | Custom CNN → ONNX | 3 Conv2D blocks (PaperCNN), 7-class classifier (~8.4 MB) |
| **Training** | PyTorch + CUDA | GPU-accelerated training on 2,444 fish images |
| **Dataset** | [Kaggle](https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia) (South Asian freshwater fish) | 7 disease classes, original Train/Test split preserved |

---

## Prerequisites

Install these before starting:

| Tool | Version | Download |
|------|---------|----------|
| **Python** | 3.11+ | https://www.python.org/downloads/ |
| **Flutter SDK** | 3.3+ | https://docs.flutter.dev/get-started/install |
| **Git** | any | https://git-scm.com/downloads |
| **Android Studio** or **VS Code** | latest | For Flutter device/emulator setup |
| **NVIDIA GPU + CUDA** | *(optional)* | For faster training — CPU works too |

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
│   │   ├── disease_model.dart
│   │   └── prediction_result_model.dart
│   ├── screens/
│   │   ├── app_shell.dart             # Bottom navigation shell
│   │   ├── home_screen.dart           # Home — scan buttons
│   │   ├── result_screen.dart         # Prediction result display
│   │   ├── disease_library_screen.dart
│   │   └── scan_history_screen.dart
│   ├── services/
│   │   ├── api_prediction_service.dart # HTTP client to backend
│   │   └── scan_history_service.dart   # In-memory scan history
│   ├── theme/
│   │   └── app_theme.dart
│   └── widgets/
│       ├── confidence_ring.dart       # Animated confidence gauge
│       ├── bubble_background.dart     # Floating bubble decoration
│       └── wave_clipper.dart          # Wave-shaped header clipper
├── backend/
│   ├── requirements.txt               # Python dependencies
│   ├── run_backend.bat                # One-click Windows launcher
│   ├── app/
│   │   ├── main.py                    # FastAPI app + routes
│   │   ├── models.py                  # Pydantic response models
│   │   ├── ml/
│   │   │   ├── fish_disease_classifier.onnx  # Trained ONNX model
│   │   │   └── class_map.json                # Class index → disease mapping
│   │   └── services/
│   │       └── prediction_service.py  # ONNX inference + preprocessing
│   ├── train/
│   │   ├── prepare_dataset.py         # Dataset scanning & splitting
│   │   ├── train_classifier.py        # PyTorch training pipeline
│   │   └── export_model.py            # Standalone ONNX export
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
│       └── (same 7 subfolders)
├── pubspec.yaml
└── README.md
```

---

## Complete Step-by-Step Guide

> **Just want to run the app?** If the ONNX model is already in `backend/app/ml/`, skip to [Step 3](#step-3--set-up-the-python-backend).
>
> **Want to retrain the model?** Follow every step from [Step 1](#step-1--clone-the-repository).
>
> **On macOS?** Jump to the [macOS Quick-Start Guide](#macos-step-by-step-guide) for platform-specific commands.

---

### Step 1 — Clone the Repository

```bash
git clone git@github.com:ATU-TechAI-Research-Hub/Fish_Disease_Detection-Flutter.git
cd Fish_Disease_Detection-Flutter
```

---

### Step 2 — Download the Kaggle Dataset (Required for Training Only)

> **Skip this step** if you only want to run the app with the pre-trained model.

1. Go to the Kaggle dataset page:
   https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia

2. Click **Download** and extract the ZIP into the project root so the folder structure is:

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

The dataset contains **2,444 images** across 7 classes.

> **Alternative — Kaggle CLI:**
> ```bash
> pip install kaggle
> kaggle datasets download -d subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia
> ```
> Then unzip into the project root.

---

### Step 3 — Set Up the Python Backend

Open a terminal and navigate to the `backend` folder:

```bash
cd backend
```

**3a. Create a virtual environment:**

```bash
python3 -m venv .venv
```

**3b. Activate it:**

| OS | Command |
|----|---------|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (CMD) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

**3c. Install dependencies:**

```bash
pip install -r requirements.txt
```

This installs: FastAPI, uvicorn, ONNX Runtime, PyTorch, torchvision, Pillow, pandas, scikit-learn, numpy, tqdm.

**3d. (Optional) Install CUDA-enabled PyTorch for GPU training:**

If you have an NVIDIA GPU:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

> Without this, training works on CPU but takes significantly longer.
> macOS with Apple Silicon uses MPS acceleration automatically — no extra install needed.

---

### Step 4 — Prepare the Dataset (Training Only)

> **Skip this step** if you already have the ONNX model in `backend/app/ml/`.

From the `backend` folder (with the venv activated):

```bash
python -m train.prepare_dataset
```

**Windows without activating the venv:**

```powershell
.venv\Scripts\python.exe -m train.prepare_dataset
```

**What this does:**

1. Scans the `Train/` and `Test/` folders for images
2. Maps each subfolder to one of the 7 disease classes
3. Splits data: **1,747 train** / **348 validation** / **349 test**
4. Writes artifacts to `backend/train/artifacts/`:
   - `class_map.json` — class index to disease ID mapping
   - `train_split.csv`, `val_split.csv`, `test_split.csv` — image manifests
   - `audit_summary.json` — dataset statistics

**Expected output:**

```
Scanning Train folder... found 1747 images
Scanning Test folder... found 697 images
Total: 2444 images across 7 classes
Artifacts saved to backend/train/artifacts/
```

---

### Step 5 — Train the CNN Model (Training Only)

> **Skip this step** if you already have the ONNX model in `backend/app/ml/`.

From the `backend` folder:

```bash
python -m train.train_classifier
```

**Windows without activating the venv:**

```powershell
.venv\Scripts\python.exe -m train.train_classifier
```

**Training defaults:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 80 | Maximum training epochs |
| `--batch-size` | 32 | Batch size |
| `--learning-rate` | 5e-4 | AdamW learning rate |
| `--weight-decay` | 1e-4 | L2 regularization |
| `--patience` | 12 | Early stopping patience |
| `--image-size` | 150 | Input image resolution |
| `--label-smoothing` | 0.1 | Label smoothing factor |
| `--mixup-alpha` | 0.2 | Mixup augmentation strength |
| `--warmup-epochs` | 5 | LR warmup epochs |

**Custom training example (fewer epochs):**

```bash
python -m train.train_classifier --epochs 50 --batch-size 64 --learning-rate 8e-4
```

**What this does:**

1. Loads train/val/test splits from Step 4
2. Builds the PaperCNN model:
   - 3 Conv2D blocks (128 → 64 → 32 filters) with BatchNorm, MaxPool, Dropout
   - Dense(256) → Dense(7)
   - ~2.2M parameters
3. Trains with: AdamW optimizer, cosine annealing LR schedule with warmup, weighted cross-entropy with label smoothing, mixup augmentation, L1 regularization, gradient clipping
4. Early stops when validation stops improving
5. Evaluates on the held-out test set
6. Exports the best model to ONNX

**Output files:**

| File | Location | Description |
|------|----------|-------------|
| `fish_disease_classifier.onnx` | `backend/app/ml/` | Trained ONNX model (~8.4 MB) |
| `class_map.json` | `backend/app/ml/` | Class mapping for inference |
| `best_model.pt` | `backend/train/artifacts/` | PyTorch checkpoint |
| `metrics.json` | `backend/train/artifacts/` | Final accuracy & training config |
| `training_history.csv` | `backend/train/artifacts/` | Per-epoch train/val metrics |
| `confusion_matrix.csv` | `backend/train/artifacts/` | Test set confusion matrix |
| `classification_report.json` | `backend/train/artifacts/` | Per-class precision/recall/F1 |

**Expected training output:**

```
Using device: cuda (NVIDIA GeForce RTX 3060)
Model: PaperCNN | Parameters: 2,201,639

Epoch  1/80 ▸ train_loss=1.42  train_acc=48.2%  val_loss=0.89  val_acc=67.5%  lr=1.0e-04
Epoch  2/80 ▸ train_loss=0.81  train_acc=70.1%  val_loss=0.52  val_acc=82.3%  lr=3.0e-04
...
Early stopping at epoch 65
Test Accuracy: 79.4%
Model exported to backend/app/ml/fish_disease_classifier.onnx (8.40 MB)
```

> **Re-export only (no retraining):** If you have `best_model.pt` and just want to regenerate the ONNX file:
> ```bash
> python -m train.export_model
> ```

---

### Step 6 — Start the Backend Server

From the `backend` folder:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Windows without activating the venv:**

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**

```
2026-02-25 10:00:00 | INFO | Loading prediction service...
2026-02-25 10:00:01 | INFO | Model ready: True (provider: onnxruntime-cpu)
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to stop)
```

**Verify the backend — open in your browser:**

| URL | What You Should See |
|-----|---------------------|
| http://127.0.0.1:8000 | `"AquaScan Fish Disease Detection API is running."` |
| http://127.0.0.1:8000/health | `"model_ready": true` |
| http://127.0.0.1:8000/docs | Swagger UI — interactive API explorer |
| http://127.0.0.1:8000/diseases | JSON list of all 7 diseases |

> **Keep this terminal open.** The backend must stay running while you use the app.

---

### Step 7 — Smoke Test (Optional)

Open a **second terminal** while the backend is running. From the `backend` folder:

```bash
python tests/smoke_test.py --image-path "../Freshwater_Fish_Disease_Aquaculture_in_south_asia/Test/Bacterial Red disease/Bacterial Red disease (1).jpg"
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\python.exe tests\smoke_test.py --image-path "..\Freshwater_Fish_Disease_Aquaculture_in_south_asia\Test\Bacterial Red disease\Bacterial Red disease (1).jpg"
```

**Expected output:**

```json
{
  "prediction": {
    "name": "Bacterial Red Disease",
    "type": "Bacterial"
  },
  "confidence": 0.85,
  "source": "onnxruntime-cpu",
  "inference_ms": 12.3,
  "top_predictions": [...]
}
```

---

### Step 8 — Set Up the Flutter Frontend

Open a **new terminal** at the project root (`Aquaculture/` folder):

```bash
flutter pub get
```

Verify your environment:

```bash
flutter doctor
```

Make sure at least one platform shows a green checkmark.

---

### Step 9 — Configure Backend URL for Your Device

The Flutter app needs to know where the backend is running. The configuration is in `lib/services/api_prediction_service.dart`:

**For Android Emulator — no changes needed.** It uses `http://10.0.2.2:8000` automatically.

**For iOS Simulator (macOS) — no changes needed.** It uses `http://127.0.0.1:8000` automatically since the simulator shares the Mac's network.

**For a Physical Android Phone:**

1. Find your computer's local IP address:

   - **Windows:** `ipconfig` — look for "IPv4 Address" under Wi-Fi
   - **macOS:** `ipconfig getifaddr en0` (Wi-Fi) or check System Settings → Wi-Fi → Details → IP Address

2. Open `lib/services/api_prediction_service.dart` and update the `_lanIp` constant:

   ```dart
   static const String _lanIp = 'http://192.168.1.74:8000';  // ← Your computer's IP
   ```

3. Make sure your phone and computer are on the **same Wi-Fi network**.

**For a Physical iPhone (macOS):**

Same as above — update `_lanIp` to your Mac's IP. The phone must be on the same Wi-Fi.

**For Windows/Chrome/macOS desktop — no changes needed.** It uses `http://127.0.0.1:8000` automatically.

---

### Step 10 — Run the Flutter App

Make sure the backend is running (Step 6), then:

| Target | Command |
|--------|---------|
| **Physical Android phone** (USB) | `flutter run` → select your device |
| **Android Emulator** | `flutter run` → select emulator |
| **iOS Simulator** (macOS only) | `flutter run` → select simulator, or `open -a Simulator && flutter run` |
| **Physical iPhone** (macOS only) | `flutter run` → select your iPhone |
| **Chrome (web)** | `flutter run -d chrome` |
| **Windows desktop** | `flutter run -d windows` |
| **macOS desktop** | `flutter run -d macos` |

**For a physical Android phone:**

1. Enable **Developer Options** on your phone (Settings → About → tap Build Number 7 times)
2. Enable **USB Debugging** in Developer Options
3. Connect via USB and accept the debugging prompt
4. Run:

```bash
flutter run
```

5. Select your phone from the device list when prompted.

**For iOS Simulator (macOS):**

1. Install Xcode from the Mac App Store
2. Open Xcode at least once and accept the license
3. Install the iOS Simulator: `xcodebuild -downloadPlatform iOS`
4. Run:

```bash
open -a Simulator
flutter run
```

**For a physical iPhone (macOS):**

1. Install Xcode from the Mac App Store
2. Connect your iPhone via USB and trust the computer on the phone
3. In Xcode, go to **Settings → Accounts** and sign in with your Apple ID
4. Run:

```bash
flutter run
```

5. On first run, go to **Settings → General → VPN & Device Management** on the iPhone and trust your developer certificate.

---

### Step 11 — Scan a Fish!

Once the app launches on your device:

1. **Home tab** → Tap **"Take a Photo"** or **"Upload from Gallery"**
2. Point the camera at a fish (or select an image from the dataset for testing)
3. Wait for the AI analysis (~1–2 seconds)
4. View the result: disease name, confidence percentage, top-3 predictions, cause, symptoms, treatment, and prevention
5. If the image doesn't contain a recognizable fish, the app shows **"No Fish Detected"** with tips for better photos

**Other tabs:**

- **Diseases** — Browse all 7 detectable conditions with details
- **History** — View past scan results from the current session

---

## macOS Step-by-Step Guide

A complete copy-paste guide for setting up and running the project on macOS (Intel or Apple Silicon).

### Prerequisites (macOS)

Install these tools first:

**1. Homebrew** (if not already installed):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Python 3.11+:**

```bash
brew install python@3.13
```

Verify:

```bash
python3 --version
```

**3. Flutter SDK:**

```bash
brew install --cask flutter
```

Or follow the official guide: https://docs.flutter.dev/get-started/install/macos

Verify:

```bash
flutter doctor
```

**4. Xcode** (required for iOS development):

```bash
xcode-select --install
```

Then install Xcode from the [Mac App Store](https://apps.apple.com/us/app/xcode/id497799835). Open Xcode once and accept the license agreement.

**5. CocoaPods** (required for Flutter iOS builds):

```bash
brew install cocoapods
```

**6. Git:**

```bash
brew install git
```

**7. (Optional) Android Studio** — only needed if you want to run on an Android emulator or physical Android device:

```bash
brew install --cask android-studio
```

---

### macOS — Full Setup Commands

```bash
# ─── Step 1: Clone the repo ───
git clone git@github.com:ATU-TechAI-Research-Hub/Fish_Disease_Detection-Flutter.git
cd Fish_Disease_Detection-Flutter

# ─── Step 2: (Optional) Download the Kaggle dataset ───
# Download from: https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia
# Unzip into the project root so you have:
#   Freshwater_Fish_Disease_Aquaculture_in_south_asia/Train/
#   Freshwater_Fish_Disease_Aquaculture_in_south_asia/Test/

# ─── Step 3: Set up the Python backend ───
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS — Train the Model (Optional)

> Skip this if the ONNX model already exists in `backend/app/ml/`.

```bash
# Make sure you're in the backend folder with the venv activated

# Prepare dataset
python -m train.prepare_dataset

# Train the CNN model
# On Apple Silicon Macs, PyTorch automatically uses MPS (Metal) for GPU acceleration
python -m train.train_classifier

# Training takes ~15-40 minutes depending on your Mac
# Apple Silicon (M1/M2/M3/M4) is significantly faster than Intel Macs
```

### macOS — Start the Backend

```bash
# From the backend folder, with the venv activated:
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify it's running — open in your browser:

- http://127.0.0.1:8000 → Should show API info
- http://127.0.0.1:8000/health → Should show `"model_ready": true`

**Keep this terminal open.**

### macOS — Smoke Test (Optional)

Open a **second terminal**:

```bash
cd Fish_Disease_Detection-Flutter/backend
source .venv/bin/activate
python tests/smoke_test.py --image-path "../Freshwater_Fish_Disease_Aquaculture_in_south_asia/Test/Bacterial Red disease/Bacterial Red disease (1).jpg"
```

### macOS — Run the Flutter App

Open a **new terminal** at the project root:

```bash
cd Fish_Disease_Detection-Flutter
flutter pub get
```

**Option A — iOS Simulator (recommended for quick testing):**

```bash
open -a Simulator
flutter run
```

**Option B — Physical iPhone (USB):**

1. Connect your iPhone via USB, trust the computer on the phone
2. The backend URL auto-resolves for same-network devices. If your phone is on the same Wi-Fi, update `_lanIp` in `lib/services/api_prediction_service.dart`:

```bash
# Find your Mac's IP address:
ipconfig getifaddr en0
```

Then set that IP in the Dart file and run:

```bash
flutter run
```

On first run, go to **Settings → General → VPN & Device Management** on your iPhone and trust the developer certificate.

**Option C — macOS Desktop:**

```bash
flutter run -d macos
```

**Option D — Chrome:**

```bash
flutter run -d chrome
```

**Option E — Android Emulator:**

```bash
# Open Android Studio → Virtual Device Manager → Start an emulator
flutter run
```

---

### macOS — Quick-Start Summary (Copy-Paste)

```bash
# ─── Terminal 1: Backend ───
git clone git@github.com:ATU-TechAI-Research-Hub/Fish_Disease_Detection-Flutter.git
cd Fish_Disease_Detection-Flutter/backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# ─── Terminal 2: Flutter ───
cd Fish_Disease_Detection-Flutter
flutter pub get
open -a Simulator        # or skip this for Chrome/macOS desktop
flutter run              # or: flutter run -d chrome / flutter run -d macos
```

---

## Command Summary — Windows (Copy-Paste Reference)

### Full Setup (Training + Running)

```powershell
# Terminal 1 — Backend setup + training
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (Optional) GPU-accelerated PyTorch
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# Prepare dataset
python -m train.prepare_dataset

# Train model (takes 10-30 min depending on GPU)
python -m train.train_classifier

# Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```powershell
# Terminal 2 — Flutter app
cd ..
flutter pub get
flutter run
```

### Quick Start (Pre-trained Model Already Available)

```powershell
# Terminal 1 — Backend
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```powershell
# Terminal 2 — Flutter
flutter pub get
flutter run
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and available endpoints |
| `GET` | `/health` | Health check — model status and runtime provider |
| `GET` | `/diseases` | List all 7 disease classes |
| `POST` | `/predict` | Upload image (`file` field, max 15 MB) → prediction |
| `GET` | `/docs` | Swagger UI (interactive API documentation) |

**Example `curl` request:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@fish_image.jpg"
```

**Response:**

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
  "confidence": 0.85,
  "source": "onnxruntime-cpu",
  "filename": "fish_image.jpg",
  "inference_ms": 12.3,
  "top_predictions": [
    {"disease_id": 1, "disease_name": "Bacterial Red Disease", "confidence": 0.85},
    {"disease_id": 6, "disease_name": "Parasitic Disease", "confidence": 0.08},
    {"disease_id": 2, "disease_name": "Bacterial Aeromoniasis", "confidence": 0.04}
  ]
}
```

---

## Disease Classes

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

## Model Architecture

| Property | Value |
|----------|-------|
| Architecture | PaperCNN (Tamut et al., 2025) |
| Input | 150 × 150 × 3 (RGB, center-cropped) |
| Conv blocks | 3 — Conv2D(128, 5×5) → Conv2D(64, 3×3) → Conv2D(32, 3×3) |
| Each block | Conv → BatchNorm → ReLU → MaxPool(2×2) → Dropout |
| Classifier | Flatten → Dense(256) → ReLU → Dropout(0.5) → Dense(7) |
| Parameters | ~2.2M |
| ONNX size | ~8.4 MB |
| Optimizer | AdamW (lr=5e-4, weight_decay=1e-4) |
| Scheduler | Cosine annealing with 5-epoch warmup |
| Augmentation | HFlip, VFlip, Rotation, ColorJitter, Affine, Grayscale, RandomErasing, Mixup |
| Inference | ONNX Runtime (auto-selects CUDA if available, falls back to CPU) |

---

## Troubleshooting

### Backend won't start

- Make sure you activated the virtual environment
- Check port 8000 is free:
  - **Windows:** `netstat -ano | findstr :8000`
  - **macOS:** `lsof -i :8000`
- Verify ONNX model exists:
  - **Windows:** `dir backend\app\ml\fish_disease_classifier.onnx`
  - **macOS:** `ls -la backend/app/ml/fish_disease_classifier.onnx`
- Kill a process occupying the port:
  - **Windows:** `Stop-Process -Id <PID> -Force` (get PID from netstat)
  - **macOS:** `kill -9 <PID>` (get PID from lsof)

### `/health` shows `model_ready: false`

- The ONNX model or `class_map.json` is missing from `backend/app/ml/`
- Run Steps 4–5 to train and export the model

### Flutter app shows "Connection Failed"

- Make sure the backend is running (Step 6)
- **Android Emulator**: Uses `http://10.0.2.2:8000` — backend must be on the same machine
- **iOS Simulator**: Uses `http://127.0.0.1:8000` — backend must be on the same Mac
- **Physical phone (Android/iPhone)**: Update `_lanIp` in `lib/services/api_prediction_service.dart` to your computer's IP, and make sure both devices are on the same Wi-Fi
- Find your IP:
  - **Windows:** `ipconfig` → look for IPv4 under Wi-Fi
  - **macOS:** `ipconfig getifaddr en0`

### 400 Bad Request from `/predict`

- The image file may be empty, too large (>15 MB), or an unsupported format
- Supported formats: JPEG, PNG, WebP, GIF, BMP

### CUDA / GPU not detected during training

- **Windows (NVIDIA GPU):** `pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`
- **macOS (Apple Silicon):** MPS acceleration is built into PyTorch — no extra install needed. Verify with: `python3 -c "import torch; print(torch.backends.mps.is_available())"`
- **macOS (Intel):** No GPU acceleration available; training runs on CPU
- Training falls back to CPU automatically if no GPU is available

### macOS — `flutter run` fails with Xcode errors

- Make sure Xcode is installed and you've accepted the license: `sudo xcodebuild -license accept`
- Install Xcode command-line tools: `xcode-select --install`
- Install CocoaPods: `brew install cocoapods`
- From the project root, run: `cd ios && pod install && cd ..`
- If you get signing errors, open `ios/Runner.xcworkspace` in Xcode, go to **Signing & Capabilities**, and select a valid development team

### macOS — `python3: command not found`

- Install Python via Homebrew: `brew install python@3.13`
- Or download from https://www.python.org/downloads/macos/

### PowerShell `&&` operator error (Windows)

PowerShell versions before 7.0 don't support `&&`. Run commands one at a time, or use `;` as separator.

---

## References

1. Tamut, J., Mangang, Y.A., & Chingakham, C. (2025). "Image Classification of Freshwater Fish Diseases in South Asian Aquaculture Using Convolutional Neural Network." *Aquaculture Journal*, 5(1), 6. https://doi.org/10.3390/aquacj5010006

---

## License

This project was developed for academic research at Atlantic Technological University (ATU).
