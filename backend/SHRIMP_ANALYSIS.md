# Shrimp Counting & Analysis

AquaScan's shrimp module is independent of fish disease classification:

`RGB image → SAM 3 ("water") → hard water gate → YOLO11x-seg → masks → count + ellipse lengths`

It does not use the fish classifier and never substitutes a generic COCO model
for the required fine-tuned shrimp model.

## Required models

### 1. SAM 3

The default is the paper's `facebook/sam3` checkpoint through Hugging Face
Transformers (`Sam3Model` and `Sam3Processor`).

1. Request checkpoint access at <https://huggingface.co/facebook/sam3>.
2. Create a Hugging Face read token.
3. Authenticate on the backend machine:

   ```powershell
   .venv\Scripts\hf.exe auth login
   ```

4. Install `transformers>=5.8.0` and PyTorch using
   `backend/requirements.txt`.

The checkpoint is gated because accepting it shares the account information
listed by Meta. AquaScan cannot accept those terms for you. The status endpoint
reports `authentication_required` until the backend has a Hugging Face token.
The official checkpoint is approximately 3.44 GB. On CUDA AquaScan loads it in
FP16 to reduce VRAM use; CPU loading remains FP32 and is substantially slower.

The first analysis downloads/loads the model and can take substantially longer.
To require an already-cached/local checkpoint, set:

```powershell
$env:AQUASCAN_SAM3_LOCAL_FILES_ONLY = "1"
```

SAM 3.1 is newer, but AquaScan does not silently switch to it because the paper
used SAM 3. A compatible alternative model ID can be set explicitly with
`AQUASCAN_SAM3_MODEL` after validation.

### 2. Fine-tuned YOLO11x-seg shrimp checkpoint

Expected default path:

```text
backend/app/ml/shrimp/best.pt
```

Or set an external path:

```powershell
$env:AQUASCAN_SHRIMP_YOLO_WEIGHTS = "D:\models\shrimp-yolo11x-seg-best.pt"
```

The checkpoint is validated at load time. It must be an Ultralytics instance
segmentation model with exactly one class named `shrimp`. Generic
`yolo11x-seg.pt`, detection-only weights, and fish classifier weights are not
accepted as deployed shrimp inference weights.

Large model artifacts are Git-ignored.

## Install and run

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\hf.exe auth login
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check configuration without loading model weights:

```powershell
curl.exe http://127.0.0.1:8000/api/shrimp/status
```

CPU fallback is supported by the libraries but SAM 3 plus YOLO11x-seg can be
slow and memory-intensive. No real-time claim is made. A CUDA-capable GPU is
recommended.

The combined backend loads TensorFlow for fish classification and PyTorch for
shrimp analysis. On constrained GPU hosts, run shrimp inference in a separate
process or pin one framework to CPU to avoid competing VRAM allocations.

## Configuration

| Environment variable | Default | Purpose |
|---|---:|---|
| `AQUASCAN_SAM3_MODEL` | `facebook/sam3` | Transformers model ID or local path |
| `AQUASCAN_SAM3_PROMPT` | `water` | One-concept text prompt |
| `AQUASCAN_SAM3_SCORE_THRESHOLD` | `0.5` | SAM candidate confidence threshold |
| `AQUASCAN_SAM3_MASK_THRESHOLD` | `0.5` | SAM binary mask threshold |
| `AQUASCAN_SAM3_MIN_WATER_AREA_RATIO` | `0.01` | Reject implausibly tiny merged masks |
| `AQUASCAN_SAM3_LOCAL_FILES_ONLY` | `0` | Disable checkpoint downloads |
| `AQUASCAN_SHRIMP_YOLO_WEIGHTS` | `app/ml/shrimp/best.pt` | Fine-tuned checkpoint |
| `AQUASCAN_SHRIMP_CONFIDENCE_THRESHOLD` | `0.5` | Retained shrimp confidence |
| `AQUASCAN_SHRIMP_IOU_THRESHOLD` | `0.7` | YOLO NMS IoU |
| `AQUASCAN_SHRIMP_MASK_THRESHOLD` | `0.5` | YOLO mask binarization |
| `AQUASCAN_SHRIMP_IMAGE_SIZE` | `640` | Stretched square inference size |
| `AQUASCAN_SHRIMP_DEVICE` | `auto` | `auto`, `cpu`, `0`, or CUDA device |
| `AQUASCAN_SHRIMP_TIMEOUT_SECONDS` | `180` | API inference timeout |
| `AQUASCAN_SHRIMP_MAX_CONCURRENT` | `1` | Concurrent GPU analyses |
| `AQUASCAN_SHRIMP_ARTIFACT_TTL_SECONDS` | `3600` | Temporary image lifetime |

The paper states that thresholds were chosen on validation and then frozen, but
does not publish their numeric values. The defaults above are application
defaults and must be tuned on the validation split before reporting research
performance.

## API

### Analyze

`POST /api/shrimp/analyze`

Multipart fields:

- `file` (required): JPEG, PNG, WebP, GIF, or BMP, up to 15 MB.
- `calibration_reference_mm` (optional): real fiducial length.
- `calibration_reference_px` (optional): measured fiducial pixel length.

Both calibration values are required together. Millimeter results are omitted
without valid calibration.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/shrimp/analyze `
  -F "file=@D:\images\bowl.jpg" `
  -F "calibration_reference_mm=10" `
  -F "calibration_reference_px=125"
```

Example response shape:

```json
{
  "success": true,
  "analysis_id": "process-unique-id",
  "total_shrimp": 12,
  "average_length_px": 41.5,
  "minimum_length_px": 34.2,
  "maximum_length_px": 49.8,
  "average_confidence": 0.947,
  "processing_time": 2.4,
  "calibration_available": false,
  "scale_mm_per_px": null,
  "sam_water_scores": [0.93],
  "detections": [
    {
      "id": 1,
      "class_name": "shrimp",
      "confidence": 0.962,
      "length_px": 42.5,
      "length_mm": null,
      "bounding_box": {"x1": 100, "y1": 120, "x2": 180, "y2": 160},
      "mask_url": "/api/shrimp/artifacts/process-unique-id/mask_001.png",
      "measurement_note": null
    }
  ],
  "images": {
    "original_url": "/api/shrimp/artifacts/process-unique-id/original.png",
    "water_mask_url": "/api/shrimp/artifacts/process-unique-id/water_mask.png",
    "water_gated_url": "/api/shrimp/artifacts/process-unique-id/water_gated.png",
    "annotated_url": "/api/shrimp/artifacts/process-unique-id/annotated.png"
  },
  "warnings": []
}
```

The image fields are temporary URLs rather than base64 data. Artifacts are
deleted at process shutdown and expire after the configured TTL.

Important error codes include `invalid_image`, `water_isolation_failed`,
`model_unavailable`, `image_too_large`, and `inference_timeout`. No-water is a
terminal failure; YOLO is not run on the full image. A valid analysis with no
shrimp returns `total_shrimp: 0` and an explanatory warning.

## Training the missing checkpoint

The paper's public CC BY 4.0 Roboflow project is:

<https://universe.roboflow.com/yathish-8yqj0/shrimp-o6rvc-8j1pm-comdo/dataset/2>

Its 707 accessible source records can be prepared with
`train/shrimp/prepare_public_dataset.py`. The utility downloads source images,
validates polygon labels, normalizes every accidental source class ID to the
single deployment class `shrimp`, and records attribution in `manifest.json`.
The current public source split is 622 train / 63 validation / 22 test. This is
not the paper's reported processed split, so results from it must not be
presented as a reproduction of the paper metrics.

Alternatively, sign in to Roboflow and export the project as **YOLOv11
Segmentation** (polygon labels), not object detection. Expected structure:

```text
shrimp-dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

`data.yaml` must resolve train/validation/test paths and contain:

```yaml
nc: 1
names: [shrimp]
```

The paper's processed set contains 1,825 train, 72 validation, and 154 test
images (2,051 total from 707 source images). Its preprocessing was auto-orient
plus stretch to 640×640. Training augmentation produced three copies per
training image using horizontal/vertical flips, 90-degree rotations, ±10-degree
horizontal/vertical shear, and ±25% saturation jitter.

Split by complete capture session **before augmentation**. Near-duplicate
frames from one continuous bowl session must not cross splits.

Train from COCO segmentation initialization:

```powershell
cd backend
.venv\Scripts\python.exe -m train.shrimp.train_yolo11_seg `
  --data D:\datasets\shrimp\data.yaml `
  --base-weights yolo11x-seg.pt `
  --epochs 100 `
  --batch-size 8 `
  --device 0
```

The script validates the single `shrimp` class and copies the best checkpoint
to `backend/app/ml/shrimp/best.pt`. Epochs, batch size, confidence, and IoU
defaults are operational starting points; the paper does not disclose all of
these hyperparameters, so tune and freeze them on validation.

YOLO11x-seg at 640 pixels can train with batch size 1 on the tested 6 GB GTX
1660 Ti, but it spills beyond dedicated VRAM on dense frames and is slow.
Larger batches are not viable on that card. Use a larger GPU for the paper's
architecture, or explicitly document any switch to YOLO11m/s/n-seg as a
hardware-driven deviation. Do not label a smaller model as YOLO11x-seg.

## Coordinate and measurement behavior

The app auto-orients the upload and preserves the original RGB frame. The gated
image is stretched to 640×640 to match the paper's training geometry. YOLO
masks and boxes are mapped back to original image coordinates before
measurement and visualization. Count equals the number of retained instance
masks, not the number of raw boxes.

For each mask AquaScan extracts external contours, keeps the largest contour,
requires at least five points, fits `cv2.fitEllipse`, and reports the larger
ellipse axis. This is an approximate geometric proxy. Merged masks, curved
bodies, and partial occlusion can make it too long.

## Verification

```powershell
cd backend
.venv\Scripts\python.exe -m pytest tests\test_shrimp_analysis.py -v
.venv\Scripts\python.exe -m pytest tests -v
```

Unit/API tests use deterministic fake SAM and YOLO services; they verify image
decoding, mask OR/gating, ellipse length, calibration, mask-based counting,
artifact serving, and the no-fallback rule. Real-model integration requires
accepted SAM 3 access and the trained shrimp checkpoint.

## Scope and limitations

The reported paper metrics describe its fixed imaging-box protocol and are not
guarantees for AquaScan or arbitrary photos. The method targets 3–4 week shrimp
in a centered white bowl viewed straight down through a fixed lid hole.

- Touching shrimp can merge into one mask.
- Pale shrimp and dense overlaps can be missed.
- Glare can break water isolation.
- Under-segmented water can remove real shrimp.
- Over-segmented water can restore dry-bowl false positives.
- Results do not establish generalization to ponds, textured tanks, older
  shrimp, or unconstrained handheld views.
- The paper reports source-level split results; session-level leakage control
  and confidence intervals remain outstanding research requirements.

## Licensing

Ultralytics is distributed under AGPL-3.0 unless covered by a separate
enterprise license. SAM 3 code and checkpoints have Meta/Hugging Face access
and license terms. Review both before distributing AquaScan commercially or as
a hosted service.
