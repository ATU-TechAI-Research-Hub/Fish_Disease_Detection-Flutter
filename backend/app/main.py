"""FastAPI entrypoint for the AquaScan fish disease detection backend.

Loading order for the inference model (offline-first):

  1. `model/model.h5`            — Keras (.h5) primary, matches the paper.
  2. `backend/app/ml/model.h5`   — Same artifact bundled with the backend
                                   (kept for backward-compat when packaging).
  3. ONNX fallback inside `backend/app/ml/fish_disease_classifier.onnx`
     — preserved so the app keeps working even before `model.h5` exists.

No external network connections are made for inference — everything runs
against local files.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import Disease, ModelStatus, PredictionResponse
from app.services.prediction_service import PredictionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("aquascan")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISEASES_FILE = PROJECT_ROOT / "assets" / "diseases.json"
LABELS_FILE = PROJECT_ROOT / "model" / "labels.json"

H5_PRIMARY = PROJECT_ROOT / "model" / "model.h5"
H5_LEGACY = PROJECT_ROOT / "backend" / "app" / "ml" / "model.h5"
ONNX_FALLBACK = PROJECT_ROOT / "backend" / "app" / "ml" / "fish_disease_classifier.onnx"

# Optional backend preference override.
# Set AQUASCAN_MODEL_PREFERENCE=onnx to skip the .h5 and force ONNX.
# Set AQUASCAN_MODEL_PREFERENCE=h5 (default) to keep the documented behaviour.
MODEL_PREFERENCE = os.environ.get("AQUASCAN_MODEL_PREFERENCE", "h5").lower()


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to `default`."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r — using default %s", name, raw, default)
        return default

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}

prediction_service: Optional[PredictionService] = None


def _resolve_h5_path() -> Optional[Path]:
    """Resolve the active `model.h5` location (project-level wins).

    Returns `None` when the user explicitly forces the ONNX backend.
    """
    if MODEL_PREFERENCE == "onnx":
        logger.info("AQUASCAN_MODEL_PREFERENCE=onnx — skipping Keras .h5.")
        return None
    for candidate in (H5_PRIMARY, H5_LEGACY):
        if candidate.exists():
            return candidate
    return H5_PRIMARY


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global prediction_service
    logger.info("Booting AquaScan prediction service...")
    prediction_service = PredictionService(
        diseases_file=DISEASES_FILE,
        labels_file=LABELS_FILE,
        h5_model_path=_resolve_h5_path(),
        onnx_model_path=ONNX_FALLBACK,
        no_fish_threshold=_env_float("AQUASCAN_NO_FISH_THRESHOLD", 0.20),
        entropy_threshold=_env_float("AQUASCAN_ENTROPY_THRESHOLD", 1.90),
        high_confidence=_env_float("AQUASCAN_HIGH_CONFIDENCE", 0.70),
        medium_confidence=_env_float("AQUASCAN_MEDIUM_CONFIDENCE", 0.45),
        enable_fish_gate=os.environ.get(
            "AQUASCAN_ENABLE_FISH_GATE", "1"
        ).strip().lower() not in ("0", "false", "no"),
        fish_gate_threshold=_env_float("AQUASCAN_FISH_GATE_THRESHOLD", 0.05),
    )
    status = prediction_service.status()
    logger.info(
        "Inference ready=%s backend=%s device=%s (path=%s)",
        status.ready, status.backend, status.device, status.model_path,
    )
    yield
    logger.info("Shutting down AquaScan.")


app = FastAPI(
    title="AquaScan – Fish Disease Detection API",
    description=(
        "Local-first FastAPI backend implementing the freshwater fish disease "
        "CNN from Tamut et al., Aquac. J. 2025 (doi:10.3390/aquacj5010006). "
        "Inference runs against a Keras .h5 model (primary) with an ONNX "
        "fallback. No external API calls are made."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    if request.url.path not in ("/", "/health"):
        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method, request.url.path, response.status_code, elapsed,
        )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/", tags=["meta"])
async def root() -> Dict[str, Any]:
    return {
        "message": "AquaScan Fish Disease Detection API is running.",
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/health", "/model/info", "/diseases", "/predict"],
        "paper": "Tamut et al., Aquac. J. 2025 (doi:10.3390/aquacj5010006)",
    }


@app.get("/health", tags=["meta"])
async def health() -> Dict[str, Any]:
    svc = prediction_service
    if svc is None:
        return {"status": "starting", "model_ready": False}
    status = svc.status()
    return {
        "status": "ok" if status.ready else "degraded",
        "model_ready": status.ready,
        "backend": status.backend,
        "device": status.device,
        "version": app.version,
    }


@app.get("/model/info", response_model=ModelStatus, tags=["meta"])
async def model_info() -> ModelStatus:
    if prediction_service is None:
        raise HTTPException(503, "Service not ready.")
    return prediction_service.status()


@app.get("/diseases", response_model=List[Disease], tags=["data"])
async def get_diseases() -> List[Disease]:
    if prediction_service is None:
        raise HTTPException(503, "Service not ready.")
    return prediction_service.get_all_diseases()


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if prediction_service is None:
        raise HTTPException(503, "Service not ready.")

    contents = await file.read()
    await file.close()

    if not contents:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            400,
            f"File too large ({len(contents) // 1024} KB). "
            f"Max: {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )

    content_type = (file.content_type or "").lower()
    if (
        content_type
        and content_type != "application/octet-stream"
        and content_type not in ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(
            400,
            f"Unsupported image type: {content_type}. "
            "Accepted: JPEG, PNG, WebP, GIF, BMP.",
        )

    filename = file.filename or "uploaded_image"
    try:
        result = await prediction_service.predict(
            image_bytes=contents, filename=filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(
        "Prediction: %s (%.1f%%, tier=%s) for %s",
        result.prediction.name,
        result.confidence * 100,
        result.confidence_tier.value,
        filename,
    )
    return result
