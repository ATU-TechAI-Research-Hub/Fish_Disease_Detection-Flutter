import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import Disease, PredictionResponse
from app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("aquascan")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISEASES_FILE = PROJECT_ROOT / "assets" / "diseases.json"
MODEL_FILE = PROJECT_ROOT / "backend" / "app" / "ml" / "fish_disease_classifier.onnx"
CLASS_MAP_FILE = PROJECT_ROOT / "backend" / "app" / "ml" / "class_map.json"

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}

prediction_service: PredictionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_service
    logger.info("Loading prediction service...")
    prediction_service = PredictionService(
        data_file=DISEASES_FILE,
        model_file=MODEL_FILE,
        class_map_file=CLASS_MAP_FILE,
    )
    logger.info("Model ready: %s (provider: %s)", prediction_service.model_ready, prediction_service.runtime_source)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AquaScan – Fish Disease Detection API",
    description="FastAPI backend powering AquaScan fish disease detection with ONNX inference.",
    version="2.0.0",
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
        logger.info("%s %s → %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
async def health() -> dict[str, object]:
    svc = prediction_service
    return {
        "status": "ok",
        "model_ready": svc.model_ready if svc else False,
        "source": svc.runtime_source if svc else "not-loaded",
        "version": "2.0.0",
    }


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "message": "AquaScan Fish Disease Detection API is running.",
        "docs": "/docs",
        "endpoints": ["/health", "/diseases", "/predict"],
    }


@app.get("/diseases", response_model=list[Disease])
async def get_diseases() -> list[Disease]:
    if not prediction_service:
        raise HTTPException(503, "Service not ready.")
    return prediction_service.get_all_diseases()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not prediction_service:
        raise HTTPException(503, "Service not ready.")

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported image type: {file.content_type}. Accepted: JPEG, PNG, WebP, GIF, BMP.")

    contents = await file.read()
    await file.close()

    if not contents:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File too large ({len(contents) // 1024} KB). Max: {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")

    filename = file.filename or "uploaded_image"
    try:
        result = await prediction_service.predict(image_bytes=contents, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info("Prediction: %s (%.1f%%) for %s", result.prediction.name, result.confidence * 100, filename)
    return result
