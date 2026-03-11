from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models import Disease, PredictionResponse
from app.services.prediction_service import PredictionService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISEASES_FILE = PROJECT_ROOT / "assets" / "diseases.json"
MODEL_FILE = PROJECT_ROOT / "backend" / "app" / "ml" / "fish_disease_classifier.onnx"
CLASS_MAP_FILE = PROJECT_ROOT / "backend" / "app" / "ml" / "class_map.json"

prediction_service = PredictionService(
    data_file=DISEASES_FILE,
    model_file=MODEL_FILE,
    class_map_file=CLASS_MAP_FILE,
)

app = FastAPI(
    title="Freshwater Fish Disease Backend",
    description="FastAPI backend for fish disease prediction prototype.",
    version="1.0.0",
)

# Allow local Flutter development clients.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_ready": prediction_service.model_ready,
        "source": prediction_service.runtime_source,
    }


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "message": "Freshwater Fish Disease Backend is running.",
        "docs": "/docs",
        "endpoints": ["/health", "/diseases", "/predict"],
    }


@app.get("/diseases", response_model=list[Disease])
async def get_diseases() -> list[Disease]:
    return prediction_service.get_all_diseases()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    contents = await file.read()
    if not contents:
        await file.close()
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "uploaded_image"
    try:
        result = await prediction_service.predict(image_bytes=contents, filename=filename)
    except ValueError as exc:
        await file.close()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        await file.close()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    await file.close()
    return result
