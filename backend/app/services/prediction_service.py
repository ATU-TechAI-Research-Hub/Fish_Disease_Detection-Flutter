import io
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from app.models import ClassProbability, Disease, PredictionResponse

CONFIDENCE_THRESHOLD = 0.40
ENTROPY_THRESHOLD = 1.6
NO_FISH_DISEASE_ID = 0


class PredictionService:
    def __init__(
        self, data_file: Path, model_file: Path | None = None, class_map_file: Path | None = None
    ) -> None:
        self._data_file = data_file
        self._diseases = self._load_diseases()
        self._disease_by_id = {disease.id: disease for disease in self._diseases}
        self._model_file = model_file
        self._class_map_file = class_map_file
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._image_size = 150
        self._class_map: dict[int, dict[str, object]] = {}
        self._runtime_source = "model-not-loaded"
        if self._model_file and self._class_map_file:
            self._load_model()

    def _load_diseases(self) -> list[Disease]:
        if not self._data_file.exists():
            raise FileNotFoundError(f"Disease data file not found: {self._data_file}")

        raw_data = json.loads(self._data_file.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list) or not raw_data:
            raise ValueError("Disease data must be a non-empty list.")

        return [Disease.model_validate(item) for item in raw_data]

    def get_all_diseases(self) -> list[Disease]:
        return [d for d in self._diseases if d.id != NO_FISH_DISEASE_ID]

    @property
    def model_ready(self) -> bool:
        return self._session is not None and self._input_name is not None and bool(self._class_map)

    @property
    def runtime_source(self) -> str:
        return self._runtime_source

    def _load_model(self) -> None:
        if not self._model_file or not self._model_file.exists():
            return
        if not self._class_map_file or not self._class_map_file.exists():
            return

        with self._class_map_file.open("r", encoding="utf-8") as file_handle:
            class_map_payload = json.load(file_handle)

        self._image_size = int(class_map_payload.get("image_size", 150))
        classes = class_map_payload.get("classes", [])
        self._class_map = {
            int(class_info["class_index"]): class_info for class_info in classes
        }

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if self._has_cuda_runtime(available_providers):
            providers.insert(0, "CUDAExecutionProvider")
        self._session = ort.InferenceSession(str(self._model_file), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        active_provider = self._session.get_providers()[0]
        self._runtime_source = (
            "onnxruntime-cuda"
            if active_provider == "CUDAExecutionProvider"
            else "onnxruntime-cpu"
        )

    def _has_cuda_runtime(self, available_providers: list[str]) -> bool:
        if "CUDAExecutionProvider" not in available_providers:
            return False

        required_dlls = ("cublasLt64_12.dll", "cudnn64_9.dll")
        path_entries = [
            Path(entry)
            for entry in os.environ.get("PATH", "").split(os.pathsep)
            if entry
        ]
        return all(
            any((p / dll).exists() for p in path_entries)
            for dll in required_dlls
        )

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

        image = ImageOps.exif_transpose(image)

        w, h = image.size
        if w == 0 or h == 0:
            raise ValueError("Image has invalid dimensions.")

        short_side = min(w, h)
        left = (w - short_side) // 2
        top = (h - short_side) // 2
        image = image.crop((left, top, left + short_side, top + short_side))

        image = image.resize(
            (self._image_size, self._image_size), Image.LANCZOS
        )

        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return arr

    @staticmethod
    def _compute_entropy(probabilities: np.ndarray) -> float:
        clipped = np.clip(probabilities, 1e-10, 1.0)
        return float(-np.sum(clipped * np.log(clipped)))

    def _build_no_fish_response(
        self, filename: str, inference_ms: float, confidence: float,
        top_predictions: list[ClassProbability],
    ) -> PredictionResponse:
        no_fish = self._disease_by_id.get(NO_FISH_DISEASE_ID)
        if no_fish is None:
            no_fish = Disease(
                id=0, name="No Fish Detected", type="Unknown",
                cause="Image not recognized.", symptoms="N/A",
                treatment="Try a clearer fish photo.", prevention="N/A",
            )
        return PredictionResponse(
            prediction=no_fish,
            confidence=round(confidence, 4),
            source=self._runtime_source,
            filename=filename,
            inference_ms=round(inference_ms, 1),
            top_predictions=top_predictions,
        )

    async def predict(self, image_bytes: bytes, filename: str) -> PredictionResponse:
        if not self.model_ready:
            raise RuntimeError(
                "Model artifacts are missing. Train the classifier before using /predict."
            )

        assert self._session is not None
        assert self._input_name is not None

        t0 = time.perf_counter()
        input_tensor = self._preprocess_image(image_bytes)
        logits = self._session.run(None, {self._input_name: input_tensor})[0][0]
        inference_ms = (time.perf_counter() - t0) * 1000
        logits = np.asarray(logits, dtype=np.float32)
        exp_scores = np.exp(logits - np.max(logits))
        probabilities = exp_scores / np.sum(exp_scores)
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions: list[ClassProbability] = []
        for idx in top_indices:
            ci = self._class_map.get(int(idx))
            if ci is not None:
                top_predictions.append(
                    ClassProbability(
                        disease_id=int(ci["disease_id"]),
                        disease_name=str(ci["disease_name"]),
                        confidence=round(float(probabilities[idx]), 4),
                    )
                )

        entropy = self._compute_entropy(probabilities)
        if confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
            return self._build_no_fish_response(
                filename, inference_ms, confidence, top_predictions,
            )

        class_info = self._class_map.get(class_index)
        if class_info is None:
            raise RuntimeError(f"Class index {class_index} not found in class_map.json.")

        disease_id = int(class_info["disease_id"])
        disease = self._disease_by_id[disease_id]

        return PredictionResponse(
            prediction=disease,
            confidence=round(confidence, 4),
            source=self._runtime_source,
            filename=filename,
            inference_ms=round(inference_ms, 1),
            top_predictions=top_predictions,
        )
