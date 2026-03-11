import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.models import Disease, PredictionResponse


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
        self._image_size = 224
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
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
        return self._diseases

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

        self._image_size = int(class_map_payload.get("image_size", 224))
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
        return all(any((path_entry / dll_name).exists() for path_entry in path_entries) for dll_name in required_dlls)

    def _resize_and_crop(self, image: Image.Image) -> Image.Image:
        resize_size = int(round(self._image_size / 0.875))
        width, height = image.size
        if width == 0 or height == 0:
            raise ValueError("Image has invalid dimensions.")

        if width < height:
            new_width = resize_size
            new_height = int(height * resize_size / width)
        else:
            new_height = resize_size
            new_width = int(width * resize_size / height)

        resized = image.resize((new_width, new_height), Image.BILINEAR)
        left = max((new_width - self._image_size) // 2, 0)
        top = max((new_height - self._image_size) // 2, 0)
        right = left + self._image_size
        bottom = top + self._image_size
        return resized.crop((left, top, right, bottom))

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

        cropped = self._resize_and_crop(image)
        image_array = np.asarray(cropped, dtype=np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))[None, ...]
        normalized = (image_array - self._mean) / self._std
        return normalized.astype(np.float32)

    async def predict(self, image_bytes: bytes, filename: str) -> PredictionResponse:
        if not self.model_ready:
            raise RuntimeError(
                "Model artifacts are missing. Train the classifier before using /predict."
            )

        assert self._session is not None
        assert self._input_name is not None

        input_tensor = self._preprocess_image(image_bytes)
        logits = self._session.run(None, {self._input_name: input_tensor})[0][0]
        logits = np.asarray(logits, dtype=np.float32)
        exp_scores = np.exp(logits - np.max(logits))
        probabilities = exp_scores / np.sum(exp_scores)
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])

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
        )
