"""Unified model loader supporting Keras `.h5` (primary) and ONNX (fallback).

Loading priority (so the application can transition smoothly without breaking
existing deployments):

  1. Keras `.h5`  ← primary, matches the paper exactly. Set via `H5 path`.
  2. ONNX `.onnx` ← optional legacy fallback for environments without TF.

Both backends expose the same interface:

    model = load_model(...)
    probabilities = model.predict(batch_tensor)   # shape (N, num_classes)
    info = model.info()                           # dict for `/health`

All inference happens locally — no network required.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when no usable model artifact can be loaded."""


@dataclass(frozen=True)
class ModelInfo:
    backend: str          # "keras-h5" or "onnxruntime"
    path: str
    num_classes: int
    image_size: int
    device: str           # "cpu" / "cuda" / "n/a"

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "path": self.path,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "device": self.device,
        }


class BaseModel(ABC):
    """Common interface for every supported inference backend."""

    @abstractmethod
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Run inference and return softmax probabilities of shape (N, K)."""

    @abstractmethod
    def info(self) -> ModelInfo:  # pragma: no cover - trivial
        ...


def _softmax_if_needed(raw: np.ndarray) -> np.ndarray:
    """Ensure outputs are valid probabilities (non-negative, sum to 1)."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[None, :]

    looks_normalized = (
        np.all(raw >= 0)
        and np.allclose(raw.sum(axis=1), 1.0, atol=1e-2)
    )
    if looks_normalized:
        return raw

    shifted = raw - raw.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class KerasH5Model(BaseModel):
    """Keras / TensorFlow `.h5` model (primary backend)."""

    def __init__(self, h5_path: Path, num_classes: int, image_size: int) -> None:
        try:
            from tensorflow import keras
        except ImportError as exc:  # pragma: no cover - explicit msg for users
            raise ModelLoadError(
                "TensorFlow is required to load .h5 models. "
                "Install it with: pip install tensorflow"
            ) from exc

        if not h5_path.exists():
            raise ModelLoadError(f"Keras model not found: {h5_path}")

        logger.info("Loading Keras .h5 model from %s ...", h5_path)
        self._model = keras.models.load_model(str(h5_path), compile=False)

        output_shape = self._model.output_shape
        if not isinstance(output_shape, tuple) or output_shape[-1] != num_classes:
            raise ModelLoadError(
                f"Model output shape {output_shape} does not match "
                f"labels.json ({num_classes} classes). Refusing to serve "
                "potentially mislabelled predictions."
            )

        self._h5_path = h5_path
        self._num_classes = num_classes
        self._image_size = image_size

        try:
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            self._device = "cuda" if any("GPU" in d.device_type for d in devices) else "cpu"
        except Exception:  # pragma: no cover - device discovery is best-effort
            self._device = "cpu"

    def predict(self, batch: np.ndarray) -> np.ndarray:
        raw = self._model.predict(batch, verbose=0)
        return _softmax_if_needed(raw)

    def info(self) -> ModelInfo:
        return ModelInfo(
            backend="keras-h5",
            path=str(self._h5_path),
            num_classes=self._num_classes,
            image_size=self._image_size,
            device=self._device,
        )


class OnnxModel(BaseModel):
    """ONNX Runtime backend (offline-only fallback / legacy support)."""

    def __init__(self, onnx_path: Path, num_classes: int, image_size: int) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - explicit msg
            raise ModelLoadError(
                "onnxruntime is required to load .onnx models. "
                "Install it with: pip install onnxruntime"
            ) from exc

        if not onnx_path.exists():
            raise ModelLoadError(f"ONNX model not found: {onnx_path}")

        logger.info("Loading ONNX fallback model from %s ...", onnx_path)
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        output_shape = self._session.get_outputs()[0].shape
        if (
            output_shape
            and isinstance(output_shape[-1], int)
            and output_shape[-1] != num_classes
        ):
            raise ModelLoadError(
                f"ONNX output shape {output_shape} does not match "
                f"labels.json ({num_classes} classes)."
            )
        self._onnx_path = onnx_path
        self._num_classes = num_classes
        self._image_size = image_size

    def _adapt_layout(self, batch: np.ndarray) -> np.ndarray:
        """Convert NHWC → NCHW if the ONNX model expects channels-first."""
        if batch.ndim != 4:
            return batch
        expected = self._input_shape
        if len(expected) == 4 and isinstance(expected[1], int) and expected[1] == 3:
            return np.transpose(batch, (0, 3, 1, 2))
        return batch

    def predict(self, batch: np.ndarray) -> np.ndarray:
        tensor = self._adapt_layout(batch.astype(np.float32))
        raw = self._session.run(None, {self._input_name: tensor})[0]
        return _softmax_if_needed(raw)

    def info(self) -> ModelInfo:
        return ModelInfo(
            backend="onnxruntime",
            path=str(self._onnx_path),
            num_classes=self._num_classes,
            image_size=self._image_size,
            device="cpu",
        )


def load_model(
    h5_path: Optional[Path],
    onnx_path: Optional[Path],
    num_classes: int,
    image_size: int = 150,
) -> BaseModel:
    """Load the first available backend: Keras (.h5) → ONNX fallback.

    Raises:
        ModelLoadError: If no usable artifact is found.
    """
    errors: list[str] = []

    if h5_path is not None and h5_path.exists():
        try:
            return KerasH5Model(h5_path, num_classes=num_classes, image_size=image_size)
        except ModelLoadError as exc:
            errors.append(f"Keras .h5: {exc}")
            logger.warning("Falling back from Keras .h5: %s", exc)

    if onnx_path is not None and onnx_path.exists():
        try:
            return OnnxModel(onnx_path, num_classes=num_classes, image_size=image_size)
        except ModelLoadError as exc:
            errors.append(f"ONNX: {exc}")
            logger.warning("ONNX fallback failed: %s", exc)

    h5_str = str(h5_path) if h5_path else "<not configured>"
    onnx_str = str(onnx_path) if onnx_path else "<not configured>"
    detail = "; ".join(errors) if errors else "no artifact files were found"
    raise ModelLoadError(
        "Could not load any inference backend. "
        f"Tried Keras: {h5_str}, ONNX: {onnx_str}. Detail: {detail}"
    )
