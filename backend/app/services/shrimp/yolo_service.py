"""Ultralytics YOLO11 instance-segmentation adapter for shrimp masks."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .config import ShrimpConfig
from .errors import ShrimpAnalysisError, ShrimpModelUnavailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YoloShrimpInstance:
    confidence: float
    bounding_box: tuple[float, float, float, float]
    mask: np.ndarray


class YoloShrimpService:
    """Lazy-loads one validated, single-class shrimp segmentation checkpoint."""

    def __init__(self, config: ShrimpConfig) -> None:
        self._config = config
        self._model: Any = None
        self._state = (
            "not_loaded" if self._config.yolo_weights.is_file() else "missing_weights"
        )
        self._device = "unloaded"
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def status(self) -> dict[str, object]:
        if not self._config.yolo_weights.is_file() and self._state != "failed":
            self._state = "missing_weights"
        return {
            "ready": self._model is not None
            or (
                self._config.yolo_weights.is_file()
                and self._state not in {"failed", "missing_weights"}
            ),
            "state": self._state,
            "model": str(self._config.yolo_weights),
            "device": self._device,
            "error": self._load_error,
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self._config.yolo_weights.is_file():
                self._state = "missing_weights"
                raise ShrimpModelUnavailableError(
                    "The trained shrimp YOLO11 segmentation checkpoint was not "
                    f"found at {self._config.yolo_weights}. Set "
                    "AQUASCAN_SHRIMP_YOLO_WEIGHTS to a fine-tuned shrimp "
                    "YOLO11x-seg best.pt file."
                )
            if self._state == "failed":
                raise ShrimpModelUnavailableError(
                    self._load_error or "Shrimp YOLO11 model unavailable."
                )

            self._state = "loading"
            try:
                import torch
                from ultralytics import YOLO

                configured = self._config.device.lower()
                if configured == "auto":
                    device_arg: Any = 0 if torch.cuda.is_available() else "cpu"
                    self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
                else:
                    device_arg = self._config.device
                    self._device = self._config.device

                logger.info(
                    "Loading shrimp YOLO11 segmentation model from %s on %s...",
                    self._config.yolo_weights,
                    self._device,
                )
                model = YOLO(str(self._config.yolo_weights), task="segment")
                names_value = model.names
                if isinstance(names_value, dict):
                    class_names = [str(names_value[key]) for key in sorted(names_value)]
                else:
                    class_names = [str(name) for name in names_value]
                if len(class_names) != 1 or class_names[0].strip().lower() != "shrimp":
                    raise ValueError(
                        "Checkpoint class metadata must contain exactly one class "
                        f"named 'shrimp'; found {class_names!r}."
                    )

                self._device_arg = device_arg
                self._model = model
                self._state = "ready"
                self._load_error = None
            except ShrimpModelUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._state = "failed"
                self._load_error = (
                    "The shrimp YOLO11 instance-segmentation checkpoint could "
                    f"not be loaded: {exc}"
                )
                logger.exception("Shrimp YOLO11 initialization failed")
                raise ShrimpModelUnavailableError(self._load_error) from exc

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def predict(self, gated_rgb: np.ndarray) -> list[YoloShrimpInstance]:
        self._ensure_loaded()

        original_height, original_width = gated_rgb.shape[:2]
        size = self._config.yolo_image_size
        # The research export was trained with aspect-ratio-distorting stretch.
        # Supplying an already-square frame avoids Ultralytics letterboxing.
        stretched = cv2.resize(
            gated_rgb, (size, size), interpolation=cv2.INTER_LINEAR
        )

        with self._inference_lock:
            try:
                predictions = self._model.predict(
                    source=stretched,
                    conf=self._config.yolo_confidence_threshold,
                    iou=self._config.yolo_iou_threshold,
                    imgsz=size,
                    device=self._device_arg,
                    retina_masks=True,
                    max_det=self._config.yolo_max_detections,
                    augment=False,
                    verbose=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Shrimp YOLO11 inference failed")
                raise ShrimpAnalysisError(
                    "Shrimp segmentation failed during YOLO11 inference. "
                    "Please try again or inspect the server logs."
                ) from exc

        if not predictions:
            return []
        result = predictions[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return []

        masks = self._numpy(result.masks.data)
        boxes = self._numpy(result.boxes.xyxy)
        confidences = self._numpy(result.boxes.conf).reshape(-1)
        classes = self._numpy(result.boxes.cls).reshape(-1)

        scale_x = original_width / float(size)
        scale_y = original_height / float(size)
        retained: list[YoloShrimpInstance] = []
        for index in range(min(len(masks), len(boxes), len(confidences))):
            confidence = float(confidences[index])
            class_index = int(classes[index]) if index < len(classes) else 0
            if class_index != 0 or confidence < self._config.yolo_confidence_threshold:
                continue

            binary_square = np.squeeze(masks[index]) >= self._config.yolo_mask_threshold
            binary_original = cv2.resize(
                binary_square.astype(np.uint8),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if not binary_original.any():
                continue

            x1, y1, x2, y2 = [float(value) for value in boxes[index]]
            retained.append(
                YoloShrimpInstance(
                    confidence=confidence,
                    bounding_box=(
                        max(0.0, x1 * scale_x),
                        max(0.0, y1 * scale_y),
                        min(float(original_width), x2 * scale_x),
                        min(float(original_height), y2 * scale_y),
                    ),
                    mask=binary_original,
                )
            )
        return retained
