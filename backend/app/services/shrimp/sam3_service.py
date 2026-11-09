"""Lazy, reusable SAM 3 text-prompted water segmentation."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .config import ShrimpConfig
from .errors import ShrimpModelUnavailableError, WaterIsolationError
from .image_processing import combine_binary_masks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WaterSegmentation:
    mask: np.ndarray
    scores: list[float]


class Sam3WaterService:
    """Loads the official Transformers SAM 3 model once and reuses it."""

    def __init__(self, config: ShrimpConfig) -> None:
        self._config = config
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._dtype: Any = None
        self._device = "unloaded"
        self._state = "not_loaded"
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def status(self) -> dict[str, object]:
        state = self._state
        error = self._load_error
        ready = state != "failed"
        model_path = Path(self._config.sam_model_id).expanduser()
        if (
            self._model is None
            and not model_path.exists()
            and not self._config.sam_local_files_only
            and self._config.sam_model_id == "facebook/sam3"
        ):
            try:
                from huggingface_hub import get_token

                if get_token() is None:
                    ready = False
                    state = "authentication_required"
                    error = (
                        "SAM 3 requires accepted access to facebook/sam3 and a "
                        "Hugging Face login on the backend machine."
                    )
            except ImportError:
                ready = False
                state = "dependency_missing"
                error = "huggingface-hub is required to access the SAM 3 checkpoint."
        return {
            "ready": ready,
            "state": state,
            "model": self._config.sam_model_id,
            "device": self._device,
            "error": error,
        }

    def _resolve_device(self, torch: Any) -> str:
        configured = self._config.device.lower()
        if configured == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return self._config.device

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return
            if self._state == "failed":
                raise ShrimpModelUnavailableError(self._load_error or "SAM 3 unavailable.")

            self._state = "loading"
            try:
                import torch
                from transformers import Sam3Model, Sam3Processor

                self._device = self._resolve_device(torch)
                self._dtype = (
                    torch.float16 if self._device.startswith("cuda") else torch.float32
                )
                logger.info(
                    "Loading SAM 3 model %s on %s with %s weights...",
                    self._config.sam_model_id,
                    self._device,
                    self._dtype,
                )
                processor = Sam3Processor.from_pretrained(
                    self._config.sam_model_id,
                    local_files_only=self._config.sam_local_files_only,
                )
                model = Sam3Model.from_pretrained(
                    self._config.sam_model_id,
                    local_files_only=self._config.sam_local_files_only,
                    dtype=self._dtype,
                )
                model.to(self._device)
                model.eval()
                self._torch = torch
                self._processor = processor
                self._model = model
                self._state = "ready"
                self._load_error = None
            except Exception as exc:  # noqa: BLE001 - convert optional ML failures
                self._state = "failed"
                self._load_error = (
                    "SAM 3 could not be loaded. Request access to facebook/sam3 "
                    "on Hugging Face, authenticate with `hf auth login`, and "
                    "install a Transformers version that includes Sam3Model. "
                    f"Underlying error: {exc}"
                )
                logger.exception("SAM 3 initialization failed")
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

    def segment_water(self, rgb: np.ndarray) -> WaterSegmentation:
        self._ensure_loaded()
        assert self._torch is not None

        image = Image.fromarray(rgb, mode="RGB")
        with self._inference_lock:
            try:
                inputs = self._processor(
                    images=image,
                    text=self._config.sam_prompt,
                    return_tensors="pt",
                )
                original_sizes = inputs.get("original_sizes").tolist()
                inputs = inputs.to(self._device)
                for key, value in inputs.items():
                    if self._torch.is_floating_point(value):
                        inputs[key] = value.to(dtype=self._dtype)
                with self._torch.inference_mode():
                    outputs = self._model(**inputs)
                result = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self._config.sam_score_threshold,
                    mask_threshold=self._config.sam_mask_threshold,
                    target_sizes=original_sizes,
                )[0]
            except ShrimpModelUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("SAM 3 water segmentation failed")
                raise WaterIsolationError(
                    "Unable to identify the water region. Please capture a "
                    "clearer top-down image with better lighting."
                ) from exc

        masks_raw = result.get("masks")
        scores_raw = result.get("scores")
        if masks_raw is None or len(masks_raw) == 0:
            raise WaterIsolationError(
                "Unable to identify the water region. Please capture a clearer "
                "top-down image with better lighting."
            )

        masks = [self._numpy(mask) for mask in masks_raw]
        scores_array = (
            self._numpy(scores_raw).reshape(-1)
            if scores_raw is not None
            else np.ones(len(masks), dtype=np.float32)
        )
        valid_masks: list[np.ndarray] = []
        valid_scores: list[float] = []
        for index, mask in enumerate(masks):
            score = float(scores_array[index]) if index < len(scores_array) else 1.0
            if score < self._config.sam_score_threshold:
                continue
            binary = np.squeeze(mask).astype(bool)
            if binary.any():
                valid_masks.append(binary)
                valid_scores.append(score)

        water_mask = combine_binary_masks(valid_masks, rgb.shape[:2])
        area_ratio = float(water_mask.mean())
        if not valid_masks or area_ratio < self._config.sam_min_water_area_ratio:
            raise WaterIsolationError(
                "Unable to identify a usable water region. Please keep the "
                "entire white bowl visible and reduce glare before trying again."
            )

        return WaterSegmentation(
            mask=water_mask,
            scores=[round(score, 4) for score in valid_scores],
        )
