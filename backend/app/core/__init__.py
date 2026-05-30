"""Shared inference core: preprocessing, labels, and model loading."""

from .fish_detector import (
    FishDetection,
    detect_fish,
    warmup as warmup_fish_gate,
)
from .labels import ClassEntry, LabelMap, load_label_map
from .model_loader import (
    BaseModel,
    KerasH5Model,
    ModelInfo,
    ModelLoadError,
    OnnxModel,
    load_model,
)
from .preprocessing import (
    PAPER_IMAGE_SIZE,
    PreprocessingError,
    is_low_quality,
    preprocess_batch,
    preprocess_image,
)

__all__ = [
    "BaseModel",
    "ClassEntry",
    "FishDetection",
    "detect_fish",
    "warmup_fish_gate",
    "KerasH5Model",
    "LabelMap",
    "ModelInfo",
    "ModelLoadError",
    "OnnxModel",
    "PAPER_IMAGE_SIZE",
    "PreprocessingError",
    "is_low_quality",
    "load_label_map",
    "load_model",
    "preprocess_batch",
    "preprocess_image",
]
