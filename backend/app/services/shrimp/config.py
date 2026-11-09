"""Environment-backed configuration for shrimp analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ShrimpConfig:
    """All model paths and operating thresholds used by the pipeline."""

    yolo_weights: Path
    sam_model_id: str = "facebook/sam3"
    sam_prompt: str = "water"
    sam_score_threshold: float = 0.5
    sam_mask_threshold: float = 0.5
    sam_min_water_area_ratio: float = 0.01
    sam_local_files_only: bool = False
    yolo_confidence_threshold: float = 0.5
    yolo_iou_threshold: float = 0.7
    yolo_mask_threshold: float = 0.5
    yolo_image_size: int = 640
    yolo_max_detections: int = 1000
    device: str = "auto"
    inference_timeout_seconds: int = 180
    max_upload_bytes: int = 15 * 1024 * 1024
    artifact_ttl_seconds: int = 60 * 60
    max_concurrent_analyses: int = 1
    blur_variance_threshold: float = 45.0

    @classmethod
    def from_env(cls, project_root: Path) -> "ShrimpConfig":
        default_weights = project_root / "backend" / "app" / "ml" / "shrimp" / "best.pt"
        configured_weights = os.getenv("AQUASCAN_SHRIMP_YOLO_WEIGHTS", "").strip()
        return cls(
            yolo_weights=(
                Path(configured_weights).expanduser()
                if configured_weights
                else default_weights
            ),
            sam_model_id=os.getenv("AQUASCAN_SAM3_MODEL", "facebook/sam3").strip(),
            sam_prompt=os.getenv("AQUASCAN_SAM3_PROMPT", "water").strip() or "water",
            sam_score_threshold=_env_float("AQUASCAN_SAM3_SCORE_THRESHOLD", 0.5),
            sam_mask_threshold=_env_float("AQUASCAN_SAM3_MASK_THRESHOLD", 0.5),
            sam_min_water_area_ratio=_env_float(
                "AQUASCAN_SAM3_MIN_WATER_AREA_RATIO", 0.01
            ),
            sam_local_files_only=_env_bool(
                "AQUASCAN_SAM3_LOCAL_FILES_ONLY", False
            ),
            yolo_confidence_threshold=_env_float(
                "AQUASCAN_SHRIMP_CONFIDENCE_THRESHOLD", 0.5
            ),
            yolo_iou_threshold=_env_float("AQUASCAN_SHRIMP_IOU_THRESHOLD", 0.7),
            yolo_mask_threshold=_env_float(
                "AQUASCAN_SHRIMP_MASK_THRESHOLD", 0.5
            ),
            yolo_image_size=_env_int("AQUASCAN_SHRIMP_IMAGE_SIZE", 640),
            yolo_max_detections=_env_int("AQUASCAN_SHRIMP_MAX_DETECTIONS", 1000),
            device=os.getenv("AQUASCAN_SHRIMP_DEVICE", "auto").strip() or "auto",
            inference_timeout_seconds=_env_int(
                "AQUASCAN_SHRIMP_TIMEOUT_SECONDS", 180
            ),
            max_upload_bytes=_env_int(
                "AQUASCAN_SHRIMP_MAX_UPLOAD_BYTES", 15 * 1024 * 1024
            ),
            artifact_ttl_seconds=_env_int(
                "AQUASCAN_SHRIMP_ARTIFACT_TTL_SECONDS", 60 * 60
            ),
            max_concurrent_analyses=max(
                1, _env_int("AQUASCAN_SHRIMP_MAX_CONCURRENT", 1)
            ),
            blur_variance_threshold=_env_float(
                "AQUASCAN_SHRIMP_BLUR_VARIANCE_THRESHOLD", 45.0
            ),
        )
