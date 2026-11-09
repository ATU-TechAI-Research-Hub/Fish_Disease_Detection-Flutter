"""Orchestrates the research pipeline from upload to structured results."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .artifact_store import ShrimpArtifactStore
from .config import ShrimpConfig
from .errors import ShrimpModelUnavailableError
from .image_processing import (
    annotate_instances,
    apply_water_gate,
    blur_variance,
    decode_image,
    encode_png,
    render_water_mask,
)
from .measurement_service import Ellipse, calibration_scale, estimate_length
from .sam3_service import Sam3WaterService
from .schemas import (
    BoundingBox,
    ComponentStatus,
    ShrimpAnalysisResponse,
    ShrimpDetection,
    ShrimpImages,
    ShrimpModelStatus,
)
from .yolo_service import YoloShrimpService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyzedInstance:
    id: int
    confidence: float
    bounding_box: tuple[float, float, float, float]
    mask: np.ndarray
    length_px: Optional[float]
    length_mm: Optional[float]
    ellipse: Optional[Ellipse]
    measurement_note: Optional[str]


class ShrimpAnalyzer:
    """Runs SAM 3 first and YOLO11 only on the hard-gated RGB image."""

    def __init__(
        self,
        config: ShrimpConfig,
        artifacts: ShrimpArtifactStore,
        sam_service: Optional[Sam3WaterService] = None,
        yolo_service: Optional[YoloShrimpService] = None,
    ) -> None:
        self.config = config
        self.artifacts = artifacts
        self.sam = sam_service or Sam3WaterService(config)
        self.yolo = yolo_service or YoloShrimpService(config)

    def status(self) -> ShrimpModelStatus:
        sam_status = ComponentStatus.model_validate(self.sam.status())
        yolo_status = ComponentStatus.model_validate(self.yolo.status())
        ready = sam_status.ready and yolo_status.ready
        message = None
        if not sam_status.ready and sam_status.state == "authentication_required":
            message = (
                "Accept access to facebook/sam3 on Hugging Face, then run "
                "`hf auth login` on the backend machine."
            )
        elif not sam_status.ready:
            message = sam_status.error or "SAM 3 model setup is required."
        elif not yolo_status.ready and yolo_status.state == "missing_weights":
            message = (
                "Add the fine-tuned single-class shrimp YOLO11 segmentation "
                "checkpoint before running analysis."
            )
        elif sam_status.state == "not_loaded":
            message = "SAM 3 is configured and will load on the first analysis."
        return ShrimpModelStatus(
            ready=ready,
            sam3=sam_status,
            yolo11=yolo_status,
            prompt=self.config.sam_prompt,
            image_size=self.config.yolo_image_size,
            message=message,
        )

    def analyze(
        self,
        image_bytes: bytes,
        *,
        reference_length_mm: Optional[float] = None,
        reference_length_px: Optional[float] = None,
    ) -> ShrimpAnalysisResponse:
        started = time.perf_counter()
        decoded = decode_image(image_bytes)
        rgb = decoded.rgb
        logger.info("Shrimp analysis: decoded %dx%d RGB image", decoded.width, decoded.height)
        warnings: list[str] = []

        sharpness = blur_variance(rgb)
        if sharpness < self.config.blur_variance_threshold:
            warnings.append(
                "The image may be blurry. Retake it with the phone steady and "
                "the shrimp in focus if masks appear incomplete."
            )

        scale = calibration_scale(reference_length_mm, reference_length_px)

        yolo_status = self.yolo.status()
        if not bool(yolo_status.get("ready")):
            raise ShrimpModelUnavailableError(
                str(yolo_status.get("error") or self.status().message)
            )

        # Stage 1: text-prompted water isolation. Failure is terminal by design;
        # the research method explicitly forbids a silent full-frame fallback.
        logger.info("Shrimp analysis: isolating water with SAM 3")
        water = self.sam.segment_water(rgb)
        gated_rgb = apply_water_gate(rgb, water.mask)

        # Stage 2: every returned item is an individual segmentation mask.
        logger.info("Shrimp analysis: running YOLO11 on the water-gated image")
        raw_instances = self.yolo.predict(gated_rgb)
        analyzed: list[AnalyzedInstance] = []
        for instance_id, instance in enumerate(raw_instances, start=1):
            measurement = estimate_length(instance.mask)
            length_px = measurement.length_px
            length_mm = (
                float(length_px * scale)
                if length_px is not None and scale is not None
                else None
            )
            analyzed.append(
                AnalyzedInstance(
                    id=instance_id,
                    confidence=float(instance.confidence),
                    bounding_box=instance.bounding_box,
                    mask=instance.mask,
                    length_px=length_px,
                    length_mm=length_mm,
                    ellipse=measurement.ellipse,
                    measurement_note=measurement.note,
                )
            )

        if not analyzed:
            warnings.append(
                "No shrimp were detected. Please check the image quality, "
                "ensure the shrimp are visible inside the water, and try again."
            )

        annotated = annotate_instances(rgb, analyzed)
        artifacts_payload: dict[str, bytes] = {
            "original": encode_png(rgb),
            "water_mask": encode_png(render_water_mask(water.mask)),
            "water_gated": encode_png(gated_rgb),
            "annotated": encode_png(annotated),
        }
        for instance in analyzed:
            artifacts_payload[f"mask_{instance.id:03d}"] = encode_png(
                instance.mask.astype(np.uint8) * 255
            )

        analysis_id, urls = self.artifacts.save(artifacts_payload)
        detections = [
            ShrimpDetection(
                id=instance.id,
                confidence=round(instance.confidence, 4),
                length_px=(
                    round(instance.length_px, 2)
                    if instance.length_px is not None
                    else None
                ),
                length_mm=(
                    round(instance.length_mm, 2)
                    if instance.length_mm is not None
                    else None
                ),
                bounding_box=BoundingBox(
                    x1=round(instance.bounding_box[0], 2),
                    y1=round(instance.bounding_box[1], 2),
                    x2=round(instance.bounding_box[2], 2),
                    y2=round(instance.bounding_box[3], 2),
                ),
                mask_url=urls[f"mask_{instance.id:03d}"],
                measurement_note=instance.measurement_note,
            )
            for instance in analyzed
        ]

        lengths = [
            instance.length_px
            for instance in analyzed
            if instance.length_px is not None
        ]
        confidences = [instance.confidence for instance in analyzed]
        elapsed = time.perf_counter() - started
        logger.info(
            "Shrimp analysis complete: %d masks in %.3fs",
            len(analyzed),
            elapsed,
        )
        return ShrimpAnalysisResponse(
            analysis_id=analysis_id,
            total_shrimp=len(analyzed),
            average_length_px=round(float(np.mean(lengths)), 2) if lengths else None,
            minimum_length_px=round(float(np.min(lengths)), 2) if lengths else None,
            maximum_length_px=round(float(np.max(lengths)), 2) if lengths else None,
            average_confidence=(
                round(float(np.mean(confidences)), 4) if confidences else None
            ),
            processing_time=round(elapsed, 3),
            calibration_available=scale is not None,
            scale_mm_per_px=round(scale, 6) if scale is not None else None,
            sam_water_scores=water.scores,
            detections=detections,
            images=ShrimpImages(
                original_url=urls["original"],
                water_mask_url=urls["water_mask"],
                water_gated_url=urls["water_gated"],
                annotated_url=urls["annotated"],
            ),
            warnings=warnings,
        )
