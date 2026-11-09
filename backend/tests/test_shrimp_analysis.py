"""Unit and mocked API tests for the two-stage shrimp pipeline."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("AQUASCAN_ENABLE_FISH_GATE", "0")
os.environ.setdefault("AQUASCAN_ASSISTANT_FAKE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.shrimp.artifact_store import ShrimpArtifactStore  # noqa: E402
from app.services.shrimp.config import ShrimpConfig  # noqa: E402
from app.services.shrimp.errors import (  # noqa: E402
    ShrimpModelUnavailableError,
    WaterIsolationError,
)
from app.services.shrimp.image_processing import (  # noqa: E402
    apply_water_gate,
    combine_binary_masks,
    decode_image,
)
from app.services.shrimp.measurement_service import (  # noqa: E402
    calibration_scale,
    estimate_length,
)
from app.services.shrimp.router import set_shrimp_runtime_for_tests  # noqa: E402
from app.services.shrimp.sam3_service import WaterSegmentation  # noqa: E402
from app.services.shrimp.shrimp_analyzer import ShrimpAnalyzer  # noqa: E402
from app.services.shrimp.yolo_service import (  # noqa: E402
    YoloShrimpInstance,
    YoloShrimpService,
)


def _image_bytes(size: tuple[int, int] = (120, 80)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (190, 220, 230)).save(buffer, "PNG")
    return buffer.getvalue()


class FakeSam:
    def status(self) -> dict[str, object]:
        return {
            "ready": True,
            "state": "ready",
            "model": "fake-sam3",
            "device": "cpu",
            "error": None,
        }

    def segment_water(self, rgb: np.ndarray) -> WaterSegmentation:
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        mask[5:-5, 5:-5] = 1
        return WaterSegmentation(mask=mask, scores=[0.94])


class FakeYolo:
    called = False

    def status(self) -> dict[str, object]:
        return {
            "ready": True,
            "state": "ready",
            "model": "fake-yolo11x-seg",
            "device": "cpu",
            "error": None,
        }

    def predict(self, gated_rgb: np.ndarray) -> list[YoloShrimpInstance]:
        self.called = True
        mask = np.zeros(gated_rgb.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (55, 40), (24, 7), 15, 0, 360, 1, -1)
        return [
            YoloShrimpInstance(
                confidence=0.91,
                bounding_box=(28.0, 25.0, 82.0, 55.0),
                mask=mask.astype(bool),
            )
        ]


class FailingSam(FakeSam):
    def segment_water(self, rgb: np.ndarray) -> WaterSegmentation:
        raise WaterIsolationError("water failed")


class TrackingSam(FakeSam):
    called = False

    def segment_water(self, rgb: np.ndarray) -> WaterSegmentation:
        self.called = True
        return super().segment_water(rgb)


def _analyzer(
    *,
    sam: object | None = None,
    yolo: object | None = None,
) -> ShrimpAnalyzer:
    config = ShrimpConfig(
        yolo_weights=Path("missing-test-checkpoint.pt"),
        blur_variance_threshold=0.0,
    )
    return ShrimpAnalyzer(
        config=config,
        artifacts=ShrimpArtifactStore(ttl_seconds=300),
        sam_service=sam or FakeSam(),  # type: ignore[arg-type]
        yolo_service=yolo or FakeYolo(),  # type: ignore[arg-type]
    )


def test_decode_and_mask_gate_preserve_original_coordinates():
    decoded = decode_image(_image_bytes((12, 8)))
    assert decoded.rgb.shape == (8, 12, 3)
    left = np.zeros((8, 12), dtype=np.uint8)
    right = np.zeros((8, 12), dtype=np.uint8)
    left[:, :4] = 1
    right[:, 8:] = 1
    combined = combine_binary_masks([left, right], (8, 12))
    gated = apply_water_gate(decoded.rgb, combined)
    assert combined.sum() == 64
    assert np.all(gated[:, 4:8] == 0)
    assert np.all(gated[:, :4] != 0)


def test_ellipse_measurement_uses_major_axis():
    mask = np.zeros((140, 180), dtype=np.uint8)
    cv2.ellipse(mask, (90, 70), (42, 11), 25, 0, 360, 1, -1)
    measurement = estimate_length(mask)
    assert measurement.length_px == pytest.approx(84.0, abs=3.0)
    assert measurement.ellipse is not None


def test_calibration_requires_both_positive_measurements():
    assert calibration_scale(10.0, 50.0) == pytest.approx(0.2)
    assert calibration_scale(None, None) is None
    with pytest.raises(ValueError):
        calibration_scale(10.0, None)
    with pytest.raises(ValueError):
        calibration_scale(-1.0, 20.0)


def test_analyzer_counts_masks_and_writes_all_visualizations():
    analyzer = _analyzer()
    try:
        result = analyzer.analyze(
            _image_bytes(),
            reference_length_mm=10.0,
            reference_length_px=100.0,
        )
        assert result.total_shrimp == len(result.detections) == 1
        assert result.detections[0].length_px is not None
        assert result.detections[0].length_mm == pytest.approx(
            result.detections[0].length_px * 0.1, abs=0.01
        )
        for url in (
            result.images.original_url,
            result.images.water_mask_url,
            result.images.water_gated_url,
            result.images.annotated_url,
            result.detections[0].mask_url,
        ):
            _, _, analysis_id, filename = url.rsplit("/", 3)
            assert analyzer.artifacts.resolve(analysis_id, filename) is not None
    finally:
        analyzer.artifacts.close()


def test_water_failure_never_calls_yolo():
    yolo = FakeYolo()
    analyzer = _analyzer(sam=FailingSam(), yolo=yolo)
    try:
        with pytest.raises(WaterIsolationError):
            analyzer.analyze(_image_bytes())
        assert yolo.called is False
    finally:
        analyzer.artifacts.close()


def test_missing_yolo_checkpoint_fails_before_expensive_sam_inference():
    sam = TrackingSam()
    config = ShrimpConfig(
        yolo_weights=Path("definitely-missing-shrimp-checkpoint.pt"),
        blur_variance_threshold=0.0,
    )
    analyzer = ShrimpAnalyzer(
        config=config,
        artifacts=ShrimpArtifactStore(ttl_seconds=300),
        sam_service=sam,  # type: ignore[arg-type]
        yolo_service=YoloShrimpService(config),
    )
    try:
        with pytest.raises(ShrimpModelUnavailableError):
            analyzer.analyze(_image_bytes())
        assert sam.called is False
    finally:
        analyzer.artifacts.close()


def test_mocked_shrimp_api_and_artifact_serving():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        analyzer = _analyzer()
        set_shrimp_runtime_for_tests(analyzer)
        response = client.post(
            "/api/shrimp/analyze",
            files={"file": ("bowl.png", _image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["total_shrimp"] == 1
        assert payload["detections"][0]["class_name"] == "shrimp"

        image_response = client.get(payload["images"]["annotated_url"])
        assert image_response.status_code == 200
        assert image_response.headers["content-type"] == "image/png"
