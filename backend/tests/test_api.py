"""Pytest suite for the AquaScan FastAPI backend.

Run from the `backend/` directory (with the venv active):

    python -m pytest tests/test_api.py -v

The suite covers:
  * request validation on /predict (empty file, bad content type, non-image)
  * /health and /model/info metadata endpoints
  * end-to-end prediction on a synthetic image (when a model is available)
  * determinism: identical input bytes → identical confidence
  * preprocessing unit checks (shape, value range, EXIF-safe decoding)
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# The heavyweight fish-presence gate (MobileNetV2) is not what we are testing
# here — disable it before the app module is imported.
os.environ.setdefault("AQUASCAN_ENABLE_FISH_GATE", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.preprocessing import (  # noqa: E402
    PreprocessingError,
    preprocess_image,
)
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # `with` runs the lifespan handler (loads the model once per module).
    with TestClient(app) as test_client:
        yield test_client


def _png_bytes(size: tuple[int, int] = (300, 200), color=(90, 140, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _model_ready(client: TestClient) -> bool:
    return bool(client.get("/health").json().get("model_ready"))


# ---------------------------------------------------------------------------
# Metadata endpoints
# ---------------------------------------------------------------------------

def test_root_lists_endpoints(client: TestClient):
    body = client.get("/").json()
    assert "/predict" in body["endpoints"]


def test_health_reports_status(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["model_ready"], bool)


def _load_labels() -> list[str]:
    """Canonical class names from model/labels.json (index order)."""
    labels_file = Path(__file__).resolve().parents[2] / "model" / "labels.json"
    with labels_file.open(encoding="utf-8") as fh:
        classes = json.load(fh)["classes"]
    ordered = sorted(classes, key=lambda c: c["class_index"])
    return [c["disease_name"] for c in ordered]


def test_model_info_matches_labels_file(client: TestClient):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_classes"] == len(_load_labels()) == 7


def test_diseases_catalog_matches_labels(client: TestClient):
    diseases = client.get("/diseases").json()
    names = {d["name"] for d in diseases}
    # Every model label must resolve to a catalogued disease entry.
    assert set(_load_labels()) <= names


# ---------------------------------------------------------------------------
# /predict request validation
# ---------------------------------------------------------------------------

def test_predict_requires_file(client: TestClient):
    assert client.post("/predict").status_code == 422


def test_predict_rejects_empty_file(client: TestClient):
    resp = client.post(
        "/predict", files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_predict_rejects_unsupported_content_type(client: TestClient):
    resp = client.post(
        "/predict", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


def test_predict_rejects_non_image_bytes(client: TestClient):
    if not _model_ready(client):
        pytest.skip("No model artifact available in this environment.")
    resp = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"this is not an image", "image/jpeg")},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /predict happy path + determinism
# ---------------------------------------------------------------------------

def test_predict_returns_full_response(client: TestClient):
    if not _model_ready(client):
        pytest.skip("No model artifact available in this environment.")

    resp = client.post(
        "/predict", files={"file": ("synthetic.png", _png_bytes(), "image/png")}
    )
    assert resp.status_code == 200
    body = resp.json()

    assert 0.0 <= body["confidence"] <= 1.0
    assert body["confidence_tier"] in ("high", "medium", "low")
    assert 1 <= len(body["top_predictions"]) <= 3
    # Top-k must be sorted by confidence, descending.
    confidences = [p["confidence"] for p in body["top_predictions"]]
    assert confidences == sorted(confidences, reverse=True)
    assert body["prediction"]["name"]


def test_predict_is_deterministic(client: TestClient):
    if not _model_ready(client):
        pytest.skip("No model artifact available in this environment.")

    image = _png_bytes(color=(30, 180, 90))
    results = [
        client.post(
            "/predict", files={"file": ("same.png", image, "image/png")}
        ).json()
        for _ in range(2)
    ]
    assert results[0]["prediction"]["id"] == results[1]["prediction"]["id"]
    assert results[0]["confidence"] == pytest.approx(
        results[1]["confidence"], abs=1e-6
    )


# ---------------------------------------------------------------------------
# Preprocessing unit checks (training/inference parity)
# ---------------------------------------------------------------------------

def test_preprocess_shape_and_range():
    tensor = preprocess_image(_png_bytes(size=(640, 480)))
    assert tensor.shape == (1, 150, 150, 3)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_handles_grayscale_and_alpha():
    for mode, color in (("L", 128), ("RGBA", (10, 20, 30, 128))):
        buf = io.BytesIO()
        Image.new(mode, (64, 64), color).save(buf, format="PNG")
        tensor = preprocess_image(buf.getvalue())
        assert tensor.shape == (1, 150, 150, 3)


def test_preprocess_rejects_garbage():
    with pytest.raises(PreprocessingError):
        preprocess_image(b"definitely not an image")
