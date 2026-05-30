"""Fish-presence gate (out-of-distribution detection).

The disease classifier is a 7-class CNN trained *only* on fish images, so it
will happily force any input (a person, a wall, food) into one of its disease
classes. To stop that, we run a general-purpose ImageNet classifier
(MobileNetV2, 1000 classes) first and ask a simpler question: "is there a fish
in this image at all?"

ImageNet contains many fish classes (goldfish, tench, sharks, rays, eel, coho,
gar, puffer, …). If the image puts meaningful probability mass on those
classes — or its single most-likely class is a fish — we let the disease
classifier run. Otherwise we short-circuit to "No Fish Detected".

The model loads lazily and fails *open*: if MobileNetV2 cannot be loaded for
any reason, the gate is skipped so the rest of the app keeps working.
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ImageNet (1000-class) indices that correspond to fish.
#   0 tench · 1 goldfish · 2 great white · 3 tiger shark · 4 hammerhead
#   5 electric ray · 6 stingray · 389 barracouta · 390 eel · 391 coho
#   392 rock beauty · 393 anemone fish · 394 sturgeon · 395 gar
#   396 lionfish · 397 puffer
FISH_CLASS_INDICES = frozenset(
    {0, 1, 2, 3, 4, 5, 6, 389, 390, 391, 392, 393, 394, 395, 396, 397}
)

# Minimum probability mass on fish classes for the image to count as a fish.
# Chance level is 1/1000; the diseased reference image scored ~0.34, while
# non-fish objects score ~0. 0.05 keeps real (even unusual) fish in.
DEFAULT_FISH_GATE_THRESHOLD = 0.05

_INPUT_SIZE = 224
_model = None
_load_failed = False
_lock = threading.Lock()


@dataclass(frozen=True)
class FishDetection:
    """Result of the fish-presence check."""

    is_fish: bool
    fish_score: float
    top_label: str
    top_prob: float
    available: bool = True


def _get_model():
    """Lazily load MobileNetV2 (ImageNet). Returns None if unavailable."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    with _lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

            _model = MobileNetV2(weights="imagenet")
            logger.info("Fish gate ready: MobileNetV2 (ImageNet) loaded.")
        except Exception as exc:  # noqa: BLE001 - fail open, never crash app
            _load_failed = True
            logger.warning(
                "Fish gate disabled (could not load MobileNetV2): %s", exc
            )
    return _model


def warmup() -> bool:
    """Eagerly load the gate model at startup. Returns True if loaded."""
    return _get_model() is not None


def _label_for_index(index: int) -> str:
    try:
        from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

        probe = np.zeros((1, 1000), dtype=np.float32)
        probe[0, index] = 1.0
        return decode_predictions(probe, top=1)[0][0][1]
    except Exception:  # noqa: BLE001
        return f"class_{index}"


def detect_fish(
    image_bytes: bytes,
    threshold: float = DEFAULT_FISH_GATE_THRESHOLD,
) -> FishDetection:
    """Decide whether `image_bytes` contains a fish.

    Fails open (``is_fish=True, available=False``) when the gate model or the
    image cannot be processed, so a gate failure never blocks a real scan.
    """
    model = _get_model()
    if model is None:
        return FishDetection(True, 0.0, "n/a", 0.0, available=False)

    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((_INPUT_SIZE, _INPUT_SIZE), Image.LANCZOS)
        arr = preprocess_input(np.asarray(img, dtype=np.float32)[None, ...])
        preds = model.predict(arr, verbose=0)[0]
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Fish gate skipped (image error): %s", exc)
        return FishDetection(True, 0.0, "n/a", 0.0, available=False)
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.warning("Fish gate skipped (inference error): %s", exc)
        return FishDetection(True, 0.0, "n/a", 0.0, available=False)

    fish_score = float(sum(preds[i] for i in FISH_CLASS_INDICES))
    top_idx = int(np.argmax(preds))
    top_prob = float(preds[top_idx])
    is_fish = top_idx in FISH_CLASS_INDICES or fish_score >= threshold
    top_label = _label_for_index(top_idx)

    return FishDetection(
        is_fish=is_fish,
        fish_score=round(fish_score, 4),
        top_label=top_label,
        top_prob=round(top_prob, 4),
        available=True,
    )
