"""Image decoding, gating, quality checks, and visualization helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import InvalidShrimpImageError

# Some phones store multi-picture JPEGs that Pillow identifies as MPO even
# when the uploaded filename/content type is ordinary .JPG. Frame zero is the
# full-resolution still and is safe to process like JPEG.
ALLOWED_IMAGE_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "GIF", "BMP"}


@dataclass(frozen=True)
class DecodedImage:
    rgb: np.ndarray
    image_format: str

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def decode_image(image_bytes: bytes) -> DecodedImage:
    """Decode bytes, apply EXIF orientation, select frame zero, and return RGB."""
    if not image_bytes:
        raise InvalidShrimpImageError("Uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image_format = (opened.format or "").upper()
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise InvalidShrimpImageError(
                    "Unsupported image format. Use JPEG, PNG, WebP, GIF, or BMP."
                )
            opened.seek(0)
            oriented = ImageOps.exif_transpose(opened)
            rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8).copy()
    except InvalidShrimpImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidShrimpImageError(
            "The uploaded file is not a readable image or is corrupted."
        ) from exc

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise InvalidShrimpImageError("The uploaded image has invalid dimensions.")
    return DecodedImage(rgb=rgb, image_format=image_format)


def combine_binary_masks(
    masks: Iterable[np.ndarray], expected_shape: tuple[int, int]
) -> np.ndarray:
    """Combine valid same-sized masks using logical OR."""
    combined = np.zeros(expected_shape, dtype=bool)
    found = False
    for mask in masks:
        array = np.asarray(mask)
        array = np.squeeze(array)
        if array.shape != expected_shape:
            array = cv2.resize(
                array.astype(np.float32),
                (expected_shape[1], expected_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        combined |= array.astype(bool)
        found = True
    if not found:
        return np.zeros(expected_shape, dtype=np.uint8)
    return combined.astype(np.uint8)


def apply_water_gate(rgb: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    if water_mask.shape != rgb.shape[:2]:
        raise ValueError("Water mask dimensions do not match the source image.")
    return np.where(water_mask[..., None].astype(bool), rgb, 0).astype(np.uint8)


def blur_variance(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def encode_png(rgb_or_gray: np.ndarray) -> bytes:
    array = np.asarray(rgb_or_gray)
    if array.ndim == 2:
        encoded_source = array
    elif array.ndim == 3 and array.shape[2] == 3:
        encoded_source = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Only grayscale or RGB arrays can be encoded.")
    ok, encoded = cv2.imencode(".png", encoded_source)
    if not ok:
        raise RuntimeError("OpenCV could not encode the result image.")
    return encoded.tobytes()


def render_water_mask(water_mask: np.ndarray) -> np.ndarray:
    return (water_mask.astype(bool).astype(np.uint8) * 255)


def _instance_color(instance_id: int) -> tuple[int, int, int]:
    hue = int((instance_id * 47) % 180)
    hsv = np.uint8([[[hue, 190, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def annotate_instances(
    original_rgb: np.ndarray,
    instances: Iterable[object],
) -> np.ndarray:
    """Overlay unique mask colors, contours, IDs, confidence, and major axes."""
    annotated = original_rgb.copy()
    overlay = original_rgb.copy()
    items = list(instances)

    for instance in items:
        color = _instance_color(int(instance.id))
        mask = np.asarray(instance.mask).astype(bool)
        overlay[mask] = color
    annotated = cv2.addWeighted(overlay, 0.42, annotated, 0.58, 0)

    for instance in items:
        instance_id = int(instance.id)
        color_rgb = _instance_color(instance_id)
        mask_u8 = np.asarray(instance.mask).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, color_rgb, 2)

        if instance.ellipse is not None:
            center, axes, angle = instance.ellipse
            ellipse_for_cv = (
                (float(center[0]), float(center[1])),
                (float(axes[0]), float(axes[1])),
                float(angle),
            )
            cv2.ellipse(annotated, ellipse_for_cv, color_rgb, 2)

        x1, y1, x2, y2 = [int(round(v)) for v in instance.bounding_box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 2)
        label = f"#{instance_id} {float(instance.confidence) * 100:.1f}%"
        text_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_y = max(text_size[1] + 8, y1)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_size[1] - 8),
            (x1 + text_size[0] + 8, label_y + 3),
            color_rgb,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated
