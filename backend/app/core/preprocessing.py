"""Image preprocessing pipeline that matches the research paper exactly.

Tamut et al., Aquac. J. 2025, 5(1), 6 describe:
  1. Resize each image to 150x150 pixels.
  2. Normalize pixel values to [0, 1] by dividing by 255.

This module is the *single* source of truth for image preprocessing. It is used
by training (Keras `ImageDataGenerator`-equivalent), evaluation, and inference,
so the same pipeline is applied at every stage. Mismatched preprocessing
between training and inference was identified as the single largest accuracy
bug in earlier iterations of this project, so we keep it centralised here.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

PAPER_IMAGE_SIZE: int = 150


class PreprocessingError(ValueError):
    """Raised when an image cannot be decoded or is unusable for inference."""


def _open_image(source: bytes | str | Path | Image.Image) -> Image.Image:
    """Open an image from bytes, a path, or an existing PIL Image."""
    if isinstance(source, Image.Image):
        return source.copy()

    try:
        if isinstance(source, (str, Path)):
            return Image.open(str(source))
        if isinstance(source, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(bytes(source)))
    except (UnidentifiedImageError, OSError) as exc:
        raise PreprocessingError(
            "The uploaded file is not a valid image. "
            "Please upload a JPEG, PNG, WebP, GIF, or BMP file."
        ) from exc

    raise PreprocessingError(
        f"Unsupported image source type: {type(source).__name__}."
    )


def preprocess_image(
    source: bytes | str | Path | Image.Image,
    image_size: int = PAPER_IMAGE_SIZE,
) -> np.ndarray:
    """Convert any input image to the exact tensor the CNN expects.

    Steps (matches paper):
      1. Decode + convert to RGB (drops alpha, supports paletted images).
      2. Auto-rotate based on EXIF (so phone photos taken in portrait mode
         are not interpreted upside-down).
      3. Resize to (image_size, image_size) using high quality LANCZOS.
      4. Normalize to float32 in [0, 1] by dividing by 255.
      5. Add the batch dimension: shape becomes (1, H, W, 3).

    Args:
        source: Raw image bytes, a file path, or an open PIL Image.
        image_size: Side length of the square network input (paper: 150).

    Returns:
        A NumPy array of shape (1, image_size, image_size, 3), dtype float32,
        with values in the range [0, 1].

    Raises:
        PreprocessingError: If the image cannot be decoded or has zero size.
    """
    if image_size <= 0:
        raise PreprocessingError(
            f"image_size must be positive, received {image_size}."
        )

    image = _open_image(source)
    image = ImageOps.exif_transpose(image).convert("RGB")

    if image.size[0] == 0 or image.size[1] == 0:
        raise PreprocessingError("The uploaded image has invalid dimensions.")

    image = image.resize((image_size, image_size), Image.LANCZOS)

    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def preprocess_batch(
    sources: list[bytes | str | Path | Image.Image],
    image_size: int = PAPER_IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess a list of images into a single (N, H, W, 3) batch."""
    if not sources:
        return np.empty((0, image_size, image_size, 3), dtype=np.float32)

    tensors = [preprocess_image(s, image_size=image_size)[0] for s in sources]
    return np.stack(tensors, axis=0).astype(np.float32)


def is_low_quality(
    source: bytes | str | Path | Image.Image,
    min_side: int = 64,
) -> Tuple[bool, str]:
    """Heuristic check for unusably small or empty images.

    Returns (is_low_quality, reason). This is used to warn users before
    running inference on clearly unusable images.
    """
    try:
        image = _open_image(source).convert("RGB")
    except PreprocessingError as exc:
        return True, str(exc)

    w, h = image.size
    if w < min_side or h < min_side:
        return True, (
            f"Image is too small ({w}x{h}). "
            f"Please use a photo at least {min_side}x{min_side} pixels."
        )
    return False, ""
