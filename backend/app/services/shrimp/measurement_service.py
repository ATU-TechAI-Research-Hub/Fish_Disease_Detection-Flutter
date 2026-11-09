"""OpenCV contour and ellipse-based shrimp length estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

Ellipse = tuple[
    tuple[float, float],
    tuple[float, float],
    float,
]


@dataclass(frozen=True)
class Measurement:
    length_px: Optional[float]
    ellipse: Optional[Ellipse]
    note: Optional[str] = None


def estimate_length(mask: np.ndarray) -> Measurement:
    """Return the major axis of an ellipse fit to the principal contour."""
    binary = np.asarray(mask).astype(bool).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return Measurement(None, None, "No contour could be extracted.")

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return Measurement(
            None,
            None,
            "The principal contour had fewer than five points; ellipse fit skipped.",
        )

    center, axes, angle = cv2.fitEllipse(contour)
    length_px = float(max(axes))
    ellipse: Ellipse = (
        (float(center[0]), float(center[1])),
        (float(axes[0]), float(axes[1])),
        float(angle),
    )
    return Measurement(length_px=length_px, ellipse=ellipse)


def calibration_scale(
    reference_length_mm: Optional[float],
    reference_length_px: Optional[float],
) -> Optional[float]:
    if reference_length_mm is None and reference_length_px is None:
        return None
    if reference_length_mm is None or reference_length_px is None:
        raise ValueError(
            "Calibration requires both reference_length_mm and reference_length_px."
        )
    if reference_length_mm <= 0 or reference_length_px <= 0:
        raise ValueError("Calibration measurements must be greater than zero.")
    return float(reference_length_mm / reference_length_px)
