"""Typed API contracts for shrimp counting and analysis."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., ge=0)
    y1: float = Field(..., ge=0)
    x2: float = Field(..., ge=0)
    y2: float = Field(..., ge=0)


class ShrimpDetection(BaseModel):
    id: int = Field(..., ge=1)
    class_name: str = "shrimp"
    confidence: float = Field(..., ge=0.0, le=1.0)
    length_px: Optional[float] = Field(default=None, ge=0.0)
    length_mm: Optional[float] = Field(default=None, ge=0.0)
    bounding_box: BoundingBox
    mask_url: str
    measurement_note: Optional[str] = None


class ShrimpImages(BaseModel):
    original_url: str
    water_mask_url: str
    water_gated_url: str
    annotated_url: str


class ShrimpAnalysisResponse(BaseModel):
    success: bool = True
    analysis_id: str
    total_shrimp: int = Field(..., ge=0)
    average_length_px: Optional[float] = Field(default=None, ge=0.0)
    minimum_length_px: Optional[float] = Field(default=None, ge=0.0)
    maximum_length_px: Optional[float] = Field(default=None, ge=0.0)
    average_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    processing_time: float = Field(..., ge=0.0)
    calibration_available: bool
    scale_mm_per_px: Optional[float] = Field(default=None, gt=0.0)
    sam_water_scores: List[float] = Field(default_factory=list)
    detections: List[ShrimpDetection] = Field(default_factory=list)
    images: ShrimpImages
    warnings: List[str] = Field(default_factory=list)
    methodology_note: str = (
        "Ellipse major-axis length is an approximate geometric proxy, not a "
        "complete biological rostrum-to-telson measurement."
    )


class ComponentStatus(BaseModel):
    ready: bool
    state: str
    model: str
    device: str
    error: Optional[str] = None


class ShrimpModelStatus(BaseModel):
    ready: bool
    sam3: ComponentStatus
    yolo11: ComponentStatus
    prompt: str
    image_size: int
    message: Optional[str] = None
