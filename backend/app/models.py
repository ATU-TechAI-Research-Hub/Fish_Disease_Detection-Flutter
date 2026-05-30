"""Pydantic request / response models for the AquaScan API."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ConfidenceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Disease(BaseModel):
    """Disease metadata (cause, symptoms, treatment, prevention)."""

    id: int
    name: str
    type: str
    cause: str
    symptoms: str
    treatment: str
    prevention: str


class ClassProbability(BaseModel):
    """Per-class probability used in the top-K list."""

    disease_id: int
    disease_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response for the `/predict` endpoint."""

    prediction: Disease
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_tier: ConfidenceTier
    source: str
    filename: str
    inference_ms: float = 0.0
    top_predictions: List[ClassProbability] = Field(default_factory=list)
    warning: Optional[str] = None
    recommendation: Optional[str] = None


class ModelStatus(BaseModel):
    """Detailed model status surfaced via `/health` and `/model/info`."""

    ready: bool
    backend: str
    model_path: str
    device: str
    num_classes: int
    image_size: int
    error: Optional[str] = None
