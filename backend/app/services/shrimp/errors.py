"""Domain exceptions surfaced by the shrimp API."""

from __future__ import annotations


class ShrimpAnalysisError(RuntimeError):
    code = "shrimp_analysis_failed"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidShrimpImageError(ShrimpAnalysisError):
    code = "invalid_image"
    status_code = 400


class ShrimpModelUnavailableError(ShrimpAnalysisError):
    code = "model_unavailable"
    status_code = 503


class WaterIsolationError(ShrimpAnalysisError):
    code = "water_isolation_failed"
    status_code = 422


class ShrimpInferenceTimeoutError(ShrimpAnalysisError):
    code = "inference_timeout"
    status_code = 504
