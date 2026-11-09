"""FastAPI routes for the independent shrimp analysis capability."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .artifact_store import ShrimpArtifactStore
from .config import ShrimpConfig
from .errors import (
    ShrimpAnalysisError,
    ShrimpInferenceTimeoutError,
)
from .schemas import ShrimpAnalysisResponse, ShrimpModelStatus
from .shrimp_analyzer import ShrimpAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shrimp", tags=["shrimp"])

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "application/octet-stream",
}

_analyzer: Optional[ShrimpAnalyzer] = None
_semaphore: Optional[asyncio.Semaphore] = None


def initialize_shrimp_runtime(project_root: Path) -> ShrimpAnalyzer:
    global _analyzer, _semaphore
    config = ShrimpConfig.from_env(project_root)
    artifacts = ShrimpArtifactStore(config.artifact_ttl_seconds)
    _analyzer = ShrimpAnalyzer(config=config, artifacts=artifacts)
    _semaphore = asyncio.Semaphore(config.max_concurrent_analyses)
    logger.info(
        "Shrimp pipeline configured (SAM=%s, YOLO=%s, device=%s)",
        config.sam_model_id,
        config.yolo_weights,
        config.device,
    )
    return _analyzer


def shutdown_shrimp_runtime() -> None:
    global _analyzer, _semaphore
    if _analyzer is not None:
        _analyzer.artifacts.close()
    _analyzer = None
    _semaphore = None


def set_shrimp_runtime_for_tests(analyzer: ShrimpAnalyzer) -> None:
    """Replace the runtime without loading real model weights in API tests."""
    global _analyzer, _semaphore
    _analyzer = analyzer
    _semaphore = asyncio.Semaphore(1)


def _runtime() -> tuple[ShrimpAnalyzer, asyncio.Semaphore]:
    if _analyzer is None or _semaphore is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "service_starting",
                "message": "Shrimp analysis service is not initialized.",
            },
        )
    return _analyzer, _semaphore


@router.get("/status", response_model=ShrimpModelStatus)
async def shrimp_status() -> ShrimpModelStatus:
    analyzer, _ = _runtime()
    return analyzer.status()


@router.get("/artifacts/{analysis_id}/{filename}", response_class=FileResponse)
async def shrimp_artifact(analysis_id: str, filename: str) -> FileResponse:
    analyzer, _ = _runtime()
    path = analyzer.artifacts.resolve(analysis_id, filename)
    if path is None:
        raise HTTPException(404, "Result image was not found or has expired.")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "private, no-store, max-age=0"},
    )


@router.post("/analyze", response_model=ShrimpAnalysisResponse)
async def analyze_shrimp(
    file: UploadFile = File(...),
    calibration_reference_mm: Optional[float] = Form(default=None),
    calibration_reference_px: Optional[float] = Form(default=None),
) -> ShrimpAnalysisResponse:
    analyzer, semaphore = _runtime()
    content_type = (file.content_type or "application/octet-stream").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unsupported_image_type",
                "message": (
                    "Unsupported image type. Use JPEG, PNG, WebP, GIF, or BMP."
                ),
            },
        )

    contents = await file.read(analyzer.config.max_upload_bytes + 1)
    await file.close()
    if not contents:
        raise HTTPException(
            400, detail={"code": "empty_image", "message": "Uploaded file is empty."}
        )
    if len(contents) > analyzer.config.max_upload_bytes:
        raise HTTPException(
            413,
            detail={
                "code": "image_too_large",
                "message": (
                    f"Image exceeds the {analyzer.config.max_upload_bytes // 1024 // 1024} "
                    "MB upload limit."
                ),
            },
        )

    await semaphore.acquire()
    release_when_done = False
    loop = asyncio.get_running_loop()
    work = loop.run_in_executor(
        None,
        lambda: analyzer.analyze(
            contents,
            reference_length_mm=calibration_reference_mm,
            reference_length_px=calibration_reference_px,
        ),
    )
    try:
        return await asyncio.wait_for(
            asyncio.shield(work),
            timeout=analyzer.config.inference_timeout_seconds,
        )
    except TimeoutError as exc:
        # Python cannot stop an already-running PyTorch worker thread. Keep the
        # concurrency slot occupied until it actually exits.
        release_when_done = True
        work.add_done_callback(lambda _: semaphore.release())
        timeout_error = ShrimpInferenceTimeoutError(
            "Shrimp analysis timed out. The server may need a CUDA GPU or a "
            "longer AQUASCAN_SHRIMP_TIMEOUT_SECONDS value."
        )
        raise HTTPException(
            timeout_error.status_code,
            detail={"code": timeout_error.code, "message": timeout_error.message},
        ) from exc
    except ShrimpAnalysisError as exc:
        raise HTTPException(
            exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            400, detail={"code": "invalid_calibration", "message": str(exc)}
        ) from exc
    finally:
        if not release_when_done:
            semaphore.release()
