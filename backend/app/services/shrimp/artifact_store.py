"""Short-lived result image storage served by the shrimp API."""

from __future__ import annotations

import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Mapping

_SAFE_ID = re.compile(r"^[a-f0-9]{32}$")
_SAFE_FILENAME = re.compile(r"^[a-zA-Z0-9_-]+\.png$")


class ShrimpArtifactStore:
    """Owns a process-scoped temp directory and removes expired analyses."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = max(60, ttl_seconds)
        self._root = Path(tempfile.mkdtemp(prefix="aquascan-shrimp-"))
        self._lock = Lock()

    @property
    def root(self) -> Path:
        return self._root

    def save(self, images: Mapping[str, bytes]) -> tuple[str, dict[str, str]]:
        analysis_id = uuid.uuid4().hex
        directory = self._root / analysis_id
        with self._lock:
            self.cleanup_expired()
            directory.mkdir(parents=False, exist_ok=False)
            urls: dict[str, str] = {}
            for name, payload in images.items():
                filename = f"{name}.png"
                if not _SAFE_FILENAME.fullmatch(filename):
                    raise ValueError(f"Unsafe artifact name: {name!r}")
                (directory / filename).write_bytes(payload)
                urls[name] = f"/api/shrimp/artifacts/{analysis_id}/{filename}"
        return analysis_id, urls

    def resolve(self, analysis_id: str, filename: str) -> Path | None:
        if not _SAFE_ID.fullmatch(analysis_id):
            return None
        if not _SAFE_FILENAME.fullmatch(filename):
            return None
        self.cleanup_expired()
        candidate = self._root / analysis_id / filename
        return candidate if candidate.is_file() else None

    def cleanup_expired(self) -> None:
        if not self._root.exists():
            return
        cutoff = time.time() - self._ttl_seconds
        for child in self._root.iterdir():
            try:
                if child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                continue

    def close(self) -> None:
        with self._lock:
            shutil.rmtree(self._root, ignore_errors=True)
