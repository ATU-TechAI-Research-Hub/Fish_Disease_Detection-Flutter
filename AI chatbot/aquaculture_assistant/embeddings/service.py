"""Lazy Sentence Transformer embeddings with normalized vectors."""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str, allow_download: bool = True) -> None:
        self.model_name = model_name
        self.allow_download = allow_download
        self._model = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformers is not installed. Install "
                    "`AI chatbot/requirements-assistant.txt` in the backend "
                    "environment."
                ) from exc
            # Try the cache first so normal runtime makes no Hub request.
            # A first-time setup may download the public model when allowed.
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception as local_error:
                if not self.allow_download:
                    raise RuntimeError(
                        "The embedding model is not available locally. "
                        "Temporarily set "
                        "AQUASCAN_ALLOW_EMBEDDING_DOWNLOAD=1 while online, "
                        "rebuild the index, then disable it again."
                    ) from local_error
                self._model = SentenceTransformer(self.model_name)
            return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
