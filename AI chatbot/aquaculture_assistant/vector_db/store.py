"""Auto-rebuilding FAISS store for aquaculture documents."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset_ingestion import (
    SourceDocument,
    discover_source_files,
    load_source_documents,
)
from ..models import RetrievedDocument

INDEX_VERSION = 2


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Character chunking that prefers paragraph/word boundaries."""
    compact = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(compact) <= size:
        return [compact] if compact else []
    chunks: list[str] = []
    start = 0
    while start < len(compact):
        end = min(start + size, len(compact))
        if end < len(compact):
            boundary = max(
                compact.rfind("\n\n", start, end),
                compact.rfind(". ", start, end),
                compact.rfind(" ", start, end),
            )
            if boundary > start + size // 2:
                end = boundary + 1
        chunk = compact[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(compact):
            break
        start = max(end - overlap, start + 1)
    return chunks


class AquacultureVectorStore:
    def __init__(
        self,
        *,
        roots: tuple[Path, ...],
        output_dir: Path,
        embeddings,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.roots = roots
        self.output_dir = output_dir
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = output_dir / "aquaculture.index.faiss"
        self.chunks_path = output_dir / "chunks.json"
        self.manifest_path = output_dir / "manifest.json"
        self._index = None
        self._chunks: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.last_rebuilt = False

    @property
    def ready(self) -> bool:
        return self._index is not None and bool(self._chunks)

    @property
    def document_count(self) -> int:
        return len(self._chunks)

    def _source_manifest(
        self, saved_manifest: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        files = discover_source_files(self.roots)
        saved_sources = {
            item.get("path"): item
            for item in (saved_manifest or {}).get("sources", [])
        }
        entries = []
        for path in files:
            stat = path.stat()
            previous = saved_sources.get(str(path), {})
            unchanged = (
                previous.get("bytes") == stat.st_size
                and previous.get("mtime_ns") == stat.st_mtime_ns
                and previous.get("sha256")
            )
            digest = (
                str(previous["sha256"])
                if unchanged
                else hashlib.sha256(path.read_bytes()).hexdigest()
            )
            entries.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return {
            "version": INDEX_VERSION,
            "embedding_model": self.embeddings.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "sources": entries,
        }

    def _saved_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.is_file():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _require_faiss():
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS is not installed. Install "
                "`AI chatbot/requirements-assistant.txt`."
            ) from exc
        return faiss

    def ensure_current(self, force: bool = False) -> bool:
        """Load the saved index or rebuild when any source document changed."""
        with self._lock:
            saved_manifest = self._saved_manifest()
            manifest = self._source_manifest(saved_manifest)
            if not manifest["sources"]:
                raise RuntimeError(
                    "No aquaculture knowledge documents were discovered."
                )
            can_load = (
                not force
                and manifest == saved_manifest
                and self.index_path.is_file()
                and self.chunks_path.is_file()
            )
            if can_load:
                try:
                    self._load()
                    self.last_rebuilt = False
                    return False
                except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
                    # Generated artifacts are disposable; recover from a
                    # partial/corrupt write instead of requiring manual repair.
                    pass
            self._rebuild(manifest)
            self.last_rebuilt = True
            return True

    def _load(self) -> None:
        faiss = self._require_faiss()
        chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(self.index_path))
        if index.ntotal != len(chunks):
            raise RuntimeError(
                "FAISS index and chunk metadata disagree; rebuild required."
            )
        self._index = index
        self._chunks = chunks

    def _rebuild(self, manifest: dict[str, Any]) -> None:
        faiss = self._require_faiss()
        chunks: list[dict[str, Any]] = []
        for source in manifest["sources"]:
            path = Path(source["path"])
            for document in load_source_documents(path):
                chunks.extend(self._chunk_document(document))
        if not chunks:
            raise RuntimeError("Knowledge sources produced no usable text.")

        vectors = self.embeddings.encode([chunk["text"] for chunk in chunks])
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise RuntimeError("Embedding output has an invalid shape.")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_index = self.index_path.with_suffix(".faiss.tmp")
        temporary_chunks = self.chunks_path.with_suffix(".json.tmp")
        temporary_manifest = self.manifest_path.with_suffix(".json.tmp")
        faiss.write_index(index, str(temporary_index))
        temporary_chunks.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        temporary_index.replace(self.index_path)
        temporary_chunks.replace(self.chunks_path)
        temporary_manifest.replace(self.manifest_path)
        self._index = index
        self._chunks = chunks

    def _chunk_document(self, document: SourceDocument) -> list[dict[str, str]]:
        return [
            {
                "text": text,
                "source": document.source,
                "title": document.title,
            }
            for text in _split_text(
                document.text,
                size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
        ]

    def search(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        with self._lock:
            self.ensure_current()
            vector = self.embeddings.encode([query])
            vector = np.ascontiguousarray(vector, dtype=np.float32)
            scores, indices = self._index.search(
                vector, min(max(k, 1), len(self._chunks))
            )
            results: list[RetrievedDocument] = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                chunk = self._chunks[int(index)]
                results.append(
                    RetrievedDocument(
                        text=chunk["text"],
                        source=chunk["source"],
                        title=chunk["title"],
                        score=float(score),
                    )
                )
            return results
