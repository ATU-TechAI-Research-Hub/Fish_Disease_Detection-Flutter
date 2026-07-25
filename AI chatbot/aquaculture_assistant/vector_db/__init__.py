"""Safe FAISS vector database (JSON metadata; no pickle deserialization)."""

from .store import AquacultureVectorStore

__all__ = ["AquacultureVectorStore"]
