"""Document discovery and parsing for the aquaculture knowledge base."""

from .loaders import (
    SUPPORTED_EXTENSIONS,
    SourceDocument,
    discover_source_files,
    load_source_documents,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SourceDocument",
    "discover_source_files",
    "load_source_documents",
]
