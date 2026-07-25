"""Local, retrieval-augmented Aquaculture AI Assistant.

The package intentionally keeps heavy dependencies lazy. Importing it never
loads Sentence Transformers, FAISS, or a multi-gigabyte GGUF model; those are
loaded only when the first assistant request needs them.
"""

from .config import AssistantConfig, ModelSpec
from .models import (
    ChatMessage,
    PredictionContext,
    RetrievedDocument,
)
from .runtime import AssistantRuntime

__all__ = [
    "AssistantConfig",
    "AssistantRuntime",
    "ChatMessage",
    "ModelSpec",
    "PredictionContext",
    "RetrievedDocument",
]
