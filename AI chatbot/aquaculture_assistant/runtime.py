"""Composition root for the optional local assistant."""

from __future__ import annotations

from .chat_history import ChatHistoryStore
from .config import AssistantConfig
from .embeddings import EmbeddingService
from .llm import LocalLlmManager
from .prediction_context import PredictionContextService
from .rag import AquacultureRetriever
from .services import AssistantService
from .vector_db import AquacultureVectorStore


class _FakeEmbeddings:
    model_name = "fake-test-embeddings"
    loaded = True


class _FakeVectorStore:
    """Dependency-free deterministic store used only by automated tests."""

    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()
        self.ready = True
        self.document_count = 1
        self.last_rebuilt = False

    def ensure_current(self, force: bool = False) -> bool:
        self.last_rebuilt = bool(force)
        return bool(force)

    def search(self, query: str, k: int = 5):
        from .models import RetrievedDocument

        return [
            RetrievedDocument(
                text=(
                    "Fish disease predictions should be checked against "
                    "symptoms, water quality, and professional diagnosis."
                ),
                source="test://aquaculture-knowledge",
                title="Aquaculture test knowledge",
                score=1.0,
            )
        ]


class AssistantRuntime:
    """Builds lightweight objects now; heavy ML models remain lazy."""

    def __init__(self, config: AssistantConfig | None = None) -> None:
        self.config = config or AssistantConfig()
        history = ChatHistoryStore(self.config.history_db)
        if self.config.fake_mode:
            vector_store = _FakeVectorStore()
        else:
            embeddings = EmbeddingService(
                self.config.embedding_model,
                allow_download=self.config.allow_embedding_download,
            )
            vector_store = AquacultureVectorStore(
                roots=self.config.knowledge_roots,
                output_dir=self.config.vector_dir,
                embeddings=embeddings,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        prediction_context = PredictionContextService(history)
        retriever = AquacultureRetriever(
            vector_store,
            default_k=self.config.retrieval_k,
        )
        llm = LocalLlmManager(self.config)
        self.service = AssistantService(
            config=self.config,
            history_store=history,
            prediction_context=prediction_context,
            retriever=retriever,
            vector_store=vector_store,
            llm=llm,
        )
