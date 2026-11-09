"""RAG + history + prediction + local LLM orchestration."""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..models import PredictionContext
from ..prompts import build_prompt


class AssistantService:
    def __init__(
        self,
        *,
        config,
        history_store,
        prediction_context,
        retriever,
        vector_store,
        llm,
    ) -> None:
        self.config = config
        self.history_store = history_store
        self.prediction_context = prediction_context
        self.retriever = retriever
        self.vector_store = vector_store
        self.llm = llm
        self._generation_lock = threading.Lock()

    @property
    def is_generating(self) -> bool:
        return self._generation_lock.locked()

    def try_begin_generation(self) -> bool:
        return self._generation_lock.acquire(blocking=False)

    def end_generation(self) -> None:
        if self._generation_lock.locked():
            self._generation_lock.release()

    def status(self) -> dict[str, Any]:
        models = self.llm.available_models()
        missing_dependencies = (
            []
            if self.config.fake_mode
            else [
                module
                for module in ("faiss", "sentence_transformers", "llama_cpp")
                if importlib.util.find_spec(module) is None
            ]
        )
        return {
            "status": "ok"
            if any(bool(model["available"]) for model in models)
            and not missing_dependencies
            else "degraded",
            "local_only": True,
            "active_model": self.llm.active_model,
            "default_model": self.config.default_model,
            "models": models,
            "embeddings_loaded": self.vector_store.embeddings.loaded,
            "index_ready": self.vector_store.ready,
            "indexed_chunks": self.vector_store.document_count,
            "last_index_rebuilt": self.vector_store.last_rebuilt,
            "embedding_model": self.config.embedding_model,
            "missing_dependencies": missing_dependencies,
            "load_error": self.llm.load_error,
            "generating": self.is_generating,
        }

    def rebuild_knowledge(self, force: bool = True) -> dict[str, Any]:
        rebuilt = self.vector_store.ensure_current(force=force)
        return {
            "rebuilt": rebuilt,
            "indexed_chunks": self.vector_store.document_count,
            "embedding_model": self.config.embedding_model,
        }

    def publish_prediction(
        self, session_id: str, prediction_payload: dict[str, Any]
    ) -> PredictionContext:
        return self.prediction_context.publish(session_id, prediction_payload)

    def history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return [
            message.to_dict()
            for message in self.history_store.get_history(session_id, limit)
        ]

    def clear_history(self, session_id: str) -> None:
        self.history_store.clear(session_id)

    def delete_session(self, session_id: str) -> None:
        self.history_store.delete_session(session_id)

    def stream_chat(
        self,
        *,
        session_id: str,
        question: str,
        model: str | None = None,
        regenerate: bool = False,
    ) -> Iterator[dict[str, Any]]:
        selected = self.config.selected_model(model)
        clean_question = question.strip()

        if regenerate:
            previous = self.history_store.last_user_message(session_id)
            if not clean_question and previous is not None:
                clean_question = previous.content
        if not clean_question:
            raise ValueError("Question cannot be empty.")
        if regenerate:
            self.history_store.delete_last_assistant(session_id)

        prior_history = self.history_store.get_history(session_id, limit=12)
        if not regenerate:
            self.history_store.append(
                session_id=session_id,
                role="user",
                content=clean_question,
            )

        prediction = self.prediction_context.get(session_id)
        documents = self.retriever.retrieve(clean_question, prediction)
        source_payloads = [
            {
                **document.to_dict(),
                "source_name": Path(document.source).name,
            }
            for document in documents
        ]
        prompt = build_prompt(
            model=selected.key,
            question=clean_question,
            prediction=prediction,
            documents=documents,
            history=prior_history,
        )

        yield {
            "type": "start",
            "model": selected.key,
            "sources": source_payloads,
            "prediction": prediction.to_dict() if prediction else None,
        }

        pieces: list[str] = []
        try:
            for token in self.llm.stream(prompt, selected.key):
                pieces.append(token)
                yield {"type": "token", "text": token}
        except Exception as exc:
            yield {
                "type": "error",
                "message": str(exc),
                "model": selected.key,
            }
            return

        answer = "".join(pieces).strip()
        if not answer:
            yield {
                "type": "error",
                "message": "The local model returned an empty response.",
                "model": selected.key,
            }
            return
        saved = self.history_store.append(
            session_id=session_id,
            role="assistant",
            content=answer,
            model=selected.key,
            sources=source_payloads,
        )
        yield {
            "type": "done",
            "message": saved.to_dict(),
            "model": selected.key,
        }

    def chat(
        self,
        *,
        session_id: str,
        question: str,
        model: str | None = None,
        regenerate: bool = False,
    ) -> dict[str, Any]:
        answer_parts: list[str] = []
        final: dict[str, Any] | None = None
        for event in self.stream_chat(
            session_id=session_id,
            question=question,
            model=model,
            regenerate=regenerate,
        ):
            if event["type"] == "token":
                answer_parts.append(str(event["text"]))
            elif event["type"] == "error":
                raise RuntimeError(str(event["message"]))
            elif event["type"] == "done":
                final = event
        return {
            "answer": "".join(answer_parts).strip(),
            "message": final["message"] if final else None,
            "model": final["model"] if final else model,
        }
