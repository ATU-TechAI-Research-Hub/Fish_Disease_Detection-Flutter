"""Context retrieval enriched with the latest CNN prediction."""

from __future__ import annotations

from ..models import PredictionContext, RetrievedDocument


class AquacultureRetriever:
    def __init__(self, vector_store, default_k: int = 5) -> None:
        self.vector_store = vector_store
        self.default_k = default_k

    def retrieve(
        self,
        question: str,
        prediction: PredictionContext | None = None,
        k: int | None = None,
    ) -> list[RetrievedDocument]:
        parts = [question.strip()]
        if prediction is not None:
            parts.extend(
                [
                    prediction.disease_name,
                    prediction.disease_type,
                    prediction.symptoms,
                ]
            )
        query = "\n".join(part for part in parts if part)
        return self.vector_store.search(query, k=k or self.default_k)
