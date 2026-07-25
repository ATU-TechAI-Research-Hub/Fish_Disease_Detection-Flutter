"""Pure assistant tests that require no FAISS, embeddings, or GGUF model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CHATBOT_ROOT = Path(__file__).resolve().parents[2] / "AI chatbot"
sys.path.insert(0, str(CHATBOT_ROOT))

from aquaculture_assistant.chat_history import ChatHistoryStore  # noqa: E402
from aquaculture_assistant.dataset_ingestion import (  # noqa: E402
    discover_source_files,
    load_source_documents,
)
from aquaculture_assistant.models import (  # noqa: E402
    PredictionContext,
    RetrievedDocument,
)
from aquaculture_assistant.llm.local_llm import (  # noqa: E402
    strip_thinking_tokens,
)
from aquaculture_assistant.prompts import build_prompt  # noqa: E402


def test_prompt_is_prediction_history_and_retrieval_aware(tmp_path):
    store = ChatHistoryStore(tmp_path / "history.sqlite3")
    store.append(
        session_id="session_123",
        role="user",
        content="This fish has red lesions.",
    )
    history = store.get_history("session_123")
    prediction = PredictionContext(
        disease_name="Bacterial Aeromoniasis",
        confidence=0.82,
        confidence_tier="high",
        filename="fish.jpg",
        timestamp="2026-01-01T00:00:00+00:00",
        symptoms="Red lesions and ulcers",
    )
    documents = [
        RetrievedDocument(
            text="Aeromoniasis may be associated with ulcers and stress.",
            source="diseases.json",
            title="Bacterial Aeromoniasis",
            score=0.9,
        )
    ]

    for model in ("llama", "mistral", "qwen"):
        prompt = build_prompt(
            model=model,
            question="How should I respond?",
            prediction=prediction,
            documents=documents,
            history=history,
        )
        assert "Bacterial Aeromoniasis" in prompt
        assert "82.0" in prompt
        assert "red lesions" in prompt.lower()
        assert "[1]" in prompt
        assert "mental health" not in prompt.lower()
        if model == "llama":
            assert not prompt.startswith("<|begin_of_text|>")
        if model == "mistral":
            assert not prompt.startswith("<s>")
        if model == "qwen":
            assert "/no_think" not in prompt.split("<|im_start|>user\n", 1)[0]
            assert "\n/no_think<|im_end|>" in prompt


def test_chat_history_and_prediction_are_persistent(tmp_path):
    path = tmp_path / "history.sqlite3"
    first = ChatHistoryStore(path)
    first.append(session_id="session_abc", role="user", content="What is Ich?")
    prediction = PredictionContext(
        disease_name="Parasitic Disease",
        confidence=0.55,
        confidence_tier="medium",
        filename="fish.png",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    first.set_prediction("session_abc", prediction)

    reopened = ChatHistoryStore(path)
    assert reopened.get_history("session_abc")[0].content == "What is Ich?"
    assert (
        reopened.get_prediction("session_abc").disease_name
        == "Parasitic Disease"
    )
    reopened.delete_session("session_abc")
    reopened.append(
        session_id="session_abc",
        role="assistant",
        content="A stale response that must not be persisted.",
    )
    reopened.set_prediction("session_abc", prediction)
    assert reopened.get_history("session_abc") == []
    assert reopened.get_prediction("session_abc") is None


def test_ingestion_discovers_and_loads_json_csv_markdown(tmp_path):
    (tmp_path / "guide.md").write_text(
        "# Oxygen\nUse aeration when dissolved oxygen is low.",
        encoding="utf-8",
    )
    (tmp_path / "diseases.json").write_text(
        json.dumps(
            [
                {
                    "name": "Gill Disease",
                    "symptoms": "Labored breathing",
                    "prevention": "Protect water quality",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "species.csv").write_text(
        "species,temperature\nTilapia,Species-specific warm water range\n",
        encoding="utf-8",
    )

    files = discover_source_files([tmp_path])
    assert {path.suffix for path in files} == {".md", ".json", ".csv"}
    documents = [
        document for path in files for document in load_source_documents(path)
    ]
    combined = "\n".join(document.text for document in documents)
    assert "dissolved oxygen" in combined
    assert "Labored breathing" in combined
    assert "Tilapia" in combined


def test_qwen_thinking_spans_are_removed_across_stream_chunks():
    chunks = [
        "<thi",
        "nk>\ninternal reasoning that must not be shown",
        "</thi",
        "nk>\n\nGrounded answer [1].",
    ]
    assert "".join(strip_thinking_tokens(iter(chunks))).strip() == (
        "Grounded answer [1]."
    )


def test_low_memory_mode_keeps_gguf_weights_memory_mapped():
    pytest.importorskip("llama_cpp")
    from aquaculture_assistant.llm import local_llm

    assert local_llm.keep_weights_memory_mapped() is True
    # Idempotent on repeat calls.
    assert local_llm.keep_weights_memory_mapped() is True

    import llama_cpp.llama_cpp as lib

    assert bool(lib.llama_model_default_params().use_extra_bufts) is False
