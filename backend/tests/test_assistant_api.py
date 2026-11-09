"""Assistant API contract tests with dependency-free fake local models."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("AQUASCAN_ENABLE_FISH_GATE", "0")
os.environ.setdefault("AQUASCAN_ASSISTANT_FAKE", "1")
os.environ.setdefault(
    "AQUASCAN_ASSISTANT_HISTORY_DB",
    str(Path(__file__).resolve().parents[1] / "outputs" / "assistant_test.sqlite3"),
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from aquaculture_assistant.api import assistant_runtime  # noqa: E402


def _session() -> str:
    return f"test_{uuid.uuid4().hex}"


def _prediction() -> dict:
    return {
        "prediction": {
            "id": 2,
            "name": "Bacterial Aeromoniasis",
            "type": "Bacterial",
            "cause": "Aeromonas bacteria and environmental stress.",
            "symptoms": "Red lesions and ulcers.",
            "treatment": "Confirm diagnosis and improve water quality.",
            "prevention": "Biosecurity and stable water quality.",
        },
        "confidence": 0.82,
        "confidence_tier": "high",
        "source": "onnxruntime",
        "filename": "fish.jpg",
        "inference_ms": 17.5,
        "top_predictions": [
            {
                "disease_id": 2,
                "disease_name": "Bacterial Aeromoniasis",
                "confidence": 0.82,
            }
        ],
    }


def test_assistant_health_and_models_are_local():
    with TestClient(app) as client:
        health = client.get("/assistant/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["local_only"] is True

        models = client.get("/assistant/models").json()["models"]
        assert {item["key"] for item in models} == {"llama", "mistral", "qwen"}
        assert all(item["available"] for item in models)


def test_prediction_context_chat_history_regenerate_and_clear():
    session_id = _session()
    with TestClient(app) as client:
        stored = client.post(
            "/assistant/prediction-context",
            json={"session_id": session_id, "prediction": _prediction()},
        )
        assert stored.status_code == 200
        assert (
            stored.json()["prediction"]["disease_name"]
            == "Bacterial Aeromoniasis"
        )

        chat = client.post(
            "/assistant/chat",
            json={
                "session_id": session_id,
                "question": "What does this prediction mean?",
                "model": "qwen",
            },
        )
        assert chat.status_code == 200
        assert chat.json()["answer"]

        history = client.get(f"/assistant/history/{session_id}").json()["messages"]
        assert [message["role"] for message in history] == ["user", "assistant"]
        assert history[-1]["sources"]

        regenerated = client.post(
            "/assistant/chat",
            json={
                "session_id": session_id,
                "question": "",
                "model": "mistral",
                "regenerate": True,
            },
        )
        assert regenerated.status_code == 200
        history = client.get(f"/assistant/history/{session_id}").json()["messages"]
        assert [message["role"] for message in history] == ["user", "assistant"]
        assert history[-1]["model"] == "mistral"

        cleared = client.delete(f"/assistant/history/{session_id}")
        assert cleared.status_code == 200
        assert (
            client.get(f"/assistant/history/{session_id}").json()["messages"]
            == []
        )
        deleted = client.delete(f"/assistant/session/{session_id}")
        assert deleted.status_code == 200
        assert (
            assistant_runtime.service.prediction_context.get(session_id) is None
        )


def test_stream_is_valid_ndjson_start_tokens_done():
    session_id = _session()
    with TestClient(app) as client:
        response = client.post(
            "/assistant/chat/stream",
            json={
                "session_id": session_id,
                "question": "How can I improve dissolved oxygen?",
                "model": "llama",
            },
        )
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "done"
    assert any(event["type"] == "token" for event in events)
    assert "".join(
        event["text"] for event in events if event["type"] == "token"
    ).strip()


def test_second_generation_is_rejected_without_queuing_history():
    session_id = _session()
    with TestClient(app) as client:
        assert assistant_runtime.service.try_begin_generation() is True
        try:
            response = client.post(
                "/assistant/chat/stream",
                json={
                    "session_id": session_id,
                    "question": "Do not queue this duplicate prompt.",
                    "model": "qwen",
                },
            )
        finally:
            assistant_runtime.service.end_generation()

        events = [
            json.loads(line) for line in response.text.splitlines() if line
        ]
        assert events == [
            {
                "type": "error",
                "message": (
                    "The local assistant is already answering another prompt. "
                    "Wait for it to finish before sending a new request."
                ),
            }
        ]
        history = client.get(
            f"/assistant/history/{session_id}"
        ).json()["messages"]
        assert history == []


def test_assistant_rejects_invalid_session_and_unknown_model():
    with TestClient(app) as client:
        invalid = client.post(
            "/assistant/chat",
            json={"session_id": "../bad", "question": "hello"},
        )
        assert invalid.status_code == 422

        unknown = client.post(
            "/assistant/chat",
            json={
                "session_id": _session(),
                "question": "hello",
                "model": "remote-api",
            },
        )
        assert unknown.status_code == 400
