"""Environment-driven configuration for the local assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CHATBOT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _absolute_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (CHATBOT_ROOT / path).resolve()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    path: Path
    stop_tokens: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.path.is_file()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _knowledge_roots() -> tuple[Path, ...]:
    defaults = [
        CHATBOT_ROOT / "knowledge",
        PROJECT_ROOT / "assets" / "diseases.json",
        PROJECT_ROOT / "model" / "labels.json",
        PROJECT_ROOT / "backend" / "ACCURACY_RESEARCH.md",
    ]
    extra = os.getenv("AQUASCAN_KNOWLEDGE_PATHS", "")
    for item in extra.split(os.pathsep):
        if item.strip():
            defaults.append(Path(item.strip()).expanduser().resolve())
    return tuple(defaults)


def _model_specs() -> dict[str, ModelSpec]:
    model_root = CHATBOT_ROOT / "models"
    return {
        "llama": ModelSpec(
            key="llama",
            display_name="Llama 3.1 8B Instruct",
            path=_absolute_path(
                os.getenv(
                    "AQUASCAN_LLAMA_MODEL",
                    model_root / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                )
            ),
            stop_tokens=("<|eot_id|>", "<|end_of_text|>"),
        ),
        "mistral": ModelSpec(
            key="mistral",
            display_name="Mistral 7B Instruct v0.2",
            path=_absolute_path(
                os.getenv(
                    "AQUASCAN_MISTRAL_MODEL",
                    model_root / "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                )
            ),
            stop_tokens=("</s>",),
        ),
        "qwen": ModelSpec(
            key="qwen",
            display_name="Qwen3 8B",
            path=_absolute_path(
                os.getenv(
                    "AQUASCAN_QWEN_MODEL",
                    model_root / "qwen3_8b_gguf" / "Qwen3-8B-Q4_K_M.gguf",
                )
            ),
            stop_tokens=("<|im_end|>", "<|endoftext|>"),
        ),
    }


@dataclass(frozen=True)
class AssistantConfig:
    """All paths are absolute and independent of the process working directory."""

    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "AQUASCAN_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    allow_embedding_download: bool = field(
        default_factory=lambda: _env_bool(
            "AQUASCAN_ALLOW_EMBEDDING_DOWNLOAD", True
        )
    )
    default_model: str = field(
        default_factory=lambda: os.getenv(
            "AQUASCAN_ASSISTANT_MODEL", "qwen"
        ).lower()
    )
    n_ctx: int = field(
        default_factory=lambda: _env_int("AQUASCAN_ASSISTANT_N_CTX", 4096)
    )
    n_threads: int = field(
        default_factory=lambda: _env_int(
            "AQUASCAN_ASSISTANT_THREADS",
            min(max((os.cpu_count() or 4) // 2, 1), 8),
        )
    )
    n_gpu_layers: int = field(
        default_factory=lambda: _env_int(
            "AQUASCAN_ASSISTANT_GPU_LAYERS", 0
        )
    )
    max_tokens: int = field(
        default_factory=lambda: _env_int(
            "AQUASCAN_ASSISTANT_MAX_TOKENS", 256
        )
    )
    generation_timeout_seconds: int = field(
        default_factory=lambda: _env_int(
            "AQUASCAN_ASSISTANT_TIMEOUT_SECONDS", 180
        )
    )
    low_memory: bool = field(
        default_factory=lambda: _env_bool(
            "AQUASCAN_ASSISTANT_LOW_MEMORY", True
        )
    )
    retrieval_k: int = field(
        default_factory=lambda: _env_int("AQUASCAN_RETRIEVAL_K", 4)
    )
    chunk_size: int = 900
    chunk_overlap: int = 140
    knowledge_roots: tuple[Path, ...] = field(default_factory=_knowledge_roots)
    vector_dir: Path = field(
        default_factory=lambda: _absolute_path(
            os.getenv(
                "AQUASCAN_VECTOR_DIR",
                CHATBOT_ROOT / "vector_db",
            )
        )
    )
    history_db: Path = field(
        default_factory=lambda: _absolute_path(
            os.getenv(
                "AQUASCAN_ASSISTANT_HISTORY_DB",
                CHATBOT_ROOT
                / "chat_history"
                / "assistant_history.sqlite3",
            )
        )
    )
    model_specs: dict[str, ModelSpec] = field(default_factory=_model_specs)
    fake_mode: bool = field(
        default_factory=lambda: os.getenv(
            "AQUASCAN_ASSISTANT_FAKE", "0"
        ).strip().lower()
        in {"1", "true", "yes"}
    )

    def selected_model(self, requested: str | None = None) -> ModelSpec:
        key = (requested or self.default_model).lower()
        if key not in self.model_specs:
            allowed = ", ".join(sorted(self.model_specs))
            raise ValueError(f"Unknown local model {key!r}. Choose: {allowed}.")
        return self.model_specs[key]
