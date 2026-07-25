"""Lazy, one-model-at-a-time llama.cpp streaming inference."""

from __future__ import annotations

import ctypes
import gc
import os
import site
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from ..config import AssistantConfig, ModelSpec

_weights_stay_memory_mapped = False
_cuda_dll_handles: list[object] = []
_cuda_native_handles: list[object] = []


def _prepare_cuda_runtime() -> None:
    """Expose pip-installed NVIDIA runtime DLLs to Windows."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for site_packages in site.getsitepackages():
        package_root = Path(site_packages)
        nvidia_root = package_root / "nvidia"
        for relative in ("cuda_runtime/bin", "cublas/bin"):
            dll_dir = nvidia_root / relative
            if dll_dir.is_dir():
                _cuda_dll_handles.append(os.add_dll_directory(str(dll_dir)))
        llama_lib = package_root / "llama_cpp" / "lib"
        if not _cuda_dll_handles or not llama_lib.is_dir():
            continue
        _cuda_dll_handles.append(os.add_dll_directory(str(llama_lib)))
        for filename in (
            "ggml-base.dll",
            "ggml-cpu.dll",
            "ggml-cuda.dll",
            "ggml.dll",
            "llama.dll",
        ):
            path = llama_lib / filename
            if path.is_file():
                _cuda_native_handles.append(ctypes.CDLL(str(path)))


_prepare_cuda_runtime()


def keep_weights_memory_mapped() -> bool:
    """Serve quantized weights straight from the memory-mapped GGUF file.

    By default llama.cpp "repacks" quantized tensors into CPU-optimised
    layouts at load time, which copies nearly the whole model into committed
    RAM. When the backend shares its process with TensorFlow on a 16 GB
    machine, that multi-gigabyte allocation is exactly what makes 7-8B
    models fail with "Failed to load model from file". Disabling the extra
    buffer types keeps weights file-backed and evictable, trading some
    prompt-processing speed for reliable loads.

    Returns True when the flag was applied (or already active).
    """
    global _weights_stay_memory_mapped
    if _weights_stay_memory_mapped:
        return True
    try:
        import llama_cpp.llama_cpp as lib
    except ImportError:
        return False
    if not hasattr(lib.llama_model_default_params(), "use_extra_bufts"):
        return False

    original = lib.llama_model_default_params

    def patched_default_params(*args, **kwargs):
        params = original(*args, **kwargs)
        params.use_extra_bufts = False
        return params

    # Llama.__init__ has no kwarg for this flag, so patch the default-params
    # factory it calls internally.
    lib.llama_model_default_params = patched_default_params
    _weights_stay_memory_mapped = True
    return True


def strip_thinking_tokens(chunks: Iterator[str]) -> Iterator[str]:
    """Remove Qwen thinking spans without leaking split tags or their content."""
    opening = "<think>"
    closing = "</think>"
    buffer = ""
    inside = False

    for chunk in chunks:
        buffer += chunk
        while buffer:
            if inside:
                end = buffer.find(closing)
                if end >= 0:
                    buffer = buffer[end + len(closing) :]
                    inside = False
                    continue
                # Preserve only a possible split closing tag.
                keep = max(
                    (
                        length
                        for length in range(1, min(len(buffer), len(closing)) + 1)
                        if closing.startswith(buffer[-length:])
                    ),
                    default=0,
                )
                buffer = buffer[-keep:] if keep else ""
                break

            start = buffer.find(opening)
            if start >= 0:
                if start:
                    yield buffer[:start]
                buffer = buffer[start + len(opening) :]
                inside = True
                continue

            # Do not emit a suffix that could be the beginning of a split tag.
            keep = max(
                (
                    length
                    for length in range(1, min(len(buffer), len(opening)) + 1)
                    if opening.startswith(buffer[-length:])
                ),
                default=0,
            )
            ready = buffer[:-keep] if keep else buffer
            if ready:
                yield ready
            buffer = buffer[-keep:] if keep else ""
            break

    if buffer and not inside:
        yield buffer


class LocalLlmManager:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self._llm = None
        self._active_key: str | None = None
        self._lock = threading.RLock()
        self._load_error: str | None = None

    @property
    def active_model(self) -> str | None:
        return self._active_key

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def available_models(self) -> list[dict[str, object]]:
        return [
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "available": spec.available or self.config.fake_mode,
                "path": str(spec.path),
                "active": spec.key == self._active_key,
            }
            for spec in self.config.model_specs.values()
        ]

    def _load(self, spec: ModelSpec):
        if self.config.fake_mode:
            self._active_key = spec.key
            return None
        if not spec.available:
            raise FileNotFoundError(
                f"{spec.display_name} was not found at {spec.path}. "
                "Install a local GGUF model or choose another configured model."
            )
        with self._lock:
            if self._llm is not None and self._active_key == spec.key:
                return self._llm
            self.unload()
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Install the assistant "
                    "requirements using the documented CPU wheel index."
                ) from exc
            if self.config.low_memory:
                keep_weights_memory_mapped()
            try:
                self._llm = Llama(
                    model_path=str(spec.path),
                    n_ctx=self.config.n_ctx,
                    n_batch=min(
                        256 if self.config.low_memory else 512,
                        self.config.n_ctx,
                    ),
                    n_threads=self.config.n_threads,
                    n_threads_batch=self.config.n_threads,
                    n_gpu_layers=self.config.n_gpu_layers,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )
            except Exception as exc:
                self._load_error = str(exc)
                raise RuntimeError(
                    f"Could not load {spec.display_name}. This usually means "
                    "the computer is low on free memory: close memory-heavy "
                    "applications (browser tabs, emulators, IDE build "
                    "daemons) and press regenerate, or switch to another "
                    f"model from the header menu. Details: {exc}"
                ) from exc
            self._active_key = spec.key
            self._load_error = None
            return self._llm

    def unload(self) -> None:
        if self._llm is not None:
            try:
                self._llm.close()
            except (AttributeError, RuntimeError):
                pass
        self._llm = None
        self._active_key = None
        gc.collect()

    def stream(self, prompt: str, model: str | None = None) -> Iterator[str]:
        spec = self.config.selected_model(model)
        if self.config.fake_mode:
            self._active_key = spec.key
            for token in (
                "The local aquaculture knowledge indicates ",
                "that this answer should be confirmed against the cited ",
                "documents and, for treatment decisions, by a fish-health ",
                "professional.",
            ):
                yield token
            return

        llm = self._load(spec)
        with self._lock:
            prompt_tokens = len(
                llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            )
            available_tokens = self.config.n_ctx - prompt_tokens - 8
            if available_tokens < 16:
                raise RuntimeError(
                    "The assistant context is too long for the configured "
                    f"{self.config.n_ctx}-token window. Clear the conversation "
                    "or increase AQUASCAN_ASSISTANT_N_CTX."
                )
            started_at = time.monotonic()
            output = llm(
                prompt,
                max_tokens=max(
                    1, min(self.config.max_tokens, available_tokens)
                ),
                temperature=0.35,
                top_p=0.90,
                top_k=40,
                repeat_penalty=1.12,
                stop=list(spec.stop_tokens),
                stream=True,
            )

            def timed_chunks() -> Iterator[str]:
                for chunk in output:
                    if (
                        time.monotonic() - started_at
                        > self.config.generation_timeout_seconds
                    ):
                        raise TimeoutError(
                            "The local assistant exceeded its "
                            f"{self.config.generation_timeout_seconds}-second "
                            "generation limit. Try a shorter question or a "
                            "smaller local model."
                        )
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text:
                        yield str(text)

            chunks: Iterator[str] = timed_chunks()
            if spec.key == "qwen":
                chunks = strip_thinking_tokens(chunks)
            yield from chunks
