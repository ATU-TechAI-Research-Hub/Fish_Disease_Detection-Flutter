# AquaScan Local Aquaculture Assistant

This directory contains the local RAG assistant used by the AquaScan Flutter
application. It replaces the original mental-health notebook prototype.
There is no appointment automation, crisis keyword logic, mental-health
dataset, notebook UI, OpenAI integration, or remote generation API.

The three existing GGUF files are retained as local model choices:

- `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
- `models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- `models/qwen3_8b_gguf/Qwen3-8B-Q4_K_M.gguf`

Model files, generated vectors, and SQLite history are Git-ignored.

## Architecture

The runtime path is:

```text
knowledge documents + assets/diseases.json + model/labels.json
    -> format-specific loaders
    -> overlapping text chunks
    -> all-MiniLM-L6-v2 normalized embeddings
    -> raw FAISS inner-product index + JSON metadata
    -> top-k retrieval enriched with the latest CNN prediction
    -> model-native prompt (Qwen, Mistral, or Llama)
    -> llama.cpp token stream
    -> FastAPI NDJSON
    -> persistent Flutter assistant panel
```

`aquaculture_assistant/` is framework-neutral except for `api/router.py`.
Heavy dependencies and GGUF models are lazy-loaded. Only one GGUF model is
kept in memory at a time; choosing another model unloads the previous one.

Conversation messages and the latest prediction context are persisted in
`chat_history/assistant_history.sqlite3`. The Flutter client keeps one random,
persistent session identifier in `shared_preferences`.

## Install

Use the backend virtual environment. Do not use the old `.venv` in this
directory; it was created on another computer and is not portable.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r "..\AI chatbot\requirements-assistant.txt"
```

`backend/run_backend.bat` performs both dependency installs automatically.

For NVIDIA acceleration, replace the portable CPU wheel with the official
CUDA 12.4 wheel (newer NVIDIA drivers remain backward compatible):

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall --no-deps --only-binary=:all: llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
.\.venv\Scripts\python.exe -m pip install "nvidia-cuda-runtime-cu12>=12.4,<12.5" "nvidia-cublas-cu12>=12.4,<12.5"
```

`run_backend.bat` detects a CUDA-enabled build and enables GPU layer offload.
If full offload exceeds the GPU's memory, set
`AQUASCAN_ASSISTANT_GPU_LAYERS` to a smaller value such as `24`.

The public MiniLM embedding model downloads to the local Hugging Face cache on
first setup. No Hugging Face token is required. Once dependencies, MiniLM, and
one or more GGUF files exist locally, retrieval and generation work offline.
The application does not call OpenAI or another hosted generation service.

## Knowledge ingestion

Put new sources in `knowledge/`, or add external paths with
`AQUASCAN_KNOWLEDGE_PATHS`. Supported formats are:

- CSV
- JSON
- TXT
- Markdown
- PDF
- Excel `.xlsx` and `.xls`

The disease catalog, class labels, and accuracy research notes are included
automatically. Before every retrieval, source fingerprints are compared with
the saved manifest; SHA-256 is recomputed when size or modification time
changes. Added, edited, or removed documents cause an automatic rebuild.
Generated files are:

```text
vector_db/
├── aquaculture.index.faiss
├── chunks.json
└── manifest.json
```

Metadata uses JSON rather than Python pickle, so loading the index does not
enable arbitrary pickle deserialization.

To force a rebuild:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/assistant/reindex
```

## Configuration

All settings are optional environment variables:

```powershell
$env:AQUASCAN_ASSISTANT_MODEL = "qwen"       # qwen, mistral, llama
$env:AQUASCAN_ASSISTANT_N_CTX = "4096"
$env:AQUASCAN_ASSISTANT_MAX_TOKENS = "256"
$env:AQUASCAN_ASSISTANT_TIMEOUT_SECONDS = "180"
$env:AQUASCAN_ASSISTANT_THREADS = "6"
$env:AQUASCAN_ASSISTANT_GPU_LAYERS = "-1"    # all layers; use 0 for CPU
$env:AQUASCAN_ASSISTANT_LOW_MEMORY = "1"     # keep GGUF weights mmap-backed
$env:AQUASCAN_RETRIEVAL_K = "4"
$env:AQUASCAN_ALLOW_EMBEDDING_DOWNLOAD = "0" # strict offline mode

$env:AQUASCAN_QWEN_MODEL = "D:\models\qwen.gguf"
$env:AQUASCAN_MISTRAL_MODEL = "D:\models\mistral.gguf"
$env:AQUASCAN_LLAMA_MODEL = "D:\models\llama.gguf"

# Multiple paths use the operating-system path separator.
$env:AQUASCAN_KNOWLEDGE_PATHS = "D:\farm-guides;D:\water-quality.pdf"
```

With a CUDA-enabled llama-cpp-python build, set
`AQUASCAN_ASSISTANT_GPU_LAYERS` to the number of layers to offload. The
provided requirements use a portable CPU wheel.

`AQUASCAN_ASSISTANT_LOW_MEMORY` (default on) stops llama.cpp from repacking
quantized weights into committed RAM, so a 7-8B model loads even when
TensorFlow shares the process and only a few GB are free. Set it to `0` on
machines with abundant RAM to regain some prompt-processing speed.

If a model still fails to load with *"Could not load ... low on free
memory"*, close memory-heavy applications (browsers, emulators, Gradle
daemons from a mobile build) and press the regenerate button on the failed
answer, or switch to a smaller model in the panel header.

## API

The router is mounted in the existing FastAPI process:

- `GET /assistant/health`
- `GET /assistant/models`
- `GET /assistant/history/{session_id}`
- `DELETE /assistant/history/{session_id}`
- `DELETE /assistant/session/{session_id}`
- `POST /assistant/prediction-context`
- `POST /assistant/chat`
- `POST /assistant/chat/stream`
- `POST /assistant/reindex`

`/assistant/chat/stream` returns one JSON object per line. Event types are
`start`, `token`, `done`, and `error`. The `start` event includes retrieved
sources and the current prediction. `/predict` also accepts the optional
multipart field `assistant_session_id`; when supplied, the complete CNN result
is saved as the current assistant context without a second request.

## Grounding and limitations

The system prompt requires citations, uncertainty disclosure, differential
diagnoses, and professional confirmation before medication. It explicitly
prevents the model from claiming pixel-level CNN explanations unless a real
attribution map exists, and it does not provide antibiotic doses.

RAG reduces hallucination but does not eliminate it. Retrieved text and model
output are informational, not a veterinary diagnosis. Confidence is a model
score, not the probability that a clinical diagnosis is correct.

The bundled models are 7B-8B Q4 files. CPU-only generation can be slow,
especially for the first response; GPU offload substantially improves latency.
Generation is serialized inside one backend process to avoid using the same
llama.cpp model concurrently. Running multiple Uvicorn workers would load a
separate multi-gigabyte model in each process and is therefore not recommended.

## Tests

Backend assistant tests use deterministic fake embeddings and generation, so
they run without loading a GGUF file:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests\test_assistant_core.py tests\test_assistant_api.py
```

The full Flutter test suite includes NDJSON parsing and assistant controller
stream/history behavior:

```powershell
flutter test
```
