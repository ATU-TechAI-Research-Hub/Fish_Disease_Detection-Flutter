"""Generate AquaScan research paper as a Word document."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def add_para(doc: Document, text: str, bold: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_figure_caption(doc: Document, number: int, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"Fig. {number}. {text}")
    run.italic = True
    run.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(10)


def add_table_caption(doc: Document, number: int, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"TABLE {number}\n{text}")
    run.bold = True
    run.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(6)


def build_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run(
        "AquaScan: An Integrated Mobile Platform for Freshwater Fish "
        "Disease Classification and Retrieval-Augmented Aquaculture Assistance"
    )
    tr.bold = True
    tr.font.size = Pt(14)

    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ar = author.add_run("Rachit Bhusal")
    ar.bold = True
    ar.font.size = Pt(12)

    affiliation = doc.add_paragraph()
    affiliation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    aff = affiliation.add_run(
        "B.Sc. Software Development, Atlantic Technological University (ATU), Ireland\n"
        "ATU Tech AI Research Hub · Summer 2026 Research Project (June–July 2026)"
    )
    aff.italic = True
    aff.font.size = Pt(10)

    supervisor = doc.add_paragraph()
    supervisor.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sup = supervisor.add_run(
        "Academic Supervisor: Faculty Supervisor, Atlantic Technological University"
    )
    sup.font.size = Pt(10)

    doc.add_paragraph()

    add_para(
        doc,
        "Abstract—Freshwater aquaculture in South Asia faces significant economic "
        "losses from bacterial, fungal, parasitic, and viral fish diseases, yet "
        "field-level diagnosis often depends on subjective visual inspection and "
        "delayed laboratory confirmation. This paper presents AquaScan, a local-first "
        "mobile and backend system that combines convolutional neural network (CNN) "
        "image classification with a retrieval-augmented generation (RAG) assistant "
        "for interpretable disease support. The disease classifier implements the "
        "architecture and preprocessing pipeline described by Tamut et al. (2025) "
        "for seven freshwater disease categories drawn from a Kaggle dataset of 2,444 "
        "images. Images are resized to 150×150 RGB, normalized to [0, 1], and "
        "classified through a Keras model.h5 primary backend with an ONNX Runtime "
        "fallback, enabling fully offline inference once deployed. A two-stage "
        "prediction pipeline adds an ImageNet MobileNetV2 fish-presence gate and an "
        "entropy-aware uncertainty filter to reject non-fish or ambiguous inputs. "
        "The Flutter client exposes camera and gallery capture, confidence-tiered "
        "results, scan history, a disease library, Firebase Google Sign-In with "
        "guest mode, and dark/light theming. A FastAPI service orchestrates "
        "prediction, health monitoring, and an embedded aquaculture assistant that "
        "indexes domain knowledge with sentence-transformers/all-MiniLM-L6-v2, stores "
        "vectors in FAISS, and generates grounded explanations through locally "
        "deployed Llama 3.1, Mistral 7B, or Qwen3 GGUF models via llama.cpp. "
        "CNN prediction context is automatically attached to assistant sessions, "
        "supporting multi-turn follow-up without external cloud APIs. On the held-out "
        "Kaggle test split (N=700 images), the best paper-aligned Keras checkpoint "
        "achieved 81.3% top-1 accuracy, 95.1% top-3 accuracy, and 0.81 macro-F1—"
        "a substantial improvement over an earlier 69.4% baseline after introducing "
        "AdamW, label smoothing, L2 regularisation, and stronger augmentation. The "
        "system was developed iteratively with explicit attention to training–inference "
        "preprocessing parity, class-imbalance mitigation, and cross-platform "
        "deployment on emulator and physical Android hardware. AquaScan demonstrates "
        "how paper-faithful CNN deployment, uncertainty-aware UX, and domain-specific "
        "RAG can be unified in a single aquaculture decision-support application "
        "suitable for academic and field prototyping.",
    )

    add_para(
        doc,
        "Index Terms—Aquaculture, fish disease classification, convolutional neural "
        "networks, mobile health informatics, retrieval-augmented generation, FAISS, "
        "Flutter, FastAPI, offline inference, South Asian freshwater fisheries.",
    )

    add_heading(doc, "I. INTRODUCTION")
    add_para(
        doc,
        "Aquaculture contributes substantially to food security and rural livelihoods "
        "across South Asia, but disease outbreaks remain a primary source of stock "
        "mortality and production loss. Early visual signs—skin hemorrhage, fin rot, "
        "gill inflammation, fungal patches, or abnormal swimming—are frequently "
        "misinterpreted by inexperienced operators, leading to delayed treatment, "
        "incorrect chemical application, and unnecessary culling. Convolutional neural "
        "networks have shown strong performance on fish disease image classification "
        "when trained on curated photographic datasets, yet practical adoption requires "
        "more than raw accuracy: mobile capture workflows, explainable recommendations, "
        "offline operability, and contextual follow-up questioning.",
    )
    add_para(
        doc,
        "Tamut, Mangang, and Chingakham (2025) proposed a compact CNN for classifying "
        "seven freshwater fish disease categories using images from South Asian "
        "aquaculture settings. Their model applies three convolutional blocks with "
        "batch normalization and dropout, followed by a dense layer and softmax output, "
        "achieving competitive accuracy on a multi-class photographic dataset. AquaScan "
        "extends this research line into a deployable software system rather than a "
        "standalone notebook experiment. The project integrates: (1) a reproducible "
        "training and evaluation pipeline; (2) a FastAPI inference service with dual "
        "model backends; (3) a Flutter Material 3 client for Android, iOS, desktop, "
        "and web; and (4) a local RAG assistant specialized for aquaculture management, "
        "water quality, biosecurity, and interpretation of CNN outputs.",
    )
    add_para(
        doc,
        "Unlike cloud-dependent veterinary AI prototypes, AquaScan is designed for "
        "offline-first operation. After initial dependency installation and optional "
        "embedding-model download, predictions and assistant responses are served "
        "entirely from local files on the host machine. The mobile application "
        "communicates only with a LAN or loopback FastAPI endpoint; no telemetry or "
        "third-party inference APIs are invoked during normal use. This design is "
        "particularly relevant for pond-side environments with intermittent connectivity.",
    )
    add_para(
        doc,
        "The principal contributions of this work are as follows:",
    )
    add_bullets(
        doc,
        [
            "A full-stack implementation of the Tamut et al. (2025) CNN preprocessing "
            "contract (150×150 resize, divide-by-255 normalization, EXIF-aware rotation) "
            "shared identically across training, evaluation, and inference.",
            "A production-oriented prediction service with Keras .h5 and ONNX fallback, "
            "confidence tiers (High/Medium/Low), top-3 softmax reporting, and structured "
            "disease metadata (cause, symptoms, treatment, prevention).",
            "A fish-presence gate using ImageNet MobileNetV2 to reduce out-of-distribution "
            "false positives when users photograph non-fish objects.",
            "An embedded aquaculture RAG assistant with FAISS retrieval, SQLite chat "
            "history, CNN prediction context injection, and selectable local LLMs.",
            "A cross-platform Flutter UX with scan history, disease library, authentication, "
            "theme control, accessibility semantics, and backend health polling.",
        ],
    )

    add_heading(doc, "II. RELATED WORK")
    add_para(
        doc,
        "Image-based fish disease recognition has been studied extensively using deep "
        "learning. Tamut et al. (2025) reported a CNN tailored to South Asian freshwater "
        "species with explicit layer dimensions and training callbacks (EarlyStopping, "
        "ReduceLROnPlateau), providing the architectural baseline for AquaScan. "
        "Complementary work in aquaculture informatics has explored transfer learning "
        "with MobileNet and ResNet families to improve generalization on limited "
        "datasets—a direction also supported in AquaScan's trainer through an optional "
        "MobileNetV2 fine-tuning mode while preserving the 150×150 deployment contract.",
    )
    add_para(
        doc,
        "Retrieval-augmented generation has emerged as a practical method for grounding "
        "large language model responses in domain documents, reducing unsupported "
        "generation in sensitive advisory contexts. Prior mental-health and medical "
        "chatbot systems have combined FAISS vector stores, MiniLM embeddings, and "
        "local GGUF inference; AquaScan adapts this pattern to aquaculture extension "
        "materials, disease catalogs, and prediction-interpretation guides. The assistant "
        "deliberately excludes crisis-keyword automation and appointment booking present "
        "in earlier mental-health prototypes, focusing instead on husbandry education "
        "and post-scan explanation.",
    )
    add_para(
        doc,
        "Mobile deployment of ML services commonly relies on either on-device TFLite "
        "models or a local/edge HTTP backend. AquaScan selects the latter to "
        "accommodate TensorFlow/Keras and llama.cpp workloads on a developer machine or "
        "edge PC, while keeping the Flutter client lightweight. This mirrors common "
        "patterns in research prototypes where GPU memory and model hot-swapping are "
        "easier to manage server-side.",
    )

    add_heading(doc, "III. METHODOLOGY")
    add_heading(doc, "A. Dataset and Disease Taxonomy", level=2)
    add_para(
        doc,
        "The classification corpus is the public Kaggle dataset Freshwater Fish Disease "
        "Aquaculture in South Asia, containing 2,444 labeled images partitioned into "
        "Train and Test folders. Seven disease folders align with the paper taxonomy: "
        "Bacterial Red Disease, Bacterial Aeromoniasis, Bacterial Gill Disease, Fungal "
        "Saprolegniasis, Healthy Fish, Parasitic Disease, and Viral White Tail Disease. "
        "Canonical label mappings are stored in model/labels.json, linking Kaggle folder "
        "names to stable disease IDs and display names used by both backend and Flutter UI.",
    )

    table = doc.add_table(rows=8, cols=3)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text = "Class Index"
    hdr[1].text = "Disease Name"
    hdr[2].text = "Type"
    rows_data = [
        ("0", "Bacterial Red Disease", "Bacterial"),
        ("1", "Bacterial Aeromoniasis", "Bacterial"),
        ("2", "Bacterial Gill Disease", "Bacterial"),
        ("3", "Fungal Saprolegniasis", "Fungal"),
        ("4", "Healthy Fish", "Healthy"),
        ("5", "Parasitic Disease", "Parasitic"),
        ("6", "Viral White Tail Disease", "Viral"),
    ]
    for i, (a, b, c) in enumerate(rows_data, start=1):
        table.rows[i].cells[0].text = a
        table.rows[i].cells[1].text = b
        table.rows[i].cells[2].text = c
    add_table_caption(doc, "I", "DISEASE CLASSES (TAMUT ET AL. TAXONOMY)")

    add_para(
        doc,
        "Rich textual metadata for each class—including cause, symptoms, treatment, and "
        "prevention—is maintained in assets/diseases.json and surfaced in the result screen "
        "and disease library. A synthetic eighth record, No Fish Detected (ID 0), is used "
        "only at inference time when the fish gate or uncertainty logic rejects an input.",
    )

    add_heading(doc, "B. CNN Architecture and Training Pipeline", level=2)
    add_para(
        doc,
        "The default paper-exact architecture (paper_cnn) comprises:",
    )
    add_bullets(
        doc,
        [
            "Conv2D(128, 5×5, ReLU) → MaxPool → BatchNorm → Dropout(0.25)",
            "Conv2D(64, 3×3, ReLU) → MaxPool → BatchNorm → Dropout(0.25)",
            "Conv2D(32, 3×3, ReLU) → MaxPool → BatchNorm → Dropout(0.25)",
            "Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(7, Softmax)",
        ],
    )
    add_para(
        doc,
        "Training uses categorical cross-entropy with optional label smoothing (0.05), "
        "Adam or AdamW optimizers, EarlyStopping on validation loss (patience 15), "
        "ReduceLROnPlateau (factor 0.5), and ModelCheckpoint persistence. Class "
        "imbalance is addressed via effective-number weighting (Cui et al., β=0.999) "
        "or inverse-frequency weights. Data augmentation includes rotation, shifts, "
        "horizontal flip, zoom, and brightness jitter; a strong augmentation preset is "
        "enabled by default. A recommended MobileNetV2 transfer-learning path freezes "
        "an ImageNet backbone for warmup epochs, then fine-tunes the final layers at a "
        "reduced learning rate while keeping the 150×150 RGB input contract unchanged.",
    )

    add_heading(doc, "C. Preprocessing as a Single Source of Truth", level=2)
    add_para(
        doc,
        "A central design decision is that backend/app/core/preprocessing.py defines the "
        "only valid image transformation pipeline. Steps are: (1) decode bytes or path; "
        "(2) EXIF-aware rotation via ImageOps.exif_transpose; (3) convert to RGB; "
        "(4) resize to 150×150 with LANCZOS; (5) scale to float32 [0,1]; (6) add batch "
        "dimension. This module is imported by training generators, evaluation batches, "
        "and live /predict inference. Earlier project iterations suffered severe "
        "confidence collapse on real camera photos because training used stretch-resize "
        "while inference used center-crop—an error eliminated by unifying both paths.",
    )

    add_heading(doc, "D. Inference Service and Uncertainty Handling", level=2)
    add_para(
        doc,
        "PredictionService orchestrates model loading, metadata lookup, and response "
        "assembly. Model resolution order: model/model.h5 (primary), backend/app/ml/model.h5 "
        "(legacy package path), then fish_disease_classifier.onnx via ONNX Runtime. "
        "Environment variable AQUASCAN_MODEL_PREFERENCE=onnx can force ONNX-only loading.",
    )
    add_para(
        doc,
        "Inference proceeds in stages:",
    )
    add_bullets(
        doc,
        [
            "Quality heuristic: flag images smaller than 64 px per side.",
            "Fish gate (optional): MobileNetV2 ImageNet classifier; sum probability on 17 "
            "fish-related classes; reject if fish_score < 0.05 (configurable).",
            "CNN softmax: argmax class, entropy computation, top-3 reporting.",
            "Uncertainty filter: if max confidence < 0.20 AND entropy > 1.90 (for 7 "
            "classes, max entropy ≈ 1.946), return No Fish Detected instead of a weak guess.",
            "Confidence tiering: High ≥ 0.70, Medium ≥ 0.45, else Low with tier-aware recommendations.",
        ],
    )

    add_heading(doc, "E. FastAPI Backend and API Surface", level=2)
    add_para(
        doc,
        "The FastAPI application (version 3.0.0) exposes REST endpoints for health, "
        "model metadata, disease catalogues, and multipart image prediction. CORS is "
        "permissive for local development. Request timing is logged via middleware; "
        "uploads are capped at 15 MB and restricted to JPEG, PNG, WebP, GIF, and BMP. "
        "When assistant_session_id is supplied on /predict, the latest prediction JSON "
        "is published into the assistant's prediction-context store for RAG grounding.",
    )

    api_table = doc.add_table(rows=9, cols=2)
    api_table.style = "Table Grid"
    api_table.rows[0].cells[0].text = "Endpoint"
    api_table.rows[0].cells[1].text = "Purpose"
    api_rows = [
        ("GET /health", "model_ready, backend type, assistant status"),
        ("GET /model/info", "paths, num_classes, image_size"),
        ("GET /diseases", "full disease metadata list"),
        ("POST /predict", "multipart image → PredictionResponse"),
        ("POST /assistant/chat/stream", "NDJSON token stream from local LLM"),
        ("GET /assistant/history/{id}", "SQLite-persisted chat"),
        ("POST /assistant/reindex", "rebuild FAISS knowledge index"),
        ("POST /assistant/prediction-context", "attach CNN result to session"),
    ]
    for i, (a, b) in enumerate(api_rows, start=1):
        api_table.rows[i].cells[0].text = a
        api_table.rows[i].cells[1].text = b
    add_table_caption(doc, "II", "PRIMARY API ENDPOINTS")

    add_heading(doc, "F. Flutter Client Architecture", level=2)
    add_para(
        doc,
        "The Flutter application (Dart 3.3+, Material 3) bootstraps ThemeController, "
        "AuthService, AssistantController, and BackendStatusService before rendering "
        "AppShell. ApiPredictionService resolves backend URLs automatically: Android "
        "emulator → 10.0.2.2:8000; desktop/web → 127.0.0.1:8000; physical devices → "
        "configurable LAN IP or --dart-define=AQUASCAN_BACKEND_URL. Multipart uploads "
        "set explicit MIME types via http_parser to avoid Android 400 errors from "
        "application/octet-stream defaults.",
    )
    add_para(
        doc,
        "Key screens and services include:",
    )
    add_bullets(
        doc,
        [
            "HomeScreen: scan actions, statistics, research card, photo tips.",
            "ScanPreviewScreen: pre-upload validation before inference.",
            "ResultScreen: confidence ring, tier badge, warnings, top-3 bars, recommendations.",
            "DiseaseLibraryScreen: browse diseases.json without scanning.",
            "ScanHistoryService (ChangeNotifier): persistent history with tab-safe updates.",
            "AssistantPanel: collapsible overlay with streaming Markdown, citations, regenerate.",
            "AuthService: Firebase Google Sign-In plus guest mode via shared_preferences.",
            "ThemeController: system/light/dark with context extensions (surfaceCard, textPrimary).",
        ],
    )

    add_heading(doc, "G. Aquaculture RAG Assistant", level=2)
    add_para(
        doc,
        "The assistant package (aquaculture_assistant/) implements a framework-neutral "
        "pipeline mounted into the same FastAPI process:",
    )
    add_para(
        doc,
        "knowledge documents + diseases.json + labels.json → format loaders (CSV, JSON, "
        "TXT, Markdown, PDF, Excel) → overlapping text chunks (default size 400, overlap 80) "
        "→ all-MiniLM-L6-v2 normalized embeddings → FAISS inner-product index with JSON "
        "metadata (no pickle) → top-k retrieval enriched with latest CNN prediction → "
        "model-native prompt (Qwen, Mistral, or Llama) → llama.cpp token stream → NDJSON → "
        "Flutter assistant panel.",
    )
    add_para(
        doc,
        "Knowledge sources under AI chatbot/knowledge/ include water_quality.md, "
        "biosecurity_and_management.md, species_and_feeding.md, "
        "common_conditions_and_differentials.md, and prediction_interpretation.md. "
        "Source fingerprints (SHA-256, size, mtime) trigger automatic index rebuilds when "
        "documents change. Chat history persists in assistant_history.sqlite3; only one "
        "GGUF model resides in memory at a time to conserve RAM on shared TensorFlow + "
        "LLM workloads.",
    )
    add_para(
        doc,
        "Supported local models (Q4_K_M quantization): Meta-Llama-3.1-8B-Instruct, "
        "Mistral-7B-Instruct-v0.2, and Qwen3-8B. GPU layer offload is configurable "
        "(AQUASCAN_ASSISTANT_GPU_LAYERS); low-memory mode keeps weights mmap-backed.",
    )

    add_heading(doc, "H. Evaluation Methodology", level=2)
    add_para(
        doc,
        "backend/train/evaluate.py scores the held-out Test folder with batch preprocessing "
        "identical to inference. Reported metrics include accuracy, log loss, balanced "
        "accuracy, Cohen's kappa, Matthews correlation coefficient, top-3 accuracy, "
        "per-class precision/recall/F1, expected calibration error (ECE), multiclass Brier "
        "score, and a 7×7 confusion matrix exported as CSV and JSON. This mirrors the "
        "reporting style of Tamut et al. (2025) and supports architecture comparison "
        "between paper_cnn and mobilenet_v2 checkpoints.",
    )

    add_para(
        doc,
        "Fig. 1 summarises the end-to-end AquaScan workflow from mobile image capture "
        "through FastAPI inference, uncertainty filtering, and optional RAG explanation.",
    )
    add_figure_caption(
        doc,
        1,
        "System architecture workflow. The Flutter client uploads an image to /predict; "
        "the backend applies fish-gate screening, CNN classification, confidence tiering, "
        "and optionally publishes the result to the aquaculture assistant session.",
    )
    add_para(
        doc,
        "Fig. 2 depicts the aquaculture assistant pipeline, adapted from a prior "
        "mental-health RAG prototype but re-engineered for fish-health knowledge bases.",
    )
    add_figure_caption(
        doc,
        2,
        "Aquaculture RAG assistant workflow. Domain documents are chunked, embedded with "
        "MiniLM, indexed in FAISS, enriched with CNN prediction context, and answered by "
        "a locally selected GGUF model streamed to the Flutter assistant panel.",
    )

    add_heading(doc, "IV. RESULTS AND DISCUSSION")
    add_heading(doc, "A. Quantitative Classification Results", level=2)
    add_para(
        doc,
        "All metrics below were measured on the untouched Kaggle Test/ folder (700 images, "
        "seven classes) using backend/train/evaluate.py with preprocessing identical to "
        "live inference. Tamut et al. (2025) report approximately 99.71% accuracy under "
        "their experimental protocol; AquaScan's measured results on this independently "
        "reimplemented pipeline are lower but honestly reported and suitable for "
        "deployment-oriented comparison.",
    )

    metrics_table = doc.add_table(rows=5, cols=5)
    metrics_table.style = "Table Grid"
    mhdr = metrics_table.rows[0].cells
    mhdr[0].text = "Model Artifact"
    mhdr[1].text = "Test Accuracy"
    mhdr[2].text = "Top-3 Accuracy"
    mhdr[3].text = "Macro F1"
    mhdr[4].text = "Notes"
    metrics_rows = [
        (
            "Legacy ONNX fallback",
            "~79.0%",
            "—",
            "—",
            "Pre-Keras PyTorch export; bundled for resilience",
        ),
        (
            "Keras v1 (paper CNN)",
            "69.4%",
            "92.0%",
            "0.70",
            "Baseline before regularisation upgrade",
        ),
        (
            "Keras v2 (promoted)",
            "81.3%",
            "95.1%",
            "0.81",
            "AdamW + label smoothing + L2 + strong augmentation",
        ),
        (
            "Tamut et al. (2025)",
            "~99.7%",
            "—",
            "—",
            "Published benchmark; different training protocol",
        ),
    ]
    for i, row in enumerate(metrics_rows, start=1):
        for j, val in enumerate(row):
            metrics_table.rows[i].cells[j].text = val
    add_table_caption(doc, "III", "HELD-OUT TEST SET CLASSIFICATION PERFORMANCE")

    add_para(
        doc,
        "The v2 checkpoint improved top-1 accuracy by 11.9 percentage points over v1 "
        "while raising top-3 accuracy from 92.0% to 95.1%. Macro-F1 rose from 0.70 to "
        "0.81, indicating better per-class balance under inverse-frequency and "
        "effective-number weighting. The weakest v2 class in manual review was Parasitic "
        "Disease (precision ≈ 0.69), while early v1 errors included low recall on "
        "Bacterial Red Disease (0.55) and low precision on Viral White Tail Disease "
        "(0.48)—both partially mitigated after regularisation and augmentation tuning.",
    )

    add_para(
        doc,
        "Fig. 3 illustrates how confidence tiers translate softmax outputs into "
        "operator-facing guidance on the ResultScreen.",
    )
    add_figure_caption(
        doc,
        3,
        "Confidence-tiered result presentation. High (≥70%), Medium (≥45%), and Low "
        "tiers change the recommended next step from direct treatment guidance to "
        "confirmatory re-photography or explicit non-diagnostic warnings.",
    )

    add_heading(doc, "B. System Integration Outcomes", level=2)
    add_para(
        doc,
        "End-to-end integration was validated through backend smoke tests "
        "(tests/smoke_test.py), CLI prediction (tests/predict_cli.py), pytest API suites, "
        "and Flutter widget/unit tests covering auth and assistant controller behaviour. "
        "The most impactful accuracy improvement observed during development was not a "
        "hyperparameter tweak but preprocessing alignment: after unifying resize semantics "
        "and adding EXIF correction, real-camera confidence scores rose substantially "
        "compared to the mismatched pipeline era documented in project challenges.",
    )

    add_heading(doc, "C. Qualitative UX and Decision Support", level=2)
    add_para(
        doc,
        "The combination of confidence tiers and textual recommendations changes how "
        "operators act on model output. High-confidence predictions surface direct treatment "
        "guidance from diseases.json; medium-confidence results encourage confirmatory "
        "re-photography; low-confidence results explicitly warn against treating the output "
        "as a diagnosis. The assistant layer allows follow-up questions—e.g., quarantine "
        "protocol after a Bacterial Red Disease hit—grounded in retrieved extension "
        "documents and the stored CNN JSON context.",
    )

    add_heading(doc, "D. Architectural Trade-offs", level=2)
    add_para(
        doc,
        "Offline-first local LLM inference trades cloud scalability for privacy and "
        "pond-side availability. Loading TensorFlow, ONNX Runtime, MiniLM, FAISS, and an "
        "8B GGUF model concurrently stresses consumer hardware; the project mitigates this "
        "via lazy loading, single-model residency, and optional fake_mode for UI testing. "
        "The fish gate adds latency on first MobileNetV2 load but prevents egregious "
        "misclassifications on non-fish inputs—a common failure mode of closed-set classifiers.",
    )

    add_heading(doc, "V. CHALLENGES AND LIMITATIONS")
    add_heading(doc, "A. Training Time and Overfitting", level=2)
    add_para(
        doc,
        "CNN training runs required 30–60+ minutes per experiment. Early models exhibited "
        "severe overfitting (high training accuracy, poor test generalization). Mitigations "
        "included AdamW, cosine-style schedules, stronger augmentation, early stopping, "
        "label smoothing, and effective-number class weights.",
    )
    add_heading(doc, "B. Cross-Stack Integration", level=2)
    add_para(
        doc,
        "Bridging PyTorch/Keras training artifacts, ONNX export, FastAPI, and Flutter "
        "introduced MIME-type, tensor-shape, and connectivity issues—particularly on "
        "physical Android devices requiring LAN IP configuration and cleartext HTTP allowances.",
    )
    add_heading(doc, "C. Dataset and Clinical Validity", level=2)
    add_para(
        doc,
        "The Kaggle corpus is photographic and geographically constrained. Model outputs "
        "are decision-support hints, not veterinary diagnoses. Viral conditions especially "
        "require laboratory confirmation; the UI text reflects this limitation.",
    )
    add_heading(doc, "D. Assistant Evaluation Gap", level=2)
    add_para(
        doc,
        "Unlike the reference mental-health chatbot paper, AquaScan does not yet include "
        "a formal multi-model LLM scoring benchmark (relevance, conciseness, context use) "
        "for aquaculture prompts. Assistant quality remains qualitatively reviewed.",
    )
    add_heading(doc, "E. Dependency and Environment Fragility", level=2)
    add_para(
        doc,
        "Python 3.14 incompatibility with compiled wheels, CUDA DLL requirements for GPU "
        "onnxruntime, and Windows symlink warnings in Flutter builds required pinned "
        "interpreter versions (3.13) and documented setup scripts (run_backend.bat).",
    )

    add_heading(doc, "VI. CONCLUSION AND FUTURE WORK")
    add_para(
        doc,
        "This paper presented AquaScan, an integrated freshwater fish disease detection "
        "platform combining a Tamut et al.-aligned CNN, uncertainty-aware inference, a "
        "Material 3 Flutter client, and a local RAG aquaculture assistant. By centralizing "
        "preprocessing, supporting dual model backends, and coupling CNN outputs with "
        "retrieval-grounded explanations, the system addresses both classification and "
        "operator education in a single offline-capable stack.",
    )
    add_para(
        doc,
        "Future work includes: on-device TFLite deployment for disconnected phones; "
        "expanded multilingual knowledge corpora; rigorous LLM evaluation on aquaculture "
        "prompt suites; active-learning loops for farmer-submitted misclassifications; "
        "and prospective field trials with fisheries extension officers.",
    )

    add_heading(doc, "REFERENCES")
    refs = [
        "[1] J. Tamut, Y. A. Mangang, and C. Chingakham, "
        '"Image Classification of Freshwater Fish Diseases in South Asian '
        'Aquaculture Using Convolutional Neural Network," Aquaculture Journal, '
        "vol. 5, no. 1, p. 6, 2025. https://doi.org/10.3390/aquacj5010006",
        "[2] Subir Biswas, Freshwater Fish Disease Aquaculture in South Asia, "
        "Kaggle Dataset, 2023.",
        "[3] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "
        '"Class-Balanced Loss Based on Effective Number of Samples," '
        "in Proc. IEEE/CVF CVPR, 2019.",
        "[4] J. Johnson, M. Douze, and H. Jégou, "
        '"Billion-Scale Similarity Search with GPUs," '
        "IEEE Trans. Big Data, vol. 7, no. 3, pp. 535–547, 2021.",
        "[5] Sentence Transformers, "
        '"sentence-transformers/all-MiniLM-L6-v2," Hugging Face, 2021.',
        "[6] G. Nagarajan, "
        '"PolicyBot — Reliable Question Answering over Policy Documents," '
        "arXiv:2511.13489, 2025.",
        "[7] Flutter Team, "
        '"Flutter: Cross-Platform UI Toolkit," Google, 2026.',
        "[8] FastAPI Project, "
        '"FastAPI: Modern Python Web Framework," 2026.',
        "[9] ggml-org, "
        '"llama.cpp: LLM Inference in C/C++," GitHub, 2026.',
        '[10] Z. Khawaja and J. Belisle-Pipon, '
        '"Your Robot Therapist Is Not Your Therapist: Understanding the Role '
        'of AI-Powered Mental Health Chatbots," Frontiers in Digital Health, '
        "vol. 5, 2023.",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(4)

    return doc


def main() -> None:
    out = Path(r"c:\Users\rachi\Downloads\AquaScan_Research_Paper.docx")
    doc = build_document()
    doc.save(str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
