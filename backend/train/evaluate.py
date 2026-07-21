"""Evaluate the trained `model.h5` against the held-out Test folder.

Reports (saved to `backend/outputs/`):
  - accuracy, average loss
  - confusion matrix (CSV + JSON)
  - per-class precision / recall / F1 (matching the paper's reporting)
  - top-1 / top-3 accuracy

Usage:
    python -m train.evaluate
    python -m train.evaluate --model-h5 ../model/model.h5 --dataset-root ../...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core import (
    LabelMap,
    load_label_map,
    load_model,
    preprocess_batch,
)

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Freshwater_Fish_Disease_Aquaculture_in_south_asia"
DEFAULT_LABELS_FILE = PROJECT_ROOT / "model" / "labels.json"
DEFAULT_MODEL_H5 = PROJECT_ROOT / "model" / "model.h5"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "backend" / "outputs"
DEFAULT_ONNX_FALLBACK = PROJECT_ROOT / "backend" / "app" / "ml" / "fish_disease_classifier.onnx"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _gather_test_images(
    test_root: Path, label_map: LabelMap
) -> Tuple[List[Path], List[int]]:
    if not test_root.exists():
        raise FileNotFoundError(
            f"Test folder not found: {test_root}. "
            "Did you download the Kaggle dataset?"
        )

    image_paths: List[Path] = []
    labels: List[int] = []
    for entry in label_map.classes:
        class_dir = test_root / entry.folder_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Required test class folder is missing: {class_dir}. "
                "Evaluation would otherwise report inflated/incomplete metrics."
            )
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(file_path)
                labels.append(entry.class_index)
    return image_paths, labels


def _predict_in_batches(
    model, image_paths: List[Path], image_size: int, batch_size: int = 32
) -> np.ndarray:
    all_probs: List[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        loaded = [Image.open(p) for p in batch_paths]
        tensors = preprocess_batch(loaded, image_size=image_size)
        probs = model.predict(tensors)
        all_probs.append(np.asarray(probs, dtype=np.float32))
    if not all_probs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_probs, axis=0)


def _top_k_accuracy(probabilities: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if len(probabilities) == 0:
        return 0.0
    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([y in row for y, row in zip(y_true, top_k)]))


def _multiclass_brier_score(
    probabilities: np.ndarray, y_true: np.ndarray, num_classes: int
) -> float:
    """Mean squared probability error across all classes (lower is better)."""
    targets = np.eye(num_classes, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def _expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Top-label ECE using equal-width confidence bins.

    ECE does not measure classification accuracy. It measures whether a model
    that says "80% confident" is correct about 80% of the time. Reporting it
    alongside accuracy prevents overconfident models from looking safer than
    they are (Guo et al., ICML 2017).
    """
    if len(probabilities) == 0:
        return 0.0
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == y_true
    boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        # Include confidence=1.0 in the final bin.
        mask = (confidences >= lower) & (
            confidences <= upper if index == num_bins - 1 else confidences < upper
        )
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(correct[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        ece += float(np.mean(mask)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def evaluate(
    model_h5: Path,
    onnx_fallback: Path,
    dataset_root: Path,
    labels_file: Path,
    outputs_dir: Path,
    batch_size: int,
) -> Dict[str, object]:
    label_map = load_label_map(labels_file)
    model = load_model(
        h5_path=model_h5,
        onnx_path=onnx_fallback,
        num_classes=label_map.num_classes,
        image_size=label_map.image_size,
    )
    info = model.info()
    print(f"Evaluating backend={info.backend} (path={info.path})")

    test_root = dataset_root / "Test"
    image_paths, labels = _gather_test_images(test_root, label_map)
    if not image_paths:
        raise RuntimeError(
            f"No test images discovered under {test_root}. "
            "Make sure the Kaggle dataset is in place."
        )
    print(f"Found {len(image_paths)} test images.")

    probabilities = _predict_in_batches(
        model, image_paths, image_size=label_map.image_size, batch_size=batch_size
    )
    predictions = np.argmax(probabilities, axis=1)
    y_true = np.asarray(labels, dtype=np.int64)

    target_names = label_map.class_names
    accuracy = float(accuracy_score(y_true, predictions))
    balanced_accuracy = float(balanced_accuracy_score(y_true, predictions))
    mcc = float(matthews_corrcoef(y_true, predictions))
    kappa = float(cohen_kappa_score(y_true, predictions))
    cm = confusion_matrix(y_true, predictions, labels=list(range(len(target_names))))
    report = classification_report(
        y_true,
        predictions,
        target_names=target_names,
        labels=list(range(len(target_names))),
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    loss_value = float(
        log_loss(
            y_true,
            probabilities,
            labels=list(range(len(target_names))),
        )
    )

    top3 = _top_k_accuracy(probabilities, y_true, k=3)
    brier = _multiclass_brier_score(
        probabilities, y_true, num_classes=len(target_names)
    )
    ece = _expected_calibration_error(probabilities, y_true)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(outputs_dir / "confusion_matrix.csv")
    (outputs_dir / "confusion_matrix.json").write_text(
        json.dumps(
            {"labels": target_names, "matrix": cm.tolist()},
            indent=2,
        ),
        encoding="utf-8",
    )
    (outputs_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    summary: Dict[str, object] = {
        "model_backend": info.backend,
        "model_path": info.path,
        "device": info.device,
        "num_test_images": len(image_paths),
        "test_accuracy": round(accuracy, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "matthews_correlation_coefficient": round(mcc, 6),
        "cohen_kappa": round(kappa, 6),
        "test_loss": round(loss_value, 6),
        "top3_accuracy": round(top3, 6),
        "multiclass_brier_score": round(brier, 6),
        "expected_calibration_error": round(ece, 6),
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 6),
                "recall": round(report[name]["recall"], 6),
                "f1": round(report[name]["f1-score"], 6),
                "support": int(report[name]["support"]),
            }
            for name in target_names
        },
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 6),
            "recall": round(report["macro avg"]["recall"], 6),
            "f1": round(report["macro avg"]["f1-score"], 6),
        },
        "weighted_avg": {
            "precision": round(report["weighted avg"]["precision"], 6),
            "recall": round(report["weighted avg"]["recall"], 6),
            "f1": round(report["weighted avg"]["f1-score"], 6),
        },
        "confusion_matrix_file": str(outputs_dir / "confusion_matrix.csv"),
        "classification_report_file": str(outputs_dir / "classification_report.json"),
    }
    (outputs_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # Keep a copy keyed by the model filename for A/B comparisons.
    model_tag = model_h5.stem.replace(".", "_")
    versioned = outputs_dir / f"evaluation_summary_{model_tag}.json"
    versioned.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["versioned_summary_file"] = str(versioned)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model.h5 on Test/ folder.")
    parser.add_argument("--model-h5", type=Path, default=DEFAULT_MODEL_H5)
    parser.add_argument("--onnx-fallback", type=Path, default=DEFAULT_ONNX_FALLBACK)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_h5=args.model_h5,
        onnx_fallback=args.onnx_fallback,
        dataset_root=args.dataset_root,
        labels_file=args.labels_file,
        outputs_dir=args.outputs_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
