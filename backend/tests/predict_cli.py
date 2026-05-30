"""Run a single image through the model directly (no FastAPI required).

Useful for verifying offline behaviour and debugging preprocessing.

    python -m tests.predict_cli --image-path path/to/fish.jpg
    python -m tests.predict_cli --image-path fish.jpg --model-h5 ../model/model.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core import load_label_map, load_model, preprocess_image


DEFAULT_LABELS_FILE = PROJECT_ROOT / "model" / "labels.json"
DEFAULT_MODEL_H5 = PROJECT_ROOT / "model" / "model.h5"
DEFAULT_ONNX_FALLBACK = PROJECT_ROOT / "backend" / "app" / "ml" / "fish_disease_classifier.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an image through the local model (no API)."
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--model-h5", type=Path, default=DEFAULT_MODEL_H5)
    parser.add_argument("--onnx-fallback", type=Path, default=DEFAULT_ONNX_FALLBACK)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.image_path.exists():
        print(f"Image not found: {args.image_path}", file=sys.stderr)
        return 1

    label_map = load_label_map(args.labels_file)
    model = load_model(
        h5_path=args.model_h5,
        onnx_path=args.onnx_fallback,
        num_classes=label_map.num_classes,
        image_size=label_map.image_size,
    )
    info = model.info()
    print(f"Backend: {info.backend} ({info.path})")

    tensor = preprocess_image(args.image_path, image_size=label_map.image_size)
    probabilities = model.predict(tensor)[0]

    top_indices = np.argsort(probabilities)[::-1][: args.top_k]
    predictions = []
    for idx in top_indices:
        entry = label_map.by_index(int(idx))
        predictions.append(
            {
                "disease_name": entry.disease_name,
                "confidence": round(float(probabilities[idx]), 4),
                "class_index": int(idx),
            }
        )

    print(json.dumps({"image": str(args.image_path), "top": predictions}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
