from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from train.train_classifier import DEFAULT_MODEL_DIR, build_model, export_to_onnx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "train" / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the best trained classifier checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "best_model.pt",
        help="Path to the trained checkpoint produced by train_classifier.py.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Training artifacts directory used for class_map/audit files.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory where ONNX artifacts are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_map = checkpoint["class_map"]
    model_name = checkpoint["model_name"]
    image_size = int(checkpoint["image_size"])

    model = build_model(model_name, num_classes=len(class_map["classes"]))
    model.load_state_dict(checkpoint["state_dict"])

    args.model_dir.mkdir(parents=True, exist_ok=True)
    export_to_onnx(
        model=model,
        output_path=args.model_dir / "fish_disease_classifier.onnx",
        image_size=image_size,
    )

    with (args.model_dir / "class_map.json").open("w", encoding="utf-8") as file_handle:
        json.dump(class_map, file_handle, indent=2)

    audit_summary = args.artifacts_dir / "audit_summary.json"
    if audit_summary.exists():
        shutil.copy2(audit_summary, args.model_dir / "audit_summary.json")

    print(f"ONNX model: {args.model_dir / 'fish_disease_classifier.onnx'}")
    print(f"Class map: {args.model_dir / 'class_map.json'}")


if __name__ == "__main__":
    main()
