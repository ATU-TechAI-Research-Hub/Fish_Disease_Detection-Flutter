"""Fine-tune YOLO11x-seg on a Roboflow YOLO segmentation export.

Run from ``backend``:

    python -m train.shrimp.train_yolo11_seg --data path/to/data.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def _validate_dataset(data_yaml: Path) -> None:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload.get("names")
    if isinstance(names, dict):
        ordered_names = [str(names[key]) for key in sorted(names)]
    elif isinstance(names, list):
        ordered_names = [str(name) for name in names]
    else:
        ordered_names = []
    if ordered_names != ["shrimp"]:
        raise ValueError(
            "The dataset must contain exactly one segmentation class named "
            f"'shrimp'; found {ordered_names!r}."
        )
    for split in ("train", "val", "test"):
        if split not in payload:
            raise ValueError(f"data.yaml is missing the {split!r} split.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AquaScan single-class shrimp segmentation model."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-weights", default="yolo11x-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--project", type=Path, default=Path("train/shrimp/runs"))
    parser.add_argument("--name", default="yolo11x-seg-shrimp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degrees", type=float, default=90.0)
    parser.add_argument("--shear", type=float, default=10.0)
    parser.add_argument("--saturation", type=float, default=0.25)
    parser.add_argument("--horizontal-flip", type=float, default=0.5)
    parser.add_argument("--vertical-flip", type=float, default=0.5)
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="Disabled by default because it is not part of the paper methodology.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("app/ml/shrimp/best.pt"),
        help="Copy the best validated checkpoint here after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_dataset(args.data)

    from ultralytics import YOLO

    model = YOLO(args.base_weights, task="segment")
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.image_size,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        task="segment",
        plots=True,
        degrees=args.degrees,
        shear=args.shear,
        hsv_h=0.0,
        hsv_s=args.saturation,
        hsv_v=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=args.horizontal_flip,
        flipud=args.vertical_flip,
        mosaic=args.mosaic,
        mixup=0.0,
        copy_paste=0.0,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but best.pt was not found in {save_dir}")
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output_checkpoint)
    print(f"Best shrimp checkpoint copied to: {args.output_checkpoint.resolve()}")


if __name__ == "__main__":
    main()
