"""Audit the fish-disease dataset before training.

High image-classification scores can be invalid when copies (or lightly edited
copies) of one photograph occur in both Train/ and Test/. This script reports:

* per-class counts and imbalance ratio
* unreadable images
* exact duplicate groups (SHA-256)
* perceptually similar Train/Test pairs (64-bit difference hash)
* duplicates carrying conflicting disease labels

The audit is read-only. It never deletes or moves source images.

Usage:
    python -m train.audit_dataset
    python -m train.audit_dataset --near-duplicate-distance 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core import load_label_map

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Freshwater_Fish_Disease_Aquaculture_in_south_asia"
)
DEFAULT_LABELS_FILE = PROJECT_ROOT / "model" / "labels.json"
DEFAULT_REPORT = PROJECT_ROOT / "backend" / "outputs" / "dataset_audit.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    split: str
    class_name: str
    width: int
    height: int
    sha256: str
    difference_hash: int


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    """Return a compact perceptual hash; nearby values imply similar images."""
    gray = ImageOps.grayscale(ImageOps.exif_transpose(image))
    resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _read_record(
    path: Path, dataset_root: Path, split: str, class_name: str
) -> ImageRecord:
    data = path.read_bytes()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        perceptual_hash = _difference_hash(image)
    return ImageRecord(
        path=str(path.relative_to(dataset_root)),
        split=split,
        class_name=class_name,
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
        difference_hash=perceptual_hash,
    )


def audit_dataset(
    dataset_root: Path,
    labels_file: Path,
    output: Path,
    near_duplicate_distance: int = 4,
) -> dict[str, object]:
    label_map = load_label_map(labels_file)
    records: list[ImageRecord] = []
    unreadable: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    for split in ("Train", "Test"):
        split_root = dataset_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_root}")
        for entry in label_map.classes:
            class_root = split_root / entry.folder_name
            if not class_root.exists():
                unreadable.append(
                    {
                        "path": str(class_root.relative_to(dataset_root)),
                        "error": "class directory is missing",
                    }
                )
                continue
            for path in _iter_images(class_root):
                try:
                    record = _read_record(
                        path, dataset_root, split, entry.disease_name
                    )
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    unreadable.append(
                        {
                            "path": str(path.relative_to(dataset_root)),
                            "error": str(exc),
                        }
                    )
                    continue
                records.append(record)
                counts[f"{split}/{entry.disease_name}"] += 1

    exact_groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        exact_groups[record.sha256].append(record)

    exact_duplicates = []
    conflicting_labels = []
    for sha256, group in exact_groups.items():
        if len(group) < 2:
            continue
        item = {
            "sha256": sha256,
            "files": [asdict(record) for record in group],
            "cross_split": len({record.split for record in group}) > 1,
        }
        exact_duplicates.append(item)
        if len({record.class_name for record in group}) > 1:
            conflicting_labels.append(item)

    train_records = [record for record in records if record.split == "Train"]
    test_records = [record for record in records if record.split == "Test"]
    exact_cross_pairs = {
        (left.path, right.path)
        for left in train_records
        for right in test_records
        if left.sha256 == right.sha256
    }
    near_duplicates = []
    for train_record in train_records:
        for test_record in test_records:
            if (train_record.path, test_record.path) in exact_cross_pairs:
                continue
            distance = _hamming_distance(
                train_record.difference_hash, test_record.difference_hash
            )
            if distance <= near_duplicate_distance:
                near_duplicates.append(
                    {
                        "train_path": train_record.path,
                        "train_class": train_record.class_name,
                        "test_path": test_record.path,
                        "test_class": test_record.class_name,
                        "hamming_distance": distance,
                        "conflicting_labels": (
                            train_record.class_name != test_record.class_name
                        ),
                    }
                )

    train_class_counts = [
        counts[f"Train/{entry.disease_name}"] for entry in label_map.classes
    ]
    nonzero_train_counts = [count for count in train_class_counts if count > 0]
    imbalance_ratio = (
        max(nonzero_train_counts) / min(nonzero_train_counts)
        if nonzero_train_counts
        else 0.0
    )

    report: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "total_readable_images": len(records),
        "counts": dict(sorted(counts.items())),
        "train_imbalance_ratio_max_to_min": round(imbalance_ratio, 4),
        "unreadable": unreadable,
        "exact_duplicate_groups": exact_duplicates,
        "exact_cross_split_group_count": sum(
            bool(group["cross_split"]) for group in exact_duplicates
        ),
        "near_duplicate_distance": near_duplicate_distance,
        "near_duplicate_cross_split_pairs": near_duplicates,
        "conflicting_exact_duplicate_groups": conflicting_labels,
        "recommendation": (
            "Remove or group cross-split duplicates before comparing models. "
            "Manually review perceptual matches; difference hashes can produce "
            "false positives for visually simple images."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dataset balance, corruption, and split leakage."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=4,
        help="Maximum 64-bit dHash Hamming distance (0-64).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.near_duplicate_distance <= 64:
        raise ValueError("--near-duplicate-distance must be between 0 and 64.")
    report = audit_dataset(
        dataset_root=args.dataset_root,
        labels_file=args.labels_file,
        output=args.output,
        near_duplicate_distance=args.near_duplicate_distance,
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit report to: {args.output}")


if __name__ == "__main__":
    main()
