"""Build a local YOLO segmentation dataset from the public research export.

The AquaScan paper dataset is published on Roboflow Universe under CC BY 4.0:

https://universe.roboflow.com/yathish-8yqj0/shrimp-o6rvc-8j1pm-comdo/dataset/2

Roboflow's source metadata contains YOLO polygon annotations. This utility
converts an exported metadata payload into the single-class layout expected by
``train_yolo11_seg.py`` and downloads the corresponding source images.

The metadata input may be either:

* a JSON array of source records;
* a gzip-compressed JSON array;
* a base64-encoded gzip payload; or
* the JSON wrapper produced by a Chrome DevTools ``Runtime.evaluate`` result.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import logging
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image


LOGGER = logging.getLogger("prepare_shrimp_dataset")
DATASET_URL = (
    "https://universe.roboflow.com/yathish-8yqj0/"
    "shrimp-o6rvc-8j1pm-comdo/dataset/2"
)
SOURCE_URL = "https://source.roboflow.com/{owner}/{source_id}/original.jpg"
SPLIT_NAMES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def _unwrap_cdp_payload(value: Any) -> Any:
    """Descend through a CDP response wrapper until its returned value."""
    current = value
    for _ in range(8):
        if isinstance(current, dict):
            if (
                current.get("type") in {"string", "object"}
                and "value" in current
            ):
                current = current["value"]
                continue
            if set(current).issubset({"result", "sessionId"}) and "result" in current:
                current = current["result"]
                continue
        break
    return current


def load_metadata(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)

    try:
        value: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = raw.decode("ascii").strip()

    value = _unwrap_cdp_payload(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            compressed = base64.b64decode(text, validate=True)
            value = json.loads(gzip.decompress(compressed).decode("utf-8"))

    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("Metadata must decode to a JSON array of source records.")
    return value


def _normalise_labels(annotations: dict[str, Any]) -> tuple[str, int]:
    """Merge all polygon classes and rewrite every valid instance as class 0."""
    lines: list[str] = []
    for annotation in annotations.values():
        if not isinstance(annotation, str):
            continue
        for raw_line in annotation.splitlines():
            parts = raw_line.split()
            # class id plus at least three x/y polygon points
            if len(parts) < 7 or (len(parts) - 1) % 2:
                LOGGER.warning("Skipping malformed polygon with %d fields", len(parts))
                continue
            try:
                coordinates = [float(value) for value in parts[1:]]
            except ValueError:
                LOGGER.warning("Skipping polygon with non-numeric coordinates")
                continue
            if not all(-0.001 <= value <= 1.001 for value in coordinates):
                LOGGER.warning("Skipping polygon with out-of-range coordinates")
                continue
            lines.append("0 " + " ".join(parts[1:]))
    text = "\n".join(lines)
    return (text + "\n" if text else ""), len(lines)


def _valid_existing_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def _download_image(row: dict[str, Any], destination: Path, timeout: float) -> None:
    if _valid_existing_image(destination):
        return
    owner = str(row.get("owner", "")).strip()
    source_id = str(row.get("id", "")).strip()
    if not owner or not source_id:
        raise ValueError("Source record is missing owner or id")

    response = requests.get(
        SOURCE_URL.format(owner=owner, source_id=source_id),
        headers={"User-Agent": "AquaScan-Dataset-Builder/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "image/" not in content_type:
        raise ValueError(f"Expected an image but received {content_type!r}")

    with Image.open(io.BytesIO(response.content)) as image:
        image.verify()
    temporary = destination.with_suffix(".part")
    temporary.write_bytes(response.content)
    temporary.replace(destination)


def _safe_source_id(row: dict[str, Any]) -> str:
    source_id = str(row.get("id", "")).strip()
    if source_id and all(character.isalnum() or character in "-_" for character in source_id):
        return source_id
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]


def build_dataset(
    rows: list[dict[str, Any]],
    output: Path,
    workers: int,
    timeout: float,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[dict[str, Any], Path, Path, str, int]] = []
    split_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()

    for row in rows:
        raw_split = str(row.get("split", "")).lower()
        if raw_split not in SPLIT_NAMES:
            raise ValueError(f"Unsupported dataset split {raw_split!r}")
        split = SPLIT_NAMES[raw_split]
        source_id = _safe_source_id(row)
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        label_text, instances = _normalise_labels(row.get("annotations", {}))
        image_path = image_dir / f"{source_id}.jpg"
        label_path = label_dir / f"{source_id}.txt"
        prepared.append((row, image_path, label_path, label_text, instances))
        split_counts[split] += 1
        instance_counts[split] += instances

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_download_image, row, image_path, timeout): (
                row,
                image_path,
                label_path,
                label_text,
            )
            for row, image_path, label_path, label_text, _ in prepared
        }
        completed = 0
        for future in as_completed(futures):
            row, _, label_path, label_text = futures[future]
            try:
                future.result()
                label_path.write_text(label_text, encoding="utf-8")
            except Exception as exc:
                failures.append(f"{row.get('id', '<unknown>')}: {exc}")
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                LOGGER.info("Prepared %d/%d images", completed, len(futures))

    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            f"{len(failures)} image downloads failed. First failures:\n{preview}"
        )

    dataset_root = output.resolve().as_posix()
    data_yaml = {
        "path": dataset_root,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "shrimp"},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "source": DATASET_URL,
        "license": "CC BY 4.0",
        "attribution": "yathish-8yqj0 — shrimp Computer Vision Model",
        "record_count": len(rows),
        "split_counts": dict(split_counts),
        "instance_counts": dict(instance_counts),
        "class_mapping": {"0": "shrimp"},
        "note": (
            "All source annotation labels were intentionally normalised to the "
            "single deployment class 'shrimp'."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AquaScan's public Roboflow shrimp segmentation dataset."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before preparing the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    rows = load_metadata(args.metadata)
    LOGGER.info("Loaded %d source records", len(rows))
    manifest = build_dataset(rows, args.output, args.workers, args.timeout)
    LOGGER.info("Dataset ready at %s", args.output.resolve())
    LOGGER.info("Split counts: %s", manifest["split_counts"])
    LOGGER.info("Instance counts: %s", manifest["instance_counts"])


if __name__ == "__main__":
    main()
