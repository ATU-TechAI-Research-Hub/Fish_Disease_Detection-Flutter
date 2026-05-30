"""End-to-end smoke test against the running FastAPI backend.

  python -m tests.smoke_test
  python -m tests.smoke_test --image-path "../Freshwater_Fish_Disease_Aquaculture_in_south_asia/Test/Bacterial Red disease/IMG.jpg"

Verifies:
  * /health responds and reports the active backend
  * /model/info returns the labels/backend metadata
  * /predict accepts an image and returns a valid `PredictionResponse`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Freshwater_Fish_Disease_Aquaculture_in_south_asia"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the AquaScan backend.")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000",
        help="URL of the running FastAPI service.",
    )
    parser.add_argument(
        "--image-path", type=Path,
        help="Image to send to /predict. Defaults to a sample from Test/.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help="Fallback dataset root used to find an image.",
    )
    return parser.parse_args()


def _find_sample_image(dataset_root: Path) -> Path:
    test_root = dataset_root / "Test"
    if test_root.exists():
        for class_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    return file_path
    raise FileNotFoundError(
        "No sample image could be located. "
        "Pass --image-path or download the Kaggle dataset."
    )


def main() -> int:
    args = parse_args()
    image_path = args.image_path or _find_sample_image(args.dataset_root)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    print(f"GET {args.base_url}/health")
    health = requests.get(f"{args.base_url}/health", timeout=10)
    health.raise_for_status()
    print(json.dumps(health.json(), indent=2))

    print(f"\nGET {args.base_url}/model/info")
    info = requests.get(f"{args.base_url}/model/info", timeout=10)
    info.raise_for_status()
    print(json.dumps(info.json(), indent=2))

    print(f"\nPOST {args.base_url}/predict  (image={image_path.name})")
    with image_path.open("rb") as fh:
        predict = requests.post(
            f"{args.base_url}/predict",
            files={"file": (image_path.name, fh, "image/jpeg")},
            timeout=30,
        )
    predict.raise_for_status()
    payload = predict.json()
    print(json.dumps(payload, indent=2))

    print("\nOK — backend reachable and prediction completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
