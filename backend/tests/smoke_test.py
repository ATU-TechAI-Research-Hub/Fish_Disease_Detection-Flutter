from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_SPLIT = PROJECT_ROOT / "backend" / "train" / "artifacts" / "test_split.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the FastAPI backend.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the running FastAPI backend.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        help="Optional image path to use for the /predict test.",
    )
    parser.add_argument(
        "--test-split",
        type=Path,
        default=DEFAULT_TEST_SPLIT,
        help="CSV used to locate a sample image when --image-path is omitted.",
    )
    return parser.parse_args()


def resolve_image_path(args: argparse.Namespace) -> Path:
    if args.image_path:
        if not args.image_path.exists():
            raise FileNotFoundError(f"Image not found: {args.image_path}")
        return args.image_path

    if args.test_split.exists():
        split_df = pd.read_csv(args.test_split)
        candidate = Path(str(split_df.iloc[0]["file_path"]))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No sample image could be resolved. Pass --image-path or generate local training artifacts first."
    )


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args)

    health_response = requests.get(f"{args.base_url}/health", timeout=10)
    health_response.raise_for_status()
    print("Health:", json.dumps(health_response.json(), indent=2))

    with image_path.open("rb") as file_handle:
        predict_response = requests.post(
            f"{args.base_url}/predict",
            files={"file": (image_path.name, file_handle, "image/jpeg")},
            timeout=30,
        )

    predict_response.raise_for_status()
    print("Predict:", json.dumps(predict_response.json(), indent=2))


if __name__ == "__main__":
    main()
