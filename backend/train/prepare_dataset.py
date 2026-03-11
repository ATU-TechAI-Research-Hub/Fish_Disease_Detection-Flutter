from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXTENSION_PRIORITY = {".png": 3, ".jpg": 2, ".jpeg": 1}
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Freshwater_Fish_Disease_Aquaculture_in_south_asia"
)
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "train" / "artifacts"
DEFAULT_DISEASES_FILE = PROJECT_ROOT / "assets" / "diseases.json"

CLASS_TO_DISEASE_ID = {
    "Bacterial Red disease": 1,
    "Bacterial diseases - Aeromoniasis": 2,
    "Bacterial gill disease": 3,
    "Fungal diseases Saprolegniasis": 4,
    "Healthy Fish": 5,
    "Parasitic diseases": 6,
    "Viral diseases White tail disease": 7,
}

CLASS_ORDER = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease",
]


def file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def choose_representative(group: pd.DataFrame) -> pd.Series:
    ranked = group.assign(
        split_priority=group["source_split"].map({"Train": 1, "Test": 0}),
        ext_priority=group["extension"].map(EXTENSION_PRIORITY).fillna(0),
    ).sort_values(
        by=["split_priority", "file_size", "ext_priority"],
        ascending=[False, False, False],
    )
    return ranked.iloc[0]


def load_disease_lookup(diseases_file: Path) -> dict[int, dict[str, object]]:
    with diseases_file.open("r", encoding="utf-8") as file_handle:
        diseases = json.load(file_handle)
    return {entry["id"]: entry for entry in diseases}


def scan_split(split_dir: Path, split_name: str) -> list[dict[str, object]]:
    if not split_dir.exists():
        return []

    records: list[dict[str, object]] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            records.append(
                {
                    "source_split": split_name,
                    "folder_name": class_dir.name,
                    "stem": file_path.stem,
                    "extension": file_path.suffix.lower(),
                    "file_path": str(file_path.resolve()),
                    "file_size": file_path.stat().st_size,
                }
            )
    return records


def resolve_train_csv(train_csv: Path, dataset_root: Path) -> dict[str, int]:
    if not train_csv.exists():
        return {
            "csv_rows": 0,
            "csv_direct_path_matches": 0,
            "csv_resolved_to_train": 0,
            "csv_missing_rows": 0,
        }

    csv_df = pd.read_csv(train_csv)
    direct_matches = 0
    resolved_to_train = 0
    missing_rows = 0

    for _, row in csv_df.iterrows():
        folder_name = str(row["Folder Name"])
        image_filename = str(row["Image Filename"])
        image_path = dataset_root / str(row["Image Path"])
        train_candidate = dataset_root / "Train" / folder_name / image_filename
        if image_path.exists():
            direct_matches += 1
        elif train_candidate.exists():
            resolved_to_train += 1
        else:
            missing_rows += 1

    return {
        "csv_rows": int(len(csv_df)),
        "csv_direct_path_matches": direct_matches,
        "csv_resolved_to_train": resolved_to_train,
        "csv_missing_rows": missing_rows,
    }


def build_class_map(diseases_lookup: dict[int, dict[str, object]]) -> dict[str, object]:
    classes = []
    for class_index, folder_name in enumerate(CLASS_ORDER):
        disease_id = CLASS_TO_DISEASE_ID[folder_name]
        disease = diseases_lookup[disease_id]
        classes.append(
            {
                "class_index": class_index,
                "folder_name": folder_name,
                "disease_id": disease_id,
                "disease_name": disease["name"],
            }
        )

    return {
        "image_size": IMAGE_SIZE,
        "mean": IMAGE_MEAN,
        "std": IMAGE_STD,
        "classes": classes,
    }


def build_clean_manifest(
    raw_df: pd.DataFrame, diseases_lookup: dict[int, dict[str, object]]
) -> tuple[pd.DataFrame, dict[str, object]]:
    clean_rows: list[dict[str, object]] = []
    exact_overlap_groups = 0

    for (folder_name, stem), group in raw_df.groupby(["folder_name", "stem"], sort=True):
        if folder_name not in CLASS_TO_DISEASE_ID:
            raise ValueError(f"Unexpected class folder: {folder_name}")

        chosen = choose_representative(group)
        train_group = group[group["source_split"] == "Train"]
        test_group = group[group["source_split"] == "Test"]

        has_test_duplicate = not test_group.empty
        if not train_group.empty and not test_group.empty:
            train_choice = choose_representative(train_group)
            test_choice = choose_representative(test_group)
            train_hash = file_md5(Path(str(train_choice["file_path"])))
            test_hash = file_md5(Path(str(test_choice["file_path"])))
            exact_overlap = train_hash == test_hash
            if exact_overlap:
                exact_overlap_groups += 1
        else:
            exact_overlap = False

        disease_id = CLASS_TO_DISEASE_ID[folder_name]
        disease = diseases_lookup[disease_id]

        clean_rows.append(
            {
                "group_key": f"{folder_name}::{stem}",
                "folder_name": folder_name,
                "class_index": CLASS_ORDER.index(folder_name),
                "disease_id": disease_id,
                "disease_name": disease["name"],
                "image_stem": stem,
                "file_path": chosen["file_path"],
                "file_name": Path(str(chosen["file_path"])).name,
                "extension": chosen["extension"],
                "file_size": int(chosen["file_size"]),
                "source_split": chosen["source_split"],
                "variant_count": int(len(group)),
                "train_variant_count": int(len(train_group)),
                "test_variant_count": int(len(test_group)),
                "has_test_duplicate": has_test_duplicate,
                "exact_test_overlap": exact_overlap,
            }
        )

    clean_df = pd.DataFrame(clean_rows).sort_values(
        by=["class_index", "image_stem"]
    ).reset_index(drop=True)

    duplicate_groups_by_class = (
        raw_df.groupby("folder_name")["stem"]
        .apply(lambda stems: int(stems.duplicated().sum()))
        .to_dict()
    )

    summary = {
        "raw_file_count": int(len(raw_df)),
        "canonical_image_count": int(len(clean_df)),
        "removed_duplicate_variants": int(len(raw_df) - len(clean_df)),
        "test_subset_groups": int(clean_df["has_test_duplicate"].sum()),
        "exact_test_overlap_groups": exact_overlap_groups,
        "duplicate_groups_by_class": duplicate_groups_by_class,
    }
    return clean_df, summary


def split_manifest(clean_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_df, holdout_df = train_test_split(
        clean_df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=clean_df["class_index"],
    )

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=holdout_df["class_index"],
    )

    return {
        "train": train_df.sort_values(by=["class_index", "image_stem"]).reset_index(
            drop=True
        ),
        "val": val_df.sort_values(by=["class_index", "image_stem"]).reset_index(
            drop=True
        ),
        "test": test_df.sort_values(by=["class_index", "image_stem"]).reset_index(
            drop=True
        ),
    }


def ensure_dataset_artifacts(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    diseases_file: Path = DEFAULT_DISEASES_FILE,
) -> dict[str, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    diseases_lookup = load_disease_lookup(diseases_file)
    class_map = build_class_map(diseases_lookup)

    train_records = scan_split(dataset_root / "Train", "Train")
    test_records = scan_split(dataset_root / "Test", "Test")
    raw_df = pd.DataFrame(train_records + test_records)
    if raw_df.empty:
        raise FileNotFoundError(
            f"No image files found under {dataset_root / 'Train'} or {dataset_root / 'Test'}."
        )

    clean_df, clean_summary = build_clean_manifest(raw_df, diseases_lookup)
    split_frames = split_manifest(clean_df)
    csv_summary = resolve_train_csv(dataset_root / "Train.csv", dataset_root)

    paths = {
        "clean_manifest": artifacts_dir / "clean_manifest.csv",
        "train_split": artifacts_dir / "train_split.csv",
        "val_split": artifacts_dir / "val_split.csv",
        "test_split": artifacts_dir / "test_split.csv",
        "class_map": artifacts_dir / "class_map.json",
        "audit_summary": artifacts_dir / "audit_summary.json",
    }

    clean_df.to_csv(paths["clean_manifest"], index=False)
    split_frames["train"].to_csv(paths["train_split"], index=False)
    split_frames["val"].to_csv(paths["val_split"], index=False)
    split_frames["test"].to_csv(paths["test_split"], index=False)

    with paths["class_map"].open("w", encoding="utf-8") as file_handle:
        json.dump(class_map, file_handle, indent=2)

    split_counts = {
        split_name: {
            "row_count": int(len(frame)),
            "class_counts": {
                str(int(class_index)): int(count)
                for class_index, count in frame["class_index"].value_counts()
                .sort_index()
                .to_dict()
                .items()
            },
        }
        for split_name, frame in split_frames.items()
    }

    audit_summary = {
        "dataset_root": str(dataset_root),
        "raw_train_files": int(len(train_records)),
        "raw_test_files": int(len(test_records)),
        "raw_class_counts": {
            split_name: {
                folder_name: int(count)
                for folder_name, count in frame["folder_name"]
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            }
            for split_name, frame in {
                "Train": pd.DataFrame(train_records),
                "Test": pd.DataFrame(test_records),
            }.items()
        },
        "class_order": CLASS_ORDER,
        "csv_resolution": csv_summary,
        "clean_manifest": clean_summary,
        "split_counts": split_counts,
    }

    with paths["audit_summary"].open("w", encoding="utf-8") as file_handle:
        json.dump(audit_summary, file_handle, indent=2)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fish disease dataset and create clean train/val/test manifests."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing Train/ Test/ and Train.csv.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory where manifest and audit artifacts are written.",
    )
    parser.add_argument(
        "--diseases-file",
        type=Path,
        default=DEFAULT_DISEASES_FILE,
        help="Path to diseases.json metadata file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_dataset_artifacts(
        dataset_root=args.dataset_root,
        artifacts_dir=args.artifacts_dir,
        diseases_file=args.diseases_file,
    )
    print("Dataset artifacts generated:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
