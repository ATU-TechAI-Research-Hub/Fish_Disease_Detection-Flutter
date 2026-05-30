"""Label / class-mapping utilities.

Loads `model/labels.json` (the canonical class list) and exposes it as a typed
structure used by training, evaluation, and the API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ClassEntry:
    class_index: int
    folder_name: str
    disease_id: int
    disease_name: str


@dataclass(frozen=True)
class LabelMap:
    image_size: int
    classes: List[ClassEntry]
    model_name: str
    paper_reference: str

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_names(self) -> List[str]:
        """Disease names ordered by class_index."""
        return [c.disease_name for c in self.classes]

    @property
    def folder_names(self) -> List[str]:
        """Dataset subfolder names ordered by class_index."""
        return [c.folder_name for c in self.classes]

    def by_index(self, class_index: int) -> ClassEntry:
        try:
            return self.classes[class_index]
        except IndexError as exc:
            raise KeyError(
                f"class_index {class_index} is out of range "
                f"(num_classes={len(self.classes)})."
            ) from exc

    def by_folder(self, folder_name: str) -> ClassEntry:
        for entry in self.classes:
            if entry.folder_name == folder_name:
                return entry
        raise KeyError(f"Unknown folder name: {folder_name!r}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_size": self.image_size,
            "model_name": self.model_name,
            "paper_reference": self.paper_reference,
            "classes": [
                {
                    "class_index": c.class_index,
                    "folder_name": c.folder_name,
                    "disease_id": c.disease_id,
                    "disease_name": c.disease_name,
                }
                for c in self.classes
            ],
        }


def load_label_map(path: str | Path) -> LabelMap:
    """Load and validate `labels.json`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"labels.json not found at: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError(f"{path} does not contain a non-empty 'classes' list.")

    classes: List[ClassEntry] = []
    for item in raw_classes:
        classes.append(
            ClassEntry(
                class_index=int(item["class_index"]),
                folder_name=str(item["folder_name"]),
                disease_id=int(item["disease_id"]),
                disease_name=str(item["disease_name"]),
            )
        )
    classes.sort(key=lambda c: c.class_index)

    expected_indices = list(range(len(classes)))
    actual_indices = [c.class_index for c in classes]
    if actual_indices != expected_indices:
        raise ValueError(
            f"Class indices in {path} must be 0..{len(classes) - 1} without gaps. "
            f"Got: {actual_indices}"
        )

    return LabelMap(
        image_size=int(payload.get("image_size", 150)),
        classes=classes,
        model_name=str(payload.get("model_name", "paper_cnn_keras")),
        paper_reference=str(payload.get("paper_reference", "")),
    )
