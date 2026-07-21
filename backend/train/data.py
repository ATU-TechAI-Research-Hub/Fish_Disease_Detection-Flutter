"""Inference-aligned data loading for model training.

Every source image is decoded by ``app.core.preprocess_image`` before optional
augmentation. This guarantees that training, evaluation and API inference all
use the same EXIF correction, RGB conversion, LANCZOS resize and /255 scaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from app.core import preprocess_batch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def discover_training_split(
    train_dir: Path,
    folder_names: Sequence[str],
    validation_split: float,
    seed: int,
) -> tuple[list[Path], np.ndarray, list[Path], np.ndarray]:
    """Create a deterministic, per-class-stratified train/validation split."""
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")
    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    rng = np.random.default_rng(seed)
    train_paths: list[Path] = []
    train_labels: list[int] = []
    val_paths: list[Path] = []
    val_labels: list[int] = []

    for class_index, folder_name in enumerate(folder_names):
        class_dir = train_dir / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Required class folder is missing: {class_dir}"
            )
        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(paths) < 2:
            raise ValueError(
                f"Class {folder_name!r} needs at least two readable image "
                "files for a train/validation split."
            )

        shuffled = np.asarray(paths, dtype=object)
        rng.shuffle(shuffled)
        val_count = min(
            max(1, int(round(len(paths) * validation_split))),
            len(paths) - 1,
        )
        class_val = [Path(path) for path in shuffled[:val_count]]
        class_train = [Path(path) for path in shuffled[val_count:]]
        train_paths.extend(class_train)
        train_labels.extend([class_index] * len(class_train))
        val_paths.extend(class_val)
        val_labels.extend([class_index] * len(class_val))

    return (
        train_paths,
        np.asarray(train_labels, dtype=np.int64),
        val_paths,
        np.asarray(val_labels, dtype=np.int64),
    )


def build_augmenter(strong: bool, seed: int):
    """Build lesion-preserving geometric and illumination augmentation."""
    from tensorflow.keras import Sequential, layers

    rotation = 30.0 / 360.0 if strong else 20.0 / 360.0
    translation = 0.15 if strong else 0.10
    zoom = 0.20 if strong else 0.10
    contrast = 0.20 if strong else 0.10
    brightness = 0.20 if strong else 0.10
    return Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(
                rotation,
                fill_mode="reflect",
                seed=seed + 1,
            ),
            layers.RandomTranslation(
                translation,
                translation,
                fill_mode="reflect",
                seed=seed + 2,
            ),
            layers.RandomZoom(
                height_factor=(-zoom, zoom),
                width_factor=(-zoom, zoom),
                fill_mode="reflect",
                seed=seed + 3,
            ),
            layers.RandomContrast(contrast, seed=seed + 4),
            layers.RandomBrightness(
                brightness,
                value_range=(0.0, 1.0),
                seed=seed + 5,
            ),
        ],
        name="training_augmentation",
    )


def make_image_sequence(
    file_paths: Sequence[Path],
    labels: np.ndarray,
    num_classes: int,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    augmenter=None,
):
    """Return a Keras Sequence with ``flow_from_directory``-compatible fields."""
    import tensorflow as tf

    class FishImageSequence(tf.keras.utils.Sequence):
        def __init__(self) -> None:
            super().__init__()
            self.paths = [Path(path) for path in file_paths]
            self.filepaths = [str(path) for path in self.paths]
            self.classes = np.asarray(labels, dtype=np.int64)
            self.samples = len(self.paths)
            self.batch_size = batch_size
            self._indices = np.arange(self.samples)
            self._rng = np.random.default_rng(seed)
            if shuffle:
                self._rng.shuffle(self._indices)

        def __len__(self) -> int:
            return int(np.ceil(self.samples / self.batch_size))

        def __getitem__(self, batch_index: int):
            start = batch_index * self.batch_size
            positions = self._indices[start : start + self.batch_size]
            batch_paths = [self.paths[int(index)] for index in positions]
            tensors = preprocess_batch(batch_paths, image_size=image_size)
            if augmenter is not None:
                tensors = augmenter(tensors, training=True).numpy()
                tensors = np.clip(tensors, 0.0, 1.0).astype(np.float32)
            targets = tf.keras.utils.to_categorical(
                self.classes[positions], num_classes=num_classes
            ).astype(np.float32)
            return tensors, targets

        def on_epoch_end(self) -> None:
            if shuffle:
                self._rng.shuffle(self._indices)

    return FishImageSequence()
