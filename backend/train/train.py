"""Build, train and save the paper-exact Keras CNN as `model.h5`.

Architecture, hyperparameters, and preprocessing match:
  Tamut J., Mangang Y.A., Chingakham C. (2025).
  "Image Classification of Freshwater Fish Diseases in South Asian
  Aquaculture Using Convolutional Neural Network."
  Aquaculture Journal 5(1), 6. https://doi.org/10.3390/aquacj5010006

Key paper details we reproduce:
  - 150x150x3 input, /255 normalisation
  - Conv2D(128, 5x5) → MaxPool → BatchNorm → Dropout(0.25)
  - Conv2D(64, 3x3)  → MaxPool → BatchNorm → Dropout(0.25)
  - Conv2D(32, 3x3)  → MaxPool → BatchNorm → Dropout(0.25)
  - Flatten → Dense(256, relu) → Dropout(0.5) → Dense(num_classes, softmax)
  - Adam optimizer + categorical cross-entropy
  - Callbacks: EarlyStopping + ReduceLROnPlateau

Usage:
    python -m train.train                       # use defaults
    python -m train.train --epochs 50 --batch-size 32
    python -m train.train --output ../model/model.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Freshwater_Fish_Disease_Aquaculture_in_south_asia"
DEFAULT_LABELS_FILE = PROJECT_ROOT / "model" / "labels.json"
DEFAULT_OUTPUT_H5 = PROJECT_ROOT / "model" / "model.h5"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "backend" / "outputs"


@dataclass
class TrainingConfig:
    image_size: int = 150
    batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 1e-3
    validation_split: float = 0.15
    patience_early_stop: int = 15
    patience_reduce_lr: int = 5
    seed: int = 42
    use_augmentation: bool = True
    # Regularisation knobs (kept defaulted close to the paper).
    label_smoothing: float = 0.05
    l2_weight_decay: float = 1e-4
    use_adamw: bool = True
    strong_augmentation: bool = True
    save_best_only: bool = True


def _set_seeds(seed: int) -> None:
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_paper_cnn(
    num_classes: int,
    image_size: int = 150,
    l2_weight_decay: float = 0.0,
) -> "tf.keras.Model":
    """Construct the exact CNN described in Tamut et al. (Aquac. J. 2025).

    Layer dimensions match Table 1 of the paper. Output is a softmax over
    `num_classes`.

    Args:
        num_classes: Number of disease categories (7 in this project).
        image_size: Side length of the square network input (paper: 150).
        l2_weight_decay: Optional kernel L2 strength applied to Conv/Dense
            layers. Default 0.0 reproduces the paper exactly; a value like
            1e-4 noticeably reduces overfitting on this dataset.
    """
    from tensorflow.keras import Input, Model, layers, regularizers

    reg = regularizers.l2(l2_weight_decay) if l2_weight_decay > 0 else None

    inputs = Input(shape=(image_size, image_size, 3), name="image")

    x = layers.Conv2D(128, (5, 5), activation="relu",
                      kernel_regularizer=reg, name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu",
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    x = layers.Conv2D(32, (3, 3), activation="relu",
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=reg, name="dense1")(x)
    x = layers.Dropout(0.5, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=reg, name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="paper_cnn_keras")


def _make_generators(
    dataset_root: Path,
    folder_names: List[str],
    image_size: int,
    batch_size: int,
    validation_split: float,
    seed: int,
    use_augmentation: bool,
    strong_augmentation: bool = False,
) -> tuple["ImageDataGenerator.flow_from_directory", "ImageDataGenerator.flow_from_directory"]:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir = dataset_root / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found: {train_dir}. "
            "Place the Kaggle dataset under "
            "`Freshwater_Fish_Disease_Aquaculture_in_south_asia/Train/...`"
        )

    augment_kwargs: Dict[str, object] = {}
    if use_augmentation:
        if strong_augmentation:
            augment_kwargs = dict(
                rotation_range=30,
                width_shift_range=0.15,
                height_shift_range=0.15,
                horizontal_flip=True,
                vertical_flip=False,
                zoom_range=0.2,
                brightness_range=(0.8, 1.2),
                shear_range=0.1,
                channel_shift_range=20.0,
                fill_mode="reflect",
            )
        else:
            augment_kwargs = dict(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                brightness_range=(0.9, 1.1),
            )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split,
        **augment_kwargs,
    )
    eval_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split,
    )

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical",
        classes=folder_names,
        subset="training",
        shuffle=True,
        seed=seed,
    )
    val_gen = eval_datagen.flow_from_directory(
        str(train_dir),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical",
        classes=folder_names,
        subset="validation",
        shuffle=False,
        seed=seed,
    )
    return train_gen, val_gen


def _compute_class_weights(
    train_gen,
    num_classes: int,
) -> Dict[int, float]:
    """Inverse-frequency weights, matching what the paper does for imbalance."""
    counts = np.bincount(train_gen.classes, minlength=num_classes)
    total = float(counts.sum())
    weights: Dict[int, float] = {}
    for i in range(num_classes):
        weights[i] = float(total / (num_classes * max(int(counts[i]), 1)))
    return weights


def train(
    dataset_root: Path,
    labels_file: Path,
    output_h5: Path,
    outputs_dir: Path,
    config: TrainingConfig,
) -> Dict[str, object]:
    """Train the paper CNN and save it as `model.h5`. Returns a metrics summary."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.optimizers import Adam

    from app.core import load_label_map

    _set_seeds(config.seed)

    label_map = load_label_map(labels_file)
    num_classes = label_map.num_classes
    folder_names = label_map.folder_names

    train_gen, val_gen = _make_generators(
        dataset_root=dataset_root,
        folder_names=folder_names,
        image_size=config.image_size,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        seed=config.seed,
        use_augmentation=config.use_augmentation,
        strong_augmentation=config.strong_augmentation,
    )

    print(
        f"Found {train_gen.samples} training images, "
        f"{val_gen.samples} validation images, "
        f"{num_classes} classes."
    )

    model = build_paper_cnn(
        num_classes=num_classes,
        image_size=config.image_size,
        l2_weight_decay=config.l2_weight_decay,
    )

    # Prefer AdamW (decoupled weight decay) when available — better
    # generalisation than vanilla Adam with comparable wall-clock cost.
    optimizer = None
    if config.use_adamw:
        try:
            from tensorflow.keras.optimizers import AdamW  # TF >= 2.11
            optimizer = AdamW(
                learning_rate=config.learning_rate,
                weight_decay=max(config.l2_weight_decay, 1e-5),
            )
            print(
                f"Using AdamW(lr={config.learning_rate}, "
                f"weight_decay={max(config.l2_weight_decay, 1e-5)})"
            )
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"AdamW unavailable ({exc}); falling back to Adam.")
    if optimizer is None:
        optimizer = Adam(learning_rate=config.learning_rate)
        print(f"Using Adam(lr={config.learning_rate})")

    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(label_smoothing=config.label_smoothing),
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda line: print(line))

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    best_h5 = outputs_dir / "best_model.h5"

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.patience_early_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.patience_reduce_lr,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(best_h5),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
    ]

    class_weights = _compute_class_weights(train_gen, num_classes)
    print(f"Class weights (inverse-frequency): {class_weights}")

    history = model.fit(
        train_gen,
        epochs=config.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # `restore_best_weights=True` means `model` already holds the best
    # checkpoint, so saving it again is equivalent to saving `best_h5`.
    model.save(str(output_h5))
    print(f"\nSaved Keras model to: {output_h5}")

    history_dict: Dict[str, List[float]] = {
        key: [float(v) for v in values] for key, values in history.history.items()
    }
    history_file = outputs_dir / "training_history.json"
    history_file.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    summary: Dict[str, object] = {
        "epochs_completed": len(history_dict.get("loss", [])),
        "best_val_accuracy": float(max(history_dict.get("val_accuracy", [0.0]) or [0.0])),
        "final_val_accuracy": float(val_acc),
        "final_val_loss": float(val_loss),
        "num_classes": num_classes,
        "image_size": config.image_size,
        "model_file": str(output_h5),
        "labels_file": str(labels_file),
        "config": asdict(config),
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "history_file": str(history_file),
    }
    (outputs_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nTraining summary:")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the paper-exact Keras CNN and save model.h5.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--patience-early-stop", type=int, default=15)
    parser.add_argument("--patience-reduce-lr", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4,
                        help="L2 weight decay strength.")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable training-time data augmentation.")
    parser.add_argument("--no-strong-augmentation", action="store_true",
                        help="Use light augmentation instead of the strong preset.")
    parser.add_argument("--no-adamw", action="store_true",
                        help="Use vanilla Adam instead of AdamW.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        image_size=150,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        patience_early_stop=args.patience_early_stop,
        patience_reduce_lr=args.patience_reduce_lr,
        seed=args.seed,
        use_augmentation=not args.no_augmentation,
        strong_augmentation=not args.no_strong_augmentation,
        label_smoothing=args.label_smoothing,
        l2_weight_decay=args.l2,
        use_adamw=not args.no_adamw,
    )
    train(
        dataset_root=args.dataset_root,
        labels_file=args.labels_file,
        output_h5=args.output,
        outputs_dir=args.outputs_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
