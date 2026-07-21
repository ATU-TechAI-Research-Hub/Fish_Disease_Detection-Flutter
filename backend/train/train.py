"""Train and save a fish-disease classifier as `model.h5`.

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
    python -m train.train                       # pretrained MobileNetV2
    python -m train.train --architecture paper_cnn
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
    # Research-backed transfer-learning preset. The input/output contract stays
    # 150x150 RGB /255 → seven probabilities, so deployment is unchanged.
    architecture: str = "mobilenet_v2"
    imagenet_weights: bool = True
    warmup_epochs: int = 8
    fine_tune_layers: int = 30
    fine_tune_learning_rate: float = 1e-5
    class_weight_strategy: str = "effective_number"
    effective_number_beta: float = 0.999


def _set_seeds(seed: int) -> None:
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Older TensorFlow versions may not expose this API; seeded execution
        # is still preferable to silently changing the experiment seed.
        pass


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


def build_mobilenet_v2(
    num_classes: int,
    image_size: int = 150,
    l2_weight_decay: float = 1e-4,
    imagenet_weights: bool = True,
) -> tuple["tf.keras.Model", "tf.keras.Model"]:
    """Build a compact ImageNet-pretrained transfer-learning classifier.

    Recent fish-disease studies consistently find that pretrained feature
    extractors generalise better than small CNNs on limited aquaculture
    datasets. MobileNetV2 is used here because it is accurate, lightweight and
    easy to deploy. The backend still supplies [0, 1] RGB tensors; the Lambda
    layer performs MobileNetV2's required [-1, 1] conversion inside the model.
    """
    from tensorflow.keras import Input, Model, layers, regularizers
    from tensorflow.keras.applications import MobileNetV2

    inputs = Input(shape=(image_size, image_size, 3), name="image")
    normalized = layers.Rescaling(
        scale=2.0, offset=-1.0, name="mobilenet_preprocess"
    )(inputs)
    backbone = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet" if imagenet_weights else None,
        input_tensor=normalized,
    )
    backbone.trainable = False

    reg = regularizers.l2(l2_weight_decay) if l2_weight_decay > 0 else None
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.35, name="head_dropout")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=reg,
        name="head_dense",
    )(x)
    x = layers.Dropout(0.30, name="classifier_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=reg,
        name="output",
    )(x)
    model = Model(inputs=inputs, outputs=outputs, name="mobilenet_v2_fish_disease")
    return model, backbone


def _make_generators(
    dataset_root: Path,
    folder_names: List[str],
    image_size: int,
    batch_size: int,
    validation_split: float,
    seed: int,
    use_augmentation: bool,
    strong_augmentation: bool = False,
) -> tuple[object, object]:
    from train.data import (
        build_augmenter,
        discover_training_split,
        make_image_sequence,
    )

    train_dir = dataset_root / "Train"
    train_paths, train_labels, val_paths, val_labels = discover_training_split(
        train_dir=train_dir,
        folder_names=folder_names,
        validation_split=validation_split,
        seed=seed,
    )
    augmenter = (
        build_augmenter(strong=strong_augmentation, seed=seed)
        if use_augmentation
        else None
    )
    train_gen = make_image_sequence(
        file_paths=train_paths,
        labels=train_labels,
        num_classes=len(folder_names),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        augmenter=augmenter,
    )
    val_gen = make_image_sequence(
        file_paths=val_paths,
        labels=val_labels,
        num_classes=len(folder_names),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 1000,
        augmenter=None,
    )
    return train_gen, val_gen


def _compute_class_weights(
    train_gen,
    num_classes: int,
    strategy: str = "effective_number",
    beta: float = 0.999,
) -> Dict[int, float]:
    """Compute class weights without allowing a majority class to dominate.

    ``effective_number`` follows Cui et al. (CVPR 2019). It is less aggressive
    than raw inverse frequency when a minority class has only a few samples,
    which usually makes optimisation more stable on small imbalanced datasets.
    """
    counts = np.bincount(train_gen.classes, minlength=num_classes)
    total = float(counts.sum())
    if strategy == "none":
        return {index: 1.0 for index in range(num_classes)}
    if strategy == "inverse_frequency":
        return {
            index: float(total / (num_classes * max(int(count), 1)))
            for index, count in enumerate(counts)
        }
    if strategy != "effective_number":
        raise ValueError(
            "class_weight_strategy must be one of: "
            "effective_number, inverse_frequency, none."
        )
    if not 0.0 <= beta < 1.0:
        raise ValueError("effective_number_beta must be in [0, 1).")

    safe_counts = np.maximum(counts.astype(np.float64), 1.0)
    effective_weights = (1.0 - beta) / (1.0 - np.power(beta, safe_counts))
    # Preserve the average loss scale so learning-rate behaviour stays stable.
    effective_weights /= np.mean(effective_weights)
    return {
        index: float(weight)
        for index, weight in enumerate(effective_weights)
    }


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

    # AdamW already applies decoupled weight decay. Applying kernel L2 at the
    # same time regularizes every update twice and can cause underfitting.
    kernel_l2 = 0.0 if config.use_adamw else config.l2_weight_decay
    backbone = None
    if config.architecture == "paper_cnn":
        model = build_paper_cnn(
            num_classes=num_classes,
            image_size=config.image_size,
            l2_weight_decay=kernel_l2,
        )
    elif config.architecture == "mobilenet_v2":
        model, backbone = build_mobilenet_v2(
            num_classes=num_classes,
            image_size=config.image_size,
            l2_weight_decay=kernel_l2,
            imagenet_weights=config.imagenet_weights,
        )
    else:
        raise ValueError(
            "architecture must be either 'mobilenet_v2' or 'paper_cnn'."
        )

    def compile_model(learning_rate: float) -> None:
        """Compile (or recompile after unfreezing) with identical metrics."""
        optimizer = None
        if config.use_adamw:
            try:
                from tensorflow.keras.optimizers import AdamW  # TF >= 2.11
                optimizer = AdamW(
                    learning_rate=learning_rate,
                    weight_decay=max(config.l2_weight_decay, 1e-5),
                )
                print(
                    f"Using AdamW(lr={learning_rate}, "
                    f"weight_decay={max(config.l2_weight_decay, 1e-5)})"
                )
            except Exception as exc:  # pragma: no cover - fallback path
                print(f"AdamW unavailable ({exc}); falling back to Adam.")
        if optimizer is None:
            optimizer = Adam(learning_rate=learning_rate)
            print(f"Using Adam(lr={learning_rate})")

        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(
                label_smoothing=config.label_smoothing
            ),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(
                    k=min(3, num_classes), name="top3_accuracy"
                ),
            ],
        )

    compile_model(config.learning_rate)
    model.summary(print_fn=lambda line: print(line))

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dataset_root_resolved = dataset_root.resolve()

    def relative_dataset_path(raw_path: str) -> str:
        path = Path(raw_path).resolve()
        try:
            return str(path.relative_to(dataset_root_resolved))
        except ValueError:
            return str(path)

    split_manifest = {
        "seed": config.seed,
        "validation_split": config.validation_split,
        "train_files": [
            relative_dataset_path(path) for path in train_gen.filepaths
        ],
        "validation_files": [
            relative_dataset_path(path) for path in val_gen.filepaths
        ],
    }
    split_file = outputs_dir / "validation_split.json"
    split_file.write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    best_h5 = outputs_dir / "best_model.h5"
    fine_tuned_h5 = outputs_dir / "best_model_fine_tuned.h5"

    def make_callbacks(checkpoint: Path, minimum_lr: float):
        # Validation loss is a better overfitting/calibration signal than raw
        # accuracy, especially with class weighting and label smoothing.
        return [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.patience_early_stop,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=config.patience_reduce_lr,
                min_lr=minimum_lr,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(checkpoint),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=False,
                verbose=0,
            ),
        ]

    class_weights = _compute_class_weights(
        train_gen,
        num_classes,
        strategy=config.class_weight_strategy,
        beta=config.effective_number_beta,
    )
    print(
        f"Class weights ({config.class_weight_strategy}): {class_weights}"
    )

    first_phase_epochs = (
        min(config.warmup_epochs, config.epochs)
        if backbone is not None
        else config.epochs
    )
    print(
        f"\nPhase 1: training classifier head for {first_phase_epochs} epoch(s)."
        if backbone is not None
        else f"\nTraining paper CNN for up to {first_phase_epochs} epoch(s)."
    )
    history = model.fit(
        train_gen,
        epochs=first_phase_epochs,
        validation_data=val_gen,
        callbacks=make_callbacks(best_h5, minimum_lr=1e-6),
        class_weight=class_weights,
        verbose=1,
    )
    history_dict: Dict[str, List[float]] = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    phase1_metrics = model.evaluate(val_gen, verbose=0, return_dict=True)

    remaining_epochs = max(config.epochs - first_phase_epochs, 0)
    if (
        backbone is not None
        and remaining_epochs > 0
        and config.fine_tune_layers > 0
    ):
        print(
            f"\nPhase 2: fine-tuning the final "
            f"{min(config.fine_tune_layers, len(backbone.layers))} backbone "
            f"layers for up to {remaining_epochs} epoch(s) at "
            f"lr={config.fine_tune_learning_rate}."
        )
        backbone.trainable = True
        freeze_until = max(len(backbone.layers) - config.fine_tune_layers, 0)
        for index, layer in enumerate(backbone.layers):
            # Keep BatchNorm statistics frozen on this small dataset. Updating
            # them during fine-tuning commonly damages pretrained features.
            layer.trainable = (
                index >= freeze_until
                and not isinstance(layer, tf.keras.layers.BatchNormalization)
            )
        compile_model(config.fine_tune_learning_rate)
        fine_history = model.fit(
            train_gen,
            epochs=remaining_epochs,
            validation_data=val_gen,
            callbacks=make_callbacks(fine_tuned_h5, minimum_lr=1e-7),
            class_weight=class_weights,
            verbose=1,
        )
        for key, values in fine_history.history.items():
            history_dict.setdefault(key, []).extend(float(v) for v in values)

        fine_metrics = model.evaluate(val_gen, verbose=0, return_dict=True)
        if float(fine_metrics["loss"]) > float(phase1_metrics["loss"]):
            print(
                "Fine-tuning did not improve validation loss; restoring the "
                "frozen-backbone checkpoint."
            )
            model = tf.keras.models.load_model(str(best_h5), compile=False)
        else:
            phase1_metrics = fine_metrics

    model.save(str(output_h5))
    print(f"\nSaved Keras model to: {output_h5}")

    history_file = outputs_dir / "training_history.json"
    history_file.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")

    # Compile=False restoration above means we evaluate from predictions rather
    # than depending on serialized optimizer state.
    val_probabilities = model.predict(val_gen, verbose=0)
    val_predictions = np.argmax(val_probabilities, axis=1)
    val_acc = float(np.mean(val_predictions == val_gen.classes))
    clipped = np.clip(val_probabilities, 1e-7, 1.0)
    val_targets = tf.keras.utils.to_categorical(
        val_gen.classes, num_classes=num_classes
    )
    val_loss = float(
        np.mean(-np.sum(val_targets * np.log(clipped), axis=1))
    )
    summary: Dict[str, object] = {
        "epochs_completed": len(history_dict.get("loss", [])),
        "best_val_accuracy": float(max(history_dict.get("val_accuracy", [0.0]) or [0.0])),
        "final_val_accuracy": float(val_acc),
        "final_val_loss": float(val_loss),
        "num_classes": num_classes,
        "image_size": config.image_size,
        "architecture": config.architecture,
        "kernel_l2_applied": kernel_l2,
        "model_file": str(output_h5),
        "labels_file": str(labels_file),
        "config": asdict(config),
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "history_file": str(history_file),
        "validation_split_file": str(split_file),
    }
    (outputs_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nTraining summary:")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an ImageNet-pretrained MobileNetV2 (recommended) or the "
            "paper-exact CNN and save model.h5."
        ),
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
    parser.add_argument(
        "--architecture",
        choices=("mobilenet_v2", "paper_cnn"),
        default="mobilenet_v2",
        help="Model family. MobileNetV2 transfer learning is recommended.",
    )
    parser.add_argument(
        "--no-imagenet-weights",
        action="store_true",
        help="Initialise MobileNetV2 randomly (normally reduces accuracy).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=8,
        help="Epochs with the pretrained backbone frozen.",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=30,
        help="Number of final MobileNetV2 layers to unfreeze.",
    )
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--class-weight-strategy",
        choices=("effective_number", "inverse_frequency", "none"),
        default="effective_number",
    )
    parser.add_argument(
        "--effective-number-beta",
        type=float,
        default=0.999,
        help="Cui et al. effective-number beta; must be in [0, 1).",
    )
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
        architecture=args.architecture,
        imagenet_weights=not args.no_imagenet_weights,
        warmup_epochs=args.warmup_epochs,
        fine_tune_layers=args.fine_tune_layers,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        class_weight_strategy=args.class_weight_strategy,
        effective_number_beta=args.effective_number_beta,
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
