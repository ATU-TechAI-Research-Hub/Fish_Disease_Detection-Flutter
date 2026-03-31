from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from train.prepare_dataset import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATASET_ROOT,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    RANDOM_STATE,
    ensure_dataset_artifacts,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "backend" / "app" / "ml"


class CenterCropSquare:
    """Center-crop to a square matching the short side, then resize."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        short = min(w, h)
        left = (w - short) // 2
        top = (h - short) // 2
        img = img.crop((left, top, left + short, top + short))
        return img.resize((self.size, self.size), Image.LANCZOS)


class FishDiseaseDataset(Dataset):
    def __init__(self, csv_path: Path, transform: transforms.Compose, mixup_alpha: float = 0.0) -> None:
        self.frame = pd.read_csv(csv_path)
        self.transform = transform
        self.mixup_alpha = mixup_alpha

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image = Image.open(row["file_path"]).convert("RGB")
        image_tensor = self.transform(image)
        label = int(row["class_index"])
        return image_tensor, label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compute_flatten_size(image_size: int) -> int:
    """Compute the feature map size after 3 conv+pool blocks dynamically."""
    s = image_size
    for k in (5, 3, 3):
        s = s - k + 1       # conv with no padding
        s = s // 2           # maxpool 2x2, stride 2
    return 32 * s * s


class PaperCNN(nn.Module):
    """Custom CNN from Tamut et al., Aquac. J. 2025 (Table 1), with improvements.

    Architecture:
      Conv2D(128, 5x5) -> ReLU -> MaxPool(2x2) -> BatchNorm -> Dropout(0.25)
      Conv2D(64, 3x3)  -> ReLU -> MaxPool(2x2) -> BatchNorm -> Dropout(0.25)
      Conv2D(32, 3x3)  -> ReLU -> MaxPool(2x2) -> BatchNorm -> Dropout(0.25)
      Flatten -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(num_classes)
    """

    def __init__(self, num_classes: int = 7, image_size: int = 150) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),
        )

        flat = _compute_flatten_size(image_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name: str, num_classes: int, image_size: int = 150) -> nn.Module:
    if model_name == "paper_cnn":
        return PaperCNN(num_classes=num_classes, image_size=image_size)
    raise ValueError(f"Unsupported model: {model_name}")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            CenterCropSquare(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            CenterCropSquare(image_size),
            transforms.ToTensor(),
        ]
    )
    return train_transform, eval_transform


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def compute_class_weights(csv_path: Path, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    df = pd.read_csv(csv_path)
    counts = df["class_index"].value_counts().sort_index()
    total = len(df)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for idx in range(num_classes):
        c = counts.get(idx, 1)
        weights[idx] = total / (num_classes * c)
    return weights.to(device)


def create_data_loaders(
    artifacts_dir: Path, image_size: int, batch_size: int, num_workers: int
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    train_transform, eval_transform = build_transforms(image_size)
    dataset_paths = {
        "train": artifacts_dir / "train_split.csv",
        "val": artifacts_dir / "val_split.csv",
        "test": artifacts_dir / "test_split.csv",
    }
    datasets = {
        "train": FishDiseaseDataset(dataset_paths["train"], train_transform),
        "val": FishDiseaseDataset(dataset_paths["val"], eval_transform),
        "test": FishDiseaseDataset(dataset_paths["test"], eval_transform),
    }

    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    sizes = {split_name: len(dataset) for split_name, dataset in datasets.items()}
    return loaders, sizes


def l1_regularization(model: nn.Module, lambda_l1: float = 1e-5) -> torch.Tensor:
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if "weight" in name and "features" in name and param.dim() >= 2:
            l1_loss = l1_loss + param.abs().sum()
    return lambda_l1 * l1_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    lambda_l1: float = 1e-5,
    mixup_alpha: float = 0.0,
    scheduler_step_fn=None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    running_correct = 0
    total_examples = 0

    progress = tqdm(loader, leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        if is_training and mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)
        else:
            targets_a = targets_b = labels
            lam = 1.0

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        if is_training and mixup_alpha > 0:
            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
        else:
            loss = criterion(logits, labels)

        if is_training:
            loss = loss + l1_regularization(model, lambda_l1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler_step_fn is not None:
                scheduler_step_fn()

        predictions = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += int((predictions == labels).sum().item())
        total_examples += batch_size

        progress.set_postfix(
            loss=f"{running_loss / max(total_examples, 1):.4f}",
            acc=f"{running_correct / max(total_examples, 1):.4f}",
        )

    average_loss = running_loss / max(total_examples, 1)
    average_accuracy = running_correct / max(total_examples, 1)
    return average_loss, average_accuracy


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_confidences: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

    return all_labels, all_predictions, all_confidences


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    model_name: str,
    image_size: int,
    class_map: dict[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "image_size": image_size,
            "mean": IMAGE_MEAN,
            "std": IMAGE_STD,
            "class_map": class_map,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )


def export_to_onnx(
    model: nn.Module, output_path: Path, image_size: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the freshwater fish disease classifier."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR,
    )
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--model-name", choices=["paper_cnn"], default="paper_cnn",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    artifact_paths = ensure_dataset_artifacts(
        dataset_root=args.dataset_root,
        artifacts_dir=args.artifacts_dir,
    )

    with artifact_paths["class_map"].open("r", encoding="utf-8") as file_handle:
        class_map = json.load(file_handle)
    class_map["model_name"] = args.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    loaders, dataset_sizes = create_data_loaders(
        artifacts_dir=args.artifacts_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_map["classes"])

    class_weights = compute_class_weights(
        args.artifacts_dir / "train_split.csv", num_classes, device
    )
    print(f"Class weights: {class_weights.tolist()}")

    model = build_model(args.model_name, num_classes=num_classes, image_size=args.image_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_name}")
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(loaders["train"])
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step_counter = [0]

    def scheduler_step_fn():
        scheduler.step()
        step_counter[0] += 1

    history_rows: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_checkpoint_path = args.artifacts_dir / "best_model.pt"
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            mixup_alpha=args.mixup_alpha,
            scheduler_step_fn=scheduler_step_fn,
        )
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )

        print(
            "train_loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f} lr={:.6f}".format(
                train_loss, train_accuracy, val_loss, val_accuracy, current_lr,
            )
        )

        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            improved = True

        if improved:
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                checkpoint_path=best_checkpoint_path,
                model_name=args.model_name,
                image_size=args.image_size,
                class_map=class_map,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(args.artifacts_dir / "training_history.csv", index=False)

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    labels, predictions, confidences = collect_predictions(
        model=model,
        loader=loaders["test"],
        device=device,
    )
    test_accuracy = float(accuracy_score(labels, predictions))

    target_names = [
        class_entry["disease_name"]
        for class_entry in sorted(
            class_map["classes"], key=lambda class_info: class_info["class_index"]
        )
    ]
    report = classification_report(
        labels, predictions, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    confusion = confusion_matrix(labels, predictions)

    confusion_df = pd.DataFrame(confusion, index=target_names, columns=target_names)
    confusion_df.to_csv(args.artifacts_dir / "confusion_matrix.csv")

    with (args.artifacts_dir / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as file_handle:
        json.dump(report, file_handle, indent=2)

    metrics = {
        "model_name": args.model_name,
        "architecture": "PaperCNN (Tamut et al., Aquac. J. 2025) - Enhanced",
        "device": str(device),
        "epochs_completed": int(len(history_df)),
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_accuracy,
        "dataset_sizes": dataset_sizes,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "average_test_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "training_config": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": args.mixup_alpha,
            "warmup_epochs": args.warmup_epochs,
            "optimizer": "AdamW",
            "scheduler": "cosine_annealing_with_warmup",
        },
    }
    with (args.artifacts_dir / "metrics.json").open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    export_to_onnx(
        model=model,
        output_path=args.model_dir / "fish_disease_classifier.onnx",
        image_size=args.image_size,
    )

    with (args.model_dir / "class_map.json").open("w", encoding="utf-8") as file_handle:
        json.dump(class_map, file_handle, indent=2)

    shutil.copy2(artifact_paths["audit_summary"], args.model_dir / "audit_summary.json")

    print("\nTraining complete.")
    print(json.dumps(metrics, indent=2))
    print(f"ONNX model: {args.model_dir / 'fish_disease_classifier.onnx'}")
    print(f"Class map: {args.model_dir / 'class_map.json'}")


if __name__ == "__main__":
    main()
