"""Evaluate a trained image classifier and export plots/CSV reports.

Example:
    python test.py --run-path runs/classifier --val-path data/val --weights best

The run directory is expected to contain ``train.log`` and
the selected ``best_model.pth`` or ``last_model.pth`` checkpoint.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


MODEL_RE = re.compile(r"model=(.*?)\s*\|\s*classes=(.*?)\s*\|\s*train=")
INPUT_SIZE_RE = re.compile(r"\binput_size=(None|\d+)")
EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)/\d+\s*\|\s*"
    r"train_loss=(?P<train_loss>[+\-\d.eE]+)\s*\|\s*"
    r"train_acc=(?P<train_acc>[+\-\d.eE]+)%?\s*\|\s*"
    r"val_loss=(?P<val_loss>[+\-\d.eE]+)\s*\|\s*"
    r"val_acc=(?P<val_acc>[+\-\d.eE]+)%?"
)


@dataclass(slots=True)
class RunInfo:
    model_name: str
    classes: list[str]
    input_size: int | None
    epochs: list[int]
    train_loss: list[float]
    train_acc: list[float]
    val_loss: list[float]
    val_acc: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a run and export curves, a confusion matrix, and metrics"
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        required=True,
        help="Run directory containing train.log and model weights",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        required=True,
        help="ImageFolder validation directory (one subdirectory per class)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--weights",
        choices=("best", "final"),
        default="best",
        help="Weights to evaluate: best_model.pth or final/last_model.pth (default: best)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint path; overrides --weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; default: <run-path>/test_results",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Override the input size recorded in the log (needed for older logs)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device, e.g. cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision on CUDA",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.input_size is not None and args.input_size < 1:
        parser.error("--input-size must be at least 1")
    return args


def locate_run_files(
    run_path: Path,
    weights: str,
    checkpoint: Path | None,
) -> tuple[Path, Path, Path]:
    run_path = run_path.expanduser().resolve()
    if run_path.is_file():
        if run_path.name != "train.log":
            raise ValueError("When --run-path is a file, it must be train.log")
        run_dir = run_path.parent
        log_path = run_path
    else:
        run_dir = run_path
        log_path = run_dir / "train.log"

    if not log_path.is_file():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    if checkpoint is not None:
        checkpoint_path = checkpoint.expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_dir / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
    else:
        checkpoint_name = "best_model.pth" if weights == "best" else "last_model.pth"
        checkpoint_path = run_dir / checkpoint_name

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Selected model checkpoint not found: {checkpoint_path}. "
            "Choose --weights best/final or pass --checkpoint explicitly."
        )
    return run_dir, log_path, checkpoint_path


def parse_train_log(log_path: Path) -> RunInfo:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    model_line_index = -1
    model_name = ""
    classes: list[str] = []
    input_size: int | None = None

    # A log may contain appended runs. Use the most recent run header and only
    # parse epochs after it.
    for index, line in enumerate(lines):
        match = MODEL_RE.search(line)
        if not match:
            continue
        parsed_classes = ast.literal_eval(match.group(2))
        if not isinstance(parsed_classes, list) or not all(
            isinstance(name, str) for name in parsed_classes
        ):
            raise ValueError(f"Invalid class list in {log_path}: {match.group(2)}")
        model_line_index = index
        model_name = match.group(1).strip()
        classes = parsed_classes
        input_size_match = INPUT_SIZE_RE.search(line)
        input_size = (
            int(input_size_match.group(1))
            if input_size_match and input_size_match.group(1) != "None"
            else None
        )

    if model_line_index < 0:
        raise ValueError(f"Could not find model/classes metadata in {log_path}")

    epochs: list[int] = []
    train_loss: list[float] = []
    train_acc: list[float] = []
    val_loss: list[float] = []
    val_acc: list[float] = []
    for line in lines[model_line_index + 1 :]:
        match = EPOCH_RE.search(line)
        if match:
            epochs.append(int(match.group("epoch")))
            train_loss.append(float(match.group("train_loss")))
            train_acc.append(float(match.group("train_acc")))
            val_loss.append(float(match.group("val_loss")))
            val_acc.append(float(match.group("val_acc")))

    if not epochs:
        raise ValueError(f"Could not find epoch metrics in {log_path}")
    return RunInfo(
        model_name,
        classes,
        input_size,
        epochs,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
    )


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_training_curves(run: RunInfo, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(run.epochs, run.train_loss, label="Train", linewidth=2)
    axes[0].plot(run.epochs, run.val_loss, label="Validation", linewidth=2)
    axes[0].set(title="Loss Curves", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(run.epochs, run.train_acc, label="Train", linewidth=2)
    axes[1].plot(run.epochs, run.val_acc, label="Validation", linewidth=2)
    axes[1].set(title="Accuracy Curves", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Training History: {run.model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_model(model_name: str, num_classes: int, checkpoint_path: Path) -> torch.nn.Module:
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def build_val_loader(
    val_path: Path,
    model: torch.nn.Module,
    expected_classes: list[str],
    input_size: int | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    val_path = val_path.expanduser().resolve()
    if not val_path.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_path}")

    data_config = resolve_model_data_config(model)
    if input_size is not None:
        data_config["input_size"] = (3, input_size, input_size)
    dataset = ImageFolder(
        val_path,
        transform=create_transform(**data_config, is_training=False),
    )
    if dataset.classes != expected_classes:
        raise ValueError(
            "Validation class folders do not match the training log.\n"
            f"Training:   {expected_classes}\n"
            f"Validation: {dataset.classes}"
        )
    if len(dataset) == 0:
        raise ValueError(f"Validation dataset is empty: {val_path}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_batches: list[np.ndarray] = []
    top_prediction_batches: list[np.ndarray] = []
    use_amp = amp and device.type == "cuda"

    for images, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            # Keep Top-1 identical to the training validation loop. ``argmax``
            # and ``topk(k=3)`` may choose a different first class when reduced
            # precision produces tied logits.
            top1_predictions = logits.argmax(dim=1, keepdim=True)
            remaining_k = min(2, logits.shape[1] - 1)
            if remaining_k > 0:
                remaining_logits = logits.clone()
                remaining_logits.scatter_(1, top1_predictions, -torch.inf)
                remaining_predictions = remaining_logits.topk(
                    k=remaining_k,
                    dim=1,
                ).indices
                top_predictions = torch.cat(
                    (top1_predictions, remaining_predictions),
                    dim=1,
                )
            else:
                top_predictions = top1_predictions
        true_batches.append(labels.numpy())
        top_prediction_batches.append(top_predictions.cpu().numpy())

    return np.concatenate(true_batches), np.concatenate(top_prediction_batches)


def calculate_topk_accuracies(
    y_true: np.ndarray,
    top_predictions: np.ndarray,
) -> dict[int, float]:
    accuracies: dict[int, float] = {}
    available_k = top_predictions.shape[1]
    for requested_k in (1, 2, 3):
        effective_k = min(requested_k, available_k)
        correct = (top_predictions[:, :effective_k] == y_true[:, None]).any(axis=1)
        accuracies[requested_k] = float(correct.mean())
    return accuracies


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def save_raw_confusion_csv(matrix: np.ndarray, classes: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *classes])
        for class_name, row in zip(classes, matrix, strict=True):
            writer.writerow([class_name, *row.tolist()])


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    column_totals = matrix.sum(axis=0, keepdims=True)
    return np.divide(
        matrix.astype(np.float64),
        column_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=column_totals != 0,
    )


def plot_normalized_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str],
    path: Path,
) -> None:
    class_count = len(classes)
    side = max(8.0, min(24.0, 0.7 * class_count + 4.0))
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Column-normalized ratio")

    ticks = np.arange(class_count)
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted class (columns)",
        ylabel="True class (rows)",
        title="Confusion Matrix (Normalized by Predicted Column)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if class_count <= 30:
        for row in range(class_count):
            for column in range(class_count):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=max(5, min(10, 140 / class_count)),
                    color="white" if value > 0.5 else "black",
                )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def save_metrics_csv(
    matrix: np.ndarray,
    classes: list[str],
    topk_accuracies: dict[int, float],
    path: Path,
) -> None:
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = safe_divide(true_positive, predicted)
    recall = safe_divide(true_positive, support)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    total = int(matrix.sum())
    correct = float(true_positive.sum())
    accuracy = correct / total
    if not np.isclose(accuracy, topk_accuracies[1]):
        raise ValueError("Top-1 accuracy does not match the confusion matrix")
    macro = (float(precision.mean()), float(recall.mean()), float(f1.mean()))
    weighted = (
        float(np.average(precision, weights=support)),
        float(np.average(recall, weights=support)),
        float(np.average(f1, weights=support)),
    )

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for index, class_name in enumerate(classes):
            writer.writerow(
                [
                    class_name,
                    f"{precision[index]:.6f}",
                    f"{recall[index]:.6f}",
                    f"{f1[index]:.6f}",
                    int(support[index]),
                ]
            )

        # Treat the K candidates per sample as multilabel predictions and
        # report global (micro) Top-K precision/recall/F1. Since every sample
        # has exactly one true label, Top-K recall equals Top-K accuracy, while
        # precision divides the number of hits by K predictions per sample.
        for k in (1, 2, 3):
            topk_recall = topk_accuracies[k]
            effective_k = min(k, len(classes))
            topk_precision = topk_recall / effective_k
            topk_f1 = (
                2 * topk_precision * topk_recall / (topk_precision + topk_recall)
                if topk_precision + topk_recall
                else 0.0
            )
            writer.writerow(
                [
                    f"Top-{k}",
                    f"{topk_precision:.6f}",
                    f"{topk_recall:.6f}",
                    f"{topk_f1:.6f}",
                    total,
                ]
            )
        writer.writerow(
            ["Macro", f"{macro[0]:.6f}", f"{macro[1]:.6f}", f"{macro[2]:.6f}", total]
        )
        writer.writerow(
            [
                "Weight",
                f"{weighted[0]:.6f}",
                f"{weighted[1]:.6f}",
                f"{weighted[2]:.6f}",
                total,
            ]
        )


def main() -> None:
    args = parse_args()
    run_dir, log_path, checkpoint_path = locate_run_files(
        args.run_path,
        args.weights,
        args.checkpoint,
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "test_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    run = parse_train_log(log_path)
    plot_training_curves(run, output_dir / "training_curves.png")

    input_size = args.input_size if args.input_size is not None else run.input_size
    if input_size is None:
        print(
            "Warning: this log does not record input_size. Using the model default. "
            "If training used --input-size, pass the same value to test.py."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")

    model = load_model(run.model_name, len(run.classes), checkpoint_path)
    loader = build_val_loader(
        args.val_path,
        model,
        run.classes,
        input_size,
        args.batch_size,
        args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    y_true, top_predictions = predict(model, loader, device, args.amp)
    y_pred = top_predictions[:, 0]
    topk_accuracies = calculate_topk_accuracies(y_true, top_predictions)

    confusion = make_confusion_matrix(y_true, y_pred, len(run.classes))
    save_raw_confusion_csv(
        confusion,
        run.classes,
        output_dir / "confusion_matrix_counts.csv",
    )
    normalized_confusion = normalize_columns(confusion)
    plot_normalized_confusion_matrix(
        normalized_confusion,
        run.classes,
        output_dir / "confusion_matrix_column_normalized.png",
    )
    save_metrics_csv(
        confusion,
        run.classes,
        topk_accuracies,
        output_dir / "metrics.csv",
    )
    accuracy = topk_accuracies[1]
    logged_accuracy = max(run.val_acc) if args.weights == "best" else run.val_acc[-1]
    logged_accuracy_name = "best" if args.weights == "best" else "final"
    accuracy_difference = 100.0 * accuracy - logged_accuracy

    print(f"Model:      {run.model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input size:  {input_size if input_size is not None else 'model default'}")
    print(f"Samples:    {len(y_true)}")
    print(f"Top-1 acc:  {topk_accuracies[1]:.2%}")
    print(f"Top-2 acc:  {topk_accuracies[2]:.2%}")
    print(f"Top-3 acc:  {topk_accuracies[3]:.2%}")
    print(f"Logged {logged_accuracy_name + ':':<6}{logged_accuracy:8.2f}%")
    print(f"Difference: {accuracy_difference:+8.2f} percentage points")
    print(f"Results:    {output_dir}")
    if abs(accuracy_difference) > 0.1:
        print(
            "Warning: evaluated accuracy differs from the logged best accuracy. "
            "Check that the validation data, input size, and checkpoint are unchanged."
        )


if __name__ == "__main__":
    main()
