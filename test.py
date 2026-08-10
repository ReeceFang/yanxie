"""Evaluate a trained image classifier and export plots/CSV reports.

Example:
    python test.py --run-path runs/classifier --data-dir data --weights best

The run directory is expected to contain ``train.log`` and
the selected ``best_model.pth`` or ``last_model.pth`` checkpoint.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from config import Config
from data import build_dataloaders
from model import build_model


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
    config_values: dict[str, object] | None
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
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root containing train/ and val/",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size recorded in the log",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the worker count recorded in the log",
    )
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
        default=None,
        help="Override the AMP setting recorded in the log",
    )
    args = parser.parse_args()

    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers is not None and args.num_workers < 0:
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
    pending_config_values: dict[str, object] | None = None
    config_values: dict[str, object] | None = None

    # A log may contain appended runs. Use the most recent run header and only
    # parse epochs after it.
    for index, line in enumerate(lines):
        message = line.split(" | INFO | ", maxsplit=1)[-1].strip()
        if message.startswith("config="):
            parsed_config = json.loads(message.removeprefix("config="))
            if not isinstance(parsed_config, dict):
                raise ValueError(f"Invalid config entry in {log_path}")
            pending_config_values = parsed_config

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
        config_values = (
            pending_config_values.copy() if pending_config_values is not None else None
        )
        pending_config_values = None
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
        config_values,
        epochs,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
    )


def build_test_config(
    args: argparse.Namespace,
    run: RunInfo,
    checkpoint_path: Path,
    output_dir: Path,
) -> Config:
    data_dir = args.data_dir.expanduser().resolve()

    if run.config_values is not None:
        config_values = run.config_values.copy()
        for key, value in config_values.items():
            if isinstance(value, str) and (
                key.endswith("_dir") or key.endswith("_path")
            ):
                config_values[key] = Path(value)

        # Keep the training configuration intact and override only values that
        # necessarily belong to this evaluation run.
        config_values.update(
            data_dir=data_dir,
            model=run.model_name,
            model_path=checkpoint_path,
            num_classes=len(run.classes),
            output_dir=output_dir,
        )
        if args.batch_size is not None:
            config_values["batch_size"] = args.batch_size
        if args.num_workers is not None:
            config_values["num_workers"] = args.num_workers
        if args.input_size is not None:
            config_values["input_size"] = args.input_size
        if args.amp is not None:
            config_values["amp"] = args.amp
        return Config(**config_values)

    # Backward-compatible values for logs created before the full Config JSON
    # was recorded. input_size was already present in the old metadata line.
    return Config(
        data_dir=data_dir,
        model=run.model_name,
        model_path=checkpoint_path,
        num_classes=len(run.classes),
        epochs=1,
        batch_size=args.batch_size if args.batch_size is not None else 32,
        num_workers=args.num_workers if args.num_workers is not None else 4,
        learning_rate=0.0,
        weight_decay=0.0,
        auto_augment="rand-m9-n3-mstd0.5",
        mixup=False,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mixup_prob=0.0,
        mixup_switch_prob=0.0,
        label_smoothing=0.0,
        output_dir=output_dir,
        seed=0,
        amp=args.amp if args.amp is not None else True,
        input_size=args.input_size if args.input_size is not None else run.input_size,
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


def _unique_destination(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination

    source_name = Path(filename)
    counter = 2
    while True:
        candidate = directory / f"{source_name.stem}-{counter}{source_name.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_result_images(
    loader: DataLoader,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    output_dir: Path,
) -> tuple[Path, int, int]:
    """Copy source images into ground-truth class and correctness folders."""
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError(
            "Result images require a sequential validation sampler (shuffle=False)"
        )
    if loader.drop_last:
        raise ValueError("Result images require drop_last=False for validation")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays")

    samples = getattr(loader.dataset, "samples", None)
    if samples is None:
        raise TypeError(
            "Validation dataset must expose ImageFolder-compatible 'samples' paths"
        )
    if len(samples) != len(loader.dataset):
        raise ValueError("Validation dataset length does not match dataset.samples")
    if len(samples) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError(
            "Validation samples, true labels, and predictions have different lengths: "
            f"samples={len(samples)}, y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    source_paths: list[Path] = []
    dataset_labels: list[int] = []
    for sample in samples:
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("Each dataset sample must contain an image path and label")
        source_path = Path(sample[0]).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Source image not found: {source_path}")
        source_paths.append(source_path)
        dataset_labels.append(int(sample[1]))

    dataset_labels_array = np.asarray(dataset_labels, dtype=np.int64)
    if not np.array_equal(dataset_labels_array, y_true):
        raise ValueError(
            "Validation dataset label order does not match inference labels; "
            "refusing to save mismatched images"
        )

    class_count = len(classes)
    if class_count == 0:
        raise ValueError("Class list is empty")
    for class_name in classes:
        if not class_name or class_name in {".", ".."} or Path(class_name).name != class_name:
            raise ValueError(f"Unsafe class name for an output directory: {class_name!r}")
    if np.any(y_true < 0) or np.any(y_true >= class_count):
        raise ValueError("A true label index is outside the class list")
    if np.any(y_pred < 0) or np.any(y_pred >= class_count):
        raise ValueError("A predicted label index is outside the class list")

    output_dir = output_dir.expanduser().resolve()
    result_dir = output_dir / "result_image"
    resolved_result_dir = result_dir.resolve()
    if resolved_result_dir.parent != output_dir:
        raise ValueError(
            f"Unsafe result_image path outside the output directory: {resolved_result_dir}"
        )
    if result_dir.exists():
        if result_dir.is_symlink() or not result_dir.is_dir():
            raise ValueError(f"result_image is not a regular directory: {result_dir}")
        shutil.rmtree(result_dir)

    for class_name in classes:
        (result_dir / class_name / "true").mkdir(parents=True, exist_ok=True)
        (result_dir / class_name / "false").mkdir(parents=True, exist_ok=True)

    correct_count = 0
    for source_path, true_index, pred_index in zip(
        source_paths,
        y_true.tolist(),
        y_pred.tolist(),
        strict=True,
    ):
        true_name = classes[true_index]
        pred_name = classes[pred_index]
        is_correct = true_index == pred_index
        correctness_dir = "true" if is_correct else "false"
        destination_dir = result_dir / true_name / correctness_dir
        output_name = f"{true_name}-{pred_name}-{source_path.name}"
        destination = _unique_destination(destination_dir, output_name)
        shutil.copy2(source_path, destination)
        correct_count += int(is_correct)

    incorrect_count = len(source_paths) - correct_count
    return result_dir, correct_count, incorrect_count


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
    micro_true_positive = correct
    micro_false_positive = float(predicted.sum() - correct)
    micro_false_negative = float(support.sum() - correct)
    micro_precision = micro_true_positive / (micro_true_positive + micro_false_positive)
    micro_recall = micro_true_positive / (micro_true_positive + micro_false_negative)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    macro = (float(precision.mean()), float(recall.mean()), float(f1.mean()))
    weighted = (
        float(np.average(precision, weights=support)),
        float(np.average(recall, weights=support)),
        float(np.average(f1, weights=support)),
    )

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["class", "precision_or_top_k_accuracy", "recall", "f1_score", "support"]
        )
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

        # Top-1 is ordinary single-label classification, so its global micro
        # precision/recall/F1 are well-defined. They are mathematically equal
        # to accuracy for a single-label multiclass task, but are calculated
        # independently above from TP/FP/FN.
        writer.writerow(
            [
                "Top-1",
                f"{micro_precision:.6f}",
                f"{micro_recall:.6f}",
                f"{micro_f1:.6f}",
                total,
            ]
        )

        # Top-2/Top-3 are ranking accuracies, not standard classification
        # precision/recall/F1 tuples. Store each accuracy in the shared value
        # column and leave recall/F1 empty.
        for k in (2, 3):
            topk_accuracy = topk_accuracies[k]
            writer.writerow(
                [
                    f"Top-{k}",
                    f"{topk_accuracy:.6f}",
                    "",
                    "",
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

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")

    test_config = build_test_config(
        args,
        run,
        checkpoint_path,
        output_dir,
    )
    if test_config.input_size is None:
        print(
            "Warning: this log does not record input_size. Using the model default. "
            "If training used --input-size, pass the same value to test.py."
        )
    model = build_model(test_config, checkpoint_is_trained=True)
    data = build_dataloaders(test_config, model)
    if data.classes != run.classes:
        raise ValueError(
            "Dataset classes do not match the training log.\n"
            f"Log:     {run.classes}\n"
            f"Dataset: {data.classes}"
        )
    loader = data.val_loader
    model.to(device)
    y_true, top_predictions = predict(model, loader, device, test_config.amp)
    y_pred = top_predictions[:, 0]
    topk_accuracies = calculate_topk_accuracies(y_true, top_predictions)
    result_image_dir, correct_image_count, incorrect_image_count = (
        save_result_images(
            loader,
            y_true,
            y_pred,
            run.classes,
            output_dir,
        )
    )

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
    print(
        f"Input size:  "
        f"{test_config.input_size if test_config.input_size is not None else 'model default'}"
    )
    print(f"Samples:    {len(y_true)}")
    print(f"Top-1 acc:  {topk_accuracies[1]:.2%}")
    print(f"Top-2 acc:  {topk_accuracies[2]:.2%}")
    print(f"Top-3 acc:  {topk_accuracies[3]:.2%}")
    print(f"Logged {logged_accuracy_name + ':':<6}{logged_accuracy:8.2f}%")
    print(f"Difference: {accuracy_difference:+8.2f} percentage points")
    print(f"Results:    {output_dir}")
    print(f"Images:     {result_image_dir}")
    print(f"Saved:      {correct_image_count + incorrect_image_count}")
    print(f"Correct:    {correct_image_count}")
    print(f"Incorrect:  {incorrect_image_count}")
    if abs(accuracy_difference) > 0.1:
        print(
            "Warning: evaluated accuracy differs from the logged best accuracy. "
            "Check that the validation data, input size, and checkpoint are unchanged."
        )


if __name__ == "__main__":
    main()
