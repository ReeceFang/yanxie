"""Train one stage of the standalone coarse-to-fine classification experiment."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from timm.data import Mixup, create_transform, resolve_model_data_config
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from common import (
    HierarchicalImageFolder,
    RunMetadata,
    build_pretrained_model,
    read_merged_classes,
    save_run_metadata,
)


@dataclass(slots=True)
class Metrics:
    loss: float
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone hierarchical-stage training")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("coarse", "fine"),
        required=True,
        help="coarse merges TXT classes; fine keeps and distinguishes only TXT classes",
    )
    parser.add_argument("--merged-classes-file", type=Path, required=True)
    parser.add_argument("--merged-class-name", default="__merged__")
    parser.add_argument("--model", required=True, help="timm model name")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--auto-augment", default="rand-m9-n3-mstd0.5")
    parser.add_argument(
        "--mixup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.mixup and args.batch_size % 2 != 0:
        parser.error("--batch-size must be even when Mixup/CutMix is enabled")
    if args.mixup_alpha < 0 or args.cutmix_alpha < 0:
        parser.error("Mixup and CutMix alpha values cannot be negative")
    if args.mixup and args.mixup_alpha == 0 and args.cutmix_alpha == 0:
        parser.error("Mixup/CutMix requires at least one positive alpha value")
    if not 0 <= args.mixup_prob <= 1:
        parser.error("--mixup-prob must be between 0 and 1")
    if not 0 <= args.mixup_switch_prob <= 1:
        parser.error("--mixup-switch-prob must be between 0 and 1")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if args.input_size is not None and args.input_size < 1:
        parser.error("--input-size must be at least 1")
    if not args.merged_class_name.strip():
        parser.error("--merged-class-name cannot be empty")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: Path, stage: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"hierarchical_{stage}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def save_weights(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def build_datasets(
    data_dir: Path,
    stage: str,
    merged_classes: tuple[str, ...],
    merged_class_name: str,
) -> tuple[HierarchicalImageFolder, HierarchicalImageFolder]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    train_dataset = HierarchicalImageFolder(
        train_dir,
        stage,
        merged_classes,
        merged_class_name,
    )
    val_dataset = HierarchicalImageFolder(
        val_dir,
        stage,
        merged_classes,
        merged_class_name,
    )
    if train_dataset.original_class_to_idx != val_dataset.original_class_to_idx:
        raise ValueError("train and val must contain the same original class folders")
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train and val produced different hierarchical label mappings")
    return train_dataset, val_dataset


def attach_transforms(
    train_dataset: HierarchicalImageFolder,
    val_dataset: HierarchicalImageFolder,
    model: nn.Module,
    input_size: int | None,
    auto_augment: str,
) -> None:
    data_config = resolve_model_data_config(model)
    if input_size is not None:
        data_config["input_size"] = (3, input_size, input_size)
    train_dataset.transform = create_transform(
        **data_config,
        is_training=True,
        auto_augment=auto_augment,
    )
    val_dataset.transform = create_transform(**data_config, is_training=False)


def make_loaders(
    train_dataset: HierarchicalImageFolder,
    val_dataset: HierarchicalImageFolder,
    batch_size: int,
    num_workers: int,
    mixup: bool,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    if mixup and len(train_dataset) < batch_size:
        raise ValueError(
            "Training subset is smaller than --batch-size while Mixup/CutMix is enabled"
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=mixup,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    mixup_fn: Callable | None,
) -> Metrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
            metric_labels = labels.argmax(dim=1)
        else:
            metric_labels = labels
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = metric_labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(metric_labels).sum().item()
        total_samples += batch_size
    return Metrics(
        loss=total_loss / total_samples,
        accuracy=100.0 * total_correct / total_samples,
    )


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(labels).sum().item()
        total_samples += batch_size
    return Metrics(
        loss=total_loss / total_samples,
        accuracy=100.0 * total_correct / total_samples,
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    use_amp = args.amp and device.type == "cuda"
    set_seed(args.seed)

    merged_classes = read_merged_classes(args.merged_classes_file)
    train_dataset, val_dataset = build_datasets(
        data_dir,
        args.stage,
        merged_classes,
        args.merged_class_name,
    )
    model = build_pretrained_model(
        args.model,
        args.model_path,
        len(train_dataset.classes),
    )
    attach_transforms(
        train_dataset,
        val_dataset,
        model,
        args.input_size,
        args.auto_augment,
    )
    train_loader, val_loader = make_loaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        args.num_workers,
        args.mixup,
        pin_memory=device.type == "cuda",
    )

    logger = setup_logger(output_dir, args.stage)
    model.to(device)
    metadata = RunMetadata(
        run_dir=output_dir,
        model_name=args.model,
        classes=train_dataset.classes,
        input_size=args.input_size,
        stage=args.stage,
        merged_classes=list(merged_classes),
        merged_class_name=args.merged_class_name,
    )
    save_run_metadata(output_dir / "run_config.json", metadata)

    mixup_fn = None
    if args.mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            label_smoothing=args.label_smoothing,
            num_classes=len(train_dataset.classes),
        )
    train_criterion = (
        SoftTargetCrossEntropy() if mixup_fn is not None else nn.CrossEntropyLoss()
    )
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    class_counts = Counter(train_dataset.targets)
    counts_by_name = {
        name: class_counts[index] for index, name in enumerate(train_dataset.classes)
    }
    logger.info(
        f"stage={args.stage} | model={args.model} | classes={train_dataset.classes} | "
        f"train={len(train_dataset)} | val={len(val_dataset)} | device={device}"
    )
    logger.info(f"merged_classes={list(merged_classes)}")
    logger.info(f"train_class_counts={counts_by_name}")

    best_accuracy = -1.0
    with logging_redirect_tqdm(loggers=[logger]):
        progress = tqdm(range(args.epochs), desc=f"Training {args.stage}", unit="epoch")
        for epoch in progress:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                train_criterion,
                optimizer,
                scaler,
                device,
                use_amp,
                mixup_fn,
            )
            val_metrics = validate(
                model,
                val_loader,
                val_criterion,
                device,
                use_amp,
            )
            scheduler.step()
            save_weights(model, output_dir / "last_model.pth")
            if val_metrics.accuracy > best_accuracy:
                best_accuracy = val_metrics.accuracy
                save_weights(model, output_dir / "best_model.pth")
            progress.set_postfix(
                train_loss=f"{train_metrics.loss:.4f}",
                val_loss=f"{val_metrics.loss:.4f}",
                val_acc=f"{val_metrics.accuracy:.2f}%",
            )
            logger.info(
                f"epoch {epoch + 1:03d}/{args.epochs:03d} | "
                f"train_loss={train_metrics.loss:.4f} | "
                f"train_acc={train_metrics.accuracy:.2f}% | "
                f"val_loss={val_metrics.loss:.4f} | "
                f"val_acc={val_metrics.accuracy:.2f}% | "
                f"best_acc={best_accuracy:.2f}%"
            )


if __name__ == "__main__":
    main()
