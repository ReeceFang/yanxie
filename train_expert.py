"""Train one high-resolution encoder with pair-specific binary heads."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from expert_model import DEFAULT_PAIRS, PairExpert, load_matching_backbone_weights
from utils import set_seed, setup_logger


class PairDataset(Dataset):
    """Duplicate source images into every configured binary pair they belong to."""

    def __init__(
        self,
        root: Path,
        pairs: Sequence[Sequence[str]],
        transform,
    ) -> None:
        source = ImageFolder(root)
        self.classes = source.classes
        self.class_to_idx = source.class_to_idx
        self.loader = source.loader
        self.transform = transform
        self.records: list[tuple[str, int, int]] = []
        self.pair_counts: Counter[int] = Counter()

        class_membership: dict[int, list[tuple[int, int]]] = {}
        for pair_id, pair in enumerate(pairs):
            for local_label, class_name in enumerate(pair):
                if class_name not in self.class_to_idx:
                    raise ValueError(
                        f"Expert pair class is missing from {root}: {class_name}"
                    )
                class_index = self.class_to_idx[class_name]
                class_membership.setdefault(class_index, []).append(
                    (pair_id, local_label)
                )

        for image_path, class_index in source.samples:
            for pair_id, local_label in class_membership.get(class_index, []):
                self.records.append((image_path, pair_id, local_label))
                self.pair_counts[pair_id] += 1
        if not self.records:
            raise ValueError(f"No samples matched expert pairs in {root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        image_path, pair_id, local_label = self.records[index]
        image = self.loader(image_path)
        return self.transform(image), pair_id, local_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a shared high-resolution expert for known confusion pairs"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", default="convnext_base")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        required=True,
        help="Base or ImageNet checkpoint used to initialize matching backbone tensors",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pair_expert"))
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
    if args.input_size < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("input-size, epochs, and batch-size must be positive")
    if not 0 <= args.freeze_backbone_epochs < args.epochs:
        parser.error("freeze-backbone-epochs must be in [0, epochs)")
    if not 0 <= args.label_smoothing < 1:
        parser.error("label-smoothing must be in [0, 1)")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def set_backbone_trainable(model: PairExpert, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def make_transforms(model: PairExpert, input_size: int):
    data_config = resolve_model_data_config(model.backbone)
    data_config["input_size"] = (3, input_size, input_size)
    train_transform = create_transform(
        **data_config,
        is_training=True,
        auto_augment=None,
        color_jitter=0.05,
        scale=(0.85, 1.0),
        ratio=(0.9, 1.1),
        re_prob=0.0,
    )
    val_transform = create_transform(**data_config, is_training=False)
    return train_transform, val_transform


def run_epoch(
    model: PairExpert,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    backbone_frozen: bool,
) -> tuple[float, float, list[tuple[int, int]]]:
    training = optimizer is not None
    model.train(training)
    if training and backbone_frozen:
        model.backbone.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    pair_correct = [0 for _ in model.pairs]
    pair_total = [0 for _ in model.pairs]
    use_amp = amp and device.type == "cuda"

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, pair_ids, labels in tqdm(
            loader,
            desc="Train" if training else "Validate",
            unit="batch",
            leave=False,
        ):
            images = images.to(device, non_blocking=device.type == "cuda")
            pair_ids = pair_ids.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, pair_ids)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            predictions = logits.argmax(dim=1)
            correct = predictions.eq(labels)
            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += correct.sum().item()
            total_samples += batch_size
            for pair_id in pair_ids.unique().tolist():
                mask = pair_ids == pair_id
                pair_total[pair_id] += mask.sum().item()
                pair_correct[pair_id] += correct[mask].sum().item()

    return (
        total_loss / total_samples,
        100.0 * total_correct / total_samples,
        list(zip(pair_correct, pair_total, strict=True)),
    )


def save_checkpoint(
    path: Path,
    model: PairExpert,
    classes: Sequence[str],
    input_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": model.model_name,
            "classes": list(classes),
            "pairs": [list(pair) for pair in model.pairs],
            "input_size": input_size,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("data-dir must contain train and val directories")

    set_seed(args.seed)
    logger = setup_logger(args.output_dir)
    model = PairExpert(args.model, DEFAULT_PAIRS)
    loaded, available = load_matching_backbone_weights(
        model.backbone,
        args.init_checkpoint,
    )
    logger.info(
        f"expert_model={args.model} | initialized_backbone_tensors={loaded}/{available} | "
        f"input_size={args.input_size}"
    )
    train_transform, val_transform = make_transforms(model, args.input_size)
    train_dataset = PairDataset(train_dir, model.pairs, train_transform)
    val_dataset = PairDataset(val_dir, model.pairs, val_transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train and val class folders must match")

    for pair_id, pair in enumerate(model.pairs):
        logger.info(
            f"pair[{pair_id}]={pair[0]} <-> {pair[1]} | "
            f"train={train_dataset.pair_counts[pair_id]} | "
            f"val={val_dataset.pair_counts[pair_id]}"
        )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model.to(device)
    set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.heads.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == "cuda"
    )

    best_accuracy = -1.0
    for epoch in range(args.epochs):
        if epoch == args.freeze_backbone_epochs:
            set_backbone_trainable(model, True)
            logger.info("Unfroze expert backbone")
        backbone_frozen = epoch < args.freeze_backbone_epochs
        train_loss, train_accuracy, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.amp,
            optimizer,
            scaler,
            backbone_frozen,
        )
        val_loss, val_accuracy, pair_results = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.amp,
            None,
            scaler,
            False,
        )
        scheduler.step()
        save_checkpoint(
            args.output_dir / "last_expert.pth",
            model,
            train_dataset.classes,
            args.input_size,
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                args.output_dir / "best_expert.pth",
                model,
                train_dataset.classes,
                args.input_size,
            )
        pair_text = " | ".join(
            f"p{pair_id}={100.0 * correct / total:.1f}%"
            for pair_id, (correct, total) in enumerate(pair_results)
            if total
        )
        logger.info(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.2f}% | "
            f"best={best_accuracy:.2f}% | {pair_text}"
        )


if __name__ == "__main__":
    main()

