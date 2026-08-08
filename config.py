"""Training configuration."""

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Config:
    data_dir: Path
    model: str
    model_path: Path
    num_classes: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    auto_augment: str
    mixup: bool
    mixup_alpha: float
    cutmix_alpha: float
    mixup_prob: float
    mixup_switch_prob: float
    label_smoothing: float
    output_dir: Path
    seed: int
    amp: bool
    input_size: int | None = None
    color_space: str = "rgb"

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.data_dir / "val"


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description="Single-GPU image classification")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="timm model name")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
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
        help="Enable Mixup/CutMix during training",
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Override the model input image size (default: model configuration)",
    )
    parser.add_argument(
        "--color-space",
        choices=("rgb", "hsv", "lab"),
        default="rgb",
        help="Color space used for model input (default: rgb)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/classifier"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.num_classes < 2:
        parser.error("--num-classes must be at least 2")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.mixup and args.batch_size % 2 != 0:
        parser.error("--batch-size must be even when Mixup/CutMix is enabled")
    if args.mixup_alpha < 0:
        parser.error("--mixup-alpha must be non-negative")
    if args.cutmix_alpha < 0:
        parser.error("--cutmix-alpha must be non-negative")
    if args.mixup and args.mixup_alpha == 0 and args.cutmix_alpha == 0:
        parser.error("Mixup/CutMix requires a positive alpha value")
    if not 0 <= args.mixup_prob <= 1:
        parser.error("--mixup-prob must be between 0 and 1")
    if not 0 <= args.mixup_switch_prob <= 1:
        parser.error("--mixup-switch-prob must be between 0 and 1")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if args.input_size is not None and args.input_size < 1:
        parser.error("--input-size must be at least 1")

    args.data_dir = args.data_dir.resolve()
    args.model_path = args.model_path.resolve()
    args.output_dir = args.output_dir.resolve()
    return Config(**vars(args))
