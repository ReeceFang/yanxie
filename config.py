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
    output_dir: Path
    seed: int
    amp: bool
    input_size: int | None = None

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
        "--input-size",
        type=int,
        default=None,
        help="Override the model input image size (default: model configuration)",
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
    if args.input_size is not None and args.input_size < 1:
        parser.error("--input-size must be at least 1")

    args.data_dir = args.data_dir.resolve()
    args.model_path = args.model_path.resolve()
    args.output_dir = args.output_dir.resolve()
    return Config(**vars(args))
