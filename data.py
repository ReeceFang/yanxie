"""ImageFolder datasets and DataLoaders."""

from dataclasses import dataclass

import torch.nn as nn
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import Config


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    classes: list[str]


class ConvertColorSpace:
    """Convert a PIL image while remaining safe for multiprocessing workers."""

    def __init__(self, color_space: str) -> None:
        self.color_space = color_space.upper()
        if self.color_space not in {"RGB", "HSV", "LAB"}:
            raise ValueError(f"Unsupported color space: {color_space}")

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert(self.color_space)


def _convert_transform_color_space(
    transform: transforms.Compose,
    color_space: str,
) -> transforms.Compose:
    """Insert color conversion after PIL augmentations and before tensorization."""
    color_space = color_space.lower()
    if color_space == "rgb":
        return transform

    operations = []
    conversion_added = False
    tensor_operations = {"ToTensor", "PILToTensor", "MaybeToTensor", "ToNumpy"}

    for operation in transform.transforms:
        if (
            not conversion_added
            and operation.__class__.__name__ in tensor_operations
        ):
            operations.append(ConvertColorSpace(color_space))
            conversion_added = True

        # The model data config contains RGB statistics. Once the channel
        # meanings change, use neutral scaling instead of RGB normalization.
        if isinstance(operation, transforms.Normalize):
            operation = transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )

        operations.append(operation)

    if not conversion_added:
        raise RuntimeError(
            "Could not insert color conversion before tensorization in the "
            "timm transform pipeline"
        )

    return transforms.Compose(operations)


def build_dataloaders(config: Config, model: nn.Module) -> DataBundle:
    if not config.train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {config.train_dir}")
    if not config.val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {config.val_dir}")

    data_config = resolve_model_data_config(model)
    if config.input_size is not None:
        data_config["input_size"] = (3, config.input_size, config.input_size)

    train_transform = create_transform(
        **data_config,
        is_training=True,
        auto_augment=config.auto_augment,
    )
    train_transform = _convert_transform_color_space(
        train_transform,
        config.color_space,
    )
    val_transform = create_transform(**data_config, is_training=False)
    val_transform = _convert_transform_color_space(
        val_transform,
        config.color_space,
    )

    train_dataset = ImageFolder(config.train_dir, transform=train_transform)
    val_dataset = ImageFolder(config.val_dir, transform=val_transform)

    if len(train_dataset.classes) != config.num_classes:
        raise ValueError(
            f"num_classes={config.num_classes}, but train contains "
            f"{len(train_dataset.classes)} classes: {train_dataset.classes}"
        )
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train and val must contain the same class folders")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=config.mixup,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return DataBundle(train_loader, val_loader, train_dataset.classes)
