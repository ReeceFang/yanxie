"""ImageFolder datasets and DataLoaders."""

from dataclasses import dataclass

import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import Config


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    classes: list[str]


def build_dataloaders(config: Config, model: nn.Module) -> DataBundle:
    if not config.train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {config.train_dir}")
    if not config.val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {config.val_dir}")

    data_config = resolve_model_data_config(model)
    if config.input_size is not None:
        data_config["input_size"] = (3, config.input_size, config.input_size)

    train_dataset = ImageFolder(
        config.train_dir,
        transform=create_transform(
            **data_config,
            is_training=True,
            auto_augment=config.auto_augment,
        ),
    )
    val_dataset = ImageFolder(
        config.val_dir,
        transform=create_transform(**data_config, is_training=False),
    )

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
