"""Training and validation loops."""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass(slots=True)
class Metrics:
    loss: float
    accuracy: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    amp: bool,
    mixup_fn: Callable | None = None,
) -> Metrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
            metric_labels = labels.argmax(dim=1)
        else:
            metric_labels = labels
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
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
    amp: bool,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
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
