"""Single-GPU image-classification training entry point."""

import torch
import torch.nn as nn
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import parse_config
from data import build_dataloaders
from engine import train_one_epoch, validate
from model import build_model
from utils import save_weights, set_seed, setup_logger


def main() -> None:
    config = parse_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this single-GPU training script")

    set_seed(config.seed)
    logger = setup_logger(config.output_dir)
    model = build_model(config).cuda()
    data = build_dataloaders(config, model)

    mixup_fn = None
    if config.mixup:
        mixup_fn = Mixup(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            prob=config.mixup_prob,
            switch_prob=config.mixup_switch_prob,
            label_smoothing=config.label_smoothing,
            num_classes=config.num_classes,
        )

    train_criterion = (
        SoftTargetCrossEntropy() if mixup_fn is not None else nn.CrossEntropyLoss()
    )
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    logger.info(
        f"model={config.model} | classes={data.classes} | "
        f"train={len(data.train_loader.dataset)} | val={len(data.val_loader.dataset)}"
    )
    logger.info(
        f"mixup={config.mixup} | mixup_alpha={config.mixup_alpha} | "
        f"cutmix_alpha={config.cutmix_alpha} | prob={config.mixup_prob} | "
        f"switch_prob={config.mixup_switch_prob} | "
        f"label_smoothing={config.label_smoothing}"
    )

    best_accuracy = -1.0
    with logging_redirect_tqdm(loggers=[logger]):
        progress = tqdm(range(config.epochs), desc="Training", unit="epoch")
        for epoch in progress:
            train_metrics = train_one_epoch(
                model,
                data.train_loader,
                train_criterion,
                optimizer,
                scaler,
                config.amp,
                mixup_fn,
            )
            val_metrics = validate(
                model,
                data.val_loader,
                val_criterion,
                config.amp,
            )
            scheduler.step()

            save_weights(model, config.output_dir / "last_model.pth")
            if val_metrics.accuracy > best_accuracy:
                best_accuracy = val_metrics.accuracy
                save_weights(model, config.output_dir / "best_model.pth")

            progress.set_postfix(
                train_loss=f"{train_metrics.loss:.4f}",
                val_loss=f"{val_metrics.loss:.4f}",
                val_acc=f"{val_metrics.accuracy:.2f}%",
            )
            logger.info(
                f"epoch {epoch + 1:03d}/{config.epochs:03d} | "
                f"train_loss={train_metrics.loss:.4f} | "
                f"train_acc={train_metrics.accuracy:.2f}% | "
                f"val_loss={val_metrics.loss:.4f} | "
                f"val_acc={val_metrics.accuracy:.2f}% | "
                f"best_acc={best_accuracy:.2f}%"
            )


if __name__ == "__main__":
    main()
