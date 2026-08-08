"""Create a timm model and load local pretrained weights."""

import timm
import torch.nn as nn
from timm.models import load_checkpoint

from config import Config


def build_model(
    config: Config,
    *,
    checkpoint_is_trained: bool = False,
) -> nn.Module:
    model = timm.create_model(
        config.model,
        pretrained=False,
    )

    if checkpoint_is_trained:
        # A training checkpoint already contains the task-specific classifier.
        # Build that classifier before loading so its shape matches the weights.
        model.reset_classifier(num_classes=config.num_classes)
        load_checkpoint(
            model,
            str(config.model_path),
            strict=True,
        )
        return model

    pretrained_num_classes = model.pretrained_cfg.get(
        "num_classes",
        model.num_classes,
    )

    model.reset_classifier(
        num_classes=pretrained_num_classes,
    )

    load_checkpoint(
        model,
        str(config.model_path),
        strict=True,
    )

    model.reset_classifier(
        num_classes=config.num_classes,
    )
    return model
