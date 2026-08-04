"""Create a timm model and load local pretrained weights."""

import timm
import torch.nn as nn
from timm.models import load_checkpoint

from config import Config


def build_model(config: Config) -> nn.Module:
    model = timm.create_model(
        config.model,
        pretrained=False,
    )

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
