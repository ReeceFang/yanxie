"""Create a timm model and load local pretrained weights."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import load_checkpoint

from config import Config


class CosineClassifier(nn.Module):
    """Classify normalized features with normalized class prototypes."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 16.0,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")

        self.in_features = in_features
        self.out_features = num_classes
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = F.normalize(features, p=2, dim=-1)
        normalized_weight = F.normalize(self.weight, p=2, dim=-1)
        return self.scale * F.linear(normalized_features, normalized_weight)


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

    linear_classifier = model.get_classifier()
    if not isinstance(linear_classifier, nn.Linear):
        raise TypeError(
            "CosineClassifier requires ConvNeXt's classifier to be nn.Linear"
        )
    if not hasattr(model, "head") or not hasattr(model.head, "fc"):
        raise TypeError("CosineClassifier currently supports ConvNeXt models only")

    model.head.fc = CosineClassifier(
        in_features=linear_classifier.in_features,
        num_classes=config.num_classes,
        scale=16.0,
    )

    return model
