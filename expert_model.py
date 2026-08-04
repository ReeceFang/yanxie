"""Shared high-resolution backbone with pair-specific binary classifier heads."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("huilvse niyan", "huise niyan"),
    ("zihongse niyan", "zonghongse niyan"),
    ("zonghongse niyan", "zonghongse shazhi niyan"),
    ("huibaise changshi", "qianhongse changyingzhi shayan"),
    ("huizise changyingzhi shayan", "qianhongse changyingzhi shayan"),
    ("qianhongse changyingzhi shayan", "qianhuise changyingzhi shayan"),
)


def pair_name(pair: Sequence[str]) -> str:
    return f"{pair[0]}|||{pair[1]}"


def load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    """Load common PyTorch checkpoint layouts and remove a DDP prefix."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise TypeError(f"Unsupported or empty checkpoint: {path}")
    if all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {
            str(key).removeprefix("module."): value
            for key, value in checkpoint.items()
        }
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise TypeError(f"Checkpoint does not contain a tensor state_dict: {path}")
    return checkpoint


def load_matching_backbone_weights(model: nn.Module, path: Path) -> tuple[int, int]:
    """Load matching tensors while intentionally ignoring the source classifier."""
    source = load_state_dict_file(path)
    target = model.state_dict()
    matching = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    if not matching:
        raise ValueError(
            f"No tensors in {path} match model {model.__class__.__name__}"
        )
    model.load_state_dict(matching, strict=False)
    return len(matching), len(target)


class PairExpert(nn.Module):
    """One image encoder shared by several independent two-class heads."""

    def __init__(
        self,
        model_name: str,
        pairs: Sequence[Sequence[str]],
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pairs = tuple((str(pair[0]), str(pair[1])) for pair in pairs)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = int(self.backbone.num_features)
        self.heads = nn.ModuleList(
            nn.Linear(feature_dim, 2) for _ in range(len(self.pairs))
        )

    def forward(
        self,
        images: torch.Tensor,
        pair_ids: torch.Tensor,
    ) -> torch.Tensor:
        if pair_ids.ndim != 1 or pair_ids.shape[0] != images.shape[0]:
            raise ValueError("pair_ids must contain one id per image")
        features = self.backbone(images)
        logits = features.new_empty((features.shape[0], 2))
        for pair_id_tensor in pair_ids.unique():
            pair_id = int(pair_id_tensor.item())
            if not 0 <= pair_id < len(self.heads):
                raise ValueError(f"Invalid pair id: {pair_id}")
            mask = pair_ids == pair_id
            logits[mask] = self.heads[pair_id](features[mask])
        return logits


class CosineClassifierForLoading(nn.Module):
    """Checkpoint-compatible version of the optional local cosine head."""

    def __init__(self, in_features: int, num_classes: int, scale: float = 16.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(
            F.normalize(features, p=2, dim=-1),
            F.normalize(self.weight, p=2, dim=-1),
        )


def load_classifier_model(
    model_name: str,
    num_classes: int,
    checkpoint_path: Path,
) -> nn.Module:
    """Load either the original linear head or the local cosine-head checkpoint."""
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    state_dict = load_state_dict_file(checkpoint_path)
    if (
        "head.fc.weight" in state_dict
        and "head.fc.bias" not in state_dict
        and hasattr(model, "head")
        and hasattr(model.head, "fc")
        and isinstance(model.head.fc, nn.Linear)
    ):
        model.head.fc = CosineClassifierForLoading(
            model.head.fc.in_features,
            num_classes,
        )
    model.load_state_dict(state_dict, strict=True)
    return model


def build_pair_lookup(
    classes: Sequence[str],
    pairs: Sequence[Sequence[str]],
) -> dict[frozenset[int], int]:
    class_to_idx = {name: index for index, name in enumerate(classes)}
    lookup: dict[frozenset[int], int] = {}
    for pair_id, pair in enumerate(pairs):
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"Invalid class pair: {pair}")
        missing = [name for name in pair if name not in class_to_idx]
        if missing:
            raise ValueError(f"Pair contains unknown classes: {missing}")
        key = frozenset((class_to_idx[pair[0]], class_to_idx[pair[1]]))
        if key in lookup:
            raise ValueError(f"Duplicate class pair: {pair}")
        lookup[key] = pair_id
    return lookup

