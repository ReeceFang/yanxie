"""Shared components for the standalone hierarchical-classification experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import timm
import torch
from timm.data import create_transform, resolve_model_data_config
from timm.models import load_checkpoint
from torchvision.datasets import ImageFolder


@dataclass(slots=True)
class RunMetadata:
    run_dir: Path
    model_name: str
    classes: list[str]
    input_size: int | None
    stage: str
    merged_classes: list[str]
    merged_class_name: str
    checkpoint: Path | None = None


def read_merged_classes(path: Path) -> tuple[str, ...]:
    """Read one exact ImageFolder class name per non-empty TXT line."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Merged-class TXT not found: {path}")
    classes = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if len(classes) < 2:
        raise ValueError(f"{path} must contain at least two non-empty class lines")
    duplicates = sorted({name for name in classes if classes.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate class names in {path}: {duplicates}")
    return classes


class HierarchicalImageFolder(ImageFolder):
    """An in-memory coarse or fine view over an unchanged ImageFolder tree."""

    def __init__(
        self,
        root: Path,
        stage: str,
        merged_classes: tuple[str, ...],
        merged_class_name: str,
        transform: Callable | None = None,
    ) -> None:
        super().__init__(root, transform=transform)
        self.original_classes = list(self.classes)
        self.original_class_to_idx = dict(self.class_to_idx)

        missing = [name for name in merged_classes if name not in self.class_to_idx]
        if missing:
            raise ValueError(
                f"Classes from the TXT were not found under {root}: {missing}. "
                f"Available classes: {self.original_classes}"
            )

        if stage == "coarse":
            if merged_class_name in self.class_to_idx:
                raise ValueError(
                    f"Merged class name {merged_class_name!r} conflicts with an "
                    "existing folder; choose another --merged-class-name"
                )
            classes = [
                name for name in self.original_classes if name not in merged_classes
            ]
            classes.append(merged_class_name)
            class_to_idx = {name: index for index, name in enumerate(classes)}
            old_to_new = {
                old_index: class_to_idx[
                    merged_class_name if name in merged_classes else name
                ]
                for name, old_index in self.original_class_to_idx.items()
            }
            samples = [(path, old_to_new[target]) for path, target in self.samples]
        elif stage == "fine":
            # TXT line order defines the fine classifier's label-index order.
            classes = list(merged_classes)
            class_to_idx = {name: index for index, name in enumerate(classes)}
            old_to_new = {
                self.original_class_to_idx[name]: new_index
                for new_index, name in enumerate(classes)
            }
            samples = [
                (path, old_to_new[target])
                for path, target in self.samples
                if target in old_to_new
            ]
        else:
            raise ValueError(f"Unsupported stage: {stage!r}")

        if not samples:
            raise ValueError(f"No samples remain for stage={stage!r} under {root}")
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.imgs = samples
        self.targets = [target for _, target in samples]


def build_pretrained_model(
    model_name: str,
    model_path: Path,
    num_classes: int,
) -> torch.nn.Module:
    """Build a timm model, load local pretrained weights, then replace its head."""
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Pretrained model weights not found: {model_path}")
    model = timm.create_model(model_name, pretrained=False)
    pretrained_num_classes = model.pretrained_cfg.get("num_classes", model.num_classes)
    model.reset_classifier(num_classes=pretrained_num_classes)
    load_checkpoint(model, str(model_path), strict=True)
    model.reset_classifier(num_classes=num_classes)
    return model


def build_eval_transform(
    model: torch.nn.Module,
    input_size: int | None,
) -> Callable:
    data_config = resolve_model_data_config(model)
    if input_size is not None:
        data_config["input_size"] = (3, input_size, input_size)
    return create_transform(**data_config, is_training=False)


def save_run_metadata(path: Path, metadata: RunMetadata) -> None:
    payload = {
        "schema_version": 1,
        "model": metadata.model_name,
        "num_classes": len(metadata.classes),
        "classes": metadata.classes,
        "input_size": metadata.input_size,
        "stage": metadata.stage,
        "merged_classes": metadata.merged_classes,
        "merged_class_name": metadata.merged_class_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_run_metadata(run_dir: Path, weights: str) -> RunMetadata:
    run_dir = run_dir.expanduser().resolve()
    metadata_path = run_dir / "run_config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "model",
        "classes",
        "input_size",
        "stage",
        "merged_classes",
        "merged_class_name",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing metadata fields in {metadata_path}: {missing}")

    classes = raw["classes"]
    merged_classes = raw["merged_classes"]
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError(f"Invalid classes in {metadata_path}")
    if not isinstance(merged_classes, list) or not all(
        isinstance(item, str) for item in merged_classes
    ):
        raise ValueError(f"Invalid merged_classes in {metadata_path}")
    if raw.get("num_classes", len(classes)) != len(classes):
        raise ValueError(f"num_classes does not match classes in {metadata_path}")

    checkpoint = run_dir / f"{weights}_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return RunMetadata(
        run_dir=run_dir,
        model_name=raw["model"],
        classes=classes,
        input_size=raw["input_size"],
        stage=raw["stage"],
        merged_classes=merged_classes,
        merged_class_name=raw["merged_class_name"],
        checkpoint=checkpoint,
    )


def load_trained_model(metadata: RunMetadata) -> torch.nn.Module:
    if metadata.checkpoint is None:
        raise ValueError("Run metadata does not specify a checkpoint")
    model = timm.create_model(
        metadata.model_name,
        pretrained=False,
        num_classes=len(metadata.classes),
    )
    state_dict = torch.load(metadata.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {metadata.checkpoint}")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    return model


def validate_hierarchy(coarse: RunMetadata, fine: RunMetadata) -> None:
    if coarse.stage != "coarse":
        raise ValueError(f"Expected a coarse run, got stage={coarse.stage!r}")
    if fine.stage != "fine":
        raise ValueError(f"Expected a fine run, got stage={fine.stage!r}")
    if set(coarse.merged_classes) != set(fine.merged_classes):
        raise ValueError("Coarse and fine runs use different merged classes")
    if coarse.merged_class_name != fine.merged_class_name:
        raise ValueError("Coarse and fine runs use different merged class names")
    if coarse.classes.count(coarse.merged_class_name) != 1:
        raise ValueError("Coarse labels do not contain exactly one merged class")
    if any(name in coarse.classes for name in coarse.merged_classes):
        raise ValueError("A merged original class exists in the coarse labels")
    if set(fine.classes) != set(coarse.merged_classes):
        raise ValueError("Fine labels do not exactly match the merged original classes")
