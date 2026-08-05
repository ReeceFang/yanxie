"""Standalone cascade inference and end-to-end validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from tqdm import tqdm

from common import (
    RunMetadata,
    build_eval_transform,
    load_run_metadata,
    load_trained_model,
    validate_hierarchy,
)


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform: Callable) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        return load_image(path, self.transform), str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hierarchical inference or evaluate an original validation set"
    )
    parser.add_argument("--coarse-run", type=Path, required=True)
    parser.add_argument("--fine-run", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        dest="inputs",
        type=Path,
        nargs="+",
        help="Image files/directories for unlabeled inference",
    )
    source.add_argument(
        "--val-dir",
        type=Path,
        help="Original labeled ImageFolder val directory for end-to-end evaluation",
    )
    parser.add_argument(
        "--weights",
        choices=("best", "last"),
        default="best",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("cascade_predictions.csv"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def load_image(path: Path, transform: Callable) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def is_supported_image(path: Path) -> bool:
    extensions = tuple(extension.lower() for extension in IMG_EXTENSIONS)
    return path.is_file() and path.suffix.lower() in extensions


def discover_unlabeled_images(inputs: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            candidates.extend(item for item in path.rglob("*") if is_supported_image(item))
        elif is_supported_image(path):
            candidates.append(path)
        elif path.exists():
            raise ValueError(f"Not a supported image file: {path}")
        else:
            raise FileNotFoundError(f"Input not found: {path}")
    return unique_sorted_images(candidates)


def discover_labeled_images(
    val_dir: Path,
    valid_classes: set[str],
) -> tuple[list[Path], dict[str, str]]:
    val_dir = val_dir.expanduser().resolve()
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    paths = unique_sorted_images(
        [item for item in val_dir.rglob("*") if is_supported_image(item)]
    )
    true_by_path: dict[str, str] = {}
    invalid: list[Path] = []
    for path in paths:
        relative_parts = path.relative_to(val_dir).parts
        if len(relative_parts) < 2 or relative_parts[0] not in valid_classes:
            invalid.append(path)
        else:
            true_by_path[str(path)] = relative_parts[0]
    if invalid:
        raise ValueError(
            "Every validation image must be inside an original class folder. "
            f"Invalid examples: {invalid[:5]}"
        )
    return paths, true_by_path


def unique_sorted_images(candidates: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in sorted(candidates):
        key = str(path).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise ValueError("No supported images were found")
    return unique


@torch.inference_mode()
def run_cascade(
    coarse_run: RunMetadata,
    fine_run: RunMetadata,
    paths: list[Path],
    true_by_path: dict[str, str] | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> list[dict]:
    coarse_model = load_trained_model(coarse_run).to(device).eval()
    fine_model = load_trained_model(fine_run).to(device).eval()
    coarse_transform = build_eval_transform(coarse_model, coarse_run.input_size)
    fine_transform = build_eval_transform(fine_model, fine_run.input_size)
    loader = DataLoader(
        ImagePathDataset(paths, coarse_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    merged_index = coarse_run.classes.index(coarse_run.merged_class_name)
    records: list[dict] = []
    for coarse_images, path_strings in tqdm(loader, desc="Cascade inference"):
        coarse_images = coarse_images.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=use_amp):
            coarse_logits = coarse_model(coarse_images)
        coarse_probabilities = coarse_logits.float().softmax(dim=1)
        coarse_confidences, coarse_predictions = coarse_probabilities.max(dim=1)
        coarse_prediction_list = coarse_predictions.cpu().tolist()
        coarse_confidence_list = coarse_confidences.cpu().tolist()

        routed_positions = [
            position
            for position, prediction in enumerate(coarse_prediction_list)
            if prediction == merged_index
        ]
        fine_results: dict[int, tuple[int, float]] = {}
        if routed_positions:
            fine_images = torch.stack(
                [
                    load_image(Path(path_strings[position]), fine_transform)
                    for position in routed_positions
                ]
            ).to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                fine_logits = fine_model(fine_images)
            fine_probabilities = fine_logits.float().softmax(dim=1)
            fine_confidences, fine_predictions = fine_probabilities.max(dim=1)
            fine_results = {
                position: (prediction, confidence)
                for position, prediction, confidence in zip(
                    routed_positions,
                    fine_predictions.cpu().tolist(),
                    fine_confidences.cpu().tolist(),
                    strict=True,
                )
            }

        for position, path_string in enumerate(path_strings):
            coarse_index = coarse_prediction_list[position]
            coarse_class = coarse_run.classes[coarse_index]
            coarse_confidence = coarse_confidence_list[position]
            routed = position in fine_results
            fine_class: str | None = None
            fine_confidence: float | None = None
            final_class = coarse_class
            final_confidence = coarse_confidence
            if routed:
                fine_index, fine_confidence = fine_results[position]
                fine_class = fine_run.classes[fine_index]
                final_class = fine_class
                # P(final class) = P(merged group) * P(fine class | merged group).
                final_confidence = coarse_confidence * fine_confidence
            record = {
                "path": path_string,
                "coarse_class": coarse_class,
                "coarse_confidence": coarse_confidence,
                "routed_to_fine": routed,
                "fine_class": fine_class if fine_class is not None else "",
                "fine_confidence": (
                    fine_confidence if fine_confidence is not None else ""
                ),
                "final_class": final_class,
                "final_confidence": final_confidence,
            }
            if true_by_path is not None:
                true_class = true_by_path[path_string]
                record["true_class"] = true_class
                record["correct"] = final_class == true_class
            records.append(record)
    return records


def save_csv(records: list[dict], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output_path


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def calculate_summary(records: list[dict], merged_classes: set[str]) -> dict:
    merged_records = [record for record in records if record["true_class"] in merged_classes]
    other_records = [record for record in records if record["true_class"] not in merged_classes]
    routed_merged = [record for record in merged_records if record["routed_to_fine"]]
    per_class_totals: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    for record in records:
        true_class = record["true_class"]
        per_class_totals[true_class] += 1
        per_class_correct[true_class] += int(record["correct"])
    return {
        "num_images": len(records),
        "routed_to_fine": sum(int(record["routed_to_fine"]) for record in records),
        "overall_accuracy": safe_ratio(
            sum(int(record["correct"]) for record in records),
            len(records),
        ),
        "merged_subset_end_to_end_accuracy": safe_ratio(
            sum(int(record["correct"]) for record in merged_records),
            len(merged_records),
        ),
        "other_classes_accuracy": safe_ratio(
            sum(int(record["correct"]) for record in other_records),
            len(other_records),
        ),
        "merged_route_recall": safe_ratio(len(routed_merged), len(merged_records)),
        "other_route_false_positive_rate": safe_ratio(
            sum(int(record["routed_to_fine"]) for record in other_records),
            len(other_records),
        ),
        "fine_accuracy_given_correct_route": safe_ratio(
            sum(int(record["correct"]) for record in routed_merged),
            len(routed_merged),
        ),
        "per_class_accuracy": {
            class_name: per_class_correct[class_name] / total
            for class_name, total in sorted(per_class_totals.items())
        },
    }


def save_summary(summary: dict, output_csv: Path) -> Path:
    summary_path = output_csv.with_name(f"{output_csv.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def main() -> None:
    args = parse_args()
    coarse_run = load_run_metadata(args.coarse_run, args.weights)
    fine_run = load_run_metadata(args.fine_run, args.weights)
    validate_hierarchy(coarse_run, fine_run)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    use_amp = args.amp and device.type == "cuda"

    true_by_path = None
    if args.val_dir is not None:
        original_classes = set(coarse_run.classes) - {coarse_run.merged_class_name}
        original_classes.update(fine_run.classes)
        paths, true_by_path = discover_labeled_images(args.val_dir, original_classes)
    else:
        paths = discover_unlabeled_images(args.inputs)

    records = run_cascade(
        coarse_run,
        fine_run,
        paths,
        true_by_path,
        args.batch_size,
        args.num_workers,
        device,
        use_amp,
    )
    output_csv = save_csv(records, args.output_csv)
    print(f"Images:          {len(records)}")
    print(f"Routed to fine:  {sum(bool(row['routed_to_fine']) for row in records)}")
    print(f"Predictions:     {output_csv}")

    if true_by_path is not None:
        summary = calculate_summary(records, set(fine_run.classes))
        summary_path = save_summary(summary, output_csv)
        print(f"Overall acc:     {format_rate(summary['overall_accuracy'])}")
        print(
            "Merged E2E acc:  "
            f"{format_rate(summary['merged_subset_end_to_end_accuracy'])}"
        )
        print(f"Merge route rec: {format_rate(summary['merged_route_recall'])}")
        print(f"Fine given route:{format_rate(summary['fine_accuracy_given_correct_route']):>8}")
        print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()
