"""Evaluate a base classifier with a gated high-resolution pair expert."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from expert_model import (
    PairExpert,
    build_pair_lookup,
    load_classifier_model,
    pair_name,
)
from test import (
    locate_run_files,
    make_confusion_matrix,
    parse_train_log,
    save_metrics_csv,
    save_raw_confusion_csv,
)


class DualTransformDataset(Dataset):
    def __init__(self, root: Path, base_transform, expert_transform) -> None:
        source = ImageFolder(root)
        self.classes = source.classes
        self.samples = source.samples
        self.loader = source.loader
        self.base_transform = base_transform
        self.expert_transform = expert_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self.loader(path)
        return (
            self.base_transform(image.copy()),
            self.expert_transform(image),
            label,
            path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base -> confidence gate -> pair expert cascade"
    )
    parser.add_argument("--base-run-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-weights",
        choices=("best", "final"),
        default="best",
    )
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--base-input-size", type=int, default=None)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--expert-confidence", type=float, default=0.55)
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        default=None,
        help="Optional JSON mapping 'class_a|||class_b' to a probability margin",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    if not 0 <= args.margin_threshold <= 1:
        parser.error("margin-threshold must be in [0, 1]")
    if not 0 <= args.expert_confidence <= 1:
        parser.error("expert-confidence must be in [0, 1]")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers non-negative")
    return args


def load_expert(path: Path) -> tuple[PairExpert, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state", "model_name", "classes", "pairs", "input_size"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Invalid expert checkpoint: {path}")
    model = PairExpert(checkpoint["model_name"], checkpoint["pairs"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint


def load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("thresholds-json must contain a JSON object")
    parsed = {str(key): float(value) for key, value in values.items()}
    if any(not 0 <= value <= 1 for value in parsed.values()):
        raise ValueError("All pair thresholds must be in [0, 1]")
    return parsed


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    run_dir, log_path, base_checkpoint = locate_run_files(
        args.base_run_path,
        args.base_weights,
        args.base_checkpoint,
    )
    run = parse_train_log(log_path)
    base_input_size = (
        args.base_input_size if args.base_input_size is not None else run.input_size
    )
    base_model = load_classifier_model(
        run.model_name,
        len(run.classes),
        base_checkpoint,
    )
    expert_model, expert_metadata = load_expert(
        args.expert_checkpoint.expanduser().resolve()
    )
    if list(expert_metadata["classes"]) != run.classes:
        raise ValueError("Expert and base checkpoints have different class orders")

    base_config = resolve_model_data_config(base_model)
    if base_input_size is not None:
        base_config["input_size"] = (3, base_input_size, base_input_size)
    expert_config = resolve_model_data_config(expert_model.backbone)
    expert_size = int(expert_metadata["input_size"])
    expert_config["input_size"] = (3, expert_size, expert_size)
    dataset = DualTransformDataset(
        args.val_path.expanduser().resolve(),
        create_transform(**base_config, is_training=False),
        create_transform(**expert_config, is_training=False),
    )
    if dataset.classes != run.classes:
        raise ValueError("Validation class folders do not match the base training log")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    pairs = tuple(tuple(pair) for pair in expert_metadata["pairs"])
    pair_lookup = build_pair_lookup(run.classes, pairs)
    pair_global_indices = [
        (run.classes.index(pair[0]), run.classes.index(pair[1])) for pair in pairs
    ]
    custom_thresholds = load_thresholds(args.thresholds_json)
    pair_thresholds = [
        custom_thresholds.get(pair_name(pair), args.margin_threshold)
        for pair in pairs
    ]

    base_model.to(device).eval()
    expert_model.to(device).eval()
    use_amp = args.amp and device.type == "cuda"
    all_true: list[np.ndarray] = []
    all_base: list[np.ndarray] = []
    all_final: list[np.ndarray] = []
    prediction_rows: list[list[object]] = []
    pair_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    known_pair_errors = 0
    top2_covered_errors = 0
    threshold_covered_errors = 0

    with torch.inference_mode():
        for base_images, expert_images, labels, paths in tqdm(
            loader,
            desc="Cascade evaluation",
            unit="batch",
        ):
            base_images = base_images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            labels_device = labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                base_logits = base_model(base_images)
            base_probabilities = base_logits.softmax(dim=1)
            top_probabilities, top_indices = base_probabilities.topk(2, dim=1)
            base_predictions = top_indices[:, 0]
            margins = top_probabilities[:, 0] - top_probabilities[:, 1]
            final_predictions = base_predictions.clone()

            batch_pair_ids = torch.full(
                (labels.shape[0],),
                -1,
                dtype=torch.long,
            )
            gated = torch.zeros(labels.shape[0], dtype=torch.bool)
            for index in range(labels.shape[0]):
                first = int(top_indices[index, 0].item())
                second = int(top_indices[index, 1].item())
                pair_id = pair_lookup.get(frozenset((first, second)))
                true_index = int(labels[index].item())
                if int(base_predictions[index].item()) != true_index:
                    true_pair_id = pair_lookup.get(
                        frozenset((int(base_predictions[index].item()), true_index))
                    )
                    if true_pair_id is not None:
                        known_pair_errors += 1
                        if true_index == second:
                            top2_covered_errors += 1
                            if float(margins[index].item()) < pair_thresholds[true_pair_id]:
                                threshold_covered_errors += 1
                if pair_id is None:
                    continue
                batch_pair_ids[index] = pair_id
                if float(margins[index].item()) < pair_thresholds[pair_id]:
                    gated[index] = True

            expert_confidences = torch.full((labels.shape[0],), float("nan"))
            applied = torch.zeros(labels.shape[0], dtype=torch.bool)
            gate_indices = gated.nonzero(as_tuple=False).flatten()
            if gate_indices.numel():
                expert_batch = expert_images[gate_indices].to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                expert_pair_ids = batch_pair_ids[gate_indices].to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    expert_logits = expert_model(expert_batch, expert_pair_ids)
                expert_probabilities = expert_logits.softmax(dim=1)
                confidence, local_predictions = expert_probabilities.max(dim=1)
                for local_index, batch_index_tensor in enumerate(gate_indices):
                    batch_index = int(batch_index_tensor.item())
                    pair_id = int(batch_pair_ids[batch_index].item())
                    expert_confidences[batch_index] = confidence[local_index].cpu()
                    pair_stats[pair_id]["gated"] += 1
                    if float(confidence[local_index].item()) < args.expert_confidence:
                        continue
                    global_prediction = pair_global_indices[pair_id][
                        int(local_predictions[local_index].item())
                    ]
                    final_predictions[batch_index] = global_prediction
                    applied[batch_index] = True
                    pair_stats[pair_id]["applied"] += 1

            labels_cpu = labels.numpy()
            base_cpu = base_predictions.cpu().numpy()
            final_cpu = final_predictions.cpu().numpy()
            top_indices_cpu = top_indices.cpu().numpy()
            margins_cpu = margins.cpu().numpy()
            for index, path in enumerate(paths):
                pair_id = int(batch_pair_ids[index].item())
                base_correct = base_cpu[index] == labels_cpu[index]
                final_correct = final_cpu[index] == labels_cpu[index]
                if pair_id >= 0 and bool(applied[index]):
                    if not base_correct and final_correct:
                        pair_stats[pair_id]["rescued"] += 1
                    if base_correct and not final_correct:
                        pair_stats[pair_id]["damaged"] += 1
                    if base_cpu[index] != final_cpu[index]:
                        pair_stats[pair_id]["changed"] += 1
                prediction_rows.append(
                    [
                        path,
                        run.classes[labels_cpu[index]],
                        run.classes[base_cpu[index]],
                        run.classes[top_indices_cpu[index, 1]],
                        f"{margins_cpu[index]:.6f}",
                        pair_name(pairs[pair_id]) if pair_id >= 0 else "",
                        int(bool(gated[index])),
                        ""
                        if torch.isnan(expert_confidences[index])
                        else f"{float(expert_confidences[index]):.6f}",
                        int(bool(applied[index])),
                        run.classes[final_cpu[index]],
                        int(final_correct),
                    ]
                )

            all_true.append(labels_cpu)
            all_base.append(base_cpu)
            all_final.append(final_cpu)

    y_true = np.concatenate(all_true)
    y_base = np.concatenate(all_base)
    y_final = np.concatenate(all_final)
    base_correct = int((y_base == y_true).sum())
    final_correct = int((y_final == y_true).sum())
    rescued = int(((y_base != y_true) & (y_final == y_true)).sum())
    damaged = int(((y_base == y_true) & (y_final != y_true)).sum())
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.expert_checkpoint.expanduser().resolve().parent / "cascade_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "cascade_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "path",
                "true_class",
                "base_prediction",
                "base_second",
                "base_margin",
                "pair",
                "gated",
                "expert_confidence",
                "expert_applied",
                "final_prediction",
                "final_correct",
            ]
        )
        writer.writerows(prediction_rows)

    pair_rows = []
    for pair_id, pair in enumerate(pairs):
        stats = pair_stats[pair_id]
        pair_rows.append(
            {
                "pair": pair_name(pair),
                "threshold": pair_thresholds[pair_id],
                "gated": stats["gated"],
                "applied": stats["applied"],
                "changed": stats["changed"],
                "rescued": stats["rescued"],
                "damaged": stats["damaged"],
                "net_gain": stats["rescued"] - stats["damaged"],
            }
        )
    with (output_dir / "cascade_pair_stats.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    base_confusion = make_confusion_matrix(y_true, y_base, len(run.classes))
    final_confusion = make_confusion_matrix(y_true, y_final, len(run.classes))
    save_raw_confusion_csv(
        base_confusion,
        run.classes,
        output_dir / "base_confusion_matrix_counts.csv",
    )
    save_raw_confusion_csv(
        final_confusion,
        run.classes,
        output_dir / "cascade_confusion_matrix_counts.csv",
    )
    base_accuracy = save_metrics_csv(
        base_confusion,
        run.classes,
        output_dir / "base_metrics.csv",
    )
    final_accuracy = save_metrics_csv(
        final_confusion,
        run.classes,
        output_dir / "cascade_metrics.csv",
    )
    summary = {
        "samples": int(len(y_true)),
        "base_correct": base_correct,
        "final_correct": final_correct,
        "base_accuracy": base_accuracy,
        "cascade_accuracy": final_accuracy,
        "accuracy_gain_percentage_points": 100.0 * (final_accuracy - base_accuracy),
        "rescued": rescued,
        "damaged": damaged,
        "net_gain": rescued - damaged,
        "known_pair_errors": known_pair_errors,
        "top2_covered_pair_errors": top2_covered_errors,
        "threshold_covered_pair_errors": threshold_covered_errors,
        "top2_pair_error_coverage": (
            top2_covered_errors / known_pair_errors if known_pair_errors else 0.0
        ),
        "pair_stats": pair_rows,
    }
    (output_dir / "cascade_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Base accuracy:    {base_accuracy:.2%}")
    print(f"Cascade accuracy: {final_accuracy:.2%}")
    print(f"Accuracy gain:    {100.0 * (final_accuracy - base_accuracy):+.2f} pp")
    print(f"Rescued/Damaged:  {rescued}/{damaged} (net {rescued - damaged:+d})")
    print(
        f"Pair coverage:    {top2_covered_errors}/{known_pair_errors} "
        f"({summary['top2_pair_error_coverage']:.1%})"
    )
    print(f"Results:          {output_dir}")


if __name__ == "__main__":
    main()

