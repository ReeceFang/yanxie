"""Evaluate a soft-routed two-stage image-classification cascade.

The first-stage model is trained on the merged dataset. Its Top-K coarse
classes form a candidate set. Direct classes retain their first-stage
probability, while merged classes are expanded by ``P(group) * P(class|group)``
using the corresponding second-stage model. The highest score in the original
unmerged class space is the final prediction.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm


@dataclass(slots=True)
class RunSpec:
    name: str
    run_dir: Path
    log_path: Path
    checkpoint_path: Path
    info: Any


class RoutedImageDataset:
    """Load selected image paths and retain their original sample indices."""

    def __init__(
        self,
        image_paths: list[str],
        sample_indices: list[int],
        transform: Callable[[Any], Any],
    ) -> None:
        self.image_paths = image_paths
        self.sample_indices = sample_indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image

        sample_index = self.sample_indices[index]
        with Image.open(self.image_paths[sample_index]) as image:
            transformed = self.transform(image.convert("RGB"))
        return transformed, sample_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Top-K soft-routed two-stage classifier cascade"
    )
    parser.add_argument(
        "--merged-run-path",
        type=Path,
        required=True,
        help="一级 merged 分类器的训练输出目录",
    )
    parser.add_argument(
        "--second-runs-path",
        type=Path,
        required=True,
        help="二级分类器根目录；其下目录名称必须与 JSON 键一致",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        required=True,
        help="未合并原始数据集的 ImageFolder 验证目录",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        required=True,
        help="类别合并 JSON 文件",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--route-top-k",
        type=int,
        default=2,
        help="一级保留多少个粗类别参与软路由（默认：2）",
    )
    parser.add_argument(
        "--stage1-temperature",
        type=float,
        default=1.0,
        help="一级 softmax 温度，未校准时保持 1.0",
    )
    parser.add_argument(
        "--stage2-temperature",
        type=float,
        default=1.0,
        help="所有二级模型的 softmax 温度，未校准时保持 1.0",
    )
    parser.add_argument(
        "--weights",
        choices=("best", "final"),
        default="best",
        help="所有分类器统一使用 best_model.pth 或 last_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果目录；默认按 Top-K 写入独立的 cascade_soft_topK_results",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="覆盖所有训练日志记录的输入尺寸；通常无需设置",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备，例如 cuda、cuda:0 或 cpu；默认自动选择",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在 CUDA 上使用自动混合精度",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.route_top_k < 1:
        parser.error("--route-top-k must be at least 1")
    if args.stage1_temperature <= 0:
        parser.error("--stage1-temperature must be positive")
    if args.stage2_temperature <= 0:
        parser.error("--stage2-temperature must be positive")
    if args.input_size is not None and args.input_size < 1:
        parser.error("--input-size must be at least 1")
    return args


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 中存在重复键: {key!r}")
        result[key] = value
    return result


def _validate_class_name(name: Any, location: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{location} 必须是非空字符串")
    if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"{location} 只能是类别名，不能包含路径: {name!r}")
    return name


def load_merge_mapping(path: Path) -> dict[str, list[str]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到类别合并 JSON: {path}")

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            raw = json.load(file, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 格式错误（第 {exc.lineno} 行，第 {exc.colno} 列）"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError("JSON 顶层必须是对象：{合并类别: [原始类别, ...]}")
    if not raw:
        raise ValueError("类别合并 JSON 不能为空")

    mapping: dict[str, list[str]] = {}
    member_owner: dict[str, str] = {}
    for branch, members in raw.items():
        branch = _validate_class_name(branch, "JSON 键")
        if not isinstance(members, list) or not members:
            raise ValueError(f"分支 {branch!r} 的值必须是非空数组")

        validated_members: list[str] = []
        for index, member in enumerate(members, start=1):
            member = _validate_class_name(
                member, f"分支 {branch!r} 的第 {index} 个类别"
            )
            previous = member_owner.get(member)
            if previous is not None:
                raise ValueError(
                    f"原始类别 {member!r} 重复出现在分支 "
                    f"{previous!r} 和 {branch!r} 中"
                )
            member_owner[member] = branch
            validated_members.append(member)
        mapping[branch] = validated_members
    return mapping


def validate_class_configuration(
    global_classes: list[str],
    merged_classes: list[str],
    mapping: dict[str, list[str]],
    second_classes: dict[str, list[str]],
) -> dict[str, int]:
    """Validate all name spaces and return original class name -> global ID."""
    if len(global_classes) != len(set(global_classes)):
        raise ValueError("原始验证集包含重复类别名")
    if len(merged_classes) != len(set(merged_classes)):
        raise ValueError("一级训练日志包含重复类别名")

    global_set = set(global_classes)
    branch_set = set(mapping)
    member_set = {member for members in mapping.values() for member in members}

    overlapping_keys = sorted(branch_set & global_set)
    if overlapping_keys:
        raise ValueError(
            "JSON 键必须是合并后的新类别名，不能与原始类别重名: "
            + ", ".join(overlapping_keys)
        )

    missing_members = sorted(member_set - global_set)
    if missing_members:
        raise ValueError(
            "JSON 中的以下类别不在原始验证集中: " + ", ".join(missing_members)
        )

    expected_merged = (global_set - member_set) | branch_set
    actual_merged = set(merged_classes)
    if actual_merged != expected_merged:
        missing = sorted(expected_merged - actual_merged)
        extra = sorted(actual_merged - expected_merged)
        details: list[str] = []
        if missing:
            details.append("一级日志缺少: " + ", ".join(missing))
        if extra:
            details.append("一级日志多出: " + ", ".join(extra))
        raise ValueError("一级类别与 JSON/原始数据集不一致；" + "；".join(details))

    missing_second_runs = sorted(branch_set - set(second_classes))
    extra_second_runs = sorted(set(second_classes) - branch_set)
    if missing_second_runs or extra_second_runs:
        details = []
        if missing_second_runs:
            details.append("缺少二级配置: " + ", ".join(missing_second_runs))
        if extra_second_runs:
            details.append("未知二级配置: " + ", ".join(extra_second_runs))
        raise ValueError("二级配置键与 JSON 不一致；" + "；".join(details))

    for branch, expected_members in mapping.items():
        logged_classes = second_classes[branch]
        if len(logged_classes) != len(set(logged_classes)):
            raise ValueError(f"二级日志 {branch!r} 包含重复类别名")
        expected_set = set(expected_members)
        logged_set = set(logged_classes)
        if logged_set != expected_set:
            missing = sorted(expected_set - logged_set)
            extra = sorted(logged_set - expected_set)
            details = []
            if missing:
                details.append("缺少: " + ", ".join(missing))
            if extra:
                details.append("多出: " + ", ".join(extra))
            raise ValueError(
                f"二级日志 {branch!r} 的类别与 JSON 不一致；" + "；".join(details)
            )

    return {name: index for index, name in enumerate(global_classes)}


def resolve_stage1_prediction(
    local_id: int,
    merged_classes: list[str],
    mapping: dict[str, list[str]],
    global_class_to_id: dict[str, int],
) -> tuple[str | None, int | None]:
    """Return (branch, final global ID); exactly one item is non-None."""
    if not 0 <= local_id < len(merged_classes):
        raise ValueError(f"一级预测 ID 越界: {local_id}")
    class_name = merged_classes[local_id]
    if class_name in mapping:
        return class_name, None
    try:
        return None, global_class_to_id[class_name]
    except KeyError as exc:
        raise ValueError(f"一级普通类别无法映射到原始全局 ID: {class_name!r}") from exc


def resolve_secondary_prediction(
    branch: str,
    local_id: int,
    second_classes: dict[str, list[str]],
    global_class_to_id: dict[str, int],
) -> tuple[str, int]:
    """Map a second-stage local ID through its train.log class ordering."""
    classes = second_classes[branch]
    if not 0 <= local_id < len(classes):
        raise ValueError(f"二级分支 {branch!r} 的预测 ID 越界: {local_id}")
    class_name = classes[local_id]
    try:
        return class_name, global_class_to_id[class_name]
    except KeyError as exc:
        raise ValueError(
            f"二级类别无法映射到原始全局 ID: {branch!r}/{class_name!r}"
        ) from exc


def build_run_spec(
    name: str,
    run_path: Path,
    weights: str,
    locate_run_files: Callable[..., tuple[Path, Path, Path]],
    parse_train_log: Callable[[Path], Any],
) -> RunSpec:
    run_dir, log_path, checkpoint_path = locate_run_files(run_path, weights, None)
    info = parse_train_log(log_path)
    return RunSpec(name, run_dir, log_path, checkpoint_path, info)


def build_eval_transform(model: Any, input_size: int | None) -> Any:
    from timm.data import create_transform, resolve_model_data_config

    data_config = resolve_model_data_config(model)
    if input_size is not None:
        data_config["input_size"] = (3, input_size, input_size)
    return create_transform(**data_config, is_training=False)


def predict_indexed_loader(
    model: Any,
    loader: Any,
    device: Any,
    amp: bool,
    description: str,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original sample indices and complete calibrated probabilities."""
    import torch

    model.eval()
    sample_index_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    use_amp = amp and device.type == "cuda"

    with torch.inference_mode():
        for images, sample_indices in tqdm(loader, desc=description, unit="batch"):
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            probabilities = (logits.float() / temperature).softmax(dim=1)

            sample_index_batches.append(sample_indices.numpy())
            probability_batches.append(probabilities.cpu().numpy())

    return (
        np.concatenate(sample_index_batches).astype(np.int64, copy=False),
        np.concatenate(probability_batches).astype(np.float64, copy=False),
    )


def initialize_soft_routes(
    stage1_probabilities: np.ndarray,
    merged_classes: list[str],
    mapping: dict[str, list[str]],
    global_class_to_id: dict[str, int],
    route_top_k: int,
) -> tuple[
    np.ndarray,
    dict[str, list[tuple[int, float]]],
    np.ndarray,
    np.ndarray,
]:
    """Create direct-class scores and collect merged Top-K candidates."""
    if stage1_probabilities.ndim != 2:
        raise ValueError("一级概率必须是二维矩阵")
    if stage1_probabilities.shape[1] != len(merged_classes):
        raise ValueError("一级概率列数与一级训练日志类别数不一致")
    if not np.all(np.isfinite(stage1_probabilities)):
        raise ValueError("一级概率中存在 NaN 或无穷值")
    if np.any(stage1_probabilities < 0):
        raise ValueError("一级概率不能为负数")

    effective_k = min(route_top_k, len(merged_classes))
    # There are only a few coarse classes, so a stable full sort is clearer and
    # deterministic even when reduced-precision logits produce tied values.
    top_ids = np.argsort(-stage1_probabilities, axis=1, kind="stable")[:, :effective_k]
    top_probs = np.take_along_axis(stage1_probabilities, top_ids, axis=1)

    sample_count = stage1_probabilities.shape[0]
    global_scores = np.zeros(
        (sample_count, len(global_class_to_id)), dtype=np.float64
    )
    routes: dict[str, list[tuple[int, float]]] = {
        branch: [] for branch in mapping
    }

    for sample_index in range(sample_count):
        for local_id, probability in zip(
            top_ids[sample_index].tolist(),
            top_probs[sample_index].tolist(),
            strict=True,
        ):
            coarse_name = merged_classes[local_id]
            if coarse_name in mapping:
                routes[coarse_name].append((sample_index, probability))
            else:
                try:
                    global_id = global_class_to_id[coarse_name]
                except KeyError as exc:
                    raise ValueError(
                        f"一级普通类别无法映射到原始全局 ID: {coarse_name!r}"
                    ) from exc
                global_scores[sample_index, global_id] = probability

    return global_scores, routes, top_ids, top_probs


def apply_secondary_probabilities(
    global_scores: np.ndarray,
    branch: str,
    sample_indices: np.ndarray,
    branch_probabilities: np.ndarray,
    secondary_probabilities: np.ndarray,
    second_classes: dict[str, list[str]],
    global_class_to_id: dict[str, int],
) -> None:
    """Write P(group) * P(original class | group) into global score space."""
    classes = second_classes[branch]
    if secondary_probabilities.shape != (len(sample_indices), len(classes)):
        raise ValueError(f"二级分支 {branch!r} 返回的概率矩阵尺寸不正确")
    if len(branch_probabilities) != len(sample_indices):
        raise ValueError(f"二级分支 {branch!r} 的一级概率数量不正确")
    if not np.all(np.isfinite(secondary_probabilities)):
        raise ValueError(f"二级分支 {branch!r} 的概率中存在 NaN 或无穷值")
    if np.any(secondary_probabilities < 0):
        raise ValueError(f"二级分支 {branch!r} 的概率不能为负数")

    for local_id, class_name in enumerate(classes):
        try:
            global_id = global_class_to_id[class_name]
        except KeyError as exc:
            raise ValueError(
                f"二级类别无法映射到原始全局 ID: {branch!r}/{class_name!r}"
            ) from exc
        global_scores[sample_indices, global_id] = (
            branch_probabilities * secondary_probabilities[:, local_id]
        )


def normalize_global_scores(global_scores: np.ndarray) -> np.ndarray:
    """Normalize the truncated Top-K score mass for confidence reporting."""
    score_sums = global_scores.sum(axis=1, keepdims=True)
    if np.any(score_sums <= 0):
        missing_count = int((score_sums[:, 0] <= 0).sum())
        raise RuntimeError(f"有 {missing_count} 个样本没有任何全局候选分数")
    return global_scores / score_sums


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def save_cascade_metrics_csv(
    matrix: np.ndarray,
    classes: list[str],
    path: Path,
) -> float:
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    total = int(matrix.sum())
    if total == 0:
        raise ValueError("不能为零样本数据集生成指标")
    accuracy = float(true_positive.sum()) / total
    macro = (float(precision.mean()), float(recall.mean()), float(f1.mean()))
    weighted = (
        float(np.average(precision, weights=support)),
        float(np.average(recall, weights=support)),
        float(np.average(f1, weights=support)),
    )

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["class", "precision_or_top_k_accuracy", "recall", "f1_score", "support"]
        )
        for index, class_name in enumerate(classes):
            writer.writerow(
                [
                    class_name,
                    f"{precision[index]:.6f}",
                    f"{recall[index]:.6f}",
                    f"{f1[index]:.6f}",
                    int(support[index]),
                ]
            )
        writer.writerow(["Top-1", f"{accuracy:.6f}", "", "", total])
        writer.writerow(
            ["Macro", f"{macro[0]:.6f}", f"{macro[1]:.6f}", f"{macro[2]:.6f}", total]
        )
        writer.writerow(
            [
                "Weight",
                f"{weighted[0]:.6f}",
                f"{weighted[1]:.6f}",
                f"{weighted[2]:.6f}",
                total,
            ]
        )
    return accuracy


def save_predictions_csv(
    path: Path,
    image_paths: list[str],
    y_true: np.ndarray,
    global_classes: list[str],
    mapping: dict[str, list[str]],
    merged_classes: list[str],
    stage1_top_ids: np.ndarray,
    stage1_top_probabilities: np.ndarray,
    secondary_details: list[list[dict[str, Any]]],
    final_global_ids: np.ndarray,
    final_probabilities: np.ndarray,
) -> None:
    member_owner = {
        member: branch for branch, members in mapping.items() for member in members
    }
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_path",
                "true_global_id",
                "true_class",
                "true_coarse_class",
                "true_coarse_in_stage1_topk",
                "stage1_candidates",
                "second_stage_details",
                "winning_source",
                "final_global_id",
                "final_class",
                "final_truncated_probability",
                "correct",
            ]
        )
        for index, image_path in enumerate(image_paths):
            true_id = int(y_true[index])
            true_class = global_classes[true_id]
            true_coarse_class = member_owner.get(true_class, true_class)
            final_id = int(final_global_ids[index])
            final_class = global_classes[final_id]
            stage1_candidates = [
                {
                    "rank": rank + 1,
                    "local_id": int(local_id),
                    "class": merged_classes[int(local_id)],
                    "probability": round(
                        float(stage1_top_probabilities[index, rank]), 8
                    ),
                }
                for rank, local_id in enumerate(stage1_top_ids[index])
            ]
            candidate_names = {item["class"] for item in stage1_candidates}
            writer.writerow(
                [
                    image_path,
                    true_id,
                    true_class,
                    true_coarse_class,
                    int(true_coarse_class in candidate_names),
                    json.dumps(stage1_candidates, ensure_ascii=False),
                    json.dumps(secondary_details[index], ensure_ascii=False),
                    member_owner.get(final_class, "direct"),
                    final_id,
                    final_class,
                    f"{final_probabilities[index, final_id]:.6f}",
                    int(final_id == true_id),
                ]
            )


def _clear_model_memory(device: Any) -> None:
    import torch

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_cascade(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    from test import (
        configure_matplotlib,
        load_model,
        locate_run_files,
        normalize_columns,
        parse_train_log,
        plot_normalized_confusion_matrix,
        save_raw_confusion_csv,
    )

    merged_run_path = args.merged_run_path.expanduser().resolve()
    second_runs_path = args.second_runs_path.expanduser().resolve()
    val_path = args.val_path.expanduser().resolve()
    if not second_runs_path.is_dir():
        raise FileNotFoundError(f"找不到二级分类器根目录: {second_runs_path}")
    if not val_path.is_dir():
        raise FileNotFoundError(f"找不到原始验证集目录: {val_path}")

    print("[1/5] 正在读取 JSON、训练日志和类别 ID 空间...", flush=True)
    mapping = load_merge_mapping(args.mapping_json)
    original_dataset = ImageFolder(val_path)
    if len(original_dataset) == 0:
        raise ValueError(f"原始验证集为空: {val_path}")
    global_classes = original_dataset.classes
    image_paths = [path for path, _ in original_dataset.samples]
    y_true = np.asarray(original_dataset.targets, dtype=np.int64)

    merged_spec = build_run_spec(
        "merged",
        merged_run_path,
        args.weights,
        locate_run_files,
        parse_train_log,
    )
    second_specs: dict[str, RunSpec] = {}
    for branch in mapping:
        second_specs[branch] = build_run_spec(
            branch,
            second_runs_path / branch,
            args.weights,
            locate_run_files,
            parse_train_log,
        )
    second_classes = {
        branch: spec.info.classes for branch, spec in second_specs.items()
    }
    global_class_to_id = validate_class_configuration(
        global_classes,
        merged_spec.info.classes,
        mapping,
        second_classes,
    )
    print(
        f"类别校验通过：原始 {len(global_classes)} 类，"
        f"一级 {len(merged_spec.info.classes)} 类，二级 {len(mapping)} 个分支。"
    )

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 CUDA 不可用；请使用 --device cpu")
    pin_memory = device.type == "cuda"
    sample_count = len(image_paths)

    print(f"[2/5] 正在运行一级 merged 分类器（{sample_count} 个样本）...", flush=True)
    merged_model = load_model(
        merged_spec.info.model_name,
        len(merged_spec.info.classes),
        merged_spec.checkpoint_path,
    ).to(device)
    merged_input_size = (
        args.input_size if args.input_size is not None else merged_spec.info.input_size
    )
    if merged_input_size is None:
        print("Warning: 一级日志未记录 input_size，将使用模型默认尺寸。")
    merged_transform = build_eval_transform(merged_model, merged_input_size)
    all_indices = list(range(sample_count))
    merged_loader = DataLoader(
        RoutedImageDataset(image_paths, all_indices, merged_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    stage1_indices, stage1_probs_ordered = predict_indexed_loader(
        merged_model,
        merged_loader,
        device,
        args.amp,
        "Stage 1",
        args.stage1_temperature,
    )
    stage1_probabilities = np.full(
        (sample_count, len(merged_spec.info.classes)), np.nan, dtype=np.float64
    )
    stage1_probabilities[stage1_indices] = stage1_probs_ordered
    if np.any(~np.isfinite(stage1_probabilities)):
        raise RuntimeError("一级推理没有返回全部样本")
    del merged_loader, merged_transform
    del merged_model
    _clear_model_memory(device)

    global_scores, routes, stage1_top_ids, stage1_top_probabilities = (
        initialize_soft_routes(
            stage1_probabilities,
            merged_spec.info.classes,
            mapping,
            global_class_to_id,
            args.route_top_k,
        )
    )
    secondary_details: list[list[dict[str, Any]]] = [
        [] for _ in range(sample_count)
    ]
    routed_sample_indices = {
        sample_index
        for routed_items in routes.values()
        for sample_index, _ in routed_items
    }
    secondary_evaluation_count = sum(len(items) for items in routes.values())
    print(
        f"[3/5] Top-{stage1_top_ids.shape[1]} 软路由完成："
        f"{len(routed_sample_indices)} 个样本需要二级推理，"
        f"共 {secondary_evaluation_count} 次样本-分支计算。"
    )
    for branch, routed_items in routes.items():
        print(f"  {branch}: {len(routed_items)} 个候选样本")

    print("[4/5] 正在逐个运行二级分类器...", flush=True)
    for branch, routed_items in routes.items():
        if not routed_items:
            print(f"跳过二级分支 {branch}：没有路由样本。")
            continue

        routed_indices = [sample_index for sample_index, _ in routed_items]
        stage1_branch_probability = {
            sample_index: probability for sample_index, probability in routed_items
        }

        spec = second_specs[branch]
        second_model = load_model(
            spec.info.model_name,
            len(spec.info.classes),
            spec.checkpoint_path,
        ).to(device)
        second_input_size = (
            args.input_size if args.input_size is not None else spec.info.input_size
        )
        if second_input_size is None:
            print(f"Warning: 二级日志 {branch!r} 未记录 input_size，将使用模型默认尺寸。")
        second_transform = build_eval_transform(second_model, second_input_size)
        second_loader = DataLoader(
            RoutedImageDataset(image_paths, routed_indices, second_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        sample_indices, secondary_probabilities = predict_indexed_loader(
            second_model,
            second_loader,
            device,
            args.amp,
            f"Stage 2 [{branch}]",
            args.stage2_temperature,
        )
        ordered_branch_probabilities = np.asarray(
            [stage1_branch_probability[int(index)] for index in sample_indices],
            dtype=np.float64,
        )
        apply_secondary_probabilities(
            global_scores,
            branch,
            sample_indices,
            ordered_branch_probabilities,
            secondary_probabilities,
            second_classes,
            global_class_to_id,
        )

        top1_local_ids = secondary_probabilities.argmax(axis=1)
        for row, sample_index in enumerate(sample_indices.tolist()):
            local_id = int(top1_local_ids[row])
            class_name = second_classes[branch][local_id]
            conditional_probability = float(secondary_probabilities[row, local_id])
            branch_probability = float(ordered_branch_probabilities[row])
            secondary_details[sample_index].append(
                {
                    "branch": branch,
                    "stage1_probability": round(branch_probability, 8),
                    "top1_local_id": local_id,
                    "top1_class": class_name,
                    "top1_conditional_probability": round(
                        conditional_probability, 8
                    ),
                    "top1_global_score": round(
                        branch_probability * conditional_probability, 8
                    ),
                }
            )

        del second_loader, second_transform
        del second_model
        _clear_model_memory(device)

    final_probabilities = normalize_global_scores(global_scores)
    final_global_ids = final_probabilities.argmax(axis=1).astype(np.int64)
    if not np.all((0 <= final_global_ids) & (final_global_ids < len(global_classes))):
        raise RuntimeError("最终预测中存在越界的全局类别 ID")

    print("[5/5] 正在生成最终指标、混淆矩阵和逐样本明细...", flush=True)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else merged_spec.run_dir
        / f"cascade_soft_top{stage1_top_ids.shape[1]}_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion = make_confusion_matrix(y_true, final_global_ids, len(global_classes))
    save_raw_confusion_csv(
        confusion,
        global_classes,
        output_dir / "confusion_matrix_counts.csv",
    )
    configure_matplotlib()
    plot_normalized_confusion_matrix(
        normalize_columns(confusion),
        global_classes,
        output_dir / "confusion_matrix_column_normalized.png",
    )
    accuracy = save_cascade_metrics_csv(
        confusion,
        global_classes,
        output_dir / "metrics.csv",
    )
    direct_accuracy = float((final_global_ids == y_true).mean())
    if not np.isclose(accuracy, direct_accuracy):
        raise RuntimeError("Top-1 准确率与混淆矩阵不一致")
    save_predictions_csv(
        output_dir / "predictions.csv",
        image_paths,
        y_true,
        global_classes,
        mapping,
        merged_spec.info.classes,
        stage1_top_ids,
        stage1_top_probabilities,
        secondary_details,
        final_global_ids,
        final_probabilities,
    )

    member_owner = {
        member: branch for branch, members in mapping.items() for member in members
    }
    true_coarse_classes = [
        member_owner.get(global_classes[int(global_id)], global_classes[int(global_id)])
        for global_id in y_true
    ]
    topk_hit_count = sum(
        true_coarse in {
            merged_spec.info.classes[int(local_id)]
            for local_id in stage1_top_ids[index]
        }
        for index, true_coarse in enumerate(true_coarse_classes)
    )
    direct_winner_count = sum(
        global_classes[int(global_id)] not in member_owner
        for global_id in final_global_ids
    )
    print(f"Merged model: {merged_spec.info.model_name}")
    print(f"Checkpoint:   {merged_spec.checkpoint_path}")
    print(f"Samples:      {sample_count}")
    print(f"Route Top-K:  {stage1_top_ids.shape[1]}")
    print(f"Coarse hit:   {topk_hit_count / sample_count:.2%}")
    print(f"2nd samples:  {len(routed_sample_indices)}")
    print(f"2nd calls:    {secondary_evaluation_count}")
    print(f"Direct wins:  {direct_winner_count}")
    print(f"Top-1 acc:    {accuracy:.2%}")
    print(f"Results:      {output_dir}")


def main() -> None:
    run_cascade(parse_args())


if __name__ == "__main__":
    main()
