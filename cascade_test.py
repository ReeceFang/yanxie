"""Evaluate a hard-routed two-stage image-classification cascade.

The first-stage model is trained on the merged dataset. Predictions whose
class name is a key in the merge JSON are routed to ``<second-runs>/<key>``;
all other predictions are final. Every prediction is mapped back to the class
IDs of the original, unmerged ImageFolder validation dataset before metrics
are calculated.
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
        description="Evaluate a hard-routed merged + second-stage classifier cascade"
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
        "--weights",
        choices=("best", "final"),
        default="best",
        help="所有分类器统一使用 best_model.pth 或 last_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果目录；默认为 <merged-run-path>/cascade_test_results",
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    sample_index_batches: list[np.ndarray] = []
    local_id_batches: list[np.ndarray] = []
    confidence_batches: list[np.ndarray] = []
    use_amp = amp and device.type == "cuda"

    with torch.inference_mode():
        for images, sample_indices in tqdm(loader, desc=description, unit="batch"):
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            local_ids = logits.argmax(dim=1)
            probabilities = logits.float().softmax(dim=1)
            confidences = probabilities.gather(1, local_ids[:, None]).squeeze(1)

            sample_index_batches.append(sample_indices.numpy())
            local_id_batches.append(local_ids.cpu().numpy())
            confidence_batches.append(confidences.cpu().numpy())

    return (
        np.concatenate(sample_index_batches).astype(np.int64, copy=False),
        np.concatenate(local_id_batches).astype(np.int64, copy=False),
        np.concatenate(confidence_batches).astype(np.float64, copy=False),
    )


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
    stage1_local_ids: np.ndarray,
    merged_classes: list[str],
    stage1_confidences: np.ndarray,
    second_branches: list[str],
    stage2_local_ids: np.ndarray,
    stage2_class_names: list[str],
    stage2_confidences: np.ndarray,
    final_global_ids: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_path",
                "true_global_id",
                "true_class",
                "stage1_local_id",
                "stage1_class",
                "stage1_confidence",
                "stage2_run",
                "stage2_local_id",
                "stage2_class",
                "stage2_confidence",
                "final_global_id",
                "final_class",
                "correct",
            ]
        )
        for index, image_path in enumerate(image_paths):
            true_id = int(y_true[index])
            stage1_id = int(stage1_local_ids[index])
            stage2_id = int(stage2_local_ids[index])
            final_id = int(final_global_ids[index])
            routed = stage2_id >= 0
            writer.writerow(
                [
                    image_path,
                    true_id,
                    global_classes[true_id],
                    stage1_id,
                    merged_classes[stage1_id],
                    f"{stage1_confidences[index]:.6f}",
                    second_branches[index] if routed else "",
                    stage2_id if routed else "",
                    stage2_class_names[index] if routed else "",
                    f"{stage2_confidences[index]:.6f}" if routed else "",
                    final_id,
                    global_classes[final_id],
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
    stage1_indices, stage1_ids_ordered, stage1_conf_ordered = predict_indexed_loader(
        merged_model,
        merged_loader,
        device,
        args.amp,
        "Stage 1",
    )
    stage1_local_ids = np.full(sample_count, -1, dtype=np.int64)
    stage1_confidences = np.full(sample_count, np.nan, dtype=np.float64)
    stage1_local_ids[stage1_indices] = stage1_ids_ordered
    stage1_confidences[stage1_indices] = stage1_conf_ordered
    if np.any(stage1_local_ids < 0):
        raise RuntimeError("一级推理没有返回全部样本")
    del merged_loader, merged_transform
    del merged_model
    _clear_model_memory(device)

    routes: dict[str, list[int]] = {branch: [] for branch in mapping}
    final_global_ids = np.full(sample_count, -1, dtype=np.int64)
    second_branches = [""] * sample_count
    stage2_local_ids = np.full(sample_count, -1, dtype=np.int64)
    stage2_class_names = [""] * sample_count
    stage2_confidences = np.full(sample_count, np.nan, dtype=np.float64)

    for sample_index, local_id in enumerate(stage1_local_ids.tolist()):
        branch, final_id = resolve_stage1_prediction(
            local_id,
            merged_spec.info.classes,
            mapping,
            global_class_to_id,
        )
        if branch is None:
            if final_id is None:
                raise RuntimeError("一级直接输出未生成全局 ID")
            final_global_ids[sample_index] = final_id
        else:
            routes[branch].append(sample_index)
            second_branches[sample_index] = branch

    direct_count = int((final_global_ids >= 0).sum())
    print(f"[3/5] 一级路由完成：直接输出 {direct_count} 个样本。")
    for branch, indices in routes.items():
        print(f"  {branch}: {len(indices)} 个样本进入二级分类器")

    print("[4/5] 正在逐个运行二级分类器...", flush=True)
    for branch, routed_indices in routes.items():
        if not routed_indices:
            print(f"跳过二级分支 {branch}：没有路由样本。")
            continue

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
        sample_indices, local_ids, confidences = predict_indexed_loader(
            second_model,
            second_loader,
            device,
            args.amp,
            f"Stage 2 [{branch}]",
        )
        for sample_index, local_id, confidence in zip(
            sample_indices.tolist(),
            local_ids.tolist(),
            confidences.tolist(),
            strict=True,
        ):
            class_name, global_id = resolve_secondary_prediction(
                branch,
                local_id,
                second_classes,
                global_class_to_id,
            )
            stage2_local_ids[sample_index] = local_id
            stage2_class_names[sample_index] = class_name
            stage2_confidences[sample_index] = confidence
            final_global_ids[sample_index] = global_id

        del second_loader, second_transform
        del second_model
        _clear_model_memory(device)

    if np.any(final_global_ids < 0):
        missing_count = int((final_global_ids < 0).sum())
        raise RuntimeError(f"有 {missing_count} 个样本没有获得最终全局类别 ID")
    if not np.all((0 <= final_global_ids) & (final_global_ids < len(global_classes))):
        raise RuntimeError("最终预测中存在越界的全局类别 ID")

    print("[5/5] 正在生成最终指标、混淆矩阵和逐样本明细...", flush=True)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else merged_spec.run_dir / "cascade_test_results"
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
        stage1_local_ids,
        merged_spec.info.classes,
        stage1_confidences,
        second_branches,
        stage2_local_ids,
        stage2_class_names,
        stage2_confidences,
        final_global_ids,
    )

    print(f"Merged model: {merged_spec.info.model_name}")
    print(f"Checkpoint:   {merged_spec.checkpoint_path}")
    print(f"Samples:      {sample_count}")
    print(f"Direct:       {direct_count}")
    print(f"Routed:       {sample_count - direct_count}")
    print(f"Top-1 acc:    {accuracy:.2%}")
    print(f"Results:      {output_dir}")


def main() -> None:
    run_cascade(parse_args())


if __name__ == "__main__":
    main()
