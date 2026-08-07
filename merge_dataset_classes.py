"""Copy an ImageFolder dataset while merging selected class directories.

Expected source layout::

    dataset/
        train/<class name>/...
        val/<class name>/...
        test/<class name>/...  # optional

The mapping JSON uses the merged class name as its key and a list of original
class names as its value. Classes absent from the mapping keep their names.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from tqdm import tqdm


SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge class directories in an ImageFolder dataset."
    )
    parser.add_argument("source_dataset", type=Path, help="原数据集路径")
    parser.add_argument("destination_dataset", type=Path, help="新数据集路径")
    parser.add_argument("mapping_json", type=Path, help="类别合并 JSON 路径")
    return parser.parse_args()


def _validate_class_name(name: Any, location: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{location} 必须是非空字符串")
    if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"{location} 只能是类别名，不能包含路径: {name!r}")
    return name


def load_mapping(path: Path) -> dict[str, str]:
    """Return an original-class -> merged-class lookup table."""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON 文件: {path}")

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误（第 {exc.lineno} 行，第 {exc.colno} 列）") from exc

    if not isinstance(raw, dict):
        raise ValueError("JSON 顶层必须是对象：{合并后的类别名: [原类别名, ...]}")

    lookup: dict[str, str] = {}
    for merged_name, original_names in raw.items():
        merged_name = _validate_class_name(merged_name, "合并后的类别名")
        if not isinstance(original_names, list) or not original_names:
            raise ValueError(f"类别 {merged_name!r} 的值必须是非空数组")

        for index, original_name in enumerate(original_names):
            original_name = _validate_class_name(
                original_name, f"类别 {merged_name!r} 的第 {index + 1} 个原类别名"
            )
            previous = lookup.get(original_name)
            if previous is not None:
                raise ValueError(
                    f"原类别 {original_name!r} 同时出现在 {previous!r} 和 "
                    f"{merged_name!r} 中"
                )
            lookup[original_name] = merged_name

    return lookup


def _unique_file_path(wanted: Path, original_class: str) -> Path:
    """Avoid overwriting files whose relative names collide after a merge."""
    if not wanted.exists():
        return wanted

    candidate = wanted.with_name(f"{original_class}__{wanted.name}")
    counter = 2
    while candidate.exists():
        candidate = wanted.with_name(
            f"{original_class}__{wanted.stem}_{counter}{wanted.suffix}"
        )
        counter += 1
    return candidate


def _copy_class(
    source: Path,
    destination: Path,
    original_class: str,
    progress: tqdm,
) -> int:
    copied = 0
    destination.mkdir(parents=True, exist_ok=True)

    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target = _unique_file_path(target, original_class)
            shutil.copy2(item, target)
            copied += 1
            progress.update(1)
    return copied


def merge_dataset(
    source: Path,
    destination: Path,
    mapping_path: Path,
    *,
    show_progress: bool = True,
) -> tuple[int, int]:
    source = source.resolve()
    destination = destination.resolve()
    mapping_path = mapping_path.resolve()

    if show_progress:
        print("[1/4] 正在检查输入路径和类别映射...", flush=True)

    if not source.is_dir():
        raise FileNotFoundError(f"找不到原数据集目录: {source}")
    if destination.exists():
        raise FileExistsError(f"新数据集路径已存在，为避免覆盖已停止: {destination}")
    if source == destination or source in destination.parents:
        raise ValueError("新数据集路径不能等于原数据集路径，也不能位于原数据集内部")

    split_names = [name for name in SPLIT_NAMES if (source / name).is_dir()]
    if "train" not in split_names or "val" not in split_names:
        raise ValueError("原数据集必须至少包含 train 和 val 目录")

    mapping = load_mapping(mapping_path)
    available_classes = {
        class_dir.name
        for split_name in split_names
        for class_dir in (source / split_name).iterdir()
        if class_dir.is_dir()
    }
    missing_classes = sorted(set(mapping) - available_classes)
    if missing_classes:
        raise ValueError("JSON 中的以下原类别在数据集中不存在: " + ", ".join(missing_classes))

    if show_progress:
        print("[2/4] 正在扫描数据文件并计算总数...", flush=True)
    total_files = sum(
        1
        for split_name in split_names
        for class_dir in (source / split_name).iterdir()
        if class_dir.is_dir()
        for item in class_dir.rglob("*")
        if item.is_file()
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    file_count = 0
    final_classes: set[str] = set()

    try:
        if show_progress:
            print("[3/4] 正在复制并合并类别...", flush=True)

        for item in source.iterdir():
            if item.name in split_names:
                continue
            target = staging / item.name
            if item.is_dir():
                shutil.copytree(item, target, copy_function=shutil.copy2)
            elif item.is_file():
                shutil.copy2(item, target)

        with tqdm(
            total=total_files,
            desc="复制数据集",
            unit="个文件",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for split_name in split_names:
                source_split = source / split_name
                destination_split = staging / split_name
                destination_split.mkdir()

                for class_dir in sorted(source_split.iterdir()):
                    if not class_dir.is_dir():
                        shutil.copy2(class_dir, destination_split / class_dir.name)
                        continue
                    final_name = mapping.get(class_dir.name, class_dir.name)
                    final_classes.add(final_name)
                    progress.set_postfix_str(
                        f"当前: {split_name}/{class_dir.name} -> {final_name}",
                        refresh=True,
                    )
                    file_count += _copy_class(
                        class_dir,
                        destination_split / final_name,
                        class_dir.name,
                        progress,
                    )

        if show_progress:
            print("[4/4] 正在完成新数据集...", flush=True)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return file_count, len(final_classes)


def main() -> int:
    args = parse_args()
    try:
        file_count, class_count = merge_dataset(
            args.source_dataset,
            args.destination_dataset,
            args.mapping_json,
        )
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(
        f"完成：已生成 {args.destination_dataset.resolve()}，"
        f"共复制 {file_count} 个文件，最终 {class_count} 个类别。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
