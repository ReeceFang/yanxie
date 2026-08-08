"""Create a smaller ImageFolder dataset containing selected classes only.

Expected source layout::

    dataset/
        train/<class name>/...
        val/<class name>/...
        test/<class name>/...  # optional

The class TXT file must contain one class name per line. Blank lines and
duplicate names are ignored.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm


SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected classes into a new ImageFolder dataset."
    )
    parser.add_argument("source_dataset", type=Path, help="原数据集路径")
    parser.add_argument("destination_dataset", type=Path, help="新数据集路径")
    parser.add_argument("class_txt", type=Path, help="类别名称 TXT 路径")
    return parser.parse_args()


def _validate_class_name(name: str, line_number: int) -> None:
    if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(
            f"TXT 第 {line_number} 行只能填写类别名，不能包含路径: {name!r}"
        )


def load_class_names(path: Path) -> list[str]:
    """Load unique, non-empty class names while preserving TXT order."""
    if not path.is_file():
        raise FileNotFoundError(f"找不到类别 TXT 文件: {path}")

    class_names: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            name = line.strip()
            if not name:
                continue
            _validate_class_name(name, line_number)
            if name not in seen:
                seen.add(name)
                class_names.append(name)

    if not class_names:
        raise ValueError("类别 TXT 为空，没有可抽取的类别")
    return class_names


def _copy_class(source: Path, destination: Path, progress: tqdm) -> int:
    copied = 0
    destination.mkdir(parents=True)

    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied += 1
            progress.update(1)
    return copied


def extract_dataset(
    source: Path,
    destination: Path,
    class_txt: Path,
    *,
    show_progress: bool = True,
) -> tuple[int, int]:
    source = source.resolve()
    destination = destination.resolve()
    class_txt = class_txt.resolve()

    if show_progress:
        print("[1/4] 正在检查输入路径和类别列表...", flush=True)

    if not source.is_dir():
        raise FileNotFoundError(f"找不到原数据集目录: {source}")
    if destination.exists():
        raise FileExistsError(f"新数据集路径已存在，为避免覆盖已停止: {destination}")
    if source == destination or source in destination.parents:
        raise ValueError("新数据集路径不能等于原数据集路径，也不能位于原数据集内部")

    split_names = [name for name in SPLIT_NAMES if (source / name).is_dir()]
    if "train" not in split_names or "val" not in split_names:
        raise ValueError("原数据集必须至少包含 train 和 val 目录")

    class_names = load_class_names(class_txt)
    missing_by_split: list[str] = []
    for split_name in split_names:
        missing = [
            name for name in class_names if not (source / split_name / name).is_dir()
        ]
        if missing:
            missing_by_split.append(f"{split_name}: {', '.join(missing)}")
    if missing_by_split:
        raise ValueError("以下类别目录不存在：" + "；".join(missing_by_split))

    if show_progress:
        print("[2/4] 正在扫描选中类别并计算文件总数...", flush=True)
    total_files = sum(
        1
        for split_name in split_names
        for class_name in class_names
        for item in (source / split_name / class_name).rglob("*")
        if item.is_file()
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    file_count = 0

    try:
        if show_progress:
            print("[3/4] 正在复制选中的类别...", flush=True)

        with tqdm(
            total=total_files,
            desc="抽取子数据集",
            unit="个文件",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for split_name in split_names:
                destination_split = staging / split_name
                destination_split.mkdir()

                for class_name in class_names:
                    progress.set_postfix_str(
                        f"当前: {split_name}/{class_name}", refresh=True
                    )
                    file_count += _copy_class(
                        source / split_name / class_name,
                        destination_split / class_name,
                        progress,
                    )

        if show_progress:
            print("[4/4] 正在完成新数据集...", flush=True)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return file_count, len(class_names)


def main() -> int:
    args = parse_args()
    try:
        file_count, class_count = extract_dataset(
            args.source_dataset,
            args.destination_dataset,
            args.class_txt,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(
        f"完成：已生成 {args.destination_dataset.resolve()}，"
        f"共抽取 {class_count} 个类别、复制 {file_count} 个文件。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
