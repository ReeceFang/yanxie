"""Evaluate a merged classifier with a vision-LLM second stage.

The first-stage classifier uses hard Top-1 routing. A direct class is mapped
to the original ImageFolder class ID immediately. A merged class is sent,
together with the original image, to a LangChain ``ChatOpenAI`` model whose
Pydantic schema constrains the response to that branch's configured choices.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .cascade_test import (
        RoutedImageDataset,
        _clear_model_memory,
        make_confusion_matrix,
        predict_indexed_loader,
        save_cascade_metrics_csv,
    )
except ImportError:
    from cascade_test import (
        RoutedImageDataset,
        _clear_model_memory,
        make_confusion_matrix,
        predict_indexed_loader,
        save_cascade_metrics_csv,
    )


class LLMBranchConfig(BaseModel):
    """One merged branch in the user-supplied LLM configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    prompt: str = Field(min_length=1)
    choices: list[str] = Field(alias="list", min_length=1)
    output_map: dict[str, str] = Field(alias="map", min_length=1)

    @model_validator(mode="after")
    def validate_choices_and_map(self) -> "LLMBranchConfig":
        if any(not isinstance(choice, str) or not choice.strip() for choice in self.choices):
            raise ValueError("list 中的每个选项都必须是非空字符串")
        if len(self.choices) != len(set(self.choices)):
            raise ValueError("list 中不能包含重复选项")
        if set(self.choices) != set(self.output_map):
            missing = sorted(set(self.choices) - set(self.output_map))
            extra = sorted(set(self.output_map) - set(self.choices))
            details: list[str] = []
            if missing:
                details.append("map 缺少: " + ", ".join(missing))
            if extra:
                details.append("map 多出: " + ", ".join(extra))
            raise ValueError("list 与 map 的键必须完全一致；" + "；".join(details))
        mapped_classes = list(self.output_map.values())
        if any(
            not isinstance(class_name, str) or not class_name.strip()
            for class_name in mapped_classes
        ):
            raise ValueError("map 的每个值都必须是非空原始类别名")
        if len(mapped_classes) != len(set(mapped_classes)):
            raise ValueError("map 的值不能重复映射到同一个原始类别")
        return self


@dataclass(slots=True)
class LLMClassificationResult:
    sample_index: int
    image_path: str
    branch: str
    choice: str
    mapped_class: str
    cache_key: str
    attempts: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_hit: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a merged classifier with a vision-LLM second stage"
    )
    parser.add_argument(
        "--merged-run-path",
        type=Path,
        required=True,
        help="一级 merged 分类器训练输出目录",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        required=True,
        help="未合并原始数据集的 ImageFolder 验证目录",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        required=True,
        help="二级大模型 prompt/list/map JSON 文件",
    )
    parser.add_argument(
        "--llm-model",
        required=True,
        help="支持图像和结构化输出的模型名称",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible API 地址",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="API key 配置文件（默认：项目根目录/.env）",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="保存 API key 的环境变量名（默认：OPENAI_API_KEY）",
    )
    parser.add_argument(
        "--structured-method",
        choices=("json_schema", "function_calling"),
        default="json_schema",
        help="结构化输出方式；兼容端点不支持 json_schema 时使用 function_calling",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--weights",
        choices=("best", "final"),
        default="best",
    )
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--llm-concurrency", type=int, default=4)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument(
        "--llm-classification-attempts",
        type=int,
        default=2,
        help="结构化解析失败时每张图片的最大尝试次数",
    )
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="默认不发送 temperature；需要时显式设置，例如 0",
    )
    parser.add_argument("--image-max-side", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--image-detail",
        choices=("auto", "low", "high"),
        default="high",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="复用 llm_cache.jsonl 中配置完全一致的成功结果",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认：<merged-run-path>/cascade_llm_results",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.input_size is not None and args.input_size < 1:
        parser.error("--input-size must be at least 1")
    if args.llm_concurrency < 1:
        parser.error("--llm-concurrency must be at least 1")
    if args.llm_max_retries < 0:
        parser.error("--llm-max-retries cannot be negative")
    if args.llm_classification_attempts < 1:
        parser.error("--llm-classification-attempts must be at least 1")
    if args.llm_timeout <= 0:
        parser.error("--llm-timeout must be positive")
    if args.image_max_side < 1:
        parser.error("--image-max-side must be at least 1")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    return args


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 中存在重复键: {key!r}")
        result[key] = value
    return result


def _validate_branch_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("JSON 分支键必须是非空字符串")
    if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"JSON 分支键只能是类别名，不能包含路径: {name!r}")
    return name


def load_llm_config(path: Path) -> dict[str, LLMBranchConfig]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到二级大模型 JSON: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            raw = json.load(file, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 格式错误（第 {exc.lineno} 行，第 {exc.colno} 列）"
        ) from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError("JSON 顶层必须是非空对象")

    config: dict[str, LLMBranchConfig] = {}
    for branch, branch_raw in raw.items():
        branch = _validate_branch_name(branch)
        try:
            config[branch] = LLMBranchConfig.model_validate(branch_raw)
        except Exception as exc:
            raise ValueError(f"分支 {branch!r} 配置错误: {exc}") from exc
    return config


def validate_class_configuration(
    global_classes: list[str],
    merged_classes: list[str],
    llm_config: dict[str, LLMBranchConfig],
) -> dict[str, int]:
    if len(global_classes) != len(set(global_classes)):
        raise ValueError("原始验证集包含重复类别名")
    if len(merged_classes) != len(set(merged_classes)):
        raise ValueError("一级训练日志包含重复类别名")

    global_set = set(global_classes)
    branch_set = set(llm_config)
    member_owner: dict[str, str] = {}
    for branch, config in llm_config.items():
        for class_name in config.output_map.values():
            previous = member_owner.get(class_name)
            if previous is not None:
                raise ValueError(
                    f"原始类别 {class_name!r} 同时出现在 "
                    f"{previous!r} 和 {branch!r} 的 map 中"
                )
            member_owner[class_name] = branch
    member_set = set(member_owner)

    overlap = sorted(branch_set & global_set)
    if overlap:
        raise ValueError(
            "合并类别键不能与原始类别重名: " + ", ".join(overlap)
        )
    missing_members = sorted(member_set - global_set)
    if missing_members:
        raise ValueError(
            "map 中的以下实际类别不在原始验证集中: "
            + ", ".join(missing_members)
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
        raise ValueError("一级类别与原始数据集/LLM JSON 不一致；" + "；".join(details))
    return {name: index for index, name in enumerate(global_classes)}


def create_choice_schema(branch: str, choices: list[str]) -> type[BaseModel]:
    """Create a Pydantic model whose choice field is a JSON Schema enum."""
    safe_name = re.sub(r"\W+", "_", branch).strip("_") or "Branch"
    literal_type = Literal.__getitem__(tuple(choices))
    return create_model(
        f"{safe_name}_Classification",
        __config__=ConfigDict(extra="forbid"),
        choice=(
            literal_type,
            Field(description="必须从给定候选项中选择且只能选择一个"),
        ),
    )


def resolve_llm_choice(
    branch: str,
    choice: str,
    llm_config: dict[str, LLMBranchConfig],
    global_class_to_id: dict[str, int],
) -> tuple[str, int]:
    try:
        mapped_class = llm_config[branch].output_map[choice]
    except KeyError as exc:
        raise ValueError(f"分支 {branch!r} 返回了未知选项: {choice!r}") from exc
    try:
        return mapped_class, global_class_to_id[mapped_class]
    except KeyError as exc:
        raise ValueError(f"map 值无法映射到原始全局 ID: {mapped_class!r}") from exc


def load_trained_model(
    model_name: str,
    num_classes: int,
    checkpoint_path: Path,
) -> Any:
    import timm
    import torch

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"不支持的 checkpoint 格式: {checkpoint_path}")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    return model


def build_eval_transform(model: Any, run_info: Any, input_size: int | None) -> Any:
    from timm.data import create_transform, resolve_model_data_config

    from data import _convert_transform_color_space

    data_config = resolve_model_data_config(model)
    effective_input_size = input_size if input_size is not None else run_info.input_size
    if effective_input_size is not None:
        data_config["input_size"] = (3, effective_input_size, effective_input_size)
    transform = create_transform(**data_config, is_training=False)
    color_space = "rgb"
    if run_info.config_values is not None:
        color_space = str(run_info.config_values.get("color_space", "rgb"))
    return _convert_transform_color_space(transform, color_space)


def encode_image_data_url(
    image_path: str,
    max_side: int,
    jpeg_quality: int,
) -> str:
    from PIL import Image, ImageOps

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_llm_prompt(config: LLMBranchConfig) -> str:
    choices = json.dumps(config.choices, ensure_ascii=False)
    return (
        config.prompt.strip()
        + "\n\n你必须从以下候选项中选择且只能选择一个：\n"
        + choices
        + "\n请通过结构化输出字段 choice 返回选择结果。"
    )


def make_cache_key(
    image_path: str,
    branch: str,
    config: LLMBranchConfig,
    model_name: str,
    base_url: str | None,
    structured_method: str,
    image_max_side: int,
    jpeg_quality: int,
    image_detail: str,
) -> str:
    path = Path(image_path)
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "branch": branch,
        "config": config.model_dump(by_alias=True),
        "model": model_name,
        "base_url": base_url,
        "structured_method": structured_method,
        "image_max_side": image_max_side,
        "jpeg_quality": jpeg_quality,
        "image_detail": image_detail,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_llm_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return cache
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                cache_key = record["cache_key"]
            except (json.JSONDecodeError, KeyError, TypeError):
                print(f"Warning: 忽略缓存第 {line_number} 行的无效记录。")
                continue
            cache[cache_key] = record
    return cache


def result_from_cache(
    record: dict[str, Any],
    sample_index: int,
    image_path: str,
    branch: str,
    config: LLMBranchConfig,
) -> LLMClassificationResult | None:
    try:
        choice = str(record["choice"])
        mapped_class = str(record["mapped_class"])
        if config.output_map[choice] != mapped_class:
            return None
        return LLMClassificationResult(
            sample_index=sample_index,
            image_path=image_path,
            branch=branch,
            choice=choice,
            mapped_class=mapped_class,
            cache_key=str(record["cache_key"]),
            attempts=int(record.get("attempts", 1)),
            input_tokens=int(record.get("input_tokens", 0)),
            output_tokens=int(record.get("output_tokens", 0)),
            total_tokens=int(record.get("total_tokens", 0)),
            cache_hit=True,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _extract_structured_choice(envelope: Any) -> tuple[str, dict[str, int]]:
    if isinstance(envelope, dict) and "parsed" in envelope:
        parsing_error = envelope.get("parsing_error")
        parsed = envelope.get("parsed")
        if parsing_error is not None or parsed is None:
            raise ValueError(f"Pydantic 结构化解析失败: {parsing_error}")
        raw = envelope.get("raw")
    else:
        parsed = envelope
        raw = None

    if isinstance(parsed, BaseModel):
        choice = str(parsed.model_dump()["choice"])
    elif isinstance(parsed, dict):
        choice = str(parsed["choice"])
    else:
        raise TypeError(f"未知的结构化输出类型: {type(parsed).__name__}")
    usage = getattr(raw, "usage_metadata", None) or {}
    return choice, {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def classify_image_with_llm(
    sample_index: int,
    image_path: str,
    branch: str,
    config: LLMBranchConfig,
    structured_llm: Any,
    cache_key: str,
    image_max_side: int,
    jpeg_quality: int,
    image_detail: str,
    max_attempts: int,
) -> LLMClassificationResult:
    data_url = encode_image_data_url(image_path, image_max_side, jpeg_quality)
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": build_llm_prompt(config)},
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": image_detail},
            },
        ],
    }
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            envelope = structured_llm.invoke([message])
            choice, usage = _extract_structured_choice(envelope)
            mapped_class = config.output_map[choice]
            return LLMClassificationResult(
                sample_index=sample_index,
                image_path=image_path,
                branch=branch,
                choice=choice,
                mapped_class=mapped_class,
                cache_key=cache_key,
                attempts=attempt,
                **usage,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"图片 {image_path!r} 在分支 {branch!r} 分类失败，"
        f"已尝试 {max_attempts} 次: {last_error}"
    ) from last_error


def save_predictions_csv(
    path: Path,
    image_paths: list[str],
    y_true: np.ndarray,
    global_classes: list[str],
    stage1_local_ids: np.ndarray,
    stage1_classes: list[str],
    stage1_confidences: np.ndarray,
    llm_results: list[LLMClassificationResult | None],
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
                "llm_branch",
                "llm_choice",
                "llm_mapped_class",
                "llm_cache_hit",
                "llm_attempts",
                "llm_input_tokens",
                "llm_output_tokens",
                "llm_total_tokens",
                "final_global_id",
                "final_class",
                "correct",
            ]
        )
        for index, image_path in enumerate(image_paths):
            true_id = int(y_true[index])
            stage1_id = int(stage1_local_ids[index])
            final_id = int(final_global_ids[index])
            result = llm_results[index]
            writer.writerow(
                [
                    image_path,
                    true_id,
                    global_classes[true_id],
                    stage1_id,
                    stage1_classes[stage1_id],
                    f"{stage1_confidences[index]:.6f}",
                    result.branch if result else "",
                    result.choice if result else "",
                    result.mapped_class if result else "",
                    int(result.cache_hit) if result else "",
                    result.attempts if result else "",
                    result.input_tokens if result else "",
                    result.output_tokens if result else "",
                    result.total_tokens if result else "",
                    final_id,
                    global_classes[final_id],
                    int(final_id == true_id),
                ]
            )


def load_api_key(env_file: Path, key_name: str) -> str:
    try:
        from dotenv import dotenv_values
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 python-dotenv，请先执行: pip install -r requirements.txt"
        ) from exc

    env_file = env_file.expanduser().resolve()
    if not env_file.is_file():
        raise FileNotFoundError(f"找不到 .env 文件: {env_file}")
    value = dotenv_values(env_file).get(key_name)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{env_file} 中没有有效的 {key_name}")
    return value.strip()


def run_cascade(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 langchain-openai，请先执行: pip install -r requirements.txt"
        ) from exc

    from test import locate_run_files, parse_train_log

    merged_run_path = args.merged_run_path.expanduser().resolve()
    val_path = args.val_path.expanduser().resolve()
    if not val_path.is_dir():
        raise FileNotFoundError(f"找不到原始验证集目录: {val_path}")

    print("[1/5] 正在校验一级模型、原始类别和 LLM JSON...", flush=True)
    llm_config = load_llm_config(args.llm_config)
    original_dataset = ImageFolder(val_path)
    if len(original_dataset) == 0:
        raise ValueError(f"原始验证集为空: {val_path}")
    global_classes = original_dataset.classes
    image_paths = [path for path, _ in original_dataset.samples]
    y_true = np.asarray(original_dataset.targets, dtype=np.int64)

    merged_run_dir, merged_log_path, checkpoint_path = locate_run_files(
        merged_run_path, args.weights, None
    )
    run_info = parse_train_log(merged_log_path)
    global_class_to_id = validate_class_configuration(
        global_classes, run_info.classes, llm_config
    )
    print(
        f"类别校验通过：原始 {len(global_classes)} 类，一级 {len(run_info.classes)} 类，"
        f"LLM 二级分支 {len(llm_config)} 个。"
    )

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 CUDA 不可用；请使用 --device cpu")
    pin_memory = device.type == "cuda"
    sample_count = len(image_paths)

    print(f"[2/5] 正在运行一级 merged 分类器（{sample_count} 个样本）...", flush=True)
    merged_model = load_trained_model(
        run_info.model_name, len(run_info.classes), checkpoint_path
    ).to(device)
    transform = build_eval_transform(merged_model, run_info, args.input_size)
    loader = DataLoader(
        RoutedImageDataset(image_paths, list(range(sample_count)), transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    sample_indices, probabilities = predict_indexed_loader(
        merged_model, loader, device, args.amp, "Stage 1", 1.0
    )
    stage1_probabilities = np.full(
        (sample_count, len(run_info.classes)), np.nan, dtype=np.float64
    )
    stage1_probabilities[sample_indices] = probabilities
    if np.any(~np.isfinite(stage1_probabilities)):
        raise RuntimeError("一级推理没有返回全部样本")
    stage1_local_ids = stage1_probabilities.argmax(axis=1).astype(np.int64)
    stage1_confidences = stage1_probabilities[
        np.arange(sample_count), stage1_local_ids
    ]
    del loader, transform, merged_model
    _clear_model_memory(device)

    routes: dict[str, list[int]] = {branch: [] for branch in llm_config}
    final_global_ids = np.full(sample_count, -1, dtype=np.int64)
    for sample_index, local_id in enumerate(stage1_local_ids.tolist()):
        stage1_class = run_info.classes[local_id]
        if stage1_class in llm_config:
            routes[stage1_class].append(sample_index)
        else:
            try:
                final_global_ids[sample_index] = global_class_to_id[stage1_class]
            except KeyError as exc:
                raise ValueError(
                    f"一级普通类别无法映射到全局 ID: {stage1_class!r}"
                ) from exc

    routed_count = sum(len(indices) for indices in routes.values())
    print(
        f"[3/5] 一级硬路由完成：直接输出 {sample_count - routed_count} 个，"
        f"需要 LLM 二级分类 {routed_count} 个。"
    )
    for branch, indices in routes.items():
        print(f"  {branch}: {len(indices)} 个样本")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else merged_run_dir / "cascade_llm_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "llm_cache.jsonl"
    cache = load_llm_cache(cache_path) if args.resume else {}
    llm_results: list[LLMClassificationResult | None] = [None] * sample_count
    pending: list[tuple[int, str, str, str]] = []
    cache_hits = 0
    for branch, indices in routes.items():
        config = llm_config[branch]
        for sample_index in indices:
            image_path = image_paths[sample_index]
            cache_key = make_cache_key(
                image_path,
                branch,
                config,
                args.llm_model,
                args.base_url,
                args.structured_method,
                args.image_max_side,
                args.jpeg_quality,
                args.image_detail,
            )
            cached_result = result_from_cache(
                cache.get(cache_key, {}),
                sample_index,
                image_path,
                branch,
                config,
            )
            if cached_result is not None:
                llm_results[sample_index] = cached_result
                cache_hits += 1
            else:
                pending.append((sample_index, image_path, branch, cache_key))

    print(
        f"[4/5] 正在运行视觉大模型：缓存命中 {cache_hits} 个，"
        f"新增 API 调用 {len(pending)} 个。",
        flush=True,
    )
    if pending:
        api_key = load_api_key(args.env_file, args.api_key_env)
        llm_kwargs: dict[str, Any] = {
            "model": args.llm_model,
            "api_key": api_key,
            "base_url": args.base_url,
            "timeout": args.llm_timeout,
            "max_retries": args.llm_max_retries,
        }
        if args.llm_temperature is not None:
            llm_kwargs["temperature"] = args.llm_temperature
        llm = ChatOpenAI(**llm_kwargs)
        structured_models = {
            branch: llm.with_structured_output(
                create_choice_schema(branch, config.choices),
                method=args.structured_method,
                include_raw=True,
                strict=True,
            )
            for branch, config in llm_config.items()
        }

        failures: list[dict[str, str]] = []

        def persist_result(result: LLMClassificationResult, cache_file: Any) -> None:
            llm_results[result.sample_index] = result
            cache_record = asdict(result)
            cache_record.pop("sample_index", None)
            cache_record.pop("cache_hit", None)
            cache_file.write(json.dumps(cache_record, ensure_ascii=False) + "\n")

        with (
            cache_path.open("a", encoding="utf-8", buffering=1) as cache_file,
            tqdm(total=len(pending), desc="Vision LLM", unit="image") as progress,
        ):
            # Verify credentials, vision input, and the selected structured
            # output method before dispatching the rest of the paid requests.
            sample_index, image_path, branch, cache_key = pending[0]
            try:
                preflight_result = classify_image_with_llm(
                    sample_index,
                    image_path,
                    branch,
                    llm_config[branch],
                    structured_models[branch],
                    cache_key,
                    args.image_max_side,
                    args.jpeg_quality,
                    args.image_detail,
                    args.llm_classification_attempts,
                )
            except Exception as exc:
                method_hint = (
                    " 可尝试 --structured-method function_calling。"
                    if args.structured_method == "json_schema"
                    else ""
                )
                raise RuntimeError(
                    f"LLM 预检失败: {branch}/{image_path}: {exc}.{method_hint}"
                ) from exc
            persist_result(preflight_result, cache_file)
            progress.set_postfix_str(f"preflight: {branch}", refresh=False)
            progress.update(1)

            futures: dict[Future[LLMClassificationResult], tuple[int, str, str]] = {}
            with ThreadPoolExecutor(max_workers=args.llm_concurrency) as executor:
                for sample_index, image_path, branch, cache_key in pending[1:]:
                    future = executor.submit(
                        classify_image_with_llm,
                        sample_index,
                        image_path,
                        branch,
                        llm_config[branch],
                        structured_models[branch],
                        cache_key,
                        args.image_max_side,
                        args.jpeg_quality,
                        args.image_detail,
                        args.llm_classification_attempts,
                    )
                    futures[future] = (sample_index, image_path, branch)

                for future in as_completed(futures):
                    sample_index, image_path, branch = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failures.append(
                            {
                                "image_path": image_path,
                                "branch": branch,
                                "error": str(exc),
                            }
                        )
                        progress.set_postfix_str(f"failed: {branch}", refresh=False)
                        progress.update(1)
                        continue
                    persist_result(result, cache_file)
                    progress.set_postfix_str(branch, refresh=False)
                    progress.update(1)

        if failures:
            error_path = output_dir / "llm_errors.jsonl"
            with error_path.open("w", encoding="utf-8") as error_file:
                for failure in failures:
                    error_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
            raise RuntimeError(
                f"有 {len(failures)} 个 LLM 请求失败；成功结果已缓存，"
                f"错误明细见 {error_path}，修复后重新运行即可断点续跑。"
            )

    for branch, indices in routes.items():
        for sample_index in indices:
            result = llm_results[sample_index]
            if result is None:
                raise RuntimeError(f"样本 {image_paths[sample_index]} 缺少 LLM 结果")
            mapped_class, global_id = resolve_llm_choice(
                branch, result.choice, llm_config, global_class_to_id
            )
            if mapped_class != result.mapped_class:
                raise RuntimeError("LLM 缓存映射与当前配置不一致")
            final_global_ids[sample_index] = global_id
    if np.any(final_global_ids < 0):
        raise RuntimeError("部分样本没有获得最终全局类别 ID")

    print("[5/5] 正在生成指标、混淆矩阵和逐样本明细...", flush=True)
    from test import (
        configure_matplotlib,
        normalize_columns,
        plot_normalized_confusion_matrix,
        save_raw_confusion_csv,
    )

    confusion = make_confusion_matrix(y_true, final_global_ids, len(global_classes))
    save_raw_confusion_csv(
        confusion, global_classes, output_dir / "confusion_matrix_counts.csv"
    )
    configure_matplotlib()
    plot_normalized_confusion_matrix(
        normalize_columns(confusion),
        global_classes,
        output_dir / "confusion_matrix_column_normalized.png",
    )
    accuracy = save_cascade_metrics_csv(
        confusion, global_classes, output_dir / "metrics.csv"
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
        run_info.classes,
        stage1_confidences,
        llm_results,
        final_global_ids,
    )

    called_results = [result for result in llm_results if result is not None]
    new_results = [result for result in called_results if not result.cache_hit]
    print(f"Merged model:  {run_info.model_name}")
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"LLM model:     {args.llm_model}")
    print(f"Samples:       {sample_count}")
    print(f"LLM routed:    {routed_count}")
    print(f"Cache hits:    {cache_hits}")
    print(f"New API calls: {len(new_results)}")
    print(f"Input tokens:  {sum(result.input_tokens for result in new_results)}")
    print(f"Output tokens: {sum(result.output_tokens for result in new_results)}")
    print(f"Top-1 acc:     {accuracy:.2%}")
    print(f"Results:       {output_dir}")


def main() -> None:
    run_cascade(parse_args())


if __name__ == "__main__":
    main()
