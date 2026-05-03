from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict


DEFAULT_JOB_CONFIG: Dict[str, Any] = {
    "name": "qwen-local-job",
    "model": {
        "id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    },
    "data": {
        "source": "data/dataset.jsonl",
        "tool_catalog": "data/tool_catalog.json",
        "split_seed": "teich-tune-v1",
    },
    "training": {
        "profile": "conservative",
        "thinking_mode": "omit",
        "epochs": 3,
        "batch_size": 1,
        "effective_batch_size": 8,
        "learning_rate": 1e-5,
        "max_seq_length": 2048,
        "num_layers": 8,
        "grad_checkpoint": True,
        "mask_prompt": "auto",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "steps_per_report": 10,
        "steps_per_eval": 25,
        "val_batches": -1,
        "save_every": 50,
        "early_stopping_patience": 3,
        "min_records_warn": 50,
    },
    "outputs": {
        "jobs_dir": "jobs",
        "sample_prompts": [
            "Summarize the assistant behavior you were trained for.",
            "Respond to a user request in the target style.",
        ],
    },
    "safety": {
        "allow_unsupported_model": False,
        "warn_lr_above": 5e-5,
        "warn_rank_above": 32,
        "warn_small_dataset_below": 50,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_json_or_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(
                f"{path} is not valid JSON. Install PyYAML or keep the config in JSON-compatible YAML."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level mapping/object.")
    return data


def load_job_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    raw = _load_json_or_yaml(config_path)
    merged = deep_merge(DEFAULT_JOB_CONFIG, raw)
    config_dir = config_path.resolve().parent
    merged["data"]["source"] = str((config_dir / merged["data"]["source"]).resolve())
    merged["data"]["tool_catalog"] = str((config_dir / merged["data"]["tool_catalog"]).resolve())
    merged["outputs"]["jobs_dir"] = str((config_dir / merged["outputs"]["jobs_dir"]).resolve())
    merged["_config_path"] = str(config_path.resolve())
    return merged


def dump_yaml_compatible_json(data: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "job"
