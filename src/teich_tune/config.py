from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict


PROFILE_TRAINING_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "epochs": 3,
        "batch_size": 1,
        "effective_batch_size": 8,
        "learning_rate": 1e-5,
        "max_seq_length": 2048,
        "num_layers": 8,
        "grad_checkpoint": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "steps_per_report": 10,
        "steps_per_eval": 25,
        "val_batches": -1,
        "save_every": 50,
        "early_stopping_patience": 3,
    },
    "expert": {
        "epochs": 4,
        "batch_size": 1,
        "effective_batch_size": 16,
        "learning_rate": 2e-5,
        "max_seq_length": 2048,
        "num_layers": 16,
        "grad_checkpoint": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "steps_per_report": 10,
        "steps_per_eval": 25,
        "val_batches": -1,
        "save_every": 50,
        "early_stopping_patience": 2,
    },
}


DEFAULT_AUTO_SPLIT: Dict[str, Any] = {
    "train_ratio": 0.9,
    "valid_ratio": 0.05,
    "test_ratio": 0.05,
    "min_train_records": 1,
    "min_valid_records": 1,
    "min_test_records": 1,
    "min_records_for_test_split": 10,
}


BASE_JOB_CONFIG: Dict[str, Any] = {
    "name": "qwen-local-job",
    "model": {
        "id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    },
    "data": {
        "source": "data/dataset.jsonl",
        "tool_catalog": "data/tool_catalog.json",
        "split_seed": "teich-tune-v1",
        "auto_split": DEFAULT_AUTO_SPLIT,
    },
    "training": {
        "profile": "conservative",
        "thinking_mode": "omit",
        "mask_prompt": "auto",
        "min_records_warn": 50,
    },
    "outputs": {
        "jobs_dir": "jobs",
        "sample_prompts": [
            "Summarize the assistant behavior you were trained for.",
            "Respond to a user request in the target style.",
        ],
        "gguf": {
            "enabled": False,
            "quants": ["q8", "q4"],
            "base_outtype": "f16",
            "llama_cpp_dir": None,
            "converter_python": None,
            "metadata": {},
        },
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


def build_default_job_config(profile: str = "conservative") -> Dict[str, Any]:
    if profile not in PROFILE_TRAINING_DEFAULTS:
        supported = ", ".join(sorted(PROFILE_TRAINING_DEFAULTS))
        raise ValueError(f"Unsupported training profile {profile!r}. Expected one of: {supported}.")
    config = copy.deepcopy(BASE_JOB_CONFIG)
    config["training"] = deep_merge(config["training"], PROFILE_TRAINING_DEFAULTS[profile])
    config["training"]["profile"] = profile
    return config


DEFAULT_JOB_CONFIG: Dict[str, Any] = build_default_job_config()


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
    requested_profile = raw.get("training", {}).get("profile", DEFAULT_JOB_CONFIG["training"]["profile"])
    merged = deep_merge(build_default_job_config(requested_profile), raw)
    config_dir = config_path.resolve().parent
    merged["data"]["source"] = str((config_dir / merged["data"]["source"]).resolve())
    merged["data"]["tool_catalog"] = str((config_dir / merged["data"]["tool_catalog"]).resolve())
    merged["outputs"]["jobs_dir"] = str((config_dir / merged["outputs"]["jobs_dir"]).resolve())
    gguf_config = merged.get("outputs", {}).get("gguf", {})
    if isinstance(gguf_config, dict) and gguf_config.get("llama_cpp_dir"):
        merged["outputs"]["gguf"]["llama_cpp_dir"] = str((config_dir / gguf_config["llama_cpp_dir"]).resolve())
    if isinstance(gguf_config, dict) and gguf_config.get("converter_python"):
        converter_python = str(gguf_config["converter_python"])
        if "/" in converter_python or converter_python.startswith("."):
            merged["outputs"]["gguf"]["converter_python"] = str((config_dir / converter_python).expanduser())
    merged["_config_path"] = str(config_path.resolve())
    return merged


def dump_yaml_compatible_json(data: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "job"
