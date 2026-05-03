from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from teich_tune.compiler import compile_dataset
from teich_tune.config import DEFAULT_JOB_CONFIG, dump_yaml_compatible_json, load_job_config, slugify
from teich_tune.dataset import load_jsonl, write_json
from teich_tune.registry import ModelSpec, resolve_model
from teich_tune.reports import append_metric, load_metrics, parse_metrics_line, write_report


class JobLockError(RuntimeError):
    pass


PYTHON_BIN = sys.executable or "python3"


def init_workspace(base_dir: str | Path) -> List[str]:
    root = Path(base_dir)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tool_catalog_path = data_dir / "tool_catalog.json"
    dataset_path = data_dir / "dataset.jsonl"
    job_path = root / "job.yaml"

    tool_catalog = {
        "write": {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write a file to the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative file path."},
                        "content": {"type": "string", "description": "File contents to write."},
                    },
                    "required": ["path", "content"],
                },
            },
        }
    }
    dataset_example = {
        "messages": [
            {
                "type": "message",
                "role": "user",
                "content": "Create PLAN.md with a short plan.",
            },
            {
                "type": "tool_call",
                "name": "write",
                "arguments": {
                    "path": "PLAN.md",
                    "content": "# Plan\n- Define requirements\n- Implement MVP\n",
                },
            },
            {
                "type": "tool_result",
                "tool_call_id": "call_example01",
                "name": "write",
                "content": "Wrote PLAN.md",
                "is_error": False,
            },
            {
                "type": "message",
                "role": "assistant",
                "thinking": "I should write the file first, then confirm the result.",
                "content": "The plan has been saved in PLAN.md.",
            },
        ],
        "tools": ["write"],
    }
    dataset_example["messages"][1]["id"] = dataset_example["messages"][2]["tool_call_id"]

    dump_yaml_compatible_json(DEFAULT_JOB_CONFIG, job_path)
    write_json(tool_catalog_path, tool_catalog)
    dataset_path.write_text(json.dumps(dataset_example) + "\n", encoding="utf-8")
    return [str(job_path), str(tool_catalog_path), str(dataset_path)]


def validate_job_config(config: Dict[str, Any]) -> Dict[str, Any]:
    allow_unsupported = bool(config["safety"].get("allow_unsupported_model"))
    model_spec = resolve_model(config["model"]["id"], allow_unsupported=allow_unsupported)
    thinking_mode = config["training"].get("thinking_mode", "omit")
    if thinking_mode not in {"omit", "include"}:
        raise ValueError("training.thinking_mode must be 'omit' or 'include'.")
    if config["training"]["profile"] not in {"conservative", "expert"}:
        raise ValueError("training.profile must be 'conservative' or 'expert'.")
    return {"model_spec": model_spec}


def job_warnings(config: Dict[str, Any], manifest: Dict[str, Any], model_spec: ModelSpec) -> List[str]:
    warnings: List[str] = list(manifest.get("warnings", []))
    training = config["training"]
    safety = config["safety"]
    if manifest.get("records", 0) < safety.get("warn_small_dataset_below", 50):
        warnings.append(
            f"Dataset has only {manifest.get('records', 0)} records; expect higher overfitting risk."
        )
    if float(training["learning_rate"]) > float(safety["warn_lr_above"]):
        warnings.append(
            f"Learning rate {training['learning_rate']} exceeds recommended warning threshold {safety['warn_lr_above']}."
        )
    if int(training["lora_rank"]) > int(safety["warn_rank_above"]):
        warnings.append(
            f"LoRA rank {training['lora_rank']} exceeds recommended warning threshold {safety['warn_rank_above']}."
        )
    if int(training["max_seq_length"]) > int(model_spec.max_seq_length):
        warnings.append(
            f"Configured max_seq_length {training['max_seq_length']} exceeds validated default {model_spec.max_seq_length}."
        )
    if training["profile"] == "expert":
        warnings.append("Expert profile enabled; conservative guardrails are warnings only.")
    return warnings


def build_job_dir(config: Dict[str, Any]) -> Path:
    jobs_dir = Path(config["outputs"]["jobs_dir"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return jobs_dir / f"{stamp}-{slugify(config['name'])}"


def compute_grad_accumulation(config: Dict[str, Any]) -> int:
    batch_size = int(config["training"]["batch_size"])
    effective = int(config["training"]["effective_batch_size"])
    return max(1, math.ceil(effective / max(1, batch_size)))


def compute_iters(train_records: int, config: Dict[str, Any]) -> int:
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    if train_records <= 0:
        raise ValueError("Training split is empty after compilation.")
    return max(1, math.ceil(train_records / max(1, batch_size)) * max(1, epochs))


def mask_prompt_setting(config: Dict[str, Any], manifest: Dict[str, Any]) -> bool:
    explicit = config["training"].get("mask_prompt", "auto")
    if explicit in {True, False}:
        return bool(explicit)
    if manifest.get("tool_records", 0) > 0:
        return False
    if config["training"].get("thinking_mode") == "include":
        return False
    return True


def lora_scale_setting(config: Dict[str, Any]) -> float:
    training = config["training"]
    explicit_scale = training.get("lora_scale")
    if explicit_scale is not None:
        return float(explicit_scale)
    rank = max(1, int(training["lora_rank"]))
    alpha = float(training["lora_alpha"])
    return alpha / rank


def build_mlx_config(config: Dict[str, Any], job_dir: Path, manifest: Dict[str, Any], model_spec: ModelSpec) -> Dict[str, Any]:
    training = config["training"]
    grad_accumulation = compute_grad_accumulation(config)
    compiled_dir = job_dir / "compiled"
    train_count = int(manifest["split_counts"].get("train", 0))
    mlx_config: Dict[str, Any] = {
        "model": config["model"]["id"],
        "train": True,
        "data": str(compiled_dir),
        "fine_tune_type": "lora" if not model_spec.quantized else "lora",
        "batch_size": int(training["batch_size"]),
        "grad_accumulation_steps": grad_accumulation,
        "iters": compute_iters(train_count, config),
        "learning_rate": float(training["learning_rate"]),
        "steps_per_report": int(training["steps_per_report"]),
        "steps_per_eval": int(training["steps_per_eval"]),
        "val_batches": int(training["val_batches"]),
        "save_every": int(training["save_every"]),
        "adapter_path": str(job_dir / "adapters"),
        "max_seq_length": int(training["max_seq_length"]),
        "num_layers": int(training["num_layers"]),
        "grad_checkpoint": bool(training["grad_checkpoint"]),
        "mask_prompt": mask_prompt_setting(config, manifest),
        "lora_parameters": {
            "rank": int(training["lora_rank"]),
            "scale": lora_scale_setting(config),
            "dropout": float(training["lora_dropout"]),
        },
    }
    return mlx_config


def ensure_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(
            f"Required command '{command}' was not found on PATH. Install the training dependencies first."
        )


@contextmanager
def local_job_lock(jobs_dir: str | Path) -> Iterable[Path]:
    lock_path = Path(jobs_dir) / ".teich-tune.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise JobLockError(f"Another teich-tune job is already running: {lock_path}") from exc
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        yield lock_path
    finally:
        if lock_path.exists():
            lock_path.unlink()


def prepare_job(config_path: str | Path, explicit_job_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    config = load_job_config(config_path)
    validated = validate_job_config(config)
    model_spec: ModelSpec = validated["model_spec"]
    job_dir = Path(explicit_job_dir) if explicit_job_dir else build_job_dir(config)
    job_dir.mkdir(parents=True, exist_ok=True)
    compiled_dir = job_dir / "compiled"
    manifest = compile_dataset(
        source=config["data"]["source"],
        tool_catalog_path=config["data"].get("tool_catalog"),
        split_seed=config["data"]["split_seed"],
        output_dir=compiled_dir,
        thinking_mode=config["training"]["thinking_mode"],
        model_spec=model_spec,
    )
    warnings = job_warnings(config, manifest, model_spec)
    config_snapshot = dict(config)
    config_snapshot["resolved_model"] = model_spec.__dict__
    config_snapshot["warnings"] = warnings
    config_snapshot["_job_dir"] = str(job_dir.resolve())
    dump_yaml_compatible_json(config_snapshot, job_dir / "config.snapshot.yaml")
    manifest_path = compiled_dir / "dataset_manifest.json"
    return {
        "config": config,
        "model_spec": model_spec,
        "job_dir": job_dir,
        "compiled_dir": compiled_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "warnings": warnings,
    }


def _run_streaming_command(
    args: List[str],
    *,
    cwd: Path,
    log_path: Path,
    metrics_path: Path,
    patience: int,
) -> int:
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to capture process stdout.")
        best_valid: Optional[float] = None
        bad_evals = 0
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            metric = parse_metrics_line(line)
            append_metric(metrics_path, metric)
            valid_loss = metric.get("valid_loss")
            if valid_loss is None:
                continue
            if best_valid is None or valid_loss < best_valid:
                best_valid = valid_loss
                bad_evals = 0
            else:
                bad_evals += 1
                if patience > 0 and bad_evals >= patience:
                    process.send_signal(signal.SIGTERM)
                    log_handle.write("Early stopping triggered by teich-tune.\n")
                    break
        return process.wait()


def run_train(config_path: str | Path) -> Path:
    config = load_job_config(config_path)
    jobs_dir = config["outputs"]["jobs_dir"]
    with local_job_lock(jobs_dir):
        prepared = prepare_job(config_path)
        job_dir: Path = prepared["job_dir"]
        manifest = prepared["manifest"]
        model_spec: ModelSpec = prepared["model_spec"]
        metrics_path = job_dir / "metrics.jsonl"
        log_path = job_dir / "train.log"
        mlx_config = build_mlx_config(prepared["config"], job_dir, manifest, model_spec)
        dump_yaml_compatible_json(mlx_config, job_dir / "mlx-lora-config.yaml")
        train_args = [
            PYTHON_BIN,
            "-m",
            "mlx_lm",
            "lora",
            "--config",
            str(job_dir / "mlx-lora-config.yaml"),
        ]
        exit_code = _run_streaming_command(
            train_args,
            cwd=Path.cwd(),
            log_path=log_path,
            metrics_path=metrics_path,
            patience=int(prepared["config"]["training"]["early_stopping_patience"]),
        )
        if exit_code != 0:
            raise RuntimeError(f"Training failed with exit code {exit_code}. See {log_path}.")
        run_eval(job_dir)
        return job_dir


def run_resume(job_dir: str | Path) -> Path:
    job_path = Path(job_dir)
    snapshot_path = job_path / "config.snapshot.yaml"
    if not snapshot_path.exists():
        raise ValueError(f"Missing {snapshot_path}.")
    config = load_job_config(snapshot_path)
    with local_job_lock(config["outputs"]["jobs_dir"]):
        manifest = json.loads((job_path / "compiled" / "dataset_manifest.json").read_text(encoding="utf-8"))
        model_spec = resolve_model(
            config["model"]["id"],
            allow_unsupported=bool(config["safety"].get("allow_unsupported_model")),
        )
        mlx_config = build_mlx_config(config, job_path, manifest, model_spec)
        adapter_file = job_path / "adapters" / "adapters.safetensors"
        if adapter_file.exists():
            mlx_config["resume_adapter_file"] = str(adapter_file)
        dump_yaml_compatible_json(mlx_config, job_path / "mlx-lora-config.yaml")
        metrics_path = job_path / "metrics.jsonl"
        log_path = job_path / "train.log"
        exit_code = _run_streaming_command(
            [
                PYTHON_BIN,
                "-m",
                "mlx_lm",
                "lora",
                "--config",
                str(job_path / "mlx-lora-config.yaml"),
            ],
            cwd=Path.cwd(),
            log_path=log_path,
            metrics_path=metrics_path,
            patience=int(config["training"]["early_stopping_patience"]),
        )
        if exit_code != 0:
            raise RuntimeError(f"Resume failed with exit code {exit_code}. See {log_path}.")
        run_eval(job_path)
    return job_path


def run_eval(job_dir: str | Path) -> Path:
    job_path = Path(job_dir)
    snapshot_path = job_path / "config.snapshot.yaml"
    if not snapshot_path.exists():
        raise ValueError(f"Missing {snapshot_path}.")
    config = load_job_config(snapshot_path)
    eval_log_path = job_path / "eval.log"
    manifest = json.loads((job_path / "compiled" / "dataset_manifest.json").read_text(encoding="utf-8"))
    adapter_path = job_path / "adapters"
    if int(manifest.get("split_counts", {}).get("test", 0)) <= 0 or not (job_path / "compiled" / "test.jsonl").exists():
        eval_log_path.write_text("Skipped evaluation: no test split was generated.\n", encoding="utf-8")
        sample_outputs = generate_samples(job_path, config)
        render_report(job_path, config, sample_outputs)
        return job_path
    with eval_log_path.open("a", encoding="utf-8") as log_handle:
        result = subprocess.run(
            [
                PYTHON_BIN,
                "-m",
                "mlx_lm",
                "lora",
                "--model",
                config["model"]["id"],
                "--adapter-path",
                str(adapter_path),
                "--data",
                str(job_path / "compiled"),
                "--test",
            ],
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
        )
        log_handle.write(result.stdout)
        log_handle.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with exit code {result.returncode}. See {eval_log_path}.")
    sample_outputs = generate_samples(job_path, config)
    render_report(job_path, config, sample_outputs)
    return job_path


def generate_samples(job_dir: str | Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    job_path = Path(job_dir)
    sample_dir = job_path / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    prompts = list(config["outputs"].get("sample_prompts", []))
    results: List[Dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        result = subprocess.run(
            [
                PYTHON_BIN,
                "-m",
                "mlx_lm",
                "generate",
                "--model",
                config["model"]["id"],
                "--adapter-path",
                str(job_path / "adapters"),
                "--prompt",
                prompt,
                "--max-tokens",
                "256",
            ],
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
        )
        output_path = sample_dir / f"sample-{index}.txt"
        output_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"Sample generation failed for prompt {index}. See {output_path}.")
        results.append({"prompt": prompt, "path": str(output_path.resolve())})
    return results


def render_report(job_dir: str | Path, config: Dict[str, Any], sample_outputs: List[Dict[str, Any]]) -> None:
    job_path = Path(job_dir)
    manifest = json.loads((job_path / "compiled" / "dataset_manifest.json").read_text(encoding="utf-8"))
    warnings = list(config.get("warnings", []))
    warnings.extend(manifest.get("warnings", []))
    if int(manifest.get("split_counts", {}).get("test", 0)) <= 0:
        warnings.append("Evaluation was skipped because no test split was generated.")
    metrics = load_metrics(job_path / "metrics.jsonl")
    write_report(
        path=job_path / "report.md",
        job_name=config["name"],
        model_id=config["model"]["id"],
        manifest=manifest,
        warnings=warnings,
        metrics=metrics,
        sample_outputs=sample_outputs,
    )


def print_report(job_dir: str | Path) -> str:
    report_path = Path(job_dir) / "report.md"
    if not report_path.exists():
        raise ValueError(f"Missing report at {report_path}.")
    return report_path.read_text(encoding="utf-8")


def compile_only(config_path: str | Path, output_dir: Optional[str | Path] = None) -> Path:
    prepared = prepare_job(config_path, explicit_job_dir=output_dir)
    return prepared["job_dir"]


def validate_only(config_path: str | Path) -> Dict[str, Any]:
    prepared = prepare_job(config_path, explicit_job_dir=Path(".teich-tune-validate"))
    return {
        "job_dir": str(prepared["job_dir"]),
        "manifest": prepared["manifest"],
        "warnings": prepared["warnings"],
    }
