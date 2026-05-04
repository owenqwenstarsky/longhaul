from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from teich_tune.compiler import compile_dataset
from teich_tune.config import DEFAULT_JOB_CONFIG, dump_yaml_compatible_json, load_job_config, slugify
from teich_tune.dataset import load_jsonl, write_json
from teich_tune.gguf import export_job_to_gguf, load_export_manifest, validate_gguf_settings
from teich_tune.registry import ModelSpec, resolve_model
from teich_tune.reports import append_metric, load_metrics, parse_metrics_line, write_report


class JobLockError(RuntimeError):
    pass


PYTHON_BIN = sys.executable or "python3"


def init_workspace(base_dir: str | Path, template: str = "chat") -> List[str]:
    root = Path(base_dir)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tool_catalog_path = data_dir / "tool_catalog.json"
    dataset_path = data_dir / "dataset.jsonl"
    job_path = root / "job.yaml"

    if template == "chat":
        tool_catalog: Dict[str, Any] = {}
        dataset_example = {
            "messages": [
                {
                    "type": "message",
                    "role": "system",
                    "content": "You are a concise, practical assistant.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "Rewrite this politely: send me the file now.",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "Could you please send me the file when you have a moment?",
                },
            ]
        }
    elif template == "tools":
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
    else:
        raise ValueError("template must be one of: chat, tools")

    dump_yaml_compatible_json(DEFAULT_JOB_CONFIG, job_path)
    write_json(tool_catalog_path, tool_catalog)
    dataset_path.write_text(json.dumps(dataset_example) + "\n", encoding="utf-8")
    return [str(job_path), str(tool_catalog_path), str(dataset_path)]


def validate_job_config(config: Dict[str, Any]) -> Dict[str, Any]:
    allow_unsupported = bool(config["safety"].get("allow_unsupported_model"))
    model_spec = resolve_model(config["model"]["id"], allow_unsupported=allow_unsupported)
    thinking_mode = config["training"].get("thinking_mode", "omit")
    split_config = config["data"].get("auto_split", {})
    if thinking_mode not in {"omit", "include"}:
        raise ValueError("training.thinking_mode must be 'omit' or 'include'.")
    if config["training"]["profile"] not in {"conservative", "expert"}:
        raise ValueError("training.profile must be 'conservative' or 'expert'.")
    if float(split_config.get("train_ratio", 0.9)) <= 0:
        raise ValueError("data.auto_split.train_ratio must be greater than 0.")
    if float(split_config.get("valid_ratio", 0.05)) < 0 or float(split_config.get("test_ratio", 0.05)) < 0:
        raise ValueError("data.auto_split valid/test ratios must be non-negative.")
    validate_gguf_settings(config)
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
        split_config=config["data"].get("auto_split"),
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


def select_eval_split(manifest: Dict[str, Any]) -> Optional[str]:
    eval_split = manifest.get("eval_split")
    if eval_split in {"test", "valid"}:
        return eval_split
    return None


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def eta_line(step: int, total_iters: int, elapsed_seconds: float) -> Optional[str]:
    if total_iters <= 0 or step <= 0 or elapsed_seconds <= 0:
        return None
    bounded_step = min(step, total_iters)
    remaining = max(0, total_iters - bounded_step)
    seconds_per_iter = elapsed_seconds / bounded_step
    eta_seconds = seconds_per_iter * remaining
    total_estimate = seconds_per_iter * total_iters
    progress = (bounded_step / total_iters) * 100
    return (
        f"[teich-tune] progress={bounded_step}/{total_iters} ({progress:.1f}%) "
        f"elapsed={format_duration(elapsed_seconds)} "
        f"eta={format_duration(eta_seconds)} "
        f"total_est={format_duration(total_estimate)}"
    )


def _run_streaming_command(
    args: List[str],
    *,
    cwd: Path,
    log_path: Path,
    metrics_path: Path,
    patience: int,
    total_iters: int,
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
        started_at = time.monotonic()
        last_eta_step = 0
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="", flush=True)
            metric = parse_metrics_line(line)
            append_metric(metrics_path, metric)
            step = metric.get("step")
            if "train_loss" in metric and isinstance(step, int) and step > last_eta_step:
                rendered_eta = eta_line(step, total_iters, time.monotonic() - started_at)
                if rendered_eta is not None:
                    log_handle.write(rendered_eta + "\n")
                    log_handle.flush()
                    print(rendered_eta, flush=True)
                last_eta_step = step
            valid_loss = metric.get("valid_loss")
            if valid_loss is None:
                continue
            if best_valid is None or valid_loss < best_valid:
                best_valid = valid_loss
                bad_evals = 0
            else:
                bad_evals += 1
                if patience > 0 and bad_evals >= patience and (not isinstance(step, int) or step < total_iters):
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
        print(f"[teich-tune] job_dir={job_dir}", flush=True)
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
            total_iters=int(mlx_config["iters"]),
        )
        if exit_code != 0:
            raise RuntimeError(f"Training failed with exit code {exit_code}. See {log_path}.")
        run_eval(job_dir)
        if bool(prepared["config"]["outputs"].get("gguf", {}).get("enabled")):
            run_export(job_dir)
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
        print(f"[teich-tune] job_dir={job_path}", flush=True)
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
            total_iters=int(mlx_config["iters"]),
        )
        if exit_code != 0:
            raise RuntimeError(f"Resume failed with exit code {exit_code}. See {log_path}.")
        run_eval(job_path)
        if bool(config["outputs"].get("gguf", {}).get("enabled")):
            run_export(job_path)
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
    eval_split = select_eval_split(manifest)
    if eval_split is None:
        eval_log_path.write_text("Skipped evaluation: no validation or test split was generated.\n", encoding="utf-8")
        sample_outputs = generate_samples(job_path, config)
        render_report(job_path, config, sample_outputs)
        return job_path
    eval_source = job_path / "compiled" / f"{eval_split}.jsonl"
    if not eval_source.exists():
        raise ValueError(f"Missing evaluation source file {eval_source}.")
    with eval_log_path.open("a", encoding="utf-8") as log_handle:
        if eval_split == "valid":
            log_handle.write("Using validation split for post-train evaluation because no test split was generated.\n")
            with tempfile.TemporaryDirectory(prefix="teich-tune-eval-") as tmpdir:
                temp_dir = Path(tmpdir)
                shutil.copyfile(eval_source, temp_dir / "test.jsonl")
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
                        str(temp_dir),
                        "--test",
                    ],
                    cwd=str(Path.cwd()),
                    capture_output=True,
                    text=True,
                )
        else:
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


def existing_sample_outputs(job_dir: str | Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    job_path = Path(job_dir)
    sample_dir = job_path / "samples"
    prompts = list(config["outputs"].get("sample_prompts", []))
    results: List[Dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        output_path = sample_dir / f"sample-{index}.txt"
        if not output_path.exists():
            continue
        results.append({"prompt": prompt, "path": str(output_path.resolve())})
    return results


def render_report(job_dir: str | Path, config: Dict[str, Any], sample_outputs: List[Dict[str, Any]]) -> None:
    job_path = Path(job_dir)
    manifest = json.loads((job_path / "compiled" / "dataset_manifest.json").read_text(encoding="utf-8"))
    warnings = list(config.get("warnings", []))
    warnings.extend(manifest.get("warnings", []))
    eval_split = select_eval_split(manifest)
    if eval_split is None:
        warnings.append("Evaluation was skipped because no validation or test split was generated.")
    elif eval_split == "valid":
        warnings.append("Evaluation used the validation split because no test split was generated.")
    metrics = load_metrics(job_path / "metrics.jsonl")
    export_manifest = load_export_manifest(job_path)
    gguf_outputs = [] if export_manifest is None else list(export_manifest.get("outputs", []))
    write_report(
        path=job_path / "report.md",
        job_name=config["name"],
        model_id=config["model"]["id"],
        manifest=manifest,
        warnings=warnings,
        metrics=metrics,
        sample_outputs=sample_outputs,
        gguf_outputs=gguf_outputs,
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


def run_export(job_dir: str | Path, quants: Optional[List[str]] = None) -> Path:
    job_path = Path(job_dir)
    snapshot_path = job_path / "config.snapshot.yaml"
    if not snapshot_path.exists():
        raise ValueError(f"Missing {snapshot_path}.")
    config = load_job_config(snapshot_path)
    export_job_to_gguf(job_path, requested_quants=quants, python_bin=PYTHON_BIN)
    render_report(job_path, config, existing_sample_outputs(job_path, config))
    return job_path
