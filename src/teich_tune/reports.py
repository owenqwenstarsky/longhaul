from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


TRAIN_LOSS_RE = re.compile(r"(?:train|training).{0,15}?loss[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
VALID_LOSS_RE = re.compile(r"(?:val|valid|validation).{0,15}?loss[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
ITER_RE = re.compile(r"(?:iter|step|steps?)\D+(\d+)", re.IGNORECASE)


def parse_metrics_line(line: str) -> Dict[str, Any]:
    metric: Dict[str, Any] = {}
    train_match = TRAIN_LOSS_RE.search(line)
    valid_match = VALID_LOSS_RE.search(line)
    iter_match = ITER_RE.search(line)
    if iter_match:
        metric["step"] = int(iter_match.group(1))
    if train_match:
        metric["train_loss"] = float(train_match.group(1))
    if valid_match:
        metric["valid_loss"] = float(valid_match.group(1))
    return metric


def append_metric(path: str | Path, payload: Dict[str, Any]) -> None:
    if not payload:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_metrics(path: str | Path) -> List[Dict[str, Any]]:
    metric_path = Path(path)
    if not metric_path.exists():
        return []
    return [json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def best_valid_loss(metrics: Iterable[Dict[str, Any]]) -> float | None:
    losses = [item["valid_loss"] for item in metrics if "valid_loss" in item]
    return min(losses) if losses else None


def write_report(
    *,
    path: str | Path,
    job_name: str,
    model_id: str,
    manifest: Dict[str, Any],
    warnings: List[str],
    metrics: List[Dict[str, Any]],
    sample_outputs: List[Dict[str, Any]],
) -> None:
    best_valid = best_valid_loss(metrics)
    lines = [
        f"# Report: {job_name}",
        "",
        f"- Model: `{model_id}`",
        f"- Records: {manifest.get('records', 0)}",
        f"- Splits: {manifest.get('split_counts', {})}",
        f"- Thinking mode: `{manifest.get('thinking_mode', 'omit')}`",
        f"- Tool records: {manifest.get('tool_records', 0)}",
        f"- Best validation loss: {best_valid if best_valid is not None else 'n/a'}",
        "",
        "## Warnings",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.extend(["", "## Sample Outputs"])
    if sample_outputs:
        for sample in sample_outputs:
            lines.append(f"- Prompt: {sample['prompt']}")
            lines.append(f"  Output path: `{sample['path']}`")
    else:
        lines.append("- None")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
