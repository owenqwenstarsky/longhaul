from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from teich_tune.dataset import ValidationResult, load_json, load_or_split_dataset, write_json, write_jsonl
from teich_tune.registry import ModelSpec


def compile_dataset(
    *,
    source: str,
    tool_catalog_path: str | None,
    split_seed: str,
    split_config: Dict[str, Any] | None,
    output_dir: str | Path,
    thinking_mode: str,
    model_spec: ModelSpec,
) -> Dict[str, Any]:
    split_records, validation = load_or_split_dataset(source, tool_catalog_path, split_seed, split_config=split_config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tool_catalog = load_json(tool_catalog_path) if tool_catalog_path else {}
    compiled_counts: Dict[str, int] = {}

    for split_name, records in split_records.items():
        split_file = output_path / f"{split_name}.jsonl"
        compiled = [compile_record(record, tool_catalog, thinking_mode, model_spec) for record in records]
        if compiled:
            write_jsonl(split_file, compiled)
        elif split_file.exists():
            split_file.unlink()
        compiled_counts[split_name] = len(compiled)

    manifest = build_manifest(
        validation,
        compiled_counts,
        thinking_mode,
        model_spec.model_id,
        split_mode="explicit" if Path(source).is_dir() else "auto",
    )
    write_json(output_path / "dataset_manifest.json", manifest)
    return manifest


def build_manifest(
    validation: ValidationResult,
    split_counts: Dict[str, int],
    thinking_mode: str,
    model_id: str,
    split_mode: str,
) -> Dict[str, Any]:
    validation.stats.split_counts = split_counts
    eval_split = "test" if split_counts.get("test", 0) > 0 else "valid" if split_counts.get("valid", 0) > 0 else None
    return {
        "model_id": model_id,
        "thinking_mode": thinking_mode,
        "split_mode": split_mode,
        "eval_split": eval_split,
        "records": validation.stats.records,
        "tool_records": validation.stats.tool_records,
        "total_messages": validation.stats.total_messages,
        "max_estimated_tokens": validation.stats.max_estimated_tokens,
        "avg_estimated_tokens": validation.stats.avg_estimated_tokens,
        "split_counts": split_counts,
        "warnings": validation.warnings,
    }


def compile_record(
    record: Dict[str, Any],
    tool_catalog: Dict[str, Dict[str, Any]],
    thinking_mode: str,
    model_spec: ModelSpec,
) -> Dict[str, Any]:
    if thinking_mode not in {"omit", "include"}:
        raise ValueError(f"Unsupported thinking mode: {thinking_mode}")
    tools_used = record.get("tools", [])
    has_tool_events = any(event["type"] != "message" for event in record["messages"])
    messages: List[Dict[str, Any]] = []
    pending_tool_calls: List[Dict[str, Any]] = []

    for event in record["messages"]:
        if event["type"] == "message":
            if pending_tool_calls:
                messages.append({"role": "assistant", "tool_calls": pending_tool_calls})
                pending_tool_calls = []
            compiled_message = {
                "role": event["role"],
                "content": render_message_content(event, thinking_mode, model_spec),
            }
            messages.append(compiled_message)
            continue

        if event["type"] == "tool_call":
            pending_tool_calls.append(
                {
                    "id": event["id"],
                    "type": "function",
                    "function": {
                        "name": event["name"],
                        "arguments": json.dumps(event["arguments"], sort_keys=True),
                    },
                }
            )
            continue

        if pending_tool_calls:
            messages.append({"role": "assistant", "tool_calls": pending_tool_calls})
            pending_tool_calls = []
        messages.append(
            {
                "role": "tool",
                "tool_call_id": event["tool_call_id"],
                "name": event["name"],
                "content": event["content"],
                "is_error": event["is_error"],
            }
        )

    if pending_tool_calls:
        messages.append({"role": "assistant", "tool_calls": pending_tool_calls})

    payload: Dict[str, Any] = {"messages": messages}
    if has_tool_events:
        payload["tools"] = [tool_catalog[name] for name in tools_used]
    return payload


def render_message_content(event: Dict[str, Any], thinking_mode: str, model_spec: ModelSpec) -> str:
    content = event.get("content", "")
    thinking = event.get("thinking")
    if thinking_mode == "omit" or not thinking:
        return content
    if not model_spec.supports_thinking:
        return f"[thinking]\n{thinking}\n[/thinking]\n\n{content}".strip()
    return f"<think>\n{thinking}\n</think>\n\n{content}".strip()
