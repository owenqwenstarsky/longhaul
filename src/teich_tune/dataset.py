from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROLE_SET = {"system", "user", "assistant"}
EVENT_SET = {"message", "tool_call", "tool_result"}
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class DatasetStats:
    records: int = 0
    tool_records: int = 0
    total_messages: int = 0
    max_estimated_tokens: int = 0
    avg_estimated_tokens: float = 0.0
    split_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ValidationResult:
    records: List[Dict[str, Any]]
    warnings: List[str]
    stats: DatasetStats


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {idx} in {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {idx} in {path} must be an object.")
        records.append(payload)
    return records


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    rendered = "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records)
    Path(path).write_text(rendered, encoding="utf-8")


def estimate_tokens_for_text(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def estimate_record_tokens(record: Dict[str, Any]) -> int:
    total = 0
    for event in record.get("messages", []):
        if event.get("type") == "message":
            total += estimate_tokens_for_text(event.get("content", ""))
            total += estimate_tokens_for_text(event.get("thinking", ""))
        elif event.get("type") == "tool_call":
            total += estimate_tokens_for_text(event.get("name", ""))
            total += estimate_tokens_for_text(json.dumps(event.get("arguments", {}), sort_keys=True))
        elif event.get("type") == "tool_result":
            total += estimate_tokens_for_text(event.get("name", ""))
            total += estimate_tokens_for_text(str(event.get("content", "")))
    return total


def deterministic_tool_call_id(record_index: int, event_index: int, name: str, arguments: Any) -> str:
    payload = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(f"{record_index}:{event_index}:{name}:{payload}".encode("utf-8")).hexdigest()
    return f"call_{digest[:12]}"


def _load_tool_catalog(path: Optional[str | Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    catalog = load_json(path)
    if not isinstance(catalog, dict):
        raise ValueError("Tool catalog must be a JSON object keyed by tool name.")
    result: Dict[str, Dict[str, Any]] = {}
    for name, schema in catalog.items():
        if not isinstance(name, str) or not isinstance(schema, dict):
            raise ValueError("Tool catalog entries must map string names to JSON objects.")
        result[name] = schema
    return result


def validate_dataset_file(dataset_path: str | Path, tool_catalog_path: Optional[str | Path]) -> ValidationResult:
    records = load_jsonl(dataset_path)
    catalog = _load_tool_catalog(tool_catalog_path)
    return validate_records(records, catalog)


def validate_records(records: List[Dict[str, Any]], tool_catalog: Dict[str, Dict[str, Any]]) -> ValidationResult:
    warnings: List[str] = []
    stats = DatasetStats(records=len(records))
    normalized: List[Dict[str, Any]] = []
    total_tokens = 0
    for record_index, raw_record in enumerate(records):
        normalized_record, record_warnings, has_tools = validate_record(
            raw_record,
            record_index=record_index,
            tool_catalog=tool_catalog,
        )
        warnings.extend(record_warnings)
        if has_tools:
            stats.tool_records += 1
        stats.total_messages += len(normalized_record["messages"])
        token_estimate = estimate_record_tokens(normalized_record)
        total_tokens += token_estimate
        stats.max_estimated_tokens = max(stats.max_estimated_tokens, token_estimate)
        normalized.append(normalized_record)
    if records:
        stats.avg_estimated_tokens = total_tokens / len(records)
    return ValidationResult(records=normalized, warnings=warnings, stats=stats)


def validate_record(
    record: Dict[str, Any],
    record_index: int,
    tool_catalog: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str], bool]:
    warnings: List[str] = []
    if not isinstance(record, dict):
        raise ValueError(f"Record {record_index + 1} must be an object.")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Record {record_index + 1} must have a non-empty messages array.")
    declared_tools = record.get("tools", [])
    if declared_tools is None:
        declared_tools = []
    if not isinstance(declared_tools, list):
        raise ValueError(f"Record {record_index + 1} tools must be a list.")
    normalized_tools: List[str] = []
    for tool_name in declared_tools:
        if not isinstance(tool_name, str):
            raise ValueError(f"Record {record_index + 1} tool names must be strings.")
        if tool_name not in tool_catalog:
            raise ValueError(f"Record {record_index + 1} references unknown tool '{tool_name}'.")
        normalized_tools.append(tool_name)

    pending_calls: Dict[str, str] = {}
    used_tools = set(normalized_tools)
    normalized_messages: List[Dict[str, Any]] = []
    has_tools = False

    for event_index, event in enumerate(messages):
        if not isinstance(event, dict):
            raise ValueError(f"Record {record_index + 1} message {event_index + 1} must be an object.")
        event_type = event.get("type")
        if event_type not in EVENT_SET:
            raise ValueError(
                f"Record {record_index + 1} message {event_index + 1} has unsupported type {event_type!r}."
            )
        if event_type == "message":
            role = event.get("role")
            if role not in ROLE_SET:
                raise ValueError(
                    f"Record {record_index + 1} message {event_index + 1} has invalid role {role!r}."
                )
            content = event.get("content", "")
            thinking = event.get("thinking")
            if thinking is not None and role != "assistant":
                raise ValueError(
                    f"Record {record_index + 1} message {event_index + 1} uses thinking outside assistant role."
                )
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise ValueError(
                    f"Record {record_index + 1} message {event_index + 1} content must be a string."
                )
            if thinking is not None and not isinstance(thinking, str):
                raise ValueError(
                    f"Record {record_index + 1} message {event_index + 1} thinking must be a string."
                )
            normalized_messages.append(
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                    **({"thinking": thinking} if thinking is not None else {}),
                }
            )
            continue

        has_tools = True
        tool_name = event.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"Record {record_index + 1} tool event {event_index + 1} needs a name.")
        if tool_name not in tool_catalog:
            raise ValueError(f"Record {record_index + 1} references unknown tool '{tool_name}'.")
        used_tools.add(tool_name)

        if event_type == "tool_call":
            arguments = event.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"Record {record_index + 1} tool_call {event_index + 1} arguments must be an object."
                )
            tool_call_id = event.get("id")
            if tool_call_id is not None and not isinstance(tool_call_id, str):
                raise ValueError(
                    f"Record {record_index + 1} tool_call {event_index + 1} id must be a string."
                )
            if tool_call_id is None:
                tool_call_id = deterministic_tool_call_id(record_index, event_index, tool_name, arguments)
                warnings.append(
                    f"Record {record_index + 1} tool_call {event_index + 1} missing id; generated {tool_call_id}."
                )
            pending_calls[tool_call_id] = tool_name
            normalized_messages.append(
                {
                    "type": "tool_call",
                    "id": tool_call_id,
                    "name": tool_name,
                    "arguments": arguments,
                }
            )
            continue

        tool_call_id = event.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            raise ValueError(
                f"Record {record_index + 1} tool_result {event_index + 1} must include tool_call_id."
            )
        if tool_call_id not in pending_calls:
            raise ValueError(
                f"Record {record_index + 1} tool_result {event_index + 1} references unknown tool_call_id {tool_call_id}."
            )
        if pending_calls[tool_call_id] != tool_name:
            raise ValueError(
                f"Record {record_index + 1} tool_result {event_index + 1} name does not match pending tool_call_id."
            )
        content = event.get("content", "")
        if not isinstance(content, str):
            raise ValueError(
                f"Record {record_index + 1} tool_result {event_index + 1} content must be a string."
            )
        is_error = event.get("is_error", False)
        if not isinstance(is_error, bool):
            raise ValueError(
                f"Record {record_index + 1} tool_result {event_index + 1} is_error must be boolean."
            )
        normalized_messages.append(
            {
                "type": "tool_result",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": content,
                "is_error": is_error,
            }
        )

    if has_tools and not normalized_tools:
        warnings.append(f"Record {record_index + 1} omitted top-level tools; inferred from events.")

    return {
        "messages": normalized_messages,
        "tools": sorted(used_tools),
    }, warnings, has_tools


def assign_split(record: Dict[str, Any], split_seed: str) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(f"{split_seed}:{canonical}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def load_or_split_dataset(
    source: str | Path,
    tool_catalog_path: Optional[str | Path],
    split_seed: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], ValidationResult]:
    source_path = Path(source)
    if source_path.is_dir():
        split_paths = {name: source_path / f"{name}.jsonl" for name in ("train", "valid", "test")}
        missing = [name for name, path in split_paths.items() if name == "train" and not path.exists()]
        if missing:
            raise ValueError(f"Expected at least {split_paths['train']} in explicit split directory.")
        catalog = _load_tool_catalog(tool_catalog_path)
        split_records: Dict[str, List[Dict[str, Any]]] = {}
        combined_records: List[Dict[str, Any]] = []
        combined_warnings: List[str] = []
        stats = DatasetStats()
        for split_name, split_path in split_paths.items():
            if split_path.exists():
                loaded = load_jsonl(split_path)
            else:
                loaded = []
            validation = validate_records(loaded, catalog)
            split_records[split_name] = validation.records
            combined_records.extend(validation.records)
            combined_warnings.extend(validation.warnings)
            stats.records += validation.stats.records
            stats.tool_records += validation.stats.tool_records
            stats.total_messages += validation.stats.total_messages
            stats.max_estimated_tokens = max(stats.max_estimated_tokens, validation.stats.max_estimated_tokens)
        if combined_records:
            total_tokens = sum(estimate_record_tokens(record) for record in combined_records)
            stats.avg_estimated_tokens = total_tokens / len(combined_records)
        validation = ValidationResult(records=combined_records, warnings=combined_warnings, stats=stats)
        return split_records, validation

    validation = validate_dataset_file(source_path, tool_catalog_path)
    split_records = {"train": [], "valid": [], "test": []}
    for record in validation.records:
        split_name = assign_split(record, split_seed=split_seed)
        split_records[split_name].append(record)
    return split_records, validation
