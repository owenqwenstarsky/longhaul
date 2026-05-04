from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from teich_tune.dataset import write_json, write_jsonl


THINK_TAG_RE = re.compile(r"(?is)^\s*<think>.*?</think>\s*")
THINK_BRACKET_RE = re.compile(r"(?is)^\s*\[thinking\].*?\[/thinking\]\s*")


def strip_reasoning_blocks(text: str) -> str:
    cleaned = THINK_TAG_RE.sub("", text)
    cleaned = THINK_BRACKET_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def convert_glm_reasoning_row(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(row.get("input", "")).strip()
    response = strip_reasoning_blocks(str(row.get("output", "")))
    if not prompt:
        raise ValueError("Row is missing a non-empty input field.")
    if not response:
        raise ValueError(f"Row {row.get('id', '<unknown>')} has no assistant content after stripping reasoning.")
    return {
        "messages": [
            {"type": "message", "role": "user", "content": prompt},
            {"type": "message", "role": "assistant", "content": response},
        ]
    }


def stable_subset_order(rows: Iterable[Dict[str, Any]], seed: str) -> List[Dict[str, Any]]:
    ranked: List[tuple[str, Dict[str, Any]]] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(f"{seed}:{row_id}:{payload}".encode("utf-8")).hexdigest()
        ranked.append((digest, row))
    ranked.sort(key=lambda item: item[0])
    return [row for _, row in ranked]


def split_records(records: List[Dict[str, Any]], train_count: int, valid_count: int, test_count: int) -> Dict[str, List[Dict[str, Any]]]:
    total = train_count + valid_count + test_count
    if total != len(records):
        raise ValueError(f"Split counts ({total}) do not match record count ({len(records)}).")
    train_end = train_count
    valid_end = train_end + valid_count
    return {
        "train": records[:train_end],
        "valid": records[train_end:valid_end],
        "test": records[valid_end:valid_end + test_count],
    }


def write_explicit_split_dataset(
    output_dir: str | Path,
    split_records_map: Dict[str, List[Dict[str, Any]]],
    metadata: Dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split_name, records in split_records_map.items():
        write_jsonl(output_path / f"{split_name}.jsonl", records)
    write_json(output_path / "source_metadata.json", metadata)
    write_json(output_path / "tool_catalog.json", {})
