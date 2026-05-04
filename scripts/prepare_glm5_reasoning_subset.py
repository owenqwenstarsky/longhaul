#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from teich_tune.dataset import estimate_record_tokens
from teich_tune.importers import convert_glm_reasoning_row, split_records, stable_subset_order, write_explicit_split_dataset


DEFAULT_DATASET = "Jackrong/GLM-5.1-Reasoning-1M-Cleaned"
DEFAULT_CONFIG = "main"
DEFAULT_SPLIT = "train"


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> Dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def collect_plain_chat_records(
    dataset: str,
    config: str,
    split: str,
    count: int,
    page_size: int,
    max_estimated_tokens: int,
) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    offset = 0
    while len(collected) < count:
        payload = fetch_rows(dataset, config, split, offset, page_size)
        rows = payload.get("rows", [])
        if not rows:
            break
        for wrapper in rows:
            row = wrapper["row"]
            try:
                record = convert_glm_reasoning_row(row)
            except ValueError:
                continue
            if estimate_record_tokens(record) > max_estimated_tokens:
                continue
            collected.append(record)
            if len(collected) >= count:
                break
        offset += len(rows)
    if len(collected) < count:
        raise RuntimeError(f"Only collected {len(collected)} usable rows, expected {count}.")
    return collected


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a plain-chat subset from GLM-5.1-Reasoning-1M-Cleaned.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-estimated-tokens", type=int, default=1800)
    parser.add_argument("--seed", default="glm5-plain-100")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    records = collect_plain_chat_records(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        count=args.count,
        page_size=max(args.page_size, args.count),
        max_estimated_tokens=args.max_estimated_tokens,
    )
    ordered = stable_subset_order(records, args.seed)
    if args.count < 20:
        train_count = max(1, args.count - 1)
        valid_count = args.count - train_count
        test_count = 0
    else:
        train_count = int(args.count * 0.9)
        valid_count = int(args.count * 0.05)
        test_count = args.count - train_count - valid_count
    splits = split_records(ordered, train_count=train_count, valid_count=valid_count, test_count=test_count)
    metadata = {
        "source_dataset": args.dataset,
        "source_config": args.config,
        "source_split": args.split,
        "requested_examples": args.count,
        "prepared_examples": len(ordered),
        "seed": args.seed,
        "train_count": train_count,
        "valid_count": valid_count,
        "test_count": test_count,
        "max_estimated_tokens": args.max_estimated_tokens,
        "thinking_removed": True,
    }
    write_explicit_split_dataset(Path(args.output_dir), splits, metadata)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "metadata": metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
