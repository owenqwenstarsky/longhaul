import json
import tempfile
import unittest
from pathlib import Path

from teich_tune.compiler import compile_record
from teich_tune.dataset import (
    deterministic_tool_call_id,
    load_or_split_dataset,
    plan_auto_split,
    validate_records,
)
from teich_tune.registry import resolve_model


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool_catalog = {
            "write": {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "Write file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        }
        self.record = {
            "messages": [
                {"type": "message", "role": "user", "content": "Create PLAN.md"},
                {
                    "type": "tool_call",
                    "name": "write",
                    "arguments": {"path": "PLAN.md", "content": "hello"},
                },
                {
                    "type": "tool_result",
                    "tool_call_id": deterministic_tool_call_id(
                        0, 1, "write", {"path": "PLAN.md", "content": "hello"}
                    ),
                    "name": "write",
                    "content": "Wrote PLAN.md",
                    "is_error": False,
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "thinking": "I should confirm the file write.",
                    "content": "The file is saved.",
                },
            ],
            "tools": ["write"],
        }

    def test_validate_generates_missing_tool_call_id(self) -> None:
        record = {
            "messages": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "tool_call", "name": "write", "arguments": {"path": "a", "content": "b"}},
            ],
            "tools": ["write"],
        }
        result = validate_records([record], self.tool_catalog)
        self.assertEqual(result.stats.records, 1)
        self.assertTrue(result.warnings)
        generated_id = result.records[0]["messages"][1]["id"]
        self.assertTrue(generated_id.startswith("call_"))

    def test_validate_rejects_unknown_tool_result_reference(self) -> None:
        broken = {
            "messages": [
                {"type": "message", "role": "user", "content": "hi"},
                {
                    "type": "tool_result",
                    "tool_call_id": "missing",
                    "name": "write",
                    "content": "nope",
                    "is_error": False,
                },
            ],
            "tools": ["write"],
        }
        with self.assertRaises(ValueError):
            validate_records([broken], self.tool_catalog)

    def test_compile_tool_record_to_mlx_tools_format(self) -> None:
        validated = validate_records([self.record], self.tool_catalog)
        compiled = compile_record(
            validated.records[0],
            self.tool_catalog,
            thinking_mode="omit",
            model_spec=resolve_model("mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
        )
        self.assertIn("tools", compiled)
        self.assertEqual(compiled["messages"][1]["role"], "assistant")
        self.assertIn("tool_calls", compiled["messages"][1])
        self.assertEqual(compiled["messages"][2]["role"], "tool")

    def test_compile_includes_thinking_for_qwen3(self) -> None:
        validated = validate_records([self.record], self.tool_catalog)
        compiled = compile_record(
            validated.records[0],
            self.tool_catalog,
            thinking_mode="include",
            model_spec=resolve_model("mlx-community/Qwen3-1.7B-4bit"),
        )
        assistant_messages = [item for item in compiled["messages"] if item["role"] == "assistant" and "content" in item]
        self.assertIn("<think>", assistant_messages[-1]["content"])

    def test_deterministic_split_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.jsonl"
            tool_path = Path(tmpdir) / "tools.json"
            dataset_path.write_text(json.dumps(self.record) + "\n", encoding="utf-8")
            tool_path.write_text(json.dumps(self.tool_catalog), encoding="utf-8")
            splits_a, _ = load_or_split_dataset(dataset_path, tool_path, "seed-1")
            splits_b, _ = load_or_split_dataset(dataset_path, tool_path, "seed-1")
            self.assertEqual(
                {name: len(records) for name, records in splits_a.items()},
                {name: len(records) for name, records in splits_b.items()},
            )

    def test_small_auto_split_uses_validation_only(self) -> None:
        counts, warnings = plan_auto_split(2, {"min_records_for_test_split": 10})
        self.assertEqual(counts, {"train": 1, "valid": 1, "test": 0})
        self.assertTrue(any("validation only" in warning for warning in warnings))

    def test_auto_split_uses_test_once_dataset_is_large_enough(self) -> None:
        counts, warnings = plan_auto_split(10, {"min_records_for_test_split": 10})
        self.assertEqual(counts, {"train": 8, "valid": 1, "test": 1})
        self.assertFalse(warnings)


if __name__ == "__main__":
    unittest.main()
