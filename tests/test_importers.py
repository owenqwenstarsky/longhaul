import unittest

from longhaul.importers import (
    convert_glm_reasoning_row,
    split_records,
    stable_subset_order,
    strip_reasoning_blocks,
)


class ImporterTests(unittest.TestCase):
    def test_strip_reasoning_blocks_removes_think_tags(self) -> None:
        text = "<think>\nReasoning here.\n</think>\n\nFinal answer."
        self.assertEqual(strip_reasoning_blocks(text), "Final answer.")

    def test_strip_reasoning_blocks_removes_bracket_tags(self) -> None:
        text = "[thinking]\nReasoning here.\n[/thinking]\n\nFinal answer."
        self.assertEqual(strip_reasoning_blocks(text), "Final answer.")

    def test_convert_glm_reasoning_row_builds_plain_chat_record(self) -> None:
        record = convert_glm_reasoning_row(
            {
                "id": "abc123",
                "input": "How do I uppercase a Python string?",
                "output": "<think>\nUse .upper().\n</think>\n\nUse `text.upper()`.",
            }
        )
        self.assertEqual(record["messages"][0]["role"], "user")
        self.assertEqual(record["messages"][1]["role"], "assistant")
        self.assertEqual(record["messages"][1]["content"], "Use `text.upper()`.")

    def test_stable_subset_order_is_deterministic(self) -> None:
        rows = [
            {"id": "b", "input": "q2", "output": "a2"},
            {"id": "a", "input": "q1", "output": "a1"},
        ]
        ordered_a = stable_subset_order(rows, "seed-1")
        ordered_b = stable_subset_order(rows, "seed-1")
        self.assertEqual([row["id"] for row in ordered_a], [row["id"] for row in ordered_b])

    def test_split_records_checks_total(self) -> None:
        with self.assertRaises(ValueError):
            split_records([{"messages": []}], train_count=0, valid_count=0, test_count=0)


if __name__ == "__main__":
    unittest.main()
