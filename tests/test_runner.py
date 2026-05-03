import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teich_tune.config import dump_yaml_compatible_json, load_job_config
from teich_tune.runner import (
    build_mlx_config,
    compile_only,
    compute_grad_accumulation,
    job_warnings,
    lora_scale_setting,
    run_eval,
    run_resume,
)


def example_config(root: Path) -> dict:
    return {
        "name": "test-job",
        "model": {"id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit"},
        "data": {
            "source": str(root / "data" / "dataset.jsonl"),
            "tool_catalog": str(root / "data" / "tool_catalog.json"),
            "split_seed": "seed-1",
        },
        "training": {
            "profile": "conservative",
            "thinking_mode": "omit",
            "epochs": 2,
            "batch_size": 1,
            "effective_batch_size": 8,
            "learning_rate": 1e-5,
            "max_seq_length": 2048,
            "num_layers": 8,
            "grad_checkpoint": True,
            "mask_prompt": "auto",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "steps_per_report": 10,
            "steps_per_eval": 25,
            "val_batches": -1,
            "save_every": 50,
            "early_stopping_patience": 2,
            "min_records_warn": 50,
        },
        "outputs": {
            "jobs_dir": str(root / "jobs"),
            "sample_prompts": ["hello"],
        },
        "safety": {
            "allow_unsupported_model": False,
            "warn_lr_above": 5e-5,
            "warn_rank_above": 32,
            "warn_small_dataset_below": 50,
        },
    }


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        data_dir = self.root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
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
                    "id": "call_123",
                    "name": "write",
                    "arguments": {"path": "PLAN.md", "content": "hello"},
                },
                {
                    "type": "tool_result",
                    "tool_call_id": "call_123",
                    "name": "write",
                    "content": "Wrote PLAN.md",
                    "is_error": False,
                },
                {"type": "message", "role": "assistant", "content": "Done."},
            ],
            "tools": ["write"],
        }
        (data_dir / "tool_catalog.json").write_text(json.dumps(self.tool_catalog), encoding="utf-8")
        (data_dir / "dataset.jsonl").write_text(json.dumps(self.record) + "\n", encoding="utf-8")
        self.config_path = self.root / "job.yaml"
        dump_yaml_compatible_json(example_config(self.root), self.config_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_compile_only_writes_manifest(self) -> None:
        job_dir = compile_only(self.config_path)
        manifest_path = Path(job_dir) / "compiled" / "dataset_manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["records"], 1)
        self.assertTrue((Path(job_dir) / "compiled" / "train.jsonl").exists())
        self.assertFalse((Path(job_dir) / "compiled" / "valid.jsonl").exists())
        self.assertFalse((Path(job_dir) / "compiled" / "test.jsonl").exists())

    def test_load_job_config_resolves_paths_relative_to_config(self) -> None:
        old_cwd = Path.cwd()
        try:
            os.chdir("/")
            loaded = load_job_config(self.config_path)
        finally:
            os.chdir(old_cwd)
        self.assertEqual(loaded["data"]["source"], str((self.root / "data" / "dataset.jsonl").resolve()))
        self.assertEqual(
            loaded["data"]["tool_catalog"], str((self.root / "data" / "tool_catalog.json").resolve())
        )

    def test_build_mlx_config_sets_grad_accumulation(self) -> None:
        job_dir = compile_only(self.config_path)
        manifest = json.loads((Path(job_dir) / "compiled" / "dataset_manifest.json").read_text(encoding="utf-8"))
        from teich_tune.registry import resolve_model

        config = load_job_config(self.config_path)
        model_spec = resolve_model(config["model"]["id"])
        mlx_config = build_mlx_config(config, Path(job_dir), manifest, model_spec)
        self.assertEqual(mlx_config["grad_accumulation_steps"], 8)
        self.assertEqual(mlx_config["lora_parameters"]["scale"], 2.0)
        self.assertNotIn("alpha", mlx_config["lora_parameters"])

    def test_lora_scale_defaults_to_alpha_over_rank(self) -> None:
        config = load_job_config(self.config_path)
        self.assertEqual(lora_scale_setting(config), 2.0)

    def test_lora_scale_prefers_explicit_override(self) -> None:
        config = load_job_config(self.config_path)
        config["training"]["lora_scale"] = 6.5
        self.assertEqual(lora_scale_setting(config), 6.5)

    def test_job_warnings_flag_small_datasets(self) -> None:
        from teich_tune.config import load_job_config
        from teich_tune.registry import resolve_model

        config = load_job_config(self.config_path)
        model_spec = resolve_model(config["model"]["id"])
        warnings = job_warnings(config, {"records": 1, "warnings": [], "tool_records": 0}, model_spec)
        self.assertTrue(any("overfitting risk" in item for item in warnings))

    @patch("teich_tune.runner.run_eval")
    @patch("teich_tune.runner._run_streaming_command", return_value=0)
    def test_resume_uses_existing_adapter_when_present(self, _run_streaming, _run_eval) -> None:
        job_dir = compile_only(self.config_path)
        job_path = Path(job_dir)
        adapter_dir = job_path / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapters.safetensors").write_text("stub", encoding="utf-8")
        resumed = run_resume(job_dir)
        mlx_config = json.loads((job_path / "mlx-lora-config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(str(resumed), str(job_path))
        self.assertEqual(mlx_config["resume_adapter_file"], str(adapter_dir / "adapters.safetensors"))

    @patch("teich_tune.runner.render_report")
    @patch("teich_tune.runner.generate_samples", return_value=[])
    def test_eval_skips_when_no_test_split(self, _generate_samples, _render_report) -> None:
        job_dir = compile_only(self.config_path)
        run_eval(job_dir)
        eval_log = (Path(job_dir) / "eval.log").read_text(encoding="utf-8")
        self.assertIn("Skipped evaluation", eval_log)


if __name__ == "__main__":
    unittest.main()
