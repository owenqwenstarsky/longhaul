import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from teich_tune.config import dump_yaml_compatible_json, load_job_config
from teich_tune.gguf import resolve_gguf_targets
from teich_tune.runner import (
    build_mlx_config,
    compile_only,
    compute_grad_accumulation,
    eta_line,
    format_duration,
    init_workspace,
    job_warnings,
    lora_scale_setting,
    _run_streaming_command,
    run_eval,
    run_export,
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
        self.assertIsNone(manifest["eval_split"])
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

    def test_load_job_config_applies_expert_profile_defaults(self) -> None:
        expert_config_path = self.root / "expert-job.yaml"
        dump_yaml_compatible_json(
            {
                "name": "expert-job",
                "training": {"profile": "expert"},
                "data": {
                    "source": "data/dataset.jsonl",
                    "tool_catalog": "data/tool_catalog.json",
                },
                "outputs": {"jobs_dir": "jobs"},
            },
            expert_config_path,
        )
        loaded = load_job_config(expert_config_path)
        self.assertEqual(loaded["training"]["learning_rate"], 2e-5)
        self.assertEqual(loaded["training"]["lora_rank"], 16)
        self.assertEqual(loaded["training"]["num_layers"], 16)

    def test_load_job_config_resolves_llama_cpp_dir_relative_to_config(self) -> None:
        llama_cpp_dir = self.root / "vendor" / "llama.cpp"
        llama_cpp_dir.mkdir(parents=True, exist_ok=True)
        converter_python = self.root / "tools" / "gguf-python" / "bin" / "python"
        converter_python.parent.mkdir(parents=True, exist_ok=True)
        converter_python.write_text("", encoding="utf-8")
        config_path = self.root / "gguf-job.yaml"
        dump_yaml_compatible_json(
            {
                "name": "gguf-job",
                "data": {
                    "source": "data/dataset.jsonl",
                    "tool_catalog": "data/tool_catalog.json",
                },
                "outputs": {
                    "jobs_dir": "jobs",
                    "gguf": {
                        "llama_cpp_dir": "vendor/llama.cpp",
                        "converter_python": "tools/gguf-python/bin/python",
                    },
                },
            },
            config_path,
        )
        loaded = load_job_config(config_path)
        self.assertEqual(loaded["outputs"]["gguf"]["llama_cpp_dir"], str(llama_cpp_dir.resolve()))
        self.assertEqual(loaded["outputs"]["gguf"]["converter_python"], str(converter_python.resolve()))

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

    def test_resolve_gguf_targets_maps_aliases_and_dedupes(self) -> None:
        self.assertEqual(
            resolve_gguf_targets(["Q8", "q4", "q8_0", "Q4-K-M", "bf16"]),
            ["q8_0", "q4_k_m", "bf16"],
        )

    def test_format_duration_renders_human_readable_eta(self) -> None:
        self.assertEqual(format_duration(7), "7s")
        self.assertEqual(format_duration(61), "1m 01s")
        self.assertEqual(format_duration(3661), "1h 01m 01s")

    def test_eta_line_includes_progress_and_total_estimate(self) -> None:
        rendered = eta_line(step=40, total_iters=100, elapsed_seconds=80.0)
        self.assertIsNotNone(rendered)
        assert rendered is not None
        self.assertIn("progress=40/100 (40.0%)", rendered)
        self.assertIn("elapsed=1m 20s", rendered)
        self.assertIn("eta=2m 00s", rendered)
        self.assertIn("total_est=3m 20s", rendered)

    def test_streaming_command_stops_early_before_final_iteration(self) -> None:
        log_path = self.root / "train.log"
        metrics_path = self.root / "metrics.jsonl"

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = iter(
                    [
                        "Iter 10: Val loss 1.000, Val took 1.000s\n",
                        "Iter 20: Val loss 1.000, Val took 1.000s\n",
                    ]
                )
                self.signals: list[int] = []

            def send_signal(self, sig: int) -> None:
                self.signals.append(sig)

            def wait(self) -> int:
                return -15 if self.signals else 0

        fake_process = FakeProcess()
        with patch("teich_tune.runner.subprocess.Popen", return_value=fake_process):
            with patch("builtins.print"):
                exit_code = _run_streaming_command(
                    ["python3", "-m", "mlx_lm", "lora"],
                    cwd=self.root,
                    log_path=log_path,
                    metrics_path=metrics_path,
                    patience=1,
                    total_iters=30,
                )

        self.assertEqual(exit_code, -15)
        self.assertEqual(fake_process.signals, [signal.SIGTERM])
        self.assertIn("Early stopping triggered by teich-tune.", log_path.read_text(encoding="utf-8"))

    def test_streaming_command_does_not_stop_on_final_validation_iteration(self) -> None:
        log_path = self.root / "train.log"
        metrics_path = self.root / "metrics.jsonl"

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = iter(
                    [
                        "Iter 10: Val loss 1.000, Val took 1.000s\n",
                        "Iter 20: Val loss 1.000, Val took 1.000s\n",
                    ]
                )
                self.signals: list[int] = []

            def send_signal(self, sig: int) -> None:
                self.signals.append(sig)

            def wait(self) -> int:
                return -15 if self.signals else 0

        fake_process = FakeProcess()
        with patch("teich_tune.runner.subprocess.Popen", return_value=fake_process):
            with patch("builtins.print"):
                exit_code = _run_streaming_command(
                    ["python3", "-m", "mlx_lm", "lora"],
                    cwd=self.root,
                    log_path=log_path,
                    metrics_path=metrics_path,
                    patience=1,
                    total_iters=20,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(fake_process.signals, [])
        self.assertNotIn("Early stopping triggered by teich-tune.", log_path.read_text(encoding="utf-8"))

    def test_init_workspace_chat_template_is_plain_chat(self) -> None:
        workspace = self.root / "starter"
        created = init_workspace(workspace, template="chat")
        self.assertEqual(len(created), 3)
        dataset = json.loads((workspace / "data" / "dataset.jsonl").read_text(encoding="utf-8").strip())
        tool_catalog = json.loads((workspace / "data" / "tool_catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(tool_catalog, {})
        self.assertTrue(all(event["type"] == "message" for event in dataset["messages"]))

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

    @patch("teich_tune.runner.run_export")
    @patch("teich_tune.runner.run_eval")
    @patch("teich_tune.runner._run_streaming_command", return_value=0)
    def test_resume_auto_exports_when_enabled(self, _run_streaming, _run_eval, mock_run_export) -> None:
        config = example_config(self.root)
        config["outputs"]["gguf"] = {
            "enabled": True,
            "quants": ["q8", "q4"],
            "base_outtype": "f16",
            "metadata": {},
        }
        dump_yaml_compatible_json(config, self.config_path)
        job_dir = compile_only(self.config_path)
        job_path = Path(job_dir)
        adapter_dir = job_path / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapters.safetensors").write_text("stub", encoding="utf-8")
        run_resume(job_dir)
        mock_run_export.assert_called_once_with(job_path)

    @patch("teich_tune.runner.render_report")
    @patch("teich_tune.runner.generate_samples", return_value=[])
    def test_eval_skips_when_no_test_split(self, _generate_samples, _render_report) -> None:
        job_dir = compile_only(self.config_path)
        run_eval(job_dir)
        eval_log = (Path(job_dir) / "eval.log").read_text(encoding="utf-8")
        self.assertIn("Skipped evaluation", eval_log)
        self.assertIn("no validation or test split", eval_log)

    @patch("teich_tune.runner.render_report")
    @patch("teich_tune.runner.generate_samples", return_value=[])
    @patch("teich_tune.runner.subprocess.run")
    def test_eval_uses_validation_split_when_test_missing(
        self,
        mock_run,
        _generate_samples,
        _render_report,
    ) -> None:
        job_root = self.root / "chat-eval"
        data_dir = job_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "messages": [
                    {"type": "message", "role": "user", "content": "Say hello."},
                    {"type": "message", "role": "assistant", "content": "Hello."},
                ]
            },
            {
                "messages": [
                    {"type": "message", "role": "user", "content": "Say goodbye."},
                    {"type": "message", "role": "assistant", "content": "Goodbye."},
                ]
            },
        ]
        (data_dir / "dataset.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        (data_dir / "tool_catalog.json").write_text("{}\n", encoding="utf-8")
        config_path = job_root / "job.yaml"
        dump_yaml_compatible_json(
            {
                "name": "chat-eval",
                "data": {
                    "source": "data/dataset.jsonl",
                    "tool_catalog": "data/tool_catalog.json",
                    "split_seed": "chat-eval-seed",
                },
                "outputs": {"jobs_dir": "jobs", "sample_prompts": []},
            },
            config_path,
        )
        job_dir = compile_only(config_path)
        manifest = json.loads((Path(job_dir) / "compiled" / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["eval_split"], "valid")

        def fake_run(args, **kwargs):
            data_dir_arg = Path(args[args.index("--data") + 1])
            copied_test = data_dir_arg / "test.jsonl"
            self.assertTrue(copied_test.exists())
            return SimpleNamespace(returncode=0, stdout="Test loss 1.234, Test ppl 3.435.\n", stderr="")

        mock_run.side_effect = fake_run
        run_eval(job_dir)
        eval_log = (Path(job_dir) / "eval.log").read_text(encoding="utf-8")
        self.assertIn("Using validation split", eval_log)

    @patch("teich_tune.gguf.resolve_converter_python", return_value="python3.10")
    @patch("teich_tune.gguf._brew_llama_cpp_prefix", return_value=None)
    @patch("teich_tune.gguf.subprocess.run")
    def test_run_export_generates_default_q8_and_q4_artifacts(
        self,
        mock_run,
        _mock_brew_prefix,
        _mock_resolve_converter_python,
    ) -> None:
        llama_cpp_dir = self.root / "vendor" / "llama.cpp"
        quantize_dir = llama_cpp_dir / "build" / "bin"
        quantize_dir.mkdir(parents=True, exist_ok=True)
        (llama_cpp_dir / "convert_hf_to_gguf.py").write_text(
            'parser.add_argument("--metadata")\n',
            encoding="utf-8",
        )
        (quantize_dir / "llama-quantize").write_text("", encoding="utf-8")

        config = example_config(self.root)
        config["outputs"]["gguf"] = {
            "enabled": True,
            "quants": ["q8", "q4"],
            "base_outtype": "f16",
            "llama_cpp_dir": str(llama_cpp_dir),
            "metadata": {"general.author": "unit-test"},
        }
        dump_yaml_compatible_json(config, self.config_path)
        job_dir = compile_only(self.config_path)
        job_path = Path(job_dir)
        adapter_dir = job_path / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapters.safetensors").write_text("stub", encoding="utf-8")

        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append([str(item) for item in args])
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        mock_run.side_effect = fake_run

        exported = run_export(job_dir)
        self.assertEqual(exported, job_path)

        self.assertEqual(len(calls), 4)
        self.assertIn("-m", calls[0])
        self.assertIn("mlx_lm", calls[0])
        self.assertIn("fuse", calls[0])
        self.assertIn("--dequantize", calls[0])

        self.assertEqual(calls[1][1], str((llama_cpp_dir / "convert_hf_to_gguf.py").resolve()))
        self.assertIn("--outtype", calls[1])
        self.assertIn("f16", calls[1])
        self.assertIn("--metadata", calls[1])
        self.assertEqual(calls[2][-1], "Q8_0")
        self.assertEqual(calls[3][-1], "Q4_K_M")

        manifest = json.loads((job_path / "exports" / "gguf" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([item["quant"] for item in manifest["outputs"]], ["Q8_0", "Q4_K_M"])
        self.assertEqual(manifest["base_gguf"]["quant"], "F16")
        report_text = (job_path / "report.md").read_text(encoding="utf-8")
        self.assertIn("## GGUF Exports", report_text)
        self.assertIn("Q8_0", report_text)
        self.assertIn("Q4_K_M", report_text)


if __name__ == "__main__":
    unittest.main()
