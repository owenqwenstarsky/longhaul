# Long Haul by TEI

`longhaul` is the CLI for Long Haul by TEI. It prepares datasets, compiles MLX-ready training files, and runs small fine-tunes on Apple Silicon.

## What v0.1 does

- Validates a simple structured JSONL dataset format.
- Resolves tool definitions from a shared catalog.
- Compiles canonical records into MLX-compatible `chat` or `tools` JSONL.
- Applies Qwen-specific `conservative` and `expert` training presets.
- Runs one local MLX job at a time with resumable artifacts and reports.
- Uses deterministic auto-splitting with validation/test fallback rules that work on small datasets.

## Install

```bash
python3 -m pip install ".[train]"
```

That installs the `longhaul` CLI plus MLX training dependencies.

If you only want dataset validation and compilation:

```bash
python3 -m pip install .
```

## Quick start

```bash
longhaul init --template chat
longhaul validate job.yaml
longhaul compile -c job.yaml
longhaul train -c job.yaml
```

Use `--template tools` if you want a starter tool-calling dataset instead of plain chat.

## GGUF export

`longhaul` can export a completed MLX LoRA job to GGUF for `llama.cpp` and other GGUF runtimes.

Requirements:

- A `llama.cpp` checkout with `convert_hf_to_gguf.py`
- A built `llama-quantize` (or `quantize`) binary
- Either `outputs.gguf.llama_cpp_dir` in `job.yaml` or `LLAMA_CPP_DIR` in the environment

Manual export:

```bash
longhaul export /path/to/job
longhaul export /path/to/job --quant q8 --quant q4_k_m --quant bf16
```

Default aliases:

- `q8` -> `Q8_0`
- `q4` -> `Q4_K_M`

Recommended config snippet:

```json
{
  "outputs": {
    "jobs_dir": "jobs",
    "sample_prompts": [
      "Summarize the assistant behavior you were trained for.",
      "Respond to a user request in the target style."
    ],
    "gguf": {
      "enabled": true,
      "quants": ["q8", "q4"],
      "base_outtype": "f16",
      "llama_cpp_dir": "../llama.cpp"
    }
  }
}
```

When `outputs.gguf.enabled` is `true`, `longhaul train` and `longhaul resume` automatically export GGUF artifacts after evaluation finishes. Exported files are written under `jobs/<job>/exports/gguf/`, and the job report includes the generated GGUF paths.

## Included examples

See [examples/README.md](/Users/owen/longhaul/examples/README.md) for two tiny starter jobs:

- [examples/chat-minimal/job.yaml](/Users/owen/longhaul/examples/chat-minimal/job.yaml)
- [examples/tool-call/job.yaml](/Users/owen/longhaul/examples/tool-call/job.yaml)
- [examples/glm5-plain-100/job.yaml](/Users/owen/longhaul/examples/glm5-plain-100/job.yaml)

## Prepared subset example

`examples/glm5-plain-100/` contains a real plain-chat subset prepared from the Hugging Face dataset `Jackrong/GLM-5.1-Reasoning-1M-Cleaned`.

- The subset uses 100 examples total with explicit `90/5/5` train/valid/test splits.
- `<think>...</think>` reasoning blocks are stripped before training.
- The prep script filters out oversized records to keep the subset suitable for a small Qwen 2.5 1.5B run.

To regenerate that subset locally:

```bash
PYTHONPATH=src python3.10 scripts/prepare_glm5_reasoning_subset.py \
  --count 100 \
  --max-estimated-tokens 1800 \
  --output-dir examples/glm5-plain-100/data
```

## Canonical dataset format

Each line is one JSON object:

```json
{
  "messages": [
    { "type": "message", "role": "user", "content": "Create PLAN.md" },
    {
      "type": "tool_call",
      "name": "write",
      "arguments": {
        "path": "PLAN.md",
        "content": "# Plan\n..."
      }
    },
    {
      "type": "tool_result",
      "tool_call_id": "call_abc123",
      "name": "write",
      "content": "Wrote PLAN.md",
      "is_error": false
    },
    {
      "type": "message",
      "role": "assistant",
      "content": "The plan has been saved."
    }
  ],
  "tools": ["write"]
}
```

Assistant reasoning can be stored separately:

```json
{
  "type": "message",
  "role": "assistant",
  "thinking": "First I should write the file, then confirm it.",
  "content": "The file is ready."
}
```

By default, `thinking` is omitted from the compiled training set.

## Profiles

- `conservative` is the default and keeps the LoRA config intentionally small.
- `expert` raises the default LoRA rank, effective batch size, and trainable layers. It is still overrideable per field in `job.yaml`.

## Auto split behavior

- Single-file datasets are split deterministically.
- Datasets with fewer than 10 records use train+valid only by default.
- Once the dataset reaches 10 or more records, Long Haul creates train, valid, and test splits.
- If no test split exists, `longhaul eval` falls back to the validation split instead of silently doing nothing.

## Artifact policy

Commit source files and example datasets. Do not commit generated run artifacts.

Ignored by default:

- `jobs/`
- `examples/*/jobs/`
- `.smoke/`
- `.longhaul-validate/`
- `__pycache__/`
- `*.egg-info/`

## Notes

- The generated `job.yaml` template is JSON text with a `.yaml` extension. JSON is valid YAML, which keeps the bootstrap dependency-free.
- v0.1 validates a small allowlist of Qwen model IDs rather than claiming blanket support.
