# Examples

These example jobs are intentionally tiny. They are meant to test the pipeline, not produce a useful fine-tune.

## Chat-only example

```bash
PYTHONPATH=src python3 -m longhaul validate examples/chat-minimal/job.yaml
PYTHONPATH=src python3 -m longhaul compile -c examples/chat-minimal/job.yaml
```

## Tool-call example

```bash
PYTHONPATH=src python3 -m longhaul validate examples/tool-call/job.yaml
PYTHONPATH=src python3 -m longhaul compile -c examples/tool-call/job.yaml
```

## Real subset example

`examples/glm5-plain-100/` is a non-thinking plain-chat subset derived from:

- https://huggingface.co/datasets/Jackrong/GLM-5.1-Reasoning-1M-Cleaned

It contains 100 rows total with explicit `train/valid/test` files and reasoning removed from the assistant targets.

```bash
PYTHONPATH=src python3 -m longhaul validate examples/glm5-plain-100/job.yaml
PYTHONPATH=src python3 -m longhaul compile -c examples/glm5-plain-100/job.yaml
```

If `mlx-lm` is installed, you can train either one:

```bash
PYTHONPATH=src python3 -m longhaul train -c examples/chat-minimal/job.yaml
PYTHONPATH=src python3 -m longhaul train -c examples/tool-call/job.yaml
PYTHONPATH=src python3 -m longhaul train -c examples/glm5-plain-100/job.yaml
```

Generated run outputs land under each example's `jobs/` directory, but those artifacts are ignored by git. Keep the example inputs, not the compiled runs.
