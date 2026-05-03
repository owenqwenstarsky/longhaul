# Examples

These example jobs are intentionally tiny. They are meant to test the pipeline, not produce a useful fine-tune.

## Chat-only example

```bash
PYTHONPATH=src python3 -m teich_tune validate examples/chat-minimal/job.yaml
PYTHONPATH=src python3 -m teich_tune compile -c examples/chat-minimal/job.yaml
```

## Tool-call example

```bash
PYTHONPATH=src python3 -m teich_tune validate examples/tool-call/job.yaml
PYTHONPATH=src python3 -m teich_tune compile -c examples/tool-call/job.yaml
```

If `mlx-lm` is installed, you can train either one:

```bash
PYTHONPATH=src python3 -m teich_tune train -c examples/chat-minimal/job.yaml
PYTHONPATH=src python3 -m teich_tune train -c examples/tool-call/job.yaml
```
