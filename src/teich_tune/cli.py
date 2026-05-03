from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from teich_tune.runner import (
    compile_only,
    init_workspace,
    print_report,
    run_eval,
    run_resume,
    run_train,
    validate_only,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="teich-tune")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a starter workspace.")
    init_parser.add_argument("--dir", default=".", help="Workspace directory to initialize.")

    validate_parser = subparsers.add_parser("validate", help="Validate config and dataset.")
    validate_parser.add_argument("config", nargs="?", default="job.yaml", help="Path to job config.")

    compile_parser = subparsers.add_parser("compile", help="Compile canonical dataset into MLX JSONL.")
    compile_parser.add_argument("-c", "--config", default="job.yaml", help="Path to job config.")
    compile_parser.add_argument("--output-dir", default=None, help="Optional job directory to write into.")

    train_parser = subparsers.add_parser("train", help="Run a new MLX fine-tune job.")
    train_parser.add_argument("-c", "--config", default="job.yaml", help="Path to job config.")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing job.")
    resume_parser.add_argument("job_dir", help="Existing job directory.")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation and sample generation for an existing job.")
    eval_parser.add_argument("job_dir", help="Existing job directory.")

    report_parser = subparsers.add_parser("report", help="Print the generated report for an existing job.")
    report_parser.add_argument("job_dir", help="Existing job directory.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            created = init_workspace(args.dir)
            print(json.dumps({"created": created}, indent=2))
            return 0
        if args.command == "validate":
            payload = validate_only(args.config)
            shutil.rmtree(payload["job_dir"], ignore_errors=True)
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "compile":
            job_dir = compile_only(args.config, output_dir=args.output_dir)
            print(json.dumps({"job_dir": str(Path(job_dir).resolve())}, indent=2))
            return 0
        if args.command == "train":
            job_dir = run_train(args.config)
            print(json.dumps({"job_dir": str(Path(job_dir).resolve())}, indent=2))
            return 0
        if args.command == "resume":
            job_dir = run_resume(args.job_dir)
            print(json.dumps({"job_dir": str(Path(job_dir).resolve())}, indent=2))
            return 0
        if args.command == "eval":
            job_dir = run_eval(args.job_dir)
            print(json.dumps({"job_dir": str(Path(job_dir).resolve())}, indent=2))
            return 0
        if args.command == "report":
            print(print_report(args.job_dir))
            return 0
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        parser.exit(status=1, message=f"error: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
