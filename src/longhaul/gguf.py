from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from longhaul.config import load_job_config


DEFAULT_GGUF_QUANTS = ("q8", "q4")
GGUF_ALIAS_MAP = {
    "q4": "q4_k_m",
    "q8": "q8_0",
}
HIGH_PRECISION_GGUF_TYPES = {"f16", "bf16", "f32"}
GGUF_MANIFEST_PATH = Path("exports") / "gguf" / "manifest.json"
PYTHON_BIN = sys.executable or "python3"


@dataclass(frozen=True)
class LlamaCppToolchain:
    root_dir: Optional[Path]
    converter_script: Path
    quantize_binary: Optional[Path]
    supports_metadata_override: bool


def normalize_gguf_quant(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("GGUF quantization targets must not be empty.")
    canonical = GGUF_ALIAS_MAP.get(normalized, normalized)
    if not canonical.replace("_", "").isalnum():
        raise ValueError(
            f"Unsupported GGUF quantization target {value!r}. "
            "Use aliases like 'q8'/'q4' or exact llama.cpp names like 'q8_0'/'q4_k_m'."
        )
    return canonical


def resolve_gguf_targets(requested: Optional[Sequence[str]]) -> List[str]:
    raw_values = list(requested) if requested else list(DEFAULT_GGUF_QUANTS)
    resolved: List[str] = []
    seen = set()
    for item in raw_values:
        canonical = normalize_gguf_quant(item)
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved


def gguf_target_label(quant: str) -> str:
    return normalize_gguf_quant(quant).upper()


def validate_gguf_settings(config: Dict[str, Any]) -> None:
    gguf_config = config.get("outputs", {}).get("gguf", {})
    if not isinstance(gguf_config, dict):
        raise ValueError("outputs.gguf must be an object/mapping when provided.")

    quants = gguf_config.get("quants", DEFAULT_GGUF_QUANTS)
    if not isinstance(quants, list) or not all(isinstance(item, str) for item in quants):
        raise ValueError("outputs.gguf.quants must be a list of strings.")
    resolve_gguf_targets(quants)

    base_outtype = normalize_gguf_quant(str(gguf_config.get("base_outtype", "f16")))
    if base_outtype not in HIGH_PRECISION_GGUF_TYPES:
        supported = ", ".join(sorted(HIGH_PRECISION_GGUF_TYPES))
        raise ValueError(f"outputs.gguf.base_outtype must be one of: {supported}.")

    metadata = gguf_config.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("outputs.gguf.metadata must be an object/mapping when provided.")

    for key in metadata.keys():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("outputs.gguf.metadata keys must be non-empty strings.")

    llama_cpp_dir = gguf_config.get("llama_cpp_dir")
    if llama_cpp_dir is not None and not isinstance(llama_cpp_dir, str):
        raise ValueError("outputs.gguf.llama_cpp_dir must be a string path when provided.")
    converter_python = gguf_config.get("converter_python")
    if converter_python is not None and not isinstance(converter_python, str):
        raise ValueError("outputs.gguf.converter_python must be a string path/command when provided.")


def _llama_cpp_candidates(configured_dir: Optional[str | Path]) -> Iterable[Path]:
    values: List[Path] = []
    if configured_dir:
        values.append(Path(configured_dir).expanduser())
    env_dir = os.getenv("LLAMA_CPP_DIR")
    if env_dir:
        values.append(Path(env_dir).expanduser())
    brew_prefix = _brew_llama_cpp_prefix()
    if brew_prefix is not None:
        values.append(brew_prefix)
    home = Path.home()
    values.extend(
        [
            Path.cwd() / "llama.cpp",
            Path.cwd().parent / "llama.cpp",
            home / "llama.cpp",
            home / "src" / "llama.cpp",
            home / "code" / "llama.cpp",
            home / "dev" / "llama.cpp",
        ]
    )

    seen: set[Path] = set()
    for path in values:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def _brew_llama_cpp_prefix() -> Optional[Path]:
    brew_bin = shutil.which("brew")
    if not brew_bin:
        return None
    result = subprocess.run(
        [brew_bin, "--prefix", "llama.cpp"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    prefix = result.stdout.strip()
    if not prefix:
        return None
    path = Path(prefix).expanduser().resolve()
    return path if path.exists() else None


def _find_converter_script(root_dir: Path) -> Optional[Path]:
    candidates = [
        root_dir / "convert_hf_to_gguf.py",
        root_dir / "bin" / "convert_hf_to_gguf.py",
        root_dir / "share" / "llama.cpp" / "convert_hf_to_gguf.py",
        root_dir / "libexec" / "convert_hf_to_gguf.py",
        root_dir / "lib" / "llama.cpp" / "convert_hf_to_gguf.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_python_command(candidate: str) -> Optional[str]:
    path = Path(candidate).expanduser()
    if path.is_file():
        return str(path if path.is_absolute() else Path.cwd() / path)
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    return None


def _python_module_check(python_bin: str, modules: Sequence[str]) -> tuple[bool, str]:
    result = subprocess.run(
        [python_bin, "-c", "import " + ", ".join(modules)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip() or result.stdout.strip() or "unknown import error"


def resolve_converter_python(configured_python: Optional[str], default_python: str, modules: Sequence[str]) -> str:
    candidates: List[str] = []
    env_python = os.getenv("LLAMA_CPP_PYTHON")
    if env_python:
        candidates.append(env_python)
    if configured_python:
        candidates.append(configured_python)
    candidates.append(default_python)
    candidates.extend(
        [
            str(Path.cwd() / ".gguf-python" / "bin" / "python"),
            str(Path.cwd() / ".gguf-python" / "bin" / "python3"),
            str(Path.cwd() / ".gguf-python" / "bin" / "python3.10"),
            str(Path.cwd() / ".venv" / "bin" / "python"),
            str(Path.cwd() / ".venv" / "bin" / "python3"),
        ]
    )
    if default_python != "python3":
        candidates.append("python3")

    seen: set[str] = set()
    failures: List[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved = _resolve_python_command(candidate)
        if not resolved:
            continue
        ok, message = _python_module_check(resolved, modules)
        if ok:
            return resolved
        failures.append(f"{resolved}: {message}")

    module_list = ", ".join(modules)
    raise RuntimeError(
        f"GGUF conversion requires a Python environment with {module_list}. "
        "Set outputs.gguf.converter_python or LLAMA_CPP_PYTHON to a suitable interpreter. "
        + (" Tried: " + " | ".join(failures) if failures else "")
    )


def _find_quantize_binary(root_dir: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if root_dir is not None:
        candidates.extend(
            [
                root_dir / "llama-quantize",
                root_dir / "quantize",
                root_dir / "build" / "bin" / "llama-quantize",
                root_dir / "build" / "bin" / "quantize",
                root_dir / "bin" / "llama-quantize",
                root_dir / "bin" / "quantize",
                root_dir / "libexec" / "bin" / "llama-quantize",
                root_dir / "libexec" / "bin" / "quantize",
            ]
        )
    for name in ("llama-quantize", "quantize"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_llama_cpp_toolchain(configured_dir: Optional[str | Path], *, require_quantize: bool) -> LlamaCppToolchain:
    converter_script: Optional[Path] = None
    root_dir: Optional[Path] = None
    for candidate in _llama_cpp_candidates(configured_dir):
        script_path = _find_converter_script(candidate)
        if script_path is not None and script_path.exists():
            root_dir = candidate
            converter_script = script_path
            break
    if converter_script is None:
        raise RuntimeError(
            "Could not find llama.cpp's convert_hf_to_gguf.py. "
            "Set outputs.gguf.llama_cpp_dir or the LLAMA_CPP_DIR environment variable."
        )

    quantize_binary = _find_quantize_binary(root_dir)
    if require_quantize and quantize_binary is None:
        raise RuntimeError(
            "Could not find llama.cpp's quantizer binary. Build llama.cpp so that "
            "`llama-quantize` (or `quantize`) is available, or point outputs.gguf.llama_cpp_dir "
            "at a checkout with built binaries."
        )

    script_text = converter_script.read_text(encoding="utf-8", errors="ignore")
    return LlamaCppToolchain(
        root_dir=root_dir,
        converter_script=converter_script,
        quantize_binary=quantize_binary,
        supports_metadata_override="--metadata" in script_text or "metadata_override" in script_text,
    )


def build_gguf_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    model_id = str(config["model"]["id"])
    metadata: Dict[str, Any] = {
        "general.name": config["name"],
        "general.author": "TEI",
        "general.basename": model_id.split("/")[-1],
        "general.finetune": "lora",
        "general.description": f"LoRA fine-tune exported by Long Haul by TEI from {model_id}.",
    }
    if "/" in model_id and not model_id.startswith(("/", ".")):
        source_url = f"https://huggingface.co/{model_id}"
        metadata["general.source.url"] = source_url
        metadata["general.source.repo_url"] = source_url

    extra_metadata = dict(config.get("outputs", {}).get("gguf", {}).get("metadata", {}))
    metadata.update(extra_metadata)
    return metadata


def load_export_manifest(job_dir: str | Path) -> Optional[Dict[str, Any]]:
    manifest_path = Path(job_dir) / GGUF_MANIFEST_PATH
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _run_logged_command(
    args: Sequence[str | Path],
    *,
    description: str,
    log_path: Path,
    cwd: Optional[Path] = None,
) -> None:
    string_args = [str(item) for item in args]
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("$ " + shlex.join(string_args) + "\n")
        result = subprocess.run(
            string_args,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            log_handle.write(result.stdout)
        if result.stderr:
            log_handle.write(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"{description} failed with exit code {result.returncode}. See {log_path}.")


def _high_precision_output_plan(
    targets: Sequence[str],
    base_outtype: str,
) -> tuple[str, List[str]]:
    requested_precisions = [item for item in targets if item in HIGH_PRECISION_GGUF_TYPES]
    quantized_targets = [item for item in targets if item not in HIGH_PRECISION_GGUF_TYPES]
    quantize_source_type = requested_precisions[0] if requested_precisions else base_outtype
    needed_precisions: List[str] = []
    if quantized_targets:
        needed_precisions.append(quantize_source_type)
    for item in requested_precisions:
        if item not in needed_precisions:
            needed_precisions.append(item)
    return quantize_source_type, needed_precisions


def export_job_to_gguf(
    job_dir: str | Path,
    *,
    requested_quants: Optional[Sequence[str]] = None,
    python_bin: str = PYTHON_BIN,
) -> Dict[str, Any]:
    job_path = Path(job_dir)
    snapshot_path = job_path / "config.snapshot.yaml"
    if not snapshot_path.exists():
        raise ValueError(f"Missing {snapshot_path}.")
    config = load_job_config(snapshot_path)
    gguf_config = dict(config.get("outputs", {}).get("gguf", {}))
    targets = resolve_gguf_targets(requested_quants if requested_quants is not None else gguf_config.get("quants"))
    quantized_targets = [item for item in targets if item not in HIGH_PRECISION_GGUF_TYPES]
    base_outtype = normalize_gguf_quant(str(gguf_config.get("base_outtype", "f16")))
    toolchain = resolve_llama_cpp_toolchain(
        gguf_config.get("llama_cpp_dir"),
        require_quantize=bool(quantized_targets),
    )
    converter_python = resolve_converter_python(
        gguf_config.get("converter_python"),
        python_bin,
        ("torch", "transformers", "numpy", "gguf"),
    )

    adapter_path = job_path / "adapters"
    if not adapter_path.exists():
        raise ValueError(f"Missing adapter directory {adapter_path}. Train the job before exporting GGUF artifacts.")

    export_root = job_path / "exports" / "gguf"
    intermediate_dir = export_root / "intermediate"
    fused_dir = export_root / "fused-model"
    export_root.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    log_path = export_root / "export.log"
    metadata_path = export_root / "metadata.override.json"
    metadata = build_gguf_metadata(config)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    fuse_args = [
        python_bin,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        str(config["model"]["id"]),
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(fused_dir),
        "--dequantize",
    ]
    _run_logged_command(
        fuse_args,
        description="MLX adapter fusion",
        log_path=log_path,
        cwd=Path.cwd(),
    )

    quantize_source_type, needed_precisions = _high_precision_output_plan(targets, base_outtype)
    precision_paths: Dict[str, Path] = {}
    requested_precisions = {item for item in targets if item in HIGH_PRECISION_GGUF_TYPES}

    for outtype in needed_precisions:
        output_path = (
            export_root / f"ggml-model-{gguf_target_label(outtype)}.gguf"
            if outtype in requested_precisions
            else intermediate_dir / f"ggml-model-{gguf_target_label(outtype)}.gguf"
        )
        precision_paths[outtype] = output_path
        convert_args: List[str | Path] = [
            converter_python,
            toolchain.converter_script,
            "--outfile",
            output_path,
            "--outtype",
            outtype,
            "--model-name",
            str(config["name"]),
        ]
        if toolchain.supports_metadata_override:
            convert_args.extend(["--metadata", metadata_path])
        convert_args.append(fused_dir)
        _run_logged_command(
            convert_args,
            description=f"GGUF conversion ({gguf_target_label(outtype)})",
            log_path=log_path,
            cwd=toolchain.root_dir,
        )

    artifacts: List[Dict[str, Any]] = []
    for target in targets:
        if target in HIGH_PRECISION_GGUF_TYPES:
            artifacts.append(
                {
                    "quant": gguf_target_label(target),
                    "path": str(precision_paths[target].resolve()),
                    "kind": "full_precision",
                }
            )
            continue
        assert toolchain.quantize_binary is not None
        output_path = export_root / f"ggml-model-{gguf_target_label(target)}.gguf"
        quantize_args = [
            toolchain.quantize_binary,
            precision_paths[quantize_source_type],
            output_path,
            gguf_target_label(target),
        ]
        _run_logged_command(
            quantize_args,
            description=f"GGUF quantization ({gguf_target_label(target)})",
            log_path=log_path,
            cwd=toolchain.root_dir,
        )
        artifacts.append(
            {
                "quant": gguf_target_label(target),
                "path": str(output_path.resolve()),
                "kind": "quantized",
            }
        )

    manifest = {
        "job_dir": str(job_path.resolve()),
        "model_id": str(config["model"]["id"]),
        "llama_cpp_dir": str(toolchain.root_dir.resolve()) if toolchain.root_dir is not None else None,
        "converter_python": converter_python,
        "fused_model_dir": str(fused_dir.resolve()),
        "metadata_override_path": str(metadata_path.resolve()),
        "metadata_override_applied": toolchain.supports_metadata_override,
        "base_gguf": {
            "quant": gguf_target_label(quantize_source_type),
            "path": str(precision_paths[quantize_source_type].resolve()),
        },
        "outputs": artifacts,
    }
    manifest_path = job_path / GGUF_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
