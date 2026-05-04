from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    approx_params_b: float
    max_seq_length: int
    supports_thinking: bool
    quantized: bool
    recommended_num_layers: int


SUPPORTED_MODELS: Dict[str, ModelSpec] = {
    "Qwen/Qwen2.5-1.5B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        family="qwen2.5",
        approx_params_b=1.5,
        max_seq_length=2048,
        supports_thinking=False,
        quantized=False,
        recommended_num_layers=8,
    ),
    "Qwen/Qwen2.5-3B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen2.5",
        approx_params_b=3.0,
        max_seq_length=2048,
        supports_thinking=False,
        quantized=False,
        recommended_num_layers=8,
    ),
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit": ModelSpec(
        model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        family="qwen2.5",
        approx_params_b=1.5,
        max_seq_length=2048,
        supports_thinking=False,
        quantized=True,
        recommended_num_layers=8,
    ),
    "mlx-community/Qwen2.5-3B-Instruct-4bit": ModelSpec(
        model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
        family="qwen2.5",
        approx_params_b=3.0,
        max_seq_length=2048,
        supports_thinking=False,
        quantized=True,
        recommended_num_layers=8,
    ),
    "Qwen/Qwen3-1.7B": ModelSpec(
        model_id="Qwen/Qwen3-1.7B",
        family="qwen3",
        approx_params_b=1.7,
        max_seq_length=2048,
        supports_thinking=True,
        quantized=False,
        recommended_num_layers=8,
    ),
    "mlx-community/Qwen3-1.7B-4bit": ModelSpec(
        model_id="mlx-community/Qwen3-1.7B-4bit",
        family="qwen3",
        approx_params_b=1.7,
        max_seq_length=2048,
        supports_thinking=True,
        quantized=True,
        recommended_num_layers=8,
    ),
}


def supported_model_ids() -> Iterable[str]:
    return SUPPORTED_MODELS.keys()


def resolve_model(model_id: str, allow_unsupported: bool = False) -> ModelSpec:
    spec = SUPPORTED_MODELS.get(model_id)
    if spec is None:
        if allow_unsupported:
            return ModelSpec(
                model_id=model_id,
                family="unknown",
                approx_params_b=0.0,
                max_seq_length=2048,
                supports_thinking=False,
                quantized="4bit" in model_id.lower() or "8bit" in model_id.lower(),
                recommended_num_layers=8,
            )
        supported = "\n".join(f"- {item}" for item in supported_model_ids())
        raise ValueError(
            f"Unsupported model id: {model_id}\n"
            "Validated models:\n"
            f"{supported}"
        )
    return spec
