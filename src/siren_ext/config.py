"""Model registry compatible with CSSLab/SIREN's MODEL_CONFIGS.

The official repo hard-codes Qwen/Llama model entries in utils/config.py.  This
extension keeps the same shape but adds Korean-capable backbones.  For models
whose layer/hidden sizes may change across releases, leave them as None and the
extractor will infer them after loading the HF model.
"""
from __future__ import annotations

from typing import Dict, Any

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Official SIREN backbones.
    "qwen3-0.6b": {
        "model_path": "Qwen/Qwen3-0.6B",
        "model_type": "qwen3",
        "num_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072,
    },
    "qwen3-1.7b": {
        "model_path": "Qwen/Qwen3-1.7B",
        "model_type": "qwen3",
        "num_layers": 28,
        "hidden_size": 2048,
        "intermediate_size": 5504,
    },
    "qwen3-4b": {
        "model_path": "Qwen/Qwen3-4B",
        "model_type": "qwen3",
        "num_layers": 36,
        "hidden_size": 2560,
        "intermediate_size": 9728,
    },
    "llama3.2-1b": {
        "model_path": "meta-llama/Llama-3.2-1B",
        "model_type": "llama",
        "num_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
    },
    "llama3.2-3b": {
        "model_path": "meta-llama/Llama-3.2-3B",
        "model_type": "llama",
        "num_layers": 28,
        "hidden_size": 3072,
        "intermediate_size": 8192,
    },
    "llama3.1-8b": {
        "model_path": "meta-llama/Llama-3.1-8B",
        "model_type": "llama",
        "num_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
    },
    # Korean / bilingual backbones for this project.
    "exaone3.5-2.4b": {
        "model_path": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "model_type": "exaone",
        "num_layers": None,
        "hidden_size": None,
        "intermediate_size": None,
    },
    "exaone3.5-7.8b": {
        "model_path": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "model_type": "exaone",
        "num_layers": None,
        "hidden_size": None,
        "intermediate_size": None,
    },
    "exaone4.0-1.2b": {
        "model_path": "LGAI-EXAONE/EXAONE-4.0-1.2B",
        "model_type": "exaone",
        "num_layers": None,
        "hidden_size": None,
        "intermediate_size": None,
    },
}

POOLING_STRATEGIES = ["mean"]
REPRESENTATION_TYPES = ["residual", "mlp"]


def resolve_model_config(model: str | None = None, model_name_or_path: str | None = None) -> Dict[str, Any]:
    if model and model in MODEL_CONFIGS:
        return dict(MODEL_CONFIGS[model])
    if model_name_or_path:
        return {
            "model_path": model_name_or_path,
            "model_type": model or "custom",
            "num_layers": None,
            "hidden_size": None,
            "intermediate_size": None,
        }
    raise ValueError("Provide --model from registry or --model-name-or-path")
