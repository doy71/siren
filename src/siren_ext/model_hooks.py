"""Generic representation extractor compatible with CSSLab/SIREN data shape.

Official SIREN exposes Qwen3RepresentationExtractor whose extract_batch() returns:
    List[Dict[layer_idx, Dict['residual_mean'|'mlp_mean', np.ndarray]]]

This module keeps that interface but makes layer/MLP discovery more tolerant so
Qwen, Llama, EXAONE, and other decoder-only HF models can be used.
"""
from __future__ import annotations

import gc
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_by_path(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_decoder_layers(model: Any) -> list[Any]:
    """Find decoder block list across common HF causal LM architectures."""
    candidates = [
        "model.layers",          # Llama, Qwen, Mistral, many EXAONE variants
        "transformer.h",         # GPT-style
        "gpt_neox.layers",
        "model.decoder.layers",
        "decoder.layers",
    ]
    for path in candidates:
        layers = _get_by_path(model, path)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return list(layers)
    raise ValueError(
        "Could not locate decoder layers. Add the architecture path to find_decoder_layers() "
        f"for model class {model.__class__.__name__}."
    )


def find_mlp_module(layer: Any) -> Any | None:
    """Find a feed-forward/MLP submodule inside a decoder block."""
    for name in ["mlp", "feed_forward", "ffn", "feedforward", "ff", "dense_h_to_4h"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    # Last resort: search names containing mlp/feed_forward.
    for module_name, module in layer.named_modules():
        lname = module_name.lower()
        if lname.endswith("mlp") or "feed_forward" in lname or lname.endswith("ffn"):
            return module
    return None


class GenericRepresentationExtractor:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        batch_size: int = 16,
        rep_types: Optional[list[str]] = None,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        max_length: int = 512,
        device_map: str | dict[str, str] | None = None,
        no_fast_tokenizer: bool = False,
    ):
        self.device = device
        self.batch_size = batch_size
        self.rep_types = rep_types if rep_types else ["residual_mean", "mlp_mean"]
        self.max_length = max_length
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }.get(torch_dtype, torch.bfloat16)
        if device_map is None:
            device_map = {"": device} if device != "cpu" else None
        load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code, "torch_dtype": dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if device == "cpu" and device_map is None:
            self.model.to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=not no_fast_tokenizer,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model.eval()
        self.layers = find_decoder_layers(self.model)
        self.num_layers = len(self.layers)
        self.residual_outputs: list[Any] = []
        self.mlp_outputs: list[Any] = []
        self.hooks: list[Any] = []

    def _residual_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            hidden_states = hidden_states.detach()
            if len(self.residual_outputs) <= layer_idx:
                self.residual_outputs.extend([None] * (layer_idx + 1 - len(self.residual_outputs)))
            self.residual_outputs[layer_idx] = hidden_states
        return hook

    def _mlp_hook(self, layer_idx: int):
        def hook(module, input, output):
            mlp_output = output[0] if isinstance(output, tuple) else output
            mlp_output = mlp_output.detach()
            if len(self.mlp_outputs) <= layer_idx:
                self.mlp_outputs.extend([None] * (layer_idx + 1 - len(self.mlp_outputs)))
            self.mlp_outputs[layer_idx] = mlp_output
        return hook

    def register_hooks(self) -> None:
        for idx, layer in enumerate(self.layers):
            if any(rt.startswith("residual") for rt in self.rep_types):
                self.hooks.append(layer.register_forward_hook(self._residual_hook(idx)))
            if any(rt.startswith("mlp") for rt in self.rep_types):
                mlp = find_mlp_module(layer)
                if mlp is not None:
                    self.hooks.append(mlp.register_forward_hook(self._mlp_hook(idx)))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _to_batch_first(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        # Some hooks can expose [T,B,H]. Convert to [B,T,H].
        if tensor.shape[0] != batch_size and tensor.dim() >= 3 and tensor.shape[1] == batch_size:
            tensor = tensor.transpose(0, 1)
        return tensor

    def extract_batch(self, texts: list[str]) -> list[dict[int, dict[str, np.ndarray]]]:
        texts = [t.strip() if t and t.strip() else " " for t in texts]
        self.residual_outputs = []
        self.mlp_outputs = []
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = self.model(**inputs, use_cache=False)
        actual_batch_size = inputs["input_ids"].shape[0]
        mask_batch = inputs["attention_mask"]
        batch_representations: list[dict[int, dict[str, np.ndarray]]] = []
        for batch_idx in range(actual_batch_size):
            sample: dict[int, dict[str, np.ndarray]] = {}
            valid_len = int(mask_batch[batch_idx].sum().item())
            for layer_idx in range(self.num_layers):
                layer_rep: dict[str, np.ndarray] = {}
                if "residual_mean" in self.rep_types and layer_idx < len(self.residual_outputs) and self.residual_outputs[layer_idx] is not None:
                    residual = self._to_batch_first(self.residual_outputs[layer_idx], actual_batch_size)[batch_idx]
                    layer_rep["residual_mean"] = residual[:valid_len].mean(dim=0).cpu().float().numpy()
                if "mlp_mean" in self.rep_types and layer_idx < len(self.mlp_outputs) and self.mlp_outputs[layer_idx] is not None:
                    mlp = self._to_batch_first(self.mlp_outputs[layer_idx], actual_batch_size)[batch_idx]
                    layer_rep["mlp_mean"] = mlp[:valid_len].mean(dim=0).cpu().float().numpy()
                if layer_rep:
                    sample[layer_idx] = layer_rep
            batch_representations.append(sample)
        return batch_representations

    def close(self) -> None:
        self.remove_hooks()
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Drop-in aliases matching CSSLab/SIREN naming.
Qwen3RepresentationExtractor = GenericRepresentationExtractor
LlamaRepresentationExtractor = GenericRepresentationExtractor
EXAONERepresentationExtractor = GenericRepresentationExtractor
