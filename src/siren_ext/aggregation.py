"""Shared feature aggregation for Ko-SIREN train and eval.

Previously ``aggregate_features`` was duplicated in ``train_siren_ko.py`` and
``evaluate_siren_ko.py`` with subtly different layer_weights key handling
(int keys in training, str keys after pickle round-trip).  This module
provides a single implementation that handles both transparently.
"""
from __future__ import annotations

import numpy as np


def aggregate_features(
    representations: list[dict[int, dict[str, np.ndarray]]],
    pooling_type: str,
    selected_neurons_dict: dict[str, list[int]],
    layer_weights: dict,
    selected_layers: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Aggregate selected-neuron features across layers into a flat matrix.

    Args:
        representations: Per-sample dicts ``{layer_idx: {pooling_type: np.ndarray}}``.
        pooling_type: Representation key, e.g. ``"residual_mean"`` or ``"mlp_mean"``.
        selected_neurons_dict: Maps ``"layer{i}_{pooling_type}"`` → list of neuron indices.
        layer_weights: Maps layer index to a scalar weight.  Keys may be ``int`` (as
            produced by ``get_layer_weights`` during training) or ``str`` (as stored in
            ``best_model.pkl`` after JSON-serialisation normalisation).  Both are handled.
        selected_layers: Ordered list of layer indices to include.  Defaults to sorted
            keys of *layer_weights*.

    Returns:
        ``(X, selected_layers)`` where *X* is ``float32`` of shape
        ``(n_samples, total_selected_neurons)``.

    Raises:
        ValueError: If a sample produces no features (wrong pooling_type or hooks missing).
    """
    if selected_layers is None:
        selected_layers = sorted(int(k) for k in layer_weights.keys())

    aggregated: list[np.ndarray] = []
    for sample_rep in representations:
        parts: list[np.ndarray] = []
        for layer_idx in selected_layers:
            key = f"layer{layer_idx}_{pooling_type}"
            if key not in selected_neurons_dict:
                continue
            if layer_idx not in sample_rep or pooling_type not in sample_rep[layer_idx]:
                continue
            layer_vec = sample_rep[layer_idx][pooling_type]
            neuron_indices = selected_neurons_dict[key]
            selected = layer_vec[neuron_indices]
            # Accept both int and str layer_weights keys.
            weight = float(
                layer_weights[layer_idx]
                if layer_idx in layer_weights
                else layer_weights[str(layer_idx)]
            )
            parts.append(selected * weight)
        if not parts:
            raise ValueError(
                f"No features aggregated for a sample with pooling_type={pooling_type!r}. "
                "Check that hooks were registered and the pooling_type matches what was used "
                "during training."
            )
        aggregated.append(np.concatenate(parts))
    return np.asarray(aggregated, dtype=np.float32), list(selected_layers)
