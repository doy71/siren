from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveMLPClassifier(nn.Module):
    """Same role as CSSLab/SIREN's AdaptiveMLPClassifier.

    Kept in an importable module so pickled best_model.pkl can be loaded by
    train/eval scripts without the brittle __main__ pickle path problem.
    """

    def __init__(self, input_dim: int, layer_dims: list[int], dropout_rates: list[float], num_classes: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim, dropout in zip(layer_dims, dropout_rates):
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        final_linear = nn.Linear(prev_dim, num_classes)
        # The final layer outputs raw logits (no activation), so use "linear"
        # nonlinearity instead of "relu" to keep the fan-in scaling correct.
        nn.init.kaiming_normal_(final_linear.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(final_linear.bias)
        layers.append(final_linear)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
