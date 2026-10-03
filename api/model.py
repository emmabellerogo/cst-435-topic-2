"""PyTorch MLP for Adult Income binary classification.

Kept separate from the training CLI so the FastAPI service can later load a
checkpoint without importing any training code. Every architecture choice
(hidden sizes, activation, dropout) comes from the YAML config, and is stored in
the checkpoint so the exact network can be rebuilt at load time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}
MIN_HIDDEN_LAYERS = 2

PathLike = Union[str, Path]


def make_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[key]()


class MLP(nn.Module):
    """[Linear -> activation -> Dropout] x N hidden layers -> Linear(1).

    Outputs one logit per row (shape ``(n,)``); apply a sigmoid for P(>50K).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_sizes = [int(h) for h in hidden_sizes]
        if len(hidden_sizes) < MIN_HIDDEN_LAYERS:
            raise ValueError(f"need at least {MIN_HIDDEN_LAYERS} hidden layers, got {hidden_sizes}")
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError(f"hidden sizes must be positive, got {hidden_sizes}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), make_activation(activation), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        self.in_dim = int(in_dim)
        self.hidden_sizes = hidden_sizes
        self.activation = str(activation).lower()
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict_proba(model: MLP, X: np.ndarray) -> np.ndarray:
    """P(income > 50K) for each row of a preprocessed float32 matrix."""
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()


def save_checkpoint(path: PathLike, model: MLP, **metadata) -> Path:
    """Save weights + everything needed to rebuild the network.

    ``metadata`` must be plain Python types (dict/list/str/int/float/None) so
    the file loads with ``weights_only=True``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": model.in_dim,
            "hidden_sizes": model.hidden_sizes,
            "activation": model.activation,
            "dropout": model.dropout,
            **metadata,
        },
        path,
    )
    return path


def load_checkpoint(path: PathLike) -> Tuple[MLP, Dict]:
    """Rebuild the MLP from a checkpoint; returns (model in eval mode, raw checkpoint)."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = MLP(ckpt["in_dim"], ckpt["hidden_sizes"], ckpt["activation"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
