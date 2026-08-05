"""Reusable PyTorch model components for NNLCI experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Configurable fully connected network used by the NNLCI workflows."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden_layers:
            layers.extend((nn.Linear(current, width), activation()))
            current = width
        layers.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ResidualBlock(nn.Module):
    """Layer-normalized GELU residual block used by the Lax ResNet branch."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.norm2 = nn.LayerNorm(width)
        self.linear2 = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.linear1(self.norm1(inputs)))
        return inputs + self.linear2(self.norm2(hidden))


class ResidualMLP(nn.Module):
    """Input projection, residual blocks, and output projection."""

    def __init__(self, input_dim: int, output_dim: int, width: int = 256, blocks: int = 6) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(blocks))
        self.output_layer = nn.Linear(width, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layer(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_layer(hidden)
