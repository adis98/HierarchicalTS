from torch import nn, Tensor
import torch
import math
import numpy as np
from functools import reduce


def num_groups_heuristic(n: int) -> int:
    """Find the divisor of n closest to sqrt(n)"""
    step = 2 if n % 2 else 1
    factors = np.array(reduce(list.__add__, ([i, n // i] for i in range(1, int(np.sqrt(n)) + 1, step) if n % i == 0)))
    best_idx = np.argmin(np.abs(factors - np.sqrt(n)))
    return factors[best_idx]


class TransformerDiffusionBackbone(nn.Module):
    def __init__(
            self,
            in_dim: int,
            t_dim: int,
            h_dim: int = 128,
            n_layers: int = 8,
            dropout: float = 0.1,
            n_heads: int = 8,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads

        self.fc_tgt = nn.Linear(in_dim, h_dim)
        self.fc_mem = nn.Linear(t_dim, h_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=h_dim, nhead=n_heads, dim_feedforward=h_dim * 2, dropout=dropout, batch_first=True
            ),
            num_layers=n_layers,
        )
        self.fc_out = nn.Linear(h_dim, in_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.fc_tgt(x)
        t = self.fc_mem(t)
        output = self.transformer.forward(x, t)
        output = self.fc_out(output)
        return output


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, in_dim: int, t_dim: int) -> None:
        super().__init__()

        self.norm = nn.GroupNorm(num_groups_heuristic(in_dim), in_dim, affine=False, eps=1e-6)

        self.timestep = nn.Linear(t_dim, in_dim * 2)
        self.timestep.bias.data[:in_dim] = 1
        self.timestep.bias.data[in_dim:] = 0

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = self.timestep(t).transpose(1, 2)

        gamma, beta = torch.chunk(t, chunks=2, dim=1)
        if x.ndim == 2:
            gamma, beta = gamma.squeeze(2), beta.squeeze(2)

        x = self.norm(x)

        x = gamma * x + beta

        return x


class LinearDiffusionBackbone(nn.Module):
    def __init__(
            self,
            seq_len: int,
            in_dim: int,
            t_dim: int,
            h_dim: int = 128,
            n_layers: int = 8,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = dropout

        self.map_in = nn.Linear(in_dim, h_dim)
        self.channel_mixers = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(n_layers)])
        self.column_mixers = nn.ModuleList([nn.Linear(seq_len, seq_len) for _ in range(n_layers)])
        self.norms = nn.ModuleList([AdaptiveGroupNorm(h_dim, t_dim) for _ in range(n_layers)])
        self.map_out = nn.Linear(h_dim, in_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.map_in(x)
        for channel_mixer, column_mixer, norm in zip(self.channel_mixers, self.column_mixers, self.norms):
            x = channel_mixer(x)
            x = dropout(x, p=self.dropout, train=self.training)
            x = x.transpose(1, 2)
            x = column_mixer(x)
            x = dropout(x, p=self.dropout, train=self.training)
            x = gelu(x)
            x = norm(x, t)
            x = x.transpose(1, 2)
        x = self.map_out(x)
        return x


class AdaBiasBilinear(nn.Module):
    def __init__(self, in_dim: int, t_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.t_dim = t_dim

        self.weight = nn.Parameter(torch.empty((out_dim, in_dim, t_dim)))
        self.bias = nn.Parameter(torch.empty(out_dim, t_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.einsum("bi,oit,bt->bo", x, self.weight, t) + torch.einsum("ot,bt->bo", self.bias, t)


class BiLinearDiffusionBackbone(nn.Module):
    def __init__(
            self, in_dim: int, t_dim: int, h_dim: int = 128, n_layers: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.dropout = dropout

        self.map_in = AdaBiasBilinear(in_dim, t_dim, h_dim)
        self.bilinears = nn.ModuleList([AdaBiasBilinear(h_dim, t_dim, h_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([AdaptiveGroupNorm(h_dim, t_dim) for _ in range(n_layers)])
        self.map_out = AdaBiasBilinear(h_dim, t_dim, in_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x, t = x.squeeze(1), t.squeeze(1)
        x = self.map_in(x, t)
        for map, norm in zip(self.bilinears, self.norms):
            x = map(x, t)
            x = gelu(x)
            x = norm(x, t.unsqueeze(1))
            x = dropout(x, p=self.dropout, train=self.training)
        x = self.map_out(x, t)
        x = x.unsqueeze(1)
        return x
