"""A small U-Net on the 24x48 grid: longitude wraps, latitude repeats its edge row."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def wrap(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    return F.pad(F.pad(x, (k, k, 0, 0), mode="circular"), (0, 0, k, k), mode="replicate")


class Block(nn.Module):
    def __init__(self, inputs: int, width: int):
        super().__init__()
        self.first = nn.Conv2d(inputs, width, 3)
        self.second = nn.Conv2d(width, width, 3)
        self.norm1, self.norm2 = nn.GroupNorm(1, width), nn.GroupNorm(1, width)
        self.skip = nn.Conv2d(inputs, width, 1) if inputs != width else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.first(wrap(x))))
        return F.gelu(self.norm2(self.second(wrap(y))) + self.skip(x))


class UNet(nn.Module):
    """Three halvings reach a 3x6 grid, so a cell can read the whole planet."""

    def __init__(self, inputs: int, width: int = 32, place: int = 4):
        super().__init__()
        # Learned per-cell channels: the ridge has none of this, and convolutions alone cannot tell Peru from Borneo.
        self.place = nn.Parameter(torch.zeros(1, place, 24, 48)) if place else None
        sizes = [width, width * 2, width * 4, width * 4]
        self.down = nn.ModuleList([Block(inputs + place, sizes[0]), Block(sizes[0], sizes[1]), Block(sizes[1], sizes[2]), Block(sizes[2], sizes[3])])
        self.up = nn.ModuleList([Block(sizes[3] + sizes[2], sizes[2]), Block(sizes[2] + sizes[1], sizes[1]), Block(sizes[1] + sizes[0], sizes[0])])
        self.out = nn.Conv2d(sizes[0], 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.place is not None:
            x = torch.cat([x, self.place.expand(len(x), -1, -1, -1)], 1)
        kept = []
        for k, block in enumerate(self.down):
            x = block(x if k == 0 else F.avg_pool2d(kept[-1], 2))
            kept.append(x)
        for k, block in enumerate(self.up):
            x = block(torch.cat([F.interpolate(x, scale_factor=2), kept[-2 - k]], 1))
        return self.out(x)[:, 0]
