"""Custom detector modules for US17, aimed at the elongated crack classes.

StripAttention applies orientation-aware attention built from asymmetric
kernels. DySample replaces nearest-neighbour upsampling in the neck with a
content-aware variant. Both preserve channel count.

Ultralytics width-scales the first argument only for modules it knows about, so
a YAML written for the 's' scale must carry the already-scaled channel count.
Each module checks this on the first forward pass rather than failing later
inside a convolution. See rdd.models.register.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class StripAttention(nn.Module):
    """Attention over long thin structures, using 1xk and kx1 kernels.

    Orientation is definitional in this taxonomy: a longitudinal crack and a
    transverse crack are different classes, so an isotropic operator blurs the
    distinction. Each orientation gets its own depthwise branch, and the
    concatenation is projected back to C channels and used as a multiplicative
    gate. At initialisation the gate sits near 0.5, so the block attenuates
    rather than destroys the signal.

    Costs roughly 2*k*C + 2*C^2 parameters against k^2*C^2 for a dense kernel
    of the same reach.

    Args:
        c:  input and output channels. Must match the incoming tensor.
        k:  strip length. 7 covers 7 cells, which at stride 8 is 56 input
            pixels, about the length of crack fragment the dataset labels.
    """

    def __init__(self, c: int, k: int = 7):
        super().__init__()
        self.c = int(c)
        self.k = int(k)
        pad = self.k // 2
        self.horizontal = nn.Conv2d(self.c, self.c, (1, self.k),
                                    padding=(0, pad), groups=self.c, bias=False)
        self.vertical = nn.Conv2d(self.c, self.c, (self.k, 1),
                                  padding=(pad, 0), groups=self.c, bias=False)
        self.project = nn.Conv2d(2 * self.c, self.c, 1, bias=False)
        self.norm = nn.BatchNorm2d(self.c)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c:
            raise RuntimeError(
                f"StripAttention was built for {self.c} channels but received "
                f"{x.shape[1]}. The channel count in the model YAML must "
                f"already be the scaled one."
            )
        a = torch.cat((self.horizontal(x), self.vertical(x)), dim=1)
        return x * self.gate(self.norm(self.project(a)))


class DySample(nn.Module):
    """Content-aware upsampling, in place of nearest-neighbour.

    Nearest-neighbour copies each cell into a 2x2 block regardless of content,
    which turns a one-cell-wide crack response into a blocky one. This predicts
    a sampling offset per output position with a 1x1 convolution and gathers by
    bilinear interpolation instead. Follows DySample (Liu et al., ICCV 2023) in
    its single-group form.

    The offset convolution is zero-initialised, so at step zero the block is
    ordinary bilinear upsampling and a transferred checkpoint starts from
    something sensible.
    """

    def __init__(self, c: int, scale: int = 2):
        super().__init__()
        self.c = int(c)
        self.scale = int(scale)
        self.offset = nn.Conv2d(self.c, 2 * self.scale * self.scale, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c:
            raise RuntimeError(
                f"DySample was built for {self.c} channels but received "
                f"{x.shape[1]}. The model YAML must carry the already-scaled "
                f"channel count."
            )
        b, _, h, w = x.shape
        s = self.scale
        oh, ow = h * s, w * s

        # Offsets live at output resolution: predict s^2 of them per input
        # cell, then pixel-shuffle into place.
        off = F.pixel_shuffle(self.offset(x), s)              # b, 2, oh, ow

        # Base grid in normalised [-1, 1] coordinates, matching what
        # align_corners=False means for an ordinary bilinear resize.
        ys = (torch.arange(oh, device=x.device, dtype=x.dtype) + 0.5) * (2.0 / oh) - 1.0
        xs = (torch.arange(ow, device=x.device, dtype=x.dtype) + 0.5) * (2.0 / ow) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)   # 1, 2, oh, ow

        # Offsets are predicted in input cells; convert to normalised units.
        scale_xy = torch.tensor([2.0 / w, 2.0 / h], device=x.device,
                                dtype=x.dtype).view(1, 2, 1, 1)
        grid = (base + off * scale_xy).permute(0, 2, 3, 1)         # b, oh, ow, 2

        return F.grid_sample(x, grid.expand(b, oh, ow, 2),
                             mode="bilinear", padding_mode="border",
                             align_corners=False)
