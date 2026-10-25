"""Custom detector modules for US17.

Two modifications aimed at the project's two weakest classes, the elongated
cracks C1 and C2:

  * ``StripAttention``  -- orientation-aware attention built from asymmetric
    kernels, for long thin structures.
  * ``DySample``        -- content-aware upsampling in the neck, in place of
    fixed nearest-neighbour interpolation.

Both preserve channel count, which matters for how Ultralytics parses them —
see ``rdd.models.register``.

A note on why the channel count is an explicit argument and is asserted.
Ultralytics' ``parse_model`` width-scales the first argument only for modules
it knows about. A module injected from outside that set receives its arguments
verbatim, so a YAML written for the 's' scale must carry the already-scaled
channel count. That is easy to get wrong and would otherwise fail deep inside a
convolution with an unhelpful shape error, so each module checks its input on
the first forward pass and says exactly what it expected.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class StripAttention(nn.Module):
    """Attention over long thin structures, using 1xk and kx1 kernels.

    Standard attention has an isotropic receptive field. A longitudinal crack
    runs along the road and a transverse crack runs across it, and in this
    project those are *different classes* — which is also why rotation and
    vertical flip are banned from augmentation. An isotropic operator blurs
    precisely the distinction the taxonomy depends on.

    So the two orientations get their own depthwise branches: a 1xk kernel that
    integrates horizontally and a kx1 kernel that integrates vertically. Their
    concatenation is projected back to C channels and used as a multiplicative
    gate, which keeps the block residual-friendly — at initialisation the gate
    sits near 0.5 and the block attenuates rather than destroys the signal.

    Cheap by construction: two depthwise strip convolutions and one pointwise,
    so roughly 2*k*C + 2*C^2 parameters rather than the k^2*C^2 a dense kernel
    of the same reach would need.

    Args:
        c:  input and output channels. Must match the incoming tensor.
        k:  strip length. 7 covers 7 cells, which at stride 8 is 56 input
            pixels — about the length of crack fragment the dataset labels.
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
                f"{x.shape[1]}. Ultralytics does not width-scale arguments for "
                f"modules outside its own set, so the channel count in the "
                f"model YAML must already be the scaled one."
            )
        a = torch.cat((self.horizontal(x), self.vertical(x)), dim=1)
        return x * self.gate(self.norm(self.project(a)))


class DySample(nn.Module):
    """Content-aware upsampling, in place of nearest-neighbour.

    The stock neck upsamples with ``nn.Upsample(scale_factor=2, 'nearest')``,
    which copies each cell into a 2x2 block regardless of what is in it. On a
    feature map carrying a one-cell-wide crack, that turns a thin response into
    a blocky one and the following convolution has to clean it up.

    This learns where to sample from instead: a 1x1 convolution predicts a
    sampling offset per output position, and the output is gathered from the
    input by bilinear interpolation at those offsets. The idea is DySample
    (Liu et al., ICCV 2023), in its simplest single-group form.

    The offset convolution is zero-initialised on purpose. At step zero the
    offsets are zero, the sampling grid is the plain bilinear grid, and the
    block behaves as ordinary bilinear upsampling — so a checkpoint transferred
    into this architecture starts somewhere sensible rather than scrambled.
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
                f"{x.shape[1]}. See StripAttention for why the YAML must carry "
                f"the already-scaled channel count."
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
