"""
Context-Gated Fusion (CGF) — Phase 3.

Implements exactly the module specified in the frozen architecture spec,
Part 2.8:

    Input:  C3 (B,128,52,52), C4 (B,256,26,26), C5 (B,512,13,13)
    Output: P3 (B,128,52,52), P4 (B,128,26,26), P5 (B,128,13,13)

    - lateral 1x1 convs unify channel dims (128/256/512 -> 128) for C3/C4/C5
    - a shared context FC + gate FC condition the top-down fusion weight
      on a pooled descriptor of the higher-level feature map
    - top-down fusion: P4 += gate * upsample(P5); P3 += gate * upsample(P4)
    - a 3x3 smoothing conv is applied to each fused level

This is a direct, unmodified port of the module design already measured
against the frozen spec (566,176 params, 1.202 GFLOPs at 416x416 input).
No architectural changes were made here relative to that specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGF(nn.Module):
    def __init__(self, channels: int = 128, ctx_dim: int = 32):
        super().__init__()
        # Lateral 1x1 convs: unify C3/C4/C5 channel counts to `channels`.
        self.lat3 = nn.Conv2d(128, channels, 1)
        self.lat4 = nn.Conv2d(256, channels, 1)
        self.lat5 = nn.Conv2d(512, channels, 1)

        # Shared context FC + gate FC: a pooled descriptor of the
        # higher-resolution... (higher-level, coarser) feature map is
        # projected to a small context vector, then expanded back to a
        # per-channel gate in [0,1] controlling how much of the upsampled
        # higher-level feature is fused into the level below it.
        self.ctx_fc = nn.Linear(channels, ctx_dim)
        self.gate_fc = nn.Linear(ctx_dim, channels)

        # Smoothing convs applied after fusion, one per output level.
        self.smooth3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3)

        # Top-down fusion, P5 -> P4
        ctx5 = self.ctx_fc(p5.mean(dim=(2, 3)))
        g5 = torch.sigmoid(self.gate_fc(ctx5)).unsqueeze(-1).unsqueeze(-1)
        p4 = p4 + g5 * F.interpolate(p5, size=p4.shape[-2:], mode="nearest")

        # Top-down fusion, P4 -> P3
        ctx4 = self.ctx_fc(p4.mean(dim=(2, 3)))
        g4 = torch.sigmoid(self.gate_fc(ctx4)).unsqueeze(-1).unsqueeze(-1)
        p3 = p3 + g4 * F.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return p3, p4, p5


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    cgf = CGF(channels=128)
    c3 = torch.randn(1, 128, 52, 52)
    c4 = torch.randn(1, 256, 26, 26)
    c5 = torch.randn(1, 512, 13, 13)
    p3, p4, p5 = cgf(c3, c4, c5)
    print("P3:", tuple(p3.shape))
    print("P4:", tuple(p4.shape))
    print("P5:", tuple(p5.shape))
    print("Total params:", count_parameters(cgf))
