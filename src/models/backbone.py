"""
EduNet backbone — Phase 2.

Implements exactly the modules specified in the frozen architecture spec,
Part 2 ("Complete architecture, layer by layer") and Part 2.6/2.7
(AdaptiveAct, DualPathBlock):

    Stem: Conv3x3 stride2 (3->32) + BN + AdaptiveAct
    C2:   2x DualPathBlock, 32->64,  stride 2 then 1   (208 -> 104)
    C3:   3x DualPathBlock, 64->128, stride 2 then 1,1 (104 -> 52)
    C4:   3x DualPathBlock, 128->256,stride 2 then 1,1 (52  -> 26)
    C5:   2x DualPathBlock, 256->512,stride 2 then 1   (26  -> 13)

This is a direct, unmodified port of the module design that was already
measured against the frozen spec (Part 2 / Part 2.13 totals:
6,667,680 params, 7.464 GFLOPs backbone-only at 416x416). No architectural
changes were made here relative to that specification.
"""

from typing import Tuple
import torch
import torch.nn as nn


class AdaptiveAct(nn.Module):
    """
    Per-channel learnable blend of two smooth nonlinearities (spec Part 2.6).

        out = alpha * [x * sigmoid(x)] + (1 - alpha) * [x * tanh(softplus(x))]

    alpha is a per-channel learnable parameter, initialized at 0.5, per the
    frozen spec's training-init section (Part 11).

    Engineering note (per spec Part 15): this is presented honestly as an
    ACON-family adaptive activation, not claimed as an unprecedented
    mechanism.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = x * torch.sigmoid(x)
        smooth = x * torch.tanh(nn.functional.softplus(x))
        return self.alpha * swish + (1 - self.alpha) * smooth


class DualPathBlock(nn.Module):
    """
    Dual-path adaptive kernel block (spec Part 2.7 / earlier module design).

        Path A (local):   BN(Conv3x3(x))
        Path B (context): BN(Pointwise(DepthwiseConv5x5, dilation=2)(x))
        Gate:              g = sigmoid(FC2(ReLU(FC1(GlobalAvgPool(y_A)))))
        Fusion:            y = g * y_A + (1-g) * y_B
        Output:            AdaptiveAct(y + Projection(x))   (residual)

    Engineering note (per spec Part 15): this is presented honestly as an
    SKNet-family selective-kernel block, not claimed as unprecedented.
    """

    def __init__(self, cin: int, cout: int, stride: int = 1, reduction: int = 4):
        super().__init__()
        self.pathA = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.pathB = nn.Sequential(
            nn.Conv2d(cin, cin, 5, stride, padding=4, dilation=2, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cout),
        )
        gate_hidden = max(cout // reduction, 8)
        self.gate_fc1 = nn.Linear(cout, gate_hidden)
        self.gate_fc2 = nn.Linear(gate_hidden, cout)

        self.proj = None
        if cin != cout or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, 0, bias=False),
                nn.BatchNorm2d(cout),
            )
        self.act = AdaptiveAct(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yA = self.pathA(x)
        yB = self.pathB(x)
        pooled = yA.mean(dim=(2, 3))
        g = torch.sigmoid(self.gate_fc2(torch.relu(self.gate_fc1(pooled))))
        g = g.unsqueeze(-1).unsqueeze(-1)
        y = g * yA + (1 - g) * yB
        residual = self.proj(x) if self.proj is not None else x
        return self.act(y + residual)


class Stem(nn.Module):
    """Conv3x3 stride2 (3->32) + BN + AdaptiveAct. 416 -> 208 (spec Part 2.1)."""

    def __init__(self, cin: int = 3, cout: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = AdaptiveAct(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class EduNet(nn.Module):
    """
    Full EduNet backbone (spec Part 2.1-2.5).

    Input:  (B, 3, 416, 416)
    Output: C3 (B,128,52,52), C4 (B,256,26,26), C5 (B,512,13,13)

    Only C3/C4/C5 are returned since those are the levels CGF (Phase 3)
    consumes, per the frozen spec. C2 is computed internally (required as
    input to C3) but not returned, matching the spec's neck input list.
    """

    def __init__(self):
        super().__init__()
        self.stem = Stem(3, 32)  # 416 -> 208

        self.c2 = nn.Sequential(
            DualPathBlock(32, 64, stride=2),  # 208 -> 104
            DualPathBlock(64, 64, stride=1),
        )
        self.c3 = nn.Sequential(
            DualPathBlock(64, 128, stride=2),  # 104 -> 52
            DualPathBlock(128, 128, stride=1),
            DualPathBlock(128, 128, stride=1),
        )
        self.c4 = nn.Sequential(
            DualPathBlock(128, 256, stride=2),  # 52 -> 26
            DualPathBlock(256, 256, stride=1),
            DualPathBlock(256, 256, stride=1),
        )
        self.c5 = nn.Sequential(
            DualPathBlock(256, 512, stride=2),  # 26 -> 13
            DualPathBlock(512, 512, stride=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c2 = self.c2(x)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        return c3, c4, c5


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    model = EduNet()
    x = torch.randn(1, 3, 416, 416)
    c3, c4, c5 = model(x)
    print("C3:", tuple(c3.shape))
    print("C4:", tuple(c4.shape))
    print("C5:", tuple(c5.shape))
    print("Total params:", count_parameters(model))
