"""
FieldHeadV3 — the fixed detection head.

WHY THIS FILE EXISTS
====================
The original `models/head.py` has three properties that, together, are the
primary cause of "the model only ever predicts a handful of classes":

  1. `self.phi = nn.Conv2d(channels, num_classes, 3, padding=1)` uses PyTorch's
     DEFAULT bias init, so at step 0 `sigmoid(phi) ~= 0.5` on ALL class
     channels. Meanwhile ~97% of the (position, class) elements are background.
     The cheapest way for the optimizer to reduce the dense BCE is to drive
     every channel toward 0, and low-mass classes never climb back out.
     This is exactly the failure mode RetinaNet's prior-probability bias init
     was introduced to prevent. It was never applied here.

     FIX: initialize the phi conv bias to  -log((1 - pi) / pi)  with pi = 0.01,
     so the model STARTS at p ~= 0.01 (already near the background prior). The
     background loss is small from step 0, so positive gradients dominate.

  2. `phi = torch.sigmoid(self.phi(t))` returns PROBABILITIES, forcing the loss
     to use `F.binary_cross_entropy` on already-sigmoided values, then clamp to
     [eps, 1-eps]. That clamp creates exactly-zero-gradient regions and is
     numerically worse than the fused logit form.

     FIX: return LOGITS. The loss uses the fused, numerically stable
     `binary_cross_entropy_with_logits`. A `phi_prob()` helper is provided for
     inference/visualisation.

  3. One shared head serves strides 8/16/32, but the box targets are
     stride-normalized (`ew = half_width / stride`). The SAME object therefore
     requires log-extent outputs that differ by log(4) between stride 8 and
     stride 32. A single shared conv cannot emit all three at once.

     FIX: a per-level learnable affine (scale, bias) on the log-extent
     channels, so each level can absorb its own offset. This is the same
     motivation as FCOS's per-level `Scale` module.

TWO FURTHER CHANGES, both standard and both flag-controlled:

  * GroupNorm in the head trunk. The original trunk is bare Conv + AdaptiveAct
    with no normalization, and this project trains at batch size 2-16 where
    BatchNorm would be unreliable. RetinaNet/FCOS both use GN in the head for
    precisely this reason.

  * Optional separate trunks for the classification (phi) and regression
    (box/logvar) branches. Sharing one trunk forces the same features to serve
    "what class is here" and "how big is it", which are known to prefer
    different representations.

BOX PARAMETERIZATION CHANGE (important — read before comparing runs)
===================================================================
The original pipeline predicts a raw value, applies `softplus`, then takes
`log` in the loss. That has two problems: softplus saturates toward 0 (killing
gradients for small objects), and it created a real bug in this repo where two
evaluators decoded with raw linear extents while training used softplus — which
is why the headline "F1 0.3200" is not comparable to the other numbers.

V3 predicts log-extent DIRECTLY:

    channels = (dx, dy, log_ew, log_eh)
    half_width_pixels  = exp(log_ew) * stride
    half_height_pixels = exp(log_eh) * stride

so the box loss is a plain SmoothL1 in log space on the raw output, and there
is exactly ONE decode path. At init the log-extent output is ~0, i.e.
half-extent ~= one stride, which is a sane starting guess.

Everything here is additive: `models/head.py` is untouched, so every existing
checkpoint and script keeps working.
"""

from typing import Tuple
import math

import torch
import torch.nn as nn

from models.backbone import AdaptiveAct


def prior_bias(prior_prob: float) -> float:
    """
    RetinaNet-style focal-loss bias init.

        b = -log((1 - p) / p)      so that      sigmoid(b) == p

    For p = 0.01 this is -4.59512. Returning a plain float (not a tensor) keeps
    it trivially unit-testable without a GPU.
    """
    if not (0.0 < prior_prob < 1.0):
        raise ValueError(
            f"prior_prob must be in (0,1), got {prior_prob}"
        )
    return -math.log((1.0 - prior_prob) / prior_prob)


def _norm_layer(kind: str, channels: int) -> nn.Module:
    if kind == "gn":
        # 32 groups is the RetinaNet/FCOS convention; fall back if channels
        # are not divisible by 32 so this never silently misconfigures.
        groups = 32 if channels % 32 == 0 else math.gcd(32, channels)
        return nn.GroupNorm(groups, channels)
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind!r}")


def _make_trunk(channels: int, depth: int, norm: str) -> nn.Sequential:
    layers = []
    for _ in range(depth):
        layers.append(
            nn.Conv2d(channels, channels, 3, padding=1, bias=(norm == "none"))
        )
        layers.append(_norm_layer(norm, channels))
        layers.append(AdaptiveAct(channels))
    return nn.Sequential(*layers)


class FieldHeadV3(nn.Module):
    """
    Shared head applied at each pyramid level.

    forward(x, level) -> (phi_logits, box, logvar)

        phi_logits : (B, num_classes, H, W)   RAW LOGITS (no sigmoid)
        box        : (B, 4, H, W)             (dx, dy, log_ew, log_eh)
        logvar     : (B, 4, H, W)             heteroscedastic log-variance

    `level` selects the per-level log-extent affine: 0 -> stride 8,
    1 -> stride 16, 2 -> stride 32. It MUST be passed correctly or the
    box extents will be systematically wrong by a factor of ~4 per level.
    """

    def __init__(
        self,
        channels: int = 128,
        num_classes: int = 16,
        num_levels: int = 3,
        trunk_depth: int = 3,
        norm: str = "gn",
        separate_trunks: bool = True,
        prior_prob: float = 0.01,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels
        self.separate_trunks = separate_trunks
        self.prior_prob = prior_prob

        self.cls_trunk = _make_trunk(channels, trunk_depth, norm)
        if separate_trunks:
            self.reg_trunk = _make_trunk(channels, trunk_depth, norm)
        else:
            self.reg_trunk = None

        self.phi = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.box = nn.Conv2d(channels, 4, 3, padding=1)
        self.logvar = nn.Conv2d(channels, 4, 3, padding=1)

        # Per-level affine on the two log-extent channels only. Initialized to
        # identity (scale 1, bias 0) so an untrained V3 head behaves exactly
        # like a plain shared head; the levels differentiate during training.
        self.ext_scale = nn.Parameter(torch.ones(num_levels, 2))
        self.ext_bias = nn.Parameter(torch.zeros(num_levels, 2))

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard RetinaNet head init: normal(0, 0.01) on every conv, zero
        # bias, EXCEPT the phi bias which gets the background prior.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # THE key line this project was missing.
        nn.init.constant_(self.phi.bias, prior_bias(self.prior_prob))

        # Start logvar at 0 => variance 1, a neutral, well-conditioned prior
        # that avoids the clamp boundaries at +/-6.
        nn.init.constant_(self.logvar.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        level: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not (0 <= level < self.num_levels):
            raise ValueError(
                f"level must be in [0,{self.num_levels - 1}], got {level}"
            )

        cls_feat = self.cls_trunk(x)
        reg_feat = self.reg_trunk(x) if self.reg_trunk is not None else cls_feat

        phi_logits = self.phi(cls_feat)
        box = self.box(reg_feat)
        logvar = self.logvar(reg_feat)

        # Apply the per-level affine to channels 2,3 (log_ew, log_eh) only.
        scale = self.ext_scale[level].view(1, 2, 1, 1)
        bias = self.ext_bias[level].view(1, 2, 1, 1)

        dxdy = box[:, :2]
        ext = box[:, 2:] * scale + bias
        box = torch.cat([dxdy, ext], dim=1)

        return phi_logits, box, logvar

    @staticmethod
    def phi_prob(phi_logits: torch.Tensor) -> torch.Tensor:
        """Inference/visualisation helper: logits -> probabilities in [0,1]."""
        return torch.sigmoid(phi_logits)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    # Self-check that needs no dataset and no GPU.
    torch.manual_seed(0)

    head = FieldHeadV3(channels=128, num_classes=16)

    expected = prior_bias(0.01)
    actual = float(head.phi.bias[0].detach())
    print(f"prior_bias(0.01)      = {expected:.6f}   (expect -4.595120)")
    print(f"head.phi.bias[0]      = {actual:.6f}")
    assert abs(actual - expected) < 1e-5, "phi bias init did not take effect"

    feats = {0: (52, 52), 1: (26, 26), 2: (13, 13)}
    for level, (h, w) in feats.items():
        x = torch.randn(2, 128, h, w)
        with torch.no_grad():
            phi_logits, box, logvar = head(x, level=level)
            # .detach() via no_grad: float() on a grad-tracking tensor raises a
            # UserWarning, and a self-test that prints warnings trains people to
            # ignore warnings.
            mean_p = float(FieldHeadV3.phi_prob(phi_logits).mean())
        print(
            f"level {level}: phi_logits {tuple(phi_logits.shape)} "
            f"box {tuple(box.shape)} logvar {tuple(logvar.shape)} "
            f"| mean phi prob = {mean_p:.4f} (expect ~0.01)"
        )
        assert mean_p < 0.05, (
            "initial phi probability should sit near the 0.01 background prior"
        )

    print("Head params:", count_parameters(head))
    print("OK: FieldHeadV3 self-check passed.")
