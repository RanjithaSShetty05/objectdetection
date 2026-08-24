"""
Pretrained backbones for FieldNet — drop-in replacements for EduNet.

WHY THIS IS THE BIGGEST SINGLE LEVER
====================================
EduNet is trained from random initialization on 2,894 images. That is the
hardest possible setting for a detector: the backbone has to learn generic
visual features (edges, textures, object parts) from scratch, from a dataset
far too small to support it, while simultaneously learning to localize. Modern
detectors essentially never do this — they start from ImageNet features.

CHANNEL COMPATIBILITY (this is the lucky part)
=============================================
`CGF` expects exactly:

    C3 (B, 128, 52, 52)   stride 8
    C4 (B, 256, 26, 26)   stride 16
    C5 (B, 512, 13, 13)   stride 32

torchvision's ResNet-18 and ResNet-34 emit precisely those channel counts at
precisely those strides for a 416x416 input:

    conv1 (s2) -> 208 | maxpool (s2) -> 104 | layer1 -> 104 (stride 4)
    layer2 -> 52  (stride 8,  128 ch)
    layer3 -> 26  (stride 16, 256 ch)
    layer4 -> 13  (stride 32, 512 ch)

So ResNet-18/34 are EXACT drop-in replacements and the neck needs no change.
ResNet-50 emits 512/1024/2048 instead, so a 1x1 projection is added for it.

TWO DETECTION-FINETUNING CONVENTIONS ARE APPLIED
===============================================
  * FrozenBatchNorm. This project trains at small batch sizes. Real BatchNorm
    computes unstable statistics from tiny batches and will actively degrade a
    pretrained backbone. Every detection framework (Detectron2, torchvision's
    own detection models, mmdet) freezes BN affine+statistics when finetuning.
    Default: on.

  * Freezing the stem and layer1. The earliest layers encode generic
    edge/colour filters that a 2,894-image dataset cannot improve on, and
    freezing them saves memory and prevents early-training corruption.
    Default: freeze through layer1.

IMPORTANT: PRETRAINED BACKBONES REQUIRE IMAGENET INPUT NORMALIZATION.
The existing dataset divides by 255 and stops. Feeding [0,1] images to an
ImageNet-pretrained network without mean/std normalization measurably hurts.
`data/dataset_v3.py` handles this; if you write your own loader, normalize with
mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225).
"""

from typing import Tuple, List

import torch
import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_SUPPORTED = ("resnet18", "resnet34", "resnet50")

# (C3, C4, C5) channel counts emitted by each torchvision ResNet variant.
_NATIVE_CHANNELS = {
    "resnet18": (128, 256, 512),
    "resnet34": (128, 256, 512),
    "resnet50": (512, 1024, 2048),
}


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d with frozen affine parameters AND frozen running statistics.

    Behaves identically in train() and eval(). Deliberately keeps the same
    buffer names as nn.BatchNorm2d (weight/bias/running_mean/running_var) so a
    pretrained state_dict loads into it without remapping.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        shift = self.bias - self.running_mean * scale
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}, frozen=True"


def _convert_bn_to_frozen(module: nn.Module) -> nn.Module:
    """Recursively replace every nn.BatchNorm2d with FrozenBatchNorm2d."""
    if isinstance(module, nn.BatchNorm2d):
        frozen = FrozenBatchNorm2d(module.num_features, eps=module.eps)
        with torch.no_grad():
            frozen.weight.copy_(module.weight)
            frozen.bias.copy_(module.bias)
            frozen.running_mean.copy_(module.running_mean)
            frozen.running_var.copy_(module.running_var)
        return frozen

    for name, child in module.named_children():
        setattr(module, name, _convert_bn_to_frozen(child))
    return module


class PretrainedResNetBackbone(nn.Module):
    """
    ImageNet-pretrained ResNet exposing (C3, C4, C5) at strides (8, 16, 32),
    projected to (128, 256, 512) channels so `CGF` consumes it unchanged.

    Args:
        arch: one of "resnet18", "resnet34", "resnet50".
        pretrained: load ImageNet weights. Requires network access on FIRST
            run only (torchvision caches to ~/.cache/torch/hub). If you are
            offline, set False — but then you lose the entire point of this
            module and should just use EduNet.
        frozen_bn: replace BatchNorm with FrozenBatchNorm2d. Strongly
            recommended at batch sizes below ~16.
        freeze_until: "none" | "stem" | "layer1" | "layer2". Parameters up to
            and including this stage get requires_grad=False.
        out_channels: target (C3, C4, C5) channels. Projection convs are added
            only where the native count differs.
    """

    def __init__(
        self,
        arch: str = "resnet34",
        pretrained: bool = True,
        frozen_bn: bool = True,
        freeze_until: str = "layer1",
        out_channels: Tuple[int, int, int] = (128, 256, 512),
    ):
        super().__init__()

        if arch not in _SUPPORTED:
            raise ValueError(
                f"arch must be one of {_SUPPORTED}, got {arch!r}"
            )

        # Imported here (not at module scope) so that this file can be parsed
        # and syntax-checked in an environment without torchvision.
        import torchvision

        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, arch)(weights=weights)

        self.arch = arch
        self.pretrained = pretrained

        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        if frozen_bn:
            _convert_bn_to_frozen(self)

        native = _NATIVE_CHANNELS[arch]
        self.out_channels = tuple(out_channels)

        # Identity where channels already match (ResNet-18/34), 1x1 projection
        # only where they do not (ResNet-50).
        self.proj3 = self._maybe_proj(native[0], out_channels[0])
        self.proj4 = self._maybe_proj(native[1], out_channels[1])
        self.proj5 = self._maybe_proj(native[2], out_channels[2])

        self._apply_freeze(freeze_until)

    @staticmethod
    def _maybe_proj(cin: int, cout: int) -> nn.Module:
        if cin == cout:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.GroupNorm(32 if cout % 32 == 0 else 1, cout),
        )

    def _apply_freeze(self, freeze_until: str) -> None:
        order = ["stem", "layer1", "layer2"]
        if freeze_until == "none":
            return
        if freeze_until not in order:
            raise ValueError(
                f"freeze_until must be 'none' or one of {order}, "
                f"got {freeze_until!r}"
            )

        to_freeze: List[str] = order[: order.index(freeze_until) + 1]
        for name in to_freeze:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self._frozen_stages = to_freeze

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)      # stride 4
        x = self.layer1(x)    # stride 4
        c3 = self.layer2(x)   # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32
        return self.proj3(c3), self.proj4(c4), self.proj5(c5)


def build_backbone(name: str = "resnet34", **kwargs) -> nn.Module:
    """
    Factory so training scripts can switch backbone with a single string.

        "edunet"  -> the original from-scratch EduNet (unchanged)
        "resnet18" / "resnet34" / "resnet50" -> ImageNet-pretrained

    EduNet ignores the pretrained/freezing kwargs, since none apply to it.
    """
    if name == "edunet":
        from models.backbone import EduNet
        return EduNet()
    return PretrainedResNetBackbone(arch=name, **kwargs)


if __name__ == "__main__":
    # Shape contract check. Confirms drop-in compatibility with CGF.
    for arch in ("resnet18", "resnet34"):
        net = PretrainedResNetBackbone(arch=arch, pretrained=False)
        c3, c4, c5 = net(torch.randn(1, 3, 416, 416))
        print(f"{arch}: C3 {tuple(c3.shape)} C4 {tuple(c4.shape)} C5 {tuple(c5.shape)}")
        assert tuple(c3.shape)[1:] == (128, 52, 52), c3.shape
        assert tuple(c4.shape)[1:] == (256, 26, 26), c4.shape
        assert tuple(c5.shape)[1:] == (512, 13, 13), c5.shape

    net = PretrainedResNetBackbone(arch="resnet50", pretrained=False)
    c3, c4, c5 = net(torch.randn(1, 3, 416, 416))
    print(f"resnet50: C3 {tuple(c3.shape)} C4 {tuple(c4.shape)} C5 {tuple(c5.shape)}")
    assert tuple(c3.shape)[1:] == (128, 52, 52), c3.shape

    print("OK: all backbones match the CGF input contract.")
