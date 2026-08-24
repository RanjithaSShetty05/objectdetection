"""
FieldNetV3 — the trainable model wrapper for the fixed stack.

    image -> backbone (EduNet | pretrained ResNet) -> CGF neck
          -> shared FieldHeadV3 -> (phi_logits, box, logvar) at strides 8/16/32

Differences from `models/fieldnet_model.TrainableFieldNet`:

  * the backbone is selectable, so from-scratch EduNet and an
    ImageNet-pretrained ResNet can be compared under an otherwise identical
    pipeline;
  * the head returns phi LOGITS rather than probabilities;
  * the head is told which pyramid level it is being applied to, so its
    per-level log-extent affine works (see models/head_v3.py).

The CGF neck is used completely unchanged — the pretrained backbones are
channel-matched to it deliberately.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.neck import CGF
from models.head_v3 import FieldHeadV3
from models.backbones_pretrained import build_backbone


class FieldNetV3(nn.Module):

    STRIDES: Tuple[int, int, int] = (8, 16, 32)

    def __init__(
        self,
        num_classes: int = 16,
        channels: int = 128,
        backbone: str = "resnet34",
        pretrained: bool = True,
        frozen_bn: bool = True,
        freeze_until: str = "layer1",
        head_norm: str = "gn",
        head_trunk_depth: int = 3,
        separate_trunks: bool = True,
        prior_prob: float = 0.01,
    ):
        super().__init__()

        if backbone == "edunet":
            self.backbone = build_backbone("edunet")
        else:
            self.backbone = build_backbone(
                backbone,
                pretrained=pretrained,
                frozen_bn=frozen_bn,
                freeze_until=freeze_until,
                out_channels=(128, 256, 512),
            )

        self.neck = CGF(channels=channels)

        self.head = FieldHeadV3(
            channels=channels,
            num_classes=num_classes,
            num_levels=len(self.STRIDES),
            trunk_depth=head_trunk_depth,
            norm=head_norm,
            separate_trunks=separate_trunks,
            prior_prob=prior_prob,
        )

        self.num_classes = num_classes
        self.strides = self.STRIDES
        self.backbone_name = backbone

    def forward(
        self, images: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        images: (B, 3, 416, 416), ImageNet-normalized if using a pretrained
        backbone.

        Returns {stride: (phi_logits, box, logvar)} for strides 8, 16, 32.
        `phi_logits` are RAW LOGITS — apply sigmoid for probabilities.
        """
        c3, c4, c5 = self.backbone(images)
        p3, p4, p5 = self.neck(c3, c4, c5)

        return {
            8: self.head(p3, level=0),
            16: self.head(p4, level=1),
            32: self.head(p5, level=2),
        }

    def param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1):
        """
        Separate LR groups for backbone vs neck+head.

        A pretrained backbone must be finetuned at a LOWER learning rate than
        randomly-initialized modules, otherwise the first few hundred steps
        destroy the ImageNet features you just paid for. 0.1x is the usual
        choice. For backbone="edunet" (random init) pass mult=1.0.
        """
        backbone_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if name.startswith("backbone.") else other_params).append(param)

        groups = [{"params": other_params, "lr": base_lr, "name": "neck_head"}]
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": base_lr * backbone_lr_mult,
                    "name": "backbone",
                }
            )
        return groups


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )


if __name__ == "__main__":
    for backbone in ("edunet", "resnet34"):
        model = FieldNetV3(
            num_classes=16,
            backbone=backbone,
            pretrained=False,  # offline-safe for this smoke test
        )
        out = model(torch.randn(1, 3, 416, 416))

        print(f"--- backbone={backbone} ---")
        for stride, (phi_logits, box, logvar) in sorted(out.items()):
            g = 416 // stride
            print(
                f"  stride {stride:2d}: phi_logits {tuple(phi_logits.shape)} "
                f"box {tuple(box.shape)} logvar {tuple(logvar.shape)}"
            )
            assert tuple(phi_logits.shape) == (1, 16, g, g), phi_logits.shape
            assert tuple(box.shape) == (1, 4, g, g), box.shape

        total = count_parameters(model)
        trainable = count_parameters(model, trainable_only=True)
        print(f"  params: {total:,} total / {trainable:,} trainable")

        with torch.no_grad():
            init_prob = float(torch.sigmoid(out[8][0]).mean())
        print(f"  initial mean phi prob: {init_prob:.4f} (expect ~0.01)")

    print("OK: FieldNetV3 shape contract satisfied for both backbones.")
