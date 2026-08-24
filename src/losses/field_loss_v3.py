"""
field_loss_v3 — the fixed loss for FieldNet.

=============================================================================
THE DIAGNOSIS
=============================================================================
The original `losses/pfl_loss.field_loss` is:

    bce_sum = F.binary_cross_entropy(phi_pred, phi_target, reduction="sum")
    loss    = bce_sum / n_pos + lambda_l1 * (l1_sum / n_pos)

`bce_sum` runs over EVERY element of a (B, 16, H, W) tensor at all three
strides — 16 * (52^2 + 26^2 + 13^2) = 56,784 elements per image — while
`n_pos` counts only the elements where phi_target > 0, typically a few
thousand. Roughly 95-97% of the gradient signal is therefore "push this
channel to zero", spread across all 16 class channels.

For a class with 114 training instances (laptop) that channel is background in
almost every image. The optimizer's fastest available loss reduction is to
zero the channel permanently, and nothing in plain BCE ever rewards climbing
back out. Combined with a phi bias that starts at sigmoid(0)=0.5 (see
models/head_v3.py), this is a textbook collapse configuration.

This is exactly the problem focal loss was invented for. It had never been
tried in this project: a repo-wide grep for "focal" returns zero hits.

=============================================================================
WHY PLAIN FOCAL LOSS IS NOT THE RIGHT ANSWER EITHER
=============================================================================
Standard focal loss assumes BINARY targets y in {0,1}:

    FL = -alpha * (1 - p_t)^gamma * log(p_t)

But PFL targets are CONTINUOUS: 1.0 at the box center, decaying smoothly to
0.0 at the box boundary. There is no p_t to define.

The correct generalization is Quality Focal Loss (QFL) from "Generalized Focal
Loss" (Li et al., NeurIPS 2020), which extends focal loss to continuous
targets y in [0,1]:

    QFL(sigma, y) = |y - sigma|^beta * BCE(sigma, y)

Check the endpoints:
    y = 0  ->  sigma^beta * (-log(1 - sigma))       = focal loss for negatives
    y = 1  ->  (1 - sigma)^beta * (-log sigma)      = focal loss for positives
    0<y<1  ->  smooth interpolation; the modulator |y - sigma|^beta -> 0
               as the prediction approaches the target, so well-fit elements
               stop contributing gradient.

=============================================================================
THE BIAS INIT AND QFL ARE MULTIPLICATIVE, NOT ADDITIVE — BOTH ARE REQUIRED
=============================================================================
This is the non-obvious part, and it is worth stating precisely because it
explains why neither fix alone would have worked.

Consider a rare-class channel (laptop: 114 instances across 2,894 images), so
the channel is pure background in ~96% of images. Take ~3,549 cells per image
across the three strides and ~50 positive cells when the object is present.
The gradient of BCE-with-logits w.r.t. the logit is (sigma - y), so we can
compare total downward (background) vs upward (positive) gradient directly:

    sigma    plain BCE down:up      QFL down:up
    0.50        1800.90 : 1          1800.90 : 1      <-- current init
    0.10         200.10 : 1             2.47 : 1
    0.05          94.78 : 1             0.26 : 1
    0.01          18.19 : 1            0.0019 : 1     <-- bias-init start

Read the first row carefully. **At sigma = 0.5, QFL changes nothing at all** —
the modulator is |y - 0.5|^2 = 0.25 for both background and positives, so it
cancels out and the ratio stays 1800:1. Adding focal loss to a head that
starts at 0.5 buys you nothing during exactly the early phase where rare-class
channels get destroyed.

Conversely, the bias init alone leaves plain BCE at 18:1 against the positives.
Its equilibrium on that channel — where the summed gradient is zero — is

    sigma* = n_pos / (n_pos + n_neg) = 0.00055

i.e. the channel's optimal plain-BCE behaviour is to emit a near-zero
CONSTANT and never discriminate. That is the collapse, quantified.

Together, though: the bias init puts the head at sigma ~= 0.01 from step 0,
where QFL's modulator is ~1e-4 for background and ~1 for unfit positives — a
~10,000x relative reweighting that flips the ratio to 0.002:1 in favour of the
positives. The head therefore has to become genuinely discriminative.

So: apply the bias init (models/head_v3.py) AND this loss. Applying one
without the other largely wastes the change.

=============================================================================
CLASS WEIGHTING
=============================================================================
QFL fixes foreground/background imbalance. It does NOT fix CROSS-CLASS
imbalance: PFL target mass scales with box AREA, so large classes (person,
desk, table, monitor) absorb far more of the positive gradient budget than
small ones (pen, pencil, mouse, book).

The user's own diagnostics prototyped inverse-PFL-mass class weights with an
alpha sweep (1.0 / 0.5 / 0.25) but never landed them in the loss. They are
implemented here properly, via `compute_class_weights`, and applied to the
whole class channel so the within-channel positive/negative balance that QFL
establishes is preserved.

    w_c = (mean_mass / mass_c) ^ alpha,   then normalized to mean 1.0

    alpha = 0.0  -> disabled (all weights 1.0)
    alpha = 0.25 -> mild correction
    alpha = 0.5  -> sqrt inverse-mass (recommended starting point)
    alpha = 1.0  -> full inverse-mass (can destabilize; use with care)

=============================================================================
BOX LOSS
=============================================================================
Two changes:

  1. V3 predicts log-extent DIRECTLY (channels are dx, dy, log_ew, log_eh),
     so the softplus-then-log round trip is gone. That removes a real bug
     class: two evaluators in this repo decoded raw linear extents while
     training used softplus, which is why the previously reported "best F1
     0.3200" is not comparable to the other numbers.

  2. A CIoU option that operates on DECODED boxes rather than on the four
     parameters independently. SmoothL1 on (dx, dy, log_ew, log_eh) treats the
     four numbers as unrelated, but mAP depends on the IoU of the assembled
     box. Every modern detector optimizes an IoU-family loss for this reason.
     Positives are additionally weighted by their PFL target value, so cells
     near an object's center — which are better positioned to regress it —
     count more. Default: "ciou".

`losses/pfl_loss.py` is left completely untouched, so every existing
experiment stays reproducible.
"""

from typing import Dict, Optional, Sequence
import json
import math

import torch
import torch.nn.functional as F


EPS = 1e-6
LOGVAR_CLAMP_MIN = -6.0
LOGVAR_CLAMP_MAX = 6.0

# exp(6) ~= 403, so a half-extent of 403 strides. Generous upper bound that
# still prevents inf/NaN if the head briefly diverges early in training.
LOG_EXT_CLAMP_MIN = -6.0
LOG_EXT_CLAMP_MAX = 6.0


# =========================================================================
# FIELD LOSS (Quality Focal Loss)
# =========================================================================

def quality_focal_field_loss(
    phi_logits: torch.Tensor,
    phi_target: torch.Tensor,
    beta: float = 2.0,
    class_weight: Optional[torch.Tensor] = None,
    normalizer: str = "pos_count",
    neg_weight: float = 1.0,
    class_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    QFL over the dense potential field.

    Args:
        phi_logits: (B, C, H, W) RAW LOGITS from FieldHeadV3.
        phi_target: (B, C, H, W) continuous targets in [0, 1].
        beta: focusing exponent. 2.0 is the GFL paper default. Must be >= 1
            to keep |y - sigma|^beta differentiable at y == sigma.
        class_weight: optional (C,) tensor, mean ~1.0. See
            `compute_class_weights`.
        normalizer: "pos_count" divides by the number of elements with
            target > 0 (comparable to the original loss); "pos_mass" divides
            by the summed target mass (smoother, favours large objects less).
        neg_weight: multiplier on the loss at cells whose target is exactly 0.
            1.0 is the original behaviour. LEAVE IT AT 1.0 if `class_mask` is in
            use; see below.
        class_mask: optional (B, C) 0/1 tensor. 0 means "this image's source
            dataset never labels this class", and that channel then contributes
            NOTHING to the loss for that image. This is the federated /
            multi-dataset supervision fix; see below.

    Returns:
        scalar loss

    WHY `class_mask` EXISTS  (measured 2026-08-22, and it is the important one)
    -------------------------------------------------------------------------
    FieldNet's data is five source datasets pooled into one file, and their label
    vocabularies are not compatible. Only source `base` (1746 images) labels all
    the classes. `stationary` (546 images) labels 3 of 14; `annotation` (215)
    labels 4; `indoor` (498) labels 9; `object` (380) labels 11. So 1639 of 3385
    images -- 48% -- come from datasets where most classes are never labelled
    even when plainly visible: a `stationary` photo of a desk with a laptop,
    monitor and mouse on it carries labels for the pens and nothing else.

    PFL targets are a hard 0 everywhere outside a labelled box. So on those
    images every correct detection of an out-of-vocabulary class is a
    full-strength "this is background" gradient. The model is explicitly taught
    to suppress objects it can see, on half the training set, and then valid asks
    for them back.

    The mask removes that supervision instead of reweighting it. A masked-out
    channel is neither positive nor negative for that image: it gets no gradient
    at all. That is the correct statement of what the data says -- "unknown", not
    "absent" -- and it is the standard remedy in multi-dataset detection.

    Two properties worth knowing:

      * The NORMALIZER IS UNAFFECTED. `pos_count` counts cells with target > 0,
        and a masked-out channel has none: the vocabulary is derived as "every
        class with at least one annotation in this source", so zero-vocabulary
        implies zero boxes implies zero positive cells. Masking therefore removes
        background loss only, and the positive gradient keeps its exact
        magnitude. (If `taxonomy.VOCAB_MIN_COUNT` is ever raised above 1 that
        invariant breaks, which is why `verify_v3.py` checks it against the real
        split rather than trusting it.)
      * It is NOT `neg_weight`, and they should not be combined casually.
        `neg_weight` down-weights ALL background in ALL channels on ALL images,
        so it pays for the unlabelled classes by also unlearning the labelled
        ones. The mask is surgical: it removes background loss for exactly the
        (image, class) pairs where the label is missing by construction, and
        leaves every other background cell at full strength. With the mask in
        place, keep `neg_weight=1.0` unless there is separate evidence for it.

    WHY `neg_weight` EXISTS  (measured, 2026-08-22 — SUPERSEDED, see above)
    ----------------------------------------------------------------------
    Kept because it is cheap and because its justification is not fully dead:
    even within a source's vocabulary, annotation is incomplete. But the original
    argument for it was that "train is 10x sparser than valid", which turned out
    to be split SELECTION bias (valid was the top 25% of train by label count),
    and the per-class version of the argument is falsified too
    (Spearman(incompleteness, AP50) = -0.079). At the converged operating point
    background is only ~0.04% of the field loss anyway. If used at all the right
    value is ~0.1, not the 0.5 this docstring used to recommend.

    Down-weighting the negative term needs no knowledge of WHERE the missing
    objects are and cannot inject a wrong box -- that part was and is true.
    Pseudo-labelling was tried first and hit a ceiling: after 80 epochs the model
    has learned to agree with the sparse labels, so it no longer fires on what is
    missing (mining recovered only 411-1835 boxes, lifting density from 2.60 to
    2.74-3.23 against a target of 7.46, and the aggressive pass was visibly
    noisy).

    Cost of down-weighting negatives: more false positives in general. Most
    failing classes here UNDER-fire, so that trade is aligned -- but desk
    over-fires and should be watched.

    Note that cells on a box's outer edge also hold target 0, since the PFL patch
    is clamp(1 - d, min=0); they carry no positive signal, so treating them as
    negatives is consistent.
    """
    if phi_logits.shape != phi_target.shape:
        raise ValueError(
            f"shape mismatch: phi_logits {tuple(phi_logits.shape)} "
            f"vs phi_target {tuple(phi_target.shape)}"
        )
    if beta < 1.0:
        raise ValueError(
            f"beta must be >= 1.0 for a differentiable modulator, got {beta}"
        )
    if not 0.0 < neg_weight <= 1.0:
        raise ValueError(
            f"neg_weight must be in (0, 1]; got {neg_weight}. Above 1.0 would "
            f"worsen the false-negative problem it exists to fix, and 0.0 would "
            f"remove background supervision entirely, so every channel would "
            f"saturate to 1."
        )

    # Fused, numerically stable — this is why the head returns logits.
    bce = F.binary_cross_entropy_with_logits(
        phi_logits, phi_target, reduction="none"
    )

    sigma = torch.sigmoid(phi_logits)

    # The QFL modulator. Gradient is allowed to flow through it, matching both
    # the reference focal-loss and GFL implementations.
    modulator = (phi_target - sigma).abs().clamp(min=EPS).pow(beta)

    loss = modulator * bce  # (B, C, H, W)

    if class_weight is not None:
        if class_weight.numel() != phi_logits.shape[1]:
            raise ValueError(
                f"class_weight has {class_weight.numel()} entries but there "
                f"are {phi_logits.shape[1]} classes"
            )
        w = class_weight.to(loss.device, loss.dtype).view(1, -1, 1, 1)
        loss = loss * w

    if neg_weight != 1.0:
        # Cells with target exactly 0 are the ones an unlabelled object would
        # land in. Scale them down; leave every positive cell untouched.
        loss = torch.where(phi_target > 0, loss, loss * neg_weight)

    if class_mask is not None:
        b, c = phi_logits.shape[0], phi_logits.shape[1]
        if tuple(class_mask.shape) != (b, c):
            raise ValueError(
                f"class_mask must be (B, C) = ({b}, {c}), got "
                f"{tuple(class_mask.shape)}. It is per-IMAGE, not per-batch: "
                f"a batch mixes images from different source datasets, which is "
                f"the entire reason the mask is needed."
            )
        loss = loss * class_mask.to(loss.device, loss.dtype).view(b, c, 1, 1)

    if normalizer == "pos_count":
        denom = (phi_target > 0).sum().clamp(min=1).to(loss.dtype)
    elif normalizer == "pos_mass":
        denom = phi_target.sum().clamp(min=1.0).to(loss.dtype)
    else:
        raise ValueError(
            f"normalizer must be 'pos_count' or 'pos_mass', got {normalizer!r}"
        )

    return loss.sum() / denom


# =========================================================================
# BOX DECODE (single source of truth, shared by loss and inference)
# =========================================================================

def decode_boxes(
    dx: torch.Tensor,
    dy: torch.Tensor,
    log_ew: torch.Tensor,
    log_eh: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """
    Turn head outputs into xyxy pixel boxes.

        cx = px + dx * stride
        cy = py + dy * stride
        hw = exp(log_ew) * stride
        hh = exp(log_eh) * stride

    px/py are the PIXEL CENTERS of the cells, i.e. (g + 0.5) * stride.

    This function is imported by both the loss and the decoder so the two can
    never drift apart — the previous softplus-vs-linear split is structurally
    impossible now.

    Returns (..., 4) xyxy.
    """
    cx = px + dx * stride
    cy = py + dy * stride
    hw = torch.exp(log_ew.clamp(LOG_EXT_CLAMP_MIN, LOG_EXT_CLAMP_MAX)) * stride
    hh = torch.exp(log_eh.clamp(LOG_EXT_CLAMP_MIN, LOG_EXT_CLAMP_MAX)) * stride
    return torch.stack([cx - hw, cy - hh, cx + hw, cy + hh], dim=-1)


def cell_centers(height: int, width: int, stride: int, device, dtype):
    """Pixel centers (py, px) of every cell in an (H, W) grid."""
    gy = torch.arange(height, device=device, dtype=dtype)
    gx = torch.arange(width, device=device, dtype=dtype)
    py = (gy + 0.5) * stride
    px = (gx + 0.5) * stride
    return py.view(-1, 1).expand(height, width), px.view(1, -1).expand(height, width)


# =========================================================================
# IoU FAMILY
# =========================================================================

def _iou_components(pred: torch.Tensor, target: torch.Tensor):
    """Shared geometry for IoU / GIoU / CIoU. Inputs (N,4) xyxy."""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    target_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = pred_area + target_area - inter + EPS
    iou = inter / union

    # Smallest enclosing box.
    ex1 = torch.minimum(px1, tx1)
    ey1 = torch.minimum(py1, ty1)
    ex2 = torch.maximum(px2, tx2)
    ey2 = torch.maximum(py2, ty2)

    return iou, union, (ex1, ey1, ex2, ey2), (px1, py1, px2, py2), (tx1, ty1, tx2, ty2)


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Complete-IoU loss, elementwise. Inputs (N,4) xyxy, returns (N,).

    CIoU = IoU - rho^2 / c^2 - alpha * v
        rho = distance between box centers
        c   = diagonal of the smallest enclosing box
        v   = aspect-ratio consistency
    loss = 1 - CIoU, in [0, ~2].
    """
    iou, _, (ex1, ey1, ex2, ey2), (px1, py1, px2, py2), (tx1, ty1, tx2, ty2) = (
        _iou_components(pred, target)
    )

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + EPS

    pw = (px2 - px1).clamp(min=EPS)
    ph = (py2 - py1).clamp(min=EPS)
    tw = (tx2 - tx1).clamp(min=EPS)
    th = (ty2 - ty1).clamp(min=EPS)

    v = (4.0 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    # alpha is treated as a constant, per the CIoU paper.
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + EPS)

    return 1.0 - (iou - rho2 / c2 - alpha * v)


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss, elementwise. Inputs (N,4) xyxy, returns (N,)."""
    iou, union, (ex1, ey1, ex2, ey2), _, _ = _iou_components(pred, target)
    enclose = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0) + EPS
    return 1.0 - (iou - (enclose - union) / enclose)


# =========================================================================
# BOX LOSS
# =========================================================================

def _object_group_ids(
    dx_t: torch.Tensor,
    dy_t: torch.Tensor,
    ew_t: torch.Tensor,
    eh_t: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    stride: int,
    batch_idx: torch.Tensor,
):
    """
    Recover a per-object group id for every positive cell.

    `box_target` stores the SAME object geometry at every cell that object won,
    but in cell-relative form (dx = (cx - px)/stride). Undoing that recovers
    absolute (cx, cy, hw, hh), which uniquely fingerprints the owning ground
    truth. So the grouping is derivable from tensors the collate already
    produces — `losses/box_targets.py` and `data/collate_v3.py` stay untouched,
    which matters because the audit directive says not to disturb the target
    generator.

    `batch_idx` is part of the key: two different images in one batch can hold
    same-size objects at the same pixel location, and merging those would halve
    both of their weights.

    Quantized to 1/16 px before `unique` because dx is float32 and the round
    trip `px + dx*stride` is not bit-exact (error ~1e-4 px at these magnitudes).
    Two genuinely distinct objects would have to agree on centre AND both
    extents to within 1/16 px to collide, which real annotations do not.

    Returns (inverse, num_groups); inverse[i] is the group of positive cell i.
    """
    cx = px + dx_t * stride
    cy = py + dy_t * stride
    key = torch.stack(
        [
            batch_idx,
            torch.round(cx * 16.0),
            torch.round(cy * 16.0),
            torch.round(ew_t * stride * 16.0),
            torch.round(eh_t * stride * 16.0),
        ],
        dim=-1,
    )
    unique, inverse = torch.unique(key, dim=0, return_inverse=True)
    return inverse, int(unique.shape[0])


def box_loss_v3(
    box_pred: torch.Tensor,
    box_target: torch.Tensor,
    pos_mask: torch.Tensor,
    stride: int,
    mode: str = "ciou",
    quality: Optional[torch.Tensor] = None,
    normalizer: str = "object",
) -> torch.Tensor:
    """
    Args:
        box_pred:   (B,4,H,W) head output = (dx, dy, log_ew, log_eh)
        box_target: (B,4,H,W) targets     = (dx, dy, ew,     eh)
                    NOTE targets carry LINEAR extents (ew = half_width/stride),
                    matching `losses/box_targets.generate_box_targets`. They are
                    log-transformed here so the existing target generator needs
                    no change.
        pos_mask:   (B,H,W) bool
        stride:     pyramid stride, needed to decode to pixels
        mode:       "ciou" | "giou" | "smooth_l1"
        quality:    optional (B,H,W) PFL target value per cell, used to weight
                    positives. Pass phi_target.amax(dim=1).
        normalizer: "object" (default) or "cell".

    WHY `normalizer="object"` IS THE DEFAULT  (audit finding #6, measured)
    ---------------------------------------------------------------------
    `normalizer="cell"` is the original behaviour: a quality-weighted mean over
    positive cells. Because a PFL field covers ~4*ew*eh cells, an object's share
    of the box gradient is proportional to its AREA. Measured on 8 training
    images with `scripts/diagnose_boxes_v3.py`:

        bag    39.82% of positive cells, 39.78% of PFL mass
        mouse   1.04% of positive cells,  1.07% of PFL mass     38x imbalance

    The same audit showed the targets at those cells are correct (IoU 1.00 for
    25/26 objects) while the PREDICTIONS there are not, and that the predicted
    extents are systematically too large on the short axis, worsening as the
    object gets thinner:

        pen    79x23 -> predicted half-height 8.28x too big, IoU 0.07
        bottle 95x43 -> 7.89x, IoU 0.05
        bag   159x78 -> 2.95x, IoU 0.27
        mouse 121x80 -> 2.47x, IoU 0.19
        every object with a short side >= 89 px -> ~1.0x, IoU 0.87-0.97

    That is the signature of a head that never received enough gradient to move
    off the large-object average, not of a bad target or a bad decode.

    "object" normalizes the quality weights WITHIN each object, so every object
    contributes total weight 1 regardless of how many cells it covers, while
    centre cells still count more than edge cells inside that object. The
    returned value stays a mean of per-cell CIoU losses in [0, ~2], so
    `lambda_box` keeps its meaning and old logs stay roughly comparable.

    Returns scalar loss.
    """
    if pos_mask.sum() == 0:
        return torch.zeros((), device=box_pred.device, dtype=box_pred.dtype)

    if normalizer not in ("object", "cell"):
        raise ValueError(
            f"normalizer must be 'object' or 'cell', got {normalizer!r}"
        )

    batch, _, h, w = box_pred.shape

    pred = box_pred.permute(0, 2, 3, 1)[pos_mask]    # (N,4)
    target = box_target.permute(0, 2, 3, 1)[pos_mask]  # (N,4)

    dx_p, dy_p, lew_p, leh_p = pred.unbind(-1)
    dx_t, dy_t, ew_t, eh_t = target.unbind(-1)

    # Cell centres are needed by the CIoU/GIoU decode AND by the object
    # grouping, so compute them once for every mode.
    py_grid, px_grid = cell_centers(h, w, stride, box_pred.device, box_pred.dtype)
    py = py_grid.unsqueeze(0).expand(batch, h, w)[pos_mask]
    px = px_grid.unsqueeze(0).expand(batch, h, w)[pos_mask]

    if mode == "smooth_l1":
        # Compare in log space, matching the head's parameterization.
        target_log = torch.stack(
            [
                dx_t,
                dy_t,
                torch.log(ew_t.clamp(min=EPS)),
                torch.log(eh_t.clamp(min=EPS)),
            ],
            dim=-1,
        )
        pred_log = torch.stack([dx_p, dy_p, lew_p, leh_p], dim=-1)
        per_element = F.smooth_l1_loss(pred_log, target_log, reduction="none").sum(-1)

    elif mode in ("ciou", "giou"):
        pred_boxes = decode_boxes(dx_p, dy_p, lew_p, leh_p, px, py, stride)
        target_boxes = decode_boxes(
            dx_t,
            dy_t,
            torch.log(ew_t.clamp(min=EPS)),
            torch.log(eh_t.clamp(min=EPS)),
            px,
            py,
            stride,
        )
        fn = ciou_loss if mode == "ciou" else giou_loss
        per_element = fn(pred_boxes, target_boxes)

    else:
        raise ValueError(
            f"mode must be 'ciou', 'giou' or 'smooth_l1', got {mode!r}"
        )

    if quality is not None:
        weight = quality[pos_mask].clamp(min=0.0)
    else:
        weight = torch.ones_like(per_element)

    if normalizer == "cell":
        # Original behaviour: area-weighted. Kept so the earlier runs can be
        # reproduced exactly for an A/B.
        return (per_element * weight).sum() / weight.sum().clamp(min=EPS)

    # --- normalizer == "object" -----------------------------------------
    batch_idx = (
        torch.arange(batch, device=box_pred.device, dtype=box_pred.dtype)
        .view(-1, 1, 1)
        .expand(batch, h, w)[pos_mask]
    )
    group, n_groups = _object_group_ids(
        dx_t, dy_t, ew_t, eh_t, px, py, stride, batch_idx
    )

    # Sum of weights within each object, then divide each cell's weight by its
    # own object's total => every object contributes exactly 1.0 of weight, but
    # centre cells still dominate edge cells inside that object.
    group_weight = torch.zeros(
        n_groups, device=box_pred.device, dtype=per_element.dtype
    ).scatter_add_(0, group, weight.to(per_element.dtype))
    weight = weight / group_weight[group].clamp(min=EPS)

    return (per_element * weight).sum() / float(n_groups)


# =========================================================================
# UNCERTAINTY LOSS
# =========================================================================

def uncertainty_loss_v3(
    box_pred: torch.Tensor,
    logvar_pred: torch.Tensor,
    box_target: torch.Tensor,
    pos_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian NLL on the four log-space regression targets, positives only.
    Kept for spec compatibility; lambda_unc defaults to 0.0 because both v2.1
    training runs disabled it.
    """
    if pos_mask.sum() == 0:
        return torch.zeros((), device=box_pred.device, dtype=box_pred.dtype)

    pred = box_pred.permute(0, 2, 3, 1)[pos_mask]
    target = box_target.permute(0, 2, 3, 1)[pos_mask]
    logvar = logvar_pred.permute(0, 2, 3, 1)[pos_mask].clamp(
        LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX
    )

    dx_t, dy_t, ew_t, eh_t = target.unbind(-1)
    target_log = torch.stack(
        [dx_t, dy_t, torch.log(ew_t.clamp(min=EPS)), torch.log(eh_t.clamp(min=EPS))],
        dim=-1,
    )

    sq_err = (pred - target_log) ** 2
    nll = sq_err / (2.0 * torch.exp(logvar)) + 0.5 * logvar
    return nll.sum(-1).mean()


# =========================================================================
# TOTAL
# =========================================================================

def total_loss_v3(
    phi_logits: torch.Tensor,
    phi_target: torch.Tensor,
    box_pred: torch.Tensor,
    logvar_pred: torch.Tensor,
    box_target: torch.Tensor,
    pos_mask: torch.Tensor,
    stride: int,
    lambda_field: float = 1.0,
    lambda_box: float = 2.0,
    lambda_unc: float = 0.0,
    qfl_beta: float = 2.0,
    class_weight: Optional[torch.Tensor] = None,
    normalizer: str = "pos_count",
    neg_weight: float = 1.0,
    box_mode: str = "ciou",
    box_normalizer: str = "object",
    quality_weight_box: bool = True,
    class_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Combined loss for ONE pyramid level. The caller sums over levels.

    `normalizer` applies to the FIELD loss ("pos_count" | "pos_mass").
    `box_normalizer` applies to the BOX loss ("object" | "cell") — see
    `box_loss_v3` for why "object" is the default. The two are deliberately
    separate knobs: classification is already healthy under "pos_count"
    (field loss 0.0046 on the overfit check), so the area bias only needed
    fixing on the regression side.

    `class_mask` is (B, C) and removes the loss for (image, class) pairs whose
    source dataset never labels that class — 48% of this dataset's images. See
    `quality_focal_field_loss` for the evidence. Like `neg_weight` it touches
    ONLY the field loss, and for the same reason: the box loss is computed on
    `pos_mask`, which is built from the GT boxes, so a class with no boxes on an
    image contributes no cells to it. There is nothing there to mask. The same
    goes for `quality`, which is `phi_target.amax(dim=1)` — masked-out channels
    are all-zero in the target, so they cannot win the max.

    `neg_weight` scales the field loss at cells whose target is 0. It predates
    the mask, is largely superseded by it, and should be left at 1.0 when the
    mask is in use — see `quality_focal_field_loss`.

    Returns a dict of the individual components plus the weighted total, so
    each can be logged separately — essential for telling apart "the field loss
    is not learning" from "the box loss is not learning".
    """
    l_field = quality_focal_field_loss(
        phi_logits,
        phi_target,
        beta=qfl_beta,
        class_weight=class_weight,
        normalizer=normalizer,
        neg_weight=neg_weight,
        class_mask=class_mask,
    )

    quality = phi_target.amax(dim=1) if quality_weight_box else None

    l_box = box_loss_v3(
        box_pred,
        box_target,
        pos_mask,
        stride,
        mode=box_mode,
        quality=quality,
        normalizer=box_normalizer,
    )

    if lambda_unc > 0.0:
        l_unc = uncertainty_loss_v3(box_pred, logvar_pred, box_target, pos_mask)
    else:
        l_unc = torch.zeros((), device=box_pred.device, dtype=box_pred.dtype)

    total = lambda_field * l_field + lambda_box * l_box + lambda_unc * l_unc

    return {
        "L_field": l_field,
        "L_box": l_box,
        "L_uncertainty": l_unc,
        "L_total": total,
    }


# =========================================================================
# CLASS WEIGHTS FROM DATASET STATISTICS
# =========================================================================

def compute_class_weights(
    annotation_file: str,
    num_classes: int = 16,
    alpha: float = 0.5,
    use_area: bool = True,
    clamp_max: float = 5.0,
) -> torch.Tensor:
    """
    Inverse-mass class weights, computed ONCE from the training annotations.

    Because PFL target mass grows with box area, `use_area=True` weights by
    total annotated area (a direct proxy for how much field mass each class
    contributes) rather than by instance count. This targets the actual
    imbalance rather than a correlate of it.

        w_c = (mean_mass / mass_c) ^ alpha,  normalized to mean 1, clamped.

    alpha=0.0 returns all ones (feature disabled).
    """
    if alpha == 0.0:
        return torch.ones(num_classes)

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    mass = [0.0] * num_classes
    for ann in data["annotations"]:
        cid = int(ann["category_id"])
        if not (0 <= cid < num_classes):
            continue
        if use_area:
            _, _, w, h = ann["bbox"]
            mass[cid] += max(float(w) * float(h), 0.0)
        else:
            mass[cid] += 1.0

    # A class with zero mass would give an infinite weight; floor it at the
    # smallest observed positive mass so the result stays finite.
    positive = [m for m in mass if m > 0]
    if not positive:
        raise ValueError(f"no annotations found in {annotation_file}")
    floor = min(positive)
    mass = [m if m > 0 else floor for m in mass]

    mean_mass = sum(mass) / num_classes
    weights = torch.tensor(
        [(mean_mass / m) ** alpha for m in mass], dtype=torch.float32
    )
    weights = weights / weights.mean()
    return weights.clamp(max=clamp_max)


if __name__ == "__main__":
    # Numeric self-checks. No dataset, no GPU required.
    torch.manual_seed(0)

    B, C, H, W = 2, 16, 26, 26

    # --- 1. QFL reduces to focal loss at the binary endpoints -------------
    logits = torch.randn(B, C, H, W)
    zeros = torch.zeros(B, C, H, W)
    sigma = torch.sigmoid(logits)

    qfl_neg = (
        sigma.pow(2.0)
        * F.binary_cross_entropy_with_logits(logits, zeros, reduction="none")
    ).sum()
    ours = quality_focal_field_loss(logits, zeros, beta=2.0)
    # zeros target => denom clamps to 1, so the sums must agree
    assert torch.allclose(ours, qfl_neg, rtol=1e-5), (ours, qfl_neg)
    print(f"QFL at y=0 matches focal-negative form: {float(ours):.4f}  OK")

    # --- 2. QFL is ~0 when the prediction matches the target -------------
    target = torch.rand(B, C, H, W)
    perfect = torch.log(target.clamp(EPS, 1 - EPS) / (1 - target.clamp(EPS, 1 - EPS)))
    near_zero = quality_focal_field_loss(perfect, target, beta=2.0)
    print(f"QFL with perfect predictions: {float(near_zero):.3e} (expect ~0)")
    assert float(near_zero) < 1e-3

    # --- 3. The load-bearing test: rare-class gradient balance -----------
    # Simulates ONE rare-class channel (laptop-like: present in ~4% of images)
    # and measures total downward (background) vs upward (positive) gradient
    # under plain BCE and under QFL, at several prediction levels.
    #
    # This is the test that justifies the whole design, and it shows that the
    # bias init and QFL are multiplicative: at sigma=0.5 QFL does nothing.
    print("\n--- rare-class channel gradient balance (down:up) ---")
    print(f"{'sigma':>7} {'plain BCE':>14} {'QFL':>14}   verdict")

    cells_per_image = 52 * 52 + 26 * 26 + 13 * 13   # 3549
    images = 100
    p_present = 114 / 2894                          # laptop base rate
    pos_cells = 50                                  # positives when present

    n_pos = images * p_present * pos_cells
    n_bg = images * cells_per_image - n_pos

    ratios = {}
    for s in (0.5, 0.10, 0.05, 0.01):
        # |d/dlogit BCE| = |sigma - y|
        plain = (n_bg * s) / (n_pos * (1 - s))
        # QFL multiplies each element by |y - sigma|^beta
        qfl = (n_bg * s * s ** 2) / (n_pos * (1 - s) * (1 - s) ** 2)
        ratios[s] = (plain, qfl)
        verdict = "QFL no help" if abs(plain - qfl) / plain < 0.01 else "QFL rebalances"
        print(f"{s:>7} {plain:>14.2f} {qfl:>14.4f}   {verdict}")

    # At the ORIGINAL init (sigma=0.5) QFL must be indistinguishable from BCE.
    plain_half, qfl_half = ratios[0.5]
    assert abs(plain_half - qfl_half) / plain_half < 0.01, (
        "at sigma=0.5 the QFL modulator cancels; if this fails the reasoning "
        "in the module docstring is wrong"
    )
    # At the BIAS-INIT start (sigma=0.01) QFL must flip the balance to favour
    # positives by a wide margin.
    plain_prior, qfl_prior = ratios[0.01]
    assert plain_prior > 10.0, "plain BCE should still suppress the rare class"
    assert qfl_prior < 0.01, "QFL should decisively favour positives at the prior"
    print(
        f"\n  => bias init alone: {plain_prior:.1f}:1 against the class (collapse)\n"
        f"  => bias init + QFL: {qfl_prior:.4f}:1, i.e. positives dominate "
        f"{1/qfl_prior:.0f}x"
    )

    # Plain BCE's equilibrium on this channel is a near-zero constant.
    sigma_star = n_pos / (n_pos + n_bg)
    print(f"  => plain-BCE equilibrium sigma* = {sigma_star:.5f} (never discriminates)")
    assert sigma_star < 0.01

    # --- 4. CIoU is 0 for identical boxes, positive otherwise ------------
    a = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    assert float(ciou_loss(a, a)) < 1e-5
    b = torch.tensor([[20.0, 20.0, 60.0, 60.0]])
    assert float(ciou_loss(a, b)) > 0
    print(f"CIoU identical={float(ciou_loss(a, a)):.2e}  shifted={float(ciou_loss(a, b)):.4f}  OK")

    # --- 5. decode_boxes round trip -------------------------------------
    stride = 8
    px = torch.tensor([100.0])
    py = torch.tensor([200.0])
    got = decode_boxes(
        torch.tensor([0.25]), torch.tensor([-0.5]),
        torch.log(torch.tensor([4.0])), torch.log(torch.tensor([2.0])),
        px, py, stride,
    )
    # cx = 100 + 0.25*8 = 102 ; cy = 200 - 0.5*8 = 196
    # hw = 4*8 = 32 ; hh = 2*8 = 16
    want = torch.tensor([[70.0, 180.0, 134.0, 212.0]])
    assert torch.allclose(got, want, atol=1e-4), (got, want)
    print("decode_boxes round trip OK")

    # --- 6. box normalizer: does a small object get heard? ---------------
    # This is the test for audit finding #6, measured as a 38x cell imbalance
    # by scripts/diagnose_boxes_v3.py. Build one LARGE and one SMALL object,
    # corrupt ONLY the small object's predicted extent, and compare how much
    # each normalizer notices.
    print("\n--- box normalizer: does a small object get heard? ---")
    stride_t, G, B_ = 16, 26, 1
    box_t = torch.zeros(B_, 4, G, G)
    posm = torch.zeros(B_, G, G, dtype=torch.bool)
    qual = torch.zeros(B_, G, G)

    def paint(cx, cy, hw, hh):
        """Write one object's PFL field and box target, as box_targets.py does."""
        n = 0
        for gy in range(G):
            for gx in range(G):
                px_ = (gx + 0.5) * stride_t
                py_ = (gy + 0.5) * stride_t
                d = max(abs(px_ - cx) / hw, abs(py_ - cy) / hh)
                if d >= 1.0:
                    continue
                box_t[0, :, gy, gx] = torch.tensor(
                    [
                        (cx - px_) / stride_t,
                        (cy - py_) / stride_t,
                        hw / stride_t,      # LINEAR extent, matching the generator
                        hh / stride_t,
                    ]
                )
                posm[0, gy, gx] = True
                qual[0, gy, gx] = (1.0 - d) ** 1.5      # PFL gamma = 1.5
                n += 1
        return n

    n_large = paint(144.0, 144.0, 112.0, 112.0)   # 224x224 px
    n_small = paint(368.0, 368.0, 24.0, 24.0)     # 48x48 px
    print(f"  large object covers {n_large} cells, small covers {n_small} "
          f"  ({n_large / n_small:.0f}x cell imbalance)")

    # The grouping must recover exactly two objects from box_target alone.
    pyg, pxg = cell_centers(G, G, stride_t, box_t.device, box_t.dtype)
    tsel = box_t.permute(0, 2, 3, 1)[posm]
    _, n_groups = _object_group_ids(
        tsel[:, 0], tsel[:, 1], tsel[:, 2], tsel[:, 3],
        pxg.unsqueeze(0).expand(B_, G, G)[posm],
        pyg.unsqueeze(0).expand(B_, G, G)[posm],
        stride_t,
        torch.zeros(int(posm.sum()), dtype=box_t.dtype),
    )
    print(f"  object grouping recovered {n_groups} objects (expect 2)")
    assert n_groups == 2, f"grouping found {n_groups} objects, expected 2"

    # A perfect prediction must score ~0 under BOTH normalizers.
    box_p = box_t.clone()
    box_p[:, 2:] = torch.log(box_t[:, 2:].clamp(min=EPS))
    for norm in ("cell", "object"):
        base = box_loss_v3(
            box_p, box_t, posm, stride_t, mode="ciou", quality=qual,
            normalizer=norm,
        )
        assert float(base) < 1e-4, f"{norm}: perfect prediction gave {float(base)}"
    print("  both normalizers score ~0 on a perfect prediction  OK")

    # Now break ONLY the small object: predict its extent e^1 = 2.72x too big,
    # which is roughly what the real checkpoint does to `bottle` and `pen`.
    broken = box_p.clone()
    small_only = posm.clone()
    small_only[0, :20, :] = False           # large object lives in rows 2..15
    assert int(small_only.sum()) == n_small
    broken[:, 2:] += small_only.unsqueeze(1).float()

    deltas = {}
    for norm in ("cell", "object"):
        deltas[norm] = float(
            box_loss_v3(broken, box_t, posm, stride_t, mode="ciou",
                        quality=qual, normalizer=norm)
        )
        print(f"  {norm:>6}: loss when only the small object is wrong = "
              f"{deltas[norm]:.4f}")

    ratio = deltas["object"] / max(deltas["cell"], 1e-12)
    print(f"  => 'object' reacts {ratio:.1f}x more strongly to the small object")
    assert ratio > 5.0, (
        "per-object normalization should amplify a small object's contribution "
        "by roughly n_large/n_small; if this fails, the grouping is broken and "
        "the loss has silently reverted to area weighting"
    )

    # --- 7. neg_weight: does it spare the positive gradient? --------------
    # The claim that justifies this knob is narrow and checkable: lowering
    # neg_weight must scale the BACKGROUND gradient and leave the POSITIVE
    # gradient bit-for-bit alone. If it silently rescaled both, it would just be
    # a learning-rate change on the field loss and would fix nothing.
    print("\n--- neg_weight: background down-weighting ---")

    nw_target = torch.zeros(B, C, H, W)
    nw_target[0, 3, 10:16, 10:16] = torch.rand(6, 6).clamp(min=0.05)
    nw_target[1, 7, 4:9, 18:23] = torch.rand(5, 5).clamp(min=0.05)
    pos_cells_nw = int((nw_target > 0).sum())
    print(f"  {pos_cells_nw} positive cells, "
          f"{nw_target.numel() - pos_cells_nw} background cells")

    # 7a. neg_weight=1.0 must be EXACTLY the old behaviour, so every previous
    # result stays reproducible and the A/B has a real control.
    base_loss = quality_focal_field_loss(logits, nw_target, beta=2.0)
    same = quality_focal_field_loss(logits, nw_target, beta=2.0, neg_weight=1.0)
    assert float(base_loss) == float(same), (float(base_loss), float(same))
    print(f"  neg_weight=1.0 is bit-identical to default: {float(same):.6f}  OK")

    # 7b. The loss must decompose as pos + nw * neg.
    nw = 0.5
    down = quality_focal_field_loss(logits, nw_target, beta=2.0, neg_weight=nw)
    pos_only = quality_focal_field_loss(
        logits, nw_target, beta=2.0, neg_weight=1e-12
    )
    neg_part = float(base_loss) - float(pos_only)
    expect = float(pos_only) + nw * neg_part
    # Relative tolerance, not absolute: this is a float32 reduction over
    # ~21k cells and the two paths sum in different orders.
    assert abs(float(down) - expect) <= 1e-4 * max(expect, 1.0), (
        float(down), expect
    )
    print(f"  loss({nw}) = pos {float(pos_only):.6f} + {nw} x neg "
          f"{neg_part:.6f} = {expect:.6f}  OK")
    print(f"  background is {neg_part / float(base_loss) * 100:.1f}% of the "
          f"field loss at this init")

    # 7c. THE LOAD-BEARING CHECK: gradients.
    grads = {}
    for w in (1.0, nw):
        lg = logits.clone().requires_grad_(True)
        quality_focal_field_loss(lg, nw_target, beta=2.0, neg_weight=w).backward()
        grads[w] = lg.grad.clone()

    pos_mask_nw = nw_target > 0
    assert torch.equal(grads[1.0][pos_mask_nw], grads[nw][pos_mask_nw]), (
        "neg_weight changed the POSITIVE gradient; it must not. If this fails, "
        "the normalizer is counting background cells and the knob is just a "
        "learning-rate scale on the field loss."
    )
    print("  positive-cell gradients unchanged (bit-identical)  OK")

    bg = ~pos_mask_nw
    assert torch.allclose(grads[nw][bg], grads[1.0][bg] * nw,
                          rtol=1e-5, atol=1e-12), (
        "background gradient did not scale by exactly neg_weight"
    )
    g1 = float(grads[1.0][bg].abs().sum() / grads[1.0][pos_mask_nw].abs().sum())
    g2 = float(grads[nw][bg].abs().sum() / grads[nw][pos_mask_nw].abs().sum())
    print(f"  background:positive gradient mass {g1:.2f}:1 -> {g2:.2f}:1 "
          f"(halved, as intended)")

    # 7d. Guard rails. 0.0 would delete background supervision entirely and
    # every channel would saturate to 1; >1.0 would worsen the very problem
    # this exists to fix.
    for bad in (0.0, -0.5, 1.5):
        try:
            quality_focal_field_loss(logits, nw_target, neg_weight=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"neg_weight={bad} should have been rejected")
    print("  neg_weight 0.0 / -0.5 / 1.5 all rejected  OK")

    # ---------------------------------------------------------------------
    # 8. class_mask — federated supervision (see the docstring above)
    # ---------------------------------------------------------------------
    print("\n8. class_mask: per-image federated supervision")

    B, C, G = 2, 8, 6
    cm_target = torch.zeros(B, C, G, G)
    cm_target[0, 3, 2, 2] = 1.0
    cm_target[0, 3, 2, 3] = 0.6
    cm_target[1, 5, 4, 1] = 0.9      # positives live in channels 3 and 5 only

    def _cm_run(mask):
        lg = torch.randn(B, C, G, G,
                         generator=torch.Generator().manual_seed(7))
        lg.requires_grad_(True)
        out = quality_focal_field_loss(lg, cm_target, beta=2.0,
                                       normalizer="pos_count", class_mask=mask)
        out.backward()
        return float(out.detach()), lg.grad.clone()

    l_none, g_none = _cm_run(None)
    l_ones, g_ones = _cm_run(torch.ones(B, C))

    # 8a. All-ones must be a no-op, or turning the feature on silently
    # invalidates every number measured before it existed.
    assert l_none == l_ones and torch.equal(g_none, g_ones), (
        f"an all-ones class_mask is not a no-op ({l_none!r} vs {l_ones!r}). "
        "Every pre-mask result would become incomparable."
    )
    print(f"  all-ones mask == no mask, bit-identical ({l_none:.10f})  OK")

    # 8b. Masking background-only channels must delete their gradient
    # entirely and leave every other channel untouched, INCLUDING through the
    # normalizer -- pos_count counts target > 0, which the mask cannot change
    # as long as masked channels hold no positives (verify_v3.py stage 11
    # checks that this holds on the real split, rather than assuming it).
    cm = torch.ones(B, C)
    cm[:, :3] = 0.0
    l_part, g_part = _cm_run(cm)

    assert float(g_part[:, :3].abs().max()) == 0.0, \
        "masked-out channels still received gradient"
    kept = list(range(3, C))
    assert torch.equal(g_part[:, kept], g_none[:, kept]), (
        "masking channels 0-2 perturbed channels 3-7. The pos_count "
        "denominator must be independent of the mask."
    )
    assert l_part < l_none, "the mask had no effect on the loss at all"
    print(f"  masking 3/{C} background-only channels: loss {l_none:.6f} -> "
          f"{l_part:.6f}, other {len(kept)} channels bit-identical  OK")

    # 8c. Wrong shape must raise. (B, C) is per-IMAGE; a (C,) mask would
    # broadcast silently and apply one vocabulary to the whole batch, which is
    # exactly the confusion the mask exists to remove.
    for bad in (torch.ones(C), torch.ones(B, C, 1, 1), torch.ones(B, C + 1)):
        try:
            quality_focal_field_loss(torch.randn(B, C, G, G), cm_target,
                                     class_mask=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"class_mask of shape {tuple(bad.shape)} should have been "
                f"rejected; (C,) in particular would broadcast silently")
    print("  (C,), (B,C,1,1) and (B,C+1) masks all rejected  OK")

    print("\nOK: all field_loss_v3 self-checks passed.")
