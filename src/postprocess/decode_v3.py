"""
decode_v3 — unified inference decode with cross-level NMS.

TWO BUGS THIS FIXES, BOTH OF WHICH INFLATE FALSE POSITIVES DIRECTLY
===================================================================

BUG 1: NO NMS ANYWHERE IN THE PIPELINE.
`scripts/evaluate_decoded_v21.py` loops over strides 8, 16 and 32,
appends every SLE detection from each into one flat `predictions` list,
sorts by score, and matches greedily against ground truth. Nothing ever
suppresses overlapping boxes.

The pyramid levels are not partitioned by object size — there is no FCOS
style size-range assignment in this codebase, and the PFL target generator
writes a field for EVERY object at EVERY level. So a single chair produces
a peak at stride 8, a peak at stride 16, and a peak at stride 32. All three
decode to roughly the same box. Greedy matching lets exactly ONE of them
claim the GT (each GT is matched at most once, `matched_gt.add(best_gt)`),
and the other two are counted as false positives.

That is not a subtle effect. With 3 levels all firing, the theoretical
ceiling on precision is ~1/3 before anything else goes wrong, and the
reported "FP 485 vs TP 117" is almost exactly the 3:1-plus-slop shape you
would predict. NMS across levels is not an optimization here; its absence
is a scoring bug.

BUG 2: THREE DIFFERENT BOX DECODES FOR ONE TRAINED MODEL.
Training applies `softplus` then `log`. `evaluate_decoded_v21.py` applies
`softplus`. `evaluate_threshold_sweep_v21.py` (the source of the headline
"F1 0.3200") applies neither and reads the raw linear value. A model
trained under one convention and scored under another is being measured on
a task it was never trained for, so those numbers are not comparable to
each other or to anything else.

Here, every path funnels through `field_loss_v3.decode_boxes`, the same
function the loss uses. `box_extent` names the convention explicitly
instead of leaving it implicit in whichever script you happened to run.

WHY `box_extent` AND `phi_is_logits` EXIST
=========================================
So this ONE evaluator can score both a v3 checkpoint (logits, log-extent)
and the existing v2.1 checkpoint (sigmoid probs, softplus extent). That
matters for the comparison: running the old checkpoint through this decoder
with NMS enabled isolates how much of the improvement is "we added NMS" and
how much is "we retrained". Without that control, any v3 number is
confounded and you cannot tell which fix earned the gain.

TWO DECODE MODES
================
  mode="peak"  (default)
      Threshold -> 3x3 max-pool local-maxima -> top-k -> decode -> NMS.
      Fully vectorized, no scipy, runs on GPU, one detection per local
      maximum. This is the CenterNet/FCOS-family decode.

  mode="sle"
      The project's original Soft Level-Set Extraction: threshold ->
      connected components -> peaks -> nearest-peak split. Preserved so
      the research contribution can be measured rather than assumed, but
      it now shares the unified box decode and gets NMS applied after.

`mode="sle"` has a structural weakness worth knowing before you interpret
its numbers: its box comes from the extent of a thresholded blob, so the
box size is a function of `tau_c`. Lower the threshold and every box grows.
That couples localization quality to a confidence hyperparameter, which is
why a threshold sweep on SLE moves precision and recall in ways that have
nothing to do with what the model learned. The `peak` mode's box comes from
the regression branch and is independent of the threshold. Both are
available; prefer `peak` for headline numbers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from losses.field_loss_v3 import decode_boxes, EPS


BOX_EXTENT_CONVENTIONS = ("log", "softplus", "linear")
DECODE_MODES = ("peak", "sle")


@dataclass
class DecodeConfig:
    """
    Inference-time decode settings.

    score_thresh: keep detections above this. For mAP computation set this
        LOW (0.05 or lower) — average precision integrates the whole
        precision/recall curve, and a high threshold truncates the curve and
        silently caps your AP. For a live demo, set it high (0.3-0.5).
    nms_iou: IoU above which the lower-scoring box is suppressed. 0.6 is the
        usual detection default; 0.5 is more aggressive.
    topk_per_level: cap on candidates taken from each pyramid level before
        NMS. Bounds worst-case cost; 300 is far more than any real image
        needs at 16 classes.
    max_detections: final cap per image after NMS. COCO convention is 100.
    class_agnostic_nms: False means boxes of different classes never
        suppress each other (correct for mAP — a laptop and a keyboard
        genuinely overlap). True is only for single-object-per-region demos.
    phi_is_logits: True for FieldNetV3, False for the original
        `TrainableFieldNet` whose head already applies sigmoid.
    box_extent: how to read box channels 2 and 3.
        "log"      -> v3: value IS log(half_extent / stride)
        "softplus" -> v2.1 training convention: softplus then log
        "linear"   -> raw value is half_extent / stride
    """

    score_thresh: float = 0.05
    nms_iou: float = 0.6
    topk_per_level: int = 300
    max_detections: int = 100
    mode: str = "peak"
    peak_kernel: int = 3
    image_size: int = 416
    class_agnostic_nms: bool = False
    phi_is_logits: bool = True
    box_extent: str = "log"

    # SLE-only
    sle_tau_c: float = 0.30
    sle_tau_peak: float = 0.10

    def __post_init__(self) -> None:
        if self.mode not in DECODE_MODES:
            raise ValueError(f"mode must be one of {DECODE_MODES}, got {self.mode!r}")
        if self.box_extent not in BOX_EXTENT_CONVENTIONS:
            raise ValueError(
                f"box_extent must be one of {BOX_EXTENT_CONVENTIONS}, "
                f"got {self.box_extent!r}"
            )
        if self.peak_kernel % 2 != 1:
            raise ValueError(f"peak_kernel must be odd, got {self.peak_kernel}")
        if not 0.0 < self.nms_iou <= 1.0:
            raise ValueError(f"nms_iou must be in (0,1], got {self.nms_iou}")


# =========================================================================
# EXTENT CONVENTION ADAPTER
# =========================================================================

def extents_to_log(ext_raw: torch.Tensor, box_extent: str) -> torch.Tensor:
    """
    Convert whichever extent convention a checkpoint was trained under into
    the log-extent that `decode_boxes` expects.

    This function is the ONLY place a convention difference is allowed to
    exist. Everything downstream sees log-extent.
    """
    if box_extent == "log":
        return ext_raw
    if box_extent == "softplus":
        return torch.log(F.softplus(ext_raw).clamp(min=EPS))
    if box_extent == "linear":
        return torch.log(ext_raw.clamp(min=EPS))
    raise ValueError(f"unknown box_extent: {box_extent!r}")


def phi_to_prob(phi: torch.Tensor, phi_is_logits: bool) -> torch.Tensor:
    return torch.sigmoid(phi) if phi_is_logits else phi


# =========================================================================
# NMS
# =========================================================================

def _iou_one_to_many(box: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """IoU of a single (4,) box against an (N,4) set. Both xyxy."""
    x1 = torch.maximum(box[0], others[:, 0])
    y1 = torch.maximum(box[1], others[:, 1])
    x2 = torch.minimum(box[2], others[:, 2])
    y2 = torch.minimum(box[3], others[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_box = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_others = (
        (others[:, 2] - others[:, 0]).clamp(min=0)
        * (others[:, 3] - others[:, 1]).clamp(min=0)
    )
    union = area_box + area_others - inter
    return inter / union.clamp(min=EPS)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """
    Greedy NMS. Returns indices of kept boxes, highest score first.

    Pure-torch fallback used when torchvision is unavailable; `batched_nms`
    below prefers torchvision's fused CUDA kernel when it can.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: List[torch.Tensor] = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = _iou_one_to_many(boxes[current], boxes[rest])
        order = rest[ious <= iou_thresh]

    return torch.stack(keep)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_thresh: float,
    class_agnostic: bool = False,
) -> torch.Tensor:
    """
    Class-aware NMS across ALL pyramid levels at once.

    Running NMS per level would not fix anything — the duplicates are
    BETWEEN levels. Boxes from strides 8/16/32 must be pooled into one
    tensor before suppression, which is what `decode_image` does.

    Class-awareness uses the standard coordinate-offset trick: shift each
    class's boxes into its own disjoint region of coordinate space so a
    single NMS pass can never compare boxes of different classes.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    if class_agnostic:
        try:
            from torchvision.ops import nms as tv_nms
            return tv_nms(boxes, scores, iou_thresh)
        except ImportError:
            return nms(boxes, scores, iou_thresh)

    try:
        from torchvision.ops import batched_nms as tv_batched_nms
        return tv_batched_nms(boxes, scores, class_ids, iou_thresh)
    except ImportError:
        pass

    offset = (boxes.max() - boxes.min() + 1.0) if boxes.numel() else 1.0
    shifted = boxes + (class_ids.to(boxes.dtype) * offset).unsqueeze(1)
    return nms(shifted, scores, iou_thresh)


# =========================================================================
# PEAK DECODE
# =========================================================================

def peak_decode_level(
    phi_logits: torch.Tensor,
    box_raw: torch.Tensor,
    stride: int,
    cfg: DecodeConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode ONE pyramid level of ONE image.

        phi_logits: (C, H, W)
        box_raw:    (4, H, W)  channels (dx, dy, ext_w, ext_h)

    Returns (boxes (N,4) xyxy pixels, scores (N,), class_ids (N,)).

    A cell is a candidate if its field value equals the max of its
    `peak_kernel` neighbourhood and clears `score_thresh`. Note that
    max-pool equality marks every cell of a tied plateau — the original SLE
    code had to add a special plateau-merging pass for this because
    fragmenting a blob there produced several boxes with different extents.
    Here it is harmless: every tied cell regresses a box for the SAME
    object, so the boxes land on top of each other and NMS collapses them.
    """
    if phi_logits.dim() != 3 or box_raw.dim() != 3:
        raise ValueError(
            f"expected (C,H,W) and (4,H,W), got {tuple(phi_logits.shape)} "
            f"and {tuple(box_raw.shape)}"
        )

    num_classes, height, width = phi_logits.shape
    device, dtype = phi_logits.device, phi_logits.dtype

    prob = phi_to_prob(phi_logits, cfg.phi_is_logits)

    pooled = F.max_pool2d(
        prob.unsqueeze(0),
        kernel_size=cfg.peak_kernel,
        stride=1,
        padding=cfg.peak_kernel // 2,
    ).squeeze(0)

    is_peak = (prob >= pooled) & (prob >= cfg.score_thresh)

    # Zero out non-peaks so topk only ever returns genuine local maxima.
    masked = torch.where(is_peak, prob, torch.zeros_like(prob))

    flat = masked.reshape(-1)
    k = int(min(cfg.topk_per_level, flat.numel()))
    if k == 0:
        empty_b = torch.zeros((0, 4), device=device, dtype=dtype)
        empty_s = torch.zeros((0,), device=device, dtype=dtype)
        empty_c = torch.zeros((0,), device=device, dtype=torch.long)
        return empty_b, empty_s, empty_c

    scores, flat_idx = flat.topk(k)

    valid = scores >= cfg.score_thresh
    scores, flat_idx = scores[valid], flat_idx[valid]

    if scores.numel() == 0:
        empty_b = torch.zeros((0, 4), device=device, dtype=dtype)
        empty_c = torch.zeros((0,), device=device, dtype=torch.long)
        return empty_b, scores, empty_c

    # Unravel flat index over (C, H, W).
    class_ids = flat_idx // (height * width)
    rem = flat_idx % (height * width)
    gy = rem // width
    gx = rem % width

    # Gather the box prediction at each peak cell.
    dx = box_raw[0, gy, gx]
    dy = box_raw[1, gy, gx]
    log_ew = extents_to_log(box_raw[2, gy, gx], cfg.box_extent)
    log_eh = extents_to_log(box_raw[3, gy, gx], cfg.box_extent)

    px = (gx.to(dtype) + 0.5) * stride
    py = (gy.to(dtype) + 0.5) * stride

    boxes = decode_boxes(dx, dy, log_ew, log_eh, px, py, stride)

    return boxes, scores, class_ids.long()


# =========================================================================
# SLE DECODE (original pipeline, unified box decode + NMS)
# =========================================================================

def sle_decode_level(
    phi_logits: torch.Tensor,
    box_raw: torch.Tensor,
    stride: int,
    cfg: DecodeConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The project's original SLE decode, with the box taken from the
    regression branch through the unified `decode_boxes` (as
    `evaluate_decoded_v21.py` intended) rather than from the blob extent.
    """
    from postprocess.sle import run_sle, SLEConfig

    device, dtype = phi_logits.device, phi_logits.dtype
    prob = phi_to_prob(phi_logits, cfg.phi_is_logits)

    sle_cfg = SLEConfig(
        tau_c=cfg.sle_tau_c,
        tau_peak=cfg.sle_tau_peak,
        stride=stride,
        image_size=cfg.image_size,
    )
    detections = run_sle(prob.detach().cpu(), sle_cfg)

    if not detections:
        return (
            torch.zeros((0, 4), device=device, dtype=dtype),
            torch.zeros((0,), device=device, dtype=dtype),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    gy = torch.tensor([d["grid_y"] for d in detections], device=device)
    gx = torch.tensor([d["grid_x"] for d in detections], device=device)
    scores = torch.tensor(
        [d["score"] for d in detections], device=device, dtype=dtype
    )
    class_ids = torch.tensor(
        [d["class_id"] for d in detections], device=device, dtype=torch.long
    )

    dx = box_raw[0, gy, gx]
    dy = box_raw[1, gy, gx]
    log_ew = extents_to_log(box_raw[2, gy, gx], cfg.box_extent)
    log_eh = extents_to_log(box_raw[3, gy, gx], cfg.box_extent)

    px = (gx.to(dtype) + 0.5) * stride
    py = (gy.to(dtype) + 0.5) * stride

    boxes = decode_boxes(dx, dy, log_ew, log_eh, px, py, stride)

    keep = scores >= cfg.score_thresh
    return boxes[keep], scores[keep], class_ids[keep]


# =========================================================================
# IMAGE / BATCH LEVEL
# =========================================================================

def _clip_and_drop_degenerate(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    image_size: int,
    min_size: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip to the frame and remove boxes with no area."""
    if boxes.numel() == 0:
        return boxes, scores, class_ids

    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(image_size))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(image_size))

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_size) & (h >= min_size)

    return boxes[keep], scores[keep], class_ids[keep]


def decode_image(
    outputs: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    index: int,
    cfg: Optional[DecodeConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Decode image `index` of a batched model output.

        outputs: {stride: (phi, box, logvar)} exactly as the model returns.

    Returns {"boxes": (N,4), "scores": (N,), "labels": (N,)} sorted by
    descending score, after cross-level class-aware NMS.

    THE ORDER MATTERS: candidates from every stride are concatenated FIRST,
    then suppressed ONCE. Suppressing per level and then concatenating would
    leave the cross-level duplicates that cause the false positives.
    """
    cfg = cfg if cfg is not None else DecodeConfig()
    decoder = peak_decode_level if cfg.mode == "peak" else sle_decode_level

    all_boxes, all_scores, all_classes = [], [], []

    for stride in sorted(outputs.keys()):
        phi, box_raw, _logvar = outputs[stride]
        boxes, scores, class_ids = decoder(
            phi[index].detach(), box_raw[index].detach(), stride, cfg
        )
        if boxes.numel():
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(class_ids)

    if not all_boxes:
        device = next(iter(outputs.values()))[0].device
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "scores": torch.zeros((0,), device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
        }

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    class_ids = torch.cat(all_classes, dim=0)

    boxes, scores, class_ids = _clip_and_drop_degenerate(
        boxes, scores, class_ids, cfg.image_size
    )

    keep = batched_nms(
        boxes, scores, class_ids, cfg.nms_iou, cfg.class_agnostic_nms
    )
    keep = keep[: cfg.max_detections]

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": class_ids[keep],
    }


def decode_batch(
    outputs: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: Optional[DecodeConfig] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Decode every image in the batch. Returns one dict per image."""
    batch_size = next(iter(outputs.values()))[0].shape[0]
    return [decode_image(outputs, i, cfg) for i in range(batch_size)]


# =========================================================================
# SELF-TEST
# =========================================================================

if __name__ == "__main__":

    torch.manual_seed(0)

    # ---- NMS: three identical boxes must collapse to one -----------------
    boxes = torch.tensor([
        [10.0, 10.0, 110.0, 110.0],
        [11.0, 11.0, 111.0, 111.0],   # IoU ~0.96 with the first
        [12.0, 12.0, 112.0, 112.0],
        [300.0, 300.0, 400.0, 400.0],  # disjoint
    ])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    ids = torch.zeros(4, dtype=torch.long)

    keep = nms(boxes, scores, 0.6)
    print(f"NMS same-class: kept {keep.tolist()} (expect [0, 3])")
    assert keep.tolist() == [0, 3], keep.tolist()

    # ---- class-aware NMS must NOT suppress across classes ----------------
    ids_mixed = torch.tensor([0, 1, 2, 0])
    keep_aware = batched_nms(boxes, scores, ids_mixed, 0.6, class_agnostic=False)
    print(f"NMS class-aware: kept {sorted(keep_aware.tolist())} (expect all 4)")
    assert len(keep_aware) == 4, keep_aware.tolist()

    keep_agnostic = batched_nms(boxes, scores, ids_mixed, 0.6, class_agnostic=True)
    print(f"NMS class-agnostic: kept {sorted(keep_agnostic.tolist())} (expect [0, 3])")
    assert sorted(keep_agnostic.tolist()) == [0, 3], keep_agnostic.tolist()

    # ---- THE HEADLINE TEST: the three-levels-one-object duplicate --------
    # Build a fake model output where the SAME object is detected at all
    # three strides, exactly the situation the old evaluator mishandles.
    # Object: center (208, 208), half-extent 40px -> box [168,168,248,248].
    C = 16
    outputs = {}
    for stride, grid in ((8, 52), (16, 26), (32, 13)):
        phi = torch.full((1, C, grid, grid), -10.0)   # logits -> prob ~0
        box = torch.zeros((1, 4, grid, grid))

        # Cell containing pixel 208 at this stride.
        g = int(208 // stride)
        phi[0, 3, g, g] = 2.0                          # class 3, prob ~0.88
        # dx, dy chosen so the decoded center lands exactly on 208.
        box[0, 0, g, g] = (208.0 - (g + 0.5) * stride) / stride
        box[0, 1, g, g] = (208.0 - (g + 0.5) * stride) / stride
        # log(half_extent / stride) so half-extent is 40px at every stride.
        box[0, 2, g, g] = torch.log(torch.tensor(40.0 / stride))
        box[0, 3, g, g] = torch.log(torch.tensor(40.0 / stride))

        outputs[stride] = (phi, box, torch.zeros_like(box))

    cfg_no_nms = DecodeConfig(score_thresh=0.05, nms_iou=1.0, mode="peak")
    cfg_nms = DecodeConfig(score_thresh=0.05, nms_iou=0.6, mode="peak")

    # nms_iou=1.0 suppresses nothing, reproducing the old behaviour.
    without = decode_image(outputs, 0, cfg_no_nms)
    with_nms = decode_image(outputs, 0, cfg_nms)

    print()
    print(f"one object, three levels, NO suppression : {len(without['boxes'])} boxes")
    for b, s in zip(without["boxes"].tolist(), without["scores"].tolist()):
        print(f"    [{b[0]:6.1f} {b[1]:6.1f} {b[2]:6.1f} {b[3]:6.1f}]  score {s:.3f}"
              "   <-- 2 of these 3 were counted as FALSE POSITIVES")
    print(f"one object, three levels, WITH NMS       : {len(with_nms['boxes'])} boxes")
    for b, s in zip(with_nms["boxes"].tolist(), with_nms["scores"].tolist()):
        print(f"    [{b[0]:6.1f} {b[1]:6.1f} {b[2]:6.1f} {b[3]:6.1f}]  score {s:.3f}")

    assert len(without["boxes"]) == 3, "expected one duplicate per level"
    assert len(with_nms["boxes"]) == 1, (
        f"NMS should collapse 3 duplicates to 1, got {len(with_nms['boxes'])}"
    )
    # 2 of 3 would have been counted as false positives. That is the bug.

    # ---- decoded geometry must be correct at EVERY stride ---------------
    expected = [168.0, 168.0, 248.0, 248.0]
    got = with_nms["boxes"][0].tolist()
    print(f"\ndecoded box {['%.1f' % v for v in got]} (expect {expected})")
    for g_, e_ in zip(got, expected):
        assert abs(g_ - e_) < 1e-3, (got, expected)
    assert int(with_nms["labels"][0]) == 3, with_nms["labels"]

    # Each level individually must decode to the same box — this is what
    # proves the per-level log-extent convention is handled consistently.
    for stride in (8, 16, 32):
        single = decode_image({stride: outputs[stride]}, 0, cfg_nms)
        b = single["boxes"][0].tolist()
        print(f"  stride {stride:2d} alone -> [{b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}]")
        for g_, e_ in zip(b, expected):
            assert abs(g_ - e_) < 1e-3, (stride, b)

    # ---- extent convention adapter --------------------------------------
    raw = torch.tensor([5.0])
    log_from_log = extents_to_log(raw, "log")
    log_from_sp = extents_to_log(raw, "softplus")
    log_from_lin = extents_to_log(raw, "linear")
    print(
        f"\nextent adapter on raw=5.0: log={float(log_from_log):.4f} "
        f"softplus={float(log_from_sp):.4f} linear={float(log_from_lin):.4f}"
    )
    # softplus(5) = 5.0067, so log(softplus(5)) ~= log(5) = 1.6094
    assert abs(float(log_from_lin) - 1.6094) < 1e-3
    assert abs(float(log_from_sp) - 1.6107) < 1e-3
    # These three differ, which is exactly why the old scripts disagreed.
    assert float(log_from_log) != float(log_from_lin)

    # ---- empty output must not crash ------------------------------------
    dead = {
        s: (
            torch.full((1, C, g, g), -30.0),
            torch.zeros((1, 4, g, g)),
            torch.zeros((1, 4, g, g)),
        )
        for s, g in ((8, 52), (16, 26), (32, 13))
    }
    out = decode_image(dead, 0, cfg_nms)
    assert out["boxes"].shape == (0, 4), out["boxes"].shape
    print("all-background image -> 0 detections, no crash")

    print("\nOK: decode_v3 self-check passed.")
