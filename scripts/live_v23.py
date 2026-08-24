import os
import sys
import cv2
import time
import torch
import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

from models.fieldnet_v3 import FieldNetV3
from postprocess.decode_v3 import DecodeConfig, decode_batch
from data.taxonomy import canonical_classes


# ============================================================
# FIELDNET V23 LIVE CAMERA
# ============================================================

IMAGE_SIZE = 416
CHANNELS = 128
BACKBONE = "resnet34"

CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "outputs", "v3_ft_mask", "best_ema.pt"
)

CONF_THRESHOLD = 0.30
DECODE_THRESHOLD = 0.02
NMS_IOU = 0.60
MAX_DETECTIONS = 100

CAMERA_INDEX = 0

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = canonical_classes(merge=True)

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("FIELDNET V23 LIVE CAMERA")
    print("=" * 70)

    print(f"checkpoint : {CKPT}")
    print(f"device     : {DEVICE}")
    print(f"classes    : {NUM_CLASSES}")
    print(f"backbone   : {BACKBONE}")
    print(f"image size : {IMAGE_SIZE}")
    print()

    if not os.path.isfile(CKPT):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CKPT}"
        )

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False,
    )

    model = FieldNetV3(
        num_classes=NUM_CLASSES,
        channels=CHANNELS,
        backbone=BACKBONE,
        pretrained=False,
    )

    # best_ema.pt may itself be a state dict or checkpoint dictionary.
    if isinstance(checkpoint, dict):

        if "ema_state_dict" in checkpoint:
            state = checkpoint["ema_state_dict"]
            print("weights     : ema_state_dict")

        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            print("weights     : model_state_dict")

        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
            print("weights     : state_dict")

        else:
            state = checkpoint
            print("weights     : checkpoint")

    else:
        state = checkpoint
        print("weights     : raw")

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing:
        print(
            f"WARNING: {len(missing)} missing keys"
        )
        print(missing[:5])

    if unexpected:
        print(
            f"WARNING: {len(unexpected)} unexpected keys"
        )
        print(unexpected[:5])

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            "Do not run camera inference."
        )

    model = model.to(DEVICE)
    model.eval()

    print()
    print("MODEL LOADED SUCCESSFULLY")
    print()
    print("Classes:")

    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i:2d}: {name}")

    print()

    return model


# ============================================================
# FRAME PREPROCESSING
# ============================================================

def frame_to_tensor(frame):

    resized = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    array = np.asarray(
        rgb,
        dtype=np.float32,
    )

    tensor = torch.from_numpy(array)

    tensor = (
        tensor
        .permute(2, 0, 1)
        .contiguous()
        / 255.0
    )

    return tensor


# ============================================================
# DRAW
# ============================================================

def draw_detections(frame, detections):

    h, w = frame.shape[:2]

    sx = w / float(IMAGE_SIZE)
    sy = h / float(IMAGE_SIZE)

    for box, score, label in zip(
        detections["boxes"],
        detections["scores"],
        detections["labels"],
    ):

        score = float(score)

        if score < CONF_THRESHOLD:
            continue

        box = box.tolist()

        x1, y1, x2, y2 = box

        x1 = int(max(0, min(w - 1, x1 * sx)))
        y1 = int(max(0, min(h - 1, y1 * sy)))
        x2 = int(max(0, min(w - 1, x2 * sx)))
        y2 = int(max(0, min(h - 1, y2 * sy)))

        cls = int(label)

        if 0 <= cls < NUM_CLASSES:
            name = CLASS_NAMES[cls]
        else:
            name = f"class_{cls}"

        text = f"{name} {score:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )

        text_y = max(
            th + 8,
            y1,
        )

        cv2.rectangle(
            frame,
            (x1, text_y - th - 8),
            (x1 + tw + 8, text_y + baseline),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x1 + 4, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# INFERENCE
# ============================================================

@torch.inference_mode()
def detect(model, frame):

    tensor = frame_to_tensor(frame)

    batch = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    outputs = model(batch)

    cfg = DecodeConfig(
        score_thresh=DECODE_THRESHOLD,
        nms_iou=NMS_IOU,
        topk_per_level=1000,
        max_detections=MAX_DETECTIONS,
        mode="peak",
        image_size=IMAGE_SIZE,
        class_agnostic_nms=False,
        phi_is_logits=True,
        box_extent="log",
    )

    decoded = decode_batch(
        outputs,
        cfg,
    )

    return decoded[0]


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():

        # fallback
        cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        DISPLAY_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        DISPLAY_HEIGHT,
    )

    print("=" * 70)
    print("CAMERA STARTED")
    print("Press Q to quit.")
    print("=" * 70)

    previous_time = time.time()
    frame_count = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Could not read camera frame.")
            break

        start = time.time()

        detections = detect(
            model,
            frame,
        )

        draw_detections(
            frame,
            detections,
        )

        elapsed = max(
            time.time() - start,
            1e-6,
        )

        fps = 1.0 / elapsed

        cv2.putText(
            frame,
            f"FieldNet v23 | {DEVICE} | FPS {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Detections: {len(detections['scores'])}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "FieldNet V23 - Live Camera",
            frame,
        )

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("Camera stopped.")


if __name__ == "__main__":
    main()