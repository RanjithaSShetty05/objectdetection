# FieldNet — Real-Time Multi-Class Object Detection

<p align="center">
  <strong>FieldNet V3</strong><br>
  Lightweight real-time object detection for webcam-based visual recognition
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-FieldNet%20V3-blue" alt="FieldNet V3">
  <img src="https://img.shields.io/badge/Backbone-ResNet34-green" alt="ResNet34">
  <img src="https://img.shields.io/badge/Classes-14-orange" alt="14 Classes">
  <img src="https://img.shields.io/badge/Inference-Webcam-purple" alt="Webcam Inference">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="PyTorch">
  <img src="https://img.shields.io/github/license/RanjithaSShetty05/objectdetection" alt="License">
</p>

---

## Overview

**FieldNet** is a real-time multi-class object detection system designed to recognize common objects through a webcam.

The final version uses a custom **FieldNet V3** detection architecture with a **ResNet34 backbone** and supports **14 object categories** after resolving source-taxonomy collisions.

This repository contains the **final inference-ready version** of the project.

> **Important:** This repository is intentionally inference-only.  
> Training, evaluation, datasets, research experiments, and development artifacts are not required to run the final webcam detector.

---

## Key Features

- 🎥 Real-time webcam object detection
- ⚡ GPU-accelerated inference when CUDA is available
- 🧠 Custom FieldNet V3 detection architecture
- 🏗️ ResNet34 backbone
- 📦 14-class object vocabulary
- 🎯 Multi-scale detection heads
- 🔍 Confidence filtering and class-wise NMS
- 🖼️ Real-time bounding-box visualization
- 💻 CPU fallback when CUDA is unavailable
- 📁 Minimal deployment-oriented project structure
- 🔒 Trained checkpoint distributed separately through GitHub Releases

---

## Supported Classes

The final taxonomy contains **14 canonical classes**:

| ID | Class |
|---:|---|
| 0 | `person` |
| 1 | `chair` |
| 2 | `desk_table` |
| 3 | `laptop` |
| 4 | `mobile_phone` |
| 5 | `book_notebook` |
| 6 | `pen` |
| 7 | `pencil` |
| 8 | `bottle` |
| 9 | `bag` |
| 10 | `keyboard` |
| 11 | `mouse` |
| 12 | `monitor` |
| 13 | `window` |

### Taxonomy Resolution

The original pooled datasets contained naming collisions where the same or closely related objects were represented under different labels.

The final taxonomy therefore uses:

- `desk` + `table` → `desk_table`
- `book` + `notebook` → `book_notebook`

This produces a consistent 14-class vocabulary for the final model.

---

# Model

## FieldNet V3

The final checkpoint uses:

| Component | Configuration |
|---|---|
| Architecture | FieldNet V3 |
| Backbone | ResNet34 |
| Feature channels | 128 |
| Detection mode | Peak decoding |
| Input size | 416 × 416 |
| NMS | Class-wise |
| Checkpoint | `best_ema.pt` |
| Training artifacts | Not included |
| Dataset | Not required for inference |

The final checkpoint contains the trained model weights required for webcam inference.

---

# Repository Structure

```text
objectdetection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── outputs/
│   └── v3_ft_mask/
│       ├── best_ema.pt
│       └── config.json
│
├── scripts/
│   └── live_v23.py
│
└── src/
    ├── data/
    │   ├── __init__.py
    │   └── taxonomy.py
    │
    ├── losses/
    │   ├── __init__.py
    │   └── field_loss_v3.py
    │
    ├── models/
    │   ├── __init__.py
    │   ├── backbone.py
    │   ├── backbones_pretrained.py
    │   ├── fieldnet_v3.py
    │   ├── head_v3.py
    │   └── neck.py
    │
    └── postprocess/
        ├── __init__.py
        └── decode_v3.py
```

The repository intentionally excludes the original training pipeline, datasets, evaluation scripts, experimental checkpoints, and research artifacts.

---

# Requirements

The final inference application requires:

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- OpenCV

Install the dependencies with:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the appropriate CUDA-enabled PyTorch build for the target machine.

> CUDA availability is detected automatically. If CUDA is unavailable, the application falls back to CPU inference.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/RanjithaSShetty05/objectdetection.git
cd objectdetection
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Download the trained checkpoint

The trained checkpoint is distributed separately through the project's GitHub Release because it is a large binary file.

### Download

[Download FieldNet V3 v1.0.0](../../releases/tag/v1.0.0)

Download:

```text
best_ema.pt
```

## 4. Place the checkpoint

Create/verify this directory:

```text
outputs/
└── v3_ft_mask/
```

Place the downloaded model here:

```text
outputs/v3_ft_mask/best_ema.pt
```

The final structure should therefore contain:

```text
objectdetection/
│
├── outputs/
│   └── v3_ft_mask/
│       ├── best_ema.pt
│       └── config.json
│
├── scripts/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Run the Webcam Detector

From the project root:

### Windows

```powershell
python scripts\live_v23.py
```

### Linux / macOS

```bash
python scripts/live_v23.py
```

The application will automatically:

1. Load the FieldNet V3 architecture
2. Load the trained checkpoint
3. Detect whether CUDA is available
4. Open the default webcam
5. Capture live frames
6. Preprocess each frame
7. Run FieldNet V3 inference
8. Decode detections
9. Apply confidence filtering
10. Apply class-wise NMS
11. Draw bounding boxes and class confidence scores
12. Display the live result

---

# Webcam Controls

The live application uses the system's default camera:

```text
Camera index: 0
```

If the system has multiple cameras, the camera index can be changed in:

```text
scripts/live_v23.py
```

For example:

```python
CAMERA_INDEX = 1
```

may select another connected camera.

---

# Inference Pipeline

The final webcam pipeline is:

```text
Webcam
   │
   ▼
Frame Capture
   │
   ▼
Image Preprocessing
   │
   ▼
416 × 416 Input
   │
   ▼
FieldNet V3
   │
   ├── ResNet34 Backbone
   │
   ├── Feature Processing
   │
   └── Detection Head
   │
   ▼
Multi-Level Feature Predictions
   │
   ▼
Peak Decoding
   │
   ▼
Confidence Filtering
   │
   ▼
Class-wise NMS
   │
   ▼
Bounding Boxes + Scores
   │
   ▼
Live Visualization
```

---

# Model Output

Each detected object is displayed using:

```text
<class_name> <confidence>
```

Example:

```text
person 0.61
laptop 0.63
mobile_phone 0.47
keyboard 0.47
```

Bounding boxes are rendered directly on the webcam stream.

---

# Checkpoint

The final trained model is:

```text
best_ema.pt
```

The checkpoint is approximately **87 MB** and is therefore distributed through the GitHub Release rather than committed directly to the Git repository.

### Release

**FieldNet Final v1.0.0**

[View Release and Download Model](../../releases/tag/v1.0.0)

The expected checkpoint location after downloading is:

```text
outputs/v3_ft_mask/best_ema.pt
```

---

# Why the Model Is Not Stored in Git

The source repository contains the complete inference code, but the trained `.pt` checkpoint is distributed separately.

This keeps Git history lightweight and prevents the large binary model from being unnecessarily stored in every repository clone.

The model can always be obtained from the project's versioned release.

---

# What Is NOT Required

For webcam inference, you do **not** need:

- Training datasets
- Validation datasets
- Test datasets
- Annotation files
- Training scripts
- Evaluation scripts
- Experiment logs
- Research notebooks
- Previous checkpoints
- Hyperparameter search files
- PowerShell training scripts
- GPU training setup
- Model retraining

The final deployment only requires:

```text
Source code
+
Python dependencies
+
best_ema.pt
+
Webcam
```

---

# Performance Notes

The final model is intended for **real-time webcam inference**.

Performance depends on:

- GPU / CPU hardware
- CUDA availability
- Webcam resolution
- Number of detections in the scene
- Background complexity
- Object size and distance
- Lighting conditions

GPU inference is recommended for real-time use.

The application automatically reports the active inference device and live FPS.

---

# Important Practical Notes

### Object distance matters

Detection quality depends on the apparent size of an object in the camera frame.

Objects that occupy very few pixels may be harder to detect reliably than larger objects.

For best results:

- Move smaller objects closer to the camera
- Keep objects reasonably well illuminated
- Avoid severe motion blur
- Avoid extreme occlusion
- Keep the object inside the camera frame

### Multiple objects

The detector can identify multiple supported classes in the same frame.

For example:

```text
person
laptop
keyboard
mobile_phone
book_notebook
```

can be detected simultaneously.

---

# Troubleshooting

## Camera does not open

Check that the webcam is available and not being used by another application.

The default camera index is:

```python
CAMERA_INDEX = 0
```

Try another index if multiple cameras are connected.

---

## Checkpoint not found

Verify that the model exists at exactly:

```text
outputs/v3_ft_mask/best_ema.pt
```

From PowerShell:

```powershell
Test-Path outputs\v3_ft_mask\best_ema.pt
```

Expected result:

```text
True
```

---

## CUDA is unavailable

The application can run on CPU, although inference may be slower.

Verify CUDA availability with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints:

```text
True
```

CUDA is available to PyTorch.

---

## Missing Python package

Run:

```bash
pip install -r requirements.txt
```

If using a virtual environment, make sure it is activated before installing the dependencies.

---

# Reproducible Deployment

A new machine can reproduce the final inference environment using:

```text
1. Clone repository
2. Install Python
3. Install requirements
4. Download v1.0.0 checkpoint
5. Place best_ema.pt in outputs/v3_ft_mask/
6. Connect webcam
7. Run live_v23.py
```

No training or dataset preparation is required.

---

# Project Status

**Status: Final inference release**

The repository represents the finalized webcam inference package for FieldNet V3.

```text
Training        → Complete
Model           → Finalized
Inference       → Working
Webcam          → Supported
GPU             → Supported
CPU fallback    → Supported
Dataset         → Not required for deployment
Evaluation      → Not required for deployment
Release         → v1.0.0
```

---

# Version

## v1.0.0 — FieldNet Final

Initial final release containing:

- FieldNet V3 inference architecture
- ResNet34 backbone
- 14-class taxonomy
- Webcam inference pipeline
- Post-processing and detection decoding
- Final trained EMA checkpoint
- Deployment documentation

---

# License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for the full license text.

---

## Final Quick Start

For someone who just wants to run the detector:

```bash
git clone https://github.com/RanjithaSShetty05/objectdetection.git
cd objectdetection
pip install -r requirements.txt
```

Download:

```text
best_ema.pt
```

from:

[FieldNet Final v1.0.0 Release](../../releases/tag/v1.0.0)

Place it at:

```text
outputs/v3_ft_mask/best_ema.pt
```

Then run:

```powershell
python scripts\live_v23.py
```

### That's it.

**No dataset.  
No training.  
No evaluation.  
No configuration.  
Just install → download model → run webcam.**
