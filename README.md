# FieldNet — Real-Time Object Detection

FieldNet is a real-time computer vision system for detecting classroom and everyday objects using a custom FieldNet V3 object detection model.

The final model supports 14 object classes and can perform real-time detection through a webcam.

## Detected Classes

The final model detects:

1. Person
2. Chair
3. Desk / Table
4. Laptop
5. Mobile Phone
6. Book / Notebook
7. Pen
8. Pencil
9. Bottle
10. Bag
11. Keyboard
12. Mouse
13. Monitor
14. Window

## Project Structure

```text
fieldnet_final/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   └── live_v23.py
│
├── src/
│   ├── data/
│   ├── losses/
│   ├── models/
│   └── postprocess/
│
└── outputs/
    └── v3_ft_mask/
        ├── best_ema.pt
        └── config.json