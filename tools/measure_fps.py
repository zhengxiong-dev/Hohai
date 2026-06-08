"""
Measure inference speed (FPS / latency) for UAV-YOLO vs baseline and competitors.

Reports preprocess/inference/postprocess latency and end-to-end FPS at 640x640
on the GPU, to support the "real-time UAV deployment" claim in the paper.

Usage:
    python tools/measure_fps.py
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
VAL_IMG_DIR = PROJECT_DIR / "datasets" / "VisDrone-YOLO" / "images" / "val"

MODELS = [
    ("YOLO11s (Baseline)", "runs/detect/visdrone_yolo11s_baseline/weights/best.pt"),
    ("UAV-YOLO (Ours)",    "runs/detect/visdrone_uav_yolo_pt/weights/best.pt"),
    ("YOLOv8s",            "runs/detect/visdrone_yolov8s/weights/best.pt"),
    ("YOLOv9s",            "runs/detect/visdrone_yolov9s/weights/best.pt"),
    ("YOLOv10s",           "runs/detect/visdrone_yolov10s/weights/best.pt"),
    ("RT-DETR-l",          "runs/detect/visdrone_rtdetr_l/weights/best.pt"),
]

WARMUP = 20
N = 200


def main():
    import torch
    from ultralytics import YOLO

    imgs = sorted(p for p in VAL_IMG_DIR.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:N + WARMUP]
    imgs = [str(p) for p in imgs]

    print(f"{'Model':<22}{'pre(ms)':>9}{'inf(ms)':>9}{'post(ms)':>10}{'total(ms)':>11}{'FPS':>8}")
    print("=" * 69)
    for name, w in MODELS:
        wp = PROJECT_DIR / w
        if not wp.exists():
            print(f"{name:<22}{'(weights missing)':>47}")
            continue
        model = YOLO(str(wp))
        # warmup
        for p in imgs[:WARMUP]:
            model.predict(p, imgsz=640, device=0, verbose=False)
        pre = inf = post = 0.0
        for p in imgs[WARMUP:WARMUP + N]:
            r = model.predict(p, imgsz=640, device=0, verbose=False)
            sp = r[0].speed  # dict ms
            pre += sp["preprocess"]; inf += sp["inference"]; post += sp["postprocess"]
        pre /= N; inf /= N; post /= N
        total = pre + inf + post
        fps = 1000.0 / total if total > 0 else 0
        print(f"{name:<22}{pre:>9.2f}{inf:>9.2f}{post:>10.2f}{total:>11.2f}{fps:>8.1f}")
    print("=" * 69)
    print(f"Measured on {N} images @ 640x640, GPU. FPS = 1000/total_latency.")


if __name__ == "__main__":
    main()
