"""
Scale-wise AP evaluation (COCO style) for UAV-YOLO on VisDrone2019-DET val.

Builds a COCO-format ground-truth file from the YOLO labels of the VisDrone
validation split, runs inference with one or more trained models, and reports
the COCO metrics — crucially AP_small / AP_medium / AP_large — so the paper can
demonstrate that the improvements concentrate on SMALL objects.

COCO area thresholds:
    small  : area < 32^2  (1024 px^2)
    medium : 32^2 <= area < 96^2
    large  : area >= 96^2

Usage:
    python tools/eval_scale_ap.py \
        --weights runs/detect/visdrone_yolo11s_baseline/weights/best.pt \
                  runs/detect/visdrone_p2_ema_pt/weights/best.pt \
        --names   Baseline  UAV-YOLO
"""

import argparse
import json
import contextlib
import io
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
VAL_IMG_DIR = PROJECT_DIR / "datasets" / "VisDrone-YOLO" / "images" / "val"
VAL_LBL_DIR = PROJECT_DIR / "datasets" / "VisDrone-YOLO" / "labels" / "val"
GT_JSON = PROJECT_DIR / "tools" / "_visdrone_val_coco_gt.json"

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def build_coco_gt():
    """Build (and cache) a COCO ground-truth json from VisDrone YOLO labels."""
    images, annotations = [], []
    categories = [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]
    ann_id = 1

    img_files = sorted(
        p for p in VAL_IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    img_id_map = {}
    for img_id, img_path in enumerate(img_files, 1):
        with Image.open(img_path) as im:
            w, h = im.size
        img_id_map[img_path.stem] = (img_id, w, h)
        images.append(
            {"id": img_id, "file_name": img_path.name, "width": w, "height": h}
        )

        lbl_path = VAL_LBL_DIR / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            x, y = cx - bw / 2.0, cy - bh / 2.0
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    GT_JSON.write_text(json.dumps(coco))
    print(f"[GT] {len(images)} images, {len(annotations)} annotations -> {GT_JSON.name}")
    return img_id_map


def collect_detections(weights, img_id_map):
    """Run a model over the val images and return COCO-format detections."""
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(
        source=str(VAL_IMG_DIR),
        imgsz=640,
        conf=0.001,
        iou=0.7,
        max_det=300,
        stream=True,
        verbose=False,
        device=0,
    )
    detections = []
    for r in results:
        stem = Path(r.path).stem
        if stem not in img_id_map:
            continue
        img_id = img_id_map[stem][0]
        b = r.boxes
        if b is None or b.shape[0] == 0:
            continue
        xywh = b.xywh.cpu().numpy()      # center x,y,w,h (abs px)
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        for (cx, cy, bw, bh), sc, c in zip(xywh, conf, cls):
            detections.append({
                "image_id": int(img_id),
                "category_id": int(c),
                "bbox": [float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)],
                "score": float(sc),
            })
    return detections


def evaluate(detections):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(GT_JSON))
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats  # 12 COCO metrics
    return {
        "mAP50-95": s[0] * 100,
        "mAP50": s[1] * 100,
        "mAP75": s[2] * 100,
        "AP_small": s[3] * 100,
        "AP_medium": s[4] * 100,
        "AP_large": s[5] * 100,
        "AR_small": s[9] * 100,
        "AR_medium": s[10] * 100,
        "AR_large": s[11] * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    args = ap.parse_args()

    names = args.names or [Path(w).parent.parent.name for w in args.weights]
    assert len(names) == len(args.weights), "names count must match weights count"

    img_id_map = build_coco_gt()

    rows = []
    for name, w in zip(names, args.weights):
        print(f"\n[EVAL] {name}: {w}")
        dets = collect_detections(w, img_id_map)
        print(f"       collected {len(dets)} detections")
        metrics = evaluate(dets)
        rows.append((name, metrics))

    cols = ["mAP50-95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large",
            "AR_small", "AR_medium", "AR_large"]
    print("\n" + "=" * 100)
    header = f"{'Model':<22}" + "".join(f"{c:>11}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<22}" + "".join(f"{m[c]:>11.2f}" for c in cols))
    print("=" * 100)

    out = PROJECT_DIR / "tools" / "scale_ap_results.json"
    out.write_text(json.dumps({n: m for n, m in rows}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
