"""
Collect best metrics from all UAV-YOLO experiments into a single summary table.

Reads each experiment's results.csv (no GPU needed) and reports the best-epoch
P / R / mAP50 / mAP50-95 for every run directory under runs/detect. Use this to
assemble the ablation and comparison tables for the paper once training is done.

Usage:
    python tools/collect_results.py
"""

import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "runs" / "detect"

# Display order and friendly labels for the experiments we care about.
GROUPS = [
    ("--- VisDrone: fair (pretrained-backbone) ---", [
        ("visdrone_yolo11s_baseline", "Baseline YOLO11s (PT)"),
        ("visdrone_p2_pt",            "+ P2 (PT)"),
        ("visdrone_ema_pt",           "+ EMA (PT)"),
        ("visdrone_bifpn_pt",         "+ BiFPN (PT)"),
        ("visdrone_p2_bifpn_pt",      "+ P2 + BiFPN (PT)"),
        ("visdrone_p2_ema_pt",        "+ P2 + EMA = UAV-YOLO (PT)"),
        ("visdrone_uav_yolo_pt",      "+ P2 + BiFPN + EMA (PT)"),
    ]),
    ("--- VisDrone: comparison SOTA ---", [
        ("visdrone_yolov8s",  "YOLOv8s"),
        ("visdrone_yolov9s",  "YOLOv9s"),
        ("visdrone_yolov10s", "YOLOv10s"),
        ("visdrone_rtdetr_l", "RT-DETR-l"),
    ]),
    ("--- VisDrone: old from-scratch (for reference) ---", [
        ("visdrone_ablation_p2",       "P2 (scratch)"),
        ("visdrone_ablation_ema",      "EMA (scratch)"),
        ("visdrone_ablation_bifpn",    "BiFPN (scratch)"),
        ("visdrone_ablation_p2_bifpn", "P2+BiFPN (scratch)"),
        ("visdrone_ablation_p2_ema",   "P2+EMA (scratch)"),
        ("visdrone_uav_yolo2",         "P2+BiFPN+EMA (scratch)"),
    ]),
    ("--- AI-TOD generalization ---", [
        ("aitod_yolo11s_baseline", "AI-TOD Baseline"),
        ("aitod_p2_ema_pt",        "AI-TOD UAV-YOLO (PT)"),
        ("aitod_uav_yolo",         "AI-TOD UAV-YOLO (scratch,old)"),
    ]),
    ("--- NWPU generalization ---", [
        ("nwpu_yolo11s_baseline", "NWPU Baseline"),
        ("nwpu_uav_yolo",         "NWPU UAV-YOLO"),
    ]),
]

KEY_MAP = {
    "P":        "metrics/precision(B)",
    "R":        "metrics/recall(B)",
    "mAP50":    "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def best_row(csv_path):
    """Return the row (dict) with the highest mAP50-95."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    key = KEY_MAP["mAP50-95"]
    best = max(rows, key=lambda r: float(r.get(key, 0) or 0))
    return best


def fmt(row):
    if row is None:
        return None
    out = {}
    for label, col in KEY_MAP.items():
        try:
            out[label] = float(row[col]) * 100
        except (KeyError, ValueError, TypeError):
            out[label] = None
    out["epoch"] = row.get("epoch", "?")
    return out


def main():
    print(f"{'Experiment':<32}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}{'@ep':>6}")
    print("=" * 73)
    for header, items in GROUPS:
        print(f"\n{header}")
        for dirname, label in items:
            csv_path = RUNS_DIR / dirname / "results.csv"
            if not csv_path.exists():
                print(f"{label:<32}{'(not run / training...)':>41}")
                continue
            m = fmt(best_row(csv_path))
            if m is None:
                print(f"{label:<32}{'(empty csv)':>41}")
                continue
            def s(v):
                return f"{v:.2f}" if isinstance(v, float) else "—"
            print(f"{label:<32}{s(m['P']):>8}{s(m['R']):>8}"
                  f"{s(m['mAP50']):>9}{s(m['mAP50-95']):>10}{str(m['epoch']):>6}")
    print("\n" + "=" * 73)
    print("Note: 'PT' = pretrained YOLO11s backbone loaded (fair comparison).")


if __name__ == "__main__":
    main()
