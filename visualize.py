"""
Visualization tools for UAV-YOLO paper figures.
- GradCAM heatmaps
- Detection result comparison
- Metrics table generation
"""

import os
import json
import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).parent
RUNS_DIR = PROJECT_DIR / "runs" / "detect"


def collect_results(runs_dir=RUNS_DIR):
    """Collect metrics from all experiment runs and generate a comparison table."""
    results = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            continue

        # Read the last line of results.csv for final metrics
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                continue
            last_row = rows[-1]

        # Also read args.yaml for model info
        args_path = run_dir / "args.yaml"
        model_info = ""
        if args_path.exists():
            with open(args_path, "r") as f:
                for line in f:
                    if line.strip().startswith("model:"):
                        model_info = line.strip().split(":", 1)[1].strip()
                        break

        # Extract key metrics
        metrics = {
            "name": run_dir.name,
            "model": model_info,
            "mAP50": _safe_float(last_row, "metrics/mAP50(B)"),
            "mAP50-95": _safe_float(last_row, "metrics/mAP50-95(B)"),
            "precision": _safe_float(last_row, "metrics/precision(B)"),
            "recall": _safe_float(last_row, "metrics/recall(B)"),
        }
        results.append(metrics)

    # Print table
    if results:
        print(f"\n{'Model':<35} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
        print("-" * 75)
        for r in results:
            print(
                f"{r['name']:<35} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} "
                f"{r['precision']:>10.4f} {r['recall']:>8.4f}"
            )

    # Save to CSV
    out_path = PROJECT_DIR / "results_summary.csv"
    if results:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {out_path}")

    return results


def compare_detections(image_path, model_paths, names, save_dir=None):
    """Run multiple models on the same image and create a comparison figure.

    Args:
        image_path: Path to test image.
        model_paths: List of model weight paths (.pt files from runs/detect/*/weights/best.pt).
        names: Display names for each model.
        save_dir: Directory to save comparison image.
    """
    from PIL import Image, ImageDraw, ImageFont

    images = []
    for model_path, name in zip(model_paths, names):
        model = YOLO(model_path)
        results = model(image_path, imgsz=640, conf=0.25)[0]

        # Get annotated image
        annotated = results.plot()  # numpy array BGR
        annotated = annotated[:, :, ::-1]  # BGR -> RGB
        img = Image.fromarray(annotated)

        # Add model name label
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.rectangle([(0, 0), (len(name) * 16, 32)], fill="black")
        draw.text((4, 4), name, fill="white", font=font)
        images.append(img)

    # Stitch images horizontally
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    total_w = sum(widths)
    max_h = max(heights)

    comparison = Image.new("RGB", (total_w, max_h), "white")
    x_offset = 0
    for img in images:
        comparison.paste(img, (x_offset, 0))
        x_offset += img.width

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "detection_comparison.png")
        comparison.save(save_path, quality=95)
        print(f"Comparison saved to: {save_path}")

    return comparison


def generate_gradcam(model_path, image_path, save_dir=None):
    """Generate GradCAM heatmap for a YOLO model.

    Uses hooks on the last conv layer of the backbone to visualize
    which regions the model focuses on.
    """
    from ultralytics import YOLO
    import cv2

    model = YOLO(model_path)
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run prediction to get results
    results = model(image_path, imgsz=640, conf=0.25)
    annotated = results[0].plot()

    # Simple activation-based visualization (GradCAM-like)
    # Hook into the backbone's last layer
    activations = {}

    def hook_fn(module, input, output):
        activations["feat"] = output.detach()

    # Register hook on the backbone's last feature map
    backbone = model.model.model
    target_layer = None
    for i, layer in enumerate(backbone):
        if hasattr(layer, "cv2") or hasattr(layer, "conv"):
            target_layer = layer

    if target_layer is not None:
        handle = target_layer.register_forward_hook(hook_fn)
        _ = model(image_path, imgsz=640, conf=0.25)
        handle.remove()

        if "feat" in activations:
            feat = activations["feat"][0]  # (C, H, W)
            heatmap = feat.mean(dim=0).cpu().numpy()  # (H, W)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, "gradcam_heatmap.png"), overlay)
                cv2.imwrite(os.path.join(save_dir, "detection_result.png"), annotated)
                print(f"GradCAM saved to: {save_dir}")

            return overlay

    print("[WARN] Could not generate GradCAM - no suitable layer found.")
    return None


def _safe_float(row, key):
    """Safely extract a float from a CSV row."""
    val = row.get(key, "0").strip()
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["table", "compare", "gradcam"], default="table")
    parser.add_argument("--image", type=str, help="Image path for compare/gradcam")
    args = parser.parse_args()

    if args.action == "table":
        collect_results()
    elif args.action == "compare":
        if not args.image:
            print("Please provide --image path")
        else:
            # Auto-find best.pt from all runs
            model_paths = sorted(RUNS_DIR.glob("*/weights/best.pt"))
            names = [p.parent.parent.name for p in model_paths]
            compare_detections(args.image, model_paths, names,
                               save_dir=str(PROJECT_DIR / "figures"))
    elif args.action == "gradcam":
        if not args.image:
            print("Please provide --image path")
        else:
            model_paths = sorted(RUNS_DIR.glob("*/weights/best.pt"))
            for mp in model_paths:
                name = mp.parent.parent.name
                generate_gradcam(str(mp), args.image,
                                 save_dir=str(PROJECT_DIR / "figures" / name))
