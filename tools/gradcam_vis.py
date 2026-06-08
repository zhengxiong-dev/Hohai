"""
Grad-CAM visualization for UAV-YOLO vs YOLO11s baseline on VisDrone.

Generates class-activation heatmaps to provide visual evidence that the EMA
attention module focuses the network on small foreground objects and suppresses
background clutter. Produces side-by-side heatmaps for the paper (Figure 7).

Usage:
    python tools/gradcam_vis.py \
        --weights runs/detect/visdrone_p2_ema_pt/weights/best.pt \
        --images <img1.jpg> <img2.jpg> ... \
        --outdir paper/gradcam_out

If --images is omitted, a few representative validation images are chosen
automatically (those with many small annotations).
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
VAL_IMG_DIR = PROJECT_DIR / "datasets" / "VisDrone-YOLO" / "images" / "val"
VAL_LBL_DIR = PROJECT_DIR / "datasets" / "VisDrone-YOLO" / "labels" / "val"
IMG_EXTS = (".jpg", ".jpeg", ".png")


def pick_busy_images(n=4):
    """Pick validation images containing the most small-object annotations."""
    scored = []
    for lbl in VAL_LBL_DIR.glob("*.txt"):
        cnt = 0
        for line in lbl.read_text().strip().splitlines():
            p = line.split()
            if len(p) == 5 and float(p[3]) * float(p[4]) < 0.0025:  # tiny boxes
                cnt += 1
        if cnt:
            scored.append((cnt, lbl.stem))
    scored.sort(reverse=True)
    out = []
    for _, stem in scored[:n]:
        for ext in IMG_EXTS:
            f = VAL_IMG_DIR / f"{stem}{ext}"
            if f.exists():
                out.append(str(f))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--outdir", default=str(PROJECT_DIR / "paper" / "gradcam_out"))
    ap.add_argument("--layer", type=int, default=-2,
                    help="negative index into model.model layers for the target")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    images = args.images or pick_busy_images(4)
    print(f"[gradcam] target images: {images}")

    from ultralytics import YOLO
    yolo = YOLO(args.weights)
    model = yolo.model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Hook the last backbone/neck conv-bearing layer for activations+gradients.
    layers = list(model.model)
    target_layer = layers[args.layer]
    feats, grads = {}, {}

    def fwd_hook(_m, _i, o):
        feats["v"] = o.detach() if not isinstance(o, (list, tuple)) else o[0].detach()

    def bwd_hook(_m, gin, gout):
        g = gout[0]
        grads["v"] = g.detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    for img_path in images:
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[skip] cannot read {img_path}")
            continue
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (640, 640)).astype(np.float32) / 255.0
        x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
        x.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        out = model(x)
        # Aggregate detection responses into a scalar objective.
        if isinstance(out, (list, tuple)):
            score = sum(o.abs().mean() for o in out if torch.is_tensor(o))
        else:
            score = out.abs().mean()
        score.backward()

        if "v" not in feats or "v" not in grads:
            print(f"[warn] no activation captured for {img_path}")
            continue
        act = feats["v"][0]                       # (C,h,w)
        grad = grads["v"][0]                      # (C,h,w)
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(0))  # (h,w)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (W, H))
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)

        stem = Path(img_path).stem
        cv2.imwrite(str(outdir / f"{stem}_orig.jpg"), bgr)
        cv2.imwrite(str(outdir / f"{stem}_gradcam.jpg"), overlay)
        print(f"[ok] {stem} -> {outdir}")

    h1.remove()
    h2.remove()
    print(f"[done] heatmaps saved to {outdir}")


if __name__ == "__main__":
    main()
