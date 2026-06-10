"""
UAV-YOLO / AFA-YOLO11: Complete Experiment Runner
==================================================

Usage:
    # ===== Baseline & UAV-YOLO =====
    python train.py --exp baseline          # YOLO11s baseline
    python train.py --exp uav_yolo_pt       # UAV-YOLO (P2+BiFPN+EMA)

    # ===== AFA-YOLO11 (Recommended for thesis) =====
    python train.py --exp sgfa_pt           # AFA: SGFA only (P2+SGFA)
    python train.py --exp afa_yolo_safm_p2_pt  # AFA: SGFA+SAFM(P2)
    python train.py --exp afa_yolo_final_pt # AFA: Final version (recommended)

    # ===== Ablation Studies =====
    python train.py --exp ablation_p2       # Ablation: P2 only
    python train.py --exp ablation_bifpn    # Ablation: BiFPN only
    python train.py --exp ablation_ema      # Ablation: EMA only
    python train.py --exp ablation_p2_bifpn # Ablation: P2+BiFPN
    python train.py --exp ablation_p2_ema   # Ablation: P2+EMA

    # ===== Comparison Experiments =====
    python train.py --exp compare_yolov8    # Comparison: YOLOv8s
    python train.py --exp compare_yolov9    # Comparison: YOLOv9s
    python train.py --exp compare_yolov10   # Comparison: YOLOv10s
    python train.py --exp compare_rtdetr    # Comparison: RT-DETR-l

    # ===== Generalization =====
    python train.py --exp nwpu_uav_yolo     # Generalization: UAV-YOLO on NWPU
    python train.py --exp nwpu_baseline     # Generalization: YOLO11s on NWPU
    python train.py --exp aitod_baseline    # Generalization: YOLO11s on AI-TOD
    python train.py --exp aitod_uav_yolo    # Generalization: UAV-YOLO on AI-TOD

    # ===== Run All =====
    python train.py --exp all               # Run ALL experiments sequentially
"""

import argparse
import os
import sys
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
DATASETS_DIR = PROJECT_DIR / "datasets"
VISDRONE_YAML = DATASETS_DIR / "VisDrone-YOLO" / "visdrone.yaml"
NWPU_YAML = DATASETS_DIR / "NWPU-YOLO" / "nwpu.yaml"
AITOD_YAML = DATASETS_DIR / "AI-TOD-YOLO" / "aitod.yaml"

# Common training settings for 4090 (24GB)
COMMON_ARGS = dict(
    imgsz=640,
    epochs=200,
    batch=8,
    device=0,
    workers=8,
    patience=50,       # early stopping patience
    save=True,
    save_period=50,
    val=True,
    plots=True,
    cos_lr=True,       # cosine learning rate scheduler
    close_mosaic=20,   # disable mosaic for last 20 epochs
    seed=42,
)


def get_experiment_config(exp_name: str) -> dict:
    """Return training configuration for a given experiment name."""

    experiments = {
        # ==================== BASELINE ====================
        "baseline": dict(
            model="yolo11s.pt",
            data=str(VISDRONE_YAML),
            name="visdrone_yolo11s_baseline",
            **COMMON_ARGS,
        ),

        # ==================== OUR MODEL ====================
        "uav_yolo": dict(
            model=str(MODELS_DIR / "uav-yolo.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_uav_yolo",
            pretrained=False,
            **COMMON_ARGS,
        ),

        # ==================== ABLATION STUDIES ====================
        "ablation_p2": dict(
            model=str(MODELS_DIR / "ablation_p2.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_ablation_p2",
            pretrained=False,
            **COMMON_ARGS,
        ),
        "ablation_bifpn": dict(
            model=str(MODELS_DIR / "ablation_bifpn.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_ablation_bifpn",
            pretrained=False,
            **COMMON_ARGS,
        ),
        "ablation_ema": dict(
            model=str(MODELS_DIR / "ablation_ema.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_ablation_ema",
            pretrained=False,
            **COMMON_ARGS,
        ),
        "ablation_p2_bifpn": dict(
            model=str(MODELS_DIR / "ablation_p2_bifpn.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_ablation_p2_bifpn",
            pretrained=False,
            **COMMON_ARGS,
        ),
        "ablation_p2_ema": dict(
            model=str(MODELS_DIR / "ablation_p2_ema.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_ablation_p2_ema",
            pretrained=False,
            **COMMON_ARGS,
        ),

        # ============ FAIR (pretrained-backbone) RE-RUNS on VisDrone ============
        # These variants load the YOLO11s COCO-pretrained backbone into the
        # improved architectures (layer-name/shape matching), so the comparison
        # against the pretrained baseline/competitors is fair. The added P2 head
        # and EMA layers are randomly initialized and learned during training.
        "p2_pt": dict(
            model=str(MODELS_DIR / "ablation_p2.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_p2_pt",
            load="yolo11s.pt",
            **COMMON_ARGS,
        ),
        "ema_pt": dict(
            model=str(MODELS_DIR / "ablation_ema.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_ema_pt",
            load="yolo11s.pt",
            **COMMON_ARGS,
        ),
        "bifpn_pt": dict(
            model=str(MODELS_DIR / "ablation_bifpn.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_bifpn_pt",
            load="yolo11s.pt",
            **COMMON_ARGS,
        ),
        "p2_bifpn_pt": dict(
            model=str(MODELS_DIR / "ablation_p2_bifpn.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_p2_bifpn_pt",
            load="yolo11s.pt",
            **COMMON_ARGS,
        ),
        "p2_ema_pt": dict(
            model=str(MODELS_DIR / "ablation_p2_ema.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_p2_ema_pt",
            load="yolo11s.pt",
            **COMMON_ARGS,
        ),
        "uav_yolo_pt": dict(
            model=str(MODELS_DIR / "uav-yolo.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_uav_yolo_pt",
            load="yolo11s.pt",
            **COMMON_ARGS,
        ),

        # ==================== AFA-YOLO11 EXPERIMENTS ====================
        # AFA-YOLO11: Aligned Fusion Adaptive YOLO11
        # Progressive experiments: SGFA only → SGFA+SAFM(P2) → Final

        "sgfa_pt": dict(
            model=str(MODELS_DIR / "afa-yolo-sgfa.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_sgfa_pt",
            load="yolo11s.pt",
            **COMMON_ARGS,
        ),
        "afa_yolo_safm_p2_pt": dict(
            model=str(MODELS_DIR / "afa-yolo-sgfa-safm-p2.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_afa_yolo_safm_p2_pt",
            load="yolo11s.pt",
            **COMMON_ARGS,
        ),
        "afa_yolo_final_pt": dict(
            model=str(MODELS_DIR / "afa-yolo-final.yaml"),
            data=str(VISDRONE_YAML),
            name="visdrone_afa_yolo_final_pt",
            load="yolo11s.pt",
            **COMMON_ARGS,
        ),

        # ==================== COMPARISON EXPERIMENTS ====================
        "compare_yolov8": dict(
            model="yolov8s.pt",
            data=str(VISDRONE_YAML),
            name="visdrone_yolov8s",
            **COMMON_ARGS,
        ),
        "compare_yolov9": dict(
            model="yolov9s.pt",
            data=str(VISDRONE_YAML),
            name="visdrone_yolov9s",
            **COMMON_ARGS,
        ),
        "compare_yolov10": dict(
            model="yolov10s.pt",
            data=str(VISDRONE_YAML),
            name="visdrone_yolov10s",
            **COMMON_ARGS,
        ),
        "compare_rtdetr": dict(
            model="rtdetr-l.pt",
            data=str(VISDRONE_YAML),
            name="visdrone_rtdetr_l",
            **COMMON_ARGS,
        ),

        # ==================== GENERALIZATION (NWPU VHR-10) ====================
        "nwpu_baseline": dict(
            model="yolo11s.pt",
            data=str(NWPU_YAML),
            name="nwpu_yolo11s_baseline",
            **{**COMMON_ARGS, "epochs": 200, "batch": 16},
        ),
        "nwpu_uav_yolo": dict(
            model=str(MODELS_DIR / "uav-yolo.yaml"),
            data=str(NWPU_YAML),
            name="nwpu_uav_yolo",
            pretrained=False,
            **{**COMMON_ARGS, "epochs": 200, "batch": 16},
        ),

        # ==================== GENERALIZATION (AI-TOD) ====================
        "aitod_baseline": dict(
            model="yolo11s.pt",
            data=str(AITOD_YAML),
            name="aitod_yolo11s_baseline",
            **{**COMMON_ARGS, "epochs": 200, "batch": 16},
        ),
        "aitod_uav_yolo": dict(
            model=str(MODELS_DIR / "uav-yolo.yaml"),
            data=str(AITOD_YAML),
            name="aitod_uav_yolo",
            pretrained=False,
            **{**COMMON_ARGS, "epochs": 200, "batch": 4},
        ),
        # Fair re-run: P2+EMA final model with pretrained backbone on AI-TOD.
        "aitod_p2_ema_pt": dict(
            model=str(MODELS_DIR / "ablation_p2_ema.yaml"),
            data=str(AITOD_YAML),
            name="aitod_p2_ema_pt",
            load="yolo11s.pt",
            # batch=4: AI-TOD images are extremely dense and the P2 head's
            # 160x160 feature maps make batch=8 exceed 24 GB VRAM (spills to
            # shared memory -> ~20x slowdown). batch=4 fits within physical VRAM.
            **{**COMMON_ARGS, "epochs": 200, "batch": 4},
        ),
    }

    return experiments.get(exp_name)


# Order for running all experiments
ALL_EXPERIMENTS = [
    # Baseline & UAV-YOLO
    "baseline",
    "uav_yolo",
    # Ablation studies
    "ablation_p2",
    "ablation_bifpn",
    "ablation_ema",
    "ablation_p2_bifpn",
    "ablation_p2_ema",
    # Pretrained backbone experiments
    "p2_pt",
    "ema_pt",
    "bifpn_pt",
    "p2_bifpn_pt",
    "p2_ema_pt",
    "uav_yolo_pt",
    # AFA-YOLO11 experiments (NEW)
    "sgfa_pt",
    "afa_yolo_safm_p2_pt",
    "afa_yolo_final_pt",
    # Comparison experiments
    "compare_yolov8",
    "compare_yolov9",
    "compare_yolov10",
    "compare_rtdetr",
    # Generalization experiments
    "nwpu_baseline",
    "nwpu_uav_yolo",
    "aitod_baseline",
    "aitod_uav_yolo",
]


def run_experiment(exp_name: str):
    """Run a single experiment."""
    from ultralytics import YOLO

    config = get_experiment_config(exp_name)
    if config is None:
        print(f"[ERROR] Unknown experiment: {exp_name}")
        print(f"Available: {list(get_experiment_config.__code__.co_consts)}")
        return None

    model_path = config.pop("model")
    pretrained = config.pop("pretrained", True)
    load_ckpt = config.pop("load", None)
    exp_display_name = config.get("name", exp_name)

    print("=" * 70)
    print(f"  EXPERIMENT: {exp_display_name}")
    print(f"  Model:      {model_path}")
    print(f"  Dataset:    {config.get('data', 'N/A')}")
    if load_ckpt:
        print(f"  Load wts:   {load_ckpt} (transfer matching layers)")
    print("=" * 70)

    # Load model
    if model_path.endswith(".pt"):
        model = YOLO(model_path)
    else:
        model = YOLO(model_path, task="detect")
        # Transfer pretrained weights into the architecture by layer matching.
        # Only layers with identical name and shape are copied; new layers
        # (e.g. P2 head, EMA) remain randomly initialized.
        if load_ckpt:
            model.load(load_ckpt)

    # Train
    project_dir = str(PROJECT_DIR / "runs" / "detect")
    results = model.train(
        project=project_dir,
        **config,
    )

    print(f"\n[DONE] {exp_display_name} completed.")
    print(f"  Results saved to: {project_dir}/{exp_display_name}")
    return results


def main():
    parser = argparse.ArgumentParser(description="UAV-YOLO Experiment Runner")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name (see script header for options)",
    )
    args = parser.parse_args()

    if args.exp == "all":
        print(f"\n{'#' * 70}")
        print(f"  RUNNING ALL {len(ALL_EXPERIMENTS)} EXPERIMENTS")
        print(f"{'#' * 70}\n")
        for i, exp_name in enumerate(ALL_EXPERIMENTS, 1):
            print(f"\n[{i}/{len(ALL_EXPERIMENTS)}] Starting: {exp_name}")
            try:
                run_experiment(exp_name)
            except Exception as e:
                print(f"[ERROR] {exp_name} failed: {e}")
                continue
        print(f"\n{'#' * 70}")
        print("  ALL EXPERIMENTS COMPLETED")
        print(f"{'#' * 70}")
    else:
        run_experiment(args.exp)


if __name__ == "__main__":
    main()
