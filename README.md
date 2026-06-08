# UAV-YOLO

Official implementation of **UAV-YOLO: A Resolution-Attention Synergy Network for Real-Time Small Object Detection in Aerial Imagery**.

UAV-YOLO is built on YOLO11s and targets small object detection in drone imagery. It adds three coordinated components:

- a **P2 detection head** at stride 4 to recover spatial detail for tiny objects,
- a **weighted bidirectional feature fusion** (BiFPN-style) to balance cross-scale information, and
- an **Efficient Multi-Scale Attention (EMA)** module to refine features at low cost.

A controlled ablation under a fair pretrained-backbone protocol shows that these components are interdependent: attention or weighted fusion added alone gives little or no benefit, while the high-resolution P2 branch makes both effective.

## Main results (VisDrone2019 validation)

| Method | Params (M) | mAP50 (%) | mAP50-95 (%) | FPS |
|---|---|---|---|---|
| YOLO11s (baseline) | 9.46 | 46.00 | 27.61 | 37.6 |
| **UAV-YOLO (ours)** | 9.58 | **48.53** | **29.21** | 37.8 |

Under the COCO scale-wise protocol, small-object AP improves from 11.84% to 14.59% (a 23% relative gain) and small-object recall from 19.53% to 24.49%, while inference speed matches the baseline.

## Repository structure

```
models/
  uav-yolo.yaml          # full model: P2 + BiFPN + EMA
  ablation_p2.yaml       # ablation variants
  ablation_ema.yaml
  ablation_bifpn.yaml
  ablation_p2_ema.yaml
  ablation_p2_bifpn.yaml
  modules.py             # EMA and BiFPN_Concat module definitions
train.py                 # unified training / experiment runner
tools/
  convert_visdrone_to_yolo.py   # VisDrone -> YOLO format
  convert_nwpu_to_yolo.py       # NWPU VHR-10 -> YOLO format
  eval_scale_ap.py              # COCO scale-wise AP (small/medium/large)
  measure_fps.py                # latency / FPS measurement
  gradcam_vis.py                # Grad-CAM visualization
  collect_results.py            # aggregate results.csv files
prepare_aitod.py         # AI-TOD preparation
setup_env.py             # environment setup helper
requirements.txt
```

Datasets, trained weights, and run outputs are **not** included in this repository (see below).

## Installation

```bash
conda create -n uav-yolo python=3.11 -y
conda activate uav-yolo
pip install -r requirements.txt
```

Our experiments use `ultralytics==8.4.26`, `torch==2.6.0+cu124`, and `torchvision==0.21.0+cu124` on a single NVIDIA RTX 4090. Install the PyTorch build matching your CUDA version from https://pytorch.org.

## Datasets

Download the datasets from their official sources, then convert them to YOLO format.

- **VisDrone2019-DET**: https://github.com/VisDrone/VisDrone-Dataset
- **AI-TOD**: https://github.com/jwwangchn/AI-TOD
- **NWPU VHR-10**: https://gcheng-nwpu.github.io/

```bash
# VisDrone -> YOLO (edit the paths at the top of the script first)
python tools/convert_visdrone_to_yolo.py

# NWPU VHR-10 -> YOLO
python tools/convert_nwpu_to_yolo.py

# AI-TOD
python prepare_aitod.py
```

Each converter writes images, labels, and a dataset YAML under `datasets/`.

## Training

`train.py` initializes every model (baseline, ablation variants, and competing
methods) from the same pretrained YOLO11s backbone so that comparisons differ
only in architecture, not in initialization.

```bash
# train the full UAV-YOLO model on VisDrone
python train.py --exp uav_yolo_pt

# train the baseline
python train.py --exp baseline

# run all experiments sequentially
python train.py --exp all
```

Training settings: input size 640x640, SGD with initial learning rate 0.01,
momentum 0.937, weight decay 5e-4, batch size 16, default Ultralytics
augmentation.

## Evaluation

```bash
# scale-wise AP under the COCO protocol (small / medium / large)
python tools/eval_scale_ap.py \
  --weights runs/detect/visdrone_yolo11s_baseline/weights/best.pt \
            runs/detect/visdrone_uav_yolo_pt/weights/best.pt \
  --names Baseline UAV-YOLO

# latency / FPS at 640x640
python tools/measure_fps.py

# Grad-CAM visualization
python tools/gradcam_vis.py --weights runs/detect/visdrone_uav_yolo_pt/weights/best.pt
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{xu2026uavyolo,
  title   = {UAV-YOLO: A Resolution-Attention Synergy Network for Real-Time Small Object Detection in Aerial Imagery},
  author  = {Xu, Junhao and Li, Zenghui},
  journal = {IEEE Access},
  year    = {2026}
}
```

## Acknowledgements

This project builds on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics). The EMA module follows Ouyang et al. (ICASSP 2023) and the weighted fusion follows EfficientDet (Tan et al., CVPR 2020).

## License

Released under the MIT License. See [LICENSE](LICENSE).
