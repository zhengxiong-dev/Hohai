"""
Convert VisDrone2019-DET annotations to YOLO format and split train/val.

VisDrone format: bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
Categories: 0-ignored, 1-pedestrian, 2-people, 3-bicycle, 4-car, 5-van,
            6-truck, 7-tricycle, 8-awning-tricycle, 9-bus, 10-motor, 11-others

We skip category 0 (ignored) and 11 (others), remap remaining to 0-9.
"""

import os
import random
import shutil
from pathlib import Path
from PIL import Image

# Config
SRC_DIR = Path("/mnt/d/Student/XJH/datasets/VisDrone/VisDrone2019-DET-train")
DST_DIR = Path("/mnt/d/Student/XJH/datasets/VisDrone-YOLO")
VAL_RATIO = 0.15
SEED = 42

# Category mapping: VisDrone category -> YOLO class index
# Skip 0 (ignored) and 11 (others)
CAT_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def convert_annotation(ann_path, img_w, img_h):
    """Convert one VisDrone annotation file to YOLO format."""
    yolo_lines = []
    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_w = int(parts[2])
            bbox_h = int(parts[3])
            # score = int(parts[4])
            category = int(parts[5])

            if category not in CAT_MAP:
                continue
            if bbox_w <= 0 or bbox_h <= 0:
                continue

            cls_id = CAT_MAP[category]
            # Convert to YOLO format: cx, cy, w, h (normalized)
            cx = (bbox_left + bbox_w / 2) / img_w
            cy = (bbox_top + bbox_h / 2) / img_h
            nw = bbox_w / img_w
            nh = bbox_h / img_h

            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}") 
    return yolo_lines


def main():
    random.seed(SEED)

    img_dir = SRC_DIR / "images"
    ann_dir = SRC_DIR / "annotations"

    # Get all image files
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    print(f"Total images: {len(img_files)}")

    # Shuffle and split
    random.shuffle(img_files)
    val_count = int(len(img_files) * VAL_RATIO)
    val_files = set(f.stem for f in img_files[:val_count])
    train_files = set(f.stem for f in img_files[val_count:])

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create output dirs
    for split in ['train', 'val']:
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {'train': 0, 'val': 0, 'skipped': 0}

    for img_path in img_files:
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.txt"

        if not ann_path.exists():
            stats['skipped'] += 1
            continue

        # Get image dimensions
        img = Image.open(img_path)
        img_w, img_h = img.size

        # Convert annotation
        yolo_lines = convert_annotation(ann_path, img_w, img_h)

        # Determine split
        split = 'val' if stem in val_files else 'train'

        # Copy image
        shutil.copy2(img_path, DST_DIR / "images" / split / img_path.name)

        # Write YOLO label
        with open(DST_DIR / "labels" / split / f"{stem}.txt", 'w') as f:
            f.write('\n'.join(yolo_lines))

        stats[split] += 1

    print(f"\nConversion complete!")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")
    print(f"  Skipped: {stats['skipped']}")

    # Write dataset YAML
    yaml_content = f"""path: {DST_DIR}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = DST_DIR / "visdrone.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nDataset YAML written to: {yaml_path}")


if __name__ == "__main__":
    main()
