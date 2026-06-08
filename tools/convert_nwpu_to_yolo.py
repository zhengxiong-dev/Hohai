"""
Convert NWPU VHR-10 annotations to YOLO format and split train/val.

NWPU format: (x1,y1),(x2,y2),class
Classes: 1-airplane, 2-ship, 3-storage tank, 4-baseball diamond, 5-tennis court,
         6-basketball court, 7-ground track field, 8-harbor, 9-bridge, 10-vehicle
Remap to 0-9.
"""

import os
import re
import random
import shutil
from pathlib import Path
from PIL import Image

SRC_DIR = Path("/mnt/d/YOLO-dataset/NWPU VHR-10/NWPU VHR-10 dataset")
DST_DIR = Path("/mnt/d/Student/XJH/datasets/NWPU-YOLO")
VAL_RATIO = 0.2
SEED = 42

CLASS_NAMES = [
    "airplane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "vehicle"
]


def parse_nwpu_line(line):
    """Parse NWPU annotation line: (x1,y1),(x2,y2),class"""
    line = line.strip()
    if not line:
        return None
    # Pattern: (x1,y1),(x2,y2),class
    match = re.match(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)', line)
    if not match:
        return None
    x1, y1, x2, y2, cls = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))
    return x1, y1, x2, y2, cls


def main():
    random.seed(SEED)

    img_dir = SRC_DIR / "positive image set"
    ann_dir = SRC_DIR / "ground truth"

    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    print(f"Total positive images: {len(img_files)}")

    # Shuffle and split
    random.shuffle(img_files)
    val_count = int(len(img_files) * VAL_RATIO)
    val_stems = set(f.stem for f in img_files[:val_count])

    print(f"Train: {len(img_files) - val_count}, Val: {val_count}")

    for split in ['train', 'val']:
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {'train': 0, 'val': 0}

    for img_path in img_files:
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.txt"

        if not ann_path.exists():
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        yolo_lines = []
        with open(ann_path, 'r') as f:
            for line in f:
                parsed = parse_nwpu_line(line)
                if parsed is None:
                    continue
                x1, y1, x2, y2, cls = parsed
                if cls < 1 or cls > 10:
                    continue

                cls_id = cls - 1  # Remap 1-10 to 0-9
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                nw = (x2 - x1) / img_w
                nh = (y2 - y1) / img_h

                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        split = 'val' if stem in val_stems else 'train'
        shutil.copy2(img_path, DST_DIR / "images" / split / img_path.name)
        with open(DST_DIR / "labels" / split / f"{stem}.txt", 'w') as f:
            f.write('\n'.join(yolo_lines))
        stats[split] += 1

    print(f"\nConversion complete!")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")

    yaml_content = f"""path: {DST_DIR}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = DST_DIR / "nwpu.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Dataset YAML written to: {yaml_path}")


if __name__ == "__main__":
    main()
