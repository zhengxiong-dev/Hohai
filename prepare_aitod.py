"""
AI-TOD 数据集准备脚本
=====================
功能：解压 + COCO JSON 转 YOLO TXT 格式 + 生成 dataset yaml

使用方法（在conda yolo环境下运行）：
    python prepare_aitod.py

耗时预估：解压约10-15分钟，转换约1-2分钟
"""

import zipfile
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

# ==================== 路径配置 ====================
AITOD_RAW = Path("/mnt/d/YOLO-dataset/aitod/AI-TOD")
OUTPUT_DIR = Path("/mnt/d/Student/XJH/datasets/AI-TOD-YOLO")

# AI-TOD 类别 (COCO id 从1开始，YOLO id 从0开始)
CATEGORIES = {
    1: 0,   # airplane    -> 0
    2: 1,   # bridge      -> 1
    3: 2,   # storage-tank-> 2
    4: 3,   # ship        -> 3
    5: 4,   # swimming-pool -> 4
    6: 5,   # vehicle     -> 5
    7: 6,   # person      -> 6
    8: 7,   # wind-mill   -> 7
}

CLASS_NAMES = [
    "airplane", "bridge", "storage-tank", "ship",
    "swimming-pool", "vehicle", "person", "wind-mill"
]


def step1_extract():
    """Step 1: 解压内层zip（annotations, train, val）"""
    print("=" * 60)
    print("  Step 1: 解压数据集")
    print("=" * 60)

    for zipname in ["annotations.zip", "train.zip", "val.zip"]:
        zippath = AITOD_RAW / zipname
        if not zippath.exists():
            print(f"  [SKIP] {zipname} 不存在")
            continue

        # 检查是否已解压
        folder = zipname.replace(".zip", "")
        if (AITOD_RAW / folder).exists():
            count = len(list((AITOD_RAW / folder).rglob("*")))
            if count > 10:
                print(f"  [SKIP] {folder}/ 已存在 ({count} files)")
                continue

        print(f"  解压 {zipname} ...")
        with zipfile.ZipFile(zippath, 'r') as z:
            z.extractall(AITOD_RAW)
        print(f"  [OK] {zipname} 解压完成")

    print()


def step2_convert():
    """Step 2: COCO JSON -> YOLO TXT 格式转换"""
    print("=" * 60)
    print("  Step 2: 转换标注格式 (COCO JSON -> YOLO TXT)")
    print("=" * 60)

    # 创建输出目录
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 处理 train 和 val
    for split in ["train", "val"]:
        json_path = AITOD_RAW / "annotations" / f"aitod_{split}_v1.json"
        if not json_path.exists():
            print(f"  [ERROR] {json_path} 不存在!")
            continue

        print(f"\n  处理 {split} 集...")
        with open(json_path, 'r') as f:
            coco = json.load(f)

        # 建立 image_id -> image_info 映射
        images = {img['id']: img for img in coco['images']}
        print(f"    图像数量: {len(images)}")
        print(f"    标注数量: {len(coco['annotations'])}")

        # 按 image_id 分组标注
        img_anns = defaultdict(list)
        for ann in coco['annotations']:
            img_anns[ann['image_id']].append(ann)

        # 转换每张图
        converted = 0
        skipped = 0
        for img_id, img_info in images.items():
            filename = img_info['file_name']
            w, h = img_info['width'], img_info['height']

            # 创建符号链接或复制图像
            src_img = AITOD_RAW / split / filename
            dst_img = OUTPUT_DIR / "images" / split / filename
            if not dst_img.exists() and src_img.exists():
                os.symlink(src_img.resolve(), dst_img)

            # 写入 YOLO 格式标注
            label_name = Path(filename).stem + ".txt"
            label_path = OUTPUT_DIR / "labels" / split / label_name

            anns = img_anns.get(img_id, [])
            lines = []
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in CATEGORIES:
                    skipped += 1
                    continue

                yolo_cls = CATEGORIES[cat_id]
                # COCO bbox: [x_min, y_min, width, height] -> YOLO: [cx, cy, w, h] normalized
                bx, by, bw, bh = ann['bbox']
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h

                # 过滤无效标注
                if nw <= 0 or nh <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1:
                    skipped += 1
                    continue

                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(label_path, 'w') as f:
                f.write("\n".join(lines))
            converted += 1

        print(f"    转换完成: {converted} 张图像, 跳过 {skipped} 个无效标注")

    print()


def step3_yaml():
    """Step 3: 生成 YOLO dataset yaml 配置文件"""
    print("=" * 60)
    print("  Step 3: 生成 dataset yaml")
    print("=" * 60)

    yaml_content = f"""# AI-TOD Dataset (Tiny Object Detection in Aerial Images)
# 8 classes, ~28K images, average object size ~12.8 pixels

path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val

nc: 8
names:
  0: airplane
  1: bridge
  2: storage-tank
  3: ship
  4: swimming-pool
  5: vehicle
  6: person
  7: wind-mill
"""

    yaml_path = OUTPUT_DIR / "aitod.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"  [OK] 已生成: {yaml_path}")
    print()


def step4_verify():
    """Step 4: 验证数据集"""
    print("=" * 60)
    print("  Step 4: 验证数据集完整性")
    print("=" * 60)

    for split in ["train", "val"]:
        img_dir = OUTPUT_DIR / "images" / split
        lbl_dir = OUTPUT_DIR / "labels" / split
        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split}: {n_img} images, {n_lbl} labels")

        # 读一个标注看看
        if n_lbl > 0:
            sample = next(lbl_dir.glob("*.txt"))
            with open(sample) as f:
                content = f.read()
            lines = [l for l in content.strip().split('\n') if l]
            print(f"    样例 ({sample.name}): {len(lines)} objects")
            if lines:
                print(f"    首行: {lines[0]}")

    print(f"\n  YAML 配置: {OUTPUT_DIR / 'aitod.yaml'}")
    print("\n" + "=" * 60)
    print("  全部完成! 可以开始训练了。")
    print("=" * 60)


if __name__ == "__main__":
    step1_extract()
    step2_convert()
    step3_yaml()
    step4_verify()
