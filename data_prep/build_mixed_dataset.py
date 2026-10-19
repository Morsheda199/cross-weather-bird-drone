"""Mixed dataset = exact union of DS-3..DS-6 (161,839 images), preserving each split."""
import os
import shutil
import random
from pathlib import Path

BASE = os.environ.get("DATA_ROOT", "data")
OUTPUT = os.path.join(BASE, "yolo_data_mixed_17")
RANDOM_SEED = 42
SPLITS = ["train", "val", "test"]

SOURCES = [
    "yolo_data_foggy_17",
    "yolo_data_sunny_17",
    "yolo_data_rain_17",
    "yolo_data_snow_17",
]

if __name__ == "__main__":
    print("="*60)
    print("Building Mixed Dataset")
    print("="*60)

    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
    os.makedirs(OUTPUT)

    counts = {"train": 0, "val": 0, "test": 0}

    for split in SPLITS:
        img_out = os.path.join(OUTPUT, split, "images")
        lbl_out = os.path.join(OUTPUT, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for src in SOURCES:
            src_img = os.path.join(BASE, src, split, "images")
            src_lbl = os.path.join(BASE, src, split, "labels")

            if not os.path.exists(src_img):
                print(f"  SKIP: {src}/{split} not found")
                continue

            images = [f for f in os.listdir(src_img) if f.endswith(".png")]
            for img_file in images:
                lbl_file = img_file.replace(".png", ".txt")
                src_i = os.path.join(src_img, img_file)
                src_l = os.path.join(src_lbl, lbl_file)
                dst_i = os.path.join(img_out, img_file)
                dst_l = os.path.join(lbl_out, lbl_file)

                if not os.path.exists(src_l):
                    continue

                shutil.copy2(src_i, dst_i)
                shutil.copy2(src_l, dst_l)
                counts[split] += 1

        print(f"  {split} done: {counts[split]} images")

    # YAML
    yaml_content = f"""path: {os.path.abspath(OUTPUT)}
train: train/images
val:   val/images
test:  test/images

nc: 2
names:
  0: bird
  1: drone
"""
    yaml_path = os.path.join(OUTPUT, "yolo_data_mixed_17.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    total = sum(counts.values())
    print(f"\nTotal images: {total}")
    print(f"Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")
    print(f"YAML: {yaml_path}")
    print("Mixed dataset build complete!")
