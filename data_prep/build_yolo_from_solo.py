"""Phase 1 (13 drones): build the paired foggy+sunny YOLO dataset from Unity SOLO output.
Sequence-disjoint 70/15/15 split at the solo-run level (seed 42); foggy and sunny share the split.
Run data_prep/split_phase1_by_weather.py afterwards to obtain DS-1 (sunny) and DS-2 (foggy)."""
import json
import glob
import os
import shutil
import random
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Raw Unity Perception (SOLO) output with sub-folders foggy/ and sunny/
DATASET_BASE = os.environ.get("SOLO_PHASE1_ROOT", "data/raw/Bird_Drone")
OUTPUT_BASE  = os.path.join(os.environ.get("DATA_ROOT", "data"), "yolo_data")
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test
MIN_BBOX_AREA = 16                   # filter boxes smaller than 4x4 px
RANDOM_SEED  = 42
IMG_W, IMG_H = 1920.0, 1080.0

BIRD_LABELS = {
    "Toucan", "Stylized Parrot", "Sparrow_PBR", "Seagull_PBR",
    "RedKite", "Pigeon_PBR", "GreatHornedOwl_PBR", "Crow_PBR", "01BirdPrefab"
}
# Everything else is drone → class 1
# Bird = 0, Drone = 1

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_class(label_name):
    return 0 if label_name in BIRD_LABELS else 1


def parse_json(json_path):
    """
    Returns list of YOLO lines for this frame.
    Empty list = background image (valid, keep it).
    Returns None if JSON has no 'captures' key (metadata file).
    """
    with open(json_path) as f:
        d = json.load(f)

    if "captures" not in d:
        return None  # metadata file, skip

    yolo_lines = []
    for cap in d["captures"]:
        for ann in cap.get("annotations", []):
            if "BoundingBox2D" not in ann.get("@type", ""):
                continue
            for v in ann.get("values", []):
                x, y   = v["origin"]
                bw, bh = v["dimension"]

                # Filter tiny boxes
                if bw * bh < MIN_BBOX_AREA:
                    continue

                # Convert to YOLO normalized xywh
                cx = (x + bw / 2) / IMG_W
                cy = (y + bh / 2) / IMG_H
                nw = bw / IMG_W
                nh = bh / IMG_H

                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                cls = get_class(v["labelName"])
                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return yolo_lines


def get_solo_folders(weather):
    """Return sorted list of solo folder names for a given weather."""
    base = os.path.join(DATASET_BASE, weather)
    folders = [f for f in os.listdir(base)
               if os.path.isdir(os.path.join(base, f)) and f.startswith("solo")]
    return sorted(folders)


def split_folders(folders, ratios, seed):
    """Split folder list into train/val/test by solo-level (not image-level)."""
    random.seed(seed)
    shuffled = folders.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    return train, val, test


def copy_sample(weather, solo, sequence, split):
    """Copy image and write label file for one sequence."""
    seq_dir   = os.path.join(DATASET_BASE, weather, solo, sequence)
    json_path = os.path.join(seq_dir, "step0.frame_data.json")
    img_path  = os.path.join(seq_dir, "step0.camera.png")

    if not os.path.exists(json_path) or not os.path.exists(img_path):
        return False

    yolo_lines = parse_json(json_path)
    if yolo_lines is None:
        return False  # metadata file

    # Unique filename: weather_solo_sequence.png
    fname = f"{weather}_{solo}_{sequence}"

    # Destination paths
    img_dst   = os.path.join(OUTPUT_BASE, split, "images", f"{fname}.png")
    label_dst = os.path.join(OUTPUT_BASE, split, "labels", f"{fname}.txt")

    os.makedirs(os.path.dirname(img_dst),   exist_ok=True)
    os.makedirs(os.path.dirname(label_dst), exist_ok=True)

    shutil.copy2(img_path, img_dst)

    with open(label_dst, "w") as f:
        f.write("\n".join(yolo_lines))

    return True


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Building YOLO dataset from Unity Solo format")
    print("=" * 60)

    # Clean output directory
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)
    os.makedirs(OUTPUT_BASE)

    # Get solo folders (foggy is the reference for splitting)
    foggy_folders = get_solo_folders("foggy")
    sunny_folders = get_solo_folders("sunny")

    print(f"Foggy solo folders: {len(foggy_folders)}")
    print(f"Sunny solo folders: {len(sunny_folders)}")

    # Solo-level split (paired — same split for foggy and sunny)
    train_f, val_f, test_f = split_folders(foggy_folders, SPLIT_RATIOS, RANDOM_SEED)

    # Sunny folders are the same names, use same split indices
    train_s = [f for f in sunny_folders if f in set(train_f)]
    val_s   = [f for f in sunny_folders if f in set(val_f)]
    test_s  = [f for f in sunny_folders if f in set(test_f)]

    print(f"\nSplit (solo folders):")
    print(f"  Train : {len(train_f)} foggy + {len(train_s)} sunny = {len(train_f)+len(train_s)} folders")
    print(f"  Val   : {len(val_f)} foggy + {len(val_s)} sunny = {len(val_f)+len(val_s)} folders")
    print(f"  Test  : {len(test_f)} foggy + {len(test_s)} sunny = {len(test_f)+len(test_s)} folders")

    # Process all sequences
    counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for weather, split_map in [
        ("foggy", {"train": train_f, "val": val_f, "test": test_f}),
        ("sunny", {"train": train_s, "val": val_s, "test": test_s}),
    ]:
        base = os.path.join(DATASET_BASE, weather)
        for split, folders in split_map.items():
            for solo in folders:
                solo_path = os.path.join(base, solo)
                sequences = sorted([
                    s for s in os.listdir(solo_path)
                    if os.path.isdir(os.path.join(solo_path, s)) and s.startswith("sequence")
                ])
                for seq in sequences:
                    ok = copy_sample(weather, solo, seq, split)
                    if ok:
                        counts[split] += 1
                    else:
                        skipped += 1

        print(f"  ✅ {weather} done")

    # Write dataset YAML
    yaml_content = f"""path: {OUTPUT_BASE}
train: train/images
val:   val/images
test:  test/images

nc: 2
names:
  0: bird
  1: drone
"""
    yaml_path = os.path.join(OUTPUT_BASE, "bird_drone.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # Final report
    total = sum(counts.values())
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Train images : {counts['train']}  ({counts['train']/total*100:.1f}%)")
    print(f"Val   images : {counts['val']}   ({counts['val']/total*100:.1f}%)")
    print(f"Test  images : {counts['test']}   ({counts['test']/total*100:.1f}%)")
    print(f"Total images : {total}")
    print(f"Skipped      : {skipped}")
    print(f"YAML saved   : {yaml_path}")
    print("=" * 60)
    print("✅ Dataset build complete!")


if __name__ == "__main__":
    main()