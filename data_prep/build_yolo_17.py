"""Phases 2-3 (17 drones): build DS-3..DS-6 (sunny, foggy, rainy, snowy) from Unity SOLO output.
Sequence-disjoint 70/15/15 split at the solo-run level (seed 42); boxes < 16 px^2 discarded."""
import json
import os
import shutil
import random

# DATASET_BASE holds the raw SOLO folders (e.g. 260325_Weather_foggy); outputs go to DATA_ROOT
DATASET_BASE = os.environ.get("SOLO_17_ROOT", os.environ.get("DATA_ROOT", "data"))
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
MIN_BBOX_AREA = 16
RANDOM_SEED = 42
IMG_W, IMG_H = 1920.0, 1080.0
MAX_SOLO_FOLDERS = 405

BIRD_LABELS = {
    "Toucan", "Stylized Parrot", "Sparrow_PBR", "Seagull_PBR",
    "RedKite", "Pigeon_PBR", "GreatHornedOwl_PBR", "Crow_PBR", "01BirdPrefab"
}

def get_class(label_name):
    return 0 if label_name in BIRD_LABELS else 1

def parse_json(json_path):
    with open(json_path) as f:
        d = json.load(f)
    if "captures" not in d:
        return None
    yolo_lines = []
    for cap in d["captures"]:
        for ann in cap.get("annotations", []):
            if "BoundingBox2D" not in ann.get("@type", ""):
                continue
            for v in ann.get("values", []):
                x, y = v["origin"]
                bw, bh = v["dimension"]
                if bw * bh < MIN_BBOX_AREA:
                    continue
                cx = (x + bw / 2) / IMG_W
                cy = (y + bh / 2) / IMG_H
                nw = bw / IMG_W
                nh = bh / IMG_H
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                cls = get_class(v["labelName"])
                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return yolo_lines

def get_solo_folders(weather_folder, max_folders=None):
    base = os.path.join(DATASET_BASE, weather_folder)
    folders = sorted([f for f in os.listdir(base)
               if os.path.isdir(os.path.join(base, f)) and f.startswith("solo")])
    if max_folders and len(folders) > max_folders:
        random.seed(RANDOM_SEED)
        folders = random.sample(folders, max_folders)
        folders = sorted(folders)
        print(f"  Capped to {max_folders} solo folders (was {len(folders)+len(folders)-max_folders})")
    print(f"  Using {len(folders)} solo folders")
    return folders

def split_folders(folders, ratios, seed):
    random.seed(seed)
    shuffled = folders.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]

def build_dataset(weather_folder, output_name):
    OUTPUT_BASE = os.path.abspath(os.path.join(DATA_ROOT, output_name))
    print(f"\n{'='*60}")
    print(f"Building : {output_name}")
    print(f"Source   : {weather_folder}")
    print(f"Output   : {OUTPUT_BASE}")
    print(f"{'='*60}")

    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)
    os.makedirs(OUTPUT_BASE)

    folders = get_solo_folders(weather_folder, max_folders=MAX_SOLO_FOLDERS)
    train_f, val_f, test_f = split_folders(folders, SPLIT_RATIOS, RANDOM_SEED)
    print(f"Split -> Train:{len(train_f)} Val:{len(val_f)} Test:{len(test_f)} solo folders")

    split_map = {"train": train_f, "val": val_f, "test": test_f}
    counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    base = os.path.join(DATASET_BASE, weather_folder)
    for split, solo_list in split_map.items():
        for solo in solo_list:
            solo_path = os.path.join(base, solo)
            sequences = sorted([s for s in os.listdir(solo_path)
                if os.path.isdir(os.path.join(solo_path, s)) and s.startswith("sequence")])
            for seq in sequences:
                seq_dir = os.path.join(solo_path, seq)
                json_path = os.path.join(seq_dir, "step0.frame_data.json")
                img_path = os.path.join(seq_dir, "step0.camera.png")
                if not os.path.exists(json_path) or not os.path.exists(img_path):
                    skipped += 1
                    continue
                yolo_lines = parse_json(json_path)
                if yolo_lines is None:
                    skipped += 1
                    continue
                fname = f"{weather_folder}_{solo}_{seq}"
                img_dst = os.path.join(OUTPUT_BASE, split, "images", f"{fname}.png")
                label_dst = os.path.join(OUTPUT_BASE, split, "labels", f"{fname}.txt")
                os.makedirs(os.path.dirname(img_dst), exist_ok=True)
                os.makedirs(os.path.dirname(label_dst), exist_ok=True)
                shutil.copy2(img_path, img_dst)
                with open(label_dst, "w") as f:
                    f.write("\n".join(yolo_lines))
                counts[split] += 1
        print(f"  {split} done: {counts[split]} images")

    yaml_content = f"""path: {OUTPUT_BASE}
train: train/images
val:   val/images
test:  test/images

nc: 2
names:
  0: bird
  1: drone
"""
    yaml_path = os.path.join(OUTPUT_BASE, f"{output_name}.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    total = sum(counts.values())
    print(f"\nTrain: {counts['train']} ({counts['train']/total*100:.1f}%)")
    print(f"Val  : {counts['val']} ({counts['val']/total*100:.1f}%)")
    print(f"Test : {counts['test']} ({counts['test']/total*100:.1f}%)")
    print(f"Total: {total} | Skipped: {skipped}")
    print(f"YAML : {yaml_path}")
    print(f"DONE : {output_name}")

if __name__ == "__main__":
    build_dataset("260326_Weather_rain", "yolo_data_rain_17")
    build_dataset("260326_Weather_Snow", "yolo_data_snow_17")
    build_dataset("260325_Weather_foggy", "yolo_data_foggy_17")
    build_dataset("260325_Weather_sunny", "yolo_data_sunny_17")
    print("\nAll datasets built successfully!")
