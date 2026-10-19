"""Phase 1: split the paired yolo_data/ set into DS-1 (sunny) and DS-2 (foggy).

Files produced by build_yolo_from_solo.py are named <weather>_<solo>_<sequence>.png,
so each weather is selected by its filename prefix. The split assignment is unchanged.
(Verified on the original data: foggy_* files in yolo_data/ are identical to DS-2.)
"""
import os, shutil

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
SRC = os.path.join(DATA_ROOT, "yolo_data")
TARGETS = {"foggy": ("yolo_data_foggy", "bird_drone_foggy.yaml"),
           "sunny": ("yolo_data_sunny", "bird_drone_sunny.yaml")}

for weather, (folder, yaml_name) in TARGETS.items():
    out = os.path.abspath(os.path.join(DATA_ROOT, folder))
    for split in ("train", "val", "test"):
        for kind, ext in (("images", ".png"), ("labels", ".txt")):
            src_dir = os.path.join(SRC, split, kind)
            dst_dir = os.path.join(out, split, kind)
            os.makedirs(dst_dir, exist_ok=True)
            for f in os.listdir(src_dir):
                if f.startswith(weather + "_") and f.endswith(ext):
                    shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        n = len(os.listdir(os.path.join(out, split, "images")))
        print(f"{folder} {split}: {n} images")
    with open(os.path.join(out, yaml_name), "w") as fh:
        fh.write(f"path: {out}\ntrain: train/images\nval: val/images\ntest: test/images\n\n"
                 "nc: 2\nnames:\n  0: bird\n  1: drone\n")
