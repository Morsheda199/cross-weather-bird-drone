"""Evaluate models on the fixed 1,062-image AOD4 bird-drone test set.

Table 9: zero-shot synthetic-to-real transfer (no real labels).
Table 10 : fine-tuned models; YOLOv8n 1% = mean +- std over the four subsets (1, 1_s1, 1_s2, 1_s3).
Use --clean to evaluate on the leakage-free 863-image test subset instead.
"""
import os, sys, statistics
from ultralytics import YOLO

DATA = os.environ.get("DATA_ROOT", "data")
RUNS = os.environ.get("RUNS", "runs")
AOD4 = os.environ.get("AOD4_ROOT", "data/AOD4_bird_drone_only")
TEST = (os.path.join(DATA, "AOD4_bd_clean_test", "aod4_bd_clean.yaml") if "--clean" in sys.argv
        else os.path.join(AOD4, "aod4_bd.yaml"))

def ev(weights):
    if not os.path.exists(weights):
        return None
    return YOLO(weights).val(data=TEST, split="test", imgsz=640, batch=16, verbose=False, plots=False).box.map50

w = lambda group, run: os.path.join(RUNS, group, run, "weights", "best.pt")
fmt = lambda v: "  -  " if v is None else f"{v:.3f}"

print(f"Test set: {TEST}\n\n== Table 9: zero-shot ==")
for arch in ["yolov8n", "yolov8s"]:
    for pre in ["fog17", "sun17", "mixed17"] + (["mixed17_scratch"] if arch == "yolov8n" else []):
        print(f"{arch} {pre:16s} {fmt(ev(w('synthetic', f'{arch}_{pre}')))}")

print("\n== Table 10: fine-tuning ==")
inits = {"yolov8n": ["random", "synonly", "coco", "cocosyn"], "yolov8s": ["random", "coco", "cocosyn"]}
for arch, names in inits.items():
    for init in names:
        cells = []
        for tag in ["1", "5", "10", "25", "50", "100"]:
            if tag == "1" and arch == "yolov8n":
                v = [ev(w("finetune", f"{arch}_{init}_{t}")) for t in ["1", "1_s1", "1_s2", "1_s3"]]
                v = [x for x in v if x is not None]
                cells.append(f"1%:{statistics.mean(v):.3f}+-{statistics.stdev(v):.3f}" if len(v) > 1 else "1%:  -  ")
            else:
                cells.append(f"{tag}%:{fmt(ev(w('finetune', f'{arch}_{init}_{tag}')))}")
        print(f"{arch} {init:8s}", "  ".join(cells))
