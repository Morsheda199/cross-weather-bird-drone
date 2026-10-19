"""Evaluate synthetic-domain models on the synthetic test splits.

Reproduces Table 3 (Phase 1), Tables 5-7 and Fig. 4 (4x4 cross-weather matrices, mAP50 and mAP50-95)
and Table 8 (mixed-weather model on each weather). Run after experiments/train_synthetic.sh.
"""
import os
from ultralytics import YOLO

DATA = os.environ.get("DATA_ROOT", "data")
RUNS = os.path.join(os.environ.get("RUNS", "runs"), "synthetic")
Y = lambda folder, name: os.path.join(DATA, folder, name)
P1 = {"Fog": Y("yolo_data_foggy", "bird_drone_foggy.yaml"), "Sun": Y("yolo_data_sunny", "bird_drone_sunny.yaml")}
W17 = {"Sun": Y("yolo_data_sunny_17", "yolo_data_sunny_17.yaml"), "Fog": Y("yolo_data_foggy_17", "yolo_data_foggy_17.yaml"),
       "Rain": Y("yolo_data_rain_17", "yolo_data_rain_17.yaml"), "Snow": Y("yolo_data_snow_17", "yolo_data_snow_17.yaml")}

def ev(run, data):
    m = YOLO(os.path.join(RUNS, run, "weights", "best.pt")).val(data=data, split="test", imgsz=640,
                                                                  verbose=False, plots=False)
    return m.box.map50, m.box.map

print("== Phase 1 (13 drones), mAP50 ==")
for arch in ["yolov8n", "yolov8s", "rtdetr-l"]:
    for tr in P1:
        for te in P1:
            print(f"{arch:9s} {tr}->{te}: {ev(f'{arch}_p1_{tr.lower()}', P1[te])[0]:.3f}")

for arch in ["yolov8n", "yolov8s"]:
    print(f"\n== {arch}: 4x4 cross-weather, rows = trained on, cols = tested on (mAP50 / mAP50-95) ==")
    for tr in W17:
        row = [ev(f"{arch}_{tr.lower()}17", W17[te]) for te in W17]
        print(f"{tr:5s}", "  ".join(f"{a:.3f}/{b:.3f}" for a, b in row))
    print(f"{arch} mixed:", "  ".join(f"{te} {ev(f'{arch}_mixed17', W17[te])[0]:.3f}" for te in W17))
