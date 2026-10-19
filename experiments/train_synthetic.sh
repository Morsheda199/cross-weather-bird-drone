#!/bin/bash
# Synthetic-domain training: Phase 1 (Tables 3-4), Phases 2-3 (Tables 5-7, Fig. 4),
# mixed-weather models (Table 8) and the random-init "Syn. only" model (Tables 9-10).
set -e
source "$(dirname "$0")/common.sh"
T() { yolo detect train "$@" $COMMON project=$RUNS/synthetic exist_ok=True; }

# Phase 1 (13 drones): DS-2 foggy, DS-1 sunny - batch 8
for m in yolov8n yolov8s rtdetr-l; do
  T model=$m.pt data=$P1_FOG batch=8 name=${m}_p1_fog
  T model=$m.pt data=$P1_SUN batch=8 name=${m}_p1_sun
done

# Phases 2-3 (17 drones): one model per weather - batch 8
for m in yolov8n yolov8s; do
  T model=$m.pt data=$FOG  batch=8 name=${m}_fog17
  T model=$m.pt data=$SUN  batch=8 name=${m}_sun17
  T model=$m.pt data=$RAIN batch=8 name=${m}_rain17
  T model=$m.pt data=$SNOW batch=8 name=${m}_snow17
done

# Mixed weather (union of DS-3..DS-6) - batch 16
T model=yolov8n.pt   data=$MIXED batch=16 name=yolov8n_mixed17          # COCO-initialized
T model=yolov8s.pt   data=$MIXED batch=16 name=yolov8s_mixed17          # COCO-initialized
T model=yolov8n.yaml data=$MIXED batch=16 name=yolov8n_mixed17_scratch  # random init (Syn. only)
