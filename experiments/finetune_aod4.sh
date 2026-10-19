#!/bin/bash
# Real-data fine-tuning on AOD4 (Table 10, Fig. 6). Batch 16, 50 epochs, seed 0, optimizer=auto.
# Initializations: Random (.yaml), COCO (.pt), COCO+Syn. (mixed model), Syn. only (YOLOv8n only).
# Real fractions: 1% (four subsets: 1, 1_s1, 1_s2, 1_s3), 5, 10, 25, 50, 100%.
set -e
source "$(dirname "$0")/common.sh"
AOD4_ROOT=${AOD4_ROOT:-data/AOD4_bird_drone_only}
SYN=$RUNS/synthetic

data_for() {  # data_for <tag>
  if [ "$1" = "100" ]; then echo "$AOD4_ROOT/aod4_bd.yaml"
  else echo "$DATA_ROOT/AOD4_bd_$1/aod4_bd_$1.yaml"; fi
}
FT() { yolo detect train "$@" batch=16 $COMMON project=$RUNS/finetune exist_ok=True; }

for tag in 1 1_s1 1_s2 1_s3 5 10 25 50 100; do
  D=$(data_for $tag)
  MODELS="yolov8n yolov8s"; case $tag in 1_s*) MODELS="yolov8n";; esac  # extra 1% subsets: YOLOv8n only
  for m in $MODELS; do
    FT model=$m.yaml                                   data=$D name=${m}_random_$tag
    FT model=$m.pt                                     data=$D name=${m}_coco_$tag
    FT model=$SYN/${m}_mixed17/weights/best.pt         data=$D name=${m}_cocosyn_$tag
  done
  FT model=$SYN/yolov8n_mixed17_scratch/weights/best.pt data=$D name=yolov8n_synonly_$tag
done
