# Cross-Weather Generalization of Bird–Drone Detectors with Multi-Weather Synthetic Data

Code for the paper **"Cross-Weather Generalization of Bird–Drone Detectors with Multi-Weather Synthetic Data: Asymmetric Transfer and Sim-to-Real Adaptation"** (Morsheda Akter, Joo YeonShin, Son JunYoung), currently under review.

The repository contains the scripts used to build the synthetic datasets, train and evaluate the detectors, fine-tune on the real AOD4 benchmark, and draw the figures. Every table and figure in the paper maps to a script below.

## Main findings
- **Cross-weather transfer is direction-dependent.** Snow-trained models are the most robust source (they transfer to fog and rain with essentially no loss), while sunny and snowy conditions are the hardest targets. Degraded-to-clear transfer is not free: for YOLOv8n, fog-, rain- and snow-trained models all lose 0.09–0.13 mAP50 on sunny data, and the sunny-trained model collapses on snow (0.656).
- **Fog↔sun asymmetry.** Fog→Sun drops 0.130 mAP50 vs. 0.023 for Sun→Fog (ρ ≈ 5.7, YOLOv8n; ρ ≈ 4.1, YOLOv8s).
- **Mixed-weather training** reaches 0.98–0.99 mAP50 on fog, sun and rain within the synthetic domain (snow: 0.946), but transfers poorly to real AOD4 imagery (zero-shot 0.117 for YOLOv8n, 0.152 for YOLOv8s).
- **Synthetic pre-training helps most when real labels are scarce.** With 1% real data, synthetic pre-training alone matches COCO initialization (0.627 vs. 0.633), and COCO + synthetic pre-training is best (0.685). All initializations converge as real data grows. Only synthetic pre-training enables zero-shot drone detection.

## Repository layout
```
data_prep/     dataset construction (synthetic DS-1..DS-6, Mixed, AOD4 subsets)
experiments/   training, fine-tuning and evaluation
figures/       scripts for Figs. 4-6
splits/        exact AOD4 image lists used in the paper (subsets, test set, leakage lists)
```

## Environment
Single NVIDIA GB10 GPU, Ultralytics 8.4. The experiments were run over several months, so individual runs used Ultralytics releases between 8.4.14 and 8.4.158 (partly inside the NVIDIA PyTorch 25.09 container, partly in a Python 3.12.3 / PyTorch 2.13.0 environment). Re-running with a different 8.4.x release may change results slightly.
```bash
pip install -r requirements.txt
export DATA_ROOT=data                          # where datasets are built
export AOD4_ROOT=data/AOD4_bird_drone_only     # AOD4 restricted to bird/drone
export RUNS=runs                               # training outputs
```
Training settings for all runs: 50 epochs, 640×640, seed 0, `optimizer=auto` (Ultralytics selects the optimizer and learning rate). Synthetic single-weather models use batch 8; mixed-weather models and all AOD4 fine-tuning use batch 16.

## Data
**Synthetic data** (Unity 3D + Unity Perception, 1920×1080, classes 0 = bird, 1 = drone). The rendered datasets are not included because of their size.

| Dataset | Weather | Drones | Birds | Images |
|---|---|---|---|---|
| DS-1 / DS-2 (Phase 1) | Sunny / Foggy | 13 | 9 | 26,100 each |
| DS-3 / DS-4 (Phase 2) | Sunny / Foggy | 17 | 9 | 40,462 / 40,378 |
| DS-5 / DS-6 (Phase 3) | Rainy / Snowy | 17 | 9 | 40,499 / 40,500 |
| Mixed | all four | 17 | 9 | 161,839 (113,040 / 24,000 / 24,799) |

Each dataset uses a sequence-disjoint 70/15/15 train/val/test split at the level of Unity Perception runs (`solo_N`, seed 42); boxes smaller than 16 px² are discarded.

```bash
python data_prep/build_yolo_from_solo.py      # Phase 1 paired foggy+sunny set   (SOLO_PHASE1_ROOT)
python data_prep/split_phase1_by_weather.py   # -> DS-1 sunny, DS-2 foggy
python data_prep/build_yolo_17.py             # DS-3..DS-6                        (SOLO_17_ROOT)
python data_prep/build_mixed_dataset.py       # Mixed = DS-3 ∪ DS-4 ∪ DS-5 ∪ DS-6
```

**Real data: AOD4** ([Roboflow](https://universe.roboflow.com/airborne-object-detection/airborne-object-detection-4-aod4)), restricted to the bird and drone classes (airplane and helicopter removed) and remapped to 0 = bird, 1 = drone. All real-world results use the same fixed 1,062-image test set. The exact image lists are in `splits/`:

| File | Content |
|---|---|
| `AOD4_bd_test.txt` | fixed test set (1,062 images) |
| `AOD4_bd_{1,5,10,25,50}_train.txt` | training subsets: 74 / 372 / 745 / 1,862 / 3,724 images (100% = 7,448) |
| `AOD4_bd_1_s{1,2,3}_train.txt` | three additional independent 1% subsets |
| `leaked_test_images.txt`, `overlap_bases.txt` | 199 test images that share a source frame with a training image via Roboflow augmentation |

```bash
python data_prep/make_aod4_subsets.py         # builds AOD4_bd_<p>/ and the leakage-free test set (863 images)
```

## Reproducing the paper
```bash
bash experiments/train_synthetic.sh           # Phase 1-3 models, mixed models, random-init mixed model
python experiments/eval_synthetic.py          # Tables 3, 5-8, Fig. 4
bash experiments/finetune_aod4.sh             # Table 10 fine-tuning runs
python experiments/eval_aod4.py               # Tables 9-10
python experiments/eval_aod4.py --clean       # same, on the leakage-free test subset
```

| Paper | Script(s) |
|---|---|
| Table 3 (Phase 1 cross-weather) | `train_synthetic.sh`, `eval_synthetic.py` |
| Table 4 (efficiency) | Phase 1 fog-trained models from `train_synthetic.sh` |
| Tables 5-7, Fig. 4 (cross-weather matrices, asymmetry, mAP50-95) | `eval_synthetic.py`; `figures/make_fig4a_heatmap_yolov8n.py`, `make_fig4b_heatmap_yolov8s.py` |
| Table 8 (mixed-weather training) | `eval_synthetic.py` |
| Table 9 (zero-shot on AOD4) | `eval_aod4.py` |
| Table 10, Fig. 6 (fine-tuning) | `finetune_aod4.sh`, `eval_aod4.py`; `figures/make_fig6_finetune.py` |
| Fig. 5 (t-SNE before/after fine-tuning) | `figures/make_fig5_tsne.py` |

Initializations in Table 10: **Random** = `yolov8n.yaml` / `yolov8s.yaml`; **COCO** = `yolov8n.pt` / `yolov8s.pt`; **COCO+Syn.** = the COCO-initialized mixed-weather model; **Syn. only** = the mixed-weather model trained from random initialization (YOLOv8n only). The YOLOv8n 1% entries are means ± std over four subsets (`1`, `1_s1`, `1_s2`, `1_s3`); all other entries are single runs.

## Key results

**Cross-weather matrix, YOLOv8n (mAP50; rows = trained on, columns = tested on)**

| | Sun | Fog | Rain | Snow |
|---|---|---|---|---|
| Sun | 0.977 | 0.954 | 0.907 | 0.656 |
| Fog | 0.843 | 0.973 | 0.947 | 0.762 |
| Rain | 0.883 | 0.975 | 0.977 | 0.785 |
| Snow | 0.845 | 0.972 | 0.966 | 0.941 |

**Zero-shot on AOD4 (mAP50)**

| Pre-training | YOLOv8n | YOLOv8s |
|---|---|---|
| Fog | 0.098 | 0.100 |
| Sun | 0.102 | 0.101 |
| Mixed | 0.117 | 0.152 |
| Mixed, random init | 0.077 | – |

**Fine-tuning on AOD4, YOLOv8n (mAP50)**

| Real % | Random | Syn. only | COCO | COCO+Syn. |
|---|---|---|---|---|
| 1 | 0.238 ± 0.038 | 0.627 ± 0.009 | 0.633 ± 0.013 | 0.685 ± 0.020 |
| 5 | 0.679 | 0.822 | 0.871 | 0.839 |
| 10 | 0.795 | 0.859 | 0.898 | 0.873 |
| 25 | 0.886 | 0.911 | 0.931 | 0.920 |
| 50 | 0.912 | 0.932 | 0.949 | 0.948 |
| 100 | 0.940 | 0.954 | 0.962 | 0.956 |

See the paper for the YOLOv8s results and the remaining tables.

## Citation
Citation details will be added upon publication.
