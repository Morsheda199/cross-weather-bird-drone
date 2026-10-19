#!/usr/bin/env python3
"""Fig. 5: t-SNE of YOLOv8n backbone features (COCO+Syn. model), before vs after fine-tuning on 10% AOD4.
250 test images per domain; random seed 0."""
import os, glob, random
import numpy as np
from ultralytics import YOLO
from sklearn.manifold import TSNE
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.environ.get('DATA_ROOT', 'data')
AOD4 = os.environ.get('AOD4_ROOT', 'data/AOD4_bird_drone_only')
RUNS = os.environ.get('RUNS', 'runs')

DOMAINS = {
    'Synthetic (Sunny)': os.path.join(DATA, 'yolo_data_sunny_17/test/images'),
    'Synthetic (Foggy)': os.path.join(DATA, 'yolo_data_foggy_17/test/images'),
    'Synthetic (Rainy)': os.path.join(DATA, 'yolo_data_rain_17/test/images'),
    'Synthetic (Snowy)': os.path.join(DATA, 'yolo_data_snow_17/test/images'),
    'Real (AOD4)':       os.path.join(AOD4, 'images/test'),
}
MODELS = {
    'Before fine-tuning (zero-shot)':
        os.path.join(RUNS, 'synthetic/yolov8n_mixed17/weights/best.pt'),
    'After fine-tuning (+10% real data)':
        os.path.join(RUNS, 'finetune/yolov8n_cocosyn_10/weights/best.pt'),
}
N_PER, SEED, BATCH = 250, 0, 64
random.seed(SEED); np.random.seed(SEED)

imgset = {}
for name, folder in DOMAINS.items():
    paths = []
    for ext in ('*.png','*.jpg','*.jpeg','*.PNG','*.JPG'):
        paths += glob.glob(os.path.join(folder, ext))
    random.shuffle(paths)
    imgset[name] = paths[:N_PER]
    print(name, len(imgset[name]), 'images')

colors = {'Synthetic (Sunny)':'#F2C14E','Synthetic (Foggy)':'#9AA5B1',
          'Synthetic (Rainy)':'#5B9BD5','Synthetic (Snowy)':'#A0C4E8',
          'Real (AOD4)':'#C0392B'}

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, (mname, mpath) in zip(axes, MODELS.items()):
    if not os.path.exists(mpath):
        print("MISSING model:", mpath)
        continue
    model = YOLO(mpath)
    feats, labels = [], []
    for name, imgs in imgset.items():
        for i in range(0, len(imgs), BATCH):
            emb = model.embed(imgs[i:i+BATCH], verbose=False)
            for e in emb:
                e = e.detach().cpu()
                if e.ndim >= 3:
                    e = e.mean(dim=(-1,-2))
                feats.append(e.numpy().ravel())
                labels.append(name)
    X = np.array(feats)
    labels = np.array(labels)
    Z = TSNE(n_components=2, perplexity=30, init='pca', random_state=SEED).fit_transform(X)
    for name in DOMAINS:
        m = labels == name
        is_real = 'Real' in name
        ax.scatter(Z[m,0], Z[m,1],
                   s=(28 if is_real else 12),
                   c=colors[name], label=name,
                   alpha=(0.85 if is_real else 0.5),
                   edgecolors=('black' if is_real else 'none'),
                   linewidths=0.3,
                   marker=('*' if is_real else 'o'),
                   zorder=(3 if is_real else 2))
    ax.set_title(mname, fontsize=11, weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#B9C6D4')

axes[0].legend(loc='upper right', fontsize=7.5, framealpha=0.9)
# fig.suptitle('Synthetic-to-Real Domain Gap: Before vs After Fine-tuning', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('fig_tsne_before_after.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_tsne_before_after.pdf', bbox_inches='tight')
print('saved fig_tsne_before_after')
