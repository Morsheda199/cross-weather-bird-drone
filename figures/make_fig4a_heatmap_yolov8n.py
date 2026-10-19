"""Fig. 4 (top): YOLOv8n 4x4 cross-weather mAP50 matrix (values from eval_synthetic.py / Table 6)."""
import numpy as np
import matplotlib.pyplot as plt

weathers = ["Sun", "Fog", "Rain", "Snow"]
M = np.array([
    [0.977, 0.954, 0.907, 0.656],
    [0.843, 0.973, 0.947, 0.762],
    [0.883, 0.975, 0.977, 0.785],
    [0.845, 0.972, 0.966, 0.941],
])
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(M, cmap="viridis", vmin=0.65, vmax=1.0)
for i in range(4):
    for j in range(4):
        v = M[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color="white" if v < 0.90 else "black", fontsize=11)
ax.set_xticks(range(4)); ax.set_xticklabels(weathers)
ax.set_yticks(range(4)); ax.set_yticklabels(weathers)
ax.set_xlabel("Tested on"); ax.set_ylabel("Trained on")
ax.set_title("Cross-Weather Transfer (mAP@50, YOLOv8n)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("mAP@50")
plt.tight_layout()
plt.savefig("fig_asymmetry.pdf", bbox_inches="tight")
plt.savefig("fig_asymmetry.png", dpi=300, bbox_inches="tight")
print("saved fig_asymmetry (complete 4x4, batch=8 consistent)")
