"""Fig. 4 (bottom): YOLOv8s 4x4 cross-weather mAP50 matrix (values from eval_synthetic.py)."""
import numpy as np
import matplotlib.pyplot as plt

weathers = ["Sun", "Fog", "Rain", "Snow"]
M = np.array([
    [0.987, 0.962, 0.919, 0.680],   # train Sun
    [0.881, 0.983, 0.966, 0.795],   # train Fog
    [0.898, 0.984, 0.983, 0.829],   # train Rain
    [0.900, 0.978, 0.974, 0.949],   # train Snow
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
ax.set_title("Cross-Weather Transfer (mAP@50, YOLOv8s)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("mAP@50")
plt.tight_layout()
plt.savefig("fig_asymmetry_y8s.pdf", bbox_inches="tight")
plt.savefig("fig_asymmetry_y8s.png", dpi=300, bbox_inches="tight")
print("saved fig_asymmetry_y8s (full 4x4)")
