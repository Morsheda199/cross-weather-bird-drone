"""Fig. 6: sim-to-real adaptation on AOD4 (re-plots the values reported in Table 10).

YOLOv8n 1% values are means over four random subsets; error bars show the standard deviation.
Zero-shot (0%) points exist only for models with synthetic pre-training.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PCT = [1, 5, 10, 25, 50, 100]
DATA = {
    "YOLOv8n": {
        "COCO + Synthetic": dict(v=[0.685, 0.839, 0.873, 0.920, 0.948, 0.956], err=0.020, zs=0.117),
        "COCO":             dict(v=[0.633, 0.871, 0.898, 0.931, 0.949, 0.962], err=0.013),
        "Synthetic only":   dict(v=[0.627, 0.822, 0.859, 0.911, 0.932, 0.954], err=0.009, zs=0.077),
        "Random":           dict(v=[0.238, 0.679, 0.795, 0.886, 0.912, 0.940], err=0.038),
    },
    "YOLOv8s": {
        "COCO + Synthetic": dict(v=[0.676, 0.850, 0.874, 0.923, 0.956, 0.959], zs=0.152),
        "COCO":             dict(v=[0.522, 0.863, 0.885, 0.935, 0.951, 0.963]),
        "Random":           dict(v=[0.221, 0.707, 0.821, 0.902, 0.930, 0.950]),
    },
}
STYLE = {"COCO + Synthetic": ("#2E5496", "o-"), "COCO": ("#2E7D5B", "s--"),
         "Synthetic only": ("#D98E04", "D-."), "Random": ("#B23A48", "^:")}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, (name, curves) in zip(axes, DATA.items()):
    for label, d in curves.items():
        color, fmt = STYLE[label]
        ax.plot(PCT, d["v"], fmt, color=color, lw=1.8, ms=6, label=label)
        if "err" in d:
            ax.errorbar([1], [d["v"][0]], yerr=[d["err"]], color=color, capsize=3, lw=1)
        if "zs" in d:
            ax.scatter([0], [d["zs"]], marker="*", s=140, color=color, edgecolors="black", linewidths=0.5, zorder=4)
    ax.set_title(name, fontsize=12, weight="bold")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 5, 10, 25, 50, 100])
    ax.set_xticklabels(["0", "1", "5", "10", "25", "50", "100"])
    ax.set_xlabel("Real labeled data (%)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("AOD4 mAP@50")
axes[0].legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("fig_finetune.png", dpi=300, bbox_inches="tight")
plt.savefig("fig_finetune.pdf", bbox_inches="tight")
print("saved fig_finetune")
