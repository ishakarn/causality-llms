"""
Plot per-epoch learning curves for OLMo-7B-Instruct and Qwen2.5-3B-Instruct.
Outputs two figures:
  1. learning_curve_overall.png  — overall val accuracy per epoch
  2. learning_curve_per_query.png — per-query-type accuracy per epoch (2×5 grid)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
epochs = [1, 2, 3, 4, 5]

olmo = {
    "overall":           [0.8028, 0.8512, 0.9066, 0.9239, 0.9239],
    "ate":               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    "backadj":           [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    "collider_bias":     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    "ett":               [0.9412, 1.0000, 0.9706, 0.9706, 0.9706],
    "correlation":       [0.8750, 0.9250, 0.9750, 1.0000, 1.0000],
    "nie":               [0.7826, 0.8696, 0.9565, 0.9130, 0.9130],
    "nde":               [0.6429, 0.9286, 0.9286, 0.9286, 0.9286],
    "det-counterfactual":[0.6000, 0.6250, 0.7500, 0.7750, 0.7750],
    "marginal":          [0.5227, 0.5909, 0.7500, 0.8636, 0.8636],
    "exp_away":          [0.4000, 0.4000, 0.6000, 0.4000, 0.4000],
}

qwen = {
    "overall":           [0.7543, 0.8720, 0.9273, 0.9308, 0.9308],
    "ate":               [0.9500, 1.0000, 1.0000, 1.0000, 1.0000],
    "backadj":           [0.9773, 1.0000, 1.0000, 1.0000, 1.0000],
    "collider_bias":     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    "ett":               [0.7941, 1.0000, 1.0000, 1.0000, 1.0000],
    "correlation":       [0.7250, 0.9000, 0.9250, 0.9500, 0.9250],
    "nie":               [0.6957, 0.8696, 0.9130, 0.9565, 0.9565],
    "nde":               [0.7857, 0.7857, 0.9286, 0.8571, 0.9286],
    "det-counterfactual":[0.5250, 0.6250, 0.8500, 0.8750, 0.8500],
    "marginal":          [0.5455, 0.7727, 0.8636, 0.8409, 0.8636],
    "exp_away":          [0.8000, 0.6000, 0.4000, 0.4000, 0.4000],
}

OLMO_COLOR = "#2196F3"   # blue
QWEN_COLOR  = "#F44336"  # red

# ── Figure 1: Overall accuracy ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(epochs, olmo["overall"], "o-", color=OLMO_COLOR, lw=2, ms=7,
        label="OLMo-3-7B-Instruct")
ax.plot(epochs, qwen["overall"], "s-", color=QWEN_COLOR,  lw=2, ms=7,
        label="Qwen2.5-3B-Instruct")

# best-checkpoint markers
ax.axvline(4, color="gray", ls="--", lw=0.9, alpha=0.6, label="best ckpt (both ep4)")

# test-set horizontal dashes
ax.axhline(0.9201, color=OLMO_COLOR, ls=":", lw=1.2, alpha=0.7)
ax.axhline(0.8993, color=QWEN_COLOR,  ls=":", lw=1.2, alpha=0.7)
ax.text(5.08, 0.9201, "test\n92.0%", color=OLMO_COLOR, va="center", fontsize=7.5)
ax.text(5.08, 0.8993, "test\n89.9%", color=QWEN_COLOR,  va="center", fontsize=7.5)

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Val Accuracy", fontsize=11)
ax.set_title("Overall Val Accuracy per Epoch", fontsize=12, fontweight="bold")
ax.set_xticks(epochs)
ax.set_xlim(0.7, 5.9)
ax.set_ylim(0.70, 0.97)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.35)
fig.tight_layout()
fig.savefig("learning_curve_overall.png", dpi=150)
print("Saved: learning_curve_overall.png")
plt.close()

# ── Figure 2: Per-query-type (2×5 grid) ────────────────────────────────────
query_types = [
    "ate", "backadj", "collider_bias", "ett", "correlation",
    "nie", "nde", "det-counterfactual", "marginal", "exp_away",
]

fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=True, sharex=True)

for ax, qt in zip(axes.flat, query_types):
    ax.plot(epochs, olmo[qt], "o-", color=OLMO_COLOR, lw=1.8, ms=5)
    ax.plot(epochs, qwen[qt], "s-", color=QWEN_COLOR,  lw=1.8, ms=5)
    ax.set_title(qt.replace("-", "-\n") if len(qt) > 12 else qt,
                 fontsize=9, fontweight="bold")
    ax.set_xticks(epochs)
    ax.set_ylim(-0.05, 1.09)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=8)

for ax in axes[1]:
    ax.set_xlabel("Epoch", fontsize=9)

# shared y-label
fig.text(0.005, 0.55, "Val Accuracy", va="center", rotation="vertical", fontsize=10)

# legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=OLMO_COLOR, marker="o", lw=1.8, ms=5, label="OLMo-3-7B-Instruct"),
    Line2D([0], [0], color=QWEN_COLOR,  marker="s", lw=1.8, ms=5, label="Qwen2.5-3B-Instruct"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 1.01))

fig.suptitle("Per-Query-Type Val Accuracy per Epoch", fontsize=12,
             fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig("learning_curve_per_query.png", dpi=150, bbox_inches="tight")
print("Saved: learning_curve_per_query.png")
plt.close()
