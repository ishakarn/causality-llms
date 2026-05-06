"""
Paper-ready graded-exposure learning curve.
Evenly-spaced categorical x-axis (each N = same visual width).
Output: outputs/plots/paper-ready/learning_curve.{png,pdf}
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plot_style import apply_defaults, model_color as mc

apply_defaults()

# ── Config ─────────────────────────────────────────────────────────────────────
OUT_DIR     = Path("outputs/plots/paper-ready")
RESULTS_DIR = Path("finetune/eval_results/graded")

SIZES    = [50, 100, 250, 500, 1000, 2000]
DATASETS = ["easy", "hard", "anticommonsense", "noncommonsense"]

DATASET_LABELS = {
    "easy":            "Easy",
    "hard":            "Hard",
    "anticommonsense": "Anti-Commonsense",
    "noncommonsense":  "Non-Commonsense",
}

MODEL_TAGS   = ["qwen3b", "llama8b", "olmo32b"]
MODEL_LABELS = {
    "qwen3b":  "Qwen2.5-3B",
    "llama8b": "Llama-3.1-8B",
    "olmo32b": "OLMo-3.1-32B",
}
MODEL_DISPLAY = {
    "qwen3b":  "Qwen2.5-3B-Instruct",
    "llama8b": "Llama-3.1-8B-Instruct",
    "olmo32b": "OLMo-3.1-32B-Instruct",
}
MODEL_STYLES = {
    "qwen3b":  dict(linestyle="-",  marker="o"),
    "llama8b": dict(linestyle="-.", marker="^"),
    "olmo32b": dict(linestyle=":",  marker="D"),
}

# ── Data ───────────────────────────────────────────────────────────────────────
def load_summary(model_tag, N, ds):
    p = RESULTS_DIR / f"{model_tag}_n{N}_lora" / ds / "score" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def wilson_ci(correct, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - m), min(1.0, c + m)


# ── Plot ───────────────────────────────────────────────────────────────────────
LEGEND_PANEL = 0   # index of the panel that gets the legend (Easy)

def plot_learning_curve():
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=True)

    handles_for_legend = []

    for ax_i, (ax, ds) in enumerate(zip(axes, DATASETS)):
        ax.set_title(DATASET_LABELS[ds], fontsize=9, fontweight="bold", pad=4)

        for model_tag in MODEL_TAGS:
            style = MODEL_STYLES[model_tag]
            color = mc(MODEL_DISPLAY[model_tag])
            xs, accs, lows, highs = [], [], [], []

            for N in SIZES:
                s = load_summary(model_tag, N, ds)
                if s is None:
                    continue
                lo, hi = wilson_ci(s["correct"], s["n"])
                xs.append(N)
                accs.append(s["acc_all"])
                lows.append(lo)
                highs.append(hi)

            if not accs:
                continue

            line, = ax.plot(xs, accs, color=color, linewidth=1.6,
                            linestyle=style["linestyle"], marker=style["marker"],
                            markersize=4.5, label=MODEL_LABELS[model_tag], zorder=3)
            ax.fill_between(xs, lows, highs, color=color, alpha=0.12, zorder=2)
            if ax_i == LEGEND_PANEL:
                handles_for_legend.append(line)

        ax.set_xscale("log")
        ax.set_xlim(SIZES[0] * 0.75, SIZES[-1] * 1.35)
        ax.set_xticks(SIZES)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs="auto", numticks=10))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticklabels([str(s) for s in SIZES], rotation=40, ha="right", fontsize=7.5)
        ax.set_ylim(0.4, 1.02)

        # prominent horizontal grid + chance line
        ax.axhline(0.5, color="gray", linewidth=0.9, linestyle="--", alpha=0.6, zorder=1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25, zorder=0)

        # prominent vertical grid — both minor (faint) and major (solid) for log feel
        ax.grid(axis="x", which="minor", linestyle="-", linewidth=0.4, alpha=0.18, zorder=0)
        ax.grid(axis="x", which="major", linestyle="-", linewidth=0.7, alpha=0.45, zorder=0)

        # thicker spines + tick marks to signal log scale
        ax.tick_params(axis="x", which="major", length=5, width=0.9)
        ax.tick_params(axis="x", which="minor", length=3, width=0.6)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.spines["left"].set_linewidth(0.9)

        if ax_i == 0:
            ax.set_ylabel("Accuracy", fontsize=8)

    # Single centered x-axis label below all panels
    fig.text(0.5, 0.01, "Training size (N)", ha="center", va="top", fontsize=8)

    # Compact legend below the x-axis
    fig.legend(handles=handles_for_legend, loc="lower center", ncol=3,
               fontsize=7.5, framealpha=0.9, edgecolor="#cccccc",
               handlelength=2.0, bbox_to_anchor=(0.5, -0.18))

    plt.tight_layout(w_pad=0.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"learning_curve.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close()


if __name__ == "__main__":
    plot_learning_curve()
