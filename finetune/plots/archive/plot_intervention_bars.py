"""
Small standalone accuracy-bar PNGs for each intervention, for use in draw.io.
Output: outputs/plots/intervention_bars/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT_DIR = Path("outputs/plots/intervention_bars")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_ORIG   = "#4C72B0"
COLOR_INTERV = "#C44E52"

INTERVENTIONS = [
    dict(filename="intervention_66_mask_numbers",    label="Mask Numbers",                orig=87, interv=61),
    dict(filename="intervention_67_word_replace",    label="Word Replace",                orig=87, interv=71),
    dict(filename="intervention_70_word_polarity",   label="Word + Polarity Mask",        orig=87, interv=55),
    dict(filename="intervention_71_word_pct_polarity", label="Word + Pct + Polarity Mask", orig=87, interv=54),
]


def _style_bar_ax(ax, orig, interv, show_chance_label=False, fontsize=8):
    ax.set_xlim(50, 102)
    ax.set_ylim(-0.6, 1.6)

    ax.barh(1, orig,   color=COLOR_ORIG,   height=0.52, zorder=3)
    ax.barh(0, interv, color=COLOR_INTERV, height=0.52, zorder=3)

    for y, val, color in [(1, orig, COLOR_ORIG), (0, interv, COLOR_INTERV)]:
        ax.text(val + 0.8, y, f"{val}%",
                va="center", ha="left", fontsize=fontsize,
                color=color, fontweight="bold", zorder=4)

    ax.axvline(50, color="#777777", linewidth=0.9, linestyle=":", zorder=2)

    ax.text(49.4, 1, "Original",   va="center", ha="right", fontsize=fontsize, color=COLOR_ORIG)
    ax.text(49.4, 0, "Intervened", va="center", ha="right", fontsize=fontsize, color=COLOR_INTERV)

    ax.set_yticks([])
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_xticklabels(["50", "60", "70", "80", "90", "100"], fontsize=6.5)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Accuracy (%)", fontsize=7, labelpad=2)


# ── Individual figures ────────────────────────────────────────────────────────
for iv in INTERVENTIONS:
    fig, ax = plt.subplots(figsize=(3.0, 1.2))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    _style_bar_ax(ax, iv["orig"], iv["interv"], show_chance_label=True)
    plt.tight_layout(pad=0.3)
    out = OUT_DIR / f"{iv['filename']}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote: {out}")


# ── Combined figure ───────────────────────────────────────────────────────────
n = len(INTERVENTIONS)
fig, axes = plt.subplots(n, 1, figsize=(3.8, 1.4 * n))
fig.patch.set_alpha(0)

for ax, iv in zip(axes, INTERVENTIONS):
    ax.patch.set_alpha(0)
    _style_bar_ax(ax, iv["orig"], iv["interv"],
                  show_chance_label=(iv is INTERVENTIONS[-1]), fontsize=7.5)
    # Intervention name on far left as a rotated label
    ax.text(-0.01, 0.5, iv["label"],
            transform=ax.transAxes,
            va="center", ha="right",
            fontsize=7, fontstyle="italic", color="#333333")

plt.tight_layout(pad=0.4, h_pad=0.5)
out = OUT_DIR / "intervention_bars_combined.png"
fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)
print(f"Wrote: {out}")
