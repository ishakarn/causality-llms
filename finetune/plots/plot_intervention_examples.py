"""
Publication-quality example figure: Original vs Intervened queries with accuracy bars.
Output: outputs/plots/paper-ready/intervention_examples.{png,pdf}
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

OUT_DIR = Path("outputs/plots/paper-ready")

# ── Content ───────────────────────────────────────────────────────────────────
ORIGINAL_TEXT = (
    "For husbands that don't set the alarm, the probability of ringing alarm is 26%. "
    "For husbands that set the alarm, the probability of ringing alarm is 76%. "
    "Will alarm set by husband decrease the chance of ringing alarm?"
)

PANELS = [
    dict(
        label="Mask Numbers",
        intervened=(
            "For husbands that don't set the alarm, the probability of ringing alarm is X%. "
            "For husbands that set the alarm, the probability of ringing alarm is X%. "
            "Will alarm set by husband decrease the chance of ringing alarm?"
        ),
        orig_acc=87, interv_acc=61,
    ),
    dict(
        label="Word Replace",
        intervened=(
            "For Boulder, the probability of maple is 26%. "
            "For dune, the probability of maple is 76%. "
            "Will dune decrease the chance of maple?"
        ),
        orig_acc=87, interv_acc=71,
    ),
    dict(
        label="Word + Polarity Mask",
        intervened=(
            "For Boulder, the probability of maple is 26%. "
            "For dune, the probability of maple is 76%. "
            "Will dune _____ the chance of maple?"
        ),
        orig_acc=87, interv_acc=55,
    ),
    dict(
        label="Word + Percentage + Polarity Mask",
        intervened=(
            "For Boulder, the probability of maple is X%. "
            "For dune, the probability of maple is X%. "
            "Will dune _____ the chance of maple?"
        ),
        orig_acc=87, interv_acc=54,
    ),
]

# ── Style ─────────────────────────────────────────────────────────────────────
COLOR_ORIG   = "#4C72B0"   # blue
COLOR_INTERV = "#C44E52"   # red
BOX_ORIG     = "#EEF3FA"   # light blue tint
BOX_INTERV   = "#FBF0F0"   # light red tint
WRAP_WIDTH   = 62          # characters per line for text wrapping
FONT_TEXT    = 8.0
FONT_LABEL   = 8.0
FONT_ACC     = 8.0


def draw_textbox(ax, text, bg_color):
    """Fill ax with a rounded FancyBboxPatch and centered wrapped text."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    patch = FancyBboxPatch(
        (0.01, 0.04), 0.98, 0.92,
        boxstyle="round,pad=0.02",
        facecolor=bg_color, edgecolor="#cccccc", linewidth=0.8,
        transform=ax.transAxes, clip_on=False, zorder=1,
    )
    ax.add_patch(patch)

    wrapped = textwrap.fill(text, width=WRAP_WIDTH)
    ax.text(
        0.5, 0.5, wrapped,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=FONT_TEXT, linespacing=1.4,
        zorder=2,
    )


def draw_bar(ax, orig_acc, interv_acc, show_chance_label):
    """Draw a 2-row horizontal bar chart for orig and intervened accuracy."""
    ax.set_xlim(50, 100)
    ax.set_ylim(-0.6, 1.6)

    # Bars (start at 0; xlim clips to 50 so they appear to start at chance)
    ax.barh(1, orig_acc,   color=COLOR_ORIG,   height=0.55, zorder=3)
    ax.barh(0, interv_acc, color=COLOR_INTERV, height=0.55, zorder=3)

    # Percentage labels at bar end
    for y, val, color in [(1, orig_acc, COLOR_ORIG), (0, interv_acc, COLOR_INTERV)]:
        ax.text(min(val + 0.8, 99), y, f"{val}%",
                va="center", ha="left", fontsize=FONT_ACC,
                color=color, fontweight="bold", zorder=4)

    # Chance line
    ax.axvline(50, color="#555555", linewidth=1.0, linestyle=":", zorder=2)
    if show_chance_label:
        ax.text(50.5, -0.52, "chance", fontsize=6.5, color="#555555", va="bottom")

    # Row labels
    ax.text(49.2, 1, "Original",   va="center", ha="right",
            fontsize=FONT_LABEL, color=COLOR_ORIG)
    ax.text(49.2, 0, "Intervened", va="center", ha="right",
            fontsize=FONT_LABEL, color=COLOR_INTERV)

    ax.set_yticks([])
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_xticklabels(["50", "60", "70", "80", "90", "100"], fontsize=6.5)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Accuracy (%)", fontsize=7, labelpad=3)


# ── Layout ────────────────────────────────────────────────────────────────────
def make_figure():
    n = len(PANELS)

    # 3 rows per panel: [header, content] — merge text+bar into one content row
    # Use GridSpec: rows alternate header / text+bar
    # col 0 = text (wide), col 1 = bar (narrower)
    fig = plt.figure(figsize=(7.2, 6.8))

    # Build outer rows: n panels × [header_row, content_row]
    outer = gridspec.GridSpec(
        n * 2, 1,
        figure=fig,
        height_ratios=[0.18, 1.0] * n,
        hspace=0.08,
    )

    for pi, panel in enumerate(PANELS):
        header_row  = pi * 2
        content_row = pi * 2 + 1

        # ── Panel header ─────────────────────────────────────────────────────
        ax_hdr = fig.add_subplot(outer[header_row])
        ax_hdr.axis("off")
        ax_hdr.set_xlim(0, 1)
        ax_hdr.set_ylim(0, 1)
        ax_hdr.add_patch(FancyBboxPatch(
            (0, 0.05), 1.0, 0.88,
            boxstyle="round,pad=0.02",
            facecolor="#f0f0f0", edgecolor="#bbbbbb", linewidth=0.8,
            transform=ax_hdr.transAxes, clip_on=False,
        ))
        ax_hdr.text(0.012, 0.5, panel["label"],
                    transform=ax_hdr.transAxes,
                    va="center", ha="left",
                    fontsize=9, fontweight="bold", color="#222222")

        # ── Content: text boxes (left) + bar (right) ─────────────────────────
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer[content_row],
            width_ratios=[3.2, 1.0],
            height_ratios=[1, 1],
            hspace=0.08,
            wspace=0.04,
        )

        ax_orig_text  = fig.add_subplot(inner[0, 0])
        ax_interv_text = fig.add_subplot(inner[1, 0])
        ax_bar         = fig.add_subplot(inner[:, 1])   # bar spans both rows

        draw_textbox(ax_orig_text,   ORIGINAL_TEXT,    BOX_ORIG)
        draw_textbox(ax_interv_text, panel["intervened"], BOX_INTERV)
        draw_bar(ax_bar, panel["orig_acc"], panel["interv_acc"],
                 show_chance_label=(pi == n - 1))

    fig.suptitle(
        "Effect of interventions on query text and model accuracy (GPT-5-Nano, easy split)",
        fontsize=9, fontweight="bold", y=1.005,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"intervention_examples.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
