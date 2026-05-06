"""
Answer-polarity scatter plots — 1×4 grid aggregated across all splits.
Panels: ATE(True) | ATE(False) | ETT(True) | ETT(False)
Extra horizontal space between panels 2 and 3 to separate query types.
Styled to match outputs/plots/paper-ready/learning_curve.png.

Usage:
    cd <repo-root>
    python polarity_analysis/grapher.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import apply_defaults

apply_defaults()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILES = {
    "balanced": Path("original_cladder_data/data/cladder-v1/cladder-v1-q-balanced.json"),
}

SPLITS = ["balanced"]

COLOR_YES = "#009e73"   # teal-green
COLOR_NO  = "#d55e00"   # vermillion

OUT_DIR = Path("polarity_analysis")

# panels: (query_type, polarity, title)
PANELS = [
    ("ate", True,  "ATE (Polarity = True)"),
    ("ate", False, "ATE (Polarity = False)"),
    ("ett", True,  "ETT (Polarity = True)"),
    ("ett", False, "ETT (Polarity = False)"),
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_points(query_type: str, polarity: bool):
    """Aggregate across all splits; return {answer: ([x...], [y...])}."""
    buckets = {"yes": ([], []), "no": ([], [])}
    for split in SPLITS:
        path = DATA_FILES[split]
        if not path.exists():
            continue
        for item in json.loads(path.read_text()):
            meta = item.get("meta", {})
            if meta.get("query_type") != query_type:
                continue
            if meta.get("polarity") is not polarity:
                continue
            probs = meta.get("given_info", {}).get("p(Y | X)")
            if not isinstance(probs, list) or len(probs) != 2:
                continue
            answer = str(item.get("answer", "")).strip().lower()
            if answer not in ("yes", "no"):
                continue
            buckets[answer][0].append(probs[0])
            buckets[answer][1].append(probs[1])
    return buckets


# ── Plot ──────────────────────────────────────────────────────────────────────
def make_figure(out_path: Path):
    # Use GridSpec to insert extra space between panels 2 and 3
    fig = plt.figure(figsize=(7.0, 1.6))
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[1, 1, 1, 1],
        wspace=0.22,   # default spacing within each pair
        left=0.08, right=0.98, top=0.93, bottom=0.18,
    )
    # Override the gap between col 1 and col 2 via a manual position tweak
    # (GridSpec wspace is global; we shift panels 2–3 right after creation)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # Shift panels 3 & 4 (ett) rightward to create a visible gap
    EXTRA_GAP = 0.025
    for ax in axes[2:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + EXTRA_GAP, pos.y0, pos.width, pos.height])

    handles = []
    for ax_i, (ax, (query_type, polarity, title)) in enumerate(zip(axes, PANELS)):
        buckets = load_points(query_type, polarity)

        s_no = ax.scatter(
            buckets["no"][0], buckets["no"][1],
            color=COLOR_NO, s=4, alpha=0.45, linewidths=0,
            label="True answer: No", zorder=3,
        )
        s_yes = ax.scatter(
            buckets["yes"][0], buckets["yes"][1],
            color=COLOR_YES, s=4, alpha=0.45, linewidths=0,
            label="True answer: Yes", zorder=4,
        )
        if ax_i == 0:
            handles = [s_yes, s_no]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
        ax.set_xlabel("p(Y | X=0)", fontsize=8)
        if ax_i == 0:
            ax.set_ylabel("p(Y | X=1)", fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.25, zorder=0)
        ax.tick_params(labelsize=7.5)
        ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8,
                linestyle="--", alpha=0.5, zorder=1)

        if ax_i == 0:
            ax.legend(
                handles=handles,
                fontsize=7.5,
                framealpha=0.45,
                edgecolor="#cccccc",
                loc="upper left",
                markerscale=2.5,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_path.with_suffix('.pdf')}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_figure(OUT_DIR / "polarity_scatter.png")
