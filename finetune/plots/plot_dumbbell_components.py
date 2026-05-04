"""
Modular dumbbell SVGs for paper assembly — all 4 models, colored by model.
Matches the color scheme and data of outputs/plots/paper-ready/dumbbell.png.
Output: outputs/plots/paper-ready/dumbbell_components/

Individual component plots have no x-axis label so they can be stacked manually.
Multi-row panels (panel_a, panel_b) include y-axis intervention labels.
"""

import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUT_DIR     = Path("outputs/plots/paper-ready/dumbbell_components")
OUTPUTS     = Path("outputs")
RESULTS_DIR = Path("finetune/eval_results/interventions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model config (mirrors plot_paper_interventions.py exactly) ────────────────
CONDITIONS = OrderedDict([
    ("qwen3b_n2000_lora",  ("Qwen2.5-3B LoRA",   "finetuned", "qwen25-3b-instruct-lora",    "#4C72B0", "D", "dashed")),
    ("olmo32b_n2000_lora", ("OLMo-3.1-32B LoRA", "finetuned", "olmo3-32b-instruct-lora",    "#DD8452", "D", "dashed")),
    ("gptoss_base",        ("GPT-OSS-20B base",  "baseline",  "gpt-oss-20b-baseline",        "#55A868", "o", "solid")),
    ("gpt5nano_base",      ("GPT-5-Nano base",   "baseline",  "gpt-5-nano-nano496-baseline", "#9467BD", "o", "solid")),
    ("gpt55_base",         ("GPT-5.5 base",      "baseline",  "gpt-5.5-baseline",            "#C44E52", "o", "solid")),
])

DEFAULT_SPLITS = ["easy", "hard", "anticommonsense", "noncommonsense"]
CHANCE_COLOR   = "#888888"
MODEL_OFFSETS  = [-0.36, -0.18, 0.0, 0.18, 0.36]   # vertical nudge per model within each row

# Use acc_valid_only for models that produce invalid (non-yes/no) responses
USE_VALID_ONLY = {"gpt5nano_base"}

# ── Panel definitions ─────────────────────────────────────────────────────────
# Panel A: interventions where accuracy drops (text/meaning disrupted)
ALL_INTERVS = OrderedDict([
    ("67_word_replace",                                          "Word Replace"),
    ("68_number_replace",                                        "Number Replace"),
    ("74_swap_percentages_within_graph_group_and_flip_answers",  "Swap Pct+Flip"),
    ("81_story_swap",                                            "Story Swap"),
    ("86_nonsense_replace",                                      "Nonsense Replace"),
])

# ── Data helpers (same logic as plot_paper_interventions.py) ──────────────────
def _weighted_acc(results):
    if not results:
        return None
    total_n = sum(n for _, n in results)
    return sum(a * n for a, n in results) / total_n if total_n else None


def _acc_key(cond_key):
    return "acc_valid_only" if cond_key in USE_VALID_ONLY else "acc_all"


def _load_baseline(cond_key, split):
    _, subdir, run_id, *_ = CONDITIONS[cond_key]
    p = OUTPUTS / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    n = d["n"] - d.get("invalid", 0) if cond_key in USE_VALID_ONLY else d["n"]
    return d[_acc_key(cond_key)], n


def _load_interv(cond_key, interv_key, split):
    p = RESULTS_DIR / cond_key / interv_key / split / "score" / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    n = d["n"] - d.get("invalid", 0) if cond_key in USE_VALID_ONLY else d["n"]
    return d[_acc_key(cond_key)], n


def get_baseline(cond_key):
    results = [r for s in DEFAULT_SPLITS if (r := _load_baseline(cond_key, s))]
    return _weighted_acc(results)


def get_interv(cond_key, interv_key):
    results = [r for s in DEFAULT_SPLITS if (r := _load_interv(cond_key, interv_key, s))]
    return _weighted_acc(results)


# ── Drawing ───────────────────────────────────────────────────────────────────
def _draw_single(ax, interv_key, xlim=(0.47, 1.02)):
    """One intervention row, all 4 models with vertical offsets."""
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, 0.55)

    for ci, cond_key in enumerate(CONDITIONS.keys()):
        label, _, _, color, marker, ls = CONDITIONS[cond_key]
        y = MODEL_OFFSETS[ci]
        b = get_baseline(cond_key)
        v = get_interv(cond_key, interv_key)
        if b is None or v is None:
            continue

        ax.plot([b, v], [y, y], color=color, linewidth=1.4,
                linestyle=ls, alpha=0.85, zorder=2)
        ax.scatter(b, y, s=38, color="white",
                   edgecolors=color, linewidths=1.6, zorder=4)
        ax.scatter(v, y, s=38, color=color, marker=marker, zorder=4)

    # chance line (no text label)
    ax.axvline(0.5, color=CHANCE_COLOR, linewidth=0.9, linestyle=":", zorder=1)

    ax.set_yticks([])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["50", "60", "70", "80", "90", "100"], fontsize=7)
    ax.tick_params(axis="x", length=2.5, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.3, zorder=0)


# ── One PNG per intervention ──────────────────────────────────────────────────
for interv_key, interv_label in ALL_INTERVS.items():
    fig, ax = plt.subplots(figsize=(2.4, 0.82))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    _draw_single(ax, interv_key)
    plt.tight_layout(pad=0.25)
    slug = interv_label.lower().replace(" ", "_").replace("+", "plus")
    out  = OUT_DIR / f"{slug}_dumbbell.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote: {out}")

# ── Original query baseline (open circles only, no intervention line) ─────────
fig, ax = plt.subplots(figsize=(2.4, 0.82))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.set_xlim(0.47, 1.02)
ax.set_ylim(-0.55, 0.55)
for ci, cond_key in enumerate(CONDITIONS):
    _, _, _, color, _, _ = CONDITIONS[cond_key]
    b = get_baseline(cond_key)
    if b is None:
        continue
    ax.scatter(b, MODEL_OFFSETS[ci], s=38, color="white",
               edgecolors=color, linewidths=1.6, zorder=4)
ax.axvline(0.5, color=CHANCE_COLOR, linewidth=0.9, linestyle=":", zorder=1)
ax.set_yticks([])
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(["50", "60", "70", "80", "90", "100"], fontsize=7)
ax.tick_params(axis="x", length=2.5, pad=1.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.3, zorder=0)
plt.tight_layout(pad=0.25)
out = OUT_DIR / "original_baseline.png"
fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)
print(f"Wrote: {out}")

# ── Legend ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 0.45))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.axis("off")

handles = []
for cond_key, (label, _, _, color, marker, ls) in CONDITIONS.items():
    # baseline open circle
    h_base = mlines.Line2D([], [], marker="o", markersize=6, linewidth=0,
                           color="white", markeredgecolor=color, markeredgewidth=1.6)
    # intervened filled marker
    h_interv = mlines.Line2D([], [], marker=marker, markersize=6, linewidth=1.2,
                              color=color, linestyle=ls,
                              label=label)
    # combine both into one handle tuple (shows as "o—◆" style)
    handles.append(h_interv)

# also add a generic baseline indicator once
h_bl = mlines.Line2D([], [], marker="o", markersize=6, linewidth=0,
                     color="white", markeredgecolor="#555555", markeredgewidth=1.4,
                     label="baseline (open)")
handles.append(h_bl)

ax.legend(
    handles=handles,
    loc="center",
    ncol=len(handles),
    fontsize=7,
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.0,
)

plt.tight_layout(pad=0.1)
out = OUT_DIR / "legend_dumbbell.png"
fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)
print(f"Wrote: {out}")


# ── Multi-row panels (panel_a, panel_b) ───────────────────────────────────────
def _draw_panel(interv_items, out_path, xlim=(0.47, 1.02)):
    """Multi-row dumbbell panel with y-axis intervention labels, no x-axis label."""
    n = len(interv_items)
    row_h = 0.75
    fig, axes = plt.subplots(n, 1, figsize=(2.4, row_h * n),
                             sharex=True, squeeze=False)
    fig.patch.set_alpha(0)

    for row_i, (interv_key, interv_label) in enumerate(interv_items):
        ax = axes[row_i][0]
        ax.patch.set_alpha(0)
        _draw_single(ax, interv_key, xlim=xlim)

        # Add intervention name as y-axis label on each row
        ax.set_ylabel(interv_label, fontsize=7.5, rotation=0,
                      ha="right", va="center", labelpad=4)
        ax.yaxis.set_label_coords(-0.02, 0.5)

        # Only bottom row keeps x-tick labels; all rows suppress x-axis label
        if row_i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)

        # Separator line between rows
        if row_i < n - 1:
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_color("#e0e0e0")
            ax.spines["bottom"].set_linewidth(0.6)

    fig.subplots_adjust(hspace=0.08, left=0.22, right=0.98,
                        top=0.98, bottom=0.06)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote: {out_path}")


PANEL_A = [
    ("67_word_replace",   "Word Replace"),
    ("68_number_replace", "Number Replace"),
    ("81_story_swap",     "Story Swap"),
]

_draw_panel(PANEL_A, OUT_DIR / "panel_a_dumbbell.png")


# ── Baseline per split (one file per split) ───────────────────────────────────
for split in DEFAULT_SPLITS:
    fig, ax = plt.subplots(figsize=(2.4, 0.82))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_xlim(0.47, 1.02)
    ax.set_ylim(-0.55, 0.55)

    for ci, cond_key in enumerate(CONDITIONS):
        _, subdir, run_id, color, *_ = CONDITIONS[cond_key]
        p = OUTPUTS / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "summary.json"
        if not p.exists():
            continue
        acc = json.load(open(p))[_acc_key(cond_key)]
        ax.scatter(acc, MODEL_OFFSETS[ci], s=38, color="white",
                   edgecolors=color, linewidths=1.6, zorder=4)

    ax.axvline(0.5, color=CHANCE_COLOR, linewidth=0.9, linestyle=":", zorder=1)
    ax.set_yticks([])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["50", "60", "70", "80", "90", "100"], fontsize=7)
    ax.tick_params(axis="x", length=2.5, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.3, zorder=0)

    plt.tight_layout(pad=0.25)
    out = OUT_DIR / f"baseline_{split}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote: {out}")
