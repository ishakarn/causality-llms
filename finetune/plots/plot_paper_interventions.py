"""
Paper-ready intervention plots for NeurIPS (single-column, short & wide).
Subset: 6 key interventions that tell the causal-reasoning story.
Output: outputs/plots/paper-ready/dumbbell.{png,pdf}
        outputs/plots/paper-ready/heatmap.{png,pdf}
"""

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR     = Path("outputs/plots/paper-ready")
OUTPUTS     = Path("outputs")
RESULTS_DIR = Path("finetune/eval_results/interventions")

# ── Shared config ─────────────────────────────────────────────────────────────
CONDITIONS = OrderedDict([
    ("qwen3b_n2000_lora",  ("Qwen2.5-3B LoRA",   "finetuned", "qwen25-3b-instruct-lora",        "#4C72B0", "D", "dashed")),
    ("olmo32b_n2000_lora", ("OLMo-3.1-32B LoRA", "finetuned", "olmo3-32b-instruct-lora",        "#DD8452", "D", "dashed")),
    ("gptoss_base",        ("GPT-OSS-20B base",  "baseline",  "gpt-oss-20b-baseline",            "#55A868", "o", "solid")),
    ("gpt5nano_base",      ("GPT-5-Nano base",   "baseline",  "gpt-5-nano-nano496-baseline",     "#9467BD", "o", "solid")),
])

QTYPE_ORDER = ["marginal", "correlation", "backadj", "ate", "ett",
               "nie", "nde", "det-counterfactual", "exp_away", "collider_bias"]
QTYPE_LABELS = {
    "marginal": "MAR", "correlation": "COR", "backadj": "BKA",
    "ate": "ATE", "ett": "ETT", "nie": "NIE", "nde": "NDE",
    "det-counterfactual": "DCF", "exp_away": "EXP", "collider_bias": "COL",
}

# 6 key interventions in narrative order
PAPER_INTERVENTIONS = OrderedDict([
    ("66_set_numbers_to_X",               "Mask Numbers"),
    ("67_word_replace",                   "Word Replace"),
    ("68_number_replace",                 "Number Replace"),
    ("70_word_replace_polarity_mask",     "Word+Polarity Mask"),
    ("71_word_replace_pct_polarity_mask", "Word+Pol Mask (%)"),
    ("81_story_swap",                     "Story Swap"),
    ("86_nonsense_replace",               "Nonsense Replace"),
])

DEFAULT_SPLITS = ["easy", "hard", "anticommonsense"]

# ── Data helpers ──────────────────────────────────────────────────────────────
def weighted_acc(results):
    if not results:
        return None
    total_n = sum(n for _, n in results)
    return sum(a * n for a, n in results) / total_n if total_n else None


def load_baseline(cond_key, split):
    _, subdir, run_id, *_ = CONDITIONS[cond_key]
    p = OUTPUTS / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["acc_all"], d["n"]


def load_interv(cond_key, interv, split):
    p = RESULTS_DIR / cond_key / interv / split / "score" / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["acc_all"], d["n"]


def load_qtype_csv(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {row["query_type"]: (row["acc_all"], row["n"]) for _, row in df.iterrows()}


def get_baseline_qtype(cond_key):
    _, subdir, run_id, *_ = CONDITIONS[cond_key]
    pooled = defaultdict(lambda: [0, 0])
    for split in DEFAULT_SPLITS:
        p = OUTPUTS / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "per_query_type.csv"
        for qt, (acc, n) in load_qtype_csv(p).items():
            pooled[qt][0] += round(acc * n)
            pooled[qt][1] += n
    return {qt: v[0] / v[1] for qt, v in pooled.items() if v[1] > 0}


def get_interv_qtype(cond_key, interv_key):
    pooled = defaultdict(lambda: [0, 0])
    for split in DEFAULT_SPLITS:
        p = RESULTS_DIR / cond_key / interv_key / split / "score" / "per_query_type.csv"
        for qt, (acc, n) in load_qtype_csv(p).items():
            pooled[qt][0] += round(acc * n)
            pooled[qt][1] += n
    return {qt: v[0] / v[1] for qt, v in pooled.items() if v[1] > 0}


def get_baseline(cond_key):
    results = [r for s in DEFAULT_SPLITS if (r := load_baseline(cond_key, s))]
    return weighted_acc(results)


def get_interv(cond_key, interv_key):
    results = [r for s in DEFAULT_SPLITS if (r := load_interv(cond_key, interv_key, s))]
    return weighted_acc(results)


# ── Plot 1: Dumbbell ──────────────────────────────────────────────────────────
def plot_dumbbell():
    MODEL_OFFSETS = [-0.30, -0.10, 0.10, 0.30]
    MODEL_LS      = ["dashed", "dashed", "solid", "solid"]
    cond_keys     = list(CONDITIONS.keys())
    interv_keys   = list(PAPER_INTERVENTIONS.keys())
    interv_labels = list(PAPER_INTERVENTIONS.values())
    n_interv      = len(interv_keys)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for ii, interv in enumerate(interv_keys):
        y_center = n_interv - 1 - ii

        for ci, cond in enumerate(cond_keys):
            label, _, _, color, *_ = CONDITIONS[cond]
            y  = y_center + MODEL_OFFSETS[ci]
            b  = get_baseline(cond)
            v  = get_interv(cond, interv)
            if b is None or v is None:
                continue

            ls = MODEL_LS[ci]
            ax.plot([b, v], [y, y], color=color, linewidth=1.4,
                    linestyle=ls, alpha=0.8, zorder=2)
            ax.scatter(b, y, color="white", edgecolors=color,
                       s=36, linewidths=1.5, zorder=4)
            ax.scatter(v, y, color=color, s=36, zorder=4,
                       marker="o" if ls == "solid" else "D")

        if ii < n_interv - 1:
            ax.axhline(y_center - 0.5, color="#e0e0e0", linewidth=0.7, zorder=1)

    ax.set_yticks(range(n_interv))
    ax.set_yticklabels(reversed(interv_labels), fontsize=9)
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(-0.55, n_interv - 0.45)
    ax.set_xlabel("Accuracy (all splits pooled)", fontsize=9)
    ax.set_title("Model accuracy under intervention",
                 fontsize=10, fontweight="bold")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.45)
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    # Legend
    handles = []
    for ci, (cond, (label, _, _, color, *_)) in enumerate(CONDITIONS.items()):
        ls = MODEL_LS[ci]
        h = mlines.Line2D([], [], color=color, linewidth=1.4, linestyle=ls,
                          marker="o" if ls == "solid" else "D", markersize=5.5,
                          markerfacecolor=color, label=label)
        handles.append(h)
    handles.append(
        mlines.Line2D([], [], color="gray", marker="o", markersize=5.5,
                      markerfacecolor="white", markeredgecolor="gray",
                      linewidth=0, label="Baseline"))
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"dumbbell.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close()


# ── Plot 2: Heatmap ───────────────────────────────────────────────────────────
def plot_heatmap():
    cond_keys     = list(CONDITIONS.keys())
    interv_keys   = list(PAPER_INTERVENTIONS.keys())
    cond_labels   = [v[0] for v in CONDITIONS.values()]
    interv_labels = list(PAPER_INTERVENTIONS.values())

    change   = np.full((len(cond_keys), len(interv_keys)), np.nan)

    for ci, cond in enumerate(cond_keys):
        for ii, interv in enumerate(interv_keys):
            b = get_baseline(cond)
            v = get_interv(cond, interv)
            if b is not None and v is not None:
                change[ci, ii] = v - b

    fig, ax = plt.subplots(figsize=(7.0, 2.2))

    vlim = 0.45
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.imshow(change, aspect="auto", cmap=cmap, norm=norm)

    for ci in range(change.shape[0]):
        for ii in range(change.shape[1]):
            if np.isnan(change[ci, ii]):
                ax.text(ii, ci, "—", ha="center", va="center", fontsize=8.5, color="#999999")
            else:
                d = change[ci, ii]
                color = "white" if abs(d) > 0.28 else "black"
                ax.text(ii, ci, f"{d:+.2f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=color)

    ax.set_xticks(range(len(interv_labels)))
    ax.set_xticklabels(interv_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(cond_labels)))
    ax.set_yticklabels(cond_labels, fontsize=9)
    ax.set_title("Accuracy change under intervention (all splits pooled)",
                 fontsize=10, fontweight="bold", pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, aspect=12)
    cbar.set_label("Accuracy change", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"heatmap.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close()


# ── Plot 3: Query-type strip ──────────────────────────────────────────────────
def plot_qtype_strip():
    interv_keys   = list(PAPER_INTERVENTIONS.keys())
    interv_labels = list(PAPER_INTERVENTIONS.values())
    qtypes        = [qt for qt in QTYPE_ORDER if qt in QTYPE_LABELS]
    qt_labels     = [QTYPE_LABELS[qt] for qt in qtypes]
    n_qt          = len(qtypes)
    n_interv      = len(interv_keys)
    model_offsets = [-0.30, -0.10, 0.10, 0.30]

    fig, axes = plt.subplots(1, n_interv, figsize=(8.5, 4.2),
                             sharey=True, sharex=True)

    for ax_i, (interv, interv_label) in enumerate(zip(interv_keys, interv_labels)):
        ax = axes[ax_i]

        for qi, qt in enumerate(qtypes):
            for ci, (cond, (label, _, _, color, marker, ls)) in enumerate(CONDITIONS.items()):
                b_map = get_baseline_qtype(cond)
                v_map = get_interv_qtype(cond, interv)
                b = b_map.get(qt)
                v = v_map.get(qt)
                if b is None or v is None:
                    continue
                y = n_qt - 1 - qi + model_offsets[ci]
                ax.scatter(v - b, y, color=color, marker=marker, s=22, zorder=4, alpha=0.9)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.35)
        ax.set_title(interv_label, fontsize=8.5, fontweight="bold", pad=5)
        ax.set_xlim(-0.65, 0.35)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.set_xlabel("Acc. change", fontsize=7.5)

        if ax_i == 0:
            ax.set_yticks(range(n_qt))
            ax.set_yticklabels(reversed(qt_labels), fontsize=8.5)

        for qi in range(n_qt - 1):
            ax.axhline(n_qt - 1 - qi - 0.5, color="#e0e0e0", linewidth=0.6)

    fig.suptitle("Per-query-type accuracy change under intervention (all splits pooled)",
                 fontsize=10, fontweight="bold", y=1.02)

    handles = [mlines.Line2D([], [], color=c, marker=m, markersize=5.5,
                             linewidth=0, label=lbl)
               for _, (lbl, _, _, c, m, _) in CONDITIONS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"qtype_strip.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_dumbbell()
    plot_heatmap()
    plot_qtype_strip()
