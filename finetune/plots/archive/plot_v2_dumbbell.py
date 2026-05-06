"""
Plot 2: Dumbbell plot — baseline vs intervened accuracy per model, one row per intervention.
Output: outputs/plots/interventions_v2/dumbbell.png
"""

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("outputs/plots/interventions_v2")

CONDITIONS = OrderedDict([
    ("qwen3b_n2000_lora",  ("Qwen2.5-3B LoRA",    "finetuned", "qwen25-3b-instruct-lora",        "#4C72B0")),
    ("olmo32b_n2000_lora", ("OLMo-3.1-32B LoRA",  "finetuned", "olmo3-32b-instruct-lora",        "#DD8452")),
    ("gptoss_base",        ("GPT-OSS-20B base",   "baseline",  "gpt-oss-20b-baseline",           "#55A868")),
])

INTERVENTIONS = OrderedDict([
    ("1_add_space_between_nonspace_characters",   "Add Spaces"),
    ("7_append_1000_high_density_unicode_chars",  "Unicode Noise"),
    ("13_remove_every_other_word_except_numbers", "Del Alt Words"),
    ("22_insert_wrong_answer_once_per_sentence",  "Insert Wrong Ans"),
    ("32_insert_common_emoji_blocks",             "Common Emoji"),
    ("33_insert_rare_emoji_blocks",               "Rare Emoji"),
    ("35_polarity_flip",                          "Polarity Flip†"),
    ("66_set_numbers_to_X",                       "Mask Numbers"),
    ("67_word_replace",                           "Word Replace"),
    ("68_number_replace",                         "Number Replace"),
    ("70_word_replace_polarity_mask",             "Word+Polarity Mask"),
    ("71_word_replace_pct_polarity_mask",         "Word+Pol Mask (%)"),
    ("81_story_swap",                             "Story Swap"),
    ("82_polarity_flip",                          "Q-Dir Flip"),
    ("83_story_swap_polarity_flip",               "Story+Q-Dir Flip"),
    ("84_symbolic_partial",                       "Symbolic (partial)"),
    ("85_symbolic_full",                          "Symbolic (full)"),
    ("86_nonsense_replace",                       "Nonsense Replace"),
])

INTERV_SPLITS = {
    "35_polarity_flip": ["anticommonsense"],
}
DEFAULT_SPLITS = ["easy", "hard", "anticommonsense"]

# vertical offsets within each intervention slot (3 models)
MODEL_OFFSETS = [-0.22, 0.0, 0.22]
MODEL_LS = ["dashed", "dashed", "solid"]
MODEL_MARKER_BASE = ["o", "o", "o"]
MODEL_MARKER_INTV = ["D", "D", "D"]


# ── Data loading ──────────────────────────────────────────────────────────────

def weighted_acc(results):
    if not results:
        return None
    total_n = sum(n for _, n in results)
    return sum(a * n for a, n in results) / total_n if total_n else None


def load_baseline(outputs, cond_key, split):
    _, subdir, run_id, _ = CONDITIONS[cond_key]
    p = outputs / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["acc_all"], d["n"]


def load_interv(results_dir, cond_key, interv, split):
    p = results_dir / cond_key / interv / split / "score" / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["acc_all"], d["n"]


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_dumbbell(outputs, results_dir, split=None):
    cond_keys    = list(CONDITIONS.keys())
    # filter interventions valid for this split
    if split:
        interv_keys = [k for k in INTERVENTIONS
                       if split in INTERV_SPLITS.get(k, DEFAULT_SPLITS)]
    else:
        interv_keys = list(INTERVENTIONS.keys())
    interv_labels = [INTERVENTIONS[k] for k in interv_keys]
    n_interv = len(interv_keys)

    fig, ax = plt.subplots(figsize=(12, 11))

    for ii, interv in enumerate(interv_keys):
        y_center = n_interv - 1 - ii   # top-to-bottom
        splits = [split] if split else INTERV_SPLITS.get(interv, DEFAULT_SPLITS)

        for ci, cond in enumerate(cond_keys):
            label, _, _, color = CONDITIONS[cond]
            y = y_center + MODEL_OFFSETS[ci]

            bl = [r for s in splits if (r := load_baseline(outputs, cond, s))]
            iv = [r for s in splits if (r := load_interv(results_dir, cond, interv, s))]
            b = weighted_acc(bl)
            v = weighted_acc(iv)

            if b is None or v is None:
                continue

            ls = MODEL_LS[ci]
            # Draw connecting line
            ax.plot([b, v], [y, y], color=color, linewidth=1.4,
                    linestyle=ls, alpha=0.75, zorder=2)
            # Baseline: open circle
            ax.scatter(b, y, color="white", edgecolors=color,
                       s=38, linewidths=1.5, zorder=4)
            # Intervened: filled circle
            ax.scatter(v, y, color=color,
                       s=38, zorder=4,
                       marker="o" if ls == "solid" else "D")

        # Horizontal separator
        if ii < n_interv - 1:
            ax.axhline(y_center - 0.5, color="#dddddd", linewidth=0.7, zorder=1)

    # Group shading for intervention categories
    # Order: 6 noise | 6 mask/replace (indices 6-11) | 5 structure (indices 12-16)
    for band_top, band_bot, label in [
        (n_interv - 0.5,  n_interv - 6.5,  "Noise / Corruption"),
        (n_interv - 6.5,  n_interv - 12.5, "Mask / Replace"),
        (n_interv - 12.5, n_interv - 17.5, "Structure / Symbol"),
    ]:
        ax.axhspan(band_bot, band_top, color="#f5f5f5", zorder=0)
        ax.text(-0.005, (band_top + band_bot) / 2, label,
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7.5, color="#888888", rotation=90)

    ax.set_yticks(range(n_interv))
    ax.set_yticklabels(reversed(interv_labels), fontsize=9.5)
    ax.set_xlim(0.3, 1.02)
    split_tag = f"  [{split} split]" if split else "  [all splits pooled]"
    ax.set_xlabel("Accuracy", fontsize=10)
    ax.set_title(f"Baseline → Intervened accuracy per model{split_tag}\n"
                 "(open circle = baseline, filled = under intervention)",
                 fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Legend
    legend_handles = []
    for ci, (cond, (label, _, _, color)) in enumerate(CONDITIONS.items()):
        ls = MODEL_LS[ci]
        marker = "o" if ls == "solid" else "D"
        h = mlines.Line2D([], [], color=color, linewidth=1.5, linestyle=ls,
                          marker=marker, markersize=6,
                          markerfacecolor=color, label=label)
        legend_handles.append(h)
    # baseline marker explanation
    legend_handles.append(
        mlines.Line2D([], [], color="gray", marker="o", markersize=6,
                      markerfacecolor="white", markeredgecolor="gray",
                      linewidth=0, label="← baseline (open)"))

    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
              title="Model", title_fontsize=9, framealpha=0.9)
    ax.annotate("† anticommonsense split only",
                xy=(0.01, 0.005), xycoords="axes fraction",
                fontsize=7.5, color="#666666")

    plt.tight_layout()
    out_dir = OUT_DIR / split if split else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "dumbbell.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    outputs     = Path("outputs")
    results_dir = Path("finetune/eval_results/interventions")
    for split in [None, "easy", "hard", "anticommonsense"]:
        plot_dumbbell(outputs, results_dir, split)
