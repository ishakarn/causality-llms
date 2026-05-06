"""
Plot 1: Change heatmap — accuracy change (intervened − baseline) per model × intervention.
Negative = worse (red), positive = improved/resistant (green).
Output: outputs/plots/interventions_v2/heatmap_drop.png
"""

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("outputs/plots/interventions_v2")

CONDITIONS = OrderedDict([
    ("qwen3b_n2000_lora",  ("Qwen2.5-3B\nLoRA",       "finetuned", "qwen25-3b-instruct-lora")),
    ("olmo32b_n2000_lora", ("OLMo-3.1-32B\nLoRA",     "finetuned", "olmo3-32b-instruct-lora")),
    ("gptoss_base",        ("GPT-OSS-20B\nbase",      "baseline",  "gpt-oss-20b-baseline")),
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

# splits available per intervention
INTERV_SPLITS = {
    "35_polarity_flip": ["anticommonsense"],
}
DEFAULT_SPLITS = ["easy", "hard", "anticommonsense"]


# ── Data loading ──────────────────────────────────────────────────────────────

def weighted_acc(results: list) -> float | None:
    """Compute weighted-average accuracy from list of (acc, n) tuples."""
    if not results:
        return None
    total_n = sum(n for _, n in results)
    if total_n == 0:
        return None
    return sum(acc * n for acc, n in results) / total_n


def load_baseline(outputs: Path, cond_key: str, split: str) -> tuple[float, int] | None:
    _, subdir, run_id = CONDITIONS[cond_key]
    p = outputs / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["acc_all"], d["n"]


def load_interv(results_dir: Path, cond_key: str, interv: str, split: str) -> tuple[float, int] | None:
    p = results_dir / cond_key / interv / split / "score" / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["acc_all"], d["n"]


def build_matrix(outputs: Path, results_dir: Path, split: str | None = None):
    cond_keys = list(CONDITIONS.keys())
    interv_keys = list(INTERVENTIONS.keys())
    change   = np.full((len(cond_keys), len(interv_keys)), np.nan)
    base_acc = np.full((len(cond_keys), len(interv_keys)), np.nan)
    int_acc  = np.full((len(cond_keys), len(interv_keys)), np.nan)

    for ci, cond in enumerate(cond_keys):
        for ii, interv in enumerate(interv_keys):
            splits = [split] if split else INTERV_SPLITS.get(interv, DEFAULT_SPLITS)
            # skip if this split isn't valid for this intervention
            valid = INTERV_SPLITS.get(interv, DEFAULT_SPLITS)
            if split and split not in valid:
                continue
            bl = [r for s in splits if (r := load_baseline(outputs, cond, s))]
            iv = [r for s in splits if (r := load_interv(results_dir, cond, interv, s))]
            b = weighted_acc(bl)
            v = weighted_acc(iv)
            if b is not None and v is not None:
                change[ci, ii]   = v - b   # negative = worse, positive = improved
                base_acc[ci, ii] = b
                int_acc[ci, ii]  = v

    return change, base_acc, int_acc


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_heatmap(change: np.ndarray, base_acc: np.ndarray, int_acc: np.ndarray,
                 split: str | None = None):
    cond_labels  = [v[0] for v in CONDITIONS.values()]
    interv_labels = list(INTERVENTIONS.values())

    fig, ax = plt.subplots(figsize=(17, 4.5))

    vlim = 0.45
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.imshow(change, aspect="auto", cmap=cmap, norm=norm)

    # Annotate cells
    for ci in range(change.shape[0]):
        for ii in range(change.shape[1]):
            if np.isnan(change[ci, ii]):
                ax.text(ii, ci, "—", ha="center", va="center", fontsize=8, color="#999999")
            else:
                d = change[ci, ii]
                color = "white" if abs(d) > 0.28 else "black"
                ax.text(ii, ci, f"{d:+.2f}", ha="center", va="center",
                        fontsize=7.5, fontweight="bold", color=color)

    ax.set_xticks(range(len(interv_labels)))
    ax.set_xticklabels(interv_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(cond_labels)))
    ax.set_yticklabels(cond_labels, fontsize=9)
    split_tag = f"  [{split} split]" if split else "  [all splits pooled]"
    ax.set_title(f"Accuracy change under intervention  (intervened − baseline){split_tag}",
                 fontsize=12, fontweight="bold", pad=10)

    # Vertical dividers: noise (0-5) | mask/replace (6-11) | structure/symbol (12-16)
    ax.axvline(5.5,  color="white", linewidth=2)   # after Rare Emoji
    ax.axvline(11.5, color="white", linewidth=2)   # after Word+Pol Mask (%)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Accuracy change  (negative = worse, positive = improved)", fontsize=9)

    ax.annotate("† anticommonsense split only", xy=(0, -0.22),
                xycoords="axes fraction", fontsize=7.5, color="#666666")

    plt.tight_layout()
    out_dir = OUT_DIR / split if split else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "heatmap_drop.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    outputs     = Path("outputs")
    results_dir = Path("finetune/eval_results/interventions")
    for split in [None, "easy", "hard", "anticommonsense"]:
        drop, base_acc, int_acc = build_matrix(outputs, results_dir, split)
        plot_heatmap(drop, base_acc, int_acc, split)
