"""
Plot intervention robustness results for base vs fine-tuned OLMo-3-7B-Instruct.

Produces:
  1. outputs/plots/interventions/overview.png
       — all interventions × both models, overall accuracy + Wilson 95% CI
  2. outputs/plots/interventions/per_intervention/<stem>.png  (one per intervention)
       — per-query-type accuracy with Wilson 95% CI, 4 bars per query type:
         base-original, base-intervention, ft-original, ft-intervention

Usage:
    python finetune/plot_interventions.py \\
        --results_dir finetune/intervention_results \\
        --out_dir     outputs/plots/interventions
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plot_style import apply_defaults, model_color, HATCH_FINETUNED, ALPHA_BAR, LINEWIDTH_BAR
from query_type_labels import abbrev, sort_key, RUNG_ORDER

INTERVENTION_LABELS = {
    "1_insert_spaces_between_chars_without_whitespace":         "char_spaces",
    "7_append_1000_high_density_unicode_chars":                 "unicode_noise",
    "13_remove_every_other_word":                               "del_alt_words",
    "22_insert_incorrect_answer_once_per_sentence":             "insert_wrong_ans",
    "33_insert_rare_emoji_blocks":                              "emoji_blocks",
    "34_polarity_decrease_to_not_decrease_flip":                "polarity_not_dec",
    "35_polarity_decrease_to_increase_flip":                    "polarity_increase",
    "37_polarity_decrease_to_not_not_decrease_same":            "polarity_notnot",
    "62_intervention_word_order":                               "shuffle_words",
    "66_set_percentages_to_x":                                  "mask_numbers",
    "67_replace_scenario_words_with_random_words_corrected":    "random_words",
    "68_replace_scenario_words_with_consistent_numbers":        "words_to_nums",
    "69_scenario_words_to_random_words_and_percentages_to_x":  "rand_words+mask_nums",
    "70_scenario_words_to_random_words_and_mask_polarity":      "rand_words+mask_pol",
    "71_scenario_words_to_random_words_and_percentages_to_x_and_mask_polarity": "full_ablation",
    "74_swap_percentages_within_graph_group_and_invert_answers":"swap_pct+inv_ans",
    "75_swap_percentages_within_graph_group_and_invert_polarity":"swap_pct+inv_pol",
    "77_remove_all_filler_words":                               "no_filler",
}

INTERVENTION_GROUPS = [
    ("Noise / surface",      ["char_spaces", "unicode_noise", "emoji_blocks",
                               "del_alt_words", "shuffle_words"]),
    ("Semantic ablations",   ["no_filler", "mask_numbers", "random_words", "words_to_nums"]),
    ("Combined ablations",   ["rand_words+mask_nums", "rand_words+mask_pol", "full_ablation"]),
    ("Polarity / ans flips", ["polarity_not_dec", "polarity_increase", "polarity_notnot",
                               "swap_pct+inv_ans", "swap_pct+inv_pol"]),
    ("Distractor",           ["insert_wrong_ans"]),
]
ORDERED_LABELS = [lbl for _, lbls in INTERVENTION_GROUPS for lbl in lbls]

# Verified by comparing intervention file answers vs original data:
#   34_polarity_decrease_to_not_decrease_flip      → 100% flipped  (†)
#   35_polarity_decrease_to_increase_flip          → 100% flipped  (†)
#   74_swap_percentages_within_graph_group_and_invert_answers → 66.7% flipped (‡)
#   75_swap_percentages_within_graph_group_and_invert_polarity → 0% flipped (no annotation)
# Accuracy in eval is already computed vs the flipped gold (read from intervention file).
ANSWER_CHANGED_FULL    = {"polarity_not_dec", "polarity_increase"}        # 100% flipped
ANSWER_CHANGED_PARTIAL = {"swap_pct+inv_ans"}                             # 66.7% flipped
ANSWER_CHANGED         = ANSWER_CHANGED_FULL | ANSWER_CHANGED_PARTIAL     # any flip


# ── Wilson 95% CI ──────────────────────────────────────────────────────────────

def wilson_ci(correct: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Returns (acc, ci_lo, ci_hi). Returns (nan,nan,nan) if n==0."""
    if n == 0:
        nan = float("nan")
        return nan, nan, nan
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_all_results(results_dir: Path) -> Dict:
    """
    Returns nested dict:
      data[model][stem] = list of {question_id, query_type, gold, pred}
    model ∈ {"base", "finetuned"}, stem ∈ {"original", <intervention_stem>}
    """
    data = {}
    for model in ("base", "finetuned"):
        data[model] = {}
        model_dir = results_dir / model
        if not model_dir.exists():
            continue
        for jpath in sorted(model_dir.glob("*.jsonl")):
            stem = jpath.stem
            data[model][stem] = load_jsonl(jpath)
    return data


def compute_stats(records: List[Dict], query_type: Optional[str] = None
                  ) -> Tuple[float, float, float, int]:
    """Compute (acc, ci_lo, ci_hi, n) for records, optionally filtered by query_type."""
    if query_type:
        records = [r for r in records if r.get("query_type") == query_type]
    n       = len(records)
    correct = sum(1 for r in records if r["pred"] == r["gold"])
    acc, lo, hi = wilson_ci(correct, n)
    return acc, lo, hi, n


# ── Overview plot ─────────────────────────────────────────────────────────────

def plot_overview(data: Dict, out_path: Path, model_name: str = "model"):
    """Two-panel overview: absolute acc (top) + acc drop (bottom), with Wilson CIs."""
    apply_defaults()
    COLOR = model_color(model_name)

    orig_base_recs = data["base"].get("original", [])
    orig_ft_recs   = data["finetuned"].get("original", [])
    orig_base_acc, orig_base_lo, orig_base_hi, _ = compute_stats(orig_base_recs)
    orig_ft_acc,   orig_ft_lo,   orig_ft_hi,   _ = compute_stats(orig_ft_recs)

    stems_in_order = []
    for _, lbls in INTERVENTION_GROUPS:
        for lbl in lbls:
            stem = next((s for s, l in INTERVENTION_LABELS.items() if l == lbl), None)
            if stem:
                stems_in_order.append((lbl, stem))

    n_int   = len(stems_in_order)
    x       = np.arange(n_int)
    width   = 0.38

    base_acc, base_lo, base_hi = [], [], []
    ft_acc,   ft_lo,   ft_hi   = [], [], []

    for lbl, stem in stems_in_order:
        a, lo, hi, _ = compute_stats(data["base"].get(stem, []))
        base_acc.append(a); base_lo.append(lo); base_hi.append(hi)
        a, lo, hi, _ = compute_stats(data["finetuned"].get(stem, []))
        ft_acc.append(a); ft_lo.append(lo); ft_hi.append(hi)

    base_acc  = np.array(base_acc,  dtype=float)
    ft_acc    = np.array(ft_acc,    dtype=float)
    base_drop = orig_base_acc - base_acc
    ft_drop   = orig_ft_acc   - ft_acc

    # Error bars for absolute acc (asymmetric Wilson)
    base_err_lo = base_acc - np.array(base_lo, dtype=float)
    base_err_hi = np.array(base_hi, dtype=float) - base_acc
    ft_err_lo   = ft_acc   - np.array(ft_lo,   dtype=float)
    ft_err_hi   = np.array(ft_hi,   dtype=float)   - ft_acc

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(
        f"Intervention Robustness — {model_name} (test split, n=288)\n"
        "Zero-shot base vs LoRA fine-tuned  |  Wilson 95% CI",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Top: absolute accuracy ─────────────────────────────────────────────
    ax = axes[0]
    ax.bar(x - width/2, base_acc, width,
           color=COLOR, alpha=ALPHA_BAR, linewidth=LINEWIDTH_BAR, edgecolor="black",
           label="Base (zero-shot)",
           yerr=[base_err_lo, base_err_hi], error_kw=dict(ecolor="black", capsize=3, lw=1))
    ax.bar(x + width/2, ft_acc, width,
           color=COLOR, alpha=ALPHA_BAR, linewidth=LINEWIDTH_BAR, edgecolor="black",
           hatch=HATCH_FINETUNED, label="Fine-tuned (LoRA)",
           yerr=[ft_err_lo, ft_err_hi], error_kw=dict(ecolor="black", capsize=3, lw=1))

    ax.axhline(orig_base_acc, color=COLOR, linestyle="--", lw=1.2, alpha=0.6,
               label=f"Base original ({orig_base_acc:.3f})")
    ax.axhline(orig_ft_acc,   color=COLOR,  linestyle="-.", lw=1.2, alpha=0.6,
               label=f"FT original ({orig_ft_acc:.3f})")
    ax.axhline(0.5, color="gray", linestyle=":", lw=0.8, alpha=0.5)

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_title("Absolute accuracy per intervention  (error bars = Wilson 95% CI)", fontsize=10)

    for i, (lbl, _) in enumerate(stems_in_order):
        if lbl in ANSWER_CHANGED_FULL:
            ax.text(x[i], 1.03, "†", ha="center", fontsize=9, color="gray")
        elif lbl in ANSWER_CHANGED_PARTIAL:
            ax.text(x[i], 1.03, "‡", ha="center", fontsize=9, color="gray")

    # ── Bottom: accuracy drop ──────────────────────────────────────────────
    ax2 = axes[1]
    # Propagate CI width as uncertainty on drop (conservative: use bar CI widths)
    ax2.bar(x - width/2, base_drop, width,
            color=COLOR, alpha=ALPHA_BAR, linewidth=LINEWIDTH_BAR, edgecolor="black",
            label="Base drop",
            yerr=[base_err_lo + (orig_base_acc - orig_base_lo),
                  base_err_hi + (orig_base_hi  - orig_base_acc)],
            error_kw=dict(ecolor="black", capsize=3, lw=1))
    ax2.bar(x + width/2, ft_drop, width,
            color=COLOR, alpha=ALPHA_BAR, linewidth=LINEWIDTH_BAR, edgecolor="black",
            hatch=HATCH_FINETUNED, label="Fine-tuned drop",
            yerr=[ft_err_lo + (orig_ft_acc - orig_ft_lo),
                  ft_err_hi + (orig_ft_hi  - orig_ft_acc)],
            error_kw=dict(ecolor="black", capsize=3, lw=1))

    ax2.axhline(0, color="black", lw=0.8)
    all_drops = [v for v in list(base_drop) + list(ft_drop) if not math.isnan(v)]
    ymax = max(max(all_drops, default=0.5) + 0.1, 0.5)
    ymin = min(min(all_drops, default=0.0) - 0.08, -0.1)
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel("Accuracy drop  (original − intervention)")
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_title("Accuracy drop vs unperturbed original  (positive = worse)", fontsize=10)

    tick_labels = [lbl.replace("+", "+\n") for lbl, _ in stems_in_order]
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=8)

    # Group separators + labels
    pos = 0
    for grp_title, lbls in INTERVENTION_GROUPS:
        n_g = len(lbls)
        for panel in axes:
            panel.axvline(pos - 0.5, color="lightgray", lw=0.7, zorder=0)
        mid = pos + (n_g - 1) / 2
        ylo = ax2.get_ylim()[0]
        yr  = ax2.get_ylim()[1] - ax2.get_ylim()[0]
        axes[1].text(mid, ylo - 0.06 * yr, grp_title,
                     ha="center", va="top", fontsize=7.5, color="dimgray", style="italic",
                     transform=axes[1].get_xaxis_transform())
        pos += n_g

    fig.text(0.01, 0.005,
             "† Gold answer 100% flipped  |  ‡ Gold answer partially flipped (66.7%)  "
             "|  Accuracy is computed vs the flipped gold in both cases",
             fontsize=7.5, color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote: {out_path}")
    plt.close(fig)


# ── Per-intervention per-query-type plot ──────────────────────────────────────

def plot_per_intervention(data: Dict, stem: str, label: str, out_dir: Path,
                          model_name: str = "model"):
    """
    For one intervention: grouped bars by query type.
    4 bars per group: base-orig, base-intv, ft-orig, ft-intv.
    Wilson 95% CIs on all bars.
    """
    apply_defaults()
    COLOR = model_color(model_name)

    query_types = sorted(RUNG_ORDER, key=sort_key) + ["overall"]

    orig_base_recs = data["base"].get("original", [])
    orig_ft_recs   = data["finetuned"].get("original", [])
    intv_base_recs = data["base"].get(stem, [])
    intv_ft_recs   = data["finetuned"].get(stem, [])

    if not intv_base_recs and not intv_ft_recs:
        print(f"  [skip] {stem}: no data")
        return

    tick_labels = []
    x_all = []
    bo_acc, bo_lo, bo_hi = [], [], []
    bi_acc, bi_lo, bi_hi = [], [], []
    fo_acc, fo_lo, fo_hi = [], [], []
    fi_acc, fi_lo, fi_hi = [], [], []
    n_labels = []  # sample sizes for annotation

    for qt in query_types:
        qt_filter = None if qt == "overall" else qt
        a, lo, hi, n = compute_stats(orig_base_recs, qt_filter)
        if n == 0:
            continue
        bo_acc.append(a); bo_lo.append(lo); bo_hi.append(hi)
        a, lo, hi, _ = compute_stats(orig_ft_recs,   qt_filter)
        fo_acc.append(a); fo_lo.append(lo); fo_hi.append(hi)
        a, lo, hi, _ = compute_stats(intv_base_recs, qt_filter)
        bi_acc.append(a); bi_lo.append(lo); bi_hi.append(hi)
        a, lo, hi, _ = compute_stats(intv_ft_recs,   qt_filter)
        fi_acc.append(a); fi_lo.append(lo); fi_hi.append(hi)

        tick_labels.append(abbrev(qt))
        n_labels.append(n)

    n_qt  = len(tick_labels)
    x     = np.arange(n_qt)
    width = 0.20   # 4 bars per group

    def err(acc_list, lo_list, hi_list):
        a = np.array(acc_list, dtype=float)
        return [a - np.array(lo_list, dtype=float),
                np.array(hi_list, dtype=float) - a]

    fig, ax = plt.subplots(figsize=(max(10, n_qt * 1.2), 5))

    kw = dict(linewidth=LINEWIDTH_BAR, edgecolor="black",
              error_kw=dict(ecolor="black", capsize=3, lw=1))

    ax.bar(x - 1.5*width, bo_acc, width, color=COLOR, alpha=ALPHA_BAR,
           label="Base — original", yerr=err(bo_acc, bo_lo, bo_hi), **kw)
    ax.bar(x - 0.5*width, bi_acc, width, color=COLOR, alpha=ALPHA_BAR,
           hatch=HATCH_FINETUNED, label="Base — intervention",
           yerr=err(bi_acc, bi_lo, bi_hi), **kw)
    ax.bar(x + 0.5*width, fo_acc, width, color=COLOR, alpha=0.55,
           label="Fine-tuned — original", yerr=err(fo_acc, fo_lo, fo_hi), **kw)
    ax.bar(x + 1.5*width, fi_acc, width, color=COLOR, alpha=0.55,
           hatch=HATCH_FINETUNED, label="Fine-tuned — intervention",
           yerr=err(fi_acc, fi_lo, fi_hi), **kw)

    # Separator before "overall"
    ax.axvline(n_qt - 1 - 0.5, color="lightgray", lw=0.8, zorder=0)
    ax.axhline(0.5, color="gray", linestyle=":", lw=0.8, alpha=0.5)

    if label in ANSWER_CHANGED_FULL:
        flag = "  † gold 100% flipped"
    elif label in ANSWER_CHANGED_PARTIAL:
        flag = "  ‡ gold 66.7% flipped"
    else:
        flag = ""
    ax.set_title(
        f"{model_name}  |  {label}{flag}  —  per-query-type accuracy\n"
        f"Base vs Fine-tuned  |  solid = original, hatched = intervention  |  Wilson 95% CI",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    # Annotate n under each tick
    for i, n in enumerate(n_labels):
        ax.text(x[i], -0.07, f"n={n}", ha="center", va="top",
                fontsize=6.5, color="gray", transform=ax.get_xaxis_transform())

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    if label in ANSWER_CHANGED:
        note = ("† Gold answer 100% flipped" if label in ANSWER_CHANGED_FULL
                else "‡ Gold answer 66.7% flipped")
        ax.text(0.01, 0.01, f"{note}  |  accuracy computed vs flipped gold",
                transform=ax.transAxes, fontsize=7.5, color="gray", va="bottom")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = out_dir / f"{stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Wrote: {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="finetune/intervention_results",
                    help="Directory containing base/ and finetuned/ subdirs of JSONL files.")
    ap.add_argument("--out_dir",     default="outputs/plots/interventions")
    ap.add_argument("--model_name",  default=None,
                    help="Human-readable model label for plot titles and color lookup. "
                         "Defaults to the basename of results_dir.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    model_name  = args.model_name or results_dir.name

    data = load_all_results(results_dir)

    # Overview plot
    plot_overview(data, out_dir / "overview.png", model_name=model_name)

    # Per-intervention plots
    per_int_dir = out_dir / "per_intervention"
    for stem, label in INTERVENTION_LABELS.items():
        if stem in data.get("base", {}) or stem in data.get("finetuned", {}):
            plot_per_intervention(data, stem, label, per_int_dir, model_name=model_name)

    print(f"\nDone. Plots in {out_dir}/")


if __name__ == "__main__":
    main()
