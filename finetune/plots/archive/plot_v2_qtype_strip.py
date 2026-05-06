"""
Plot 3: Per-query-type strip plot — accuracy drop per query type, faceted by intervention.
Auto-selects the TOP_N interventions by mean accuracy drop across all model conditions.
Output: outputs/plots/interventions_v2/qtype_strip.png
"""

import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("outputs/plots/interventions_v2")
TOP_N   = 5   # number of interventions to show, ranked by mean drop

CONDITIONS = OrderedDict([
    ("qwen3b_n2000_lora",  ("Qwen2.5-3B LoRA",    "finetuned", "qwen25-3b-instruct-lora",        "#4C72B0", "D",  "dashed")),
    ("olmo32b_n2000_lora", ("OLMo-3.1-32B LoRA",  "finetuned", "olmo3-32b-instruct-lora",        "#DD8452", "D",  "dashed")),
    ("gptoss_base",        ("GPT-OSS-20B base",   "baseline",  "gpt-oss-20b-baseline",           "#55A868", "o",  "solid")),
])

ALL_INTERVENTIONS = {
    "1_add_space_between_nonspace_characters":   "Add Spaces",
    "7_append_1000_high_density_unicode_chars":  "Unicode Noise",
    "13_remove_every_other_word_except_numbers": "Del Alt Words",
    "22_insert_wrong_answer_once_per_sentence":  "Insert Wrong Ans",
    "32_insert_common_emoji_blocks":             "Common Emoji",
    "33_insert_rare_emoji_blocks":               "Rare Emoji",
    "35_polarity_flip":                          "Polarity Flip†",
    "66_set_numbers_to_X":                       "Mask Numbers",
    "67_word_replace":                           "Word Replace",
    "68_number_replace":                         "Number Replace",
    "70_word_replace_polarity_mask":             "Word+Polarity Mask",
    "71_word_replace_pct_polarity_mask":         "Word+Pol Mask (%)",
    "81_story_swap":                             "Story Swap",
    "82_polarity_flip":                          "Q-Dir Flip",
    "83_story_swap_polarity_flip":               "Story+Q-Dir Flip",
    "84_symbolic_partial":                       "Symbolic (partial)",
    "85_symbolic_full":                          "Symbolic (full)",
}

INTERV_SPLITS = {
    "35_polarity_flip": ["anticommonsense"],
}
DEFAULT_SPLITS = ["easy", "hard", "anticommonsense"]

QTYPE_ORDER = ["marginal", "correlation", "backadj", "ate", "ett",
               "nie", "nde", "det-counterfactual", "exp_away", "collider_bias"]
QTYPE_LABELS = {
    "marginal": "MAR", "correlation": "COR", "backadj": "BKA",
    "ate": "ATE", "ett": "ETT", "nie": "NIE", "nde": "NDE",
    "det-counterfactual": "DCF", "exp_away": "EXP", "collider_bias": "COL",
}

MODEL_OFFSETS = [-0.22, 0.0, 0.22]


# ── Data loading ──────────────────────────────────────────────────────────────

def weighted_acc(results: list):
    if not results:
        return None
    total_n = sum(n for _, n in results)
    return sum(a * n for a, n in results) / total_n if total_n else None


def load_summary(path: Path):
    """Returns (acc_all, n) or None."""
    if not path.exists():
        return None
    import json
    d = json.load(open(path))
    return d["acc_all"], d["n"]


def load_overall_baseline(outputs: Path, cond_key: str, split: str):
    _, subdir, run_id, *_ = CONDITIONS[cond_key]
    p = outputs / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "summary.json"
    return load_summary(p)


def load_overall_interv(results_dir: Path, cond_key: str, interv: str, split: str):
    p = results_dir / cond_key / interv / split / "score" / "summary.json"
    return load_summary(p)


def select_top_interventions(outputs: Path, results_dir: Path, n: int = TOP_N,
                             split: str | None = None):
    """Rank interventions valid for `split` by mean drop across conditions, return top-n."""
    drops = {}
    for interv in ALL_INTERVENTIONS:
        valid = INTERV_SPLITS.get(interv, DEFAULT_SPLITS)
        if split and split not in valid:
            continue
        splits = [split] if split else valid
        model_drops = []
        for cond in CONDITIONS:
            bl = weighted_acc([r for s in splits if (r := load_overall_baseline(outputs, cond, s))])
            iv = weighted_acc([r for s in splits if (r := load_overall_interv(results_dir, cond, interv, s))])
            if bl is not None and iv is not None:
                model_drops.append(bl - iv)
        if model_drops:
            drops[interv] = np.mean(model_drops)
    # most negative change = worst interventions, rank ascending
    ranked = sorted(drops, key=drops.get)
    return OrderedDict((k, ALL_INTERVENTIONS[k]) for k in ranked[:n])


def load_qtype_csv(path: Path) -> dict:
    """Returns {query_type: (acc, n)}."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {row["query_type"]: (row["acc_all"], row["n"]) for _, row in df.iterrows()}


def get_baseline_qtype(outputs: Path, cond_key: str, splits: list) -> dict:
    _, subdir, run_id, *_ = CONDITIONS[cond_key]
    pooled = defaultdict(lambda: [0, 0])  # qt -> [correct, n]
    for split in splits:
        p = outputs / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "per_query_type.csv"
        for qt, (acc, n) in load_qtype_csv(p).items():
            pooled[qt][0] += round(acc * n)
            pooled[qt][1] += n
    return {qt: v[0] / v[1] for qt, v in pooled.items() if v[1] > 0}


def get_interv_qtype(results_dir: Path, cond_key: str, interv: str, splits: list) -> dict:
    pooled = defaultdict(lambda: [0, 0])
    for split in splits:
        p = results_dir / cond_key / interv / split / "score" / "per_query_type.csv"
        for qt, (acc, n) in load_qtype_csv(p).items():
            pooled[qt][0] += round(acc * n)
            pooled[qt][1] += n
    return {qt: v[0] / v[1] for qt, v in pooled.items() if v[1] > 0}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_qtype_strip(outputs: Path, results_dir: Path, split: str | None = None):
    focus = select_top_interventions(outputs, results_dir, TOP_N, split)
    split_tag = f" [{split}]" if split else " [all splits pooled]"
    print(f"Top-{TOP_N} interventions{split_tag}: {list(focus.keys())}")

    n_interv = len(focus)
    qtypes = [qt for qt in QTYPE_ORDER if qt in QTYPE_LABELS]
    qt_labels = [QTYPE_LABELS[qt] for qt in qtypes]
    n_qt = len(qtypes)

    fig, axes = plt.subplots(1, n_interv, figsize=(4.2 * n_interv, 7),
                             sharey=True, sharex=True)

    for ax_i, (interv, interv_label) in enumerate(focus.items()):
        ax = axes[ax_i]
        splits = [split] if split else INTERV_SPLITS.get(interv, DEFAULT_SPLITS)

        for qi, qt in enumerate(qtypes):
            for ci, (cond, (label, _, _, color, marker, ls)) in enumerate(CONDITIONS.items()):
                b_map = get_baseline_qtype(outputs, cond, splits)
                v_map = get_interv_qtype(results_dir, cond, interv, splits)

                b = b_map.get(qt)
                v = v_map.get(qt)
                if b is None or v is None:
                    continue

                change = v - b   # negative = worse, positive = improved
                y = n_qt - 1 - qi + MODEL_OFFSETS[ci]
                ax.scatter(change, y, color=color, marker=marker, s=28, zorder=4, alpha=0.9)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
        ax.set_title(interv_label, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(-0.65, 0.35)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        if ax_i == 0:
            ax.set_yticks(range(n_qt))
            ax.set_yticklabels(reversed(qt_labels), fontsize=9.5)
        ax.set_xlabel("Accuracy change\n(negative = worse)", fontsize=8.5)

        # Separator lines
        for qi in range(n_qt - 1):
            ax.axhline(n_qt - 1 - qi - 0.5, color="#dddddd", linewidth=0.6)

    split_label = f" [{split} split]" if split else " [all splits pooled]"
    fig.suptitle(f"Per-query-type accuracy change under intervention{split_label}\n"
                 "(one dot per model, negative x = accuracy fell)",
                 fontsize=12, fontweight="bold", y=1.01)

    # Legend
    handles = []
    for cond, (label, _, _, color, marker, ls) in CONDITIONS.items():
        h = mlines.Line2D([], [], color=color, marker=marker, markersize=6,
                          linewidth=0, label=label)
        handles.append(h)
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.06),
               title="Model condition", title_fontsize=9)

    plt.tight_layout()
    out_dir = OUT_DIR / split if split else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "qtype_strip.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out}")


def _draw_strip_row(axes, interv_items, outputs, results_dir, split,
                    qtypes, qt_labels, first_col=True):
    """Draw one row of strip panels; returns axes list."""
    n_qt = len(qtypes)
    for ax_i, (interv, interv_label) in enumerate(interv_items):
        ax = axes[ax_i]
        splits = [split] if split else INTERV_SPLITS.get(interv, DEFAULT_SPLITS)

        for qi, qt in enumerate(qtypes):
            for ci, (cond, (label, _, _, color, marker, ls)) in enumerate(CONDITIONS.items()):
                b_map = get_baseline_qtype(outputs, cond, splits)
                v_map = get_interv_qtype(results_dir, cond, interv, splits)
                b = b_map.get(qt)
                v = v_map.get(qt)
                if b is None or v is None:
                    continue
                change = v - b
                y = n_qt - 1 - qi + MODEL_OFFSETS[ci]
                ax.scatter(change, y, color=color, marker=marker, s=28, zorder=4, alpha=0.9)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
        ax.set_title(interv_label, fontsize=9, fontweight="bold", pad=5)
        ax.set_xlim(-0.65, 0.35)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        if ax_i == 0 and first_col:
            ax.set_yticks(range(n_qt))
            ax.set_yticklabels(reversed(qt_labels), fontsize=9)
        else:
            ax.set_yticks([])

        ax.set_xlabel("Acc. change\n(neg = worse)", fontsize=7.5)

        for qi in range(n_qt - 1):
            ax.axhline(n_qt - 1 - qi - 0.5, color="#dddddd", linewidth=0.6)


def plot_qtype_strip_all(outputs: Path, results_dir: Path, split: str | None = None):
    """Show all interventions in a 2-row grid: noise on top, semantic/structural below."""
    # Split ALL_INTERVENTIONS into two groups matching the category bands
    noise_keys      = ["1_add_space_between_nonspace_characters",
                       "7_append_1000_high_density_unicode_chars",
                       "13_remove_every_other_word_except_numbers",
                       "22_insert_wrong_answer_once_per_sentence",
                       "32_insert_common_emoji_blocks",
                       "33_insert_rare_emoji_blocks",
                       "66_set_numbers_to_X",
                       "67_word_replace",
                       "68_number_replace",
                       "70_word_replace_polarity_mask",
                       "71_word_replace_pct_polarity_mask"]
    semantic_keys   = ["35_polarity_flip",
                       "81_story_swap",
                       "82_polarity_flip",
                       "83_story_swap_polarity_flip",
                       "84_symbolic_partial",
                       "85_symbolic_full"]

    if split:
        noise_keys    = [k for k in noise_keys    if split in INTERV_SPLITS.get(k, DEFAULT_SPLITS)]
        semantic_keys = [k for k in semantic_keys if split in INTERV_SPLITS.get(k, DEFAULT_SPLITS)]

    noise_items    = [(k, ALL_INTERVENTIONS[k]) for k in noise_keys]
    semantic_items = [(k, ALL_INTERVENTIONS[k]) for k in semantic_keys]
    n_cols = max(len(noise_items), len(semantic_items))

    qtypes    = [qt for qt in QTYPE_ORDER if qt in QTYPE_LABELS]
    qt_labels = [QTYPE_LABELS[qt] for qt in qtypes]
    n_qt      = len(qtypes)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 13),
                             sharey=True, sharex=True)

    # pad shorter row with invisible axes
    def pad_row(items, n):
        return items + [(None, None)] * (n - len(items))

    for ax_i, (interv, label) in enumerate(pad_row(noise_items, n_cols)):
        ax = axes[0, ax_i]
        if interv is None:
            ax.set_visible(False)
            continue
        _draw_strip_row([ax], [(interv, label)], outputs, results_dir, split,
                        qtypes, qt_labels, first_col=(ax_i == 0))

    for ax_i, (interv, label) in enumerate(pad_row(semantic_items, n_cols)):
        ax = axes[1, ax_i]
        if interv is None:
            ax.set_visible(False)
            continue
        _draw_strip_row([ax], [(interv, label)], outputs, results_dir, split,
                        qtypes, qt_labels, first_col=(ax_i == 0))

    # Row group labels
    for row_i, row_label in enumerate(["Noise / Corruption", "Semantic / Structural"]):
        axes[row_i, 0].annotate(row_label, xy=(-0.35, 0.5),
                                xycoords="axes fraction", ha="right", va="center",
                                fontsize=9, color="#555555", fontweight="bold", rotation=90)

    split_tag = f" [{split} split]" if split else " [all splits pooled]"
    fig.suptitle(f"Per-query-type accuracy change — all interventions{split_tag}\n"
                 "(one dot per model, negative x = accuracy fell)",
                 fontsize=12, fontweight="bold", y=1.01)

    handles = [mlines.Line2D([], [], color=c, marker=m, markersize=6,
                             linewidth=0, label=lbl)
               for _, (lbl, _, _, c, m, _) in CONDITIONS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.03),
               title="Model condition", title_fontsize=9)

    plt.tight_layout()
    out_dir = OUT_DIR / split if split else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "qtype_strip_all.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    outputs     = Path("outputs")
    results_dir = Path("finetune/eval_results/interventions")
    for split in [None, "easy", "hard", "anticommonsense"]:
        plot_qtype_strip(outputs, results_dir, split)
        plot_qtype_strip_all(outputs, results_dir, split)
