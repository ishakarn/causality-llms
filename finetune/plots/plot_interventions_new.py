"""
Intervention robustness plots for Phase 1 battery.

Reads from:
  finetune/eval_results/interventions/<condition>/<intervention>/<split>/score/summary.json
  finetune/eval_results/interventions/<condition>/<intervention>/<split>/score/per_query_type.csv
  outputs/cladder-v1-q-<split>/baseline/<model_run>/score/summary.json   (clean base)
  outputs/cladder-v1-q-<split>/finetuned/<model_run>/score/summary.json  (clean lora)

Produces (outputs/plots/interventions/):
  heatmap_acc.png          — abs accuracy heatmap, rows=model×cond, cols=interventions
  heatmap_drop.png         — accuracy drop vs clean, same layout
  bar_per_model.png        — per-model base vs lora bars per intervention, with CI
  split_combined_{model}.png — base+lora side-by-side per intervention, one panel per split, CI
  qtype_drop_{model}.png   — per-query-type accuracy drop per intervention, base vs lora

Usage:
    python finetune/plots/plot_interventions_new.py [--models qwen3b olmo32b gptoss]
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plot_style import apply_defaults, model_color
from query_type_labels import abbrev, RUNG_ORDER

# ── Constants ─────────────────────────────────────────────────────────────────

INTERVENTIONS = [
    "1_insert_spaces_between_chars_without_whitespace",
    "7_append_1000_high_density_unicode_chars",
    "13_remove_every_other_word",
    "22_insert_incorrect_answer_once_per_sentence",
    "33_insert_rare_emoji_blocks",
    "35_polarity_flip",
    "66_set_numbers_to_X",
]
INTERV_SHORT = {
    "1_insert_spaces_between_chars_without_whitespace": "char_spaces",
    "7_append_1000_high_density_unicode_chars":         "unicode_noise",
    "13_remove_every_other_word":                       "del_alt_words",
    "22_insert_incorrect_answer_once_per_sentence":     "insert_wrong",
    "33_insert_rare_emoji_blocks":                      "emoji_blocks",
    "35_polarity_flip":                                 "polarity_flip†",
    "66_set_numbers_to_X":                              "mask_numbers",
}
INTERV_GROUPS = [
    ("Surface noise",    ["1_insert_spaces_between_chars_without_whitespace",
                          "7_append_1000_high_density_unicode_chars",
                          "33_insert_rare_emoji_blocks"]),
    ("Semantic",         ["13_remove_every_other_word",
                          "66_set_numbers_to_X"]),
    ("Adversarial",      ["22_insert_incorrect_answer_once_per_sentence"]),
    ("Label flip",       ["35_polarity_flip"]),
]

CLEAN_BASELINE_RUNS = {
    "qwen3b":  "qwen25-3b-instruct-baseline",
    "olmo32b": "olmo3-32b-instruct-baseline",
    "gptoss":  "gpt-oss-20b-baseline",
}
CLEAN_LORA_RUNS = {
    "qwen3b":  "qwen25-3b-instruct-lora",
    "olmo32b": "olmo3-32b-instruct-lora",
    "gptoss":  None,
}
MODEL_DISPLAY = {
    "qwen3b":  "Qwen2.5-3B",
    "olmo32b": "OLMo-3.1-32B",
    "gptoss":  "GPT-OSS-20B",
}
SPLITS = ["easy", "hard", "anticommonsense", "noncommonsense"]
SPLIT_DISPLAY = {"easy": "Easy", "hard": "Hard",
                 "anticommonsense": "Anti-CS", "noncommonsense": "Non-CS"}
QUERY_TYPES = [qt for qt in RUNG_ORDER]   # canonical order from query_type_labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def wilson_ci(correct, n, z=1.96):
    if n == 0:
        nan = float("nan")
        return nan, nan, nan
    p = correct / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def load_summary(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def load_qtype_csv(path) -> dict:
    """Returns {query_type: {'n': int, 'correct': int, 'acc_all': float}}"""
    p = Path(path)
    if not p.exists():
        return {}
    result = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            qt = row["query_type"]
            n  = int(row["n"])
            result[qt] = {
                "n":       n,
                "correct": round(float(row["acc_all"]) * n),  # back-compute
                "acc_all": float(row["acc_all"]),
            }
    return result


def acc(summary: dict) -> float:
    return summary.get("acc_all", float("nan"))


def mean_nonan(vals):
    v = [x for x in vals if not math.isnan(x)]
    return sum(v) / len(v) if v else float("nan")


def pool_correct_n(summaries: list):
    """Pool n and correct across multiple summary dicts."""
    total_n = total_c = 0
    for s in summaries:
        if s:
            total_n += s.get("n", 0)
            total_c += s.get("correct", 0)
    return total_c, total_n


def draw_group_separators(ax, groups, x_positions, orientation="v"):
    pos = 0
    for _, items in groups:
        if pos > 0:
            xv = x_positions[pos] - (x_positions[1] - x_positions[0]) * 0.5
            if orientation == "v":
                ax.axvline(xv, color="lightgray", lw=0.8, zorder=0)
            else:
                ax.axhline(xv, color="lightgray", lw=0.8, zorder=0)
        pos += len(items)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_intervention_results(base_dir: Path, models: list) -> dict:
    """
    data[model][condition][iname][split] = summary dict
    condition ∈ {'base', 'lora'}
    """
    data = {}
    for model in models:
        data[model] = {"base": {}, "lora": {}}
        for iname in INTERVENTIONS:
            data[model]["base"][iname] = {}
            data[model]["lora"][iname] = {}
            for split in SPLITS:
                bp = base_dir / f"{model}_base" / iname / split / "score" / "summary.json"
                lp = base_dir / f"{model}_n2000_lora" / iname / split / "score" / "summary.json"
                data[model]["base"][iname][split] = load_summary(bp)
                data[model]["lora"][iname][split]  = load_summary(lp)
    return data


def load_qtype_results(base_dir: Path, models: list) -> dict:
    """
    qtype_data[model][condition][iname][split][query_type] = {n, correct, acc_all}
    """
    qtype_data = {}
    for model in models:
        qtype_data[model] = {"base": {}, "lora": {}}
        for iname in INTERVENTIONS:
            qtype_data[model]["base"][iname] = {}
            qtype_data[model]["lora"][iname] = {}
            for split in SPLITS:
                bp = base_dir / f"{model}_base" / iname / split / "score" / "per_query_type.csv"
                lp = base_dir / f"{model}_n2000_lora" / iname / split / "score" / "per_query_type.csv"
                qtype_data[model]["base"][iname][split] = load_qtype_csv(bp)
                qtype_data[model]["lora"][iname][split]  = load_qtype_csv(lp)
    return qtype_data


def load_clean_baselines(outputs_dir: Path, models: list) -> dict:
    """clean[model][condition][split] = acc float"""
    clean = {}
    for model in models:
        clean[model] = {"base": {}, "lora": {}}
        for split in SPLITS:
            run = CLEAN_BASELINE_RUNS.get(model)
            p   = outputs_dir / f"cladder-v1-q-{split}" / "baseline" / run / "score" / "summary.json" if run else None
            clean[model]["base"][split] = acc(load_summary(p)) if p else float("nan")

            run = CLEAN_LORA_RUNS.get(model)
            p   = outputs_dir / f"cladder-v1-q-{split}" / "finetuned" / run / "score" / "summary.json" if run else None
            clean[model]["lora"][split] = acc(load_summary(p)) if p else float("nan")
    return clean


def load_clean_qtype(outputs_dir: Path, models: list) -> dict:
    """clean_qt[model][condition][split][query_type] = {n, correct, acc_all}"""
    clean_qt = {}
    for model in models:
        clean_qt[model] = {"base": {}, "lora": {}}
        for split in SPLITS:
            run = CLEAN_BASELINE_RUNS.get(model)
            p   = outputs_dir / f"cladder-v1-q-{split}" / "baseline" / run / "score" / "per_query_type.csv" if run else None
            clean_qt[model]["base"][split] = load_qtype_csv(p) if p else {}

            run = CLEAN_LORA_RUNS.get(model)
            p   = outputs_dir / f"cladder-v1-q-{split}" / "finetuned" / run / "score" / "per_query_type.csv" if run else None
            clean_qt[model]["lora"][split] = load_qtype_csv(p) if p else {}
    return clean_qt


# ── Plot 1: Heatmaps ──────────────────────────────────────────────────────────

def plot_heatmaps(data, clean, models, out_dir):
    apply_defaults()
    rows = []
    for model in models:
        rows.append((model, f"{MODEL_DISPLAY[model]}\nbase", "base"))
        if CLEAN_LORA_RUNS.get(model):
            rows.append((model, f"{MODEL_DISPLAY[model]}\nlora", "lora"))

    n_rows, n_cols = len(rows), len(INTERVENTIONS)
    acc_mat  = np.full((n_rows, n_cols), float("nan"))
    drop_mat = np.full((n_rows, n_cols), float("nan"))

    for r, (model, _, cond) in enumerate(rows):
        for c, iname in enumerate(INTERVENTIONS):
            c_n, c_c = pool_correct_n([data[model][cond][iname][sp] for sp in SPLITS])
            a = c_c / c_n if c_n else float("nan")
            acc_mat[r, c] = a
            clean_vals = [clean[model][cond][sp] for sp in SPLITS if not math.isnan(clean[model][cond][sp])]
            clean_avg  = mean_nonan(clean_vals)
            drop_mat[r, c] = clean_avg - a

    col_labels = [INTERV_SHORT[i] for i in INTERVENTIONS]
    row_labels  = [r[1] for r in rows]

    for title, mat, cmap, vmin, vmax, fname in [
        ("Intervention accuracy (pooled across splits)",
         acc_mat,  "RdYlGn", 0.3, 0.95, "heatmap_acc.png"),
        ("Accuracy drop vs clean baseline  (positive = worse)",
         drop_mat, "RdYlGn_r", -0.05, 0.4, "heatmap_drop.png"),
    ]:
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.4), max(4, n_rows * 0.85)))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n_cols)); ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(n_rows)); ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        for r in range(n_rows):
            for c in range(n_cols):
                v = mat[r, c]
                if not math.isnan(v):
                    txt = f"{v:+.2f}" if "drop" in fname else f"{v:.2f}"
                    brightness = (v - vmin) / (vmax - vmin)
                    fg = "white" if (brightness < 0.3 or brightness > 0.75) else "black"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=7.5, color=fg)
        pos = 0
        for _, items in INTERV_GROUPS:
            if pos > 0:
                ax.axvline(pos - 0.5, color="white", lw=1.5)
            pos += len(items)
        ax.text(0.5, -0.1, "† polarity_flip: gold answer already flipped in data file",
                transform=ax.transAxes, ha="center", fontsize=7.5, color="gray")
        plt.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"Wrote: {p}")
        plt.close(fig)


# ── Plot 2: Bar chart per model, base vs lora, CI ─────────────────────────────

def plot_bars_per_model(data, clean, models, out_dir):
    apply_defaults()
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1,
                             figsize=(max(12, len(INTERVENTIONS) * 1.6), 4.5 * n_models),
                             sharex=False)
    if n_models == 1:
        axes = [axes]

    x = np.arange(len(INTERVENTIONS))
    width = 0.35

    for ax, model in zip(axes, models):
        color = model_color(model)
        has_lora = CLEAN_LORA_RUNS.get(model) is not None

        def bars_for_cond(cond):
            accs, ci_lo, ci_hi = [], [], []
            for iname in INTERVENTIONS:
                c, n = pool_correct_n([data[model][cond][iname][sp] for sp in SPLITS])
                a, lo, hi = wilson_ci(c, n)
                accs.append(a); ci_lo.append(lo); ci_hi.append(hi)
            return np.array(accs, dtype=float), np.array(ci_lo, dtype=float), np.array(ci_hi, dtype=float)

        b_acc, b_lo, b_hi = bars_for_cond("base")
        l_acc, l_lo, l_hi = bars_for_cond("lora") if has_lora else (None, None, None)

        def err(a, lo, hi):
            return [a - lo, hi - a]

        offset = width / 2 if has_lora else 0
        ax.bar(x - offset, b_acc, width, color=color, alpha=0.75,
               edgecolor="black", linewidth=0.6, label="Base (zero-shot)",
               yerr=err(b_acc, b_lo, b_hi),
               error_kw=dict(ecolor="black", capsize=3, lw=1))
        if has_lora:
            ax.bar(x + offset, l_acc, width, color=color, alpha=0.75,
                   edgecolor="black", linewidth=0.6, hatch="//",
                   label="Fine-tuned N=2000",
                   yerr=err(l_acc, l_lo, l_hi),
                   error_kw=dict(ecolor="black", capsize=3, lw=1))

        # Clean baseline reference lines
        cb = mean_nonan([clean[model]["base"][sp] for sp in SPLITS])
        cl = mean_nonan([clean[model]["lora"][sp]  for sp in SPLITS]) if has_lora else float("nan")
        if not math.isnan(cb):
            ax.axhline(cb, color=color, ls="--", lw=1.2, alpha=0.6, label=f"Base clean ({cb:.2f})")
        if not math.isnan(cl):
            ax.axhline(cl, color=color, ls="-.", lw=1.2, alpha=0.6, label=f"Lora clean ({cl:.2f})")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([INTERV_SHORT[i] for i in INTERVENTIONS], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.2, 1.08)
        ax.set_yticks(np.arange(0.2, 1.05, 0.1))
        ax.set_title(f"{MODEL_DISPLAY[model]}  —  pooled across 4 splits, Wilson 95% CI", fontsize=9)
        ax.legend(fontsize=7.5, ncol=2, loc="lower right")
        draw_group_separators(ax, INTERV_GROUPS, x)

    fig.suptitle("Intervention Robustness — Base vs Fine-tuned", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = out_dir / "bar_per_model.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"Wrote: {p}")
    plt.close(fig)


# ── Plot 3: Split combined — base + lora together per model ───────────────────

def plot_split_combined(data, clean, models, out_dir):
    """
    One figure per model. 4 panels (one per split).
    Each panel: 2 bars per intervention (base solid, lora hatched), Wilson 95% CI.
    Dashed reference lines for clean base and lora.
    """
    apply_defaults()

    for model in models:
        color    = model_color(model)
        has_lora = CLEAN_LORA_RUNS.get(model) is not None
        n_splits = len(SPLITS)
        fig, axes = plt.subplots(1, n_splits, figsize=(max(14, len(INTERVENTIONS) * 1.5), 5),
                                 sharey=True)

        x     = np.arange(len(INTERVENTIONS))
        width = 0.35

        for ax, split in zip(axes, SPLITS):
            b_accs, b_los, b_his = [], [], []
            l_accs, l_los, l_his = [], [], []

            for iname in INTERVENTIONS:
                bs = data[model]["base"][iname][split]
                ls = data[model]["lora"][iname][split]
                bc, bn = bs.get("correct", 0), bs.get("n", 0)
                lc, ln = ls.get("correct", 0), ls.get("n", 0)
                a, lo, hi = wilson_ci(bc, bn)
                b_accs.append(a); b_los.append(lo); b_his.append(hi)
                a, lo, hi = wilson_ci(lc, ln)
                l_accs.append(a); l_los.append(lo); l_his.append(hi)

            b_accs = np.array(b_accs, dtype=float)
            l_accs = np.array(l_accs, dtype=float)

            def err(a, lo, hi):
                return [a - np.array(lo, dtype=float), np.array(hi, dtype=float) - a]

            offset = width / 2 if has_lora else 0
            ax.bar(x - offset, b_accs, width, color=color, alpha=0.75,
                   edgecolor="black", linewidth=0.5, label="Base",
                   yerr=err(b_accs, b_los, b_his),
                   error_kw=dict(ecolor="black", capsize=2, lw=0.8))
            if has_lora:
                ax.bar(x + offset, l_accs, width, color=color, alpha=0.75,
                       edgecolor="black", linewidth=0.5, hatch="//",
                       label="Lora N=2000",
                       yerr=err(l_accs, l_los, l_his),
                       error_kw=dict(ecolor="black", capsize=2, lw=0.8))

            cb = clean[model]["base"][split]
            cl = clean[model]["lora"][split] if has_lora else float("nan")
            if not math.isnan(cb):
                ax.axhline(cb, color=color, ls="--", lw=1.0, alpha=0.6, label=f"Base clean ({cb:.2f})")
            if not math.isnan(cl):
                ax.axhline(cl, color=color, ls="-.", lw=1.0, alpha=0.6, label=f"Lora clean ({cl:.2f})")
            ax.axhline(0.5, color="gray", ls=":", lw=0.7, alpha=0.4)

            ax.set_xticks(x)
            ax.set_xticklabels([INTERV_SHORT[i] for i in INTERVENTIONS],
                               rotation=30, ha="right", fontsize=6.5)
            ax.set_title(SPLIT_DISPLAY[split], fontsize=9, fontweight="bold")
            ax.legend(fontsize=6, loc="lower right")
            draw_group_separators(ax, INTERV_GROUPS, x)

        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim(0.2, 1.08)
        axes[0].set_yticks(np.arange(0.2, 1.05, 0.1))
        fig.suptitle(f"{MODEL_DISPLAY[model]}  —  Base vs Fine-tuned per split, Wilson 95% CI",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        p = out_dir / f"split_combined_{model}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"Wrote: {p}")
        plt.close(fig)


# ── Plot 4: Per-query-type accuracy drop ──────────────────────────────────────

def plot_qtype_drop(data_qt, clean_qt, models, out_dir):
    """
    One figure per model. 2 panels: base (top) and lora (bottom).
    X = query types. Grouped bars: one bar per intervention per query type.
    Y = accuracy drop vs clean (positive = worse). Wilson 95% CI error bars.
    """
    apply_defaults()
    n_interv = len(INTERVENTIONS)
    INTERV_COLORS = plt.cm.tab10(np.linspace(0, 1, n_interv))

    for model in models:
        has_lora = CLEAN_LORA_RUNS.get(model) is not None
        n_panels = 2 if has_lora else 1
        fig, axes = plt.subplots(n_panels, 1,
                                 figsize=(max(14, len(QUERY_TYPES) * 1.5), 5.5 * n_panels),
                                 sharex=True)
        if n_panels == 1:
            axes = [axes]

        qt_list = QUERY_TYPES
        n_qt    = len(qt_list)
        group_w = 0.8                        # total width of one query-type group
        bar_w   = group_w / n_interv
        x       = np.arange(n_qt)

        for ax_idx, cond in enumerate(["base", "lora"][:n_panels]):
            ax = axes[ax_idx]

            # Clean accuracy per query type (pooled across splits)
            clean_acc_qt = {}
            for qt in qt_list:
                total_c, total_n = 0, 0
                for split in SPLITS:
                    row = clean_qt[model][cond].get(split, {}).get(qt, {})
                    total_n += row.get("n", 0)
                    total_c += row.get("correct", 0)
                a, _, _ = wilson_ci(total_c, total_n)
                clean_acc_qt[qt] = a

            all_drops = []
            for i_idx, iname in enumerate(INTERVENTIONS):
                drops, err_lo, err_hi = [], [], []
                for qt in qt_list:
                    total_c, total_n = 0, 0
                    for split in SPLITS:
                        row = data_qt[model][cond][iname].get(split, {}).get(qt, {})
                        total_n += row.get("n", 0)
                        total_c += row.get("correct", 0)
                    a, lo, hi = wilson_ci(total_c, total_n)
                    drop = clean_acc_qt.get(qt, float("nan")) - a
                    drops.append(drop)
                    # CI on drop: propagate both clean and intervention CI
                    # (conservative: sum of half-widths)
                    ci_half = (hi - lo) / 2 if not math.isnan(a) else float("nan")
                    err_lo.append(ci_half); err_hi.append(ci_half)

                drops  = np.array(drops,  dtype=float)
                err_lo = np.array(err_lo, dtype=float)
                err_hi = np.array(err_hi, dtype=float)
                all_drops.extend(drops[~np.isnan(drops)].tolist())

                valid = ~np.isnan(drops)
                if not valid.any():
                    continue

                offset = (i_idx - (n_interv - 1) / 2) * bar_w
                color  = INTERV_COLORS[i_idx]
                ax.bar(x[valid] + offset, drops[valid], bar_w,
                       color=color, alpha=0.8, edgecolor="black", linewidth=0.4,
                       label=INTERV_SHORT[iname],
                       yerr=[err_lo[valid], err_hi[valid]],
                       error_kw=dict(ecolor="black", capsize=2, lw=0.7))

            ax.axhline(0, color="black", lw=0.9)
            # Separator between each query type group
            for xi in x[1:]:
                ax.axvline(xi - 0.5, color="lightgray", lw=0.8, zorder=0)
            ax.set_ylabel("Accuracy drop\n(clean − intervened)")
            cond_label = "Base (zero-shot)" if cond == "base" else "Fine-tuned N=2000"
            ax.set_title(cond_label, fontsize=9)
            ax.legend(fontsize=6.5, ncol=4, loc="upper right",
                      title="Intervention", title_fontsize=7)
            ymax = max(0.4, max((abs(v) for v in all_drops), default=0.4) + 0.08)
            ax.set_ylim(-0.2, ymax)
            ax.axhline(0.1, color="gray", ls=":", lw=0.7, alpha=0.5)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([abbrev(qt) for qt in qt_list], fontsize=8)
        axes[-1].set_xlabel("Query type")
        fig.suptitle(
            f"{MODEL_DISPLAY[model]}  —  Accuracy drop per query type  (Wilson 95% CI, pooled across 4 splits)\n"
            "Positive = worse than clean baseline",
            fontsize=11, fontweight="bold")
        plt.tight_layout()
        p = out_dir / f"qtype_drop_{model}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"Wrote: {p}")
        plt.close(fig)


# ── Plot 5: Per-intervention, all models, per query type ─────────────────────

# Conditions in display order for cross-model plots
ALL_CONDITIONS = [
    ("qwen3b",  "base", "Qwen2.5-3B base"),
    ("qwen3b",  "lora", "Qwen2.5-3B lora"),
    ("olmo32b", "base", "OLMo-3.1-32B base"),
    ("olmo32b", "lora", "OLMo-3.1-32B lora"),
    ("gptoss",  "base", "GPT-OSS-20B base"),
]
COND_COLORS = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452", "#55A868"]
COND_HATCHES = ["", "//", "", "//", ""]


def load_single_intervention_qtype(results_dir: Path, iname: str,
                                   models: list, splits: list) -> dict:
    """qtype_data[model][cond][split][query_type] = {n, correct, acc_all}"""
    qtype_data = {}
    for model in models:
        qtype_data[model] = {"base": {}, "lora": {}}
        for split in splits:
            bp = results_dir / f"{model}_base"       / iname / split / "score" / "per_query_type.csv"
            lp = results_dir / f"{model}_n2000_lora" / iname / split / "score" / "per_query_type.csv"
            qtype_data[model]["base"][split] = load_qtype_csv(bp)
            qtype_data[model]["lora"][split] = load_qtype_csv(lp)
    return qtype_data


def plot_all_models_per_intervention(results_dir: Path, outputs_dir: Path,
                                     iname: str, splits: list,
                                     models: list, out_dir: Path,
                                     short_name: str = None):
    """
    One figure: X=query_types, grouped bars=all model conditions.
    Two panels: top=accuracy, bottom=accuracy drop vs clean.
    Splits are pooled (only those in `splits` list).
    """
    apply_defaults()
    short_name = short_name or iname

    data_qt  = load_single_intervention_qtype(results_dir, iname, models, splits)
    clean_qt = load_clean_qtype(outputs_dir, models)

    qt_list = QUERY_TYPES
    n_qt    = len(qt_list)
    n_cond  = len(ALL_CONDITIONS)
    group_w = 0.8
    bar_w   = group_w / n_cond
    x       = np.arange(n_qt)

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_qt * 1.6), 10), sharex=True)

    for panel, use_drop in enumerate([False, True]):
        ax = axes[panel]

        for ci, (model, cond, label) in enumerate(ALL_CONDITIONS):
            if cond == "lora" and CLEAN_LORA_RUNS.get(model) is None:
                continue  # gptoss has no lora

            # Pool across splits for intervention
            accs, ci_los, ci_his = [], [], []
            for qt in qt_list:
                total_c, total_n = 0, 0
                for split in splits:
                    row = data_qt[model][cond].get(split, {}).get(qt, {})
                    total_c += row.get("correct", 0)
                    total_n += row.get("n", 0)
                a, lo, hi = wilson_ci(total_c, total_n)
                accs.append(a); ci_los.append(lo); ci_his.append(hi)

            accs   = np.array(accs,   dtype=float)
            ci_los = np.array(ci_los, dtype=float)
            ci_his = np.array(ci_his, dtype=float)

            if use_drop:
                # Pool clean across same splits
                clean_accs = []
                for qt in qt_list:
                    total_c, total_n = 0, 0
                    for split in splits:
                        row = clean_qt[model][cond].get(split, {}).get(qt, {})
                        total_c += row.get("correct", 0)
                        total_n += row.get("n", 0)
                    ca, _, _ = wilson_ci(total_c, total_n)
                    clean_accs.append(ca)
                clean_accs = np.array(clean_accs, dtype=float)
                vals = clean_accs - accs  # positive = worse
                err  = (ci_his - ci_los) / 2
            else:
                vals = accs
                err  = np.array([(accs - ci_los), (ci_his - accs)])

            valid  = ~np.isnan(vals)
            offset = (ci - (n_cond - 1) / 2) * bar_w

            if use_drop:
                yerr_arg = np.clip(err[valid], 0, None)
            else:
                yerr_arg = [np.clip(err[0][valid], 0, None),
                            np.clip(err[1][valid], 0, None)]

            ax.bar(x[valid] + offset, vals[valid], bar_w,
                   color=COND_COLORS[ci], alpha=0.82,
                   hatch=COND_HATCHES[ci],
                   edgecolor="black", linewidth=0.4,
                   label=label,
                   yerr=yerr_arg,
                   error_kw=dict(ecolor="black", capsize=2, lw=0.7))

        ax.axhline(0 if use_drop else 0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
        for xi in x[1:]:
            ax.axvline(xi - 0.5, color="lightgray", lw=0.8, zorder=0)

        if use_drop:
            ax.set_ylabel("Accuracy drop\n(clean − intervened)")
            ax.set_title("Accuracy drop vs clean baseline  (positive = worse)", fontsize=9)
        else:
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy under intervention  (Wilson 95% CI)", fontsize=9)
            ax.set_ylim(0.2, 1.05)
            ax.set_yticks(np.arange(0.2, 1.05, 0.1))

        ax.legend(fontsize=7.5, ncol=n_cond, loc="upper right" if use_drop else "lower right")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([abbrev(qt) for qt in qt_list], fontsize=8)
    axes[-1].set_xlabel("Query type")

    split_label = "+".join(splits)
    fig.suptitle(
        f"Intervention: {short_name}  —  All models per query type"
        f"  (pooled across {split_label}, Wilson 95% CI)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    safe = iname.replace("/", "_")
    p = out_dir / f"allmodels_qtype_{safe}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Wrote: {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",      nargs="+", default=["qwen3b", "olmo32b", "gptoss"])
    ap.add_argument("--results_dir", default="finetune/eval_results/interventions")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--out_dir",     default="outputs/plots/interventions")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    outputs_dir = Path(args.outputs_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir} …")
    data    = load_intervention_results(results_dir, args.models)
    data_qt = load_qtype_results(results_dir, args.models)

    print(f"Loading clean baselines from {outputs_dir} …")
    clean    = load_clean_baselines(outputs_dir, args.models)
    clean_qt = load_clean_qtype(outputs_dir, args.models)

    print("Heatmaps …")
    plot_heatmaps(data, clean, args.models, out_dir)

    print("Bar charts per model …")
    plot_bars_per_model(data, clean, args.models, out_dir)

    print("Split combined plots …")
    plot_split_combined(data, clean, args.models, out_dir)

    print("Query-type drop plots (per model) …")
    plot_qtype_drop(data_qt, clean_qt, args.models, out_dir)

    print("All-models per-query-type plots (per intervention) …")
    # Original 7 interventions — pool all 4 splits
    for iname in INTERVENTIONS:
        plot_all_models_per_intervention(
            results_dir, outputs_dir, iname,
            splits=SPLITS, models=args.models,
            out_dir=out_dir,
            short_name=INTERV_SHORT.get(iname, iname),
        )
    # story_swap — easy + hard only
    plot_all_models_per_intervention(
        results_dir, outputs_dir, "81_story_swap",
        splits=["easy", "hard"], models=args.models,
        out_dir=out_dir, short_name="story_swap",
    )
    # 82 + 83 — easy + hard only
    plot_all_models_per_intervention(
        results_dir, outputs_dir, "82_polarity_flip",
        splits=["easy", "hard"], models=args.models,
        out_dir=out_dir, short_name="polarity_flip_v2",
    )
    plot_all_models_per_intervention(
        results_dir, outputs_dir, "83_story_swap_polarity_flip",
        splits=["easy", "hard"], models=args.models,
        out_dir=out_dir, short_name="story_swap_polarity_flip",
    )
    # 84 + 85 — easy + hard only
    plot_all_models_per_intervention(
        results_dir, outputs_dir, "84_symbolic_partial",
        splits=["easy", "hard"], models=args.models,
        out_dir=out_dir, short_name="symbolic_partial",
    )
    plot_all_models_per_intervention(
        results_dir, outputs_dir, "85_symbolic_full",
        splits=["easy", "hard"], models=args.models,
        out_dir=out_dir, short_name="symbolic_full",
    )

    print(f"\nDone. All plots → {out_dir}/")


if __name__ == "__main__":
    main()
