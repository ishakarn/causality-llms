"""
Five overview plots comparing all models across all datasets.

Plots generated:
  1. cross_model_heatmap.png    — models × datasets, overall accuracy as color
  2. lora_gain.png              — accuracy gain (LoRA − base) per model
  3. query_type_heatmap.png     — models × query types, accuracy averaged across datasets
  4. param_efficiency.png       — model size vs accuracy (base and LoRA)
  5. dataset_difficulty.png     — per query type: 4 dataset bars averaged across LoRA models

Usage:
    python finetune/plot_overview.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plot_style import ALPHA_BAR, HATCH_FINETUNED, LINEWIDTH_BAR, apply_defaults, model_color
from query_type_labels import abbrev, sort_key

apply_defaults()

OUTPUTS = Path("outputs")
OUT_DIR = OUTPUTS / "plots" / "overview"

DATASETS = [
    "cladder-v1-q-easy",
    "cladder-v1-q-hard",
    "cladder-v1-q-noncommonsense",
    "cladder-v1-q-anticommonsense",
]
DS_LABELS = {
    "cladder-v1-q-easy":            "Easy",
    "cladder-v1-q-hard":            "Hard",
    "cladder-v1-q-noncommonsense":  "Non-CS",
    "cladder-v1-q-anticommonsense": "Anti-CS",
}
DATASET_COLORS = {
    "cladder-v1-q-easy":            "#0072b2",
    "cladder-v1-q-hard":            "#e69f00",
    "cladder-v1-q-noncommonsense":  "#009e73",
    "cladder-v1-q-anticommonsense": "#d55e00",
}


def _jsonl(ds: str, subdir: str, run_id: str, suffix: str = "") -> Path:
    return OUTPUTS / ds / subdir / run_id / f"{run_id}{suffix}.jsonl"


# All model series in display order
MODELS = [
    {
        "label":   "GPT-5-Nano",
        "key":     "gpt-5-nano",
        "size_b":  None,
        "is_lora": False,
        "base_key": None,
        "resolve": lambda ds: _jsonl(ds, "baseline", "gpt-5-nano-baseline"),
    },
    {
        "label":   "GPT-OSS-20B",
        "key":     "gpt-oss-20b",
        "size_b":  20,
        "is_lora": False,
        "base_key": None,
        "resolve": lambda ds: _jsonl(ds, "baseline", "gpt-oss-20b-baseline"),
    },
    {
        "label":   "OLMo-7B (base)",
        "key":     "olmo3-7b-base",
        "size_b":  7,
        "is_lora": False,
        "base_key": None,
        "resolve": lambda ds: _jsonl(ds, "baseline", "olmo3-7b-instruct-baseline"),
    },
    {
        "label":   "OLMo-7B (LoRA)",
        "key":     "olmo3-7b-lora",
        "size_b":  7,
        "is_lora": True,
        "base_key": "olmo3-7b-base",
        "resolve": lambda ds: _jsonl(ds, "finetuned", "olmo3-7b-instruct-lora",
                                     f"__{ds}"),
    },
    {
        "label":   "Qwen-3B (base)",
        "key":     "qwen25-3b-base",
        "size_b":  3,
        "is_lora": False,
        "base_key": None,
        "resolve": lambda ds: _jsonl(ds, "baseline", "qwen25-3b-instruct-baseline"),
    },
    {
        "label":   "Qwen-3B (LoRA)",
        "key":     "qwen25-3b-lora",
        "size_b":  3,
        "is_lora": True,
        "base_key": "qwen25-3b-base",
        "resolve": lambda ds: _jsonl(ds, "finetuned", "qwen25-3b-instruct-lora",
                                     f"__{ds}"),
    },
    {
        "label":   "Llama-8B (base)",
        "key":     "llama31-8b-base",
        "size_b":  8,
        "is_lora": False,
        "base_key": None,
        "resolve": lambda ds: _jsonl(ds, "baseline", "llama31-8b-instruct-baseline"),
    },
    {
        "label":   "Llama-8B (LoRA)",
        "key":     "llama31-8b-lora",
        "size_b":  8,
        "is_lora": True,
        "base_key": "llama31-8b-base",
        "resolve": lambda ds: _jsonl(ds, "finetuned", "llama31-8b-instruct-lora",
                                     f"__{ds}"),
    },
    {
        "label":   "OLMo-32B (base)",
        "key":     "olmo3-32b-base",
        "size_b":  32,
        "is_lora": False,
        "base_key": None,
        "resolve": lambda ds: _jsonl(ds, "baseline", "olmo3-32b-instruct-baseline"),
    },
    {
        "label":   "OLMo-32B (LoRA)",
        "key":     "olmo3-32b-lora",
        "size_b":  32,
        "is_lora": True,
        "base_key": "olmo3-32b-base",
        "resolve": lambda ds: _jsonl(ds, "finetuned", "olmo3-32b-instruct-lora",
                                     f"__{ds}"),
    },
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def compute_stats(rows: List[Dict]) -> Dict:
    """Returns overall acc + per query_type acc."""
    by_type: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "k": 0})
    total_n = total_k = 0
    for r in rows:
        qt = r.get("query_type", "UNKNOWN")
        by_type[qt]["n"] += 1
        total_n += 1
        if r.get("pred") == r.get("gold"):
            by_type[qt]["k"] += 1
            total_k += 1
    per_type = {qt: v["k"] / v["n"] for qt, v in by_type.items() if v["n"]}
    overall = total_k / total_n if total_n else 0.0
    return {"overall": overall, "per_type": per_type, "n": total_n}


def load_all() -> Dict[str, Dict[str, Dict]]:
    """Returns data[model_key][dataset] = stats dict."""
    data = {}
    for m in MODELS:
        data[m["key"]] = {}
        for ds in DATASETS:
            path = m["resolve"](ds)
            if not path.exists():
                continue
            rows = load_jsonl(path)
            data[m["key"]][ds] = compute_stats(rows)
    return data


# ── Plot 1: Cross-model heatmap ───────────────────────────────────────────────

def _draw_group_separator(ax, sep_row: int, n_rows_above: int, n_rows_below: int,
                          label_above: str, label_below: str) -> None:
    """Draw a thick separator between two groups of rows and label each group on the left.

    Labels are placed just outside the left spine using a blended transform
    (x in axes-fraction, y in data coords).  x=-0.32 sits comfortably to the
    left of the y-tick labels without ballooning the figure width.
    """
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    ax.axhline(sep_row + 0.5, color="white", linewidth=3.5, zorder=4)

    mid_above = (n_rows_above - 1) / 2
    mid_below = n_rows_above + (n_rows_below - 1) / 2

    for y, label in [(mid_above, label_above), (mid_below, label_below)]:
        ax.text(-0.32, y, label,
                ha="right", va="center", fontsize=8.5, fontstyle="italic",
                rotation=90, transform=trans, clip_on=False)


def plot_cross_model_heatmap(
    data: Dict,
    out_dir: Path,
    group_by_exposure: bool = False,
    out_name: str = "cross_model_heatmap.png",
) -> None:
    """Models × datasets overall-accuracy heatmap.

    group_by_exposure=True reorders rows so zero-shot models (CLADDER unseen)
    appear first, followed by LoRA fine-tuned models, separated by a labelled
    divider line.
    """
    if group_by_exposure:
        unseen     = [m for m in MODELS if not m["is_lora"]]
        seen       = [m for m in MODELS if m["is_lora"]]
        row_models = unseen + seen
        sep_after  = len(unseen) - 1
    else:
        row_models = MODELS
        sep_after  = None

    model_labels = [m["label"] for m in row_models]
    ds_labels    = [DS_LABELS[d] for d in DATASETS]
    n_rows       = len(row_models)

    matrix = np.full((n_rows, len(DATASETS)), np.nan)
    for i, m in enumerate(row_models):
        for j, ds in enumerate(DATASETS):
            if ds in data[m["key"]]:
                matrix[i, j] = data[m["key"]][ds]["overall"]

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(ds_labels, fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(model_labels, fontsize=10)

    for i in range(n_rows):
        for j in range(len(DATASETS)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "black" if 0.6 < val < 0.9 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    if group_by_exposure:
        _draw_group_separator(ax, sep_after, len(unseen), len(seen),
                              "zero-shot (CLADDER unseen)",
                              "fine-tuned (CLADDER seen)")
    else:
        for i, m in enumerate(row_models):
            if m["is_lora"]:
                ax.axhline(i - 0.5, color="white", linewidth=2)

    ax.set_title("Overall accuracy — all models × all datasets",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out = out_dir / out_name
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out.name}")


# ── Plot 2: LoRA gain ─────────────────────────────────────────────────────────

def plot_lora_gain(data: Dict, out_dir: Path) -> None:
    lora_models = [m for m in MODELS if m["is_lora"] and m["base_key"]]

    labels, gains, colors = [], [], []
    for m in lora_models:
        ft_accs, base_accs = [], []
        for ds in DATASETS:
            if ds in data[m["key"]] and ds in data.get(m["base_key"], {}):
                ft_accs.append(data[m["key"]][ds]["overall"])
                base_accs.append(data[m["base_key"]][ds]["overall"])
        if ft_accs:
            gain = np.mean(ft_accs) - np.mean(base_accs)
            labels.append(m["label"].replace(" (LoRA)", ""))
            gains.append(gain)
            colors.append(model_color(m["label"]))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, gains, color=colors, alpha=ALPHA_BAR,
                  edgecolor="black", linewidth=LINEWIDTH_BAR,
                  hatch=HATCH_FINETUNED)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy gain (LoRA − zero-shot base)", fontsize=11)
    ax.set_title("Fine-tuning lift per model\n(averaged across 4 datasets)",
                 fontsize=13, fontweight="bold")

    for bar, gain in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2,
                gain + 0.005, f"+{gain:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, max(gains) * 1.2)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    out = out_dir / "lora_gain.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out.name}")


# ── Plot 3: Query-type heatmap ────────────────────────────────────────────────

def plot_query_type_heatmap(
    data: Dict,
    out_dir: Path,
    agg: str = "mean",
    group_by_exposure: bool = False,
    out_name: str = "query_type_heatmap.png",
) -> None:
    """
    agg: "mean" or "median" — how to aggregate accuracy across the 4 datasets.
    group_by_exposure: if True, rows are reordered so models that have NOT been
        fine-tuned on CLADDER appear first, followed by LoRA fine-tuned models,
        with a thick separator between the two groups.
    """
    agg_fn = np.median if agg == "median" else np.mean
    agg_label = "median" if agg == "median" else "avg"

    # Collect all query types
    all_qt = set()
    for m in MODELS:
        for ds in DATASETS:
            if ds in data[m["key"]]:
                all_qt.update(data[m["key"]][ds]["per_type"].keys())
    qt_order = sorted(all_qt, key=sort_key)

    # Determine row order
    if group_by_exposure:
        unseen = [m for m in MODELS if not m["is_lora"]]
        seen   = [m for m in MODELS if m["is_lora"]]
        row_models = unseen + seen
        separator_after = len(unseen) - 1   # draw thick line after this row index
    else:
        row_models = MODELS
        separator_after = None

    model_labels = [m["label"] for m in row_models]
    n_rows = len(row_models)
    matrix = np.full((n_rows, len(qt_order)), np.nan)

    for i, m in enumerate(row_models):
        for j, qt in enumerate(qt_order):
            vals = [
                data[m["key"]][ds]["per_type"][qt]
                for ds in DATASETS
                if ds in data[m["key"]] and qt in data[m["key"]][ds]["per_type"]
            ]
            if vals:
                matrix[i, j] = agg_fn(vals)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label=f"Accuracy ({agg_label} across datasets)", shrink=0.8)

    ax.set_xticks(range(len(qt_order)))
    ax.set_xticklabels([abbrev(qt) for qt in qt_order], fontsize=10,
                       rotation=30, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(model_labels, fontsize=10)

    for i in range(n_rows):
        for j in range(len(qt_order)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "black" if 0.5 < val < 0.92 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)

    # Light vertical grid lines between query types
    for j in range(len(qt_order)):
        ax.axvline(j - 0.5, color="white", linewidth=0.5, alpha=0.5)

    if group_by_exposure:
        _draw_group_separator(ax, separator_after, len(unseen), len(seen),
                              "zero-shot (CLADDER unseen)",
                              "fine-tuned (CLADDER seen)")
    else:
        # Original behaviour: thin lines between base/lora pairs
        for i in range(n_rows):
            if row_models[i]["is_lora"]:
                ax.axhline(i - 0.5, color="white", linewidth=1.5)

    title_agg = "median" if agg == "median" else "averaged"
    ax.set_title(
        f"Per query-type accuracy — all models ({title_agg} across 4 datasets)",
        fontsize=13, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    out = out_dir / out_name
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out.name}")


# ── Plot 4: Parameter efficiency scatter ──────────────────────────────────────

def plot_param_efficiency(data: Dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Open-source models only (GPT sizes are proprietary)
    os_models = [m for m in MODELS if m["size_b"] is not None
                 and "gpt" not in m["key"]]

    for m in os_models:
        accs = [data[m["key"]][ds]["overall"]
                for ds in DATASETS if ds in data[m["key"]]]
        if not accs:
            continue
        avg = np.mean(accs)
        color = model_color(m["label"])
        marker = "D" if m["is_lora"] else "o"
        hatch = HATCH_FINETUNED if m["is_lora"] else None
        size = 180

        ax.scatter(m["size_b"], avg, color=color, marker=marker,
                   s=size, edgecolors="black", linewidth=0.8,
                   zorder=3, label=m["label"])
        ax.annotate(m["label"], (m["size_b"], avg),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5)

    # Connect base → lora pairs with arrows
    lora_models = [m for m in os_models if m["is_lora"] and m["base_key"]]
    for m in lora_models:
        base_accs = [data[m["base_key"]][ds]["overall"]
                     for ds in DATASETS if ds in data.get(m["base_key"], {})]
        ft_accs   = [data[m["key"]][ds]["overall"]
                     for ds in DATASETS if ds in data[m["key"]]]
        if base_accs and ft_accs:
            ax.annotate("", xy=(m["size_b"], np.mean(ft_accs)),
                        xytext=(m["size_b"], np.mean(base_accs)),
                        arrowprops=dict(arrowstyle="->", color=model_color(m["label"]),
                                        lw=1.5))

    # GPT reference lines
    for m in MODELS:
        if "gpt" not in m["key"]:
            continue
        accs = [data[m["key"]][ds]["overall"]
                for ds in DATASETS if ds in data[m["key"]]]
        if not accs:
            continue
        color = model_color(m["label"])
        ax.axhline(np.mean(accs), linestyle="--", linewidth=1.2,
                   color=color, alpha=0.7, label=f"{m['label']} (zero-shot)")

    ax.set_xlabel("Model parameters (B)", fontsize=11)
    ax.set_ylabel("Average accuracy across 4 datasets", fontsize=11)
    ax.set_title("Parameter efficiency: model size vs accuracy\n"
                 "(○ = zero-shot base, ◆ = LoRA fine-tuned, arrows show gain)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 36)
    ax.set_ylim(0.45, 1.0)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = out_dir / "param_efficiency.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out.name}")


# ── Plot 5: Dataset difficulty profile ───────────────────────────────────────

def plot_dataset_difficulty(data: Dict, out_dir: Path) -> None:
    """Per query type: 4 bars (datasets), averaged across all LoRA models."""
    lora_keys = [m["key"] for m in MODELS if m["is_lora"]]

    all_qt = set()
    for key in lora_keys:
        for ds in DATASETS:
            if ds in data[key]:
                all_qt.update(data[key][ds]["per_type"].keys())
    qt_order = sorted(all_qt, key=sort_key)

    # matrix[qt][ds] = mean across LoRA models
    qt_ds: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for key in lora_keys:
        for ds in DATASETS:
            if ds not in data[key]:
                continue
            for qt, acc in data[key][ds]["per_type"].items():
                qt_ds[qt][ds].append(acc)

    n_qt = len(qt_order)
    n_ds = len(DATASETS)
    x = np.arange(n_qt)
    total_w = 0.80
    w = total_w / n_ds
    offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n_ds)

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, ds in enumerate(DATASETS):
        vals = [np.mean(qt_ds[qt][ds]) if qt_ds[qt][ds] else np.nan
                for qt in qt_order]
        ax.bar(x + offsets[i], vals, width=w * 0.92,
               color=DATASET_COLORS[ds], alpha=ALPHA_BAR,
               label=DS_LABELS[ds], edgecolor="black", linewidth=LINEWIDTH_BAR)

    ax.axvline(len(qt_order) - 0.5, color="#aaaaaa", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([abbrev(qt) for qt in qt_order],
                       rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy (avg across LoRA models)", fontsize=11)
    ax.set_title("Dataset difficulty per query type\n"
                 "(averaged across all LoRA fine-tuned models)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right", title="Dataset")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    out = out_dir / "dataset_difficulty.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data…")
    data = load_all()

    # Report coverage
    for m in MODELS:
        found = [DS_LABELS[ds] for ds in DATASETS if ds in data[m["key"]]]
        missing = [DS_LABELS[ds] for ds in DATASETS if ds not in data[m["key"]]]
        status = f"✓ {found}" + (f"  MISSING: {missing}" if missing else "")
        print(f"  {m['label']:<28} {status}")

    print("\nGenerating plots…")
    plot_cross_model_heatmap(data, OUT_DIR)
    plot_cross_model_heatmap(data, OUT_DIR,
                             group_by_exposure=True,
                             out_name="cross_model_heatmap_grouped.png")
    plot_lora_gain(data, OUT_DIR)
    # Original mean-aggregated heatmap (base/lora pairs grouped)
    plot_query_type_heatmap(data, OUT_DIR)
    # Median-aggregated + grouped by CLADDER exposure
    plot_query_type_heatmap(data, OUT_DIR,
                            agg="median",
                            group_by_exposure=True,
                            out_name="query_type_heatmap_median.png")
    plot_param_efficiency(data, OUT_DIR)
    plot_dataset_difficulty(data, OUT_DIR)
    print(f"\nDone. All plots → {OUT_DIR}/")


if __name__ == "__main__":
    main()
