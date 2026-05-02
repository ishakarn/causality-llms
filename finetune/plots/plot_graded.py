"""
Plot graded-exposure experiment results.

Reads eval results from:
    finetune/eval_results/graded/<model_tag>_n<N>_lora/<dataset>/score/

Produces 3 figures in outputs/plots/graded/:
  1. learning_curve_overall_<tag>.png   — overall accuracy vs N, one panel per dataset
  2. learning_curve_per_qtype_<tag>.png — 2×5 grid; one query-type per panel
  3. heatmap_<model_tag>.png            — query_type × N heatmap per model

Usage:
    python finetune/plots/plot_graded.py --models qwen3b
    python finetune/plots/plot_graded.py --models qwen3b llama8b olmo32b
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── shared style ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plot_style import apply_defaults
from query_type_labels import abbrev, RUNG_ORDER

apply_defaults()

# ── constants ─────────────────────────────────────────────────────────────────
SIZES    = [50, 100, 250, 500, 1000, 2000]
DATASETS = ["easy", "hard", "anticommonsense", "noncommonsense"]
GRADED_SPLITS_DIR = Path("finetune/splits_graded")

DATASET_LABELS = {
    "easy":            "Easy",
    "hard":            "Hard",
    "anticommonsense": "Anti-Commonsense",
    "noncommonsense":  "Non-Commonsense",
}

DATASET_COLORS = {
    "easy":            "#0072b2",
    "hard":            "#d55e00",
    "anticommonsense": "#009e73",
    "noncommonsense":  "#e69f00",
}

# Display label and line style per model tag
MODEL_STYLES = {
    "qwen3b":  dict(linestyle="-",  marker="o", label="Qwen2.5-3B"),
    "olmo7b":  dict(linestyle="--", marker="s", label="OLMo-3-7B"),
    "llama8b": dict(linestyle="-.", marker="^", label="Llama-3.1-8B"),
    "olmo32b": dict(linestyle=":",  marker="D", label="OLMo-3.1-32B"),
}

# When only one model is shown, give each dataset a distinct linestyle too
DATASET_LINESTYLES = {
    "easy":            "-",
    "hard":            "--",
    "anticommonsense": "-.",
    "noncommonsense":  ":",
}
DATASET_MARKERS = {
    "easy":            "o",
    "hard":            "s",
    "anticommonsense": "^",
    "noncommonsense":  "D",
}

MODEL_DISPLAY = {
    "qwen3b":  "Qwen2.5-3B-Instruct",
    "olmo7b":  "OLMo-3-7B-Instruct",
    "llama8b": "Llama-3.1-8B-Instruct",
    "olmo32b": "OLMo-3.1-32B-Instruct",
}

# Canonical query-type order (Rung 1 → 2 → 3)
QUERY_TYPE_ORDER = RUNG_ORDER


def load_train_qt_counts(graded_dir: Path = GRADED_SPLITS_DIR) -> dict[str, dict[int, int]]:
    """
    Returns {query_type: {N: count}} for all graded training splits.
    Used to annotate x-tick labels in the per-query-type plot.
    """
    counts: dict[str, dict[int, int]] = {}
    for n in SIZES:
        path = graded_dir / f"n{n}" / "train.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            qt = json.loads(line)["query_type"]
            counts.setdefault(qt, {})
            counts[qt][n] = counts[qt].get(n, 0) + 1
    return counts


def wilson_ci(correct: int, n: int, z: float = 1.96):
    """Wilson 95% confidence interval, clipped to [0, 1]."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _setup_log_xaxis(ax, sizes):
    """Configure a log-scale x-axis so ticks land exactly on data points."""
    ax.set_xscale("log")
    ax.set_xlim(sizes[0] * 0.6, sizes[-1] * 1.5)
    ax.set_xticks(sizes)
    ax.xaxis.set_minor_locator(ticker.NullLocator())   # no minor ticks
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=8)


# ── data loading ──────────────────────────────────────────────────────────────

def load_summary(path: Path) -> dict | None:
    p = path / "score" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_per_qtype(path: Path) -> pd.DataFrame | None:
    p = path / "score" / "per_query_type.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_all_results(base_dir: Path, model_tags: list[str]) -> dict:
    results = {}
    for model_tag in model_tags:
        results[model_tag] = {}
        for N in SIZES:
            run_name = f"{model_tag}_n{N}_lora"
            results[model_tag][N] = {}
            for ds in DATASETS:
                ds_dir  = base_dir / run_name / ds
                summary = load_summary(ds_dir)
                per_qt  = load_per_qtype(ds_dir)
                if summary is not None:
                    results[model_tag][N][ds] = {
                        "summary":  summary,
                        "per_qtype": per_qt,
                    }
    return results


# ── Figure 1: overall learning curves ─────────────────────────────────────────

def plot_overall_curves(results: dict, out_path: Path) -> None:
    """
    Overall accuracy vs N.

    Single model: 1 panel, 4 lines (one per dataset, colored by dataset).
    Multi-model:  4 panels (one per dataset), 1 line per model colored by
                  model, styled per MODEL_STYLES. Legend on every panel.
    """
    from plot_style import model_color as mc

    model_tags  = list(results.keys())
    model_names = ", ".join(MODEL_DISPLAY.get(mt, mt) for mt in model_tags)
    multi_model = len(model_tags) > 1

    if not multi_model:
        # ── Single model: one panel, all 4 datasets overlaid ──────────────────
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        fig.suptitle(f"Graded Exposure — Overall Accuracy vs Training Size\n{model_names}",
                     fontsize=12, y=1.03)

        for ds in DATASETS:
            color = DATASET_COLORS[ds]
            accs, lows, highs, ns_plotted = [], [], [], []
            for N in SIZES:
                entry = results[model_tags[0]].get(N, {}).get(ds)
                if entry is None:
                    continue
                s = entry["summary"]
                lo, hi = wilson_ci(s["correct"], s["n"])
                accs.append(s["acc_all"])
                lows.append(lo)
                highs.append(hi)
                ns_plotted.append(N)
            if not accs:
                continue
            ax.plot(ns_plotted, accs, color=color, linewidth=2,
                    linestyle="-", marker=DATASET_MARKERS[ds], markersize=6,
                    label=DATASET_LABELS[ds])
            ax.fill_between(ns_plotted, lows, highs, color=color, alpha=0.15)

        _setup_log_xaxis(ax, SIZES)
        ax.set_ylim(0.4, 1.02)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Training examples (N)")
        ax.legend(fontsize=9, loc="lower right")

    else:
        # ── Multi-model: 4 panels (one per dataset), lines = models ───────────
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        fig.suptitle(f"Graded Exposure — Overall Accuracy vs Training Size\n{model_names}",
                     fontsize=12, y=1.03)

        for ax_idx, (ax, ds) in enumerate(zip(axes, DATASETS)):
            ax.set_title(DATASET_LABELS[ds], fontsize=11)
            ax.set_xlabel("Training examples (N)")
            _setup_log_xaxis(ax, SIZES)
            ax.set_ylim(0.4, 1.02)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
            ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

            for model_tag in model_tags:
                style = MODEL_STYLES.get(model_tag,
                                         dict(linestyle="-", marker="o", label=model_tag))
                color = mc(MODEL_DISPLAY.get(model_tag, model_tag))
                accs, lows, highs, ns_plotted = [], [], [], []
                for N in SIZES:
                    entry = results[model_tag].get(N, {}).get(ds)
                    if entry is None:
                        continue
                    s = entry["summary"]
                    lo, hi = wilson_ci(s["correct"], s["n"])
                    accs.append(s["acc_all"])
                    lows.append(lo)
                    highs.append(hi)
                    ns_plotted.append(N)
                if not accs:
                    continue
                ax.plot(ns_plotted, accs, color=color, linewidth=2,
                        linestyle=style["linestyle"], marker=style["marker"],
                        markersize=6, label=style["label"])
                ax.fill_between(ns_plotted, lows, highs, color=color, alpha=0.15)

            if ax_idx == 0:
                ax.set_ylabel("Accuracy")
            ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved → {out_path}")


# ── Figure 2: per-query-type learning curves ──────────────────────────────────

def plot_per_qtype_curves(results: dict, out_path: Path,
                          graded_dir: Path = GRADED_SPLITS_DIR) -> None:
    """
    2×5 grid — one subplot per query type (3-letter labels from query_type_labels).
    X=N (log), Y=accuracy.
    Lines = datasets (4 colors), linestyle = model.
    X-tick labels show both total N and the per-query-type training count (in
    parentheses) so rare types like exp_away/collider_bias are easy to spot.
    """
    model_tags  = list(results.keys())
    model_names = ", ".join(MODEL_DISPLAY.get(mt, mt) for mt in model_tags)
    train_qt_counts = load_train_qt_counts(graded_dir)

    all_qtypes = set()
    for mt in model_tags:
        for N in SIZES:
            for ds in DATASETS:
                entry = results[mt].get(N, {}).get(ds)
                if entry and entry["per_qtype"] is not None:
                    all_qtypes.update(entry["per_qtype"]["query_type"].tolist())

    ordered_qt = [q for q in QUERY_TYPE_ORDER if q in all_qtypes]
    ordered_qt += sorted(all_qtypes - set(QUERY_TYPE_ORDER))

    ncols = 5
    nrows = (len(ordered_qt) + ncols - 1) // ncols

    # sharex=False so each subplot gets its own x-axis tick label object —
    # with sharex=True, set_xticklabels() writes to the shared ticker and the
    # last subplot would overwrite all others, making every panel show
    # identical per-query-type training counts.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.4, nrows * 3.0),
                             sharex=False, sharey=True)
    fig.suptitle(f"Per-Query-Type Learning Curves — {model_names}",
                 fontsize=12, y=1.01)
    axes_flat = axes.flat

    for i, qt in enumerate(ordered_qt):
        ax = axes_flat[i]
        ax.set_title(abbrev(qt), fontsize=11, fontweight="bold")
        _setup_log_xaxis(ax, SIZES)
        ax.set_ylim(0.3, 1.05)
        ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

        multi_model = len(model_tags) > 1

        for ds in DATASETS:
            color = DATASET_COLORS[ds]
            for model_tag in model_tags:
                style = MODEL_STYLES.get(model_tag,
                                         dict(linestyle="-", marker="o", label=model_tag))
                # Single model: solid lines, vary marker per dataset.
                # Multi-model: model-specific linestyle+marker.
                ls = style["linestyle"] if multi_model else "-"
                mk = DATASET_MARKERS[ds] if not multi_model else style["marker"]
                accs, ns_plotted = [], []
                for N in SIZES:
                    entry = results[model_tag].get(N, {}).get(ds)
                    if entry is None or entry["per_qtype"] is None:
                        continue
                    row = entry["per_qtype"][entry["per_qtype"]["query_type"] == qt]
                    if row.empty:
                        continue
                    accs.append(float(row["acc_all"].iloc[0]))
                    ns_plotted.append(N)

                if not accs:
                    continue

                ax.plot(ns_plotted, accs, color=color, linewidth=1.8,
                        linestyle=ls, marker=mk, markersize=5,
                        label=DATASET_LABELS[ds] if model_tag == model_tags[0] else None)

        # Legend on every subplot so colors are identifiable without cross-referencing
        ax.legend(fontsize=6, loc="lower right", framealpha=0.7)
        if i % ncols == 0:
            ax.set_ylabel("Accuracy")
        # X-tick labels: show total N and per-query-type train count.
        # tick_params(labelbottom=True) is needed because sharex=True suppresses
        # labels on all but the bottom row by default.
        qt_counts_for = train_qt_counts.get(qt, {})
        tick_labels = [
            f"{n}\n({qt_counts_for.get(n, 0)})"
            for n in SIZES
        ]
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=7)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("N  (train count)")

    for j in range(i + 1, nrows * ncols):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved → {out_path}")


# ── Figure 3: heatmap query_type × N ──────────────────────────────────────────

def plot_heatmap(results: dict, out_path: Path, model_tag: str) -> None:
    """
    One heatmap panel per dataset (4 panels). Rows = query types (3-letter labels,
    sorted by avg accuracy). Columns = N values.
    """
    all_qtypes = set()
    for N in SIZES:
        for ds in DATASETS:
            entry = results[model_tag].get(N, {}).get(ds)
            if entry and entry["per_qtype"] is not None:
                all_qtypes.update(entry["per_qtype"]["query_type"].tolist())

    ordered_qt = [q for q in QUERY_TYPE_ORDER if q in all_qtypes]
    ordered_qt += sorted(all_qtypes - set(QUERY_TYPE_ORDER))

    display_name = MODEL_DISPLAY.get(model_tag, model_tag)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f"Query-Type × Training Size Accuracy — {display_name}",
                 fontsize=13, y=1.02)

    for ax, ds in zip(axes, DATASETS):
        mat = np.full((len(ordered_qt), len(SIZES)), np.nan)
        for j, N in enumerate(SIZES):
            entry = results[model_tag].get(N, {}).get(ds)
            if entry is None or entry["per_qtype"] is None:
                continue
            for i, qt in enumerate(ordered_qt):
                row = entry["per_qtype"][entry["per_qtype"]["query_type"] == qt]
                if not row.empty:
                    mat[i, j] = float(row["acc_all"].iloc[0])

        # Sort rows by mean accuracy descending
        row_means  = np.nanmean(mat, axis=1)
        sort_idx   = np.argsort(-row_means)
        mat_sorted = mat[sort_idx]
        qt_labels  = [abbrev(ordered_qt[k]) for k in sort_idx]

        im = ax.imshow(mat_sorted, vmin=0.4, vmax=1.0, aspect="auto",
                       cmap="RdYlGn", interpolation="nearest")

        ax.set_title(DATASET_LABELS[ds], fontsize=11)
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels([str(s) for s in SIZES], rotation=45, fontsize=8)
        ax.set_yticks(range(len(qt_labels)))
        ax.set_yticklabels(qt_labels, fontsize=9)

        for i in range(mat_sorted.shape[0]):
            for j in range(mat_sorted.shape[1]):
                val = mat_sorted[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.6 or val > 0.9 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     format=ticker.PercentFormatter(xmax=1.0, decimals=0))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3b"],
                    help="Model tags: qwen3b, olmo7b, llama8b, olmo32b")
    ap.add_argument("--results_dir",      default="finetune/eval_results/graded")
    ap.add_argument("--graded_splits_dir", default="finetune/splits_graded",
                    help="Directory containing splits_graded/n*/train.jsonl for count annotations")
    ap.add_argument("--out_dir",     default="outputs/plots/graded")
    args = ap.parse_args()

    base_dir = Path(args.results_dir)
    out_dir  = Path(args.out_dir)

    print(f"Loading results from: {base_dir}")
    results = load_all_results(base_dir, args.models)

    total = 0
    for mt in args.models:
        for N in SIZES:
            for ds in DATASETS:
                if results[mt].get(N, {}).get(ds):
                    total += 1
                    acc = results[mt][N][ds]["summary"]["acc_all"]
                    print(f"  {mt}  N={N:<5}  {ds:<20}  acc={acc:.3f}")
    if total == 0:
        print("No results found yet — run eval_graded_seq.sh first.")
        return

    print(f"\nGenerating plots → {out_dir}/")
    tag = "_".join(args.models)

    plot_overall_curves(results,   out_dir / f"learning_curve_overall_{tag}.png")
    plot_per_qtype_curves(results, out_dir / f"learning_curve_per_qtype_{tag}.png",
                          graded_dir=Path(args.graded_splits_dir))

    for mt in args.models:
        has_data = any(
            results[mt].get(N, {}).get(ds)
            for N in SIZES for ds in DATASETS
        )
        if has_data:
            plot_heatmap(results, out_dir / f"heatmap_{mt}.png", mt)

    print("\nDone.")


if __name__ == "__main__":
    main()
