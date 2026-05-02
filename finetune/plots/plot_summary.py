"""
Summary plots across all datasets:

1. Cross-dataset heatmap  — rows=models, cols=datasets, overall accuracy as color
2. Fine-tuning delta chart — LoRA−base gain per query type for OLMo & Qwen
3. GPT vs fine-tuned       — GPT zero-shot vs open-source fine-tuned, per dataset

Usage:
    python finetune/plot_summary.py
    python finetune/plot_summary.py --outputs_root outputs --out_dir outputs/plots/summary
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from query_type_labels import abbrev, sort_key
from plot_style import model_color, HATCH_FINETUNED, ALPHA_BAR, LINEWIDTH_BAR, apply_defaults

apply_defaults()

DATASETS = [
    "cladder-v1-q-easy",
    "cladder-v1-q-hard",
    "cladder-v1-q-balanced",
    "cladder-v1-q-commonsense",
    "cladder-v1-q-noncommonsense",
    "cladder-v1-q-anticommonsense",
]

DATASET_LABELS = {
    "cladder-v1-q-easy":           "Easy",
    "cladder-v1-q-hard":           "Hard",
    "cladder-v1-q-balanced":       "Balanced",
    "cladder-v1-q-commonsense":    "Common-\nsense",
    "cladder-v1-q-noncommonsense": "Non-\ncommon",
    "cladder-v1-q-anticommonsense":"Anti-\ncommon",
}

_DISPLAY = {
    "gpt-5-nano":        "GPT-5-Nano",
    "gpt-oss-20b":       "GPT-OSS-20B",
    "gpt54":             "GPT-5.4",
    "olmo3-7b-instruct": "OLMo-3-7B",
    "qwen25-3b-instruct":"Qwen 2.5-3B",
}

def display(run_id: str) -> str:
    low = run_id.lower()
    for k, v in _DISPLAY.items():
        if k in low:
            return v
    return run_id

def short_label(run_id: str, is_ft: bool) -> str:
    return display(run_id) + (" (LoRA)" if is_ft else " (base)")


# ── Data helpers ──────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def compute_per_type(rows: List[Dict]) -> pd.DataFrame:
    by_type: Dict[str, Dict] = {}
    total_n = total_correct = 0
    for r in rows:
        qt = r.get("query_type", "UNKNOWN")
        if qt not in by_type:
            by_type[qt] = {"n": 0, "correct": 0}
        by_type[qt]["n"] += 1
        total_n += 1
        if r.get("pred") is not None and r.get("pred") == r.get("gold"):
            by_type[qt]["correct"] += 1
            total_correct += 1
    out = []
    for qt, s in by_type.items():
        n, k = s["n"], s["correct"]
        acc = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        out.append({"query_type": qt, "n": n, "correct": k,
                    "acc": acc, "ci_lo": lo, "ci_hi": hi})
    acc_all = total_correct / total_n if total_n else 0.0
    lo, hi = wilson_ci(total_correct, total_n)
    out.append({"query_type": "overall", "n": total_n, "correct": total_correct,
                "acc": acc_all, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out).set_index("query_type")


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_models(dataset: str, outputs_root: Path) -> List[Dict]:
    found = []
    for subdir, is_ft in [("baseline", False), ("finetuned", True)]:
        d = outputs_root / dataset / subdir
        if not d.exists():
            continue
        for model_dir in sorted(d.glob("*")):
            jsonls = list(model_dir.glob("*.jsonl"))
            if jsonls:
                found.append({"jsonl": jsonls[0], "is_finetuned": is_ft,
                               "run_id": model_dir.name})
    return found


def load_all(outputs_root: Path) -> Dict[str, List[Dict]]:
    """Returns {dataset: [entry, ...]} where entry has run_id, is_finetuned, df."""
    result = {}
    for ds in DATASETS:
        raw = discover_models(ds, outputs_root)
        entries = []
        for e in raw:
            rows = load_jsonl(e["jsonl"])
            entries.append({
                "run_id":       e["run_id"],
                "is_finetuned": e["is_finetuned"],
                "df":           compute_per_type(rows),
            })
        result[ds] = entries
    return result


# ── Plot 1: Cross-dataset heatmap ─────────────────────────────────────────────

def plot_cross_dataset_heatmap(all_data: Dict[str, List[Dict]], out_path: Path) -> None:
    # Collect unique model series across all datasets
    seen, model_keys = set(), []
    for ds, entries in all_data.items():
        for e in entries:
            key = (e["run_id"], e["is_finetuned"])
            if key not in seen:
                seen.add(key)
                model_keys.append(key)

    ds_labels  = [DATASET_LABELS[d] for d in DATASETS]
    row_labels = [short_label(r, ft) for r, ft in model_keys]

    mat = np.full((len(model_keys), len(DATASETS)), np.nan)
    for col, ds in enumerate(DATASETS):
        entry_map = {(e["run_id"], e["is_finetuned"]): e for e in all_data[ds]}
        for row, key in enumerate(model_keys):
            e = entry_map.get(key)
            if e and "overall" in e["df"].index:
                mat[row, col] = e["df"].loc["overall", "acc"]

    fig, ax = plt.subplots(figsize=(len(DATASETS) * 1.6 + 1, len(model_keys) * 0.75 + 1.2))

    im = ax.imshow(mat, aspect="auto", vmin=0.45, vmax=0.95,
                   cmap="RdYlGn", interpolation="nearest")

    for i in range(len(model_keys)):
        for j in range(len(DATASETS)):
            v = mat[i, j]
            if not np.isnan(v):
                text_color = "black" if 0.5 < v < 0.82 else "white"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(ds_labels, fontsize=9)
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels(row_labels, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Overall Accuracy", fontsize=9)
    ax.set_title("Overall accuracy across datasets & models", fontsize=12,
                 fontweight="bold", pad=10)

    # Horizontal dividers between model families
    # GPT baselines / OLMo+Qwen base / OLMo+Qwen LoRA
    prev_ft = None
    for i, (run_id, is_ft) in enumerate(model_keys):
        is_gpt = any(k in run_id.lower() for k in ("gpt-5", "gpt-oss"))
        if i > 0 and is_ft != prev_ft:
            ax.axhline(i - 0.5, color="white", linewidth=2)
        prev_ft = is_ft

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}")


# ── Plot 2: Fine-tuning delta chart ───────────────────────────────────────────

def plot_delta(all_data: Dict[str, List[Dict]], out_path: Path) -> None:
    """LoRA − base accuracy delta per query type, averaged across all datasets."""

    target_models = {
        "olmo3-7b-instruct": "OLMo-3-7B-Instruct",
        "qwen25-3b-instruct": "Qwen 2.5-3B-Instruct",
    }

    # Collect per-query-type deltas: {model_key: {qt: [delta, ...]}}
    from collections import defaultdict
    deltas: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for ds, entries in all_data.items():
        entry_map: Dict[str, Dict] = {}
        for e in entries:
            low = e["run_id"].lower()
            for mk in target_models:
                if mk in low:
                    role = "ft" if e["is_finetuned"] else "base"
                    entry_map[f"{mk}_{role}"] = e

        for mk in target_models:
            base_e = entry_map.get(f"{mk}_base")
            ft_e   = entry_map.get(f"{mk}_ft")
            if not base_e or not ft_e:
                continue
            all_qt = set(base_e["df"].index) & set(ft_e["df"].index) - {"overall"}
            for qt in all_qt:
                delta = ft_e["df"].loc[qt, "acc"] - base_e["df"].loc[qt, "acc"]
                deltas[mk][qt].append(delta)
            # overall
            d_overall = ft_e["df"].loc["overall","acc"] - base_e["df"].loc["overall","acc"]
            deltas[mk]["overall"].append(d_overall)

    # Average across datasets
    avg: Dict[str, Dict[str, float]] = {}
    for mk, qt_map in deltas.items():
        avg[mk] = {qt: float(np.mean(v)) for qt, v in qt_map.items()}

    all_qt = set()
    for qt_map in avg.values():
        all_qt.update(qt_map.keys())
    all_qt.discard("overall")
    type_order = sorted(all_qt, key=sort_key) + ["overall"]

    n_models = len(avg)
    x = np.arange(len(type_order))
    total_w = 0.7
    w = total_w / n_models
    offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n_models)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (mk, qt_map) in enumerate(avg.items()):
        vals = [qt_map.get(qt, 0.0) for qt in type_order]
        color = model_color(mk)
        ax.bar(x + offsets[i], vals, width=w * 0.92,
               color=color, alpha=ALPHA_BAR + 0.05,
               edgecolor="black", linewidth=LINEWIDTH_BAR,
               label=target_models[mk])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(len(type_order) - 1.5, color="#aaaaaa", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([abbrev(qt) for qt in type_order], fontsize=9)
    ax.set_ylabel("Accuracy gain (LoRA − base)", fontsize=10)
    ax.set_title("Fine-tuning gain per query type (averaged across all datasets)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, max(max(v.values()) for v in avg.values()) + 0.07)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}")


# ── Plot 3: GPT vs fine-tuned ─────────────────────────────────────────────────

def plot_gpt_vs_finetuned(all_data: Dict[str, List[Dict]], out_path: Path) -> None:
    """
    Two groups per dataset bar cluster:
      Left group:  GPT-5-Nano (base), GPT-OSS-20B (base)
      Right group: OLMo LoRA, Qwen LoRA
    Dashed line = OLMo/Qwen base (for context).
    """
    GPT_KEYS    = ("gpt-5-nano", "gpt-oss-20b")
    OPENFT_KEYS = ("olmo3-7b-instruct", "qwen25-3b-instruct")

    # For each dataset, gather overall accuracy for each series
    series: Dict[str, Dict[str, float]] = {}   # run_label -> {dataset: acc}

    for ds, entries in all_data.items():
        for e in entries:
            low = e["run_id"].lower()
            label = short_label(e["run_id"], e["is_finetuned"])
            if "overall" not in e["df"].index:
                continue
            acc = e["df"].loc["overall", "acc"]
            if label not in series:
                series[label] = {}
            series[label][ds] = acc

    # Define display order: GPT baselines first, then open-source LoRA
    def _group(label: str) -> int:
        l = label.lower()
        if "gpt" in l and "base" in l:   return 0
        if "lora" in l:                   return 1
        return 2  # open-source base (context lines only)

    ordered = sorted(series.keys(), key=_group)

    ds_x = np.arange(len(DATASETS))
    n_bar_series = sum(1 for lbl in ordered if _group(lbl) in (0, 1))
    total_w = 0.82
    w = total_w / n_bar_series

    bar_series = [lbl for lbl in ordered if _group(lbl) in (0, 1)]
    offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n_bar_series)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Separator between GPT group and LoRA group
    n_gpt = sum(1 for lbl in bar_series if _group(lbl) == 0)

    for i, lbl in enumerate(bar_series):
        vals = [series[lbl].get(ds, np.nan) for ds in DATASETS]
        run_id = lbl.split(" (")[0].lower().replace("-", "").replace(".", "").replace(" ", "")
        # find original run_id for color
        color = "#666666"
        for entries in all_data.values():
            for e in entries:
                if display(e["run_id"]).lower().replace("-","").replace(".","").replace(" ","") == run_id.replace("3b","").replace("7b","") or \
                   display(e["run_id"]) in lbl:
                    color = model_color(e["run_id"])
                    break
            else:
                continue
            break

        is_ft = "(LoRA)" in lbl
        ax.bar(ds_x + offsets[i], vals, width=w * 0.92,
               color=color, alpha=ALPHA_BAR,
               hatch=HATCH_FINETUNED if is_ft else None,
               edgecolor="black", linewidth=LINEWIDTH_BAR,
               label=lbl.replace("\n", " "))

    # Draw vertical gap between GPT and LoRA groups
    gap_x = (offsets[n_gpt - 1] + offsets[n_gpt]) / 2
    for xi in ds_x:
        ax.axvline(xi + gap_x, color="#cccccc", linewidth=1, linestyle="--", zorder=0)

    # Dashed reference lines for open-source base
    base_series = [lbl for lbl in ordered if _group(lbl) == 2]
    line_styles = ["--", ":"]
    for li, lbl in enumerate(base_series):
        vals = [series[lbl].get(ds, np.nan) for ds in DATASETS]
        color = "#666666"
        for entries in all_data.values():
            for e in entries:
                if display(e["run_id"]) in lbl:
                    color = model_color(e["run_id"])
                    break
            else:
                continue
            break
        ax.plot(ds_x, vals, linestyle=line_styles[li % 2], color=color,
                linewidth=1.5, alpha=0.6, marker="o", markersize=4,
                label=lbl.replace("\n", " ") + " (ref)")

    ax.axhline(0.5, linestyle=":", linewidth=1, color="black", alpha=0.4)
    ax.set_xticks(ds_x)
    ax.set_xticklabels([DATASET_LABELS[d].replace("\n", " ") for d in DATASETS], fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Overall Accuracy", fontsize=10)
    ax.set_title("GPT zero-shot vs. fine-tuned open-source models — per dataset",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Two-part legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, loc="lower right", ncol=2)

    # Annotate the two groups
    ax.text(ds_x[0] + offsets[0] - 0.05, 0.97, "◀ GPT zero-shot",
            fontsize=7.5, color="#555555", va="top")
    ax.text(ds_x[0] + offsets[n_gpt] - 0.05, 0.97, "Fine-tuned ▶",
            fontsize=7.5, color="#555555", va="top")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--out_dir",      default="outputs/plots/summary")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    out_dir      = Path(args.out_dir)

    print("Loading data…")
    all_data = load_all(outputs_root)
    for ds, entries in all_data.items():
        print(f"  {ds}: {len(entries)} model series")

    print()
    plot_cross_dataset_heatmap(all_data, out_dir / "cross_dataset_heatmap.png")
    plot_delta(all_data, out_dir / "finetuning_delta.png")
    plot_gpt_vs_finetuned(all_data, out_dir / "gpt_vs_finetuned.png")


if __name__ == "__main__":
    main()
