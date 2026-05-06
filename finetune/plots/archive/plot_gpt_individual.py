"""
Individual per-query-type accuracy plots for GPT baseline models.
Produces one plot per dataset per model, matching the style of
plot_comparison.py (same dimensions, colors from plot_style.py).

Usage:
    python finetune/plot_gpt_individual.py
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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

GPT_MODELS = [
    ("gpt-5-nano-baseline",  "GPT-5-Nano (zero-shot)"),
    ("gpt-oss-20b-baseline", "GPT-OSS-20B (zero-shot)"),
]

# OLMo and Qwen: show the LoRA model as a single bar (consistent with GPT plots)
FINETUNED_MODELS = [
    ("olmo3-7b-instruct-lora",  "OLMo-3-7B-Instruct (LoRA)"),
    ("qwen25-3b-instruct-lora", "Qwen 2.5-3B-Instruct (LoRA)"),
]


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def plot_single(df: pd.DataFrame, run_id: str, label: str,
                dataset: str, out_path: Path, is_finetuned: bool = False) -> None:
    all_types = [t for t in df.index if t != "overall"]
    type_order = sorted(all_types, key=sort_key) + ["overall"]

    x   = np.arange(len(type_order))
    w   = 0.55
    color = model_color(run_id)

    fig, ax = plt.subplots(figsize=(15, 6.5))

    df_ord = df.reindex(type_order)
    acc = df_ord["acc"].to_numpy(dtype=float)
    lo  = np.clip((df_ord["acc"] - df_ord["ci_lo"]).to_numpy(dtype=float), 0, None)
    hi  = np.clip((df_ord["ci_hi"] - df_ord["acc"]).to_numpy(dtype=float), 0, None)

    ax.bar(x, acc, width=w, color=color, alpha=ALPHA_BAR, label=label,
           hatch=HATCH_FINETUNED if is_finetuned else None,
           yerr=np.vstack([lo, hi]),
           error_kw={"elinewidth": 1.1, "capsize": 3, "ecolor": "#333333"},
           edgecolor="black", linewidth=LINEWIDTH_BAR)

    ns = df_ord["n"].to_numpy(dtype=int)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{abbrev(qt)}\n(n={int(n)})" for qt, n in zip(type_order, ns)],
        rotation=35, ha="right", fontsize=9,
    )

    ax.axvline(len(type_order) - 1.5, color="#aaaaaa", linewidth=1, linestyle="--")
    ax.axhline(0.5, linestyle=":", linewidth=1.2, color="black", alpha=0.7)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("Accuracy (Wilson 95% CI)", fontsize=11)
    ax.set_xlabel("Query type", fontsize=10)
    ax.set_title(f"{dataset}: {label} — per query type accuracy",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")

    overall_acc = df.loc["overall", "acc"] if "overall" in df.index else float("nan")
    ax.text(0.01, 0.97, f"Overall: {overall_acc:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    # CSV
    csv_path = out_path.with_suffix(".csv")
    df.reindex(type_order).to_csv(csv_path)

    print(f"  Wrote: {out_path.name}  (overall={overall_acc:.4f})")


def main() -> None:
    root = Path("outputs")

    for dataset in DATASETS:
        print(f"\n=== {dataset} ===")

        for run_id, label in GPT_MODELS:
            jsonl = root / dataset / "baseline" / run_id / f"{run_id}.jsonl"
            if not jsonl.exists():
                print(f"  [skip] {run_id} — file not found")
                continue
            rows = load_jsonl(jsonl)
            df   = compute_per_type(rows)
            out  = root / "plots" / "newdata" / dataset / f"{dataset}__{run_id}.png"
            plot_single(df, run_id, label, dataset, out, is_finetuned=False)

        for run_id, label in FINETUNED_MODELS:
            jsonl = root / dataset / "finetuned" / run_id / f"{run_id}__{dataset}.jsonl"
            if not jsonl.exists():
                print(f"  [skip] {run_id} — file not found")
                continue
            rows = load_jsonl(jsonl)
            df   = compute_per_type(rows)
            out  = root / "plots" / "newdata" / dataset / f"{dataset}__{run_id}.png"
            plot_single(df, run_id, label, dataset, out, is_finetuned=True)


if __name__ == "__main__":
    main()
