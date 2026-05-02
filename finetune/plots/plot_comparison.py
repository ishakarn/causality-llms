"""
Side-by-side comparison: baseline zero-shot vs LoRA fine-tuned.

Two subplots in one figure — same query types, same y-axis, Wilson 95% CI.
The baseline JSONL covers all 2880 samples; this script filters it to the
test-split question_ids so both panels show predictions on identical examples.

Usage (Instruct variant):
    python finetune/plot_comparison.py \\
        --baseline  outputs/queries_easy__allenai__Olmo-3-7B-Instruct__seed0.jsonl \\
        --finetuned finetune/eval_results/olmo3-7b-instruct-lora/olmo3-7b-instruct-lora__test.jsonl \\
        --test_split finetune/splits/test.jsonl \\
        --out        outputs/plots/instruct_base_vs_finetuned.png

Usage (Think variant, once that run is done):
    python finetune/plot_comparison.py \\
        --baseline  outputs/queries_easy__allenai__Olmo-3-7B-Think__seed0.jsonl \\
        --finetuned finetune/eval_results/olmo3-7b-think-lora/olmo3-7b-think-lora__test.jsonl \\
        --test_split finetune/splits/test.jsonl \\
        --out        outputs/plots/think_base_vs_finetuned.png
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from query_type_labels import abbrev, sort_key, RUNG_ORDER
from plot_style import model_color, HATCH_FINETUNED, ALPHA_BAR, LINEWIDTH_BAR, apply_defaults

apply_defaults()


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def model_label_from_rows(rows: List[Dict], fallback: str = "model") -> str:
    if not rows:
        return fallback
    r = rows[0]
    mid = r.get("model_id_str") or r.get("model_name_or_path") or r.get("model") or ""
    return mid.split("/")[-1] if mid else fallback


def filter_to_test_ids(rows: List[Dict], test_qids: set) -> List[Dict]:
    filtered = [r for r in rows if r.get("question_id") in test_qids]
    return filtered


# ── Statistics ────────────────────────────────────────────────────────────────

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
    """Compute per-query-type accuracy + Wilson CI from a list of prediction dicts."""
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

    # Overall row
    acc_all = total_correct / total_n if total_n else 0.0
    lo, hi = wilson_ci(total_correct, total_n)
    out.append({"query_type": "overall", "n": total_n, "correct": total_correct,
                "acc": acc_all, "ci_lo": lo, "ci_hi": hi})

    return pd.DataFrame(out).set_index("query_type")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(
    base_df: pd.DataFrame,
    ft_df: pd.DataFrame,
    base_label: str,
    ft_label: str,
    color: str,
    out_path: str,
    title: str,
) -> None:
    # Query type order: canonical rung order, with "overall" pinned last
    present = [t for t in base_df.index if t != "overall"]
    type_order = sorted(present, key=sort_key) + ["overall"]

    x = np.arange(len(type_order))
    w = 0.38   # paired bars

    fig, ax = plt.subplots(figsize=(15, 6.5))

    def _bars(ax, df, label, offset, hatch=None):
        df_ord = df.reindex(type_order)
        acc = df_ord["acc"].to_numpy(dtype=float)
        lo  = np.clip((df_ord["acc"] - df_ord["ci_lo"]).to_numpy(dtype=float), 0, None)
        hi  = np.clip((df_ord["ci_hi"] - df_ord["acc"]).to_numpy(dtype=float), 0, None)
        ax.bar(
            x + offset, acc, width=w,
            color=color, alpha=ALPHA_BAR, label=label, hatch=hatch,
            yerr=np.vstack([lo, hi]),
            error_kw={"elinewidth": 1.1, "capsize": 3, "ecolor": "#333333"},
            edgecolor="black", linewidth=LINEWIDTH_BAR,
        )

    _bars(ax, base_df, base_label, -w / 2, hatch=None)
    _bars(ax, ft_df,   ft_label,   +w / 2, hatch=HATCH_FINETUNED)

    # x-axis: show n from base_df (same test ids so counts match)
    base_ns = base_df.reindex(type_order)["n"].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{abbrev(qt)}\n(n={int(n)})" for qt, n in zip(type_order, base_ns)],
        rotation=35, ha="right", fontsize=9,
    )

    ax.axvline(len(type_order) - 1.5, color="#aaaaaa", linewidth=1, linestyle="--")
    ax.axhline(0.5, linestyle=":", linewidth=1.2, color="black", alpha=0.7)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("Accuracy (Wilson 95% CI)", fontsize=11)
    ax.set_xlabel("Query type", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",   required=True,
                    help="Baseline JSONL (full dataset OK — filtered to test ids)")
    ap.add_argument("--finetuned",  required=True,
                    help="Fine-tuned predictions JSONL")
    ap.add_argument("--test_split", default=None,
                    help="Test split JSONL used to filter the baseline. Omit to use all baseline records.")
    ap.add_argument("--out",        default="outputs/plots/base_vs_finetuned.png")
    ap.add_argument("--title",      default=None)
    ap.add_argument("--base_label", default=None,
                    help="Override label for baseline subplot")
    ap.add_argument("--ft_label",   default=None,
                    help="Override label for fine-tuned subplot")
    args = ap.parse_args()

    # Load baseline — optionally filter to test split
    base_rows_all = load_jsonl(args.baseline)
    if args.test_split:
        test_records = load_jsonl(args.test_split)
        test_qids = {r["question_id"] for r in test_records}
        print(f"Test split: {len(test_qids)} question_ids")
        base_rows = filter_to_test_ids(base_rows_all, test_qids)
        print(f"Baseline: {len(base_rows_all)} total → {len(base_rows)} on test set")
    else:
        base_rows = base_rows_all
        print(f"Baseline: {len(base_rows)} records (no split filter)")

    # Load fine-tuned predictions
    ft_rows = load_jsonl(args.finetuned)
    print(f"Fine-tuned: {len(ft_rows)} predictions")

    # Labels
    base_model = model_label_from_rows(base_rows_all, "baseline")
    ft_model   = model_label_from_rows(ft_rows, "fine-tuned")
    base_label = args.base_label or f"{base_model}\n(zero-shot)"
    ft_label   = args.ft_label   or f"{ft_model}\n(LoRA fine-tuned, test set)"
    title      = args.title      or f"{base_model}: zero-shot vs LoRA fine-tuned — per query type accuracy"

    # Color — same hue for both bars; fine-tuned gets hatch in plot_comparison()
    color = model_color(base_model)

    # Compute stats
    base_df = compute_per_type(base_rows)
    ft_df   = compute_per_type(ft_rows)

    # Print summary table
    print(f"\n{'Query type':<22} {'Base acc':>10} {'FT acc':>10} {'Delta':>8}")
    print("-" * 54)
    common = sorted(set(base_df.index) & set(ft_df.index))
    for qt in common:
        b = base_df.loc[qt, "acc"]
        f = ft_df.loc[qt, "acc"]
        sign = "+" if f - b >= 0 else ""
        print(f"  {qt:<20} {b:>10.4f} {f:>10.4f} {sign}{f-b:>7.4f}")

    plot_comparison(
        base_df=base_df,
        ft_df=ft_df,
        base_label=base_label,
        ft_label=ft_label,
        color=color,
        out_path=args.out,
        title=title,
    )

    # Save CSV
    csv_path = Path(args.out).with_suffix(".csv")
    merged = base_df.add_prefix("base_").join(ft_df.add_prefix("ft_"), how="outer")
    merged["delta_acc"] = merged["ft_acc"] - merged["base_acc"]
    merged.to_csv(csv_path)
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
