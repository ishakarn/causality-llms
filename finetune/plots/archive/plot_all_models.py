"""
Per-dataset multi-model comparison plot.

Auto-discovers all baseline and finetuned JSONL files for a dataset and plots
accuracy per query type for every model in a single grouped bar chart.

Usage:
    python finetune/plot_all_models.py --dataset cladder-v1-q-easy
    python finetune/plot_all_models.py --dataset cladder-v1-q-easy --out outputs/plots/newdata/cladder-v1-q-easy/all_models.png

    # Run all available datasets:
    for d in cladder-v1-q-easy cladder-v1-q-hard cladder-v1-q-anticommonsense; do
        python finetune/plot_all_models.py --dataset "$d"
    done
"""

import argparse
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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def model_label_from_rows(rows: List[Dict], fallback: str) -> str:
    if not rows:
        return fallback
    r = rows[0]
    mid = r.get("model_id_str") or r.get("model_name_or_path") or r.get("model") or ""
    return mid.split("/")[-1] if mid else fallback


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_models(dataset: str, outputs_root: Path) -> List[Dict]:
    """
    Returns list of dicts: {label, jsonl_path, is_finetuned, model_id}
    sorted: baseline models first, then finetuned, alphabetical within group.
    """
    found = []
    base_dir = outputs_root / dataset / "baseline"
    ft_dir   = outputs_root / dataset / "finetuned"

    for d in sorted(base_dir.glob("*")) if base_dir.exists() else []:
        jsonls = list(d.glob("*.jsonl"))
        if jsonls:
            found.append({"jsonl": jsonls[0], "is_finetuned": False, "run_id": d.name})

    for d in sorted(ft_dir.glob("*")) if ft_dir.exists() else []:
        jsonls = list(d.glob("*.jsonl"))
        if jsonls:
            found.append({"jsonl": jsonls[0], "is_finetuned": True, "run_id": d.name})

    return found


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_all_models(entries: List[Dict], dataset: str, out_path: Path) -> None:
    """
    entries: list of {label, df (pd.DataFrame), is_finetuned, color}
    """
    # Collect type_order from first entry
    all_types = set()
    for e in entries:
        all_types.update(e["df"].index)
    all_types.discard("overall")
    type_order = sorted(all_types, key=sort_key) + ["overall"]
    x = np.arange(len(type_order))

    n_models = len(entries)
    total_width = 0.80
    w = total_width / n_models
    offsets = np.linspace(-(total_width - w) / 2, (total_width - w) / 2, n_models)

    fig, ax = plt.subplots(figsize=(max(16, 2 * len(type_order)), 6.5))

    for i, e in enumerate(entries):
        df_ord = e["df"].reindex(type_order)
        acc = df_ord["acc"].to_numpy(dtype=float)
        lo  = np.clip((df_ord["acc"] - df_ord["ci_lo"]).to_numpy(dtype=float), 0, None)
        hi  = np.clip((df_ord["ci_hi"] - df_ord["acc"]).to_numpy(dtype=float), 0, None)
        ax.bar(
            x + offsets[i], acc, width=w * 0.92,
            color=e["color"], alpha=ALPHA_BAR,
            hatch=HATCH_FINETUNED if e["is_finetuned"] else None,
            label=e["label"],
            yerr=np.vstack([lo, hi]),
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#333333"},
            edgecolor="black", linewidth=LINEWIDTH_BAR,
        )

    # x-tick labels — use n from first baseline entry
    ref_df = entries[0]["df"].reindex(type_order)
    ns = ref_df["n"].fillna(0).to_numpy(dtype=int)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{abbrev(qt)}\n(n={int(n)})" for qt, n in zip(type_order, ns)],
        rotation=35, ha="right", fontsize=9,
    )

    ax.axvline(len(type_order) - 1.5, color="#aaaaaa", linewidth=1, linestyle="--")
    ax.axhline(0.5, linestyle=":", linewidth=1.2, color="black", alpha=0.7)
    ax.set_ylim(-0.02, 1.12)
    ax.set_ylabel("Accuracy (Wilson 95% CI)", fontsize=11)
    ax.set_xlabel("Query type", fontsize=10)
    ax.set_title(f"{dataset}: all models — per query type accuracy",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

# Model-id substrings → display name
_DISPLAY_NAMES = {
    "olmo3-7b-instruct": "OLMo-3-7B-Instruct",
    "qwen25-3b-instruct": "Qwen 2.5-3B-Instruct",
    "gpt-5-nano": "GPT-5-Nano",
    "gpt-oss-20b": "GPT-OSS-20B",
    "gpt54": "GPT-5.4",
}

def _display(run_id: str) -> str:
    low = run_id.lower()
    for k, v in _DISPLAY_NAMES.items():
        if k in low:
            return v
    return run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset name, e.g. cladder-v1-q-easy")
    ap.add_argument("--outputs_root", default="outputs",
                    help="Root of outputs directory (default: outputs)")
    ap.add_argument("--out", default=None,
                    help="Output PNG path. Default: outputs/plots/newdata/<dataset>/all_models.png")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    out_path = Path(args.out) if args.out else \
        Path(f"outputs/plots/newdata/{args.dataset}/all_models.png")

    raw_entries = discover_models(args.dataset, outputs_root)
    if not raw_entries:
        print(f"No JSONL files found for dataset '{args.dataset}' under {outputs_root}/")
        sys.exit(1)

    entries = []
    for e in raw_entries:
        rows = load_jsonl(e["jsonl"])
        model_str = model_label_from_rows(rows, e["run_id"])
        display = _display(e["run_id"])
        suffix = " (LoRA)" if e["is_finetuned"] else " (base)"
        label = display + suffix
        color = model_color(e["run_id"])
        df = compute_per_type(rows)
        entries.append({
            "label": label,
            "df": df,
            "is_finetuned": e["is_finetuned"],
            "color": color,
            "run_id": e["run_id"],
        })
        n = len(rows)
        acc_overall = df.loc["overall", "acc"] if "overall" in df.index else float("nan")
        print(f"  {'FT' if e['is_finetuned'] else 'BS'} | {label:<40} n={n:>6}  acc={acc_overall:.4f}")

    print(f"\nPlotting {len(entries)} model series…")
    plot_all_models(entries, args.dataset, out_path)

    # Save merged CSV
    csv_path = out_path.with_suffix(".csv")
    dfs = []
    for e in entries:
        tmp = e["df"][["n", "acc", "ci_lo", "ci_hi"]].copy()
        tmp.columns = [f"{e['run_id']}__{c}" for c in tmp.columns]
        dfs.append(tmp)
    merged = pd.concat(dfs, axis=1)
    merged.to_csv(csv_path)
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
