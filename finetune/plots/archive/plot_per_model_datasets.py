"""
Per-model cross-dataset comparison plots.

One plot per model. X-axis = query types. 4 bars per query type = 4 datasets.
Lets you see whether a model's per-query-type profile shifts across datasets.

Excluded datasets: balanced, commonsense (as requested).

Usage:
    python finetune/plot_per_model_datasets.py
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
from plot_style import ALPHA_BAR, LINEWIDTH_BAR, apply_defaults

apply_defaults()

# ── Config ────────────────────────────────────────────────────────────────────

DATASETS = [
    "cladder-v1-q-easy",
    "cladder-v1-q-hard",
    "cladder-v1-q-noncommonsense",
    "cladder-v1-q-anticommonsense",
]

DATASET_LABELS = {
    "cladder-v1-q-easy":           "Easy",
    "cladder-v1-q-hard":           "Hard",
    "cladder-v1-q-noncommonsense": "Non-commonsense",
    "cladder-v1-q-anticommonsense":"Anti-commonsense",
}

# Colorblind-safe dataset colors (Wong palette)
DATASET_COLORS = {
    "cladder-v1-q-easy":           "#0072b2",  # deep blue
    "cladder-v1-q-hard":           "#e69f00",  # amber
    "cladder-v1-q-noncommonsense": "#009e73",  # teal
    "cladder-v1-q-anticommonsense":"#d55e00",  # vermillion
}

# 6 model series: (plot title, jsonl resolver fn, hatch)
def _jsonl(outputs: Path, dataset: str, subdir: str, run_id: str,
           suffix: str = "") -> Path:
    return outputs / dataset / subdir / run_id / f"{run_id}{suffix}.jsonl"

MODELS = [
    {
        "title":   "GPT-4-0613 (zero-shot)",
        "out":     "gpt4-0613-baseline",
        "hatch":   None,
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "baseline", "gpt4-0613-baseline"),
    },
    {
        "title":   "GPT-5-Nano (zero-shot)",
        "out":     "gpt-5-nano-baseline",
        "hatch":   None,
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "baseline", "gpt-5-nano-baseline"),
    },
    {
        "title":   "GPT-OSS-20B (zero-shot)",
        "out":     "gpt-oss-20b-baseline",
        "hatch":   None,
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "baseline", "gpt-oss-20b-baseline"),
    },
    {
        "title":   "GPT-5.4 (zero-shot)",
        "out":     "gpt54-baseline",
        "hatch":   None,
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "baseline", "gpt54-baseline"),
    },
    {
        "title":   "OLMo-3-7B-Instruct (base, zero-shot)",
        "out":     "olmo3-7b-instruct-baseline",
        "hatch":   None,
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "baseline", "olmo3-7b-instruct-baseline"),
    },
    {
        "title":   "OLMo-3-7B-Instruct (LoRA fine-tuned)",
        "out":     "olmo3-7b-instruct-lora",
        "hatch":   "////",
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "finetuned", "olmo3-7b-instruct-lora",
            f"__{ds}"),
    },
    {
        "title":   "Qwen 2.5-3B-Instruct (base, zero-shot)",
        "out":     "qwen25-3b-instruct-baseline",
        "hatch":   None,
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "baseline", "qwen25-3b-instruct-baseline"),
    },
    {
        "title":   "Qwen 2.5-3B-Instruct (LoRA fine-tuned)",
        "out":     "qwen25-3b-instruct-lora",
        "hatch":   "////",
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "finetuned", "qwen25-3b-instruct-lora",
            f"__{ds}"),
    },
    {
        "title":   "OLMo-3.1-32B-Instruct (base, zero-shot)",
        "out":     "olmo3-32b-instruct-baseline",
        "hatch":   None,
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "baseline", "olmo3-32b-instruct-baseline"),
    },
    {
        "title":   "OLMo-3.1-32B-Instruct (LoRA fine-tuned)",
        "out":     "olmo3-32b-instruct-lora",
        "hatch":   "////",
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "finetuned", "olmo3-32b-instruct-lora",
            f"__{ds}"),
    },
    {
        "title":   "Llama-3.1-8B-Instruct (base, zero-shot)",
        "out":     "llama31-8b-instruct-baseline",
        "hatch":   None,
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "baseline", "llama31-8b-instruct-baseline"),
    },
    {
        "title":   "Llama-3.1-8B-Instruct (LoRA fine-tuned)",
        "out":     "llama31-8b-instruct-lora",
        "hatch":   "////",
        "resolve": lambda outputs, ds: _jsonl(
            outputs, ds, "finetuned", "llama31-8b-instruct-lora",
            f"__{ds}"),
    },
]

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


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_model(model_cfg: Dict, dataset_dfs: Dict[str, pd.DataFrame],
               out_path: Path) -> None:
    all_types = set()
    for df in dataset_dfs.values():
        all_types.update(df.index)
    all_types.discard("overall")
    type_order = sorted(all_types, key=sort_key) + ["overall"]

    n_ds = len(DATASETS)
    x = np.arange(len(type_order))
    total_w = 0.80
    w = total_w / n_ds
    offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n_ds)

    fig, ax = plt.subplots(figsize=(15, 6.5))

    for i, ds in enumerate(DATASETS):
        df = dataset_dfs.get(ds)
        if df is None:
            continue
        df_ord = df.reindex(type_order)
        acc = df_ord["acc"].to_numpy(dtype=float)
        lo  = np.clip((df_ord["acc"] - df_ord["ci_lo"]).to_numpy(dtype=float), 0, None)
        hi  = np.clip((df_ord["ci_hi"] - df_ord["acc"]).to_numpy(dtype=float), 0, None)

        ax.bar(
            x + offsets[i], acc, width=w * 0.92,
            color=DATASET_COLORS[ds],
            hatch=model_cfg["hatch"],
            alpha=ALPHA_BAR,
            label=DATASET_LABELS[ds],
            yerr=np.vstack([lo, hi]),
            error_kw={"elinewidth": 0.9, "capsize": 2.5, "ecolor": "#333333"},
            edgecolor="black", linewidth=LINEWIDTH_BAR,
        )

    # x-tick labels — use n from first available dataset
    ref_df = next(iter(dataset_dfs.values())).reindex(type_order)
    ns = ref_df["n"].fillna(0).to_numpy(dtype=int)
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
    ax.set_title(f"{model_cfg['title']} — accuracy per query type across datasets",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right", title="Dataset")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    outputs = Path("outputs")
    out_dir = outputs / "plots" / "per_model_datasets"

    for model_cfg in MODELS:
        print(f"\n=== {model_cfg['title']} ===")

        dataset_dfs = {}
        for ds in DATASETS:
            jsonl = model_cfg["resolve"](outputs, ds)
            if not jsonl.exists():
                print(f"  [skip] {ds} — {jsonl} not found")
                continue
            rows = load_jsonl(jsonl)
            dataset_dfs[ds] = compute_per_type(rows)
            acc = dataset_dfs[ds].loc["overall", "acc"]
            print(f"  {DATASET_LABELS[ds]:<20} overall={acc:.4f}")

        if not dataset_dfs:
            print("  No data found, skipping.")
            continue

        out_path = out_dir / f"{model_cfg['out']}__datasets.png"
        plot_model(model_cfg, dataset_dfs, out_path)


if __name__ == "__main__":
    main()
