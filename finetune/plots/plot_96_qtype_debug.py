"""
Debug plot: per-query-type accuracy change for 96_probability_expander.
Shows baseline vs. intervention accuracy per query type per model.
Output: outputs/plots/paper-ready/96_qtype_debug.{png,pdf}
"""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

OUTPUTS     = Path("outputs")
RESULTS_DIR = Path("finetune/eval_results/interventions")
OUT_DIR     = Path("outputs/plots/paper-ready")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTERV = "96_probability_expander"
SPLITS = ["easy", "hard", "anticommonsense", "noncommonsense"]

CONDITIONS = {
    "qwen3b_n2000_lora":  ("Qwen2.5-3B LoRA",   "finetuned", "qwen25-3b-instruct-lora",  "#4C72B0", "D"),
    "llama8b_n2000_lora": ("Llama-3.1-8B LoRA", "finetuned", "llama31-8b-instruct-lora", "#56B4E9", "D"),
    "olmo32b_n2000_lora": ("OLMo-3.1-32B LoRA", "finetuned", "olmo3-32b-instruct-lora",  "#DD8452", "D"),
    "gptoss_base":        ("GPT-OSS-20B",        "baseline",  "gpt-oss-20b-baseline",      "#55A868", "o"),
    "gpt5nano_base":      ("GPT-5-Nano",         "baseline",  "gpt-5-nano-baseline",       "#9467BD", "o"),
}

QTYPE_ORDER = ["marginal", "correlation", "ate", "ett", "nie", "nde", "exp_away"]

USE_VALID_ONLY = {"gpt5nano_base"}

def acc_key(cond_key):
    return "acc_valid_only" if cond_key in USE_VALID_ONLY else "acc_all"

def load_qtype_csv(path, cond_key):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    col = acc_key(cond_key)
    return {row["query_type"]: (row[col], row["n"]) for _, row in df.iterrows()}

def pool_qtype(paths_and_cond):
    """Weighted pool per_query_type.csv across splits."""
    pooled = defaultdict(lambda: [0, 0])
    for path, cond_key in paths_and_cond:
        for qt, (acc, n) in load_qtype_csv(path, cond_key).items():
            pooled[qt][0] += round(acc * n)
            pooled[qt][1] += n
    return {qt: v[0] / v[1] for qt, v in pooled.items() if v[1] > 0}

def get_baseline_qtype(cond_key):
    _, subdir, run_id, *_ = CONDITIONS[cond_key]
    pairs = []
    for split in SPLITS:
        p = OUTPUTS / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "per_query_type.csv"
        pairs.append((p, cond_key))
    return pool_qtype(pairs)

def get_interv_qtype(cond_key):
    pairs = []
    for split in SPLITS:
        p = RESULTS_DIR / cond_key / INTERV / split / "score" / "per_query_type.csv"
        pairs.append((p, cond_key))
    return pool_qtype(pairs)


# ── Plot ──────────────────────────────────────────────────────────────────────
qtypes = [qt for qt in QTYPE_ORDER]
n_qt   = len(qtypes)
cond_keys = list(CONDITIONS.keys())
n_models  = len(cond_keys)

x = np.arange(n_qt)
width = 0.13
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5), sharey=False)

# Panel 1: absolute accuracy (baseline vs intervention)
ax = axes[0]
ax.set_title("Accuracy by query type", fontsize=9, fontweight="bold")

for ci, cond_key in enumerate(cond_keys):
    label, _, _, color, marker = CONDITIONS[cond_key]
    b_map = get_baseline_qtype(cond_key)
    v_map = get_interv_qtype(cond_key)
    for qi, qt in enumerate(qtypes):
        b = b_map.get(qt)
        v = v_map.get(qt)
        if b is None or v is None:
            continue
        xpos = x[qi] + offsets[ci]
        ax.scatter(xpos, b, color="white", edgecolors=color, s=28, linewidths=1.3, zorder=4)
        ax.scatter(xpos, v, color=color, marker=marker, s=28, zorder=5)
        ax.plot([xpos, xpos], [b, v], color=color, linewidth=0.9, alpha=0.7, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(qtypes, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Accuracy", fontsize=8)
ax.set_ylim(0.4, 1.05)
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# Panel 2: accuracy change (intervention - baseline)
ax = axes[1]
ax.set_title("Accuracy change (96 − baseline)", fontsize=9, fontweight="bold")

for ci, cond_key in enumerate(cond_keys):
    label, _, _, color, marker = CONDITIONS[cond_key]
    b_map = get_baseline_qtype(cond_key)
    v_map = get_interv_qtype(cond_key)
    for qi, qt in enumerate(qtypes):
        b = b_map.get(qt)
        v = v_map.get(qt)
        if b is None or v is None:
            continue
        xpos = x[qi] + offsets[ci]
        ax.scatter(xpos, v - b, color=color, marker=marker, s=28, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(qtypes, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Δ Accuracy", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))

# Shared legend
handles = []
for cond_key in cond_keys:
    label, _, _, color, marker = CONDITIONS[cond_key]
    handles.append(mlines.Line2D([], [], color=color, marker=marker,
                                 markersize=5.5, linewidth=0, label=label))
handles.append(mlines.Line2D([], [], color="gray", marker="o", markersize=5.5,
                              markerfacecolor="white", markeredgecolor="gray",
                              linewidth=0, label="open = baseline"))
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.08), framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout(w_pad=1.5)
for ext in ("png", "pdf"):
    out = OUT_DIR / f"96_qtype_debug.{ext}"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote: {out}")
plt.close()
