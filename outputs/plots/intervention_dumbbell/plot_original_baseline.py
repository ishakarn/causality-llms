"""
One open circle per model at their baseline accuracy — represents the
'original query' row in the intervention dumbbell figure.
Output: outputs/plots/intervention_dumbbell/original_baseline.png
"""

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUTS     = Path("outputs")
RESULTS_DIR = Path("finetune/eval_results/interventions")
OUT_DIR     = Path("outputs/plots/intervention_dumbbell")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = OrderedDict([
    ("qwen3b_n2000_lora",  ("Qwen2.5-3B LoRA",   "finetuned", "qwen25-3b-instruct-lora",    "#4C72B0", "D", "dashed")),
    ("olmo32b_n2000_lora", ("OLMo-3.1-32B LoRA", "finetuned", "olmo3-32b-instruct-lora",    "#DD8452", "D", "dashed")),
    ("gptoss_base",        ("GPT-OSS-20B base",  "baseline",  "gpt-oss-20b-baseline",        "#55A868", "o", "solid")),
    ("gpt5nano_base",      ("GPT-5-Nano base",   "baseline",  "gpt-5-nano-nano496-baseline", "#9467BD", "o", "solid")),
])

DEFAULT_SPLITS = ["easy", "hard", "anticommonsense"]
MODEL_OFFSETS  = [-0.27, -0.09, 0.09, 0.27]
USE_VALID_ONLY = {"gpt5nano_base"}
CHANCE_COLOR   = "#888888"


def _acc_key(cond_key):
    return "acc_valid_only" if cond_key in USE_VALID_ONLY else "acc_all"


def _load_baseline(cond_key, split):
    _, subdir, run_id, *_ = CONDITIONS[cond_key]
    p = OUTPUTS / f"cladder-v1-q-{split}" / subdir / run_id / "score" / "summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    n = d["n"] - d.get("invalid", 0) if cond_key in USE_VALID_ONLY else d["n"]
    return d[_acc_key(cond_key)], n


def get_baseline(cond_key):
    results = [r for s in DEFAULT_SPLITS if (r := _load_baseline(cond_key, s))]
    if not results:
        return None
    total_n = sum(n for _, n in results)
    return sum(a * n for a, n in results) / total_n


fig, ax = plt.subplots(figsize=(2.4, 0.82))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

ax.set_xlim(0.47, 1.02)
ax.set_ylim(-0.55, 0.55)

for ci, cond_key in enumerate(CONDITIONS):
    _, _, _, color, _, _ = CONDITIONS[cond_key]
    y = MODEL_OFFSETS[ci]
    b = get_baseline(cond_key)
    if b is None:
        continue
    ax.scatter(b, y, s=38, color="white", edgecolors=color, linewidths=1.6, zorder=4)

ax.axvline(0.5, color=CHANCE_COLOR, linewidth=0.9, linestyle=":", zorder=1)

ax.set_yticks([])
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(["50", "60", "70", "80", "90", "100"], fontsize=7)
ax.tick_params(axis="x", length=2.5, pad=1.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.3, zorder=0)

plt.tight_layout(pad=0.25)
out = OUT_DIR / "original_baseline.png"
fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)
print(f"Wrote: {out}")
