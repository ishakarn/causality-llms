"""
Compile all intervention + baseline results into a single CSV.

Columns: model, condition_type, intervention, split, n, correct, invalid,
         acc_all, acc_valid_only, delta_acc_all (vs. pooled baseline)

Output: outputs/results_summary.csv

Usage:
    cd <repo-root>
    python finetune/compile_results.py
"""

import json
import csv
from collections import defaultdict
from pathlib import Path

OUTPUTS     = Path("outputs")
RESULTS_DIR = Path("finetune/eval_results/interventions")
OUT_PATH    = Path("outputs/results_summary.csv")

SPLITS = ["easy", "hard", "anticommonsense", "noncommonsense"]

CONDITIONS = {
    "qwen3b_n2000_lora":  ("Qwen2.5-3B LoRA",   "finetuned", "qwen25-3b-instruct-lora"),
    "llama8b_n2000_lora": ("Llama-3.1-8B LoRA", "finetuned", "llama31-8b-instruct-lora"),
    "olmo32b_n2000_lora": ("OLMo-3.1-32B LoRA", "finetuned", "olmo3-32b-instruct-lora"),
    "gptoss_base":        ("GPT-OSS-20B",        "baseline",  "gpt-oss-20b-baseline"),
    "gpt5nano_base":      ("GPT-5-Nano",         "baseline",  "gpt-5-nano-baseline"),
    "gpt55_base":         ("GPT-5.5",            "baseline",  "gpt-5.5-baseline"),
}

INTERVENTIONS = [
    "67_word_replace",
    "68_number_replace",
    "81_story_swap",
    "86_nonsense_replace",
    "94_drop_background",
    "96_probability_expander",
    "100_drop_graph_structure",
]


def load_summary(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pooled_row(summaries):
    """Weighted pool a list of summary dicts into one aggregate dict."""
    total_n = total_c = total_inv = 0
    for s in summaries:
        if s is None:
            continue
        total_n   += s["n"]
        total_c   += s["correct"]
        total_inv += s.get("invalid", 0)
    if total_n == 0:
        return None
    valid_n = total_n - total_inv
    return {
        "n":             total_n,
        "correct":       total_c,
        "invalid":       total_inv,
        "acc_all":       total_c / total_n,
        "acc_valid_only": total_c / valid_n if valid_n > 0 else None,
    }


rows = []

for cond_key, (label, cond_type, run_id) in CONDITIONS.items():
    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_by_split = {}
    for split in SPLITS:
        p = OUTPUTS / f"cladder-v1-q-{split}" / cond_type / run_id / "score" / "summary.json"
        baseline_by_split[split] = load_summary(p)

    baseline_pooled = pooled_row(list(baseline_by_split.values()))

    for split in SPLITS:
        s = baseline_by_split[split]
        if s is None:
            continue
        rows.append({
            "model":          label,
            "model_key":      cond_key,
            "condition_type": cond_type,
            "intervention":   "baseline",
            "split":          split,
            "n":              s["n"],
            "correct":        s["correct"],
            "invalid":        s.get("invalid", 0),
            "acc_all":        round(s["acc_all"], 4),
            "acc_valid_only": round(s["acc_valid_only"], 4),
            "delta_acc_all":  "",
        })

    if baseline_pooled:
        rows.append({
            "model":          label,
            "model_key":      cond_key,
            "condition_type": cond_type,
            "intervention":   "baseline",
            "split":          "pooled",
            "n":              baseline_pooled["n"],
            "correct":        baseline_pooled["correct"],
            "invalid":        baseline_pooled["invalid"],
            "acc_all":        round(baseline_pooled["acc_all"], 4),
            "acc_valid_only": round(baseline_pooled["acc_valid_only"], 4) if baseline_pooled["acc_valid_only"] else "",
            "delta_acc_all":  "",
        })

    # ── Interventions ─────────────────────────────────────────────────────────
    for interv in INTERVENTIONS:
        interv_by_split = {}
        for split in SPLITS:
            p = RESULTS_DIR / cond_key / interv / split / "score" / "summary.json"
            interv_by_split[split] = load_summary(p)

        interv_pooled = pooled_row(list(interv_by_split.values()))

        for split in SPLITS:
            s = interv_by_split[split]
            b = baseline_by_split[split]
            if s is None:
                continue
            delta = round(s["acc_all"] - b["acc_all"], 4) if b else ""
            rows.append({
                "model":          label,
                "model_key":      cond_key,
                "condition_type": cond_type,
                "intervention":   interv,
                "split":          split,
                "n":              s["n"],
                "correct":        s["correct"],
                "invalid":        s.get("invalid", 0),
                "acc_all":        round(s["acc_all"], 4),
                "acc_valid_only": round(s["acc_valid_only"], 4),
                "delta_acc_all":  delta,
            })

        if interv_pooled:
            b_acc = baseline_pooled["acc_all"] if baseline_pooled else None
            delta = round(interv_pooled["acc_all"] - b_acc, 4) if b_acc else ""
            rows.append({
                "model":          label,
                "model_key":      cond_key,
                "condition_type": cond_type,
                "intervention":   interv,
                "split":          "pooled",
                "n":              interv_pooled["n"],
                "correct":        interv_pooled["correct"],
                "invalid":        interv_pooled["invalid"],
                "acc_all":        round(interv_pooled["acc_all"], 4),
                "acc_valid_only": round(interv_pooled["acc_valid_only"], 4) if interv_pooled["acc_valid_only"] else "",
                "delta_acc_all":  delta,
            })

# ── Write CSV ────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fieldnames = ["model", "model_key", "condition_type", "intervention", "split",
              "n", "correct", "invalid", "acc_all", "acc_valid_only", "delta_acc_all"]

with open(OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows → {OUT_PATH}")

# ── Quick summary table to stdout ────────────────────────────────────────────
print()
print(f"{'Model':<22} {'Intervention':<55} {'Pooled Acc':>10} {'Delta':>8}")
print("-" * 100)
for r in rows:
    if r["split"] == "pooled":
        delta = f"{r['delta_acc_all']:+.1%}" if r["delta_acc_all"] != "" else "   —"
        acc   = f"{r['acc_all']:.1%}"
        print(f"{r['model']:<22} {r['intervention']:<55} {acc:>10} {delta:>8}")
