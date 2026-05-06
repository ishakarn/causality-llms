#!/bin/bash
# Re-score polarity_flip using original gold answers (text is unchanged in the
# intervened files — only the label was flipped, so model accuracy should be
# measured against the ORIGINAL answer, not the flipped one).

set -euo pipefail

WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

module load conda/latest
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$CONDA_PREFIX")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cladder_olmo

INTERVENTION="35_polarity_flip"
SPLITS=(easy hard anticommonsense noncommonsense)
CONDITIONS=(qwen3b_base qwen3b_n2000_lora olmo32b_base olmo32b_n2000_lora gptoss_base)

# Build original-gold lookup per split
python3 - <<'PYEOF'
import json, csv, os
from pathlib import Path

WORKDIR = Path(".")
INTERVENTION = "35_polarity_flip"
SPLITS = ["easy", "hard", "anticommonsense", "noncommonsense"]
CONDITIONS = ["qwen3b_base", "qwen3b_n2000_lora", "olmo32b_base", "olmo32b_n2000_lora", "gptoss_base"]

# Original data files (before any intervention)
ORIG_DATA = {
    "easy":             "data/cladder-v1-q-easy.json",
    "hard":             "data/cladder-v1-q-hard.json",
    "anticommonsense":  "data/cladder-v1-q-anticommonsense.json",
    "noncommonsense":   "data/cladder-v1-q-noncommonsense.json",
}

def load_orig(split):
    p = WORKDIR / ORIG_DATA[split]
    if not p.exists():
        # fallback: check for merged file
        print(f"  WARNING: {p} not found, trying data/queries_{split}.json")
        return {}
    d = json.loads(p.read_text())
    if isinstance(d, list):
        return {str(r["question_id"]): r["answer"] for r in d}
    if "queries" in d:
        return {str(r["question_id"]): r["answer"] for r in d["queries"]}
    return {}

for split in SPLITS:
    orig_gold = load_orig(split)
    print(f"\n[{split}] loaded {len(orig_gold)} original gold answers")

    for cond in CONDITIONS:
        base = WORKDIR / "finetune/eval_results/interventions" / cond / INTERVENTION / split
        jsonl_name = f"{cond}__{INTERVENTION}__{split}.jsonl"
        jsonl_path = base / jsonl_name
        score_dir  = base / "score"

        if not jsonl_path.exists():
            print(f"  [{cond}] MISSING jsonl — skipping")
            continue

        # Read existing records, swap gold
        records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        n_fixed = 0
        for r in records:
            qid = str(r.get("question_id", ""))
            orig = orig_gold.get(qid)
            if orig and r["gold"] != orig:
                r["gold"] = orig
                n_fixed += 1

        # Write a corrected temp jsonl for scoring (same path — scorer reads it)
        tmp = base / f"_rescored_{jsonl_name}"
        tmp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        print(f"  [{cond}] fixed {n_fixed}/{len(records)} gold labels → {tmp.name}")

        # Delete old score dir so scorer rewrites it
        import shutil
        if score_dir.exists():
            shutil.rmtree(score_dir)

        # Score against corrected gold
        import subprocess
        result = subprocess.run(
            ["python", "cladder_score_yesno.py",
             "--pred_jsonl", str(tmp),
             "--out_dir",    str(score_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [{cond}] SCORER ERROR:\n{result.stderr[:400]}")
            continue

        # Read back acc
        summary = json.loads((score_dir / "summary.json").read_text())
        print(f"  [{cond}] acc={summary['acc_all']:.3f}  (was ~{1-summary['acc_all']:.3f} with flipped gold)")

        # Clean up temp file
        tmp.unlink()

print("\nDone re-scoring polarity_flip.")
PYEOF
