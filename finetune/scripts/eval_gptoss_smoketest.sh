#!/bin/bash
# =============================================================================
# Smoke-test for GPT-OSS-20B think + nothink modes.
#
# Runs BOTH modes on the first 10 records of cladder-v1-q-easy only.
# Total compute: ~5 minutes. Use this to verify:
#   1. Model loads and generates output
#   2. think mode stores full reasoning trace in raw_response
#   3. nothink mode appends the user suffix and produces short answers
#   4. Scoring and JSONL output are well-formed
#
# Results saved to:
#   outputs/cladder-v1-q-easy/baseline/gpt-oss-20b-think-test/
#   outputs/cladder-v1-q-easy/baseline/gpt-oss-20b-nothink-test/
#
# Submit:
#   sbatch finetune/eval_gptoss_smoketest.sh
# =============================================================================

#SBATCH --job-name=cladder-gptoss-smoke
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=finetune/slurm_logs/%j_gptoss_smoke.out
#SBATCH --error=finetune/slurm_logs/%j_gptoss_smoke.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

set -euo pipefail

WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

WS=/scratch/workspace/ikarn_umass_edu-olmo_cladder_cache
mkdir -p "$WS/.cache/huggingface" "$WS/.cache/torch"

export HF_HOME="$WS/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$WS/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="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"

module load conda/latest
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$CONDA_PREFIX")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cladder_olmo
echo "[env] python: $(which python)"

pip install -q vllm

echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

mkdir -p finetune/slurm_logs

MODEL="openai/gpt-oss-20b"
DATA="data/cladder-v1-q-easy.json"
N=10  # records per mode

echo "=== Smoke test: $MODEL — $N records, both modes ==="
echo ""

# ── Think mode ────────────────────────────────────────────────────────────────
echo "--- think mode ---"
THINK_OUT="outputs/cladder-v1-q-easy/baseline/gpt-oss-20b-think-test"
mkdir -p "$THINK_OUT"

python finetune/infer_vllm.py \
    --model         "$MODEL" \
    --data_file     "$DATA" \
    --out_jsonl     "${THINK_OUT}/gpt-oss-20b-think-test.jsonl" \
    --run_id        "gpt-oss-20b-think-test" \
    --max_model_len 5120 \
    --thinking_mode think \
    --max_samples   $N \
    --overwrite

echo ""
echo "--- think: first 3 raw_response lengths (should be long CoT) ---"
python - <<'EOF'
import json
path = "outputs/cladder-v1-q-easy/baseline/gpt-oss-20b-think-test/gpt-oss-20b-think-test.jsonl"
with open(path) as f:
    for i, line in enumerate(f):
        if i >= 3: break
        r = json.loads(line)
        resp = r.get("raw_response", "")
        print(f"  [{i}] pred={r['pred']}  gold={r['gold']}  response_len={len(resp)}  preview={resp[:80]!r}")
EOF

# ── Nothink mode ──────────────────────────────────────────────────────────────
echo ""
echo "--- nothink mode ---"
NOTHINK_OUT="outputs/cladder-v1-q-easy/baseline/gpt-oss-20b-nothink-test"
mkdir -p "$NOTHINK_OUT"

python finetune/infer_vllm.py \
    --model         "$MODEL" \
    --data_file     "$DATA" \
    --out_jsonl     "${NOTHINK_OUT}/gpt-oss-20b-nothink-test.jsonl" \
    --run_id        "gpt-oss-20b-nothink-test" \
    --max_model_len 2048 \
    --thinking_mode nothink \
    --max_samples   $N \
    --overwrite

echo ""
echo "--- nothink: first 3 raw_response lengths (should be short) ---"
python - <<'EOF'
import json
path = "outputs/cladder-v1-q-easy/baseline/gpt-oss-20b-nothink-test/gpt-oss-20b-nothink-test.jsonl"
with open(path) as f:
    for i, line in enumerate(f):
        if i >= 3: break
        r = json.loads(line)
        resp = r.get("raw_response", "")
        print(f"  [{i}] pred={r['pred']}  gold={r['gold']}  response_len={len(resp)}  preview={resp[:80]!r}")
EOF

echo ""
echo "=== Smoke test complete ==="
echo "  think   → $THINK_OUT"
echo "  nothink → $NOTHINK_OUT"
