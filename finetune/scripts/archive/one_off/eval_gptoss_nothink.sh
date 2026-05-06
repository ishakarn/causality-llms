#!/bin/bash
# =============================================================================
# SLURM job: GPT-OSS-20B with thinking DISABLED across all 6 cladder-v1 datasets.
#
# Two-pronged suppression of chain-of-thought:
#   1. enable_thinking=False passed to apply_chat_template (may be silently
#      ignored, but included as belt-and-suspenders)
#   2. USER_SUFFIX_REASONING appended to the user message:
#      "Respond with one word only — yes or no:"
#      This is the actual enforcement — forces the answer token to come next.
#
# Because CoT is suppressed, 512 output tokens is more than enough.
#
# Outputs saved to:
#   outputs/{DATASET}/baseline/gpt-oss-20b-nothink/
#
# Submit:
#   sbatch finetune/eval_gptoss_nothink.sh
# =============================================================================

#SBATCH --job-name=cladder-gptoss-nothink
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=finetune/slurm_logs/%j_gptoss_nothink.out
#SBATCH --error=finetune/slurm_logs/%j_gptoss_nothink.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

set -euo pipefail

WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

# ── Scratch cache ─────────────────────────────────────────────────────────────
WS=/scratch/workspace/ikarn_umass_edu-olmo_cladder_cache
mkdir -p "$WS/.cache/huggingface" "$WS/.cache/torch"

export HF_HOME="$WS/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$WS/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="${HF_TOKEN}"

# ── Conda ─────────────────────────────────────────────────────────────────────
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

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL="openai/gpt-oss-20b"
RUN_ID="gpt-oss-20b-nothink"

# max_model_len: prompt (~300 tok) + 512 answer tokens (no CoT) = 1024 with buffer
MAX_MODEL_LEN=2048

echo "=== Model: $MODEL  mode: nothink ==="
echo ""

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS=(
    data/cladder-v1-q-easy.json
    data/cladder-v1-q-hard.json
    data/cladder-v1-q-anticommonsense.json
    data/cladder-v1-q-balanced.json
    data/cladder-v1-q-commonsense.json
    data/cladder-v1-q-noncommonsense.json
)

# ── Loop ──────────────────────────────────────────────────────────────────────
for DATA in "${DATASETS[@]}"; do
    DATASET=$(basename "$DATA" .json)
    echo ""
    echo "========================================================"
    echo "  Dataset: $DATASET"
    echo "========================================================"

    BASE_OUT="outputs/${DATASET}/baseline/${RUN_ID}"
    mkdir -p "$BASE_OUT"

    JSONL="${BASE_OUT}/${RUN_ID}.jsonl"

    # ── Step 1: Inference ─────────────────────────────────────────────────────
    echo "[1/2] Inference (thinking disabled, max_tokens=512)…"
    python finetune/infer_vllm.py \
        --model          "$MODEL" \
        --data_file      "$DATA" \
        --out_jsonl      "$JSONL" \
        --run_id         "$RUN_ID" \
        --max_model_len  "$MAX_MODEL_LEN" \
        --thinking_mode  nothink \
        --overwrite

    # ── Step 2: Score ─────────────────────────────────────────────────────────
    echo "[2/2] Scoring…"
    python cladder_score_yesno.py \
        --pred_jsonl "$JSONL" \
        --out_dir    "${BASE_OUT}/score"

    echo "  → done: $DATASET"
done

echo ""
echo "=== All datasets complete ==="
echo "  Results: outputs/*/baseline/${RUN_ID}/"
