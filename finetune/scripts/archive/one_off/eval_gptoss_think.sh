#!/bin/bash
# =============================================================================
# SLURM job: GPT-OSS-20B with thinking ENABLED across all 6 cladder-v1 datasets.
#
# Thinking is on by default for this model; we pass --thinking_mode think to
# set enable_thinking=True explicitly and allocate enough tokens (4096) to
# capture the full reasoning trace.  The complete CoT is stored in raw_response.
#
# Outputs saved to:
#   outputs/{DATASET}/baseline/gpt-oss-20b-think/
#
# Submit:
#   sbatch finetune/eval_gptoss_think.sh
# =============================================================================

#SBATCH --job-name=cladder-gptoss-think
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=finetune/slurm_logs/%j_gptoss_think.out
#SBATCH --error=finetune/slurm_logs/%j_gptoss_think.err
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
RUN_ID="gpt-oss-20b-think"

# max_model_len: CLADDER prompt (~300 tok) + 4096 CoT output + buffer = 5120
MAX_MODEL_LEN=5120

echo "=== Model: $MODEL  mode: think ==="
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
    echo "[1/2] Inference (thinking enabled, max_tokens=4096)…"
    python finetune/infer_vllm.py \
        --model          "$MODEL" \
        --data_file      "$DATA" \
        --out_jsonl      "$JSONL" \
        --run_id         "$RUN_ID" \
        --max_model_len  "$MAX_MODEL_LEN" \
        --thinking_mode  think \
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
