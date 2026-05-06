#!/bin/bash
# =============================================================================
# SLURM job: LoRA fine-tuning on CLADDER (generic — any model).
#
# Select a model by setting CONFIG before submitting:
#
#   sbatch finetune/scripts/finetune.sh                                         # OLMo instruct (default)
#   CONFIG=finetune/configs/olmo3-7b-think.yaml     sbatch --export=ALL finetune/scripts/finetune.sh
#   CONFIG=finetune/configs/qwen25-3b-instruct.yaml sbatch --export=ALL finetune/scripts/finetune.sh
#
# Resume from a checkpoint:
#   CONFIG=finetune/configs/qwen25-3b-instruct.yaml \
#   RESUME=finetune/checkpoints/qwen25-3b-instruct-lora/checkpoint-10 \
#     sbatch --export=ALL finetune/scripts/finetune.sh
#
# All models share the same output dirs (differentiated by run_name in config):
#   HF model downloads  → scratch workspace (never home dir)
#   LoRA checkpoints    → finetune/checkpoints/<run_name>/
#   Eval results        → finetune/eval_results/<run_name>/
# =============================================================================

#SBATCH --job-name=cladder-lora-ft
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=finetune/slurm_logs/%j_finetune.out
#SBATCH --error=finetune/slurm_logs/%j_finetune.err
#SBATCH --mail-type=BEGIN,END,FAIL   # email on job start, finish, or crash
#SBATCH --mail-user=ikarn@umass.edu  # ← confirm this is your address

set -euo pipefail

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

# ── Cache: all large files go to scratch, NOT home dir ───────────────────────
# Mirrors the variables in set_scratch_cache.sh.
WS=/scratch/workspace/ikarn_umass_edu-olmo_cladder_cache
mkdir -p "$WS/.cache/huggingface" "$WS/.cache/torch" "$WS/.cache/pip"

export HF_HOME="$WS/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$WS/.cache/torch"
export PIP_CACHE_DIR="$WS/.cache/pip"
export TOKENIZERS_PARALLELISM=false    # suppress tokenizer fork warning
export HF_TOKEN="${HF_TOKEN}"

echo "[cache] HF_HOME=$HF_HOME"
echo "[cache] TORCH_HOME=$TORCH_HOME"

# ── Conda environment ─────────────────────────────────────────────────────────
# `conda activate` needs the shell function that conda init installs.
# Sourcing the conda.sh hook is the reliable way to do this in a SLURM script.
module load conda/latest
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$CONDA_PREFIX")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cladder_olmo
echo "[env] python: $(which python)"
echo "[env] torch: $(python -c 'import torch; print(torch.__version__)')"

# ── Install fine-tuning dependencies (safe to re-run; no-ops if installed) ───
echo "[deps] checking peft / accelerate / datasets …"
pip install -q peft accelerate datasets

# ── GPU diagnostics ───────────────────────────────────────────────────────────
echo ""
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Config: read model_id and run_name from the yaml ─────────────────────────
CONFIG="${CONFIG:-finetune/configs/olmo3-7b-instruct.yaml}"
MODEL=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model_id'])")
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('run_name','lora'))")

echo "=== Config ==="
echo "  CONFIG   = $CONFIG"
echo "  MODEL    = $MODEL"
echo "  RUN_NAME = $RUN_NAME"
echo ""

mkdir -p finetune/slurm_logs finetune/checkpoints finetune/eval_results

# ── Step 1: Stratified data split ─────────────────────────────────────────────
# Writes finetune/splits/{train,val,test}.jsonl (80/10/10, stratified by query_type).
# Skipped if splits already exist so re-runs are fast.
if [ ! -f "finetune/splits/train.jsonl" ]; then
    echo "=== Step 1: prepare splits ==="
    python finetune/prepare_data.py \
        --queries_json data/queries_easy.json \
        --out_dir      finetune/splits \
        --train_frac   0.8 \
        --val_frac     0.1 \
        --seed         42
else
    echo "=== Step 1: splits already exist, skipping ==="
fi

# ── Step 2: LoRA fine-tuning ──────────────────────────────────────────────────
echo ""
echo "=== Step 2: train ==="

RESUME_ARG=""
# Auto-detect latest checkpoint if RESUME not explicitly set.
# This makes resubmitting the same sbatch command safe after preemption or
# a time-limit hit — the job picks up from the last completed epoch automatically.
if [ -n "${RESUME:-}" ]; then
    echo "  Resuming from (explicit): $RESUME"
    RESUME_ARG="--resume_from_checkpoint $RESUME"
else
    LATEST_CKPT=$(ls -td "finetune/checkpoints/${RUN_NAME}/checkpoint-"* 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_CKPT" ]; then
        echo "  Auto-resuming from latest checkpoint: $LATEST_CKPT"
        RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
    fi
fi

python finetune/train.py \
    --config   "$CONFIG" \
    --model_id "$MODEL" \
    --run_name "$RUN_NAME" \
    $RESUME_ARG

# ── Step 3: Evaluate best checkpoint on test split ────────────────────────────
echo ""
echo "=== Step 3: evaluate ==="
ADAPTER="finetune/checkpoints/${RUN_NAME}/final_adapter"

python finetune/evaluate.py \
    --config     "$CONFIG" \
    --checkpoint "$ADAPTER" \
    --split      test \
    --out_dir    "finetune/eval_results/${RUN_NAME}"

# ── Step 4: Score with the existing per-query-type scorer ─────────────────────
echo ""
echo "=== Step 4: score ==="
PRED_JSONL="finetune/eval_results/${RUN_NAME}/${RUN_NAME}__test.jsonl"
SCORE_DIR="finetune/eval_results/${RUN_NAME}/score"

python cladder_score_yesno.py \
    --pred_jsonl "$PRED_JSONL" \
    --out_dir    "$SCORE_DIR"

echo ""
echo "=== Done ==="
echo "  summary : $SCORE_DIR/summary.json"
echo "  per-type: $SCORE_DIR/per_query_type.csv"
echo "  per-epoch logit-mode CSVs: finetune/checkpoints/${RUN_NAME}/per_query_eval/"
