#!/bin/bash
# =============================================================================
# SLURM job: zero-shot baseline inference for openai/gpt-oss-20b on all 6
# cladder-v1 datasets.  No fine-tuning — baseline only.
#
# Submit:
#   sbatch finetune/eval_gptoss.sh
#
# Requires A100-80GB for the 20B model (~40 GB weights in bf16).
# Weights cached to scratch HF_HOME on first run.
# =============================================================================

#SBATCH --job-name=cladder-gptoss
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=finetune/slurm_logs/%j_gptoss.out
#SBATCH --error=finetune/slurm_logs/%j_gptoss.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

set -euo pipefail

WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

# ── Scratch cache ─────────────────────────────────────────────────────────────
WS=${SCRATCH_CACHE:-/scratch/workspace/$(whoami)-cladder-cache}
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
BASELINE_ID="gpt-oss-20b-baseline"

echo "=== Model: $MODEL ==="
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

    BASE_OUT="outputs/${DATASET}/baseline/${BASELINE_ID}"
    PLOT_OUT="outputs/plots/newdata/${DATASET}"
    mkdir -p "$BASE_OUT" "$PLOT_OUT"

    BASE_JSONL="${BASE_OUT}/${BASELINE_ID}.jsonl"

    # ── Step 1: Inference ─────────────────────────────────────────────────────
    echo "[1/3] Baseline inference (vLLM)…"
    python finetune/infer_vllm.py \
        --model     "$MODEL" \
        --data_file "$DATA" \
        --out_jsonl "$BASE_JSONL" \
        --run_id    "$BASELINE_ID" \
        --max_model_len 2048 \
        --overwrite

    # ── Step 2: Score ─────────────────────────────────────────────────────────
    echo "[2/3] Scoring…"
    python cladder_score_yesno.py \
        --pred_jsonl "$BASE_JSONL" \
        --out_dir    "${BASE_OUT}/score"

    # ── Step 3: Regenerate all-models plot for this dataset ───────────────────
    echo "[3/3] Plotting (all models)…"
    python finetune/plots/plot_all_models.py --dataset "$DATASET"

    echo "  → done: $DATASET"
done

echo ""
echo "=== All datasets complete ==="
echo "  Plots: outputs/plots/newdata/"
