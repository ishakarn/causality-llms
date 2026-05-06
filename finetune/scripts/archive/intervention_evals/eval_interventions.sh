#!/bin/bash
# =============================================================================
# SLURM job: evaluate (base + LoRA) on all interventions — generic, any model.
#
#   sbatch finetune/eval_interventions.sh                                         # OLMo instruct (default)
#   CONFIG=finetune/configs/qwen25-3b-instruct.yaml sbatch --export=ALL finetune/eval_interventions.sh
#
# Runs both models sequentially on a single A100:
#   base model  → ~18 intervention files × 288 test examples
#   finetuned   → same
#
# Requires: finetune.sh has completed for the chosen CONFIG (final_adapter must exist).
# =============================================================================

#SBATCH --job-name=cladder-interventions
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=finetune/slurm_logs/%j_interventions.out
#SBATCH --error=finetune/slurm_logs/%j_interventions.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

set -euo pipefail

WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

# ── Scratch cache (never home dir) ────────────────────────────────────────────
WS=${SCRATCH_CACHE:-/scratch/workspace/$(whoami)-cladder-cache}
mkdir -p "$WS/.cache/huggingface" "$WS/.cache/torch"

export HF_HOME="$WS/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$WS/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="${HF_TOKEN}"

echo "[cache] HF_HOME=$HF_HOME"

# ── Conda environment ─────────────────────────────────────────────────────────
module load conda/latest
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$CONDA_PREFIX")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cladder_olmo
echo "[env] python: $(which python)"
echo "[env] torch: $(python -c 'import torch; print(torch.__version__)')"

# ── Deps ──────────────────────────────────────────────────────────────────────
pip install -q peft accelerate datasets

# ── GPU info ──────────────────────────────────────────────────────────────────
echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

mkdir -p finetune/slurm_logs finetune/intervention_results

# ── Config: read run_name from the yaml ───────────────────────────────────────
CONFIG="${CONFIG:-finetune/configs/olmo3-7b-instruct.yaml}"
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('run_name','lora'))")
OUT_DIR="finetune/intervention_results/${RUN_NAME}"

echo "=== Config ==="
echo "  CONFIG   = $CONFIG"
echo "  RUN_NAME = $RUN_NAME"
echo "  OUT_DIR  = $OUT_DIR"
echo ""

# ── Run: base model and/or finetuned ─────────────────────────────────────────
# MODE: both (default) | base | finetuned
MODE="${MODE:-both}"
echo "=== Running intervention eval (mode=${MODE}) ==="
python finetune/eval_interventions.py \
    --mode    "$MODE" \
    --config  "$CONFIG" \
    --out_dir "$OUT_DIR"

# ── Plot results ──────────────────────────────────────────────────────────────
echo ""
echo "=== Plotting ==="
python finetune/plots/plot_interventions.py \
    --results_dir "$OUT_DIR" \
    --model_name  "$RUN_NAME" \
    --out_dir     "outputs/plots/interventions/${RUN_NAME}"

echo ""
echo "=== Done ==="
echo "  summary CSV : $OUT_DIR/summary.csv"
echo "  plots       : outputs/plots/interventions/${RUN_NAME}/"
