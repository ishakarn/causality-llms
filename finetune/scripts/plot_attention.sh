#!/bin/bash
# =============================================================================
# SLURM job: attention score plots for CLADDER queries (base + finetuned).
#
# Runs both passes in a single job so the model is loaded only once each:
#   Pass 1 — zero-shot base model
#   Pass 2 — base + LoRA adapter merged (fine-tuned)
#
# Select model via CONFIG (default: OLMo 3 7B Instruct).
# QUERY_IDS defaults to one representative from each query type in the test set.
#
# Usage:
#   sbatch finetune/plot_attention.sh
#   CONFIG=finetune/configs/qwen25-3b-instruct.yaml sbatch --export=ALL finetune/plot_attention.sh
#
# Override queries or adapter:
#   QUERY_IDS="23190 329" ADAPTER=... sbatch --export=ALL finetune/plot_attention.sh
#
# Outputs → finetune/attention_plots/<model_slug>/
#            finetune/attention_plots/<model_slug>_finetuned/
# =============================================================================

#SBATCH --job-name=cladder-attn
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=finetune/slurm_logs/%j_attention.out
#SBATCH --error=finetune/slurm_logs/%j_attention.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

set -euo pipefail

WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

# ── Cache ─────────────────────────────────────────────────────────────────────
WS=/scratch/workspace/ikarn_umass_edu-olmo_cladder_cache
mkdir -p "$WS/.cache/huggingface" "$WS/.cache/torch" "$WS/.cache/pip"
export HF_HOME="$WS/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$WS/.cache/torch"
export PIP_CACHE_DIR="$WS/.cache/pip"
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="${HF_TOKEN}"

# ── Conda ─────────────────────────────────────────────────────────────────────
module load conda/latest
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$CONDA_PREFIX")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cladder_olmo
pip install -q matplotlib peft

echo "[env] python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-finetune/configs/olmo3-7b-instruct.yaml}"
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('run_name','lora'))")
ADAPTER="${ADAPTER:-finetune/checkpoints/${RUN_NAME}/final_adapter}"

# One representative test-set query per query type (stratified split, seed=42):
#   ate=23190  backadj=13239  collider_bias=17897  correlation=14947
#   det-counterfactual=37522  ett=22573  exp_away=7399  marginal=11827
#   nde=329  nie=11380
QUERY_IDS="${QUERY_IDS:-23190 13239 17897 14947 37522 22573 7399 11827 329 11380}"

echo "=== Config ==="
echo "  CONFIG     = $CONFIG"
echo "  RUN_NAME   = $RUN_NAME"
echo "  ADAPTER    = $ADAPTER"
echo "  QUERY_IDS  = $QUERY_IDS"
echo ""

mkdir -p finetune/slurm_logs finetune/attention_plots

COMMON_ARGS=(
    --config       "$CONFIG"
    --query_ids    $QUERY_IDS
    --queries_json data/queries_easy.json
    --out_dir      finetune/attention_plots
)

# Run base + finetuned side-by-side comparison (highlighted text)
if [ -d "$ADAPTER" ]; then
    echo "=== Highlighted text attention (base vs fine-tuned) ==="
    python finetune/plots/plot_attention_highlight.py "${COMMON_ARGS[@]}" \
        --adapter "$ADAPTER" \
        --compare
else
    echo "=== Highlighted text attention (base only — adapter not found at ${ADAPTER}) ==="
    python finetune/plots/plot_attention_highlight.py "${COMMON_ARGS[@]}"
fi

echo ""
echo "=== Done ==="
echo "  plots → finetune/attention_plots/"
