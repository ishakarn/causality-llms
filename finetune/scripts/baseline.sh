#!/bin/bash
# =============================================================================
# SLURM job: zero-shot baseline inference on CLADDER (generic — any model).
#
# Select a model by setting CONFIG before submitting:
#
#   sbatch finetune/scripts/baseline.sh                                         # OLMo instruct (default)
#   CONFIG=finetune/configs/olmo3-7b-think.yaml     sbatch --export=ALL finetune/baseline.sh
#   CONFIG=finetune/configs/qwen25-3b-instruct.yaml sbatch --export=ALL finetune/baseline.sh
#
# Outputs (per model, differentiated by run_name):
#   outputs/baseline/<run_name>/<run_name>.jsonl
#   outputs/baseline/<run_name>/score/
# =============================================================================

#SBATCH --job-name=cladder-baseline
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=finetune/slurm_logs/%j_baseline.out
#SBATCH --error=finetune/slurm_logs/%j_baseline.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

set -euo pipefail

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

# ── Cache: all large files go to scratch, NOT home dir ───────────────────────
WS=/scratch/workspace/ikarn_umass_edu-olmo_cladder_cache
mkdir -p "$WS/.cache/huggingface" "$WS/.cache/torch" "$WS/.cache/pip"

export HF_HOME="$WS/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$WS/.cache/torch"
export PIP_CACHE_DIR="$WS/.cache/pip"
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

pip install -q protobuf

# ── GPU diagnostics ───────────────────────────────────────────────────────────
echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Config: read model_id and run_name from yaml ──────────────────────────────
CONFIG="${CONFIG:-finetune/configs/olmo3-7b-instruct.yaml}"
# Dataset to evaluate (default: queries_easy); override with DATA env var
DATA="${DATA:-data/queries_easy.json}"
DATASET=$(basename "$DATA" .json)   # e.g. queries_easy, cladder-v1-q-easy

MODEL=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model_id'])")
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('run_name','lora'))")
# Baseline run id: strip the trailing -lora suffix if present
BASELINE_ID="${RUN_NAME%-lora}-baseline"
OUT_DIR="outputs/${DATASET}/baseline/${BASELINE_ID}"

echo "=== Config ==="
echo "  CONFIG      = $CONFIG"
echo "  DATA        = $DATA"
echo "  DATASET     = $DATASET"
echo "  MODEL       = $MODEL"
echo "  BASELINE_ID = $BASELINE_ID"
echo "  OUT_DIR     = $OUT_DIR"
echo ""

mkdir -p finetune/slurm_logs "$OUT_DIR"

# ── Step 1: Logit-mode inference over full dataset ────────────────────────────
echo "=== Step 1: baseline inference ==="
python cladder_infer_yesno.py \
    --model         "$MODEL" \
    --queries_json  "$DATA" \
    --out_dir       "$OUT_DIR" \
    --run_id        "$BASELINE_ID" \
    --decision_mode logit \
    --dtype         bf16

PRED_JSONL="$OUT_DIR/${BASELINE_ID}.jsonl"

# ── Step 2: Score per-query-type accuracy ─────────────────────────────────────
echo ""
echo "=== Step 2: score ==="
python cladder_score_yesno.py \
    --pred_jsonl "$PRED_JSONL" \
    --out_dir    "$OUT_DIR/score"

echo ""
echo "=== Done ==="
echo "  predictions : $PRED_JSONL"
echo "  summary     : $OUT_DIR/score/summary.json"
echo "  per-type    : $OUT_DIR/score/per_query_type.csv"
