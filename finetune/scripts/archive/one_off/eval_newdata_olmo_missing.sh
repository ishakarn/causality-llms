#!/bin/bash
# =============================================================================
# SLURM job: OLMo 7B Instruct inference on 3 missing cladder-v1 datasets:
#   balanced, commonsense, noncommonsense
# =============================================================================

#SBATCH --job-name=cladder-olmo-missing
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=finetune/slurm_logs/%j_olmo_missing.out
#SBATCH --error=finetune/slurm_logs/%j_olmo_missing.err
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

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="finetune/configs/olmo3-7b-instruct.yaml"
MODEL=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model_id'])")
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('run_name','lora'))")
OUTPUT_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('output_dir','finetune/checkpoints'))")
BASELINE_ID="${RUN_NAME%-lora}-baseline"
LORA_PATH="${OUTPUT_DIR}/${RUN_NAME}/final_adapter"

echo "=== Config ==="
echo "  MODEL       = $MODEL"
echo "  RUN_NAME    = $RUN_NAME"
echo "  BASELINE_ID = $BASELINE_ID"
echo "  LORA_PATH   = $LORA_PATH"
echo ""

# ── Missing datasets only ─────────────────────────────────────────────────────
DATASETS=(
    data/cladder-v1-q-balanced.json
    data/cladder-v1-q-commonsense.json
    data/cladder-v1-q-noncommonsense.json
)

# ── Loop over datasets ────────────────────────────────────────────────────────
for DATA in "${DATASETS[@]}"; do
    DATASET=$(basename "$DATA" .json)
    echo ""
    echo "========================================================"
    echo "  Dataset: $DATASET"
    echo "========================================================"

    BASE_OUT="outputs/${DATASET}/baseline/${BASELINE_ID}"
    FT_OUT="outputs/${DATASET}/finetuned/${RUN_NAME}"
    PLOT_OUT="outputs/plots/newdata/${DATASET}"
    mkdir -p "$BASE_OUT" "$FT_OUT" "$PLOT_OUT"

    BASE_JSONL="${BASE_OUT}/${BASELINE_ID}.jsonl"
    FT_JSONL="${FT_OUT}/${RUN_NAME}__${DATASET}.jsonl"

    # ── Step 1: Baseline inference (vLLM) ─────────────────────────────────────
    echo "[1/4] Baseline inference (vLLM)…"
    python finetune/infer_vllm.py \
        --model     "$MODEL" \
        --data_file "$DATA" \
        --out_jsonl "$BASE_JSONL" \
        --run_id    "$BASELINE_ID" \
        --overwrite

    # ── Step 2: Finetuned inference (vLLM + LoRA) ─────────────────────────────
    echo "[2/4] Finetuned inference (vLLM + LoRA)…"
    python finetune/infer_vllm.py \
        --model     "$MODEL" \
        --lora_path "$LORA_PATH" \
        --data_file "$DATA" \
        --out_jsonl "$FT_JSONL" \
        --run_id    "${RUN_NAME}__${DATASET}" \
        --overwrite

    # ── Step 3: Score both ────────────────────────────────────────────────────
    echo "[3/4] Scoring…"
    python cladder_score_yesno.py \
        --pred_jsonl "$BASE_JSONL" \
        --out_dir    "${BASE_OUT}/score"

    python cladder_score_yesno.py \
        --pred_jsonl "$FT_JSONL" \
        --out_dir    "${FT_OUT}/score"

    # ── Step 4: Plot comparison ───────────────────────────────────────────────
    echo "[4/4] Plotting…"
    python finetune/plots/plot_comparison.py \
        --baseline  "$BASE_JSONL" \
        --finetuned "$FT_JSONL" \
        --out        "${PLOT_OUT}/${DATASET}__${RUN_NAME}.png" \
        --title      "${DATASET}: zero-shot vs LoRA (${RUN_NAME})"

    echo "  → done: $DATASET"
done

echo ""
echo "=== All missing datasets complete ==="
echo "  Plots: outputs/plots/newdata/"
