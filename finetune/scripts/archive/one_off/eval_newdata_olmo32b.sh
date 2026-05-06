#!/bin/bash
# =============================================================================
# SLURM job: OLMo-3.1-32B-Instruct baseline + finetuned inference on all 4
# cladder-v1 datasets used in per-model comparison plots.
#
# Requires A100-80GB (32B in bf16 = ~64 GB weights).
#
# Outputs:
#   outputs/{dataset}/baseline/olmo3-32b-instruct-baseline/
#   outputs/{dataset}/finetuned/olmo3-32b-instruct-lora/
#
# Submit:
#   sbatch finetune/eval_newdata_olmo32b.sh
# =============================================================================

#SBATCH --job-name=cladder-olmo32b-eval
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=finetune/slurm_logs/%j_olmo32b_eval.out
#SBATCH --error=finetune/slurm_logs/%j_olmo32b_eval.err
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
export HF_TOKEN="${HF_TOKEN}"

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

MODEL="allenai/OLMo-3.1-32B-Instruct"
RUN_NAME="olmo3-32b-instruct-lora"
BASELINE_ID="olmo3-32b-instruct-baseline"
LORA_PATH="finetune/checkpoints/${RUN_NAME}/final_adapter"

echo "=== OLMo-3.1-32B-Instruct: baseline + finetuned ==="
echo "  MODEL       = $MODEL"
echo "  BASELINE_ID = $BASELINE_ID"
echo "  LORA_PATH   = $LORA_PATH"
echo ""

DATASETS=(
    data/cladder-v1-q-easy.json
    data/cladder-v1-q-hard.json
    data/cladder-v1-q-noncommonsense.json
    data/cladder-v1-q-anticommonsense.json
)

for DATA in "${DATASETS[@]}"; do
    DATASET=$(basename "$DATA" .json)
    echo ""
    echo "========================================================"
    echo "  Dataset: $DATASET"
    echo "========================================================"

    BASE_OUT="outputs/${DATASET}/baseline/${BASELINE_ID}"
    FT_OUT="outputs/${DATASET}/finetuned/${RUN_NAME}"
    mkdir -p "$BASE_OUT" "$FT_OUT"

    BASE_JSONL="${BASE_OUT}/${BASELINE_ID}.jsonl"
    FT_JSONL="${FT_OUT}/${RUN_NAME}__${DATASET}.jsonl"

    # ── Step 1: Baseline inference ────────────────────────────────────────────
    echo "[1/3] Baseline inference…"
    python finetune/infer_vllm.py \
        --model     "$MODEL" \
        --data_file "$DATA" \
        --out_jsonl "$BASE_JSONL" \
        --run_id    "$BASELINE_ID" \
        --overwrite

    # ── Step 2: Finetuned inference (LoRA) ────────────────────────────────────
    echo "[2/3] Finetuned inference (LoRA)…"
    python finetune/infer_vllm.py \
        --model     "$MODEL" \
        --lora_path "$LORA_PATH" \
        --data_file "$DATA" \
        --out_jsonl "$FT_JSONL" \
        --run_id    "${RUN_NAME}__${DATASET}" \
        --overwrite

    # ── Step 3: Score both ────────────────────────────────────────────────────
    echo "[3/3] Scoring…"
    python cladder_score_yesno.py \
        --pred_jsonl "$BASE_JSONL" \
        --out_dir    "${BASE_OUT}/score"

    python cladder_score_yesno.py \
        --pred_jsonl "$FT_JSONL" \
        --out_dir    "${FT_OUT}/score"

    echo "  → done: $DATASET"
done

echo ""
echo "=== All datasets complete — running per-model plot ==="
python finetune/plots/plot_per_model_datasets.py

echo ""
echo "=== Done ==="
echo "  Plots: outputs/plots/per_model_datasets/"
