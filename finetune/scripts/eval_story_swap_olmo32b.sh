#!/bin/bash
# Eval 81_story_swap for olmo32b base + lora (easy + hard only).

#SBATCH --job-name=story-swap-olmo32b
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=finetune/slurm_logs/%j_story_swap_olmo32b.out
#SBATCH --error=finetune/slurm_logs/%j_story_swap_olmo32b.err
#SBATCH --mail-type=END,FAIL
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
pip install -q vllm

echo ""; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader; echo ""
mkdir -p finetune/slurm_logs

MODEL_ID="allenai/OLMo-3.1-32B-Instruct"
MODEL_TAG="olmo32b"
LORA_RUN="olmo32b_n2000_lora"
ADAPTER="finetune/checkpoints/graded/${LORA_RUN}/final_adapter"
INTERVENTION="81_story_swap"
SPLITS=(easy hard)

score_condition() {
    local COND_TAG="$1"
    for SPLIT in "${SPLITS[@]}"; do
        local OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
        local RUN_ID="${COND_TAG}__${INTERVENTION}__${SPLIT}"
        local OUT_JSONL="${OUT_DIR}/${RUN_ID}.jsonl"
        local SCORE_FILE="${OUT_DIR}/score/summary.json"
        [ -f "$OUT_JSONL" ] || continue
        [ -f "$SCORE_FILE" ] && continue
        python cladder_score_yesno.py --pred_jsonl "$OUT_JSONL" --out_dir "${OUT_DIR}/score"
        ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")")
        echo "  → scored $SPLIT acc=$ACC"
    done
}

# ── Base condition ────────────────────────────────────────────────────────────
COND_TAG="${MODEL_TAG}_base"
echo "=== Base ==="
BASE_PAIRS=(); BASE_RUN_IDS=()
for SPLIT in "${SPLITS[@]}"; do
    DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
    [ -f "$DATA_FILE" ] || { echo "WARNING: $DATA_FILE not found"; continue; }
    SCORE_FILE="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}/score/summary.json"
    [ -f "$SCORE_FILE" ] && { echo "  [skip] $SPLIT (already scored)"; continue; }
    OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
    mkdir -p "$OUT_DIR"
    BASE_PAIRS+=("${DATA_FILE}:${OUT_DIR}/${COND_TAG}__${INTERVENTION}__${SPLIT}.jsonl")
    BASE_RUN_IDS+=("${COND_TAG}__${INTERVENTION}__${SPLIT}")
done
if [ ${#BASE_PAIRS[@]} -gt 0 ]; then
    python finetune/infer_vllm_multi.py \
        --model "$MODEL_ID" --pairs "${BASE_PAIRS[@]}" --run_ids "${BASE_RUN_IDS[@]}" \
        --max_model_len 8192 --overwrite
fi
score_condition "$COND_TAG"

# ── LoRA condition ────────────────────────────────────────────────────────────
if [ ! -d "$ADAPTER" ]; then
    echo "WARNING: Adapter not found at $ADAPTER — skipping lora"
else
    COND_TAG="$LORA_RUN"
    echo "=== LoRA ==="
    LORA_PAIRS=(); LORA_RUN_IDS=()
    for SPLIT in "${SPLITS[@]}"; do
        DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
        [ -f "$DATA_FILE" ] || continue
        SCORE_FILE="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}/score/summary.json"
        [ -f "$SCORE_FILE" ] && { echo "  [skip] $SPLIT (already scored)"; continue; }
        OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
        mkdir -p "$OUT_DIR"
        LORA_PAIRS+=("${DATA_FILE}:${OUT_DIR}/${COND_TAG}__${INTERVENTION}__${SPLIT}.jsonl")
        LORA_RUN_IDS+=("${COND_TAG}__${INTERVENTION}__${SPLIT}")
    done
    if [ ${#LORA_PAIRS[@]} -gt 0 ]; then
        python finetune/infer_vllm_multi.py \
            --model "$MODEL_ID" --lora_path "$ADAPTER" \
            --pairs "${LORA_PAIRS[@]}" --run_ids "${LORA_RUN_IDS[@]}" \
            --max_model_len 8192 --overwrite
    fi
    score_condition "$COND_TAG"
fi

echo "=== Done: story_swap for olmo32b ==="
