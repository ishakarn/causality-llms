#!/bin/bash
# =============================================================================
# SLURM array job: intervention battery eval — one task per intervention.
#
# Array index 0-6 maps to one of 7 interventions.
# Each task loads the model TWICE (base + lora) and runs all 4 splits.
# Skip logic: any (condition, split) pair with an existing score is skipped.
#
# Required env var:
#   MODEL_TAG — qwen3b | olmo7b | llama8b | olmo32b
#
# Usage:
#   MODEL_TAG=qwen3b  sbatch --export=ALL finetune/scripts/eval_interventions_array.sh
#   MODEL_TAG=olmo32b sbatch --export=ALL --mem=80G finetune/scripts/eval_interventions_array.sh
# =============================================================================

#SBATCH --job-name=cladder-interv
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=finetune/slurm_logs/%A_%a_interv_%x.out
#SBATCH --error=finetune/slurm_logs/%A_%a_interv_%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

set -euo pipefail

WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

WS=${SCRATCH_CACHE:-/scratch/workspace/$(whoami)-cladder-cache}
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
pip install -q vllm

echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
mkdir -p finetune/slurm_logs

# ── Model selection ───────────────────────────────────────────────────────────
MODEL_TAG="${MODEL_TAG:?ERROR: MODEL_TAG env var required}"
case "$MODEL_TAG" in
    qwen3b)  MODEL_ID="Qwen/Qwen2.5-3B-Instruct" ;;
    olmo7b)  MODEL_ID="allenai/Olmo-3-7B-Instruct" ;;
    olmo32b) MODEL_ID="allenai/OLMo-3.1-32B-Instruct" ;;
    llama8b) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
    *)
        echo "ERROR: Unknown MODEL_TAG=$MODEL_TAG"
        exit 1
        ;;
esac

# ── Intervention index → name ─────────────────────────────────────────────────
INTERVENTIONS=(
    "1_insert_spaces_between_chars_without_whitespace"
    "7_append_1000_high_density_unicode_chars"
    "13_remove_every_other_word"
    "22_insert_incorrect_answer_once_per_sentence"
    "33_insert_rare_emoji_blocks"
    "35_polarity_flip"
    "66_set_numbers_to_X"
)
INTERVENTION="${INTERVENTIONS[$SLURM_ARRAY_TASK_ID]}"
SPLITS=(easy hard anticommonsense noncommonsense)
LORA_N=2000
LORA_RUN="${MODEL_TAG}_n${LORA_N}_lora"
ADAPTER="finetune/checkpoints/graded/${LORA_RUN}/final_adapter"

echo "=== Intervention Eval (array task $SLURM_ARRAY_TASK_ID) ==="
echo "  MODEL_TAG    : $MODEL_TAG"
echo "  Model        : $MODEL_ID"
echo "  Intervention : $INTERVENTION"
echo "  LoRA adapter : $ADAPTER"
echo ""

score_condition() {
    local COND_TAG="$1"
    for SPLIT in "${SPLITS[@]}"; do
        local OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
        local RUN_ID="${COND_TAG}__${INTERVENTION}__${SPLIT}"
        local OUT_JSONL="${OUT_DIR}/${RUN_ID}.jsonl"
        local SCORE_FILE="${OUT_DIR}/score/summary.json"

        [ -f "$OUT_JSONL" ] || continue
        [ -f "$SCORE_FILE" ] && continue

        python cladder_score_yesno.py \
            --pred_jsonl "$OUT_JSONL" \
            --out_dir    "${OUT_DIR}/score"

        ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
        echo "  → scored $SPLIT  acc=$ACC"
    done
}

# ── Condition: base ───────────────────────────────────────────────────────────
COND_TAG_BASE="${MODEL_TAG}_base"
echo "========================================================"
echo "  Condition: base"
echo "========================================================"

BASE_PAIRS=()
BASE_RUN_IDS=()
for SPLIT in "${SPLITS[@]}"; do
    DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
    [ -f "$DATA_FILE" ] || { echo "  WARNING: $DATA_FILE not found — skipping $SPLIT"; continue; }
    SCORE_FILE="finetune/eval_results/interventions/${COND_TAG_BASE}/${INTERVENTION}/${SPLIT}/score/summary.json"
    if [ -f "$SCORE_FILE" ]; then
        ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
        echo "  [skip] $SPLIT  (acc=$ACC)"
        continue
    fi
    OUT_DIR="finetune/eval_results/interventions/${COND_TAG_BASE}/${INTERVENTION}/${SPLIT}"
    OUT_JSONL="${OUT_DIR}/${COND_TAG_BASE}__${INTERVENTION}__${SPLIT}.jsonl"
    mkdir -p "$OUT_DIR"
    BASE_PAIRS+=("${DATA_FILE}:${OUT_JSONL}")
    BASE_RUN_IDS+=("${COND_TAG_BASE}__${INTERVENTION}__${SPLIT}")
done

if [ ${#BASE_PAIRS[@]} -gt 0 ]; then
    python finetune/infer_vllm_multi.py \
        --model         "$MODEL_ID" \
        --pairs         "${BASE_PAIRS[@]}" \
        --run_ids       "${BASE_RUN_IDS[@]}" \
        --max_model_len 8192 \
        --overwrite
fi
score_condition "$COND_TAG_BASE"
echo ""

# ── Condition: lora ───────────────────────────────────────────────────────────
if [ ! -d "$ADAPTER" ]; then
    echo "WARNING: Adapter not found at $ADAPTER — skipping lora condition"
else
    COND_TAG_LORA="$LORA_RUN"
    echo "========================================================"
    echo "  Condition: lora ($LORA_RUN)"
    echo "========================================================"

    LORA_PAIRS=()
    LORA_RUN_IDS=()
    for SPLIT in "${SPLITS[@]}"; do
        DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
        [ -f "$DATA_FILE" ] || { echo "  WARNING: $DATA_FILE not found — skipping $SPLIT"; continue; }
        SCORE_FILE="finetune/eval_results/interventions/${COND_TAG_LORA}/${INTERVENTION}/${SPLIT}/score/summary.json"
        if [ -f "$SCORE_FILE" ]; then
            ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
            echo "  [skip] $SPLIT  (acc=$ACC)"
            continue
        fi
        OUT_DIR="finetune/eval_results/interventions/${COND_TAG_LORA}/${INTERVENTION}/${SPLIT}"
        OUT_JSONL="${OUT_DIR}/${COND_TAG_LORA}__${INTERVENTION}__${SPLIT}.jsonl"
        mkdir -p "$OUT_DIR"
        LORA_PAIRS+=("${DATA_FILE}:${OUT_JSONL}")
        LORA_RUN_IDS+=("${COND_TAG_LORA}__${INTERVENTION}__${SPLIT}")
    done

    if [ ${#LORA_PAIRS[@]} -gt 0 ]; then
        python finetune/infer_vllm_multi.py \
            --model         "$MODEL_ID" \
            --lora_path     "$ADAPTER" \
            --pairs         "${LORA_PAIRS[@]}" \
            --run_ids       "${LORA_RUN_IDS[@]}" \
            --max_model_len 8192 \
            --overwrite
    fi
    score_condition "$COND_TAG_LORA"
fi

echo ""
echo "=== Done: $INTERVENTION for $MODEL_TAG ==="
