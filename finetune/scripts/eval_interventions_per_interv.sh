#!/bin/bash
# =============================================================================
# SLURM job: intervention eval for ONE intervention across all splits + conditions.
#
# Loads the model TWICE per job (base + lora), running all 4 splits each time.
# This avoids the 56× model-reload cost of the old per-file loop.
#
# Required env vars:
#   MODEL_TAG    — qwen3b | olmo7b | llama8b | olmo32b
#   INTERVENTION — e.g. "1_insert_spaces_between_chars_without_whitespace"
#
# Usage (submit one job per intervention):
#   for INAME in \
#     "1_insert_spaces_between_chars_without_whitespace" \
#     "7_append_1000_high_density_unicode_chars" \
#     "13_remove_every_other_word" \
#     "22_insert_incorrect_answer_once_per_sentence" \
#     "33_insert_rare_emoji_blocks" \
#     "35_polarity_flip" \
#     "66_set_numbers_to_X"; do
#       MODEL_TAG=olmo32b INTERVENTION=$INAME \
#         sbatch --export=ALL --mem=80G \
#         finetune/scripts/eval_interventions_per_interv.sh
#   done
#
# Note on polarity_flip (intervention 35):
#   The data file's `answer` field already holds the flipped ground truth.
#   infer_vllm_multi.py reads record["answer"] as gold, so scoring is correct
#   without any extra handling.
# =============================================================================

#SBATCH --job-name=cladder-interv
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/slurm_logs/%j_interv_%x.out
#SBATCH --error=finetune/slurm_logs/%j_interv_%x.err
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

echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
mkdir -p finetune/slurm_logs

# ── Model selection ───────────────────────────────────────────────────────────
MODEL_TAG="${MODEL_TAG:?ERROR: MODEL_TAG env var required}"
INTERVENTION="${INTERVENTION:?ERROR: INTERVENTION env var required}"

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

SPLITS=(easy hard anticommonsense noncommonsense)
LORA_N=2000
LORA_RUN="${MODEL_TAG}_n${LORA_N}_lora"
ADAPTER="finetune/checkpoints/graded/${LORA_RUN}/final_adapter"

echo "=== Intervention Eval ==="
echo "  MODEL_TAG    : $MODEL_TAG"
echo "  Model        : $MODEL_ID"
echo "  Intervention : $INTERVENTION"
echo "  LoRA adapter : $ADAPTER"
echo ""

# Helper: build space-separated list of data_file:out_jsonl pairs for one condition
build_pairs_and_ids() {
    local COND_TAG="$1"
    local PAIRS=()
    local RUN_IDS=()
    for SPLIT in "${SPLITS[@]}"; do
        DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
        if [ ! -f "$DATA_FILE" ]; then
            echo "  WARNING: $DATA_FILE not found — skipping $SPLIT" >&2
            continue
        fi
        OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
        RUN_ID="${COND_TAG}__${INTERVENTION}__${SPLIT}"
        OUT_JSONL="${OUT_DIR}/${RUN_ID}.jsonl"
        mkdir -p "$OUT_DIR"
        PAIRS+=("${DATA_FILE}:${OUT_JSONL}")
        RUN_IDS+=("$RUN_ID")
    done
    echo "${PAIRS[@]}"
    echo "${RUN_IDS[@]}"
}

score_condition() {
    local COND_TAG="$1"
    for SPLIT in "${SPLITS[@]}"; do
        OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
        RUN_ID="${COND_TAG}__${INTERVENTION}__${SPLIT}"
        OUT_JSONL="${OUT_DIR}/${RUN_ID}.jsonl"
        SCORE_FILE="${OUT_DIR}/score/summary.json"

        if [ ! -f "$OUT_JSONL" ]; then
            echo "  [no jsonl] $SPLIT — skipping score"
            continue
        fi
        if [ -f "$SCORE_FILE" ]; then
            ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
            echo "  [already scored] $SPLIT  acc=$ACC"
            continue
        fi

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
echo "  Condition: base  ($COND_TAG_BASE)"
echo "========================================================"

BASE_PAIRS=()
BASE_RUN_IDS=()
for SPLIT in "${SPLITS[@]}"; do
    DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
    [ -f "$DATA_FILE" ] || { echo "  WARNING: $DATA_FILE not found — skipping"; continue; }
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
        --max_model_len 4096 \
        --overwrite
fi

echo "  Scoring base condition…"
score_condition "$COND_TAG_BASE"
echo ""

# ── Condition: lora (N=2000) ──────────────────────────────────────────────────
if [ ! -d "$ADAPTER" ]; then
    echo "WARNING: Adapter not found at $ADAPTER — skipping lora condition"
else
    COND_TAG_LORA="$LORA_RUN"
    echo "========================================================"
    echo "  Condition: lora  ($COND_TAG_LORA)"
    echo "========================================================"

    LORA_PAIRS=()
    LORA_RUN_IDS=()
    for SPLIT in "${SPLITS[@]}"; do
        DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
        [ -f "$DATA_FILE" ] || { echo "  WARNING: $DATA_FILE not found — skipping"; continue; }
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
            --max_model_len 4096 \
            --overwrite
    fi

    echo "  Scoring lora condition…"
    score_condition "$COND_TAG_LORA"
fi

echo ""
echo "=== Done: $INTERVENTION for $MODEL_TAG ==="
