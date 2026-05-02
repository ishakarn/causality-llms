#!/bin/bash
# =============================================================================
# SLURM array job: intervention battery eval for openai/gpt-oss-20b.
#
# Array index 0-6 maps to 7 interventions.
# Base condition only (no LoRA fine-tuning for gpt-oss).
# Uses text-parse decode (generate mode) — gpt-oss is a reasoning model.
#
# Usage:
#   sbatch finetune/scripts/eval_interventions_array_gptoss.sh
# =============================================================================

#SBATCH --job-name=cladder-interv-gptoss
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=finetune/slurm_logs/%A_%a_interv_gptoss.out
#SBATCH --error=finetune/slurm_logs/%A_%a_interv_gptoss.err
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

MODEL_ID="openai/gpt-oss-20b"
MODEL_TAG="gptoss"
SPLITS=(easy hard anticommonsense noncommonsense)
COND_TAG="${MODEL_TAG}_base"

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

echo "=== gpt-oss-20b Intervention Eval (task $SLURM_ARRAY_TASK_ID) ==="
echo "  Intervention : $INTERVENTION"
echo ""

# Build pairs, skipping already-scored splits
PAIRS=()
RUN_IDS=()
for SPLIT in "${SPLITS[@]}"; do
    DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
    [ -f "$DATA_FILE" ] || { echo "  WARNING: $DATA_FILE not found — skipping $SPLIT"; continue; }

    SCORE_FILE="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}/score/summary.json"
    if [ -f "$SCORE_FILE" ]; then
        ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
        echo "  [skip] $SPLIT  (acc=$ACC)"
        continue
    fi

    OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
    OUT_JSONL="${OUT_DIR}/${COND_TAG}__${INTERVENTION}__${SPLIT}.jsonl"
    mkdir -p "$OUT_DIR"
    PAIRS+=("${DATA_FILE}:${OUT_JSONL}")
    RUN_IDS+=("${COND_TAG}__${INTERVENTION}__${SPLIT}")
done

if [ ${#PAIRS[@]} -gt 0 ]; then
    # infer_vllm_multi.py auto-detects gpt-oss as a reasoning model and uses
    # text-parse decode (generate mode), loading the model only once for all splits.
    python finetune/infer_vllm_multi.py \
        --model         "$MODEL_ID" \
        --pairs         "${PAIRS[@]}" \
        --run_ids       "${RUN_IDS[@]}" \
        --max_model_len 8192 \
        --overwrite
fi

# Score all splits
for SPLIT in "${SPLITS[@]}"; do
    OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
    RUN_ID="${COND_TAG}__${INTERVENTION}__${SPLIT}"
    OUT_JSONL="${OUT_DIR}/${RUN_ID}.jsonl"
    SCORE_FILE="${OUT_DIR}/score/summary.json"

    [ -f "$OUT_JSONL" ] || continue
    [ -f "$SCORE_FILE" ] && continue

    python cladder_score_yesno.py \
        --pred_jsonl "$OUT_JSONL" \
        --out_dir    "${OUT_DIR}/score"

    ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
    echo "  → scored $SPLIT  acc=$ACC"
done

echo ""
echo "=== Done: $INTERVENTION for gpt-oss-20b ==="
