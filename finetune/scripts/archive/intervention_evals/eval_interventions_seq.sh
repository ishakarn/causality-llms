#!/bin/bash
# =============================================================================
# SLURM job: Intervention battery evaluation — Phase 1.
#
# Evaluates TWO conditions for one model:
#   1. base  — zero-shot (no adapter)
#   2. lora  — LoRA fine-tuned at N=2000 (graded checkpoint)
#
# Runs all 7 interventions × 4 splits = 28 files per condition.
# Skips any (condition, intervention, split) triple that already has a score.
#
# ── IMPORTANT: polarity_flip (intervention 35) ───────────────────────────────
# For intervention 35, the data file's `answer` field already contains the
# FLIPPED ground truth (yes↔no).  infer_vllm.py writes `record["answer"]`
# as `gold`, so scoring against the intervention file is correct by
# construction — no special post-processing needed.  Scoring a polarity-flipped
# file against the *original* answer would be wrong; we never pass the original
# file here.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   sbatch finetune/scripts/eval_interventions_seq.sh                   # qwen3b
#   MODEL_TAG=olmo7b   sbatch --export=ALL finetune/scripts/eval_interventions_seq.sh
#   MODEL_TAG=llama8b  sbatch --export=ALL finetune/scripts/eval_interventions_seq.sh
#   MODEL_TAG=olmo32b  sbatch --export=ALL --mem=80G --constraint=a100 \
#       finetune/scripts/eval_interventions_seq.sh
#
# ── Output layout ─────────────────────────────────────────────────────────────
#   finetune/eval_results/interventions/<condition>/<iid>_<iname>/<split>/
#       <run_id>.jsonl
#       score/summary.json
#       score/per_query_type.csv
# =============================================================================

#SBATCH --job-name=cladder-interv-eval
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/slurm_logs/%j_interv_eval.out
#SBATCH --error=finetune/slurm_logs/%j_interv_eval.err
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
MODEL_TAG="${MODEL_TAG:-qwen3b}"
case "$MODEL_TAG" in
    qwen3b)  MODEL_ID="Qwen/Qwen2.5-3B-Instruct" ;;
    olmo7b)  MODEL_ID="allenai/Olmo-3-7B-Instruct" ;;
    olmo32b) MODEL_ID="allenai/OLMo-3.1-32B-Instruct" ;;
    llama8b) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
    *)
        echo "ERROR: Unknown MODEL_TAG=$MODEL_TAG. Use: qwen3b, olmo7b, olmo32b, llama8b"
        exit 1
        ;;
esac

# ── Intervention and split definitions ───────────────────────────────────────
# Keys: intervention id + name (matches filename prefix)
INTERVENTIONS=(
    "1_insert_spaces_between_chars_without_whitespace"
    "7_append_1000_high_density_unicode_chars"
    "13_remove_every_other_word"
    "22_insert_incorrect_answer_once_per_sentence"
    "33_insert_rare_emoji_blocks"
    "35_polarity_flip"
    "66_set_numbers_to_X"
)
SPLITS=(easy hard anticommonsense noncommonsense)

# Intervention 35 flips the ground-truth label.  The intervention file's
# `answer` field already reflects this flip.  infer_vllm.py reads that field
# as `gold`, so no post-processing is required here — but we track the id for
# documentation clarity.
POLARITY_FLIP_INAME="35_polarity_flip"

# ── Conditions: base (no adapter) + N=2000 LoRA ───────────────────────────────
LORA_N=2000
LORA_RUN="${MODEL_TAG}_n${LORA_N}_lora"
ADAPTER="finetune/checkpoints/graded/${LORA_RUN}/final_adapter"

echo "=== Intervention Battery Eval (Phase 1) ==="
echo "  MODEL_TAG : $MODEL_TAG"
echo "  Model     : $MODEL_ID"
echo "  LoRA run  : $LORA_RUN"
echo "  Adapter   : $ADAPTER"
echo ""

if [ ! -d "$ADAPTER" ]; then
    echo "WARNING: LoRA adapter not found at $ADAPTER — will skip lora condition"
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
for CONDITION in "base" "lora"; do
    if [ "$CONDITION" = "lora" ] && [ ! -d "$ADAPTER" ]; then
        echo "  [skip] lora condition — adapter missing"
        continue
    fi

    LORA_ARG=""
    COND_TAG="${MODEL_TAG}_base"
    if [ "$CONDITION" = "lora" ]; then
        LORA_ARG="--lora_path $ADAPTER"
        COND_TAG="${LORA_RUN}"
    fi

    echo "========================================================"
    echo "  Condition: $CONDITION  ($COND_TAG)"
    echo "========================================================"

    for INAME in "${INTERVENTIONS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            DATA_FILE="data/intervened_datasets/${SPLIT}/${INAME}_${SPLIT}.json"

            if [ ! -f "$DATA_FILE" ]; then
                echo "  [skip] $INAME / $SPLIT — data file not found"
                continue
            fi

            OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INAME}/${SPLIT}"
            SCORE_FILE="${OUT_DIR}/score/summary.json"
            RUN_ID="${COND_TAG}__${INAME}__${SPLIT}"
            OUT_JSONL="${OUT_DIR}/${RUN_ID}.jsonl"

            mkdir -p "$OUT_DIR"

            # Skip if already scored
            if [ -f "$SCORE_FILE" ]; then
                ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
                echo "  [skip] $INAME / $SPLIT  (acc=$ACC)"
                continue
            fi

            # Note for polarity_flip: the data file's 'answer' is already the
            # flipped ground truth.  infer_vllm.py uses record['answer'] as gold,
            # so scoring is correct without any additional handling.
            if [ "$INAME" = "$POLARITY_FLIP_INAME" ]; then
                echo "  [infer] $INAME / $SPLIT  [NOTE: gold = flipped answer from data file]"
            else
                echo "  [infer] $INAME / $SPLIT"
            fi

            # All interventions can inflate prompt length significantly:
            #   1_insert_spaces  → 1 token per char → up to ~2200 tokens
            #   7_unicode        → ~1500 appended tokens
            #   33_emoji         → emoji blocks between every word
            # Use 4096 uniformly to safely cover the worst case.
            python finetune/infer_vllm.py \
                --model         "$MODEL_ID" \
                $LORA_ARG \
                --data_file     "$DATA_FILE" \
                --out_jsonl     "$OUT_JSONL" \
                --run_id        "$RUN_ID" \
                --max_model_len 4096 \
                --overwrite

            python cladder_score_yesno.py \
                --pred_jsonl "$OUT_JSONL" \
                --out_dir    "${OUT_DIR}/score"

            ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
            echo "  → $INAME / $SPLIT  acc=$ACC"
        done
    done
    echo ""
done

echo "=== All done for $MODEL_TAG ==="
