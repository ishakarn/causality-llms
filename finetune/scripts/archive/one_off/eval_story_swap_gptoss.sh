#!/bin/bash
# Eval 81_story_swap for gptoss base only (easy + hard only).

#SBATCH --job-name=story-swap-gptoss
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=finetune/slurm_logs/%j_story_swap_gptoss.out
#SBATCH --error=finetune/slurm_logs/%j_story_swap_gptoss.err
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

echo ""; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader; echo ""
mkdir -p finetune/slurm_logs

MODEL_ID="openai/gpt-oss-20b"
COND_TAG="gptoss_base"
INTERVENTION="81_story_swap"
SPLITS=(easy hard)

PAIRS=(); RUN_IDS=()
for SPLIT in "${SPLITS[@]}"; do
    DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
    [ -f "$DATA_FILE" ] || { echo "WARNING: $DATA_FILE not found"; continue; }
    SCORE_FILE="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}/score/summary.json"
    [ -f "$SCORE_FILE" ] && { echo "  [skip] $SPLIT (already scored)"; continue; }
    OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
    mkdir -p "$OUT_DIR"
    PAIRS+=("${DATA_FILE}:${OUT_DIR}/${COND_TAG}__${INTERVENTION}__${SPLIT}.jsonl")
    RUN_IDS+=("${COND_TAG}__${INTERVENTION}__${SPLIT}")
done

if [ ${#PAIRS[@]} -gt 0 ]; then
    python finetune/infer_vllm_multi.py \
        --model "$MODEL_ID" --pairs "${PAIRS[@]}" --run_ids "${RUN_IDS[@]}" \
        --max_model_len 8192 --overwrite
fi

# Score
for SPLIT in "${SPLITS[@]}"; do
    OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
    RUN_ID="${COND_TAG}__${INTERVENTION}__${SPLIT}"
    OUT_JSONL="${OUT_DIR}/${RUN_ID}.jsonl"
    SCORE_FILE="${OUT_DIR}/score/summary.json"
    [ -f "$OUT_JSONL" ] || continue
    [ -f "$SCORE_FILE" ] && continue
    python cladder_score_yesno.py --pred_jsonl "$OUT_JSONL" --out_dir "${OUT_DIR}/score"
    ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")")
    echo "  → scored $SPLIT acc=$ACC"
done

echo "=== Done: story_swap for gptoss ==="
