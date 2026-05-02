#!/bin/bash
# Generic SLURM template: Qwen2.5-3B-Instruct base + LoRA
# Required env vars (set via --export or parent script):
#   INTERV_NAME  — intervention directory name, e.g. 1_add_space_between_nonspace_characters
#   SPLITS_LIST  — space-separated splits, e.g. "easy hard anticommonsense"

#SBATCH --job-name=interv-qwen
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/slurm_logs/%j_interv_qwen_%x.out
#SBATCH --error=finetune/slurm_logs/%j_interv_qwen_%x.err
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

MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
MODEL_TAG="qwen3b"
LORA_RUN="qwen3b_n2000_lora"
ADAPTER="finetune/checkpoints/graded/${LORA_RUN}/final_adapter"

IFS=' ' read -ra INTERVENTIONS <<< "${INTERV_NAME}"
IFS=' ' read -ra SPLITS      <<< "${SPLITS_LIST}"

echo "=== Intervention: ${INTERV_NAME} | Splits: ${SPLITS_LIST} ==="

score_condition() {
    local COND_TAG="$1"
    for INTERV in "${INTERVENTIONS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            local OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/${SPLIT}"
            local RUN_ID="${COND_TAG}__${INTERV}__${SPLIT}"
            local OUT_JSONL="${OUT_DIR}/${RUN_ID}.jsonl"
            local SCORE_FILE="${OUT_DIR}/score/summary.json"
            [ -f "$OUT_JSONL" ] || continue
            rm -rf "${OUT_DIR}/score"
            python cladder_score_yesno.py --pred_jsonl "$OUT_JSONL" --out_dir "${OUT_DIR}/score"
            ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")")
            echo "  → scored ${INTERV}/${SPLIT}  acc=$ACC"
        done
    done
}

# ── Base ──────────────────────────────────────────────────────────────────────
COND_TAG="${MODEL_TAG}_base"
echo "=== Base ==="
PAIRS=(); RUN_IDS=()
for INTERV in "${INTERVENTIONS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERV}_${SPLIT}.json"
        [ -f "$DATA_FILE" ] || { echo "WARNING: $DATA_FILE not found"; continue; }
        OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/${SPLIT}"
        mkdir -p "$OUT_DIR"
        PAIRS+=("${DATA_FILE}:${OUT_DIR}/${COND_TAG}__${INTERV}__${SPLIT}.jsonl")
        RUN_IDS+=("${COND_TAG}__${INTERV}__${SPLIT}")
    done
done
[ ${#PAIRS[@]} -gt 0 ] && python finetune/infer_vllm_multi.py \
    --model "$MODEL_ID" --pairs "${PAIRS[@]}" --run_ids "${RUN_IDS[@]}" \
    --max_model_len 8192 --overwrite
score_condition "$COND_TAG"

# ── LoRA ──────────────────────────────────────────────────────────────────────
if [ ! -d "$ADAPTER" ]; then
    echo "WARNING: Adapter not found — skipping lora"
else
    COND_TAG="$LORA_RUN"
    echo "=== LoRA ==="
    PAIRS=(); RUN_IDS=()
    for INTERV in "${INTERVENTIONS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERV}_${SPLIT}.json"
            [ -f "$DATA_FILE" ] || continue
            OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/${SPLIT}"
            mkdir -p "$OUT_DIR"
            PAIRS+=("${DATA_FILE}:${OUT_DIR}/${COND_TAG}__${INTERV}__${SPLIT}.jsonl")
            RUN_IDS+=("${COND_TAG}__${INTERV}__${SPLIT}")
        done
    done
    [ ${#PAIRS[@]} -gt 0 ] && python finetune/infer_vllm_multi.py \
        --model "$MODEL_ID" --lora_path "$ADAPTER" \
        --pairs "${PAIRS[@]}" --run_ids "${RUN_IDS[@]}" \
        --max_model_len 8192 --overwrite
    score_condition "$COND_TAG"
fi

echo "=== Done: ${INTERV_NAME} for qwen3b ==="
