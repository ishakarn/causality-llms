#!/bin/bash
# One-off: score gptoss_base emoji/noncommonsense (the single remaining split).
# Uses --qos=short for higher scheduling priority.

#SBATCH --job-name=gptoss-emoji-nc
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=finetune/slurm_logs/%j_gptoss_emoji_nc.out
#SBATCH --error=finetune/slurm_logs/%j_gptoss_emoji_nc.err
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

MODEL_ID="openai/gpt-oss-20b"
COND_TAG="gptoss_base"
INTERVENTION="33_insert_rare_emoji_blocks"
SPLIT="noncommonsense"

DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERVENTION}_${SPLIT}.json"
OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERVENTION}/${SPLIT}"
OUT_JSONL="${OUT_DIR}/${COND_TAG}__${INTERVENTION}__${SPLIT}.jsonl"
SCORE_FILE="${OUT_DIR}/score/summary.json"

if [ -f "$SCORE_FILE" ]; then
    ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
    echo "Already scored (acc=$ACC) — nothing to do."
    exit 0
fi

mkdir -p "$OUT_DIR"

echo "=== gpt-oss-20b emoji/noncommonsense ==="
python finetune/infer_vllm_multi.py \
    --model         "$MODEL_ID" \
    --pairs         "${DATA_FILE}:${OUT_JSONL}" \
    --run_ids       "${COND_TAG}__${INTERVENTION}__${SPLIT}" \
    --max_model_len 8192 \
    --overwrite

python cladder_score_yesno.py \
    --pred_jsonl "$OUT_JSONL" \
    --out_dir    "${OUT_DIR}/score"

ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")" 2>/dev/null || echo "?")
echo "=== Done: acc=$ACC ==="
