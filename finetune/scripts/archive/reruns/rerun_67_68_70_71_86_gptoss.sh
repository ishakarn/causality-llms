#!/bin/bash
# Re-run 67/68/70/71/86 for gptoss base on updated intervention files.

#SBATCH --job-name=rerun-gptoss
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/slurm_logs/%j_rerun_gptoss.out
#SBATCH --error=finetune/slurm_logs/%j_rerun_gptoss.err
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

echo ""; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader; echo ""
mkdir -p finetune/slurm_logs

MODEL_ID="openai/gpt-oss-20b"
COND_TAG="gptoss_base"
INTERVENTIONS=(67_word_replace 68_number_replace 70_word_replace_polarity_mask 71_word_replace_pct_polarity_mask 86_nonsense_replace)
SPLITS=(easy hard anticommonsense)

PAIRS=(); RUN_IDS=()
for INTERV in "${INTERVENTIONS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        DATA_FILE="data/intervened_datasets/${SPLIT}/${INTERV}_${SPLIT}.json"
        [ -f "$DATA_FILE" ] || { echo "WARNING: $DATA_FILE not found"; continue; }
        SCORE_FILE="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/${SPLIT}/score/summary.json"
        [ -f "$SCORE_FILE" ] && { echo "  [skip] ${INTERV}/${SPLIT}"; continue; }
        OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/${SPLIT}"
        mkdir -p "$OUT_DIR"
        PAIRS+=("${DATA_FILE}:${OUT_DIR}/${COND_TAG}__${INTERV}__${SPLIT}.jsonl")
        RUN_IDS+=("${COND_TAG}__${INTERV}__${SPLIT}")
    done
done

[ ${#PAIRS[@]} -gt 0 ] && python finetune/infer_vllm_multi.py \
    --model "$MODEL_ID" --pairs "${PAIRS[@]}" --run_ids "${RUN_IDS[@]}" \
    --max_model_len 8192 --overwrite

for INTERV in "${INTERVENTIONS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/${SPLIT}"
        OUT_JSONL="${OUT_DIR}/${COND_TAG}__${INTERV}__${SPLIT}.jsonl"
        SCORE_FILE="${OUT_DIR}/score/summary.json"
        [ -f "$OUT_JSONL" ] || continue
        [ -f "$SCORE_FILE" ] && continue
        python cladder_score_yesno.py --pred_jsonl "$OUT_JSONL" --out_dir "${OUT_DIR}/score"
        ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")")
        echo "  → scored ${INTERV}/${SPLIT}  acc=$ACC"
    done
done

echo "=== Done: 67/68/70/71/86 rerun for gptoss ==="
