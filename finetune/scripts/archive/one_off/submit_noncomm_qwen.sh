#!/bin/bash
# Submit one job per intervention for qwen3b base + lora on noncommonsense split.

WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

INTERVENTIONS=(67_word_replace 68_number_replace 70_word_replace_polarity_mask 71_word_replace_pct_polarity_mask 81_story_swap 86_nonsense_replace)
MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
ADAPTER="finetune/checkpoints/graded/qwen3b_n2000_lora/final_adapter"

COMMON_HEADER='#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

set -euo pipefail
cd /work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test

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
mkdir -p finetune/slurm_logs'

for INTERV in "${INTERVENTIONS[@]}"; do
    for COND in base lora; do
        if [ "$COND" = "base" ]; then
            COND_TAG="qwen3b_base"
            LORA_ARG=""
        else
            COND_TAG="qwen3b_n2000_lora"
            LORA_ARG="--lora_path $ADAPTER"
        fi

        LOG_SLUG="noncomm_qwen_${INTERV}_${COND}"
        TMPSCRIPT=$(mktemp /tmp/slurm_qwen_XXXXXX.sh)

        cat > "$TMPSCRIPT" <<SCRIPT
$COMMON_HEADER
#SBATCH --job-name=nc-qw-${INTERV:0:2}-${COND}
#SBATCH --output=finetune/slurm_logs/%j_${LOG_SLUG}.out
#SBATCH --error=finetune/slurm_logs/%j_${LOG_SLUG}.err

DATA_FILE="data/intervened_datasets/noncommonsense/${INTERV}_noncommonsense.json"
SCORE_FILE="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/noncommonsense/score/summary.json"
[ -f "\$SCORE_FILE" ] && { echo "  [skip] ${INTERV}/noncommonsense"; exit 0; }
[ -f "\$DATA_FILE" ] || { echo "ERROR: \$DATA_FILE not found"; exit 1; }

OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/noncommonsense"
mkdir -p "\$OUT_DIR"
OUT_JSONL="\${OUT_DIR}/${COND_TAG}__${INTERV}__noncommonsense.jsonl"

python finetune/infer_vllm_multi.py \
    --model "${MODEL_ID}" ${LORA_ARG} \
    --pairs "\${DATA_FILE}:\${OUT_JSONL}" \
    --run_ids "${COND_TAG}__${INTERV}__noncommonsense" \
    --max_model_len 8192 --overwrite

python cladder_score_yesno.py --pred_jsonl "\$OUT_JSONL" --out_dir "\${OUT_DIR}/score"
ACC=\$(python -c "import json; s=json.load(open('\${OUT_DIR}/score/summary.json')); print(f\"{s['acc_all']:.3f}\")")
echo "  → scored ${INTERV}/noncommonsense  acc=\$ACC"
echo "=== Done: ${INTERV} ${COND_TAG} noncommonsense ==="
SCRIPT

        JOB_ID=$(sbatch "$TMPSCRIPT" | awk '{print $4}')
        echo "Submitted $JOB_ID  →  ${INTERV} / ${COND_TAG} / noncommonsense"
        rm "$TMPSCRIPT"
    done
done
