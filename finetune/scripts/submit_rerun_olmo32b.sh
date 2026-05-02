#!/bin/bash
# Rerun interventions 67, 68, 81, 86 for olmo32b base + lora on all splits.

WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

INTERVENTIONS=(67_word_replace 68_number_replace 81_story_swap 86_nonsense_replace)
SPLITS=(easy hard anticommonsense noncommonsense)
MODEL_ID="allenai/OLMo-3.1-32B-Instruct"
ADAPTER="finetune/checkpoints/graded/olmo32b_n2000_lora/final_adapter"

COMMON_HEADER='#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
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
export HF_TOKEN="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"

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
            COND_TAG="olmo32b_base"
            LORA_ARG=""
        else
            COND_TAG="olmo32b_n2000_lora"
            LORA_ARG="--lora_path $ADAPTER"
        fi

        LOG_SLUG="rerun_olmo32b_${INTERV}_${COND}"
        TMPSCRIPT=$(mktemp /tmp/slurm_olmo_XXXXXX.sh)

        cat > "$TMPSCRIPT" <<SCRIPT
$COMMON_HEADER
#SBATCH --job-name=ro32-${INTERV:0:2}-${COND}
#SBATCH --output=finetune/slurm_logs/%j_${LOG_SLUG}.out
#SBATCH --error=finetune/slurm_logs/%j_${LOG_SLUG}.err

PAIRS=(); RUN_IDS=()
SPLITS=(${SPLITS[@]})
for SPLIT in "\${SPLITS[@]}"; do
    DATA_FILE="data/intervened_datasets/\${SPLIT}/${INTERV}_\${SPLIT}.json"
    [ -f "\$DATA_FILE" ] || { echo "WARNING: \$DATA_FILE not found"; continue; }
    SCORE_FILE="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/\${SPLIT}/score/summary.json"
    [ -f "\$SCORE_FILE" ] && { echo "  [skip] ${INTERV}/\${SPLIT}"; continue; }
    OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/\${SPLIT}"
    mkdir -p "\$OUT_DIR"
    PAIRS+=("\${DATA_FILE}:\${OUT_DIR}/${COND_TAG}__${INTERV}__\${SPLIT}.jsonl")
    RUN_IDS+=("${COND_TAG}__${INTERV}__\${SPLIT}")
done

[ \${#PAIRS[@]} -gt 0 ] && python finetune/infer_vllm_multi.py \\
    --model "${MODEL_ID}" ${LORA_ARG} \\
    --pairs "\${PAIRS[@]}" --run_ids "\${RUN_IDS[@]}" \\
    --max_model_len 8192 --overwrite

for SPLIT in "\${SPLITS[@]}"; do
    OUT_DIR="finetune/eval_results/interventions/${COND_TAG}/${INTERV}/\${SPLIT}"
    OUT_JSONL="\${OUT_DIR}/${COND_TAG}__${INTERV}__\${SPLIT}.jsonl"
    SCORE_FILE="\${OUT_DIR}/score/summary.json"
    [ -f "\$OUT_JSONL" ] || continue
    [ -f "\$SCORE_FILE" ] && continue
    python cladder_score_yesno.py --pred_jsonl "\$OUT_JSONL" --out_dir "\${OUT_DIR}/score"
    ACC=\$(python -c "import json; s=json.load(open('\$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")")
    echo "  → scored ${INTERV}/\${SPLIT}  acc=\$ACC"
done

echo "=== Done: ${INTERV} ${COND_TAG} ==="
SCRIPT

        JOB_ID=$(sbatch "$TMPSCRIPT" | awk '{print $4}')
        echo "Submitted $JOB_ID  →  ${INTERV} / ${COND_TAG}"
        rm "$TMPSCRIPT"
    done
done
