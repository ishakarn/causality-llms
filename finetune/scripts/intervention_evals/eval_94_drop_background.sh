#!/bin/bash
# Submission driver for: 94_drop_background
# One job per model per split for faster GPU allocation.
# Usage: bash finetune/scripts/intervention_evals/eval_94_drop_background.sh

TEMPLATES=$(dirname "$0")
WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

export INTERV_NAME="94_drop_background"

for SPLIT in easy hard anticommonsense noncommonsense; do
    export SPLITS_LIST="$SPLIT"
    sbatch --job-name="interv-qwen-94-${SPLIT:0:5}"   --export=ALL "$TEMPLATES/_run_qwen.sh"
    sbatch --job-name="interv-llama-94-${SPLIT:0:5}"  --export=ALL "$TEMPLATES/_run_llama8b.sh"
    sbatch --job-name="interv-o32b-94-${SPLIT:0:5}"   --export=ALL "$TEMPLATES/_run_olmo32b.sh"
    sbatch --job-name="interv-goss-94-${SPLIT:0:5}"   --export=ALL "$TEMPLATES/_run_gptoss.sh"
    echo "Submitted 4 jobs for: ${INTERV_NAME} / ${SPLIT}"
done
