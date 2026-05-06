#!/bin/bash
# Submission driver for: 96_probability_expander
# One job per model per split for faster GPU allocation.
# Usage: bash finetune/scripts/intervention_evals/eval_96_probability_expander.sh

TEMPLATES=$(dirname "$0")
WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

export INTERV_NAME="96_probability_expander"

for SPLIT in easy hard anticommonsense noncommonsense; do
    export SPLITS_LIST="$SPLIT"
    sbatch --job-name="interv-qwen-96-${SPLIT:0:5}"   --export=ALL "$TEMPLATES/_run_qwen.sh"
    sbatch --job-name="interv-llama-96-${SPLIT:0:5}"  --export=ALL "$TEMPLATES/_run_llama8b.sh"
    sbatch --job-name="interv-o32b-96-${SPLIT:0:5}"   --export=ALL "$TEMPLATES/_run_olmo32b.sh"
    sbatch --job-name="interv-goss-96-${SPLIT:0:5}"   --export=ALL "$TEMPLATES/_run_gptoss.sh"
    echo "Submitted 4 jobs for: ${INTERV_NAME} / ${SPLIT}"
done
