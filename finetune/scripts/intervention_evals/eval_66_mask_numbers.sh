#!/bin/bash
# Submission driver for: 66_set_numbers_to_X
# Splits: easy hard anticommonsense
# Usage: bash finetune/scripts/intervention_evals/eval_66_mask_numbers.sh

TEMPLATES=$(dirname "$0")
WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

export INTERV_NAME="66_set_numbers_to_X"
export SPLITS_LIST="easy hard anticommonsense"

JOB_NAME="66_set_numbers_to_X"

sbatch --job-name="interv-qwen-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_qwen.sh"
sbatch --job-name="interv-o32b-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_olmo32b.sh"
sbatch --job-name="interv-gptoss-${JOB_NAME:0:18}" --export=ALL "$TEMPLATES/_run_gptoss.sh"

echo "Submitted 3 jobs for: ${INTERV_NAME}  splits: ${SPLITS_LIST}"
