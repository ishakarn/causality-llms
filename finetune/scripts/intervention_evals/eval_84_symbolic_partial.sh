#!/bin/bash
# Submission driver for: 84_symbolic_partial
# Splits: easy hard
# Usage: bash finetune/scripts/intervention_evals/eval_84_symbolic_partial.sh

TEMPLATES=$(dirname "$0")
WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

export INTERV_NAME="84_symbolic_partial"
export SPLITS_LIST="easy hard"

JOB_NAME="84_symbolic_partial"

sbatch --job-name="interv-qwen-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_qwen.sh"
sbatch --job-name="interv-o32b-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_olmo32b.sh"
sbatch --job-name="interv-gptoss-${JOB_NAME:0:18}" --export=ALL "$TEMPLATES/_run_gptoss.sh"

echo "Submitted 3 jobs for: ${INTERV_NAME}  splits: ${SPLITS_LIST}"
