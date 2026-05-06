#!/bin/bash
# Submission driver for: 32_insert_common_emoji_blocks
# Splits: easy hard anticommonsense
# Usage: bash finetune/scripts/intervention_evals/eval_32_common_emoji.sh

TEMPLATES=$(dirname "$0")
WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

export INTERV_NAME="32_insert_common_emoji_blocks"
export SPLITS_LIST="easy hard anticommonsense"

JOB_NAME="32_insert_common_emoji_blocks"

sbatch --job-name="interv-qwen-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_qwen.sh"
sbatch --job-name="interv-o32b-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_olmo32b.sh"
sbatch --job-name="interv-gptoss-${JOB_NAME:0:18}" --export=ALL "$TEMPLATES/_run_gptoss.sh"

echo "Submitted 3 jobs for: ${INTERV_NAME}  splits: ${SPLITS_LIST}"
