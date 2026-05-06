#!/bin/bash
# Submission driver for: 13_remove_every_other_word_except_numbers
# Splits: easy hard anticommonsense
# Usage: bash finetune/scripts/intervention_evals/eval_13_del_alt_words.sh

TEMPLATES=$(dirname "$0")
WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

export INTERV_NAME="13_remove_every_other_word_except_numbers"
export SPLITS_LIST="easy hard anticommonsense"

JOB_NAME="13_remove_every_other_word_except_numbers"

sbatch --job-name="interv-qwen-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_qwen.sh"
sbatch --job-name="interv-o32b-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_olmo32b.sh"
sbatch --job-name="interv-gptoss-${JOB_NAME:0:18}" --export=ALL "$TEMPLATES/_run_gptoss.sh"

echo "Submitted 3 jobs for: ${INTERV_NAME}  splits: ${SPLITS_LIST}"
