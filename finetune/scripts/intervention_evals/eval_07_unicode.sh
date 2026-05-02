#!/bin/bash
# Submission driver for: 7_append_1000_high_density_unicode_chars
# Splits: easy hard anticommonsense
# Usage: bash finetune/scripts/intervention_evals/eval_07_unicode.sh

TEMPLATES=$(dirname "$0")
WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

export INTERV_NAME="7_append_1000_high_density_unicode_chars"
export SPLITS_LIST="easy hard anticommonsense"

JOB_NAME="7_append_1000_high_density_unicode_chars"

sbatch --job-name="interv-qwen-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_qwen.sh"
sbatch --job-name="interv-o32b-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_olmo32b.sh"
sbatch --job-name="interv-gptoss-${JOB_NAME:0:18}" --export=ALL "$TEMPLATES/_run_gptoss.sh"

echo "Submitted 3 jobs for: ${INTERV_NAME}  splits: ${SPLITS_LIST}"
