#!/bin/bash
# Submission driver for: 83_story_swap_polarity_flip
# Splits: easy hard
# Usage: bash finetune/scripts/intervention_evals/eval_83_story_swap_polarity.sh

TEMPLATES=$(dirname "$0")
WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

export INTERV_NAME="83_story_swap_polarity_flip"
export SPLITS_LIST="easy hard"

JOB_NAME="83_story_swap_polarity_flip"

sbatch --job-name="interv-qwen-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_qwen.sh"
sbatch --job-name="interv-o32b-${JOB_NAME:0:20}"   --export=ALL "$TEMPLATES/_run_olmo32b.sh"
sbatch --job-name="interv-gptoss-${JOB_NAME:0:18}" --export=ALL "$TEMPLATES/_run_gptoss.sh"

echo "Submitted 3 jobs for: ${INTERV_NAME}  splits: ${SPLITS_LIST}"
