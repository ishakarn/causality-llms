#!/bin/bash
# Submit baseline re-runs for all models × all splits (with background in text).
# Usage: bash finetune/scripts/submit_baseline_reruns.sh

SCRIPTS=$(dirname "$0")
WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

for SPLIT in easy hard anticommonsense noncommonsense; do
    export SPLIT="$SPLIT"
    sbatch --job-name="qwen-base-${SPLIT:0:5}"   --export=ALL "$SCRIPTS/rerun_baselines_qwen_base.sh"
    sbatch --job-name="qwen-lora-${SPLIT:0:5}"   --export=ALL "$SCRIPTS/rerun_baselines_qwen_lora.sh"
    sbatch --job-name="llama-base-${SPLIT:0:5}"  --export=ALL "$SCRIPTS/rerun_baselines_llama8b_base.sh"
    sbatch --job-name="llama-lora-${SPLIT:0:5}"  --export=ALL "$SCRIPTS/rerun_baselines_llama8b_lora.sh"
    sbatch --job-name="o32b-base-${SPLIT:0:5}"   --export=ALL "$SCRIPTS/rerun_baselines_olmo32b_base.sh"
    sbatch --job-name="o32b-lora-${SPLIT:0:5}"   --export=ALL "$SCRIPTS/rerun_baselines_olmo32b_lora.sh"
    sbatch --job-name="goss-base-${SPLIT:0:5}"   --export=ALL "$SCRIPTS/rerun_baselines_gptoss.sh"
    echo "Submitted 7 jobs for split: ${SPLIT}"
done

echo ""
echo "NOTE: gpt-5-nano and gpt-5.5 baselines require batch API re-submission."
