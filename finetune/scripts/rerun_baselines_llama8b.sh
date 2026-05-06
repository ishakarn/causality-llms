#!/bin/bash
# Re-run Llama-3.1-8B-Instruct LoRA baseline on all 4 splits (with background in text).
# Required env var: SPLIT

#SBATCH --job-name=base-rerun-llama
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint="a40|a100"
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/slurm_logs/%j_base_rerun_llama_%x.out
#SBATCH --error=finetune/slurm_logs/%j_base_rerun_llama_%x.err

set -euo pipefail

WORKDIR=/work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test
cd "$WORKDIR"

WS=/scratch/workspace/ikarn_umass_edu-olmo_cladder_cache
mkdir -p "$WS/.cache/huggingface" "$WS/.cache/torch"
export HF_HOME="$WS/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$WS/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="${HF_TOKEN}"

module load conda/latest
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$CONDA_PREFIX")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cladder_olmo
pip install -q vllm

echo ""; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader; echo ""
mkdir -p finetune/slurm_logs

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
LORA_ADAPTER="finetune/checkpoints/graded/llama8b_n2000_lora/final_adapter"

SPLIT="${SPLIT}"
DATA="data/cladder-v1-q-${SPLIT}.json"
echo "=== Split: ${SPLIT} ==="

# ── LoRA ──────────────────────────────────────────────────────────────────────
RUN_ID="llama31-8b-instruct-lora"
OUT_DIR="outputs/cladder-v1-q-${SPLIT}/finetuned/${RUN_ID}"
mkdir -p "$OUT_DIR"
python finetune/infer_vllm_multi.py \
    --model "$MODEL_ID" --lora_path "$LORA_ADAPTER" \
    --pairs "${DATA}:${OUT_DIR}/${RUN_ID}.jsonl" \
    --run_ids "$RUN_ID" \
    --max_model_len 8192 --overwrite
python cladder_score_yesno.py --pred_jsonl "${OUT_DIR}/${RUN_ID}.jsonl" --out_dir "${OUT_DIR}/score"
echo "  lora scored: $(python -c "import json; print(f\"{json.load(open('${OUT_DIR}/score/summary.json'))['acc_all']:.3f}\")")"

echo "=== Done: llama8b / ${SPLIT} ==="
