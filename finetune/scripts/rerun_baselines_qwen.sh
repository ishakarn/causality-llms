#!/bin/bash
# Re-run Qwen2.5-3B-Instruct baseline on all 4 splits (with background in text).
# Required env var: SPLIT

#SBATCH --job-name=base-rerun-qwen
#SBATCH --partition=gpu-preempt
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint="a16|a40|a100"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/slurm_logs/%j_base_rerun_qwen_%x.out
#SBATCH --error=finetune/slurm_logs/%j_base_rerun_qwen_%x.err

set -euo pipefail

WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")/../.."; pwd)}
cd "$WORKDIR"

WS=${SCRATCH_CACHE:-/scratch/workspace/$(whoami)-cladder-cache}
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

MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
LORA_ADAPTER="finetune/checkpoints/graded/qwen3b_n2000_lora/final_adapter"

SPLIT="${SPLIT}"
DATA="data/cladder-v1-q-${SPLIT}.json"
echo "=== Split: ${SPLIT} ==="

# ── Base ──────────────────────────────────────────────────────────────────────
RUN_ID="qwen25-3b-instruct-baseline"
OUT_DIR="outputs/cladder-v1-q-${SPLIT}/baseline/${RUN_ID}"
mkdir -p "$OUT_DIR"
python finetune/infer_vllm_multi.py \
    --model "$MODEL_ID" \
    --pairs "${DATA}:${OUT_DIR}/${RUN_ID}.jsonl" \
    --run_ids "$RUN_ID" \
    --max_model_len 8192 --overwrite
python cladder_score_yesno.py --pred_jsonl "${OUT_DIR}/${RUN_ID}.jsonl" --out_dir "${OUT_DIR}/score"
echo "  base scored: $(python -c "import json; print(f\"{json.load(open('${OUT_DIR}/score/summary.json'))['acc_all']:.3f}\")")"

# ── LoRA ──────────────────────────────────────────────────────────────────────
RUN_ID="qwen25-3b-instruct-lora"
OUT_DIR="outputs/cladder-v1-q-${SPLIT}/finetuned/${RUN_ID}"
mkdir -p "$OUT_DIR"
python finetune/infer_vllm_multi.py \
    --model "$MODEL_ID" --lora_path "$LORA_ADAPTER" \
    --pairs "${DATA}:${OUT_DIR}/${RUN_ID}.jsonl" \
    --run_ids "$RUN_ID" \
    --max_model_len 8192 --overwrite
python cladder_score_yesno.py --pred_jsonl "${OUT_DIR}/${RUN_ID}.jsonl" --out_dir "${OUT_DIR}/score"
echo "  lora scored: $(python -c "import json; print(f\"{json.load(open('${OUT_DIR}/score/summary.json'))['acc_all']:.3f}\")")"

echo "=== Done: qwen3b / ${SPLIT} ==="
