#!/bin/bash
# =============================================================================
# SLURM job: Unsupervised CLM pretraining on CLADDER question text.
#
# Trains OLMo-3-7B-Instruct (LoRA) on background+given_info+question text
# from the aggregate CLADDER dataset — no answer supervision.
#
# After training, evaluate zero-shot accuracy on the easy split:
#   python finetune/infer_vllm.py \
#       --model allenai/Olmo-3-7B-Instruct \
#       --lora_path finetune/checkpoints/pretrain_clm/olmo7b_pretrain_clm/final_adapter \
#       --data_file data/cladder-v1-q-easy.json \
#       --out_jsonl finetune/eval_results/pretrain_clm/olmo7b_pretrain_clm__easy.jsonl
#
# Submit:
#   sbatch finetune/pretrain_clm.sh
#
# Override model config:
#   CONFIG=finetune/configs/pretrain_clm_qwen.yaml sbatch --export=ALL finetune/pretrain_clm.sh
# =============================================================================

#SBATCH --job-name=cladder-pretrain-clm
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/slurm_logs/%j_pretrain_clm.out
#SBATCH --error=finetune/slurm_logs/%j_pretrain_clm.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

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
pip install -q peft accelerate datasets

echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
mkdir -p finetune/slurm_logs

CONFIG="${CONFIG:-finetune/configs/pretrain_clm.yaml}"
echo "=== CLM Pretraining ==="
echo "  Config: $CONFIG"
echo ""

# ── Step 1: Pretrain ──────────────────────────────────────────────────────────
echo "=== Step 1: Unsupervised CLM pretraining (NO answer supervision) ==="
RUN_NAME_EARLY=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['run_name'])")
OUT_DIR_EARLY=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('output_dir','finetune/checkpoints/pretrain_clm'))")
LATEST_CKPT=$(ls -td "${OUT_DIR_EARLY}/${RUN_NAME_EARLY}/checkpoint-"* 2>/dev/null | head -1 || true)
RESUME_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    echo "  Auto-resuming from: $LATEST_CKPT"
    RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
fi
python finetune/pretrain_clm.py --config "$CONFIG" $RESUME_ARG

# Extract fields from config for downstream steps
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['run_name'])")
MODEL_ID=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model_id'])")
ADAPTER="finetune/checkpoints/pretrain_clm/${RUN_NAME}/final_adapter"

echo ""
echo "=== Step 2: Zero-shot evaluation on easy split ==="
EVAL_DIR="finetune/eval_results/pretrain_clm/${RUN_NAME}"
mkdir -p "$EVAL_DIR"

python finetune/infer_vllm.py \
    --model     "$MODEL_ID" \
    --lora_path "$ADAPTER" \
    --data_file "data/cladder-v1-q-easy.json" \
    --out_jsonl "${EVAL_DIR}/${RUN_NAME}__easy.jsonl"

echo ""
echo "=== Step 3: Score ==="
python cladder_score_yesno.py \
    --pred_jsonl "${EVAL_DIR}/${RUN_NAME}__easy.jsonl" \
    --out_dir    "${EVAL_DIR}/score"

echo ""
echo "=== Done: $RUN_NAME ==="
echo "  Adapter : $ADAPTER"
echo "  Score   : ${EVAL_DIR}/score/summary.json"
