#!/bin/bash
# =============================================================================
# SLURM job array: evaluate graded-exposure checkpoints on big datasets.
#
# For each training size N, runs logit-mode inference on 4 large CLADDER
# datasets (easy, hard, anticommonsense, noncommonsense) and scores each.
#
# Array layout — select via MODEL_TAG env var (default: qwen3b):
#   qwen3b  indices 0–5  → n=50,100,250,500,1000,2000
#   olmo7b  indices 0–5  → n=50,100,250,500,1000,2000
#
# Output layout:
#   finetune/eval_results/graded/<model_tag>_n<N>_lora/<dataset>/
#       <run_name>__<dataset>.jsonl
#       score/summary.json
#       score/per_query_type.csv
#
# Submit (Qwen):
#   sbatch finetune/eval_graded.sh
#
# Submit (OLMo-7B):
#   MODEL_TAG=olmo7b MODEL_ID=allenai/Olmo-3-7B-Instruct \
#       sbatch --export=ALL finetune/scripts/eval_graded.sh
# =============================================================================

#SBATCH --job-name=cladder-graded-eval
#SBATCH --array=0-5
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=finetune/slurm_logs/%A_%a_graded_eval.out
#SBATCH --error=finetune/slurm_logs/%A_%a_graded_eval.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikarn@umass.edu

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

echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
mkdir -p finetune/slurm_logs

# ── Model selection ───────────────────────────────────────────────────────────
MODEL_TAG="${MODEL_TAG:-qwen3b}"
if [ "$MODEL_TAG" = "qwen3b" ]; then
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
elif [ "$MODEL_TAG" = "olmo7b" ]; then
    MODEL_ID="${MODEL_ID:-allenai/Olmo-3-7B-Instruct}"
else
    echo "ERROR: Unknown MODEL_TAG=$MODEL_TAG. Use 'qwen3b' or 'olmo7b'."
    exit 1
fi

# ── Training size for this array task ────────────────────────────────────────
SIZES=(50 100 250 500 1000 2000)
IDX=${SLURM_ARRAY_TASK_ID}
N=${SIZES[$IDX]}
RUN_NAME="${MODEL_TAG}_n${N}_lora"
ADAPTER="finetune/checkpoints/graded/${RUN_NAME}/final_adapter"

echo "=== Graded Eval Job ==="
echo "  Array index : $IDX"
echo "  MODEL_TAG   : $MODEL_TAG"
echo "  Model       : $MODEL_ID"
echo "  N           : $N"
echo "  Run name    : $RUN_NAME"
echo "  Adapter     : $ADAPTER"
echo ""

if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: Adapter not found at $ADAPTER"
    echo "Run graded_exposure.sh first and wait for it to complete."
    exit 1
fi

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS=(
    "data/cladder-v1-q-easy.json"
    "data/cladder-v1-q-hard.json"
    "data/cladder-v1-q-anticommonsense.json"
    "data/cladder-v1-q-noncommonsense.json"
)

for DATA_PATH in "${DATASETS[@]}"; do
    DATASET=$(basename "$DATA_PATH" .json | sed 's/cladder-v1-q-//')   # easy, hard, ...
    OUT_DIR="finetune/eval_results/graded/${RUN_NAME}/${DATASET}"
    OUT_JSONL="${OUT_DIR}/${RUN_NAME}__${DATASET}.jsonl"
    mkdir -p "$OUT_DIR"

    echo "------------------------------------------------------------"
    echo "  Dataset: $DATASET  (→ $OUT_JSONL)"
    echo "------------------------------------------------------------"

    python finetune/infer_vllm.py \
        --model     "$MODEL_ID" \
        --lora_path "$ADAPTER" \
        --data_file "$DATA_PATH" \
        --out_jsonl "$OUT_JSONL" \
        --run_id    "${RUN_NAME}__${DATASET}" \
        --overwrite

    python cladder_score_yesno.py \
        --pred_jsonl "$OUT_JSONL" \
        --out_dir    "${OUT_DIR}/score"

    echo "  → $DATASET done"
    echo ""
done

echo "=== All datasets done for N=$N ($RUN_NAME) ==="
