#!/bin/bash
# =============================================================================
# SLURM job: evaluate graded checkpoints on big datasets — sequential version.
#
# Loops over all N values and all 4 datasets in a single job.
# Skips any (N, dataset) pair that already has a score file.
#
# Select model via MODEL_TAG env var:
#   sbatch finetune/scripts/eval_graded_seq.sh                        # Qwen-3B (default)
#   MODEL_TAG=llama8b  sbatch --export=ALL finetune/scripts/eval_graded_seq.sh
#   MODEL_TAG=olmo7b   sbatch --export=ALL finetune/scripts/eval_graded_seq.sh
#   MODEL_TAG=olmo32b  sbatch --export=ALL --mem=80G --constraint=a100 finetune/scripts/eval_graded_seq.sh
#
# Output layout:
#   finetune/eval_results/graded/<model_tag>_n<N>_lora/<dataset>/
#       <run_name>__<dataset>.jsonl
#       score/summary.json  +  score/per_query_type.csv
# =============================================================================

#SBATCH --job-name=cladder-graded-eval
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/slurm_logs/%j_graded_eval_seq.out
#SBATCH --error=finetune/slurm_logs/%j_graded_eval_seq.err
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
export HF_TOKEN="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"

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
case "$MODEL_TAG" in
    qwen3b)  MODEL_ID="Qwen/Qwen2.5-3B-Instruct" ;;
    olmo7b)  MODEL_ID="allenai/Olmo-3-7B-Instruct" ;;
    olmo32b) MODEL_ID="allenai/OLMo-3.1-32B-Instruct" ;;
    llama8b) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
    *)
        echo "ERROR: Unknown MODEL_TAG=$MODEL_TAG. Use: qwen3b, olmo7b, olmo32b, llama8b"
        exit 1
        ;;
esac

SIZES=(${SIZES_OVERRIDE:-50 100 250 500 1000 2000})
DATASETS=(
    "data/cladder-v1-q-easy.json"
    "data/cladder-v1-q-hard.json"
    "data/cladder-v1-q-anticommonsense.json"
    "data/cladder-v1-q-noncommonsense.json"
)

echo "=== Graded Eval (sequential) ==="
echo "  MODEL_TAG : $MODEL_TAG"
echo "  Model     : $MODEL_ID"
echo "  Sizes     : ${SIZES[*]}"
echo ""

for N in "${SIZES[@]}"; do
    RUN_NAME="${MODEL_TAG}_n${N}_lora"
    ADAPTER="finetune/checkpoints/graded/${RUN_NAME}/final_adapter"

    if [ ! -d "$ADAPTER" ]; then
        echo "WARNING: Adapter not found at $ADAPTER — skipping N=$N"
        continue
    fi

    echo "========================================================"
    echo "  N=$N  adapter=$ADAPTER"
    echo "========================================================"

    for DATA_PATH in "${DATASETS[@]}"; do
        DATASET=$(basename "$DATA_PATH" .json | sed 's/cladder-v1-q-//')
        OUT_DIR="finetune/eval_results/graded/${RUN_NAME}/${DATASET}"
        SCORE_FILE="${OUT_DIR}/score/summary.json"
        OUT_JSONL="${OUT_DIR}/${RUN_NAME}__${DATASET}.jsonl"
        mkdir -p "$OUT_DIR"

        # Skip if already scored
        if [ -f "$SCORE_FILE" ]; then
            ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")")
            echo "  [skip] $DATASET  (already done, acc=$ACC)"
            continue
        fi

        echo "  [infer] $DATASET"
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

        ACC=$(python -c "import json; s=json.load(open('$SCORE_FILE')); print(f\"{s['acc_all']:.3f}\")")
        echo "  → $DATASET  acc=$ACC"
    done
    echo ""
done

echo "=== All done. Generating plots... ==="
python finetune/plots/plot_graded.py --models "$MODEL_TAG"
echo "  Plots → outputs/plots/graded/"
