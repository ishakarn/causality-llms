#!/bin/bash
# =============================================================================
# SLURM job: graded supervised exposure — sequential version for short QOS.
#
# Runs all N values for one model in a single job (no array).
# Use this when short QOS is needed (QOS rejects job arrays).
#
# Select model via MODEL_TAG env var:
#   sbatch finetune/scripts/graded_exposure_seq.sh                        # Qwen-3B (default)
#   MODEL_TAG=olmo7b   sbatch --export=ALL finetune/scripts/graded_exposure_seq.sh
#   MODEL_TAG=llama8b  sbatch --export=ALL finetune/scripts/graded_exposure_seq.sh
#
# OLMo-32B needs 80 GB VRAM — split across two jobs:
#   SIZES_OVERRIDE="50 100 250 500"   MODEL_TAG=olmo32b \
#       sbatch --export=ALL --mem=80G --constraint=a100 finetune/scripts/graded_exposure_seq.sh
#   SIZES_OVERRIDE="1000 2000"        MODEL_TAG=olmo32b \
#       sbatch --export=ALL --mem=80G --constraint=a100 finetune/scripts/graded_exposure_seq.sh
#
# Skip already-finished N values by overriding SIZES_OVERRIDE:
#   SIZES_OVERRIDE="500 1000 2000" MODEL_TAG=qwen3b sbatch --export=ALL ...
# =============================================================================

#SBATCH --job-name=cladder-graded-seq
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/slurm_logs/%j_graded_seq.out
#SBATCH --error=finetune/slurm_logs/%j_graded_seq.err
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
pip install -q peft accelerate datasets

echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
mkdir -p finetune/slurm_logs

# ── Model selection ───────────────────────────────────────────────────────────
MODEL_TAG="${MODEL_TAG:-qwen3b}"
case "$MODEL_TAG" in
    qwen3b)
        MODEL="Qwen/Qwen2.5-3B-Instruct"
        BASE_CONFIG="finetune/configs/qwen25-3b-instruct.yaml"
        ;;
    olmo7b)
        MODEL="allenai/Olmo-3-7B-Instruct"
        BASE_CONFIG="finetune/configs/olmo3-7b-instruct.yaml"
        ;;
    olmo32b)
        MODEL="allenai/OLMo-3.1-32B-Instruct"
        BASE_CONFIG="finetune/configs/olmo3-32b-instruct.yaml"
        ;;
    llama8b)
        MODEL="meta-llama/Llama-3.1-8B-Instruct"
        BASE_CONFIG="finetune/configs/llama31-8b-instruct.yaml"
        ;;
    *)
        echo "ERROR: Unknown MODEL_TAG=$MODEL_TAG. Use: qwen3b, olmo7b, olmo32b, llama8b"
        exit 1
        ;;
esac

SIZES=(${SIZES_OVERRIDE:-50 100 250 500 1000 2000})

echo "=== Graded Exposure (sequential) ==="
echo "  MODEL_TAG   : $MODEL_TAG"
echo "  Model       : $MODEL"
echo "  Base config : $BASE_CONFIG"
echo "  Sizes       : ${SIZES[*]}"
echo ""

for N in "${SIZES[@]}"; do
    RUN_NAME="${MODEL_TAG}_n${N}_lora"
    SPLITS_DIR="finetune/splits_graded/n${N}"
    CKPT_DIR="finetune/checkpoints/graded/${RUN_NAME}"
    EVAL_DIR="finetune/eval_results/graded/${RUN_NAME}"

    echo "========================================================"
    echo "  N=$N  run=$RUN_NAME"
    echo "========================================================"

    if [ ! -f "${SPLITS_DIR}/train.jsonl" ]; then
        echo "ERROR: ${SPLITS_DIR}/train.jsonl not found — run prepare_graded.py first"
        exit 1
    fi

    # Skip if already fully evaluated
    if [ -f "${EVAL_DIR}/score/summary.json" ]; then
        ACC=$(python -c "import json; s=json.load(open('${EVAL_DIR}/score/summary.json')); print(f\"{s['acc_all']:.3f}\")")
        echo "  → Already done (acc=$ACC), skipping."
        echo ""
        continue
    fi

    # Write per-N config
    TEMP_CONFIG=$(mktemp /tmp/graded_config_XXXXXX.yaml)
    python - <<PYEOF
import yaml, pathlib
cfg = yaml.safe_load(open("$BASE_CONFIG"))
cfg["run_name"]   = "$RUN_NAME"
cfg["splits_dir"] = "$SPLITS_DIR"
cfg["output_dir"] = "finetune/checkpoints/graded"
cfg["disable_per_query_callback"] = True
pathlib.Path("$TEMP_CONFIG").write_text(yaml.dump(cfg))
print(f"  Temp config: run_name={cfg['run_name']}  splits_dir={cfg['splits_dir']}")
PYEOF

    # ── Train ─────────────────────────────────────────────────────────────────
    echo "--- Train ---"
    python finetune/train.py \
        --config   "$TEMP_CONFIG" \
        --model_id "$MODEL" \
        --run_name "$RUN_NAME"

    # ── Evaluate ──────────────────────────────────────────────────────────────
    echo "--- Evaluate ---"
    mkdir -p "$EVAL_DIR"
    python finetune/evaluate.py \
        --config     "$TEMP_CONFIG" \
        --checkpoint "${CKPT_DIR}/final_adapter" \
        --split      test \
        --out_dir    "$EVAL_DIR" \
        --overwrite

    # ── Score ─────────────────────────────────────────────────────────────────
    echo "--- Score ---"
    PRED_JSONL="${EVAL_DIR}/${RUN_NAME}__test.jsonl"
    python cladder_score_yesno.py \
        --pred_jsonl "$PRED_JSONL" \
        --out_dir    "${EVAL_DIR}/score"

    rm -f "$TEMP_CONFIG"

    echo "  → N=$N done: $(python -c "import json; s=json.load(open('${EVAL_DIR}/score/summary.json')); print(f\"acc={s['acc_all']:.3f}\")")"
    echo ""
done

echo "=== All N complete for $MODEL_TAG ==="
