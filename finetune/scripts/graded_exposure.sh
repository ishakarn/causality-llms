#!/bin/bash
# =============================================================================
# SLURM job array: graded supervised exposure experiment.
#
# Runs LoRA fine-tuning for OLMo-3-7B-Instruct and Qwen-2.5-3B-Instruct
# at 6 training sizes (50, 100, 250, 500, 1000, 2000).
#
# Array layout (12 jobs total):
#   Index 0–5:  OLMo-3-7B-Instruct  at n=50,100,250,500,1000,2000
#   Index 6–11: Qwen-2.5-3B-Instruct at n=50,100,250,500,1000,2000
#
# Outputs:
#   finetune/checkpoints/graded/<model>_n<N>/   ← LoRA adapter
#   finetune/eval_results/graded/<model>_n<N>/  ← test predictions + score
#
# Prerequisites:
#   python finetune/prepare_graded.py    (run once to create all splits)
#
# Submit:
#   sbatch finetune/graded_exposure.sh
#   sbatch --array=0-5  finetune/graded_exposure.sh   # OLMo only
#   sbatch --array=6-11 finetune/graded_exposure.sh   # Qwen only
# =============================================================================

#SBATCH --job-name=cladder-graded
#SBATCH --array=0-11
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=finetune/slurm_logs/%A_%a_graded.out
#SBATCH --error=finetune/slurm_logs/%A_%a_graded.err
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

# ── Lookup tables ──────────────────────────────────────────────────────────────
SIZES=(50 100 250 500 1000 2000)

MODELS=(
    "allenai/Olmo-3-7B-Instruct"       # index 0–5
    "allenai/Olmo-3-7B-Instruct"
    "allenai/Olmo-3-7B-Instruct"
    "allenai/Olmo-3-7B-Instruct"
    "allenai/Olmo-3-7B-Instruct"
    "allenai/Olmo-3-7B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"         # index 6–11
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
)

MODEL_TAGS=(
    "olmo7b" "olmo7b" "olmo7b" "olmo7b" "olmo7b" "olmo7b"
    "qwen3b" "qwen3b" "qwen3b" "qwen3b" "qwen3b" "qwen3b"
)

BASE_CONFIGS=(
    "finetune/configs/olmo3-7b-instruct.yaml"   # 0–5
    "finetune/configs/olmo3-7b-instruct.yaml"
    "finetune/configs/olmo3-7b-instruct.yaml"
    "finetune/configs/olmo3-7b-instruct.yaml"
    "finetune/configs/olmo3-7b-instruct.yaml"
    "finetune/configs/olmo3-7b-instruct.yaml"
    "finetune/configs/qwen25-3b-instruct.yaml"  # 6–11
    "finetune/configs/qwen25-3b-instruct.yaml"
    "finetune/configs/qwen25-3b-instruct.yaml"
    "finetune/configs/qwen25-3b-instruct.yaml"
    "finetune/configs/qwen25-3b-instruct.yaml"
    "finetune/configs/qwen25-3b-instruct.yaml"
)

IDX=${SLURM_ARRAY_TASK_ID}
SIZE_IDX=$((IDX % 6))
N=${SIZES[$SIZE_IDX]}
MODEL=${MODELS[$IDX]}
MODEL_TAG=${MODEL_TAGS[$IDX]}
BASE_CONFIG=${BASE_CONFIGS[$IDX]}
RUN_NAME="${MODEL_TAG}_n${N}_lora"
SPLITS_DIR="finetune/splits_graded/n${N}"
CKPT_DIR="finetune/checkpoints/graded/${RUN_NAME}"
EVAL_DIR="finetune/eval_results/graded/${RUN_NAME}"

echo "=== Graded Exposure Job ==="
echo "  Array index : $IDX"
echo "  Model       : $MODEL"
echo "  N           : $N"
echo "  Run name    : $RUN_NAME"
echo "  Splits dir  : $SPLITS_DIR"
echo "  Checkpoint  : $CKPT_DIR"
echo ""

# Verify splits exist
if [ ! -f "${SPLITS_DIR}/train.jsonl" ]; then
    echo "ERROR: ${SPLITS_DIR}/train.jsonl not found."
    echo "Run: python finetune/prepare_graded.py"
    exit 1
fi

# Write a per-job config by patching the base config
TEMP_CONFIG=$(mktemp /tmp/graded_config_XXXXXX.yaml)
python - <<PYEOF
import yaml, pathlib
cfg = yaml.safe_load(open("$BASE_CONFIG"))
cfg["run_name"]   = "$RUN_NAME"
cfg["splits_dir"] = "$SPLITS_DIR"
cfg["output_dir"] = "finetune/checkpoints/graded"
# Small N jobs can use fewer workers; large N is fast anyway
cfg["disable_per_query_callback"] = False
pathlib.Path("$TEMP_CONFIG").write_text(yaml.dump(cfg))
print(f"  Wrote temp config: {cfg['run_name']}  splits_dir={cfg['splits_dir']}")
PYEOF

# ── Step 1: Train ─────────────────────────────────────────────────────────────
echo "=== Step 1: Train (n=$N) ==="
python finetune/train.py \
    --config   "$TEMP_CONFIG" \
    --model_id "$MODEL" \
    --run_name "$RUN_NAME"

# ── Step 2: Evaluate on shared test split ─────────────────────────────────────
echo ""
echo "=== Step 2: Evaluate ==="
mkdir -p "$EVAL_DIR"
ADAPTER="${CKPT_DIR}/final_adapter"

python finetune/evaluate.py \
    --config     "$TEMP_CONFIG" \
    --checkpoint "$ADAPTER" \
    --split      test \
    --out_dir    "$EVAL_DIR" \
    --overwrite

# ── Step 3: Score ─────────────────────────────────────────────────────────────
echo ""
echo "=== Step 3: Score ==="
PRED_JSONL="${EVAL_DIR}/${RUN_NAME}__test.jsonl"

python cladder_score_yesno.py \
    --pred_jsonl "$PRED_JSONL" \
    --out_dir    "${EVAL_DIR}/score"

rm -f "$TEMP_CONFIG"

echo ""
echo "=== Done: $RUN_NAME ==="
echo "  Score: ${EVAL_DIR}/score/summary.json"
