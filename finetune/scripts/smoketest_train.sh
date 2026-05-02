#!/bin/bash
# =============================================================================
# Smoke test: verify tokenization + chat template + 1 training step for any
# model config before committing to a full fine-tuning run.
#
# Checks:
#   1. Tokenizer loads and chat template works
#   2. Token boundary is clean (no drift between prompt and full string)
#   3. yes/no IDs resolve to single tokens
#   4. 1 epoch of training on 10 examples completes without error
#   5. infer_vllm.py runs logit-mode inference on 5 records
#
# Select model via CONFIG:
#   CONFIG=finetune/configs/llama31-8b-instruct.yaml sbatch --export=ALL finetune/smoketest_train.sh
#   CONFIG=finetune/configs/olmo3-7b-instruct.yaml   sbatch --export=ALL finetune/smoketest_train.sh
#
# Default: llama31-8b-instruct
# =============================================================================

#SBATCH --job-name=cladder-smoketest
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=finetune/slurm_logs/%j_smoketest.out
#SBATCH --error=finetune/slurm_logs/%j_smoketest.err
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
echo "[env] python: $(which python)"

pip install -q peft accelerate datasets vllm

echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

mkdir -p finetune/slurm_logs

CONFIG="${CONFIG:-finetune/configs/llama31-8b-instruct.yaml}"
MODEL=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model_id'])")
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('run_name','lora'))")
SMOKE_RUN="${RUN_NAME}-smoke"
SMOKE_DIR="finetune/checkpoints/${SMOKE_RUN}"

echo "=== Smoke test: $MODEL ==="
echo "  CONFIG   = $CONFIG"
echo "  RUN_NAME = $SMOKE_RUN"
echo ""

# ── Step 1: Tokenizer + boundary check ───────────────────────────────────────
echo "=== Step 1: tokenizer + boundary check ==="
python - <<PYEOF
import sys, yaml
from transformers import AutoTokenizer
from finetune.train import build_prompt, get_yesno_ids, get_system_msg

cfg = yaml.safe_load(open("$CONFIG"))
model_id = cfg["model_id"]
print(f"  Loading tokenizer: {model_id}")
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=cfg.get("trust_remote_code", True))
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"

system_msg = get_system_msg(model_id)
print(f"  system_msg: {system_msg!r}")

yes_ids, no_ids = get_yesno_ids(tok)
print(f"  yes_ids={yes_ids}  no_ids={no_ids}")
if not yes_ids or not no_ids:
    print("  ERROR: could not resolve yes/no token IDs!", file=sys.stderr)
    sys.exit(1)

# Test boundary on a real example
import json
records = [json.loads(l) for l in open("finetune/splits/val.jsonl") if l.strip()][:5]
for i, rec in enumerate(records):
    prompt_str = build_prompt(tok, rec["text"], system_msg)
    full_str   = prompt_str + rec["answer"]
    prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
    full_ids   = tok(full_str,   add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        print(f"  ERROR: boundary drift on example {i}!", file=sys.stderr)
        print(f"    prompt tail: {prompt_ids[-5:]}", file=sys.stderr)
        print(f"    full at boundary: {full_ids[len(prompt_ids)-5:len(prompt_ids)+3]}", file=sys.stderr)
        sys.exit(1)
    answer_tok = full_ids[len(prompt_ids)]
    print(f"  [ex {i}] answer={rec['answer']!r}  answer_tok_id={answer_tok}  "
          f"decoded={tok.decode([answer_tok])!r}  boundary=OK")

print("  Tokenizer check PASSED")
PYEOF

echo ""
echo "=== Step 2: 1-epoch training on 10 examples ==="

# Write a temp config with 1 epoch, 10 examples, tiny output dir
python - <<PYEOF
import yaml, json, pathlib

cfg = yaml.safe_load(open("$CONFIG"))
cfg["num_train_epochs"] = 1
cfg["run_name"] = "$SMOKE_RUN"
cfg["splits_dir"] = "/tmp"         # points to smoke_train.jsonl / smoke_val.jsonl
cfg["disable_per_query_callback"] = True
cfg["per_device_train_batch_size"] = 1
cfg["gradient_accumulation_steps"] = 1
cfg["per_device_eval_batch_size"] = 1
cfg["logging_steps"] = 1
cfg["save_total_limit"] = 1

pathlib.Path("/tmp/smoke_config.yaml").write_text(yaml.dump(cfg))

# Write 10-example splits to /tmp
for split in ("train", "val"):
    src = pathlib.Path(f"finetune/splits/{split}.jsonl")
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()][:10]
    out = pathlib.Path(f"/tmp/{split}.jsonl")   # train.py expects splits_dir/train.jsonl
    out.write_text("\n".join(json.dumps(r) for r in rows))
print("  Wrote /tmp/smoke_config.yaml and /tmp/{train,val}.jsonl")
PYEOF

python finetune/train.py \
    --config   /tmp/smoke_config.yaml \
    --model_id "$MODEL" \
    --run_name "$SMOKE_RUN"

echo ""
echo "=== Step 3: vLLM logit-mode inference on 5 records ==="
python finetune/infer_vllm.py \
    --model       "$MODEL" \
    --data_file   data/cladder-v1-q-easy.json \
    --out_jsonl   "/tmp/${SMOKE_RUN}_infer.jsonl" \
    --run_id      "$SMOKE_RUN" \
    --max_samples 5 \
    --overwrite

echo ""
echo "--- Inference results (pred vs gold) ---"
python - <<PYEOF
import json
path = "/tmp/$SMOKE_RUN_infer.jsonl"
with open(f"/tmp/$SMOKE_RUN" + "_infer.jsonl") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        print(f"  [{i}] pred={r['pred']}  gold={r['gold']}  mode={r['decision_mode']}")
PYEOF

echo ""
echo "=== Smoke test PASSED for $MODEL ==="
echo "  Checkpoints: $SMOKE_DIR"
echo "  To clean up: rm -rf $SMOKE_DIR"
