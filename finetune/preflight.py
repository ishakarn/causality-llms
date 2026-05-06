"""
Pre-flight sanity check for the LoRA fine-tuning pipeline.

Run this on an interactive GPU node BEFORE submitting the SLURM job.
It catches the most common failure modes in ~2-3 minutes.

    salloc -p gpu-preempt --gpus=1 -c 4 --mem 24G -t 00:10:00 --constraint=a100-80g
    conda activate cladder_olmo
    python finetune/preflight.py --config finetune/config.yaml

Exit code 0 = all checks passed, safe to sbatch.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
import yaml


def ok(msg):  print(f"  [OK]   {msg}")
def fail(msg, exc=None):
    print(f"  [FAIL] {msg}")
    if exc:
        traceback.print_exc()
    sys.exit(1)
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def check_cuda():
    section("1. CUDA / GPU")
    if not torch.cuda.is_available():
        fail("No CUDA device found — are you on a GPU node?")
    name   = torch.cuda.get_device_name(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb  = free_b  / 1e9
    total_gb = total_b / 1e9
    ok(f"GPU: {name}  ({free_gb:.1f} GB free / {total_gb:.1f} GB total)")
    if total_gb < 35:
        fail(f"GPU has only {total_gb:.1f} GB — LoRA bf16 of 7B needs ~22 GB peak; "
             "try to get an A100-40g or A100-80g node.")


def check_deps():
    section("2. Python dependencies")
    required = ["transformers", "peft", "accelerate", "datasets", "yaml"]
    for pkg in required:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError as e:
            fail(f"Missing: {pkg}  →  pip install {pkg}", e)


def check_data(cfg):
    section("3. Data")
    data_path = cfg["data_path"]
    if not Path(data_path).exists():
        fail(f"data_path not found: {data_path}")
    data = json.loads(Path(data_path).read_text())
    records = data["queries"] if isinstance(data, dict) and "queries" in data else data
    ok(f"{data_path} — {len(records)} records")

    splits_dir = Path(cfg["splits_dir"])
    for split in ("train", "val", "test"):
        p = splits_dir / f"{split}.jsonl"
        if p.exists():
            n = sum(1 for _ in open(p))
            ok(f"split/{split}.jsonl — {n} records")
        else:
            print(f"  [WARN] {p} not found — run prepare_data.py first (Step 1 in finetune.sh)")


def check_tokenizer(cfg):
    section("4. Tokenizer")
    from transformers import AutoTokenizer
    model_id = cfg["model_id"]
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=cfg.get("trust_remote_code", True)
        )
        ok(f"loaded tokenizer for {model_id}")
    except Exception as e:
        fail(f"tokenizer load failed for {model_id}", e)

    if tok.pad_token_id is None:
        print(f"  [INFO] no pad_token; will use eos_token={tok.eos_token!r}")
    if not getattr(tok, "chat_template", None):
        print(f"  [WARN] no chat_template found — fallback prompt format will be used")
    else:
        ok("chat_template present")

    # Check yes/no single-token ids
    def single_ids(variants):
        seen, out = set(), []
        for v in variants:
            ids = tok(v, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0]); out.append(ids[0])
        return out

    yes_ids = single_ids([" yes", "yes", " Yes", "YES"])
    no_ids  = single_ids([" no",  "no",  " No",  "NO"])
    if not yes_ids:
        fail("Could not find a single-token id for 'yes' — logit-mode eval will break")
    if not no_ids:
        fail("Could not find a single-token id for 'no' — logit-mode eval will break")
    ok(f"yes_ids={yes_ids}  no_ids={no_ids}")
    return tok


def check_model_and_lora(cfg, tok):
    section("5. Model load + LoRA target modules")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    model_id    = cfg["model_id"]
    torch_dtype = torch.bfloat16 if cfg.get("dtype", "bf16") == "bf16" else torch.float16

    print(f"  Loading {model_id} (this takes ~1-2 min on first run) …")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
    except Exception as e:
        fail(f"model load failed for {model_id}", e)

    free_b, total_b = torch.cuda.mem_get_info(0)
    ok(f"base model loaded — GPU free: {free_b/1e9:.1f} GB / {total_b/1e9:.1f} GB")

    # Enumerate all named modules to verify LoRA targets exist
    all_module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    target_modules   = cfg["lora_target_modules"]
    missing = [t for t in target_modules if t not in all_module_names]
    if missing:
        print(f"\n  [FAIL] These LoRA target_modules are NOT in the model: {missing}")
        print(f"\n  Available leaf module names (sample):")
        for n in sorted(all_module_names)[:40]:
            print(f"    {n}")
        print(f"\n  Update lora_target_modules in finetune/config.yaml to use names from the list above.")
        sys.exit(1)
    ok(f"LoRA target modules all present: {target_modules}")

    # Apply LoRA
    try:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        ok("LoRA applied successfully")
        model.print_trainable_parameters()
    except Exception as e:
        fail("LoRA application failed", e)

    # One forward pass to verify end-to-end
    section("6. Forward pass smoke test")
    model.eval()
    try:
        sample_text = "Imagine a world. Will A cause B?"
        enc = tok(sample_text, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(**enc)
        ok(f"forward pass OK — logit shape: {out.logits.shape}")
    except Exception as e:
        fail("forward pass failed", e)

    # One training step to verify gradient flow
    section("7. Training step smoke test (gradient checkpointing + LoRA)")
    model.train()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        from transformers import DataCollatorForSeq2Seq
        import torch.nn.functional as F

        # Build a tiny fake batch (prompt + answer token)
        prompt_ids = tok(sample_text, add_special_tokens=False)["input_ids"]
        answer_ids = tok(" yes",      add_special_tokens=False)["input_ids"]
        full_ids   = prompt_ids + answer_ids
        labels     = [-100] * len(prompt_ids) + answer_ids

        input_ids = torch.tensor([full_ids], device=next(model.parameters()).device)
        lbls      = torch.tensor([labels],   device=next(model.parameters()).device)
        attn_mask = torch.ones_like(input_ids)

        out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=lbls)
        loss = out.loss
        loss.backward()
        ok(f"backward pass OK — loss={loss.item():.4f}")

        # Check that LoRA params actually have gradients
        lora_grads = [(n, p.grad) for n, p in model.named_parameters()
                      if "lora_" in n and p.requires_grad]
        grad_ok = [g is not None for _, g in lora_grads]
        if not all(grad_ok):
            fail(f"Some LoRA params have no gradient! "
                 f"({sum(grad_ok)}/{len(grad_ok)} have grads)")
        ok(f"LoRA gradients flowing — {len(lora_grads)} lora params have grads")
    except Exception as e:
        fail("training step failed", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="finetune/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    print(f"\nPre-flight check for: {cfg['model_id']}")
    print(f"Config: {args.config}")

    check_cuda()
    check_deps()
    check_data(cfg)
    tok = check_tokenizer(cfg)
    check_model_and_lora(cfg, tok)

    print(f"\n{'='*60}")
    print(f"  ALL CHECKS PASSED — safe to: sbatch finetune/finetune.sh")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
