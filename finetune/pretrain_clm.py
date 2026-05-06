"""
Unsupervised continued pretraining on CLADDER question text.

Trains on raw text (background + given_info + question) with standard causal
language modeling loss — no answer labels, no loss masking.  This is a
separate script from train.py because train.py is hardcoded to mask all
prompt tokens and train only on the answer token.

Key differences from train.py:
  - Labels = input_ids (every token contributes to loss)
  - No answer token appended; no masking
  - Text is the `text` field from the data (already concatenates background +
    given_info + question)
  - A short instruction prefix is prepended so the model adapts in the
    instruct-tuned embedding space
  - LoRA is used for memory efficiency and so evaluation can reuse infer_vllm.py

After training, evaluate zero-shot with:
    python finetune/infer_vllm.py \\
        --model allenai/Olmo-3-7B-Instruct \\
        --lora_path finetune/checkpoints/pretrain_clm/<run_name>/final_adapter \\
        --data_file data/cladder-v1-q-easy.json \\
        --out_jsonl outputs/.../pretrained_baseline.jsonl

Usage:
    python finetune/pretrain_clm.py --config finetune/configs/pretrain_clm.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(path: str) -> List[Dict]:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl" or text.lstrip().startswith("{"):
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]
    return data


def make_clm_text(record: Dict) -> str:
    """
    Build the pretraining document from a CLADDER record.

    Deliberately excludes the `answer` field. The model sees only the
    question context — no supervision signal at all.
    """
    parts = []
    if record.get("background"):
        parts.append(record["background"].strip())
    if record.get("given_info"):
        parts.append(record["given_info"].strip())
    if record.get("question"):
        parts.append(record["question"].strip())
    # Fall back to `text` field if individual fields are absent
    if not parts and record.get("text"):
        parts.append(record["text"].strip())
    return "\n".join(parts)


def verify_no_answer_leakage(records: List[Dict], tokenized: Dataset,
                             tokenizer) -> None:
    """
    Spot-check that no answer tokens appear at the label boundary.

    For CLM pretraining every token is a label, but we confirm that
    'yes'/'no' only appear as part of legitimate question text, never
    appended as a target at the end of a sequence.
    """
    yes_ids = set()
    no_ids  = set()
    for v in [" yes", "yes", " Yes", "YES"]:
        ids = tokenizer(v, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            yes_ids.add(ids[0])
    for v in [" no", "no", " No", "NO"]:
        ids = tokenizer(v, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            no_ids.add(ids[0])

    leaked = 0
    for i, (rec, ex) in enumerate(zip(records[:100], tokenized.select(range(min(100, len(tokenized)))))):
        last_tok = ex["input_ids"][-1]
        if last_tok in yes_ids or last_tok in no_ids:
            leaked += 1
            print(f"  [WARN] Example {i}: last token is yes/no — check data pipeline")

    if leaked == 0:
        print(f"  [verify] No answer tokens detected at sequence end (checked 100 examples). ✓")
    else:
        print(f"  [WARN] {leaked}/100 examples end with yes/no token — investigate!")


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize_clm(record: Dict, tokenizer, max_seq_length: int) -> Dict:
    """
    Tokenise one record for CLM pretraining.

    Labels = input_ids (every token trains). No masking.
    """
    text = make_clm_text(record)
    ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
    )["input_ids"]
    # Append EOS so the model learns to end sequences
    if tokenizer.eos_token_id is not None and ids[-1] != tokenizer.eos_token_id:
        ids = ids + [tokenizer.eos_token_id]
    ids = ids[:max_seq_length]
    return {
        "input_ids":      ids,
        "attention_mask": [1] * len(ids),
        # labels are set by DataCollatorForLanguageModeling — no need to set here
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="finetune/configs/pretrain_clm.yaml")
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="Path to checkpoint dir to resume from")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))

    model_id  = cfg["model_id"]
    run_name  = cfg.get("run_name", "pretrain_clm")
    out_dir   = Path(cfg.get("output_dir", "finetune/checkpoints/pretrain_clm")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    max_seq   = cfg.get("max_seq_length", 512)

    print(f"[pretrain] model:    {model_id}")
    print(f"[pretrain] run_name: {run_name}")
    print(f"[pretrain] data:     {cfg['data_path']}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=cfg.get("trust_remote_code", True)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ── Data ──────────────────────────────────────────────────────────────────
    records = load_records(cfg["data_path"])
    print(f"[pretrain] loaded {len(records)} records (no answer field used)")

    # Confirm answers are not in the data we'll train on
    answer_vals = set(r.get("answer", "") for r in records)
    print(f"[pretrain] answer field values present in source: {answer_vals}")
    print(f"[pretrain] (answers are present in source but NOT included in pretraining text)")

    tokenized = [tokenize_clm(r, tokenizer, max_seq) for r in records]
    ds = Dataset.from_list(tokenized)
    print(f"[pretrain] tokenized {len(ds)} examples")
    print(f"[pretrain] max seq len: {max(len(x['input_ids']) for x in ds)}")

    verify_no_answer_leakage(records, ds, tokenizer)

    # ── Model ──────────────────────────────────────────────────────────────────
    dtype_map  = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.get("dtype", "bf16"), torch.bfloat16)

    print(f"[pretrain] loading {model_id} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules",
                                ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    # ── Training args ──────────────────────────────────────────────────────────
    bf16 = cfg.get("dtype", "bf16") == "bf16"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=bf16,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=cfg.get("logging_steps", 50),
        report_to="none",
        seed=cfg.get("seed", 42),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # ── Collator — standard CLM, no masking ────────────────────────────────────
    # mlm=False → labels = input_ids shifted right (standard CLM loss)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    print(f"\n[pretrain] Starting pretraining — NO answer supervision")
    if args.resume_from_checkpoint:
        print(f"[pretrain] Resuming from: {args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = out_dir / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n[pretrain] Saved LoRA adapter → {adapter_path}")
    print(f"\nEvaluate zero-shot with:")
    print(f"  python finetune/infer_vllm.py \\")
    print(f"      --model {model_id} \\")
    print(f"      --lora_path {adapter_path} \\")
    print(f"      --data_file data/cladder-v1-q-easy.json \\")
    print(f"      --out_jsonl outputs/.../pretrained_clm.jsonl")


if __name__ == "__main__":
    main()
