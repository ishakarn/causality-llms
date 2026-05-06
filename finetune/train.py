"""
LoRA fine-tuning of OLMo 7B (Instruct or Think) on CLADDER yes/no task.

Fine-tuning method: LoRA in bf16 on a single A100.
  - Only adapter weights (~0.5% of params) are trained.
  - Loss is masked so only the answer token (yes/no) and EOS contribute.
  - A custom callback evaluates per-query-type accuracy via logit mode after
    each epoch (same decision rule as cladder_infer_yesno.py).

Usage:
    python finetune/train.py --config finetune/config.yaml

CLI overrides (all optional, override config.yaml values):
    --model_id   allenai/Olmo-3-7B-Think
    --run_name   olmo3-7b-think-lora
    --num_train_epochs 3
    --resume_from_checkpoint finetune/checkpoints/olmo3-7b-instruct-lora/checkpoint-10
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_system_msg(model_id: str) -> str:
    """Match the system prompts used in cladder_infer_yesno.py."""
    if "think" in model_id.lower():
        return (
            "You must answer with exactly one token: yes or no.\n"
            "Do not provide any explanation, reasoning, analysis, or extra text.\n"
            "If you are unsure, still output yes or no."
        )
    return "Answer with only 'yes' or 'no'. No explanation. No punctuation."


def build_prompt(tokenizer, text: str, system_msg: str) -> str:
    """Build the prompt string (no answer) using chat template if available."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{system_msg}\n\n{text}\n\nAnswer:"


def tokenize_example(
    record: Dict,
    tokenizer,
    system_msg: str,
    max_seq_length: int,
) -> Dict:
    """
    Tokenise one CLADDER record for causal-LM fine-tuning.

    Labels are -100 (ignored) for all prompt tokens; only the answer
    token ("yes"/"no") and EOS carry loss.  This teaches the model to
    produce the right answer without memorising the prompt text.
    """
    prompt_str = build_prompt(tokenizer, record["text"], system_msg)
    full_str   = prompt_str + record["answer"]

    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(full_str,   add_special_tokens=False)["input_ids"]

    # Verify the token boundary is clean — some tokenizers (e.g. Llama-3.1)
    # can retokenize differently at a string junction. Fail loudly if so.
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            f"Token boundary drift detected for model. "
            f"prompt tail={prompt_ids[-5:]}, "
            f"full at boundary={full_ids[max(0,len(prompt_ids)-5):len(prompt_ids)+2]}. "
            f"The answer token mask will be wrong — fix build_prompt or tokenization."
        )

    if tokenizer.eos_token_id is not None:
        full_ids = full_ids + [tokenizer.eos_token_id]

    # Mask prompt tokens out of the loss
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # Truncate (the prompt is ~350 tokens max; answer is 1 token — no real risk)
    input_ids      = full_ids[:max_seq_length]
    labels         = labels[:max_seq_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        # token_type_ids intentionally omitted — Llama-3.1 and other models
        # raise TypeError if an unexpected key is passed to forward().
    }


def get_yesno_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """
    Resolve the single-token vocabulary IDs for yes/no variants.
    Mirrors the logic in cladder_infer_yesno.py for consistency.
    """
    def single_token_ids(variants: List[str]) -> List[int]:
        seen, out = set(), []
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
                out.append(ids[0])
        return out

    yes_ids = single_token_ids([" yes", "yes", " Yes", "YES"])
    no_ids  = single_token_ids([" no",  "no",  " No",  "NO"])
    if not yes_ids or not no_ids:
        raise RuntimeError("Could not resolve single-token ids for 'yes'/'no'")
    return yes_ids, no_ids


# ── Per-query-type evaluation callback ───────────────────────────────────────

class PerQueryEvalCallback(TrainerCallback):
    """
    After each Trainer eval phase, runs logit-mode yes/no inference over the
    raw val records and logs per-query-type accuracy.

    Decision rule: pred = "yes" if max(logit[yes_ids]) >= max(logit[no_ids]).
    This matches --decision_mode logit in cladder_infer_yesno.py.

    Metrics logged:
        eval_acc_all          — overall accuracy on val set
        eval_acc_<query_type> — per-type accuracy (10 values)

    Note: these metrics are logged via trainer.log() and appear in
    trainer_state.json / console.  Checkpoint selection uses eval_loss
    (set in config), which tracks accuracy tightly for a 1-token task.
    """

    def __init__(
        self,
        val_records: List[Dict],
        tokenizer,
        system_msg: str,
        yes_ids: List[int],
        no_ids: List[int],
        csv_dir: Path,
    ):
        self.val_records = val_records
        self.tokenizer   = tokenizer
        self.system_msg  = system_msg
        self.yes_ids     = yes_ids
        self.no_ids      = no_ids
        self.csv_dir     = csv_dir
        self.trainer     = None          # set after Trainer is constructed

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if model is None:
            return

        model.eval()
        device = next(model.parameters()).device
        yes_t  = torch.tensor(self.yes_ids, device=device)
        no_t   = torch.tensor(self.no_ids,  device=device)

        by_type = defaultdict(lambda: {"n": 0, "correct": 0})
        n_total, n_correct = 0, 0

        for record in self.val_records:
            prompt = build_prompt(self.tokenizer, record["text"], self.system_msg)
            enc    = self.tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                logits = model(**enc).logits[0, -1, :]   # last-token logits

            pred    = "yes" if logits[yes_t].max() >= logits[no_t].max() else "no"
            correct = int(pred == record["answer"])
            qt      = record["query_type"]

            by_type[qt]["n"]       += 1
            by_type[qt]["correct"] += correct
            n_total   += 1
            n_correct += correct

        overall_acc = n_correct / n_total if n_total else 0.0

        epoch_str = f"{state.epoch:.0f}" if state.epoch is not None else "?"
        print(f"\n[Epoch {epoch_str}] Logit-mode val accuracy (per query type):")

        metrics: Dict[str, float] = {}
        rows = []
        for qt in sorted(by_type):
            s   = by_type[qt]
            acc = s["correct"] / s["n"] if s["n"] else 0.0
            metrics[f"eval_acc_{qt}"] = round(acc, 4)
            rows.append({"query_type": qt, "n": s["n"], "correct": s["correct"], "acc": acc})
            print(f"  {qt:<24} {acc:.4f}  ({s['correct']}/{s['n']})")

        metrics["eval_acc_all"] = round(overall_acc, 4)
        print(f"  {'overall':<24} {overall_acc:.4f}  ({n_correct}/{n_total})\n")

        if self.trainer is not None:
            self.trainer.log(metrics)

        # Write per-epoch CSV
        epoch_int = int(state.epoch) if state.epoch is not None else 0
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.csv_dir / f"per_query_type_epoch{epoch_int:02d}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["query_type", "n", "correct", "acc"])
            w.writeheader()
            for row in rows:
                w.writerow({**row, "acc": f"{row['acc']:.4f}"})
        print(f"  [callback] wrote {csv_path}")


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: Dict) -> Dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    # Apply only the CLI overrides that were explicitly set (not None)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="finetune/config.yaml")
    # Optional CLI overrides — all correspond to config.yaml keys
    ap.add_argument("--model_id",              default=None)
    ap.add_argument("--run_name",              default=None)
    ap.add_argument("--num_train_epochs",      type=int,   default=None)
    ap.add_argument("--learning_rate",         type=float, default=None)
    ap.add_argument("--output_dir",            default=None)
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="Path to a checkpoint dir to resume from")
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "resume_from_checkpoint")}
    cfg = load_config(args.config, overrides)

    set_seed(cfg["seed"])

    model_id   = cfg["model_id"]
    run_name   = cfg.get("run_name", "olmo-lora")
    output_dir = Path(cfg["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(cfg["splits_dir"])
    system_msg = get_system_msg(model_id)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    print(f"[setup] model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=cfg.get("trust_remote_code", True)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"   # left-pad causes issues with CausalLM loss

    yes_ids, no_ids = get_yesno_ids(tokenizer)
    print(f"[setup] yes_ids={yes_ids}  no_ids={no_ids}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_records = load_jsonl(str(splits_dir / "train.jsonl"))
    val_records   = load_jsonl(str(splits_dir / "val.jsonl"))
    max_seq       = cfg["max_seq_length"]

    print(f"[data] tokenising {len(train_records)} train / {len(val_records)} val examples …")

    def to_hf_dataset(records: List[Dict]) -> Dataset:
        tokenized = [tokenize_example(r, tokenizer, system_msg, max_seq) for r in records]
        return Dataset.from_list(tokenized)

    train_ds = to_hf_dataset(train_records)
    val_ds   = to_hf_dataset(val_records)

    max_train_len = max(len(x["input_ids"]) for x in train_ds)
    max_val_len   = max(len(x["input_ids"]) for x in val_ds)
    print(f"[data] max seq len — train: {max_train_len}, val: {max_val_len}")

    # ── Base model ─────────────────────────────────────────────────────────────
    dtype_map  = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.get("dtype", "bf16"), torch.bfloat16)

    print(f"[model] loading {model_id} ({cfg.get('dtype', 'bf16')}) …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )

    # ── LoRA ───────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Required for gradient checkpointing + LoRA (PEFT docs requirement)
    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    # ── TrainingArguments ──────────────────────────────────────────────────────
    bf16 = cfg.get("dtype", "bf16") == "bf16"
    fp16 = cfg.get("dtype", "bf16") == "fp16"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        logging_steps=cfg["logging_steps"],
        report_to=cfg.get("report_to", "none"),
        seed=cfg["seed"],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # ── Callback ───────────────────────────────────────────────────────────────
    callbacks = []
    if not cfg.get("disable_per_query_callback", False):
        eval_cb = PerQueryEvalCallback(
            val_records=val_records,
            tokenizer=tokenizer,
            system_msg=system_msg,
            yes_ids=yes_ids,
            no_ids=no_ids,
            csv_dir=output_dir / "per_query_eval",
        )
        callbacks.append(eval_cb)

    # ── DataCollator ───────────────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    if callbacks:
        callbacks[0].trainer = trainer   # back-reference so callback can call trainer.log()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save final adapter ─────────────────────────────────────────────────────
    final_dir = output_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] saved LoRA adapter → {final_dir}")
    print(f"[done] score with: python finetune/evaluate.py --config finetune/config.yaml --split test")


if __name__ == "__main__":
    main()
