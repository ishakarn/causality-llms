"""
Post-training evaluation of a LoRA-fine-tuned OLMo model on CLADDER.

Uses logit-mode yes/no decisions (same as --decision_mode logit in
cladder_infer_yesno.py), so results are directly comparable to baseline
inference runs.

Output JSONL is compatible with cladder_score_yesno.py.

Usage:
    # Evaluate best checkpoint on test split
    python finetune/evaluate.py \
        --config     finetune/config.yaml \
        --split      test

    # Evaluate a specific checkpoint
    python finetune/evaluate.py \
        --config     finetune/config.yaml \
        --checkpoint finetune/checkpoints/olmo3-7b-instruct-lora/checkpoint-10 \
        --split      val

    # Then score with the existing scorer:
    python cladder_score_yesno.py \
        --pred_jsonl finetune/eval_results/olmo3-7b-instruct-lora/olmo3-7b-instruct-lora__test.jsonl \
        --out_dir    finetune/eval_results/olmo3-7b-instruct-lora/score
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Helpers (mirrors train.py / cladder_infer_yesno.py) ──────────────────────

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_system_msg(model_id: str) -> str:
    if "think" in model_id.lower():
        return (
            "You must answer with exactly one token: yes or no.\n"
            "Do not provide any explanation, reasoning, analysis, or extra text.\n"
            "If you are unsure, still output yes or no."
        )
    return "Answer with only 'yes' or 'no'. No explanation. No punctuation."


def build_prompt(tokenizer, text: str, system_msg: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{system_msg}\n\n{text}\n\nAnswer:"


def get_yesno_ids(tokenizer) -> Tuple[List[int], List[int]]:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def load_json_or_jsonl(path: str) -> List[Dict]:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl" or text.lstrip().startswith("{"):
        # JSONL: one JSON object per line
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    return json.loads(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      default="finetune/config.yaml")
    ap.add_argument(
        "--checkpoint", default=None,
        help="Path to LoRA adapter dir. Defaults to <output_dir>/<run_name>/final_adapter",
    )
    ap.add_argument("--split",    choices=["val", "test"], default="test")
    ap.add_argument(
        "--data_file", default=None,
        help="Optional: evaluate on an arbitrary JSON/JSONL data file instead of the training split.",
    )
    ap.add_argument("--out_dir",  default=None,
                    help="Output dir. Defaults to <output_dir>/<run_name>/eval_results/")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg        = yaml.safe_load(Path(args.config).read_text())
    model_id   = cfg["model_id"]
    run_name   = cfg.get("run_name", "olmo-lora")
    output_dir = Path(cfg["output_dir"]) / run_name

    checkpoint = args.checkpoint or str(output_dir / "final_adapter")
    out_dir    = Path(args.out_dir) if args.out_dir else output_dir / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_file:
        dataset_stem = Path(args.data_file).stem   # e.g. cladder-v1-q-easy
        run_id    = f"{run_name}__{dataset_stem}"
        records   = load_json_or_jsonl(args.data_file)
        print(f"[eval] data_file={args.data_file}  records={len(records)}")
    else:
        run_id    = f"{run_name}__{args.split}"
        splits_dir = Path(cfg["splits_dir"])
        records   = load_jsonl(str(splits_dir / f"{args.split}.jsonl"))
        print(f"[eval] split={args.split}  records={len(records)}")

    out_jsonl = out_dir / f"{run_id}.jsonl"
    if out_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"{out_jsonl} already exists. Use --overwrite.")
    print(f"[eval] checkpoint={checkpoint}")

    system_msg = get_system_msg(model_id)

    # ── Load tokenizer from adapter dir (it was saved there by train.py) ──────
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=cfg.get("trust_remote_code", True)
    )

    # ── Load base model + LoRA adapter ────────────────────────────────────────
    dtype_map   = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.get("dtype", "bf16"), torch.bfloat16)

    print(f"[eval] loading base model: {model_id} …")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )

    print(f"[eval] loading LoRA adapter: {checkpoint} …")
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    yes_ids, no_ids = get_yesno_ids(tokenizer)
    device = next(model.parameters()).device
    yes_t  = torch.tensor(yes_ids, device=device)
    no_t   = torch.tensor(no_ids,  device=device)

    # ── Run logit-mode inference ───────────────────────────────────────────────
    results = []
    for record in tqdm(records, desc=f"eval {run_id}"):
        prompt = build_prompt(tokenizer, record["text"], system_msg)
        enc    = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**enc).logits[0, -1, :]

        yes_score = logits[yes_t].max().item()
        no_score  = logits[no_t].max().item()
        pred      = "yes" if yes_score >= no_score else "no"

        results.append({
            "run_id":        run_id,
            "question_id":   record.get("question_id"),
            "model_id":      record.get("model_id"),
            "query_type":    record.get("query_type"),
            "gold":          record.get("answer"),
            "pred":          pred,
            "raw_response":  f"[logit] yes={yes_score:.4f} no={no_score:.4f}",
            "model_id_str":  model_id,
            "checkpoint":    checkpoint,
            "split":         Path(args.data_file).stem if args.data_file else args.split,
            "decision_mode": "logit",
        })

    with open(out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[eval] wrote {len(results)} predictions → {out_jsonl}")
    print(
        f"[eval] score with:\n"
        f"  python cladder_score_yesno.py \\\n"
        f"    --pred_jsonl {out_jsonl} \\\n"
        f"    --out_dir {out_dir / (run_id + '_score')}"
    )


if __name__ == "__main__":
    main()
