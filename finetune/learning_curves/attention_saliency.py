"""
Gradient saliency + raw attention analysis for the yes/no logit decision.

For N val examples this script computes:
  1. Gradient × input saliency  — which tokens drive logit(yes) - logit(no)
  2. Raw attention weights at the last prompt position (the position used to
     read yes/no logits), per layer and per head
  3. Attention entropy per layer — low entropy = peaked / attention-sink-like

Outputs (all in --out_dir):
  saliency.csv          — per-example, per-token normalized saliency score
  attn_entropy.csv      — per-example, per-layer attention entropy (avg over heads)
  attn_topk.csv         — per-example, per-layer: top-k tokens attended to at last pos
  attn_weights_raw.csv  — per-example, per-layer, per-head: full last-position attn row
                          (one row per head; can be large — gated by --save_raw)

Usage:
    # Fine-tuned checkpoint (default: final_adapter)
    python finetune/learning_curves/attention_saliency.py \\
        --model qwen25-3b-instruct --n_examples 20

    # Base model only (no adapter)
    python finetune/learning_curves/attention_saliency.py \\
        --model qwen25-3b-instruct --base_only --n_examples 20

    # Specific checkpoint
    python finetune/learning_curves/attention_saliency.py \\
        --model qwen25-3b-instruct \\
        --checkpoint finetune/checkpoints/qwen25-3b-instruct-lora/checkpoint-144

Requires a GPU. On Unity HPC run via srun or a short SLURM job.
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Point HF to scratch cache if not already set by the calling environment
# (SLURM scripts set this; interactive runs would otherwise hit ~/.cache)
_WS = "${SCRATCH_CACHE:-/scratch/workspace/$(whoami)-cladder-cache}"
os.environ.setdefault("HF_HOME",             f"{_WS}/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE",  f"{_WS}/.cache/huggingface/hub")
os.environ.setdefault("HF_DATASETS_CACHE",  f"{_WS}/.cache/huggingface/datasets")

import torch
import yaml

ROOT     = Path(__file__).resolve().parents[2]
FINETUNE = ROOT / "finetune"

MODELS = {
    "olmo3-7b-instruct": {
        "config":           FINETUNE / "configs/olmo3-7b-instruct.yaml",
        "checkpoints_dir":  FINETUNE / "checkpoints/olmo3-7b-instruct-lora",
    },
    "olmo3-7b-think": {
        "config":           FINETUNE / "configs/olmo3-7b-think.yaml",
        "checkpoints_dir":  FINETUNE / "checkpoints/olmo3-7b-think-lora",
    },
    "qwen25-3b-instruct": {
        "config":           FINETUNE / "configs/qwen25-3b-instruct.yaml",
        "checkpoints_dir":  FINETUNE / "checkpoints/qwen25-3b-instruct-lora",
    },
}

TOP_K = 5   # number of top attended tokens to report per layer


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path) -> List[Dict]:
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
    def single_token_ids(variants):
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
        raise RuntimeError("Could not resolve single-token ids for yes/no")
    return yes_ids, no_ids


def attention_entropy(attn_row: torch.Tensor) -> float:
    """Shannon entropy of a probability vector (attention weights for one position).

    Low entropy → peaked / attention-sink-like.
    Maximum entropy = log(seq_len).
    """
    p = attn_row.clamp(min=1e-9)
    return -(p * p.log()).sum().item()


# ── Per-example analysis ───────────────────────────────────────────────────────

def analyse_example(
    record: Dict,
    model,
    tokenizer,
    system_msg: str,
    yes_t: torch.Tensor,
    no_t: torch.Tensor,
    device: torch.device,
    save_raw: bool,
) -> Dict:
    """
    Returns a dict with keys:
      tokens         list[str]
      saliency       list[float]  normalised abs(grad*embed) per token
      layer_entropy  list[float]  one per layer, avg over heads
      layer_topk     list[list[dict]]  one per layer, TOP_K entries {rank, token, pos, weight}
      layer_head_attn list[list[list[float]]]  [layer][head][pos]  (only if save_raw)
      gold           str
      pred           str
      yes_logit      float
      no_logit       float
    """
    prompt = build_prompt(tokenizer, record["text"], system_msg)
    enc    = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids   = enc["input_ids"]           # (1, L)
    attn_mask   = enc["attention_mask"]      # (1, L)
    tokens      = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    L           = input_ids.shape[1]

    # ── 1. Gradient × input saliency ─────────────────────────────────────────
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids).detach().requires_grad_(True)  # (1, L, d)

    # forward with inputs_embeds so grad flows only through this pass
    out = model(inputs_embeds=embeds, attention_mask=attn_mask)
    logits_last = out.logits[0, -1, :]                  # (vocab,)
    yes_score   = logits_last[yes_t].max()
    no_score    = logits_last[no_t].max()
    decision    = yes_score - no_score                  # positive → pred yes
    decision.backward()

    # (grad * embed).sum(d) → scalar per token; take abs and normalise
    with torch.no_grad():
        sal = (embeds.grad * embeds).sum(-1).squeeze(0)  # (L,)
        sal = sal.abs()
        sal = (sal / sal.sum()).cpu().tolist()

    # ── 2. Attention weights ──────────────────────────────────────────────────
    with torch.no_grad():
        out_attn = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_attentions=True,
        )

    # out_attn.attentions: tuple of n_layers tensors, each (1, heads, L, L)
    layer_entropy  = []
    layer_topk     = []
    layer_head_attn = [] if save_raw else None

    for layer_idx, attn_tensor in enumerate(out_attn.attentions):
        # attn_tensor: (1, heads, L, L) — last dim is "key" positions
        heads_attn = attn_tensor[0, :, -1, :].cpu()  # (heads, L)  last query pos

        # per-head entropy
        head_entropies = [attention_entropy(heads_attn[h]) for h in range(heads_attn.shape[0])]
        avg_entropy    = sum(head_entropies) / len(head_entropies)
        layer_entropy.append(avg_entropy)

        # top-k tokens (averaged over heads)
        avg_attn = heads_attn.mean(0)  # (L,)
        topk_vals, topk_idx = avg_attn.topk(min(TOP_K, L))
        topk = [
            {
                "rank":   rank + 1,
                "pos":    int(idx),
                "token":  tokens[idx] if idx < len(tokens) else "?",
                "weight": float(val),
            }
            for rank, (idx, val) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()))
        ]
        layer_topk.append(topk)

        if save_raw:
            layer_head_attn.append(heads_attn.tolist())  # [heads][L]

    pred = "yes" if yes_score.item() >= no_score.item() else "no"

    return {
        "tokens":           tokens,
        "saliency":         sal,
        "layer_entropy":    layer_entropy,
        "layer_topk":       layer_topk,
        "layer_head_attn":  layer_head_attn,
        "gold":             record.get("answer", "?"),
        "pred":             pred,
        "yes_logit":        yes_score.item(),
        "no_logit":         no_score.item(),
        "question_id":      record.get("question_id", "?"),
        "query_type":       record.get("query_type", "?"),
    }


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_saliency_csv(results: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["example_idx", "question_id", "query_type", "gold", "pred",
                    "pos", "token", "saliency"])
        for i, r in enumerate(results):
            for pos, (tok, sal) in enumerate(zip(r["tokens"], r["saliency"])):
                w.writerow([i, r["question_id"], r["query_type"],
                             r["gold"], r["pred"], pos, tok, f"{sal:.6f}"])
    print(f"  wrote {path}")


def write_entropy_csv(results: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        n_layers = len(results[0]["layer_entropy"]) if results else 0
        w.writerow(["example_idx", "question_id", "query_type", "gold", "pred",
                    "layer", "entropy_avg_heads", "max_possible_entropy"])
        for i, r in enumerate(results):
            L = len(r["tokens"])
            max_ent = math.log(L) if L > 1 else 1.0
            for layer_idx, ent in enumerate(r["layer_entropy"]):
                w.writerow([i, r["question_id"], r["query_type"],
                             r["gold"], r["pred"],
                             layer_idx, f"{ent:.6f}", f"{max_ent:.6f}"])
    print(f"  wrote {path}")


def write_topk_csv(results: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["example_idx", "question_id", "query_type", "gold", "pred",
                    "layer", "rank", "pos", "token", "attn_weight"])
        for i, r in enumerate(results):
            for layer_idx, topk in enumerate(r["layer_topk"]):
                for entry in topk:
                    w.writerow([i, r["question_id"], r["query_type"],
                                 r["gold"], r["pred"],
                                 layer_idx, entry["rank"], entry["pos"],
                                 entry["token"], f"{entry['weight']:.6f}"])
    print(f"  wrote {path}")


def write_raw_attn_csv(results: List[Dict], path: Path) -> None:
    """Write full per-head attention rows. Can be large."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["example_idx", "question_id", "layer", "head", "pos", "token", "attn_weight"])
        for i, r in enumerate(results):
            if r["layer_head_attn"] is None:
                continue
            tokens = r["tokens"]
            for layer_idx, heads in enumerate(r["layer_head_attn"]):
                for head_idx, attn_row in enumerate(heads):
                    for pos, weight in enumerate(attn_row):
                        tok = tokens[pos] if pos < len(tokens) else "?"
                        w.writerow([i, r["question_id"], layer_idx, head_idx,
                                    pos, tok, f"{weight:.6f}"])
    print(f"  wrote {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      choices=list(MODELS.keys()), required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="LoRA adapter path. Defaults to final_adapter. Ignored with --base_only.")
    ap.add_argument("--base_only",  action="store_true",
                    help="Load base model without any LoRA adapter (for comparison).")
    ap.add_argument("--n_examples", type=int, default=20,
                    help="Number of val examples to analyse (default 20).")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--out_dir",    default=None,
                    help="Output directory. Defaults to learning_curves/attention_analysis/<model>/")
    ap.add_argument("--save_raw",   action="store_true",
                    help="Also write full per-head attention weights (large file).")
    args = ap.parse_args()

    cfg_path = MODELS[args.model]["config"]
    cfg      = yaml.safe_load(Path(cfg_path).read_text())
    model_id = cfg["model_id"]

    ckpt_dir = MODELS[args.model]["checkpoints_dir"]
    if args.base_only:
        checkpoint = None
        label = "base"
    else:
        checkpoint = args.checkpoint or str(ckpt_dir.parent / (ckpt_dir.name.replace("-lora", "-lora")) / "final_adapter")
        # resolve default: <checkpoints_dir>/final_adapter
        if args.checkpoint is None:
            checkpoint = str(ckpt_dir / "final_adapter")
        label = "finetuned"

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).parent / "attention_analysis" / f"{args.model}_{label}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    val_path = FINETUNE / "splits/val.jsonl"
    records  = load_jsonl(str(val_path))
    random.seed(args.seed)
    sample   = random.sample(records, min(args.n_examples, len(records)))
    print(f"[setup] {len(sample)} examples from {val_path}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[setup] loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=cfg.get("trust_remote_code", True)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype_map   = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.get("dtype", "bf16"), torch.bfloat16)

    print(f"[setup] loading base model {model_id} ({cfg.get('dtype','bf16')}) …")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
        attn_implementation="eager",   # SDPA/flash do not return attention weights
    )

    if checkpoint is not None:
        from peft import PeftModel
        print(f"[setup] loading LoRA adapter from {checkpoint} …")
        model = PeftModel.from_pretrained(base_model, checkpoint)
    else:
        model = base_model
        print("[setup] using base model (no adapter)")

    model.eval()
    device  = next(model.parameters()).device
    yes_ids, no_ids = get_yesno_ids(tokenizer)
    yes_t   = torch.tensor(yes_ids, device=device)
    no_t    = torch.tensor(no_ids,  device=device)
    system_msg = get_system_msg(model_id)

    print(f"[setup] yes_ids={yes_ids}  no_ids={no_ids}")
    print(f"[setup] device={device}  label={label}")

    # ── Analyse ───────────────────────────────────────────────────────────────
    results = []
    for idx, record in enumerate(sample):
        print(f"  [{idx+1}/{len(sample)}] qid={record.get('question_id','?')}  "
              f"type={record.get('query_type','?')}  gold={record.get('answer','?')}")
        r = analyse_example(
            record, model, tokenizer, system_msg,
            yes_t, no_t, device, save_raw=args.save_raw,
        )
        results.append(r)
        print(f"    pred={r['pred']}  yes_logit={r['yes_logit']:.3f}  "
              f"no_logit={r['no_logit']:.3f}")

    # ── Write CSVs ────────────────────────────────────────────────────────────
    write_saliency_csv(results, out_dir / "saliency.csv")
    write_entropy_csv (results, out_dir / "attn_entropy.csv")
    write_topk_csv    (results, out_dir / "attn_topk.csv")
    if args.save_raw:
        write_raw_attn_csv(results, out_dir / "attn_weights_raw.csv")

    # Print a quick summary: mean entropy per layer across examples
    if results:
        n_layers = len(results[0]["layer_entropy"])
        print(f"\n[summary] mean attention entropy at last position (avg over {len(results)} examples):")
        print(f"  {'layer':>6}  {'entropy':>10}  {'normalised':>10}")
        L = len(results[0]["tokens"])
        max_ent = math.log(L) if L > 1 else 1.0
        for layer in range(n_layers):
            vals = [r["layer_entropy"][layer] for r in results if layer < len(r["layer_entropy"])]
            mean_ent = sum(vals) / len(vals)
            print(f"  {layer:>6}  {mean_ent:>10.4f}  {mean_ent/max_ent:>10.4f}")

    print(f"\n[done] output in {out_dir}")


if __name__ == "__main__":
    main()
