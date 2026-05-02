"""
Fast vLLM-based yes/no inference for CLADDER datasets.

Handles both base model and LoRA fine-tuned model via --lora_path.
Outputs JSONL compatible with cladder_score_yesno.py and plot_comparison.py.

Usage:
  # Base model
  python finetune/infer_vllm.py \
      --model  allenai/Olmo-3-7B-Instruct \
      --data_file data/cladder-v1-q-easy.json \
      --out_jsonl outputs/cladder-v1-q-easy/baseline/olmo3-7b-instruct-baseline/olmo3-7b-instruct-baseline.jsonl

  # Fine-tuned (LoRA)
  python finetune/infer_vllm.py \
      --model     allenai/Olmo-3-7B-Instruct \
      --lora_path finetune/checkpoints/olmo3-7b-instruct-lora/final_adapter \
      --data_file data/cladder-v1-q-easy.json \
      --out_jsonl outputs/cladder-v1-q-easy/finetuned/olmo3-7b-instruct-lora/olmo3-7b-instruct-lora__cladder-v1-q-easy.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ── Constants ─────────────────────────────────────────────────────────────────

SYSTEM_INSTRUCT = "Answer with only 'yes' or 'no'. No explanation. No punctuation."
SYSTEM_THINK = (
    "You must answer with exactly one token: yes or no.\n"
    "Do not provide any explanation, reasoning, analysis, or extra text.\n"
    "If you are unsure, still output yes or no."
)
# For reasoning/OSS models that ignore the system prompt and generate analysis:
# appending this suffix to the user message forces the answer token to come next.
USER_SUFFIX_REASONING = "\n\nRespond with one word only — yes or no:"

DEFAULT_BATCH_SIZE = 1000   # prompts per llm.generate() call — keeps RAM usage flat
DEFAULT_MAX_MODEL_LEN = 768  # CLADDER prompts are short; caps KV cache allocation
REASONING_MAX_TOKENS = 1024  # auto-mode reasoning models (legacy)
REASONING_MAX_MODEL_LEN = 2048
THINK_MAX_TOKENS = 4096      # dedicated thinking run — room for full CoT
THINK_MAX_MODEL_LEN = 5120   # prompt (~300 tok) + 4096 thinking + buffer
NOTHINK_MAX_TOKENS = 512     # non-thinking run — answer should come quickly


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path: str) -> List[Dict]:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl" or text.lstrip().startswith("{"):
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    return json.loads(text)


def get_system_msg(model_id: str) -> str:
    return SYSTEM_THINK if "think" in model_id.lower() else SYSTEM_INSTRUCT


def is_reasoning_model(model_id: str) -> bool:
    """Models that generate chain-of-thought and need a user-message suffix."""
    return "gpt-oss" in model_id.lower()


def build_prompt(record: Dict, tokenizer, system_msg: str,
                 thinking_mode: str = "auto") -> str:
    """
    thinking_mode:
      "think"   — enable_thinking=True passed to chat template
      "nothink" — enable_thinking=False + USER_SUFFIX_REASONING appended to
                  user message (prompt-level enforcement, since the flag is
                  silently ignored by some models)
      "auto"    — no enable_thinking kwarg; model uses its default behaviour
    """
    user_content = record["text"]
    if thinking_mode == "nothink":
        user_content = user_content + USER_SUFFIX_REASONING
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_content},
    ]
    if getattr(tokenizer, "chat_template", None):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if thinking_mode == "think":
            kwargs["enable_thinking"] = True
        elif thinking_mode == "nothink":
            kwargs["enable_thinking"] = False  # belt-and-suspenders; may be ignored
        return tokenizer.apply_chat_template(messages, **kwargs)
    return f"{system_msg}\n\n{record['text']}\n\nAnswer:"


def get_yesno_ids(tokenizer) -> Tuple[List[int], List[int]]:
    def single_ids(variants):
        seen, out = set(), []
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
                out.append(ids[0])
        return out
    yes_ids = single_ids([" yes", "yes", " Yes", "YES"])
    no_ids  = single_ids([" no",  "no",  " No",  "NO"])
    if not yes_ids or not no_ids:
        raise RuntimeError("Could not resolve single-token IDs for yes/no")
    return yes_ids, no_ids


def decode_output(output, record: Dict, yes_ids: List[int], no_ids: List[int],
                  model_id: str, run_id: str) -> Dict:
    # Scan all generated token positions for yes/no logprobs.
    # Some models (e.g. GPT-OSS) prepend a newline before the answer token,
    # so position 0 may not contain yes/no even though position 1 does.
    all_lps = output.outputs[0].logprobs or []
    yes_score = no_score = float("-inf")
    logit_pos = -1
    for pos, lp in enumerate(all_lps):
        ys = max((lp[tid].logprob for tid in yes_ids if tid in lp), default=float("-inf"))
        ns = max((lp[tid].logprob for tid in no_ids  if tid in lp), default=float("-inf"))
        if ys != float("-inf") or ns != float("-inf"):
            yes_score, no_score, logit_pos = ys, ns, pos
            break

    if yes_score == float("-inf") and no_score == float("-inf"):
        # Nothing found in logprobs — fall back to text match
        gen = output.outputs[0].text.strip().lower()
        pred = "yes" if gen.startswith("yes") else ("no" if gen.startswith("no") else None)
        raw  = f"[fallback] generated={gen!r}"
    else:
        pred = "yes" if yes_score >= no_score else "no"
        raw  = f"[logit@{logit_pos}] yes={yes_score:.4f} no={no_score:.4f}"

    return {
        "run_id":        run_id,
        "question_id":   record.get("question_id"),
        "model_id":      record.get("model_id"),
        "query_type":    record.get("query_type"),
        "gold":          record.get("answer"),
        "pred":          pred,
        "raw_response":  raw,
        "model_id_str":  model_id,
        "decision_mode": "logit",
    }


def decode_output_text(output, record: Dict, model_id: str, run_id: str) -> Dict:
    """Text-based decoder for reasoning models that generate CoT before the answer."""
    import re
    gen = output.outputs[0].text.strip()
    # Search for the last standalone yes/no in the generated text (after reasoning)
    matches = re.findall(r"\b(yes|no)\b", gen.lower())
    pred = matches[-1] if matches else None
    return {
        "run_id":        run_id,
        "question_id":   record.get("question_id"),
        "model_id":      record.get("model_id"),
        "query_type":    record.get("query_type"),
        "gold":          record.get("answer"),
        "pred":          pred,
        "raw_response":  gen,  # full text — reasoning traces need this
        "model_id_str":  model_id,
        "decision_mode": "generate",
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    llm: LLM,
    records: List[Dict],
    tokenizer,
    system_msg: str,
    yes_ids: List[int],
    no_ids: List[int],
    model_id: str,
    run_id: str,
    lora_request: Optional[LoRARequest],
    out_file,
    batch_size: int,
    thinking_mode: str = "auto",
) -> Tuple[int, int]:
    """Stream results directly to out_file in batches to avoid RAM accumulation.

    thinking_mode:
      "think"   — full CoT enabled; text-parse decoder; max_tokens=THINK_MAX_TOKENS
      "nothink" — CoT suppressed via prompt suffix; text-parse decoder; max_tokens=NOTHINK_MAX_TOKENS
      "auto"    — reasoning models (is_reasoning_model) → text-parse with REASONING_MAX_TOKENS;
                  all others → logit-mode
    """
    is_reasoning = is_reasoning_model(model_id)
    use_text_decode = thinking_mode in ("think", "nothink") or (thinking_mode == "auto" and is_reasoning)

    if thinking_mode == "think":
        sampling_params = SamplingParams(max_tokens=THINK_MAX_TOKENS, temperature=0)
    elif thinking_mode == "nothink":
        sampling_params = SamplingParams(max_tokens=NOTHINK_MAX_TOKENS, temperature=0)
    elif is_reasoning:  # auto + reasoning model — legacy behaviour
        sampling_params = SamplingParams(max_tokens=REASONING_MAX_TOKENS, temperature=0)
    else:
        sampling_params = SamplingParams(
            max_tokens=10,  # allow prefix tokens (newline etc.) before yes/no
            temperature=0,
            logprobs=20,  # vLLM v0.18 cap; yes/no always appear in top-20
        )

    n_total = n_correct = n_invalid = 0

    for start in tqdm(range(0, len(records), batch_size), desc="batches"):
        batch_records = records[start : start + batch_size]
        batch_prompts = [
            build_prompt(r, tokenizer, system_msg, thinking_mode=thinking_mode)
            for r in batch_records
        ]

        outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)

        for record, output in zip(batch_records, outputs):
            if use_text_decode:
                row = decode_output_text(output, record, model_id, run_id)
            else:
                row = decode_output(output, record, yes_ids, no_ids, model_id, run_id)
            out_file.write(json.dumps(row) + "\n")
            n_total += 1
            if row["pred"] is None:
                n_invalid += 1
            elif row["pred"] == row["gold"]:
                n_correct += 1

        out_file.flush()

    return n_total, n_correct, n_invalid


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         required=True, help="HF model id")
    ap.add_argument("--data_file",     required=True, help="JSON/JSONL data file")
    ap.add_argument("--out_jsonl",     required=True, help="Output JSONL path")
    ap.add_argument("--lora_path",     default=None,  help="Path to LoRA adapter dir (omit for base model)")
    ap.add_argument("--run_id",        default=None,  help="Override run_id written to each record")
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--batch_size",    type=int, default=DEFAULT_BATCH_SIZE,
                    help="Prompts per llm.generate() call (default: 1000)")
    ap.add_argument("--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LEN,
                    help="Max token length — caps KV cache; 768 is safe for CLADDER prompts")
    ap.add_argument("--thinking_mode", choices=["think", "nothink", "auto"], default="auto",
                    help="think=enable CoT (max_tokens=4096, full trace stored), "
                         "nothink=suppress CoT via prompt suffix+enable_thinking=False, "
                         "auto=detect from model name (default)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Limit to first N records (for smoke-testing before a full run)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_jsonl)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} already exists. Use --overwrite.")
    out.parent.mkdir(parents=True, exist_ok=True)

    dataset_stem = Path(args.data_file).stem
    run_id = args.run_id or (
        f"{Path(args.lora_path).parent.name}__{dataset_stem}"
        if args.lora_path else
        f"{args.model.split('/')[-1]}__{dataset_stem}"
    )

    records = load_data(args.data_file)
    if args.max_samples is not None:
        records = records[: args.max_samples]
        print(f"[infer] --max_samples={args.max_samples}: using first {len(records)} records")
    print(f"[infer] data_file={args.data_file}  records={len(records)}")
    print(f"[infer] run_id={run_id}  batch_size={args.batch_size}  max_model_len={args.max_model_len}")

    thinking_mode = args.thinking_mode
    system_msg = get_system_msg(args.model)

    # Warn if max_model_len is too small for the thinking run
    if thinking_mode == "think" and args.max_model_len < THINK_MAX_MODEL_LEN:
        print(f"[infer] WARNING: thinking mode with max_model_len={args.max_model_len} "
              f"may truncate reasoning traces. Recommend --max_model_len {THINK_MAX_MODEL_LEN}")

    if thinking_mode == "think":
        print("[infer] thinking mode ENABLED — full CoT; storing complete reasoning trace")
    elif thinking_mode == "nothink":
        print("[infer] thinking mode DISABLED — appending user suffix + enable_thinking=False")
    else:
        if is_reasoning_model(args.model):
            print("[infer] reasoning model detected (auto) — text-parse mode")

    # Always load tokenizer from the base model — LoRA doesn't change the tokenizer,
    # and the adapter's saved tokenizer_config may have version-incompatible fields.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    yes_ids, no_ids = get_yesno_ids(tokenizer)
    print(f"[infer] yes_ids={yes_ids}  no_ids={no_ids}")

    # Build LLM
    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,   # slightly conservative to leave headroom
        enforce_eager=True,            # avoids FakeTensorMode torch.compile issue
        max_model_len=args.max_model_len,
    )
    lora_request = None
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        lora_request = LoRARequest("adapter", 1, args.lora_path)
        print(f"[infer] LoRA adapter: {args.lora_path}")
    else:
        print(f"[infer] base model (no LoRA)")

    print(f"[infer] loading model: {args.model}")
    llm = LLM(**llm_kwargs)

    with open(out, "w") as f:
        n_total, n_correct, n_invalid = run_inference(
            llm=llm,
            records=records,
            tokenizer=tokenizer,
            system_msg=system_msg,
            yes_ids=yes_ids,
            no_ids=no_ids,
            model_id=args.model,
            run_id=run_id,
            lora_request=lora_request,
            out_file=f,
            batch_size=args.batch_size,
            thinking_mode=thinking_mode,
        )

    print(f"[infer] wrote {n_total} records → {out}")
    print(f"[infer] acc={n_correct/n_total:.4f}  invalid={n_invalid}")


if __name__ == "__main__":
    main()
