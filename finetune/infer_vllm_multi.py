"""
Multi-file vLLM inference — loads the model once, runs N data files sequentially.

Identical inference logic to infer_vllm.py but accepts a list of (data_file,
out_jsonl) pairs so the expensive model-load step happens only once.

Usage:
  python finetune/infer_vllm_multi.py \
      --model allenai/OLMo-3.1-32B-Instruct \
      --pairs data/file1.json:outputs/file1.jsonl \
              data/file2.json:outputs/file2.jsonl \
      [--lora_path finetune/checkpoints/graded/olmo32b_n2000_lora/final_adapter] \
      [--max_model_len 8192]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ── Reuse constants and helpers from infer_vllm ────────────────────────────────
SYSTEM_INSTRUCT = "Answer with only 'yes' or 'no'. No explanation. No punctuation."
SYSTEM_THINK = (
    "You must answer with exactly one token: yes or no.\n"
    "Do not provide any explanation, reasoning, analysis, or extra text.\n"
    "If you are unsure, still output yes or no."
)
USER_SUFFIX_REASONING = "\n\nRespond with one word only — yes or no:"
DEFAULT_MAX_MODEL_LEN = 4096
REASONING_MAX_TOKENS = 1024


def load_data(path: str) -> List[Dict]:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl" or text.lstrip().startswith("{"):
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]
    return data


def is_reasoning_model(model_id: str) -> bool:
    return "gpt-oss" in model_id.lower()


def get_system_msg(model_id: str) -> str:
    return SYSTEM_THINK if "think" in model_id.lower() else SYSTEM_INSTRUCT


def build_prompt(record: Dict, tokenizer, system_msg: str,
                 reasoning: bool = False) -> str:
    user_content = record["text"]
    if reasoning:
        user_content = user_content + USER_SUFFIX_REASONING
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_content},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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


def decode_output_text(output, record: Dict, model_id: str, run_id: str) -> Dict:
    import re
    gen = output.outputs[0].text.strip()
    matches = re.findall(r"\b(yes|no)\b", gen.lower())
    pred = matches[-1] if matches else None
    return {
        "run_id":        run_id,
        "question_id":   record.get("question_id"),
        "model_id":      record.get("model_id"),
        "query_type":    record.get("query_type"),
        "gold":          record.get("answer"),
        "pred":          pred,
        "raw_response":  gen,
        "model_id_str":  model_id,
        "decision_mode": "generate",
    }


def decode_output(output, record: Dict, yes_ids: List[int], no_ids: List[int],
                  model_id: str, run_id: str) -> Dict:
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
        gen  = output.outputs[0].text.strip().lower()
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


def run_one_file(
    llm: LLM,
    data_file: str,
    out_jsonl: str,
    run_id: str,
    tokenizer,
    system_msg: str,
    yes_ids: List[int],
    no_ids: List[int],
    model_id: str,
    lora_request: Optional[LoRARequest],
    sampling_params: SamplingParams,
    batch_size: int,
    overwrite: bool,
    reasoning: bool = False,
) -> None:
    out = Path(out_jsonl)
    if out.exists() and not overwrite:
        # Read existing to report acc
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        n_correct = sum(1 for r in rows if r.get("pred") == r.get("gold"))
        print(f"  [skip] {data_file} → already exists (acc={n_correct/len(rows):.3f})")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    records = load_data(data_file)
    print(f"  [infer] {data_file}  ({len(records)} records)")

    n_total = n_correct = n_invalid = 0
    with open(out, "w") as f:
        for start in tqdm(range(0, len(records), batch_size), desc=Path(data_file).stem, leave=False):
            batch = records[start : start + batch_size]
            prompts = [build_prompt(r, tokenizer, system_msg, reasoning=reasoning) for r in batch]
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            for record, output in zip(batch, outputs):
                if reasoning:
                    row = decode_output_text(output, record, model_id, run_id)
                else:
                    row = decode_output(output, record, yes_ids, no_ids, model_id, run_id)
                f.write(json.dumps(row) + "\n")
                n_total += 1
                if row["pred"] is None:
                    n_invalid += 1
                elif row["pred"] == row["gold"]:
                    n_correct += 1
            f.flush()

    acc = n_correct / n_total if n_total else 0
    print(f"  → wrote {n_total} records  acc={acc:.4f}  invalid={n_invalid}  → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         required=True)
    ap.add_argument("--pairs",         required=True, nargs="+",
                    help="data_file:out_jsonl pairs, colon-separated")
    ap.add_argument("--run_ids",       nargs="+", default=None,
                    help="Optional run_id per pair (same order). Defaults to derived from filenames.")
    ap.add_argument("--lora_path",     default=None)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--batch_size",    type=int, default=500)
    ap.add_argument("--overwrite",     action="store_true")
    args = ap.parse_args()

    # Parse pairs
    pairs = []
    for p in args.pairs:
        if ":" not in p:
            raise ValueError(f"Each --pairs entry must be 'data_file:out_jsonl', got: {p!r}")
        data_file, out_jsonl = p.split(":", 1)
        pairs.append((data_file, out_jsonl))

    run_ids = args.run_ids or [None] * len(pairs)
    if len(run_ids) != len(pairs):
        raise ValueError("--run_ids must have the same number of entries as --pairs")

    reasoning = is_reasoning_model(args.model)

    print(f"[multi-infer] model:         {args.model}")
    print(f"[multi-infer] lora_path:     {args.lora_path or '(none)'}")
    print(f"[multi-infer] max_model_len: {args.max_model_len}")
    print(f"[multi-infer] reasoning:     {reasoning}")
    print(f"[multi-infer] files to run:  {len(pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    yes_ids, no_ids = get_yesno_ids(tokenizer)
    system_msg = get_system_msg(args.model)
    print(f"[multi-infer] yes_ids={yes_ids}  no_ids={no_ids}")

    # Check which files still need processing (before loading the model)
    pending = []
    for (data_file, out_jsonl), rid in zip(pairs, run_ids):
        score_file = Path(out_jsonl).parent / "score" / "summary.json"
        if score_file.exists():
            acc = json.loads(score_file.read_text()).get("acc_all", "?")
            print(f"  [skip] {data_file}  (already scored, acc={acc:.3f})")
        elif Path(out_jsonl).exists() and not args.overwrite:
            print(f"  [skip] {data_file}  (jsonl exists, no score yet — will score after)")
            pending.append((data_file, out_jsonl, rid, True))  # skip_infer=True
        else:
            pending.append((data_file, out_jsonl, rid, False))

    if not pending:
        print("[multi-infer] Nothing to do — all files already scored.")
        return

    # Build LLM
    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )
    lora_request = None
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        lora_request = LoRARequest("adapter", 1, args.lora_path)
        print(f"[multi-infer] LoRA adapter: {args.lora_path}")
    else:
        print(f"[multi-infer] base model (no LoRA)")

    print(f"[multi-infer] loading model…")
    llm = LLM(**llm_kwargs)

    if reasoning:
        sampling_params = SamplingParams(max_tokens=REASONING_MAX_TOKENS, temperature=0)
    else:
        sampling_params = SamplingParams(max_tokens=10, temperature=0, logprobs=20)

    for data_file, out_jsonl, rid, skip_infer in pending:
        if rid is None:
            stem = Path(data_file).stem
            rid = (
                f"{Path(args.lora_path).parent.name}__{stem}"
                if args.lora_path else
                f"{args.model.split('/')[-1]}__{stem}"
            )
        if not skip_infer:
            run_one_file(
                llm=llm,
                data_file=data_file,
                out_jsonl=out_jsonl,
                run_id=rid,
                tokenizer=tokenizer,
                system_msg=system_msg,
                yes_ids=yes_ids,
                no_ids=no_ids,
                model_id=args.model,
                lora_request=lora_request,
                sampling_params=sampling_params,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                reasoning=reasoning,
            )

    print(f"\n[multi-infer] Done — {len(pending)} file(s) processed.")


if __name__ == "__main__":
    main()
