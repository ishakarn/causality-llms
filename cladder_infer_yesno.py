import argparse
import json
import os
import random
import re
from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Data loading / schema helpers
# -----------------------------

REQUIRED_KEYS = {
    "question_id",
    "model_id",
    "query_type",
    "background",
    "given_info",
    "question",
    "text",
    "answer",
}


def load_queries(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text())
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    if not isinstance(data, list):
        raise ValueError("Expected list of query objects or dict with key 'queries'.")
    return data


def validate_schema(records: List[Dict[str, Any]], strict: bool = True) -> None:
    missing_any = False
    for i, r in enumerate(records[:50]):  # cheap spot check
        missing = REQUIRED_KEYS - set(r.keys())
        if missing:
            missing_any = True
            msg = f"[schema] record {i} missing keys: {sorted(missing)}"
            if strict:
                raise ValueError(msg)
            else:
                print(msg)
    if not missing_any:
        print("[schema] looks good (spot-checked 50 records)")


def get_prompt(q: Dict[str, Any], prompt_field: str = "text") -> str:
    if prompt_field in q and isinstance(q[prompt_field], str):
        return q[prompt_field]
    # fallback
    for k in ["text", "question", "prompt", "query", "input"]:
        if k in q and isinstance(q[k], str):
            return q[k]
    return str(q)


# -----------------------------
# Run id / output naming
# -----------------------------

def slugify(s: str) -> str:
    """
    Make a filesystem-friendly slug.
    - Replace '/' with '__' so HF ids become stable filenames
    - Replace other junk with '_'
    """
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def default_run_id(model: str, queries_json: str, seed: int, tag: Optional[str] = None) -> str:
    dataset = Path(queries_json).stem  # e.g., queries_easy
    model_slug = slugify(model)
    rid = f"{dataset}__{model_slug}__seed{seed}"
    if tag:
        rid += f"__{slugify(tag)}"
    return rid

def slice_tag(start: int, limit: Optional[int]) -> Optional[str]:
    if start == 0 and limit is None:
        return None
    if limit is None:
        return f"start{start}"
    return f"start{start}_n{limit}"


def append_slice_suffix(name: str, slice_suffix: Optional[str]) -> str:
    if not slice_suffix:
        return name
    token = slugify(slice_suffix)
    if re.search(rf"(^|__){re.escape(token)}($|__)", name):
        return name
    return f"{name}__{token}"

def resolve_out_jsonl(
    out_jsonl: Optional[str],
    out_dir: str,
    run_id: str,
    slice_suffix: Optional[str] = None,
) -> str:
    """
    Priority:
      1) If --out_jsonl is provided:
         - if it's a directory => <dir>/<run_id>.jsonl
         - else use as-is
      2) Else => <out_dir>/<run_id>.jsonl
    """
    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if out_jsonl is None:
        return str(out_dir_p / f"{run_id}.jsonl")

    p = Path(out_jsonl).expanduser()
    # If user passed something ending with '/', treat it as a directory even if it doesn't exist yet
    if str(out_jsonl).endswith(os.sep) or (p.exists() and p.is_dir()):
        p.mkdir(parents=True, exist_ok=True)
        return str(p / f"{run_id}.jsonl")

    # Explicit file path: inject slice suffix into filename stem so sampled runs remain distinct
    stem = append_slice_suffix(p.stem, slice_suffix)
    suffix = p.suffix or ".jsonl"
    p = p.with_name(f"{stem}{suffix}")

    # If parent dir doesn't exist, create it
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


# -----------------------------
# Prompting + parsing
# -----------------------------

DEFAULT_SYSTEM_INSTRUCT = "Answer with only 'yes' or 'no'. No explanation. No punctuation."
DEFAULT_SYSTEM_THINK = (
    "You must answer with exactly one token: yes or no.\n"
    "Do not provide any explanation, reasoning, analysis, or extra text.\n"
    "If you are unsure, still output yes or no."
)
DEFAULT_SYSTEM = "AUTO"
DEFAULT_USER_PREFIX = ""

SYSTEM_ALIASES = {
    "DEFAULT_SYSTEM_INSTRUCT": DEFAULT_SYSTEM_INSTRUCT,
    "INSTRUCT": DEFAULT_SYSTEM_INSTRUCT,
    "DEFAULT_SYSTEM_THINK": DEFAULT_SYSTEM_THINK,
    "THINK": DEFAULT_SYSTEM_THINK,
}

YESNO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)
THINK_STRIP_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_yes_no(text: str) -> Optional[str]:
    """
    Robustly extract first yes/no token from the model output.
    Returns "yes" / "no" or None.
    """
    if not text:
        return None
    t = text.strip().lower()
    # first non-space token
    first = t.split()[0] if t.split() else ""
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    m = YESNO_RE.search(t)
    return m.group(1).lower() if m else None


def resolve_system_prompt(system_arg: str, model_name: str) -> str:
    raw = (system_arg or "").strip()
    if raw.upper() == DEFAULT_SYSTEM:
        return DEFAULT_SYSTEM_THINK if "think" in model_name.lower() else DEFAULT_SYSTEM_INSTRUCT

    alias = SYSTEM_ALIASES.get(raw) or SYSTEM_ALIASES.get(raw.upper())
    return alias if alias is not None else system_arg


def build_chat_input(tokenizer, prompt: str, system_msg: str,
                     user_prefix: str = "", thinking: bool = False) -> str:
    """
    Return a string that is ready to be tokenized.
    Uses chat template only when tokenizer.chat_template is set; otherwise falls back.
    Pass thinking=True to leave Qwen3 thinking mode enabled (default is off).
    """
    user_content = (user_prefix + "\n" if user_prefix else "") + prompt

    has_template = bool(getattr(tokenizer, "chat_template", None))
    if has_template:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
        kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
        # Qwen3 thinking is ON by default; disable unless explicitly requested
        if "enable_thinking" in (tokenizer.chat_template or ""):
            kwargs["enable_thinking"] = thinking
        return tokenizer.apply_chat_template(messages, **kwargs)

    # Base model fallback (no chat template)
    return f"{system_msg}\n\n{user_content}\n\nAnswer:"


# -----------------------------
# Model loading
# -----------------------------

@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: str


def load_model(model_name: str, dtype: str = "bf16") -> ModelBundle:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = torch.float32

    print(f"[load] tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Avoid "Setting pad_token_id to eos_token_id" spam during generate()
    # if tok.pad_token_id is None and tok.eos_token_id is not None:
    #     tok.pad_token = tok.eos_token

    print(f"[load] model: {model_name} (device={device}, dtype={torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    if device == "cuda":
        print(f"[load] gpu: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"[load] cuda mem free/total (GB): {free/1e9:.2f}/{total/1e9:.2f}")

    return ModelBundle(tokenizer=tok, model=model, device=device)


# -----------------------------
# Inference
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_one(
    mb: ModelBundle,
    prompt: str,
    system_msg: str,
    user_prefix: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    decision_mode: str,
    thinking: bool = False,
) -> Tuple[str, str]:
    """
    Returns:
      raw_generation (decoded new tokens only),
      parsed_pred ("yes"/"no"/None as string in second return)
    """
    tok, model = mb.tokenizer, mb.model

    rendered = build_chat_input(tok, prompt, system_msg=system_msg,
                                user_prefix=user_prefix, thinking=thinking)
    enc = tok(rendered, return_tensors="pt").to(model.device)

    if decision_mode == "logit":
        yes_variants = [" yes", "yes", " Yes", "YES"]
        no_variants = [" no", "no", " No", "NO"]

        def single_token_ids(variants: List[str]) -> List[int]:
            out = []
            for v in variants:
                ids = tok(v, add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    out.append(ids[0])
            # de-dup preserving order
            seen = set()
            uniq = []
            for i in out:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            return uniq

        yes_ids = single_token_ids(yes_variants)
        no_ids = single_token_ids(no_variants)
        if not yes_ids or not no_ids:
            raise RuntimeError("Could not resolve single-token ids for yes/no with this tokenizer")

        with torch.no_grad():
            logits = model(**enc).logits[0, -1, :]

        yes_score = torch.max(logits[yes_ids]).item()
        no_score = torch.max(logits[no_ids]).item()
        pred = "yes" if yes_score >= no_score else "no"
        raw = f"[logit] yes={yes_score:.4f} no={no_score:.4f}"
        return raw, pred

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p,
    )

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    # Decode only the generated continuation (not the prompt/template)
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    raw = tok.decode(gen_ids, skip_special_tokens=True).strip()

    # For thinking mode strip <think>...</think> before parsing yes/no
    parse_text = THINK_STRIP_RE.sub("", raw).strip() if thinking else raw
    pred = parse_yes_no(parse_text)
    return raw, (pred or "")


def run_inference(
    mb: ModelBundle,
    records: List[Dict[str, Any]],
    out_jsonl: str,
    run_id: str,
    model_id_str: str,
    prompt_field: str,
    system_msg: str,
    user_prefix: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    decision_mode: str,
    limit: Optional[int],
    start: int,
    thinking: bool = False,
) -> None:
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subset = records[start:]
    if limit is not None:
        subset = subset[:limit]

    print(f"[run] run_id: {run_id}")
    print(f"[run] writing: {out_path}")
    print(f"[run] records: {len(subset)} (start={start}, limit={limit})")

    with out_path.open("w") as f:
        for i, q in enumerate(tqdm(subset, desc=f"infer {run_id}", total=len(subset))):
            prompt = get_prompt(q, prompt_field=prompt_field)
            raw, pred = generate_one(
                mb,
                prompt=prompt,
                system_msg=system_msg,
                user_prefix=user_prefix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                decision_mode=decision_mode,
                thinking=thinking,
            )

            rec = {
                "run_id": run_id,
                "idx": start + i,
                "question_id": q.get("question_id"),
                "model_id": q.get("model_id"),
                "query_type": q.get("query_type"),
                "gold": q.get("answer"),
                "pred": pred if pred != "" else None,
                "raw_response": raw,
                # both are useful: HF id + what transformers reports as name_or_path
                "model_id_str": model_id_str,
                "model_name_or_path": getattr(mb.model.config, "name_or_path", None) or "unknown",
                # keep prompt settings for reproducibility
                "prompt_field": prompt_field,
                "system": system_msg,
                "user_prefix": user_prefix,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "decision_mode": decision_mode,
                "seed": None,  # filled in main if you want (kept for backwards compat)
            }
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--queries_json", required=True, help="Path to queries_easy.json")

    # NEW: output handling
    ap.add_argument("--out_dir", default="outputs", help="Directory to write outputs (default: outputs/)")
    ap.add_argument("--run_id", default=None, help="Optional explicit run id (otherwise auto-generated)")
    ap.add_argument("--tag", default=None, help="Optional extra tag appended to auto run_id (e.g., think, dpo)")
    ap.add_argument(
        "--out_jsonl",
        default=None,
        help="Optional explicit output path. If omitted, uses <out_dir>/<run_id>.jsonl. "
             "If a directory path (or endswith /), writes <dir>/<run_id>.jsonl.",
    )

    ap.add_argument("--prompt_field", default="text", help="Which field to use as prompt (default: text)")
    ap.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help=(
            "System instruction for yes/no. "
            "Use AUTO (default), DEFAULT_SYSTEM_INSTRUCT, DEFAULT_SYSTEM_THINK, or custom text."
        ),
    )
    ap.add_argument("--user_prefix", default=DEFAULT_USER_PREFIX, help="Optional prefix inserted before prompt")

    ap.add_argument("--max_new_tokens", type=int, default=8, help="Small is enough for yes/no")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument(
        "--decision_mode",
        choices=["auto", "generate", "logit"],
        default="auto",
        help="How to produce yes/no: auto (Think->logit, else generate), generate, or logit.",
    )
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    ap.add_argument("--limit", type=int, default=None, help="Run only first N examples (after --start)")
    ap.add_argument("--start", type=int, default=0, help="Start offset into dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing out_jsonl if it exists")
    ap.add_argument("--thinking", action="store_true",
                    help="Enable chain-of-thought thinking (Qwen3). Strips <think>...</think> before parsing yes/no.")

    ap.add_argument("--no_strict_schema", action="store_true", help="Don’t fail if schema differs")
    args = ap.parse_args()

    set_seed(args.seed)

    args.system = resolve_system_prompt(args.system, args.model)
    if args.decision_mode == "auto":
        args.decision_mode = "logit" if "think" in args.model.lower() else "generate"

    records = load_queries(args.queries_json)
    validate_schema(records, strict=(not args.no_strict_schema))

    # auto-tag slice (limit/start) so small runs don't overwrite full runs
    auto_slice = slice_tag(args.start, args.limit)

    # Build base run_id (auto or explicit), then always append slice suffix when sampling.
    base_run_id = args.run_id or default_run_id(args.model, args.queries_json, args.seed, args.tag)
    run_id = append_slice_suffix(base_run_id, auto_slice)
    out_jsonl = resolve_out_jsonl(args.out_jsonl, args.out_dir, run_id, slice_suffix=auto_slice)

    # don't silently overwrite
    out_path = Path(out_jsonl)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {out_jsonl}\n"
            f"Use --overwrite to overwrite, or pass --run_id/--tag to create a new name."
        )

    print(f"[run] model: {args.model}")
    print(f"[run] run_id: {run_id}")
    print(f"[run] out_jsonl: {out_jsonl}")
    print(f"[run] decision_mode: {args.decision_mode}")

    mb = load_model(args.model, dtype=args.dtype)

    run_inference(
        mb=mb,
        records=records,
        out_jsonl=out_jsonl,
        run_id=run_id,
        model_id_str=args.model,
        prompt_field=args.prompt_field,
        system_msg=args.system,
        user_prefix=args.user_prefix,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        decision_mode=args.decision_mode,
        limit=args.limit,
        start=args.start,
        thinking=args.thinking,
    )


if __name__ == "__main__":
    main()