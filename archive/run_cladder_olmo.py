import argparse, json, os
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_queries(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text())
    # Accept either a dict with "queries" or a raw list
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    if not isinstance(data, list):
        raise ValueError("Expected list of queries or dict with key 'queries'.")
    return data


def extract_prompt(q: Dict[str, Any]) -> str:
    # Try common keys; fall back to string casting
    for k in ["prompt", "query", "question", "input", "text"]:
        if k in q and isinstance(q[k], str):
            return q[k]
    # Sometimes CLADDER entries store as {"messages":[...]}
    if "messages" in q and isinstance(q["messages"], list):
        # if already chat-like, let caller handle; for now collapse into last user turn
        for m in reversed(q["messages"]):
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                return m["content"]
    return str(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--queries_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    queries = load_queries(args.queries_json)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def format_input(prompt: str) -> torch.Tensor:
        # Prefer chat template if available
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return tok(text, return_tensors="pt").to(model.device)
        else:
            return tok(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
    )

    with out_path.open("w") as f:
        for i, q in enumerate(queries):
            prompt = extract_prompt(q)
            enc = format_input(prompt)
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            text = tok.decode(out[0], skip_special_tokens=True)

            rec = {
                "idx": i,
                "id": q.get("id", q.get("qid", None)),
                "prompt": prompt,
                "response": text,
                "meta": {k: q[k] for k in q.keys() if k not in {"prompt","query","question","input","text","messages"}},
                "model": args.model,
            }
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()