"""
Convert OpenAI Batch API response files for GPT-5-nano into the JSONL format
expected by cladder_score_yesno.py and plot_all_models.py.

Usage:
    python finetune/convert_gpt5nano.py \
        --responses data/gpt-5-nano_responses/responses_queries_easy_all_batched_gpt-5-nano.jsonl \
        --data      data/cladder-v1-q-easy.json \
        --out       outputs/cladder-v1-q-easy/baseline/gpt-5-nano-baseline/gpt-5-nano-baseline.jsonl
"""

import argparse
import json
import re
from pathlib import Path


RUN_ID = "gpt-5-nano-baseline"
MODEL_ID_STR = "gpt-5-nano-2025-08-07"


def extract_text(response_body: dict) -> str:
    """Pull the assistant message text out of an OpenAI Responses API body."""
    for item in response_body.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text", "")
    return ""


def parse_yesno(text: str):
    """Return 'yes', 'no', or None."""
    t = text.strip().lower()
    # exact match first
    if t in ("yes", "no"):
        return t
    # starts-with match
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    # search for standalone yes/no word
    m = re.search(r"\b(yes|no)\b", t)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="Batch API JSONL response file")
    ap.add_argument("--data",      required=True, help="Original data JSON (for query_type/gold)")
    ap.add_argument("--out",       required=True, help="Output JSONL path")
    args = ap.parse_args()

    # Build lookup: question_id -> {query_type, answer, model_id}
    data = json.loads(Path(args.data).read_text())
    lookup = {str(r["question_id"]): r for r in data}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_total = n_correct = n_invalid = 0

    with open(args.responses) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            batch_rec = json.loads(line)
            custom_id = str(batch_rec["custom_id"])
            orig = lookup.get(custom_id, {})

            body = batch_rec.get("response", {}).get("body", {})
            raw_text = extract_text(body)
            pred = parse_yesno(raw_text)

            row = {
                "run_id":        RUN_ID,
                "question_id":   int(custom_id) if custom_id.isdigit() else custom_id,
                "model_id":      orig.get("model_id"),
                "query_type":    orig.get("query_type"),
                "gold":          orig.get("answer"),
                "pred":          pred,
                "raw_response":  raw_text.strip(),
                "model_id_str":  MODEL_ID_STR,
                "decision_mode": "generate",
            }
            fout.write(json.dumps(row) + "\n")
            n_total += 1
            if pred is None:
                n_invalid += 1
            elif pred == orig.get("answer"):
                n_correct += 1

    acc = n_correct / n_total if n_total else 0
    print(f"  records={n_total}  correct={n_correct}  invalid={n_invalid}  acc={acc:.4f}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
