"""
Convert gpt-5.4 batch API response JSONL files into the standard prediction
format expected by cladder_score_yesno.py.

Input:  data/gpt-5.4_responses/responses_queries_{split}_batched_gpt-5.4.json
Output: outputs/cladder-v1-q-{split}/baseline/gpt54-baseline/gpt54-baseline.jsonl

Usage:
    python finetune/convert_gpt54_responses.py [--splits easy hard anticommonsense]
"""

import argparse
import json
from pathlib import Path

RUN_ID      = "gpt54-baseline"
MODEL_STR   = "gpt-5.4-2026-03-05"


def convert_split(split: str, responses_dir: Path, data_dir: Path, out_dir: Path):
    resp_file = responses_dir / f"responses_queries_{split}_batched_gpt-5.4.json"
    data_file = data_dir / f"cladder-v1-q-{split}.json"

    if not resp_file.exists():
        print(f"  [skip] {resp_file} not found")
        return
    if not data_file.exists():
        print(f"  [skip] {data_file} not found")
        return

    # Load original data → lookup by question_id
    raw = json.loads(data_file.read_text())
    records = raw if isinstance(raw, list) else raw.get("queries", raw)
    orig = {str(r["question_id"]): r for r in records}

    # Load responses
    responses = [json.loads(l) for l in resp_file.read_text().splitlines() if l.strip()]

    out_file = out_dir / f"cladder-v1-q-{split}" / "baseline" / RUN_ID / f"{RUN_ID}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_correct = n_invalid = n_missing = 0
    with open(out_file, "w") as f:
        for resp in responses:
            cid  = str(resp["custom_id"])
            orig_rec = orig.get(cid)
            if orig_rec is None:
                n_missing += 1
                continue

            text = resp["response"]["body"]["output"][0]["content"][0]["text"].strip()
            tl   = text.lower()
            if tl in ("yes", "no"):
                pred = tl
            else:
                # Try to extract yes/no from longer text
                import re
                m = re.findall(r"\b(yes|no)\b", tl)
                pred = m[-1] if m else None

            gold = orig_rec["answer"].lower()
            row  = {
                "run_id":        RUN_ID,
                "question_id":   orig_rec["question_id"],
                "model_id":      orig_rec.get("model_id"),
                "query_type":    orig_rec["query_type"],
                "gold":          gold,
                "pred":          pred,
                "raw_response":  text,
                "model_id_str":  MODEL_STR,
                "decision_mode": "generate",
            }
            f.write(json.dumps(row) + "\n")
            n_total += 1
            if pred is None:
                n_invalid += 1
            elif pred == gold:
                n_correct += 1

    acc = n_correct / n_total if n_total else 0
    print(f"  {split:20s}: {n_total:6d} records  acc={acc:.4f}  invalid={n_invalid}  missing_lookup={n_missing}")
    print(f"    → {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=["easy", "hard", "anticommonsense"],
                    help="Splits to convert (default: easy hard anticommonsense)")
    ap.add_argument("--responses_dir", default="data/gpt-5.4_responses")
    ap.add_argument("--data_dir",      default="data")
    ap.add_argument("--out_dir",       default="outputs")
    args = ap.parse_args()

    responses_dir = Path(args.responses_dir)
    data_dir      = Path(args.data_dir)
    out_dir       = Path(args.out_dir)

    print(f"Converting gpt-5.4 responses → {RUN_ID}")
    for split in args.splits:
        convert_split(split, responses_dir, data_dir, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
