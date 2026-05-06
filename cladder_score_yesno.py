import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score_rows(rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      per_type_df: query_type breakdown
      summary: dict with overall stats
    """
    n = len(rows)
    invalid = 0
    correct = 0

    by_type = defaultdict(lambda: {"n": 0, "correct": 0, "invalid": 0})

    for r in rows:
        qt = r.get("query_type", "UNKNOWN")
        gold = r.get("gold")
        pred = r.get("pred")

        by_type[qt]["n"] += 1

        if pred is None:
            invalid += 1
            by_type[qt]["invalid"] += 1
            continue

        if pred == gold:
            correct += 1
            by_type[qt]["correct"] += 1

    overall_acc = correct / n if n else 0.0
    valid_n = n - invalid
    valid_acc = (correct / valid_n) if valid_n else 0.0
    invalid_rate = invalid / n if n else 0.0

    per_rows = []
    for qt, s in sorted(by_type.items(), key=lambda kv: kv[0]):
        nn = s["n"]
        inv = s["invalid"]
        val = nn - inv
        acc = (s["correct"] / nn) if nn else 0.0
        vacc = (s["correct"] / val) if val else 0.0
        per_rows.append(
            {
                "query_type": qt,
                "n": nn,
                "invalid": inv,
                "invalid_rate": inv / nn if nn else 0.0,
                "acc_all": acc,
                "acc_valid_only": vacc,
            }
        )

    per_df = pd.DataFrame(per_rows).sort_values("n", ascending=False)

    summary = {
        "n": n,
        "correct": correct,
        "invalid": invalid,
        "invalid_rate": invalid_rate,
        "acc_all": overall_acc,
        "acc_valid_only": valid_acc,
    }
    return per_df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True, help="Output from cladder_infer_yesno.py")
    ap.add_argument("--out_dir", default=None, help="Optional: write summary.json + per_type.csv here")
    args = ap.parse_args()

    rows = read_jsonl(args.pred_jsonl)
    per_df, summary = score_rows(rows)

    print("\n=== Overall ===")
    for k, v in summary.items():
        print(f"{k:>14}: {v}")

    print("\n=== Per query_type (top 10 by n) ===")
    print(per_df.head(10).to_string(index=False))

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        per_df.to_csv(out / "per_query_type.csv", index=False)
        print(f"\nWrote: {out/'summary.json'}")
        print(f"Wrote: {out/'per_query_type.csv'}")


if __name__ == "__main__":
    main()