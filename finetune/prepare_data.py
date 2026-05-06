"""
Stratified train / val / test split for CLADDER queries_easy.json.

Each of the 10 query_types is split independently so that the type
distribution is preserved in every split.

Usage:
    python finetune/prepare_data.py \
        --queries_json data/queries_easy.json \
        --out_dir      finetune/splits \
        --train_frac   0.8 \
        --val_frac     0.1 \
        --seed         42
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_queries(path: str) -> List[Dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]
    return data


def stratified_split(
    records: List[Dict],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split each query_type independently, then merge and shuffle."""
    rng = random.Random(seed)
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_type[r["query_type"]].append(r)

    train, val, test = [], [], []
    for qt in sorted(by_type):
        items = by_type[qt][:]
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(records: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(records):>5} records → {path}")


def print_split_stats(name: str, records: List[Dict]) -> None:
    types = Counter(r["query_type"] for r in records)
    answers = Counter(r["answer"] for r in records)
    yes_n = answers.get("yes", 0)
    no_n = answers.get("no", 0)
    print(f"\n{name}  (n={len(records)}, yes={yes_n}, no={no_n}):")
    for qt, n in sorted(types.items(), key=lambda kv: -kv[1]):
        print(f"  {qt:<22} {n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries_json", default="data/queries_easy.json")
    ap.add_argument("--out_dir", default="finetune/splits")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert args.train_frac + args.val_frac < 1.0, "train + val must be < 1"

    records = load_queries(args.queries_json)
    print(f"Loaded {len(records)} records from {args.queries_json}")

    train, val, test = stratified_split(
        records, args.train_frac, args.val_frac, args.seed
    )

    for name, split in [("train", train), ("val", val), ("test", test)]:
        print_split_stats(name, split)

    print("\nWriting splits:")
    write_jsonl(train, f"{args.out_dir}/train.jsonl")
    write_jsonl(val,   f"{args.out_dir}/val.jsonl")
    write_jsonl(test,  f"{args.out_dir}/test.jsonl")

    total = len(train) + len(val) + len(test)
    print(f"\nSplit sizes: train={len(train)}, val={len(val)}, test={len(test)}, total={total}")


if __name__ == "__main__":
    main()
