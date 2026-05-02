"""
Create stratified training subsets of size N from the existing train split
for graded-exposure experiments.

Each subset is a proportional stratified sample by query_type drawn from the
EXISTING train.jsonl (not re-split from raw data). Val and test are shared
across all runs — they are copied verbatim from finetune/splits/ so that
accuracy is directly comparable across all N conditions.

Output layout:
    finetune/splits_graded/
        n50/   train.jsonl (50 examples, stratified)
               val.jsonl   (copy of finetune/splits/val.jsonl)
               test.jsonl  (copy of finetune/splits/test.jsonl)
        n100/  ...
        ...

Usage:
    python finetune/prepare_graded.py
    python finetune/prepare_graded.py --sizes 50 100 250 500 1000 2000 \\
        --base_splits finetune/splits --out_dir finetune/splits_graded
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


SIZES = [50, 100, 250, 500, 1000, 2000]


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def stratified_sample(records: List[Dict], n: int, seed: int) -> List[Dict]:
    """
    Draw exactly n examples proportional to query_type distribution.

    For each type, floor(n * type_frac) examples are taken. Remainders are
    allocated one-at-a-time to the types with the largest fractional parts
    until we reach exactly n.
    """
    rng = random.Random(seed)
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_type[r["query_type"]].append(r)

    # Shuffle within each type for reproducibility
    for qt in by_type:
        rng.shuffle(by_type[qt])

    total = len(records)
    # Compute proportional allocations
    exact = {qt: n * len(items) / total for qt, items in by_type.items()}
    floors = {qt: int(v) for qt, v in exact.items()}
    remainder = n - sum(floors.values())

    # Distribute remainder to types with largest fractional parts
    fracs = sorted(by_type.keys(), key=lambda qt: -(exact[qt] - floors[qt]))
    for qt in fracs[:remainder]:
        floors[qt] += 1

    sampled = []
    for qt, count in floors.items():
        available = by_type[qt]
        if count > len(available):
            raise ValueError(
                f"Requested {count} examples for query_type={qt} "
                f"but only {len(available)} available in train split. "
                f"Reduce N or check your splits."
            )
        sampled.extend(available[:count])

    rng.shuffle(sampled)
    return sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES,
                    help="Training subset sizes to generate")
    ap.add_argument("--base_splits", default="finetune/splits",
                    help="Directory containing the base train/val/test splits")
    ap.add_argument("--out_dir", default="finetune/splits_graded",
                    help="Output root directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(args.base_splits)
    out_root = Path(args.out_dir)

    train_all = load_jsonl(base / "train.jsonl")
    print(f"Base train split: {len(train_all)} examples")
    print(f"Query type distribution:")
    for qt, n in sorted(Counter(r["query_type"] for r in train_all).items()):
        print(f"  {qt:<25} {n}")

    for N in sorted(args.sizes):
        if N > len(train_all):
            print(f"\n[skip] N={N} exceeds train size ({len(train_all)}), skipping")
            continue

        subset = stratified_sample(train_all, N, seed=args.seed)
        out_dir = out_root / f"n{N}"
        out_dir.mkdir(parents=True, exist_ok=True)

        write_jsonl(subset, out_dir / "train.jsonl")
        shutil.copy(base / "val.jsonl",  out_dir / "val.jsonl")
        shutil.copy(base / "test.jsonl", out_dir / "test.jsonl")

        qt_counts = Counter(r["query_type"] for r in subset)
        print(f"\nn={N:>5}  ({out_dir})")
        print(f"  query types represented: {len(qt_counts)}/10")
        print(f"  type counts: { {qt: qt_counts[qt] for qt in sorted(qt_counts)} }")

    print(f"\nDone. Splits written to {out_root}/")
    print("Val and test are identical across all N — accuracy is directly comparable.")


if __name__ == "__main__":
    main()
