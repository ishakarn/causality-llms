"""
Generate a table of query-type counts across all graded splits and CLADDER
eval datasets, saved as a CSV and printed to stdout.

Output columns:
  query_type | test | val | train_n50 | train_n100 | ... | train_n2000
  (plus the 4 large CLADDER eval datasets for reference)

Usage:
    python finetune/plots/make_query_type_table.py
    python finetune/plots/make_query_type_table.py \
        --splits_dir finetune/splits \
        --graded_dir finetune/splits_graded \
        --out_csv    outputs/plots/graded/query_type_counts.csv
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

SIZES = [50, 100, 250, 500, 1000, 2000]
EVAL_DATASETS = ["easy", "hard", "anticommonsense", "noncommonsense"]

QUERY_TYPE_ORDER = [
    # Rung 1 — associational
    "marginal", "correlation", "backadj",
    # Rung 2 — interventional
    "ate", "ett", "nde", "nie", "exp_away", "collider_bias",
    # Rung 3 — counterfactual
    "det-counterfactual",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def qt_counts(records: list[dict]) -> Counter:
    return Counter(r["query_type"] for r in records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir",  default="finetune/splits")
    ap.add_argument("--graded_dir",  default="finetune/splits_graded")
    ap.add_argument("--data_dir",    default="data")
    ap.add_argument("--out_csv",     default="outputs/plots/graded/query_type_counts.csv")
    ap.add_argument("--no_eval",     action="store_true",
                    help="Skip loading the large CLADDER eval datasets")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    graded_dir = Path(args.graded_dir)
    data_dir   = Path(args.data_dir)

    # ── Collect all counts ────────────────────────────────────────────────────
    rows: dict[str, dict] = {}

    # Shared val / test
    for split in ("val", "test"):
        records = load_jsonl(splits_dir / f"{split}.jsonl")
        rows[split] = {"label": split, "n_total": len(records),
                       **qt_counts(records)}

    # Per-N training splits
    for n in SIZES:
        path = graded_dir / f"n{n}" / "train.jsonl"
        if not path.exists():
            continue
        records = load_jsonl(path)
        rows[f"train_n{n}"] = {"label": f"train n={n}", "n_total": len(records),
                               **qt_counts(records)}

    # Large CLADDER eval datasets (reference columns)
    if not args.no_eval:
        for ds in EVAL_DATASETS:
            path = data_dir / f"cladder-v1-q-{ds}.json"
            if not path.exists():
                continue
            records = load_json(path)
            rows[f"eval_{ds}"] = {"label": f"eval/{ds}", "n_total": len(records),
                                  **qt_counts(records)}

    # ── Build DataFrame ───────────────────────────────────────────────────────
    all_qtypes = sorted(
        {k for row in rows.values() for k in row
         if k not in ("label", "n_total")},
        key=lambda q: (QUERY_TYPE_ORDER.index(q) if q in QUERY_TYPE_ORDER else 999, q),
    )

    records_out = []
    for key, data in rows.items():
        rec = {"split": data["label"], "n_total": data["n_total"]}
        for qt in all_qtypes:
            rec[qt] = data.get(qt, 0)
        records_out.append(rec)

    df = pd.DataFrame(records_out)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n=== Query-type counts across graded splits ===\n")
    print(df.to_string(index=False))
    print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved → {out}")

    # ── Also print a transposed view (query_type as rows) for readability ──
    df_T = df.set_index("split")[all_qtypes].T
    df_T.index.name = "query_type"
    print("\n=== Transposed (query_type × split) ===\n")
    print(df_T.to_string())
    print()


if __name__ == "__main__":
    main()
