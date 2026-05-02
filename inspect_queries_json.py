import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


TEXTY_HINTS = {"text", "question", "background", "given_info", "context", "prompt"}


def _is_scalar(x):
    return isinstance(x, (str, int, float, bool)) or x is None


def _type_name(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int) and not isinstance(x, bool):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "str"
    if isinstance(x, list):
        return "list"
    if isinstance(x, dict):
        return "dict"
    return type(x).__name__


def load_json(path: str):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    if not isinstance(data, list):
        raise ValueError("Expected a list of query objects (or dict with key 'queries').")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to queries_easy.json")
    ap.add_argument("--topk", type=int, default=20, help="Top-k values to show for categorical fields")
    ap.add_argument("--out_csv", default=None, help="Optional: write flattened CSV (best-effort)")
    args = ap.parse_args()

    records = load_json(args.path)
    n = len(records)
    print(f"\nLoaded {n} records from {args.path}\n")

    # 1) Union of keys
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)
    print("All keys found:")
    print("  " + ", ".join(all_keys))
    print()

    # 2) Missingness + type counts
    missing = Counter()
    type_counts = {k: Counter() for k in all_keys}
    value_counts = {k: Counter() for k in all_keys}

    # For length stats (for string fields)
    str_lens = defaultdict(list)

    for r in records:
        for k in all_keys:
            if k not in r or r[k] is None:
                missing[k] += 1
                type_counts[k]["missing"] += 1
                continue

            v = r[k]
            t = _type_name(v)
            type_counts[k][t] += 1

            # Categorical-ish counts: only for scalars and short strings
            if _is_scalar(v):
                if isinstance(v, str):
                    if len(v) <= 80:
                        value_counts[k][v] += 1
                    str_lens[k].append(len(v))
                else:
                    value_counts[k][v] += 1
            elif isinstance(v, list):
                # capture list length distribution
                value_counts[k][f"list_len={len(v)}"] += 1
            elif isinstance(v, dict):
                value_counts[k]["dict"] += 1

            # If key looks texty, also track length even if long
            if isinstance(v, str) and (k in TEXTY_HINTS or any(h in k.lower() for h in TEXTY_HINTS)):
                str_lens[k].append(len(v))

    # 3) Print summary table
    rows = []
    for k in all_keys:
        miss = missing[k]
        present = n - miss
        top_types = type_counts[k].most_common(4)
        rows.append(
            {
                "field": k,
                "present": present,
                "missing": miss,
                "missing_%": round(100.0 * miss / n, 2),
                "types(top)": "; ".join([f"{t}:{c}" for t, c in top_types]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["missing_%", "field"], ascending=[True, True])
    print("Field presence / missingness / types:")
    print(df.to_string(index=False))
    print()

    # 4) Show categorical distributions for a few useful fields
    # Prioritize query_type + any low-cardinality fields
    def approx_cardinality(k):
        # number of distinct categorical buckets we collected
        return len(value_counts[k])

    candidate_fields = []
    for k in all_keys:
        card = approx_cardinality(k)
        # show if it looks categorical (small-ish)
        if 1 <= card <= 50:
            candidate_fields.append((k, card))
    candidate_fields.sort(key=lambda x: (x[1], x[0]))

    # Ensure query_type is shown if present
    if "query_type" in all_keys and ("query_type", approx_cardinality("query_type")) not in candidate_fields:
        candidate_fields.insert(0, ("query_type", approx_cardinality("query_type")))

    shown = 0
    print("Top categorical-like fields (value counts):")
    for k, card in candidate_fields:
        if shown >= 12:
            break
        # Skip obviously-not-categorical long text fields
        if any(h in k.lower() for h in ["text", "background", "given", "question"]):
            continue
        print(f"\n- {k} (distinct buckets observed: {card})")
        for v, c in value_counts[k].most_common(args.topk):
            print(f"    {repr(v)}: {c}")
        shown += 1
    print()

    # 5) Text length stats
    print("Text length stats (chars) for text-like fields:")
    tl_rows = []
    for k, lens in str_lens.items():
        if not lens:
            continue
        s = pd.Series(lens)
        tl_rows.append(
            {
                "field": k,
                "count": int(s.count()),
                "min": int(s.min()),
                "p25": int(s.quantile(0.25)),
                "median": int(s.median()),
                "p75": int(s.quantile(0.75)),
                "max": int(s.max()),
                "mean": round(float(s.mean()), 2),
            }
        )
    if tl_rows:
        tldf = pd.DataFrame(tl_rows).sort_values("median", ascending=False)
        print(tldf.to_string(index=False))
    else:
        print("(none found)")
    print()

    # 6) Optional: best-effort flatten to CSV for eyeballing
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Keep dict/list columns as JSON strings so CSV is readable
        flat = []
        for r in records:
            rr = {}
            for k in all_keys:
                v = r.get(k, None)
                if isinstance(v, (dict, list)):
                    rr[k] = json.dumps(v, ensure_ascii=False)
                else:
                    rr[k] = v
            flat.append(rr)

        pd.DataFrame(flat).to_csv(out_csv, index=False)
        print(f"Wrote flattened CSV to: {out_csv}")


if __name__ == "__main__":
    main()