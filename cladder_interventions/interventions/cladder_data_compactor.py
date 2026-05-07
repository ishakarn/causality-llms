"""
Preprocessing utility: flatten CLadder v1 JSON into a compact query format.

The raw CLadder files (cladder-v1-q-{split}.json) store metadata under a
nested 'meta' key and do not include the background text directly — that lives
in the separate cladder-v1-meta-models.json file keyed by model_id.

This script joins both files and writes a flat list of query dicts, each with:
    question_id, model_id, graph_id, story_id, query_type, polarity, rung,
    background, given_info, question, text, answer

The 'text' field concatenates background + given_info + question into a single
ready-to-use prompt (the format consumed by all downstream intervention scripts
and the vLLM inference pipeline).

Outputs one queries_{split}.json per split in data/.

Usage:
    python cladder_data_compactor.py
    python cladder_data_compactor.py --splits easy hard anticommonsense noncommonsense
"""

import argparse
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def compact_split(split: str) -> None:
    questions   = json.load(open(DATA_DIR / f"cladder-v1-q-{split}.json"))
    meta_models = json.load(open(DATA_DIR / "cladder-v1-meta-models.json"))
    bg_by_id    = {m["model_id"]: m["background"] for m in meta_models}

    queries = []
    for item in questions:
        meta       = item["meta"]
        background = bg_by_id.get(meta["model_id"], "")
        given_info = item["given_info"]
        question   = item["question"]
        queries.append({
            "question_id": item["question_id"],
            "model_id":    meta["model_id"],
            "graph_id":    meta["graph_id"],
            "story_id":    meta["story_id"],
            "query_type":  meta["query_type"],
            "polarity":    meta["polarity"],
            "rung":        meta["rung"],
            "background":  background,
            "given_info":  given_info,
            "question":    question,
            "text":        (background + " " + given_info + " " + question).strip(),
            "answer":      item["answer"],
        })

    out = DATA_DIR / f"queries_{split}.json"
    with open(out, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"  {split}: {len(queries)} queries → {out}")


def main():
    parser = argparse.ArgumentParser(description="Flatten CLadder v1 JSON into compact query format")
    parser.add_argument("--splits", nargs="+",
                        default=["easy", "hard", "anticommonsense", "noncommonsense"],
                        choices=["easy", "hard", "anticommonsense", "noncommonsense"])
    args = parser.parse_args()

    for split in args.splits:
        compact_split(split)
    print("Done.")


if __name__ == "__main__":
    main()
