"""
Drop-graph-structure intervention (100) for the CLadder dataset.

Strips the causal graph description from the background field, leaving only
the story title. The original background typically reads:

    "The <story> scenario involves ... We know that X causes Y ..."

After the intervention only the title clause before the first colon is kept:

    "The <story> scenario."

This removes all explicit structural information (variable relationships,
direction of causation) while keeping the story context and all numerical
given_info intact. It tests whether models rely on the graph structure
stated in the background or can reason from the given facts alone.

Causal structure, numerical values, and gold label are unchanged.

Usage:
    python 100_drop_graph_structure.py
    python 100_drop_graph_structure.py --splits easy hard anticommonsense noncommonsense
"""

import argparse
import json
from pathlib import Path


DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


def drop_graph_structure(background: str) -> str:
    """Keep only the title clause (text before the first colon)."""
    colon = background.find(":")
    if colon == -1:
        return background
    return background[:colon].rstrip() + "."


def process_split(split: str) -> None:
    in_path  = DATA_DIR / f"cladder-v1-q-{split}.json"
    out_dir  = OUTPUT_DIR / split
    out_path = out_dir / f"100_drop_graph_structure_{split}.json"

    questions = json.load(open(in_path))
    processed = []
    for q in questions:
        new_q = dict(q)
        new_bg = drop_graph_structure(q.get("background", ""))
        new_q["background"]     = new_bg
        new_q["original_text"]  = q.get("text", "")
        new_q["text"]           = (new_bg + " " + q["given_info"] + " " + q["question"]).strip()
        processed.append(new_q)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"  [100] {split}: {len(processed)} questions → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CLadder drop-graph-structure intervention (100)")
    parser.add_argument("--splits", nargs="+",
                        default=["easy", "hard", "anticommonsense", "noncommonsense"],
                        choices=["easy", "hard", "anticommonsense", "noncommonsense"])
    args = parser.parse_args()

    for split in args.splits:
        process_split(split)
    print("Done.")


if __name__ == "__main__":
    main()
