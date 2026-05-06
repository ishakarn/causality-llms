"""
Nonsense-replace intervention (86) for the CLadder dataset.

Replaces every story-specific variable name/phrase with a novel 4-letter
nonsense string that:
  - is exactly 4 lowercase letters
  - does not appear in CLadder's own nonsense vocabulary (31 words)
  - does not consist of a single repeated character (e.g. 'aaaa')
  - is drawn from a fixed 200-word pool generated with seed=42

Within each question the assignment is deterministic (seed = question_id)
and consistent — the same source variable always maps to the same string.
Causal structure, numerical values, and gold label are preserved exactly.

Outputs:
  86_nonsense_replace_{split}.json   — one JSON array per split
  nonsense_vocab.txt                 — 200-word generated pool + 31 CLadder
                                       words, with explicit no-overlap check

Usage:
    python nonsense_replace.py
    python nonsense_replace.py --splits easy hard anticommonsense
"""

import argparse
import json
import random
import re
import string
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from story_swap import (
    load_all_story_yamls,
    get_yaml_value,
    map_meta_vars_to_yaml_vars,
    apply_replacements,
    add_pair,
    EXTENDED_YAML_SUFFIXES,
    get_anticommonsense_phrase_forms,
)


STORIES_DIR = Path(__file__).parent.parent / "cladder-main" / "assets" / "stories"
DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_DIR  = Path(__file__).parent


# ── CLadder's existing nonsense vocabulary (extracted from nonsense0–9.yml) ───

CLADDER_NONSENSE = frozenset({
    "cwoi", "glimx", "gwet",  "gyzp",  "hwax",  "jyka",  "kraz",  "kwox",
    "kwoz", "lirg",  "muvq",  "muvy",  "pexu",  "qwiu",  "rixq",  "rukz",
    "swoq", "swoy",  "tijv",  "tijw",  "uvzi",  "vubr",  "wibl",  "xevo",
    "xevu", "xyfo",  "yomx",  "yupt",  "zory",  "zuph",  "zupj",
})


# ── Novel 4-letter nonsense pool ──────────────────────────────────────────────

def _build_pool(size: int = 200, seed: int = 42) -> list[str]:
    """
    Generate `size` novel 4-letter lowercase strings.
    Excludes CLadder's existing vocabulary and all-same-character strings
    (e.g. 'aaaa') which could confuse models.
    """
    rng      = random.Random(seed)
    letters  = string.ascii_lowercase
    pool     = []
    excluded = set(CLADDER_NONSENSE) | {c * 4 for c in letters}

    while len(pool) < size:
        word = "".join(rng.choices(letters, k=4))
        if word not in excluded:
            pool.append(word)
            excluded.add(word)

    return pool


NONSENSE_POOL: list[str] = _build_pool()


# ── Slot helpers (shared structure with word_replace.py) ─────────────────────

def get_var_slots(meta_to_yaml: dict) -> list[tuple[str, str]]:
    slots = []
    for meta_var in meta_to_yaml:
        slots.append((meta_var, "name"))
        slots.append((meta_var, "val0"))
        slots.append((meta_var, "val1"))
    return slots


def assign_nonsense(slots: list, seed: int) -> dict:
    return {slot: word
            for slot, word in zip(slots, random.Random(seed).sample(NONSENSE_POOL, len(slots)))}


# ── Replacement dict builder ──────────────────────────────────────────────────

def build_replacements(variable_mapping: dict, source_yaml: dict,
                       meta_to_yaml: dict, slot_to_word: dict) -> dict[str, str]:
    replacements = {}

    for meta_var, yaml_var in meta_to_yaml.items():
        src_name  = (variable_mapping.get(f"{meta_var}name") or "").strip()
        src_noun0 = (variable_mapping.get(f"{meta_var}0")    or "").strip()
        src_noun1 = (variable_mapping.get(f"{meta_var}1")    or "").strip()

        tgt_name = slot_to_word[(meta_var, "name")]
        tgt_val0 = slot_to_word[(meta_var, "val0")]
        tgt_val1 = slot_to_word[(meta_var, "val1")]

        add_pair(replacements, src_name,  tgt_name)
        add_pair(replacements, src_noun0, tgt_val0)
        add_pair(replacements, src_noun1, tgt_val1)

        yaml_noun0 = get_yaml_value(source_yaml, f"{yaml_var}0_noun")
        yaml_noun1 = get_yaml_value(source_yaml, f"{yaml_var}1_noun")
        yaml_matches_model = (
            (not src_noun0 or not yaml_noun0 or yaml_noun0 == src_noun0) and
            (not src_noun1 or not yaml_noun1 or yaml_noun1 == src_noun1)
        )

        if yaml_matches_model:
            src_forms = {f"{yaml_var}{s}": get_yaml_value(source_yaml, f"{yaml_var}{s}")
                         for s in EXTENDED_YAML_SUFFIXES}
        else:
            src_forms = get_anticommonsense_phrase_forms(yaml_var, src_noun1, source_yaml)

        for suffix in EXTENDED_YAML_SUFFIXES:
            src_val = src_forms.get(f"{yaml_var}{suffix}", "")
            tgt_val = tgt_val0 if suffix.startswith("0") else tgt_val1
            add_pair(replacements, src_val, tgt_val)

            if " instead of " in (src_val or ""):
                src_prefix = src_val.split(" instead of ")[0]
                add_pair(replacements, src_prefix, tgt_val)

    return replacements


def make_nonsense_map(meta_to_yaml: dict, slot_to_word: dict) -> dict:
    result = {}
    for meta_var in meta_to_yaml:
        result[f"{meta_var}name"] = slot_to_word[(meta_var, "name")]
        result[f"{meta_var}0"]    = slot_to_word[(meta_var, "val0")]
        result[f"{meta_var}1"]    = slot_to_word[(meta_var, "val1")]
    return result


# ── Per-question processing ───────────────────────────────────────────────────

def process_question(question: dict, model: dict,
                     source_yaml: dict, meta_to_yaml: dict) -> dict:
    slots        = get_var_slots(meta_to_yaml)
    slot_to_word = assign_nonsense(slots, seed=question["question_id"])
    replacements = build_replacements(model["variable_mapping"], source_yaml,
                                      meta_to_yaml, slot_to_word)

    new_back  = apply_replacements(model.get("background", ""), replacements)
    new_given = apply_replacements(question["given_info"], replacements)
    new_quest = apply_replacements(question["question"],   replacements)

    new_q = dict(question)
    new_q["background"]    = new_back
    new_q["given_info"]    = new_given
    new_q["question"]      = new_quest
    new_q["text"]          = (new_back + " " + new_given + " " + new_quest).strip()
    new_q["nonsense_map"]  = make_nonsense_map(meta_to_yaml, slot_to_word)
    return new_q


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_intervention(questions: list[dict], story_yamls: dict,
                     meta_model_by_id: dict) -> tuple[list[dict], list[dict]]:
    processed, skipped = [], []

    for question in questions:
        model       = meta_model_by_id[question["model_id"]]
        source_yaml = story_yamls.get(model["story_id"])
        if source_yaml is None:
            skipped.append(question)
            continue

        meta_to_yaml = map_meta_vars_to_yaml_vars(model["variable_mapping"], source_yaml)
        processed.append(process_question(question, model, source_yaml, meta_to_yaml))

    return processed, skipped


def write_vocab_file(path: Path) -> None:
    lines = [
        "CLadder nonsense vocabulary (31 words from nonsense0–9.yml)",
        "─" * 60,
    ]
    lines.append("  " + "  ".join(sorted(CLADDER_NONSENSE)))
    lines.append("")
    lines.append(f"Generated novel pool (200 words, seed=42, 4 letters, no all-same-char)")
    lines.append("─" * 60)
    for i in range(0, len(NONSENSE_POOL), 10):
        lines.append("  " + "  ".join(NONSENSE_POOL[i:i+10]))
    lines.append("")
    overlap = set(NONSENSE_POOL) & CLADDER_NONSENSE
    lines.append(f"Overlap between generated pool and CLadder vocabulary: "
                 f"{'NONE' if not overlap else sorted(overlap)}")
    path.write_text("\n".join(lines) + "\n")
    print(f"  Vocab file: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="CLadder nonsense-replace intervention (86)")
    parser.add_argument("--splits", nargs="+", default=["easy", "hard"],
                        choices=["easy", "hard", "anticommonsense", "noncommonsense"])
    args = parser.parse_args()

    print("Loading story YAMLs...")
    story_yamls = load_all_story_yamls(STORIES_DIR)
    print(f"  {len(story_yamls)} stories loaded")

    print("Loading meta-models...")
    meta_models      = json.load(open(DATA_DIR / "cladder-v1-meta-models.json"))
    meta_model_by_id = {m["model_id"]: m for m in meta_models}

    write_vocab_file(OUTPUT_DIR / "nonsense_vocab.txt")

    for split in args.splits:
        print(f"\nProcessing {split} split...")
        questions          = json.load(open(DATA_DIR / f"cladder-v1-q-{split}.json"))
        processed, skipped = run_intervention(questions, story_yamls, meta_model_by_id)

        out_path = OUTPUT_DIR / split / f"86_nonsense_replace_{split}.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(processed, f, indent=2)

        print(f"  Questions in:  {len(questions)}")
        print(f"  Processed:     {len(processed)}")
        print(f"  Skipped:       {len(skipped)}")
        print(f"  Saved to:      {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
