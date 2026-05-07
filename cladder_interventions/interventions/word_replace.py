"""
Word-replace and number-replace interventions.

67  word_replace    Replace all variable names/phrases with random nouns.
68  number_replace  Replace all variable names/phrases with random 3-digit numbers.

For each question the variable names and noun forms are detected via the source
story YAML and the model's variable_mapping, then replaced using the same
two-phase sentinel approach as story_swap.py (longest-first, no cascading).

Replacement targets are drawn deterministically (seed = question_id).

word_map / number_map fields store the slot → value assignments per question so
interventions are fully reversible.

Usage:
    python word_replace.py
    python word_replace.py --splits easy hard
"""

import argparse
import json
import random
import re
import yaml
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


# ── Word pool ─────────────────────────────────────────────────────────────────
# Common English nouns with no causal, scientific, or directional associations.

WORD_POOL = [
    # Stones / minerals
    "marble",   "pebble",    "boulder",   "crystal",  "shard",    "gem",
    "flint",    "slate",     "quartz",    "granite",  "obsidian", "jasper",
    "topaz",    "opal",      "amber",     "jade",     "coral",    "agate",
    "garnet",   "onyx",      "pumice",    "feldspar", "mica",     "gypsum",
    "chalk",    "cobble",    "gravel",    "basalt",   "schist",   "chert",
    # Wood / plants
    "twig",     "plank",     "bark",      "reed",     "fern",     "moss",
    "acorn",    "elm",       "birch",     "cedar",    "maple",    "willow",
    "ivy",      "sage",      "thyme",     "clover",   "briar",    "ash",
    "oak",      "pine",      "frond",     "sprig",    "stalk",    "bough",
    "knot",     "timber",    "splint",    "larch",    "yew",      "balsa",
    # Weather / water
    "mist",     "frost",     "gale",      "dew",      "hail",     "fog",
    "crest",    "foam",      "ripple",    "gust",     "flurry",   "vapor",
    "sleet",    "tide",      "brine",     "silt",     "delta",    "eddy",
    "torrent",  "droplet",
    # Household objects
    "bucket",   "barrel",    "kettle",    "lantern",  "anvil",    "pestle",
    "ladle",    "trowel",    "spade",     "clasp",    "latch",    "hinge",
    "thread",   "needle",    "spindle",   "bobbin",   "flask",    "vial",
    "urn",      "cask",      "crate",     "bale",     "spool",    "coil",
    "reel",     "bead",      "pod",       "hull",     "prism",    "wedge",
    "pulley",   "crank",     "bellows",   "mortar",   "piston",   "rivet",
    # Geography / terrain
    "ridge",    "knoll",     "bluff",     "gully",    "dell",     "glen",
    "moor",     "fen",       "vale",      "ledge",    "crag",     "gorge",
    "ravine",   "summit",    "slope",     "mesa",     "basin",    "dune",
    "fjord",    "inlet",     "shoal",     "reef",     "atoll",    "butte",
    "plateau",  "tundra",    "steppe",    "islet",    "spit",     "bluff",
    # Fabrics / materials
    "linen",    "velvet",    "silk",      "tweed",    "burlap",   "muslin",
    "gauze",    "satin",     "denim",     "flannel",  "tulle",    "suede",
    "hemp",     "twill",     "lace",      "wool",     "crepe",    "damask",
    "plaid",    "nylon",     "rayon",     "twine",    "cord",     "strand",
    "mesh",     "lattice",   "weft",      "warp",     "taffeta",  "organza",
    # Miscellaneous neutral nouns
    "token",    "cipher",    "bracket",   "notch",    "groove",   "pivot",
    "seam",     "stitch",    "fringe",    "tassel",   "flange",   "gusset",
    "tack",     "ferrule",   "fleck",     "mote",     "speck",    "orb",
    "disc",     "slab",
]
WORD_POOL = list(dict.fromkeys(WORD_POOL))  # deduplicate, preserve order


# ── Variable slot enumeration ─────────────────────────────────────────────────

def get_var_slots(meta_to_yaml: dict) -> list[tuple[str, str]]:
    """Return ordered list of (meta_var, slot_type) for all variable slots."""
    slots = []
    for meta_var in meta_to_yaml:
        slots.append((meta_var, "name"))
        slots.append((meta_var, "val0"))
        slots.append((meta_var, "val1"))
    return slots


def assign_words(slots: list, seed: int) -> dict:
    """Assign one unique random word per slot."""
    return {slot: word for slot, word in
            zip(slots, random.Random(seed).sample(WORD_POOL, len(slots)))}


def assign_numbers(slots: list, seed: int, pct_values: set) -> dict:
    """Assign one unique 3-digit number per slot, avoiding pct_values."""
    available = [n for n in range(100, 1000) if n not in pct_values]
    chosen    = random.Random(seed).sample(available, len(slots))
    return {slot: n for slot, n in zip(slots, chosen)}


# ── Replacement dict builder ──────────────────────────────────────────────────

def build_replacements(variable_mapping: dict, source_yaml: dict,
                       meta_to_yaml: dict, slot_to_value: dict) -> dict[str, str]:
    """
    Build {source_phrase: target_value} for all variable phrase forms.

    Mirrors story_swap.build_replacement_dict but uses slot_to_value targets
    instead of a target story YAML.  Extended YAML forms are only used when the
    YAML's noun values agree with the model's variable_mapping (same guard as
    story_swap — anticommonsense models with substituted X phrases skip YAML forms).
    """
    replacements = {}

    for meta_var, yaml_var in meta_to_yaml.items():
        src_name  = (variable_mapping.get(f"{meta_var}name") or "").strip()
        src_noun0 = (variable_mapping.get(f"{meta_var}0")    or "").strip()
        src_noun1 = (variable_mapping.get(f"{meta_var}1")    or "").strip()

        tgt_name  = str(slot_to_value.get((meta_var, "name"), ""))
        tgt_val0  = str(slot_to_value.get((meta_var, "val0"), ""))
        tgt_val1  = str(slot_to_value.get((meta_var, "val1"), ""))

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


# ── word_map / number_map helpers ─────────────────────────────────────────────

def make_word_map(meta_to_yaml: dict, slot_to_word: dict) -> dict:
    """Build word_map dict for JSON output: Xname, X0, X1, V2name, ..."""
    result = {}
    for meta_var in meta_to_yaml:
        result[f"{meta_var}name"] = slot_to_word[(meta_var, "name")]
        result[f"{meta_var}0"]    = slot_to_word[(meta_var, "val0")]
        result[f"{meta_var}1"]    = slot_to_word[(meta_var, "val1")]
    return result


def make_number_map(meta_to_yaml: dict, slot_to_number: dict) -> dict:
    """Build number_map dict for JSON output."""
    result = {}
    for meta_var in meta_to_yaml:
        result[f"{meta_var}name"] = slot_to_number[(meta_var, "name")]
        result[f"{meta_var}0"]    = slot_to_number[(meta_var, "val0")]
        result[f"{meta_var}1"]    = slot_to_number[(meta_var, "val1")]
    return result


# ── Per-question processing ───────────────────────────────────────────────────

def _prepare(question: dict, model: dict, story_yamls: dict):
    """Return (source_yaml, meta_to_yaml) or None if YAML missing."""
    source_yaml = story_yamls.get(model["story_id"])
    if source_yaml is None:
        return None, None
    meta_to_yaml = map_meta_vars_to_yaml_vars(model["variable_mapping"], source_yaml)
    return source_yaml, meta_to_yaml


def process_67(question: dict, model: dict, source_yaml: dict,
               meta_to_yaml: dict) -> dict:
    slots        = get_var_slots(meta_to_yaml)
    slot_to_word = assign_words(slots, seed=question["question_id"])
    replacements = build_replacements(model["variable_mapping"], source_yaml,
                                      meta_to_yaml, slot_to_word)

    new_back  = apply_replacements(model.get("background", ""), replacements)
    new_given = apply_replacements(question["given_info"], replacements)
    new_quest = apply_replacements(question["question"],   replacements)

    new_q = dict(question)
    new_q["background"] = new_back
    new_q["given_info"] = new_given
    new_q["question"]   = new_quest
    new_q["text"]       = (new_back + " " + new_given + " " + new_quest).strip()
    new_q["word_map"]   = make_word_map(meta_to_yaml, slot_to_word)
    return new_q


def process_68(question: dict, model: dict, source_yaml: dict,
               meta_to_yaml: dict) -> dict:
    pct_values   = {int(m) for m in re.findall(r'\b(\d+)%', question["given_info"])}
    slots        = get_var_slots(meta_to_yaml)
    slot_to_num  = assign_numbers(slots, seed=question["question_id"],
                                  pct_values=pct_values)
    replacements = build_replacements(model["variable_mapping"], source_yaml,
                                      meta_to_yaml, slot_to_num)

    new_back  = apply_replacements(model.get("background", ""), replacements)
    new_given = apply_replacements(question["given_info"], replacements)
    new_quest = apply_replacements(question["question"],   replacements)

    new_q = dict(question)
    new_q["background"]  = new_back
    new_q["given_info"]  = new_given
    new_q["question"]    = new_quest
    new_q["text"]        = (new_back + " " + new_given + " " + new_quest).strip()
    new_q["number_map"]  = make_number_map(meta_to_yaml, slot_to_num)
    return new_q


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_interventions(questions: list[dict], story_yamls: dict,
                      meta_model_by_id: dict) -> dict[int, tuple[list, list]]:
    """
    Run word_replace and number_replace over the question list.
    Returns {67: (processed, skipped), 68: (processed, skipped)}.
    """
    results = {67: ([], []), 68: ([], [])}

    for question in questions:
        model = meta_model_by_id[question["model_id"]]
        source_yaml, meta_to_yaml = _prepare(question, model, story_yamls)

        if source_yaml is None:
            for key in results:
                results[key][1].append(question)
            continue

        results[67][0].append(process_67(question, model, source_yaml, meta_to_yaml))
        results[68][0].append(process_68(question, model, source_yaml, meta_to_yaml))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CLadder word-replace and number-replace interventions")
    parser.add_argument("--splits", nargs="+", default=["easy", "hard"],
                        choices=["easy", "hard", "anticommonsense", "noncommonsense"])
    args = parser.parse_args()

    print("Loading story YAMLs...")
    story_yamls = load_all_story_yamls(STORIES_DIR)
    print(f"  {len(story_yamls)} stories loaded")

    print("Loading meta-models...")
    meta_models      = json.load(open(DATA_DIR / "cladder-v1-meta-models.json"))
    meta_model_by_id = {m["model_id"]: m for m in meta_models}

    prefixes = {67: "67_word_replace", 68: "68_number_replace"}

    for split in args.splits:
        print(f"\nProcessing {split} split...")
        questions   = json.load(open(DATA_DIR / f"cladder-v1-q-{split}.json"))
        all_results = run_interventions(questions, story_yamls, meta_model_by_id)

        split_dir = OUTPUT_DIR / split
        split_dir.mkdir(exist_ok=True)
        for iid, (processed, skipped) in all_results.items():
            out_path = split_dir / f"{prefixes[iid]}_{split}.json"
            with open(out_path, "w") as f:
                json.dump(processed, f, indent=2)
            print(f"  [{iid}] processed={len(processed)}  skipped={len(skipped)}"
                  f"  → {split}/{out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
