"""
Story-swap intervention (81) for the CLadder dataset.

For each question, the variable names and narrative framing are replaced with
those from a different story that uses the same causal graph topology.
The causal structure, numerical values, and answer are preserved exactly.

Usage:
    python story_swap.py
    python story_swap.py --splits easy hard --seed 42
"""

import argparse
import json
import random
import yaml
from collections import defaultdict
from pathlib import Path


STORIES_DIR = Path(__file__).parent.parent / "cladder-main" / "assets" / "stories"
DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_DIR  = Path(__file__).parent

with open(STORIES_DIR / "anticommonsense.yml") as _f:
    _AC_DATA: dict = yaml.safe_load(_f)


# ── Anticommonsense phrase-form resolution ────────────────────────────────────

def get_anticommonsense_phrase_forms(yaml_var: str, src_noun1: str,
                                     source_yaml: dict) -> dict:
    """
    For a variable whose noun values don't match the source YAML (i.e. it was
    anticommonsense-substituted), return the extended phrase forms from
    anticommonsense.yml, formatted with the correct subject/plural.

    yaml_var  : YAML letter of the variable ('X' or 'Y').
    src_noun1 : val1 noun from variable_mapping (e.g. 'liking spicy food').
    source_yaml: the story YAML (used to get the subject field).

    Returns a dict with keys like X0_wheresentence, X1_wherepartial, etc.
    Returns {} if no match found (V2/V3 variables, or unknown substitution).
    """
    var_choices = _AC_DATA.get("variables", {}).get(yaml_var, {})
    matching = None
    for fields in var_choices.values():
        # Match on the val=1 noun field first (most reliable); fall back to the name field.
        # For most choices Xname == X1_noun, but for some (e.g. 'speaks english') they differ:
        #   Xname='ability to speak english', X1_noun='speaking english'
        if (fields.get(f"{yaml_var}1_noun") == src_noun1 or
                fields.get(f"{yaml_var}name") == src_noun1):
            matching = fields
            break
    if matching is None:
        return {}

    subject = source_yaml.get(f"{yaml_var}subject", "")
    plural  = _AC_DATA.get("plurals", {}).get(subject, f"{subject}s")

    result = {}
    for key, template in matching.items():
        try:
            result[key] = str(template).format(subject=subject, plural=plural)
        except (KeyError, ValueError):
            result[key] = str(template)
    return result


# ── YAML loading ─────────────────────────────────────────────────────────────

def load_all_story_yamls(stories_dir: Path) -> dict[str, dict]:
    """
    Load every story YAML, keyed by story_id field.
    Also indexes by filename stem as a fallback for stories where the YAML's
    story_id field differs from the dataset's story_id (e.g. blood_pressure.yml
    has story_id 'simpson_blood_pressure' but the dataset calls it 'blood_pressure').
    """
    yamls = {}
    for path in sorted(stories_dir.glob("*.yml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        story_id  = data.get("story_id")
        file_stem = path.stem  # filename without .yml

        if story_id and story_id != "??":
            yamls[story_id] = data

        # Also register under the filename so dataset story_ids always resolve
        if file_stem not in yamls:
            yamls[file_stem] = data

    return yamls


# ── Variable correspondence: meta-model Vn ↔ YAML letter (X/Y/Z/W) ──────────

def get_yaml_value(yaml_data: dict, key: str) -> str:
    """Return a YAML field's string value, or '' if missing/nan."""
    val = yaml_data.get(key)
    if val is None or str(val) in ("nan", "None"):
        return ""
    return str(val)


def find_yaml_var_for_vn(vn_name: str, vn_noun0: str, vn_noun1: str,
                          yaml_data: dict) -> str | None:
    """
    Find which YAML variable letter (X, Y, Z, W) corresponds to a meta-model
    variable (V1/V2/V3/V4) by matching name and noun values.

    Matching rules:
    - Name must match when both sides have a value.
    - Nouns must match only when BOTH the model and the YAML have a value for
      them.  If the YAML has no noun forms (e.g. the unobserved confounder W in
      the IV topology has a name but no noun fields), a name-only match is
      accepted.
    """
    for letter in ["X", "Y", "Z", "W"]:
        yaml_name  = get_yaml_value(yaml_data, f"{letter}name")
        yaml_noun0 = get_yaml_value(yaml_data, f"{letter}0_noun")
        yaml_noun1 = get_yaml_value(yaml_data, f"{letter}1_noun")

        # Name: must match when both sides have a value
        if vn_name and yaml_name and vn_name != yaml_name:
            continue
        if not (vn_name and yaml_name):
            continue  # need at least a name match to avoid spurious hits

        # Nouns: only reject if BOTH sides have a value and they disagree
        if vn_noun0 and yaml_noun0 and vn_noun0 != yaml_noun0:
            continue
        if vn_noun1 and yaml_noun1 and vn_noun1 != yaml_noun1:
            continue

        return letter

    return None


def map_meta_vars_to_yaml_vars(variable_mapping: dict, source_yaml: dict) -> dict[str, str]:
    """
    Return a dict mapping each meta-model variable name to its YAML letter.

    X and Y are always X and Y. V1..V4 are identified by matching their
    name/noun values against the source YAML.

    Example return value: {'X': 'X', 'Y': 'Y', 'V1': 'Z'}
    """
    result = {"X": "X", "Y": "Y"}

    for vn in ["V1", "V2", "V3", "V4"]:
        vn_name  = variable_mapping.get(f"{vn}name", "") or ""
        vn_noun0 = variable_mapping.get(f"{vn}0",    "") or ""
        vn_noun1 = variable_mapping.get(f"{vn}1",    "") or ""

        if not any([vn_name, vn_noun0, vn_noun1]):
            continue  # this meta-variable is not part of this topology

        yaml_var = find_yaml_var_for_vn(vn_name, vn_noun0, vn_noun1, source_yaml)
        if yaml_var is not None:
            result[vn] = yaml_var

    return result


# ── Building the text-replacement dictionary ──────────────────────────────────

# YAML text form suffixes beyond the noun/name forms that appear in question text.
EXTENDED_YAML_SUFFIXES = [
    "0_wherepartial",       "1_wherepartial",
    "0_wheresentence",      "1_wheresentence",
    "0_sentence",           "1_sentence",
    "0_sentence_condition", "1_sentence_condition",
]


def first_letter_upper(text: str) -> str:
    """Capitalise only the first character; leave the rest unchanged."""
    return text[0].upper() + text[1:] if text else text


def add_pair(replacements: dict, src: str, tgt: str) -> None:
    """
    Add a source→target pair and its first-letter-capitalised variant.
    No-ops when src or tgt is empty, or when they are equal.
    """
    if not src or not tgt or src == tgt:
        return
    replacements[src] = tgt
    cap_src = first_letter_upper(src)
    cap_tgt = first_letter_upper(tgt)
    if cap_src != src:
        replacements[cap_src] = cap_tgt


def build_replacement_dict(source_yaml: dict, target_yaml: dict,
                           variable_mapping: dict,
                           meta_to_yaml: dict[str, str]) -> dict[str, str]:
    """
    Build a {source_token: target_token} mapping for every text form of every
    variable in this question.

    Source noun/name tokens come from the model's variable_mapping — these are
    the actual values embedded in the question text, including any anticommonsense
    substitutions (e.g. Xname='drinking coffee' instead of the YAML's 'husband').

    Extended forms (wherepartial, wheresentence, sentence, sentence_condition)
    are taken from the source YAML, but only when the YAML's noun values agree
    with the model's variable_mapping.  For anticommonsense models the YAML
    extended forms are wrong, so they are skipped.

    Also adds the prefix of each sentence_condition form (the text before
    " instead of ") as a separate entry, because the question generator
    sometimes uses that truncated form rather than the full condition.
    """
    replacements = {}

    for meta_var, yaml_var in meta_to_yaml.items():
        # ── Noun / name forms from the model's variable_mapping ──────────────
        src_name  = (variable_mapping.get(f"{meta_var}name") or "").strip()
        src_noun0 = (variable_mapping.get(f"{meta_var}0")    or "").strip()
        src_noun1 = (variable_mapping.get(f"{meta_var}1")    or "").strip()

        tgt_name  = get_yaml_value(target_yaml, f"{yaml_var}name")
        tgt_noun0 = get_yaml_value(target_yaml, f"{yaml_var}0_noun")
        tgt_noun1 = get_yaml_value(target_yaml, f"{yaml_var}1_noun")

        add_pair(replacements, src_name,  tgt_name)
        add_pair(replacements, src_noun0, tgt_noun0)
        add_pair(replacements, src_noun1, tgt_noun1)

        # ── Extended forms from the source YAML ──────────────────────────────
        # Only safe to use when the source YAML's noun values match the model's
        # variable_mapping (i.e. this is a standard, non-anticommonsense model).
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
            # Anticommonsense substitution: get extended forms from anticommonsense.yml
            src_forms = get_anticommonsense_phrase_forms(yaml_var, src_noun1, source_yaml)

        for suffix in EXTENDED_YAML_SUFFIXES:
            src_val = src_forms.get(f"{yaml_var}{suffix}", "")
            tgt_val = get_yaml_value(target_yaml, f"{yaml_var}{suffix}")
            add_pair(replacements, src_val, tgt_val)

            if " instead of " in (src_val or ""):
                src_prefix = src_val.split(" instead of ")[0]
                tgt_prefix = tgt_val.split(" instead of ")[0] if " instead of " in (tgt_val or "") else tgt_val
                add_pair(replacements, src_prefix, tgt_prefix)

    return replacements


def apply_replacements(text: str, replacements: dict[str, str]) -> str:
    """
    Substitute all tokens in text using a two-phase placeholder approach.

    Phase 1 replaces each source token with a unique sentinel (longest first,
    to avoid short tokens clobbering parts of longer ones).
    Phase 2 replaces sentinels with their target values.

    The two-phase approach prevents cascading: if a target value happens to
    equal another source token, it will never be double-replaced.
    """
    sentinel_to_target = {}
    for i, source in enumerate(sorted(replacements, key=len, reverse=True)):
        if source in text:
            sentinel = f"\x00SWAP{i}\x00"
            text = text.replace(source, sentinel)
            sentinel_to_target[sentinel] = replacements[source]

    for sentinel, target in sentinel_to_target.items():
        text = text.replace(sentinel, target)

    return text


# ── Story selection ───────────────────────────────────────────────────────────

def build_topology_story_index(meta_models: list[dict]) -> dict[str, list[str]]:
    """Map each graph_id to a sorted list of story_ids that appear in the dataset."""
    index: dict[str, set] = defaultdict(set)
    for model in meta_models:
        index[model["graph_id"]].add(model["story_id"])
    return {gid: sorted(stories) for gid, stories in index.items()}


def select_target_story(source_story_id: str, graph_id: str,
                        topology_index: dict[str, list[str]],
                        seed: int) -> str | None:
    """
    Pick a swap-target story deterministically from the same topology.
    The seed is the question_id so every run produces the same pairings.
    """
    candidates = [s for s in topology_index.get(graph_id, [])
                  if s != source_story_id]
    if not candidates:
        return None
    return random.Random(seed).choice(candidates)


# ── Per-question swap ─────────────────────────────────────────────────────────

def swap_question(question: dict, model: dict,
                  source_yaml: dict, target_story_id: str, target_yaml: dict,
                  meta_to_yaml: dict[str, str]) -> dict:
    """
    Return a new question dict with all story-specific text replaced.
    background, given_info, question, and text fields are updated.
    answer, query_type, and all numerical content are left untouched.
    """
    replacements = build_replacement_dict(source_yaml, target_yaml,
                                          model["variable_mapping"], meta_to_yaml)

    swapped_background = apply_replacements(model["background"], replacements)
    swapped_given_info = apply_replacements(question["given_info"], replacements)
    swapped_question   = apply_replacements(question["question"],   replacements)

    new_q = dict(question)
    new_q["background"] = swapped_background
    new_q["given_info"] = swapped_given_info
    new_q["question"]   = swapped_question
    new_q["text"]       = (swapped_background + " " + swapped_given_info + " " + swapped_question).strip()

    return new_q


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_swap_intervention(questions: list[dict], story_yamls: dict[str, dict],
                          topology_index: dict[str, list[str]],
                          meta_model_by_id: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """
    Apply story-swap to every question.
    Returns (swapped_questions, skipped_questions).
    A question is skipped only if its story has no YAML or no valid swap partner.
    """
    swapped = []
    skipped = []

    for question in questions:
        model          = meta_model_by_id[question["model_id"]]
        source_story   = model["story_id"]
        graph_id       = model["graph_id"]

        source_yaml = story_yamls.get(source_story)
        if source_yaml is None:
            skipped.append(question)
            continue

        target_story = select_target_story(source_story, graph_id, topology_index,
                                           seed=question["question_id"])
        if target_story is None:
            skipped.append(question)
            continue

        target_yaml = story_yamls.get(target_story)
        if target_yaml is None:
            skipped.append(question)
            continue

        meta_to_yaml = map_meta_vars_to_yaml_vars(model["variable_mapping"], source_yaml)
        swapped_q    = swap_question(question, model, source_yaml,
                                     target_story, target_yaml, meta_to_yaml)
        swapped.append(swapped_q)

    return swapped, skipped


def main():
    parser = argparse.ArgumentParser(description="CLadder story-swap intervention")
    parser.add_argument("--splits", nargs="+", default=["easy", "hard"],
                        choices=["easy", "hard", "anticommonsense", "noncommonsense"])
    args = parser.parse_args()

    print("Loading story YAMLs...")
    story_yamls = load_all_story_yamls(STORIES_DIR)
    print(f"  {len(story_yamls)} stories loaded")

    print("Loading meta-models...")
    meta_models      = json.load(open(DATA_DIR / "cladder-v1-meta-models.json"))
    meta_model_by_id = {m["model_id"]: m for m in meta_models}
    topology_index   = build_topology_story_index(meta_models)
    print(f"  {len(meta_models)} models across {len(topology_index)} topologies")

    for split in args.splits:
        print(f"\nProcessing {split} split...")
        questions          = json.load(open(DATA_DIR / f"cladder-v1-q-{split}.json"))
        swapped, skipped   = run_swap_intervention(questions, story_yamls,
                                                   topology_index, meta_model_by_id)

        out_path = OUTPUT_DIR / split / f"81_story_swap_{split}.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(swapped, f, indent=2)

        print(f"  Questions in:  {len(questions)}")
        print(f"  Swapped:       {len(swapped)}")
        print(f"  Skipped:       {len(skipped)}")
        print(f"  Saved to:      {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
