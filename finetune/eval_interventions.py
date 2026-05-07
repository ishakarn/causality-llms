"""
Evaluate OLMo-3-7B-Instruct (base and LoRA fine-tuned) on all intervention
files to test whether fine-tuning represents genuine learning or memorization.

For each intervention file the script:
  1. Filters records to the test-split question_ids (held-out set)
  2. Runs logit-mode yes/no inference
  3. Writes a per-intervention JSONL and computes accuracy

A final CSV summary is written comparing:
  - baseline (original text, zero-shot)
  - each intervention × {base_model, finetuned}

Usage (run via eval_interventions.sh or directly):

  # Base model only
  python finetune/eval_interventions.py --mode base

  # Fine-tuned model only
  python finetune/eval_interventions.py --mode finetuned

  # Both sequentially (default)
  python finetune/eval_interventions.py --mode both
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

INTERVENTION_DIR = Path("data/Olmo 3 7B Interventions")
TEST_SPLIT       = Path("finetune/splits/test.jsonl")
CONFIG           = Path("finetune/config.yaml")
OUT_DIR          = Path("finetune/intervention_results")

# Human-readable short names for each intervention file
INTERVENTION_LABELS = {
    "1_insert_spaces_between_chars_without_whitespace":         "char_spaces",
    "7_append_1000_high_density_unicode_chars":                 "unicode_noise",
    "13_remove_every_other_word":                               "del_alt_words",
    "22_insert_incorrect_answer_once_per_sentence":             "insert_wrong_ans",
    "33_insert_rare_emoji_blocks":                              "emoji_blocks",
    "34_polarity_decrease_to_not_decrease_flip":                "polarity_not_dec",
    "35_polarity_decrease_to_increase_flip":                    "polarity_increase",
    "37_polarity_decrease_to_not_not_decrease_same":            "polarity_notnot",
    "62_intervention_word_order":                               "shuffle_words",
    "66_set_percentages_to_x":                                  "mask_numbers",
    "67_replace_scenario_words_with_random_words_corrected":    "random_words",
    "68_replace_scenario_words_with_consistent_numbers":        "words_to_nums",
    "69_scenario_words_to_random_words_and_percentages_to_x":  "rand_words+mask_nums",
    "70_scenario_words_to_random_words_and_mask_polarity":      "rand_words+mask_pol",
    "71_scenario_words_to_random_words_and_percentages_to_x_and_mask_polarity": "full_ablation",
    "74_swap_percentages_within_graph_group_and_invert_answers":"swap_pct+inv_ans",
    "75_swap_percentages_within_graph_group_and_invert_polarity":"swap_pct+inv_pol",
    "77_remove_all_filler_words":                               "no_filler",
}

# Interventions that change the gold answer (polarity/answer flips)
ANSWER_CHANGED_INTERVENTIONS = {
    "34_polarity_decrease_to_not_decrease_flip",
    "35_polarity_decrease_to_increase_flip",
    "74_swap_percentages_within_graph_group_and_invert_answers",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return list(data.values())


def get_system_msg(model_id: str) -> str:
    if "think" in model_id.lower():
        return (
            "You must answer with exactly one token: yes or no.\n"
            "Do not provide any explanation, reasoning, analysis, or extra text.\n"
            "If you are unsure, still output yes or no."
        )
    return "Answer with only 'yes' or 'no'. No explanation. No punctuation."


def build_prompt(tokenizer, text: str, system_msg: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{system_msg}\n\n{text}\n\nAnswer:"


def get_yesno_ids(tokenizer) -> Tuple[List[int], List[int]]:
    def single_token_ids(variants):
        seen, out = set(), []
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
                out.append(ids[0])
        return out
    yes_ids = single_token_ids([" yes", "yes", " Yes", "YES"])
    no_ids  = single_token_ids([" no",  "no",  " No",  "NO"])
    if not yes_ids or not no_ids:
        raise RuntimeError("Could not resolve single-token ids for 'yes'/'no'")
    return yes_ids, no_ids


def run_inference(model, tokenizer, records: List[Dict], system_msg: str,
                  yes_t, no_t, device, desc: str) -> List[Dict]:
    results = []
    for record in tqdm(records, desc=desc, leave=False):
        text   = record.get("text", "")
        prompt = build_prompt(tokenizer, text, system_msg)
        enc    = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1, :]
        yes_score = logits[yes_t].max().item()
        no_score  = logits[no_t].max().item()
        pred      = "yes" if yes_score >= no_score else "no"
        results.append({
            "question_id": record.get("question_id"),
            "query_type":  record.get("query_type"),
            "gold":        record.get("answer"),
            "pred":        pred,
        })
    return results


def accuracy(results: List[Dict]) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r["pred"] == r["gold"]) / len(results)


def per_qt_accuracy(results: List[Dict]) -> Dict[str, float]:
    by_qt: Dict[str, List] = {}
    for r in results:
        by_qt.setdefault(r["query_type"], []).append(r)
    return {qt: accuracy(rows) for qt, rows in by_qt.items()}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(cfg):
    model_id    = cfg["model_id"]
    dtype_map   = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.get("dtype", "bf16"), torch.bfloat16)
    print(f"[load] base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=cfg.get("trust_remote_code", True)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(cfg):
    from peft import PeftModel
    model_id    = cfg["model_id"]
    run_name    = cfg.get("run_name", "olmo-lora")
    checkpoint  = str(Path(cfg["output_dir"]) / run_name / "final_adapter")
    dtype_map   = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.get("dtype", "bf16"), torch.bfloat16)
    print(f"[load] base model for LoRA: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=cfg.get("trust_remote_code", True)
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    print(f"[load] LoRA adapter: {checkpoint}")
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    return model, tokenizer


# ── Core eval loop ────────────────────────────────────────────────────────────

def eval_all_interventions(model, tokenizer, model_label: str,
                           test_qids: set, cfg: dict, out_subdir: Path):
    """Run model on original data + all intervention files. Write JSONL + return summary rows."""
    out_subdir.mkdir(parents=True, exist_ok=True)

    model_id   = cfg["model_id"]
    system_msg = get_system_msg(model_id)
    yes_ids, no_ids = get_yesno_ids(tokenizer)
    device = next(model.parameters()).device
    yes_t  = torch.tensor(yes_ids, device=device)
    no_t   = torch.tensor(no_ids,  device=device)

    summary_rows = []

    # ── Original (unmodified) data as baseline ─────────────────────────────
    orig_data  = load_json(cfg["data_path"])
    orig_test  = [r for r in orig_data if r.get("question_id") in test_qids]
    print(f"\n[{model_label}] original data: {len(orig_test)} test records")

    orig_results = run_inference(model, tokenizer, orig_test, system_msg,
                                 yes_t, no_t, device, f"{model_label} | original")
    orig_acc = accuracy(orig_results)
    orig_qt  = per_qt_accuracy(orig_results)

    out_path = out_subdir / "original.jsonl"
    with open(out_path, "w") as f:
        for r in orig_results:
            f.write(json.dumps({**r, "intervention": "original", "model": model_label}) + "\n")

    summary_rows.append({
        "model":        model_label,
        "intervention": "original",
        "label":        "original",
        "n":            len(orig_results),
        "acc":          orig_acc,
        "acc_drop":     0.0,
        **{f"acc_{qt}": orig_qt.get(qt, float("nan")) for qt in orig_qt},
    })
    print(f"  → acc={orig_acc:.4f}")

    # ── Each intervention file ─────────────────────────────────────────────
    int_files = sorted(INTERVENTION_DIR.glob("*.json"))
    for int_file in int_files:
        stem  = int_file.stem
        label = INTERVENTION_LABELS.get(stem, stem)

        records  = load_json(int_file)
        test_recs = [r for r in records if r.get("question_id") in test_qids]

        if not test_recs:
            print(f"  [skip] {stem}: no test records")
            continue

        print(f"\n[{model_label}] {label} ({len(test_recs)} records)")
        results = run_inference(model, tokenizer, test_recs, system_msg,
                                yes_t, no_t, device, f"{model_label} | {label}")
        acc    = accuracy(results)
        qt_acc = per_qt_accuracy(results)
        drop   = orig_acc - acc  # positive = degradation

        out_path = out_subdir / f"{stem}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps({**r, "intervention": stem, "label": label,
                                    "model": model_label}) + "\n")

        row = {
            "model":        model_label,
            "intervention": stem,
            "label":        label,
            "n":            len(results),
            "acc":          acc,
            "acc_drop":     drop,
            **{f"acc_{qt}": qt_acc.get(qt, float("nan")) for qt in qt_acc},
        }
        summary_rows.append(row)
        print(f"  → acc={acc:.4f}  drop={drop:+.4f}")

    return summary_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "finetuned", "both"], default="both")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    cfg      = yaml.safe_load(Path(args.config).read_text())
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_records = load_jsonl(str(TEST_SPLIT))
    test_qids    = {r["question_id"] for r in test_records}
    print(f"Test split: {len(test_qids)} question_ids")

    all_summary = []

    if args.mode in ("base", "both"):
        model, tokenizer = load_base_model(cfg)
        rows = eval_all_interventions(
            model, tokenizer, "base",
            test_qids, cfg, out_dir / "base"
        )
        all_summary.extend(rows)
        del model
        torch.cuda.empty_cache()

    if args.mode in ("finetuned", "both"):
        model, tokenizer = load_finetuned_model(cfg)
        rows = eval_all_interventions(
            model, tokenizer, "finetuned",
            test_qids, cfg, out_dir / "finetuned"
        )
        all_summary.extend(rows)
        del model
        torch.cuda.empty_cache()

    # ── Write summary CSV ──────────────────────────────────────────────────
    if all_summary:
        csv_path = out_dir / "summary.csv"
        all_keys = list(all_summary[0].keys())
        # collect all qt keys across rows
        qt_keys = sorted({k for row in all_summary for k in row if k.startswith("acc_")})
        fieldnames = ["model", "intervention", "label", "n", "acc", "acc_drop"] + qt_keys

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_summary:
                writer.writerow(row)

        print(f"\n[done] summary → {csv_path}")
        print_summary_table(all_summary)


def print_summary_table(rows: List[Dict]):
    print(f"\n{'Model':<12}  {'Intervention':<38}  {'Acc':>6}  {'Drop':>7}")
    print("-" * 70)
    for r in rows:
        acc  = r.get("acc", float("nan"))
        drop = r.get("acc_drop", 0.0)
        sign = "+" if drop >= 0 else ""
        print(f"{r['model']:<12}  {r['label']:<38}  {acc:.4f}  {sign}{drop:.4f}")


if __name__ == "__main__":
    main()
