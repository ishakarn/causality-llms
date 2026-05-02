"""
Compute pred_yes% per epoch for each checkpoint by running logit-mode
inference on the validation split.

Usage:
    python compute_pred_yes_per_epoch.py --model olmo3-7b-instruct
    python compute_pred_yes_per_epoch.py --model qwen25-3b-instruct
    python compute_pred_yes_per_epoch.py --all

Outputs:
    learning_curves/<model>_pred_yes_per_epoch.csv
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINETUNE = ROOT / "finetune"

MODELS = {
    "olmo3-7b-instruct": {
        "config": FINETUNE / "configs/olmo3-7b-instruct.yaml",
        "checkpoints_dir": FINETUNE / "checkpoints/olmo3-7b-instruct-lora",
        "steps_per_epoch": 144,
        "n_epochs": 5,
    },
    "qwen25-3b-instruct": {
        "config": FINETUNE / "configs/qwen25-3b-instruct.yaml",
        "checkpoints_dir": FINETUNE / "checkpoints/qwen25-3b-instruct-lora",
        "steps_per_epoch": 144,
        "n_epochs": 5,
    },
}

VAL_SPLIT = FINETUNE / "splits/val.json"
OUT_DIR = Path(__file__).parent


def run_logit_inference(model_id, adapter_path, val_path, out_jsonl):
    """Run cladder_infer_yesno.py in logit mode on val split."""
    import subprocess
    cmd = [
        sys.executable,
        str(ROOT / "cladder_infer_yesno.py"),
        "--model_id", model_id,
        "--adapter_path", str(adapter_path),
        "--data_file", str(val_path),
        "--output_file", str(out_jsonl),
        "--mode", "logit",
        "--split", "val",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return False
    return True


def compute_pred_yes(jsonl_path):
    preds = [json.loads(l)["pred"] for l in open(jsonl_path)]
    n = len(preds)
    yes = sum(p == "yes" for p in preds)
    return n, yes, yes / n


def process_model(model_name):
    cfg = MODELS[model_name]
    ckpt_dir = cfg["checkpoints_dir"]
    steps = cfg["steps_per_epoch"]
    n_epochs = cfg["n_epochs"]

    # load model_id from config yaml
    import yaml
    with open(cfg["config"]) as f:
        config = yaml.safe_load(f)
    model_id = config["model_id"]

    rows = []
    for epoch in range(1, n_epochs + 1):
        step = epoch * steps
        ckpt_path = ckpt_dir / f"checkpoint-{step}"
        if not ckpt_path.exists():
            print(f"  [skip] {ckpt_path} not found")
            continue

        tmp_jsonl = OUT_DIR / f"_tmp_{model_name}_ep{epoch}.jsonl"
        print(f"Epoch {epoch}: checkpoint-{step}")

        ok = run_logit_inference(model_id, ckpt_path, VAL_SPLIT, tmp_jsonl)
        if not ok or not tmp_jsonl.exists():
            print(f"  [skip] inference failed for epoch {epoch}")
            continue

        n, yes, yes_pct = compute_pred_yes(tmp_jsonl)
        no_pct = 1 - yes_pct
        rows.append({"epoch": epoch, "n": n, "pred_yes": yes, "pred_yes_pct": round(yes_pct, 4), "pred_no_pct": round(no_pct, 4)})
        tmp_jsonl.unlink(missing_ok=True)
        print(f"  pred_yes={yes}/{n} ({yes_pct:.1%})")

    out_csv = OUT_DIR / f"{model_name}_pred_yes_per_epoch.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "n", "pred_yes", "pred_yes_pct", "pred_no_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out_csv}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for name in MODELS:
            print(f"\n=== {name} ===")
            process_model(name)
    elif args.model:
        process_model(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
