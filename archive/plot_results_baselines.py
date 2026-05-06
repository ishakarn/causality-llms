import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from query_type_labels import abbrev, sort_key, RUNG_ORDER
from plot_style import model_color, ALPHA_BAR, LINEWIDTH_BAR, apply_defaults

apply_defaults()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	with Path(path).open("r") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rows.append(json.loads(line))
	return rows


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
	if n == 0:
		return 0.0, 0.0
	p = k / n
	z2 = z * z
	denom = 1.0 + z2 / n
	center = (p + z2 / (2.0 * n)) / denom
	half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
	lo = max(0.0, center - half)
	hi = min(1.0, center + half)
	return lo, hi


def short_model_name(model_id: str) -> str:
	if not model_id:
		return "unknown"
	return model_id.split("/")[-1]


def summarize_one_run(pred_jsonl: str, label: str = "") -> pd.DataFrame:
	rows = read_jsonl(pred_jsonl)
	if not rows:
		raise ValueError(f"No rows found in {pred_jsonl}")

	model_id = rows[0].get("model_id_str") or rows[0].get("model_name_or_path") or "unknown"
	model_label = label.strip() if label else short_model_name(str(model_id))

	per_type: Dict[str, Dict[str, int]] = {}
	total_n = 0
	total_correct = 0
	for r in rows:
		qt = r.get("query_type", "UNKNOWN")
		if qt not in per_type:
			per_type[qt] = {"n": 0, "correct": 0}
		per_type[qt]["n"] += 1
		total_n += 1

		pred = r.get("pred")
		gold = r.get("gold")
		if pred is not None and pred == gold:
			per_type[qt]["correct"] += 1
			total_correct += 1

	out_rows = []
	for qt, s in per_type.items():
		n = s["n"]
		k = s["correct"]
		acc = (k / n) if n else 0.0
		ci_lo, ci_hi = wilson_interval(k, n)
		out_rows.append(
			{
				"model": model_label,
				"model_id": model_id,
				"query_type": qt,
				"n": n,
				"correct": k,
				"acc": acc,
				"ci_lo": ci_lo,
				"ci_hi": ci_hi,
				"err_low": acc - ci_lo,
				"err_high": ci_hi - acc,
				"source_file": str(Path(pred_jsonl).name),
			}
		)

	all_acc = (total_correct / total_n) if total_n else 0.0
	all_ci_lo, all_ci_hi = wilson_interval(total_correct, total_n)
	out_rows.append(
		{
			"model": model_label,
			"model_id": model_id,
			"query_type": "all",
			"n": total_n,
			"correct": total_correct,
			"acc": all_acc,
			"ci_lo": all_ci_lo,
			"ci_hi": all_ci_hi,
			"err_low": all_acc - all_ci_lo,
			"err_high": all_ci_hi - all_acc,
			"source_file": str(Path(pred_jsonl).name),
		}
	)

	return pd.DataFrame(out_rows)


def parse_run_arg(item: str) -> Tuple[str, str]:
	if "=" in item:
		label, path = item.split("=", 1)
		return label.strip(), path.strip()
	return "", item.strip()




def plot_per_query(df: pd.DataFrame, out_png: str, title: str) -> None:
	model_order = sorted(df["model"].unique().tolist())

	present = df["query_type"].unique().tolist()
	query_order = sorted(present, key=sort_key)   # canonical rung order
	n_by_type = df.groupby("query_type", as_index=False)["n"].max()
	count_map = dict(zip(n_by_type["query_type"], n_by_type["n"]))

	x = np.arange(len(query_order))
	bar_width = min(0.82 / max(1, len(model_order)), 0.35)
	offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2.0) * bar_width

	plt.figure(figsize=(max(12, len(query_order) * 1.4), 6.8))
	ax = plt.gca()

	for idx, model in enumerate(model_order):
		sub = df[df["model"] == model].copy()
		sub = sub.set_index("query_type").reindex(query_order).reset_index()

		y = sub["acc"].to_numpy(dtype=float)
		low = sub["err_low"].to_numpy(dtype=float)
		high = sub["err_high"].to_numpy(dtype=float)
		xpos = x + offsets[idx]

		ax.bar(
			xpos,
			y,
			width=bar_width * 0.95,
			alpha=ALPHA_BAR,
			label=model,
			color=model_color(model),
			yerr=np.vstack([low, high]),
			error_kw={"elinewidth": 1.1, "capsize": 3},
			edgecolor="black",
			linewidth=LINEWIDTH_BAR,
		)

	xticklabels = [f"{abbrev(qt)}\n(n={count_map.get(qt, 0)})" for qt in query_order]
	ax.set_xticks(x)
	ax.set_xticklabels(xticklabels, rotation=30, ha="right")
	ax.set_ylim(-0.02, 1.02)
	ax.set_ylabel("Accuracy")
	ax.set_xlabel("Query Type (count shown per type)")
	ax.set_title(title)
	ax.axhline(0.5, linestyle=":", linewidth=1.3, color="black", alpha=0.8)
	ax.grid(axis="y", linestyle=":", alpha=0.5)
	ax.legend(title="Model", loc="best")
	plt.tight_layout()

	out_path = Path(out_png)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=220)
	plt.close()


def main() -> None:
	ap = argparse.ArgumentParser(
		description=(
			"Plot per-query-type grouped bar accuracy for multiple runs/models with "
			"Wilson 95% CI error bars."
		)
	)
	ap.add_argument(
		"--run",
		nargs="+",
		required=True,
		help=(
			"One or more runs as PATH or LABEL=PATH. Example: "
			"--run Think=outputs/think.jsonl Instruct=outputs/instruct.jsonl"
		),
	)
	ap.add_argument("--out", default="outputs/plots/per_query_models_wilson95_bars.png")
	ap.add_argument("--out_csv", default="outputs/plots/per_query_models_wilson95_bars.csv")
	ap.add_argument(
		"--title",
		default="OLMo 3 7B Baselines (Instruct vs Think): Per-query Accuracy (Wilson 95% CI)",
	)
	args = ap.parse_args()

	frames: List[pd.DataFrame] = []
	for run_item in args.run:
		label, path = parse_run_arg(run_item)
		frames.append(summarize_one_run(path, label=label))

	df = pd.concat(frames, ignore_index=True)
	plot_per_query(df, args.out, args.title)

	out_csv = Path(args.out_csv)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	df.sort_values(["query_type", "model"]).to_csv(out_csv, index=False)

	print(f"Wrote plot: {args.out}")
	print(f"Wrote table: {args.out_csv}")
	print("\nQuery counts by type:")
	counts = df.groupby("query_type", as_index=False)["n"].max().sort_values("n", ascending=False)
	print(counts.to_string(index=False))


if __name__ == "__main__":
	main()
