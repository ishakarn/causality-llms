"""
Three alternative plot styles for per-dataset multi-model accuracy comparison.

1. Heatmap        — models × query types, accuracy as color
2. Slope chart    — base → LoRA jump per query type (OLMo & Qwen only)
3. Small multiples — one panel per query type, all models as bars

Usage:
    python finetune/plot_styles.py --dataset cladder-v1-q-easy
    python finetune/plot_styles.py --dataset cladder-v1-q-anticommonsense --outputs_root outputs
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from query_type_labels import abbrev, sort_key, RUNG
from plot_style import model_color, HATCH_FINETUNED, ALPHA_BAR, LINEWIDTH_BAR, apply_defaults

apply_defaults()


# ── Data helpers (same as plot_all_models.py) ─────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def compute_per_type(rows: List[Dict]) -> pd.DataFrame:
    by_type: Dict[str, Dict] = {}
    total_n = total_correct = 0
    for r in rows:
        qt = r.get("query_type", "UNKNOWN")
        if qt not in by_type:
            by_type[qt] = {"n": 0, "correct": 0}
        by_type[qt]["n"] += 1
        total_n += 1
        if r.get("pred") is not None and r.get("pred") == r.get("gold"):
            by_type[qt]["correct"] += 1
            total_correct += 1
    out = []
    for qt, s in by_type.items():
        n, k = s["n"], s["correct"]
        acc = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        out.append({"query_type": qt, "n": n, "correct": k,
                    "acc": acc, "ci_lo": lo, "ci_hi": hi})
    acc_all = total_correct / total_n if total_n else 0.0
    lo, hi = wilson_ci(total_correct, total_n)
    out.append({"query_type": "overall", "n": total_n, "correct": total_correct,
                "acc": acc_all, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out).set_index("query_type")


def discover_models(dataset: str, outputs_root: Path) -> List[Dict]:
    found = []
    for subdir, is_ft in [("baseline", False), ("finetuned", True)]:
        d = outputs_root / dataset / subdir
        if not d.exists():
            continue
        for model_dir in sorted(d.glob("*")):
            jsonls = list(model_dir.glob("*.jsonl"))
            if jsonls:
                found.append({"jsonl": jsonls[0], "is_finetuned": is_ft,
                               "run_id": model_dir.name})
    return found


_DISPLAY_NAMES = {
    "gpt-5-nano":        "GPT-5-Nano",
    "gpt-oss-20b":       "GPT-OSS-20B",
    "olmo3-7b-instruct": "OLMo-3-7B",
    "qwen25-3b-instruct":"Qwen 2.5-3B",
}

def _display(run_id: str) -> str:
    low = run_id.lower()
    for k, v in _DISPLAY_NAMES.items():
        if k in low:
            return v
    return run_id

def _short_label(run_id: str, is_ft: bool) -> str:
    return _display(run_id) + ("\n(LoRA)" if is_ft else "\n(base)")


# ── Plot 1: Heatmap ───────────────────────────────────────────────────────────

def plot_heatmap(entries: List[Dict], dataset: str, out_path: Path) -> None:
    # Build type_order (query types + overall)
    all_types = set()
    for e in entries:
        all_types.update(e["df"].index)
    all_types.discard("overall")
    type_order = sorted(all_types, key=sort_key) + ["overall"]
    type_labels = [abbrev(qt) for qt in type_order]

    model_labels = [_short_label(e["run_id"], e["is_finetuned"]) for e in entries]

    # Matrix: rows = models, cols = query types
    mat = np.full((len(entries), len(type_order)), np.nan)
    for i, e in enumerate(entries):
        for j, qt in enumerate(type_order):
            if qt in e["df"].index:
                mat[i, j] = e["df"].loc[qt, "acc"]

    fig, ax = plt.subplots(figsize=(max(12, len(type_order) * 1.1), max(4, len(entries) * 0.85)))

    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1,
                   cmap="RdYlGn", interpolation="nearest")

    # Annotate cells
    for i in range(len(entries)):
        for j in range(len(type_order)):
            v = mat[i, j]
            if not np.isnan(v):
                text_color = "black" if 0.35 < v < 0.75 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels(type_labels, fontsize=9)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(model_labels, fontsize=9)

    # Divider before "overall" column
    ax.axvline(len(type_order) - 1.5, color="white", linewidth=2)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Accuracy", fontsize=9)
    ax.set_title(f"{dataset} — accuracy heatmap", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}")


# ── Plot 2: Slope chart ───────────────────────────────────────────────────────

def plot_slope(entries: List[Dict], dataset: str, out_path: Path) -> None:
    """Base → LoRA slope per query type for OLMo and Qwen."""
    # Pair up base+ft per model family
    pairs = {}  # display_name -> {base: df, ft: df, color: str}
    for e in entries:
        name = _display(e["run_id"])
        if name not in ("OLMo-3-7B", "Qwen 2.5-3B"):
            continue
        if name not in pairs:
            pairs[name] = {"color": model_color(e["run_id"])}
        key = "ft" if e["is_finetuned"] else "base"
        pairs[name][key] = e["df"]

    if not pairs:
        print("  [slope] no OLMo/Qwen pairs found, skipping")
        return

    all_types = set()
    for p in pairs.values():
        if "base" in p:
            all_types.update(p["base"].index)
    all_types.discard("overall")
    type_order = sorted(all_types, key=sort_key)

    fig, ax = plt.subplots(figsize=(6, 7))
    x_base, x_ft = 0, 1

    for name, p in pairs.items():
        if "base" not in p or "ft" not in p:
            continue
        color = p["color"]
        for qt in type_order:
            if qt not in p["base"].index or qt not in p["ft"].index:
                continue
            y0 = p["base"].loc[qt, "acc"]
            y1 = p["ft"].loc[qt, "acc"]
            # faint connecting line
            ax.plot([x_base, x_ft], [y0, y1], color=color, alpha=0.35,
                    linewidth=1.2, zorder=1)
            # dots
            ax.scatter(x_base, y0, color=color, s=28, zorder=3, alpha=0.7)
            ax.scatter(x_ft,   y1, color=color, s=28, zorder=3,
                       marker="D", alpha=0.9)
            # label only on the right side
            ax.text(x_ft + 0.03, y1, abbrev(qt), fontsize=6.5,
                    va="center", color=color, alpha=0.85)

        # Bold overall line
        if "overall" in p["base"].index and "overall" in p["ft"].index:
            y0 = p["base"].loc["overall", "acc"]
            y1 = p["ft"].loc["overall", "acc"]
            ax.plot([x_base, x_ft], [y0, y1], color=color, linewidth=3,
                    zorder=4, solid_capstyle="round")
            ax.scatter(x_base, y0, color=color, s=80, zorder=5)
            ax.scatter(x_ft,   y1, color=color, s=80, zorder=5, marker="D")

    ax.set_xticks([x_base, x_ft])
    ax.set_xticklabels(["Base\n(zero-shot)", "LoRA\n(fine-tuned)"], fontsize=10)
    ax.set_xlim(-0.25, 1.45)
    ax.set_ylim(-0.02, 1.08)
    ax.axhline(0.5, linestyle=":", linewidth=1, color="black", alpha=0.4)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(f"{dataset}\nbase → LoRA per query type", fontsize=11,
                 fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=p["color"], label=name)
        for name, p in pairs.items()
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}")


# ── Plot 3: Small multiples ───────────────────────────────────────────────────

def plot_small_multiples(entries: List[Dict], dataset: str, out_path: Path) -> None:
    all_types = set()
    for e in entries:
        all_types.update(e["df"].index)
    all_types.discard("overall")
    type_order = sorted(all_types, key=sort_key) + ["overall"]

    n_panels = len(type_order)
    ncols = 4
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.8),
                             sharey=True)
    axes_flat = axes.flatten() if n_panels > 1 else [axes]

    n_models = len(entries)
    x = np.arange(n_models)
    w = 0.72

    for panel_i, qt in enumerate(type_order):
        ax = axes_flat[panel_i]
        rung = RUNG.get(qt, 0)
        rung_colors = {1: "#e8f4f8", 2: "#fef9e7", 3: "#fdf2f8", 0: "#f5f5f5"}
        ax.set_facecolor(rung_colors.get(rung, "#f5f5f5"))

        for i, e in enumerate(entries):
            if qt not in e["df"].index:
                continue
            row = e["df"].loc[qt]
            acc = row["acc"]
            lo  = np.clip(acc - row["ci_lo"], 0, None)
            hi  = np.clip(row["ci_hi"] - acc, 0, None)
            ax.bar(i, acc, width=w,
                   color=model_color(e["run_id"]),
                   hatch=HATCH_FINETUNED if e["is_finetuned"] else None,
                   alpha=ALPHA_BAR, edgecolor="black", linewidth=LINEWIDTH_BAR,
                   yerr=[[lo], [hi]],
                   error_kw={"elinewidth": 1, "capsize": 2.5, "ecolor": "#333"})

        ax.axhline(0.5, linestyle=":", linewidth=0.9, color="black", alpha=0.5)
        ax.set_ylim(0, 1.08)
        ax.set_xticks([])
        ax.set_title(abbrev(qt), fontsize=9, fontweight="bold", pad=3)
        if panel_i % ncols == 0:
            ax.set_ylabel("Acc", fontsize=8)

    # Hide unused panels
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared legend at bottom
    handles = [
        mpatches.Patch(
            facecolor=model_color(e["run_id"]),
            hatch=HATCH_FINETUNED if e["is_finetuned"] else None,
            edgecolor="black", linewidth=0.5,
            label=_short_label(e["run_id"], e["is_finetuned"]).replace("\n", " "),
            alpha=ALPHA_BAR,
        )
        for e in entries
    ]
    fig.legend(handles=handles, fontsize=8, loc="lower center",
               ncol=min(n_models, 4), bbox_to_anchor=(0.5, -0.01),
               frameon=True)

    fig.suptitle(f"{dataset} — per query type", fontsize=12,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",      required=True)
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--out_dir",      default=None,
                    help="Output dir for all 3 plots (default: outputs/plots/newdata/<dataset>/)")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(f"outputs/plots/newdata/{args.dataset}")

    raw = discover_models(args.dataset, outputs_root)
    if not raw:
        print(f"No JSONL files found for '{args.dataset}'")
        return

    entries = []
    for e in raw:
        rows = load_jsonl(e["jsonl"])
        entries.append({
            "run_id":      e["run_id"],
            "is_finetuned": e["is_finetuned"],
            "df":          compute_per_type(rows),
        })
        acc = entries[-1]["df"].loc["overall", "acc"] if "overall" in entries[-1]["df"].index else float("nan")
        print(f"  {'FT' if e['is_finetuned'] else 'BS'} {e['run_id']:<45} acc={acc:.4f}")

    print()
    plot_heatmap(entries, args.dataset, out_dir / "heatmap.png")
    plot_slope(entries, args.dataset, out_dir / "slope.png")
    plot_small_multiples(entries, args.dataset, out_dir / "small_multiples.png")


if __name__ == "__main__":
    main()
