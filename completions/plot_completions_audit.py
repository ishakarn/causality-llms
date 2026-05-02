"""
Plot next-token rank per token for each model from the completions audit.
Lower rank = model predicted the correct token more easily.
NF (not found in top 1000) is plotted at a sentinel value above the axis.
Output: completions/outputs/rank_plot.png
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "GPT-OSS-20B":    ("completions/outputs/gptoss20b_full_audit.txt",    "#55A868"),
    "OLMo-3.1-32B":   ("completions/outputs/olmo32b_base_full_audit.txt", "#DD8452"),
}

NF_SENTINEL = 1500   # y-value used to plot NF tokens
OUT = Path("completions/outputs/rank_plot.png")


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_results(path: Path) -> list[tuple[str, int | None]]:
    """Return [(token, rank_or_None)] in order from the FINAL section."""
    text = path.read_text()
    marker = "FINAL SIMPLIFIED TOKEN -> RANK LIST"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Could not find final results section in {path}")
    section = text[idx:]
    results = []
    for line in section.splitlines():
        m = re.match(r"^'(.+)':\s+(\d+|NF)$", line)
        if not m:
            continue
        token = m.group(1)
        raw   = m.group(2)
        rank  = None if raw == "NF" else int(raw)
        results.append((token, rank))
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot():
    all_data = {name: parse_results(Path(path)) for name, (path, _) in MODELS.items()}

    # Align on token sequence (use first model's tokens as x-axis labels)
    ref_name  = list(MODELS.keys())[0]
    tokens    = [t for t, _ in all_data[ref_name]]
    n         = len(tokens)
    x         = np.arange(n)

    # Build readable x-labels: show token + index to disambiguate repeats
    x_labels = [f"{tok}\n({i+1})" for i, tok in enumerate(tokens)]

    fig, ax = plt.subplots(figsize=(max(14, n * 0.45), 6))

    offsets = [-0.15, 0.15]
    for (name, (_, color)), offset in zip(MODELS.items(), offsets):
        ranks  = [r for _, r in all_data[name]]
        y_vals = [NF_SENTINEL if r is None else r for r in ranks]
        is_nf  = [r is None for r in ranks]

        # Plot found tokens
        found_x = [x[i] + offset for i in range(n) if not is_nf[i]]
        found_y = [y_vals[i] for i in range(n) if not is_nf[i]]
        ax.scatter(found_x, found_y, color=color, s=40, zorder=4, alpha=0.85, label=name)

        # Plot NF tokens as X markers at the top
        nf_x = [x[i] + offset for i in range(n) if is_nf[i]]
        nf_y = [NF_SENTINEL] * len(nf_x)
        ax.scatter(nf_x, nf_y, color=color, marker="x", s=50, zorder=4,
                   linewidths=1.5, alpha=0.7)

        # Connect dots with a light line
        ax.plot([x[i] + offset for i in range(n)], y_vals,
                color=color, linewidth=0.6, alpha=0.35, zorder=2)

    # NF line annotation
    ax.axhline(NF_SENTINEL, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.text(n - 0.5, NF_SENTINEL + 30, "NF (>1000)", ha="right",
            fontsize=8, color="#888888")

    # Rank=1 line
    ax.axhline(1, color="#cccccc", linewidth=0.6, linestyle=":")

    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0.5, NF_SENTINEL + 300)
    ax.set_yticks([1, 5, 10, 50, 100, 500, 1000, NF_SENTINEL])
    ax.set_yticklabels(["1", "5", "10", "50", "100", "500", "1000", "NF"])
    ax.set_ylabel("Rank of correct token\n(lower = model predicted it more easily)", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Token in query (in order)", fontsize=9)

    ax.set_title("Next-token rank per token: GPT-OSS-20B vs OLMo-3.1-32B\n"
                 "(× = not found in top 1000 candidates; lower rank = better prediction)",
                 fontsize=11, fontweight="bold")

    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Legend
    handles = []
    for name, (_, color) in MODELS.items():
        handles.append(mlines.Line2D([], [], color=color, marker="o", markersize=6,
                                     linewidth=1, label=name))
    handles.append(mlines.Line2D([], [], color="gray", marker="x", markersize=6,
                                  linewidth=0, label="NF (not in top 1000)"))
    ax.legend(handles=handles, fontsize=9, loc="upper left")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {OUT}")

    # Print summary stats
    print("\nSummary:")
    for name, (path, _) in MODELS.items():
        data   = all_data[name]
        ranks  = [r for _, r in data if r is not None]
        nf_cnt = sum(1 for _, r in data if r is None)
        print(f"  {name}: median_rank={np.median(ranks):.0f}  mean_rank={np.mean(ranks):.1f}"
              f"  rank1={sum(1 for r in ranks if r==1)}  NF={nf_cnt}/{len(data)}")


if __name__ == "__main__":
    plot()
