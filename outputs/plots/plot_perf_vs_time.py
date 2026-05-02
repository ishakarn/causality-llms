"""
Performance vs time plot: CLadder accuracy across model generations.
Output: outputs/plots/perf_vs_time.png and perf_vs_time.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from adjustText import adjust_text

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("outputs/plots/perf_vs_time_data.csv")
df["date"] = pd.to_datetime(df["release_date"])
df["acc"]  = df["overall_accuracy"] / 100.0

# ── Visual encoding maps ──────────────────────────────────────────────────────
EXPOSURE_COLOR = {
    "yes":     "#C44E52",
    "no":      "#4C72B0",
    "unknown": "#888888",
}
TYPE_MARKER = {
    "base": "o",
    "lora": "^",
    "cot":  "s",
}

def model_type(name):
    if "LoRA" in name:       return "lora"
    if "CausalCoT" in name:  return "cot"
    return "base"

def exposure_color(val):
    return EXPOSURE_COLOR.get(str(val).strip().lower(), "#888888")

# ── Display-only date jitter (days) to un-stack crowded clusters ──────────────
# Feb 2025 cluster: 6 models at same date — spread ±40 days for readability
DATE_JITTER = {
    "OLMo-3.1-7B base":           -38,
    "OLMo-3.1-7B LoRA (N=2000)":  -22,
    "Qwen2.5-3B LoRA (N=2000)":    -8,
    "OLMo-3.1-32B base":            8,
    "OLMo-3.1-32B LoRA (N=2000)":  22,
    "LLaMA-3.1-8B LoRA (N=2000)":  38,
    # Nov 2023 cluster
    "GPT-3.5-turbo-1106":          -18,
    "GPT-4-1106":                    0,
    "GPT-4-1106 + CausalCoT":       18,
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

cladder_pub = datetime(2023, 12, 1)
ax.axvline(mdates.date2num(cladder_pub), color="#888888", linewidth=1.4,
           linestyle="--", zorder=1)
ax.text(cladder_pub, 0.975, "CLadder published\n(arXiv: Dec 2023)",
        ha="center", va="top", fontsize=11, color="#555555",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

texts   = []
point_xs = []
point_ys = []

for _, row in df.iterrows():
    color  = exposure_color(row["suspected_exposure"])
    marker = TYPE_MARKER[model_type(row["model_name"])]
    jitter = DATE_JITTER.get(row["model_name"], 0)
    x      = mdates.date2num(row["date"]) + jitter
    y      = row["acc"]
    uncertain = str(row["release_date_uncertain"]).upper() == "TRUE"

    ax.scatter(x, y, color=color, marker=marker,
               s=110, zorder=5, edgecolors="white", linewidths=0.8,
               alpha=0.95)
    if uncertain:
        ax.errorbar(x, y, xerr=20, fmt="none",
                    ecolor=color, elinewidth=1, capsize=3, alpha=0.5, zorder=4)

    t = ax.text(x, y, row["model_name"],
                fontsize=9, color="#222222", zorder=6)
    texts.append(t)
    point_xs.append(x)
    point_ys.append(y)

adjust_text(
    texts,
    x=point_xs,
    y=point_ys,
    ax=ax,
    expand=(1.3, 1.6),
    force_text=(0.4, 0.8),
    force_points=(0.3, 0.6),
    force_explode=(0.1, 0.2),
    arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.7,
                    shrinkA=4, shrinkB=4),
    time_lim=5,
)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(mdates.date2num(datetime(2020, 1, 1)),
            mdates.date2num(datetime(2026, 8, 1)))
ax.set_ylim(0.45, 1.02)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
plt.xticks(rotation=0, ha="center", fontsize=10)
ax.set_ylabel("CLadder accuracy (overall)", fontsize=13)
ax.set_xlabel("Approximate model release date", fontsize=13)
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.grid(axis="x", linestyle=":", alpha=0.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.tick_params(axis="y", labelsize=10)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mlines.Line2D([0],[0], marker="o", color="w", markerfacecolor="#4C72B0",
                  markersize=9, label="Base / zero-shot"),
    mlines.Line2D([0],[0], marker="^", color="w", markerfacecolor="#4C72B0",
                  markersize=9, label="LoRA fine-tuned (N=2000)"),
    mlines.Line2D([0],[0], marker="s", color="w", markerfacecolor="#4C72B0",
                  markersize=9, label="Prompted (CausalCoT)"),
    mpatches.Patch(facecolor="white", edgecolor="white", label=""),
    mpatches.Patch(facecolor="#4C72B0", label="No suspected CLadder exposure"),
    mpatches.Patch(facecolor="#C44E52", label="Suspected CLadder exposure"),
    mlines.Line2D([0],[0], color="#888888", linewidth=1, linestyle="--",
                  label="CLadder publication (Dec 2023)"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="upper left",
          framealpha=0.9, edgecolor="#dddddd")

# ── Dataset mismatch note ─────────────────────────────────────────────────────
note = ("Note: GPT-3/3.5/4 evaluated on CLadder v1.5 balanced split (incl. non-sensical questions).\n"
        "Our models evaluated on easy + hard + anticommonsense splits. Absolute numbers\n"
        "are not directly comparable, but the trend across model generations is informative.")
ax.text(0.01, 0.02, note, transform=ax.transAxes,
        fontsize=9, color="#666666", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#dddddd", alpha=0.9))

plt.tight_layout()
import os
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/perf_vs_time.png", dpi=200, bbox_inches="tight")
plt.savefig("outputs/plots/perf_vs_time.pdf", bbox_inches="tight")
plt.close()
print("Wrote: outputs/plots/perf_vs_time.png")
print("Wrote: outputs/plots/perf_vs_time.pdf")
