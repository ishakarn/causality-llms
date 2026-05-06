"""
Shared visual style for all CLADDER experiment plots.

Import this in every plotting script so colors, hatching, and
figure defaults stay identical across all figures.

    from plot_style import model_color, HATCH_FINETUNED, apply_defaults
"""

import matplotlib as mpl

# ── Model color palette (Wong 2011 colorblind-safe) ───────────────────────────
# Matched by lowercase substring so any form of the name resolves correctly.
_PALETTE = [
    # OLMo models
    (["olmo-3-7b-instruct", "olmo3-7b-instruct",
      "7b-instruct", "7b instruct"],                    "#0072b2"),  # deep blue
    (["7b-think",    "7b think"],                       "#cc79a7"),  # rose-magenta
    (["3.1-32b-instruct", "3.1-32b instruct"],          "#009e73"),  # teal-green
    (["3.1-32b-think",    "3.1-32b think"],             "#d55e00"),  # vermillion
    (["1125-32b", "32b-base", "32b base", "olmo-3-1125"], "#e69f00"), # amber-orange
    # OpenAI models
    (["gpt-5-nano", "gpt5nano"],                        "#cc79a7"),  # rose-magenta
    (["gpt-oss-20b", "gpt-oss"],                        "#009e73"),  # teal-green
    (["gpt54", "gpt-5.4", "gpt5.4"],                   "#f0e442"),  # yellow
    # Qwen models
    (["qwen2.5-3b", "qwen25-3b"],                       "#e69f00"),  # amber-orange
    (["qwen2.5-7b", "qwen25-7b"],                       "#009e73"),  # teal-green
    # Gemma models
    (["gemma-3-4b", "gemma3-4b"],                       "#d55e00"),  # vermillion
    (["gemma-3-12b", "gemma3-12b"],                     "#e69f00"),  # amber-orange
    # Llama models
    (["llama-3.1-8b", "llama31-8b", "llama3.1-8b"],    "#56b4e9"),  # sky blue
]

COLOR_FALLBACK   = "#666666"   # neutral gray for unknown models
HATCH_FINETUNED  = "////"      # dense diagonal lines inside fine-tuned bars
ALPHA_BAR        = 0.88        # bar fill opacity
LINEWIDTH_BAR    = 0.6         # bar edge linewidth


def model_color(model_id_or_label: str) -> str:
    """Return the canonical color for any form of a model name/id."""
    name = model_id_or_label.lower()
    for substrings, color in _PALETTE:
        if any(s in name for s in substrings):
            return color
    return COLOR_FALLBACK


def apply_defaults() -> None:
    """Set rcParams that apply to all figures (call once at script start)."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        120,
    })


def save_fig(path, fig=None, pdf: bool = False, dpi: int = 220) -> None:
    """Save the current figure (or *fig*) to *path*.

    Always writes a PNG at the given path.  When *pdf=True* an additional PDF
    is saved alongside it (same name, ``.pdf`` extension) for vector-quality
    paper figures.  The parent directory is created if it does not exist.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = fig.savefig if fig is not None else plt.savefig
    save_fn(path, dpi=dpi, bbox_inches="tight")
    if pdf:
        save_fn(path.with_suffix(".pdf"), bbox_inches="tight")
