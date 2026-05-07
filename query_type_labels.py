"""
Canonical 3-letter acronyms for CLADDER query types.

Import this wherever query types appear in plots, tables, or filenames
so the labels stay identical across all outputs.

Usage:
    from query_type_labels import QT_ABBREV, abbrev, RUNG_ORDER

    label = abbrev("det-counterfactual")   # → "DCF"
    sorted_types = RUNG_ORDER              # canonical plot order
"""

# Full query_type string → 3-letter acronym
QT_ABBREV = {
    "marginal":           "MAR",
    "correlation":        "COR",
    "backadj":            "BDA",
    "ate":                "ATE",
    "ett":                "ETT",
    "nie":                "NIE",
    "nde":                "NDE",
    "det-counterfactual": "DCF",
    "exp_away":           "EXP",
    "collider_bias":      "COL",
    # aggregate rows — both names used across scripts
    "all":                "ALL",
    "overall":            "ALL",
}

# Reverse mapping: acronym → full name
QT_FULL = {v: k for k, v in QT_ABBREV.items()}

# Canonical display order: Rung 1 → Rung 2 → Rung 3, then rare types
RUNG_ORDER = [
    "marginal",           # Rung 1
    "correlation",        # Rung 1
    "backadj",            # Rung 1
    "ate",                # Rung 2
    "ett",                # Rung 2
    "nie",                # Rung 2
    "nde",                # Rung 2
    "det-counterfactual", # Rung 3
    "exp_away",           # Rung 1 (rare)
    "collider_bias",      # Rung 1 (rare)
]

# Rung membership (for coloring / grouping)
RUNG = {
    "marginal":           1,
    "correlation":        1,
    "backadj":            1,
    "ate":                2,
    "ett":                2,
    "nie":                2,
    "nde":                2,
    "det-counterfactual": 3,
    "exp_away":           1,
    "collider_bias":      1,
}


def abbrev(query_type: str) -> str:
    """Return the 3-letter acronym for a query type, or the input uppercased if unknown."""
    return QT_ABBREV.get(query_type, query_type.upper()[:3])


def sort_key(query_type: str) -> int:
    """Sort key matching RUNG_ORDER; unknown types go last."""
    try:
        return RUNG_ORDER.index(query_type)
    except ValueError:
        return len(RUNG_ORDER)
