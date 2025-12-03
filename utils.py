import pandas as pd
import numpy as np


def norm_name(name: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def guess_column(df: pd.DataFrame, candidates) -> str | None:
    """
    Best-effort guess of a column name in df based on a list of candidates.
    Uses normalized (space/underscore-insensitive, lowercase) matching.
    """
    norm_map = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = norm_name(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def mode_or_na(s: pd.Series):
    """Return most common non-null value or <NA>."""
    s = s.dropna()
    if s.empty:
        return pd.NA
    return s.value_counts().idxmax()


def concat_unique(series: pd.Series, sep: str = "|") -> str:
    """Pipe-join unique non-empty values."""
    vals = [str(x) for x in series.dropna().unique() if str(x).strip() != ""]
    if not vals:
        return ""
    return sep.join(sorted(vals))


def lc_counts(series: pd.Series):
    """
    Count lifecycle categories: Active / Obsolete / Unknown.
    Everything else is ignored.
    """
    s = series.astype(str).str.strip().str.lower()
    active = s.eq("active").sum()
    obsolete = s.eq("obsolete").sum()
    unknown = s.eq("unknown").sum()
    total = len(s)
    return active, obsolete, unknown, total


def classify_generic_lc(active: int, obsolete: int, unknown: int) -> str:
    """Classify generic lifecycle based on counts."""
    if active > 0:
        if obsolete == 0 and unknown == 0:
            return "Active-only"
        else:
            return "Active+Others"
    elif obsolete > 0:
        if unknown == 0:
            return "Obsolete-only"
        else:
            return "Obsolete+Unknown"
    elif unknown > 0:
        return "Unknown-only"
    else:
        return "No LC data"


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric, coercing invalid entries to NaN."""
    return pd.to_numeric(series, errors="coerce")
