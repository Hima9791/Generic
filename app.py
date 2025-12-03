import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ============================================================
# Streamlit App: Generic Lifecycle + Stock/LT/Price Dashboard
# - Re-implements and extends your two Python scripts as an app
# - Fully column-agnostic via UI mapping (no renaming needed)
# - Dynamic counts, tables, and charts driven by your selections
# ============================================================

st.set_page_config(
    page_title="Generic Lifecycle & Stock/LT/Price Dashboard",
    layout="wide",
)


# --------------------------
# Helper utilities
# --------------------------
def _norm_name(name: str) -> str:
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
    """Best-effort guess of a column from a candidate list."""
    norm_map = {_norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_name(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def mode_or_na(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return pd.NA
    return s.value_counts().idxmax()


def concat_unique(series: pd.Series, sep: str = "|") -> str:
    vals = [str(x) for x in series.dropna().unique() if str(x).strip() != ""]
    if not vals:
        return ""
    return sep.join(sorted(vals))


def lc_counts(series: pd.Series):
    """Count Active / Obsolete / Unknown lifecycle values."""
    s = series.astype(str).str.strip().str.lower()
    active = s.eq("active").sum()
    obsolete = s.eq("obsolete").sum()
    unknown = s.eq("unknown").sum()
    total = len(s)
    return active, obsolete, unknown, total


def classify_generic_lc(active: int, obsolete: int, unknown: int) -> str:
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
    return pd.to_numeric(series, errors="coerce")


# --------------------------
# 1) Lifecycle / LC engine
# --------------------------
def build_lifecycle_summaries(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Reimplementation of Script 1 logic as reusable functions.

    cfg keys (values are column names or None):
        generic, partid, partnum, company, partlc,
        partintro, familyintro, stage, risk, country, pkg_norm
    """
    generic_col = cfg.get("generic")
    partid_col = cfg.get("partid")
    partnum_col = cfg.get("partnum")
    company_col = cfg.get("company")
    partlc_col = cfg.get("partlc")
    partintro_col = cfg.get("partintro")
    familyintro_col = cfg.get("familyintro")
    stage_col = cfg.get("stage")
    risk_col = cfg.get("risk")
    country_col = cfg.get("country")
    pkg_norm_col = cfg.get("pkg_norm")

    if not generic_col:
        raise ValueError("Primary key / Generic column is required for lifecycle analysis.")

    df2 = df.copy()

    # Convert year-like columns to numeric if present
    for col in [partintro_col, familyintro_col]:
        if col and col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

    # ----- Genric-level summary -----
    def compute_generic_metrics(group: pd.DataFrame) -> pd.Series:
        out: dict[str, object] = {}
        out["TotalRows"] = len(group)

        base_count_col = partid_col or partnum_col
        if base_count_col and base_count_col in group.columns:
            out["TotalParts"] = group[base_count_col].nunique()
        else:
            out["TotalParts"] = len(group)

        if partnum_col and partnum_col in group.columns:
            out["DistinctPartNumbers"] = group[partnum_col].nunique()
        else:
            out["DistinctPartNumbers"] = pd.NA

        if company_col and company_col in group.columns:
            out["NumCompanies"] = group[company_col].nunique()
        else:
            out["NumCompanies"] = pd.NA

        # Lifecycle counts & shares
        if partlc_col and partlc_col in group.columns:
            active, obsolete, unknown, total = lc_counts(group[partlc_col])
        else:
            active = obsolete = unknown = 0
            total = len(group)

        total_parts = out["TotalParts"] or 0
        out["ActiveParts"] = int(active)
        out["ObsoleteParts"] = int(obsolete)
        out["UnknownParts"] = int(unknown)
        out["ActiveShare"] = active / total_parts if total_parts else np.nan
        out["ObsoleteShare"] = obsolete / total_parts if total_parts else np.nan
        out["UnknownShare"] = unknown / total_parts if total_parts else np.nan

        if partlc_col and partlc_col in group.columns:
            out["DistinctPartLC"] = concat_unique(group[partlc_col])
            out["GenericLCStatus"] = classify_generic_lc(active, obsolete, unknown)
            out["HasActiveAndObsolete"] = bool(active > 0 and obsolete > 0)
        else:
            out["DistinctPartLC"] = ""
            out["GenericLCStatus"] = "No LC column"
            out["HasActiveAndObsolete"] = False

        if partintro_col and partintro_col in group.columns:
            out["MinPartIntro"] = group[partintro_col].min()
            out["MaxPartIntro"] = group[partintro_col].max()
        else:
            out["MinPartIntro"] = pd.NA
            out["MaxPartIntro"] = pd.NA

        if familyintro_col and familyintro_col in group.columns:
            out["MinFamilyIntro"] = group[familyintro_col].min()
            out["MaxFamilyIntro"] = group[familyintro_col].max()
        else:
            out["MinFamilyIntro"] = pd.NA
            out["MaxFamilyIntro"] = pd.NA

        if stage_col and stage_col in group.columns:
            out["MainStage"] = mode_or_na(group[stage_col])
            out["DistinctStage"] = concat_unique(group[stage_col])
        else:
            out["MainStage"] = pd.NA
            out["DistinctStage"] = ""

        if risk_col and risk_col in group.columns:
            out["MainRisk"] = mode_or_na(group[risk_col])
        else:
            out["MainRisk"] = pd.NA

        if pkg_norm_col and pkg_norm_col in group.columns:
            out["DistinctPackages"] = group[pkg_norm_col].nunique()
            out["PackageList"] = concat_unique(group[pkg_norm_col])
        else:
            out["DistinctPackages"] = pd.NA
            out["PackageList"] = ""

        return pd.Series(out)

    genric_summary = (
        df2.groupby(generic_col, dropna=False)
        .apply(compute_generic_metrics)
        .reset_index()
        .rename(columns={generic_col: "Genric"})
    )

    # ----- Company-level LC summary -----
    if company_col and company_col in df2.columns:
        def compute_company_metrics(group: pd.DataFrame) -> pd.Series:
            out: dict[str, object] = {}
            out["TotalRows"] = len(group)

            base_count_col = partid_col or partnum_col
            if base_count_col and base_count_col in group.columns:
                out["TotalParts"] = group[base_count_col].nunique()
            else:
                out["TotalParts"] = len(group)

            if generic_col in group.columns:
                out["DistinctGenerics"] = group[generic_col].nunique()
            else:
                out["DistinctGenerics"] = pd.NA

            if partlc_col and partlc_col in group.columns:
                active, obsolete, unknown, total = lc_counts(group[partlc_col])
            else:
                active = obsolete = unknown = 0
                total = len(group)

            out["ActiveParts"] = int(active)
            out["ObsoleteParts"] = int(obsolete)
            out["UnknownParts"] = int(unknown)
            total_parts = out["TotalParts"] or 0
            out["ActiveShare"] = active / total_parts if total_parts else np.nan
            out["ObsoleteShare"] = obsolete / total_parts if total_parts else np.nan

            if partlc_col and partlc_col in group.columns:
                out["AllObsoleteFlag"] = bool(active == 0 and unknown == 0 and total_parts > 0)
                out["MostlyObsoleteFlag"] = bool(obsolete > 0 and out["ObsoleteShare"] >= 0.7)
            else:
                out["AllObsoleteFlag"] = False
                out["MostlyObsoleteFlag"] = False

            if country_col and country_col in group.columns:
                tokens: set[str] = set()
                for val in group[country_col].dropna():
                    for c in str(val).split("|"):
                        c = c.strip()
                        if c:
                            tokens.add(c)
                out["Countries"] = "|".join(sorted(tokens))
                out["NumCountries"] = len(tokens)
            else:
                out["Countries"] = ""
                out["NumCountries"] = 0

            if pkg_norm_col and pkg_norm_col in group.columns:
                out["DistinctPackages"] = group[pkg_norm_col].nunique()
            else:
                out["DistinctPackages"] = pd.NA

            return pd.Series(out)

        company_summary = (
            df2.groupby(company_col, dropna=False)
            .apply(compute_company_metrics)
            .reset_index()
            .rename(columns={company_col: "CompanyName"})
        )
    else:
        company_summary = pd.DataFrame()

    # ----- Location-level LC summary (country split by "|") -----
    if country_col and country_col in df2.columns:
        loc_df = df2.copy()
        loc_df["LocationToken"] = loc_df[country_col].astype(str).str.split("|")
        loc_df = loc_df.explode("LocationToken")
        loc_df["LocationToken"] = loc_df["LocationToken"].astype(str).str.strip()
        loc_df = loc_df[loc_df["LocationToken"] != ""]
    else:
        loc_df = None

    if loc_df is not None and not loc_df.empty:

        def compute_location_metrics(group: pd.DataFrame) -> pd.Series:
            out: dict[str, object] = {}
            out["TotalRows"] = len(group)

            base_count_col = partid_col or partnum_col
            if base_count_col and base_count_col in group.columns:
                out["TotalParts"] = group[base_count_col].nunique()
            else:
                out["TotalParts"] = len(group)

            if partlc_col and partlc_col in group.columns:
                active, obsolete, unknown, total = lc_counts(group[partlc_col])
            else:
                active = obsolete = unknown = 0
                total = len(group)

            out["ActiveParts"] = int(active)
            out["ObsoleteParts"] = int(obsolete)
            out["UnknownParts"] = int(unknown)
            total_parts = out["TotalParts"] or 0
            out["ActiveShare"] = active / total_parts if total_parts else np.nan
            out["ObsoleteShare"] = obsolete / total_parts if total_parts else np.nan
            out["AlwaysObsoleteFlag"] = bool(active == 0 and unknown == 0 and total_parts > 0)

            if company_col and company_col in group.columns:
                out["DistinctCompanies"] = group[company_col].nunique()
            else:
                out["DistinctCompanies"] = pd.NA

            if pkg_norm_col and pkg_norm_col in group.columns:
                out["DistinctPackages"] = group[pkg_norm_col].nunique()
            else:
                out["DistinctPackages"] = pd.NA

            return pd.Series(out)

        location_summary = (
            loc_df.groupby("LocationToken", dropna=False)
            .apply(compute_location_metrics)
            .reset_index()
            .rename(columns={"LocationToken": "Location"})
        )
    else:
        location_summary = pd.DataFrame(columns=["Location"])

    # ----- Package-level LC summary -----
    if pkg_norm_col and pkg_norm_col in df2.columns:

        def compute_pkg_metrics(group: pd.DataFrame) -> pd.Series:
            out: dict[str, object] = {}
            out["TotalRows"] = len(group)

            base_count_col = partid_col or partnum_col
            if base_count_col and base_count_col in group.columns:
                out["TotalParts"] = group[base_count_col].nunique()
            else:
                out["TotalParts"] = len(group)

            if partlc_col and partlc_col in group.columns:
                active, obsolete, unknown, total = lc_counts(group[partlc_col])
            else:
                active = obsolete = unknown = 0
                total = len(group)

            out["ActiveParts"] = int(active)
            out["ObsoleteParts"] = int(obsolete)
            out["UnknownParts"] = int(unknown)
            total_parts = out["TotalParts"] or 0
            out["ActiveShare"] = active / total_parts if total_parts else np.nan
            out["ObsoleteShare"] = obsolete / total_parts if total_parts else np.nan
            out["AllObsoleteFlag"] = bool(active == 0 and unknown == 0 and total_parts > 0)

            if generic_col and generic_col in group.columns:
                out["DistinctGenerics"] = group[generic_col].nunique()
            else:
                out["DistinctGenerics"] = pd.NA

            return pd.Series(out)

        pkg_file_summary = (
            df2.groupby(pkg_norm_col, dropna=False)
            .apply(compute_pkg_metrics)
            .reset_index()
            .rename(columns={pkg_norm_col: "Package_Normalized"})
        )

        pkg_genric_summary = (
            df2.groupby([generic_col, pkg_norm_col], dropna=False)
            .apply(compute_pkg_metrics)
            .reset_index()
            .rename(
                columns={
                    generic_col: "Genric",
                    pkg_norm_col: "Package_Normalized",
                }
            )
        )
        pkg_genric_summary["Genric_Package"] = (
            pkg_genric_summary["Genric"].astype(str)
            + "|"
            + pkg_genric_summary["Package_Normalized"].astype(str)
        )
    else:
        pkg_file_summary = pd.DataFrame(columns=["Package_Normalized"])
        pkg_genric_summary = pd.DataFrame(
            columns=["Genric", "Package_Normalized", "Genric_Package"]
        )

    # ----- Join generic summary back to rows -----
    df_with_genric = df2.merge(
        genric_summary,
        left_on=generic_col,
        right_on="Genric",
        how="left",
        suffixes=("", "_Genric"),
    )

    return {
        "genric_summary": genric_summary,
        "company_summary": company_summary,
        "location_summary": location_summary,
        "pkg_file_summary": pkg_file_summary,
        "pkg_genric_summary": pkg_genric_summary,
        "df_with_genric": df_with_genric,
    }


# --------------------------
# 2) Stock/LT/Price engine
# --------------------------
def calc_generic_summary_price(
    df: pd.DataFrame,
    genric_col: str,
    supplier_col: str,
    qty_col: str,
    hasstock_col: str,
    minprice_col: str,
    avgprice_col: str,
    minlt_col: str,
    maxlt_col: str,
) -> pd.DataFrame:
    """Per-Genric summary (reimplemented from Script 2)."""
    df = df.copy()
    df["_has_stock"] = (df[hasstock_col] == 1) | (df[hasstock_col] == True)
    df["_qty_num"] = safe_numeric(df[qty_col]).fillna(0)
    df["_minp_num"] = safe_numeric(df[minprice_col])
    df["_avgp_num"] = safe_numeric(df[avgprice_col])
    df["_minlt_num"] = safe_numeric(df[minlt_col])
    df["_maxlt_num"] = safe_numeric(df[maxlt_col])

    group = df.groupby(genric_col, dropna=False)

    rows = []
    for genric, g in group:
        n_rows = len(g)
        n_suppliers = g[supplier_col].nunique(dropna=True)

        # stock
        g_stock = g[g["_has_stock"]]
        total_stock = g_stock["_qty_num"].sum()
        stock_rows = g_stock.shape[0]
        stock_rows_ratio = stock_rows / n_rows if n_rows else np.nan

        # price stats (ignore non-positive prices)
        g_price_valid = g[g["_minp_num"] > 0]
        if not g_price_valid.empty:
            min_minPrice = g_price_valid["_minp_num"].min()
            med_minPrice = g_price_valid["_minp_num"].median()
            max_minPrice = g_price_valid["_minp_num"].max()
            med_avgPrice = g_price_valid["_avgp_num"].median()
        else:
            min_minPrice = med_minPrice = max_minPrice = med_avgPrice = np.nan

        # price spread ratio
        if pd.notna(min_minPrice) and pd.notna(max_minPrice) and min_minPrice > 0:
            price_spread_ratio = max_minPrice / min_minPrice
        else:
            price_spread_ratio = np.nan

        # LT stats
        g_lt_min_valid = g[~g["_minlt_num"].isna()]
        g_lt_max_valid = g[~g["_maxlt_num"].isna()]
        min_MinLT = g_lt_min_valid["_minlt_num"].min() if not g_lt_min_valid.empty else np.nan
        med_MinLT = g_lt_min_valid["_minlt_num"].median() if not g_lt_min_valid.empty else np.nan
        max_MaxLT = g_lt_max_valid["_maxlt_num"].max() if not g_lt_max_valid.empty else np.nan

        if pd.notna(min_MinLT) and pd.notna(max_MaxLT):
            lt_span = max_MaxLT - min_MinLT
        else:
            lt_span = np.nan

        rows.append(
            {
                genric_col: genric,
                "Genric_rows": n_rows,
                "Genric_n_suppliers": n_suppliers,
                "Genric_total_stock": float(total_stock),
                "Genric_stock_rows": stock_rows,
                "Genric_stock_rows_ratio": stock_rows_ratio,
                "Genric_min_minPrice": min_minPrice,
                "Genric_med_minPrice": med_minPrice,
                "Genric_max_minPrice": max_minPrice,
                "Genric_med_avgPrice": med_avgPrice,
                "Genric_min_MinLT_Week": min_MinLT,
                "Genric_med_MinLT_Week": med_MinLT,
                "Genric_max_MaxLT_Week": max_MaxLT,
                "Genric_LT_span_weeks": lt_span,
                "Genric_price_spread_ratio": price_spread_ratio,
            }
        )

    generic_summary = pd.DataFrame(rows)

    # Stock buckets (quantile-based)
    pos_stock = generic_summary["Genric_total_stock"] > 0
    if pos_stock.any():
        q1 = generic_summary.loc[pos_stock, "Genric_total_stock"].quantile(0.33)
        q2 = generic_summary.loc[pos_stock, "Genric_total_stock"].quantile(0.66)

        def stock_bucket(row):
            if row["Genric_total_stock"] <= 0:
                return "NO_STOCK"
            if row["Genric_total_stock"] <= q1:
                return "LOW_STOCK"
            if row["Genric_total_stock"] <= q2:
                return "MEDIUM_STOCK"
            return "HIGH_STOCK"

    else:

        def stock_bucket(row):  # type: ignore[no-redef]
            return "NO_STOCK"

    generic_summary["Genric_stock_bucket"] = generic_summary.apply(stock_bucket, axis=1)

    # Price spread flags
    def price_spread_flag(v):
        if pd.isna(v):
            return "NO_PRICE_DATA"
        if v > 1000:
            return "SPREAD>1000x"
        if v > 100:
            return "SPREAD>100x"
        if v > 10:
            return "SPREAD>10x"
        return "SPREAD_NORMAL"

    generic_summary["Genric_price_spread_flag"] = generic_summary["Genric_price_spread_ratio"].apply(
        price_spread_flag
    )

    # LT span flag
    def lt_span_flag(v):
        if pd.isna(v):
            return "NO_LT_DATA"
        if v >= 52:
            return "SPAN>=52w"
        if v >= 26:
            return "SPAN>=26w"
        if v >= 8:
            return "SPAN>=8w"
        if v > 0:
            return "SPAN>0w"
        return "SPAN_ZERO"

    generic_summary["Genric_LT_span_flag"] = generic_summary["Genric_LT_span_weeks"].apply(lt_span_flag)

    return generic_summary


def calc_generic_supplier_summary_price(
    df: pd.DataFrame,
    genric_col: str,
    supplier_col: str,
    qty_col: str,
    hasstock_col: str,
    minprice_col: str,
    avgprice_col: str,
    minlt_col: str,
    maxlt_col: str,
) -> pd.DataFrame:
    """Per-(Genric, Supplier) summary."""
    df = df.copy()
    df["_has_stock"] = (df[hasstock_col] == 1) | (df[hasstock_col] == True)
    df["_qty_num"] = safe_numeric(df[qty_col]).fillna(0)
    df["_minp_num"] = safe_numeric(df[minprice_col])
    df["_avgp_num"] = safe_numeric(df[avgprice_col])
    df["_minlt_num"] = safe_numeric(df[minlt_col])
    df["_maxlt_num"] = safe_numeric(df[maxlt_col])

    group = df.groupby([genric_col, supplier_col], dropna=False)

    rows = []
    for (genric, sup), g in group:
        n_rows = len(g)
        g_stock = g[g["_has_stock"]]
        total_stock = g_stock["_qty_num"].sum()
        stock_rows = g_stock.shape[0]

        g_price_valid = g[g["_minp_num"] > 0]
        if not g_price_valid.empty:
            min_minPrice = g_price_valid["_minp_num"].min()
            med_minPrice = g_price_valid["_minp_num"].median()
            med_avgPrice = g_price_valid["_avgp_num"].median()
        else:
            min_minPrice = med_minPrice = med_avgPrice = np.nan

        g_lt_min_valid = g[~g["_minlt_num"].isna()]
        g_lt_max_valid = g[~g["_maxlt_num"].isna()]
        min_MinLT = g_lt_min_valid["_minlt_num"].min() if not g_lt_min_valid.empty else np.nan
        med_MinLT = g_lt_min_valid["_minlt_num"].median() if not g_lt_min_valid.empty else np.nan
        max_MaxLT = g_lt_max_valid["_maxlt_num"].max() if not g_lt_max_valid.empty else np.nan

        rows.append(
            {
                genric_col: genric,
                supplier_col: sup,
                "GS_rows": n_rows,
                "GS_stock_rows": stock_rows,
                "GS_total_stock": float(total_stock),
                "GS_min_minPrice": min_minPrice,
                "GS_med_minPrice": med_minPrice,
                "GS_med_avgPrice": med_avgPrice,
                "GS_min_MinLT_Week": min_MinLT,
                "GS_med_MinLT_Week": med_MinLT,
                "GS_max_MaxLT_Week": max_MaxLT,
            }
        )

    gs = pd.DataFrame(rows)
    return gs


def add_supply_concentration_price(
    generic_summary: pd.DataFrame,
    gs: pd.DataFrame,
    genric_col: str,
    supplier_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Given generic_summary and (Genric, Supplier) summary, compute stock shares & concentration flags."""

    # Join Genric_total_stock into gs
    gs = gs.merge(
        generic_summary[[genric_col, "Genric_total_stock"]],
        on=genric_col,
        how="left",
    )

    # Stock share in generic
    def _share(row):
        total_gen_stock = row["Genric_total_stock"]
        if pd.isna(total_gen_stock) or total_gen_stock <= 0:
            return np.nan
        return row["GS_total_stock"] / total_gen_stock

    gs["GS_stock_share_in_genric"] = gs.apply(_share, axis=1)

    # Concentration metrics per Genric
    conc_rows = []
    for genric, g in gs.groupby(genric_col, dropna=False):
        total_stock = g["Genric_total_stock"].iloc[0]

        g_with_stock = g[g["GS_total_stock"] > 0]
        n_suppliers_stock = g_with_stock.shape[0]

        top_supplier = None
        top_share = np.nan

        if total_stock is not None and total_stock > 0 and not g_with_stock.empty:
            g_sorted = g_with_stock.sort_values("GS_stock_share_in_genric", ascending=False)
            top_row = g_sorted.iloc[0]
            top_supplier = top_row[supplier_col]
            top_share = top_row["GS_stock_share_in_genric"]

        # concentration bucket
        if total_stock is None or total_stock <= 0:
            bucket = "NO_STOCK"
        elif n_suppliers_stock <= 1:
            bucket = "SINGLE_SUPPLIER"
        elif pd.isna(top_share):
            bucket = "UNKNOWN"
        elif top_share <= 0.6:
            bucket = "LOW_CONCENTRATION"
        elif top_share <= 0.9:
            bucket = "MEDIUM_CONCENTRATION"
        elif top_share <= 0.97:
            bucket = "HIGH_CONCENTRATION_90_97"
        else:
            bucket = "VERY_HIGH_CONCENTRATION_97PLUS"

        conc_rows.append(
            {
                genric_col: genric,
                "Genric_stock_suppliers_with_stock": n_suppliers_stock,
                "Genric_top_stock_supplier": top_supplier,
                "Genric_top_stock_share": top_share,
                "Genric_supply_concentration_bucket": bucket,
            }
        )

    conc_df = pd.DataFrame(conc_rows)

    # merge back into generic_summary
    generic_summary = generic_summary.merge(conc_df, on=genric_col, how="left")
    return generic_summary, gs


def calc_supplier_summary_price(
    df: pd.DataFrame,
    genric_col: str,
    supplier_col: str,
    qty_col: str,
    hasstock_col: str,
    minprice_col: str,
    avgprice_col: str,
    minlt_col: str,
    maxlt_col: str,
) -> pd.DataFrame:
    """Per-supplier summary across all generics."""
    df = df.copy()
    df["_has_stock"] = (df[hasstock_col] == 1) | (df[hasstock_col] == True)
    df["_qty_num"] = safe_numeric(df[qty_col]).fillna(0)
    df["_minp_num"] = safe_numeric(df[minprice_col])
    df["_avgp_num"] = safe_numeric(df[avgprice_col])
    df["_minlt_num"] = safe_numeric(df[minlt_col])
    df["_maxlt_num"] = safe_numeric(df[maxlt_col])

    group = df.groupby(supplier_col, dropna=False)

    rows = []
    for sup, g in group:
        n_rows = len(g)
        n_generics = g[genric_col].nunique(dropna=True)

        g_stock = g[g["_has_stock"]]
        total_stock = g_stock["_qty_num"].sum()
        stock_rows = g_stock.shape[0]
        stock_rows_ratio = stock_rows / n_rows if n_rows else np.nan

        g_price_valid = g[g["_minp_num"] > 0]
        if not g_price_valid.empty:
            med_minPrice = g_price_valid["_minp_num"].median()
            med_avgPrice = g_price_valid["_avgp_num"].median()
        else:
            med_minPrice = med_avgPrice = np.nan

        g_lt_min_valid = g[~g["_minlt_num"].isna()]
        min_MinLT = g_lt_min_valid["_minlt_num"].min() if not g_lt_min_valid.empty else np.nan
        med_MinLT = g_lt_min_valid["_minlt_num"].median() if not g_lt_min_valid.empty else np.nan

        rows.append(
            {
                supplier_col: sup,
                "Supplier_rows": n_rows,
                "Supplier_n_generics": n_generics,
                "Supplier_total_stock": float(total_stock),
                "Supplier_stock_rows": stock_rows,
                "Supplier_stock_rows_ratio": stock_rows_ratio,
                "Supplier_med_minPrice": med_minPrice,
                "Supplier_med_avgPrice": med_avgPrice,
                "Supplier_min_MinLT_Week": min_MinLT,
                "Supplier_med_MinLT_Week": med_MinLT,
            }
        )

    return pd.DataFrame(rows)


def build_stock_summaries(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Wrapper around Script 2 logic.

    cfg keys: genric, supplier, qty, hasstock, minprice, avgprice, minlt, maxlt
    """
    genric_col = cfg["genric"]
    supplier_col = cfg["supplier"]
    qty_col = cfg["qty"]
    hasstock_col = cfg["hasstock"]
    minprice_col = cfg["minprice"]
    avgprice_col = cfg["avgprice"]
    minlt_col = cfg["minlt"]
    maxlt_col = cfg["maxlt"]

    df2 = df.copy()

    generic_summary = calc_generic_summary_price(
        df2,
        genric_col,
        supplier_col,
        qty_col,
        hasstock_col,
        minprice_col,
        avgprice_col,
        minlt_col,
        maxlt_col,
    )

    gs = calc_generic_supplier_summary_price(
        df2,
        genric_col,
        supplier_col,
        qty_col,
        hasstock_col,
        minprice_col,
        avgprice_col,
        minlt_col,
        maxlt_col,
    )

    generic_summary, gs = add_supply_concentration_price(
        generic_summary,
        gs,
        genric_col,
        supplier_col,
    )

    supplier_summary = calc_supplier_summary_price(
        df2,
        genric_col,
        supplier_col,
        qty_col,
        hasstock_col,
        minprice_col,
        avgprice_col,
        minlt_col,
        maxlt_col,
    )

    # join generic flags back into input
    flags_cols = [
        "Genric_rows",
        "Genric_n_suppliers",
        "Genric_total_stock",
        "Genric_stock_rows",
        "Genric_stock_rows_ratio",
        "Genric_stock_bucket",
        "Genric_min_minPrice",
        "Genric_med_minPrice",
        "Genric_max_minPrice",
        "Genric_med_avgPrice",
        "Genric_price_spread_ratio",
        "Genric_price_spread_flag",
        "Genric_min_MinLT_Week",
        "Genric_med_MinLT_Week",
        "Genric_max_MaxLT_Week",
        "Genric_LT_span_weeks",
        "Genric_LT_span_flag",
        "Genric_stock_suppliers_with_stock",
        "Genric_top_stock_supplier",
        "Genric_top_stock_share",
        "Genric_supply_concentration_bucket",
    ]

    df_with_flags = df2.merge(
        generic_summary[[genric_col] + flags_cols],
        on=genric_col,
        how="left",
    )

    # meta info
    total_rows = len(df2)
    n_generics = df2[genric_col].nunique(dropna=True)
    n_suppliers = df2[supplier_col].nunique(dropna=True)

    df2["_has_stock_tmp"] = (df2[hasstock_col] == 1) | (df2[hasstock_col] == True)
    df2["_qty_tmp"] = safe_numeric(df2[qty_col]).fillna(0)

    meta = {
        "Metric": [
            "Total rows",
            "Unique generics",
            "Unique suppliers",
            "Rows with stock",
            "Rows without stock",
            "Total stock across file",
        ],
        "Value": [
            total_rows,
            n_generics,
            n_suppliers,
            int(df2["_has_stock_tmp"].sum()),
            int((~df2["_has_stock_tmp"]).sum()),
            float(df2.loc[df2["_has_stock_tmp"], "_qty_tmp"].sum()),
        ],
    }
    meta_df = pd.DataFrame(meta)

    # rename key column in generic_summary to standard "Genric" label
    generic_summary = generic_summary.rename(columns={genric_col: "Genric"})
    gs = gs.rename(columns={genric_col: "Genric", supplier_col: "Supplier"})

    return {
        "generic_summary": generic_summary,
        "generic_supplier_summary": gs,
        "supplier_summary": supplier_summary.rename(columns={supplier_col: "Supplier"}),
        "meta": meta_df,
        "df_with_flags": df_with_flags,
    }


# --------------------------
# 3) Chart helpers
# --------------------------
def draw_lifecycle_stacked_chart(
    summary_df: pd.DataFrame,
    category_col: str,
    title: str,
    top_n: int = 30,
):
    """Wide-to-long transform and stacked LC bar chart."""
    if summary_df is None or summary_df.empty:
        st.info("No data available for chart.")
        return

    lc_cols = [c for c in ["ActiveParts", "ObsoleteParts", "UnknownParts"] if c in summary_df.columns]
    if not lc_cols:
        st.info("Lifecycle columns (ActiveParts/ObsoleteParts/UnknownParts) not found in summary.")
        return

    df = summary_df[[category_col] + lc_cols].copy()
    df["TotalParts"] = df[lc_cols].sum(axis=1)
    df = df[df["TotalParts"] > 0]

    if df.empty:
        st.info("No non-zero lifecycle counts to chart.")
        return

    df = df.sort_values("TotalParts", ascending=False).head(top_n)

    long_df = df.melt(
        id_vars=[category_col],
        value_vars=lc_cols,
        var_name="Lifecycle",
        value_name="Count",
    )
    long_df["Lifecycle"] = long_df["Lifecycle"].str.replace("Parts", "", regex=False)

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{category_col}:N", sort="-y"),
            y=alt.Y("Count:Q", stack="zero"),
            color=alt.Color("Lifecycle:N"),
            tooltip=[category_col, "Lifecycle", "Count"],
        )
        .properties(title=title)
    )

    st.altair_chart(chart, use_container_width=True)


def draw_adhoc_chart(df: pd.DataFrame, dataset_label: str):
    """Generic 'group by / aggregate / chart' builder used in the Ad-hoc Explorer."""
    if df is None or df.empty:
        st.info("No data in selected dataset.")
        return

    st.subheader(f"Ad-hoc chart on: {dataset_label}")

    cols = list(df.columns)
    if not cols:
        st.info("Dataset has no columns.")
        return

    # dimension candidates: all columns
    dim_options = cols

    # numeric candidates
    num_options = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    gb1 = st.selectbox(
        "Group by (X axis / main category)",
        dim_options,
        key=f"{dataset_label}_gb1",
    )

    gb2 = st.selectbox(
        "Optional second group (color)",
        ["<None>"] + dim_options,
        key=f"{dataset_label}_gb2",
    )

    agg_mode = st.selectbox(
        "Aggregation function",
        ["Count rows", "Sum", "Mean", "Median"],
        key=f"{dataset_label}_agg",
    )

    metric_col = None
    if agg_mode != "Count rows":
        if not num_options:
            st.warning("No numeric columns available for aggregation; falling back to row count.")
            agg_mode = "Count rows"
        else:
            metric_col = st.selectbox(
                "Value column",
                num_options,
                key=f"{dataset_label}_metric",
            )

    top_n = st.slider(
        "Top N categories (applied on primary group)",
        5,
        100,
        20,
        key=f"{dataset_label}_topn",
    )

    group_cols = [gb1]
    use_second = gb2 != "<None>"
    if use_second:
        group_cols.append(gb2)

    g = df.copy()

    if agg_mode == "Count rows":
        agg_df = g.groupby(group_cols, dropna=False).size().reset_index(name="value")
    else:
        func_map = {"Sum": "sum", "Mean": "mean", "Median": "median"}
        agg_func = func_map[agg_mode]
        agg_df = (
            g.groupby(group_cols, dropna=False)[metric_col]
            .agg(agg_func)
            .reset_index(name="value")
        )

    # Restrict to top N on primary group
    if use_second:
        ranking = (
            agg_df.groupby(gb1)["value"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .head(top_n)
        )
        agg_df = agg_df.merge(ranking[[gb1]], on=gb1, how="inner")
    else:
        agg_df = agg_df.sort_values("value", ascending=False).head(top_n)

    if agg_df.empty:
        st.info("No data left after aggregation and Top-N filtering.")
        return

    chart = alt.Chart(agg_df).mark_bar()

    enc_kwargs = {
        "x": alt.X(f"{gb1}:N", sort="-y"),
        "y": alt.Y("value:Q"),
        "tooltip": [gb1, "value"],
    }
    if use_second:
        enc_kwargs["color"] = alt.Color(f"{gb2}:N")
        enc_kwargs["tooltip"].append(gb2)

    chart = chart.encode(**enc_kwargs).properties(title=f"{agg_mode} of selected metric")

    st.altair_chart(chart, use_container_width=True)
    st.caption("Aggregated data used for the chart:")
    st.dataframe(agg_df)


# --------------------------
# 4) Main Streamlit app
# --------------------------
def main():
    st.title("Generic Lifecycle & Stock / Lead-Time / Price Dashboard")
    st.caption(
        "Upload your two Excel files (G_A / Price_G). "
        "Map the columns once via the UI, then explore dynamic counts and charts "
        "per Generic, Supplier, Package, Country, or any other dimension."
    )

    with st.sidebar:
        st.header("1. Upload input files")
        lc_file = st.file_uploader(
            "Lifecycle / Generic file (Script 1 input, e.g. G_A.xlsx)",
            type=["xls", "xlsx"],
            key="lc_file",
        )
        stock_file = st.file_uploader(
            "Stock / LT / Price file (Script 2 input, e.g. Price_G.xlsx)",
            type=["xls", "xlsx"],
            key="stock_file",
        )

    df_lc = df_stock = None

    if lc_file is not None:
        try:
            df_lc = pd.read_excel(lc_file)
        except Exception as e:
            st.error(f"Failed to read lifecycle/generic file: {e}")

    if stock_file is not None:
        try:
            df_stock = pd.read_excel(stock_file)
        except Exception as e:
            st.error(f"Failed to read stock/price file: {e}")

    tabs = st.tabs(
        [
            "Lifecycle / LC (Script 1)",
            "Stock / LT / Price (Script 2)",
            "Ad-hoc Explorer",
            "Help",
        ]
    )

    # ---------------- Lifecycle tab ----------------
    with tabs[0]:
        st.header("Lifecycle / LC Analysis (Generic / Company / Package / Location)")

        if df_lc is None:
            st.info("Upload the lifecycle/generic file in the sidebar to start.")
        else:
            st.write(f"**File shape:** {df_lc.shape[0]} rows × {df_lc.shape[1]} columns")
            with st.expander("Preview lifecycle input (first 200 rows)", expanded=False):
                st.dataframe(df_lc.head(200))

            st.subheader("Column mapping (you can choose any columns – names are just hints)")

            lc_col1, lc_col2, lc_col3 = st.columns(3)
            lc_col4, lc_col5, lc_col6 = st.columns(3)
            lc_col7, lc_col8 = st.columns(2)

            # Candidate lists reused from your script, but only as *hints*
            GENERIC_COL_CANDIDATES = ["Genric", "Generic", "GenericKey"]
            PARTID_COL_CANDIDATES = ["PartID", "Part Id", "Part_ID"]
            PARTNUM_COL_CANDIDATES = ["PartNumber", "Part Number"]
            COMPANY_COL_CANDIDATES = ["CompanyName", "Company Name", "Manufacturer"]
            PARTLC_COL_CANDIDATES = ["PartLC", "Lifecycle", "LifeCycleStatus"]
            PARTINTRO_COL_CANDIDATES = ["PartIntro", "Part Introduction", "Part_Intro_Year"]
            FAMILYINTRO_COL_CANDIDATES = [
                "FamilyIntro",
                "Family Introduction",
                "Family_Intro_Year",
            ]
            STAGE_COL_CANDIDATES = ["Stage"]
            RISK_COL_CANDIDATES = ["Risk"]
            COUNTRY_COL_CANDIDATES = ["Country", "CountryName"]
            PKG_NORM_COL_CANDIDATES = [
                "Package_NormalizedPackageName",
                "Package_Norm",
                "PackageNorm",
            ]

            cols_list = list(df_lc.columns)
            none_option = "<None>"

            # Required: primary key / generic
            generic_guess = guess_column(df_lc, GENERIC_COL_CANDIDATES) or cols_list[0]
            generic_col = lc_col1.selectbox(
                "Primary key column (Genric / Family / etc.)",
                cols_list,
                index=cols_list.index(generic_guess),
            )

            # Optional mappings
            partid_guess = guess_column(df_lc, PARTID_COL_CANDIDATES)
            partid_col = lc_col2.selectbox(
                "PartID column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(partid_guess)) if partid_guess else 0,
            )
            partnum_guess = guess_column(df_lc, PARTNUM_COL_CANDIDATES)
            partnum_col = lc_col3.selectbox(
                "PartNumber column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(partnum_guess)) if partnum_guess else 0,
            )

            company_guess = guess_column(df_lc, COMPANY_COL_CANDIDATES)
            company_col = lc_col4.selectbox(
                "Company / Manufacturer column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(company_guess)) if company_guess else 0,
            )

            partlc_guess = guess_column(df_lc, PARTLC_COL_CANDIDATES)
            partlc_col = lc_col5.selectbox(
                "Lifecycle column (Active / Obsolete / Unknown)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(partlc_guess)) if partlc_guess else 0,
            )

            partintro_guess = guess_column(df_lc, PARTINTRO_COL_CANDIDATES)
            partintro_col = lc_col6.selectbox(
                "Part Intro Year column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(partintro_guess)) if partintro_guess else 0,
            )

            familyintro_guess = guess_column(df_lc, FAMILYINTRO_COL_CANDIDATES)
            familyintro_col = lc_col7.selectbox(
                "Family Intro Year column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(familyintro_guess)) if familyintro_guess else 0,
            )

            stage_guess = guess_column(df_lc, STAGE_COL_CANDIDATES)
            stage_col = lc_col8.selectbox(
                "Stage column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(stage_guess)) if stage_guess else 0,
            )

            risk_col = st.selectbox(
                "Risk column (optional)",
                [none_option] + cols_list,
                index=(
                    1 + cols_list.index(guess_column(df_lc, RISK_COL_CANDIDATES))
                    if guess_column(df_lc, RISK_COL_CANDIDATES)
                    else 0
                ),
            )

            country_col = st.selectbox(
                "Country / Location column (optional, '|' separated)",
                [none_option] + cols_list,
                index=(
                    1 + cols_list.index(guess_column(df_lc, COUNTRY_COL_CANDIDATES))
                    if guess_column(df_lc, COUNTRY_COL_CANDIDATES)
                    else 0
                ),
            )

            pkg_norm_col = st.selectbox(
                "Normalized package column (optional)",
                [none_option] + cols_list,
                index=(
                    1 + cols_list.index(guess_column(df_lc, PKG_NORM_COL_CANDIDATES))
                    if guess_column(df_lc, PKG_NORM_COL_CANDIDATES)
                    else 0
                ),
            )

            lc_cfg = {
                "generic": generic_col,
                "partid": None if partid_col == none_option else partid_col,
                "partnum": None if partnum_col == none_option else partnum_col,
                "company": None if company_col == none_option else company_col,
                "partlc": None if partlc_col == none_option else partlc_col,
                "partintro": None if partintro_col == none_option else partintro_col,
                "familyintro": None if familyintro_col == none_option else familyintro_col,
                "stage": None if stage_col == none_option else stage_col,
                "risk": None if risk_col == none_option else risk_col,
                "country": None if country_col == none_option else country_col,
                "pkg_norm": None if pkg_norm_col == none_option else pkg_norm_col,
            }

            # Build summaries
            try:
                lc_results = build_lifecycle_summaries(df_lc, lc_cfg)
            except Exception as e:
                st.error(f"Error while computing lifecycle summaries: {e}")
                lc_results = None

            if lc_results is not None:
                genric_summary = lc_results["genric_summary"]

                # High-level metrics
                st.subheader("High-level metrics")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows", len(df_lc))
                m2.metric("Unique generics", df_lc[generic_col].nunique(dropna=True))

                base_count_col = lc_cfg["partid"] or lc_cfg["partnum"]
                if base_count_col:
                    m3.metric(
                        f"Unique parts ({base_count_col})",
                        df_lc[base_count_col].nunique(dropna=True),
                    )
                if lc_cfg["partlc"]:
                    active, obsolete, unknown, total = lc_counts(df_lc[lc_cfg["partlc"]])
                    m4.metric("Active parts", active)

                # Lifecycle stacked chart per generic
                st.subheader("Lifecycle distribution by generic")
                top_n_gen = st.slider(
                    "Top N generics by total parts",
                    5,
                    100,
                    30,
                    key="lc_topn_gen",
                )
                draw_lifecycle_stacked_chart(
                    genric_summary,
                    category_col="Genric",
                    title="Lifecycle by Generic (Active / Obsolete / Unknown)",
                    top_n=top_n_gen,
                )

                # Summary tables chooser
                st.subheader("Summary tables")
                table_choice = st.selectbox(
                    "Choose which table to show",
                    [
                        "Genric summary",
                        "Company LC summary",
                        "Location LC summary",
                        "Package LC summary (file-level)",
                        "Package LC summary (Genric-level)",
                        "Input with Genric annotations",
                    ],
                )

                if table_choice == "Genric summary":
                    st.dataframe(genric_summary)
                elif table_choice == "Company LC summary":
                    st.dataframe(lc_results["company_summary"])
                elif table_choice == "Location LC summary":
                    st.dataframe(lc_results["location_summary"])
                elif table_choice == "Package LC summary (file-level)":
                    st.dataframe(lc_results["pkg_file_summary"])
                elif table_choice == "Package LC summary (Genric-level)":
                    st.dataframe(lc_results["pkg_genric_summary"])
                else:
                    st.dataframe(lc_results["df_with_genric"])

    # ---------------- Stock / Price tab ----------------
    with tabs[1]:
        st.header("Stock / Lead-Time / Price Analysis")

        if df_stock is None:
            st.info("Upload the stock/price file in the sidebar to start.")
        else:
            st.write(f"**File shape:** {df_stock.shape[0]} rows × {df_stock.shape[1]} columns")
            with st.expander("Preview stock/price input (first 200 rows)", expanded=False):
                st.dataframe(df_stock.head(200))

            st.subheader("Column mapping (Script 2 roles)")

            cols_list = list(df_stock.columns)

            # Candidate lists from Script 2
            GENRIC_COL_CANDIDATES = ["Genric", "Generic", "GenericKey", "GenricKey"]
            SUPPLIER_COL_CANDIDATES = ["ZCompanyName", "CompanyName", "Supplier", "Vendor"]
            QTY_COL_CANDIDATES = ["totalQuantity", "Total Quantity", "Qty", "Quantity"]
            HASSTOCK_COL_CANDIDATES = ["hasStock", "HasStock", "InStockFlag", "StockFlag"]
            MINPRICE_COL_CANDIDATES = ["minPrice", "Min Price", "Minimum Price"]
            AVGPRICE_COL_CANDIDATES = ["avgPrice", "Average Price", "Avg Price"]
            MINLT_WEEK_COL_CANDIDATES = [
                "MinLT_Week",
                "Min LT Week",
                "MinLeadTimeWeek",
                "MinLT",
            ]
            MAXLT_WEEK_COL_CANDIDATES = [
                "MaxLT_Week",
                "Max LT Week",
                "MaxLeadTimeWeek",
                "MaxLT",
            ]

            col1, col2, col3, col4 = st.columns(4)
            col5, col6, col7, col8 = st.columns(4)

            genric_guess = guess_column(df_stock, GENRIC_COL_CANDIDATES) or cols_list[0]
            genric_col = col1.selectbox(
                "Generic / Key column",
                cols_list,
                index=cols_list.index(genric_guess),
            )

            supplier_guess = guess_column(df_stock, SUPPLIER_COL_CANDIDATES) or cols_list[1]
            supplier_col = col2.selectbox(
                "Supplier / Company column",
                cols_list,
                index=cols_list.index(supplier_guess),
            )

            qty_guess = guess_column(df_stock, QTY_COL_CANDIDATES) or cols_list[2]
            qty_col = col3.selectbox(
                "Quantity column",
                cols_list,
                index=cols_list.index(qty_guess),
            )

            hasstock_guess = guess_column(df_stock, HASSTOCK_COL_CANDIDATES) or cols_list[3]
            hasstock_col = col4.selectbox(
                "Has stock flag column (1/0 or True/False)",
                cols_list,
                index=cols_list.index(hasstock_guess),
            )

            minprice_guess = guess_column(df_stock, MINPRICE_COL_CANDIDATES) or cols_list[4]
            minprice_col = col5.selectbox(
                "Min price column",
                cols_list,
                index=cols_list.index(minprice_guess),
            )

            avgprice_guess = guess_column(df_stock, AVGPRICE_COL_CANDIDATES) or cols_list[5]
            avgprice_col = col6.selectbox(
                "Avg price column",
                cols_list,
                index=cols_list.index(avgprice_guess),
            )

            minlt_guess = guess_column(df_stock, MINLT_WEEK_COL_CANDIDATES) or cols_list[6]
            minlt_col = col7.selectbox(
                "Min lead time (weeks) column",
                cols_list,
                index=cols_list.index(minlt_guess),
            )

            maxlt_guess = guess_column(df_stock, MAXLT_WEEK_COL_CANDIDATES) or cols_list[7]
            maxlt_col = col8.selectbox(
                "Max lead time (weeks) column",
                cols_list,
                index=cols_list.index(maxlt_guess),
            )

            stock_cfg = {
                "genric": genric_col,
                "supplier": supplier_col,
                "qty": qty_col,
                "hasstock": hasstock_col,
                "minprice": minprice_col,
                "avgprice": avgprice_col,
                "minlt": minlt_col,
                "maxlt": maxlt_col,
            }

            try:
                stock_results = build_stock_summaries(df_stock, stock_cfg)
            except Exception as e:
                st.error(f"Error while computing stock/price summaries: {e}")
                stock_results = None

            if stock_results is not None:
                generic_summary = stock_results["generic_summary"]
                generic_supplier_summary = stock_results["generic_supplier_summary"]
                supplier_summary = stock_results["supplier_summary"]
                meta_df = stock_results["meta"]
                df_with_flags = stock_results["df_with_flags"]

                st.subheader("High-level metrics")
                m1, m2, m3, m4 = st.columns(4)

                m1.metric("Rows", len(df_stock))
                m2.metric("Unique generics", df_stock[genric_col].nunique(dropna=True))
                m3.metric("Unique suppliers", df_stock[supplier_col].nunique(dropna=True))
                m4.metric("Total stock", float(df_with_flags[stock_cfg["qty"]].fillna(0).sum()))

                st.subheader("Top generics by total stock & concentration")
                top_n_stock = st.slider(
                    "Top N generics (by total stock)",
                    5,
                    100,
                    30,
                    key="stock_topn_gen",
                )
                g_chart = generic_summary.dropna(subset=["Genric_total_stock"]).copy()
                g_chart = g_chart.sort_values("Genric_total_stock", ascending=False).head(
                    top_n_stock
                )

                if not g_chart.empty:
                    chart = (
                        alt.Chart(g_chart)
                        .mark_bar()
                        .encode(
                            x=alt.X("Genric:N", sort="-y"),
                            y=alt.Y("Genric_total_stock:Q", title="Total stock"),
                            color=alt.Color(
                                "Genric_supply_concentration_bucket:N",
                                title="Concentration bucket",
                            ),
                            tooltip=[
                                "Genric",
                                "Genric_total_stock",
                                "Genric_stock_suppliers_with_stock",
                                "Genric_supply_concentration_bucket",
                            ],
                        )
                        .properties(title="Total stock by generic (colored by concentration)")
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No stock data available to chart.")

                st.subheader("Generics with large price spread (min vs. max)")
                spread_threshold = st.selectbox(
                    "Highlight generics with price spread flag",
                    ["SPREAD>10x", "SPREAD>100x", "SPREAD>1000x"],
                )
                spread_df = generic_summary[
                    generic_summary["Genric_price_spread_flag"].isin(
                        [spread_threshold, "SPREAD>100x", "SPREAD>1000x"]
                        if spread_threshold == "SPREAD>10x"
                        else (
                            [spread_threshold, "SPREAD>1000x"]
                            if spread_threshold == "SPREAD>100x"
                            else [spread_threshold]
                        )
                    )
                ]
                st.dataframe(spread_df)

                st.subheader("Summary tables")
                table_choice = st.selectbox(
                    "Choose which table to show",
                    [
                        "Generic summary",
                        "Generic+Supplier summary",
                        "Supplier summary",
                        "Meta summary",
                        "Input with flags",
                    ],
                )

                if table_choice == "Generic summary":
                    st.dataframe(generic_summary)
                elif table_choice == "Generic+Supplier summary":
                    st.dataframe(generic_supplier_summary)
                elif table_choice == "Supplier summary":
                    st.dataframe(supplier_summary)
                elif table_choice == "Meta summary":
                    st.dataframe(meta_df)
                else:
                    st.dataframe(df_with_flags)

    # ---------------- Ad-hoc Explorer tab ----------------
    with tabs[2]:
        st.header("Ad-hoc Explorer (build any chart from any table)")

        datasets: dict[str, pd.DataFrame] = {}

        if df_lc is not None:
            datasets["LC: Input file"] = df_lc
            # We can recompute LC results with same config if it exists in this rerun
            try:
                lc_results  # type: ignore[name-defined]
                if lc_results is not None:
                    datasets["LC: Genric summary"] = lc_results["genric_summary"]
                    if not lc_results["company_summary"].empty:
                        datasets["LC: Company summary"] = lc_results["company_summary"]
                    if not lc_results["location_summary"].empty:
                        datasets["LC: Location summary"] = lc_results["location_summary"]
                    if not lc_results["pkg_file_summary"].empty:
                        datasets["LC: Package summary (file-level)"] = lc_results[
                            "pkg_file_summary"
                        ]
                    if not lc_results["pkg_genric_summary"].empty:
                        datasets["LC: Package summary (Genric-level)"] = lc_results[
                            "pkg_genric_summary"
                        ]
                    datasets["LC: Input with Genric annotations"] = lc_results[
                        "df_with_genric"
                    ]
            except NameError:
                pass

        if df_stock is not None:
            datasets["Stock: Input file"] = df_stock
            try:
                stock_results  # type: ignore[name-defined]
                if stock_results is not None:
                    datasets["Stock: Generic summary"] = stock_results["generic_summary"]
                    datasets["Stock: Generic+Supplier summary"] = stock_results[
                        "generic_supplier_summary"
                    ]
                    datasets["Stock: Supplier summary"] = stock_results["supplier_summary"]
                    datasets["Stock: Input with flags"] = stock_results["df_with_flags"]
            except NameError:
                pass

        if not datasets:
            st.info(
                "Upload at least one file and build the base summaries in the first two tabs "
                "to unlock the ad-hoc explorer."
            )
        else:
            ds_name = st.selectbox("Choose dataset", list(datasets.keys()))
            draw_adhoc_chart(datasets[ds_name], ds_name)

    # ---------------- Help tab ----------------
    with tabs[3]:
        st.header("How to use this app")

        st.markdown(
            """
1. **Upload your two Excel files** in the sidebar:
   - The lifecycle/generic file (Script 1 input, e.g. `G_A.xlsx`).
   - The stock/lead-time/price file (Script 2 input, e.g. `Price_G.xlsx`).

2. In each tab, **map the columns**:
   - The app tries to guess the right columns using your original scripts.
   - You can override everything – you are **not restricted** to exact column names.
   - For example, you can choose any column as the primary key (Generic / Family / Die / etc.).

3. After mapping:
   - The **Lifecycle tab** builds generic/company/package/location summaries and a stacked LC chart.
   - The **Stock tab** builds generic/supplier summaries, supply concentration flags, and price/lead-time insights.

4. In the **Ad-hoc Explorer**:
   - Pick *any* table (raw input, generic summary, supplier summary, etc.).
   - Choose how to group (1 or 2 dimensions), pick the aggregation (Count / Sum / Mean / Median),
     and instantly get a chart + aggregated table.

You can put this file on GitHub as `app.py` and run it on Streamlit Cloud.
Just make sure your `requirements.txt` includes at least:
`streamlit`, `pandas`, `numpy`, `altair`, and `openpyxl`.
"""
        )


if __name__ == "__main__":
    main()
