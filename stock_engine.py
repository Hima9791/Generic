import numpy as np
import pandas as pd

from utils import safe_numeric


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
    """Per-Genric summary."""
    df = df.copy()
    df["_has_stock"] = (df[hasstock_col] == 1) | (df[hasstock_col] == True)
    df["_qty_num"] = safe_numeric(df[qty_col]).fillna(0)
    df["_minp_num"] = safe_numeric(df[minprice_col])
    df["_avgp_num"] = safe_numeric(df[avgprice_col])
    df["_minlt_num"] = safe_numeric(df[minlt_col])
    df["_maxlt_num"] = safe_numeric(df[maxlt_col])

    group = df.groupby(genric_col, dropna=False)

    base = pd.DataFrame(
        {
            "Genric_rows": group.size(),
            "Genric_n_suppliers": group[supplier_col].nunique(dropna=True),
        }
    )

    stock_group = df.loc[df["_has_stock"]].groupby(genric_col, dropna=False)
    base["Genric_total_stock"] = stock_group["_qty_num"].sum()
    base["Genric_stock_rows"] = stock_group.size()
    base["Genric_stock_rows_ratio"] = base["Genric_stock_rows"] / base[
        "Genric_rows"
    ]

    price_valid = df[df["_minp_num"] > 0].groupby(genric_col, dropna=False)
    base["Genric_min_minPrice"] = price_valid["_minp_num"].min()
    base["Genric_med_minPrice"] = price_valid["_minp_num"].median()
    base["Genric_max_minPrice"] = price_valid["_minp_num"].max()
    base["Genric_med_avgPrice"] = price_valid["_avgp_num"].median()

    lt_group = df.groupby(genric_col, dropna=False)
    base["Genric_min_MinLT_Week"] = lt_group["_minlt_num"].min()
    base["Genric_med_MinLT_Week"] = lt_group["_minlt_num"].median()
    base["Genric_max_MaxLT_Week"] = lt_group["_maxlt_num"].max()

    base["Genric_total_stock"] = base["Genric_total_stock"].fillna(0).astype(float)
    base["Genric_stock_rows"] = base["Genric_stock_rows"].fillna(0).astype(int)
    base["Genric_stock_rows_ratio"] = base["Genric_stock_rows_ratio"]
    base["Genric_price_spread_ratio"] = base["Genric_max_minPrice"] / base[
        "Genric_min_minPrice"
    ]
    base["Genric_LT_span_weeks"] = (
        base["Genric_max_MaxLT_Week"] - base["Genric_min_MinLT_Week"]
    )

    generic_summary = base.reset_index().rename(columns={genric_col: genric_col})

    pos_stock = generic_summary["Genric_total_stock"] > 0
    if pos_stock.any():
        q1 = generic_summary.loc[pos_stock, "Genric_total_stock"].quantile(0.33)
        q2 = generic_summary.loc[pos_stock, "Genric_total_stock"].quantile(0.66)
        generic_summary["Genric_stock_bucket"] = np.select(
            [~pos_stock, generic_summary["Genric_total_stock"] <= q1, generic_summary["Genric_total_stock"] <= q2],
            ["NO_STOCK", "LOW_STOCK", "MEDIUM_STOCK"],
            default="HIGH_STOCK",
        )
    else:
        generic_summary["Genric_stock_bucket"] = "NO_STOCK"

    generic_summary["Genric_price_spread_flag"] = pd.Series(
        np.select(
            [
                generic_summary["Genric_price_spread_ratio"].isna(),
                generic_summary["Genric_price_spread_ratio"] > 1000,
                generic_summary["Genric_price_spread_ratio"] > 100,
                generic_summary["Genric_price_spread_ratio"] > 10,
            ],
            ["NO_PRICE_DATA", "SPREAD>1000x", "SPREAD>100x", "SPREAD>10x"],
            default="SPREAD_NORMAL",
        )
    )

    generic_summary["Genric_LT_span_flag"] = pd.Series(
        np.select(
            [
                generic_summary["Genric_LT_span_weeks"].isna(),
                generic_summary["Genric_LT_span_weeks"] >= 52,
                generic_summary["Genric_LT_span_weeks"] >= 26,
                generic_summary["Genric_LT_span_weeks"] >= 8,
                generic_summary["Genric_LT_span_weeks"] > 0,
            ],
            ["NO_LT_DATA", "SPAN>=52w", "SPAN>=26w", "SPAN>=8w", "SPAN>0w"],
            default="SPAN_ZERO",
        )
    )

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

    keys = [genric_col, supplier_col]
    group = df.groupby(keys, dropna=False)

    base = pd.DataFrame(
        {
            "GS_rows": group.size(),
            "GS_stock_rows": df.loc[df["_has_stock"]].groupby(keys, dropna=False).size(),
            "GS_total_stock": df.loc[df["_has_stock"]]
            .groupby(keys, dropna=False)["_qty_num"]
            .sum(),
        }
    )

    price_valid = df[df["_minp_num"] > 0].groupby(keys, dropna=False)
    base["GS_min_minPrice"] = price_valid["_minp_num"].min()
    base["GS_med_minPrice"] = price_valid["_minp_num"].median()
    base["GS_med_avgPrice"] = price_valid["_avgp_num"].median()

    base["GS_min_MinLT_Week"] = group["_minlt_num"].min()
    base["GS_med_MinLT_Week"] = group["_minlt_num"].median()
    base["GS_max_MaxLT_Week"] = group["_maxlt_num"].max()

    base["GS_total_stock"] = base["GS_total_stock"].fillna(0).astype(float)
    base["GS_stock_rows"] = base["GS_stock_rows"].fillna(0).astype(int)

    return base.reset_index()


def add_supply_concentration_price(
    generic_summary: pd.DataFrame,
    gs: pd.DataFrame,
    genric_col: str,
    supplier_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add supplier concentration metrics per Genric."""
    gs = gs.merge(
        generic_summary[[genric_col, "Genric_total_stock"]],
        on=genric_col,
        how="left",
    )

    def _share(row):
        total_gen_stock = row["Genric_total_stock"]
        if pd.isna(total_gen_stock) or total_gen_stock <= 0:
            return np.nan
        return row["GS_total_stock"] / total_gen_stock

    gs["GS_stock_share_in_genric"] = gs.apply(_share, axis=1)

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

    base = pd.DataFrame(
        {
            "Supplier_rows": group.size(),
            "Supplier_n_generics": group[genric_col].nunique(dropna=True),
        }
    )

    stock_group = df.loc[df["_has_stock"]].groupby(supplier_col, dropna=False)
    base["Supplier_total_stock"] = stock_group["_qty_num"].sum()
    base["Supplier_stock_rows"] = stock_group.size()
    base["Supplier_stock_rows_ratio"] = base["Supplier_stock_rows"] / base[
        "Supplier_rows"
    ]

    price_valid = df[df["_minp_num"] > 0].groupby(supplier_col, dropna=False)
    base["Supplier_med_minPrice"] = price_valid["_minp_num"].median()
    base["Supplier_med_avgPrice"] = price_valid["_avgp_num"].median()

    base["Supplier_min_MinLT_Week"] = group["_minlt_num"].min()
    base["Supplier_med_MinLT_Week"] = group["_minlt_num"].median()

    base["Supplier_total_stock"] = base["Supplier_total_stock"].fillna(0).astype(float)
    base["Supplier_stock_rows"] = base["Supplier_stock_rows"].fillna(0).astype(int)

    return base.reset_index()


def build_stock_summaries(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Wrapper around all stock/price calculations.

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

    generic_summary = generic_summary.rename(columns={genric_col: "Genric"})
    gs = gs.rename(columns={genric_col: "Genric", supplier_col: "Supplier"})
    supplier_summary = supplier_summary.rename(columns={supplier_col: "Supplier"})

    return {
        "generic_summary": generic_summary,
        "generic_supplier_summary": gs,
        "supplier_summary": supplier_summary,
        "meta": meta_df,
        "df_with_flags": df_with_flags,
    }
