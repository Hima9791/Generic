import pandas as pd
import numpy as np

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

    rows = []
    for genric, g in group:
        n_rows = len(g)
        n_suppliers = g[supplier_col].nunique(dropna=True)

        g_stock = g[g["_has_stock"]]
        total_stock = g_stock["_qty_num"].sum()
        stock_rows = g_stock.shape[0]
        stock_rows_ratio = stock_rows / n_rows if n_rows else np.nan

        g_price_valid = g[g["_minp_num"] > 0]
        if not g_price_valid.empty:
            min_minPrice = g_price_valid["_minp_num"].min()
            med_minPrice = g_price_valid["_minp_num"].median()
            max_minPrice = g_price_valid["_minp_num"].max()
            med_avgPrice = g_price_valid["_avgp_num"].median()
        else:
            min_minPrice = med_minPrice = max_minPrice = med_avgPrice = np.nan

        if pd.notna(min_minPrice) and pd.notna(max_minPrice) and min_minPrice > 0:
            price_spread_ratio = max_minPrice / min_minPrice
        else:
            price_spread_ratio = np.nan

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

    generic_summary["Genric_price_spread_flag"] = generic_summary[
        "Genric_price_spread_ratio"
    ].apply(price_spread_flag)

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

    generic_summary["Genric_LT_span_flag"] = generic_summary[
        "Genric_LT_span_weeks"
    ].apply(lt_span_flag)

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

    return pd.DataFrame(rows)


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
