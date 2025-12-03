import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


def draw_lifecycle_stacked_chart(
    summary_df: pd.DataFrame,
    category_col: str,
    title: str,
    top_n: int = 30,
):
    """Stacked bar chart: Active / Obsolete / Unknown per category."""
    if summary_df is None or summary_df.empty:
        st.info("No data available for lifecycle chart.")
        return

    lc_cols = [c for c in ["ActiveParts", "ObsoleteParts", "UnknownParts"] if c in summary_df.columns]
    if not lc_cols:
        st.info("Lifecycle columns not found (ActiveParts / ObsoleteParts / UnknownParts).")
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
    """Generic 'group-by / aggregate / chart' builder with % of total and % within group."""
    if df is None or df.empty:
        st.info("No data in selected dataset.")
        return

    st.subheader(f"Ad-hoc chart on: {dataset_label}")

    cols = list(df.columns)
    if not cols:
        st.info("Dataset has no columns.")
        return

    dim_options = cols
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
        [
            "Count rows",
            "Count distinct of column",
            "Sum",
            "Mean",
            "Median",
        ],
        key=f"{dataset_label}_agg",
    )

    metric_col = None
    if agg_mode == "Count distinct of column":
        metric_col = st.selectbox(
            "Column to count distinct values",
            cols,
            key=f"{dataset_label}_distinct_metric",
        )
    elif agg_mode != "Count rows":
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
    elif agg_mode == "Count distinct of column":
        agg_df = (
            g.groupby(group_cols, dropna=False)[metric_col]
            .nunique(dropna=True)
            .reset_index(name="value")
        )
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

    chart_type = st.selectbox(
        "Chart type",
        ["Bar", "Line", "Area", "Scatter"],
        key=f"{dataset_label}_chart_type",
    )

    # Percentages toggles
    show_pct_total = st.checkbox(
        "Add % of grand total",
        value=True,
        key=f"{dataset_label}_pct_total",
    )
    show_pct_group = st.checkbox(
        f"Add % within primary group ({gb1})",
        value=True,
        key=f"{dataset_label}_pct_group",
    )

    # Compute percentages
    grand_total = agg_df["value"].sum()
    if pd.notna(grand_total) and grand_total != 0:
        agg_df["Pct_of_total"] = (agg_df["value"] / grand_total * 100).round(2)

    pct_group_col_name = None
    if use_second:
        group_total = agg_df.groupby(gb1)["value"].transform("sum")
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_vals = np.where(
                group_total != 0,
                (agg_df["value"] / group_total * 100).round(2),
                np.nan,
            )
        pct_group_col_name = f"Pct_within_{gb1}"
        agg_df[pct_group_col_name] = pct_vals

    mark_map = {
        "Bar": alt.Chart(agg_df).mark_bar(),
        "Line": alt.Chart(agg_df).mark_line(point=True),
        "Area": alt.Chart(agg_df).mark_area(),
        "Scatter": alt.Chart(agg_df).mark_circle(size=60),
    }
    chart = mark_map.get(chart_type, alt.Chart(agg_df).mark_bar())

    tooltip_cols = [gb1, "value"]
    if use_second:
        tooltip_cols.append(gb2)
    if show_pct_total and "Pct_of_total" in agg_df.columns:
        tooltip_cols.append("Pct_of_total")
    if show_pct_group and pct_group_col_name and pct_group_col_name in agg_df.columns:
        tooltip_cols.append(pct_group_col_name)

    enc_kwargs = {
        "x": alt.X(f"{gb1}:N", sort="-y"),
        "y": alt.Y("value:Q"),
        "tooltip": tooltip_cols,
    }
    if use_second:
        enc_kwargs["color"] = alt.Color(f"{gb2}:N")

    chart = chart.encode(**enc_kwargs).properties(title=f"{agg_mode} of selected metric")

    st.altair_chart(chart, use_container_width=True)
    st.caption("Aggregated data used for the chart (including counts and percentages):")
    st.dataframe(agg_df)


def relation_builder_and_explorer(
    df_lc,
    lc_results,
    df_stock,
    stock_results,
):
    """
    Full control over RELATIONS (joins) + charts.

    - You choose left table, right table.
    - You choose key columns on each side.
    - You choose join type (inner/left/right/outer).
    - Joined dataset gets stored and is available for ad-hoc charts.
    """
    st.header("Relations & Ad-hoc Explorer")

    if "joined_datasets" not in st.session_state:
        st.session_state["joined_datasets"] = {}

    datasets: dict[str, pd.DataFrame] = {}

    # Base datasets
    if df_lc is not None:
        datasets["LC: Input file"] = df_lc
        if lc_results is not None:
            datasets["LC: Genric summary"] = lc_results["genric_summary"]
            if not lc_results["company_summary"].empty:
                datasets["LC: Company summary"] = lc_results["company_summary"]
            if not lc_results["location_summary"].empty:
                datasets["LC: Location summary"] = lc_results["location_summary"]
            if not lc_results["pkg_file_summary"].empty:
                datasets["LC: Package summary (file-level)"] = lc_results["pkg_file_summary"]
            if not lc_results["pkg_genric_summary"].empty:
                datasets["LC: Package summary (Genric-level)"] = lc_results["pkg_genric_summary"]
            datasets["LC: Input with Genric annotations"] = lc_results["df_with_genric"]

    if df_stock is not None:
        datasets["Stock: Input file"] = df_stock
        if stock_results is not None:
            datasets["Stock: Generic summary"] = stock_results["generic_summary"]
            datasets["Stock: Generic+Supplier summary"] = stock_results["generic_supplier_summary"]
            datasets["Stock: Supplier summary"] = stock_results["supplier_summary"]
            datasets["Stock: Input with flags"] = stock_results["df_with_flags"]

    # User-defined joined datasets
    for name, jdf in st.session_state["joined_datasets"].items():
        datasets[name] = jdf

    if not datasets:
        st.info("No datasets yet. Build LC/Stock summaries first in the other tabs.")
        return

    # ===== RELATION BUILDER (FULL CONTROL) =====
    with st.expander("Create / manage relations (joins)", expanded=True):
        ds_names = list(datasets.keys())
        left_name = st.selectbox("Left table", ds_names, key="join_left_table")
        right_name = st.selectbox("Right table", ds_names, key="join_right_table")

        if left_name == right_name:
            st.info("Choose two different tables to join.")
        else:
            left_cols = list(datasets[left_name].columns)
            right_cols = list(datasets[right_name].columns)

            left_keys = st.multiselect(
                "Left key column(s)",
                left_cols,
                default=[],
                key="join_left_keys",
            )
            right_keys = st.multiselect(
                "Right key column(s) (same length & order as left)",
                right_cols,
                default=[],
                key="join_right_keys",
            )

            join_type = st.selectbox(
                "Join type",
                ["inner", "left", "right", "outer"],
                key="join_type",
            )

            default_label = f"Join: {left_name} × {right_name}"
            join_label = st.text_input(
                "Name of joined dataset",
                value=default_label,
                key="join_label",
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Create / update joined dataset", key="join_button"):
                    if not left_keys or not right_keys:
                        st.error("Select at least one key column on both sides.")
                    elif len(left_keys) != len(right_keys):
                        st.error("Number of left and right key columns must match.")
                    else:
                        try:
                            join_df = datasets[left_name].merge(
                                datasets[right_name],
                                left_on=left_keys,
                                right_on=right_keys,
                                how=join_type,
                                suffixes=("_L", "_R"),
                            )
                            st.session_state["joined_datasets"][join_label] = join_df
                            datasets[join_label] = join_df
                            st.success(
                                f"Joined dataset '{join_label}' created "
                                f"({join_df.shape[0]} rows × {join_df.shape[1]} columns)."
                            )
                        except Exception as e:
                            st.error(f"Join failed: {e}")

            with c2:
                if st.button("Clear all joined datasets", key="clear_joins"):
                    st.session_state["joined_datasets"] = {}
                    st.success("All joined datasets cleared (you can recreate them).")

    # ===== AD-HOC CHART EXPLORER =====
    st.markdown("---")
    st.subheader("Ad-hoc charts on any dataset (including your relations)")
    ds_name = st.selectbox("Dataset for chart", list(datasets.keys()), key="explorer_ds")
    draw_adhoc_chart(datasets[ds_name], ds_name)
