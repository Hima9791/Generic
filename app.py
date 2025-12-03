import streamlit as st
import pandas as pd

from utils import guess_column, lc_counts
from lc_engine import build_lifecycle_summaries
from stock_engine import build_stock_summaries
from charts_explorer import (
    draw_lifecycle_stacked_chart,
    relation_builder_and_explorer,
)

from charts_explorer import (
    draw_lifecycle_stacked_chart,
    relation_builder_and_explorer,
)


st.set_page_config(
    page_title="Generic LC & Stock/LT/Price Dashboard",
    layout="wide",
)


def main():
    st.title("Generic Lifecycle & Stock / Lead-Time / Price Dashboard")

    st.caption(
        "You now have full control: separate engines, free choice of relations, "
        "and ad-hoc charts with counts and percentages on any dataset."
    )

    with st.sidebar:
        st.header("Upload input files")
        lc_file = st.file_uploader(
            "Lifecycle / Generic file (e.g. G_A.xlsx)",
            type=["xls", "xlsx"],
            key="lc_file",
        )
        stock_file = st.file_uploader(
            "Stock / LT / Price file (e.g. Price_G.xlsx)",
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
            "Lifecycle / LC",
            "Stock / LT / Price",
            "Relations & Explorer",
            "Help",
        ]
    )

    lc_results = None
    stock_results = None

    # -------- Lifecycle tab --------
    with tabs[0]:
        st.header("Lifecycle / LC Analysis")

        if df_lc is None:
            st.info("Upload the lifecycle/generic file in the sidebar.")
        else:
            st.write(f"**Lifecycle file shape:** {df_lc.shape[0]} rows × {df_lc.shape[1]} columns")
            with st.expander("Preview lifecycle input (first 200 rows)", expanded=False):
                st.dataframe(df_lc.head(200))

            st.subheader("Column mapping (Script 1 roles)")

            cols_list = list(df_lc.columns)
            none_option = "<None>"

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

            c1, c2, c3 = st.columns(3)
            c4, c5, c6 = st.columns(3)
            c7, c8 = st.columns(2)

            generic_guess = guess_column(df_lc, GENERIC_COL_CANDIDATES) or cols_list[0]
            generic_col = c1.selectbox(
                "Primary key (Genric / Family / etc.)",
                cols_list,
                index=cols_list.index(generic_guess),
            )

            partid_guess = guess_column(df_lc, PARTID_COL_CANDIDATES)
            partid_col = c2.selectbox(
                "PartID column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(partid_guess)) if partid_guess else 0,
            )

            partnum_guess = guess_column(df_lc, PARTNUM_COL_CANDIDATES)
            partnum_col = c3.selectbox(
                "PartNumber column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(partnum_guess)) if partnum_guess else 0,
            )

            company_guess = guess_column(df_lc, COMPANY_COL_CANDIDATES)
            company_col = c4.selectbox(
                "Company / Manufacturer column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(company_guess)) if company_guess else 0,
            )

            partlc_guess = guess_column(df_lc, PARTLC_COL_CANDIDATES)
            partlc_col = c5.selectbox(
                "Lifecycle column (Active / Obsolete / Unknown)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(partlc_guess)) if partlc_guess else 0,
            )

            partintro_guess = guess_column(df_lc, PARTINTRO_COL_CANDIDATES)
            partintro_col = c6.selectbox(
                "Part Intro Year column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(partintro_guess)) if partintro_guess else 0,
            )

            familyintro_guess = guess_column(df_lc, FAMILYINTRO_COL_CANDIDATES)
            familyintro_col = c7.selectbox(
                "Family Intro Year column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(familyintro_guess)) if familyintro_guess else 0,
            )

            stage_guess = guess_column(df_lc, STAGE_COL_CANDIDATES)
            stage_col = c8.selectbox(
                "Stage column (optional)",
                [none_option] + cols_list,
                index=(1 + cols_list.index(stage_guess)) if stage_guess else 0,
            )

            risk_col = st.selectbox(
                "Risk column (optional)",
                [none_option] + cols_list,
            )

            country_col = st.selectbox(
                "Country / Location column (optional, '|' separated)",
                [none_option] + cols_list,
            )

            pkg_norm_col = st.selectbox(
                "Normalized package column (optional)",
                [none_option] + cols_list,
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

            try:
                lc_results = build_lifecycle_summaries(df_lc, lc_cfg)
            except Exception as e:
                st.error(f"Error while computing lifecycle summaries: {e}")
                lc_results = None

            if lc_results is not None:
                genric_summary = lc_results["genric_summary"]

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
                    active, obsolete, unknown, _ = lc_counts(df_lc[lc_cfg["partlc"]])
                    m4.metric("Active parts", active)

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

                st.subheader("Summary tables")
                table_choice = st.selectbox(
                    "Table",
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

    # -------- Stock tab --------
    with tabs[1]:
        st.header("Stock / Lead-Time / Price Analysis")

        if df_stock is None:
            st.info("Upload the stock/price file in the sidebar.")
        else:
            st.write(f"**Stock file shape:** {df_stock.shape[0]} rows × {df_stock.shape[1]} columns")
            with st.expander("Preview stock/price input (first 200 rows)", expanded=False):
                st.dataframe(df_stock.head(200))

            st.subheader("Column mapping (Script 2 roles)")
            cols_list = list(df_stock.columns)

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

            c1, c2, c3, c4 = st.columns(4)
            c5, c6, c7, c8 = st.columns(4)

            genric_guess = guess_column(df_stock, GENRIC_COL_CANDIDATES) or cols_list[0]
            genric_col_s = c1.selectbox(
                "Generic / Key column",
                cols_list,
                index=cols_list.index(genric_guess),
            )

            supplier_guess = guess_column(df_stock, SUPPLIER_COL_CANDIDATES) or cols_list[1]
            supplier_col = c2.selectbox(
                "Supplier / Company column",
                cols_list,
                index=cols_list.index(supplier_guess),
            )

            qty_guess = guess_column(df_stock, QTY_COL_CANDIDATES) or cols_list[2]
            qty_col = c3.selectbox(
                "Quantity column",
                cols_list,
                index=cols_list.index(qty_guess),
            )

            hasstock_guess = guess_column(df_stock, HASSTOCK_COL_CANDIDATES) or cols_list[3]
            hasstock_col = c4.selectbox(
                "Has stock flag column (1/0 or True/False)",
                cols_list,
                index=cols_list.index(hasstock_guess),
            )

            minprice_guess = guess_column(df_stock, MINPRICE_COL_CANDIDATES) or cols_list[4]
            minprice_col = c5.selectbox(
                "Min price column",
                cols_list,
                index=cols_list.index(minprice_guess),
            )

            avgprice_guess = guess_column(df_stock, AVGPRICE_COL_CANDIDATES) or cols_list[5]
            avgprice_col = c6.selectbox(
                "Avg price column",
                cols_list,
                index=cols_list.index(avgprice_guess),
            )

            minlt_guess = guess_column(df_stock, MINLT_WEEK_COL_CANDIDATES) or cols_list[6]
            minlt_col = c7.selectbox(
                "Min lead time (weeks) column",
                cols_list,
                index=cols_list.index(minlt_guess),
            )

            maxlt_guess = guess_column(df_stock, MAXLT_WEEK_COL_CANDIDATES) or cols_list[7]
            maxlt_col = c8.selectbox(
                "Max lead time (weeks) column",
                cols_list,
                index=cols_list.index(maxlt_guess),
            )

            stock_cfg = {
                "genric": genric_col_s,
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
                supplier_summary = stock_results["supplier_summary"]
                meta_df = stock_results["meta"]
                df_with_flags = stock_results["df_with_flags"]

                st.subheader("High-level metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows", len(df_stock))
                m2.metric("Unique generics", df_stock[genric_col_s].nunique(dropna=True))
                m3.metric("Unique suppliers", df_stock[supplier_col].nunique(dropna=True))
                m4.metric("Total stock", float(df_with_flags[qty_col].fillna(0).sum()))

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
                    import altair as alt

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

                st.subheader("Summary tables")
                table_choice = st.selectbox(
                    "Table",
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
                    st.dataframe(stock_results["generic_supplier_summary"])
                elif table_choice == "Supplier summary":
                    st.dataframe(supplier_summary)
                elif table_choice == "Meta summary":
                    st.dataframe(meta_df)
                else:
                    st.dataframe(df_with_flags)

    # -------- Relations & Explorer tab --------
    with tabs[2]:
        relation_builder_and_explorer(
            df_lc=df_lc,
            lc_results=lc_results,
            df_stock=df_stock,
            stock_results=stock_results,
        )

    # -------- Help tab --------
    with tabs[3]:
        st.header("How to use")

        st.markdown(
            """
**1. Engines are now split into files**

- `lc_engine.py` → all lifecycle logic (Script 1).
- `stock_engine.py` → all stock / price / LT logic (Script 2).
- `utils.py` → shared helpers.
- `charts_explorer.py` → charts + relation builder.
- `app.py` → only UI wiring.

**2. Full control over relations**

- Go to **Relations & Explorer** tab.
- Choose **Left table** and **Right table** (any base table or summary).
- Pick **key column(s)** on each side (you can use multi-column join).
- Choose **join type** (inner/left/right/outer).
- Name the relation and click **Create/Update joined dataset**.
- That relation is stored and appears in the dataset list for ad-hoc charts.
- You can also **clear all joined datasets** and re-create them as you wish.

**3. Charts with counts and percentages**

- In the same tab, select the dataset (raw, summary, or joined).
- Use the Ad-hoc chart controls:
  - Group by 1 or 2 dimensions.
  - Choose aggregation: Count, Sum, Mean, Median, Count distinct.
  - See **value**, **% of total**, and **% within primary group** in the table and tooltips.
"""
        )


if __name__ == "__main__":
    main()
