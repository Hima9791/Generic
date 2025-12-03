import pandas as pd
import numpy as np

from utils import mode_or_na, concat_unique, lc_counts, classify_generic_lc


def build_lifecycle_summaries(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Lifecycle engine.

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

    # ----- Generic-level summary -----
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
            active, obsolete, unknown, _ = lc_counts(group[partlc_col])
        else:
            active = obsolete = unknown = 0

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
                active, obsolete, unknown, _ = lc_counts(group[partlc_col])
            else:
                active = obsolete = unknown = 0

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

    # ----- Location-level LC summary -----
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
                active, obsolete, unknown, _ = lc_counts(group[partlc_col])
            else:
                active = obsolete = unknown = 0

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

    # ----- Package-level LC summaries -----
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
                active, obsolete, unknown, _ = lc_counts(group[partlc_col])
            else:
                active = obsolete = unknown = 0

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
