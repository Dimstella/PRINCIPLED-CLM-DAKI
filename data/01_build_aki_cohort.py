"""
01_build_aki_cohort.py
----------------------
Build the **AKI cohort** for a given suspect or negative-control drug.

For every drug we need to:
  1. Load raw MIMIC-IV tables (KDIGO, chemistry, vitals/OMAR, Charlson,
     medications, patients).
  2. Apply the KDIGO-based AKI-worsening filter to identify patients who
     developed AKI after the drug was initiated (> 1 day) or within 7 days
     of stopping it.
  3. Clean and aggregate each feature category:
       - Charlson comorbidities (binary flags + CCI score)
       - Laboratory tests (mean during treatment; first/last during treatment)
       - Vital signs / OMAR (mean weight, BMI, systolic BP, diastolic BP)
       - Concomitant medications (duration in days, ATC-code pivot table)
  4. Intersect subject IDs across all feature tables and produce a single
     merged parquet file: ``<drug>_aki.parquet.gzip``

Usage
-----
    python 01_build_aki_cohort.py --drug vancomycin --work-dir /path/to/data
    python 01_build_aki_cohort.py --drug all        --work-dir /path/to/data

The ``--drug all`` flag runs the pipeline for every drug in ALL_DRUGS.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd

from atc_mapping import apply_name_unification, build_atc_lookup, map_drugs_to_atc, apply_atc_from_cache
from config import (
    ALL_DRUGS,
    ATC_SIMILARITY_THRESHOLD,
    CHARLSON_COMORBIDITY_COLS,
    CHARLSON_SELECT_COLS,
    CHEM_DROP_COLS,
    CONCOMITANT_ATC_CODES,
    LAB_COLS,
    MED_DROP_COLS,
    OMAR_RESULT_NAMES,
    PATIENTS_DROP_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# I/O helpers
# ===========================================================================

def _read(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV (plain / zipped) file."""
    if not path.exists():
        raise FileNotFoundError(f"Expected input file not found: {path}")
    if path.suffix in (".parquet", ".gzip"):
        return pd.read_parquet(path)
    if path.name.endswith(".csv.zip"):
        return pd.read_csv(path, compression="zip")
    return pd.read_csv(path)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="gzip")
    logger.info("  saved  %s  (%d rows, %d cols)", path.name, len(df), df.shape[1])


def _resolve_input(work_dir: Path, candidates: list[str]) -> Path:
    """Return the first candidate path that exists, raise if none do."""
    for name in candidates:
        p = work_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of these input files found in {work_dir}:\n  " + "\n  ".join(candidates)
    )


# ===========================================================================
# Step 1 – KDIGO: identify AKI-worsening patients
# ===========================================================================

def process_kdigo(work_dir: Path, drug: str, cfg: dict) -> tuple[pd.DataFrame, list, list]:
    """
    Load the KDIGO table, apply the AKI-worsening time filter, and return
    (kdigo_filtered, subject_ids, hadm_ids).

    The three valid AKI-worsening categories (defined per drug in config.py):
      - 'AKI worsening DURING treatment (after 1 day)' : chartdatetime <= stoptime
      - 'AKI worsening WITHIN 7 FROM STOP'             : chartdatetime <= stoptime + 7 days
      - '<Drug> Ongoing AKI worsening'                 : chartdatetime <= stoptime
    """
    logger.info("[KDIGO] loading for drug=%s", drug)
    slug = cfg["slug"]
    kdigo_path = _resolve_input(
        work_dir,
        [
            f"kdigo_{slug}_AKI.parquet.gzip",
            f"kdigo_{slug}.parquet.gzip",
            f"kdigo_{slug}.csv",
        ],
    )
    kdigo = _read(kdigo_path)

    # Drop raw creatinine / urine-output sub-columns (keep aki_flag + stage)
    drop_candidates = [
        "creat_low_past_7day", "creat_low_past_48hr", "creat",
        "aki_stage_creat", "uo_rt_6hr", "uo_rt_12hr", "uo_rt_24hr",
        "aki_stage_uo", "aki_stage_crrt",
        "aki_flag1", "aki_flag2", "aki_flag3", "aki_flag4",
    ]
    kdigo.drop(
        columns=[c for c in drop_candidates if c in kdigo.columns],
        inplace=True,
    )

    stoptime_col = cfg["stoptime_col"]
    aki_flags = cfg["aki_flags"]

    kdigo[stoptime_col] = pd.to_datetime(kdigo[stoptime_col])
    kdigo["chartdatetime"] = pd.to_datetime(kdigo["chartdatetime"])

    flag_during, flag_7days, flag_ongoing = aki_flags

    mask = (
        ((kdigo["aki_flag"] == flag_7days)
         & (kdigo["chartdatetime"] <= kdigo[stoptime_col] + datetime.timedelta(days=7)))
        | ((kdigo["aki_flag"] == flag_during)
           & (kdigo["chartdatetime"] <= kdigo[stoptime_col]))
        | ((kdigo["aki_flag"] == flag_ongoing)
           & (kdigo["chartdatetime"] <= kdigo[stoptime_col]))
    )
    kdigo = kdigo[mask].copy()

    subject_ids = kdigo["subject_id"].unique().tolist()
    hadm_ids = kdigo["hadm_id"].unique().tolist()
    logger.info("  %d patients after KDIGO filter", len(subject_ids))
    return kdigo, subject_ids, hadm_ids


# ===========================================================================
# Step 2 – Charlson comorbidities
# ===========================================================================

def process_charlson(
    work_dir: Path, drug: str, cfg: dict,
    subject_ids: list, hadm_ids: list,
) -> pd.DataFrame:
    """
    Load the Charlson table, filter to our cohort, binarise comorbidities,
    and compute mean CCI per patient.

    Returns one row per subject_id.
    """
    logger.info("[Charlson] loading for drug=%s", drug)
    slug = cfg["slug"]
    charlson_path = _resolve_input(
        work_dir,
        [
            f"charlson_{slug}_AKI.parquet.gzip",
            f"charlson_{slug}.parquet.gzip",
            f"charlson_{slug}.csv",
        ],
    )
    charlson = _read(charlson_path)

    # Restrict to cohort
    charlson = charlson[
        charlson["subject_id"].isin(subject_ids)
        & charlson["hadm_id"].isin(hadm_ids)
    ]

    # Select relevant columns (gracefully skip missing ones)
    cols = [c for c in CHARLSON_SELECT_COLS if c in charlson.columns]
    ch = charlson[cols].drop_duplicates()

    comorbidity_cols = [c for c in CHARLSON_COMORBIDITY_COLS if c in ch.columns]
    agg_cols = [c for c in comorbidity_cols if c != "charlson_comorbidity_index"]
    cci_in_cols = "charlson_comorbidity_index" in ch.columns

    # Aggregate: sum per patient, then binarise
    group_cols = agg_cols + (["charlson_comorbidity_index"] if cci_in_cols else [])
    cls = (
        ch.groupby("subject_id")[group_cols]
        .sum()
        .reset_index()
    )

    # Merge admission count for mean CCI calculation
    counts = (
        ch.groupby("subject_id")["subject_id"]
        .count()
        .rename("_count")
        .reset_index()
    )
    cls = cls.merge(counts, on="subject_id")
    if cci_in_cols:
        cls["charlson_comorbidity_index"] = (
            cls["charlson_comorbidity_index"] / cls["_count"]
        ).round()
    cls.drop(columns=["_count"], inplace=True)

    # Binarise comorbidity flag columns
    cls[agg_cols] = cls[agg_cols].clip(upper=1)

    # Add age_score from the (already-grouped) charlson if available
    if "age_score" in ch.columns:
        age_df = ch.groupby("subject_id")["age_score"].first().reset_index()
        cls = cls.merge(age_df, on="subject_id", how="left")

    logger.info("  %d patients in Charlson", cls["subject_id"].nunique())
    return cls


# ===========================================================================
# Step 3 – Chemistry (laboratory tests)
# ===========================================================================

def process_chemistry(
    work_dir: Path, drug: str, cfg: dict,
    subject_ids: list, hadm_ids: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce two chemistry DataFrames:
      - cm_mean  : one row per patient, mean of each lab during treatment
      - cm_t0t1  : one row per patient, first and last values during treatment

    Returns (cm_mean, cm_t0t1).
    """
    logger.info("[Chemistry] loading for drug=%s", drug)
    slug = cfg["slug"]
    flag_col = cfg["flag_col"]
    drug_label = drug.capitalize()

    chem_path = _resolve_input(
        work_dir,
        [
            f"chemistry_t0_{slug}.csv",
            f"chem_{slug}_AKI.parquet.gzip",
            f"chemistry_{slug}.csv",
        ],
    )
    chem = _read(chem_path)

    # Drop unused columns
    chem.drop(
        columns=[c for c in CHEM_DROP_COLS if c in chem.columns],
        inplace=True,
        errors="ignore",
    )

    # Filter to cohort
    id_mask = chem["subject_id"].isin(subject_ids)
    if "hadm_id" in chem.columns:
        id_mask &= chem["hadm_id"].isin(hadm_ids)
    chem = chem[id_mask].copy()

    # Extract the DURING-treatment window
    during_label = f"DURING {drug_label}"
    if flag_col in chem.columns:
        cm_during = chem[chem[flag_col] == during_label].copy()
        cm_during.drop(columns=[flag_col], inplace=True, errors="ignore")
    else:
        # Fallback: assume the entire table is already DURING
        cm_during = chem.copy()

    available_lab_cols = [c for c in LAB_COLS if c in cm_during.columns]

    # ---- Mean representation ----
    cm_mean = (
        cm_during.dropna(subset=["glucose"])
        .groupby("subject_id")[available_lab_cols]
        .mean()
        .reset_index()
    )

    # ---- First / last (t0-t1) representation ----
    if "charttime" in cm_during.columns:
        cm_sorted = cm_during.sort_values(["subject_id", "charttime"])
    elif "chartdatetime" in cm_during.columns:
        cm_sorted = cm_during.sort_values(["subject_id", "chartdatetime"])
    else:
        cm_sorted = cm_during.copy()

    agg_dict = {}
    for col in available_lab_cols:
        agg_dict[f"{col}_first"] = (col, "first")
        agg_dict[f"{col}_last"] = (col, "last")

    cm_t0t1 = cm_sorted.groupby("subject_id").agg(**agg_dict).reset_index()
    cm_t0t1 = cm_t0t1.dropna(subset=["glucose_first"])

    logger.info(
        "  cm_mean: %d patients | cm_t0t1: %d patients",
        cm_mean["subject_id"].nunique(),
        cm_t0t1["subject_id"].nunique(),
    )
    return cm_mean, cm_t0t1


# ===========================================================================
# Step 4 – Vital signs (OMAR)
# ===========================================================================

def process_omar(
    work_dir: Path, drug: str, cfg: dict,
    subject_ids: list, kdigo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the OMAR (vital signs) table, apply the same time filter used for
    KDIGO, pivot BMI/Weight and Blood Pressure into numeric columns, and
    return a wide DataFrame (one row per patient).
    """
    logger.info("[OMAR] loading for drug=%s", drug)
    slug = cfg["slug"]
    flag_col = cfg["flag_col"]
    stoptime_col = cfg["stoptime_col"]
    drug_label = drug.capitalize()
    aki_flags = cfg["aki_flags"]
    flag_during_label, flag_7days_label, flag_ongoing_label = aki_flags

    omar_path = _resolve_input(
        work_dir,
        [
            f"omar_{slug}_AKI.parquet.gzip",
            f"omar_{slug}.parquet.gzip",
            f"omar_{slug}.csv",
        ],
    )
    omar = _read(omar_path)

    # Restrict to KDIGO-filtered subjects
    omar = omar[omar["subject_id"].isin(subject_ids)].copy()

    # Merge aki_flag from kdigo
    df_index = kdigo[["subject_id", "aki_flag"]].drop_duplicates()
    omar = omar.merge(df_index, on="subject_id", how="left")

    # Apply the same time filter as KDIGO
    omar[stoptime_col] = pd.to_datetime(omar[stoptime_col])
    omar["chartdatetime"] = pd.to_datetime(omar["chartdatetime"])

    mask = (
        ((omar["aki_flag"] == flag_7days_label)
         & (omar["chartdatetime"] <= omar[stoptime_col] + datetime.timedelta(days=7)))
        | ((omar["aki_flag"] == flag_during_label)
           & (omar["chartdatetime"] <= omar[stoptime_col]))
        | ((omar["aki_flag"] == flag_ongoing_label)
           & (omar["chartdatetime"] <= omar[stoptime_col]))
    )
    omar = omar[mask].copy()

    # Keep only relevant result types
    omar = omar[omar["result_name"].isin(OMAR_RESULT_NAMES)]

    # Filter to DURING window
    during_label = f"DURING {drug_label}"
    if flag_col in omar.columns:
        omar_during = omar[omar[flag_col] == during_label].copy()
        omar_during.drop(columns=[flag_col], inplace=True, errors="ignore")
    else:
        omar_during = omar.copy()

    omar_during = omar_during[["subject_id", "result_name", "result_value"]].copy()

    # -- BMI / Weight --
    omar_bmi = omar_during[
        omar_during["result_name"].isin(["BMI (kg/m2)", "Weight (Lbs)"])
    ].copy()
    omar_bmi["result_value"] = pd.to_numeric(omar_bmi["result_value"], errors="coerce")
    omar_bmi_wide = pd.pivot_table(
        omar_bmi,
        index="subject_id",
        columns="result_name",
        values="result_value",
        aggfunc="mean",
        fill_value=0,
    ).round()

    # -- Blood Pressure --
    omar_bp = omar_during[omar_during["result_name"] == "Blood Pressure"].copy()
    omar_bp[["Small_bp", "Big_bp"]] = (
        omar_bp["result_value"].str.split("/", expand=True)
    )
    for col in ["Small_bp", "Big_bp"]:
        omar_bp[col] = pd.to_numeric(omar_bp[col], errors="coerce")
    omar_bp_wide = (
        omar_bp.groupby("subject_id")[["Big_bp", "Small_bp"]]
        .mean()
        .round()
        .reset_index()
    )

    omar_out = omar_bmi_wide.merge(omar_bp_wide, on="subject_id", how="outer")
    logger.info("  %d patients in OMAR", omar_out["subject_id"].nunique())
    return omar_out


# ===========================================================================
# Step 5 – Medications → ATC pivot table
# ===========================================================================

def process_medications(
    work_dir: Path, drug: str, cfg: dict,
    subject_ids: list, hadm_ids: list,
    atc_csv_path: Path,
    mapping_cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load the medication table, unify drug names, map to ATC codes, and
    build a subject × ATC pivot table with each cell = total days of
    exposure.  Only ATC codes/prefixes listed in CONCOMITANT_ATC_CODES
    are kept (matching the paper's approach).

    Parameters
    ----------
    mapping_cache_path : Path or None
        If a validated Excel mapping file already exists at this path it
        is used directly (no fuzzy matching).  Otherwise fuzzy matching
        runs and the result is saved there for manual validation.
    """
    logger.info("[Medications] loading for drug=%s", drug)
    slug = cfg["slug"]
    flag_col = cfg["flag_col"]
    drug_label = drug.capitalize()

    med_path = _resolve_input(
        work_dir,
        [
            f"medication_{slug}.parquet.gzip",
            f"medication_{slug}.csv.zip",
            f"medication_{slug}_AKI.parquet.gzip",
        ],
    )
    med = _read(med_path)

    # Drop columns that are not needed
    med.drop(
        columns=[c for c in MED_DROP_COLS if c in med.columns],
        inplace=True,
        errors="ignore",
    )

    # Restrict to cohort
    id_mask = med["subject_id"].isin(subject_ids)
    if "hadm_id" in med.columns:
        id_mask &= med["hadm_id"].isin(hadm_ids)
    med = med[id_mask].copy()

    # Restrict to the DURING window
    during_label = f"DURING {drug_label}"
    if flag_col in med.columns:
        med = med[med[flag_col] == during_label].copy()

    # Unify drug names
    med = apply_name_unification(med)

    # Map to ATC codes
    if mapping_cache_path is not None and mapping_cache_path.exists():
        logger.info("  using cached ATC mapping: %s", mapping_cache_path)
        med = apply_atc_from_cache(med, mapping_cache_path)
    else:
        atc_lookup = build_atc_lookup(atc_csv_path)
        med = map_drugs_to_atc(
            med, atc_lookup,
            threshold=ATC_SIMILARITY_THRESHOLD,
            mapping_cache_path=mapping_cache_path,
        )

    # Remove N/A codes
    med = med[med["ATC"] != "N/A"].copy()

    # Calculate drug duration
    if "drug_stoptime" in med.columns and "drug_starttime" in med.columns:
        med["drug_duration_days"] = (
            pd.to_datetime(med["drug_stoptime"]) - pd.to_datetime(med["drug_starttime"])
        ).dt.days
    elif "drug_duration_days" not in med.columns:
        med["drug_duration_days"] = 1  # fallback

    # Remove rows without a valid dose
    if "dose_val_rx" in med.columns:
        med = med[~med["dose_val_rx"].isnull()]

    # Explode list-valued ATC cells (can happen if ATC is a list)
    med = med.explode("ATC")
    med = med[med["ATC"] != "N/A"].copy()

    # Pivot: subject × ATC = total days
    med_atc = med[["subject_id", "ATC", "drug_duration_days"]]
    med_pivot = pd.pivot_table(
        med_atc,
        index="subject_id",
        columns="ATC",
        values="drug_duration_days",
        aggfunc="sum",
        fill_value=0,
    )

    # Keep only literature-relevant ATC codes
    col_ls = []
    for atc_prefix in CONCOMITANT_ATC_CODES:
        for col in med_pivot.columns:
            if col.startswith(atc_prefix) or col == atc_prefix:
                col_ls.append(col)

    if col_ls:
        med_pivot = med_pivot[col_ls]
        med_pivot = med_pivot.loc[:, ~med_pivot.columns.duplicated()].copy()

    med_out = med_pivot.reset_index()
    logger.info(
        "  %d patients, %d ATC features",
        med_out["subject_id"].nunique(),
        med_out.shape[1] - 1,
    )
    return med_out


# ===========================================================================
# Step 6 – Patients demographic table
# ===========================================================================

def process_patients(
    work_dir: Path, drug: str, cfg: dict,
    subject_ids: list,
) -> pd.DataFrame:
    slug = cfg["slug"]
    patients_path = _resolve_input(
        work_dir,
        [
            f"patients_{slug}.csv",
            f"patients_{slug}.parquet.gzip",
            "patients.csv",
        ],
    )
    patients = _read(patients_path)
    patients = patients[patients["subject_id"].isin(subject_ids)].copy()
    patients.drop(
        columns=[c for c in PATIENTS_DROP_COLS if c in patients.columns],
        inplace=True,
        errors="ignore",
    )
    logger.info("  %d patients in patients table", patients["subject_id"].nunique())
    return patients


# ===========================================================================
# Step 7 – Final merge and intersection
# ===========================================================================

def build_final_dataset(
    omar_during: pd.DataFrame,
    cm_mean: pd.DataFrame,
    cls: pd.DataFrame,
    med_atc: pd.DataFrame,
    patients: pd.DataFrame,
    use_mean_labs: bool = True,
) -> pd.DataFrame:
    """
    Intersect all tables on subject_id, merge, and return the final dataset.

    Parameters
    ----------
    use_mean_labs : bool
        If True use cm_mean (mean lab values); if False use cm_t0t1
        (first/last values).  Callers should call this function twice
        to produce both representations.
    """
    cm = cm_mean  # caller decides which representation to pass

    # Intersect subject IDs across all feature tables
    sid = set(omar_during["subject_id"].unique())
    for df in [cm, cls, med_atc, patients]:
        sid &= set(df["subject_id"].unique())
    sid = list(sid)
    logger.info("  intersection: %d patients with all features present", len(sid))

    # Filter each table
    dfs = [omar_during, cm, cls, med_atc, patients]
    dfs = [d[d["subject_id"].isin(sid)] for d in dfs]

    final = dfs[0]
    for d in dfs[1:]:
        # Avoid duplicate columns (keep the left copy)
        overlap = [c for c in d.columns if c in final.columns and c != "subject_id"]
        d = d.drop(columns=overlap)
        final = final.merge(d, on="subject_id", how="inner")

    return final


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_aki_pipeline(
    drug: str,
    work_dir: Path,
    atc_csv_path: Path,
    mapping_cache_path: Path | None = None,
) -> None:
    """
    Full AKI-cohort ETL for a single drug.
    Produces:
      <work_dir>/<drug>_aki_mean.parquet.gzip   (mean lab representation)
      <work_dir>/<drug>_aki_t0t1.parquet.gzip   (first/last lab representation)
    """
    if drug not in ALL_DRUGS:
        raise ValueError(f"Unknown drug '{drug}'. Choices: {list(ALL_DRUGS)}")

    cfg = ALL_DRUGS[drug]
    logger.info("=" * 60)
    logger.info("AKI pipeline  |  drug = %s", drug)
    logger.info("=" * 60)

    # 1. KDIGO
    kdigo, subject_ids, hadm_ids = process_kdigo(work_dir, drug, cfg)

    # 2. Charlson
    cls = process_charlson(work_dir, drug, cfg, subject_ids, hadm_ids)

    # 3. Chemistry
    cm_mean, cm_t0t1 = process_chemistry(work_dir, drug, cfg, subject_ids, hadm_ids)

    # 4. OMAR
    omar_during = process_omar(work_dir, drug, cfg, subject_ids, kdigo)

    # 5. Medications
    med_atc = process_medications(
        work_dir, drug, cfg, subject_ids, hadm_ids,
        atc_csv_path=atc_csv_path,
        mapping_cache_path=mapping_cache_path,
    )

    # 6. Patients
    patients = process_patients(work_dir, drug, cfg, subject_ids)

    # 7. Final dataset – mean labs
    final_mean = build_final_dataset(omar_during, cm_mean, cls, med_atc, patients)
    _write(final_mean, work_dir / f"{drug}_aki_mean.parquet.gzip")

    # 7b. Final dataset – t0-t1 labs
    final_t0t1 = build_final_dataset(omar_during, cm_t0t1, cls, med_atc, patients)
    _write(final_t0t1, work_dir / f"{drug}_aki_t0t1.parquet.gzip")

    logger.info("Done: %s  (%d / %d patients)", drug, len(final_mean), len(final_t0t1))


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the AKI cohort parquet files for one or all drugs."
    )
    parser.add_argument(
        "--drug",
        required=True,
        help="Drug name (e.g. 'vancomycin') or 'all' to run every drug.",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        type=Path,
        help="Directory containing raw MIMIC-IV parquet/CSV files.",
    )
    parser.add_argument(
        "--atc-csv",
        default=None,
        type=Path,
        help="Path to atc_codes.csv.  Defaults to <work-dir>/atc_codes.csv.",
    )
    parser.add_argument(
        "--mapping-cache",
        default=None,
        type=Path,
        help=(
            "Path to a validated Drug_Name->ATC_Code Excel file. "
            "If it exists it is used directly; otherwise fuzzy matching runs "
            "and saves its output here for review."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    work_dir: Path = args.work_dir.resolve()
    atc_csv: Path = (
        args.atc_csv.resolve()
        if args.atc_csv
        else work_dir / "atc_codes.csv"
    )

    drugs = list(ALL_DRUGS.keys()) if args.drug == "all" else [args.drug]

    for d in drugs:
        mapping_cache = (
            args.mapping_cache
            if args.mapping_cache
            else work_dir / f"atc_med_map_85_{d}_AKI.xlsx"
        )
        run_aki_pipeline(
            drug=d,
            work_dir=work_dir,
            atc_csv_path=atc_csv,
            mapping_cache_path=mapping_cache,
        )


if __name__ == "__main__":
    main()
