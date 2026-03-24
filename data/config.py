"""
config.py
---------
Central configuration for the DAKI preprocessing pipeline.

All drug names, their MIMIC-IV file-name slugs, AKI-flag column names, and
the reference list of concomitant-drug ATC codes from the literature are
defined here so that every other module stays drug-agnostic.
"""

# ---------------------------------------------------------------------------
# Drug registry
# ---------------------------------------------------------------------------
# Each entry maps a canonical drug name to:
#   slug          - identifier used in raw MIMIC-IV CSV/parquet file names
#   flag_col      - name of the <drug>_flag column that marks BEFORE/DURING/AFTER
#   stoptime_col  - name of the stoptime column in KDIGO / chemistry / omar tables
#   aki_flags     - the three AKI-worsening categories produced by the KDIGO query
# ---------------------------------------------------------------------------

POSITIVE_DRUGS = {
    "ibuprofen": {
        "slug": "ibuprofen",
        "flag_col": "ibuprofen_flag",
        "stoptime_col": "stoptime_ibuprofen",
        "aki_flags": [
            "AKI worsening WITHIN 7 FROM STOP",
            "AKI worsening DURING treatment (after 1 day)",
            "Ibuprofen Ongoing AKI worsening",
        ],
    },
    "ketorolac": {
        "slug": "ketorolac",
        "flag_col": "ketorolac_flag",
        "stoptime_col": "stoptime_ketorolac",
        "aki_flags": [
            "AKI worsening WITHIN 7 FROM STOP",
            "AKI worsening DURING treatment (after 1 day)",
            "Ketorolac Ongoing AKI worsening",
        ],
    },
    "vancomycin": {
        "slug": "vancomycin",
        "flag_col": "vancomycin_flag",
        "stoptime_col": "stoptime_vancomycin",
        "aki_flags": [
            "AKI worsening WITHIN 7 FROM STOP",
            "AKI worsening DURING treatment (after 1 day)",
            "Vancomycin Ongoing AKI worsening",
        ],
    },
    "lisinopril": {
        "slug": "lisinopril",
        "flag_col": "lisinopril_flag",
        "stoptime_col": "stoptime_lisinopril",
        "aki_flags": [
            "AKI worsening WITHIN 7 FROM STOP",
            "AKI worsening DURING treatment (after 1 day)",
            "Lisinopril Ongoing AKI worsening",
        ],
    },
    "furosemide": {
        "slug": "furosemide",
        "flag_col": "furosemide_flag",
        "stoptime_col": "stoptime_furosemide",
        "aki_flags": [
            "AKI worsening WITHIN 7 FROM STOP",
            "AKI worsening DURING treatment (after 1 day)",
            "Furosemide Ongoing AKI worsening",
        ],
    },
    "pantoprazole": {
        "slug": "pantoprazole",
        "flag_col": "pantoprazole_flag",
        "stoptime_col": "stoptime_pantoprazole",
        "aki_flags": [
            "AKI worsening WITHIN 7 FROM STOP",
            "AKI worsening DURING treatment (after 1 day)",
            "Pantoprazole Ongoing AKI worsening",
        ],
    },
    "omeprazole": {
        "slug": "omeprazole",
        "flag_col": "omeprazole_flag",
        "stoptime_col": "stoptime_omeprazole",
        "aki_flags": [
            "AKI worsening WITHIN 7 FROM STOP",
            "AKI worsening DURING treatment (after 1 day)",
            "Omeprazole Ongoing AKI worsening",
        ],
    },
    "allopurinol": {
        "slug": "allopurinol",
        "flag_col": "allopurinol_flag",
        "stoptime_col": "stoptime_allopurinol",
        "aki_flags": [
            "AKI worsening WITHIN 7 FROM STOP",
            "AKI worsening DURING treatment (after 1 day)",
            "Allopurinol Ongoing AKI worsening",
        ],
    },
}

NEGATIVE_DRUGS = {
    "simethicone": {
        "slug": "simethicone",
        "flag_col": "simethicone_flag",
        "stoptime_col": "stoptime_simethicone",
    },
    "prochlorperazine": {
        "slug": "prochlorperazine",
        "flag_col": "prochlorperazine_flag",
        "stoptime_col": "stoptime_prochlorperazine",
    },
    "lactulose": {
        "slug": "lactulose",
        "flag_col": "lactulose_flag",
        "stoptime_col": "stoptime_lactulose",
    },
}

ALL_DRUGS = {**POSITIVE_DRUGS, **NEGATIVE_DRUGS}

# ---------------------------------------------------------------------------
# Chemistry columns
# ---------------------------------------------------------------------------
CHEM_DROP_COLS = ["albumin", "globulin", "total_protein", "calcium"]

LAB_COLS = [
    "aniongap",
    "bicarbonate",
    "bun",
    "chloride",
    "creatinine",
    "glucose",
    "sodium",
    "potassium",
]

# ---------------------------------------------------------------------------
# Charlson / demographics columns
# ---------------------------------------------------------------------------
CHARLSON_COMORBIDITY_COLS = [
    "myocardial_infarct",
    "congestive_heart_failure",
    "peripheral_vascular_disease",
    "cerebrovascular_disease",
    "dementia",
    "chronic_pulmonary_disease",
    "rheumatic_disease",
    "peptic_ulcer_disease",
    "mild_liver_disease",
    "diabetes_without_cc",
    "diabetes_with_cc",
    "paraplegia",
    "renal_disease",
    "malignant_cancer",
    "severe_liver_disease",
    "metastatic_solid_tumor",
    "aids",
    "charlson_comorbidity_index",
]

CHARLSON_SELECT_COLS = ["subject_id", "age_score"] + CHARLSON_COMORBIDITY_COLS

# ---------------------------------------------------------------------------
# Vital-signs / OMAR columns
# ---------------------------------------------------------------------------
OMAR_RESULT_NAMES = ["BMI (kg/m2)", "Blood Pressure", "Weight (Lbs)"]

# ---------------------------------------------------------------------------
# Columns to drop from the final patients table
# ---------------------------------------------------------------------------
PATIENTS_DROP_COLS = ["anchor_year_group", "anchor_year", "dod"]

# ---------------------------------------------------------------------------
# Medication columns to drop before saving
# ---------------------------------------------------------------------------
MED_DROP_COLS = [
    "drug_terminology",
    "medication_terminology",
    "Unnamed: 0.1",
    "Unnamed: 0",
    "ndc",
    "prod_strength",
    "drug",
    "medication",
    "formulary_drug_cd",
    "expiration_unit",
    "doses_per_24_hrs",
]

# ---------------------------------------------------------------------------
# ATC similarity threshold (fuzzywuzzy, %)
# ---------------------------------------------------------------------------
ATC_SIMILARITY_THRESHOLD = 85

# ---------------------------------------------------------------------------
# Concomitant drugs from the literature (ATC codes / prefixes)
# Source: Perazella & Rosner (2022); Rolland et al. (2021);
#         Fernandez-Llaneza et al. (2024)
# ---------------------------------------------------------------------------
CONCOMITANT_ATC_CODES = [
    "J01GB03", "A07AA01", "J01GB06", "J01XA01", "J01CR05", "A07AA10",
    "A01AB05", "J05AB05", "J05AF07", "J05AF08", "M01A",   "M01AH",
    "N02BE01", "L01XA01", "L01XX02", "L01BA04", "V08AB",  "L04AD01",
    "L04AD02", "M05BA03", "M05BX04", "M05BA08", "J01CE",  "J01DB",
    "J01DC",  "J01EE01", "J01EB04", "J01MA",  "J01FA",  "J04AB02",
    "A02BC",  "A02BA",   "M01AH",   "L01XC17", "L01XC18", "L01XC28",
    "L01XC32","L01XC23", "L01XC31", "L01XC11", "L01XC16", "L01XC07",
    "L01XE05","L01XE04", "C03CA01", "C03CA02", "C03AA03", "J05AB01",
    "J05AF06","J05AE02", "J05AE07", "J05AD01", "N03AA02", "N03AF01",
    "N03AB02","L01XX03", "L01BA04", "N05AN01", "M04AA01", "A07EC",
    "C03",    "C03CA01", "C03AA03", "C03DA01", "C09",     "C09AA05",
    "C09AA04","C09CA04", "C09CA03", "C09CA06", "J01",     "J01CA04",
    "J01GB03","J01XA01", "J01EE01", "J01GB06", "V08",     "V08AB11",
    "V08AB09","L01",     "L01XA02", "L01CB01", "L01AA01", "L01XA01",
    "L01BA01","M01",     "M01AE03", "M01AE13", "L04",     "L04AD02",
    "L04AD01","L04AX04", "J05",     "J05AB01", "J05AB11",
]
