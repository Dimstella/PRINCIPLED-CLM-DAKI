"""
matching/config.py
------------------
Central registry of every drug-specific confounder set used in the
matching benchmark (Dimitsaki et al., 2026).

For each positive drug we store:
  - drop_own_atc  : ATC column(s) of the drug itself that must be removed
                    from the dataset before matching (prevents data leakage)
  - extra_drop    : any extra raw timestamp columns present in some pair
                    datasets that are not features
  - dags          : a dict  { "Literature": [...], "Clinician": [...] }
                    mapping each DAG source to its list of ATC drug-family
                    names used as confounder families, plus the list of
                    extra numeric confounders beyond weight/gender/age/pre_AKI

All Charlson comorbidity columns are always included; they are appended
automatically by the pipeline and do not need to be listed here.
"""

# ---------------------------------------------------------------------------
# Feature column groups (shared across all drugs)
# ---------------------------------------------------------------------------

LAB_COLS = [
    "Big_bp", "Small_bp", "glucose", "sodium", "creatinine",
    "potassium", "bun", "bicarbonate", "chloride", "aniongap",
]

DEM_COLS = ["Weight (Lbs)", "gender", "anchor_age"]

CHARLSON_COLS = [
    "congestive_heart_failure", "renal_disease", "diabetes_without_cc",
    "chronic_pulmonary_disease", "myocardial_infarct", "diabetes_with_cc",
    "peripheral_vascular_disease", "cerebrovascular_disease",
    "malignant_cancer", "mild_liver_disease", "metastatic_solid_tumor",
    "rheumatic_disease", "peptic_ulcer_disease", "dementia",
    "paraplegia", "severe_liver_disease", "aids",
]

# Continuous columns that are z-score normalised in the preprocessor
SCALE_COLS = [
    "glucose", "Big_bp", "bicarbonate", "bun", "BMI (kg/m2)",
    "Weight (Lbs)", "chloride", "sodium", "potassium",
    "Small_bp", "creatinine", "aniongap",
]

# Columns that should NOT be binarised (all others with values > 1 → 1)
NO_BINARISE_COLS = SCALE_COLS + ["anchor_age"]

# Columns always dropped before feature construction
DROP_FROM_FEATURES = ["charlson_comorbidity_index", "BMI (kg/m2)"]

# Columns that identify the unit but are not features
ID_COLS = ["subject_id", "y", "t"]

# Legacy timestamp columns that appear in some datasets
TIMESTAMP_COLS = [
    "starttime_lisinopril", "stoptime_lisinopril",
    "starttime_kidney_disease", "stoptime_kidney_disease",
    "kidney_disease_flag",
]

# ---------------------------------------------------------------------------
# Per-drug configuration
# ---------------------------------------------------------------------------

DRUG_CONFIG = {
    # ------------------------------------------------------------------
    "ibuprofen": {
        # The drug's own ATC code column must be removed from the feature
        # matrix (it would be a perfect predictor of treatment).
        "drop_own_atc": [],          # ibuprofen does not appear as its own confounder column
        "extra_drop": TIMESTAMP_COLS,
        "dags": {
            "Literature": {
                "drug_families": [
                    "ANTIINFLAMMATORY AND ANTIRHEUMATIC PRODUCTS",
                    "ANTIHYPERTENSIVES",
                    "CONTRAST MEDIA",
                    "DRUGS USED IN DIABETES",
                    "DIURETICS",
                    "CORTICOSTEROIDS FOR SYSTEMIC USE",
                    "TOPICAL PRODUCTS FOR JOINT AND MUSCULAR PAIN",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": [
                    "ANTIINFLAMMATORY AND ANTIRHEUMATIC PRODUCTS",
                    "CONTRAST MEDIA",
                    "ANTIHYPERTENSIVES",
                    "DIURETICS",
                    "CORTICOSTEROIDS FOR SYSTEMIC USE",
                    "TOPICAL PRODUCTS FOR JOINT AND MUSCULAR PAIN",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    # ------------------------------------------------------------------
    "ketorolac": {
        "drop_own_atc": [],
        "extra_drop": TIMESTAMP_COLS,
        "dags": {
            "Literature": {
                "drug_families": [
                    "CONTRAST MEDIA",
                    "ANTIHYPERTENSIVES",
                    "DIURETICS",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "glucose", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": [
                    "CONTRAST MEDIA",
                    "ANTIHYPERTENSIVES",
                    "DIURETICS",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "sodium", "pre_AKI"],
            },
        },
    },
    # ------------------------------------------------------------------
    "vancomycin": {
        "drop_own_atc": [],
        "extra_drop": TIMESTAMP_COLS,
        # Literature and Clinician agree for vancomycin
        "dags": {
            "Literature": {
                "drug_families": [
                    "CONTRAST MEDIA",
                    "ANTIHYPERTENSIVES",
                    "DIURETICS",
                    "VACCINES",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["glucose", "Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": [
                    "CONTRAST MEDIA",
                    "ANTIHYPERTENSIVES",
                    "DIURETICS",
                    "VACCINES",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["glucose", "Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    # ------------------------------------------------------------------
    "lisinopril": {
        "drop_own_atc": [],
        "extra_drop": [],
        "dags": {
            "Literature": {
                "drug_families": [
                    "DIURETICS",
                    "CONTRAST MEDIA",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": [
                    "AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM",
                    "ANTIEPILEPTICS",
                    "DIURETICS",
                    "CONTRAST MEDIA",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    # ------------------------------------------------------------------
    "furosemide": {
        # Furosemide's own ATC code (C03CA01) is dropped to prevent leakage
        "drop_own_atc": ["C03CA01"],
        "extra_drop": [],
        # Literature and Clinician agree for furosemide
        "dags": {
            "Literature": {
                "drug_families": [
                    "DIURETICS",
                    "DRUGS FOR ACID RELATED DISORDERS",
                    "CONTRAST MEDIA",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": [
                    "DIURETICS",
                    "DRUGS FOR ACID RELATED DISORDERS",
                    "CONTRAST MEDIA",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    # ------------------------------------------------------------------
    "pantoprazole": {
        # Pantoprazole's own ATC code (A02BC02) is dropped
        "drop_own_atc": ["A02BC02"],
        "extra_drop": [],
        # Literature and Clinician agree for pantoprazole/omeprazole
        "dags": {
            "Literature": {
                "drug_families": [
                    "DIURETICS",
                    "CONTRAST MEDIA",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": [
                    "DIURETICS",
                    "CONTRAST MEDIA",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    # ------------------------------------------------------------------
    "omeprazole": {
        # Omeprazole's own ATC code (A02BC01) is dropped
        "drop_own_atc": ["A02BC01"],
        "extra_drop": [],
        "dags": {
            "Literature": {
                "drug_families": [
                    "DIURETICS",
                    "CONTRAST MEDIA",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": [
                    "DIURETICS",
                    "CONTRAST MEDIA",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
        },
    },
    # ------------------------------------------------------------------
    "allopurinol": {
        "drop_own_atc": [],
        "extra_drop": [],
        "dags": {
            "Literature": {
                "drug_families": [
                    "DIURETICS",
                    "AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                    "CONTRAST MEDIA",
                ],
                "extra_confounders": ["Weight (Lbs)", "gender", "anchor_age", "pre_AKI"],
            },
            "Clinician": {
                "drug_families": [
                    "AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM",
                    "DRUGS FOR ACID RELATED DISORDERS",
                    "IMMUNOSUPPRESSANTS",
                    "ANTIBACTERIALS FOR SYSTEMIC USE",
                    "PSYCHOLEPTICS",
                    "CONTRAST MEDIA",
                ],
                # Clinician adds full blood-panel labs as explicit confounders
                "extra_confounders": [
                    "Big_bp", "Small_bp", "sodium", "creatinine",
                    "potassium", "bun", "chloride", "bicarbonate",
                    "Weight (Lbs)", "gender", "anchor_age", "pre_AKI",
                ],
            },
        },
    },
}

# Negative control drugs (no DAG split needed – they define no confounders)
NEGATIVE_DRUGS = ["simethicone", "prochlorperazine", "lactulose"]

# All pair combinations the benchmark runs over
POSITIVE_DRUGS = list(DRUG_CONFIG.keys())
