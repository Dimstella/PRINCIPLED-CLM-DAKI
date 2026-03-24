# `analysis/`

Post-CML analysis pipeline for DAKI pharmacovigilance.

Covers the complete downstream workflow:
1. **CATE estimation** — ATE + bootstrap CI for every drug-pair × lab-repr × DAG × preprocessing scenario
2. **HTE estimation** — per-patient ITE for each drug using its single best-performing CML architecture
3. **Statistical tests** — Kruskal-Wallis tests, clinical binning, ITE summary tables, Plotly PDF boxplots

---

## Files

| File | Replaces | Purpose |
|---|---|---|
| `cate_estimation.py` | 8 per-drug CATE notebooks | Full ATE pipeline (48 datasets × 2 DAGs × 3 scenarios) |
| `hte_estimation.py` | ITE_clustering.ipynb §1 (load → learner → ITE → save) | Per-patient HTE for all 8 drugs |
| `statistical_tests.py` | ITE_clustering.ipynb §2 (interpretability + summarisation) | Kruskal-Wallis, clinical bins, PDF boxplots, Excel summaries |

---

## Prerequisites

```bash
pip install econml xgboost scikit-learn pandas numpy scipy plotly kaleido openpyxl
```

kaleido is required only for saving Plotly figures as PDF files.

---

## Recommended execution order

```
1.  data/preprocessing/01_build_aki_cohort.py
2.  data/preprocessing/02_build_noaki_cohort.py
3.  data/preprocessing/03_merge_datasets.py
4.  matching/run_benchmark.py              ← matching benchmark
5.  analysis/cate_estimation.py            ← ATE / CATE estimation
6.  analysis/hte_estimation.py             ← per-patient ITE
7.  analysis/statistical_tests.py         ← statistical analysis + plots
```

---

## cate_estimation.py

Produces one Excel workbook per positive drug, with one sheet per negative control.
Each sheet contains ATE, CI, and tagging columns for every combination of:

| Axis | Values |
|---|---|
| Lab representation | mean, t0t1 |
| DAG source | Literature, Clinician |
| Preprocessing scenario | Original, Binary, Normalised |
| Meta-learner | S-Learner, T-Learner, X-Learner, DML |

```bash
python cate_estimation.py \
    --work-dir /data/final_datasets \
    --out-dir  /results/cate
```

---

## hte_estimation.py

Uses one per-drug configuration (HTE_CONFIG dict) that encodes the best
(control, lab_repr, CML architecture, model) combination from the paper.

Outputs: `<drug>_<control>_ite.parquet.gzip` with:
- `ite` column: per-patient ITE estimate
- Raw (unscaled) lab and vital values restored for downstream interpretability

```bash
# All 8 drugs
python hte_estimation.py \
    --work-dir /data/final_datasets \
    --out-dir  /results/hte

# Single drug
python hte_estimation.py \
    --work-dir /data \
    --out-dir  /results/hte \
    --drug     vancomycin
```

### Best configuration per drug

| Drug | Control | Lab repr | Learner | Models |
|---|---|---|---|---|
| ibuprofen | prochlorperazine | mean | DML | SVR, SVR, SVR |
| ketorolac | lactulose | mean | X-Learner | SVC, SGDClf, LR, SVR, SVR |
| vancomycin | prochlorperazine | t0t1 | S-Learner | RandomForest |
| lisinopril | prochlorperazine | t0t1 | X-Learner | RF, SVC, GBC, RF, RF |
| furosemide | lactulose | t0t1 | X-Learner | DTC, SVC, SVC, RF, RF |
| pantoprazole | simethicone | t0t1 | S-Learner | RandomForest |
| omeprazole | prochlorperazine | mean | S-Learner | RandomForest |
| allopurinol | prochlorperazine | mean | S-Learner | RandomForest |

---

## statistical_tests.py

For each drug, loads the ITE parquet file and runs:

**1. ITE classification**
`ITE < -0.1` → Protective effect | `-0.1 ≤ ITE ≤ 0.1` → No effect | `ITE > 0.1` → Adverse effect

**2. Kruskal-Wallis tests** (≥3 classes) or Welch's t-test (2 classes) per feature:
- Vitals: Big_bp, Small_bp → 1×2 subplot → `<drug>_ite_vital.pdf`
- Demographics: Age, Weight → 1×2 subplot → `<drug>_ite_dem.pdf`
- Labs: glucose, sodium, creatinine, potassium, BUN, bicarbonate, chloride, aniongap → 2×4 subplot → `<drug>_ite_labs.pdf`

**3. Clinical binning** (ACC/AHA 2017 guidelines)

| Feature | Bins |
|---|---|
| Age | <18, 18–39, 40–59, 60–79, ≥80 |
| Weight | <60 kg, 60–79, 80–99, ≥100 |
| Glucose | <100, 100–125, ≥126 |
| Sodium | <135, 135–145, >145 |
| Creatinine | <1.2, 1.2–1.9, ≥2.0 |
| Potassium | <3.5, 3.5–5.0, >5.0 |
| BUN | <7, 7–20, >20 |
| Bicarbonate | <22, 22–29, >29 |
| Chloride | <98, 98–106, >106 |
| Anion gap | <8, 8–16, >16 |
| SBP | <120, 120–129, 130–139, ≥140 |
| DBP | <80, 80–89, ≥90 |

**4. ITE summary tables**: count, mean_ITE, std_ITE per category → `<drug>_ite_summary.xlsx`

```bash
python statistical_tests.py \
    --ite-dir /results/hte \
    --out-dir /results/stats
```

---

## Design decisions

**Raw lab value restoration in `hte_estimation.py`.** The learner receives z-score-scaled features (required for SVR, SVC convergence). After HTE prediction, the raw unscaled values are read back from the original parquet and written into the ITE dataframe. This is critical: clinical bin thresholds (e.g. glucose ≥126 = diabetes) are defined on the original clinical scale, not on normalised scores.

**`HTE_CONFIG` as a pure data dictionary.** In the notebook each drug had its own set of hard-coded model calls. The config dict encodes the exact same choices while making it trivial to change one drug's configuration without touching pipeline code.

**`lab_repr` awareness in `apply_clinical_bins`.** For `mean` datasets the lab columns are named `glucose`, `sodium`, etc. For `t0t1` datasets the relevant values are in `glucose_last`, `sodium_last`, etc. `apply_clinical_bins` accepts `lab_repr` and picks the correct column map automatically, eliminating the manual per-drug column renaming the notebook required.

**`plot_boxplots` is layout-agnostic.** Vitals (1×2), demographics (1×2), and labs (2×4) all go through the same function. The grid dimensions are parameters, not baked into three separate copy-pasted code blocks.

**Separation of `hte_estimation.py` and `statistical_tests.py`.** The notebook mixed HTE computation and statistical characterisation in the same cell flow. Splitting them means you can re-run just the statistical analysis (e.g., to change the ITE threshold or add a new feature group) without re-fitting the models.
