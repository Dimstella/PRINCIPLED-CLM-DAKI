# `cml/`

Causal Machine Learning pipeline for DAKI pharmacovigilance.  


---

## Files

| File | Replaces | Purpose |
|---|---|---|
| `config.py` | shared setup cells in all notebooks | Model registries, best-model configs (Tables 12–15), confounder DAGs, shared constants |
| `learners.py` | `s_learner()`, `t_learner()`, `x_learner()`, `dml_learner()` helpers | Uniform wrappers around EconML with bootstrap CI |
| `model_selection.py` | `Select_best_ML_models_for_CML_Simethicone.ipynb` | Full benchmark of all classifier/regressor combinations for each architecture |

The CATE estimation pipeline that uses these modules lives in `analysis/cate_estimation.py`.

---

## Prerequisites

```bash
pip install econml xgboost scikit-learn pandas numpy joblib openpyxl
```

---

## Model selection

Run the benchmark to find the best model for each architecture on a given dataset:

```python
import pandas as pd
import numpy as np
import sys; sys.path.insert(0, "cml/")
from model_selection import run_model_selection, save_model_selection_results

df = pd.read_parquet("data/ibuprofen_simethicone.parquet.gzip")
X = df.drop(columns=["y","t","subject_id"]).to_numpy()
y = df["y"].to_numpy()
t = df["t"].to_numpy()

results = run_model_selection(X, y, t)
save_model_selection_results(results, "results/ibuprofen_sim_model_selection.xlsx")
```

Or from the CLI (coming in a future notebook wrapper).

---

## Architecture overview

### S-Learner
Single model trained on `[X | T]`.  
Performance: average classification metrics over 5-fold CV.  
Best model selected by F1 score.

### T-Learner
Two separate models, one per treatment arm.  
All (classifier × classifier) pairs evaluated.  
Best combo selected by average F1.

### X-Learner
Two first-stage classifiers + two second-stage regressors.  
Best combo ranked by the composite `Overall` score:  
`Overall = (Accuracy + Precision + Recall + F1 + mean_R²) / 5`

### DML (Double Machine Learning)
Three regression models: outcome, treatment, final estimator.  
Best triplet ranked by composite score `S_F` (Equation 1 in the paper):

```
S = 1 / (log(10 + MSE) + log(10 + RMSE) − 1) + R²
```

---

## Bootstrap CI
All CATE/ATE estimates use `inference='bootstrap'` with n=100 samples  
(paper §2.5).  CI width ≥ 0.85 → non-robust estimate (paper §2.5.3).
