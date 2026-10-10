# BRGI Datathon 2026 — Team VAWAR

Forecasting the **Business Resilience & Growth Index (BRGI)** of Indian districts 1–4 quarters after the 2024Q4 origin, as p10/p50/p90 quantiles.
The forecasts are scored on horizon-weighted pinball loss.

| Final file | Notebook that reproduces it | Public LB |
|---|---|---|
| `submissions/solution_v6_w85.csv` | `notebooks/brgi_solution_v6_w85.ipynb` | 2.98849 |
| `submissions/solution_v8_w85.csv` | `notebooks/brgi_solution_v8_w85.ipynb` | ≈ 2.99 (same configuration, with the night-light reconstruction fix) |

Full methodology, results and business recommendations: `report/BRGI_Datathon_Solution_Report_v8.pdf`.

## Approach in brief
- **BRGI = 35·rank(credit yoy) + 20·rank(deposit yoy) + 30·rank(night-light yoy) + 15·rank(resilience)**, verified exactly on the data.
  So the features forecast those future cross-sectional ranks from the raw RBI bank data and the NASA Black Marble night-light data.
- **Point-in-time correct**: at origin O, BRGI labels and bank data are used only up to O−1, and night lights only up to the 2nd month of O (release_calendar.csv).
- **LightGBM quantile models** with a *full* feature set for districts with history and a *no-bank* feature set for unseen districts.
  A direct model and a model anchored on the last BRGI are blended, each over 5 seeds.
- **Rebuilt origins 2024Q1–Q3** from the raw files, since train.csv has no rows for them. Every rebuilt column was checked against existing origins.
- **Interval scaling ×0.85** around p50. Quantiles are sorted and clipped to [0, 100].
- Validation: rolling-origin backtests at 2021Q4, 2022Q4 and 2023Q4, with 15% of districts held out to simulate unseen ones.

## How to run
**Kaggle (recommended):** import a notebook from `notebooks/`, attach the competition data and click *Run All*. It takes about 8 minutes on CPU and writes `solution.csv`.

**Locally:**
```bash
pip install -r requirements.txt
export BRGI_DATA=/path/to/competition/csvs/   # train.csv, test.csv, bank_quarterly.csv, ...
cd src && python final.py ../solution_v8_w85.csv '[{"num_leaves":15,"r_seen":600,"r_un":400},{"num_leaves":15,"r_seen":600,"r_un":400,"anchor":"l1","anchor_un":"nbr"}]' 0.85
```

## Source files (`src/`)
| File | Purpose |
|---|---|
| `common.py` | quarter arithmetic, competition metric, data loading |
| `features.py` | point-in-time feature engineering |
| `augment.py` | rebuilds the missing origins 2024Q1–Q3 from the raw files |
| `model.py` | LightGBM quantile models, feature sets, quantile post-processing |
| `final.py` | trains on all labelled data and writes the submission |
| `cv2.py`, `blend.py`, `calib.py` | rolling-origin backtests, blend and interval-calibration analysis |

`src/` contains the v8 code (with the night-light fix in `augment.py`). Each notebook is self-contained and reproduces its own submission file exactly.

## Compliance
No external data and no AutoML or no-code tools. All cleaning, feature engineering and modelling code is written by the team.
Seeds are fixed and LightGBM runs in deterministic mode. Districts are not de-anonymised.
