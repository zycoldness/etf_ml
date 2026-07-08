# ETF V11 Validation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ETF_V11_2021_validation_framework.ipynb`, a notebook-first validation framework for the ETF ML rotation strategy starting from 2021.

**Architecture:** The notebook loads or rebuilds a weekly ETF panel, trains fixed-start LightGBM models, scores out-of-sample rows, replays top-N ETF portfolios with turnover and cost, then exports validation tables and a markdown report. It keeps the first implementation self-contained in the notebook while organizing code into clear function sections.

**Tech Stack:** Jupyter notebook JSON, Python, pandas, numpy, LightGBM with sklearn fallback, optional JoinQuant `jqdata`, CSV/pickle outputs.

---

### Task 1: Create The Notebook Skeleton

**Files:**
- Create: `机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb`

- [ ] **Step 1: Create notebook with markdown sections**

Use `nbformat` to create a valid notebook with these sections:

```text
0. Experiment Notes
1. Config
2. Feature Definitions
3. JoinQuant Data Helpers
4. Load Or Build Weekly Panel
5. Train And Score
6. Portfolio Replay
7. Baselines
8. Common OOS And Stability
9. Diagnostics And Latest Targets
10. Export Report
```

- [ ] **Step 2: Verify notebook JSON is valid**

Run:

```powershell
python - <<'PY'
import nbformat
nbformat.read(r"机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb", as_version=4)
print("ok")
PY
```

Expected: `ok`.

### Task 2: Add Config And Feature Definitions

**Files:**
- Modify: `机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb`

- [ ] **Step 1: Add config cell**

Include:

```python
START_DATE = "2021-01-01"
END_DATE = "2026-06-30"
OUT_DIR = "etf_ml_v11_2021_validation_outputs"
TRAIN_WINDOWS = [
    ("train20210101_20231231", "2021-01-01", "2023-12-31"),
    ("train20210101_20241231", "2021-01-01", "2024-12-31"),
    ("train20210101_20251231", "2021-01-01", "2025-12-31"),
]
TOP_N_LIST = [1, 3, 5]
MIN_AVG_MONEY_20_GRID = [50000000.0]
COST_RATE = 0.0001
TARGET_SPECS = [("ret5d", "future_ret_5d", "future_ret_5d")]
```

- [ ] **Step 2: Add V10 feature lists**

Define raw price, trend, rank, and context feature lists, producing 96 feature columns.

- [ ] **Step 3: Verify feature count**

Run the first config/feature cells in a Python extraction or notebook runner and confirm the output includes `features: 96`.

### Task 3: Add Data Loading And Panel Validation

**Files:**
- Modify: `机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb`

- [ ] **Step 1: Add panel candidate paths**

Include V10/V2 local and relative candidates:

```python
PANEL_CANDIDATES = [
    os.path.join(OUT_DIR, "etf_ml_v11_weekly_panel.csv"),
    "etf_ml_v10_recent_window_stability_outputs/etf_ml_v10_weekly_panel.csv",
    "etf_ml_v2_outputs/etf_ml_v2_weekly_panel.csv",
    "etf_ml_v2_outputs_v2/etf_ml_v2_weekly_panel.csv",
    r"G:\etf_ml_v10_recent_window_stability_outputs\etf_ml_v10_weekly_panel.csv",
    r"G:\etf_ml_v2_outputs\etf_ml_v2_weekly_panel.csv",
    r"G:\etf_ml_v2_outputs_v2\etf_ml_v2_weekly_panel.csv",
]
```

- [ ] **Step 2: Add `load_or_build_panel()`**

Implement cached-panel loading first. Include JoinQuant rebuild functions but only use them if no cache exists and `jqdata` is available.

- [ ] **Step 3: Add panel checks**

Output shape, feature date min/max, weekly counts, target columns, missing ratio summary, and duplicate-column warnings.

### Task 4: Add Training And Scoring

**Files:**
- Modify: `机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb`

- [ ] **Step 1: Add `clean_train_df`, `train_lgb_or_fallback`, `predict_model`, and bundle helpers**

Use the V10 LightGBM parameters and fallback to `RandomForestRegressor`.

- [ ] **Step 2: Add train/score loop**

For each money threshold, train window, and target spec:

```python
train_mask = (
    (df["feature_date"] >= train_start)
    & (df[label_date_col] <= train_end)
    & df[target_col].notna()
)
score_mask = df["feature_date"] > train_end
```

Export model bundles to `models/` and append score rows to `etf_ml_v11_score_panel.csv`.

- [ ] **Step 3: Verify manifest and score outputs**

Expected files:

```text
etf_ml_v11_model_manifest.csv
etf_ml_v11_score_panel.csv
```

### Task 5: Add Portfolio Replay With Cost And Turnover

**Files:**
- Modify: `机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb`

- [ ] **Step 1: Add `select_targets` and `calculate_turnover`**

Use equal-weight holdings. Turnover is half the absolute weight change between previous and current target weights.

- [ ] **Step 2: Add weekly replay**

For each model/date/topN:

```python
gross_ret = selected["eval_ret"].mean()
cost_drag = turnover * COST_RATE
net_ret = gross_ret - cost_drag
```

- [ ] **Step 3: Export weekly proxy**

Expected file:

```text
etf_ml_v11_weekly_proxy.csv
```

### Task 6: Add Baselines

**Files:**
- Modify: `机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb`

- [ ] **Step 1: Add random top-N baseline**

Use multiple seeds, report mean/median/min/max random returns per week.

- [ ] **Step 2: Add simple factor baselines**

Use `ret_20` and `trend_score_25` if columns exist.

- [ ] **Step 3: Include baseline columns in weekly proxy**

Columns should include median, random mean, random median, `ret20_topn_ret`, and `trend25_topn_ret` where available.

### Task 7: Add Summary, Common OOS, Stability, And Diagnostics

**Files:**
- Modify: `机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb`

- [ ] **Step 1: Add return summary functions**

Compute cumulative return, mean return, annualized weekly Sharpe-like ratio, max drawdown, win rate, turnover, and cost drag.

- [ ] **Step 2: Add yearly and common-OOS summaries**

Export all-model OOS, 2026-common OOS, and 2025-common OOS where possible.

- [ ] **Step 3: Add stability diagnostics**

Compute holding overlap and return correlation across training windows for matching top-N portfolios.

- [ ] **Step 4: Add drawdown event diagnostics and latest targets**

Export worst periods and current latest top selections.

### Task 8: Add Report Export And Notebook Validation

**Files:**
- Modify: `机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb`

- [ ] **Step 1: Add report markdown generation**

Write `etf_ml_v11_report.md` summarizing:

```text
- data range and panel size
- best model by topN
- common OOS results
- cost impact
- stability observations
- recommended next experiment
```

- [ ] **Step 2: Validate notebook can be parsed**

Run:

```powershell
python - <<'PY'
import nbformat
nb = nbformat.read(r"机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb", as_version=4)
print(len(nb.cells))
PY
```

Expected: a positive cell count.

- [ ] **Step 3: Verify no stale cost assumption remains**

Run:

```powershell
rg -n "0\\.0001%|0\\.000001" "机器学习策略/notebooks/ETF_V11_2021_validation_framework.ipynb" "docs/superpowers/specs/2026-07-08-etf-v11-validation-design.md"
```

Expected: no matches.

