# ETF V11 2021 Validation Framework Design

## Goal

Build a notebook-first validation experiment for the ETF machine-learning rotation strategy. The first notebook should verify whether the existing `ret5d + money5000w + LightGBM` ETF selector remains usable after stability, robustness, common-OOS, and transaction-cost checks.

The experiment starts from `2021-01-01` because the ETF universe before 2021 is smaller and less representative of the current tradable ETF ecosystem.

## Primary Artifact

Create one notebook:

```text
ETF_V11_2021_validation_framework.ipynb
```

The notebook is the main experiment surface. Code inside the notebook should still be organized into clear function sections so that stable pieces can later be extracted into `.py` modules if needed.

## Scope

Included:

- Reproduce the V10 baseline logic.
- Use data from `2021-01-01` onward.
- Keep the first-stage model fixed: `future_ret_5d` target, 96 V10 features, LightGBM, weekly rebalance.
- Validate fixed-start training windows:
  - `2021-01-01` to `2023-12-31`
  - `2021-01-01` to `2024-12-31`
  - `2021-01-01` to `2025-12-31`
- Evaluate `top1`, `top3`, and `top5`.
- Use liquidity filters such as `money_mean_20 >= 50,000,000`.
- Add turnover and cost-aware returns.
- Use transaction cost rate 万分之一. In decimal form this is `0.0001` per traded notional, equivalent to `0.01%` or `1 bp`.
- Compare common OOS periods, especially `2026-01-01` onward where available.
- Compare against random top-N, universe median, and simple momentum/trend baselines.
- Export summary, yearly, weekly, cost, stability, and report outputs.

Excluded from V11:

- New feature engineering beyond the V10 feature set.
- Model architecture search.
- Complex ensemble rules.
- Full live-trading integration.

These can be handled in V12/V13 after V11 establishes whether the baseline is trustworthy.

## Notebook Structure

### 0. Experiment Notes

Document the purpose, constraints, assumptions, and difference from V10:

- V11 is a validation notebook, not an optimization notebook.
- All data starts from 2021.
- The first goal is trustworthiness, not maximum backtest return.

### 1. Config

Define:

- `START_DATE = "2021-01-01"`
- `END_DATE = "2026-06-30"` or latest available date in cache.
- `TRAIN_WINDOWS`
- `TOP_N_LIST = [1, 3, 5]`
- `MIN_AVG_MONEY_20_GRID = [50000000.0]`
- `COST_RATE = 0.0001`
- `TARGET_SPECS = [("ret5d", "future_ret_5d", "future_ret_5d")]`
- Output directory such as `etf_ml_v11_2021_validation_outputs`.

### 2. Data Loading And Panel Checks

Load existing cached panel if available. Candidate inputs should include the V10 output panel and the V2 weekly panel paths. If no cache exists, provide a JoinQuant-only rebuild path using `get_all_securities`, `get_price`, and `get_trade_days`.

Validation checks:

- `feature_date` min/max.
- weekly row count.
- ETF count by week.
- label availability.
- missing feature ratios.
- duplicate columns.
- target leakage columns excluded from model features.

### 3. Feature And Label Definitions

Reuse V10 feature definitions:

- price and return features
- volatility and drawdown features
- liquidity and volume features
- trend annualized return, trend R2, trend score, trend volatility
- cross-sectional rank features
- pool context features

Use `future_ret_5d` as the primary target. Keep `target_alpha_5d` available for later experiments, but do not make it the primary V11 target.

### 4. Train And Score

For each training window:

1. Filter training rows with `feature_date >= train_start`.
2. Use only labels whose future return is observable by the training end date.
3. Train LightGBM with V10 parameters.
4. Score rows after the training end date.
5. Export model manifest and score panel.

The training split must be based on label availability, not only feature date.

### 5. Portfolio Replay

For each model and each week:

- sort by score descending
- select `top1`, `top3`, and `top5`
- use equal weight
- compute gross weekly return
- compute turnover from previous holdings
- apply transaction cost using `COST_RATE`
- record selected ETF codes, names, groups if available, gross return, net return, median return, random return, and simple baseline returns

### 6. Common OOS Comparison

Build common-period comparisons so models are compared over the same calendar dates. Required views:

- all available OOS per model
- common OOS from `2026-01-01` onward
- optional common OOS from `2025-01-01` onward for models trained by `2024-12-31`

Do not rank model quality using unequal OOS lengths without labeling that comparison as non-common-period.

### 7. Robustness Diagnostics

Compute:

- cumulative return
- annual/yearly return
- max drawdown
- weekly win rate
- weekly Sharpe-like ratio
- max single-period loss
- turnover
- cost drag
- excess versus universe median
- excess versus random top-N
- excess versus simple momentum/trend baselines
- holding overlap across training windows
- return correlation across training windows
- score gaps between rank 1/2/3
- drawdown event table

### 8. Baselines

Include:

- universe median return
- random top-N with multiple seeds
- simple `ret_20` top-N
- simple `trend_score_25` top-N if available

Random baselines should use multiple seeds and report average, median, and range.

### 9. Outputs

Write:

```text
etf_ml_v11_weekly_panel.csv
etf_ml_v11_model_manifest.csv
etf_ml_v11_score_panel.csv
etf_ml_v11_weekly_proxy.csv
etf_ml_v11_summary.csv
etf_ml_v11_yearly.csv
etf_ml_v11_common_oos.csv
etf_ml_v11_cost_sensitivity.csv
etf_ml_v11_stability.csv
etf_ml_v11_drawdown_events.csv
etf_ml_v11_latest_targets.csv
etf_ml_v11_report.md
```

## Success Criteria

V11 is successful if it can answer:

- Does the V10 baseline reproduce from cached panel data?
- Does `top3` remain more reliable than `top1` under common OOS comparison?
- Does the strategy still beat universe median, random top-N, and simple momentum/trend baselines after cost?
- Are results stable enough across fixed-start windows to justify model or portfolio optimization?
- What is the recommended next experiment: portfolio rule improvement, model ensemble, or feature/label optimization?

## Risk Controls

- Treat missing ETF names/groups as a data-quality issue; do not make final conclusions about theme concentration until names/groups are fixed.
- Flag any week where label coverage is incomplete.
- Keep model training and portfolio replay timing explicit.
- Do not use rows after the evaluation date to compute features, labels, ranks, or baselines.
- Do not make live-trading claims from notebook proxy results alone.

## Next Step After Approval

Write an implementation plan for `ETF_V11_2021_validation_framework.ipynb`, then implement the notebook and verify it can run at least through cached-panel loading, train/score, replay, and summary generation.
