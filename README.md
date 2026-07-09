# ETF ML

ETF machine-learning rotation research focused on validation from 2021 onward.

## Contents

- `notebooks/ETF_V11_2021_validation_framework.ipynb`
  - V11 validation notebook.
  - Fixed 2021 start.
  - `future_ret_5d + money5000w + LightGBM`.
  - `top1 / top3 / top5`.
  - Transaction cost set to 万分之一: `COST_RATE = 0.0001`.
  - Common-OOS, turnover, cost, random baseline, simple momentum/trend baseline, and stability diagnostics.
- `research_scripts/build_etf_v11_notebook.py`
  - Rebuilds the V11 notebook from the prior V10 research notebook template.
- `notebooks/ETF_V12_live_readiness_validation.ipynb`
  - Live-readiness validation from V11 outputs.
  - Flags missing score panels, ETF metadata, capacity, turnover, and theme concentration blockers.
- `research_scripts/build_etf_v12_notebook.py`
  - Rebuilds the V12 notebook.
- `notebooks/ETF_V13_live_gate_validation.ipynb`
  - Full live-gate validation after `etf_ml_v11_score_panel.csv` and `etf_ml_v11_weekly_panel.csv` are available.
  - Tests ML score vs momentum/trend/equal-weight/random/median baselines, cost stress, turnover, capacity, common OOS, and target-schedule export.
- `research_scripts/build_etf_v13_live_gate_notebook.py`
  - Rebuilds the V13 notebook.
- `strategies/joinquant_etf_ml_v13_targets_strategy.py`
  - JoinQuant target-schedule strategy template. Paste the generated `TARGET_SCHEDULE` from V13 before backtesting.
- `notebooks/ETF_V14_label_experiment.ipynb`
  - Label-only model experiment for fixed `top3`.
  - Generated from the original V10 notebook so ETF universe, feature engineering, liquidity filter, listing-age filter,复权,停牌处理, and weekly-date logic stay V10-aligned.
  - Compares `ret5d`, `alpha5d`, `rank5d`, `ret10d`, `alpha10d`, and `rank10d`.
  - Matches label horizon to holding period: 5-day labels use weekly rebalance and `future_ret_5d`; 10-day labels use 2-week rebalance and `future_ret_10d`.
  - Does not require V11 exports: if `etf_ml_v14_weekly_panel.csv` is absent, it rebuilds through the copied V10 panel-building functions in JoinQuant research.
  - Keeps V10 features, model settings, and liquidity floor fixed to avoid mixed-dimension overfitting.
- `research_scripts/build_etf_v14_label_experiment_notebook.py`
  - Rebuilds the V14 notebook from the original V10 notebook source. Set `ETF_V10_SOURCE_NOTEBOOK` if the V10 source is not at `G:/ETF_V10_recent_window_stability实验.ipynb`.
- `docs/specs/2026-07-08-etf-v11-validation-design.md`
  - Experiment design.
- `docs/plans/2026-07-08-etf-v11-validation-notebook.md`
  - Implementation plan.

## Notes

The notebook prefers cached panel data when available. If no cache exists, panel rebuilding requires a JoinQuant research environment with `jqdata`.

