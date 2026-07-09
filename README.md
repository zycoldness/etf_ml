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
  - Compares `ret5d`, `alpha5d`, `rank5d`, and explicitly blocks `ret10d` when the panel lacks 10-day labels.
  - Keeps V10 features, model settings, liquidity floor, and evaluation rules fixed to avoid mixed-dimension overfitting.
- `research_scripts/build_etf_v14_label_experiment_notebook.py`
  - Rebuilds the V14 notebook.
- `docs/specs/2026-07-08-etf-v11-validation-design.md`
  - Experiment design.
- `docs/plans/2026-07-08-etf-v11-validation-notebook.md`
  - Implementation plan.

## Notes

The notebook prefers cached panel data when available. If no cache exists, panel rebuilding requires a JoinQuant research environment with `jqdata`.

