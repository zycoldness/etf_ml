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
- `docs/specs/2026-07-08-etf-v11-validation-design.md`
  - Experiment design.
- `docs/plans/2026-07-08-etf-v11-validation-notebook.md`
  - Implementation plan.

## Notes

The notebook prefers cached panel data when available. If no cache exists, panel rebuilding requires a JoinQuant research environment with `jqdata`.

