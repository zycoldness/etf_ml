# ETF V12 Live Readiness Validation Design

## Goal

Create `ETF_V12_live_readiness_validation.ipynb`, a comprehensive notebook that decides whether the ETF ML strategy is ready for paper trading or still research-only.

V12 does not replace V11. It consumes V11 outputs and, when available, richer `score_panel` / `weekly_panel` data to run additional baselines, robustness checks, walk-forward validation, capacity checks, drawdown attribution, portfolio-rule tests, and a final live-readiness scorecard.

## Inputs

Primary V11 output files:

- `etf_ml_v11_model_manifest.csv`
- `etf_ml_v11_weekly_proxy.csv`
- `etf_ml_v11_weekly_cost_proxy.csv`
- `etf_ml_v11_summary.csv`
- `etf_ml_v11_yearly.csv`
- `etf_ml_v11_cost_summary.csv`
- `etf_ml_v11_common_oos.csv`
- `etf_ml_v11_cost_stability.csv`
- `etf_ml_v11_latest_targets.csv`
- `etf_ml_v11_drawdown_events.csv`
- `etf_ml_v11_score_gap_diagnostics.csv`

Optional richer files:

- `etf_ml_v11_score_panel.csv`
- `etf_ml_v11_weekly_panel.csv`

When optional files are unavailable, V12 must still run and mark dependent checks as `BLOCKED_MISSING_DATA` rather than producing fake pass/fail conclusions.

## Validation Sections

### 1. Data Quality

Check:

- ETF names available.
- ETF group classification available.
- weekly date coverage.
- duplicate model/week/topN rows.
- missing return columns.
- suspicious returns beyond configured thresholds.
- label coverage if weekly panel exists.

Output: `etf_ml_v12_data_quality.csv`.

### 2. Baseline Completeness

Compare the strategy against:

- universe median.
- random topN.
- simple `ret_20` topN if score panel exists.
- simple `ret_60` topN if score panel exists.
- simple `trend_score_25` topN if score panel exists.
- simple `trend_score_60` topN if score panel exists.
- equal-weight ETF universe if score panel exists.

Output:

- `etf_ml_v12_baseline_weekly.csv`
- `etf_ml_v12_baseline_summary.csv`

### 3. Common OOS

Build fair common-period comparisons:

- `2021-2023` vs `2021-2024` on `2025-01-01` onward.
- all available windows on `2026-01-01` onward.

Output: `etf_ml_v12_common_oos.csv`.

### 4. Robustness Grid

Test sensitivity to:

- topN: `1, 3, 5, 7` when score panel exists; otherwise available topN only.
- cost rates: `0.0001, 0.0003, 0.0005, 0.001`.
- training windows.
- liquidity thresholds if score panel includes `money_mean_20`.

Output: `etf_ml_v12_robustness_grid.csv`.

### 5. Walk-Forward

Use actual forward yearly slices:

- `2021-2023` model trades 2024.
- `2021-2024` model trades 2025.
- `2021-2025` model trades 2026.

Output:

- `etf_ml_v12_walkforward_weekly.csv`
- `etf_ml_v12_walkforward.csv`

### 6. Turnover And Capacity

Check:

- average turnover.
- max turnover.
- annualized turnover estimate.
- cost drag at each configured cost rate.
- capacity at 1m, 5m, and 10m notional if selected ETF liquidity is available.

Output: `etf_ml_v12_turnover_capacity.csv`.

### 7. Drawdown Attribution

Attribute worst periods by:

- ETF code.
- ETF name if available.
- ETF group if available.
- score gap.
- random and median comparison.

Output: `etf_ml_v12_drawdown_attribution.csv`.

### 8. Portfolio Rules

Evaluate:

- current top3/top5 equal weight.
- top5 as stabilizer.
- model consensus / voting across training windows when same dates exist.
- score-gap risk flags.
- theme-cap readiness, blocked if ETF groups are missing.

Output:

- `etf_ml_v12_rule_weekly.csv`
- `etf_ml_v12_portfolio_rules.csv`

### 9. Live Readiness Scorecard

Score each requirement:

- data quality.
- baseline superiority.
- common OOS robustness.
- walk-forward robustness.
- cost robustness.
- turnover acceptability.
- capacity availability.
- drawdown explainability.
- portfolio-rule improvement.

Final status:

- `PASS_PAPER`: ready for 1-3 months paper trading.
- `RESEARCH_ONLY`: signal exists but more validation or rule work is needed.
- `REJECT`: insufficient evidence.

Output:

- `etf_ml_v12_live_readiness_scorecard.csv`
- `etf_ml_v12_report.md`

## Default Decision Rule

Default to `RESEARCH_ONLY` unless:

- common OOS beats median and random for top3 or top5;
- walk-forward is positive in most yearly slices;
- cost stress through `0.0005` does not destroy the edge;
- turnover is explainable and not extreme after rule selection;
- data quality does not block drawdown attribution;
- at least one simple momentum/trend baseline is beaten or explicitly unavailable and marked as a blocker.

