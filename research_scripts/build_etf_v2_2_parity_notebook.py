import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_NOTEBOOK = PROJECT_DIR / "notebooks" / "ETF_V2_2_jq_parity_annual_update.ipynb"


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def build_notebook():
    cells = [
        markdown("""# ETF V2.2: JQ Parity and Annual Update Evaluation

## Purpose

This is not a new alpha-model experiment. It evaluates the already exported V2.1 model bank under the execution rule used by `joinquant_etf_ml_v2_1_parity.py`:

- Friday close data produces the signal.
- Monday open is the offline execution proxy.
- Final Top3 uses the same minimum-lot candidate fallback rule.
- Every calendar year uses only the model trained through the prior year-end.

The primary result is the stitched annual update path, not any static model's full OOS curve.
"""),
        code("""# =========================
# 0. Config
# =========================
import os
import gc
import datetime
import numpy as np
import pandas as pd
from jqdata import *

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

OUT_DIR = "etf_ml_v2_2_jq_parity_outputs"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# V2.1 output folders. Run ETF_V2_1 first; V2.2 does not retrain models.
RUN_SPECS = [
    ("money2000w_baseline", "etf_ml_v2_1_money2000w_baseline_outputs"),
    ("money2000w_ret5_volume_confirm", "etf_ml_v2_1_money2000w_ret5_volume_confirm_outputs"),
    ("money5000w_baseline", "etf_ml_v2_1_money5000w_baseline_outputs"),
    ("money5000w_ret5_volume_confirm", "etf_ml_v2_1_money5000w_ret5_volume_confirm_outputs"),
]

# The production-style annual update schedule. Do not select this after seeing the test year.
LIVE_TRAIN_TAG_BY_YEAR = {
    2024: "train20210101_20231231",
    2025: "train20210101_20241231",
    2026: "train20210101_20251231",
}

STOCK_NUM = 3
TOP_CANDIDATES = 10
INITIAL_CAPITAL = 1000000.0  # Set equal to the JoinQuant backtest capital for lot-size parity.
MIN_ORDER_LOT = 100
FIXED_SLIPPAGE = 0.001
COMMISSION_RATE = 0.0001
EVAL_START_YEAR = 2024

SIGNAL_CSV = os.path.join(OUT_DIR, "v2_2_signal_top10.csv")
EXECUTION_CSV = os.path.join(OUT_DIR, "v2_2_open_execution_proxy_weekly.csv")
YEARLY_CSV = os.path.join(OUT_DIR, "v2_2_annual_update_yearly.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "v2_2_annual_update_summary.csv")
LATEST_CSV = os.path.join(OUT_DIR, "v2_2_latest_targets.csv")

print("output:", OUT_DIR)
print("annual model schedule:", LIVE_TRAIN_TAG_BY_YEAR)
"""),
        code("""# =========================
# 1. Load V2.1 model-bank scores
# =========================
SCORE_COLUMNS = ["code", "feature_date", "next_date", "future_ret_5d", "score", "model_name", "model_file", "target_name", "train_tag"]


def load_score_panel(profile, source_dir):
    path = os.path.join(source_dir, "etf_ml_v2_score_panel.csv")
    if not os.path.exists(path):
        raise IOError("V2.1 score panel not found: " + path)
    out = pd.read_csv(path, usecols=SCORE_COLUMNS)
    out["feature_date"] = pd.to_datetime(out["feature_date"])
    out["next_date"] = pd.to_datetime(out["next_date"])
    out["profile"] = profile
    return out


score_parts = []
for profile, source_dir in RUN_SPECS:
    one = load_score_panel(profile, source_dir)
    score_parts.append(one)
    print(profile, one.shape, one["train_tag"].unique())
score_df = pd.concat(score_parts, ignore_index=True, sort=False)


def build_annual_signal_panel(all_scores):
    parts = []
    for year, train_tag in sorted(LIVE_TRAIN_TAG_BY_YEAR.items()):
        one = all_scores[(all_scores["train_tag"] == train_tag) & (all_scores["feature_date"].dt.year == year)].copy()
        one = one[one["feature_date"] >= pd.Timestamp("%s-01-01" % year)]
        parts.append(one)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out[out["feature_date"].dt.year >= EVAL_START_YEAR].copy()
    return out


signal_df = build_annual_signal_panel(score_df)
print("signal panel:", signal_df.shape)
print(signal_df.groupby(["profile", "train_tag"])["feature_date"].agg(["min", "max", "count"]))
"""),
        code("""# =========================
# 2. Friday signal -> Monday-open execution helpers
# =========================
def get_entry_exit_dates(feature_date):
    feature_date = pd.Timestamp(feature_date).normalize()
    days = pd.to_datetime(get_trade_days(
        start_date=feature_date.strftime("%Y-%m-%d"),
        end_date=(feature_date + datetime.timedelta(days=20)).strftime("%Y-%m-%d")
    ))
    days = [pd.Timestamp(x).normalize() for x in days if pd.Timestamp(x).normalize() > feature_date]
    if len(days) < 6:
        return None, None
    return days[0], days[5]


def fetch_open_prices(codes, entry_date, exit_date):
    if len(codes) == 0:
        return pd.DataFrame(columns=["code", "entry_open", "exit_open", "entry_paused"])
    try:
        px = get_price(
            codes,
            start_date=entry_date,
            end_date=exit_date,
            frequency="daily",
            fields=["open", "paused"],
            panel=False,
            fq="pre",
            skip_paused=False,
        )
    except Exception as err:
        print("open price query failed", entry_date, err)
        return pd.DataFrame(columns=["code", "entry_open", "exit_open", "entry_paused"])
    if px is None or px.empty:
        return pd.DataFrame(columns=["code", "entry_open", "exit_open", "entry_paused"])
    px["time"] = pd.to_datetime(px["time"]).dt.normalize()
    rows = []
    for code, one in px.groupby("code"):
        entry = one[one["time"] == entry_date]
        exit_ = one[one["time"] == exit_date]
        if entry.empty or exit_.empty:
            continue
        rows.append({
            "code": code,
            "entry_open": float(entry.iloc[-1]["open"]),
            "exit_open": float(exit_.iloc[-1]["open"]),
            "entry_paused": bool(entry.iloc[-1].get("paused", False)),
        })
    return pd.DataFrame(rows)


def select_tradeable_top3(ranked, price_df):
    target_value = INITIAL_CAPITAL / float(STOCK_NUM)
    price_map = price_df.set_index("code").to_dict("index") if not price_df.empty else {}
    selected = []
    skipped = []
    for code in ranked:
        if len(selected) >= STOCK_NUM:
            break
        item = price_map.get(code)
        if item is None:
            skipped.append(code + ":missing_open")
            continue
        entry = item.get("entry_open", np.nan)
        exit_ = item.get("exit_open", np.nan)
        if item.get("entry_paused", False) or pd.isnull(entry) or entry <= 0 or pd.isnull(exit_) or exit_ <= 0:
            skipped.append(code + ":paused_or_bad_open")
            continue
        if target_value < entry * MIN_ORDER_LOT:
            skipped.append(code + ":min_lot")
            continue
        selected.append(code)
    return selected, skipped, price_map
"""),
        code("""# =========================
# 3. Build strict-parity execution proxy
# =========================
def build_execution_proxy(signal_data):
    rows = []
    groups = signal_data.groupby(["profile", "train_tag", "feature_date"])
    for (profile, train_tag, feature_date), one in tqdm(groups, desc="JQ-like weekly execution"):
        ranked_df = one.sort_values("score", ascending=False).head(TOP_CANDIDATES)
        ranked = ranked_df["code"].tolist()
        entry_date, exit_date = get_entry_exit_dates(feature_date)
        if entry_date is None:
            continue
        price_df = fetch_open_prices(ranked, entry_date, exit_date)
        selected, skipped, price_map = select_tradeable_top3(ranked, price_df)
        if len(selected) == 0:
            rows.append({
                "profile": profile, "train_tag": train_tag, "feature_date": feature_date,
                "entry_date": entry_date, "exit_date": exit_date, "target_count": 0,
                "ranked_top10": ",".join(ranked), "targets": "", "skipped": ";".join(skipped),
                "gross_ret": np.nan, "net_ret": np.nan,
            })
            continue
        ret_list = []
        for code in selected:
            item = price_map[code]
            # Same fixed price slippage and commission convention as the JQ parity strategy.
            buy_price = item["entry_open"] + FIXED_SLIPPAGE
            sell_price = max(0.000001, item["exit_open"] - FIXED_SLIPPAGE)
            one_ret = sell_price / buy_price * (1.0 - COMMISSION_RATE) ** 2 - 1.0
            ret_list.append(one_ret)
        rows.append({
            "profile": profile, "train_tag": train_tag, "feature_date": feature_date,
            "entry_date": entry_date, "exit_date": exit_date, "target_count": len(selected),
            "ranked_top10": ",".join(ranked), "targets": ",".join(selected), "skipped": ";".join(skipped),
            "gross_ret": float(np.mean([(price_map[c]["exit_open"] / price_map[c]["entry_open"] - 1.0) for c in selected])),
            "net_ret": float(np.mean(ret_list)),
        })
    return pd.DataFrame(rows)


execution_df = build_execution_proxy(signal_df)
execution_df["feature_date"] = pd.to_datetime(execution_df["feature_date"])
execution_df.to_csv(EXECUTION_CSV, index=False)

signal_top10 = execution_df[["profile", "train_tag", "feature_date", "entry_date", "ranked_top10", "targets", "skipped"]].copy()
signal_top10.to_csv(SIGNAL_CSV, index=False)
print("execution rows:", execution_df.shape)
display(execution_df.tail(10))
"""),
        code("""# =========================
# 4. Report the annual-update path
# =========================
def summarize_returns(ret):
    ret = pd.Series(ret).dropna().astype(float)
    if len(ret) == 0:
        return {"weeks": 0, "cum_ret": np.nan, "max_drawdown": np.nan, "win_rate": np.nan}
    nav = (1.0 + ret).cumprod()
    return {
        "weeks": int(len(ret)),
        "cum_ret": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "win_rate": float((ret > 0).mean()),
    }


summary_rows = []
yearly_rows = []
for profile, one in execution_df.groupby("profile"):
    one = one.sort_values("feature_date")
    row = {"profile": profile}
    row.update(summarize_returns(one["net_ret"]))
    row["mean_target_count"] = float(one["target_count"].mean())
    row["incomplete_target_weeks"] = int((one["target_count"] < STOCK_NUM).sum())
    summary_rows.append(row)
    for year, year_df in one.groupby(one["feature_date"].dt.year):
        yrow = {"profile": profile, "year": int(year)}
        yrow.update(summarize_returns(year_df["net_ret"]))
        yrow["mean_target_count"] = float(year_df["target_count"].mean())
        yearly_rows.append(yrow)

summary_df = pd.DataFrame(summary_rows).sort_values("cum_ret", ascending=False)
yearly_df = pd.DataFrame(yearly_rows).sort_values(["profile", "year"])
summary_df.to_csv(SUMMARY_CSV, index=False)
yearly_df.to_csv(YEARLY_CSV, index=False)

latest_date = execution_df["feature_date"].max()
latest_df = execution_df[execution_df["feature_date"] == latest_date].copy()
latest_df.to_csv(LATEST_CSV, index=False)

display(summary_df)
display(yearly_df)
display(latest_df)
print("saved:", SIGNAL_CSV, EXECUTION_CSV, YEARLY_CSV, SUMMARY_CSV, LATEST_CSV)
"""),
        markdown("""## Decision Rule

Use this notebook only after verifying the same pkl in `joinquant_etf_ml_v2_1_parity.py`.

1. Compare the notebook `ranked_top10` and `targets` with `ETF_ML_TARGETS` logs for the same signal date.
2. Resolve any target mismatch before interpreting return differences.
3. Compare the weekly open proxy with JQ weekly portfolio returns; open proxy is an execution approximation, not the final backtest result.
4. Feature decisions use the stitched annual path: 2024 model -> 2024, 2025 model -> 2025, 2026 model -> 2026. Do not choose a static model using its later OOS return.
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.6.7", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT_NOTEBOOK)


if __name__ == "__main__":
    build_notebook()
