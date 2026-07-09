import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_NOTEBOOK = PROJECT_DIR / "notebooks" / "ETF_V14_label_experiment.ipynb"


def source(text):
    return text.strip("\n").splitlines(True)


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source(text)}


cells = []

cells.append(markdown_cell(
    """
# ETF_V14 Label Experiment

V14-A tests one optimization dimension only: **training target / label**.

Fixed controls:

- Trading rule: top3 equal weight
- Rebalance: label-matched (`5d` labels weekly, `10d` labels every 2 weeks)
- Liquidity floor: `money_mean_20 >= 5000w`
- Features: V10 feature set, unchanged
- Model parameters: V10 LightGBM params, unchanged when LightGBM is available
- Evaluation return: label-matched (`future_ret_5d` for 5-day labels, `future_ret_10d` for 10-day labels)

Candidate labels:

- `ret5d`: absolute future 5-day return
- `alpha5d`: future 5-day return minus weekly ETF-pool median return
- `rank5d`: weekly cross-sectional percentile rank of future 5-day return
- `ret10d` / `alpha10d` / `rank10d`: only if `future_ret_10d` exists in the input panel; otherwise explicitly blocked

The value of this experiment is not "find the prettiest curve"; it tells us
whether the model should learn absolute return, excess return, or cross-sectional
ranking before we touch features, parameters, or weak-signal filters. Longer
labels are evaluated with matching holding periods so we do not accidentally
test a 10-day target with a 5-day trading rule.

Data rule:

- The notebook prefers an existing V14 panel cache only to save time.
- If no cache exists, it rebuilds the ETF weekly panel directly in JoinQuant
  research via `jqdata`, including both 5-day and 10-day labels.
"""
))

cells.append(code_cell(
    r'''
# =========================
# 0. Config and IO
# =========================
import os
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from jqdata import *  # noqa: F401,F403
    JQ_IMPORT_ERROR = None
except Exception as err:
    JQ_IMPORT_ERROR = err

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

pd.set_option("display.max_columns", 180)
pd.set_option("display.width", 240)

OUT_DIR = "etf_ml_v14_label_experiment_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_DIR_CANDIDATES = [
    "G:/",
    ".",
    "etf_ml_v14_label_experiment_outputs",
    "../etf_ml_v14_label_experiment_outputs",
    "etf_ml_v11_2021_validation_outputs",
    "../etf_ml_v11_2021_validation_outputs",
]

WEEKLY_PANEL_FILES = [
    "etf_ml_v14_weekly_panel.csv",
    "etf_ml_v11_weekly_panel.csv",  # legacy cache name; not required
]
PANEL_CACHE_CSV = os.path.join(OUT_DIR, "etf_ml_v14_weekly_panel.csv")
DATA_START_DATE = "2021-01-01"
DATA_END_DATE = "2026-06-30"
LISTING_DAYS_MIN = 60
PRICE_HISTORY_COUNT = 140
JQ_CHUNK_SIZE = 80
LABEL_HORIZONS = [5, 10]
MIN_AVG_MONEY_20 = 50000000.0
TOP_N = 3
COST_RATE = 0.0001
COST_STRESS_RATE = 0.0005
RANDOM_SEEDS = list(range(50))
RANDOM_SEED = 42

TRAIN_WINDOWS = [
    ("train20210101_20231231", "2021-01-01", "2023-12-31"),
    ("train20210101_20241231", "2021-01-01", "2024-12-31"),
    ("train20210101_20251231", "2021-01-01", "2025-12-31"),
    ("train20230101_20241231", "2023-01-01", "2024-12-31"),
    ("train20230101_20251231", "2023-01-01", "2025-12-31"),
    ("train20240101_20251231", "2024-01-01", "2025-12-31"),
]

COMMON_OOS_STARTS = ["2024-01-01", "2025-01-01", "2026-01-01"]

TREND_WINDOWS = [10, 20, 25, 60]
POOL_CONTEXT_WINDOW = 25

BASE_PRICE_FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_5", "vol_20", "vol_60",
    "close_to_ma20", "close_to_ma60", "ma5_to_ma20", "ma20_to_ma60",
    "drawdown_20", "drawdown_60",
    "amp_20", "amp_60",
    "money_mean_20", "money_ratio_5_20", "money_ratio_20_60",
    "volume_ratio_5_20", "volume_ratio_20_60",
    "max_ret_20", "min_ret_20",
]

TREND_FEATURE_COLS = []
for w in TREND_WINDOWS:
    TREND_FEATURE_COLS.extend([
        "trend_ann_%s" % w,
        "trend_r2_%s" % w,
        "trend_score_%s" % w,
        "trend_vol_%s" % w,
        "trend_score_vol_adj_%s" % w,
        "trend_simple_ann_%s" % w,
    ])

RAW_FEATURE_COLS = BASE_PRICE_FEATURE_COLS + TREND_FEATURE_COLS
RANK_FEATURE_COLS = ["rank_" + c for c in RAW_FEATURE_COLS]
CONTEXT_FEATURE_COLS = [
    "pool_breadth_%s" % POOL_CONTEXT_WINDOW,
    "pool_median_vol_%s" % POOL_CONTEXT_WINDOW,
]
FEATURE_COLS = RAW_FEATURE_COLS + RANK_FEATURE_COLS + CONTEXT_FEATURE_COLS

LGB_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": 5,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "min_gain_to_split": 0.0,
    "verbose": -1,
    "seed": RANDOM_SEED,
}
NUM_BOOST_ROUND = 180


def find_input_file(fnames):
    for fname in fnames:
        for base in INPUT_DIR_CANDIDATES:
            path = os.path.join(base, fname)
            if os.path.exists(path):
                return path
    return None


def jq_ready():
    return all(name in globals() for name in ["get_price", "get_all_securities", "get_trade_days"])


def chunked(seq, size):
    seq = list(seq)
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def normalize_price_df(px):
    if px is None:
        return pd.DataFrame()
    df = px.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame().reset_index()
    elif isinstance(df, pd.DataFrame):
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif "time" not in df.columns and "date" not in df.columns and "datetime" not in df.columns:
            df = df.reset_index()
    else:
        try:
            df = df.to_frame(filter_observations=False).reset_index()
        except Exception:
            return pd.DataFrame()

    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in ["index", "level_0", "datetime", "date"]:
            rename_map[c] = "time"
        elif lc in ["level_1", "security", "order_book_id"]:
            rename_map[c] = "code"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.normalize()
    if "code" in df.columns:
        df["code"] = df["code"].astype(str)
    return df


def get_jq_trade_days(start_date, end_date):
    tdays = get_trade_days(start_date=start_date, end_date=end_date)  # noqa: F405
    return pd.to_datetime(list(tdays)).normalize()


def weekly_feature_dates(trade_days):
    s = pd.Series(pd.to_datetime(trade_days).normalize()).dropna().sort_values()
    if s.empty:
        return []
    return sorted(s.groupby(s.dt.to_period("W-FRI")).max().dropna().tolist())


def get_etf_universe(date):
    sec = get_all_securities(["etf"], date=date)  # noqa: F405
    if sec is None or sec.empty:
        return []
    sec = sec.copy()
    if "start_date" in sec.columns:
        start_dates = pd.to_datetime(sec["start_date"], errors="coerce")
        sec = sec[start_dates <= pd.Timestamp(date) - pd.Timedelta(days=LISTING_DAYS_MIN)]
    return sorted(sec.index.astype(str).tolist())


def trend_metrics(close, w):
    y = pd.Series(close).dropna().astype(float).tail(w)
    if len(y) < max(5, int(w * 0.6)):
        return {"ann": np.nan, "r2": np.nan, "score": np.nan, "vol": np.nan, "score_vol_adj": np.nan, "simple_ann": np.nan}
    ret = y.pct_change().dropna()
    vol = float(ret.std() * math.sqrt(252.0)) if len(ret) > 1 else np.nan
    x = np.arange(len(y), dtype=float)
    log_y = np.log(y.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(log_y) < max(5, int(w * 0.6)):
        return {"ann": np.nan, "r2": np.nan, "score": np.nan, "vol": vol, "score_vol_adj": np.nan, "simple_ann": np.nan}
    x = np.arange(len(log_y), dtype=float)
    slope, intercept = np.polyfit(x, log_y.values, 1)
    pred = intercept + slope * x
    ss_res = float(((log_y.values - pred) ** 2).sum())
    ss_tot = float(((log_y.values - log_y.values.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ann = float(np.exp(slope * 252.0) - 1.0)
    simple_ann = float((y.iloc[-1] / y.iloc[0]) ** (252.0 / max(1, len(y) - 1)) - 1.0) if y.iloc[0] > 0 else np.nan
    score = ann * max(0.0, r2)
    score_vol_adj = score / max(vol, 0.01) if pd.notnull(vol) else np.nan
    return {"ann": ann, "r2": float(r2), "score": score, "vol": vol, "score_vol_adj": score_vol_adj, "simple_ann": simple_ann}


def pct_ret(close, n):
    s = pd.Series(close).dropna().astype(float)
    if len(s) <= n or s.iloc[-n - 1] <= 0:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-n - 1] - 1.0)


def calc_feature_record(code, g):
    g = g.sort_values("time").copy()
    close = g["close"].astype(float)
    high = g["high"].astype(float) if "high" in g.columns else close
    low = g["low"].astype(float) if "low" in g.columns else close
    volume = g["volume"].astype(float) if "volume" in g.columns else pd.Series(np.nan, index=g.index)
    money = g["money"].astype(float) if "money" in g.columns else pd.Series(np.nan, index=g.index)
    if len(close.dropna()) < 65:
        return None
    rec = {"code": code}
    for n in [1, 5, 10, 20, 60]:
        rec["ret_%s" % n] = pct_ret(close, n)
    daily_ret = close.pct_change()
    for n in [5, 20, 60]:
        rec["vol_%s" % n] = float(daily_ret.tail(n).std() * math.sqrt(252.0)) if daily_ret.tail(n).notnull().sum() > 1 else np.nan
    ma5 = close.tail(5).mean()
    ma20 = close.tail(20).mean()
    ma60 = close.tail(60).mean()
    rec["close_to_ma20"] = float(close.iloc[-1] / ma20 - 1.0) if ma20 > 0 else np.nan
    rec["close_to_ma60"] = float(close.iloc[-1] / ma60 - 1.0) if ma60 > 0 else np.nan
    rec["ma5_to_ma20"] = float(ma5 / ma20 - 1.0) if ma20 > 0 else np.nan
    rec["ma20_to_ma60"] = float(ma20 / ma60 - 1.0) if ma60 > 0 else np.nan
    for n in [20, 60]:
        c = close.tail(n)
        h = high.tail(n)
        l = low.tail(n)
        rec["drawdown_%s" % n] = float(close.iloc[-1] / c.max() - 1.0) if c.max() > 0 else np.nan
        rec["amp_%s" % n] = float(h.max() / l.min() - 1.0) if l.min() > 0 else np.nan
    rec["money_mean_20"] = float(money.tail(20).mean())
    rec["money_ratio_5_20"] = float(money.tail(5).mean() / money.tail(20).mean()) if money.tail(20).mean() > 0 else np.nan
    rec["money_ratio_20_60"] = float(money.tail(20).mean() / money.tail(60).mean()) if money.tail(60).mean() > 0 else np.nan
    rec["volume_ratio_5_20"] = float(volume.tail(5).mean() / volume.tail(20).mean()) if volume.tail(20).mean() > 0 else np.nan
    rec["volume_ratio_20_60"] = float(volume.tail(20).mean() / volume.tail(60).mean()) if volume.tail(60).mean() > 0 else np.nan
    rec["max_ret_20"] = float(daily_ret.tail(20).max())
    rec["min_ret_20"] = float(daily_ret.tail(20).min())
    for w in TREND_WINDOWS:
        tm = trend_metrics(close, w)
        rec["trend_ann_%s" % w] = tm["ann"]
        rec["trend_r2_%s" % w] = tm["r2"]
        rec["trend_score_%s" % w] = tm["score"]
        rec["trend_vol_%s" % w] = tm["vol"]
        rec["trend_score_vol_adj_%s" % w] = tm["score_vol_adj"]
        rec["trend_simple_ann_%s" % w] = tm["simple_ann"]
    return rec


def fetch_feature_frame(feature_date, codes):
    fields = ["open", "close", "high", "low", "volume", "money"]
    rows = []
    for part in chunked(codes, JQ_CHUNK_SIZE):
        try:
            px = get_price(part, end_date=feature_date, frequency="daily", fields=fields, count=PRICE_HISTORY_COUNT, panel=False, fq="pre", skip_paused=False)  # noqa: F405
        except Exception as err:
            print("get_price feature failed", feature_date, len(part), err)
            continue
        df = normalize_price_df(px)
        if df.empty or "code" not in df.columns or "close" not in df.columns:
            continue
        for code, g in df.groupby("code"):
            rec = calc_feature_record(code, g)
            if rec is not None:
                rec["feature_date"] = pd.Timestamp(feature_date)
                rec["rebalance_date"] = pd.Timestamp(feature_date)
                rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "money_mean_20" in out.columns:
        out = out[out["money_mean_20"] >= MIN_AVG_MONEY_20].copy()
    for col in RAW_FEATURE_COLS:
        if col in out.columns:
            out["rank_" + col] = out[col].rank(pct=True)
    out["pool_breadth_%s" % POOL_CONTEXT_WINDOW] = float((out.get("ret_20", pd.Series(dtype=float)) > 0).mean())
    out["pool_median_vol_%s" % POOL_CONTEXT_WINDOW] = float(out.get("vol_20", pd.Series(dtype=float)).median())
    return out


def fetch_future_returns_multi(feature_date, codes, horizons, trade_days):
    trade_days = pd.to_datetime(trade_days).normalize()
    feature_date = pd.Timestamp(feature_date).normalize()
    idx_arr = np.where(trade_days == feature_date)[0]
    if len(idx_arr) == 0:
        return pd.DataFrame()
    start_idx = int(idx_arr[0])
    max_h = max(horizons)
    if start_idx + max_h >= len(trade_days):
        return pd.DataFrame()
    horizon_dates = {h: pd.Timestamp(trade_days[start_idx + h]).normalize() for h in horizons}
    rows = []
    for part in chunked(codes, JQ_CHUNK_SIZE):
        try:
            px = get_price(part, start_date=feature_date, end_date=horizon_dates[max_h], frequency="daily", fields=["close"], panel=False, fq="pre", skip_paused=False)  # noqa: F405
        except Exception as err:
            print("get_price label failed", feature_date, len(part), err)
            continue
        df = normalize_price_df(px)
        if df.empty or "code" not in df.columns or "close" not in df.columns:
            continue
        for code, g in df.groupby("code"):
            g = g.sort_values("time")
            base = g[g["time"] <= feature_date]["close"].dropna()
            if base.empty or float(base.iloc[-1]) <= 0:
                continue
            rec = {"code": code, "feature_date": feature_date}
            base_close = float(base.iloc[-1])
            for h, dt in horizon_dates.items():
                fut = g[g["time"] <= dt]["close"].dropna()
                rec["next_date_%sd" % h] = dt
                rec["future_ret_%sd" % h] = float(fut.iloc[-1] / base_close - 1.0) if not fut.empty else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def build_weekly_panel_from_jq():
    if not jq_ready():
        detail = "jqdata unavailable"
        if JQ_IMPORT_ERROR is not None:
            detail += ": %s" % JQ_IMPORT_ERROR
        raise RuntimeError(
            "No cached V14 panel found and %s. Run this notebook in JoinQuant research, "
            "or provide etf_ml_v14_weekly_panel.csv." % detail
        )
    trade_days = get_jq_trade_days(DATA_START_DATE, DATA_END_DATE)
    feature_dates = weekly_feature_dates(trade_days)
    rows = []
    print("building V14 weekly panel from JoinQuant", DATA_START_DATE, DATA_END_DATE, "weeks", len(feature_dates))
    for i, feature_date in enumerate(feature_dates):
        if i % 20 == 0:
            print("panel week", i + 1, "/", len(feature_dates), feature_date)
        codes = get_etf_universe(feature_date)
        if not codes:
            continue
        feature_df = fetch_feature_frame(feature_date, codes)
        if feature_df.empty:
            continue
        label_df = fetch_future_returns_multi(feature_date, feature_df["code"].tolist(), LABEL_HORIZONS, trade_days)
        if label_df.empty:
            continue
        week = feature_df.merge(label_df, on=["code", "feature_date"], how="left")
        if "next_date_5d" in week.columns:
            week["next_date"] = week["next_date_5d"]
        rows.append(week)
    panel = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if panel.empty:
        raise RuntimeError("JoinQuant panel rebuild produced no rows; check ETF universe and date range.")
    panel.to_csv(PANEL_CACHE_CSV, index=False)
    try:
        panel.to_csv("etf_ml_v14_weekly_panel.csv", index=False)
    except Exception:
        pass
    print("built panel", panel.shape, "saved", PANEL_CACHE_CSV)
    return panel


panel_path = find_input_file(WEEKLY_PANEL_FILES)
if panel_path is not None:
    panel_raw = pd.read_csv(panel_path)
    print("loaded cached panel", panel_raw.shape, "from", panel_path)
else:
    panel_raw = build_weekly_panel_from_jq()
'''
))

cells.append(code_cell(
    r'''
# =========================
# 1. Helpers and Label Construction
# =========================
def save_csv(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    try:
        df.to_csv(name, index=False)
    except Exception:
        pass
    print("saved", path, df.shape)
    return path


def unique_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def summarize_returns(ret):
    r = pd.Series(ret).dropna().astype(float)
    if r.empty:
        return {
            "periods": 0, "cum_ret": np.nan, "mean_ret": np.nan,
            "max_drawdown": np.nan, "win_rate": np.nan, "sharpe52": np.nan,
            "max_loss": np.nan,
        }
    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    std = float(r.std())
    return {
        "periods": int(len(r)),
        "cum_ret": float(nav.iloc[-1] - 1.0),
        "mean_ret": float(r.mean()),
        "max_drawdown": float(dd.min()),
        "win_rate": float((r > 0).mean()),
        "sharpe52": float(r.mean() / std * math.sqrt(52.0)) if std > 0 else np.nan,
        "max_loss": float(r.min()),
    }


def add_summary(prefix, ret):
    s = summarize_returns(ret)
    return {prefix + "_" + k: v for k, v in s.items()}


def target_weights(codes):
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {c: w for c in codes}


def calc_turnover(prev_w, new_w):
    keys = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys))


panel = panel_raw.copy()
for col in ["feature_date", "rebalance_date", "next_date", "next_date_5d", "next_date_10d"]:
    if col in panel.columns:
        panel[col] = pd.to_datetime(panel[col], errors="coerce").dt.normalize()

panel = panel[panel["feature_date"] >= pd.Timestamp("2021-01-01")].copy()
if "money_mean_20" in panel.columns:
    panel = panel[panel["money_mean_20"] >= MIN_AVG_MONEY_20].copy()

if "future_ret_5d" not in panel.columns:
    raise ValueError("future_ret_5d is required for V14")

if "target_alpha_5d" not in panel.columns:
    panel["target_alpha_5d"] = panel["future_ret_5d"] - panel.groupby("feature_date")["future_ret_5d"].transform("median")

panel["target_rank_5d"] = panel.groupby("feature_date")["future_ret_5d"].rank(pct=True)

if "future_ret_10d" in panel.columns:
    if "target_alpha_10d" not in panel.columns:
        panel["target_alpha_10d"] = panel["future_ret_10d"] - panel.groupby("feature_date")["future_ret_10d"].transform("median")
    panel["target_rank_10d"] = panel.groupby("feature_date")["future_ret_10d"].rank(pct=True)

available_features = [c for c in unique_keep_order(FEATURE_COLS) if c in panel.columns]
missing_features = [c for c in unique_keep_order(FEATURE_COLS) if c not in panel.columns]
print("panel after filters", panel.shape)
print("feature count", len(available_features), "missing", len(missing_features))

label_specs = [
    {"label_name": "ret5d", "target_col": "future_ret_5d", "eval_ret_col": "future_ret_5d", "next_col": "next_date", "horizon_days": 5, "rebalance_interval_weeks": 1},
    {"label_name": "alpha5d", "target_col": "target_alpha_5d", "eval_ret_col": "future_ret_5d", "next_col": "next_date", "horizon_days": 5, "rebalance_interval_weeks": 1},
    {"label_name": "rank5d", "target_col": "target_rank_5d", "eval_ret_col": "future_ret_5d", "next_col": "next_date", "horizon_days": 5, "rebalance_interval_weeks": 1},
]
if "future_ret_10d" in panel.columns:
    next_10d_col = "next_date_10d" if "next_date_10d" in panel.columns else "next_date"
    label_specs.extend([
        {"label_name": "ret10d", "target_col": "future_ret_10d", "eval_ret_col": "future_ret_10d", "next_col": next_10d_col, "horizon_days": 10, "rebalance_interval_weeks": 2},
        {"label_name": "alpha10d", "target_col": "target_alpha_10d", "eval_ret_col": "future_ret_10d", "next_col": next_10d_col, "horizon_days": 10, "rebalance_interval_weeks": 2},
        {"label_name": "rank10d", "target_col": "target_rank_10d", "eval_ret_col": "future_ret_10d", "next_col": next_10d_col, "horizon_days": 10, "rebalance_interval_weeks": 2},
    ])

blocked_labels = []
if "future_ret_10d" not in panel.columns:
    for missing_label in ["ret10d", "alpha10d", "rank10d"]:
        blocked_labels.append({"label_name": missing_label, "status": "BLOCKED_MISSING_DATA", "detail": "future_ret_10d not found; rebuild panel with LABEL_HORIZONS=[5, 10]."})

display(pd.DataFrame(label_specs))
if blocked_labels:
    display(pd.DataFrame(blocked_labels))
'''
))

cells.append(code_cell(
    r'''
# =========================
# 2. Model Backends
# =========================
class NumpyRidgeModel(object):
    def __init__(self, coef, mean, scale, feature_cols):
        self.coef = coef
        self.mean = mean
        self.scale = scale
        self.feature_cols = feature_cols

    def predict(self, X):
        arr = np.asarray(X[self.feature_cols].values, dtype=float)
        arr = (arr - self.mean) / self.scale
        arr = np.column_stack([np.ones(len(arr)), arr])
        return arr.dot(self.coef)


def fit_numpy_ridge(X, y, feature_cols, alpha=10.0):
    arr = np.asarray(X[feature_cols].values, dtype=float)
    mean = np.nanmean(arr, axis=0)
    arr = np.where(np.isnan(arr), mean, arr)
    scale = np.nanstd(arr, axis=0)
    scale = np.where((scale == 0) | np.isnan(scale), 1.0, scale)
    arr = (arr - mean) / scale
    arr = np.column_stack([np.ones(len(arr)), arr])
    y = np.asarray(y, dtype=float)
    eye = np.eye(arr.shape[1])
    eye[0, 0] = 0.0
    coef = np.linalg.solve(arr.T.dot(arr) + alpha * eye, arr.T.dot(y))
    return NumpyRidgeModel(coef, mean, scale, feature_cols)


def clean_train_df(df, target_col, eval_ret_col, next_col):
    cols = unique_keep_order(["code", "feature_date", next_col, target_col, eval_ret_col] + available_features)
    cols = [c for c in cols if c in df.columns]
    out = df.loc[:, cols].replace([np.inf, -np.inf], np.nan)
    out = out[~out[target_col].isnull()].copy()
    return out


def prepare_X(df):
    X_raw = df.reindex(columns=available_features).replace([np.inf, -np.inf], np.nan)
    fill_values = X_raw.median().to_dict()
    X = X_raw.fillna(pd.Series(fill_values)).fillna(0).astype(np.float32)
    return X, fill_values


def train_model(train_df, target_col):
    X, fill_values = prepare_X(train_df)
    y = train_df[target_col].astype(np.float32).values
    try:
        import lightgbm as lgb
        dtrain = lgb.Dataset(X[available_features], label=y, feature_name=list(available_features), free_raw_data=True)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        backend = "lightgbm"
    except Exception as err:
        print("LightGBM unavailable or failed; using numpy_ridge smoke-test backend:", err)
        model = fit_numpy_ridge(X, y, available_features, alpha=10.0)
        backend = "numpy_ridge"
    return model, fill_values, backend


def predict_model(model, fill_values, df, chunk_size=20000):
    preds = []
    for start in range(0, len(df), chunk_size):
        part = df.iloc[start:start + chunk_size]
        X_raw = part.reindex(columns=available_features).replace([np.inf, -np.inf], np.nan)
        X = X_raw.fillna(pd.Series(fill_values)).fillna(0).astype(np.float32)
        preds.append(np.asarray(model.predict(X[available_features])).reshape(-1))
    return np.concatenate(preds) if preds else np.asarray([])
'''
))

cells.append(code_cell(
    r'''
# =========================
# 3. Train and Score Label Experiments
# =========================
score_parts = []
manifest_rows = []

for spec in label_specs:
    label_name = spec["label_name"]
    target_col = spec["target_col"]
    eval_ret_col = spec["eval_ret_col"]
    next_col = spec["next_col"]
    horizon_days = int(spec["horizon_days"])
    rebalance_interval_weeks = int(spec["rebalance_interval_weeks"])
    if target_col not in panel.columns:
        blocked_labels.append({"label_name": label_name, "status": "BLOCKED_MISSING_DATA", "detail": "%s missing" % target_col})
        continue
    if eval_ret_col not in panel.columns:
        blocked_labels.append({"label_name": label_name, "status": "BLOCKED_MISSING_DATA", "detail": "%s missing" % eval_ret_col})
        continue
    if next_col not in panel.columns:
        next_col = "next_date"
    for train_tag, train_start, train_end in TRAIN_WINDOWS:
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        train_mask = (panel["feature_date"] >= train_start_ts) & (panel[next_col] <= train_end_ts)
        score_mask = panel["feature_date"] > train_end_ts
        train_df = clean_train_df(panel.loc[train_mask], target_col, eval_ret_col, next_col)
        score_base_cols = unique_keep_order(["code", "feature_date", next_col, eval_ret_col, target_col] + available_features)
        score_base_cols = [c for c in score_base_cols if c in panel.columns]
        score_base = panel.loc[score_mask, score_base_cols].copy()
        if train_df.empty or score_base.empty:
            manifest_rows.append({
                "label_name": label_name,
                "target_col": target_col,
                "eval_ret_col": eval_ret_col,
                "horizon_days": horizon_days,
                "rebalance_interval_weeks": rebalance_interval_weeks,
                "train_tag": train_tag,
                "status": "SKIPPED_EMPTY",
                "train_samples": int(len(train_df)),
                "score_samples": int(len(score_base)),
            })
            continue
        print("training", label_name, train_tag, "train", train_df.shape, "score", score_base.shape)
        model, fill_values, backend = train_model(train_df, target_col)
        scored = score_base[["code", "feature_date", next_col, eval_ret_col]].copy()
        if next_col != "next_date":
            scored = scored.rename(columns={next_col: "next_date"})
        scored["eval_ret"] = score_base[eval_ret_col].values
        scored["score"] = predict_model(model, fill_values, score_base)
        scored["label_name"] = label_name
        scored["target_col"] = target_col
        scored["eval_ret_col"] = eval_ret_col
        scored["horizon_days"] = horizon_days
        scored["rebalance_interval_weeks"] = rebalance_interval_weeks
        scored["train_tag"] = train_tag
        scored["train_start"] = train_start
        scored["train_end"] = train_end
        scored["model_name"] = "%s_%s" % (label_name, train_tag)
        scored["backend"] = backend
        score_parts.append(scored)
        manifest_rows.append({
            "label_name": label_name,
            "target_col": target_col,
            "eval_ret_col": eval_ret_col,
            "horizon_days": horizon_days,
            "rebalance_interval_weeks": rebalance_interval_weeks,
            "train_tag": train_tag,
            "backend": backend,
            "status": "TRAINED",
            "train_start": train_start,
            "train_end": train_end,
            "train_samples": int(len(train_df)),
            "train_weeks": int(train_df["feature_date"].nunique()),
            "score_samples": int(len(scored)),
            "score_weeks": int(scored["feature_date"].nunique()),
            "feature_count": int(len(available_features)),
        })

score_df = pd.concat(score_parts, ignore_index=True, sort=False) if score_parts else pd.DataFrame()
manifest_df = pd.DataFrame(manifest_rows)
blocked_labels_df = pd.DataFrame(blocked_labels)

save_csv(score_df, "etf_ml_v14_score_panel.csv")
save_csv(manifest_df, "etf_ml_v14_model_manifest.csv")
save_csv(blocked_labels_df, "etf_ml_v14_blocked_labels.csv")
display(manifest_df)
display(blocked_labels_df)
'''
))

cells.append(code_cell(
    r'''
# =========================
# 4. Top3 Matched-Holding Evaluation
# =========================
def baseline_ret(gdf, col, eval_ret_col):
    if col not in gdf.columns or eval_ret_col not in gdf.columns:
        return np.nan
    x = gdf.dropna(subset=[col, eval_ret_col]).copy()
    if x.empty:
        return np.nan
    return float(x.sort_values(col, ascending=False).head(min(TOP_N, len(x)))[eval_ret_col].mean())


def scheduled_score_data(score_data):
    parts = []
    if score_data.empty:
        return score_data.copy()
    for model_name, gdf0 in score_data.groupby("model_name"):
        interval = int(gdf0["rebalance_interval_weeks"].iloc[0]) if "rebalance_interval_weeks" in gdf0.columns else 1
        interval = max(1, interval)
        dates = sorted(pd.to_datetime(gdf0["feature_date"]).dropna().unique())
        keep_dates = set(pd.Timestamp(dt).normalize() for idx, dt in enumerate(dates) if idx % interval == 0)
        gdf = gdf0.copy()
        feature_dates = pd.to_datetime(gdf["feature_date"]).dt.normalize()
        parts.append(gdf.loc[feature_dates.isin(keep_dates)].copy())
    return pd.concat(parts, ignore_index=True, sort=False) if parts else score_data.iloc[0:0].copy()


score_eval_df = scheduled_score_data(score_df)
weekly_rows = []
if not score_eval_df.empty:
    feature_baseline_cols = ["code", "feature_date", "future_ret_5d", "future_ret_10d", "ret_20", "ret_60", "trend_score_25", "trend_score_60"]
    feature_baseline_cols = [c for c in feature_baseline_cols if c in panel.columns]
    panel_eval = panel[feature_baseline_cols].copy()
    for (model_name, label_name, train_tag, feature_date), gdf0 in score_eval_df.groupby(["model_name", "label_name", "train_tag", "feature_date"]):
        eval_ret_col = gdf0["eval_ret_col"].iloc[0]
        horizon_days = int(gdf0["horizon_days"].iloc[0])
        rebalance_interval_weeks = int(gdf0["rebalance_interval_weeks"].iloc[0])
        gdf = gdf0.dropna(subset=["score", "eval_ret"]).sort_values("score", ascending=False)
        if gdf.empty:
            continue
        top = gdf.head(min(TOP_N, len(gdf)))
        codes = top["code"].astype(str).tolist()
        universe = gdf
        vals = []
        n = min(TOP_N, len(universe))
        for seed in RANDOM_SEEDS:
            vals.append(float(universe.sample(n, random_state=seed)["eval_ret"].mean()))
        base_week = panel_eval[panel_eval["feature_date"] == feature_date]
        weekly_rows.append({
            "model_name": model_name,
            "label_name": label_name,
            "train_tag": train_tag,
            "backend": top["backend"].iloc[0],
            "feature_date": feature_date,
            "top_n": TOP_N,
            "horizon_days": horizon_days,
            "rebalance_interval_weeks": rebalance_interval_weeks,
            "eval_ret_col": eval_ret_col,
            "gross_ret": float(top["eval_ret"].mean()),
            "median_ret": float(universe["eval_ret"].median()),
            "equal_weight_ret": float(universe["eval_ret"].mean()),
            "random_mean_ret": float(np.mean(vals)),
            "random_median_ret": float(np.median(vals)),
            "ret20_top3_ret": baseline_ret(base_week, "ret_20", eval_ret_col),
            "ret60_top3_ret": baseline_ret(base_week, "ret_60", eval_ret_col),
            "trend25_top3_ret": baseline_ret(base_week, "trend_score_25", eval_ret_col),
            "trend60_top3_ret": baseline_ret(base_week, "trend_score_60", eval_ret_col),
            "targets": "|".join(codes),
            "target_count": int(len(codes)),
        })

weekly_gross_df = pd.DataFrame(weekly_rows).sort_values(["label_name", "train_tag", "feature_date"]) if weekly_rows else pd.DataFrame()

cost_parts = []
for model_name, gdf0 in weekly_gross_df.groupby("model_name"):
    gdf = gdf0.sort_values("feature_date").copy()
    prev = {}
    rows = []
    for _, r in gdf.iterrows():
        codes = [c for c in str(r["targets"]).split("|") if c]
        new = target_weights(codes)
        traded = calc_turnover(prev, new)
        prev = new
        rec = r.to_dict()
        rec["turnover_traded_notional"] = traded
        rec["cost_rate"] = COST_RATE
        rec["cost_drag"] = traded * COST_RATE
        rec["net_ret"] = rec["gross_ret"] - rec["cost_drag"]
        rec["stress_net_ret"] = rec["gross_ret"] - traded * COST_STRESS_RATE
        rows.append(rec)
    cost_parts.append(pd.DataFrame(rows))
weekly_df = pd.concat(cost_parts, ignore_index=True, sort=False) if cost_parts else pd.DataFrame()

save_csv(weekly_gross_df, "etf_ml_v14_weekly_gross.csv")
save_csv(weekly_df, "etf_ml_v14_weekly_cost.csv")
display(weekly_df.head(20))
'''
))

cells.append(code_cell(
    r'''
# =========================
# 5. Summary, Common OOS, Decision Matrix
# =========================
summary_rows = []
for keys, gdf in weekly_df.groupby(["label_name", "train_tag", "model_name", "backend"]):
    label_name, train_tag, model_name, backend = keys
    rec = {"label_name": label_name, "train_tag": train_tag, "model_name": model_name, "backend": backend}
    rec["horizon_days"] = int(gdf["horizon_days"].iloc[0]) if "horizon_days" in gdf.columns else np.nan
    rec["rebalance_interval_weeks"] = int(gdf["rebalance_interval_weeks"].iloc[0]) if "rebalance_interval_weeks" in gdf.columns else np.nan
    rec["eval_ret_col"] = gdf["eval_ret_col"].iloc[0] if "eval_ret_col" in gdf.columns else ""
    rec.update(add_summary("net", gdf["net_ret"]))
    rec.update(add_summary("gross", gdf["gross_ret"]))
    rec.update(add_summary("stress_net", gdf["stress_net_ret"]))
    rec.update(add_summary("median", gdf["median_ret"]))
    rec.update(add_summary("equal_weight", gdf["equal_weight_ret"]))
    rec.update(add_summary("random_mean", gdf["random_mean_ret"]))
    rec.update(add_summary("ret20", gdf["ret20_top3_ret"]))
    rec.update(add_summary("ret60", gdf["ret60_top3_ret"]))
    rec.update(add_summary("trend25", gdf["trend25_top3_ret"]))
    rec.update(add_summary("trend60", gdf["trend60_top3_ret"]))
    rec["avg_turnover"] = float(gdf["turnover_traded_notional"].mean())
    rec["net_excess_vs_random"] = rec["net_cum_ret"] - rec["random_mean_cum_ret"]
    rec["net_excess_vs_equal_weight"] = rec["net_cum_ret"] - rec["equal_weight_cum_ret"]
    rec["net_excess_vs_median"] = rec["net_cum_ret"] - rec["median_cum_ret"]
    rec["beats_all_simple"] = bool(
        rec["net_cum_ret"] > rec["ret20_cum_ret"] and
        rec["net_cum_ret"] > rec["ret60_cum_ret"] and
        rec["net_cum_ret"] > rec["trend25_cum_ret"] and
        rec["net_cum_ret"] > rec["trend60_cum_ret"]
    )
    summary_rows.append(rec)
summary_df = pd.DataFrame(summary_rows).sort_values(["net_cum_ret"], ascending=False) if summary_rows else pd.DataFrame()

common_rows = []
for start in COMMON_OOS_STARTS:
    sub = weekly_df[pd.to_datetime(weekly_df["feature_date"]) >= pd.Timestamp(start)].copy()
    for keys, gdf in sub.groupby(["label_name", "train_tag", "model_name", "backend"]):
        label_name, train_tag, model_name, backend = keys
        rec = {"common_oos_start": start, "label_name": label_name, "train_tag": train_tag, "model_name": model_name, "backend": backend, "weeks": int(len(gdf))}
        rec["horizon_days"] = int(gdf["horizon_days"].iloc[0]) if "horizon_days" in gdf.columns else np.nan
        rec["rebalance_interval_weeks"] = int(gdf["rebalance_interval_weeks"].iloc[0]) if "rebalance_interval_weeks" in gdf.columns else np.nan
        rec["eval_ret_col"] = gdf["eval_ret_col"].iloc[0] if "eval_ret_col" in gdf.columns else ""
        rec.update(add_summary("net", gdf["net_ret"]))
        rec.update(add_summary("stress_net", gdf["stress_net_ret"]))
        rec.update(add_summary("median", gdf["median_ret"]))
        rec.update(add_summary("equal_weight", gdf["equal_weight_ret"]))
        rec.update(add_summary("random_mean", gdf["random_mean_ret"]))
        rec["beats_random"] = bool(rec["net_cum_ret"] > rec["random_mean_cum_ret"])
        rec["beats_equal_weight"] = bool(rec["net_cum_ret"] > rec["equal_weight_cum_ret"])
        rec["beats_median"] = bool(rec["net_cum_ret"] > rec["median_cum_ret"])
        rec["positive_stress"] = bool(rec["stress_net_cum_ret"] > 0)
        common_rows.append(rec)
common_oos_df = pd.DataFrame(common_rows).sort_values(["common_oos_start", "net_cum_ret"], ascending=[True, False]) if common_rows else pd.DataFrame()

decision_rows = []
for label_name, gdf in summary_df.groupby("label_name"):
    common = common_oos_df[common_oos_df["label_name"] == label_name]
    decision_rows.append({
        "label_name": label_name,
        "trained_models": int(gdf["model_name"].nunique()),
        "horizon_days": int(gdf["horizon_days"].iloc[0]) if "horizon_days" in gdf.columns else np.nan,
        "rebalance_interval_weeks": int(gdf["rebalance_interval_weeks"].iloc[0]) if "rebalance_interval_weeks" in gdf.columns else np.nan,
        "eval_ret_col": gdf["eval_ret_col"].iloc[0] if "eval_ret_col" in gdf.columns else "",
        "best_net_cum_ret": float(gdf["net_cum_ret"].max()),
        "median_net_cum_ret": float(gdf["net_cum_ret"].median()),
        "worst_net_cum_ret": float(gdf["net_cum_ret"].min()),
        "best_max_drawdown": float(gdf.loc[gdf["net_cum_ret"].idxmax(), "net_max_drawdown"]),
        "beats_all_simple_rate": float(gdf["beats_all_simple"].mean()),
        "common_oos_pass_rate": float((common["beats_random"] & common["beats_equal_weight"] & common["beats_median"] & common["positive_stress"]).mean()) if len(common) else np.nan,
        "avg_turnover_median": float(gdf["avg_turnover"].median()),
    })
decision_df = pd.DataFrame(decision_rows).sort_values(["common_oos_pass_rate", "median_net_cum_ret"], ascending=[False, False]) if decision_rows else pd.DataFrame()

save_csv(summary_df, "etf_ml_v14_summary.csv")
save_csv(common_oos_df, "etf_ml_v14_common_oos.csv")
save_csv(decision_df, "etf_ml_v14_label_decision_matrix.csv")
display(summary_df.head(30))
display(decision_df)
'''
))

cells.append(code_cell(
    r'''
# =========================
# 6. Report
# =========================
def fmt_pct(x):
    if pd.isnull(x):
        return "nan"
    return "%.2f%%" % (100.0 * float(x))


backend_set = sorted(manifest_df.get("backend", pd.Series(dtype=str)).dropna().unique().tolist()) if not manifest_df.empty else []
final_status = "LABEL_EXPERIMENT_COMPLETE" if not decision_df.empty else "LABEL_EXPERIMENT_FAILED"

lines = []
lines.append("# ETF V14 Label Experiment Report")
lines.append("")
lines.append("Final status: **%s**" % final_status)
lines.append("")
lines.append("## Scope Guardrails")
lines.append("- Fixed topN: `%d`" % TOP_N)
lines.append("- Fixed features: V10 feature set, available feature count `%d`" % len(available_features))
lines.append("- Fixed cost: base `%s`, stress `%s`" % (COST_RATE, COST_STRESS_RATE))
lines.append("- Backends used: `%s`" % ", ".join(backend_set))
lines.append("- This experiment changes only training labels.")
lines.append("- Holding-period match: `5d` labels use weekly rebalance and `future_ret_5d`; `10d` labels use 2-week rebalance and `future_ret_10d`.")
lines.append("")
if blocked_labels:
    lines.append("## Blocked Labels")
    for r in blocked_labels:
        lines.append("- `%s`: `%s` - %s" % (r["label_name"], r["status"], r["detail"]))
    lines.append("")
lines.append("## Label Decision Matrix")
if not decision_df.empty:
    for _, r in decision_df.iterrows():
        lines.append("- `%s`: horizon=%sd rebalance=%sw eval=%s best=%s median=%s worst=%s common_pass=%.2f simple_pass=%.2f median_turnover=%.3f" % (
            r["label_name"],
            int(r["horizon_days"]) if not pd.isnull(r["horizon_days"]) else -1,
            int(r["rebalance_interval_weeks"]) if not pd.isnull(r["rebalance_interval_weeks"]) else -1,
            r["eval_ret_col"],
            fmt_pct(r["best_net_cum_ret"]),
            fmt_pct(r["median_net_cum_ret"]),
            fmt_pct(r["worst_net_cum_ret"]),
            float(r["common_oos_pass_rate"]),
            float(r["beats_all_simple_rate"]),
            float(r["avg_turnover_median"]),
        ))
else:
    lines.append("- No trained labels.")
lines.append("")
lines.append("## Best Rows")
if not summary_df.empty:
    for _, r in summary_df.head(15).iterrows():
        lines.append("- `%s` `%s`: horizon=%sd rebalance=%sw net=%s mdd=%s random=%s equal=%s stress=%s turnover=%.3f backend=%s" % (
            r["label_name"], r["train_tag"], int(r["horizon_days"]), int(r["rebalance_interval_weeks"]), fmt_pct(r["net_cum_ret"]), fmt_pct(r["net_max_drawdown"]),
            fmt_pct(r["random_mean_cum_ret"]), fmt_pct(r["equal_weight_cum_ret"]),
            fmt_pct(r["stress_net_cum_ret"]), float(r["avg_turnover"]), r["backend"]
        ))
lines.append("")
lines.append("## Interpretation Rules")
lines.append("- A label is valuable if it improves median/common-OOS behavior, not merely one best row.")
lines.append("- If a label underperforms `ret5d`, that is useful evidence to avoid adding that objective to the production ensemble.")
lines.append("- If `rank5d` is steadier but lower-return, use it as an ensemble stabilizer rather than the sole target.")
lines.append("- `ret10d`/`alpha10d`/`rank10d` remain untested until the weekly panel is rebuilt with 10-day labels; once available, they are evaluated as 2-week holding strategies.")

report_text = "\n".join(lines) + "\n"
report_path = os.path.join(OUT_DIR, "etf_ml_v14_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
try:
    with open("etf_ml_v14_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
except Exception:
    pass

print(report_text)
print("V14 report:", report_path)
'''
))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
OUT_NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT_NOTEBOOK)
print("cells", len(cells))
