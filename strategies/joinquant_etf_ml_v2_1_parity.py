"""ETF V2.1 parity backtest.

Signal uses the previous Friday's data. Monday execution is equal-weighted
across the final tradeable candidates, matching the V2.2 offline simulator.
"""

from jqdata import *
import datetime
import pickle
import numpy as np
import pandas as pd


MODEL_FILE = "model_etf_ml_v2_1_lgb_ret5d_train20210101_20241231_money2000w_baseline.pkl"
STOCK_NUM_OVERRIDE = None
TOP_CANDIDATE_LOG_N = 10
MIN_ORDER_LOT = 100


def initialize(context):
    g.model_file = MODEL_FILE
    bundle = pickle.loads(read_file(g.model_file))
    validate_bundle(bundle)

    g.model = bundle["model"]
    g.feature_cols = list(bundle["feature_cols"])
    g.raw_feature_cols = list(bundle.get("raw_feature_cols", []))
    g.rank_feature_cols = list(bundle.get("rank_feature_cols", []))
    g.context_feature_cols = list(bundle.get("context_feature_cols", []))
    g.fill_values = dict(bundle.get("fill_values", {}))
    g.lookback_days = int(bundle.get("lookback_days", 60))
    g.trend_windows = list(bundle.get("trend_windows", [10, 20, 25, 60]))
    g.trend_weight_end = float(bundle.get("trend_weight_end", 2.0))
    g.price_history_count = int(bundle.get(
        "price_history_count",
        max(g.lookback_days + 1, max(g.trend_windows) + 1 if len(g.trend_windows) else g.lookback_days + 1)
    ))
    g.pool_context_window = int(bundle.get("pool_context_window", 25))
    g.min_listing_days = int(bundle.get("min_listing_days", 180))
    g.min_avg_money_20 = float(bundle.get("min_avg_money_20", 20000000.0))
    #g.min_avg_money_20 = 50000000
    g.stock_num = int(bundle.get("stock_num", 3))
    if STOCK_NUM_OVERRIDE is not None:
        g.stock_num = int(STOCK_NUM_OVERRIDE)
    g.benchmark = bundle.get("benchmark", "000985.XSHG")
    g.exclude_name_keywords = list(bundle.get("exclude_name_keywords", []))
    g.volume_ab_mode = bundle.get("volume_ab_mode", "baseline")

    g.hold_list = []
    g.target_list = []
    g.target_list_date = None

    set_benchmark(g.benchmark)
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0.001))
    try:
        set_order_cost(OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0001,
            close_commission=0.0001,
            close_today_commission=0,
            min_commission=0
        ), type="fund")
    except Exception:
        set_order_cost(OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0001,
            close_commission=0.0001,
            close_today_commission=0,
            min_commission=0
        ), type="stock")
    log.set_level("order", "error")

    log.info(
        "loaded ETF ML model file=%s objective=%s target=%s features=%s raw=%s rank=%s "
        "lookback=%s history_count=%s stock_num=%s min_money20=%.0f mode=%s benchmark=%s" % (
            g.model_file,
            bundle.get("objective", ""),
            bundle.get("target_col", ""),
            len(g.feature_cols),
            len(g.raw_feature_cols),
            len(g.rank_feature_cols),
            g.lookback_days,
            g.price_history_count,
            g.stock_num,
            g.min_avg_money_20,
            g.volume_ab_mode,
            g.benchmark,
        )
    )
    log.info("feature cols: %s" % ",".join(g.feature_cols))

    run_daily(prepare_hold_list, "9:05")
    run_weekly(weekly_rebalance, 1, "9:35")


def validate_bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError("model bundle should be dict")
    for key in ["model", "feature_cols", "fill_values"]:
        if key not in bundle:
            raise ValueError("model bundle missing key: %s" % key)
    if len(bundle.get("feature_cols", [])) == 0:
        raise ValueError("model bundle feature_cols is empty")


def prepare_hold_list(context):
    g.hold_list = [
        position.security
        for position in context.portfolio.positions.values()
        if position.total_amount > 0
    ]


def weekly_rebalance(context):
    ranked_list = get_ranked_candidates(context)
    target_list, skipped = select_tradeable_targets(context, ranked_list)
    g.target_list = list(target_list)
    g.target_list_date = context.current_dt.date()
    log.info("ETF_ML_TARGETS signal_date=%s ranked=%s targets=%s skipped=%s" % (
        str(context.previous_date), ",".join(ranked_list[:TOP_CANDIDATE_LOG_N]),
        ",".join(target_list), ";".join(skipped)
    ))

    if len(target_list) == 0:
        log.warn("ETF_ML_TARGETS empty after execution filters")
        return

    for stock in list(g.hold_list):
        if stock not in target_list and stock in context.portfolio.positions:
            close_position(context.portfolio.positions[stock])

    # The research proxy assumes each selected ETF is reset to the same weight.
    target_value = context.portfolio.total_value / float(len(target_list))
    for stock in target_list:
        order = order_target_value(stock, target_value)
        if order is None:
            log.warn("ETF_ML_ORDER no order returned: %s" % stock)


def get_ranked_candidates(context):
    feature_date = context.previous_date
    etfs = get_etf_universe(feature_date)
    if len(etfs) == 0:
        return []

    feature_df = build_feature_panel(etfs, feature_date)
    if feature_df is None or feature_df.empty:
        return []

    score_df = score_feature_panel(feature_df)
    if score_df.empty:
        return []

    top_log = score_df.head(TOP_CANDIDATE_LOG_N)
    log.info("ETF_ML_TOP score snapshot: %s" % format_top_scores(top_log))
    return score_df.index.tolist()


def select_tradeable_targets(context, ranked_list):
    selected = []
    skipped = []
    if len(ranked_list) == 0:
        return selected, skipped

    target_value = context.portfolio.total_value / float(max(1, g.stock_num))
    try:
        current_data = get_current_data()
    except Exception as err:
        log.warn("current_data unavailable: %s" % err)
        return selected, ["current_data_unavailable"]

    for stock in ranked_list:
        if len(selected) >= g.stock_num:
            break
        try:
            data = current_data[stock]
            price = float(data.last_price)
            if data.paused or pd.isnull(price) or price <= 0:
                skipped.append(stock + ":paused_or_no_price")
                continue
            if target_value < price * MIN_ORDER_LOT:
                skipped.append(stock + ":min_lot")
                continue
            selected.append(stock)
        except Exception:
            skipped.append(stock + ":quote_error")
    return selected, skipped


def get_etf_universe(date):
    try:
        sec_df = get_all_securities(["etf"], date=date)
    except Exception as err:
        log.warn("get_all_securities etf failed: %s" % err)
        return []
    if sec_df is None or sec_df.empty:
        return []

    out = []
    current_data = get_current_data()
    for code, row in sec_df.iterrows():
        try:
            start_date = row.get("start_date", None)
            if pd.isnull(start_date):
                start_date = get_security_info(code).start_date
            start_date = pd.Timestamp(start_date).date()
            feature_date = pd.Timestamp(date).date()
            if feature_date - start_date < datetime.timedelta(days=g.min_listing_days):
                continue
            name = str(row.get("display_name", "")) or str(get_security_info(code).display_name)
            if should_exclude_name(name):
                continue
            try:
                if current_data[code].paused:
                    continue
            except Exception:
                pass
            out.append(code)
        except Exception:
            continue
    return out


def should_exclude_name(name):
    for kw in g.exclude_name_keywords:
        if kw and kw in name:
            return True
    return False


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_feature_panel(etfs, date):
    rows = []
    fields = ["open", "high", "low", "close", "volume", "money"]
    for etf_chunk in chunks(etfs, 120):
        try:
            price_df = get_price(
                etf_chunk,
                end_date=date,
                frequency="daily",
                fields=fields,
                count=g.price_history_count,
                panel=False,
                fq="pre",
                skip_paused=False
            )
        except Exception as err:
            log.warn("get_price feature chunk failed: %s" % err)
            continue
        if price_df is None or price_df.empty:
            continue
        for code, one in price_df.groupby("code"):
            rec = calc_one_etf_features(code, one)
            if rec is not None:
                rows.append(rec)
    if len(rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("code")
    df = df.replace([np.inf, -np.inf], np.nan)
    if "money_mean_20" in df.columns:
        df = df[df["money_mean_20"] >= g.min_avg_money_20]
    df = add_pool_context_features(df)
    df = add_rank_features(df)
    df = add_bundle_derived_features(df)
    return df


def calc_one_etf_features(code, price_df):
    df = price_df.sort_values("time").copy()
    if len(df) < max(40, int(g.lookback_days * 0.8)):
        return None
    for col in ["open", "high", "low", "close", "volume", "money"]:
        if col not in df.columns:
            return None
    close = pd.Series(df["close"].astype(float).values)
    high = pd.Series(df["high"].astype(float).values)
    low = pd.Series(df["low"].astype(float).values)
    volume = pd.Series(df["volume"].astype(float).values)
    money = pd.Series(df["money"].astype(float).values)
    if close.isnull().any() or close.iloc[-1] <= 0:
        return None

    ret = close.pct_change()
    rec = {"code": code}
    rec["ret_1"] = safe_ret(close, 1)
    rec["ret_5"] = safe_ret(close, 5)
    rec["ret_10"] = safe_ret(close, 10)
    rec["ret_20"] = safe_ret(close, 20)
    rec["ret_60"] = safe_ret(close, 60)
    rec["vol_5"] = ret.tail(5).std()
    rec["vol_20"] = ret.tail(20).std()
    rec["vol_60"] = ret.tail(60).std()
    rec["close_to_ma20"] = safe_ratio(close.iloc[-1], close.tail(20).mean()) - 1
    rec["close_to_ma60"] = safe_ratio(close.iloc[-1], close.tail(60).mean()) - 1
    rec["ma5_to_ma20"] = safe_ratio(close.tail(5).mean(), close.tail(20).mean()) - 1
    rec["ma20_to_ma60"] = safe_ratio(close.tail(20).mean(), close.tail(60).mean()) - 1
    rec["drawdown_20"] = safe_ratio(close.iloc[-1], close.tail(20).max()) - 1
    rec["drawdown_60"] = safe_ratio(close.iloc[-1], close.tail(60).max()) - 1
    rec["amp_20"] = (high.tail(20) / low.tail(20) - 1).replace([np.inf, -np.inf], np.nan).mean()
    rec["amp_60"] = (high.tail(60) / low.tail(60) - 1).replace([np.inf, -np.inf], np.nan).mean()
    rec["money_mean_20"] = money.tail(20).mean()
    rec["money_ratio_5_20"] = safe_ratio(money.tail(5).mean(), money.tail(20).mean()) - 1
    rec["money_ratio_20_60"] = safe_ratio(money.tail(20).mean(), money.tail(60).mean()) - 1
    rec["volume_ratio_5_20"] = safe_ratio(volume.tail(5).mean(), volume.tail(20).mean()) - 1
    rec["volume_ratio_20_60"] = safe_ratio(volume.tail(20).mean(), volume.tail(60).mean()) - 1
    rec["max_ret_20"] = ret.tail(20).max()
    rec["min_ret_20"] = ret.tail(20).min()
    for w in g.trend_windows:
        tm = calc_trend_metrics(close, int(w))
        rec["trend_ann_%s" % w] = tm["ann"]
        rec["trend_r2_%s" % w] = tm["r2"]
        rec["trend_score_%s" % w] = tm["score"]
        rec["trend_vol_%s" % w] = tm["vol"]
        rec["trend_score_vol_adj_%s" % w] = tm["score_vol_adj"]
        rec["trend_simple_ann_%s" % w] = tm["simple_ann"]
    return rec


def safe_ret(close, days):
    if len(close) <= days:
        return np.nan
    base = close.iloc[-days - 1]
    if pd.isnull(base) or base <= 0:
        return np.nan
    return close.iloc[-1] / base - 1


def safe_ratio(a, b):
    if pd.isnull(a) or pd.isnull(b) or b == 0:
        return np.nan
    return float(a) / float(b)


def calc_trend_metrics(close, days):
    if len(close) <= days:
        return {
            "ann": np.nan,
            "r2": np.nan,
            "score": np.nan,
            "vol": np.nan,
            "score_vol_adj": np.nan,
            "simple_ann": np.nan,
        }
    recent = pd.Series(close.iloc[-(days + 1):].astype(float).values)
    if recent.isnull().any() or (recent <= 0).any():
        return {
            "ann": np.nan,
            "r2": np.nan,
            "score": np.nan,
            "vol": np.nan,
            "score_vol_adj": np.nan,
            "simple_ann": np.nan,
        }
    y = np.log(recent.values)
    x = np.arange(len(y))
    weights = np.linspace(1.0, g.trend_weight_end, len(y))
    try:
        slope, intercept = np.polyfit(x, y, 1, w=weights)
    except Exception:
        return {
            "ann": np.nan,
            "r2": np.nan,
            "score": np.nan,
            "vol": np.nan,
            "score_vol_adj": np.nan,
            "simple_ann": np.nan,
        }
    ann = np.exp(slope * 250.0) - 1.0
    fit = slope * x + intercept
    ss_res = np.sum(weights * (y - fit) ** 2)
    ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    r2 = max(0.0, min(1.0, float(r2)))
    daily_returns = recent.pct_change().dropna()
    vol = float(daily_returns.std() * np.sqrt(250.0)) if len(daily_returns) > 1 else np.nan
    period_ret = recent.iloc[-1] / recent.iloc[0] - 1.0
    simple_ann = (1.0 + period_ret) ** (250.0 / float(days)) - 1.0 if 1.0 + period_ret > 0 else np.nan
    score = ann * r2
    score_vol_adj = score * 0.20 / max(vol, 0.05) if not pd.isnull(vol) else np.nan
    return {
        "ann": ann,
        "r2": r2,
        "score": score,
        "vol": vol,
        "score_vol_adj": score_vol_adj,
        "simple_ann": simple_ann,
    }


def add_pool_context_features(df):
    out = df.copy()
    if len(out) == 0:
        return out
    ann_col = "trend_ann_%s" % g.pool_context_window
    vol_col = "trend_vol_%s" % g.pool_context_window
    breadth_col = "pool_breadth_%s" % g.pool_context_window
    median_vol_col = "pool_median_vol_%s" % g.pool_context_window
    if ann_col in out.columns:
        out[breadth_col] = float((out[ann_col] > 0).mean())
    if vol_col in out.columns:
        out[median_vol_col] = float(out[vol_col].median())
    return out


def add_rank_features(df):
    out = df.copy()
    for col in g.raw_feature_cols:
        if col in out.columns:
            rank_col = "rank_" + col
            out[rank_col] = out[col].rank(pct=True)
    return out


def add_bundle_derived_features(df):
    out = df.copy()
    if "ret_5_x_rank_money_ratio_5_20" in g.feature_cols:
        out["ret_5_x_rank_money_ratio_5_20"] = (
            out["ret_5"] * out["rank_money_ratio_5_20"]
        )
    if "trend_score_20_x_rank_money_ratio_5_20" in g.feature_cols:
        out["trend_score_20_x_rank_money_ratio_5_20"] = (
            out["trend_score_20"] * out["rank_money_ratio_5_20"]
        )
    return out.replace([np.inf, -np.inf], np.nan)


def score_feature_panel(feature_df):
    missing = [col for col in g.feature_cols if col not in feature_df.columns]
    if missing:
        raise ValueError("required bundle features not generated: %s" % ",".join(missing))
    X = feature_df.reindex(columns=g.feature_cols).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(pd.Series(g.fill_values)).fillna(0)
    scores = np.asarray(g.model.predict(X[g.feature_cols])).reshape(-1)
    out = feature_df.copy()
    out["score"] = scores
    out = out.sort_values("score", ascending=False)
    return out


def format_top_scores(df):
    parts = []
    for code, row in df.iterrows():
        parts.append("%s=%.6f" % (code, float(row["score"])))
    return "; ".join(parts)


def close_position(position):
    if hasattr(position, "closeable_amount") and position.closeable_amount <= 0:
        return False
    order = order_target_value(position.security, 0)
    if order is not None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False
