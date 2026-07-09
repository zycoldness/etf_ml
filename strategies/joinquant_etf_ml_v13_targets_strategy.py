"""
JoinQuant ETF ML V13 target-schedule strategy template.

Usage:
1. Run ETF_V13_live_gate_validation.ipynb.
2. Open etf_ml_v13_joinquant_target_schedule.py.
3. Paste the generated TARGET_SCHEDULE dict below.
4. Backtest this strategy in JoinQuant.

Timing:
- TARGET_SCHEDULE keys are signal feature dates.
- The strategy rebalances at the next trading day's open by checking
  context.previous_date.
- This avoids using same-day close information for same-day trading.
"""


TARGET_SCHEDULE = {
    # Example:
    # "2026-06-18": {"513350.XSHG": 0.333333, "159217.XSHE": 0.333333, "159297.XSHE": 0.333333},
}

BENCHMARK = "000985.XSHG"
MAX_POSITION_WEIGHT = 0.40
MIN_ORDER_VALUE = 1000


def initialize(context):
    set_benchmark(BENCHMARK)
    set_option("use_real_price", True)
    log.set_level("order", "error")

    cost = OrderCost(
        open_tax=0,
        close_tax=0,
        open_commission=0.0001,
        close_commission=0.0001,
        close_today_commission=0,
        min_commission=0,
    )
    try:
        set_order_cost(cost, type="fund")
    except Exception:
        set_order_cost(cost, type="stock")

    run_daily(rebalance, time="open")


def _signal_key(context):
    return str(context.previous_date)


def _normalize_targets(raw_targets):
    clean = {}
    total = 0.0
    for code, weight in raw_targets.items():
        try:
            w = float(weight)
        except Exception:
            continue
        if w <= 0:
            continue
        w = min(w, MAX_POSITION_WEIGHT)
        clean[str(code)] = w
        total += w
    if total <= 0:
        return {}
    if total > 1.0:
        clean = {code: weight / total for code, weight in clean.items()}
    return clean


def _can_sell(code, current_data):
    info = current_data[code]
    if info.paused:
        return False
    return True


def _can_buy(code, current_data):
    info = current_data[code]
    if info.paused:
        return False
    if info.last_price >= info.high_limit:
        return False
    return True


def rebalance(context):
    key = _signal_key(context)
    if key not in TARGET_SCHEDULE:
        return

    targets = _normalize_targets(TARGET_SCHEDULE.get(key, {}))
    current_data = get_current_data()
    total_value = context.portfolio.total_value

    current_codes = set(context.portfolio.positions.keys())
    target_codes = set(targets.keys())

    for code in sorted(current_codes - target_codes):
        if code not in current_data:
            continue
        if _can_sell(code, current_data):
            order_target_value(code, 0)

    for code, weight in sorted(targets.items()):
        if code not in current_data:
            continue
        if not _can_buy(code, current_data):
            log.info("skip buy %s on %s: paused or limit-up" % (code, context.current_dt))
            continue
        target_value = total_value * weight
        if target_value < MIN_ORDER_VALUE:
            continue
        order_target_value(code, target_value)


def after_trading_end(context):
    key = _signal_key(context)
    if key in TARGET_SCHEDULE:
        log.info("rebalanced from signal date %s: %s" % (key, TARGET_SCHEDULE[key]))
