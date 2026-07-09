import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_NOTEBOOK = PROJECT_DIR / "notebooks" / "ETF_V13_live_gate_validation.ipynb"


def source(text):
    return text.strip("\n").splitlines(True)


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source(text)}


cells = []

cells.append(markdown_cell(
    """
# ETF_V13 Live Gate Validation

V13 closes the V12 blockers after `etf_ml_v11_score_panel.csv` and
`etf_ml_v11_weekly_panel.csv` are available.

It validates:

- ML score vs simple momentum/trend/equal-weight/random/median baselines
- cost and turnover under multiple cost assumptions
- selected ETF liquidity and capacity
- theme concentration when ETF names/groups are available
- implementable consensus and single-model portfolio rules
- exportable target schedules for a JoinQuant strategy backtest

Timing convention: signals are built on `feature_date`. A JoinQuant strategy
should rebalance on the next trading day using `context.previous_date` as the
signal date. Do not use same-day close to trade same-day open.
"""
))

cells.append(code_cell(
    r'''
# =========================
# 0. Config and IO
# =========================
try:
    from jqdata import *  # noqa: F401,F403
except Exception:
    print("jqdata unavailable locally; ETF names/groups will be enriched only when running in JoinQuant.")

import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

pd.set_option("display.max_columns", 180)
pd.set_option("display.width", 240)

OUT_DIR = "etf_ml_v13_live_gate_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_DIR_CANDIDATES = [
    "G:/",
    ".",
    "etf_ml_v11_2021_validation_outputs",
    "../etf_ml_v11_2021_validation_outputs",
]

COST_RATE_BASE = 0.0001
COST_RATE_GRID = [0.0001, 0.0003, 0.0005, 0.0010]
TOP_N_LIST = [1, 3, 5, 7]
CAPITAL_GRID = [1_000_000, 5_000_000, 10_000_000, 20_000_000]
MAX_MONEY20_PARTICIPATION = 0.05
RANDOM_BASELINE_SEEDS = list(range(50))
COMMON_OOS_STARTS = ["2024-01-01", "2025-01-01", "2026-01-01"]
PREFERRED_RULE = "consensus_score"
PREFERRED_TOP_N = 3


def find_input_file(fname):
    for base in INPUT_DIR_CANDIDATES:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            return path
    return None


def read_required(fname):
    path = find_input_file(fname)
    if path is None:
        raise FileNotFoundError("Missing required input: %s" % fname)
    df = pd.read_csv(path)
    print("loaded", fname, df.shape, "from", path)
    return df


def read_optional(fname):
    path = find_input_file(fname)
    if path is None:
        print("optional missing", fname)
        return None
    df = pd.read_csv(path)
    print("loaded", fname, df.shape, "from", path)
    return df


score_panel_raw = read_required("etf_ml_v11_score_panel.csv")
weekly_panel_raw = read_required("etf_ml_v11_weekly_panel.csv")
weekly_cost_raw = read_optional("etf_ml_v11_weekly_cost_proxy.csv")
latest_targets_raw = read_optional("etf_ml_v11_latest_targets.csv")
score_gap_raw = read_optional("etf_ml_v11_score_gap_diagnostics.csv")

print("V13 inputs ready.")
'''
))

cells.append(code_cell(
    r'''
# =========================
# 1. Helpers
# =========================
ETF_GROUP_RULES = [
    ("bank", ["\u94f6\u884c"]),
    ("innovative_drug", ["\u521b\u65b0\u836f", "\u65b0\u836f", "\u533b\u836f", "\u533b\u7597", "\u751f\u7269", "\u836f"]),
    ("oil_gas", ["\u6cb9\u6c14", "\u77f3\u6cb9", "\u80fd\u6e90", "\u5316\u5de5"]),
    ("coal", ["\u7164\u70ad"]),
    ("semiconductor", ["\u534a\u5bfc", "\u82af\u7247", "\u96c6\u6210\u7535\u8def"]),
    ("software_ai", ["\u8f6f\u4ef6", "\u4eba\u5de5\u667a\u80fd", "AI", "\u4e91\u8ba1\u7b97", "\u5927\u6570\u636e", "\u8ba1\u7b97\u673a", "\u4fe1\u521b", "\u6e38\u620f", "\u901a\u4fe1", "\u7535\u5b50"]),
    ("broker", ["\u8bc1\u5238", "\u5238\u5546"]),
    ("military", ["\u519b\u5de5", "\u56fd\u9632", "\u822a\u7a7a\u822a\u5929", "\u901a\u7528\u822a\u7a7a"]),
    ("gold", ["\u9ec4\u91d1"]),
    ("nonferrous", ["\u6709\u8272", "\u7a00\u6709\u91d1\u5c5e", "\u7a00\u571f"]),
    ("consumer", ["\u6d88\u8d39", "\u98df\u54c1", "\u9152"]),
    ("real_estate", ["\u5730\u4ea7", "\u623f\u5730\u4ea7"]),
    ("new_energy", ["\u65b0\u80fd\u6e90", "\u5149\u4f0f", "\u7535\u6c60", "\u9502\u7535", "\u50a8\u80fd", "\u79d1\u521b\u65b0\u80fd"]),
    ("cross_border", ["\u7eb3\u6307", "\u6807\u666e", "\u6052\u751f", "\u6e2f\u80a1", "\u65e5\u7ecf", "\u5fb7\u56fd", "\u6cd5\u56fd", "\u6d77\u5916", "QDII", "\u4e9a\u592a"]),
    ("broad", ["\u6caa\u6df1300", "\u4e2d\u8bc1500", "\u4e2d\u8bc11000", "\u521b\u4e1a\u677f", "\u79d1\u521b", "\u4e0a\u8bc150", "A500"]),
]


def save_csv(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    try:
        df.to_csv(name, index=False)
    except Exception:
        pass
    print("saved", path, df.shape)
    return path


def missing_or_blank(series):
    return series.isnull() | (series.astype(str).str.strip() == "")


def classify_etf_group(name):
    if name is None or pd.isnull(name) or str(name).strip() == "":
        return np.nan
    text = str(name)
    for group_name, keys in ETF_GROUP_RULES:
        for kw in keys:
            if kw and kw in text:
                return group_name
    return "other"


def try_get_jq_etf_names(date=None):
    try:
        sec = get_all_securities(["etf"], date=date)  # noqa: F405
    except Exception as err:
        print("JoinQuant ETF metadata unavailable:", err)
        return {}
    if sec is None or sec.empty:
        return {}
    return {code: str(row.get("display_name", "")) for code, row in sec.iterrows()}


def enrich_names(df, date=None):
    if df is None or df.empty or "code" not in df.columns:
        return df
    out = df.copy()
    if "name" not in out.columns:
        out["name"] = np.nan
    if "etf_group" not in out.columns:
        out["etf_group"] = np.nan
    needs_name = missing_or_blank(out["name"])
    if needs_name.any():
        name_map = try_get_jq_etf_names(date=date)
        if name_map:
            out.loc[needs_name, "name"] = out.loc[needs_name, "code"].map(name_map)
    needs_group = missing_or_blank(out["etf_group"])
    out.loc[needs_group, "etf_group"] = out.loc[needs_group, "name"].map(classify_etf_group)
    return out


def normalize_dates(df, cols=("feature_date", "rebalance_date", "next_date", "next_date_5d")):
    if df is None:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.normalize()
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


def parse_targets(x):
    if pd.isnull(x):
        return []
    return [z for z in str(x).split("|") if z and z != "nan"]


def target_weights(codes):
    codes = [c for c in codes if c]
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {c: w for c in codes}


def turnover(prev_w, new_w):
    keys = set(prev_w.keys()) | set(new_w.keys())
    traded = float(sum(abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys))
    return traded, traded / 2.0


def zscore(s):
    x = pd.Series(s).astype(float)
    std = x.std()
    if not std or pd.isnull(std):
        return x * 0.0
    return (x - x.mean()) / std


print("helpers ready")
'''
))

cells.append(code_cell(
    r'''
# =========================
# 2. Build Enriched Score Panel
# =========================
score_panel = normalize_dates(score_panel_raw)
weekly_panel = normalize_dates(weekly_panel_raw)

merge_cols = [
    "code", "feature_date", "money_mean_20", "ret_20", "ret_60",
    "trend_score_25", "trend_score_60", "rank_money_mean_20",
]
merge_cols = [c for c in merge_cols if c in weekly_panel.columns]
feature_merge = weekly_panel[merge_cols].drop_duplicates(["code", "feature_date"]).copy()

for c in ["money_mean_20", "ret_20", "ret_60", "trend_score_25", "trend_score_60", "rank_money_mean_20"]:
    if c in score_panel.columns:
        score_panel = score_panel.drop(columns=[c])

score_panel = score_panel.merge(feature_merge, on=["code", "feature_date"], how="left")
latest_date_for_meta = score_panel["feature_date"].max() if "feature_date" in score_panel.columns else None
score_panel = enrich_names(score_panel, date=latest_date_for_meta)

if "eval_ret" not in score_panel.columns and "future_ret_5d" in score_panel.columns:
    score_panel["eval_ret"] = score_panel["future_ret_5d"]

score_panel["score_rank_pct"] = score_panel.groupby(["model_name", "feature_date"])["score"].rank(pct=True)

data_quality_rows = []


def qcheck(check, status, detail, severity="INFO"):
    data_quality_rows.append({"check": check, "status": status, "severity": severity, "detail": detail})


qcheck("score_panel_rows", "PASS" if len(score_panel) > 0 else "FAIL", "rows=%d" % len(score_panel), "ERROR" if len(score_panel) == 0 else "INFO")
qcheck("weekly_panel_rows", "PASS" if len(weekly_panel) > 0 else "FAIL", "rows=%d" % len(weekly_panel), "ERROR" if len(weekly_panel) == 0 else "INFO")
for c in ["score", "eval_ret", "money_mean_20", "ret_20", "ret_60", "trend_score_25", "trend_score_60"]:
    status = "PASS" if c in score_panel.columns and score_panel[c].notna().sum() > 0 else "FAIL"
    qcheck("score_panel_%s" % c, status, "non_null=%d" % (score_panel[c].notna().sum() if c in score_panel.columns else 0), "ERROR" if status == "FAIL" else "INFO")

name_missing = float(missing_or_blank(score_panel["name"]).mean()) if "name" in score_panel.columns else 1.0
group_missing = float(missing_or_blank(score_panel["etf_group"]).mean()) if "etf_group" in score_panel.columns else 1.0
group_valid = score_panel.loc[~missing_or_blank(score_panel["etf_group"]), "etf_group"].astype(str) if "etf_group" in score_panel.columns else pd.Series(dtype=str)
qcheck("metadata_names", "PASS" if name_missing < 0.05 else "FAIL", "missing_ratio=%.3f" % name_missing, "ERROR" if name_missing >= 0.05 else "INFO")
qcheck("metadata_groups", "PASS" if group_missing < 0.10 else "FAIL", "missing_ratio=%.3f unique_groups=%d" % (group_missing, group_valid.nunique()), "ERROR" if group_missing >= 0.10 else "INFO")

data_quality_df = pd.DataFrame(data_quality_rows)
save_csv(score_panel, "etf_ml_v13_score_panel_enriched.csv")
save_csv(data_quality_df, "etf_ml_v13_data_quality.csv")
display(data_quality_df)
'''
))

cells.append(code_cell(
    r'''
# =========================
# 3. Weekly Rule Returns
# =========================
RULES = [
    ("ml_score", "score", False),
    ("ret_20", "ret_20", False),
    ("ret_60", "ret_60", False),
    ("trend_score_25", "trend_score_25", False),
    ("trend_score_60", "trend_score_60", False),
]


def select_by_col(gdf, score_col, top_n):
    if score_col not in gdf.columns:
        return pd.DataFrame()
    x = gdf.dropna(subset=[score_col, "eval_ret"]).copy()
    if x.empty:
        return x
    return x.sort_values(score_col, ascending=False).head(min(int(top_n), len(x)))


def row_from_selection(rule_name, model_name, feature_date, top_n, selected, universe, score_col):
    codes = selected["code"].astype(str).tolist()
    names = selected["name"].astype(str).tolist() if "name" in selected.columns else []
    groups = selected["etf_group"].astype(str).tolist() if "etf_group" in selected.columns else []
    money = selected["money_mean_20"].dropna().astype(float) if "money_mean_20" in selected.columns else pd.Series(dtype=float)
    rec = {
        "rule_name": rule_name,
        "model_name": model_name,
        "feature_date": feature_date,
        "top_n": int(top_n),
        "gross_ret": float(selected["eval_ret"].astype(float).mean()) if len(selected) else np.nan,
        "median_ret": float(universe["eval_ret"].median()) if len(universe) else np.nan,
        "equal_weight_ret": float(universe["eval_ret"].mean()) if len(universe) else np.nan,
        "target_count": int(len(codes)),
        "targets": "|".join(codes),
        "target_names": "|".join([x for x in names if x and x != "nan"]),
        "target_groups": "|".join([x for x in groups if x and x != "nan"]),
        "min_selected_money20": float(money.min()) if len(money) else np.nan,
        "mean_selected_money20": float(money.mean()) if len(money) else np.nan,
        "score_col": score_col,
    }
    if len(universe.dropna(subset=["eval_ret"])) > 0:
        vals = []
        n = min(int(top_n), len(universe))
        for seed in RANDOM_BASELINE_SEEDS:
            vals.append(float(universe.sample(n, random_state=seed)["eval_ret"].mean()))
        rec["random_mean_ret"] = float(np.mean(vals))
        rec["random_median_ret"] = float(np.median(vals))
    else:
        rec["random_mean_ret"] = np.nan
        rec["random_median_ret"] = np.nan
    if score_col == "score" and len(universe) >= 3:
        sx = universe.sort_values("score", ascending=False)
        rec["gap_1_2"] = float(sx.iloc[0]["score"] - sx.iloc[1]["score"])
        rec["gap_1_3"] = float(sx.iloc[0]["score"] - sx.iloc[2]["score"])
        rec["score_std_top10"] = float(sx.head(10)["score"].std())
    else:
        rec["gap_1_2"] = np.nan
        rec["gap_1_3"] = np.nan
        rec["score_std_top10"] = np.nan
    return rec


rows = []
for (model_name, feature_date), gdf0 in score_panel.groupby(["model_name", "feature_date"]):
    universe = gdf0.dropna(subset=["eval_ret"]).copy()
    if universe.empty:
        continue
    for top_n in TOP_N_LIST:
        for rule_name, score_col, _ in RULES:
            selected = select_by_col(universe, score_col, top_n)
            if not selected.empty:
                rows.append(row_from_selection(rule_name, model_name, feature_date, top_n, selected, universe, score_col))

weekly_rule_df = pd.DataFrame(rows)

# Implementable model consensus: average within-date z-scored model scores by ETF code.
consensus_rows = []
for feature_date, gdf0 in score_panel.groupby("feature_date"):
    x = gdf0.dropna(subset=["score", "eval_ret"]).copy()
    if x.empty:
        continue
    x["score_z"] = x.groupby("model_name")["score"].transform(zscore)
    agg_cols = {
        "score_z": "mean",
        "eval_ret": "first",
        "money_mean_20": "first",
        "ret_20": "first",
        "ret_60": "first",
        "trend_score_25": "first",
        "trend_score_60": "first",
        "name": "first",
        "etf_group": "first",
    }
    agg_cols = {k: v for k, v in agg_cols.items() if k in x.columns}
    uni = x.groupby(["code", "feature_date"], as_index=False).agg(agg_cols).rename(columns={"score_z": "consensus_score"})
    for top_n in TOP_N_LIST:
        selected = select_by_col(uni, "consensus_score", top_n)
        if not selected.empty:
            consensus_rows.append(row_from_selection("consensus_score", "consensus_models", feature_date, top_n, selected, uni, "consensus_score"))

consensus_weekly_df = pd.DataFrame(consensus_rows)
weekly_rule_df = pd.concat([weekly_rule_df, consensus_weekly_df], ignore_index=True, sort=False)
weekly_rule_df = weekly_rule_df.sort_values(["rule_name", "model_name", "top_n", "feature_date"]).reset_index(drop=True)

save_csv(weekly_rule_df, "etf_ml_v13_rule_weekly_gross.csv")
print(weekly_rule_df.shape)
display(weekly_rule_df.head(20))
'''
))

cells.append(code_cell(
    r'''
# =========================
# 4. Costs, Summaries, Robustness
# =========================
def apply_costs(df, cost_rate):
    out_parts = []
    for keys, gdf0 in df.groupby(["rule_name", "model_name", "top_n"]):
        gdf = gdf0.sort_values("feature_date").copy()
        prev = {}
        rows = []
        for _, r in gdf.iterrows():
            codes = parse_targets(r.get("targets", ""))
            new_w = target_weights(codes)
            traded, one_way = turnover(prev, new_w)
            prev = new_w
            rec = r.to_dict()
            rec["cost_rate"] = float(cost_rate)
            rec["turnover_traded_notional"] = traded
            rec["one_way_turnover"] = one_way
            rec["cost_drag"] = traded * float(cost_rate)
            rec["net_ret"] = float(rec["gross_ret"]) - rec["cost_drag"] if pd.notnull(rec["gross_ret"]) else np.nan
            rows.append(rec)
        out_parts.append(pd.DataFrame(rows))
    return pd.concat(out_parts, ignore_index=True, sort=False) if out_parts else pd.DataFrame()


def summarize_weekly(df):
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, gdf in df.groupby(["rule_name", "model_name", "top_n", "cost_rate"]):
        rule_name, model_name, top_n, cost_rate = keys
        rec = {"rule_name": rule_name, "model_name": model_name, "top_n": int(top_n), "cost_rate": float(cost_rate)}
        rec.update(add_summary("net", gdf["net_ret"]))
        rec.update(add_summary("gross", gdf["gross_ret"]))
        rec.update(add_summary("median", gdf["median_ret"]))
        rec.update(add_summary("equal_weight", gdf["equal_weight_ret"]))
        rec.update(add_summary("random_mean", gdf["random_mean_ret"]))
        rec["avg_turnover"] = float(gdf["turnover_traded_notional"].mean())
        rec["max_turnover"] = float(gdf["turnover_traded_notional"].max())
        rec["total_cost_drag"] = float(gdf["cost_drag"].sum())
        rec["net_excess_vs_median"] = rec["net_cum_ret"] - rec["median_cum_ret"]
        rec["net_excess_vs_equal_weight"] = rec["net_cum_ret"] - rec["equal_weight_cum_ret"]
        rec["net_excess_vs_random"] = rec["net_cum_ret"] - rec["random_mean_cum_ret"]
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(["cost_rate", "top_n", "net_cum_ret"], ascending=[True, True, False])


weekly_cost_df = apply_costs(weekly_rule_df, COST_RATE_BASE)
summary_df = summarize_weekly(weekly_cost_df)

robust_parts = [apply_costs(weekly_rule_df, cr) for cr in COST_RATE_GRID]
robust_weekly_df = pd.concat(robust_parts, ignore_index=True, sort=False)
robust_summary_df = summarize_weekly(robust_weekly_df)

save_csv(weekly_cost_df, "etf_ml_v13_rule_weekly_cost.csv")
save_csv(summary_df, "etf_ml_v13_rule_summary.csv")
save_csv(robust_summary_df, "etf_ml_v13_cost_robustness.csv")
display(summary_df.head(30))
'''
))

cells.append(code_cell(
    r'''
# =========================
# 5. Common OOS and Baseline Gate
# =========================
common_rows = []
for start in COMMON_OOS_STARTS:
    sub = weekly_cost_df[pd.to_datetime(weekly_cost_df["feature_date"]) >= pd.Timestamp(start)].copy()
    if sub.empty:
        continue
    for keys, gdf in sub.groupby(["rule_name", "model_name", "top_n"]):
        rule_name, model_name, top_n = keys
        rec = {"common_oos_start": start, "rule_name": rule_name, "model_name": model_name, "top_n": int(top_n), "weeks": int(len(gdf))}
        rec.update(add_summary("net", gdf["net_ret"]))
        rec.update(add_summary("median", gdf["median_ret"]))
        rec.update(add_summary("equal_weight", gdf["equal_weight_ret"]))
        rec.update(add_summary("random_mean", gdf["random_mean_ret"]))
        rec["beats_median"] = bool(rec["net_cum_ret"] > rec["median_cum_ret"]) if pd.notnull(rec["net_cum_ret"]) and pd.notnull(rec["median_cum_ret"]) else False
        rec["beats_equal_weight"] = bool(rec["net_cum_ret"] > rec["equal_weight_cum_ret"]) if pd.notnull(rec["net_cum_ret"]) and pd.notnull(rec["equal_weight_cum_ret"]) else False
        rec["beats_random"] = bool(rec["net_cum_ret"] > rec["random_mean_cum_ret"]) if pd.notnull(rec["net_cum_ret"]) and pd.notnull(rec["random_mean_cum_ret"]) else False
        common_rows.append(rec)

common_oos_df = pd.DataFrame(common_rows).sort_values(["common_oos_start", "top_n", "net_cum_ret"], ascending=[True, True, False])

baseline_compare_rows = []
base = summary_df[(summary_df["cost_rate"] == COST_RATE_BASE) & (summary_df["top_n"].isin([3, 5]))].copy()
for model_name in sorted(base["model_name"].dropna().unique()):
    model_part = base[base["model_name"] == model_name]
    for top_n in [3, 5]:
        ml = model_part[(model_part["rule_name"] == "ml_score") & (model_part["top_n"] == top_n)]
        if ml.empty:
            continue
        ml_ret = float(ml.iloc[0]["net_cum_ret"])
        for rule_name in ["ret_20", "ret_60", "trend_score_25", "trend_score_60"]:
            b = model_part[(model_part["rule_name"] == rule_name) & (model_part["top_n"] == top_n)]
            if b.empty:
                continue
            baseline_compare_rows.append({
                "model_name": model_name,
                "top_n": top_n,
                "baseline_rule": rule_name,
                "ml_net_cum_ret": ml_ret,
                "baseline_net_cum_ret": float(b.iloc[0]["net_cum_ret"]),
                "ml_beats_baseline": bool(ml_ret > float(b.iloc[0]["net_cum_ret"])),
            })

baseline_compare_df = pd.DataFrame(baseline_compare_rows)

save_csv(common_oos_df, "etf_ml_v13_common_oos.csv")
save_csv(baseline_compare_df, "etf_ml_v13_baseline_compare.csv")
display(common_oos_df.head(30))
display(baseline_compare_df)
'''
))

cells.append(code_cell(
    r'''
# =========================
# 6. Capacity and Theme Concentration
# =========================
capacity_rows = []
for _, r in weekly_cost_df.iterrows():
    top_n = int(r["top_n"])
    min_money = r.get("min_selected_money20", np.nan)
    mean_money = r.get("mean_selected_money20", np.nan)
    for capital in CAPITAL_GRID:
        pos_value = float(capital) / max(1, top_n)
        min_participation = pos_value / float(min_money) if pd.notnull(min_money) and min_money > 0 else np.nan
        mean_participation = pos_value / float(mean_money) if pd.notnull(mean_money) and mean_money > 0 else np.nan
        capacity_rows.append({
            "rule_name": r["rule_name"],
            "model_name": r["model_name"],
            "feature_date": r["feature_date"],
            "top_n": top_n,
            "capital": int(capital),
            "position_value": pos_value,
            "min_selected_money20": min_money,
            "mean_selected_money20": mean_money,
            "min_money20_participation": min_participation,
            "mean_money20_participation": mean_participation,
            "capacity_ok": bool(pd.notnull(min_participation) and min_participation <= MAX_MONEY20_PARTICIPATION),
        })

capacity_weekly_df = pd.DataFrame(capacity_rows)
capacity_summary_rows = []
for keys, gdf in capacity_weekly_df.groupby(["rule_name", "model_name", "top_n", "capital"]):
    rule_name, model_name, top_n, capital = keys
    capacity_summary_rows.append({
        "rule_name": rule_name,
        "model_name": model_name,
        "top_n": int(top_n),
        "capital": int(capital),
        "weeks": int(len(gdf)),
        "ok_rate": float(gdf["capacity_ok"].mean()),
        "max_min_money20_participation": float(gdf["min_money20_participation"].max()),
        "median_min_selected_money20": float(gdf["min_selected_money20"].median()),
    })
capacity_summary_df = pd.DataFrame(capacity_summary_rows).sort_values(["capital", "top_n", "ok_rate"], ascending=[True, True, False])

concentration_rows = []
for _, r in weekly_cost_df.iterrows():
    groups = [g for g in str(r.get("target_groups", "")).split("|") if g and g != "nan"]
    names = [n for n in str(r.get("target_names", "")).split("|") if n and n != "nan"]
    unique_groups = len(set(groups))
    max_group_share = max([groups.count(g) for g in set(groups)]) / float(len(groups)) if groups else np.nan
    concentration_rows.append({
        "rule_name": r["rule_name"],
        "model_name": r["model_name"],
        "feature_date": r["feature_date"],
        "top_n": int(r["top_n"]),
        "target_count": int(r["target_count"]),
        "target_names_available": bool(len(names) >= int(r["target_count"])),
        "target_groups_available": bool(len(groups) >= int(r["target_count"])),
        "unique_groups": unique_groups,
        "max_group_share": max_group_share,
        "target_groups": r.get("target_groups", ""),
        "target_names": r.get("target_names", ""),
    })
concentration_df = pd.DataFrame(concentration_rows)

save_csv(capacity_weekly_df, "etf_ml_v13_capacity_weekly.csv")
save_csv(capacity_summary_df, "etf_ml_v13_capacity_summary.csv")
save_csv(concentration_df, "etf_ml_v13_theme_concentration.csv")
display(capacity_summary_df.head(40))
display(concentration_df.head(20))
'''
))

cells.append(code_cell(
    r'''
# =========================
# 7. Export Live Targets and JoinQuant Schedule
# =========================
def target_rows_for_rule(rule_name=PREFERRED_RULE, top_n=PREFERRED_TOP_N):
    part = weekly_cost_df[(weekly_cost_df["rule_name"] == rule_name) & (weekly_cost_df["top_n"] == top_n)].copy()
    if part.empty:
        return pd.DataFrame(), {}
    part = part.sort_values("feature_date")
    latest_date = part["feature_date"].max()
    latest = part[part["feature_date"] == latest_date].copy()
    if rule_name != "consensus_score":
        latest = latest.sort_values("net_ret", ascending=False).head(1)
    rows = []
    schedule = {}
    for _, r in part.iterrows():
        codes = parse_targets(r["targets"])
        if not codes:
            continue
        weights = target_weights(codes)
        schedule[str(pd.Timestamp(r["feature_date"]).date())] = weights
    for _, r in latest.iterrows():
        codes = parse_targets(r["targets"])
        weights = target_weights(codes)
        names = [n for n in str(r.get("target_names", "")).split("|") if n and n != "nan"]
        groups = [g for g in str(r.get("target_groups", "")).split("|") if g and g != "nan"]
        for rank, code in enumerate(codes, 1):
            rows.append({
                "asof_feature_date": r["feature_date"],
                "rule_name": r["rule_name"],
                "model_name": r["model_name"],
                "top_n": int(r["top_n"]),
                "rank": rank,
                "code": code,
                "target_weight": weights.get(code, np.nan),
                "name": names[rank - 1] if rank - 1 < len(names) else "",
                "etf_group": groups[rank - 1] if rank - 1 < len(groups) else "",
                "min_selected_money20": r.get("min_selected_money20", np.nan),
                "mean_selected_money20": r.get("mean_selected_money20", np.nan),
            })
    return pd.DataFrame(rows), schedule


live_targets_df, schedule = target_rows_for_rule(PREFERRED_RULE, PREFERRED_TOP_N)
save_csv(live_targets_df, "etf_ml_v13_live_targets.csv")

schedule_path = os.path.join(OUT_DIR, "etf_ml_v13_joinquant_target_schedule.py")
with open(schedule_path, "w", encoding="utf-8") as f:
    f.write("# Generated by ETF_V13_live_gate_validation.ipynb\n")
    f.write("TARGET_SCHEDULE = ")
    f.write(json.dumps(schedule, ensure_ascii=False, indent=2, sort_keys=True))
    f.write("\n")
try:
    with open("etf_ml_v13_joinquant_target_schedule.py", "w", encoding="utf-8") as f:
        f.write("# Generated by ETF_V13_live_gate_validation.ipynb\n")
        f.write("TARGET_SCHEDULE = ")
        f.write(json.dumps(schedule, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
except Exception:
    pass

print("schedule weeks", len(schedule), "path", schedule_path)
display(live_targets_df)
'''
))

cells.append(code_cell(
    r'''
# =========================
# 8. Live Gate Scorecard and Report
# =========================
score_rows = []


def score_item(item, status, detail, severity="INFO"):
    score_rows.append({"item": item, "status": status, "severity": severity, "detail": detail})


fail_quality = data_quality_df[data_quality_df["status"] == "FAIL"]
score_item("data_quality", "PASS" if fail_quality.empty else "FAIL", "failed_checks=%s" % fail_quality["check"].tolist(), "ERROR" if not fail_quality.empty else "INFO")

preferred = summary_df[(summary_df["rule_name"] == PREFERRED_RULE) & (summary_df["top_n"] == PREFERRED_TOP_N) & (summary_df["cost_rate"] == COST_RATE_BASE)]
if preferred.empty:
    score_item("preferred_rule_exists", "FAIL", "%s top%d missing" % (PREFERRED_RULE, PREFERRED_TOP_N), "ERROR")
else:
    pr = preferred.sort_values("net_cum_ret", ascending=False).iloc[0]
    status = "PASS" if pr["net_cum_ret"] > pr["random_mean_cum_ret"] and pr["net_cum_ret"] > pr["equal_weight_cum_ret"] else "FAIL"
    score_item("preferred_beats_core_baselines", status, "net=%.3f random=%.3f equal_weight=%.3f" % (pr["net_cum_ret"], pr["random_mean_cum_ret"], pr["equal_weight_cum_ret"]), "ERROR" if status == "FAIL" else "INFO")

ml_base = baseline_compare_df.copy()
if ml_base.empty:
    score_item("ml_vs_simple_baselines", "FAIL", "No baseline comparison rows.", "ERROR")
else:
    pass_rate = float(ml_base["ml_beats_baseline"].mean())
    score_item("ml_vs_simple_baselines", "PASS" if pass_rate >= 0.60 else "FAIL", "pass_rate=%.3f rows=%d" % (pass_rate, len(ml_base)), "ERROR" if pass_rate < 0.60 else "INFO")

coos = common_oos_df[(common_oos_df["rule_name"].isin(["ml_score", "consensus_score"])) & (common_oos_df["top_n"].isin([3, 5]))].copy()
if coos.empty:
    score_item("common_oos", "FAIL", "No common OOS rows for ML/consensus top3/top5.", "ERROR")
else:
    good = coos[coos["beats_median"] & coos["beats_random"] & coos["beats_equal_weight"] & (coos["net_cum_ret"] > 0)]
    score_item("common_oos", "PASS" if len(good) >= max(1, int(0.5 * len(coos))) else "FAIL", "good=%d total=%d" % (len(good), len(coos)), "ERROR" if len(good) == 0 else "INFO")

stress = robust_summary_df[(robust_summary_df["rule_name"].isin(["ml_score", "consensus_score"])) & (robust_summary_df["top_n"].isin([3, 5])) & (robust_summary_df["cost_rate"] <= 0.0005)]
stress_good = stress[stress["net_cum_ret"] > 0]
score_item("cost_stress_to_0005", "PASS" if len(stress_good) >= max(1, int(0.7 * len(stress))) else "FAIL", "positive=%d total=%d" % (len(stress_good), len(stress)), "ERROR" if len(stress_good) == 0 else "INFO")

turn = summary_df[(summary_df["rule_name"].isin(["ml_score", "consensus_score"])) & (summary_df["top_n"].isin([3, 5]))]
high_turn = turn[turn["avg_turnover"] > 1.5]
score_item("turnover", "WARN" if not high_turn.empty else "PASS", "rows avg_turnover>1.5: %d" % len(high_turn), "WARN" if not high_turn.empty else "INFO")

cap10 = capacity_summary_df[(capacity_summary_df["rule_name"] == PREFERRED_RULE) & (capacity_summary_df["top_n"] == PREFERRED_TOP_N) & (capacity_summary_df["capital"] == 10_000_000)]
if cap10.empty:
    score_item("capacity_1000w", "FAIL", "No capacity row for preferred rule.", "ERROR")
else:
    ok_rate = float(cap10["ok_rate"].max())
    score_item("capacity_1000w", "PASS" if ok_rate >= 0.95 else "WARN", "ok_rate=%.3f at max participation %.2f%%" % (ok_rate, MAX_MONEY20_PARTICIPATION * 100), "WARN" if ok_rate < 0.95 else "INFO")

conc = concentration_df[(concentration_df["rule_name"] == PREFERRED_RULE) & (concentration_df["top_n"] == PREFERRED_TOP_N)]
if conc.empty or (not conc["target_groups_available"].all()):
    score_item("theme_concentration", "BLOCKED_MISSING_DATA", "ETF groups unavailable for all preferred-rule weeks.", "WARN")
else:
    too_conc = conc[conc["max_group_share"] > 0.67]
    score_item("theme_concentration", "PASS" if too_conc.empty else "WARN", "weeks max_group_share>0.67: %d" % len(too_conc), "WARN" if not too_conc.empty else "INFO")

score_item("joinquant_schedule_export", "PASS" if len(schedule) > 0 else "FAIL", "weeks=%d" % len(schedule), "ERROR" if len(schedule) == 0 else "INFO")

scorecard_df = pd.DataFrame(score_rows)
if (scorecard_df["status"] == "FAIL").any():
    final_status = "RESEARCH_ONLY"
elif (scorecard_df["status"] == "BLOCKED_MISSING_DATA").any():
    final_status = "PAPER_ONLY_BLOCKED_METADATA"
elif (scorecard_df["status"] == "WARN").any():
    final_status = "PAPER_READY_WITH_WARNINGS"
else:
    final_status = "PAPER_READY"
scorecard_df["final_status"] = final_status

save_csv(scorecard_df, "etf_ml_v13_live_gate_scorecard.csv")


def fmt_pct(x):
    if pd.isnull(x):
        return "nan"
    return "%.2f%%" % (100.0 * float(x))


lines = []
lines.append("# ETF V13 Live Gate Report")
lines.append("")
lines.append("Final status: **%s**" % final_status)
lines.append("")
lines.append("## Scorecard")
for _, r in scorecard_df.iterrows():
    lines.append("- `%s`: **%s** - %s" % (r["item"], r["status"], r["detail"]))
lines.append("")
lines.append("## Best Rules")
for _, r in summary_df[summary_df["cost_rate"] == COST_RATE_BASE].sort_values("net_cum_ret", ascending=False).head(15).iterrows():
    lines.append("- `%s` `%s` top%d: net=%s mdd=%s random=%s equal=%s turnover=%.3f" % (
        r["rule_name"], r["model_name"], int(r["top_n"]), fmt_pct(r["net_cum_ret"]),
        fmt_pct(r["net_max_drawdown"]), fmt_pct(r["random_mean_cum_ret"]),
        fmt_pct(r["equal_weight_cum_ret"]), float(r["avg_turnover"])
    ))
lines.append("")
lines.append("## Required Next Actions")
if (scorecard_df["status"] == "FAIL").any() or (scorecard_df["status"] == "BLOCKED_MISSING_DATA").any():
    lines.append("- Do not trade live capital yet. Fix failed or blocked scorecard rows first.")
if name_missing >= 0.05 or group_missing >= 0.10:
    lines.append("- Run this notebook inside JoinQuant or provide ETF name/group metadata so theme concentration can be audited.")
if not capacity_summary_df.empty:
    lines.append("- Use `etf_ml_v13_capacity_summary.csv` to choose a capital cap before paper trading.")
lines.append("- Backtest `strategies/joinquant_etf_ml_v13_targets_strategy.py` with the generated `TARGET_SCHEDULE` before any real-money trade.")

report_text = "\n".join(lines) + "\n"
report_path = os.path.join(OUT_DIR, "etf_ml_v13_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
try:
    with open("etf_ml_v13_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
except Exception:
    pass

print(report_text)
print("V13 report:", report_path)
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
