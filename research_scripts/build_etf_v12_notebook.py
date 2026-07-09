import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_NOTEBOOK = PROJECT_DIR / "notebooks" / "ETF_V12_live_readiness_validation.ipynb"


def source(text):
    return text.strip("\n").splitlines(True)


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source(text)}


cells = []

cells.append(markdown_cell(
    """
# ETF_V12 Live Readiness Validation

## 目的

V11 证明 ETF ML 轮动策略有信号，但还不能直接实盘。V12 的目标是把“能不能进入小资金纸面跟踪 / paper trading”拆成一组明确验证：

- 数据质量和 ETF 元数据完整性
- random / median / 简单动量趋势基准
- 共同 OOS
- 参数、成本、topN 鲁棒性
- walk-forward 年度验证
- 换手和容量
- 回撤归因
- 组合规则和模型投票
- 最终 live-readiness scorecard

本 notebook 优先读取 V11 输出文件。若 `score_panel` / `weekly_panel` 不存在，相关验证会标记为 `BLOCKED_MISSING_DATA`，不会伪造结论。
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
    print("jqdata is unavailable; metadata enrichment from JoinQuant will be skipped unless cached names exist.")

import os
import math
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 220)

OUT_DIR = "etf_ml_v12_live_readiness_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_DIR_CANDIDATES = [
    "G:/",
    ".",
    "etf_ml_v11_2021_validation_outputs",
    "../etf_ml_v11_2021_validation_outputs",
]

TOP_N_GRID = [1, 3, 5, 7]
COST_RATE_GRID = [0.0001, 0.0003, 0.0005, 0.0010]
CAPITAL_GRID = [1_000_000, 5_000_000, 10_000_000]
SUSPICIOUS_RET_ABS = 0.35
RANDOM_BASELINE_SEEDS = list(range(50))

REQUIRED_V11_FILES = [
    "model_manifest", "weekly_proxy", "weekly_cost_proxy", "summary", "yearly",
    "cost_summary", "common_oos", "cost_stability", "latest_targets",
    "drawdown_events", "score_gap_diagnostics",
]
OPTIONAL_V11_FILES = ["score_panel", "weekly_panel", "money_threshold_summary", "candidate_models", "target_overlap"]


def find_input_file(stem):
    fname = "etf_ml_v11_%s.csv" % stem
    for base in INPUT_DIR_CANDIDATES:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            return path
    return None


def read_csv_or_none(stem):
    path = find_input_file(stem)
    if path is None:
        return None
    df = pd.read_csv(path)
    print("loaded", stem, df.shape, "from", path)
    return df


v11 = {}
for stem in REQUIRED_V11_FILES + OPTIONAL_V11_FILES:
    v11[stem] = read_csv_or_none(stem)

missing_required = [k for k in REQUIRED_V11_FILES if v11.get(k) is None]
if missing_required:
    raise FileNotFoundError("Missing required V11 files: %s" % missing_required)

model_manifest = v11["model_manifest"]
weekly_proxy = v11["weekly_proxy"]
weekly_cost = v11["weekly_cost_proxy"]
v11_summary = v11["summary"]
v11_yearly = v11["yearly"]
v11_cost_summary = v11["cost_summary"]
v11_common_oos = v11["common_oos"]
v11_cost_stability = v11["cost_stability"]
latest_targets = v11["latest_targets"]
drawdown_events = v11["drawdown_events"]
score_gap = v11["score_gap_diagnostics"]
score_panel = v11.get("score_panel")
weekly_panel = v11.get("weekly_panel")


def normalize_dates(df, cols=("feature_date", "next_date", "next_date_5d", "peak_date", "trough_date")):
    if df is None:
        return None
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


for name in list(v11.keys()):
    v11[name] = normalize_dates(v11[name])

weekly_proxy = v11["weekly_proxy"]
weekly_cost = v11["weekly_cost_proxy"]
latest_targets = v11["latest_targets"]
drawdown_events = v11["drawdown_events"]
score_gap = v11["score_gap_diagnostics"]
score_panel = v11.get("score_panel")
weekly_panel = v11.get("weekly_panel")

print("V12 inputs ready.")
'''
))

cells.append(code_cell(
    r'''
# =========================
# 1. Shared Metrics and Metadata Helpers
# =========================
ETF_GROUP_RULES = [
    ("bank", ["银行"]),
    ("innovative_drug", ["创新药", "新药", "医药", "医疗", "生物", "药"]),
    ("oil_gas", ["油气", "石油", "能源"]),
    ("coal", ["煤炭"]),
    ("semiconductor", ["半导", "芯片", "集成电路"]),
    ("software_ai", ["软件", "人工智能", "AI", "云计算", "大数据", "计算机", "信创"]),
    ("broker", ["证券", "券商"]),
    ("military", ["军工", "国防", "航空航天", "通用航空"]),
    ("gold", ["黄金"]),
    ("nonferrous", ["有色", "稀有金属", "稀土"]),
    ("consumer", ["消费", "食品", "酒"]),
    ("real_estate", ["地产", "房地产"]),
    ("new_energy", ["新能源", "光伏", "电池", "锂电", "储能"]),
    ("cross_border", ["纳指", "标普", "恒生", "港股", "日经", "德国", "法国", "海外", "QDII"]),
    ("broad", ["沪深300", "中证500", "中证1000", "创业板", "科创", "上证50", "A500"]),
]


# Keep these keywords as Unicode escapes so this generator stays robust across
# Windows code pages and JoinQuant notebook exports.
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


def missing_or_blank(series):
    return series.isnull() | (series.astype(str).str.strip() == "")


def group_quality(series):
    valid = series[~missing_or_blank(series)].astype(str)
    missing_ratio = float(missing_or_blank(series).mean())
    unique_groups = int(valid.nunique()) if len(valid) else 0
    max_group_share = float(valid.value_counts(normalize=True).iloc[0]) if len(valid) else 1.0
    return missing_ratio, unique_groups, max_group_share


def classify_etf_group(name):
    if name is None or pd.isnull(name) or str(name).strip() == "":
        return np.nan
    s = str(name)
    for group, keys in ETF_GROUP_RULES:
        if any(k in s for k in keys):
            return group
    return "other"


def try_jq_name_map(date=None):
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
    needs = missing_or_blank(out["name"])
    if needs.any():
        name_map = try_jq_name_map(date=date)
        if name_map:
            out.loc[needs, "name"] = out.loc[needs, "code"].map(name_map)
    group_missing = missing_or_blank(out["etf_group"])
    out.loc[group_missing, "etf_group"] = out.loc[group_missing, "name"].map(classify_etf_group)
    return out


def split_targets(x):
    if pd.isnull(x):
        return []
    return [z for z in str(x).split("|") if z and z != "nan"]


def max_drawdown_from_returns(ret):
    r = pd.Series(ret).dropna().astype(float)
    if r.empty:
        return np.nan
    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def summarize_returns(ret):
    r = pd.Series(ret).dropna().astype(float)
    if r.empty:
        return {"periods": 0, "cum_ret": np.nan, "mean_ret": np.nan, "max_drawdown": np.nan, "win_rate": np.nan, "sharpe52": np.nan, "max_loss": np.nan}
    std = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    sharpe = np.nan if pd.isnull(std) or std <= 0 else float(r.mean() / std * math.sqrt(52))
    return {
        "periods": int(len(r)),
        "cum_ret": float((1.0 + r).prod() - 1.0),
        "mean_ret": float(r.mean()),
        "max_drawdown": max_drawdown_from_returns(r),
        "win_rate": float((r > 0).mean()),
        "sharpe52": sharpe,
        "max_loss": float(r.min()),
    }


def add_prefixed(rec, prefix, stats):
    for k, v in stats.items():
        rec["%s_%s" % (prefix, k)] = v
    return rec


def save_csv(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    try:
        df.to_csv(name, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    print("saved", path, df.shape)
    return path


latest_targets = enrich_names(latest_targets, date=latest_targets["feature_date"].max() if "feature_date" in latest_targets.columns else None)
drawdown_events = enrich_names(drawdown_events.rename(columns={"targets": "code"}) if "targets" in drawdown_events.columns and "code" not in drawdown_events.columns else drawdown_events)
if score_panel is not None:
    score_panel = enrich_names(score_panel, date=score_panel["feature_date"].max() if "feature_date" in score_panel.columns else None)

print("helpers ready")
'''
))

cells.append(code_cell(
    r'''
# =========================
# 2. Data Quality
# =========================
quality_rows = []

def qcheck(name, status, detail, severity="INFO"):
    quality_rows.append({"check": name, "status": status, "severity": severity, "detail": detail})

qcheck("required_files", "PASS", "All required V11 files loaded.")
qcheck("score_panel_available", "PASS" if score_panel is not None else "BLOCKED_MISSING_DATA", "score_panel is required for top7, simple momentum baselines, liquidity threshold grid, and capacity.")
qcheck("weekly_panel_available", "PASS" if weekly_panel is not None else "BLOCKED_MISSING_DATA", "weekly_panel is useful for label coverage and raw feature diagnostics.")

for label, df in [("weekly_cost", weekly_cost), ("weekly_proxy", weekly_proxy), ("latest_targets", latest_targets)]:
    qcheck("%s_rows" % label, "PASS" if df is not None and len(df) > 0 else "FAIL", "%s rows=%s" % (label, 0 if df is None else len(df)), "ERROR" if df is None or len(df) == 0 else "INFO")

if "name" in latest_targets.columns:
    name_missing = float(missing_or_blank(latest_targets["name"]).mean())
    qcheck("latest_target_names", "PASS" if name_missing < 0.05 else "FAIL", "missing_ratio=%.3f" % name_missing, "ERROR" if name_missing >= 0.05 else "INFO")
else:
    qcheck("latest_target_names", "FAIL", "name column missing", "ERROR")

if "etf_group" in latest_targets.columns:
    group_missing, group_unique, group_max_share = group_quality(latest_targets["etf_group"])
    qcheck("latest_target_groups", "PASS" if group_missing < 0.10 else "FAIL", "missing_ratio=%.3f" % group_missing, "ERROR" if group_missing >= 0.10 else "INFO")
    qcheck(
        "latest_target_group_diversity",
        "PASS" if group_unique > 1 or len(latest_targets) < 5 else "WARN",
        "unique_groups=%d max_group_share=%.3f" % (group_unique, group_max_share),
        "WARN" if group_unique <= 1 and len(latest_targets) >= 5 else "INFO",
    )
else:
    qcheck("latest_target_groups", "FAIL", "etf_group column missing", "ERROR")

dup_cols = ["model_name", "feature_date", "top_n", "group_cap"]
if all(c in weekly_cost.columns for c in dup_cols):
    dups = int(weekly_cost.duplicated(dup_cols).sum())
    qcheck("weekly_cost_duplicates", "PASS" if dups == 0 else "FAIL", "duplicate_rows=%d" % dups, "ERROR" if dups else "INFO")

for col in ["gross_ret", "net_ret", "median_ret", "random_mean_ret"]:
    if col in weekly_cost.columns:
        miss = float(weekly_cost[col].isnull().mean())
        extreme = int((weekly_cost[col].abs() > SUSPICIOUS_RET_ABS).sum())
        qcheck("%s_missing" % col, "PASS" if miss < 0.01 else "WARN", "missing_ratio=%.3f" % miss, "WARN" if miss >= 0.01 else "INFO")
        qcheck("%s_extreme_abs_gt_%s" % (col, SUSPICIOUS_RET_ABS), "PASS" if extreme == 0 else "WARN", "count=%d" % extreme, "WARN" if extreme else "INFO")

if weekly_panel is not None:
    if "future_ret_5d" in weekly_panel.columns:
        by_week = weekly_panel.groupby("feature_date")["future_ret_5d"].apply(lambda s: s.notnull().mean()).reset_index(name="label_coverage")
        qcheck("weekly_panel_label_coverage", "PASS" if by_week["label_coverage"].min() > 0.8 else "WARN", "min=%.3f median=%.3f" % (by_week["label_coverage"].min(), by_week["label_coverage"].median()), "WARN")
    else:
        qcheck("weekly_panel_label_coverage", "BLOCKED_MISSING_DATA", "future_ret_5d missing from weekly_panel", "WARN")

data_quality_df = pd.DataFrame(quality_rows)
save_csv(data_quality_df, "etf_ml_v12_data_quality.csv")
display(data_quality_df)
'''
))

cells.append(code_cell(
    r'''
# =========================
# 3. Baseline Completeness and Baseline Summaries
# =========================
def baseline_top_ret(gdf, factor_col, top_n):
    if factor_col not in gdf.columns or "eval_ret" not in gdf.columns:
        return np.nan
    x = gdf.dropna(subset=[factor_col, "eval_ret"]).copy()
    if x.empty:
        return np.nan
    return float(x.sort_values(factor_col, ascending=False).head(min(int(top_n), len(x)))["eval_ret"].mean())


def random_stats(gdf, top_n):
    x = gdf.dropna(subset=["eval_ret"]).copy()
    if x.empty:
        return {"random_mean_ret": np.nan, "random_median_ret": np.nan}
    n = min(int(top_n), len(x))
    vals = [float(x.sample(n, random_state=seed)["eval_ret"].mean()) for seed in RANDOM_BASELINE_SEEDS]
    return {"random_mean_ret": float(np.mean(vals)), "random_median_ret": float(np.median(vals))}


def build_baseline_weekly_from_score(score_data):
    if score_data is None or score_data.empty:
        return pd.DataFrame()
    rows = []
    factor_cols = ["ret_20", "ret_60", "trend_score_25", "trend_score_60"]
    for (model_name, dt), gdf in score_data.groupby(["model_name", "feature_date"]):
        if "eval_ret" not in gdf.columns:
            continue
        for top_n in TOP_N_GRID:
            rec = {"model_name": model_name, "feature_date": dt, "top_n": top_n}
            rec["median_ret"] = float(gdf["eval_ret"].median())
            rec["equal_weight_ret"] = float(gdf["eval_ret"].mean())
            rec.update(random_stats(gdf, top_n))
            for col in factor_cols:
                rec["%s_topn_ret" % col] = baseline_top_ret(gdf, col, top_n)
            rows.append(rec)
    return pd.DataFrame(rows)


baseline_weekly_df = build_baseline_weekly_from_score(score_panel)

if baseline_weekly_df.empty:
    rows = []
    for _, r in weekly_cost.iterrows():
        rows.append({
            "model_name": r["model_name"],
            "feature_date": r["feature_date"],
            "top_n": r["top_n"],
            "median_ret": r.get("median_ret", np.nan),
            "random_mean_ret": r.get("random_mean_ret", np.nan),
            "random_median_ret": r.get("random_median_ret", np.nan),
            "equal_weight_ret": np.nan,
            "ret_20_topn_ret": np.nan,
            "ret_60_topn_ret": np.nan,
            "trend_score_25_topn_ret": np.nan,
            "trend_score_60_topn_ret": np.nan,
        })
    baseline_weekly_df = pd.DataFrame(rows)


def summarize_baseline(baseline_weekly):
    rows = []
    ret_cols = [c for c in baseline_weekly.columns if c.endswith("_ret")]
    for (model_name, top_n), gdf in baseline_weekly.groupby(["model_name", "top_n"]):
        rec = {"model_name": model_name, "top_n": int(top_n)}
        for col in ret_cols:
            add_prefixed(rec, col.replace("_ret", ""), summarize_returns(gdf[col]))
        rows.append(rec)
    return pd.DataFrame(rows)


baseline_summary_df = summarize_baseline(baseline_weekly_df)

baseline_availability_rows = []
for col in ["ret_20_topn_ret", "ret_60_topn_ret", "trend_score_25_topn_ret", "trend_score_60_topn_ret", "equal_weight_ret"]:
    available = col in baseline_weekly_df.columns and baseline_weekly_df[col].notnull().any()
    baseline_availability_rows.append({"baseline": col, "status": "PASS" if available else "BLOCKED_MISSING_DATA", "non_null_rows": int(baseline_weekly_df[col].notnull().sum()) if col in baseline_weekly_df.columns else 0})
baseline_availability_df = pd.DataFrame(baseline_availability_rows)

save_csv(baseline_weekly_df, "etf_ml_v12_baseline_weekly.csv")
save_csv(baseline_summary_df, "etf_ml_v12_baseline_summary.csv")
save_csv(baseline_availability_df, "etf_ml_v12_baseline_availability.csv")

display(baseline_availability_df)
display(baseline_summary_df.head(20))
'''
))

cells.append(code_cell(
    r'''
# =========================
# 4. Common OOS and Cost Robustness
# =========================
def common_oos_for_models(data, models, start_date, top_n):
    sub = data[(data["model_name"].isin(models)) & (data["top_n"] == top_n) & (data["feature_date"] >= pd.Timestamp(start_date))].copy()
    if sub.empty:
        return pd.DataFrame()
    date_sets = [set(g["feature_date"]) for _, g in sub.groupby("model_name")]
    if not date_sets:
        return pd.DataFrame()
    common_dates = set.intersection(*date_sets)
    if not common_dates:
        return pd.DataFrame()
    return sub[sub["feature_date"].isin(common_dates)].copy()


models_by_end = dict(zip(model_manifest["train_end"], model_manifest["model_name"]))
model_2023 = models_by_end.get("2023-12-31")
model_2024 = models_by_end.get("2024-12-31")
model_2025 = models_by_end.get("2025-12-31")

common_specs = []
if model_2023 and model_2024:
    common_specs.append(("pair_2023_2024_from_2025", [model_2023, model_2024], "2025-01-01"))
if model_2023 and model_2024 and model_2025:
    common_specs.append(("all_windows_from_2026", [model_2023, model_2024, model_2025], "2026-01-01"))

common_rows = []
for spec_name, models, start in common_specs:
    for top_n in sorted(weekly_cost["top_n"].dropna().unique()):
        common = common_oos_for_models(weekly_cost, models, start, int(top_n))
        for model_name, gdf in common.groupby("model_name"):
            rec = {"common_spec": spec_name, "start_date": start, "model_name": model_name, "top_n": int(top_n), "common_weeks": int(len(gdf))}
            add_prefixed(rec, "net", summarize_returns(gdf["net_ret"]))
            add_prefixed(rec, "gross", summarize_returns(gdf["gross_ret"]))
            add_prefixed(rec, "median", summarize_returns(gdf["median_ret"]))
            add_prefixed(rec, "random", summarize_returns(gdf["random_mean_ret"]))
            rec["beats_median"] = rec["net_cum_ret"] > rec["median_cum_ret"] if pd.notnull(rec["median_cum_ret"]) else False
            rec["beats_random"] = rec["net_cum_ret"] > rec["random_cum_ret"] if pd.notnull(rec["random_cum_ret"]) else False
            common_rows.append(rec)

v12_common_oos_df = pd.DataFrame(common_rows)
if not v12_common_oos_df.empty:
    v12_common_oos_df = v12_common_oos_df.sort_values(["common_spec", "top_n", "net_cum_ret"], ascending=[True, True, False])


def rebuild_cost_grid_from_weekly(base_weekly):
    rows = []
    data = base_weekly.copy()
    if "gross_ret" not in data.columns and "portfolio_ret" in data.columns:
        data["gross_ret"] = data["portfolio_ret"]
    for cost_rate in COST_RATE_GRID:
        x = data.copy()
        x["stress_cost_rate"] = cost_rate
        if "turnover_traded_notional" in x.columns:
            x["stress_cost_drag"] = x["turnover_traded_notional"].astype(float) * cost_rate
        else:
            x["stress_cost_drag"] = x.get("cost_drag", 0)
        x["stress_net_ret"] = x["gross_ret"].astype(float) - x["stress_cost_drag"].astype(float)
        for (model_name, top_n), gdf in x.groupby(["model_name", "top_n"]):
            rec = {"model_name": model_name, "top_n": int(top_n), "cost_rate": cost_rate}
            add_prefixed(rec, "stress_net", summarize_returns(gdf["stress_net_ret"]))
            add_prefixed(rec, "gross", summarize_returns(gdf["gross_ret"]))
            rec["avg_turnover"] = float(gdf["turnover_traded_notional"].mean()) if "turnover_traded_notional" in gdf.columns else np.nan
            rec["total_cost_drag"] = float(gdf["stress_cost_drag"].sum())
            rows.append(rec)
    return pd.DataFrame(rows)


v12_robustness_grid_df = rebuild_cost_grid_from_weekly(weekly_cost)

save_csv(v12_common_oos_df, "etf_ml_v12_common_oos.csv")
save_csv(v12_robustness_grid_df, "etf_ml_v12_robustness_grid.csv")
display(v12_common_oos_df)
display(v12_robustness_grid_df.head(30))
'''
))

cells.append(code_cell(
    r'''
# =========================
# 5. Walk-forward
# =========================
walk_specs = [
    ("wf_2024_train2021_2023", "2023-12-31", 2024),
    ("wf_2025_train2021_2024", "2024-12-31", 2025),
    ("wf_2026_train2021_2025", "2025-12-31", 2026),
]

walk_weekly_parts = []
for spec_name, train_end, year in walk_specs:
    model_name = models_by_end.get(train_end)
    if not model_name:
        continue
    part = weekly_cost[(weekly_cost["model_name"] == model_name) & (pd.to_datetime(weekly_cost["feature_date"]).dt.year == year)].copy()
    part["walkforward_spec"] = spec_name
    part["walkforward_year"] = year
    walk_weekly_parts.append(part)

walkforward_weekly_df = pd.concat(walk_weekly_parts, ignore_index=True, sort=False) if walk_weekly_parts else pd.DataFrame()

walk_rows = []
if not walkforward_weekly_df.empty:
    for (spec_name, year, top_n), gdf in walkforward_weekly_df.groupby(["walkforward_spec", "walkforward_year", "top_n"]):
        rec = {"walkforward_spec": spec_name, "year": int(year), "top_n": int(top_n), "weeks": int(len(gdf))}
        add_prefixed(rec, "net", summarize_returns(gdf["net_ret"]))
        add_prefixed(rec, "gross", summarize_returns(gdf["gross_ret"]))
        add_prefixed(rec, "median", summarize_returns(gdf["median_ret"]))
        add_prefixed(rec, "random", summarize_returns(gdf["random_mean_ret"]))
        rec["beats_median"] = rec["net_cum_ret"] > rec["median_cum_ret"] if pd.notnull(rec["median_cum_ret"]) else False
        rec["beats_random"] = rec["net_cum_ret"] > rec["random_cum_ret"] if pd.notnull(rec["random_cum_ret"]) else False
        walk_rows.append(rec)

walkforward_df = pd.DataFrame(walk_rows)
save_csv(walkforward_weekly_df, "etf_ml_v12_walkforward_weekly.csv")
save_csv(walkforward_df, "etf_ml_v12_walkforward.csv")
display(walkforward_df)
'''
))

cells.append(code_cell(
    r'''
# =========================
# 6. Turnover, Capacity, and Drawdown Attribution
# =========================
capacity_rows = []
for (model_name, top_n), gdf in weekly_cost.groupby(["model_name", "top_n"]):
    rec = {"model_name": model_name, "top_n": int(top_n)}
    rec["weeks"] = int(len(gdf))
    rec["avg_turnover"] = float(gdf["turnover_traded_notional"].mean()) if "turnover_traded_notional" in gdf.columns else np.nan
    rec["max_turnover"] = float(gdf["turnover_traded_notional"].max()) if "turnover_traded_notional" in gdf.columns else np.nan
    rec["annualized_turnover_est"] = rec["avg_turnover"] * 52 if pd.notnull(rec["avg_turnover"]) else np.nan
    rec["total_cost_drag"] = float(gdf["cost_drag"].sum()) if "cost_drag" in gdf.columns else np.nan
    capacity_rows.append(rec)

turnover_capacity_df = pd.DataFrame(capacity_rows)

if score_panel is not None and "money_mean_20" in score_panel.columns:
    liq_rows = []
    for _, row in weekly_cost.iterrows():
        codes = split_targets(row.get("targets", ""))
        if not codes:
            continue
        sp = score_panel[(score_panel["model_name"] == row["model_name"]) & (score_panel["feature_date"] == row["feature_date"]) & (score_panel["code"].isin(codes))]
        if sp.empty:
            continue
        liq = sp["money_mean_20"].dropna().astype(float)
        if liq.empty:
            continue
        rec = {"model_name": row["model_name"], "top_n": int(row["top_n"]), "feature_date": row["feature_date"], "min_selected_money20": float(liq.min()), "mean_selected_money20": float(liq.mean())}
        for cap in CAPITAL_GRID:
            rec["capital_%d_max_participation" % cap] = float((cap / int(row["top_n"])) / liq.min())
        liq_rows.append(rec)
    liq_df = pd.DataFrame(liq_rows)
    if not liq_df.empty:
        liq_summary = liq_df.groupby(["model_name", "top_n"]).agg(
            min_selected_money20=("min_selected_money20", "min"),
            median_selected_money20=("min_selected_money20", "median"),
        ).reset_index()
        for cap in CAPITAL_GRID:
            col = "capital_%d_max_participation" % cap
            liq_summary[col + "_p95"] = liq_df.groupby(["model_name", "top_n"])[col].quantile(0.95).values
        turnover_capacity_df = turnover_capacity_df.merge(liq_summary, on=["model_name", "top_n"], how="left")
else:
    turnover_capacity_df["capacity_status"] = "BLOCKED_MISSING_SCORE_PANEL_OR_MONEY"

drawdown_attr = drawdown_events.copy()
if "targets" in drawdown_attr.columns:
    drawdown_attr["primary_code"] = drawdown_attr["targets"].map(lambda x: split_targets(x)[0] if split_targets(x) else np.nan)
if "primary_code" in drawdown_attr.columns:
    latest_name_map = latest_targets.dropna(subset=["code"]).drop_duplicates("code").set_index("code")["name"].to_dict() if "name" in latest_targets.columns else {}
    latest_group_map = latest_targets.dropna(subset=["code"]).drop_duplicates("code").set_index("code")["etf_group"].to_dict() if "etf_group" in latest_targets.columns else {}
    drawdown_attr["name_from_latest"] = drawdown_attr["primary_code"].map(latest_name_map)
    drawdown_attr["group_from_latest"] = drawdown_attr["primary_code"].map(latest_group_map)
if not score_gap.empty:
    gap_cols = ["model_name", "feature_date", "gap_1_2", "gap_1_3", "score_std_top10"]
    drawdown_attr = drawdown_attr.merge(score_gap[gap_cols], on=["model_name", "feature_date"], how="left")

save_csv(turnover_capacity_df, "etf_ml_v12_turnover_capacity.csv")
save_csv(drawdown_attr, "etf_ml_v12_drawdown_attribution.csv")
display(turnover_capacity_df)
display(drawdown_attr.head(30))
'''
))

cells.append(code_cell(
    r'''
# =========================
# 7. Portfolio Rules and Model Consensus
# =========================
rule_rows = []
rule_weekly_parts = []

# Existing equal-weight rules from V11.
for (model_name, top_n), gdf in weekly_cost.groupby(["model_name", "top_n"]):
    rec = {"rule_name": "single_model_top%d_equal_weight" % int(top_n), "model_name": model_name, "top_n": int(top_n), "blocked": False}
    add_prefixed(rec, "net", summarize_returns(gdf["net_ret"]))
    add_prefixed(rec, "gross", summarize_returns(gdf["gross_ret"]))
    rec["avg_turnover"] = float(gdf["turnover_traded_notional"].mean()) if "turnover_traded_notional" in gdf.columns else np.nan
    rule_rows.append(rec)
    temp = gdf.copy()
    temp["rule_name"] = rec["rule_name"]
    rule_weekly_parts.append(temp)

# Consensus across training-window models: average returns of models that agree on a date/topN.
for top_n, gdf0 in weekly_cost.groupby("top_n"):
    pivot = gdf0.pivot_table(index="feature_date", columns="model_name", values="net_ret", aggfunc="mean")
    if pivot.shape[1] < 2:
        continue
    consensus_ret = pivot.mean(axis=1, skipna=True)
    consensus_df = pd.DataFrame({"feature_date": consensus_ret.index, "net_ret": consensus_ret.values})
    consensus_df["gross_ret"] = consensus_df["net_ret"]
    consensus_df["top_n"] = int(top_n)
    consensus_df["model_name"] = "consensus_average_models"
    consensus_df["rule_name"] = "consensus_average_top%d" % int(top_n)
    rec = {"rule_name": "consensus_average_top%d" % int(top_n), "model_name": "consensus_average_models", "top_n": int(top_n), "blocked": False}
    add_prefixed(rec, "net", summarize_returns(consensus_df["net_ret"]))
    add_prefixed(rec, "gross", summarize_returns(consensus_df["gross_ret"]))
    rec["avg_turnover"] = np.nan
    rule_rows.append(rec)
    rule_weekly_parts.append(consensus_df)

# Theme cap readiness check.
if "etf_group" in latest_targets.columns:
    group_missing, group_unique, group_max_share = group_quality(latest_targets["etf_group"])
else:
    group_missing, group_unique, group_max_share = 1.0, 0, 1.0
rule_rows.append({
    "rule_name": "theme_cap_readiness",
    "model_name": "all",
    "top_n": np.nan,
    "blocked": bool(group_missing > 0.10 or (len(latest_targets) >= 5 and group_unique <= 1)),
    "block_reason": "ETF group missing ratio %.3f; unique_groups=%d; max_group_share=%.3f" % (group_missing, group_unique, group_max_share),
})

portfolio_rules_df = pd.DataFrame(rule_rows)
rule_weekly_df = pd.concat(rule_weekly_parts, ignore_index=True, sort=False) if rule_weekly_parts else pd.DataFrame()

save_csv(rule_weekly_df, "etf_ml_v12_rule_weekly.csv")
save_csv(portfolio_rules_df, "etf_ml_v12_portfolio_rules.csv")
display(portfolio_rules_df.sort_values("net_cum_ret", ascending=False, na_position="last").head(30))
'''
))

cells.append(code_cell(
    r'''
# =========================
# 8. Live Readiness Scorecard and Report
# =========================
score_rows = []

def score_item(item, status, detail, severity="INFO"):
    score_rows.append({"item": item, "status": status, "severity": severity, "detail": detail})

name_fail = data_quality_df[(data_quality_df["check"] == "latest_target_names") & (data_quality_df["status"] != "PASS")]
group_fail = data_quality_df[(data_quality_df["check"] == "latest_target_groups") & (data_quality_df["status"] != "PASS")]
group_diversity_warn = data_quality_df[(data_quality_df["check"] == "latest_target_group_diversity") & (data_quality_df["status"] != "PASS")]
score_item("data_quality_names", "PASS" if name_fail.empty else "FAIL", "ETF names must be available for live drawdown attribution.", "ERROR" if not name_fail.empty else "INFO")
score_item("data_quality_groups", "PASS" if group_fail.empty else "FAIL", "ETF groups must be available for theme cap and concentration checks.", "ERROR" if not group_fail.empty else "INFO")
score_item("data_quality_group_diversity", "PASS" if group_diversity_warn.empty else "WARN", "ETF group classification should not collapse all selected funds into one bucket.", "WARN" if not group_diversity_warn.empty else "INFO")

baseline_blocked = baseline_availability_df[baseline_availability_df["status"] != "PASS"]
score_item("simple_baselines", "PASS" if baseline_blocked.empty else "BLOCKED_MISSING_DATA", "Blocked baselines: %s" % baseline_blocked["baseline"].tolist(), "WARN" if not baseline_blocked.empty else "INFO")

if not v12_common_oos_df.empty:
    strong_common = v12_common_oos_df[(v12_common_oos_df["top_n"].isin([3, 5])) & (v12_common_oos_df["beats_median"]) & (v12_common_oos_df["beats_random"])]
    score_item("common_oos_top3_top5", "PASS" if not strong_common.empty else "FAIL", "top3/top5 rows beating median and random: %d" % len(strong_common), "ERROR" if strong_common.empty else "INFO")
else:
    score_item("common_oos_top3_top5", "FAIL", "No common OOS rows.", "ERROR")

if not walkforward_df.empty:
    wf = walkforward_df[walkforward_df["top_n"].isin([3, 5])]
    good = wf[(wf["net_cum_ret"] > 0) & (wf["beats_median"]) & (wf["beats_random"])]
    score_item("walkforward_top3_top5", "PASS" if len(good) >= max(1, int(len(wf) * 0.5)) else "FAIL", "good_slices=%d total_slices=%d" % (len(good), len(wf)), "ERROR" if len(good) == 0 else "INFO")
else:
    score_item("walkforward_top3_top5", "FAIL", "No walk-forward rows.", "ERROR")

stress = v12_robustness_grid_df[(v12_robustness_grid_df["top_n"].isin([3, 5])) & (v12_robustness_grid_df["cost_rate"] <= 0.0005)]
stress_good = stress[stress["stress_net_cum_ret"] > 0]
score_item("cost_stress_to_0005", "PASS" if len(stress_good) >= max(1, int(len(stress) * 0.6)) else "FAIL", "positive_rows=%d total_rows=%d" % (len(stress_good), len(stress)), "ERROR" if len(stress_good) == 0 else "INFO")

if "avg_turnover" in turnover_capacity_df.columns:
    high_turn = turnover_capacity_df[turnover_capacity_df["avg_turnover"] > 1.9]
    score_item("turnover", "WARN" if not high_turn.empty else "PASS", "rows avg_turnover>1.9: %d" % len(high_turn), "WARN" if not high_turn.empty else "INFO")
else:
    score_item("turnover", "BLOCKED_MISSING_DATA", "turnover column unavailable.", "WARN")

if "capacity_status" in turnover_capacity_df.columns and (turnover_capacity_df["capacity_status"] == "BLOCKED_MISSING_SCORE_PANEL_OR_MONEY").any():
    score_item("capacity", "BLOCKED_MISSING_DATA", "Need score_panel with money_mean_20 for selected ETF capacity.", "WARN")
else:
    score_item("capacity", "PASS", "Capacity metrics available.")

rules = portfolio_rules_df[portfolio_rules_df["rule_name"].astype(str).str.startswith("consensus_average")]
score_item("portfolio_rules", "PASS" if not rules.empty and rules["net_cum_ret"].max() > 0 else "WARN", "consensus rules tested=%d" % len(rules), "WARN")

scorecard_df = pd.DataFrame(score_rows)

if (scorecard_df["status"] == "FAIL").any():
    final_status = "RESEARCH_ONLY"
elif (scorecard_df["status"] == "BLOCKED_MISSING_DATA").any() or (scorecard_df["status"] == "WARN").any():
    final_status = "RESEARCH_ONLY"
else:
    final_status = "PASS_PAPER"

scorecard_df["final_status"] = final_status
save_csv(scorecard_df, "etf_ml_v12_live_readiness_scorecard.csv")


def fmt_pct(x):
    if pd.isnull(x):
        return "nan"
    return "%.2f%%" % (100 * float(x))


lines = []
lines.append("# ETF V12 Live Readiness Report")
lines.append("")
lines.append("Final status: **%s**" % final_status)
lines.append("")
lines.append("## Scorecard")
for _, r in scorecard_df.iterrows():
    lines.append("- `%s`: **%s** - %s" % (r["item"], r["status"], r["detail"]))
lines.append("")
lines.append("## Best V12 Common OOS")
if not v12_common_oos_df.empty:
    for _, r in v12_common_oos_df.sort_values("net_cum_ret", ascending=False).head(10).iterrows():
        lines.append("- `%s` `%s` top%d weeks=%d net=%s random=%s median=%s" % (
            r["common_spec"], r["model_name"], int(r["top_n"]), int(r["common_weeks"]),
            fmt_pct(r["net_cum_ret"]), fmt_pct(r["random_cum_ret"]), fmt_pct(r["median_cum_ret"])
        ))
lines.append("")
lines.append("## Walk-forward")
if not walkforward_df.empty:
    for _, r in walkforward_df.sort_values(["top_n", "year"]).iterrows():
        lines.append("- `%s` top%d: net=%s random=%s median=%s" % (
            r["walkforward_spec"], int(r["top_n"]), fmt_pct(r["net_cum_ret"]), fmt_pct(r["random_cum_ret"]), fmt_pct(r["median_cum_ret"])
        ))
lines.append("")
lines.append("## Required Next Fixes")
if not name_fail.empty or not group_fail.empty:
    lines.append("- Fix ETF metadata enrichment so names and groups are available for every target.")
if not baseline_blocked.empty:
    lines.append("- Provide `etf_ml_v11_score_panel.csv` or rerun V11 with score panel export to evaluate simple momentum/trend baselines and capacity.")
if (scorecard_df["status"] == "FAIL").any():
    lines.append("- Do not start live trading. Continue research and portfolio-rule validation.")
else:
    lines.append("- Consider 1-3 months paper trading before any capital allocation.")

report_text = "\n".join(lines) + "\n"
report_path = os.path.join(OUT_DIR, "etf_ml_v12_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
try:
    with open("etf_ml_v12_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
except Exception:
    pass

print(report_text)
print("V12 report:", report_path)
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
