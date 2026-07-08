import glob
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_NOTEBOOK = ROOT / "机器学习策略" / "notebooks" / "ETF_V11_2021_validation_framework.ipynb"


def source(text):
    return text.strip("\n").splitlines(True)


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source(text)}


def replace_train_windows(src):
    block = '''TRAIN_WINDOWS = [
    # Fixed 2021 start. ETF universe before 2021 is intentionally excluded.
    ("train20210101_20231231", "2021-01-01", "2023-12-31"),
    ("train20210101_20241231", "2021-01-01", "2024-12-31"),
    ("train20210101_20251231", "2021-01-01", "2025-12-31"),
]

'''
    return re.sub(r"TRAIN_WINDOWS = \[.*?\]\n\n", block, src, flags=re.S)


def patch_config_cell(src):
    src = patch_v10_names(src)
    src = src.replace("TOP_N_LIST = [1, 3]", "TOP_N_LIST = [1, 3, 5]")
    src = replace_train_windows(src)
    src = src.replace(
        'GROUP_CAP_LIST = [0]  # No theme cap. Current V2-like ETF pool keeps duplicate themes by design.',
        'GROUP_CAP_LIST = [0]  # No theme cap for V11 validation.\nCOST_RATE = 0.0001  # 万分之一 = 0.01% = 1 bp, applied to traded notional.',
    )
    src = src.replace(
        'print("V11 focus: training-window stability only; feature set and label stay fixed.")',
        'print("V11 focus: 2021-start validation with cost, topN, common OOS, and stability checks.")',
    )
    return src


def patch_v10_names(src):
    src = src.replace("ETF_V10", "ETF_V11").replace("V10", "V11")
    src = src.replace("etf_ml_v10_recent_window_stability_outputs", "etf_ml_v11_2021_validation_outputs")
    src = src.replace("etf_ml_v10_", "etf_ml_v11_")
    src = src.replace("model_etf_ml_v10_", "model_etf_ml_v11_")
    src = src.replace("etf_ml_v11_recent_window_stability", "etf_ml_v11_2021_validation")
    return src


V11_COST_AND_COMMON_OOS = r'''
# =========================
# 7. V11 cost, turnover, common-OOS, and baseline diagnostics
# =========================
V11_COST_WEEKLY_CSV = os.path.join(OUT_DIR, "etf_ml_v11_weekly_cost_proxy.csv")
V11_COST_SUMMARY_CSV = os.path.join(OUT_DIR, "etf_ml_v11_cost_summary.csv")
V11_COMMON_OOS_CSV = os.path.join(OUT_DIR, "etf_ml_v11_common_oos.csv")
V11_STABILITY_CSV = os.path.join(OUT_DIR, "etf_ml_v11_cost_stability.csv")
V11_REPORT_MD = os.path.join(OUT_DIR, "etf_ml_v11_report.md")
RANDOM_BASELINE_SEEDS = list(range(30))
COMMON_OOS_STARTS = ["2025-01-01", "2026-01-01"]


def _parse_targets(x):
    if pd.isnull(x):
        return []
    return [z for z in str(x).split("|") if z]


def _weights_from_targets(targets):
    codes = _parse_targets(targets)
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {c: w for c in codes}


def _turnover(prev_w, new_w):
    keys = set(prev_w.keys()) | set(new_w.keys())
    traded = float(sum(abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys))
    return traded, traded / 2.0


def _baseline_top_ret(gdf, factor_col, top_n):
    if factor_col not in gdf.columns:
        return np.nan
    x = gdf.dropna(subset=[factor_col, "eval_ret"]).copy()
    if x.empty:
        return np.nan
    return float(x.sort_values(factor_col, ascending=False).head(min(top_n, len(x)))["eval_ret"].mean())


def _random_baseline_stats(gdf, top_n):
    x = gdf.dropna(subset=["eval_ret"]).copy()
    if x.empty:
        return {"random_mean_ret": np.nan, "random_median_ret": np.nan, "random_min_ret": np.nan, "random_max_ret": np.nan}
    n = min(top_n, len(x))
    vals = [float(x.sample(n, random_state=seed)["eval_ret"].mean()) for seed in RANDOM_BASELINE_SEEDS]
    s = pd.Series(vals)
    return {"random_mean_ret": float(s.mean()), "random_median_ret": float(s.median()), "random_min_ret": float(s.min()), "random_max_ret": float(s.max())}


def build_v11_cost_weekly(weekly_proxy, score_data):
    if weekly_proxy is None or weekly_proxy.empty:
        return pd.DataFrame()
    out_parts = []
    score_scheduled = scheduled_score_data(score_data) if score_data is not None and not score_data.empty else pd.DataFrame()
    score_scheduled["feature_date"] = pd.to_datetime(score_scheduled["feature_date"]) if not score_scheduled.empty else pd.Series(dtype="datetime64[ns]")
    for (model_name, top_n, group_cap), gdf0 in weekly_proxy.groupby(["model_name", "top_n", "group_cap"]):
        gdf = gdf0.sort_values("feature_date").copy()
        prev_w = {}
        rows = []
        for _, row in gdf.iterrows():
            new_w = _weights_from_targets(row.get("targets", ""))
            traded, one_way = _turnover(prev_w, new_w)
            prev_w = new_w
            gross_ret = float(row.get("portfolio_ret", np.nan))
            cost_drag = traded * COST_RATE
            rec = row.to_dict()
            rec["gross_ret"] = gross_ret
            rec["turnover_traded_notional"] = traded
            rec["one_way_turnover"] = one_way
            rec["cost_rate"] = COST_RATE
            rec["cost_drag"] = cost_drag
            rec["net_ret"] = gross_ret - cost_drag if pd.notnull(gross_ret) else np.nan
            score_week = score_scheduled[(score_scheduled["model_name"] == model_name) & (score_scheduled["feature_date"] == pd.Timestamp(row["feature_date"]))]
            if score_week.empty:
                rec.update({"random_mean_ret": np.nan, "random_median_ret": np.nan, "random_min_ret": np.nan, "random_max_ret": np.nan})
                rec["ret20_topn_ret"] = np.nan
                rec["trend25_topn_ret"] = np.nan
            else:
                rec.update(_random_baseline_stats(score_week, int(top_n)))
                rec["ret20_topn_ret"] = _baseline_top_ret(score_week, "ret_20", int(top_n))
                rec["trend25_topn_ret"] = _baseline_top_ret(score_week, "trend_score_25", int(top_n))
            rows.append(rec)
        out_parts.append(pd.DataFrame(rows))
    out = pd.concat(out_parts, ignore_index=True, sort=False) if out_parts else pd.DataFrame()
    out.to_csv(V11_COST_WEEKLY_CSV, index=False)
    return out


def _summary_with_prefix(ret_series, prefix):
    s = summarize_returns(ret_series)
    return {prefix + "_" + k: v for k, v in s.items()}


def build_v11_cost_summary(cost_weekly):
    rows = []
    if cost_weekly is None or cost_weekly.empty:
        return pd.DataFrame()
    group_cols = ["model_name", "target_name", "train_tag", "top_n", "group_cap", "min_avg_money_20", "money_threshold_tag"]
    grouped = cost_weekly.copy()
    grouped[group_cols] = grouped[group_cols].fillna("__MISSING__")
    for keys, gdf in grouped.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec = {k: (np.nan if v == "__MISSING__" else v) for k, v in rec.items()}
        rec.update(_summary_with_prefix(gdf["net_ret"], "net"))
        rec.update(_summary_with_prefix(gdf["gross_ret"], "gross"))
        rec.update(_summary_with_prefix(gdf["median_ret"], "median"))
        rec.update(_summary_with_prefix(gdf["random_mean_ret"], "random"))
        rec.update(_summary_with_prefix(gdf["ret20_topn_ret"], "ret20"))
        rec.update(_summary_with_prefix(gdf["trend25_topn_ret"], "trend25"))
        rec["avg_turnover_traded_notional"] = float(gdf["turnover_traded_notional"].mean())
        rec["avg_one_way_turnover"] = float(gdf["one_way_turnover"].mean())
        rec["total_cost_drag"] = float(gdf["cost_drag"].sum())
        rec["net_excess_vs_median"] = rec.get("net_cum_ret", np.nan) - rec.get("median_cum_ret", np.nan)
        rec["net_excess_vs_random"] = rec.get("net_cum_ret", np.nan) - rec.get("random_cum_ret", np.nan)
        rec["net_excess_vs_ret20"] = rec.get("net_cum_ret", np.nan) - rec.get("ret20_cum_ret", np.nan)
        rec["net_excess_vs_trend25"] = rec.get("net_cum_ret", np.nan) - rec.get("trend25_cum_ret", np.nan)
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["top_n", "net_cum_ret"], ascending=[True, False])
    out.to_csv(V11_COST_SUMMARY_CSV, index=False)
    return out


def build_v11_common_oos(cost_weekly):
    rows = []
    if cost_weekly is None or cost_weekly.empty:
        return pd.DataFrame()
    data = cost_weekly.copy()
    data["feature_date"] = pd.to_datetime(data["feature_date"])
    for start in COMMON_OOS_STARTS:
        sub = data[data["feature_date"] >= pd.Timestamp(start)].copy()
        if sub.empty:
            continue
        sub[["top_n", "target_name"]] = sub[["top_n", "target_name"]].fillna("__MISSING__")
        for (top_n, target_name), top_df in sub.groupby(["top_n", "target_name"]):
            target_name_out = np.nan if target_name == "__MISSING__" else target_name
            date_sets = [set(pd.to_datetime(g["feature_date"]).dt.normalize()) for _, g in top_df.groupby("model_name")]
            if not date_sets:
                continue
            common_dates = set.intersection(*date_sets) if len(date_sets) > 1 else date_sets[0]
            if not common_dates:
                continue
            common = top_df[top_df["feature_date"].isin(common_dates)].copy()
            for model_name, gdf in common.groupby("model_name"):
                rec = {"common_oos_start": start, "model_name": model_name, "top_n": int(float(top_n)), "target_name": target_name_out, "common_weeks": len(common_dates)}
                rec.update(_summary_with_prefix(gdf["net_ret"], "net"))
                rec.update(_summary_with_prefix(gdf["gross_ret"], "gross"))
                rec.update(_summary_with_prefix(gdf["median_ret"], "median"))
                rec.update(_summary_with_prefix(gdf["random_mean_ret"], "random"))
                rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["common_oos_start", "top_n", "net_cum_ret"], ascending=[True, True, False])
    out.to_csv(V11_COMMON_OOS_CSV, index=False)
    return out


def build_v11_cost_stability(cost_weekly):
    rows = []
    if cost_weekly is None or cost_weekly.empty:
        return pd.DataFrame()
    for top_n, top_df in cost_weekly.groupby("top_n"):
        models = sorted(top_df["model_name"].dropna().unique())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a = top_df[top_df["model_name"] == models[i]][["feature_date", "targets", "net_ret"]].copy()
                b = top_df[top_df["model_name"] == models[j]][["feature_date", "targets", "net_ret"]].copy()
                m = a.merge(b, on="feature_date", suffixes=("_a", "_b"))
                if m.empty:
                    continue
                overlaps = []
                for _, r in m.iterrows():
                    sa = set(_parse_targets(r["targets_a"]))
                    sb = set(_parse_targets(r["targets_b"]))
                    denom = max(1, min(len(sa), len(sb)))
                    overlaps.append(len(sa & sb) / denom)
                rows.append({
                    "top_n": int(top_n),
                    "model_a": models[i],
                    "model_b": models[j],
                    "common_periods": int(len(m)),
                    "target_overlap_rate": float(np.mean(overlaps)),
                    "ret_corr": float(m["net_ret_a"].corr(m["net_ret_b"])) if len(m) >= 3 else np.nan,
                    "cum_ret_a_common": float((1.0 + m["net_ret_a"].astype(float)).prod() - 1.0),
                    "cum_ret_b_common": float((1.0 + m["net_ret_b"].astype(float)).prod() - 1.0),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["top_n", "common_periods", "target_overlap_rate"], ascending=[True, False, False])
    out.to_csv(V11_STABILITY_CSV, index=False)
    return out


v11_cost_weekly_df = build_v11_cost_weekly(weekly_proxy_df, score_df)
v11_cost_summary_df = build_v11_cost_summary(v11_cost_weekly_df)
v11_common_oos_df = build_v11_common_oos(v11_cost_weekly_df)
v11_cost_stability_df = build_v11_cost_stability(v11_cost_weekly_df)

print("V11 cost weekly:", v11_cost_weekly_df.shape)
print("V11 cost summary:", v11_cost_summary_df.shape)
print("V11 common OOS:", v11_common_oos_df.shape)
print("V11 cost stability:", v11_cost_stability_df.shape)
display(v11_cost_summary_df.head(20))
'''


V11_REPORT = r'''
# =========================
# 8. V11 report export
# =========================
def fmt_pct(x):
    if pd.isnull(x):
        return "nan"
    return "%.2f%%" % (100.0 * float(x))


def build_v11_report():
    lines = []
    lines.append("# ETF V11 Validation Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("- Data start: `%s`" % START_DATE)
    lines.append("- Data end: `%s`" % END_DATE)
    lines.append("- Cost rate: `%.6f` per traded notional, 万分之一 / 1 bp" % COST_RATE)
    lines.append("- Train windows: `%s`" % str(TRAIN_WINDOWS))
    lines.append("- Top N: `%s`" % str(TOP_N_LIST))
    lines.append("")
    lines.append("## Data")
    lines.append("")
    if "panel" in globals() and panel is not None and not panel.empty:
        lines.append("- Panel rows: `%d`" % len(panel))
        lines.append("- Weeks: `%d`" % panel["feature_date"].nunique())
        lines.append("- Feature date range: `%s` to `%s`" % (pd.to_datetime(panel["feature_date"]).min().date(), pd.to_datetime(panel["feature_date"]).max().date()))
    if "manifest_df" in globals() and manifest_df is not None and not manifest_df.empty:
        lines.append("- Models trained: `%d`" % len(manifest_df))
    lines.append("")
    lines.append("## Best Cost-Aware Rows")
    lines.append("")
    if "v11_cost_summary_df" in globals() and v11_cost_summary_df is not None and not v11_cost_summary_df.empty:
        cols = ["model_name", "top_n", "net_cum_ret", "net_max_drawdown", "net_win_rate", "avg_turnover_traded_notional", "net_excess_vs_random", "net_excess_vs_median"]
        for _, r in v11_cost_summary_df.sort_values("net_cum_ret", ascending=False).head(10)[cols].iterrows():
            lines.append("- `%s` top%d: net %s, mdd %s, win %s, turnover %.3f, excess random %s, excess median %s" % (
                r["model_name"], int(r["top_n"]), fmt_pct(r["net_cum_ret"]), fmt_pct(r["net_max_drawdown"]), fmt_pct(r["net_win_rate"]),
                float(r["avg_turnover_traded_notional"]), fmt_pct(r["net_excess_vs_random"]), fmt_pct(r["net_excess_vs_median"])
            ))
    else:
        lines.append("- No cost-aware summary generated.")
    lines.append("")
    lines.append("## Common OOS")
    lines.append("")
    if "v11_common_oos_df" in globals() and v11_common_oos_df is not None and not v11_common_oos_df.empty:
        cols = ["common_oos_start", "model_name", "top_n", "common_weeks", "net_cum_ret", "net_max_drawdown", "random_cum_ret", "median_cum_ret"]
        for _, r in v11_common_oos_df.head(20)[cols].iterrows():
            lines.append("- `%s` `%s` top%d weeks=%d: net %s, mdd %s, random %s, median %s" % (
                r["common_oos_start"], r["model_name"], int(r["top_n"]), int(r["common_weeks"]),
                fmt_pct(r["net_cum_ret"]), fmt_pct(r["net_max_drawdown"]), fmt_pct(r["random_cum_ret"]), fmt_pct(r["median_cum_ret"])
            ))
    else:
        lines.append("- No common OOS rows generated.")
    lines.append("")
    lines.append("## Stability")
    lines.append("")
    if "v11_cost_stability_df" in globals() and v11_cost_stability_df is not None and not v11_cost_stability_df.empty:
        for _, r in v11_cost_stability_df.head(12).iterrows():
            lines.append("- top%d `%s` vs `%s`: periods=%d, overlap=%.3f, ret_corr=%s" % (
                int(r["top_n"]), r["model_a"], r["model_b"], int(r["common_periods"]), float(r["target_overlap_rate"]),
                "nan" if pd.isnull(r["ret_corr"]) else "%.3f" % float(r["ret_corr"])
            ))
    else:
        lines.append("- No stability rows generated.")
    lines.append("")
    lines.append("## Next Experiment Recommendation")
    lines.append("")
    lines.append("- If top3 beats median/random in common OOS after cost, test portfolio rules next: top5, score-gap filter, weak-signal cash, and training-window ensemble.")
    lines.append("- If common OOS is weak or unstable, do not add features yet; first inspect drawdown events and ETF theme concentration.")
    lines.append("- If names/groups are missing, fix ETF metadata before making theme-cap conclusions.")
    text = "\\n".join(lines) + "\\n"
    with open(V11_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(text)
    return text


v11_report_text = build_v11_report()
print("V11 report:", V11_REPORT_MD)
print(v11_report_text[:2000])
print("V11 outputs:")
for p in [V11_COST_WEEKLY_CSV, V11_COST_SUMMARY_CSV, V11_COMMON_OOS_CSV, V11_STABILITY_CSV, V11_REPORT_MD]:
    print(" -", p)
'''


def main():
    matches = glob.glob(r"G:\ETF_V10_recent_window_stability*.ipynb")
    if not matches:
        raise FileNotFoundError("Cannot find G:\\ETF_V10_recent_window_stability*.ipynb")
    with open(matches[0], "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if "source" in cell:
            cell["source"] = source(patch_v10_names("".join(cell.get("source", []))))
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    nb["cells"][0] = markdown_cell(
        """
# ETF_V11 2021 Validation Framework

## 目的

验证 ETF 机器学习轮动策略在 2021 年之后的 ETF 生态中是否具备可用性、稳定性和鲁棒性。

本 notebook 以 V10 为基础，但只保留固定 2021 起点训练窗口，并新增：

- `top1 / top3 / top5`
- 成本按万分之一，即 `COST_RATE = 0.0001`
- 换手与成本后收益
- 多 seed random topN
- 简单 `ret_20` / `trend_score_25` baseline
- 共同 OOS 比较
- 训练窗口间持仓重合率和收益相关性

第一阶段目标是验证策略是否可信，不做新特征和模型调参。
"""
    )

    nb["cells"][1]["source"] = source(patch_config_cell("".join(nb["cells"][1]["source"])))

    insert_at = 7
    nb["cells"].insert(insert_at, code_cell(V11_COST_AND_COMMON_OOS))
    nb["cells"].insert(insert_at + 1, code_cell(V11_REPORT))

    OUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(OUT_NOTEBOOK)
    print("cells", len(nb["cells"]))


if __name__ == "__main__":
    main()
