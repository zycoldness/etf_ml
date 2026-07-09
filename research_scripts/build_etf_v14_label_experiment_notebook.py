import copy
import json
import os
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_NOTEBOOK = PROJECT_DIR / "notebooks" / "ETF_V14_label_experiment.ipynb"

V10_SOURCE_CANDIDATES = [
    Path(os.environ["ETF_V10_SOURCE_NOTEBOOK"]) if os.environ.get("ETF_V10_SOURCE_NOTEBOOK") else None,
    Path("G:/ETF_V10_recent_window_stability实验.ipynb"),
    PROJECT_DIR.parent / ".tmp" / "ETF_V10_source.ipynb",
    PROJECT_DIR / "notebooks" / "ETF_V10_recent_window_stability实验.ipynb",
]


def read_v10_source():
    for path in V10_SOURCE_CANDIDATES:
        if path is not None and path.exists():
            print("V10 source:", path)
            return json.loads(path.read_text(encoding="utf-8"))
    checked = [str(p) for p in V10_SOURCE_CANDIDATES if p is not None]
    raise FileNotFoundError(
        "Missing V10 source notebook. Set ETF_V10_SOURCE_NOTEBOOK or place the V10 notebook at one of: %s"
        % checked
    )


def replace_block(src, start_pattern, end_pattern, replacement):
    m = re.search(start_pattern + r".*?" + end_pattern, src, flags=re.S)
    if not m:
        raise ValueError("pattern not found: %s ... %s" % (start_pattern, end_pattern))
    return src[:m.start()] + replacement + src[m.end():]


def patch_markdown(src):
    src = src.replace("ETF_V10 Recent Window Stability 实验", "ETF_V14 Label Experiment (V10-aligned)")
    src = src.replace("V10", "V14")
    src = src.replace("etf_ml_v10_", "etf_ml_v14_")
    src = src.replace("ret5d", "ret5d / alpha5d / rank5d / ret10d / alpha10d / rank10d")
    src = src.replace("top1/top3", "top3")
    src = src.replace("rebalance: weekly", "rebalance: label-matched weekly cadence")
    return src + (
        "\n\n**Data alignment rule:** this notebook is generated from the V10 notebook and keeps the V10 "
        "ETF universe, feature engineering, filtering,复权,停牌处理 and weekly-date logic. V14 only changes "
        "label horizons/targets and the evaluation cadence for 10-day labels.\n"
    )


def patch_config_cell(src):
    src = src.replace("ETF_V10", "ETF_V14").replace("V10", "V14")
    src = src.replace("etf_ml_v10_recent_window_stability_outputs", "etf_ml_v14_label_experiment_outputs")
    src = src.replace("etf_ml_v10_", "etf_ml_v14_")
    src = src.replace("model_etf_ml_v10_", "model_etf_ml_v14_")
    src = src.replace("LABEL_HORIZONS = [5]", "LABEL_HORIZONS = [5, 10]")
    src = src.replace("TOP_N_LIST = [1, 3]", "TOP_N_LIST = [3]")
    src = src.replace(
        'GROUP_CAP_LIST = [0]  # No theme cap. Current V2-like ETF pool keeps duplicate themes by design.',
        'GROUP_CAP_LIST = [0]  # Keep V10 no-theme-cap setting; V14 fixes top3 only.',
    )
    src = re.sub(
        r"TARGET_SPECS = \[.*?\]\n\nTOP_N_LIST",
        """TARGET_SPECS = [
    # V14 changes labels only. Data construction remains V10-aligned.
    ("ret5d", "future_ret_5d", "future_ret_5d"),
    ("alpha5d", "target_alpha_5d", "future_ret_5d"),
    ("rank5d", "target_rank_5d", "future_ret_5d"),
    ("ret10d", "future_ret_10d", "future_ret_10d"),
    ("alpha10d", "target_alpha_10d", "future_ret_10d"),
    ("rank10d", "target_rank_10d", "future_ret_10d"),
]

TOP_N_LIST""",
        src,
        flags=re.S,
    )
    src = re.sub(
        r'TARGET_REBALANCE_RULES = \{.*?\}\n\nBASE_PRICE_FEATURE_COLS',
        '''TARGET_REBALANCE_RULES = {
    "ret3d": {"rebalance_mode": "weekly", "rebalance_interval_weeks": 1},
    "ret5d": {"rebalance_mode": "weekly", "rebalance_interval_weeks": 1},
    "ret10d": {"rebalance_mode": "weekly", "rebalance_interval_weeks": 2},
    "ret20d": {"rebalance_mode": "monthly", "rebalance_interval_weeks": 4},
    "alpha5d": {"rebalance_mode": "weekly", "rebalance_interval_weeks": 1},
    "alpha10d": {"rebalance_mode": "weekly", "rebalance_interval_weeks": 2},
    "alpha20d": {"rebalance_mode": "monthly", "rebalance_interval_weeks": 4},
    "rank5d": {"rebalance_mode": "weekly", "rebalance_interval_weeks": 1},
    "rank10d": {"rebalance_mode": "weekly", "rebalance_interval_weeks": 2},
}

BASE_PRICE_FEATURE_COLS''',
        src,
        flags=re.S,
    )
    src = src.replace(
        'print("V14 focus: training-window stability only; feature set and label stay fixed.")',
        'print("V14 focus: V10-aligned data with label-only experiment and matched 10d holding period.")',
    )
    src += """

V14_DATA_ALIGNMENT = {
    "source_notebook": "ETF_V10_recent_window_stability",
    "allowed_changes": ["version/output names", "LABEL_HORIZONS", "TARGET_SPECS", "TOP_N_LIST", "10d rebalance cadence"],
    "lookback_days": LOOKBACK_DAYS,
    "min_listing_days": MIN_LISTING_DAYS,
    "price_history_count": PRICE_HISTORY_COUNT,
    "exclude_name_keywords": list(EXCLUDE_NAME_KEYWORDS),
    "trend_windows": list(TREND_WINDOWS),
}
assert LOOKBACK_DAYS == 60
assert MIN_LISTING_DAYS == 180
assert PRICE_HISTORY_COUNT == max(LOOKBACK_DAYS + 1, max(TREND_WINDOWS) + 1)
assert LABEL_HORIZONS == [5, 10]
assert TOP_N_LIST == [3]
print("V14 data alignment guardrails:", V14_DATA_ALIGNMENT)
"""
    return src


def patch_panel_cell(src):
    src = src.replace("ETF_V10", "ETF_V14").replace("V10", "V14")
    src = src.replace("etf_ml_v10_", "etf_ml_v14_")
    src = src.replace("model_etf_ml_v10_", "model_etf_ml_v14_")
    src = re.sub(
        r"def ensure_alpha_columns\(panel\):.*?\n\ndef ensure_horizon_labels",
        """def ensure_alpha_columns(panel):
    out = panel.copy()
    for h in LABEL_HORIZONS:
        ret_col = "future_ret_%sd" % h
        alpha_col = "target_alpha_%sd" % h
        rank_col = "target_rank_%sd" % h
        if ret_col in out.columns and alpha_col not in out.columns:
            out[alpha_col] = out[ret_col] - out.groupby("feature_date")[ret_col].transform("median")
        if ret_col in out.columns and rank_col not in out.columns:
            out[rank_col] = out.groupby("feature_date")[ret_col].rank(pct=True)
    return out


def ensure_horizon_labels""",
        src,
        flags=re.S,
    )
    src = src.replace(
        'print("available target cols:", [c for c in panel_df.columns if c.startswith("future_ret_") or c.startswith("target_alpha_")])',
        'print("available target cols:", [c for c in panel_df.columns if c.startswith("future_ret_") or c.startswith("target_alpha_") or c.startswith("target_rank_")])',
    )
    src += """

print("V14 alignment check: panel was built/loaded through V10 panel functions; label horizons =", LABEL_HORIZONS)
"""
    return src


def patch_code(src):
    src = src.replace("ETF_V10", "ETF_V14").replace("V10", "V14")
    src = src.replace("etf_ml_v10_recent_window_stability_outputs", "etf_ml_v14_label_experiment_outputs")
    src = src.replace("etf_ml_v10_", "etf_ml_v14_")
    src = src.replace("model_etf_ml_v10_", "model_etf_ml_v14_")
    src = src.replace("V14 只验证训练窗口稳定性", "V14 only changes labels while keeping V10 data口径")
    return src


def clear_execution(cell):
    out = copy.deepcopy(cell)
    if out.get("cell_type") == "code":
        out["execution_count"] = None
        out["outputs"] = []
    return out


def build_v14_notebook():
    nb = read_v10_source()
    out = copy.deepcopy(nb)
    for idx, cell in enumerate(out["cells"]):
        cell = clear_execution(cell)
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            if idx == 0:
                src = patch_markdown(src)
            else:
                src = src.replace("V10", "V14").replace("etf_ml_v10_", "etf_ml_v14_")
        elif cell.get("cell_type") == "code":
            if "OUT_DIR =" in src and "LABEL_HORIZONS" in src and "TARGET_SPECS" in src:
                src = patch_config_cell(src)
            elif "def build_weekly_panel" in src and "def ensure_horizon_labels" in src:
                src = patch_panel_cell(src)
            else:
                src = patch_code(src)
        cell["source"] = src.splitlines(True)
        out["cells"][idx] = cell
    out["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    OUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT_NOTEBOOK)
    print("cells", len(out["cells"]))


if __name__ == "__main__":
    build_v14_notebook()
