#!/usr/bin/env python3
"""
実験66: 日証金（research/jsf_features.py）の特徴量の EDA。値の確認と、単独の分離力。

運用者の指示（2026-10-09）「先に日証金。新たに追加される特徴量の中身と、値の確認を含む EDA」。
この台本は集計だけを出す（公開ログに銘柄ごとの値は出さない。docs/DATA_JSF.md の利用条件）。
出力は "[json] <節の名前> <JSON>" の行（節ごとに1行）。EDA のページはそこから作る。

  0. 元データの形: 行数・銘柄数・申込日の範囲、列の有無、文言の語彙（制限措置など）
  1. 値の確認
     a. hist と daily（確報）が重なる申込日で、残高が一致する割合（hist の最後の日は速報）
     b. 残高の恒等式: 残高_t − 残高_{t−1} = 新規_t − 返済_t（融資・貸株）、差引 = 融資 − 貸株
     c. 負の値・欠測の割合、daily の申込日の抜け（営業日のうち無い日）
  2. 母集団への付き方: 取引日 T に「T より前の最新の申込日」を付けたときの充足（年・月別）、遅れ（暦日）
  3. 特徴量の分布（分位点）と、既存の列（credit_ratio など）との順位相関
  4. 単独の分離力: ROC-AUC（全体・年別）、十分位ごとの正例率、窓ごとの AUC と上位10% の超過収益（e20.screen）

    exp=e66_jsf_eda.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import build_dataset as B  # noqa: E402
import jsf_features as JF  # noqa: E402
import lab  # noqa: E402
import trading_calendar as TC  # noqa: E402
import walkforward as WF  # noqa: E402
from e20_annual_trajectory import OUTCOME, screen  # noqa: E402
from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS  # noqa: E402

OUT_DIR = os.path.join(lab.DATA_DIR, "oof")
COMPARE_WITH = ["credit_ratio", "short_ratio", "short_ratio_20", "ss_ratio", "ss_chg_20",
                "volume_trend", "log_trading_value", "log_market_cap", "r_high", "vol_20d",
                "ret_20d", "alert_days"]
QS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(name: str, obj) -> None:
    print(f"[json] {name} " + json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_js),
          flush=True)


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return str(x)


def r(x, nd: int = 4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def vocab(s: pd.Series, top: int = 12) -> dict:
    """文言の語彙（件数）。空・「－」は "(空)" にまとめる。"""
    t = s.astype("string").fillna("").str.strip()
    t = t.where(~t.isin(["", "－", "-", "―"]), "(空)")
    vc = t.value_counts().head(top)
    return {str(k): int(v) for k, v in vc.items()}


def quantiles(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return {"n": 0}
    q = s.quantile(QS)
    return {"n": int(len(s)), "mean": r(s.mean()), "sd": r(s.std()),
            **{f"q{int(p * 100):02d}": r(v) for p, v in q.items()}}


def auc(y: np.ndarray, x: np.ndarray):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100 or len(np.unique(y[ok])) < 2 or np.nanstd(x[ok]) == 0:
        return None
    return r(roc_auc_score(y[ok], x[ok]))


def deciles(x: pd.Series, y: pd.Series, k: int = 10) -> list:
    """十分位ごとの件数・正例率（値の重なりで区切りが減ることがある）。"""
    ok = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[ok], y[ok]
    if len(xs) < 200:
        return []
    try:
        bins = pd.qcut(xs.rank(method="first"), k, labels=False)
    except ValueError:
        return []
    out = []
    for b in sorted(pd.unique(bins)):
        m = bins == b
        out.append({"bin": int(b), "n": int(m.sum()), "pos": r(ys[m].mean()),
                    "lo": r(xs[m].min()), "hi": r(xs[m].max())})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=lab.DATA_DIR)
    ap.add_argument("--no-daily", action="store_true", help="hist だけで作る（確認用）")
    args = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    result = {}

    # ---------------- 0. 元データ ---------------- #
    hist_path = os.path.join(args.data_dir, "jsf_hist.parquet")
    bal_path = os.path.join(args.data_dir, "jsf_balance.parquet")
    len_path = os.path.join(args.data_dir, "jsf_lending.parquet")
    if not os.path.exists(hist_path):
        raise SystemExit(f"{hist_path} が無い（Release data-jsf）")
    raw_h = pd.read_parquet(hist_path)
    raw_h["申込日"] = pd.to_datetime(raw_h["申込日"])
    rows_per_code = raw_h.groupby("code").size()
    sec0 = {
        "hist": {"rows": int(len(raw_h)), "codes": int(raw_h["code"].nunique()),
                 "from": str(raw_h["申込日"].min().date()), "to": str(raw_h["申込日"].max().date()),
                 "rows_per_code": {"min": int(rows_per_code.min()), "median": r(rows_per_code.median(), 1),
                                   "max": int(rows_per_code.max())},
                 "columns": list(map(str, raw_h.columns)),
                 "vocab": {c: vocab(raw_h[c]) for c in ("制限措置", "臨時措置", "特別措置", "市場区分", "貸借区分")
                           if c in raw_h.columns},
                 "fetched_days": sorted(set(str(x)[:10] for x in raw_h.get("_fetched_at", pd.Series(dtype=str)).dropna().unique()))[:40]},
    }
    if os.path.exists(bal_path) and not args.no_daily:
        raw_b = pd.read_parquet(bal_path)
        raw_b["申込日"] = pd.to_datetime(raw_b["申込日"])
        sec0["daily"] = {"rows": int(len(raw_b)), "codes": int(raw_b["code"].nunique()),
                         "from": str(raw_b["申込日"].min().date()), "to": str(raw_b["申込日"].max().date()),
                         "app_dates": int(raw_b["申込日"].nunique()),
                         "columns": list(map(str, raw_b.columns)),
                         "vocab": {c: vocab(raw_b[c]) for c in ("取引所区分名", "速報／確報", "上場区分")
                                   if c in raw_b.columns}}
    if os.path.exists(len_path) and not args.no_daily:
        raw_l = pd.read_parquet(len_path)
        raw_l["貸借申込日"] = pd.to_datetime(raw_l["貸借申込日"])
        fee = pd.to_numeric(raw_l.get("当日品貸料率（円）"), errors="coerce")
        sec0["lending"] = {"rows": int(len(raw_l)), "codes": int(raw_l["code"].nunique()),
                           "from": str(raw_l["貸借申込日"].min().date()), "to": str(raw_l["貸借申込日"].max().date()),
                           "rows_with_fee": int((fee > 0).sum()), "columns": list(map(str, raw_l.columns)),
                           "vocab": {c: vocab(raw_l[c]) for c in ("取引所区分", "制限", "応札倍率ランク")
                                     if c in raw_l.columns}}
    emit("source", sec0)
    log(f"hist {sec0['hist']['rows']:,}行・{sec0['hist']['codes']:,}銘柄 / daily "
        f"{sec0.get('daily', {}).get('rows', 0):,}行")

    # ---------------- 1. 値の確認 ---------------- #
    hist = JF.load_hist(hist_path)
    daily = JF.load_daily(bal_path, len_path) if ("daily" in sec0) else None
    cal = TC.load(args.data_dir)
    sec1 = {}
    if daily is not None:
        both = hist.merge(daily, on=["app_date", "Code"], suffixes=("_h", "_d"))
        last_h = hist.groupby("Code")["app_date"].transform("max") == hist["app_date"]
        both["_last"] = both.set_index(["app_date", "Code"]).index.isin(
            hist.loc[last_h].set_index(["app_date", "Code"]).index)
        def agree(sub: pd.DataFrame) -> dict:
            o = {"n": int(len(sub))}
            for k in ("loan_bal", "stock_bal", "loan_new", "stock_new"):
                a, b = sub[f"{k}_h"], sub[f"{k}_d"]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum():
                    o[k] = {"equal": r((a[ok] == b[ok]).mean()),
                            "rel_diff_median": r(((a[ok] - b[ok]).abs() / (b[ok].abs() + 1)).median()),
                            "rel_diff_q95": r(((a[ok] - b[ok]).abs() / (b[ok].abs() + 1)).quantile(0.95))}
            return o
        sec1["hist_vs_daily"] = {"overlap_rows": int(len(both)),
                                 "overlap_dates": int(both["app_date"].nunique()),
                                 "all": agree(both),
                                 "hist_last_day(速報)": agree(both[both["_last"]]),
                                 "hist_not_last(確報)": agree(both[~both["_last"]])}
    # 恒等式（hist）
    h = hist.sort_values(["Code", "app_date"])
    prev = h.groupby("Code")[["loan_bal", "stock_bal"]].shift(1)
    d_loan = h["loan_bal"] - prev["loan_bal"]
    d_stock = h["stock_bal"] - prev["stock_bal"]
    e_loan = h["loan_new"].fillna(0) - h["loan_ret"].fillna(0)
    e_stock = h["stock_new"].fillna(0) - h["stock_ret"].fillna(0)
    ok_l = np.isfinite(d_loan) & np.isfinite(e_loan)
    ok_s = np.isfinite(d_stock) & np.isfinite(e_stock)
    net_ok = np.isfinite(h["net_bal"]) & np.isfinite(h["loan_bal"]) & np.isfinite(h["stock_bal"])
    sec1["identity"] = {
        "loan: Δ残高 = 新規−返済": {"n": int(ok_l.sum()), "exact": r((d_loan[ok_l] == e_loan[ok_l]).mean()),
                                  "within_1pct": r(((d_loan[ok_l] - e_loan[ok_l]).abs() <= 0.01 * (h["loan_bal"][ok_l].abs() + 1)).mean())},
        "stock: Δ残高 = 新規−返済": {"n": int(ok_s.sum()), "exact": r((d_stock[ok_s] == e_stock[ok_s]).mean()),
                                   "within_1pct": r(((d_stock[ok_s] - e_stock[ok_s]).abs() <= 0.01 * (h["stock_bal"][ok_s].abs() + 1)).mean())},
        "net = loan − stock": {"n": int(net_ok.sum()),
                               "exact": r((h["net_bal"][net_ok] == (h["loan_bal"] - h["stock_bal"])[net_ok]).mean())},
    }
    sec1["negative_or_missing(hist)"] = {
        k: {"missing": r(h[k].isna().mean()), "negative": r((h[k] < 0).mean()), "zero": r((h[k] == 0).mean())}
        for k in ("loan_bal", "stock_bal", "loan_new", "stock_new", "fee_ann", "price")}
    sec1["fee(hist)"] = {"rows_with_fee": int((h["fee_ann"] > 0).sum()),
                         "share_rows_with_fee": r((h["fee_ann"] > 0).mean()),
                         "codes_with_any_fee": int(h.loc[h["fee_ann"] > 0, "Code"].nunique()),
                         "fee_ann_quantiles(when>0)": quantiles(h.loc[h["fee_ann"] > 0, "fee_ann"])}
    sec1["restrict(hist)"] = {str(int(k)) if np.isfinite(k) else "nan": int(v)
                              for k, v in h["restrict"].value_counts(dropna=False).items()}
    if daily is not None:
        dd = sorted(daily["app_date"].unique())
        days = [d.date() for d in pd.to_datetime(dd)]
        n_trading = cal.count_between(days[0], days[-1]) if cal and len(days) > 1 else None
        sec1["daily_dates"] = {"app_dates": len(days), "from": str(days[0]), "to": str(days[-1]),
                               "trading_days_in_span": n_trading,
                               "missing_days": (None if n_trading is None else int(n_trading + 1 - len(days)))}
        sec1["daily_fee"] = {"rows_with_fee": int((daily["fee_ann"] > 0).sum()),
                             "fee_ann_quantiles(when>0)": quantiles(daily.loc[daily["fee_ann"] > 0, "fee_ann"])}
    emit("checks", sec1)
    log("値の確認を出した")

    # ---------------- 2. 母集団への付き方 ---------------- #
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    p = JF.panel(hist, daily)
    vol = JF.avg_volume(JF.load_bars(args.data_dir, codes=p["Code"].unique()))
    feats = JF.features_on_panel(p, vol)
    df = JF.attach(frame, feats)
    have = df["jsf_ratio"].notna()
    yr = df["Date"].dt.year
    ym = df["Date"].dt.strftime("%Y-%m")
    since = df["Date"] >= pd.Timestamp("2023-10-01")
    sec2 = {
        "rows": int(len(df)), "covered": int(have.sum()), "covered_share": r(have.mean()),
        "rows_since_2023_10": int(since.sum()), "covered_share_since_2023_10": r(have[since].mean()),
        "by_year": {str(y): {"rows": int((yr == y).sum()), "covered": int((have & (yr == y)).sum())}
                    for y in sorted(yr.unique())},
        "by_month_since": {m: {"rows": int((ym == m).sum()), "covered": int((have & (ym == m)).sum())}
                           for m in sorted(ym[since].unique())},
        "lag_days": {str(int(k)): int(v) for k, v in df.loc[have, "jsf_lag"].value_counts().sort_index().items()},
        "label_rate": {"covered": r(df.loc[have, "label"].mean()), "uncovered_since_2023_10": r(df.loc[since & ~have, "label"].mean()),
                       "all": r(df["label"].mean())},
        "codes_covered": int(df.loc[have, "Code"].nunique()),
        "codes_since_2023_10": int(df.loc[since, "Code"].nunique()),
        "panel": {"rows": int(len(p)), "codes": int(p["Code"].nunique()),
                  "from": str(p["app_date"].min().date()), "to": str(p["app_date"].max().date()),
                  "source": {str(k): int(v) for k, v in p["source"].value_counts().items()}},
        "avg_vol_missing_on_panel": r(feats["jsf_loan_v"].isna().mean()),
    }
    emit("coverage", sec2)
    log(f"母集団 {len(df):,}件 / 付いた行 {have.sum():,}件（{have.mean()*100:.1f}%。2023-10 以降では "
        f"{have[since].mean()*100:.1f}%）")

    # ---------------- 3. 分布と相関 ---------------- #
    cols = JF.columns("all")
    sub = df[have]
    sec3 = {"quantiles": {c: quantiles(sub[c]) for c in cols},
            "missing_share_on_covered": {c: r(sub[c].isna().mean()) for c in cols},
            "fee_share": r((sub["jsf_fee"] > 0).mean()),
            "restrict_levels": {str(int(k)) if np.isfinite(k) else "nan": int(v)
                                for k, v in sub["jsf_restrict"].value_counts(dropna=False).items()}}
    comp = [c for c in COMPARE_WITH if c in sub.columns]
    corr = {}
    for c in cols:
        corr[c] = {}
        for k in comp:
            x, y = sub[c], pd.to_numeric(sub[k], errors="coerce")
            ok = np.isfinite(x) & np.isfinite(y)
            corr[c][k] = r(x[ok].corr(y[ok], method="spearman"), 3) if ok.sum() > 300 else None
    sec3["spearman_with_existing"] = corr
    ok = np.isfinite(sub["jsf_ratio"]) & np.isfinite(pd.to_numeric(sub.get("credit_ratio"), errors="coerce"))
    if ok.sum() > 300:
        sec3["ratio_vs_credit_ratio"] = {
            "n": int(ok.sum()),
            "spearman": r(sub.loc[ok, "jsf_ratio"].corr(np.log(pd.to_numeric(sub.loc[ok, "credit_ratio"], errors="coerce").clip(lower=1e-3)), method="spearman"), 3),
            "pearson_log": r(sub.loc[ok, "jsf_ratio"].corr(np.log(pd.to_numeric(sub.loc[ok, "credit_ratio"], errors="coerce").clip(lower=1e-3))), 3)}
    emit("distribution", sec3)
    log("分布と相関を出した")

    # ---------------- 4. 単独の分離力 ---------------- #
    y_all = sub["label"].to_numpy(dtype=float)
    years = sorted(int(v) for v in sub["Date"].dt.year.unique())
    sec4 = {"auc": {}, "deciles": {}}
    for c in cols:
        x = sub[c].to_numpy(dtype=float)
        sec4["auc"][c] = {"all": auc(y_all, x),
                          **{str(y): auc(y_all[(sub["Date"].dt.year == y).to_numpy()],
                                         x[(sub["Date"].dt.year == y).to_numpy()]) for y in years}}
        sec4["deciles"][c] = deciles(sub[c], sub["label"])
    folds = WF.make_folds(df["Date"], min_train_months=OOF_MIN_TRAIN_MONTHS, test_months=OOF_TEST_MONTHS,
                          step_months=OOF_STEP_MONTHS, embargo_days=B.RISE_HORIZON)
    windows = [(np.datetime64(f.test_start), np.datetime64(f.test_end)) for f in folds]
    scr = screen(sub.reset_index(drop=True), cols, windows)
    keep = ["feature", "coverage", "n", "auc_pooled", "auc_win", "auc_se", "auc_win_gt05", "n_win",
            "edge_pt", "edge_se", "edge_z", "edge_win_pos"]
    sec4["screen"] = [{k: (r(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v))
                       for k, v in row.items() if k in keep} for row in scr.to_dict("records")]
    sec4["windows"] = [{"start": str(pd.Timestamp(s).date()), "end": str(pd.Timestamp(e).date()),
                        "rows": int(((sub["Date"] >= s) & (sub["Date"] <= e)).sum())} for s, e in windows]
    sec4["outcome"] = OUTCOME
    emit("signal", sec4)
    log("分離力を出した")

    result = {"source": sec0, "checks": sec1, "coverage": sec2, "distribution": sec3, "signal": sec4,
              "columns": cols, "built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    with open(os.path.join(OUT_DIR, "e66_jsf_eda.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1, default=_js)
    log(f"記録: {os.path.join(OUT_DIR, 'e66_jsf_eda.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
