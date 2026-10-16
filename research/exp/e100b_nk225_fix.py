#!/usr/bin/env python3
"""
実験100b: 日経225 ETF（13210）の未調整の分割を直したら、本番モデルの線の緩さはどれだけ消えるか（検証だけ）

probe_macro_etf（run 38909054969）: 13210 の終値が 2026-10-05 に −99.0%（1:100 の分割）。日次バーは差分で取り足して
いるので、分割の後に J-Quants が付け直した過去の AdjC を手元は持っていない（AdjC ÷ C は全期間で一定）。その結果、
市場環境の列 nk225_ret_20 / nk225_ret_120 は 10/05 以降 −99% で、学習の範囲（最小 −17%）の外の値が全候補に入っている。
実験100（run 38907694687）では lgbm_k15 の TreeSHAP で nk225_ret_120 が +0.138（対数オッズ）を全行に足していた。

やること:
  1. ETF の終値から分割の日と倍率を推定し（1日の変化が 1/2 未満か 2 倍超）、分割前の値を倍率で割って系列をつなぐ。
     直した系列から nk225_ret_20 / nk225_ret_120 を作り直す
  2. 実運用の行（live_features 10/13〜16）の2列を直した値に置き換え、規則の3本で採点し直す → 百分位の分布・3本すべて95以上
  3. 学習の後の期間（dataset_predict、trainTo の翌日〜）も同じく直して週ごとに → 直す前 / 後
  4. 日ごとの「3本すべて95以上」の件数（10/05 以降）を直す前 / 後で並べる（件数だけ）
  5. ほかの銘柄の未調整の分割（2026年、C と AdjC が同じ日に ±50% 超）が、候補の行にいくつ入っているか（件数だけ）

公開ログには件数・割合・特徴量名・ETF の変化率だけ出す（銘柄ごとの値・銘柄名は出さない）。本番は何も変えない。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import e100_line_drift as E  # noqa: E402

NK_CODE = B.MACRO_ETFS["nk225"]
NK_COLS = ("nk225_ret_20", "nk225_ret_120")
log = E.log


def load_bars(data_dir: str, years=(2025, 2026)) -> pd.DataFrame:
    parts = []
    for y in years:
        p = os.path.join(data_dir, f"bars_{y}.parquet")
        if os.path.exists(p):
            parts.append(pd.read_parquet(p, columns=["Date", "Code", "C", "AdjC"]))
    df = pd.concat(parts, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str).str.strip()
    for c in ("C", "AdjC"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def find_splits(s: pd.Series) -> List[Tuple[pd.Timestamp, float]]:
    """1日の変化が 1/2 未満（分割）か 2 倍超（併合）の日と、その倍率（分割前の値 ÷ 分割後の値。分割なら >1）。"""
    r = s / s.shift()
    out = []
    for d, v in r.items():
        if np.isfinite(v) and (v < 0.5 or v > 2.0):
            out.append((pd.Timestamp(d), float(1.0 / v)))
    return out


#: 分割・併合の倍率の候補（日本の上場株・ETF でよくある比率）。終値の変化から推定した倍率を、対数で最も近いものに丸める
NICE_FACTORS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0, 100.0)


def nice_factor(f: float) -> float:
    """推定した倍率（分割前 ÷ 分割後）を候補に丸める。その日の値動き（数%）で 95 と出ても 100 に戻す。"""
    if f >= 1.0:
        return min(NICE_FACTORS, key=lambda c: abs(np.log(f / c)))
    return 1.0 / min(NICE_FACTORS, key=lambda c: abs(np.log((1.0 / f) / c)))


def corrected_series(s: pd.Series) -> Tuple[pd.Series, List[Tuple[pd.Timestamp, float]]]:
    """分割の前の値を倍率（候補に丸める）で割ってつなぐ。倍率は 1:100 なら 100、併合 10:1 なら 0.1。"""
    s = s.sort_index()
    splits = find_splits(s)
    out = s.copy()
    for d, f in splits:
        out.loc[out.index < d] = out.loc[out.index < d] / nice_factor(f)
    return out, splits


def nk_features(bars: pd.DataFrame, code: str = NK_CODE) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """直す前（AdjC）と直した後の nk225_ret_20 / nk225_ret_120（日付ごと）。"""
    sub = (bars[bars["Code"] == code].sort_values("Date").drop_duplicates("Date", keep="last")
           .set_index("Date"))
    raw = sub["AdjC"].fillna(sub["C"])
    fixed, splits = corrected_series(raw)
    before = pd.DataFrame({"Date": raw.index, "nk225_ret_20": raw.pct_change(20).to_numpy() * 100,
                           "nk225_ret_120": raw.pct_change(120).to_numpy() * 100})
    after = pd.DataFrame({"Date": fixed.index, "nk225_ret_20": fixed.pct_change(20).to_numpy() * 100,
                          "nk225_ret_120": fixed.pct_change(120).to_numpy() * 100})
    return before, after, splits


def with_fixed(rows: pd.DataFrame, after: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """行の nk225 の2列を、直した系列のその日の値に置き換える。"""
    out = rows.copy()
    m = out[["Date"]].merge(after, on="Date", how="left")
    for c in NK_COLS:
        if c in cols:
            out[c] = m[c].to_numpy()
    return out


def daily_pass(rows: pd.DataFrame, pcts: Dict[str, np.ndarray]) -> pd.DataFrame:
    d = rows[["Date"]].copy()
    d["pass"] = E.pass_mask(pcts)
    d["ge95"] = np.asarray(pcts[E.MAIN], dtype=float) >= E.LINE
    g = d.groupby("Date").agg(n=("pass", "size"), ge95=("ge95", "sum"), passed=("pass", "sum")).reset_index()
    return g


def unadjusted_splits(bars: pd.DataFrame, year: int = 2026) -> Dict[str, pd.Timestamp]:
    """C と AdjC が同じ日に同じ向きに ±50% 超で動いた銘柄（分割・併合が AdjC に未調整の疑い）と、その日。"""
    b = bars[bars["Date"] >= pd.Timestamp(f"{year}-01-01")].sort_values(["Code", "Date"])
    la = np.log(b["AdjC"]).groupby(b["Code"]).diff()
    lc = np.log(b["C"]).groupby(b["Code"]).diff()
    both = (la.abs() > np.log(1.5)) & (lc.abs() > np.log(1.5)) & (np.sign(la) == np.sign(lc))
    hit = b.loc[both.fillna(False).to_numpy(), ["Code", "Date"]]
    return {c: pd.Timestamp(d) for c, d in hit.groupby("Code")["Date"].min().items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="日経225 ETF の未調整の分割を直して採点し直す")
    ap.add_argument("--data-dir", default=E.DATA_DIR)
    ap.add_argument("--live-from", default=E.LIVE_FROM)
    ap.add_argument("--live-to", default=E.LIVE_TO)
    args = ap.parse_args(argv)

    cols = E.features_of(E.MAIN)
    models = {a: E.load_model(a, args.data_dir) for a in E.RULE_MODELS}
    oof = E.load_oof_wide(args.data_dir)
    refs = {a: oof[f"s_{a}"].to_numpy(dtype=float) for a in E.RULE_MODELS}
    bars = load_bars(args.data_dir)
    before, after, splits = nk_features(bars)

    print("=" * 100)
    print(f"実験100b: 日経225 ETF（{NK_CODE}）の未調整の分割を直したら（規則の3本 {' / '.join(E.RULE_MODELS)}、線 {E.LINE:g}）")
    print("=" * 100)
    print("\n■ 1. ETF の系列")
    if not splits:
        print("  分割・併合の段差は見つからない（直すものが無い）")
    for d, f in splits:
        print(f"  {d.date()}: 分割前 ÷ 分割後 = {f:.1f}（{nice_factor(f):g} に丸める）")
    ab = before.merge(after, on="Date", suffixes=("_before", "_after"))
    tail = ab[ab["Date"] >= pd.Timestamp("2026-09-25")]
    print("  nk225_ret_20 / ret_120（直す前 → 後）:")
    for r in tail.to_dict("records"):
        print(f"    {pd.Timestamp(r['Date']).date()}  ret_20 {r['nk225_ret_20_before']:+7.1f}% → {r['nk225_ret_20_after']:+6.1f}%   "
              f"ret_120 {r['nk225_ret_120_before']:+7.1f}% → {r['nk225_ret_120_after']:+6.1f}%")

    # §2 実運用の行
    live = E.live_rows(os.path.join(args.data_dir, "live_features.parquet"), cols, args.live_from, args.live_to)
    X0 = live[cols].to_numpy(dtype=float)
    live_fix = with_fixed(live, after, cols)
    X1 = live_fix[cols].to_numpy(dtype=float)
    p0 = E.pct_all(E.score_all(models, X0), refs)
    p1 = E.pct_all(E.score_all(models, X1), refs)
    print(f"\n■ 2. 実運用の行（{args.live_from}〜{args.live_to}、{len(live)}件）を直して採点し直す")
    E.print_dist("直す前（画面のまま）", E.summary_of(p0))
    E.print_dist("nk225 の2列を直した後", E.summary_of(p1))
    for a in E.RULE_MODELS:
        d0, d1 = E.dist(p0[a]), E.dist(p1[a])
        print(f"    {a:9s} 中央 {d0['median']:.1f} → {d1['median']:.1f}、95以上 {d0['ge95']:.1f}% → {d1['ge95']:.1f}%")
    # 行ごとの差
    diff = np.asarray(p1[E.MAIN], dtype=float) - np.asarray(p0[E.MAIN], dtype=float)
    print(f"  {E.MAIN} の百分位の差（後 − 前）: 中央 {np.median(diff):+.1f}pt、平均 {diff.mean():+.1f}pt、最小 {diff.min():+.1f}、最大 {diff.max():+.1f}")

    # §3 学習後の期間（学習側の特徴量）
    pred_path = E.ensure_predict_dataset(os.path.join(args.data_dir, "dataset_predict.parquet"), args.data_dir)
    pr = pd.read_parquet(pred_path, columns=["Code", "Date"] + cols)
    pr["Code"] = pr["Code"].astype(str)
    pr["Date"] = pd.to_datetime(pr["Date"])
    train_to = oof["Date"].max()
    pr = pr[pr["Date"] > train_to].reset_index(drop=True)
    q0 = E.pct_all(E.score_all(models, pr[cols].to_numpy(dtype=float)), refs)
    pr_fix = with_fixed(pr, after, cols)
    q1 = E.pct_all(E.score_all(models, pr_fix[cols].to_numpy(dtype=float)), refs)
    w0, w1 = E.weekly_table(pr, q0), E.weekly_table(pr, q1)
    print(f"\n■ 3. 学習の後の期間（{train_to.date()} の翌日〜、学習側の特徴量）: 週ごと（直す前 → 後）")
    print(f"  {'週（月曜）':12s}{'件数':>6s}{'中央% 前':>9s}{'後':>7s}{'L15≥95 前':>11s}{'後':>7s}{'3本≥95 前':>11s}{'後':>7s}")
    for a, b in zip(w0.to_dict("records"), w1.to_dict("records")):
        print(f"  {str(a['week']):12s}{int(a['n']):>6d}{a['median']:>9.1f}{b['median']:>7.1f}{a['ge95']:>11.1f}{b['ge95']:>7.1f}"
              f"{a['pass']:>11.1f}{b['pass']:>7.1f}")

    # §4 日ごとの件数（分割の日以降）
    first_split = min((d for d, _ in splits), default=pd.Timestamp("2026-10-05"))
    d0, d1 = daily_pass(pr, q0), daily_pass(pr, q1)
    dd = d0.merge(d1, on="Date", suffixes=("_before", "_after"))
    dd = dd[dd["Date"] >= first_split - pd.Timedelta(days=7)]
    print(f"\n■ 4. 日ごとの件数（{first_split.date()} の1週間前から。発火 = 候補の数）")
    print(f"  {'選定日':12s}{'発火':>5s}{'L15≥95 前':>11s}{'後':>5s}{'3本≥95 前':>11s}{'後':>5s}")
    for r in dd.to_dict("records"):
        print(f"  {pd.Timestamp(r['Date']).date()!s:12s}{int(r['n_before']):>5d}{int(r['ge95_before']):>11d}{int(r['ge95_after']):>5d}"
              f"{int(r['passed_before']):>11d}{int(r['passed_after']):>5d}")

    # §5 ほかの銘柄の未調整の分割が候補に入っているか
    un = unadjusted_splits(bars)
    cand = pr[["Code", "Date"]].copy()
    cand["split_day"] = cand["Code"].map(un)
    hit = cand[cand["split_day"].notna() & (cand["Date"] >= cand["split_day"])]
    hit_live = hit[(hit["Date"] >= pd.Timestamp(args.live_from)) & (hit["Date"] <= pd.Timestamp(args.live_to))]
    print(f"\n■ 5. ほかの銘柄の未調整の分割・併合（2026年、C と AdjC が同じ日に ±50% 超）: {len(un)}銘柄。"
          f"学習後の候補の行のうち、その銘柄の段差の後の行 {len(hit)}件 / {len(pr)}件"
          f"（実運用の期間 {len(hit_live)}件 / {len(live)}件）")
    print("\n■ 注意")
    print("  ・倍率は終値の変化から推定した（整数に丸める）。J-Quants の AdjFactor を取っていないため。本番の直し方は別に決める")
    print("  ・直したのは nk225 の2列だけ。ほかの銘柄の未調整の段差は、その銘柄の列（株価位置・リターン）に残る")
    return 0


if __name__ == "__main__":
    sys.exit(main())
