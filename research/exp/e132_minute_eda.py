#!/usr/bin/env python3
"""
実験132の前段: 分足から作った33列（research/minute_features.py）の EDA（モデルは回さない）。

運用者の依頼（2026-11-19）「取り組む前に、新たに作成した列で、異常値や欠損値などがないかEDAを実施してください」。
学習データ（lab.frame。実験132と同じ行）に、実験132と同じ作りで33列を付けて、次を調べる。

  1. 行の数（全行・分足の期間の行・ラベルのある行・月ごと）
  2. 欠測（期間ごとの付き方・欠測の理由の内訳・月ごとの推移・ラベルのある行での付き方）
  3. 分布（件数・平均・標準偏差・歪度・尖度・最小・0.1/1/5/25/50/75/95/99/99.9% 点・最大）
  4. 異常値（無限・範囲の外・四分位範囲の3倍の外・ちょうど 0 / 1 の割合・終値との差が ±30% を超える件数）
  5. 値の整合（70%範囲の下限 ≦ POC の中点 ≦ 上限・下限 < 上限・時間帯の割合の合計 = 1・
     日足の売買代金 ÷ 出来高（分割・併合を換算）から出した VWAP との突き合わせ = 予測日 D に揃っているか）
  6. 時期による変化（半年ごとの中央値）
  7. 相関（33列どうしで順位相関の絶対値が 0.95 以上の組・既存の373列とのいちばん高い順位相関）
  8. 極端な値の行の傾向（その日の売買代金の順位との関係。薄商いに偏っていないか）

公開ログには件数・割合・分位・相関・日付の範囲だけを出す（銘柄・日付ごとの値は出さない）。分足は J-Quants のアドオンのデータ。
結果の JSON は research/_data/oof/e132eda_summary.json（actions/cache。公開しない）。

    exp=e132_minute_eda.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import minute_features as MF  # noqa: E402
import volume_profile as VP  # noqa: E402
import volume_profile_data as VD  # noqa: E402

MINUTE_DIR = os.path.join(lab.DATA_DIR, "minute")
OUT = os.path.join(lab.DATA_DIR, "oof", "e132eda_summary.json")
BASE_PRESET = "all_plus_prog_listing_vol_book_shape_dd_recent_tx_financial"
QS = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
IQR_K = 3.0
BIG_GAP = 0.30
STATUS_JA = {
    "ok": "集計できた", "calendar_history_missing": "期間の営業日が足りない（分足の始まりの直後）",
    "daily_missing": "日足が無い（上場して間もない等）", "invalid_daily": "日足に欠損・不整合",
    "close_missing": "その日の終値が無い", "share_basis_unknown": "分割・併合の係数が分からない",
    "intraday_missing": "分足が置き場に無い・足りない", "intraday_invalid": "分足に不正な値",
    "daily_totals_mismatch": "分足の合計が日足と合わない", MF.BEFORE_MINUTE: "分足の始まりより前",
}


def section(title: str) -> None:
    print(f"\n■ {title}", flush=True)


def is_share(col: str) -> bool:
    return "_share_" in col


def window_of(col: str) -> int:
    return int(col.rsplit("_", 1)[1][:-1])


# --------------------------------------------------------------------------- #
# 2. 欠測
# --------------------------------------------------------------------------- #
def missing_report(period: pd.DataFrame) -> dict:
    out = {"by_window": {}, "status": {}, "by_month": [], "torn": {}, "labeled": {}}
    n = len(period)
    for w in MF.WINDOWS:
        col = MF.column("vwap_gap", w)
        att = int(period[col].notna().sum())
        out["by_window"][f"{w}d"] = {"attached": att, "rate": att / n if n else float("nan")}
        counts = period[f"vp_minute_status_{w}d"].value_counts()
        out["status"][f"{w}d"] = {k: int(v) for k, v in counts.items()}
        cols = [MF.column(m, w) for m in MF.METRICS]
        have = period[cols].notna().sum(axis=1)
        out["torn"][f"{w}d"] = int(((have > 0) & (have < len(cols))).sum())
        lab_rows = period[period["label"].notna()]
        out["labeled"][f"{w}d"] = {"rows": int(len(lab_rows)), "attached": int(lab_rows[col].notna().sum())}
    month = period["Date"].dt.strftime("%Y-%m")
    for m, g in period.groupby(month):
        out["by_month"].append({"month": m, "rows": int(len(g)),
                                **{f"{w}d": float(g[MF.column('vwap_gap', w)].notna().mean()) for w in MF.WINDOWS}})
    return out


def print_missing(rep: dict, rows: int) -> None:
    for w in MF.WINDOWS:
        b = rep["by_window"][f"{w}d"]
        lab_ = rep["labeled"][f"{w}d"]
        print(f"  {w:>2}営業日: 付いた行 {b['attached']:,} / {rows:,}（{b['rate']*100:.1f}%）・ラベルのある行では "
              f"{lab_['attached']:,} / {lab_['rows']:,}・期間の中で11指標の一部だけ欠ける行 {rep['torn'][f'{w}d']}")
        for k, v in sorted(rep["status"][f"{w}d"].items(), key=lambda kv: -kv[1]):
            print(f"      {STATUS_JA.get(k, k):<34} {v:>7,}（{v / rows * 100:5.1f}%）")
    print("  月ごとの付き方（行数 / 1・5・20営業日の付いた割合）")
    for r in rep["by_month"]:
        print(f"    {r['month']}  {r['rows']:>5,}行  " + "  ".join(f"{r[f'{w}d']*100:5.1f}%" for w in MF.WINDOWS))


# --------------------------------------------------------------------------- #
# 3. 分布 / 4. 異常値
# --------------------------------------------------------------------------- #
def distribution(period: pd.DataFrame) -> dict:
    out = {}
    for c in MF.COLUMNS:
        v = pd.to_numeric(period[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not len(v):
            out[c] = {"n": 0}
            continue
        q = v.quantile(list(QS))
        out[c] = {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()), "skew": float(v.skew()),
                  "kurt": float(v.kurt()), "min": float(v.min()), "max": float(v.max()),
                  **{f"q{p*100:g}": float(q[p]) for p in QS}}
    return out


def outliers(period: pd.DataFrame) -> dict:
    out = {}
    for c in MF.COLUMNS:
        raw = pd.to_numeric(period[c], errors="coerce")
        v = raw.replace([np.inf, -np.inf], np.nan).dropna()
        rec = {"n": int(len(v)), "inf": int(np.isinf(raw.to_numpy(dtype=float)).sum())}
        if len(v):
            q1, q3 = v.quantile([0.25, 0.75])
            iqr = q3 - q1
            rec["iqr_low"] = int((v < q1 - IQR_K * iqr).sum())
            rec["iqr_high"] = int((v > q3 + IQR_K * iqr).sum())
            if is_share(c):
                rec["out_of_range"] = int(((v < -1e-9) | (v > 1 + 1e-9)).sum())
                rec["exact0"] = int((v == 0).sum())
                rec["exact1"] = int((v == 1).sum())
            else:
                rec["out_of_range"] = int((v <= -1).sum())
                rec["big"] = int((v.abs() > BIG_GAP).sum())
        out[c] = rec
    return out


def print_distribution(dist: dict, outs: dict) -> None:
    print(f"  {'列':<36}{'件数':>7}{'平均':>9}{'標準偏差':>9}{'歪度':>7}{'最小':>9}{'0.1%':>9}{'1%':>9}{'5%':>9}"
          f"{'50%':>9}{'95%':>9}{'99%':>9}{'99.9%':>9}{'最大':>9}")
    for c in MF.COLUMNS:
        d = dist[c]
        if not d.get("n"):
            print(f"  {c:<36}{0:>7}")
            continue
        print(f"  {c:<36}{d['n']:>7,}{d['mean']:>9.4f}{d['std']:>9.4f}{d['skew']:>7.2f}{d['min']:>9.4f}"
              f"{d['q0.1']:>9.4f}{d['q1']:>9.4f}{d['q5']:>9.4f}{d['q50']:>9.4f}{d['q95']:>9.4f}{d['q99']:>9.4f}"
              f"{d['q99.9']:>9.4f}{d['max']:>9.4f}")
    section(f"4. 異常値（無限・範囲の外・四分位範囲の{IQR_K:g}倍の外・割合がちょうど 0 / 1・終値との差が ±{BIG_GAP*100:g}% 超）")
    print(f"  {'列':<36}{'無限':>6}{'範囲外':>7}{'IQR下':>7}{'IQR上':>7}{'外の割合':>9}{'=0':>7}{'=1':>7}{'±30%超':>8}")
    for c in MF.COLUMNS:
        o = outs[c]
        if not o["n"]:
            continue
        rate = (o["iqr_low"] + o["iqr_high"]) / o["n"]
        print(f"  {c:<36}{o['inf']:>6}{o['out_of_range']:>7}{o['iqr_low']:>7}{o['iqr_high']:>7}{rate*100:>8.2f}%"
              f"{o.get('exact0', ''):>7}{o.get('exact1', ''):>7}{o.get('big', ''):>8}")


# --------------------------------------------------------------------------- #
# 5. 値の整合
# --------------------------------------------------------------------------- #
def consistency(period: pd.DataFrame) -> dict:
    out = {}
    for w in MF.WINDOWS:
        lo, hi, poc = (period[MF.column(m, w)] for m in ("value_low_gap", "value_high_gap", "poc_gap"))
        ok = lo.notna() & hi.notna() & poc.notna()
        shares = period[[MF.column(f"{b}_share", w) for b in VP.BUCKETS]].sum(axis=1, min_count=4)
        out[f"{w}d"] = {"rows": int(ok.sum()),
                        "poc_outside_value_area": int(((poc < lo - 1e-12) | (poc > hi + 1e-12))[ok].sum()),
                        "low_not_below_high": int((lo >= hi)[ok].sum()),
                        "bucket_sum_max_dev": float((shares - 1).abs().max()) if shares.notna().any() else float("nan")}
    return out


def vwap_check(period: pd.DataFrame, quotes: dict, sessions: list) -> dict:
    """33列の VWAP（(1 + vwap_gap) × 終値）と、日足の売買代金 ÷ 出来高（分割・併合を換算）の比。D に揃っていれば一致する。"""
    out = {}
    iso = period["Date"].dt.strftime("%Y-%m-%d").to_numpy()
    codes = period["Code"].map(VP.normalize_code).to_numpy()
    for w in MF.WINDOWS:
        col = MF.column("vwap_gap", w)
        gaps = period[col].to_numpy(dtype=float)
        rel = []
        shifted = []
        for i in np.flatnonzero(np.isfinite(gaps)):
            daily = quotes.get(codes[i], {})
            days = VP.lookback(sessions, iso[i], w)
            factors = VP.share_factors(days, daily) if days and all(d in daily for d in days) else None
            if factors is None or daily[iso[i]].close is None:
                continue
            traded = [d for d in days if daily[d].valid]
            want = sum(daily[d].turnover for d in traded) / sum(daily[d].volume / factors[d] for d in traded)
            got = (1 + gaps[i]) * daily[iso[i]].close
            rel.append(abs(got / want - 1))
            # 1日ずらした日足（前の営業日）でも同じ比になるなら、揃い方を見分けられていない
            prev = VP.lookback(sessions, iso[i], w + 1)[:-1] if w == 1 else []
            if prev and prev[0] in daily and daily[prev[0]].valid:
                p = daily[prev[0]]
                shifted.append(abs(got / (p.turnover / p.volume) - 1))
        rel = np.array(rel)
        out[f"{w}d"] = {"checked": int(len(rel)), "max_rel_diff": float(rel.max()) if len(rel) else float("nan"),
                        "over_1e-6": int((rel > 1e-6).sum())}
        if shifted:
            out[f"{w}d"]["vs_previous_day_median_rel_diff"] = float(np.median(shifted))
    return out


# --------------------------------------------------------------------------- #
# 6. 時期による変化 / 7. 相関 / 8. 極端な値の行
# --------------------------------------------------------------------------- #
def half_year(dates: pd.Series) -> pd.Series:
    return dates.dt.year.astype(str) + np.where(dates.dt.month <= 6, "H1", "H2")


def drift(period: pd.DataFrame) -> dict:
    h = half_year(period["Date"])
    return {c: {k: float(v) for k, v in period.groupby(h)[c].median().items() if np.isfinite(v)} for c in MF.COLUMNS}


def correlations(period: pd.DataFrame, base: list) -> dict:
    full = period[period[list(MF.COLUMNS)].notna().all(axis=1)]
    out = {"rows": int(len(full)), "pairs_ge_0.95": [], "vs_base": {}}
    if len(full) < 30:
        return out
    ranks = full[list(MF.COLUMNS) + base].rank()
    corr = ranks.corr()
    cols = list(MF.COLUMNS)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if np.isfinite(r) and abs(r) >= 0.95:
                out["pairs_ge_0.95"].append({"a": a, "b": b, "rho": float(r)})
    for c in cols:
        s = corr.loc[c, base].abs().dropna()
        out["vs_base"][c] = {"col": s.idxmax() if len(s) else None, "rho": float(s.max()) if len(s) else float("nan")}
    return out


def extremes(period: pd.DataFrame, turnover: pd.Series) -> dict:
    """四分位範囲の3倍の外の値（7指標）を持つ行が、その日の売買代金の小さい側に偏っていないか（行の中の順位で）。"""
    pr = turnover.rank(pct=True)
    out = {}
    for w in MF.WINDOWS:
        flag = pd.Series(False, index=period.index)
        have = period[MF.column("vwap_gap", w)].notna()
        for m in MF.PROFILE_METRICS:
            v = period[MF.column(m, w)]
            q1, q3 = v.quantile([0.25, 0.75])
            iqr = q3 - q1
            flag |= (v < q1 - IQR_K * iqr) | (v > q3 + IQR_K * iqr)
        flag &= have
        rest = have & ~flag
        out[f"{w}d"] = {"extreme_rows": int(flag.sum()), "rows": int(have.sum()),
                        "turnover_pct_median_extreme": float(pr[flag].median()) if flag.any() else float("nan"),
                        "turnover_pct_median_rest": float(pr[rest].median()) if rest.any() else float("nan"),
                        "extreme_in_bottom20": float((pr[flag] <= 0.2).mean()) if flag.any() else float("nan")}
    return out


def main(argv=None) -> int:
    t0 = time.time()
    print("=" * 78)
    print("実験132の前段: 分足の価格帯別出来高の33列の EDA（学習データの行。モデルは回さない）")
    print("=" * 78)
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"]).astype("datetime64[ns]")
    frame["Code"] = frame["Code"].astype(str)
    base = [c for c in F.columns(BASE_PRESET) if c in frame.columns]
    att, report = MF.build(frame[["Code", "Date"]], MINUTE_DIR, lab.DATA_DIR, with_status=True)
    # 行の並びは frame と同じ（MF.build が保つ）。位置で横に並べる
    df = pd.concat([frame[["Code", "Date", "label"] + base].reset_index(drop=True),
                    att[list(MF.COLUMNS) + MF.STATUS_COLUMNS].reset_index(drop=True)], axis=1)
    start = pd.Timestamp(report["minute_from"])
    period = df[df["Date"] >= start].reset_index(drop=True)
    summary = {"report": report}

    section("1. 行の数")
    print(f"  学習データ {len(df):,}行（{df['Date'].min().date()} 〜 {df['Date'].max().date()}）・キーの重複 "
          f"{int(df.duplicated(['Code', 'Date']).sum())}")
    print(f"  分足 {report['minute_from']} 〜 {report['minute_to']}（{report['minute_days']}日）。その期間の行 {len(period):,}・"
          f"うちラベルのある行 {int(period['label'].notna().sum()):,}・銘柄 {period['Code'].nunique():,}")
    per_day = period.groupby("Date").size()
    print(f"  1日の行数: 中央 {per_day.median():g}（最小 {per_day.min()}・最大 {per_day.max()}）・日数 {len(per_day)}")
    summary["rows"] = {"all": int(len(df)), "period": int(len(period)), "labeled": int(period["label"].notna().sum())}

    section("2. 欠測（分足の期間の行。理由は期間ごとの集計の状態）")
    miss = missing_report(period)
    print_missing(miss, len(period))
    summary["missing"] = miss

    section("3. 分布（付いた行）")
    dist = distribution(period)
    outs = outliers(period)
    print_distribution(dist, outs)
    summary["distribution"] = dist
    summary["outliers"] = outs

    section("5. 値の整合")
    cons = consistency(period)
    for w, c in cons.items():
        print(f"  {w:>3}: 行 {c['rows']:,}・POC が70%範囲の外 {c['poc_outside_value_area']}・下限が上限以上 "
              f"{c['low_not_below_high']}・時間帯の割合の合計と1の差の最大 {c['bucket_sum_max_dev']:.2e}")
    years = sorted({int(d[:4]) for d in period["Date"].dt.strftime("%Y-%m-%d")} | {start.year})
    bars = [os.path.join(lab.DATA_DIR, VD.bars_file(y)) for y in years]
    sessions = MF.trading_days(bars, report["minute_from"])
    days = sorted({d for x in period["Date"].dt.strftime("%Y-%m-%d") for d in VP.lookback(sessions, x, max(MF.WINDOWS))})
    codes = sorted(period["Code"].map(VP.normalize_code).unique())
    quotes = VD.load_quotes(bars, codes, days)
    vw = vwap_check(period, quotes, sessions)
    for w, c in vw.items():
        extra = (f"・1日前の日足と比べたときの差の中央 {c['vs_previous_day_median_rel_diff']*100:.2f}%"
                 if "vs_previous_day_median_rel_diff" in c else "")
        print(f"  {w:>3}: VWAP を日足（売買代金 ÷ 出来高）と突き合わせた行 {c['checked']:,}・相対差の最大 "
              f"{c['max_rel_diff']:.2e}・1e-6 を超えた行 {c['over_1e-6']}{extra}")
    summary["consistency"] = cons
    summary["vwap_check"] = vw

    section("6. 時期による変化（半年ごとの中央値）")
    dr = drift(period)
    halves = sorted({h for c in dr.values() for h in c})
    print(f"  {'列':<36}" + "".join(f"{h:>9}" for h in halves))
    for c in MF.COLUMNS:
        print(f"  {c:<36}" + "".join(f"{dr[c].get(h, float('nan')):>9.4f}" for h in halves))
    summary["drift"] = dr

    section("7. 相関（33列がすべてそろう行。順位相関）")
    cr = correlations(period, base)
    print(f"  行 {cr['rows']:,}・33列どうしで |ρ| ≥ 0.95 の組 {len(cr['pairs_ge_0.95'])}")
    for p in cr["pairs_ge_0.95"]:
        print(f"    {p['a']} ↔ {p['b']}  {p['rho']:+.3f}")
    over = sorted(((c, v) for c, v in cr["vs_base"].items() if np.isfinite(v["rho"])), key=lambda kv: -kv[1]["rho"])
    print(f"  既存の373列といちばん高い |ρ|（上位10列）: " + " / ".join(f"{c}↔{v['col']} {v['rho']:.2f}" for c, v in over[:10]))
    print(f"  |ρ| ≥ 0.9 の列 {sum(1 for _, v in over if v['rho'] >= 0.9)} / {len(over)}・0.95 以上 "
          f"{sum(1 for _, v in over if v['rho'] >= 0.95)}")
    summary["correlations"] = cr

    section(f"8. 極端な値（7指標のどれかが四分位範囲の{IQR_K:g}倍の外）の行と、その日の売買代金の順位（行の中の百分位）")
    iso = period["Date"].dt.strftime("%Y-%m-%d").to_numpy()
    cs = period["Code"].map(VP.normalize_code).to_numpy()
    turnover = pd.Series([quotes.get(c, {}).get(d).turnover if quotes.get(c, {}).get(d) else np.nan
                          for c, d in zip(cs, iso)], index=period.index, dtype=float)
    ex = extremes(period, turnover)
    for w, e in ex.items():
        print(f"  {w:>3}: 極端な値の行 {e['extreme_rows']:,} / {e['rows']:,}・売買代金の百分位の中央 極端 "
              f"{e['turnover_pct_median_extreme']*100:.0f} / ほか {e['turnover_pct_median_rest']*100:.0f}・下位20% に入る割合 "
              f"{e['extreme_in_bottom20']*100:.0f}%")
    summary["extremes"] = ex

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[{time.strftime('%H:%M:%S')}] {time.time()-t0:.0f}秒 / 記録 {OUT}（公開しない）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
