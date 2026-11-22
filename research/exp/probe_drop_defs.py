#!/usr/bin/env python3
"""
「下がった」の定義の候補ごとに、該当する銘柄が1日に何件出るか（反発を狙う戦略の準備）。

運用者の依頼（2026-11-22）「下がった銘柄の反発を狙う戦略を立てたいです。まずは「下がった」の定義ですね。
反発を狙うので、上昇トレンド下降トレンドどちらもありえると思います。」

定義を決める前なので、**事後のリターン（反発したか）は一切見ない**。見るのは件数・トレンド別の内訳・
特定の日への集中・決算の近さ・定義どうしの重なりだけ（docs/MODEL_ADOPTION_RULES.md の冒頭
「実験の結果を見てから規則を変えると、結果に合う規則を選ぶことになる」）。

母集団（ブレイク予測と同じ。research/build_dataset.py）
  - 市場区分（月末ごとの master_hist を時点でさかのぼって当てる）が「その他」（ETF・REIT 等）・TOKYO PRO MARKET でない
  - 普通株（5桁コードの末尾が 0）
  - 20営業日平均の売買代金が 0.1億円（1,000万円）以上
  - 価格は調整済み終値（AdjC）

候補の定義（その日の終値で判定）
  dd20_X   直近20営業日の終値の最高値から −X% 以上下（高値からの下落率）
  dd20rel_X  株価 ÷ TOPIX の比が、直近20営業日の最高値から −X% 以上下（市場全体の下げを除いた、個別の下げ）
  dd20s_K  上の下落を、下落前のボラ（t−80〜t−21 の日次対数リターンの標準偏差 × √20）で割って −Kσ 以下
  r5_X     5営業日の騰落率が −X% 以下（下げの速さ）
  ma25_X   25日移動平均からの乖離率が −X% 以下
  down5    5日以上続けて前日比マイナス
  rsi14    RSI(14, Wilder) が 30 以下
  bb2      終値が 20日移動平均 − 2×標準偏差 以下（ボリンジャーバンド −2σ）
  r1_X     1日の騰落率が −X% 以下（急落）

同じ銘柄が続けて該当するので、該当した日から20営業日は同じ銘柄を数えない（ブレイク予測の BREAKOUT_COOLDOWN と同じ）。

トレンドの分け方（下げる前の t−20 で測る。下げた後に測ると、下げそのものが線を割らせて「下降」に寄る）
  上昇: t−20 の終値 > 200日線、かつ 200日線が20営業日前より上
  下降: t−20 の終値 < 200日線、かつ 200日線が20営業日前より下
  混在: どちらでもない / 不明: 200日線が引けない（上場から日が浅い）

公開ログには定義ごとの集計と、集中の大きい日付（その日の件数だけ）を出す。銘柄ごとの値は出さない。

    python3 research/exp/probe_drop_defs.py [--start 2018-01-01] [--out summary.json]

結果（2026-11-22、手元で実行、20秒。docs/MODEL_REBOUND_DROP.md）
------------------------------------------------------------------
  2018-01-04〜2026-11-20 の 2,167営業日、1日の母集団 平均 2,819銘柄
  dd20_15: 62,908件・1日平均 29.0件（中央値 16）・0件の日 0.2%・上昇 41% / 下降 34%・上位1%の日に 15.9%・決算の前後 17.3%
  市場の急落の日への集中: dd20s_2.5 35.4%・ma25_15 36.8% と大きく、dd20rel_15（TOPIX比）は 6.0%
  下げる前に上昇だった件のうち、該当日に200日線の下にいたのは dd20_15 で 39.5%（下げた日に測ると「下降」に寄る）
  運用者の選択（同日）: 20日高値から −15%・素の株価で判定し TOPIX の下げは別の列・トレンドは t−20 で測る
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_data")
EXCLUDED_MARKETS = {"その他", "TOKYO PRO MARKET"}
MIN_VA = 1e7          # 20日平均売買代金の下限（円）= 0.1億円
COOLDOWN = 20
EARN_WINDOW = 5       # 決算予定日が t−4〜t に入るか


def load_wide():
    paths = sorted(glob.glob(os.path.join(DATA, "bars_*.parquet")))
    bars = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "AdjC", "Va"]) for p in paths],
                     ignore_index=True)
    bars["Code"] = bars["Code"].astype(str)
    bars = bars.drop_duplicates(["Date", "Code"], keep="last")
    close = bars.pivot(index="Date", columns="Code", values="AdjC").sort_index()
    va = bars.pivot(index="Date", columns="Code", values="Va").reindex_like(close)
    return close, va


def market_ok(close: pd.DataFrame) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(DATA, "master_hist_*.parquet")))
    mh = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "MktNm"]) for p in paths], ignore_index=True)
    mh["Code"] = mh["Code"].astype(str)
    mh["Date"] = pd.to_datetime(mh["Date"])
    ok = (~mh["MktNm"].isin(EXCLUDED_MARKETS)).astype(float)
    wide = (mh.assign(ok=ok).drop_duplicates(["Date", "Code"], keep="last")
              .pivot(index="Date", columns="Code", values="ok"))
    idx = close.index.union(wide.index)
    # 月末の区分を次の月末まで使い（ffill）、最初の月末より前は最初の区分を当てる（bfill）
    wide = wide.reindex(idx).ffill().bfill().reindex(close.index)
    wide = wide.reindex(columns=close.columns)
    return wide.eq(1.0)


def load_topix(index: pd.Index) -> pd.Series:
    paths = sorted(glob.glob(os.path.join(DATA, "topix_*.parquet")))
    t = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    t["Date"] = pd.to_datetime(t["Date"])
    return t.drop_duplicates("Date", keep="last").set_index("Date")["topix"].reindex(index).ffill()


def rsi_wilder(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def down_streak(close: pd.DataFrame) -> np.ndarray:
    down = (close < close.shift(1)).to_numpy()
    out = np.zeros(down.shape, dtype=np.int16)
    for i in range(1, len(out)):
        out[i] = np.where(down[i], out[i - 1] + 1, 0)
    return out


def dedupe(fire: np.ndarray, cooldown: int = COOLDOWN) -> np.ndarray:
    """該当した日から cooldown 営業日は同じ銘柄を数えない。"""
    out = np.zeros_like(fire)
    last = np.full(fire.shape[1], -10**9)
    for i in range(len(fire)):
        hit = fire[i] & (i - last > cooldown)
        out[i] = hit
        last = np.where(hit, i, last)
    return out


def earnings_near(close: pd.DataFrame) -> np.ndarray:
    paths = sorted(glob.glob(os.path.join(DATA, "earndate_*.parquet")))
    e = pd.concat([pd.read_parquet(p, columns=["SchDate", "Code"]) for p in paths], ignore_index=True)
    e["Code"] = e["Code"].astype(str)
    e["SchDate"] = pd.to_datetime(e["SchDate"], errors="coerce")
    e = e.dropna().drop_duplicates()
    # 予定日を、その日以降で最初の営業日に寄せる（休日の予定日を落とさない）
    dates = close.index.values
    pos = np.searchsorted(dates, e["SchDate"].values)
    keep = pos < len(dates)
    e = e[keep]
    pos = pos[keep]
    col = pd.Index(close.columns).get_indexer(e["Code"])
    m = np.zeros(close.shape, dtype=bool)
    ok = col >= 0
    m[pos[ok], col[ok]] = True
    # t−4〜t のどこかに予定日があれば True（rolling max）
    return (pd.DataFrame(m).rolling(EARN_WINDOW, min_periods=1).max().to_numpy() > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()

    close, va = load_wide()
    print(f"[load] {close.shape[0]:,}営業日 × {close.shape[1]:,}銘柄 "
          f"{close.index.min():%Y-%m-%d}〜{close.index.max():%Y-%m-%d} ({time.time() - t0:.0f}s)")

    common = np.array([(len(c) != 5) or c.endswith("0") for c in close.columns])
    univ = (market_ok(close).to_numpy()
            & common[None, :]
            & (va.rolling(20, min_periods=15).mean().to_numpy() >= MIN_VA)
            & close.notna().to_numpy())

    logr = np.log(close / close.shift(1))
    sigma = logr.rolling(60, min_periods=40).std().shift(21)
    hi20 = close.rolling(20, min_periods=15).max()
    dd20 = close / hi20 - 1
    rel = close.div(load_topix(close.index), axis=0)
    dd20rel = rel / rel.rolling(20, min_periods=15).max() - 1
    dd20s = np.log1p(dd20) / (sigma * np.sqrt(20))
    r5 = close / close.shift(5) - 1
    r1 = close / close.shift(1) - 1
    ma25dev = close / close.rolling(25, min_periods=20).mean() - 1
    ma20 = close.rolling(20, min_periods=20).mean()
    sd20 = close.rolling(20, min_periods=20).std()
    rsi = rsi_wilder(close)
    streak = down_streak(close)

    ma200 = close.rolling(200, min_periods=180).mean()
    c_pre, m_pre, m_pre2 = close.shift(20), ma200.shift(20), ma200.shift(40)
    up_pre = ((c_pre > m_pre) & (m_pre > m_pre2)).to_numpy()
    dn_pre = ((c_pre < m_pre) & (m_pre < m_pre2)).to_numpy()
    known_pre = (c_pre.notna() & m_pre.notna() & m_pre2.notna()).to_numpy()
    below_now = (close < ma200).to_numpy()
    earn = earnings_near(close)

    defs = {
        "dd20_10": dd20 <= -0.10, "dd20_15": dd20 <= -0.15, "dd20_20": dd20 <= -0.20,
        "dd20rel_10": dd20rel <= -0.10, "dd20rel_15": dd20rel <= -0.15,
        "dd20s_2.0": dd20s <= -2.0, "dd20s_2.5": dd20s <= -2.5,
        "r5_10": r5 <= -0.10, "r5_15": r5 <= -0.15,
        "ma25_10": ma25dev <= -0.10, "ma25_15": ma25dev <= -0.15, "ma25_20": ma25dev <= -0.20,
        "down5": pd.DataFrame(streak >= 5),
        "rsi14_30": rsi <= 30,
        "bb2": close <= ma20 - 2 * sd20,
        "r1_7": r1 <= -0.07, "r1_10": r1 <= -0.10,
    }

    in_period = (close.index >= pd.Timestamp(args.start))
    n_days = int(in_period.sum())
    univ_p = univ[in_period]
    print(f"[universe] {args.start}〜 {n_days:,}営業日 / 1日の母集団 平均 {univ_p.sum(1).mean():,.0f}銘柄 "
          f"(最小 {univ_p.sum(1).min():,} / 最大 {univ_p.sum(1).max():,})")

    rows, raw_sets = [], {}
    dates = close.index[in_period]
    for name, cond in defs.items():
        fire_raw = np.asarray(cond, dtype=bool) & univ
        ev = dedupe(fire_raw)[in_period]
        raw_sets[name] = fire_raw[in_period]
        per_day = ev.sum(1)
        tot = int(per_day.sum())
        up = int((ev & up_pre[in_period]).sum())
        dn = int((ev & dn_pre[in_period]).sum())
        unk = int((ev & ~known_pre[in_period]).sum())
        mixed = tot - up - dn - unk
        top_k = max(1, int(round(n_days * 0.01)))
        order = np.argsort(per_day)[::-1]
        conc = per_day[order[:top_k]].sum() / tot if tot else np.nan
        top3 = [(f"{dates[i]:%Y-%m-%d}", int(per_day[i])) for i in order[:3]]
        up_flip = (ev & up_pre[in_period] & below_now[in_period]).sum() / up if up else np.nan
        rows.append({
            "def": name, "events": tot, "per_day_mean": per_day.mean(),
            "per_day_median": float(np.median(per_day)), "p90": float(np.percentile(per_day, 90)),
            "zero_day_share": float((per_day == 0).mean()),
            "per_month_median": float(pd.Series(per_day, index=dates).resample("ME").sum().median()),
            "up": up / tot if tot else np.nan, "down": dn / tot if tot else np.nan,
            "mixed": mixed / tot if tot else np.nan, "unknown": unk / tot if tot else np.nan,
            "up_below_ma200_now": up_flip,
            "top1pct_day_share": conc, "top3_days": top3,
            "earnings_t4_t": float((ev & earn[in_period]).sum() / tot) if tot else np.nan,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = df.drop(columns=["top3_days"]).copy()
    for c in ["zero_day_share", "up", "down", "mixed", "unknown", "up_below_ma200_now",
              "top1pct_day_share", "earnings_t4_t"]:
        show[c] = (show[c] * 100).round(1)
    show["per_day_mean"] = show["per_day_mean"].round(1)
    print(show.to_string(index=False))
    print()
    for r in rows:
        print(f"  {r['def']:>10}  件数の多い日: " + ", ".join(f"{d} {n}件" for d, n in r["top3_days"]))

    # 定義どうしの重なり（該当した 銘柄×日 の Jaccard。間引く前）
    keys = ["dd20_15", "dd20rel_15", "dd20s_2.5", "r5_10", "ma25_15", "down5", "rsi14_30", "bb2", "r1_10"]
    jac = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            inter = (raw_sets[a] & raw_sets[b]).sum()
            union = (raw_sets[a] | raw_sets[b]).sum()
            jac.loc[a, b] = inter / union if union else np.nan
    print("\n[重なり] Jaccard（該当した 銘柄×日、間引く前）")
    print((jac * 100).round(0).to_string())

    adopted = adopted_breakdown(close, univ, in_period, up_pre, dn_pre, known_pre, earn)
    print(f"\n[done] {time.time() - t0:.0f}s")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"start": args.start, "days": n_days, "rows": rows,
                       "jaccard": (jac.round(4)).to_dict(), "adopted": adopted},
                      fh, ensure_ascii=False, indent=1, default=float)


def adopted_breakdown(close, univ, in_period, up_pre, dn_pre, known_pre, earn) -> dict:
    """
    採用した定義（運用者の選択 2026-11-22: 20日高値から −15%・素の株価で判定・下げる前の t−20 でトレンド）の内訳。

    市場の下げは別の列（mkt_chg = 銘柄の20日高値の日から当日までの TOPIX の騰落率）で持ち、ここでは帯に分けて数える。
    excess = (1 + 株価の下落率) / (1 + mkt_chg) − 1 は、市場の分を除いた個別の下げ。
    """
    c = close.to_numpy()
    topix = load_topix(close.index).to_numpy()
    hi20 = close.rolling(20, min_periods=15).max().to_numpy()
    dd = c / hi20 - 1
    ev = dedupe((dd <= -0.15) & univ)
    ev[~in_period] = False
    ii, jj = np.nonzero(ev)
    peak = np.empty(len(ii), dtype=int)
    for k, (i, j) in enumerate(zip(ii, jj)):
        lo = max(0, i - 19)
        peak[k] = lo + int(np.nanargmax(c[lo:i + 1, j]))
    mkt = topix[ii] / topix[peak] - 1
    depth = dd[ii, jj]
    excess = (1 + depth) / (1 + mkt) - 1
    trend = np.where(~known_pre[ii, jj], "不明", np.where(up_pre[ii, jj], "上昇",
                     np.where(dn_pre[ii, jj], "下降", "混在")))
    mkt_band = pd.cut(mkt, [-np.inf, -0.10, -0.05, 0.0, np.inf],
                      labels=["TOPIX −10%以下", "−10〜−5%", "−5〜0%", "TOPIX 上昇"])
    f = pd.DataFrame({"date": close.index[ii], "trend": trend, "mkt_band": mkt_band, "mkt": mkt,
                      "depth": depth, "excess": excess, "days_since_peak": ii - peak,
                      "earn": earn[ii, jj]})
    n = len(f)
    order = ["上昇", "下降", "混在", "不明"]
    ct = pd.crosstab(f.trend, f.mkt_band, margins=True, margins_name="計").reindex(order + ["計"])
    print(f"\n[採用した定義] 20日高値から −15%（素の株価）: {n:,}件")
    print("  トレンド（t−20）× 同じ期間の TOPIX の騰落率（件数）")
    print(ct.to_string())
    print("  （割合 %）")
    print((ct / n * 100).round(1).to_string())
    by_trend = f.groupby("trend").agg(件数=("depth", "size"), 決算の前後=("earn", "mean"),
                                      高値からの日数_中央=("days_since_peak", "median"),
                                      下落率_中央=("depth", "median"), 個別の下げ_中央=("excess", "median"),
                                      TOPIX_中央=("mkt", "median")).reindex(order)
    for col in ["決算の前後", "下落率_中央", "個別の下げ_中央", "TOPIX_中央"]:
        by_trend[col] = (by_trend[col] * 100).round(1)
    print("  トレンドごと（決算の前後 = 決算予定日が t−4〜t に入る件数の %。率は中央値の %）")
    print(by_trend.to_string())
    yearly = f.assign(year=f.date.dt.year).groupby("year").agg(
        件数=("depth", "size"), 上昇=("trend", lambda s: (s == "上昇").mean() * 100),
        下降=("trend", lambda s: (s == "下降").mean() * 100),
        TOPIX10以下=("mkt", lambda s: (s <= -0.10).mean() * 100)).round(1)
    print("  年ごと（上昇・下降・TOPIX −10%以下 は件数に占める %。2026年は11/20まで）")
    print(yearly.to_string())
    return {"events": n, "crosstab": ct.to_dict(), "by_trend": by_trend.to_dict(),
            "yearly": yearly.to_dict()}


if __name__ == "__main__":
    main()
