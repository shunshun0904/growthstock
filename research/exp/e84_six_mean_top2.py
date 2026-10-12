#!/usr/bin/env python3
"""
実験84: 6本とも95以上（1.2σ・1.5σ）の中で、6本の平均の最上位2件を選んだらどうなるか。収益の分位点と、6本の平均の分位
ごとの成績も。

運用者の問い（2026-10-12）「さらに六本の平均で最上位の2銘柄した場合はどうなりますか？ただ最上位の場合の分位点も見たいです」。

選び方（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  A15  = 1.2σ・1.5σ の lgbm・xgb・cat の6本とも 95以上（実験82・83 と同じ）
  avg6 = その6本の百分位の平均（丸める前の値で並べ、同点は lgbm のスコア順）
  月に2銘柄: 各月、A15 の中で avg6 の上位2件（後から見た月の上位なので、その日のうちには決められない。実験79・80 の
             「月に2銘柄」と同じ作り）
  日ごとの上位2件: 発火8件以上の日に、A15 の中で avg6 の高い順に2件、3枠・先着順（その日のうちに決められる形）
  比べるもの: 同じ数を今の並べ方（1.2σ の3本の最小。同点は lgbm のスコア順）で選んだとき、A15 から無作為に選んだとき
  出口: +10% の指値・届かなければ 20営業日目の終値（実験83 の tp10c）。参考に今の +20%・5日平均（tp20a）と指値なし（c20）
  判定: 月に2銘柄は「窓ごとの差（avg6 − 今の並べ方）の平均が正・上の窓が過半」と「無作為に月2件選んだときの平均以上に
        なる割合 p < 0.05」の両方。日ごとの上位2件も同じ（取引の窓ごとの平均の差と、無作為な並べ方での売買との比べ）
  分位点: 選んだ取引の収益の 10 / 25 / 50 / 75 / 90% 点。選んだ行の avg6 の水準（同じ窓の候補全体の中で上位何%か）
  avg6 の四分位: 窓ごとに A15 の行を avg6 の順位で4つに分け、成績を比べる（上の四分位 − 下の四分位、窓ごと。見るだけ）
  百分位は前向き（実験79 の window_split、丸めは画面と同じ）。テスト窓 3〜11。参考に全期間の百分位

公開ログには集計だけ（群ごとの件数・平均・分位点、窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e84_six_mean_top2.py
    args="--draws 2000"       （既定）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402

MODELS = E79.MODELS
SIX = E82.TREES + E82.T15                      # lgbm, xgb, cat, lgbm_k15, xgb_k15, cat_k15
FLAG = "A15"
MAIN = "tp10c"
REF_EXITS = ("tp20a", "c20")
TOP_N = 2
DRAWS = 2000
SIM_DRAWS = 200
QS = (10, 25, 50, 75, 90)
SEED = 42
OUT_DIR = E79.OUT_DIR

log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s


# --------------------------------------------------------------------------- #
# 6本の平均
# --------------------------------------------------------------------------- #

def build(df: pd.DataFrame, parts: list, models=MODELS) -> pd.DataFrame:
    """実験83 の行に、6本の平均 avg6（丸める前）・avg6r（画面と同じ丸めの値の平均）と、同じ窓の候補全体の中での上位何%（avg6_top）を足す。"""
    ft = E83.build(df, parts)
    idx = [list(models).index(a) for a in SIX]
    raw = np.concatenate([np.asarray(P, dtype=float)[:, idx].mean(axis=1) for _, P in parts])
    rnd = np.concatenate([E82.rounded(np.asarray(P, dtype=float)[:, idx]).mean(axis=1) for _, P in parts])
    ft["avg6"] = raw
    ft["avg6r"] = rnd
    # 同じ窓の候補全体の中で、avg6 がそれより高い行の割合（%）。0 に近いほど最上位
    ft["avg6_top"] = ft.groupby("window")["avg6"].rank(ascending=False, method="min").sub(1).div(
        ft.groupby("window")["avg6"].transform("size")).mul(100)
    return ft


def qtiles(x) -> dict:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(x):
        return {f"p{q}": float("nan") for q in QS} | {"n": 0}
    out = {f"p{q}": float(np.percentile(x, q)) for q in QS}
    out.update(n=int(len(x)), mean=float(x.mean()), min=float(x.min()), max=float(x.max()))
    return out


def fmt_q(q: dict) -> str:
    return "  ".join(f"{q_}% {q[f'p{q_}']:+6.2f}" for q_ in QS)


# --------------------------------------------------------------------------- #
# 月に2銘柄
# --------------------------------------------------------------------------- #

def monthly_by(ft: pd.DataFrame, flag: str, key: str, n: int = TOP_N) -> pd.DataFrame:
    """各月、flag を満たす行を key の高い順（同点は lgbm のスコア順）に上位 n 件。"""
    sub = ft[ft[flag].to_numpy(dtype=bool)].assign(_m=lambda x: pd.to_datetime(x["Date"]).dt.to_period("M"))
    sub = sub.sort_values(["_m", key, "score"], ascending=[True, False, False], kind="mergesort")
    return sub.groupby("_m", sort=False).head(n).reset_index(drop=True)


def random_monthly(ft: pd.DataFrame, flag: str, col: str, n: int = TOP_N, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """各月、flag を満たす行から無作為に n 件（足りなければ全部）選んだときの、選んだ行の col の平均（draws 回）。"""
    rng = np.random.default_rng(seed)
    sub = ft[ft[flag].to_numpy(dtype=bool)]
    vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
    months = pd.to_datetime(sub["Date"]).dt.to_period("M").to_numpy()
    groups = [np.where(months == m)[0] for m in pd.unique(months)]
    out = np.empty(draws)
    for i in range(draws):
        pick = np.concatenate([g if len(g) <= n else rng.choice(g, n, replace=False) for g in groups])
        out[i] = np.nanmean(vals[pick])
    return out


def window_diff(a: pd.DataFrame, b: pd.DataFrame, col: str) -> dict:
    ma, mb = a.groupby("window")[col].mean(), b.groupby("window")[col].mean()
    j = ma.index.intersection(mb.index)
    return summarize((ma[j] - mb[j]).to_numpy(dtype=float))


def group_line(name: str, g: pd.DataFrame, months: int, width: int = 30) -> str:
    r = pd.to_numeric(g[MAIN], errors="coerce").dropna()
    hit = g.loc[r.index, "hit10"].astype(bool).mean() * 100 if len(r) else float("nan")
    refs = "  ".join(f"{E83.EXIT_JA[c].split('（')[0]} {pd.to_numeric(g[c], errors='coerce').mean():+.2f}%"
                     for c in REF_EXITS)
    return (f"  {LT._l(name, width)}{len(g):>5,}件（月 {len(g) / months:4.2f}件）  +10% {r.mean():+6.2f}%（中央 {r.median():+5.2f}）"
            f"  勝率 {(r > 0).mean() * 100:4.1f}%  到達 {hit:4.1f}%  −10%割れ {(r < -10).mean() * 100:4.1f}%  ｜ 参考 {refs}")


# --------------------------------------------------------------------------- #
# avg6 の四分位
# --------------------------------------------------------------------------- #

def quartiles_in_window(ft: pd.DataFrame, flag: str, key: str = "avg6", min_rows: int = 8) -> pd.Series:
    """窓ごとに flag の行を key の順位で4つ（0 = 下 〜 3 = 上）に分ける。行が少ない窓は NaN。"""
    out = pd.Series(np.nan, index=ft.index)
    sub = ft[ft[flag].to_numpy(dtype=bool)]
    for _, g in sub.groupby("window"):
        if len(g) < min_rows:
            continue
        rk = g[key].rank(method="first")
        out.loc[g.index] = pd.qcut(rk, 4, labels=False)
    return out


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def report(ft: pd.DataFrame, bar_days=None, ff: pd.DataFrame | None = None, draws: int = DRAWS,
           sim_draws: int = SIM_DRAWS, seed: int = SEED) -> dict:
    summary: dict = {}
    months = E82.n_months(ft)
    a15 = ft[ft[FLAG].to_numpy(dtype=bool)]
    print(f"\n■ 0. 前向き {len(ft):,}行・{months}か月。A15（6本とも95以上）{len(a15):,}行。6本の平均（画面の丸め）は "
          f"A15 の中で 10% 点 {np.percentile(a15['avg6r'], 10):.1f}・中央 {np.median(a15['avg6r']):.1f}・90% 点 "
          f"{np.percentile(a15['avg6r'], 90):.1f}")
    summary["a15_avg6r"] = qtiles(a15["avg6r"])

    # ---- 1. 月に2銘柄
    print(f"\n■ 1. 月に{TOP_N}銘柄（各月の A15 の中の上位。後から見た月の上位なので、その日のうちには決められない）")
    top6 = monthly_by(ft, FLAG, "avg6")
    topm = monthly_by(ft, FLAG, "pmin3")
    print(group_line("A15 全体", a15, months))
    print(group_line("6本の平均の上位2件", top6, months))
    print(group_line("今の並べ方（3本の最小）の上位2件", topm, months))
    rnd = random_monthly(ft, FLAG, MAIN, draws=draws, seed=seed)
    m6 = float(pd.to_numeric(top6[MAIN], errors="coerce").mean())
    p_rnd = float((np.sum(rnd >= m6) + 1) / (len(rnd) + 1))
    print(f"  無作為に月{TOP_N}件（{draws}回）: +10% の平均 {rnd.mean():+.2f}%（5〜95% {np.percentile(rnd, 5):+.2f} 〜 "
          f"{np.percentile(rnd, 95):+.2f}%）。6本の平均の上位2件がそれ以上になる割合 p = {p_rnd:.3f}")
    d6m = window_diff(top6, topm, MAIN)
    d6a = window_diff(top6, a15, MAIN)
    print(f"  窓ごとの差: 6本の平均の上位2件 − 今の並べ方の上位2件 {fmt_s(d6m)}、− A15 全体 {fmt_s(d6a)}")
    need = d6m["n"] // 2 + 1
    ok_m = bool(d6m["n"] and d6m["mean"] > 0 and d6m["wins"] >= need and p_rnd < 0.05)
    print(f"  判定: {'通る' if ok_m else '通らない'}（差の平均が正 {'○' if d6m['n'] and d6m['mean'] > 0 else '×'} / 上の窓が過半 "
          f"{'○' if d6m['wins'] >= need else '×'}（{need}窓以上） / 無作為との比べ p < 0.05 {'○' if p_rnd < 0.05 else '×'}）")
    print("  収益の分位点（+10% の指値・20日目の終値。上の方は +10% で止まる）")
    q6, qm, qa = qtiles(top6[MAIN]), qtiles(topm[MAIN]), qtiles(a15[MAIN])
    print(f"    6本の平均の上位2件   {fmt_q(q6)}  最悪 {q6['min']:+.1f}%")
    print(f"    今の並べ方の上位2件  {fmt_q(qm)}  最悪 {qm['min']:+.1f}%")
    print(f"    A15 全体             {fmt_q(qa)}  最悪 {qa['min']:+.1f}%")
    print("  収益の分位点（参考: 指値なし・20日目の終値。値動きそのものの散らばり）")
    print(f"    6本の平均の上位2件   {fmt_q(qtiles(top6['c20']))}")
    print(f"    A15 全体             {fmt_q(qtiles(a15['c20']))}")
    lv = qtiles(top6["avg6r"])
    tp = qtiles(top6["avg6_top"])
    print(f"  選んだ行の6本の平均（画面の丸め）: 10% 点 {lv['p10']:.1f}・中央 {lv['p50']:.1f}・90% 点 {lv['p90']:.1f}。"
          f"同じ窓の候補全体の中で上位 {tp['p50']:.2f}%（中央。10〜90% 点 {tp['p10']:.2f}〜{tp['p90']:.2f}%）")
    summary["monthly"] = {"top6": {"q": q6, "mean": m6, "n": int(len(top6)), "months": int(top6["_m"].nunique())},
                          "topm": {"q": qm, "n": int(len(topm))}, "a15": {"q": qa},
                          "random": {"mean": float(rnd.mean()), "p": p_rnd},
                          "vs_topm": d6m, "vs_a15": d6a, "pass": ok_m, "level": lv, "top_pct": tp,
                          "ref": {c: float(pd.to_numeric(top6[c], errors="coerce").mean()) for c in REF_EXITS}}

    # ---- 2. 日ごとの上位2件（3枠）
    if bar_days is not None and len(bar_days):
        n_days = LT.n_bar_days(bar_days, pd.to_datetime(ft["Date"]).min(), pd.to_datetime(ft["Date"]).max())
        print(f"\n■ 2. 日ごとの上位{TOP_N}件（発火{LT.SKIP_BREAKS}件以上の日・{LT.SLOTS}枠・先着順。その日のうちに決められる形。"
              f"{n_days}営業日）")
        sims = {}
        for key, name in (("avg6", "6本の平均の順"), ("pmin3", "今の並べ方（3本の最小）の順")):
            st = E83.trade_sim(ft, FLAG, MAIN, "hit10", "hitday10", bar_days, n_days, key=key, keep_trades=True)
            sims[key] = st
            q = qtiles(st.get("rets", []))
            print(f"  {LT._l(name, 28)}候補 {st['picks']:>4}件 → 買えた {st['n']:>4}件（月 {st['per_month']:.1f}件）  1取引 "
                  f"{st['mean']:+6.2f}%（中央 {st['median']:+5.2f}）  勝率 {st['win']:4.1f}%  到達 {st['hit']:4.1f}%  "
                  f"−10%割れ {st['bad10']:4.1f}%  月の収益 {st['monthly']:+5.2f}%")
            print(f"    分位点 {fmt_q(q)}  最悪 {q.get('min', float('nan')):+.1f}%")
        rng = np.random.default_rng(seed)
        r_mean, r_month = [], []
        ft2 = ft.copy()
        for _ in range(sim_draws):
            ft2["_rnd"] = rng.random(len(ft2))
            st = E83.trade_sim(ft2, FLAG, MAIN, "hit10", "hitday10", bar_days, n_days, key="_rnd")
            r_mean.append(st.get("mean", np.nan))
            r_month.append(st.get("monthly", np.nan))
        r_mean, r_month = np.array(r_mean, dtype=float), np.array(r_month, dtype=float)
        s6 = sims["avg6"]
        p_mean = float((np.sum(r_mean >= s6["mean"]) + 1) / (len(r_mean) + 1))
        p_month = float((np.sum(r_month >= s6["monthly"]) + 1) / (len(r_month) + 1))
        print(f"  無作為な並べ方（{sim_draws}回）: 1取引 {np.nanmean(r_mean):+.2f}%（5〜95% {np.nanpercentile(r_mean, 5):+.2f}〜"
              f"{np.nanpercentile(r_mean, 95):+.2f}）・月の収益 {np.nanmean(r_month):+.2f}%。6本の平均の順がそれ以上になる割合 "
              f"1取引 p = {p_mean:.3f}・月の収益 p = {p_month:.3f}")
        wa, wb = sims["avg6"]["by_window"], sims["pmin3"]["by_window"]
        j = sorted(set(wa) & set(wb))
        dd = summarize(np.array([wa[w] - wb[w] for w in j], dtype=float))
        need = dd["n"] // 2 + 1
        ok_d = bool(dd["n"] and dd["mean"] > 0 and dd["wins"] >= need and p_mean < 0.05)
        print(f"  取引の窓ごとの平均の差（6本の平均の順 − 今の並べ方）: {fmt_s(dd)} → 判定: {'通る' if ok_d else '通らない'}")
        summary["daily"] = {k: {kk: vv for kk, vv in v.items() if kk != "rets"} | {"q": qtiles(v.get("rets", []))}
                            for k, v in sims.items()}
        summary["daily"].update(random={"mean": float(np.nanmean(r_mean)), "monthly": float(np.nanmean(r_month)),
                                        "p_mean": p_mean, "p_month": p_month}, vs_pmin3=dd, passed=ok_d)

    # ---- 3. avg6 の四分位
    print("\n■ 3. A15 の中の6本の平均の四分位（窓ごとに順位で4つ。見るだけ）")
    qt = quartiles_in_window(ft, FLAG)
    summary["quartiles"] = {}
    names = ("下（0〜25%）", "25〜50%", "50〜75%", "上（75〜100%）")
    for k in range(4):
        g = ft[qt == k]
        if not len(g):
            continue
        q = qtiles(g[MAIN])
        summary["quartiles"][k] = q | {"avg6r_med": float(np.median(g["avg6r"]))}
        print(group_line(f"{names[k]}（6本の平均 中央 {np.median(g['avg6r']):.1f}）", g, months, width=36))
        print(f"      分位点 {fmt_q(q)}")
    top_q, low_q = ft[qt == 3], ft[qt == 0]
    dq = window_diff(top_q, low_q, MAIN)
    summary["quartiles"]["top_minus_low"] = dq
    print(f"  上の四分位 − 下の四分位（窓ごと）: {fmt_s(dq)}")

    # ---- 4. 全期間の百分位（参考）
    if ff is not None and len(ff):
        mf = E82.n_months(ff)
        print("\n■ 4. 参考: 全期間の百分位（画面の物差し。先の分布も含む）・月に2銘柄")
        t6f, tmf = monthly_by(ff, FLAG, "avg6"), monthly_by(ff, FLAG, "pmin3")
        print(group_line("A15 全体", ff[ff[FLAG].to_numpy(dtype=bool)], mf))
        print(group_line("6本の平均の上位2件", t6f, mf))
        print(group_line("今の並べ方の上位2件", tmf, mf))
        print(f"    6本の平均の上位2件の分位点 {fmt_q(qtiles(t6f[MAIN]))}")
        summary["full"] = {"top6": qtiles(t6f[MAIN]), "topm": qtiles(tmf[MAIN]),
                           "vs_topm": window_diff(t6f, tmf, MAIN)}
        print(f"    窓ごとの差（6本の平均 − 今の並べ方）{fmt_s(summary['full']['vs_topm'])}")

    print("\n■ まとめ")
    m = summary["monthly"]
    print(f"  月に2銘柄（6本の平均の上位2件・+10%）: {m['top6']['n']}件・1取引 {m['top6']['mean']:+.2f}%"
          f"（中央 {m['top6']['q']['p50']:+.2f}・10% 点 {m['top6']['q']['p10']:+.2f}）、無作為に月2件 {m['random']['mean']:+.2f}%"
          f"（p = {m['random']['p']:.3f}）、今の並べ方との窓ごとの差 {fmt_s(m['vs_topm'])} → {'通る' if m['pass'] else '通らない'}")
    if "daily" in summary:
        d = summary["daily"]
        print(f"  日ごとの上位2件（3枠）: 6本の平均の順 1取引 {d['avg6']['mean']:+.2f}%・月 {d['avg6']['monthly']:+.2f}%、"
              f"今の並べ方 {d['pmin3']['mean']:+.2f}%・月 {d['pmin3']['monthly']:+.2f}%、無作為 {d['random']['mean']:+.2f}%"
              f"（p = {d['random']['p_mean']:.3f}）→ {'通る' if d['passed'] else '通らない'}")
    print(f"  四分位: 上 − 下 {fmt_s(summary['quartiles']['top_minus_low'])}")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。手数料・税・すべりは含まない")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験84: 6本とも95以上の中で6本の平均の上位2件")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--sim-draws", type=int, default=SIM_DRAWS)
    args = ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件（モデルごとの件数 {sorted(set(info0['rows'].values()))}）")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = E83.attach_exits(df, bars, targets=(10.0, 20.0))
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・5日平均と ret_o1_20 の一致 {chk['hold5_match']:,}/"
        f"{chk['hold5_both']:,}・到達の順序の食い違い {chk['order_violations']}件")
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    split = E79.window_split(df, folds)
    ft = build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）")
    ff = build(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))])
    summary = report(ft, bar_days=bar_days, ff=ff, draws=args.draws, sim_draws=args.sim_draws)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e84_summary.json")

    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=conv)
    log(f"記録: {path} / 全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
