#!/usr/bin/env python3
"""
実験87: 月に資金の10%（平均）を稼ぐには、どんな選び方にすればよいか（過去の out-of-fold での組み合わせ最適化）。

運用者の問い（2026-10-12）「逆問題として解いてほしいです。毎月リターン10%を確保するには、どのような選び方にすればよいか。
売り方は、指値10%か15%か20%、非到達時、20営業時点で売る。というのは変えられないです。これは過去のoofのきかんを見た上での
組み合わせ最適化として解いてもらえれば良いです」。運用者の選択: 月リターン = 資金全体に対する%（資金を枠の数で等分して1件
ずつ入れる。複利にしない）、確保 = 月の平均で10%以上、件数の制約なし（枠の数も探す）、確かめ方 = 全期間の最適と、前半で
選んで後半に当てた結果の両方。

探す範囲（先に決めたもの。結果を見てから広げない・狭めない）
------------------------------------------
  候補の条件: モデルの組の百分位がすべて線以上（画面と同じ丸め）
      組（8つ）= LGB だけ（1.2σ）/ 1.2σ の3本 / 1.2σ の4本（logit も）/ 1.2σ・1.5σ の6本 / 1.2σ・1.8σ の6本 /
                 1.2σ・1.5σ・1.8σ の9本 / 12本すべて / 1.5σ の3本
      線（5つ）= 90 / 95 / 97 / 98 / 99
  日の条件: 発火（その日の候補の数）が 1（問わない）/ 8 / 15 / 20 件以上
  その日に買う上限: 上位 1 / 2 / 3 件（組の百分位の最小の高い順、同点は lgbm のスコア順）
  枠: 同時に 1 / 2 / 3 / 4 / 5 件（1件に資金の 1/枠）。空きが無ければ見送る（画面の3枠と同じ回し方: 翌営業日の寄りで買い、
      指値に届けば届いた日の翌営業日から、届かなければ20営業日の翌営業日から枠が空く）
  月の上限: 新しく買うのは月に 1 / 2 件まで / 上限なし（月は選定日の月）
  売り: +10% / +15% / +20% の指値、届かなければ 20営業日目の終値
  計 8 × 5 × 4 × 3 × 5 × 3 × 3 = 21,600 通り
  物差し: 月の損益（資金全体に対する%）= その月に選んだ取引の収益 × 1/枠 の合計。月の平均 = 合計 ÷ 月数（取引の無い月も数える）
  件数の下限: 全期間で24件以上（前半で選ぶときは前半で12件以上）
  前半 = テスト窓 3〜6（2022年5月〜2024年4月）、後半 = 窓 7〜11（2024年5月〜2026年9月）
  百分位は前向き（実験79 の window_split）。参考に全期間の百分位（画面の物差し）でも同じ探し方（全期間の上位だけ）
  手数料・税・すべりは含まない

出すもの
  1. 全期間で月の平均が最も高い組み合わせ（上位10。同じ取引になる組み合わせはまとめる）と、10% に届くか。前半・後半の内訳、
     1取引の平均、取引の無い月、10% 以上の月、最悪の月、累計の一番大きな落ち込み。全組み合わせの月の平均の散らばり
  2. 枠の数 × 売り、月の上限ごとの最良（届く上限の地図）
  3. 前半で選んだ最良（と上位10）を後半にそのまま当てた結果。後半での順位、前半と後半の月の平均の順位相関
  4. 比べる相手: 画面の形（1.2σ の3本 95・発火8件以上・上位2件・3枠・+10%）と、月2件の先着（1.2σ の3本 95）
  5. 画面の物差しでの上位5
確かめ: 速い回し方が、画面の形で実験83 の trade_sim と、月2件の先着で実験84 の first_by と同じ取引になること

公開ログには集計だけ（組み合わせごとの件数・月の平均など）。銘柄・日付ごとの値は出さない。

    exp=e87_monthly_return_search.py
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
import models as M  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e84_six_mean_top2 as E84  # noqa: E402

MODELS = E79.MODELS
#: (記号, 表示, モデル)
SETS = (
    ("L1", "LGB だけ（1.2σ）", ("lgbm",)),
    ("T3", "1.2σ の3本", E82.TREES),
    ("B4", "1.2σ の4本（logit も）", E82.BASE4),
    ("S15", "1.2σ・1.5σ の6本", E82.TREES + E82.T15),
    ("S18", "1.2σ・1.8σ の6本", E82.TREES + E82.T18),
    ("N9", "1.2σ・1.5σ・1.8σ の9本", E82.TREES + E82.T15 + E82.T18),
    ("A12", "12本すべて", MODELS),
    ("K15", "1.5σ の3本", E82.T15),
)
SET_JA = {c: ja for c, ja, _ in SETS}
LINES = (90.0, 95.0, 97.0, 98.0, 99.0)
BREAKS = (1, 8, 15, 20)
TOPK = (1, 2, 3)
SLOTS = (1, 2, 3, 4, 5)
CAPS = (1, 2, 0)                    # 0 = 上限なし
TARGETS = (10.0, 15.0, 20.0)
HOLD = LT.HOLD                      # 20営業日
GOAL = 10.0                         # 月の平均（資金全体に対する%）
MIN_FULL, MIN_H1 = 24, 12
H1 = (3, 4, 5, 6)                   # 前半のテスト窓
H2 = (7, 8, 9, 10, 11)              # 後半のテスト窓
TOP_SHOW = 10
OUT_DIR = E79.OUT_DIR

log = E79.log
summarize = E79.summarize


# --------------------------------------------------------------------------- #
# 行
# --------------------------------------------------------------------------- #

def build(df: pd.DataFrame, parts: list, models=MODELS) -> pd.DataFrame:
    """
    parts: [(行番号, 丸める前の百分位の行列)]。1行ごとに、組ごとの百分位の最小（画面と同じ丸め。欠けがあれば NaN = 満たさない）、
    lgbm のスコア、売りごとの収益（%）・到達の印・到達日、発火（その日の行数）。
    """
    i = {a: k for k, a in enumerate(models)}
    outs = []
    for rows, P in parts:
        R = E82.rounded(P)
        src = df.iloc[rows]
        g = src[["Code", "Date", "window"]].copy().reset_index(drop=True)
        g["Date"] = pd.to_datetime(g["Date"])
        g["score"] = src[f"score_{M.BASELINE}"].to_numpy(dtype=float)
        for code, _, grp in SETS:
            g[f"min_{code}"] = R[:, [i[a] for a in grp]].min(axis=1)
        for t in TARGETS:
            k = E83.tp_key(t)
            g[f"tp{k}c"] = pd.to_numeric(src[f"tp{k}c"], errors="coerce").to_numpy(dtype=float)
            g[f"hit{k}"] = src[f"hit{k}"].to_numpy(dtype=bool)
            g[f"hitday{k}"] = pd.to_numeric(src[f"hitday{k}"], errors="coerce").to_numpy(dtype=float)
        outs.append(g)
    ft = pd.concat(outs, ignore_index=True)
    ft["n_break"] = ft.groupby("Date")["Code"].transform("size").to_numpy()
    return ft


# --------------------------------------------------------------------------- #
# 売買（速い回し方。live_track.simulate と同じ約束）
# --------------------------------------------------------------------------- #

def simulate(day_idx, ret, hit, hitday, month, slots: int, cap: int, hold: int = HOLD) -> list:
    """
    日付・順位の順に並んだ候補を先着で枠へ入れ、買えた候補の位置を返す。翌営業日（day_idx + 1）に買い、指値に hitday 営業日目
    （1 = 買った日）で届けばその翌営業日から、届かなければ hold 営業日の翌営業日から枠が空く。収益が無い（日足が足りない）候補は
    枠をふさがずに飛ばす。cap > 0 なら、同じ月（選定日の月）に買うのは cap 件まで。
    """
    free = [-1.0] * slots
    used: dict = {}
    taken = []
    for j in range(len(day_idx)):
        r = ret[j]
        if r != r:                                   # NaN
            continue
        m = month[j]
        if cap and used.get(m, 0) >= cap:
            continue
        buy = day_idx[j] + 1
        s = -1
        for k in range(slots):
            if free[k] <= buy:
                s = k
                break
        if s < 0:
            continue
        free[s] = buy + (hitday[j] if hit[j] else hold)
        used[m] = used.get(m, 0) + 1
        taken.append(j)
    return taken


def candidates(ft: pd.DataFrame, code: str, line: float, min_break: int) -> pd.DataFrame:
    """組 code の百分位がすべて line 以上・発火 min_break 件以上の行を、日付・順位の順に。rank = その日の順位（1 から）。"""
    key = f"min_{code}"
    ok = (ft[key].to_numpy(dtype=float) >= line) & (ft["n_break"].to_numpy() >= min_break)
    c = ft[ok].sort_values(["Date", key, "score"], ascending=[True, False, False], kind="mergesort")
    c = c.assign(rank=c.groupby("Date").cumcount().to_numpy() + 1)
    return c


def months_index(ft: pd.DataFrame) -> tuple[pd.PeriodIndex, dict, np.ndarray]:
    """期間の月（取引の無い月も含む）、月 → 通し番号、月ごとのテスト窓。"""
    d = pd.to_datetime(ft["Date"])
    allm = pd.period_range(d.min().to_period("M"), d.max().to_period("M"), freq="M")
    pos = {m: k for k, m in enumerate(allm)}
    w = ft.assign(_m=d.dt.to_period("M")).groupby("_m")["window"].agg(lambda x: int(x.mode().iloc[0]))
    win = np.array([int(w.get(m, 0)) for m in allm])
    return allm, pos, win


def metrics(rets: np.ndarray, mpos: np.ndarray, win_of_trade: np.ndarray, slots: int, n_months: int,
            month_win: np.ndarray, h1=H1, h2=H2) -> dict:
    """取引の収益（%）と月の通し番号から、資金全体に対する月の損益（%）の要約。"""
    monthly = np.bincount(mpos, weights=rets / slots, minlength=n_months) if len(rets) else np.zeros(n_months)
    in1, in2 = np.isin(month_win, h1), np.isin(month_win, h2)
    t1, t2 = np.isin(win_of_trade, h1), np.isin(win_of_trade, h2)
    cum = np.cumsum(monthly)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    n = int(len(rets))
    nan = float("nan")
    return {"n": n, "n_h1": int(t1.sum()), "n_h2": int(t2.sum()),
            "avg": float(monthly.mean()), "avg_h1": float(monthly[in1].mean()) if in1.any() else nan,
            "avg_h2": float(monthly[in2].mean()) if in2.any() else nan,
            "trade_mean": float(rets.mean()) if n else nan, "trade_med": float(np.median(rets)) if n else nan,
            "win": float((rets > 0).mean() * 100) if n else nan,
            "zero_months": int((np.bincount(mpos, minlength=n_months) == 0).sum()) if n else int(n_months),
            "ge_goal": int((monthly >= GOAL).sum()), "worst": float(monthly.min()), "best": float(monthly.max()),
            "max_dd": float((peak - cum).max()), "sd": float(monthly.std(ddof=1)) if n_months > 1 else nan}


def search(ft: pd.DataFrame, bar_days, sets=SETS, lines=LINES, breaks=BREAKS, topks=TOPK, slots_list=SLOTS, caps=CAPS,
           targets=TARGETS, h1=H1, h2=H2) -> pd.DataFrame:
    """すべての組み合わせを回し、1組み合わせ1行の表を返す。"""
    idx = {pd.Timestamp(d): k for k, d in enumerate(bar_days)}
    allm, pos, month_win = months_index(ft)
    ft = ft.assign(_day=pd.to_datetime(ft["Date"]).map(lambda d: idx.get(pd.Timestamp(d), -1)).to_numpy(dtype=float),
                   _mpos=pd.to_datetime(ft["Date"]).dt.to_period("M").map(pos).to_numpy(dtype=int))
    rows = []
    for code, _, _ in sets:
        for line in lines:
            for b in breaks:
                c = candidates(ft, code, line, b)
                c = c[c["_day"].to_numpy() >= 0]
                day, mp, win = c["_day"].to_numpy(dtype=float), c["_mpos"].to_numpy(dtype=int), c["window"].to_numpy()
                rank = c["rank"].to_numpy()
                for k_top in topks:
                    sel = rank <= k_top
                    d_s, m_s, w_s = day[sel], mp[sel], win[sel]
                    for t in targets:
                        k = E83.tp_key(t)
                        r_s = c[f"tp{k}c"].to_numpy(dtype=float)[sel]
                        h_s = c[f"hit{k}"].to_numpy(dtype=bool)[sel]
                        hd_s = c[f"hitday{k}"].to_numpy(dtype=float)[sel]
                        for s in slots_list:
                            for cap in caps:
                                tk = np.array(simulate(d_s, r_s, h_s, hd_s, m_s, s, cap), dtype=int)
                                rr = r_s[tk] if len(tk) else np.zeros(0)
                                met = metrics(rr, m_s[tk] if len(tk) else np.zeros(0, dtype=int),
                                              w_s[tk] if len(tk) else np.zeros(0, dtype=int), s, len(allm), month_win,
                                              h1=h1, h2=h2)
                                met.update(set=code, line=line, breaks=b, topk=k_top, slots=s, cap=cap, target=t,
                                           signals=int(sel.sum()))
                                rows.append(met)
    return pd.DataFrame(rows)


def run_one(ft: pd.DataFrame, bar_days, code: str, line: float, b: int, k_top: int, s: int, cap: int, t: float):
    """1つの組み合わせの、買えた取引の行（確かめ用）。"""
    idx = {pd.Timestamp(d): k for k, d in enumerate(bar_days)}
    c = candidates(ft, code, line, b)
    c = c[c["rank"] <= k_top]
    day = np.array([idx.get(pd.Timestamp(d), -1) for d in c["Date"]], dtype=float)
    c, day = c[day >= 0], day[day >= 0]
    k = E83.tp_key(t)
    m = c["Date"].dt.to_period("M").map(lambda p: p.ordinal).to_numpy()
    tk = simulate(day, c[f"tp{k}c"].to_numpy(dtype=float), c[f"hit{k}"].to_numpy(dtype=bool),
                  c[f"hitday{k}"].to_numpy(dtype=float), m, s, cap)
    return c.iloc[tk]


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def label(r) -> str:
    cap = "月の上限なし" if int(r["cap"]) == 0 else f"月{int(r['cap'])}件まで"
    brk = "発火を問わない" if int(r["breaks"]) <= 1 else f"発火{int(r['breaks'])}件以上"
    return (f"{SET_JA[r['set']]} {r['line']:g}以上・{brk}・上位{int(r['topk'])}件・枠{int(r['slots'])}・{cap}・"
            f"+{r['target']:g}%")


def dedupe(res: pd.DataFrame, cols=("n", "avg", "avg_h1", "avg_h2", "trade_mean")) -> pd.DataFrame:
    """同じ取引になる組み合わせ（指標がすべて同じ）をまとめ、まとめた数を same に入れる（最初に出た組み合わせを残す）。"""
    sig = res[list(cols)].round(9).astype(str).agg("|".join, axis=1)
    out = res.assign(_sig=sig)
    cnt = out.groupby("_sig")["_sig"].transform("size")
    return out.assign(same=cnt.to_numpy()).drop_duplicates("_sig").drop(columns="_sig")


def line_of(r, extra: str = "") -> str:
    return (f"    月 {r['avg']:+6.2f}%（前半 {r['avg_h1']:+6.2f} / 後半 {r['avg_h2']:+6.2f}）  {int(r['n']):>4}件  1取引 "
            f"{r['trade_mean']:+6.2f}%（中央 {r['trade_med']:+5.2f}・勝率 {r['win']:4.1f}%）  取引の無い月 {int(r['zero_months'])}・"
            f"10%以上の月 {int(r['ge_goal'])}・最悪の月 {r['worst']:+6.2f}%・落ち込み {r['max_dd']:.1f}%{extra}")


def report(res: pd.DataFrame, n_months: int, checks: dict | None = None, res_full: pd.DataFrame | None = None,
           n_months_full: int | None = None) -> dict:
    summary: dict = {"combos": int(len(res)), "months": n_months}
    ok = res[res["n"] >= MIN_FULL]
    print(f"\n■ 0. {len(res):,}通りを回した（前向き・{n_months}か月）。全期間で{MIN_FULL}件以上の組み合わせ {len(ok):,}通り")
    if checks:
        for k, v in checks.items():
            print(f"  確かめ: {v['name']}: 速い回し方 {v['n_fast']}件・平均 {v['mean_fast']:+.2f}% / 元の関数 {v['n_ref']}件・"
                  f"平均 {v['mean_ref']:+.2f}% → {'一致' if v['same'] else '食い違い'}")
        summary["checks"] = checks
    q = ok["avg"].quantile([0.1, 0.5, 0.9]).to_dict() if len(ok) else {}
    if len(ok):
        print(f"  月の平均の散らばり（{MIN_FULL}件以上の組み合わせ）: 10% 点 {q[0.1]:+.2f}%・中央 {q[0.5]:+.2f}%・90% 点 {q[0.9]:+.2f}%・"
              f"最大 {ok['avg'].max():+.2f}%。月の平均がプラス {(ok['avg'] > 0).mean() * 100:.1f}%")
    summary["spread"] = {"q10": q.get(0.1), "q50": q.get(0.5), "q90": q.get(0.9),
                         "max": float(ok["avg"].max()) if len(ok) else None}

    # ---- 1. 全期間の上位
    reach = ok[ok["avg"] >= GOAL]
    print(f"\n■ 1. 全期間で月の平均が高い組み合わせ（後知恵）。目標 {GOAL:g}% に届いたのは {len(reach)}通り")
    top = dedupe(ok.sort_values(["avg", "n"], ascending=[False, False], kind="mergesort")).head(TOP_SHOW)
    for k, (_, r) in enumerate(top.iterrows(), 1):
        print(f"  {k:>2}. {label(r)}" + (f"（同じ取引になる組み合わせ {int(r['same'])}通り）" if r["same"] > 1 else ""))
        print(line_of(r))
    summary["top"] = top.drop(columns=[c for c in top.columns if c.startswith("_")]).to_dict("records")
    summary["reach"] = int(len(reach))

    # ---- 2. 地図
    print("\n■ 2. 枠の数 × 売りごとの最良（全期間・件数の下限あり）")
    grid = []
    for s in SLOTS:
        parts = []
        for t in TARGETS:
            g = ok[(ok["slots"] == s) & (ok["target"] == t)]
            if not len(g):
                parts.append(f"+{t:g}% —")
                continue
            b = g.sort_values(["avg", "n"], ascending=[False, False], kind="mergesort").iloc[0]
            grid.append({"slots": s, "target": t, "avg": float(b["avg"]), "n": int(b["n"]), "label": label(b)})
            parts.append(f"+{t:g}% 月 {b['avg']:+5.2f}%（{int(b['n'])}件）")
        print(f"  枠{s}: " + "  ".join(parts))
    print("  月の上限ごとの最良:")
    for cap in CAPS:
        g = ok[ok["cap"] == cap]
        if len(g):
            b = g.sort_values(["avg", "n"], ascending=[False, False], kind="mergesort").iloc[0]
            print(f"    {'上限なし' if cap == 0 else f'月{cap}件まで'}: {label(b)}")
            print(line_of(b))
            grid.append({"cap": cap, "avg": float(b["avg"]), "n": int(b["n"]), "label": label(b)})
    summary["grid"] = grid

    # ---- 3. 前半で選んで後半に当てる
    e1 = res[res["n_h1"] >= MIN_H1]
    print(f"\n■ 3. 前半（窓 3〜6）で選び、後半（窓 7〜11）にそのまま当てる。前半で{MIN_H1}件以上の組み合わせ {len(e1):,}通り")
    if len(e1):
        s1 = e1.sort_values(["avg_h1", "n_h1"], ascending=[False, False], kind="mergesort")
        b1 = s1.iloc[0]
        h2_rank = float((e1["avg_h2"] > b1["avg_h2"]).mean() * 100)
        rho = float(e1["avg_h1"].corr(e1["avg_h2"], method="spearman"))
        top1 = dedupe(s1).head(TOP_SHOW)
        print(f"  前半の最良: {label(b1)}")
        print(line_of(b1, f"・後半の件数 {int(b1['n_h2'])}"))
        print(f"  → 後半の月の平均 {b1['avg_h2']:+.2f}%（前半 {b1['avg_h1']:+.2f}%）。後半で、この組み合わせより上だった組み合わせは"
              f" {h2_rank:.1f}%")
        print(f"  前半の上位{TOP_SHOW}（まとめた後）の後半の月の平均: 平均 {top1['avg_h2'].mean():+.2f}%・"
              f"{top1['avg_h2'].min():+.2f}〜{top1['avg_h2'].max():+.2f}%（前半では {top1['avg_h1'].min():+.2f}〜"
              f"{top1['avg_h1'].max():+.2f}%）")
        print(f"  前半と後半の月の平均の順位相関（スピアマン）: {rho:+.2f}")
        b2 = e1.sort_values(["avg_h2", "n_h2"], ascending=[False, False], kind="mergesort").iloc[0]
        print(f"  参考: 後半だけで見た最良（後知恵）: 後半 {b2['avg_h2']:+.2f}%（前半 {b2['avg_h1']:+.2f}%）  {label(b2)}")
        summary["holdout"] = {"best_h1": {"label": label(b1), "avg_h1": float(b1["avg_h1"]), "avg_h2": float(b1["avg_h2"]),
                                          "n_h1": int(b1["n_h1"]), "n_h2": int(b1["n_h2"]), "h2_above_pct": h2_rank},
                              "top_h1_h2_mean": float(top1["avg_h2"].mean()), "spearman": rho,
                              "best_h2": {"label": label(b2), "avg_h2": float(b2["avg_h2"]), "avg_h1": float(b2["avg_h1"])}}

    # ---- 4. 比べる相手
    print("\n■ 4. 比べる相手（同じ回し方）")
    base = {"screen": ("画面の形: 1.2σ の3本 95・発火8件以上・上位2件・枠3・+10%", ("T3", 95.0, 8, 2, 3, 0, 10.0)),
            "month2": ("月2件の先着: 1.2σ の3本 95・枠5（ふさがらない）・月2件まで・+10%", ("T3", 95.0, 1, 3, 5, 2, 10.0))}
    summary["base"] = {}
    for key, (name, (code, line, b, k_top, s, cap, t)) in base.items():
        g = res[(res["set"] == code) & (res["line"] == line) & (res["breaks"] == b) & (res["topk"] == k_top)
                & (res["slots"] == s) & (res["cap"] == cap) & (res["target"] == t)]
        if len(g):
            r = g.iloc[0]
            print(f"  {name}")
            print(line_of(r))
            summary["base"][key] = r.to_dict()

    # ---- 5. 画面の物差し
    if res_full is not None and len(res_full):
        okf = res_full[res_full["n"] >= MIN_FULL]
        print(f"\n■ 5. 参考: 全期間の百分位（画面の物差し。先の分布も含む）・{n_months_full}か月。{MIN_FULL}件以上 {len(okf):,}通り。"
              f"目標に届いたのは {int((okf['avg'] >= GOAL).sum())}通り")
        topf = dedupe(okf.sort_values(["avg", "n"], ascending=[False, False], kind="mergesort")).head(5)
        for k, (_, r) in enumerate(topf.iterrows(), 1):
            print(f"  {k}. {label(r)}")
            print(line_of(r))
        summary["full_top"] = topf.to_dict("records")

    print("\n■ まとめ")
    if len(ok):
        b = ok.sort_values(["avg", "n"], ascending=[False, False], kind="mergesort").iloc[0]
        print(f"  全期間で最も高い月の平均は {b['avg']:+.2f}%（目標 {GOAL:g}%）: {label(b)}")
    if "holdout" in summary:
        h = summary["holdout"]
        print(f"  前半で選んだ最良は、前半 {h['best_h1']['avg_h1']:+.2f}% → 後半 {h['best_h1']['avg_h2']:+.2f}%")
    print("  注意: 21,600通りから選んだ最良は、たまたま当たった分だけ高く出る。手数料・税・すべりは含まない")
    return summary


def check_against(ft: pd.DataFrame, ft83: pd.DataFrame, bar_days) -> dict:
    """速い回し方が、実験83 の trade_sim（画面の形）・実験84 の first_by（月2件の先着）と同じ取引になるか。"""
    out = {}
    n_days = LT.n_bar_days(bar_days, pd.to_datetime(ft83["Date"]).min(), pd.to_datetime(ft83["Date"]).max())
    a = run_one(ft, bar_days, "T3", 95.0, LT.SKIP_BREAKS, LT.TOP_K, LT.SLOTS, 0, 10.0)
    ref = E83.trade_sim(ft83, "R0", "tp10c", "hit10", "hitday10", bar_days, n_days, keep_trades=True)
    ra, rb = np.sort(a["tp10c"].to_numpy(dtype=float)), np.sort(np.asarray(ref.get("rets", []), dtype=float))
    out["screen"] = {"name": "画面の形（実験83 の trade_sim）", "n_fast": int(len(ra)), "n_ref": int(len(rb)),
                     "mean_fast": float(ra.mean()) if len(ra) else float("nan"),
                     "mean_ref": float(rb.mean()) if len(rb) else float("nan"),
                     "same": bool(len(ra) == len(rb) and np.allclose(ra, rb))}
    m2 = run_one(ft, bar_days, "T3", 95.0, 1, 3, 5, 2, 10.0)
    fb = E84.first_by(ft83, ft83["R0"].to_numpy(dtype=bool), "pmin3")
    rc = np.sort(m2["tp10c"].to_numpy(dtype=float))
    rd = np.sort(pd.to_numeric(fb["tp10c"], errors="coerce").dropna().to_numpy(dtype=float))
    out["month2"] = {"name": "月2件の先着（実験84 の first_by）", "n_fast": int(len(rc)), "n_ref": int(len(rd)),
                     "mean_fast": float(rc.mean()) if len(rc) else float("nan"),
                     "mean_ref": float(rd.mean()) if len(rd) else float("nan"),
                     "same": bool(len(rc) == len(rd) and np.allclose(rc, rd))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験87: 月の平均で資金の10%を稼ぐ選び方の組み合わせ最適化")
    ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    for f in folds:
        if 3 <= f.index <= 11:
            log(f"  窓{f.index}: {f.test_start} 〜 {f.test_end}（{'前半' if f.index in H1 else '後半'}）")
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = E83.attach_exits(df, bars, targets=TARGETS)
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・到達の順序の食い違い {chk['order_violations']}件・"
        f"到達の割合 " + "・".join(f"+{k}% {v:.1f}%" for k, v in chk["hit_share"].items()))
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    split = E79.window_split(df, folds)
    parts = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
    ft = build(df, parts)
    ft83 = E83.build(df, parts)
    checks = check_against(ft, ft83, bar_days)
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）。回し始める")
    t1 = time.time()
    res = search(ft, bar_days)
    log(f"前向き {len(res):,}通り {time.time() - t1:.0f}秒")
    ff = build(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))])
    t1 = time.time()
    res_full = search(ff, bar_days)
    log(f"画面の物差し {len(res_full):,}通り {time.time() - t1:.0f}秒")
    summary = report(res, len(months_index(ft)[0]), checks=checks, res_full=res_full,
                     n_months_full=len(months_index(ff)[0]))
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_parquet(os.path.join(OUT_DIR, "e87_search.parquet"), index=False)
    path = os.path.join(OUT_DIR, "e87_summary.json")

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
