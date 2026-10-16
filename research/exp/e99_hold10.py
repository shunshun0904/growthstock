#!/usr/bin/env python3
"""
実験99: 保有期間を 20 営業日 → 10 営業日にしたら（+10% の指値に 10 日で届かなければ 10 日目の終値で売り、次の候補へ）。
銘柄の選び方（1.5σ の3本がすべて95以上・上位2件・3枠・発火の数は問わない）とモデルは変えない。出口の日数だけ変える。

運用者の仮説（2026-10-16）「基準日を過ぎて、上がってそのままの勢いで 10% に行かなければ調整に入り、その後 20 日待っても
10% に行かないどころか低い水準でうろうろするのでは。それなら 10 日で利確して次の銘柄に行ったほうが良いのでは。もう 10 日
待つことと、他の銘柄に切り替えること、どちらのほうが 10% に到達する期待値が高いか」。

比べ方（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  行: 本番の out-of-fold（Release の 12 本）に、実験88 と同じ前向きの百分位（テスト窓より前の窓の分布。窓 3〜11）を付け、
      規則の行 = 1.5σ の3本（lgbm_k15 / xgb_k15 / cat_k15）がすべて 95 以上。発火の数は問わない（2026-10-12 から）
  出口: +10% の指値（日中の高値が買値の +10% に届いたらその日に +10%）。届かなければ H 営業日目の終値（1 = 買った日）。
      H = 5 / 10 / 15 / 20（今は 20）。live_track.forward で測る（画面・保有の推移と同じ約束。20 日目の含み損の持ち越しは
      しない = 実験83・88 と同じ tp10c の形）
  §1 到達の時間: 規則の行（20 日ぶんの日足がある行）で、+10% に k 日目までに届いた割合（k = 1〜20）。
      「10 日までに届かなかった行」のうち 11〜20 日目に届く割合 = 仮説の「もう 10 日待つ」の到達確率
  §2 出口 H ごとの成績（規則の行・枠なし）: 件数・1取引の平均・中央・SE・勝率・到達・−10%割れ。窓ごとの差 H − 20
  §3 仮説の直接比較（規則の行）:
      A「もう 10 日待つ」= 10 日目までに届かなかった行の、10 日目の終値から出口（11〜20 日目に届けば +10% の値、届かなければ
        20 日目の終値）までの収益。10 日目の含み（< 0 / 0〜+5% / +5% 以上）で分けても出す
      B「次の銘柄に切り替える」= 規則の行の最初の 10 日の収益（+10% に届けば +10%、届かなければ 10 日目の終値）。
        切り替え先がすぐにあるかは、規則の行のある日の間隔（営業日）で示す
      窓ごとの差 B − A（両方に MIN_GROUP 行以上ある窓）
  §4 画面の 3 枠の形（今の規則・上位 2 件・同時に 3 件・先着・発火を問わない）を H = 5 / 10 / 15 / 20 で回す:
      取引数・1取引の平均・勝率・到達・資金全体の月あたり（取引の収益の合計 ÷ 3 枠 ÷ 月数。複利なし）。
      窓ごとの差 H − 20（1取引の平均と、窓の月あたり）
  §5 参考: 2026-10-16 からの指値（時価総額 1000 億円以上は +5%）で同じことを H = 10 / 20 で
  判定（先に決めた）: 次の 2 つがそろえば「10 日の案を運用者に出す」。片方だけ・どちらも無ければ結果を見せて運用者が決める
      (a) 3 枠の形で、窓ごとの「資金全体の月あたり」の差 H10 − H20 の平均が正で、H10 が上の窓が過半（n//2 + 1 以上）
      (b) §3 の窓ごとの差 B − A の平均が正で、B が上の窓が過半（切り替えたほうが、もう 10 日待つより良い）
  出すもの: 件数・平均・割合だけ。銘柄・日付ごとの値は出さない。手数料・税・すべりは含まない

    exp=e99_hold10.py            （引数なし。数分）
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
import e87_monthly_return_search as E87  # noqa: E402

MODELS = E79.MODELS
RULE_MODELS = LT.RULE_MODELS            # 1.5σ の3本
LINE = LT.AGREE_PCT                     # 95
TP = LT.TAKE_PROFIT                     # +10%
HOLDS = (5, 10, 15, 20)
BASE_HOLD = LT.HOLD                     # 20（今）
ALT_HOLD = 10                           # 仮説
TOP_K = LT.TOP_K
SLOTS = LT.SLOTS
MIN_BREAK = LT.SKIP_BREAKS              # 1 = 発火の数は問わない
MIN_GROUP = E82.MIN_GROUP               # 窓ごとの差に要る行数
HAZARD_DAYS = (1, 2, 3, 5, 7, 10, 12, 15, 20)
BUCKETS = ((None, 0.0, "10日目に含み損"), (0.0, 5.0, "0〜+5%"), (5.0, None, "+5% 以上"))
OUT_DIR = E79.OUT_DIR
log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s


# --------------------------------------------------------------------------- #
# 出口（保有 H 日）
# --------------------------------------------------------------------------- #

def col(prefix: str, name: str, h: int) -> str:
    return f"{prefix}{name}_h{h}"


def attach_holds(df: pd.DataFrame, bars: pd.DataFrame, holds=HOLDS, tp: float = TP, prefix: str = "",
                 mcap: pd.Series | None = None) -> pd.DataFrame:
    """
    保有 H 日ごとに hitday（+tp% に届いた日。1 = 買った日。届かなければ NaN）・ret（届けば +tp%、届かなければ H 日目の
    終値の収益 %。日足が足りなければ NaN）・close（H 日目の終値の収益 %）を足す。mcap（億円）を渡すと、行ごとの指値を
    live_track.take_profit_of で決める（2026-10-16 からの規則。tgt 列に %）。
    """
    keys = df[["Code", "Date"]].copy()
    if mcap is not None:
        keys["mcap"] = pd.to_numeric(mcap, errors="coerce").to_numpy(dtype=float)
    out = df.copy()
    for h in holds:
        fw = LT.forward(bars, keys, target=tp, hold=h)
        fw = fw[["Code", "Date", "hit_day", "ret", "ret_close", "target"]].rename(columns={
            "hit_day": col(prefix, "hitday", h), "ret": col(prefix, "ret", h), "ret_close": col(prefix, "close", h),
            "target": f"{prefix}tgt"})
        if f"{prefix}tgt" in out.columns:
            fw = fw.drop(columns=[f"{prefix}tgt"])
        out = out.merge(fw, on=["Code", "Date"], how="left")
    return out


def build(df: pd.DataFrame, parts: list, models=MODELS, prefixes=("",)) -> pd.DataFrame:
    """
    parts: [(行番号, 丸める前の百分位の行列)]。行ごとに、規則の3本の百分位の最小（画面と同じ丸め。欠けがあれば NaN）、
    基準モデルのスコア、保有 H 日ごとの出口の列、発火（その日の行数）。
    """
    i = {a: k for k, a in enumerate(models)}
    outs = []
    for rows, P in parts:
        R = E82.rounded(P)
        src = df.iloc[rows]
        g = src[["Code", "Date", "window"]].copy().reset_index(drop=True)
        g["Date"] = pd.to_datetime(g["Date"])
        g["score"] = src[f"score_{M.BASELINE}"].to_numpy(dtype=float)
        g["min_K15"] = R[:, [i[a] for a in RULE_MODELS]].min(axis=1)
        for p in prefixes:
            if f"{p}tgt" in src.columns:
                g[f"{p}tgt"] = pd.to_numeric(src[f"{p}tgt"], errors="coerce").to_numpy(dtype=float)
            for h in HOLDS:
                for name in ("hitday", "ret", "close"):
                    c = col(p, name, h)
                    if c in src.columns:
                        g[c] = pd.to_numeric(src[c], errors="coerce").to_numpy(dtype=float)
        outs.append(g)
    ft = pd.concat(outs, ignore_index=True)
    ft["n_break"] = ft.groupby("Date")["Code"].transform("size").to_numpy()
    return ft


def rule_rows(ft: pd.DataFrame) -> pd.DataFrame:
    ok = (ft["min_K15"].to_numpy(dtype=float) >= LINE) & (ft["n_break"].to_numpy() >= MIN_BREAK)
    return ft[ok]


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #

def stats(r: pd.Series, hit: pd.Series | None = None) -> dict:
    r = pd.to_numeric(r, errors="coerce").dropna()
    n = int(len(r))
    nan = float("nan")
    out = {"n": n, "mean": float(r.mean()) if n else nan, "med": float(r.median()) if n else nan,
           "win": float((r > 0).mean() * 100) if n else nan, "bad10": float((r < -10).mean() * 100) if n else nan,
           "se": float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else nan, "hit": nan}
    if hit is not None and n:
        out["hit"] = float(pd.Series(hit).loc[r.index].astype(bool).mean() * 100)
    return out


def window_diff(g: pd.DataFrame, a: str, b: str, min_rows: int = MIN_GROUP) -> dict:
    """窓ごとの (a の平均 − b の平均)。両方の列が min_rows 行以上ある窓だけ。"""
    diffs = []
    for _, w in g.groupby("window"):
        x = pd.to_numeric(w[a], errors="coerce").dropna()
        y = pd.to_numeric(w[b], errors="coerce").dropna()
        if len(x) >= min_rows and len(y) >= min_rows:
            diffs.append(float(x.mean() - y.mean()))
    return summarize(np.array(diffs, dtype=float))


def hazard(g: pd.DataFrame, prefix: str = "", hold: int = BASE_HOLD) -> dict:
    """
    +10% に k 日目までに届いた割合（hold 日ぶんの日足がある行）。cond = 10 日までに届かなかった行のうち 11〜hold 日目に届く割合。
    """
    c = pd.to_numeric(g[col(prefix, "close", hold)], errors="coerce")
    full = g[c.notna()]
    hd = pd.to_numeric(full[col(prefix, "hitday", hold)], errors="coerce").to_numpy(dtype=float)
    n = int(len(full))
    cum = {int(k): float(np.mean(hd <= k) * 100) if n else float("nan") for k in range(1, hold + 1)}
    not10 = ~(hd <= ALT_HOLD)
    late = (hd > ALT_HOLD) & (hd <= hold)
    return {"n": n, "cum": cum, "hit10": cum.get(ALT_HOLD, float("nan")), "hit20": cum.get(hold, float("nan")),
            "n_not10": int(not10.sum()), "n_late": int(late.sum()),
            "cond_late": float(late.sum() / not10.sum() * 100) if not10.sum() else float("nan"),
            "median_hitday": float(np.nanmedian(hd)) if np.isfinite(hd).any() else float("nan")}


def wait_vs_switch(g: pd.DataFrame, prefix: str = "") -> dict:
    """
    A「もう 10 日待つ」: 10 日目までに届かなかった行（20 日ぶんの日足あり）の、10 日目の終値から出口までの収益。
    B「切り替える」: 規則の行の最初の 10 日の収益（全行）。窓ごとの差 B − A。
    """
    r10 = pd.to_numeric(g[col(prefix, "ret", ALT_HOLD)], errors="coerce")
    r20 = pd.to_numeric(g[col(prefix, "ret", BASE_HOLD)], errors="coerce")
    hd20 = pd.to_numeric(g[col(prefix, "hitday", BASE_HOLD)], errors="coerce")
    full = r10.notna() & r20.notna()
    not10 = full & ~(hd20 <= ALT_HOLD)
    seg = ((1.0 + r20 / 100.0) / (1.0 + r10 / 100.0) - 1.0) * 100.0
    a = g[not10].assign(seg=seg[not10], late=(hd20[not10] > ALT_HOLD))
    out = {"A": stats(a["seg"], a["late"]), "B": stats(r10[r10.notna()], hd20[r10.notna()] <= ALT_HOLD),
           "n_full": int(full.sum()), "buckets": {}}
    tmp = g.assign(_seg=seg.where(not10), _r10=r10.where(r10.notna()))
    out["diff"] = window_diff(tmp, "_r10", "_seg")
    for lo, hi, name in BUCKETS:
        m = not10.to_numpy().copy()               # &= で元の配列を壊さない
        if lo is not None:
            m &= (r10 >= lo).to_numpy()
        if hi is not None:
            m &= (r10 < hi).to_numpy()
        out["buckets"][name] = stats(seg[m], (hd20[m] > ALT_HOLD))
    return out


def gaps(g: pd.DataFrame, bar_days) -> dict:
    """規則の行がある日の間隔（営業日）。切り替え先がすぐにあるかの目安。"""
    idx = {pd.Timestamp(d): k for k, d in enumerate(bar_days)}
    days = sorted({pd.Timestamp(d) for d in g["Date"]})
    pos = np.array([idx[d] for d in days if d in idx], dtype=float)
    d = np.diff(pos)
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "median": float(np.median(d)), "mean": float(d.mean()),
            "le1": float((d <= 1).mean() * 100), "le5": float((d <= 5).mean() * 100), "le10": float((d <= 10).mean() * 100),
            "days_with_rule": int(len(days))}


def trade(ft: pd.DataFrame, bar_days, hold: int, months: int, prefix: str = "") -> tuple[dict, pd.DataFrame]:
    """画面の 3 枠の形（今の規則・上位 TOP_K 件・SLOTS 枠・先着・発火を問わない）を保有 hold 日で。"""
    idx = {pd.Timestamp(d): k for k, d in enumerate(bar_days)}
    c = rule_rows(ft).sort_values(["Date", "min_K15", "score"], ascending=[True, False, False], kind="mergesort")
    c = c.assign(rank=c.groupby("Date").cumcount().to_numpy() + 1)
    c = c[c["rank"] <= TOP_K]
    day = np.array([idx.get(pd.Timestamp(d), -1) for d in c["Date"]], dtype=float)
    c, day = c[day >= 0], day[day >= 0]
    ret = pd.to_numeric(c[col(prefix, "ret", hold)], errors="coerce").to_numpy(dtype=float)
    hitday = pd.to_numeric(c[col(prefix, "hitday", hold)], errors="coerce").to_numpy(dtype=float)
    hit = np.isfinite(hitday)
    m = c["Date"].dt.to_period("M").map(lambda p: p.ordinal).to_numpy()
    tk = E87.simulate(day, ret, hit, hitday, m, SLOTS, 0, hold=hold)
    t = c.iloc[tk].copy()
    t["_ret"] = ret[tk]
    t["_hit"] = hit[tk]
    s = stats(t["_ret"], t["_hit"])
    s["picks"] = int(len(c))
    s["capital_pm"] = float(t["_ret"].sum() / SLOTS / months) if months else float("nan")
    # 窓ごとの月あたり（その窓の取引の合計 ÷ 枠 ÷ その窓の月数）
    wm = ft.assign(_m=pd.to_datetime(ft["Date"]).dt.to_period("M")).groupby("window")["_m"].nunique()
    per = {}
    for w, grp in t.groupby("window"):
        n_m = int(wm.get(w, 0))
        per[int(w)] = {"n": int(len(grp)), "mean": float(grp["_ret"].mean()),
                       "capital_pm": float(grp["_ret"].sum() / SLOTS / n_m) if n_m else float("nan")}
    s["per_window"] = per
    return s, t


def trade_diff(a: dict, b: dict, key: str) -> dict:
    """窓ごとの a − b（両方に取引がある窓）。key は mean（1取引の平均）か capital_pm（月あたり）。"""
    ws = sorted(set(a["per_window"]) & set(b["per_window"]))
    return summarize(np.array([a["per_window"][w][key] - b["per_window"][w][key] for w in ws], dtype=float))


def judge(trade_cap: dict, switch: dict) -> dict:
    """先に決めた判定。(a) 3枠の月あたりの差 H10 − H20、(b) 窓ごとの差 B − A。"""
    def ok(s):
        return bool(s["n"]) and s["mean"] > 0 and s["wins"] >= s["n"] // 2 + 1
    a_ok, b_ok = ok(trade_cap), ok(switch)
    return {"a": a_ok, "b": b_ok,
            "verdict": "10日の案を運用者に出す" if a_ok and b_ok else "結果を見せて運用者が決める"}


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def line(name: str, s: dict, width: int = 34) -> str:
    hit = f"  到達 {s['hit']:4.1f}%" if np.isfinite(s.get("hit", np.nan)) else ""
    return (f"  {LT._l(name, width)}{s['n']:>5}件  1取引 {s['mean']:+6.2f}%（中央 {s['med']:+5.2f}・SE {s['se']:.2f}）  勝率 "
            f"{s['win']:4.1f}%{hit}  −10%割れ {s['bad10']:4.1f}%") if s["n"] else f"  {LT._l(name, width)}    0件"


def report(ft: pd.DataFrame, bar_days, prefixes=("",)) -> dict:
    summary: dict = {}
    months = int(pd.to_datetime(ft["Date"]).dt.to_period("M").nunique())
    g = rule_rows(ft)
    print(f"\n■ 0. 前向き {len(ft):,}行・{months}か月（テスト窓 {int(ft['window'].min())}〜{int(ft['window'].max())}）。"
          f"今の規則の行（1.5σ の3本がすべて{LINE:g}以上・発火を問わない） {len(g):,}件")
    summary["rows"] = {"all": int(len(ft)), "rule": int(len(g)), "months": months}

    # ---- 1. 到達の時間
    hz = hazard(g)
    hz_all = hazard(ft)
    summary["hazard"] = {"rule": hz, "all": hz_all}
    print(f"\n■ 1. +{TP:g}% に k 日目までに届いた割合（規則の行・{BASE_HOLD}日ぶんの日足がある {hz['n']:,}件。1 = 買った日）")
    print("  " + "  ".join(f"{k}日 {hz['cum'][k]:4.1f}%" for k in HAZARD_DAYS))
    print(f"  {ALT_HOLD}日までに届かなかった {hz['n_not10']:,}件のうち、{ALT_HOLD + 1}〜{BASE_HOLD}日目に届いたのは {hz['n_late']:,}件"
          f"（{hz['cond_late']:.1f}%）。届いた日の中央値 {hz['median_hitday']:.0f}日")
    print(f"  参考・候補全体 {hz_all['n']:,}件: {ALT_HOLD}日まで {hz_all['hit10']:.1f}% / {BASE_HOLD}日まで {hz_all['hit20']:.1f}% / "
          f"{ALT_HOLD}日まで届かなかった行のうち後半に届く {hz_all['cond_late']:.1f}%")

    # ---- 2. 出口 H ごと
    print(f"\n■ 2. 出口を H 日にした成績（規則の行・枠なし。+{TP:g}% に届けば +{TP:g}%、届かなければ H 日目の終値）")
    summary["holds"] = {}
    for h in HOLDS:
        s = stats(g[col("", "ret", h)], pd.to_numeric(g[col("", "hitday", h)], errors="coerce").notna())
        summary["holds"][h] = s
        print(line(f"H = {h}日{'（今）' if h == BASE_HOLD else ''}", s))
    summary["hold_diffs"] = {}
    for h in HOLDS:
        if h == BASE_HOLD:
            continue
        d = window_diff(g, col("", "ret", h), col("", "ret", BASE_HOLD))
        summary["hold_diffs"][h] = d
        print(f"  窓ごとの差 H{h} − H{BASE_HOLD}: {fmt_s(d)}")

    # ---- 3. 待つ vs 切り替える
    ws = wait_vs_switch(g)
    gp = gaps(g, bar_days)
    summary["switch"] = ws
    summary["gaps"] = gp
    print(f"\n■ 3. 仮説: {ALT_HOLD}日までに届かなかったとき、もう{BASE_HOLD - ALT_HOLD}日待つ（A）か、次の銘柄に切り替える（B）か")
    print(line(f"A もう{BASE_HOLD - ALT_HOLD}日待つ（{ALT_HOLD}日目の終値から出口まで）", ws["A"], 44)
          + f"  ← 到達 = {ALT_HOLD + 1}〜{BASE_HOLD}日目に +{TP:g}%")
    print(line(f"B 切り替える（規則の行の最初の{ALT_HOLD}日）", ws["B"], 44) + f"  ← 到達 = {ALT_HOLD}日以内に +{TP:g}%")
    print(f"  窓ごとの差 B − A: {fmt_s(ws['diff'])}")
    print(f"  A を{ALT_HOLD}日目の含みで分ける:")
    for name, s in ws["buckets"].items():
        print(line("  " + name, s, 44))
    if gp.get("n"):
        print(f"  切り替え先: 規則の行のある日は {gp['days_with_rule']:,}日。次の規則の日までの間隔は中央 {gp['median']:.0f}営業日"
              f"（平均 {gp['mean']:.1f}。1日以内 {gp['le1']:.0f}% / 5日以内 {gp['le5']:.0f}% / 10日以内 {gp['le10']:.0f}%）")

    # ---- 4. 3枠の形
    print(f"\n■ 4. 画面の3枠の形（今の規則・上位{TOP_K}件・同時に{SLOTS}件・先着・発火を問わない）を保有 H 日で")
    summary["trade"] = {}
    for h in HOLDS:
        s, _ = trade(ft, bar_days, h, months)
        summary["trade"][h] = s
        print(line(f"H = {h}日{'（今）' if h == BASE_HOLD else ''}", s) + f"  資金全体で月 {s['capital_pm']:+.2f}%（選んだ {s['picks']}件）")
    summary["trade_diffs"] = {}
    for h in HOLDS:
        if h == BASE_HOLD:
            continue
        dm = trade_diff(summary["trade"][h], summary["trade"][BASE_HOLD], "mean")
        dc = trade_diff(summary["trade"][h], summary["trade"][BASE_HOLD], "capital_pm")
        summary["trade_diffs"][h] = {"mean": dm, "capital_pm": dc}
        print(f"  窓ごとの差 H{h} − H{BASE_HOLD}: 1取引の平均 {fmt_s(dm)} / 月あたり {fmt_s(dc)}")

    # ---- 5. 参考: 時価総額の規則
    if "cap_" in prefixes:
        print(f"\n■ 5. 参考: 2026-10-16 からの指値（時価総額 {LT.LARGE_CAP:g}億円以上は +{LT.TAKE_PROFIT_LARGE:g}%）で同じことを")
        summary["cap"] = {"holds": {}, "trade": {}}
        gc = g
        for h in (ALT_HOLD, BASE_HOLD):
            s = stats(gc[col("cap_", "ret", h)], pd.to_numeric(gc[col("cap_", "hitday", h)], errors="coerce").notna())
            summary["cap"]["holds"][h] = s
            print(line(f"規則の行・H = {h}日", s))
        for h in (ALT_HOLD, BASE_HOLD):
            s, _ = trade(ft, bar_days, h, months, prefix="cap_")
            summary["cap"]["trade"][h] = s
            print(line(f"3枠の形・H = {h}日", s) + f"  資金全体で月 {s['capital_pm']:+.2f}%")
        summary["cap"]["switch"] = wait_vs_switch(g, prefix="cap_")
        print(f"  待つ vs 切り替える: A {summary['cap']['switch']['A']['mean']:+.2f}%（{summary['cap']['switch']['A']['n']}件）/ "
              f"B {summary['cap']['switch']['B']['mean']:+.2f}%（{summary['cap']['switch']['B']['n']}件）/ "
              f"窓ごとの差 B − A {fmt_s(summary['cap']['switch']['diff'])}")
        summary["cap"]["trade_diff_capital"] = trade_diff(summary["cap"]["trade"][ALT_HOLD], summary["cap"]["trade"][BASE_HOLD],
                                                           "capital_pm")
        print(f"  3枠の月あたりの差 H{ALT_HOLD} − H{BASE_HOLD}: {fmt_s(summary['cap']['trade_diff_capital'])}")

    # ---- 6. 判定
    jd = judge(summary["trade_diffs"][ALT_HOLD]["capital_pm"], ws["diff"])
    summary["judge"] = jd
    print(f"\n■ 6. 判定（先に決めた読み方）")
    print(f"  (a) 3枠の月あたり H{ALT_HOLD} − H{BASE_HOLD} {fmt_s(summary['trade_diffs'][ALT_HOLD]['capital_pm'])} → {'○' if jd['a'] else '×'}")
    print(f"  (b) 切り替える − もう{BASE_HOLD - ALT_HOLD}日待つ（B − A） {fmt_s(ws['diff'])} → {'○' if jd['b'] else '×'}")
    print(f"  → {jd['verdict']}")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。手数料・税・すべりは含まない。規則・モデルは変えていない")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験99: 保有期間を 10 日にしたら（規則・モデルは変えない）")
    ap.add_argument("--no-cap", action="store_true", help="時価総額の規則の参考（§5）を出さない")
    args = ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df = attach_holds(df, bars)
    prefixes = [""]
    if not args.no_cap and "market_cap" in fr.columns:
        mc = fr[["Code", "Date", "market_cap"]].copy()
        mc["Code"] = mc["Code"].astype(str)
        mc["Date"] = pd.to_datetime(mc["Date"])
        mc = mc.drop_duplicates(["Code", "Date"])
        m = df[["Code", "Date"]].merge(mc, on=["Code", "Date"], how="left")["market_cap"]
        df = attach_holds(df, bars, prefix="cap_", mcap=m)
        prefixes.append("cap_")
    n20 = int(pd.to_numeric(df[col("", "close", BASE_HOLD)], errors="coerce").notna().sum())
    log(f"出口: {BASE_HOLD}日目の終値 {n20:,}/{len(df):,}行")
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    split = E79.window_split(df, folds)
    ft = build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()], prefixes=tuple(prefixes))
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）。今の規則の行 {len(rule_rows(ft)):,}件（実験88 は 559件）")
    summary = report(ft, bar_days, prefixes=tuple(prefixes))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e99_summary.json")

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
