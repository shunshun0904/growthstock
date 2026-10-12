#!/usr/bin/env python3
"""
実験85: 月に2銘柄の枠を「6本とも95以上 → 3本が95以上 → LGB が95以上」と、条件を段階的に緩めながら埋めたらどうなるか。

運用者の話（2026-10-12）「もちろん6モデルともに95以上の条件は満たした上でのことです。なければ、条件を徐々に緩和させる
イメージで3モデル95以上、lgbm95以上のような感じです」。緩める時期は運用者の回答「月の後半（16日以降）から」。

段（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  1段目 = A15（1.2σ・1.5σ の lgbm・xgb・cat の6本とも 95以上）。段の中は6本の平均の高い順
  2段目 = R0（1.2σ の3本が95以上）で1段目でない。段の中は3本の最小の高い順
  3段目 = LGB（1.2σ の lgbm）が95以上で R0 でない。段の中は lgbm のスコアの高い順
  （どの段も同点は lgbm のスコア順。百分位は前向き・画面と同じ丸め。テスト窓 3〜11）
  出口: +10% の指値・届かなければ 20営業日目の終値（運用者の今の売り）

選び方（月に2件まで）
  主: 段階的・16日から緩める（その日のうちに決められる形）: 15日までは1段目だけを出た順に買う。16日以降は、まだ2件に
      届いていなければ、その日の候補のうち一番上の段から買う（1段目はいつでも買う）
  比べる: 今のやり方 = R0 の先着2件（R0 を出た順に。同じ日は3本の最小の順）
  参考: 段階的・3段目は月末の5営業日だけ / 段階的・1段目に6本の平均の線（訓練の期間で A15 が月平均2件になる高さ。
        実験84 と同じ）/ 後から見た段階的な上位2件（各月、段の上から順に2件。月末にならないと決まらない）
  判定: 主 − 今のやり方 の窓ごとの差の平均が正・上の窓が過半（実験84 の B と同じ。差の標準誤差も並べる）
  出すもの: 件数・買った月（2件 / 1件 / 0件）・段ごとの件数と成績・+10% の成績（平均・中央・勝率・到達・−10%割れ・分位点）
  参考に全期間の百分位（画面の物差し）でも主と今のやり方・後から見た段階的な上位2件を

公開ログには集計だけ（件数・平均・分位点、窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e85_tier_cascade.py
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
MAIN = "tp10c"
TOP_N = 2
RELAX_DAY = 16                 # この日（暦の日）以降に下の段へ緩める
LAST_DAYS = 5                  # 参考: 3段目は月末のこの営業日数だけ
LINE = 95.0
OUT_DIR = E79.OUT_DIR
TIER_JA = {1: "1段目（6本とも95以上）", 2: "2段目（3本が95以上）", 3: "3段目（LGB が95以上）"}

log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s
qtiles, fmt_q, months_of, window_diff = E84.qtiles, E84.fmt_q, E84.months_of, E84.window_diff


# --------------------------------------------------------------------------- #
# 段
# --------------------------------------------------------------------------- #

def build(df: pd.DataFrame, parts: list, models=MODELS) -> pd.DataFrame:
    """実験84 の行（A15・R0・6本の平均）に、lgbm の百分位（画面の丸め）と L95（lgbm が95以上）を足す。"""
    ft = E84.build(df, parts, models)
    i = list(models).index(M.BASELINE)
    ft["lgbm_r"] = np.concatenate([E82.rounded(np.asarray(P, dtype=float)[:, [i]])[:, 0] for _, P in parts])
    ft["L95"] = ft["lgbm_r"] >= LINE
    return ft


def tiers(ft: pd.DataFrame, top_mask: np.ndarray | None = None) -> pd.DataFrame:
    """
    段（tier: 1 / 2 / 3、どれでもなければ 0）と段の中の並べ方の値（tkey: 1段目 = 6本の平均、2段目 = 3本の最小、3段目 = lgbm の
    スコア）を足す。top_mask を渡すと1段目をそれに置き換える（線ありの参考。外れた A15 は2段目へ）。
    """
    a15 = ft["A15"].to_numpy(dtype=bool) if top_mask is None else np.asarray(top_mask, dtype=bool)
    r0 = ft["R0"].to_numpy(dtype=bool)
    l95 = ft["L95"].to_numpy(dtype=bool)
    t = np.where(a15, 1, np.where(r0, 2, np.where(l95, 3, 0)))
    key = np.where(t == 1, ft["avg6"].to_numpy(dtype=float),
                   np.where(t == 2, ft["pmin3"].to_numpy(dtype=float), ft["score"].to_numpy(dtype=float)))
    return ft.assign(tier=t, tkey=key)


def last_days_of_month(bar_days, n: int = LAST_DAYS) -> set:
    """各月の最後の n 営業日（日足のある日）。"""
    s = pd.Series(pd.to_datetime(list(bar_days)))
    out = set()
    for _, g in s.groupby(s.dt.to_period("M")):
        out.update(pd.Timestamp(d) for d in g.sort_values().iloc[-n:])
    return out


def cascade(ft: pd.DataFrame, relax_day: int = RELAX_DAY, n: int = TOP_N, tier3_days: set | None = None) -> pd.DataFrame:
    """
    その日のうちに決められる段階的な選び方。月ごとに日付の順に見て、その日に買える段（1段目はいつでも、2・3段目は暦の日が
    relax_day 以降。tier3_days を渡すと3段目はその日だけ）の候補を、段の上から（段の中は tkey、同点は lgbm のスコア）枠が
    埋まるまで買う。月に n 件まで。
    """
    sub = ft[ft["tier"] > 0].assign(_m=months_of, _d=lambda x: pd.to_datetime(x["Date"]))
    picks = []
    for _, g in sub.groupby("_m", sort=True):
        left = n
        for d, day in g.groupby("_d", sort=True):
            if left <= 0:
                break
            ok = day["tier"] == 1
            if d.day >= relax_day:
                ok |= day["tier"] == 2
                ok |= (day["tier"] == 3) if tier3_days is None else ((day["tier"] == 3) & (d in tier3_days))
            c = day[ok].sort_values(["tier", "tkey", "score"], ascending=[True, False, False], kind="mergesort")
            take = c.head(left)
            picks.append(take)
            left -= len(take)
    out = pd.concat(picks) if picks else sub.iloc[0:0]
    return out.reset_index(drop=True)


def hindsight(ft: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """後から見た段階的な上位: 各月、段の上から（段の中は tkey、同点は lgbm のスコア）n 件。"""
    sub = ft[ft["tier"] > 0].assign(_m=months_of)
    sub = sub.sort_values(["_m", "tier", "tkey", "score"], ascending=[True, True, False, False], kind="mergesort")
    return sub.groupby("_m", sort=False).head(n).reset_index(drop=True)


def first_r0(ft: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """今のやり方: R0 の先着 n 件（同じ日は3本の最小の順）。"""
    return E84.first_by(ft, ft["R0"].to_numpy(dtype=bool), "pmin3", n)


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def describe(name: str, g: pd.DataFrame, months: int, width: int = 40) -> dict:
    r = pd.to_numeric(g[MAIN], errors="coerce").dropna()
    hit = g.loc[r.index, "hit10"].astype(bool).mean() * 100 if len(r) else float("nan")
    per_m = g.assign(_m=months_of).groupby("_m").size() if len(g) else pd.Series(dtype=int)
    two, one = int((per_m >= 2).sum()), int((per_m == 1).sum())
    zero = months - two - one
    q = qtiles(r)
    print(f"  {LT._l(name, width)}{len(g):>4}件（2件の月 {two}・1件 {one}・0件 {zero}）  +10% {q['mean']:+6.2f}%"
          f"（中央 {q['p50']:+5.2f}）  勝率 {(r > 0).mean() * 100 if len(r) else float('nan'):4.1f}%  到達 {hit:4.1f}%  "
          f"−10%割れ {(r < -10).mean() * 100 if len(r) else float('nan'):4.1f}%")
    print(f"      分位点 {fmt_q(q)}")
    out = {"n": int(len(g)), "two": two, "one": one, "zero": zero, "q": q, "hit": float(hit),
           "win": float((r > 0).mean() * 100) if len(r) else float("nan"),
           "bad10": float((r < -10).mean() * 100) if len(r) else float("nan")}
    if "tier" in g.columns and len(g):
        parts = []
        for t in (1, 2, 3):
            x = pd.to_numeric(g.loc[g["tier"] == t, MAIN], errors="coerce").dropna()
            out[f"tier{t}"] = {"n": int(len(x)), "mean": float(x.mean()) if len(x) else float("nan")}
            parts.append(f"{TIER_JA[t]} {len(x)}件 {x.mean():+.2f}%" if len(x) else f"{TIER_JA[t]} 0件")
        print("      段ごと: " + " / ".join(parts))
    return out


def report(ft: pd.DataFrame, thr: dict | None = None, bar_days=None, ff: pd.DataFrame | None = None) -> dict:
    summary: dict = {}
    months = E82.n_months(ft)
    ft = tiers(ft)
    cnt = ft["tier"].value_counts()
    print(f"\n■ 0. 前向き {len(ft):,}行・{months}か月。段ごとの行: 1段目 {int(cnt.get(1, 0))}・2段目 {int(cnt.get(2, 0))}・"
          f"3段目 {int(cnt.get(3, 0))}。出口はどれも +10% の指値・届かなければ20日目の終値")
    for t in (1, 2, 3):
        g = ft[ft["tier"] == t]
        r = pd.to_numeric(g[MAIN], errors="coerce").dropna()
        print(f"  {TIER_JA[t]}の行全体: 月 {len(g) / months:.1f}件・+10% {r.mean():+.2f}%（中央 {r.median():+.2f}・勝率 "
              f"{(r > 0).mean() * 100:.1f}%）")
    print(f"\n■ 1. 月に{TOP_N}件の選び方（その日のうちに決められる形と、後から見た形）")
    cur = first_r0(ft)
    main = cascade(ft)
    sel = {"cur": ("今のやり方: R0 の先着2件", cur),
           "main": (f"段階的・{RELAX_DAY}日から緩める（主）", main)}
    if bar_days is not None and len(bar_days):
        sel["last5"] = (f"段階的・3段目は月末の{LAST_DAYS}営業日だけ", cascade(ft, tier3_days=last_days_of_month(bar_days)))
    if thr:
        ft_thr = tiers(ft, top_mask=E84.threshold_mask(ft, thr))
        sel["thr"] = ("段階的・1段目に6本の平均の線", cascade(ft_thr))
    sel["hind"] = ("後から見た段階的な上位2件", hindsight(ft))
    summary["select"] = {}
    for k, (name, g) in sel.items():
        summary["select"][k] = describe(name, g, months)
    print("  窓ごとの差（− 今のやり方）: " + "、".join(
        f"{sel[k][0]} {fmt_s(window_diff(sel[k][1], cur, MAIN))}" for k in sel if k != "cur"))
    d = window_diff(main, cur, MAIN)
    need = d["n"] // 2 + 1
    ok = bool(d["n"] and d["mean"] > 0 and d["wins"] >= need)
    print(f"  判定（主 − 今のやり方）: {fmt_s(d)} → {'通る' if ok else '通らない'}（差の平均が正 "
          f"{'○' if d['n'] and d['mean'] > 0 else '×'} / 上の窓が過半 {'○' if d['wins'] >= need else '×'}（{need}窓以上））")
    summary["main_vs_cur"] = d
    summary["passed"] = ok

    if ff is not None and len(ff):
        mf = E82.n_months(ff)
        f2 = tiers(ff)
        print(f"\n■ 2. 参考: 全期間の百分位（画面の物差し。先の分布も含む）・窓 {int(ff['window'].min())}〜{int(ff['window'].max())}")
        fs = {"cur": ("今のやり方: R0 の先着2件", first_r0(f2)),
              "main": (f"段階的・{RELAX_DAY}日から緩める", cascade(f2)),
              "hind": ("後から見た段階的な上位2件", hindsight(f2))}
        summary["full"] = {k: describe(name, g, mf) for k, (name, g) in fs.items()}
        summary["full"]["main_vs_cur"] = window_diff(fs["main"][1], fs["cur"][1], MAIN)
        print(f"  窓ごとの差（段階的・{RELAX_DAY}日から − 今のやり方）: {fmt_s(summary['full']['main_vs_cur'])}")

    print("\n■ まとめ（+10% の指値・20日目の終値）")
    s = summary["select"]
    print(f"  段階的・{RELAX_DAY}日から緩める: {s['main']['n']}件・1取引 {s['main']['q']['mean']:+.2f}%（中央 {s['main']['q']['p50']:+.2f}・"
          f"10% 点 {s['main']['q']['p10']:+.2f}）、今のやり方 {s['cur']['n']}件・{s['cur']['q']['mean']:+.2f}%（中央 "
          f"{s['cur']['q']['p50']:+.2f}・10% 点 {s['cur']['q']['p10']:+.2f}）。差 {fmt_s(d)} → {'通る' if ok else '通らない'}")
    print(f"  後から見た段階的な上位2件: {s['hind']['q']['mean']:+.2f}%（中央 {s['hind']['q']['p50']:+.2f}）")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。手数料・税・すべりは含まない")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験85: 段階的に条件を緩めて月2件を埋める")
    ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
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
    thr = E84.train_thresholds(df, split)
    ff = build(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))])
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）")
    summary = report(ft, thr=thr, bar_days=bar_days, ff=ff)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e85_summary.json")

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
