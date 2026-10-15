#!/usr/bin/env python3
"""
実験95: 実験93 の後工程の候補の線を「1.5σ の3本の最小が90以上」に緩めて、同じ物差し・同じ判定を回す（検証だけ。本番は変えない）。

運用者の指示（2026-10-15）「これだと厳しすぎるので、厳密な候補というよりは、ワッサーステイン用のサンプルの条件にはここは
緩めたいです。「1.5σモデルの3つのうちの最低が90以上」という条件にしてほしいです。それなら先ほどのバックテストでの検証も
サンプルが増えると思います」。買う規則（1.5σ の3本がすべて95以上）は変えない。後工程が選ぶ候補の集まりだけを 90以上に広げる。

比べ方（先に決めたもの。線以外は実験93 と同じ。結果を見てから動かさない）
------------------------------------------
  候補: その窓の物差し（前向きの百分位・画面と同じ丸め）で 1.5σ の3本の最小が **90以上** の行（実験93 は95以上）
  判断日: 候補が同じ日に2件以上ある日（収益と2つの分布が出せる候補だけ数える）
  近傍の母集団・距離・見積もり（近傍50件の収益の下側25%点）・選び方（Q = 見積もりの最上位、M = 今の並べ方 = 候補の中の
      3本の最小の順の先頭、R = 無作為 2,000回）・判定の3条件: 実験93 と同じ（e93 の関数をそのまま呼ぶ）
      95以上の候補がある日は、M はその中の先頭（画面の並びの先頭と同じ）
  判定（「分布で見て後工程として効く」= 3つがそろう）
    1. Q の取引の25%点 ≥ M の取引の25%点
    2. Q の25%点が無作為の25%点の分布で片側 p < 0.05
    3. Q の 10・50・75%点がどれも無作為の同じ分位点の中央以上
  参考（判定に使わない）
    - 判断日の内訳: 95以上の候補が1件以上ある日 / 90〜95 の候補だけの日。それぞれで Q・M・無作為の分布と同じ条件
    - Q が選んだ候補の3本の最小（95以上 / 90〜95 の割合）。95以上の候補がある日に Q が 90〜95 の候補を選んだ割合
    - 実験93 の線（95以上）との件数の比較、ほかの物差し（近傍の中央値・平均・25%点の近い100件）
  出すもの: 件数・分位点・割合・面積。銘柄・日付ごとの値は出さない

    exp=e95_knn_p25_line90.py
    args="--draws 2000"       （既定）

結果（2026-10-15、run 38738871533、a500d0f、49秒。docs/MODEL_ADOPTION_RULES.md §55）
------------------------------------------------------------------
  判定: 今の並べ方・無作為と見分けられない（3条件とも満たさない）。判断日は 96 → 225日に増えた（線以上の行 1,188件）が、
      Q（近傍50件の25%点の最上位）25%点 −2.47・10%点 −8.36・中央 +5.20・平均 +2.80%・−10%割れ 8.0%・到達 43.1%、
      M（今の並べ方の先頭）−1.92・−7.96・+5.20・+2.78%・7.1%・40.9%、無作為 −2.86・−8.03・+3.76・+2.20%。
      Q の25%点が無作為より上 p = 0.189、10%点は無作為の中央（−7.96）より下
  優越からのずれ: Q 対 M 下側 0.15 / 上側 0.18pt（ほぼ同じ分布）、Q 対 無作為 下側 0.12 / 上側 0.72pt
  内訳: 95以上の候補がある日 180日・90〜95 だけの日 45日。Q が選んだ候補の3本の最小が95以上なのは 46.2%（M は 80.0%）で、
      95以上の候補がある日に Q が 90〜95 の候補を選んだのは 42.2%。95以上の候補がある180日でも Q 25%点 −2.51 対 M −2.15・
      p = 0.259（×××）、90〜95 だけの45日は Q = M = −1.67・p = 0.193
  参考: 近傍の中央値 p = 0.060（××○）、平均 p = 0.137（×××）、25%点の近い100件 p = 0.189（××○）。M 自身は無作為より上 p = 0.052
  日の復元抽出の25%点の差 −1.84〜+1.23pt、前半 +0.00pt・後半 −0.05pt
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
import e83_tp10_rule as E83  # noqa: E402
import e91_wasserstein_knn_pick as E91  # noqa: E402
import e93_knn_quantile_pick as E93  # noqa: E402

LINE = 90.0                            # 後工程の候補の線（運用者の指示）。実験93 は E93.LINE = 95
STRICT = E93.LINE                      # 買う規則の線（95）。内訳と比較に使う
POOL_LINE = E93.POOL_LINE              # 近傍の母集団は実験93 と同じ（85以上）
MAIN_KEY = f"p{E93.MAIN_P}_{E93.K_MAIN}"
DRAWS = E93.DRAWS
SEED = E93.SEED
OUT_DIR = E79.OUT_DIR
log = E79.log


# --------------------------------------------------------------------------- #
# 内訳
# --------------------------------------------------------------------------- #

def day_flags(c: pd.DataFrame, strict: float = STRICT) -> pd.Series:
    """判断日ごとに、厳しい線（買う規則）以上の候補が1件以上あるか。"""
    if c.empty:
        return pd.Series(dtype=bool)
    return c.groupby("Date")["min15"].max() >= strict


def pick_min15(c: pd.DataFrame, key: str = MAIN_KEY, ascending: bool = False) -> pd.DataFrame:
    """判断日ごとの、Q（見積もり key の最上位）と M（今の並べ方の先頭）の3本の最小。e93.day_picks と同じ並べ方。"""
    rows = []
    if c.empty:
        return pd.DataFrame(rows)
    for d, g in c.groupby("Date", sort=True):
        if len(g) < 2:
            continue
        gq = g.sort_values([key, "min15", "score"], ascending=[ascending, False, False], kind="mergesort")
        gm = g.sort_values(["min15", "score"], ascending=False, kind="mergesort")
        rows.append({"Date": d, "Q_min15": float(gq["min15"].iat[0]), "M_min15": float(gm["min15"].iat[0])})
    return pd.DataFrame(rows)


def breakdown(c: pd.DataFrame, key: str = MAIN_KEY, strict: float = STRICT) -> dict:
    """Q が選んだ候補の3本の最小の内訳と、厳しい線以上の候補がある日に Q が下の帯を選んだ割合。"""
    pm = pick_min15(c, key)
    if pm.empty:
        return {"days": 0}
    flags = day_flags(c, strict)
    has = pm["Date"].map(flags).to_numpy(dtype=bool)
    q_strict = pm["Q_min15"].to_numpy(dtype=float) >= strict
    return {"days": int(len(pm)), "days_strict": int(has.sum()), "days_band_only": int((~has).sum()),
            "Q_strict_share": float(q_strict.mean() * 100),
            "Q_band_when_strict": float((~q_strict[has]).mean() * 100) if has.any() else float("nan"),
            "M_strict_share": float((pm["M_min15"].to_numpy(dtype=float) >= strict).mean() * 100)}


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def _print_eval(ev: dict, draws: int, name_q: str) -> None:
    print(E93.line(f"Q: {name_q}（{ev['days']}日）", ev["Q"]))
    print(E93.line(f"M: 今の並べ方の先頭（{ev['days']}日）", ev["M"]))
    print(E93.line(f"無作為（{draws:,}回を合わせた分布）", ev["R"]))
    print(f"  条件: {E93.fmt_cond(ev)}")


def report(c: pd.DataFrame, info: dict, draws: int = DRAWS, seed: int = SEED, n_days_strict_line: int | None = None) -> dict:
    summary: dict = {"info": {k: v for k, v in info.items() if k != "pool"}}
    pool = np.array(info.get("pool", []), dtype=float)
    print(f"\n■ 0. 件数（候補の線 {LINE:.0f}以上・前向き。実験93 は {STRICT:.0f}以上で 559件・96日）")
    print(f"  線以上の行 {info['rule_rows']}件（収益と2つの分布が出せる行 {info['rule_ok']}件）。候補が2件以上の日 {info['days2']}日、"
          f"うち近傍の母集団が{E93.MIN_POOL}件に満たず外した日 {info['days_small_pool']}日 → 判断日 {info['days']}日")
    if len(c):
        per_day = c.groupby("Date").size()
        print(f"  判断日の候補の数: 中央 {per_day.median():.0f}件（最大 {per_day.max()}件・2件の日 {int((per_day == 2).sum())}日）")
        print("  窓ごとの判断日: " + "  ".join(f"窓{w} {n}日" for w, n in info.get("per_window", {}).items()))
    if len(pool):
        print(f"  近傍の母集団の大きさ: 最小 {pool.min():.0f} / 中央 {np.median(pool):.0f} / 最大 {pool.max():.0f}件")
    if n_days_strict_line is not None:
        print(f"  参考: 同じデータで線を {STRICT:.0f} にしたときの判断日 {n_days_strict_line}日")

    print(f"\n■ 1. 判定（主: 近傍{E93.K_MAIN}件の{E93.MAIN_P}%点で選ぶ Q。候補は3本の最小 {LINE:.0f}以上）")
    ev = E93.evaluate(c, MAIN_KEY, False, draws, seed)
    summary["main"] = ev
    if not ev["days"]:
        print("  判断日が0日 → 判定できない")
        verdict = "判定できない"
    else:
        _print_eval(ev, draws, f"近傍の{E93.MAIN_P}%点の最上位")
        verdict = "分布で見て後工程として効く" if ev["effective"] else "今の並べ方・無作為と見分けられない"
        print(f"  判定: {verdict}")
        print(f"  Q と M が同じ候補を選んだ日 {ev['same_share']:.1f}%。候補全体の平均 {ev['all']:+.2f}%・後から見た最良 {ev['best']:+.2f}%")
        print(f"  参考: {E93.fmt_pair(ev)}")
        print(f"  参考: {E93.fmt_area(ev)}")
        print(f"  参考: M の{E93.MAIN_P}%点が無作為より上になる p = {ev['p_M']:.3f}")

    print(f"\n■ 2. 参考: 判断日の内訳（{STRICT:.0f}以上の候補がある日 / {LINE:.0f}〜{STRICT:.0f} だけの日。判定に使わない）")
    bd = breakdown(c, MAIN_KEY, STRICT)
    summary["breakdown"] = bd
    if bd.get("days"):
        print(f"  判断日 {bd['days']}日 = {STRICT:.0f}以上の候補がある日 {bd['days_strict']}日 + {LINE:.0f}〜{STRICT:.0f} だけの日 {bd['days_band_only']}日")
        print(f"  Q が選んだ候補の3本の最小: {STRICT:.0f}以上 {bd['Q_strict_share']:.1f}%（M は {bd['M_strict_share']:.1f}%）。"
              f"{STRICT:.0f}以上の候補がある日に Q が {LINE:.0f}〜{STRICT:.0f} の候補を選んだ割合 {bd['Q_band_when_strict']:.1f}%")
        flags = day_flags(c, STRICT)
        summary["subsets"] = {}
        for name, mask in ((f"{STRICT:.0f}以上の候補がある日", flags), (f"{LINE:.0f}〜{STRICT:.0f} だけの日", ~flags)):
            days = flags.index[mask.to_numpy()]
            sub = c[c["Date"].isin(days)]
            e = E93.evaluate(sub, MAIN_KEY, False, draws, seed) if len(sub) else {"days": 0}
            summary["subsets"][name] = e
            print(f"  {name}")
            if e.get("days"):
                _print_eval(e, draws, f"近傍の{E93.MAIN_P}%点の最上位")
                print(f"    {E93.fmt_area(e)}")
            else:
                print("    判断日が無い")

    print("\n■ 3. 参考: ほかの物差しで選ぶ形（判定に使わない）")
    summary["ref"] = {}
    rows = [(f"p50_{E93.K_MAIN}", f"近傍の中央値・近い{E93.K_MAIN}件", False),
            (f"mean_{E93.K_MAIN}", f"近傍の平均・近い{E93.K_MAIN}件", False)]
    rows += [(f"p{E93.MAIN_P}_{k}", f"近傍の{E93.MAIN_P}%点・近い{k}件", False) for k in E93.K_REF if k != E93.K_MAIN]
    for col, name, asc in rows:
        if col not in c.columns:
            continue
        e = E93.evaluate(c, col, asc, draws, seed)
        summary["ref"][col] = e
        if e["days"]:
            cd = e["conds"]
            print(f"  {LT._l(name, 40)}Q の{E93.MAIN_P}%点 {e['Q'][f'p{E93.MAIN_P}']:+6.2f}（M {e['M'][f'p{E93.MAIN_P}']:+6.2f}）・"
                  f"無作為より上 p = {e['p_Q']:.3f}・条件 {E93.mark(cd['not_below_M'])}{E93.mark(cd['p_random'])}"
                  f"{E93.mark(cd['no_worse_elsewhere'])}・Q 対 M 下側 {e['area_M']['down']:.2f} / 上側 {e['area_M']['up']:.2f}pt・"
                  f"平均 {e['Q']['mean']:+.2f}%")

    print("\n■ まとめ")
    if ev["days"]:
        print(f"  候補の線 {LINE:.0f}以上・判断日 {ev['days']}日: {E93.MAIN_P}%点 Q {ev['Q'][f'p{E93.MAIN_P}']:+.2f} / M "
              f"{ev['M'][f'p{E93.MAIN_P}']:+.2f} / 無作為の中央 {ev['null_q'][E93.MAIN_P]['p50']:+.2f}（p = {ev['p_Q']:.3f}）→ {verdict}")
    else:
        print("  判断日が0日 → 判定できない")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ（実験91・93 と同じデータ。線は実験93 の結果を見た後に運用者が決めた）。"
          "手数料・税・すべりは含まない")
    summary["verdict"] = verdict
    return summary


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験95: 後工程の候補の線を90以上に緩めて、実験93 と同じ判定を回す")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    df = E91.attach_vol(df, fr)
    df["score"] = pd.to_numeric(df[f"score_{M.BASELINE}"], errors="coerce")
    del fr
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=250),
                        extra=("C", "AdjVo", "Vo"))
    df, chk = E83.attach_exits(df, bars, targets=(E93.TARGET,))
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行")
    dist = E91.event_distributions(bars, df[["Code", "Date"]])
    del bars
    split = E79.window_split(df, folds)
    starts = {f.index: pd.Timestamp(f.test_start) for f in folds}
    c, info = E93.collect(df, dist, split, starts, line=LINE, pool_line=POOL_LINE)
    _, info95 = E93.collect(df, dist, split, starts, line=STRICT, pool_line=POOL_LINE)
    log(f"線 {LINE:.0f}以上の行 {info['rule_rows']}件・判断日 {info['days']}日（線 {STRICT:.0f}: {info95['rule_rows']}件・{info95['days']}日。"
        f"実験93 は 559件・96日）")
    summary = report(c, info, draws=args.draws, n_days_strict_line=info95["days"])
    summary["exit_check"] = chk
    summary["strict_line"] = {k: v for k, v in info95.items() if k != "pool"}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e95_summary.json")

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
