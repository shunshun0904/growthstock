#!/usr/bin/env python3
"""
画面の戦略パネルの出口を +10% の指値・届かなければ 20営業日目の終値にしたときの、指値に届く割合と届くまでの日数
（運用者の指示 2026-10-12「実際の売りに合わせて +10%に変えますか？」「お願いします」）。

画面の手順の文言（「到達は約X割、届くときの中央値はY営業日」）の数字を、今の本番モデルの OOF で測る。規則は画面と
同じ（live_track.decide: 発火8件以上・3モデルすべて 90 以上・上位2件、3枠・先着順）。百分位は live_track の見込みと
同じ全期間の分布。参考に運用者の目安（3モデルとも 95 以上）と、以前の出口（+20% の指値・5日平均）も並べる。
選定日から20営業日が過ぎていない取引は数えない。公開ログには集計だけ（件数・割合・分位）。

    exp=probe_tp10_panel.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live_track as LT  # noqa: E402

EXITS = ((10.0, "close", "+10% の指値・20日目の終値（新）"), (20.0, "ma5", "+20% の指値・5日平均（以前）"))
PANEL_AGREE = 90.0       # 記録した時の画面の規則の線（1.2σ の3本）


def exit_summary(pf: pd.DataFrame, bar_days, n_days: int) -> dict:
    """選定した行（候補）と、3枠で買えた取引の、指値に届く割合・届いた日の分位・収益。終わっていない取引は除く。"""
    done = pf[pf["status"].isin(LT.HIT_LABELS + ("満了",))]
    hit = done["status"].isin(LT.HIT_LABELS)
    d = pd.to_numeric(done.loc[hit, "hit_day"], errors="coerce").dropna()
    r = pd.to_numeric(done["ret"], errors="coerce").dropna()
    sim = LT.simulate(pf, bar_days)
    taken = sim[(sim["taken"] == LT.TAKEN) & sim["status"].isin(LT.HIT_LABELS + ("満了",))]
    st = LT.trade_stats(taken, n_days)
    th = taken["status"].isin(LT.HIT_LABELS)
    td = pd.to_numeric(taken.loc[th, "hit_day"], errors="coerce").dropna()
    q = (lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan"))
    return {"n": int(len(done)), "hit": float(hit.mean() * 100) if len(done) else float("nan"),
            "day25": q(d, 25), "day50": q(d, 50), "day75": q(d, 75),
            "mean": float(r.mean()) if len(r) else float("nan"), "med": float(r.median()) if len(r) else float("nan"),
            "win": float((r > 0).mean() * 100) if len(r) else float("nan"),
            "t_n": st["n"], "t_hit": st["hit"], "t_day50": q(td, 50), "t_mean": st["mean"], "t_med": st["median"],
            "t_win": st["win"], "t_monthly": st["monthly"], "t_per_month": st["per_month"]}


def main() -> int:
    # 規則は記録した時（2026-10-12）の画面の規則: 1.2σ の3本・90以上。その後 1.5σ の3本・95以上になったので明示して固定する
    rows = LT.oof_rows(LT.DATA_DIR, LT.MODEL_DIR, algos=LT.BOOST)
    bars = LT.load_bars(LT.DATA_DIR, start=pd.Timestamp("2021-01-01"))
    bar_days = sorted(pd.Timestamp(d) for d in bars["Date"].unique())
    print(f"[oof] 候補 {len(rows):,}件（{rows['Date'].min().date()} 〜 {rows['Date'].max().date()}）。"
          f"規則: 発火{LT.SKIP_BREAKS}件以上・上位{LT.TOP_K}件・{LT.SLOTS}枠・{LT.HOLD}営業日。今の指値の設定 +{LT.TAKE_PROFIT:g}%"
          f"・届かなければ {LT.FALLBACK}")
    for agree in (PANEL_AGREE, 95.0):
        r, days, picks = LT.decide(rows, agree=agree, models=LT.BOOST)
        n_days = LT.n_bar_days(bar_days, r["Date"].min(), r["Date"].max())
        print(f"\n■ 3モデルすべて {agree:g} 以上（{'画面の規則' if agree == PANEL_AGREE else '運用者の目安'}）: 選んだ候補 {len(picks):,}件")
        for target, fb, name in EXITS:
            pf = LT.forward(bars, picks, target=target, fallback=fb)
            s = exit_summary(pf, bar_days, n_days)
            print(f"  {LT._l(name, 32)}候補 {s['n']:>4}件: 到達 {s['hit']:4.1f}%・到達日 25/50/75% 点 {s['day25']:.0f} / "
                  f"{s['day50']:.0f} / {s['day75']:.0f}営業日（1 = 買った日）・1件 {s['mean']:+.2f}%（中央 {s['med']:+.2f}）・"
                  f"勝率 {s['win']:.1f}%")
            print(f"  {LT._l('', 32)}3枠で買えた {s['t_n']:>4}件（月 {s['t_per_month']:.1f}件）: 到達 {s['t_hit']:4.1f}%・到達日の中央 "
                  f"{s['t_day50']:.0f}営業日・1取引 {s['t_mean']:+.2f}%（中央 {s['t_med']:+.2f}）・勝率 {s['t_win']:.1f}%・"
                  f"月の収益 {s['t_monthly']:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
