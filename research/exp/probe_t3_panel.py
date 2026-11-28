#!/usr/bin/env python3
"""
選定の基準を「1.2σ の3本（lgbm・xgb・cat）がすべて95以上」にするときの、画面の文言用の数字（見るだけ。判定はしない）。

運用者の依頼（2026-10-23）「選定の基準ですが1.2σにしてほしいです」→ AskUserQuestion で「1.2σ の3本 95以上」（線は今の 95 の
まま）・「すぐ切り替える」。画面の今日の戦略・惜しい候補・際どい印（src/lib/strategy.js・src/views/PredictionView.jsx）の数字は、
1.5σ の3本 95以上を 2026-10-12 の OOF（249列のころ）で測ったもの（probe_k15_baseline・probe_k15_border・実験88）。今の OOF
（360列）で、1.2σ の3本 95以上の同じ数字を測る。並べて 1.5σ の3本 95以上も今の OOF で出す（見るだけ）。

出すもの（出口はどれも +10% の指値・届かなければ20営業日目の終値）
  0. 確かめ: 前向きの 1.5σ の3本 95以上の件数（実験110 は 523件。同じ OOF なら同じ）
  1. 件数と 0件の月（前向き・画面の物差し）、+10% の成績（行全体・月2件の先着・画面の3枠の形）: probe_k15_baseline の 1〜3
  2. 画面の文言用（画面の物差し・1.2σ の3本 95以上）: 行全体で +10% に届く割合と届いた日の分位、発火の帯（20件以上 /
     8〜19件 / 7件以下）、惜しい帯（3本の最小 85〜95・発火8件以上の日）の幅（5 未満 / 5 以上）、out-of-fold の期間
  3. 前向きの発火の帯（7件以下 / 8件以上）と、画面の3枠の形を発火を問わず回した取引・発火8件以上の日だけの取引（実験88 の形）
  4. 際どい形 × 一番低いモデル（前向き・発火を問わない。probe_k15_border の形）と、strategy.js の BORDER_STATS の行

公開ログには集計だけ（件数・平均・割合・分位）。銘柄・日付ごとの値は出さない。

    exp=probe_t3_panel.py

結果（2026-10-23、手元・33秒。12本の OOF は 360列の学習で Release と同じ版。docs/MODEL_ADOPTION_RULES.md §75）
------------------------------------------------------------------
  確かめ: 前向きの 1.5σ の3本 95以上 523件（実験110 と同じ）
  前向き: 1.2σ の3本 95以上 467件 +2.59%（0件の月 12 / 53）・月2件の先着 73件 −1.03%・3枠（発火8件以上）89件 +0.50%、
        1.5σ の3本 95以上 523件 +2.72%（6 / 53）・84件 +0.16%・107件 +2.03%。窓ごとの差（1.5σ − 1.2σ）+1.09 ± 0.58 /
        +1.39 ± 0.96 / +2.21 ± 0.95pt。3枠・発火を問わない: 1.2σ 100取引 +0.15%・1.5σ 121取引 +1.61%
  画面の物差し: 1.2σ 339件 +2.78%（届く 37.5%・届いた日の中央 9営業日）、1.5σ 328件 +3.57%。1.2σ の帯: 発火20件以上 204件
        +3.54%・8〜19件 112件 +1.50%・7件以下 23件 +2.30%。惜しい帯 幅5未満 335件 +1.49%・幅5以上 576件 +2.28%
  前向きの発火の帯（1.2σ）: 7件以下 31件 +1.49%（−10%割れ 16.1%）・8件以上 436件 +2.67%
  際どい形（前向き・1.2σ）: 2本95以上 462件 +2.00%（LightGBM 212件 +1.69% / XGBoost 140件 +1.98% / CatBoost 110件 +2.61%）、
        1本以下・最小 85〜95 700件 +1.56%（277件 +1.50% / 257件 +1.15% / 166件 +2.30%）。赤の組なし
"""
from __future__ import annotations

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
import e83_tp10_rule as E83  # noqa: E402
import e87_monthly_return_search as E87  # noqa: E402
import probe_k15_baseline as PKB  # noqa: E402
import probe_k15_border as PKX  # noqa: E402

MODELS = E79.MODELS
LINE = PKB.LINE                 # 95
CODES = (("T3", "1.2σ の3本 95以上"), ("K15", "1.5σ の3本 95以上（今までの規則・見るだけ）"))
E110_K15 = 523                  # 実験110 の前向きの今の規則の行（360列の OOF）

log = E79.log


def period_of(ff: pd.DataFrame) -> tuple[str, str]:
    """out-of-fold の期間（月）。"""
    d = pd.to_datetime(ff["Date"])
    return (str(d.min().to_period("M")), str(d.max().to_period("M"))) if len(d) else ("-", "-")


def panel(ff: pd.DataFrame, code: str) -> dict:
    """画面の文言用（画面の物差し）: 行全体・発火の帯・惜しい帯の幅。probe_k15_baseline.panel_block を組 code で。"""
    k = PKB.rule_mask(ff, code)
    nb = ff["n_break"].to_numpy()
    out = {"all": PKB.stats(ff[k]), "bands": {}}
    for lo, hi, name in PKB.BANDS:
        m = k & (nb >= lo)
        if hi is not None:
            m = m & (nb <= hi)
        out["bands"][name] = PKB.stats(ff[m])
    mn = ff[f"min_{code}"].to_numpy(dtype=float)
    sp = ff[f"max_{code}"].to_numpy(dtype=float) - mn
    near = (mn >= PKB.NEAR_LO) & (mn < LINE) & (nb >= LT.PREV_SKIP_BREAKS)
    out["near"] = {"narrow": PKB.stats(ff[near & (sp < PKB.WIDE)]), "wide": PKB.stats(ff[near & (sp >= PKB.WIDE)])}
    return out


def forward(ft: pd.DataFrame, bar_days, code: str) -> dict:
    """前向き（実験88 の形）: 行全体・発火7件以下 / 8件以上、画面の3枠の形を発火を問わず / 発火8件以上の日だけ回した取引。"""
    k = PKB.rule_mask(ft, code)
    nb = ft["n_break"].to_numpy()
    months = int(pd.to_datetime(ft["Date"]).dt.to_period("M").nunique())
    out = {"rows": PKB.stats(ft[k]), "quiet": PKB.stats(ft[k & (nb < LT.PREV_SKIP_BREAKS)]),
           "busy": PKB.stats(ft[k & (nb >= LT.PREV_SKIP_BREAKS)])}
    for name, b in (("any", 1), ("busy8", LT.PREV_SKIP_BREAKS)):
        t = E87.run_one(ft, bar_days, code, LINE, b, LT.TOP_K, LT.SLOTS, 0, 10.0)
        s = PKB.stats(t)
        r = pd.to_numeric(t["tp10c"], errors="coerce").dropna()
        s["capital_pm"] = float(r.sum() / LT.SLOTS / months) if months else float("nan")
        out[f"slots_{name}"] = s
    return out


def short(s: dict) -> str:
    if not s["n"]:
        return "0件"
    return f"{s['n']}件 {s['mean']:+.2f}%（勝率 {s['win']:.1f}%・−10%割れ {s['bad10']:.1f}%）"


def print_panel(p: dict, ja: str, period: tuple[str, str]) -> None:
    a = p["all"]
    print(f"  {ja}（画面の物差し・out-of-fold {period[0]}〜{period[1]}）: 行全体 {a['n']}件 +10% {a['mean']:+.2f}%"
          f"（中央 {a['med']:+.2f}・勝率 {a['win']:.1f}%・−10%割れ {a['bad10']:.1f}%）。+10% に届く {a['hit']:.1f}%・"
          f"届いた日 25/50/75% 点 {a['day25']:.0f} / {a['day50']:.0f} / {a['day75']:.0f}営業日（1 = 買った日）")
    for name, s in p["bands"].items():
        print(f"    {LT._l(name, 14)}{short(s)}")
    print(f"    惜しい帯（最小 {PKB.NEAR_LO:g}〜{LINE:g}・発火{LT.PREV_SKIP_BREAKS}件以上の日）幅{PKB.WIDE:g}未満 "
          f"{short(p['near']['narrow'])}・幅{PKB.WIDE:g}以上 {short(p['near']['wide'])}")


def print_forward(f: dict, ja: str) -> None:
    print(f"  {ja}（前向き）: 行全体 {short(f['rows'])}。発火{LT.PREV_SKIP_BREAKS - 1}件以下 {short(f['quiet'])}・"
          f"{LT.PREV_SKIP_BREAKS}件以上 {short(f['busy'])}")
    for key, name in (("slots_any", "画面の3枠の形・発火を問わない"), ("slots_busy8", f"画面の3枠の形・発火{LT.PREV_SKIP_BREAKS}件以上の日だけ")):
        s = f[key]
        print(f"    {LT._l(name, 30)}{s['n']}取引 1取引 {s['mean']:+.2f}%（勝率 {s['win']:.1f}%）・資金全体で月 "
              f"{s['capital_pm']:+.2f}%（1件に1/3）")


def main(argv=None) -> int:
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = E83.attach_exits(df, bars, targets=E87.TARGETS)
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・到達の順序の食い違い {chk['order_violations']}件")
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    split = E79.window_split(df, folds)
    parts_f = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
    parts_s = [(np.arange(len(df)), E79.pct_matrix(df, MODELS))]
    ft = E87.build(df, parts_f)
    ff = E87.build(df, parts_s)
    for code, _ in CODES:
        ft = PKB.add_spread(ft, parts_f, code)
        ff = PKB.add_spread(ff, parts_s, code)

    k15 = int(PKB.rule_mask(ft, "K15").sum())
    print(f"■ 0. 確かめ: 前向きの 1.5σ の3本 95以上 {k15}件（実験110 は {E110_K15}件）"
          + ("" if k15 == E110_K15 else "  ← 実験110 と違う（OOF が変わった？）"))

    print("\n■ 1. 件数と +10% の成績（probe_k15_baseline の 1〜3。左が 1.5σ・右が 1.2σ の差は「1.5σ − 1.2σ」）")
    fw = PKB.scale_block(ft, bar_days, "前向き")
    sw = PKB.scale_block(ff, bar_days, "画面の物差し")
    fw.pop("table")
    sw.pop("table")

    period = period_of(ff)
    print("\n■ 2. 画面の文言用（画面の物差し）")
    pn = {code: panel(ff, code) for code, _ in CODES}
    for code, ja in CODES:
        print_panel(pn[code], ja, period)

    print("\n■ 3. 前向きの発火の帯と、画面の3枠の形（実験88 の形）")
    fwd = {code: forward(ft, bar_days, code) for code, _ in CODES}
    for code, ja in CODES:
        print_forward(fwd[code], ja)

    print("\n■ 4. 際どい形 × 一番低いモデル（前向き・発火を問わない。1.2σ の3本。strategy.js の BORDER_STATS）")
    ftx = PKX.add_shapes(ft, parts_f)
    PKX.shape_block(ftx, "T3", "4a. 形ごと（前向き・発火を問わない・1.2σ の3本）")
    print("\n■ 4b. 形 × 一番低いモデル（画面に出す数字）")
    t3 = PKX.family_table(ftx, "T3")
    PKX.print_family(t3, "T3")
    reds = [f"{fam}・一番低い {a}" for fam, _, _ in PKX.FAMILIES for a in PKX.GROUPS["T3"] if t3[fam][a]["warn"]]
    print(f"  赤の注意の組（{PKX.WARN_MIN_N}件以上で平均 {PKX.WARN_RET:g}% 以下）: {'、'.join(reds) if reds else 'なし'}")
    print("  BORDER_STATS（js）:")
    for s in PKX.js_lines(t3, "T3"):
        print("js:" + s)

    print("\n■ まとめ")
    a, b = pn["T3"]["all"], pn["K15"]["all"]
    print(f"  画面の物差し: 1.2σ の3本 95以上 {a['n']}件 {a['mean']:+.2f}%（届く {a['hit']:.1f}%）、1.5σ の3本 95以上 {b['n']}件 "
          f"{b['mean']:+.2f}%（届く {b['hit']:.1f}%）")
    x, y = fwd["T3"], fwd["K15"]
    print(f"  前向き: 1.2σ の3本 95以上 {short(x['rows'])}・3枠 {x['slots_any']['n']}取引 {x['slots_any']['mean']:+.2f}%、"
          f"1.5σ の3本 95以上 {short(y['rows'])}・3枠 {y['slots_any']['n']}取引 {y['slots_any']['mean']:+.2f}%")
    print("  注意: 見るだけ。選定の基準は運用者が決めた（1.2σ の3本 95以上）。手数料・税・すべりは含まない")
    log(f"全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
