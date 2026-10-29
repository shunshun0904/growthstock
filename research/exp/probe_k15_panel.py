#!/usr/bin/env python3
"""
選定の基準を「1.5σ の3本（lgbm_k15・xgb_k15・cat_k15）がすべて95以上」に戻すときの、画面の文言用の数字（見るだけ。判定はしない）。

運用者の依頼（2026-10-29）「候補銘柄の選定モデルに、1.2σではなく、1.5σのモデルにしてほしいです。」→ AskUserQuestion「すぐ
（今夜から）」（測り直しで 1.2σ の3本のほうが上なら、切り替える前に知らせる）。2026-10-23 の切り替え（1.5σ → 1.2σ。
docs/MODEL_ADOPTION_RULES.md §75・probe_t3_panel）の逆。今の OOF（360列。最後の再学習は 10/24）で、1.5σ の3本 95以上の画面の
文言用の数字と際どい形の数字（strategy.js の BORDER_STATS）を測り、1.2σ の3本 95以上と並べる。

probe_t3_panel との違い: 12本ではなく、規則に使う木の6本（1.2σ・1.5σ の lgbm / xgb / cat）だけを突き合わせる。外した 1.8σ の
4本の OOF は 2026-10-24 から作り直していないので、12本で突き合わせると、その後の再学習で増えた新しい行が落ちる。

出すもの（出口はどれも +10% の指値・届かなければ20営業日目の終値）
  0. 確かめ: 6本がそろう行・窓・期間
  1. 件数と 0件の月・重なり・+10% の成績（行全体・月2件の先着・画面の3枠の形）と窓ごとの差（1.5σ − 1.2σ）: 前向き・画面の物差し
  2. 画面の文言用（画面の物差し）: 行全体の +10% に届く割合と届いた日の分位・発火の帯・惜しい帯の幅（両方の規則）
  3. 前向きの発火の帯と、画面の3枠の形を発火を問わず / 発火8件以上の日だけ回した取引（実験88 の形。両方の規則）
  4. 際どい形 × 一番低いモデル（前向き・発火を問わない・1.5σ の3本）と strategy.js の BORDER_STATS の行
  5. まとめ: 1.5σ の3本のほうが 1.2σ の3本より上か（前向きの行全体・3枠の形。窓ごとの差）

公開ログには集計だけ（件数・平均・割合・分位）。銘柄・日付ごとの値は出さない。

    exp=probe_k15_panel.py
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
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e87_monthly_return_search as E87  # noqa: E402
import probe_k15_baseline as PKB  # noqa: E402
import probe_k15_border as PKX  # noqa: E402
import probe_t3_panel as T3P  # noqa: E402

#: 規則に使う木の6本（1.2σ の3本・1.5σ の3本）。外したモデルの OOF は読まない
SIX = tuple(E82.TREES) + tuple(E82.T15)
CODES = (("K15", "1.5σ の3本 95以上（新しい規則）"), ("T3", "1.2σ の3本 95以上（今までの規則）"))
#: E87.build に渡す組（6本だけでできるもの）
SETS = tuple(s for s in E87.SETS if s[0] in dict(CODES))
LINE = PKB.LINE                 # 95

log = E79.log


def load(data_dir: str = lab.DATA_DIR):
    """6本の OOF・実収益・窓・出口を付けた行（前向きと画面の物差しの2通りの百分位）。"""
    wide, info0 = E79.load_oofs(data_dir, models=SIX)
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    bars = LT.load_bars(data_dir, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = E83.attach_exits(df, bars, targets=E87.TARGETS)
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    return df, folds, bar_days, info0, chk


def frames(df: pd.DataFrame, folds: list):
    """前向き（ft。テスト窓より前の窓の分布での百分位）と画面の物差し（ff。全期間の分布）の行。最大の列も足す。"""
    split = E79.window_split(df, folds, models=SIX)
    parts_f = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
    parts_s = [(np.arange(len(df)), E79.pct_matrix(df, SIX))]
    ft = E87.build(df, parts_f, models=SIX, sets=SETS)
    ff = E87.build(df, parts_s, models=SIX, sets=SETS)
    for code, _ in CODES:
        ft = PKB.add_spread(ft, parts_f, code, models=SIX)
        ff = PKB.add_spread(ff, parts_s, code, models=SIX)
    return ft, ff, parts_f


def summary(fw: dict, fwd: dict) -> dict:
    """1.5σ の3本が上か（前向き）。行全体・画面の3枠の形の平均と、窓ごとの差（1.5σ − 1.2σ）。"""
    k, t = fwd["K15"], fwd["T3"]
    rows_up = bool(k["rows"]["mean"] > t["rows"]["mean"])
    slots_up = bool(k["slots_any"]["mean"] > t["slots_any"]["mean"])
    return {"rows_up": rows_up, "slots_up": slots_up, "k15_better": rows_up and slots_up,
            "diff": fw.get("diff", {})}


def main(argv=None) -> int:
    t0 = time.time()
    df, folds, bar_days, info0, chk = load()
    ft, ff, parts_f = frames(df, folds)
    d = pd.to_datetime(df["Date"])
    print(f"■ 0. 確かめ: 6本（{' / '.join(SIX)}）がそろう行 {info0['kept']:,}件・窓に入る {len(df):,}件・"
          f"{int(df['window'].nunique())}窓（{d.min().date()} 〜 {d.max().date()}）。出口: 20日目の終値 "
          f"{chk['with_close']:,}/{chk['rows']:,}行・到達の順序の食い違い {chk['order_violations']}件")

    print("\n■ 1. 件数と +10% の成績（窓ごとの差は「1.5σ − 1.2σ」）")
    fw = PKB.scale_block(ft, bar_days, "前向き")
    sw = PKB.scale_block(ff, bar_days, "画面の物差し")
    fw.pop("table")
    sw.pop("table")

    period = T3P.period_of(ff)
    print("\n■ 2. 画面の文言用（画面の物差し）")
    pn = {code: T3P.panel(ff, code) for code, _ in CODES}
    for code, ja in CODES:
        T3P.print_panel(pn[code], ja, period)

    print("\n■ 3. 前向きの発火の帯と、画面の3枠の形（実験88 の形）")
    fwd = {code: T3P.forward(ft, bar_days, code) for code, _ in CODES}
    for code, ja in CODES:
        T3P.print_forward(fwd[code], ja)

    print("\n■ 4. 際どい形 × 一番低いモデル（前向き・発火を問わない。1.5σ の3本。strategy.js の BORDER_STATS）")
    ftx = PKX.add_shapes(ft, parts_f, models=SIX)
    PKX.shape_block(ftx, "K15", "4a. 形ごと（前向き・発火を問わない・1.5σ の3本）")
    print("\n■ 4b. 形 × 一番低いモデル（画面に出す数字）")
    k15 = PKX.family_table(ftx, "K15")
    PKX.print_family(k15, "K15")
    reds = [f"{fam}・一番低い {a}" for fam, _, _ in PKX.FAMILIES for a in PKX.GROUPS["K15"] if k15[fam][a]["warn"]]
    print(f"  赤の注意の組（{PKX.WARN_MIN_N}件以上で平均 {PKX.WARN_RET:g}% 以下）: {'、'.join(reds) if reds else 'なし'}")
    print("  BORDER_STATS（js）:")
    for s in PKX.js_lines(k15, "K15"):
        print("js:" + s)

    sm = summary(fw, fwd)
    print("\n■ 5. まとめ（見るだけ。選定の基準は運用者が決めた。手数料・税・すべりは含まない）")
    a, b = pn["K15"]["all"], pn["T3"]["all"]
    print(f"  画面の物差し: 1.5σ の3本 95以上 {a['n']}件 {a['mean']:+.2f}%（届く {a['hit']:.1f}%）、1.2σ の3本 95以上 {b['n']}件 "
          f"{b['mean']:+.2f}%（届く {b['hit']:.1f}%）")
    x, y = fwd["K15"], fwd["T3"]
    print(f"  前向き: 1.5σ の3本 95以上 {T3P.short(x['rows'])}・3枠 {x['slots_any']['n']}取引 {x['slots_any']['mean']:+.2f}%、"
          f"1.2σ の3本 95以上 {T3P.short(y['rows'])}・3枠 {y['slots_any']['n']}取引 {y['slots_any']['mean']:+.2f}%")
    print("  前向きの窓ごとの差（1.5σ − 1.2σ）: " + "、".join(
        f"{ja} {E79.fmt_s(sm['diff'][k_])}" for k_, ja in (("rows", "行全体"), ("first2", "月2件の先着"), ("screen", "画面の3枠の形"))
        if k_ in sm["diff"]))
    print(f"  → 前向きの行全体と画面の3枠の形の平均は、{'どちらも 1.5σ の3本のほうが上' if sm['k15_better'] else '1.5σ の3本のほうが上とは言えない（運用者に知らせる）'}")
    log(f"全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
