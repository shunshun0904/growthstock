#!/usr/bin/env python3
"""
実験37: ROE と進捗率は、帯ごとに効いているか。

きっかけ（運用者、2026-09-22）
  クニミネ工業(5388) は EPS成長率 49.6% と強いのに、ROE 6.19%・
  進捗率の基準比 −3.82% が弱く、モデルの決算寄与は −0.14 だった。
  この2つが本当に帯ごとに効いているのかを確かめたい。

測る軸
  ROE_q0            直近の ROE（%）。API の値が無ければ TTM 純利益 ÷ 自己資本
  progress_vs_base  進捗率 − 四半期 × 25。通期会社予想営業利益に対する
                    当期累計の進み具合を、四半期の線形ペースからの差で見る
                    （0 なら計画どおり、負なら遅れ）

見るもの
  1. 帯ごとの実収益（母集団 / 基準通過 / 惜しい帯）。単調か、山形か
  2. 実験20・23 と同じ両側スクリーニング（窓ごとの上位/下位10%の超過、z）
  3. **基準を通ったあとに、さらにこの2つで絞ると良くなるか**（運用の問い）
  4. クニミネの形（EPS成長は高いのに ROE が低い）が実在する悪い組み合わせか

物差しは ret_o1_20（翌営業日の寄りで買い、20営業日）。
結果は research/_data/oof/e37_*.csv。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import e23_dimension_screen as E23  # noqa: E402
import walkforward as WF  # noqa: E402
from e23_dimension_screen import screen_both  # noqa: E402
from e27_timing_multi import OOF_DIR, log  # noqa: E402
from e31_strategy import prepare  # noqa: E402
from e36_score_line import hist_pct  # noqa: E402

AXES = {
    "ROE_q0": ("ROE（%）", [-1e9, 0, 5, 8, 12, 18, 1e9],
               ["赤字(<0)", "0〜5", "5〜8", "8〜12", "12〜18", "18〜"]),
    "progress_vs_base": ("進捗率の基準比（pt）", [-1e9, -20, -10, 0, 10, 20, 1e9],
                         ["−20未満", "−20〜−10", "−10〜0", "0〜10", "10〜20", "20〜"]),
    "eps_growth_q0": ("EPS成長率（%）", [-1e9, 0, 10, 25, 50, 100, 1e9],
                      ["減益(<0)", "0〜10", "10〜25", "25〜50", "50〜100", "100〜"]),
}

HEAD = (f"  {'帯':<14}{'件数':>7}{'持ち切り':>10}{'勝率':>7}{'正例率':>7}{'SE':>7}"
        f"{'−10%割れ':>9}")


def band_line(nm: str, g: pd.DataFrame, min_n: int = 20) -> str:
    if len(g) < min_n:
        return f"  {nm:<14}{len(g):>7}   （件数が少ない）"
    r = g["r"]
    return (f"  {nm:<14}{len(g):>7}{r.mean():>+9.2f}%{(r > 0).mean()*100:>6.0f}%"
            f"{g['label'].mean()*100:>6.0f}%{r.std(ddof=1)/np.sqrt(len(r)):>7.2f}"
            f"{(r < -10).mean()*100:>8.1f}%")


def bands(d: pd.DataFrame, col: str) -> pd.Series:
    _, edges, names = AXES[col]
    return pd.cut(d[col], edges, labels=names, right=False)


def main() -> int:
    d = prepare()
    d = hist_pct(d)
    fr = lab.frame()
    fr["Date"] = pd.to_datetime(fr["Date"])
    need = [c for c in AXES if c not in d.columns]
    if need:
        d = d.merge(fr[["Code", "Date"] + need], on=["Code", "Date"], how="left")
    d = d[d["r"].notna() & d["hp_min"].notna()].copy()
    log(f"対象 {len(d):,}件 / {d['Date'].min().date()}〜{d['Date'].max().date()}")
    for c in AXES:
        log(f"  {c:<18}充足 {d[c].notna().mean()*100:.1f}%")

    pops = {
        "母集団（高値更新の全件）": d,
        "基準通過（3モデル90以上）": d[d["hp_min"] >= 90],
        "惜しい帯（85〜90）": d[(d["hp_min"] >= 85) & (d["hp_min"] < 90)],
    }
    rows = []
    for col, (ja, _, _) in AXES.items():
        print(f"\n=== {ja} の帯ごと ===")
        for pname, pop in pops.items():
            g = pop[pop[col].notna()].copy()
            if len(g) < 100:
                print(f"\n  ▼ {pname}   （件数が少ない: {len(g)}）")
                continue
            print(f"\n  ▼ {pname}（{len(g):,}件、全体 {g['r'].mean():+.2f}%）")
            print(HEAD)
            g["帯"] = bands(g, col)
            for b, gg in g.groupby("帯", observed=True):
                print(band_line(str(b), gg))
                rows.append({"axis": col, "pop": pname, "band": str(b), "n": len(gg),
                             "hold": gg["r"].mean(), "win": (gg["r"] > 0).mean()*100,
                             "label": gg["label"].mean()*100,
                             "lose10": (gg["r"] < -10).mean()*100})
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e37_bands.csv"), index=False)

    # スクリーニングは out-of-fold の部分集合ではなく**母集団の全期間**で回す。
    # 部分集合だと窓が3本しか取れず、z がまったく当てにならない。
    print(f"\n=== 両側スクリーニング（実験20・23 と同じ。窓ごとの上位/下位10%の超過）===")
    folds = WF.make_folds(fr["Date"], min_train_months=E23.OOF_MIN_TRAIN_MONTHS,
                          test_months=E23.OOF_TEST_MONTHS,
                          step_months=E23.OOF_STEP_MONTHS,
                          embargo_days=E23.B.RISE_HORIZON)
    wins = [(np.datetime64(f.test_start), np.datetime64(f.test_end)) for f in folds]
    cols = [c for c in ("guidance_op_growth", "op_margin_q0", "sales_growth_q0",
                        "ROE_q0", "progress_vs_base", "ROA_q0", "equity_ratio_q0",
                        "eps_growth_q0") if c in fr.columns]
    print(f"  母集団 {len(fr):,}件 / 評価窓 {len(wins)}本 / 物差し ret_o1_20")
    print(f"  {'特徴量':<20}{'充足':>7}{'窓AUC':>8}{'上位10%':>10}{'z':>7}{'勝窓':>7}"
          f"{'下位10%':>10}{'z':>7}{'勝窓':>7}")
    sc = screen_both(fr, cols, wins)
    for _, r in sc.iterrows():
        n = int(r["n_win"])
        print(f"  {r['feature']:<20}{r['coverage']*100:>6.0f}%{r['auc_win']:>8.3f}"
              f"{r['top_pt']:>+9.2f}pt{r['top_z']:>+7.2f}{int(r['top_pos']):>5}/{n:<2}"
              f"{r['bot_pt']:>+9.2f}pt{r['bot_z']:>+7.2f}{int(r['bot_pos']):>5}/{n:<2}")
    sc.to_csv(os.path.join(OOF_DIR, "e37_screen.csv"), index=False)
    print(f"  ※ 足切りは |z| > 2（docs/MODEL_ADOPTION_RULES.md）。"
          f"上位10%が負 = その指標が高いほど悪い")

    print(f"\n=== 基準を通ったあと、さらに絞ると良くなるか（3モデル90以上、"
          f"{len(pops['基準通過（3モデル90以上）']):,}件）===")
    sel = pops["基準通過（3モデル90以上）"]
    print(HEAD)
    print(band_line("絞らない", sel))
    for col, (ja, _, _) in AXES.items():
        g = sel[sel[col].notna()]
        if len(g) < 100:
            continue
        med = g[col].median()
        print(f"  --- {ja}（通過組の中央値 {med:.2f}）---")
        print(band_line("中央値以上", g[g[col] >= med]))
        print(band_line("中央値未満", g[g[col] < med]))
    # 進捗率は 0（計画どおり）に意味のある線がある
    g = sel[sel["progress_vs_base"].notna()]
    print(f"  --- 進捗率: 計画に対して進んでいるか ---")
    print(band_line("0以上（計画以上）", g[g["progress_vs_base"] >= 0]))
    print(band_line("0未満（遅れ）", g[g["progress_vs_base"] < 0]))

    print(f"\n=== クニミネの形: EPS成長は高いのに ROE が低い ===")
    g = d[d["eps_growth_q0"].notna() & d["ROE_q0"].notna()].copy()
    hi_eps = g["eps_growth_q0"] >= 25
    lo_roe = g["ROE_q0"] < 8
    print(f"  母集団 {len(g):,}件（EPS成長25%以上 {int(hi_eps.sum()):,}件 / "
          f"ROE 8%未満 {int(lo_roe.sum()):,}件）")
    print(HEAD)
    for nm, m in (("EPS高・ROE低", hi_eps & lo_roe), ("EPS高・ROE高", hi_eps & ~lo_roe),
                  ("EPS低・ROE低", ~hi_eps & lo_roe), ("EPS低・ROE高", ~hi_eps & ~lo_roe)):
        print(band_line(nm, g[m]))
    print(f"\n  同じ切り口を基準通過組（3モデル90以上）で:")
    s = g[g["hp_min"] >= 90]
    he, lr = s["eps_growth_q0"] >= 25, s["ROE_q0"] < 8
    print(HEAD)
    for nm, m in (("EPS高・ROE低", he & lr), ("EPS高・ROE高", he & ~lr),
                  ("EPS低・ROE低", ~he & lr), ("EPS低・ROE高", ~he & ~lr)):
        print(band_line(nm, s[m]))
    log(f"記録: {OOF_DIR}/e37_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
