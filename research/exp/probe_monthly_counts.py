#!/usr/bin/env python3
"""
月ごとの候補数: 6本とも95以上（A15）と3本とも95以上（R0）の候補が月に何件あったか、0件の月はいつか。

運用者の問い（2026-10-12）「シンプルに6モデル95以上、ないしは3モデル95以上だと候補がない月の推移をみたいです。
候補数の推移グラフとかでよいかもしれません」。

物差しは2つ（丸めは画面と同じ小数1桁、線は 95）
  画面の物差し: 全期間の百分位（実験84・85 の「画面の物差し」と同じ。今の画面と同じ作りで、先の分布も含む）。窓 1〜11
  前向き: その窓より前の分布での百分位（実験79 の window_split。その時点で見えたはずの値に近い）。テスト窓 3〜11
数えるもの（月ごと）: 発火（候補全体）の件数・候補のある日数・発火8件以上の日数、物差しごとの R0 と A15 の件数（行 = 銘柄 × 日）
  と銘柄数、発火8件以上の日に限った R0 と A15 の件数。0件の月の数・続いた最長・年ごと
確かめ: 前向きの R0・A15 の合計を実験82・83 の数（543件・333件）と比べる

公開ログには月ごとの件数だけ。銘柄・日付ごとの値は出さない。

    exp=probe_monthly_counts.py

結果（2026-10-12、run 38207797677、f5da447、31秒。docs/MODEL_ADOPTION_RULES.md §44）
------------------------------------------------------------------
  確かめ: 前向きの合計 R0 543件・A15 333件（実験82・83 と同じ）。前向きの0件の月 R0 11・A15 14 も実験84〜86 と同じ
  0件の月: 画面の物差し R0 29 / 65か月（最長 6か月続く、2021-05〜10）・A15 34 / 65（最長 11か月、2021-12〜2022-10）
           前向き R0 11 / 53（最長 2か月）・A15 14 / 53（最長 4か月、2022-06〜09）
           発火8件以上の日に限ると 画面の物差し R0 33・A15 36、前向き R0 15・A15 17
  年ごと（R0 / A15）: 画面の物差し 2021年 7 / 7（8か月）・2022年 9 / 11・2023年 5 / 6・2024年 2 / 3・2025年 3 / 4・
           2026年 3 / 3（9か月。9月は8日までの6日分）。前向き 2022年 3 / 4（8か月）・2023年 1 / 3・2024年 1 / 1・2025年 3 / 3・
           2026年 3 / 3
  集中: 4か月（2024年1・2月、2025年7・8月）に R0 の 55〜61%、A15 の 57〜66% が入る（前向き / 画面の物差し）
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

MODELS = E79.MODELS
RULES = ("R0", "A15")
RULE_JA = {"R0": "3本とも95以上（R0）", "A15": "6本とも95以上（A15）"}
SCALES = (("s", "画面の物差し"), ("f", "前向き"))
MIN_BREAK = LT.SKIP_BREAKS
CSV_COLS = ("month", "breaks", "days", "hot_days") + tuple(
    f"{s}_{k}" for s, _ in SCALES for k in ("r0", "a15", "r0_codes", "a15_codes", "r0_hot", "a15_hot"))

log = E79.log


def flag_rows(df: pd.DataFrame, parts: list, models=MODELS) -> pd.DataFrame:
    """parts: [(行番号, 丸める前の百分位の行列)]。行ごとに Date・Code・window と R0・A15 の印（画面と同じ丸め）。"""
    outs = []
    for rows, P in parts:
        fl = E82.flags_of(E82.rounded(P), models)
        g = df.iloc[rows][["Code", "Date", "window"]].reset_index(drop=True)
        g["Date"] = pd.to_datetime(g["Date"])
        for k in RULES:
            g[k] = np.asarray(fl[k], dtype=bool)
        outs.append(g)
    return pd.concat(outs, ignore_index=True)


def monthly(df: pd.DataFrame, scales: dict, min_break: int = MIN_BREAK) -> pd.DataFrame:
    """
    月ごとの件数。df = 候補全体（発火を数える）、scales = {"s": flag_rows の結果, "f": ...}。物差しが覆わない月は NaN。
    発火 = その日の候補の行数（live_track.decide と同じ数え方）。
    """
    d = pd.to_datetime(df["Date"])
    per_day = d.value_counts().sort_index()
    m_all = pd.period_range(d.min().to_period("M"), d.max().to_period("M"), freq="M")
    day_m = per_day.index.to_period("M")
    out = pd.DataFrame(index=m_all)
    out["breaks"] = pd.Series(per_day.to_numpy(), index=day_m).groupby(level=0).sum().reindex(m_all, fill_value=0)
    out["days"] = pd.Series(1, index=day_m).groupby(level=0).sum().reindex(m_all, fill_value=0)
    out["hot_days"] = pd.Series((per_day >= min_break).astype(int).to_numpy(), index=day_m).groupby(level=0).sum() \
        .reindex(m_all, fill_value=0)
    hot_dates = set(per_day.index[per_day >= min_break])
    for s, fl in scales.items():
        g = fl.assign(_m=fl["Date"].dt.to_period("M"), _hot=fl["Date"].isin(hot_dates))
        covered = pd.Index(g["_m"].unique())
        for k in RULES:
            x = g[g[k]]
            cnt = x.groupby("_m").size().reindex(m_all, fill_value=0).astype(float)
            codes = x.groupby("_m")["Code"].nunique().reindex(m_all, fill_value=0).astype(float)
            hot = x[x["_hot"]].groupby("_m").size().reindex(m_all, fill_value=0).astype(float)
            miss = ~m_all.isin(covered)
            cnt[miss] = codes[miss] = hot[miss] = np.nan
            out[f"{s}_{k.lower()}"], out[f"{s}_{k.lower()}_codes"], out[f"{s}_{k.lower()}_hot"] = cnt, codes, hot
    return out


def zero_runs(x: pd.Series) -> dict:
    """0件の月の数・続いた最長（月数と始まり・終わりの月）。NaN の月は数えない。"""
    v = x.dropna()
    z = (v == 0).to_numpy()
    best, cur, best_end = 0, 0, None
    for i, flag in enumerate(z):
        cur = cur + 1 if flag else 0
        if cur > best:
            best, best_end = cur, i
    out = {"months": int(len(v)), "zero": int(z.sum()), "longest": int(best)}
    if best:
        out["longest_from"] = str(v.index[best_end - best + 1])
        out["longest_to"] = str(v.index[best_end])
    return out


def report(tbl: pd.DataFrame) -> dict:
    summary: dict = {}
    print(f"\n■ 1. 月ごとの件数（行 = 銘柄 × 日。発火{MIN_BREAK}件以上 = その日の候補が{MIN_BREAK}件以上）")
    print("  月        発火  日数  8件以上の日 ｜ 画面の物差し R0  A15（銘柄 R0 / A15・8件以上の日の R0 / A15）"
          "｜ 前向き R0  A15（銘柄 R0 / A15・8件以上の日の R0 / A15）")

    def f(v):
        return "  —" if pd.isna(v) else f"{int(v):>3}"

    for m, r in tbl.iterrows():
        print(f"  {m}  {int(r['breaks']):>5}  {int(r['days']):>3}  {int(r['hot_days']):>3}      ｜ "
              f"{f(r['s_r0'])} {f(r['s_a15'])}（{f(r['s_r0_codes'])} /{f(r['s_a15_codes'])}・{f(r['s_r0_hot'])} /{f(r['s_a15_hot'])}）"
              f"｜ {f(r['f_r0'])} {f(r['f_a15'])}（{f(r['f_r0_codes'])} /{f(r['f_a15_codes'])}・{f(r['f_r0_hot'])} /"
              f"{f(r['f_a15_hot'])}）")
    print(f"\n■ 2. 0件の月（月の数・続いた最長）")
    for s, name in SCALES:
        for k in RULES:
            for suf, extra in (("", ""), ("_hot", f"・発火{MIN_BREAK}件以上の日に限る")):
                col = f"{s}_{k.lower()}{suf}"
                z = zero_runs(tbl[col])
                summary[col] = z
                tail = f"・最長 {z['longest']}か月（{z['longest_from']} 〜 {z['longest_to']}）" if z["longest"] else ""
                print(f"  {LT._l(name + '・' + RULE_JA[k] + extra, 60)}{z['zero']:>2} / {z['months']}か月{tail}")
    print("\n■ 3. 年ごとの0件の月（R0 / A15）")
    yr = tbl.index.year
    for s, name in SCALES:
        parts = []
        for y in sorted(set(yr)):
            t = tbl[yr == y]
            r0, a15 = t[f"{s}_r0"].dropna(), t[f"{s}_a15"].dropna()
            if len(r0):
                parts.append(f"{y}年 {int((r0 == 0).sum())} / {int((a15 == 0).sum())}（{len(r0)}か月）")
        print(f"  {name}: " + "  ".join(parts))
    print("\n■ 4. 合計（確かめ: 前向きの R0・A15 は実験82・83 では 543件・333件）")
    for s, name in SCALES:
        print(f"  {name}: R0 {int(np.nansum(tbl[f'{s}_r0']))}件・A15 {int(np.nansum(tbl[f'{s}_a15']))}件"
              f"（覆う月 {int(tbl[f'{s}_r0'].notna().sum())}）")
        summary[f"{s}_total"] = {"r0": int(np.nansum(tbl[f"{s}_r0"])), "a15": int(np.nansum(tbl[f"{s}_a15"]))}
    print("\n■ 5. グラフ用（csv）")
    print("csv:" + ",".join(CSV_COLS))
    for m, r in tbl.iterrows():
        vals = [str(m)] + ["" if pd.isna(r[c]) else str(int(r[c])) for c in CSV_COLS[1:]]
        print("csv:" + ",".join(vals))
    return summary


def main(argv=None) -> int:
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = wide.copy()
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    split = E79.window_split(df, folds)
    scales = {"s": flag_rows(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))]),
              "f": flag_rows(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])}
    log(f"候補 {len(df):,}件（窓 {int(df['window'].min())}〜{int(df['window'].max())}）・前向き {len(scales['f']):,}件"
        f"（テスト窓 {len(split)}つ）")
    tbl = monthly(df, scales)
    report(tbl)
    log(f"全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
