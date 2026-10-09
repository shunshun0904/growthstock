#!/usr/bin/env python3
"""
実験71: 除外の列を「前の窓だけ」で選び直す（先読みの無い確認。実験68 / 実験41 §7-1 の宿題）。

運用者の依頼（2026-10-09）「除外の列を前の窓だけで選び直す形で、先読みの無い確認をする」。確認（AskUserQuestion）で
候補は非線形の32列だけ、を選択。

実験68 は nl_tra1_120 を全窓を見て選んでいた（しきい値は前の窓からだが、列の選び方に先読み）。ここでは窓ごとに

  1. それより前の窓の**選定の中**（3モデル 90以上 / lgbm 単体 95以上）で、候補の各列について
     「列の値が前の窓の母集団の中央値より上の行」と「下の行」の ret_o1_20 の平均の差を取る
  2. |差| が最大の列を選ぶ（両側とも 30件以上ある列だけ）。差が負なら上半分を外す、正なら下半分を外す
  3. その窓の選定に、同じ中央値（前の窓の母集団）で外す規則を当て、残りの平均 − 外さない平均（pt）を測る

全部が「前の窓」だけで決まるので、窓ごとの結果は先読みを含まない。比べる相手は、実験68 の固定の規則
（nl_tra1_120 の上半分を外す）と「外さない」。どの列がいつ選ばれたかも記録する（tra1 が前の窓だけでも
選ばれるなら、実験68 の結果は偶然の列ではない）。25% の版（前の窓の母集団の 75% 点 / 25% 点で切る）も並べる。

スコアは実験66 の腕 T（本番の239列・本番のパラメータ・種3つの平均）、切り方 0 / 2 / 4 か月。

  python3 research/exp/e71_exclusion_pit.py [--shifts 0,2,4] [--seeds 3] [--min-prior 100]
  結果は research/_data/oof/e71_*。本番の設定には書かない。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import nonlinear_features as NL  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e66_nonlinear_screen as E66  # noqa: E402
import e68_tra1_exclusion as E68  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
OUTCOME = lab.OUTCOME
POOL: List[str] = list(NL.NL_COLS)
FIXED_COL = E68.COL
#: 切る位置: (名前, 悪い側が上のときの百分位, 悪い側が下のときの百分位)
CUTS = (("半分", 50.0, 50.0), ("25%", 75.0, 25.0))
MIN_HALF = 30


def prior_threshold(ref: pd.DataFrame, col: str, fold: int, q: float, min_rows: int = 500) -> float:
    """それより前の窓の母集団での百分位 q。参照が足りなければ NaN。"""
    prev = pd.to_numeric(ref.loc[ref["fold"] < fold, col], errors="coerce").dropna()
    if len(prev) < min_rows:
        return float("nan")
    return float(np.percentile(prev, q))


def choose_column(prior_sel: pd.DataFrame, ref: pd.DataFrame, fold: int, pool: List[str],
                  min_half: int = MIN_HALF) -> Optional[Tuple[str, str, float]]:
    """
    前の窓の選定 prior_sel の中で、列ごとに「前の窓の母集団の中央値で分けた上半分 − 下半分」の
    ret_o1_20 の差を取り、|差| 最大の列を返す。(列, 外す側 'top'/'bottom', 差)。候補が無ければ None。
    """
    r = pd.to_numeric(prior_sel[OUTCOME], errors="coerce").to_numpy(dtype=float)
    best = None
    for c in pool:
        thr = prior_threshold(ref, c, fold, 50.0)
        if not np.isfinite(thr):
            continue
        x = pd.to_numeric(prior_sel[c], errors="coerce").to_numpy(dtype=float)
        hi, lo = (x >= thr) & np.isfinite(x), (x < thr) & np.isfinite(x)
        if hi.sum() < min_half or lo.sum() < min_half:
            continue
        d = np.nanmean(r[hi]) - np.nanmean(r[lo])
        if not np.isfinite(d):
            continue
        if best is None or abs(d) > abs(best[2]):
            best = (c, "top" if d < 0 else "bottom", float(d))
    return best


def run_selection(sel: pd.DataFrame, ref: pd.DataFrame, pool: List[str], min_prior: int) -> pd.DataFrame:
    """窓ごとに列を選び直して外す。固定の tra1 規則と「外さない」も同じ窓で並べる。"""
    base = pd.to_numeric(sel[OUTCOME], errors="coerce").to_numpy(dtype=float)
    fo = sel["fold"].to_numpy()
    rows = []
    for k in sorted(np.unique(fo)):
        cur = fo == k
        prior = sel[fo < k]
        if len(prior) < min_prior:
            continue
        rec = {"fold": int(k), "n": int(cur.sum()), "mean_all": float(np.nanmean(base[cur])) * 100}
        pick = choose_column(prior, ref, int(k), pool)
        for name, q_top, q_bot in CUTS:
            # 前の窓だけで選んだ列
            if pick is None:
                rec[f"pit_{name}_col"] = None
                rec[f"pit_{name}_gain"] = np.nan
                rec[f"pit_{name}_excl"] = 0
            else:
                c, side, d = pick
                thr = prior_threshold(ref, c, int(k), q_top if side == "top" else q_bot)
                x = pd.to_numeric(sel[c], errors="coerce").to_numpy(dtype=float)
                m = cur & np.isfinite(x) & ((x >= thr) if side == "top" else (x <= thr))
                keep = cur & ~m
                rec[f"pit_{name}_col"] = f"{c}:{side}"
                rec[f"pit_{name}_gain"] = (float(np.nanmean(base[keep])) - float(np.nanmean(base[cur]))) * 100 if keep.sum() else np.nan
                rec[f"pit_{name}_excl"] = int(m.sum())
                rec["pick_diff_pt"] = d * 100
            # 固定の規則（実験68）: tra1 の上側を外す
            thr = prior_threshold(ref, FIXED_COL, int(k), q_top)
            x = pd.to_numeric(sel[FIXED_COL], errors="coerce").to_numpy(dtype=float)
            m = cur & np.isfinite(x) & (x >= thr)
            keep = cur & ~m
            rec[f"fixed_{name}_gain"] = (float(np.nanmean(base[keep])) - float(np.nanmean(base[cur]))) * 100 if keep.sum() else np.nan
            rec[f"fixed_{name}_excl"] = int(m.sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, label: str) -> Dict[str, float]:
    out = {}
    for kind in ("pit", "fixed"):
        for name, _, _ in CUTS:
            g = res[f"{kind}_{name}_gain"].dropna()
            out[f"{kind}_{name}_mean"] = float(g.mean()) if len(g) else np.nan
            out[f"{kind}_{name}_won"] = int((g > 1e-12).sum())
            out[f"{kind}_{name}_n"] = int(len(g))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験71: 除外の列を前の窓だけで選び直す")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-prior", type=int, default=100, help="列を選ぶのに要る、前の窓の選定の最小件数")
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    os.makedirs(OOF_DIR, exist_ok=True)

    df = E66.load_frame()
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    base_cols = F.columns(F.DEFAULT_PRESET)
    feat = df[["Code", "Date"] + POOL]
    print("=" * 78)
    print(f"実験71 除外の列を前の窓だけで選び直す（候補 {len(POOL)}列・実験66 の腕 T のスコア・種{len(seeds)}つ・"
          f"ずらし {shifts}か月・前の窓の選定 {args.min_prior}件以上で選ぶ）: {len(df):,}件")
    print("=" * 78)
    all_rows, picks = [], []
    for sh in shifts:
        oofs = E68.load_oofs(df, base_cols, sh, seeds)
        ref = oofs["lgbm"][["Code", "Date", "fold"]].merge(feat, on=["Code", "Date"], how="left")
        sels = E68.selections(oofs)
        print(f"\n■ ずらし{sh}か月")
        for name in ("3モデル 90以上", "lgbm 単体 95以上"):
            s = sels[name]
            if not len(s):
                continue
            s = s.merge(feat, on=["Code", "Date"], how="left")
            res = run_selection(s, ref, POOL, args.min_prior)
            if not len(res):
                print(f"  ◆ {name}: 測れる窓なし")
                continue
            res.insert(0, "selection", name)
            res.insert(0, "shift", sh)
            all_rows.append(res)
            sm = summarize(res, name)
            print(f"  ◆ {name}（{int(res['n'].sum()):,}件 / 測れる窓 {len(res)}）")
            print(f"    {'窓':>3}{'件数':>6}{'平均':>8} | {'前の窓で選んだ列（半分）':<34}{'外す':>5}{'差pt':>8} | "
                  f"{'25%':>8} | {'固定 tra1 半分':>8}{'25%':>8}")
            for _, r in res.iterrows():
                print(f"    {int(r['fold']):>3}{int(r['n']):>6}{r['mean_all']:>+7.2f}% | "
                      f"{str(r['pit_半分_col']):<34}{int(r['pit_半分_excl']):>5}{r['pit_半分_gain']:>+8.2f} | "
                      f"{r['pit_25%_gain']:>+8.2f} | {r['fixed_半分_gain']:>+8.2f}{r['fixed_25%_gain']:>+8.2f}")
            print(f"    まとめ: 前の窓で選んだ列 半分 {sm['pit_半分_mean']:+.2f}pt（勝ち {sm['pit_半分_won']}/{sm['pit_半分_n']}）/ "
                  f"25% {sm['pit_25%_mean']:+.2f}pt（{sm['pit_25%_won']}/{sm['pit_25%_n']}）｜ 固定 tra1 半分 "
                  f"{sm['fixed_半分_mean']:+.2f}pt（{sm['fixed_半分_won']}/{sm['fixed_半分_n']}）/ 25% {sm['fixed_25%_mean']:+.2f}pt"
                  f"（{sm['fixed_25%_won']}/{sm['fixed_25%_n']}）")
            picks += [{"shift": sh, "selection": name, "fold": int(r["fold"]), "col": r["pit_半分_col"]}
                      for _, r in res.iterrows()]
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out.to_csv(os.path.join(OOF_DIR, "e71_exclusion_pit.csv"), index=False)
    if len(out):
        print("\n■ 切り方3通り × 選定2つをまとめて（窓ごとの差 pt の平均 / 勝ち窓）")
        print(f"  {'規則':<34}{'窓':>5}{'平均':>8}{'勝ち窓':>9}{'外す件数':>9}")
        for kind, ja in (("pit", "前の窓で選んだ列"), ("fixed", "固定 nl_tra1_120 上側")):
            for name, _, _ in CUTS:
                g = out[f"{kind}_{name}_gain"].dropna()
                print(f"  {ja + ' ' + name:<34}{len(g):>5}{g.mean():>+8.2f}{int((g > 1e-12).sum()):>5}/{len(g):<3}"
                      f"{int(out[f'{kind}_{name}_excl'].sum()):>9}")
        pk = pd.Series([p["col"] for p in picks if p["col"]])
        print("\n■ 前の窓で選ばれた列（列:外す側 → 回数、全切り方・全選定）")
        for c, n in pk.value_counts().items():
            print(f"  {c:<32}{n:>4}")
        n_tra1 = int(pk.str.startswith(FIXED_COL + ":top").sum())
        print(f"  → {FIXED_COL} の上側が選ばれた窓: {n_tra1}/{len(pk)}")
    print(f"記録: {OOF_DIR}/e71_exclusion_pit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
