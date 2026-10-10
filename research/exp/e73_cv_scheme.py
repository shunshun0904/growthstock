#!/usr/bin/env python3
"""
実験73: 探索の分割を層別（year_cap_date）から前進分割（walkforward）に変えると、lgbm の精度はどう変わるか。

運用者（2026-10-10）: 探索の 5分割 CV を「本番の窓と同じ前進分割」に変えた（docs/MODEL_ADOPTION_RULES.md §32）うえで
「まずは lgbm だけで精度はどう変わるか見たい」。

腕（列は本番の239列で同じ。データも同じ。**パラメータだけ**が違う）
  S  層別（year_cap_date。2026-10-10 までの本番）で 50試行 × 5分割
  W  前進分割（walkforward。2026-10-10 からの本番）で 50試行 × 5分割（検証窓は打ち切り日から遡って 6か月 × 5本）
探索はどちらもホールドアウトより前（同じ打ち切り日）。Optuna の種も同じ。

物差し（実験67・68・72 と同じ。本番と同じ OOF（36/6/6か月・エンバーゴ20営業日・種42）と、境界を 0/2/4か月ずらした
32窓 × 種3つの平均）: PR-AUC・正例率に対するリフト・ROC-AUC・日内 AUC・上位10% の超過リターン。
窓ごとの差 W−S は、全窓と、**探索の打ち切り日より後に始まる窓（探索に一切使っていない期間）** を分けて出す。
前の窓は、どちらの腕もその期間のデータでパラメータを選んでいる（同じ条件だが、絶対値は楽観側）。

ついでに、W の探索の CV（前進分割の5窓）と、同じ期間の OOF の窓の PR-AUC を並べる。分割を変えた狙いの1つは
「探索の CV の値が本番の OOF と同じ物差しになる」ことなので、それが実際にそうかを見る。

期待（結果を見る前に）
  実験で何度も見たとおり、同じ列・同じデータなら探索し直しの揺れは PR-AUC で ±0.005 程度。分割を変えても
  選ばれるパラメータの違いはその範囲に収まり、32窓の W−S は SE の 2倍以内に入る見込み。前進分割は直近 2.5年で
  選ぶので、打ち切り日より後の窓でだけ W が少し上なら「直近に合わせた」効果、そこでも差が無ければ「どちらでも同じ」。

    exp=e73_cv_scheme.py
試運転: exp=e73_cv_scheme.py args="--tag e73smoke --n-trials 2 --shifts 0"

公開ログには件数・割合・日付・精度だけを出す。本番の設定（research/lgbm_params.json、features.py）には書かない。
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
import features as F  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e68_edinet_ab as E68  # noqa: E402
from e72_jsf_edinet_only import safe_metrics, windows_on  # noqa: E402

ALGO = "lgbm"
ARMS = ("S", "W")
SCHEME_OF = {"S": "year_cap_date", "W": "walkforward"}
LABELS = {"S": "S 層別（year_cap_date。2026-10-10 までの本番）で探索",
          "W": "W 前進分割（walkforward。2026-10-10 からの本番）で探索"}
TAG = "e73"
log = E68.log


def tune_arm(arm: str, df: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict, compute: bool):
    """E68.tune は lgbm の分割を E68.CV_SCHEME から読むので、腕ごとに切り替えて呼ぶ（探索の記録は腕ごとに別ファイル）。"""
    saved = E68.CV_SCHEME
    E68.CV_SCHEME = SCHEME_OF[arm]
    try:
        return E68.tune(ALGO, arm, df, cols, cutoff, n_trials, stamp, compute)
    finally:
        E68.CV_SCHEME = saved


def window_starts(dates: pd.Series, shifts: list) -> dict:
    """(ずらし, 窓の番号) → 窓の初日。打ち切り日より後に始まる窓を分けるのに使う。"""
    out = {}
    for sh in shifts:
        for f in E41.folds_for(dates, sh):
            out[(sh, int(f.index))] = pd.Timestamp(f.test_start)
    return out


def split_windows(s: pd.DataFrame, starts: dict, cutoff) -> pd.DataFrame:
    """窓ごとの表に holdout（窓の初日が打ち切り日より後）の列を付ける。"""
    s = s.copy()
    s["start"] = [starts.get((int(sh), int(f)), pd.NaT) for sh, f in zip(s["shift"], s["fold"])]
    s["holdout"] = s["start"] > pd.Timestamp(cutoff)
    return s


def run(df: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict, shifts: list, compute: bool):
    recs, params = {}, {}
    for arm in ARMS:
        recs[arm] = tune_arm(arm, df, cols, cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
        params[arm] = recs[arm]["params"]
    prod, wins = {}, {}
    for arm in ARMS:
        o = E68.oof(ALGO, "base", df, cols, params[arm], 0, (E68.PROD_SEED,), compute)
        if o is None:
            return None
        prod[arm] = o
    for sh in shifts:
        for arm in ARMS:
            o = E68.oof(ALGO, "base", df, cols, params[arm], sh, E68.SEEDS[ALGO], compute)
            if o is None:
                return None
            wins[(sh, arm)] = windows_on(o, None)
    return {"recs": recs, "params": params, "prod": prod, "wins": wins}


def report(res: dict, shifts: list, cutoff, starts: dict) -> dict:
    summary = {"cutoff": str(pd.Timestamp(cutoff).date()), "tuning": {}, "production_oof": {}, "windows": {}}
    print("\n" + "=" * 78)
    print("■ 1. 探索の CV（S は層別で楽観側、W は前進分割。互いに比べる値ではない。パラメータ選び用）")
    print(f"  {'腕':<4}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC")
    for arm in ARMS:
        rec = res["recs"][arm]
        cv = rec["_cv"]
        fs = cv.get("fold_scores") or []
        print(f"  {arm:<4}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}{rec.get('_seconds', 0):>6}  "
              + " / ".join(f"{v:.4f}" for v in fs))
        summary["tuning"][arm] = {k: cv.get(k) for k in ("scheme", "mean_pr_auc", "std", "mean_roc_auc",
                                                          "fold_scores", "fold_pos_rate", "fold_windows")}
        if cv.get("fold_windows"):
            print("      検証窓: " + " / ".join(f"{w['valid_from']}〜{w['valid_to']}（正例率 {w['pos_rate']*100:.1f}%）"
                                              for w in cv["fold_windows"]))
    pa, pb = res["params"]["S"], res["params"]["W"]
    print("  選ばれたパラメータ: " + ("S と W で同じ" if E68.params_hash(pa) == E68.params_hash(pb)
                                  else f"S {E68.short(pa)} / W {E68.short(pb)}"))

    print("\n■ 2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。全行）")
    print(f"  {'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'件数':>8}{'正例率':>8}")
    ms = {}
    for arm in ARMS:
        o = res["prod"][arm]
        m = safe_metrics(o)
        ms[arm] = m
        summary["production_oof"][arm] = {**{k: float(v) for k, v in m.items()}, "n": int(len(o))}
        print(f"  {arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
              f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{len(o):>8,}"
              f"{m['rate']*100:>7.1f}%")
    print(f"  W−S {ms['W']['pr_auc'] - ms['S']['pr_auc']:>+8.4f}{'':>7}{ms['W']['roc_auc'] - ms['S']['roc_auc']:>+8.4f}"
          f"{ms['W']['day_auc'] - ms['S']['day_auc']:>+8.4f}{ms['W']['ret_o1_20_mean'] - ms['S']['ret_o1_20_mean']:>+10.2f}pt")

    rows_ = []
    for sh in shifts:
        w = None
        for arm in ARMS:
            x = res["wins"][(sh, arm)].rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
            if arm != ARMS[0]:
                x = x.drop(columns=["rate", "n"])
            w = x if w is None else w.merge(x, on="fold")
        if w is None or len(w) == 0:
            continue
        for _, r in w.iterrows():
            rows_.append({"shift": sh, "fold": int(r["fold"]), "rate": float(r["rate"]), "n": int(r["n"]),
                          **{f"{m}_{arm}": float(r[f"{m}_{arm}"]) for m in ("pr", "roc") for arm in ARMS}})
    s = split_windows(pd.DataFrame(rows_), starts, cutoff)
    s.to_csv(E68.path("auc_by_window.csv"), index=False)
    print(f"\n■ 3. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り × 種3つの平均。"
          f"探索の打ち切り日 {pd.Timestamp(cutoff).date()}）")
    print(f"  {'':<28}{'窓':>4}{'正例率':>8}{'PR S':>9}{'PR W':>9}{'W−S':>9}{'SE':>8}{'W が上':>8}{'ROC S':>9}{'ROC W':>9}{'W−S':>9}")
    for name, g in (("全窓", s), ("打ち切り日より後に始まる窓", s[s["holdout"]]), ("打ち切り日より前の窓", s[~s["holdout"]])):
        if not len(g):
            print(f"  {name:<28}{0:>4}")
            continue
        d_pr = g["pr_W"] - g["pr_S"]
        d_roc = g["roc_W"] - g["roc_S"]
        se = float(d_pr.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
        se_roc = float(d_roc.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
        print(f"  {name:<28}{len(g):>4}{g['rate'].mean()*100:>7.1f}%{g['pr_S'].mean():>9.4f}{g['pr_W'].mean():>9.4f}"
              f"{d_pr.mean():>+9.4f}{se:>8.4f}{int((d_pr > 0).sum()):>5}/{len(g):<3}{g['roc_S'].mean():>8.4f}"
              f"{g['roc_W'].mean():>9.4f}{d_roc.mean():>+9.4f}")
        summary["windows"][name] = {"n": int(len(g)), "rate": float(g["rate"].mean()),
                                    "pr_S": float(g["pr_S"].mean()), "pr_W": float(g["pr_W"].mean()),
                                    "d_pr": float(d_pr.mean()), "se_pr": se, "wins": int((d_pr > 0).sum()),
                                    "roc_S": float(g["roc_S"].mean()), "roc_W": float(g["roc_W"].mean()),
                                    "d_roc": float(d_roc.mean()), "se_roc": se_roc}
    hold = s[s["holdout"]].sort_values(["shift", "fold"])
    if len(hold):
        print("  打ち切り日より後の窓の内訳: " + " / ".join(
            f"ずらし{int(r['shift'])} 窓{int(r['fold'])}（{r['start'].date()}〜）S {r['pr_S']:.4f} W {r['pr_W']:.4f}"
            for _, r in hold.iterrows()))

    # W の探索の CV（前進分割の5窓）と、同じ期間の OOF の窓（ずらし0）
    cvw = res["recs"]["W"]["_cv"]
    if cvw.get("fold_windows"):
        print("\n■ 4. W の探索の CV の窓 と、同じ期間の OOF の窓（ずらし0）の PR-AUC（分割を変えた狙い: 探索の値が OOF と同じ物差しになる）")
        o = res["prod"]["W"]
        od = pd.to_datetime(o["Date"])
        pairs = []
        for w, v in zip(cvw["fold_windows"], cvw.get("fold_scores") or []):
            sub = o[(od >= pd.Timestamp(w["valid_from"])) & (od <= pd.Timestamp(w["valid_to"]))]
            if len(sub) >= 100 and sub["label"].nunique() == 2:
                from sklearn.metrics import average_precision_score
                pr = float(average_precision_score(sub["label"], sub["score"]))
                pairs.append((w["valid_from"], w["valid_to"], float(v), pr, float(sub["label"].mean())))
        for a, b, v, pr, rate in pairs:
            print(f"  {a}〜{b}: 探索の CV {v:.4f} / OOF {pr:.4f} / 正例率 {rate*100:.1f}%")
        summary["cv_vs_oof"] = [{"from": a, "to": b, "cv": v, "oof": pr, "rate": rate} for a, b, v, pr, rate in pairs]
    with open(E68.path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験73: 探索の分割（層別 / 前進分割）で lgbm の精度はどう変わるか")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--budget-min", type=float, default=290.0)
    args = ap.parse_args(argv)
    E68.TAG = args.tag
    E68.RUN = f"{args.tag}_lgbm"
    E68.DEADLINE = time.time() + args.budget_min * 60 if args.budget_min > 0 else None
    E68.TIMED_OUT = False
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    cols = F.columns(F.DEFAULT_PRESET)

    print("=" * 78)
    print("実験73 探索の分割（層別 / 前進分割）で lgbm の精度はどう変わるか（50試行 × 5分割 + 本番と同じ OOF + 32窓）")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  列: {F.DEFAULT_PRESET} {len(cols)}列 / 探索 {args.n_trials}試行 × {E68.N_SPLITS}分割 / 窓のずらし {shifts}か月 / "
          f"種 {E68.SEEDS[ALGO]} / 上限 {args.budget_min:g}分 / 保存 {E68.RUN}_*")
    print("=" * 78)
    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    stamp = {"built_utc": f"dataset-to-{df['Date'].max().date()}"}
    df.attrs["built_utc"] = stamp["built_utc"]
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"\n  行 {len(df):,}（{df['Date'].min().date()} 〜 {df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(df['Date'] <= cutoff).sum():,}行")
    starts = window_starts(df["Date"], shifts)
    t0 = time.time()
    res = run(df, cols, cutoff, args.n_trials, stamp, shifts, compute=True)
    log(f"[{ALGO}] {'そろった' if res else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    if res:
        report(res, shifts, cutoff, starts)
    if E68.TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {E68.OOF_DIR}/{E68.RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
