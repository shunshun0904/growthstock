#!/usr/bin/env python3
"""
実験73: 探索の分割（層別 / 前進分割）と目的関数（PR-AUC / リフト）で、lgbm の精度はどう変わるか。

運用者（2026-10-10）: 探索の 5分割 CV を「時系列の層別 k 分割」にしたい → 形は「本番の窓と同じ前進分割」
（docs/MODEL_ADOPTION_RULES.md §32）。「まずは lgbm だけで精度はどう変わるか見たい」「いきなり変えるよりも、
まずは検証してほしい。本番導入するかはその結果次第」。本番は層別（year_cap_date）・PR-AUC のまま、これはその検証。
1回目（S と W）の結果を見て、前進分割では窓ごとの正例率（9.8〜23.9%）が PR-AUC の平均を引っ張るので「目的関数を
正例率で割ったリフトにする」案を出し、運用者の了承（「あくまで 5cv のですよね？であれば試す価値はある」）で WL を足した。

腕（列は本番の239列で同じ。データも同じ。**パラメータだけ**が違う）
  S   層別（year_cap_date。今の本番）・目的関数 PR-AUC で 50試行 × 5分割
  W   前進分割（walkforward。候補）・目的関数 PR-AUC で 50試行 × 5分割（検証窓は打ち切り日から遡って 6か月 × 5本）
  WL  前進分割・目的関数 リフト（PR-AUC ÷ 検証窓の正例率）で 50試行 × 5分割。窓の重みをそろえる
探索はどれもホールドアウトより前（同じ打ち切り日）。Optuna の種も同じ。目的関数は探索の中だけの話で、
本番の評価（OOF・窓）は PR-AUC のまま。

物差し（実験67・68・72 と同じ。本番と同じ OOF（36/6/6か月・エンバーゴ20営業日・種42）と、境界を 0/2/4か月ずらした
32窓 × 種3つの平均）: PR-AUC・正例率に対するリフト・ROC-AUC・日内 AUC・上位10% の超過リターン。
窓ごとの差は、全窓と、**探索の打ち切り日より後に始まる窓（探索に一切使っていない期間）** を分けて出す。
前の窓は、どの腕もその期間のデータでパラメータを選んでいる（同じ条件だが、絶対値は楽観側。前進分割の腕は
その期間の窓で選んでいるぶん、前の窓では有利に見える）。

ついでに、前進分割の腕の探索の CV（5窓）と、同じ期間の OOF の窓の PR-AUC を並べる。分割を変えた狙いの1つは
「探索の CV の値が本番の OOF と同じ物差しになる」ことなので、それが実際にそうかを見る。

期待（結果を見る前に）
  同じ列・同じデータなら探索し直しの揺れは PR-AUC で ±0.005 程度。分割・目的関数を変えても、探索に使っていない
  窓での差はその範囲に収まる見込み。WL は正例率の低い窓（2024-08〜2025-02）を W より重く見るので、その窓に近い
  地合いの窓で W より上なら「そろえた」効果。

    exp=e73_cv_scheme.py
試運転: exp=e73_cv_scheme.py args="--tag e73smoke --n-trials 2 --shifts 0"

公開ログには件数・割合・日付・精度だけを出す。本番の設定（research/lgbm_params.json、features.py）には書かない。

結果（1回目 S / W。2026-10-10、run 38073499515、33分。docs/MODEL_ADOPTION_RULES.md §32）
--------
- 探索の CV: S 0.3442 ± 0.027（層別・楽観側）、W 0.2995 ± 0.114（前進分割。窓ごと 0.25 / 0.32 / 0.22 / 0.20 / 0.51 で、
  正例率 9.8〜23.9% の違いがそのまま出る）。W の CV は同じ期間の OOF の窓と 0.01 以内で一致した（狙いどおり）
- 選ばれたパラメータは大きく違う: S は葉 85・min_child 85・列 57%、W は葉 12・min_child 17・列 86%（浅い木）
- 本番と同じ OOF（全行 16,320件）: S 0.2767（1.55x）/ W 0.2807（1.57x）。ROC +0.002、日内 +0.005、上位10% +0.11pt
- 32窓: W−S +0.0052 ± 0.0016（W が上 22/32）。ただし**探索の打ち切り日より後に始まる 6窓では +0.0016 ± 0.0035（2/6）**で
  差が無い。前の 26窓の +0.0060（20/26）は、W がその期間（2023-02〜2025-08）の窓で選んだパラメータなので、選んだぶんの
  楽観が入る
- 結論: 探索に使っていない期間では精度は変わらない（差 ±0.004 の範囲）。前進分割の利点は「探索の値が本番の OOF と同じ
  物差しになる」ことで、精度の根拠にはならない。本番に入れるかは運用者の判断（§32）
結果（2回目 WL。2026-10-10、run 38077508167、21分。S / W は1回目の保存済みを再利用）
--------
- WL の探索: 目的関数（リフト）1.748、PR-AUC の平均 0.2979 ± 0.107。窓ごと 0.249 / 0.317 / 0.209 / 0.217 / 0.499 で、
  正例率 9.8% の窓が W の 0.198 → 0.217 に上がり、23.9% の窓が 0.512 → 0.499 に下がった（狙いどおり窓の重みがそろう）。
  選ばれたパラメータは S と W の中間（葉 35・min_child 130・列 78%・reg_lambda 2.6）
- 本番と同じ OOF（ずらし0・種42）: S 0.2767 / W 0.2807 / WL 0.2852（ROC 0.637 / 0.639 / 0.646）。この1本では WL が上
- 32窓: WL−S +0.0016 ± 0.0015（16/32）、WL−W −0.0036 ± 0.0012（10/32）。**探索の打ち切り日より後の 6窓では WL−S −0.0035 ±
  0.0012（1/6）、WL−W −0.0051 ± 0.0030（2/6）** で、WL がいちばん下。ROC も同じ向き（6窓 S 0.6572 / W 0.6591 / WL 0.6572）
- 結論: リフトにしても、探索に使っていない期間の精度は上がらない（むしろ 0.004 ほど下）。W・WL とも S を上回らず、
  分割・目的関数を変える精度上の根拠は無い（§32）
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
ARMS = ("S", "W", "WL")
#: 腕 → (分割, 目的関数)
ARM_SPEC = {"S": ("year_cap_date", "pr_auc"), "W": ("walkforward", "pr_auc"), "WL": ("walkforward", "lift")}
LABELS = {"S": "S 層別（year_cap_date。今の本番）・PR-AUC で探索",
          "W": "W 前進分割（walkforward。候補）・PR-AUC で探索",
          "WL": "WL 前進分割・リフト（PR-AUC ÷ 正例率）で探索"}
#: 表に出す差（「後 − 前」）
PAIRS = (("S", "W"), ("S", "WL"), ("W", "WL"))
TAG = "e73"
log = E68.log


def tune_arm(arm: str, df: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict, compute: bool):
    """E68.tune は lgbm の分割・目的関数を E68.CV_SCHEME / CV_OBJECTIVE から読むので、腕ごとに切り替えて呼ぶ
    （探索の記録は腕ごとに別ファイル）。呼んだあと元に戻す。"""
    saved = (E68.CV_SCHEME, E68.CV_OBJECTIVE)
    E68.CV_SCHEME, E68.CV_OBJECTIVE = ARM_SPEC[arm]
    try:
        return E68.tune(ALGO, arm, df, cols, cutoff, n_trials, stamp, compute)
    finally:
        E68.CV_SCHEME, E68.CV_OBJECTIVE = saved


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


def _pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = b - a
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"n": int(len(d)), "d": float(d.mean()), "se": se, "wins": int((d > 0).sum())}


def report(res: dict, shifts: list, cutoff, starts: dict) -> dict:
    summary = {"cutoff": str(pd.Timestamp(cutoff).date()), "tuning": {}, "production_oof": {}, "windows": {}}
    print("\n" + "=" * 78)
    print("■ 1. 探索の CV（S は層別で楽観側、W / WL は前進分割。腕どうしで比べる値ではない。パラメータ選び用）")
    print(f"  {'腕':<4}{'目的関数':>10}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC")
    for arm in ARMS:
        rec = res["recs"][arm]
        cv = rec["_cv"]
        fs = cv.get("fold_scores") or []
        obj = cv.get("objective", "pr_auc")
        print(f"  {arm:<4}{obj:>10}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}{rec.get('_seconds', 0):>6}  "
              + " / ".join(f"{v:.4f}" for v in fs)
              + (f"  （目的関数の値 {cv['objective_value']:.4f}）" if "objective_value" in cv else ""))
        summary["tuning"][arm] = {k: cv.get(k) for k in ("scheme", "objective", "objective_value", "mean_pr_auc", "std",
                                                          "mean_roc_auc", "fold_scores", "fold_pos_rate", "fold_windows")}
        if cv.get("fold_windows") and ARM_SPEC[arm][0] == "walkforward":
            print("      検証窓: " + " / ".join(f"{w['valid_from']}〜{w['valid_to']}（正例率 {w['pos_rate']*100:.1f}%）"
                                              for w in cv["fold_windows"]))
    print("  選ばれたパラメータ: " + " / ".join(f"{arm} {E68.short(res['params'][arm])}" for arm in ARMS))

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
    for x, y in PAIRS:
        print(f"  {y}−{x:<4}{ms[y]['pr_auc'] - ms[x]['pr_auc']:>+7.4f}{'':>7}{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}"
              f"{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

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
    print(f"  {'':<28}{'窓':>4}{'正例率':>8}" + "".join(f"{'PR ' + arm:>9}" for arm in ARMS)
          + "".join(f"{'ROC ' + arm:>9}" for arm in ARMS))
    groups = (("全窓", s), ("打ち切り日より後に始まる窓", s[s["holdout"]]), ("打ち切り日より前の窓", s[~s["holdout"]]))
    for name, g in groups:
        if not len(g):
            print(f"  {name:<28}{0:>4}")
            continue
        print(f"  {name:<28}{len(g):>4}{g['rate'].mean()*100:>7.1f}%"
              + "".join(f"{g[f'pr_{arm}'].mean():>9.4f}" for arm in ARMS)
              + "".join(f"{g[f'roc_{arm}'].mean():>9.4f}" for arm in ARMS))
        summary["windows"][name] = {"n": int(len(g)), "rate": float(g["rate"].mean()),
                                    **{f"pr_{arm}": float(g[f"pr_{arm}"].mean()) for arm in ARMS},
                                    **{f"roc_{arm}": float(g[f"roc_{arm}"].mean()) for arm in ARMS}}
    print(f"  {'':<28}{'':>4}{'':>8}" + "".join(f"{y + '−' + x:>18}" for x, y in PAIRS) + "   （PR の差 ± SE、上の窓）")
    for name, g in groups:
        if not len(g):
            continue
        cells = []
        for x, y in PAIRS:
            st = _pair_stats(g[f"pr_{x}"].to_numpy(), g[f"pr_{y}"].to_numpy())
            summary["windows"][name][f"pr_{y}-{x}"] = st
            summary["windows"][name][f"roc_{y}-{x}"] = _pair_stats(g[f"roc_{x}"].to_numpy(), g[f"roc_{y}"].to_numpy())
            cells.append(f"{st['d']:+.4f}±{st['se']:.4f} {st['wins']}/{st['n']}")
        print(f"  {name:<28}{'':>4}{'':>8}" + "".join(f"{c:>18}" for c in cells))
    hold = s[s["holdout"]].sort_values(["shift", "fold"])
    if len(hold):
        print("  打ち切り日より後の窓の内訳: " + " / ".join(
            f"ずらし{int(r['shift'])} 窓{int(r['fold'])}（{r['start'].date()}〜）" + " ".join(f"{arm} {r[f'pr_{arm}']:.4f}" for arm in ARMS)
            for _, r in hold.iterrows()))

    print("\n■ 4. 前進分割の腕の探索の CV の窓 と、同じ期間の OOF の窓（ずらし0）の PR-AUC（探索の値が OOF と同じ物差しになるか）")
    from sklearn.metrics import average_precision_score
    summary["cv_vs_oof"] = {}
    for arm in ARMS:
        cv = res["recs"][arm]["_cv"]
        if ARM_SPEC[arm][0] != "walkforward" or not cv.get("fold_windows"):
            continue
        o = res["prod"][arm]
        od = pd.to_datetime(o["Date"])
        pairs = []
        for w, v in zip(cv["fold_windows"], cv.get("fold_scores") or []):
            sub = o[(od >= pd.Timestamp(w["valid_from"])) & (od <= pd.Timestamp(w["valid_to"]))]
            if len(sub) >= 100 and sub["label"].nunique() == 2:
                pr = float(average_precision_score(sub["label"], sub["score"]))
                pairs.append((w["valid_from"], w["valid_to"], float(v), pr, float(sub["label"].mean())))
        for a, b, v, pr, rate in pairs:
            print(f"  {arm:<3} {a}〜{b}: 探索の CV {v:.4f} / OOF {pr:.4f} / 正例率 {rate*100:.1f}%")
        summary["cv_vs_oof"][arm] = [{"from": a, "to": b, "cv": v, "oof": pr, "rate": rate} for a, b, v, pr, rate in pairs]
    with open(E68.path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験73: 探索の分割（層別 / 前進分割）と目的関数で lgbm の精度はどう変わるか")
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
    print("実験73 探索の分割（層別 / 前進分割）と目的関数（PR-AUC / リフト）で lgbm の精度はどう変わるか"
          "（50試行 × 5分割 + 本番と同じ OOF + 32窓）")
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
