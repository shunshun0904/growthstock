#!/usr/bin/env python3
"""
実験43: 時点整合の修正（信用残・投資部門別を公表日で結合）で OOF がどう変わるか。

2026-09-24 まで、週次の信用残を基準日（金曜）で、投資部門別を集計期間の末日で
結合していて、学習だけが公表前の値を見ていた（信用残で母集団の 38.5%、
投資部門別で 85.7% の行）。予測では同じ新しさの値は手に入らない。
修正（research/availability.py）でこれまでの OOF の見込みがどれだけ動くかを測る。

腕（それ以外はすべて同じ）
  A  修正前。availability.LEGACY = True（信用残は基準日、投資部門別は末日で結合）
  B  修正後（本番）。どちらも公表日で結合
  共通: 本番の205列（all_plus）、本番のパラメータ（lgbm_params.json /
        multi_params.json を読むだけ。書かない）、ブースティング3モデル、種3つの平均

窓: 本番の out-of-fold と同じ作り（36/6/6か月）を、境界を 0/2/4か月ずらした3通り
    （実験41 §10 と同じ folds_for / oof_folds）。**1本の OOF で決めない**

見るもの（窓ごと・切り方ごと）
  1. A と B で値が変わった行の割合（credit_ratio・inv_*。ほかの列は1つも変わらないことを確かめる）
  2. 分離力: モデルごとの PR-AUC / ROC-AUC。窓ごとの B−A と、その窓平均・SE・B が勝った窓の数
  3. 運用の規則（research/live_track.py と同じ。3モデル90以上・発火8件以上・上位2件・
     枠3・+20%か20営業日）での取引: 件数・1取引の平均・勝率・月あたり（概算）。窓ごとも

  --shifts 0,2,4  --seeds 3  --algos lgbm,xgb,cat
  結果は research/_data/oof/e43_*。本番の設定には書かない。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import availability as AV  # noqa: E402
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e25_auc_noise import average  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
LEGACY_PATH = os.path.join(lab.DATA_DIR, "dataset_legacy.parquet")
#: 修正で値が変わってよい列。これ以外が A と B で違えば、比較が結合の違いだけでなくなる
CHANGED = ("credit_ratio", "credit_ratio_r")
CHANGED_PREFIX = ("inv_",)
ARMS = ("A", "B")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def allowed_to_change(col: str) -> bool:
    return col in CHANGED or col.startswith(CHANGED_PREFIX)


def build_legacy(data_dir: str) -> pd.DataFrame:
    """修正前の結合でデータセットを作る（保存済みなら読む）。"""
    if not os.path.exists(LEGACY_PATH):
        log("腕 A のデータセットを作る（availability.LEGACY = True）")
        AV.LEGACY = True
        try:
            B.build(data_dir, LEGACY_PATH)
        finally:
            AV.LEGACY = False
    return pd.read_parquet(LEGACY_PATH)


def frames(data_dir: str):
    """(腕 A の frame, 腕 B の frame, 値が変わった行の割合)。"""
    fb = lab.frame()
    fb["Date"] = pd.to_datetime(fb["Date"])
    fb["Code"] = fb["Code"].astype(str)
    la = build_legacy(data_dir)
    la["Date"] = pd.to_datetime(la["Date"])
    la["Code"] = la["Code"].astype(str)
    if len(la) != len(fb.dropna(subset=["label"])) and len(la) != len(fb):
        raise SystemExit(f"腕 A と B で行数が違う（A {len(la):,} / B {len(fb):,}）")
    key = ["Code", "Date"]
    extra = [c for c in fb.columns if c not in la.columns]       # 実収益など lab が足した列
    fa = la.merge(fb[key + extra], on=key, how="inner", validate="one_to_one")
    if len(fa) != len(la):
        raise SystemExit("腕 A の行が B と突き合わない")
    fb = fb.set_index(key).loc[fa.set_index(key).index].reset_index()
    changed = {}
    for c in la.columns:
        if c in key or c not in fb.columns:
            continue
        a, b = fa[c], fb[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            diff = ~(np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                                equal_nan=True))
        else:
            diff = ~((a == b) | (a.isna() & b.isna())).to_numpy()
        share = float(diff.mean())
        if share > 0:
            changed[c] = share
    bad = {c: v for c, v in changed.items() if not allowed_to_change(c)}
    if bad:
        raise SystemExit(f"結合の修正と関係の無い列が変わっている: {bad}")
    return fa, fb, changed


def fingerprint(df: pd.DataFrame, cols: list) -> str:
    """
    学習に使う中身の指紋。保存名に入れて、中身が変わったのに前の out-of-fold を
    読むことを防ぐ（research/_data/oof は Actions のキャッシュで次の実行に引き継がれる）。
    """
    h = pd.util.hash_pandas_object(df[["Code", "Date", "label"] + list(cols)], index=False)
    # 行の並びも入れる（並びが違えば結果も違う。ab_oof.fingerprint と同じ）
    return hashlib.sha1(h.to_numpy(dtype=np.uint64).tobytes()).hexdigest()[:8]


def oof_arm(df: pd.DataFrame, cols: list, arm: str, shift: int, algos, seeds) -> dict:
    """腕・切り方ごとの out-of-fold（種の平均）。保存済みなら読む（中身の指紋が同じときだけ）。"""
    folds = E41.folds_for(df["Date"], shift)
    fp = fingerprint(df, cols)
    out = {}
    for a in algos:
        par = E27.prod_params(a)
        parts = []
        for sd in seeds:
            path = os.path.join(OOF_DIR, f"e43_{arm}_{fp}_{a}_sh{shift}_s{sd}.parquet")
            if os.path.exists(path):
                parts.append(pd.read_parquet(path))
                continue
            t0 = time.time()
            o = E41.oof_folds(a, df, cols, par, sd, folds)
            o.to_parquet(path, index=False)
            parts.append(o)
            log(f"  腕{arm} {a} ずらし{shift}か月 種{sd}: {len(o):,}件 {time.time()-t0:.0f}秒")
        out[a] = average(parts)
        out[a]["Date"] = pd.to_datetime(out[a]["Date"])
        out[a]["Code"] = out[a]["Code"].astype(str)
    return out


def auc_by_window(o: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for f, g in o.groupby("fold"):
        y = g["label"].to_numpy(dtype=int)
        if y.min() == y.max():
            continue
        rows.append({"fold": int(f), "pr": average_precision_score(y, g["score"]),
                     "roc": roc_auc_score(y, g["score"])})
    return pd.DataFrame(rows)


def rule_trades(oofs: dict, bars: pd.DataFrame, bar_days) -> pd.DataFrame:
    """運用の規則で取った取引（live_track と同じ関数）。fold は選定日の窓。"""
    base = None
    for a in L.BOOST:
        o = oofs[a][["Code", "Date", "fold", "label", "score"]].rename(columns={"score": f"s_{a}"})
        o[f"p_{a}"] = L.pct_of(o[f"s_{a}"].to_numpy(), o[f"s_{a}"].to_numpy())
        base = o if base is None else base.merge(o.drop(columns=["label", "fold"]),
                                                 on=["Code", "Date"], how="inner")
    base["score"] = base["s_lgbm"]
    rows, _, picks = L.decide(base)
    pf = L.forward(bars, picks)
    sim = L.simulate(pf, bar_days)
    t = sim[sim["taken"] == L.TAKEN].copy()
    return t


def pair_line(name: str, a: np.ndarray, b: np.ndarray, fmt: str = "{:.4f}") -> str:
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return (f"  {name:<18}{fmt.format(a.mean()):>9}{fmt.format(b.mean()):>9}"
            f"{d.mean():>+10.4f}{se:>9.4f}{(d > 0).sum():>5}/{len(d):<3}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験43: 時点整合の修正の前後")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default=",".join(L.BOOST))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    algos = [a for a in args.algos.split(",") if a]
    os.makedirs(OOF_DIR, exist_ok=True)

    fa, fb, changed = frames(lab.DATA_DIR)
    cols = F.columns(F.DEFAULT_PRESET)
    miss = [c for c in cols if c not in fb.columns]
    if miss:
        raise SystemExit(f"{F.DEFAULT_PRESET} の列がデータセットに無い: {miss[:5]}")
    print("=" * 78)
    print(f"実験43 時点整合の修正の前後（{F.DEFAULT_PRESET} {len(cols)}列・本番のパラメータ・"
          f"種{len(seeds)}つ・ずらし {shifts}か月）")
    print("=" * 78)
    print("\n■ 1. 修正で値が変わった行の割合（ほかの列は1つも変わっていない）")
    for c, v in sorted(changed.items(), key=lambda x: -x[1]):
        print(f"  {c:<20}{v*100:>6.1f}%")

    bars = L.load_bars(lab.DATA_DIR, start=pd.Timestamp("2021-01-01"))
    bar_days = sorted(pd.Timestamp(d) for d in bars["Date"].unique())
    summary = []
    for sh in shifts:
        res = {arm: oof_arm(df, cols, arm, sh, algos, seeds)
               for arm, df in (("A", fa), ("B", fb))}
        print(f"\n■ 2. 分離力（窓ごと。ずらし{sh}か月）")
        print(f"  {'':<18}{'A 修正前':>9}{'B 修正後':>9}{'B−A':>10}{'SE':>9}{'Bが上':>9}")
        for a in algos:
            wa, wb = auc_by_window(res["A"][a]), auc_by_window(res["B"][a])
            m = wa.merge(wb, on="fold", suffixes=("_a", "_b"))
            print(pair_line(f"{a} PR-AUC", m["pr_a"].to_numpy(), m["pr_b"].to_numpy()))
            print(pair_line(f"{a} ROC-AUC", m["roc_a"].to_numpy(), m["roc_b"].to_numpy()))
            for _, r in m.iterrows():
                summary.append({"shift": sh, "algo": a, "fold": int(r["fold"]),
                                "pr_a": r["pr_a"], "pr_b": r["pr_b"],
                                "roc_a": r["roc_a"], "roc_b": r["roc_b"]})
        if set(L.BOOST) <= set(algos):
            print(f"\n■ 3. 運用の規則での取引（ずらし{sh}か月）")
            print(f"  {'腕':<10}{'取引':>6}{'1取引の平均':>12}{'勝率':>7}{'月あたり':>10}"
                  f"{'窓の平均の最小':>14}{'平均が正の窓':>12}")
            for arm in ARMS:
                t = rule_trades(res[arm], bars, bar_days)
                n_days = sum(1 for d in bar_days
                             if res[arm]["lgbm"]["Date"].min() <= d <= res[arm]["lgbm"]["Date"].max())
                st = L.trade_stats(t, n_days)
                per = t.groupby("fold")["ret"].mean() if len(t) else pd.Series(dtype=float)
                print(f"  {arm + (' 修正前' if arm == 'A' else ' 修正後'):<10}{st['n']:>6}"
                      f"{st['mean']:>+11.2f}%{st['win']:>6.0f}%{st['monthly']:>+9.2f}%"
                      f"{(per.min() if len(per) else np.nan):>+13.2f}%"
                      f"{int((per > 0).sum()):>7}/{len(per):<4}")
                t.assign(shift=sh, arm=arm).to_csv(
                    os.path.join(OOF_DIR, f"e43_trades_{arm}_sh{sh}.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OOF_DIR, "e43_auc_by_window.csv"), index=False)

    if summary:
        s = pd.DataFrame(summary)
        print("\n■ 4. 切り方3通りをまとめた窓ごとの B−A（PR-AUC）")
        print(f"  {'':<10}{'窓の数':>7}{'B−A の平均':>12}{'SE':>9}{'Bが上':>9}")
        for a, g in s.groupby("algo"):
            d = (g["pr_b"] - g["pr_a"]).to_numpy()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            print(f"  {a:<10}{len(d):>7}{d.mean():>+12.4f}{se:>9.4f}{(d > 0).sum():>5}/{len(d)}")
    log(f"記録: {OOF_DIR}/e43_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
