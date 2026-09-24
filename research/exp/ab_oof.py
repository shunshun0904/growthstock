#!/usr/bin/env python3
"""
腕（データセットと列の組）どうしを、本番と同じ out-of-fold の作りで比べる共通部品。

実験43（時点整合の修正）と同じ手順を、実験44（空売り残高報告）・実験45（取り込みの
重複除去の修正）で使う。**1本の OOF で決めない**（運用者の指示）ので、

  窓: 36/6/6か月、境界を 0/2/4か月ずらした3通り（実験41 §10 の folds_for / oof_folds）
  モデル: ブースティング3つ（本番のパラメータを読むだけ。書かない）、種3つの平均
  見るもの: 窓ごとの PR-AUC / ROC-AUC の差（平均・SE・勝った窓の数）と、運用の規則
           （research/live_track.py と同じ）での取引。どちらも切り方ごと

途中の out-of-fold は research/_data/oof/<tag>_<腕>_<中身の指紋>_... に置く。
Actions のキャッシュで次の実行に引き継がれるので、中身が変わったら別の名前になる
ように指紋を入れる。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import live_track as L  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e25_auc_noise import average  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fingerprint(df: pd.DataFrame, cols: List[str]) -> str:
    """学習に使う中身の指紋（行・ラベル・列の値）。"""
    h = pd.util.hash_pandas_object(df[["Code", "Date", "label"] + list(cols)], index=False)
    return f"{int(h.to_numpy(dtype=np.uint64).sum(dtype=np.uint64)) & 0xFFFFFFFF:08x}"


def oof_arm(tag: str, df: pd.DataFrame, cols: List[str], arm: str, shift: int,
            algos, seeds) -> Dict[str, pd.DataFrame]:
    """腕・切り方ごとの out-of-fold（種の平均）。保存済みなら読む（指紋が同じときだけ）。"""
    folds = E41.folds_for(df["Date"], shift)
    fp = fingerprint(df, cols)
    out = {}
    for a in algos:
        par = E27.prod_params(a)
        parts = []
        for sd in seeds:
            path = os.path.join(OOF_DIR, f"{tag}_{arm}_{fp}_{a}_sh{shift}_s{sd}.parquet")
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


def rule_trades(oofs: Dict[str, pd.DataFrame], bars: pd.DataFrame, bar_days) -> pd.DataFrame:
    """運用の規則で取った取引（live_track と同じ関数）。fold は選定日の窓。"""
    base = None
    for a in L.BOOST:
        o = oofs[a][["Code", "Date", "fold", "label", "score"]].rename(columns={"score": f"s_{a}"})
        o[f"p_{a}"] = L.pct_of(o[f"s_{a}"].to_numpy(), o[f"s_{a}"].to_numpy())
        base = o if base is None else base.merge(o.drop(columns=["label", "fold"]),
                                                 on=["Code", "Date"], how="inner")
    base["score"] = base["s_lgbm"]
    _, _, picks = L.decide(base)
    pf = L.forward(bars, picks)
    sim = L.simulate(pf, bar_days)
    return sim[sim["taken"] == L.TAKEN].copy()


def pair_line(name: str, a: np.ndarray, b: np.ndarray) -> str:
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return (f"  {name:<18}{a.mean():>9.4f}{b.mean():>9.4f}"
            f"{d.mean():>+10.4f}{se:>9.4f}{(d > 0).sum():>5}/{len(d):<3}")


def compare(tag: str, arms: Dict[str, Tuple[pd.DataFrame, List[str]]], base: str,
            labels: Dict[str, str], shifts: List[int], seeds: List[int],
            algos: List[str]) -> pd.DataFrame:
    """
    base の腕とほかの腕を窓ごとに比べて表示する。窓ごとの結果を返す。

    arms: 腕 -> (frame, 列)。frame は Code / Date / label と列を持つ（lab.frame() の形）。
    """
    os.makedirs(OOF_DIR, exist_ok=True)
    bars = L.load_bars(lab.DATA_DIR, start=pd.Timestamp("2021-01-01"))
    bar_days = sorted(pd.Timestamp(d) for d in bars["Date"].unique())
    others = [a for a in arms if a != base]
    summary = []
    for sh in shifts:
        res = {arm: oof_arm(tag, df, cols, arm, sh, algos, seeds)
               for arm, (df, cols) in arms.items()}
        for arm in others:
            print(f"\n■ 分離力（窓ごと。ずらし{sh}か月）: {labels[arm]} − {labels[base]}")
            print(f"  {'':<18}{base:>9}{arm:>9}{'差':>10}{'SE':>9}{'上の窓':>9}")
            for a in algos:
                wa, wb = auc_by_window(res[base][a]), auc_by_window(res[arm][a])
                m = wa.merge(wb, on="fold", suffixes=("_a", "_b"))
                print(pair_line(f"{a} PR-AUC", m["pr_a"].to_numpy(), m["pr_b"].to_numpy()))
                print(pair_line(f"{a} ROC-AUC", m["roc_a"].to_numpy(), m["roc_b"].to_numpy()))
                for _, r in m.iterrows():
                    summary.append({"shift": sh, "arm": arm, "algo": a, "fold": int(r["fold"]),
                                    "pr_base": r["pr_a"], "pr_arm": r["pr_b"],
                                    "roc_base": r["roc_a"], "roc_arm": r["roc_b"]})
        if set(L.BOOST) <= set(algos):
            print(f"\n■ 運用の規則での取引（ずらし{sh}か月）")
            print(f"  {'腕':<24}{'取引':>6}{'1取引の平均':>12}{'勝率':>7}{'月あたり':>10}"
                  f"{'窓の平均の最小':>14}{'平均が正の窓':>12}")
            for arm in arms:
                t = rule_trades(res[arm], bars, bar_days)
                lg = res[arm]["lgbm"]
                n_days = sum(1 for d in bar_days if lg["Date"].min() <= d <= lg["Date"].max())
                st = L.trade_stats(t, n_days)
                per = t.groupby("fold")["ret"].mean() if len(t) else pd.Series(dtype=float)
                print(f"  {labels[arm]:<24}{st['n']:>6}{st['mean']:>+11.2f}%{st['win']:>6.0f}%"
                      f"{st['monthly']:>+9.2f}%{(per.min() if len(per) else np.nan):>+13.2f}%"
                      f"{int((per > 0).sum()):>7}/{len(per):<4}")
                t.assign(shift=sh, arm=arm).to_csv(
                    os.path.join(OOF_DIR, f"{tag}_trades_{arm}_sh{sh}.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OOF_DIR, f"{tag}_auc_by_window.csv"), index=False)
    if len(s):
        print("\n■ 切り方3通りをまとめた窓ごとの差（PR-AUC）")
        print(f"  {'':<24}{'窓の数':>7}{'差の平均':>10}{'SE':>9}{'上の窓':>9}")
        for (arm, a), g in s.groupby(["arm", "algo"]):
            d = (g["pr_arm"] - g["pr_base"]).to_numpy()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            print(f"  {labels[arm] + ' ' + a:<24}{len(d):>7}{d.mean():>+10.4f}{se:>9.4f}"
                  f"{(d > 0).sum():>5}/{len(d)}")
    log(f"記録: {OOF_DIR}/{tag}_*")
    return s
