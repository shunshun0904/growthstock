#!/usr/bin/env python3
"""
実験25: 種だけを振ったとき、out-of-fold の PR-AUC などはどれだけ動くか。

第2の採否規則（docs/MODEL_ADOPTION_RULES.md §3-4）の「PR-AUC の改善幅が
種のレンジを超える」の、レンジを測る。実験11 は窓平均のレンジ（0.143pt）を
測ったが、PR-AUC はその後ラベル・物差し・パラメータが変わったので測り直す。

同一設定（本番の特徴量 `all`、本番のパラメータ、本番の out-of-fold と同じ窓）で
種を 5 つ回し、
  - 種1つずつの指標のレンジ
  - 種3つの確率平均（A/B で使う形）10通りの指標のレンジ
を出す。種ごとの out-of-fold は保存して実験26 の腕 A に使い回す。
"""

from __future__ import annotations

import itertools
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
import e19_freshdata as E19  # noqa: E402
from e18_horizon import edge  # noqa: E402
from e21_annual_ab import PARAMS, PRESET  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

SEEDS5 = (42, 7, 123, 2024, 31337)
OUTCOMES = ("ret_o1_20", "ret_o1_40")
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")


def metrics(o: pd.DataFrame) -> dict:
    y = o["label"].to_numpy(dtype=int)
    s = o["score"].to_numpy(dtype=float)
    m = {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
         "day_auc": lab.auc_in_day(o)}
    m["lift"] = m["pr_auc"] / y.mean()
    for oc in OUTCOMES:
        e = edge(o, oc)
        m[f"{oc}_mean"] = e["thr_fold_mean"]
        m[f"{oc}_won"] = e["thr_folds_won"]
        m[f"{oc}_n"] = e["thr_folds"]
        m[f"{oc}_worst"] = e["thr_worst"]
    return m


def single_seed_oof(df: pd.DataFrame, cols: list, params: dict, seed: int,
                    path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    E19.SEEDS = (seed,)
    o = E19.oof_for(df, cols, params)
    o.to_parquet(path, index=False)
    return o


def average(oofs: list) -> pd.DataFrame:
    o = oofs[0].copy()
    o["score"] = np.mean([x["score"].to_numpy(dtype=float) for x in oofs], axis=0)
    return o


def show(title: str, rows: list) -> None:
    keys = [("pr_auc", "PR-AUC", 1), ("lift", "PR/正例率", 1), ("roc_auc", "ROC-AUC", 1),
            ("day_auc", "日内AUC", 1), ("ret_o1_20_mean", "窓平均20", 1),
            ("ret_o1_20_won", "勝ち窓20", 1), ("ret_o1_20_worst", "最悪20", 1),
            ("ret_o1_40_mean", "窓平均40", 1), ("ret_o1_40_won", "勝ち窓40", 1),
            ("ret_o1_40_worst", "最悪40", 1)]
    print(f"\n=== {title}（{len(rows)}通り）===")
    print(f"  {'指標':<12}{'平均':>10}{'標準偏差':>10}{'最小':>10}{'最大':>10}{'レンジ':>10}")
    out = {}
    for k, t, _ in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": v.mean(), "sd": v.std(ddof=1), "min": v.min(), "max": v.max(),
                  "range": v.max() - v.min()}
        print(f"  {t:<12}{v.mean():>10.4f}{v.std(ddof=1):>10.4f}{v.min():>10.4f}"
              f"{v.max():>10.4f}{v.max()-v.min():>10.4f}")
    return out


def main() -> int:
    os.makedirs(OOF_DIR, exist_ok=True)
    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    cols = [c for c in F.columns(PRESET) if c in df.columns]
    with open(PARAMS, encoding="utf-8") as fh:
        params = {k: v for k, v in json.load(fh)[PRESET].items() if not k.startswith("_")}
    print(f"母集団 {len(df):,}件 / 特徴量 {PRESET}（{len(cols)}列）/ 本番のパラメータ / 種 {SEEDS5}")

    oofs = {}
    for s in SEEDS5:
        t0 = time.time()
        oofs[s] = single_seed_oof(df, cols, params, s, os.path.join(OOF_DIR, f"e25_A_s{s}.parquet"))
        print(f"  種 {s}: {len(oofs[s]):,}件 / {time.time()-t0:.0f}秒")

    singles = [metrics(oofs[s]) for s in SEEDS5]
    r1 = show("種1つずつ", singles)
    triples = [metrics(average([oofs[s] for s in c])) for c in itertools.combinations(SEEDS5, 3)]
    r3 = show("種3つの確率平均（A/B で使う形）", triples)
    quint = metrics(average([oofs[s] for s in SEEDS5]))
    print(f"\n種5つの確率平均: PR-AUC {quint['pr_auc']:.4f} / 窓平均20 {quint['ret_o1_20_mean']:+.2f}pt "
          f"勝ち {quint['ret_o1_20_won']}/{quint['ret_o1_20_n']} 最悪 {quint['ret_o1_20_worst']:+.2f}pt")
    with open(os.path.join(OOF_DIR, "e25_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"single": r1, "triple": r3, "quint": quint}, fh, ensure_ascii=False, indent=1)
    print(f"\n記録: {OOF_DIR}/e25_A_s*.parquet, e25_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
