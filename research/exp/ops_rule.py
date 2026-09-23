#!/usr/bin/env python3
"""
運用の規則で実収益を測る: ブースティング3モデル（LightGBM / XGBoost /
CatBoost）のスコアが、それぞれの過去窓の分布で pct パーセンタイル以上の
行だけを選ぶ（docs/MODEL_ADOPTION_RULES.md §7）。

単一モデルの上位10%（lab.threshold_edge）と違い、3モデルの合議なので
選定数は少なく（13〜18%）、正例率が高い。採否には使わず、記録する。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

BOOST = ("lgbm", "xgb", "cat")
OUTCOMES = ("ret_o1_20", "ret_o1_40")

#: 運用の選定基準（2026-09-22 に運用者が決めた2つ）。以後の性能比較はこの2つで出す。
#:   1) LightGBM 単体でスコアが上位 5%
#:   2) ブースティング3モデルすべてでスコアが上位 10%
#: docs/MODEL_ADOPTION_RULES.md §7 / docs/MODEL_SELECTION_EDA.md
RULES = (
    ("lgbm 単体 95以上", ("lgbm",), 95.0),
    ("3モデル 90以上", BOOST, 90.0),
)


def consensus(oofs: Dict[str, pd.DataFrame], pct: float = 85.0,
              min_rows: int = 100, models=BOOST, keep: bool = False) -> dict:
    """
    oofs: {algo: out-of-fold（Code, Date, fold, label, score, ret_o1_*）}。
    models のモデルすべてが pct パーセンタイル以上の行を選ぶ。
    models=("lgbm",) なら LightGBM 単体（lab.threshold_edge と同じ選び方）。
    モデル同士は同じ行・同じ窓であること（同じ df から作った out-of-fold）。

    keep=True なら、選んだ行そのものを out["rows"] に入れて返す
    （選んだ後の値動きを追う実験41 用）。集計の数字は変わらない。
    """
    first = models[0]
    base = oofs[first][["Code", "Date", "fold", "label"] + list(OUTCOMES)].copy()
    for a in models:
        base = base.merge(oofs[a][["Code", "Date", "score"]].rename(columns={"score": f"s_{a}"}),
                          on=["Code", "Date"], how="inner")
    per_fold = {oc: {} for oc in OUTCOMES}
    picks = []
    n_all = 0
    for f in sorted(base["fold"].unique()):
        prev = base[base["fold"] < f]
        if len(prev) < 500:
            continue
        cur = base[base["fold"] == f]
        ok = np.ones(len(cur), dtype=bool)
        for a in models:
            ok &= (cur[f"s_{a}"] > np.percentile(prev[f"s_{a}"], pct)).to_numpy()
        sel = cur[ok]
        if len(sel) < 5:
            continue
        n_all += len(cur)
        picks.append(sel)
        for oc in OUTCOMES:
            r = pd.to_numeric(cur[oc], errors="coerce")
            if r.notna().sum() < min_rows:
                continue
            per_fold[oc][int(f)] = float((pd.to_numeric(sel[oc], errors="coerce").mean() - r.mean()) * 100)
    if not picks:
        return {"n": 0}
    sel = pd.concat(picks)
    out = {"pct": pct, "n": int(len(sel)), "rate": float(len(sel) / n_all),
           "label_rate": float(sel["label"].mean()),
           "ret20": float(pd.to_numeric(sel["ret_o1_20"], errors="coerce").mean() * 100)}
    for oc in OUTCOMES:
        v = np.array(list(per_fold[oc].values()))
        out[oc] = {"fold_mean": float(v.mean()) if len(v) else float("nan"),
                   "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"),
                   "won": int((v > 0).sum()), "n_folds": int(len(v)),
                   "worst": float(v.min()) if len(v) else float("nan"),
                   "per_fold": {int(k): float(x) for k, x in per_fold[oc].items()}}
    if keep:
        out["rows"] = sel.reset_index(drop=True)
    return out


def fmt(name: str, r: dict) -> str:
    if not r.get("n"):
        return f"  {name:<30} （選定なし）"
    a, b = r["ret_o1_20"], r["ret_o1_40"]
    return (f"  {name:<30}{r['n']:>7,}{r['rate']*100:>6.1f}%{r['label_rate']*100:>7.1f}%"
            f"{r['ret20']:>+8.2f}%{a['fold_mean']:>+9.2f}pt{a['se']:>6.2f}{a['won']:>4}/{a['n_folds']:<2}"
            f"{a['worst']:>+8.2f}pt{b['fold_mean']:>+9.2f}pt")


HEADER = (f"  {'腕':<30}{'選定数':>7}{'選定率':>7}{'正例率':>8}{'ret20':>9}"
          f"{'窓平均超過20':>10}{'SE':>6}{'勝ち':>7}{'最悪':>9}{'超過40':>10}")


def report(oofs: Dict[str, pd.DataFrame], label: str = "") -> Dict[str, dict]:
    """RULES の2つの基準で測って印字する。戻り値は {規則名: consensus の結果}。"""
    print(HEADER if not label else f"{label}\n{HEADER}")
    out = {}
    for name, models, pct in RULES:
        if any(a not in oofs for a in models):
            continue
        out[name] = consensus(oofs, pct, models=models)
        print(fmt(name, out[name]))
    return out
