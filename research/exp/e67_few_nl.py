#!/usr/bin/env python3
"""
実験67: 非線形時系列解析の列を「先に決めた少数」だけ足す腕（実験66 の続き）。

運用者の依頼（2026-10-09）「先に決めた少数だけを足す腕。群ごとに 1 本で、判定は切り方をずらした窓で行う」。
確認（AskUserQuestion）で「6列: 群ごとに最大 |z| の列」を選択。

列の選び方（結果を見る前に決めた規則）
  実験66 の全行スクリーニング（ずらし0か月の11窓、ret_o1_20 の上位/下位10% の z。
  docs/FEATURE_IDEAS_NONLINEAR.md §3）で、群ごとに最大 |z| の1列。
    A 持続性    nl_dfa_abs120   （|z| 3.90）
    B 複雑さ    nl_sampen_120   （4.09）
    C 非線形性  nl_bds2_120     （4.97）
    D 再帰性    nl_rqa_entr_120 （1.50。足切り未満だが規則を優先）
    E 経路の形  nl_er_60        （3.06）
    F 出来高    nl_tv_burst_60  （2.98）
  32列を一括で足した実験66 では3モデルとも分離力が動かなかった（§27）。LightGBM の colsample_bytree は 0.37
  なので、効かない列が 26本混ざると当たりの列に木が当たる機会が減る、という仮説を測る。

判定
  列を選んだ窓（ずらし0か月）は判定に使わない（同じ窓で選んで同じ窓で測ると先読みになる。実験41 §7-1 の注意）。
  **ずらし2・4か月の21窓**で §7（docs/MODEL_ADOPTION_RULES.md、2026-09-26 改訂）: 3モデル中2つ以上で
  V6 − T の PR-AUC が正、対照 P6 を上回り、上の窓が過半（21窓なら 11以上）。ずらし0 は参考として並べる。

腕（本番のパラメータ・木200本・種3つ）
  T   本番の239列（実験66 の T の out-of-fold と中身が同じなので、保存済みがあれば流用する）
  V6  T + 6列（245列）
  P6  対照: V6 の6列を同じ日の銘柄どうしで入れ替えたもの（列を足しただけで動く幅）

  python3 research/exp/e67_few_nl.py [--shifts 0,2,4] [--seeds 3] [--algos lgbm,xgb,cat]
  結果は research/_data/oof/e67_*。本番の設定には書かない。
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import nonlinear_features as NL  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e66_nonlinear_screen as E66  # noqa: E402
from e44_shortsale import permuted  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
TAG = "e67"
#: 群 -> 列（結果を見る前に決めた規則: 実験66 の全行スクリーニングで群ごとに最大 |z|）
FEW_BY_FAMILY: Dict[str, str] = {
    "A 持続性": "nl_dfa_abs120",
    "B 複雑さ": "nl_sampen_120",
    "C 非線形性": "nl_bds2_120",
    "D 再帰性": "nl_rqa_entr_120",
    "E 経路の形": "nl_er_60",
    "F 出来高": "nl_tv_burst_60",
}
FEW_COLS: List[str] = list(FEW_BY_FAMILY.values())
LABELS = {"T": "T 本番の239列", "V6": "V6 T + 非線形6列", "P6": "P6 対照（6列を日付内で入れ替え）"}
#: 判定に使う切り方（列を選んだ ずらし0 は外す）
JUDGE_SHIFTS = (2, 4)
PERM_SEED = E66.PERM_SEED


def reuse_t(fp: str, src_tag: str = "e66") -> int:
    """実験66 の T の out-of-fold（同じ指紋・同じパラメータ）を、このタグの名前で使えるようにする。"""
    n = 0
    for p in sorted(glob.glob(os.path.join(OOF_DIR, f"{src_tag}_T_{fp}_*.parquet"))):
        q = os.path.join(OOF_DIR, os.path.basename(p).replace(f"{src_tag}_T_", f"{TAG}_T_", 1))
        if not os.path.exists(q):
            try:
                os.link(p, q)
            except OSError:
                shutil.copyfile(p, q)
            n += 1
    return n


def judge(s: pd.DataFrame, judge_shifts=JUDGE_SHIFTS, base_arm: str = "T", arm: str = "V6",
          ctrl: str = "P6") -> pd.DataFrame:
    """
    {TAG}_auc_by_window.csv（shift / arm / algo / fold / pr_base / pr_arm / roc_base / roc_arm）から、
    判定に使う切り方の窓だけで V6 − T と P6 − T をまとめ、§7 の条件をモデルごとに判定する。
    """
    rows = []
    for algo, g in s.groupby("algo"):
        rec = {"algo": algo}
        for which, name in (("judge", "判定"), ("ref", "参考")):
            part = g[g["shift"].isin(judge_shifts)] if which == "judge" else g[~g["shift"].isin(judge_shifts)]
            for a in (arm, ctrl):
                h = part[part["arm"] == a]
                d = (h["pr_arm"] - h["pr_base"]).to_numpy()
                r = (h["roc_arm"] - h["roc_base"]).to_numpy()
                rec[f"{which}_{a}_n"] = len(d)
                rec[f"{which}_{a}_pr"] = d.mean() if len(d) else np.nan
                rec[f"{which}_{a}_pr_se"] = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                rec[f"{which}_{a}_pr_up"] = int((d > 0).sum())
                rec[f"{which}_{a}_roc"] = r.mean() if len(r) else np.nan
                rec[f"{which}_{a}_roc_up"] = int((r > 0).sum())
        n = rec[f"judge_{arm}_n"]
        rec["ok"] = bool(n > 0 and rec[f"judge_{arm}_pr"] > 0
                         and rec[f"judge_{arm}_pr"] > rec[f"judge_{ctrl}_pr"]
                         and rec[f"judge_{arm}_pr_up"] > n / 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def show_judge(res: pd.DataFrame, arm: str = "V6", ctrl: str = "P6") -> None:
    print(f"\n■ 判定（ずらし {list(JUDGE_SHIFTS)}か月の窓だけ。列を選んだ ずらし0 は参考）")
    print(f"  {'':<6}{'窓':>4}{arm + '−T PR':>12}{'SE':>8}{'上の窓':>8}{ctrl + '−T PR':>12}{'上の窓':>8}"
          f"{arm + '−T ROC':>13}{'上の窓':>8}{'§7':>5}  |{'参考 ずらし0 ' + arm + '−T':>22}{'上の窓':>8}")
    for _, r in res.iterrows():
        n, n0 = int(r[f"judge_{arm}_n"]), int(r[f"ref_{arm}_n"])
        print(f"  {r['algo']:<6}{n:>4}{r[f'judge_{arm}_pr']:>+12.4f}{r[f'judge_{arm}_pr_se']:>8.4f}"
              f"{int(r[f'judge_{arm}_pr_up']):>5}/{n:<3}{r[f'judge_{ctrl}_pr']:>+12.4f}"
              f"{int(r[f'judge_{ctrl}_pr_up']):>5}/{n:<3}{r[f'judge_{arm}_roc']:>+13.4f}"
              f"{int(r[f'judge_{arm}_roc_up']):>5}/{n:<3}{'○' if r['ok'] else '×':>5}  |"
              f"{r[f'ref_{arm}_pr']:>+22.4f}{int(r[f'ref_{arm}_pr_up']):>5}/{n0:<3}")
    k = int(res["ok"].sum())
    print(f"  §7 の条件（正・対照を上回る・上の窓が過半）を満たすモデル: {k}/{len(res)} → "
          + ("採用の条件に達する（2つ以上）" if k >= 2 else "採用しない（2つ未満）"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験67: 非線形の列を少数だけ足す腕")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default="lgbm,xgb,cat")
    ap.add_argument("--judge-only", action="store_true", help="保存済みの CSV から判定だけ出す")
    args = ap.parse_args(argv)
    os.makedirs(OOF_DIR, exist_ok=True)
    assert set(FEW_COLS) <= set(NL.NL_COLS) and len(set(FEW_COLS)) == 6
    if args.judge_only:
        s = pd.read_csv(os.path.join(OOF_DIR, f"{TAG}_auc_by_window.csv"))
        show_judge(judge(s))
        return 0

    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    algos = [a for a in args.algos.split(",") if a]
    df = E66.load_frame()
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    base = F.columns(F.DEFAULT_PRESET)
    v6 = base + FEW_COLS
    fp_t = AB.fingerprint(df, base)
    n_reused = reuse_t(fp_t)
    print("=" * 78)
    print(f"実験67 少数の列（種{len(seeds)}つ・ずらし {shifts}か月・{algos}）: {len(df):,}件 / "
          f"T の指紋 {fp_t}（実験66 から流用 {n_reused}本）")
    for fam, c in FEW_BY_FAMILY.items():
        print(f"  {fam:<10} {c:<18} {NL.NL_DESC[c][:60]}")
    for arm, cols in (("T", base), ("V6", v6), ("P6", v6)):
        print(f"  {LABELS[arm]:<30}{len(cols)}列  指紋 {F.signature(cols)}")
    print("=" * 78)
    fp = permuted(df, FEW_COLS, seed=PERM_SEED)
    arms = {"T": (df, base), "V6": (df, v6), "P6": (fp, v6)}
    s = AB.compare(TAG, arms, "T", LABELS, shifts, seeds, algos)
    if len(s):
        res = judge(s)
        res.to_csv(os.path.join(OOF_DIR, f"{TAG}_judge.csv"), index=False)
        show_judge(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
