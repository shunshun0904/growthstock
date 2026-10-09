#!/usr/bin/env python3
"""
実験68: 時間反転非対称性 nl_tra1_120 を「選んだ後の除外規則」として測る（実験41 §9c / §9e と同じ形）。

運用者の依頼（2026-10-09）「上位帯の中で実収益の差が大きかった時間反転非対称性 nl_tra1_120 を、実験41 の
除外規則の形で測る」。実験66 §4 で、本番 OOF の上位10% の中で nl_tra1_120 の上半分は下半分より ret_o1_20 が
−2.65pt（正の窓 1/11）だった。列は label（+1.2σ 到達）ではなく収益の大きさに効くので、モデルの列としてではなく
**選んだ後に外す規則**として測る。

作り（実験41 §9e と同じ）
  スコア    実験66 の腕 T（本番の239列・本番のパラメータ・木200本・種3つの平均）の out-of-fold。
            窓の切り方 0 / 2 / 4 か月の3通り
  選定      ① 3モデル 90以上（運用の規則。ops_rule.consensus）② lgbm 単体 95以上 ③ lgbm 窓の中の上位10%
            （③は実験66 §4 で差を見つけた帯そのもの。①②が運用の形）
  除外      nl_tra1_120 が「それより前の窓の母集団」の上位10% / 25% / 50% に入る行を外す。しきい値に先の値は
            使わない（E41.exclusion と同じ。最初の窓は参照が無いので外さない）。対照として下位10% を外す規則も
  見るもの  外したあとの平均 ret_o1_20 − 外さない平均（pt）、窓ごとに外したほうが良かった数、外した件数、
            外した行の平均収益、−10% 以下（大負け）と正例の割合の変化

注意（実験41 §7-1 と同じ）: 列そのものは実験66 §4 で全窓を見て選んだ。しきい値は前の窓だけから決めるが、
列の選び方に先読みがある。切り方3通りで向きがそろうかを見るが、独立の確認にはならない。

  python3 research/exp/e68_tra1_exclusion.py [--shifts 0,2,4] [--seeds 3]
  結果は research/_data/oof/e68_*。本番の設定には書かない。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import ab_oof as AB  # noqa: E402
import ops_rule as OR  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e66_nonlinear_screen as E66  # noqa: E402
from e25_auc_noise import average  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
COL = "nl_tra1_120"
OUTCOME = lab.OUTCOME
BIG = -0.10
#: 除外の規則: (名前, 側, 母集団の百分位)。top は「百分位以上を外す」、bottom は「百分位以下を外す」
RULES: Tuple[Tuple[str, str, float], ...] = (
    ("上位10% を外す", "top", 90.0),
    ("上位25% を外す", "top", 75.0),
    ("上半分を外す", "top", 50.0),
    ("対照: 下位10% を外す", "bottom", 10.0),
)
SELECTIONS = ("3モデル 90以上", "lgbm 単体 95以上", "lgbm 窓の中の上位10%")


def log(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_oofs(df: pd.DataFrame, base: List[str], shift: int, seeds, src_tag: str = "e66") -> Dict[str, pd.DataFrame]:
    """実験66 の腕 T の out-of-fold（3モデル・種の平均）。"""
    fp = AB.fingerprint(df, base)
    out = {}
    for a in OR.BOOST:
        parts = []
        for sd in seeds:
            p = os.path.join(OOF_DIR, f"{src_tag}_T_{fp}_{a}_sh{shift}_s{sd}.parquet")
            if not os.path.exists(p):
                raise SystemExit(f"{p} がありません。先に e66_nonlinear_screen.py --stage ab を回してください")
            parts.append(pd.read_parquet(p))
        o = average(parts)
        o["Date"] = pd.to_datetime(o["Date"])
        o["Code"] = o["Code"].astype(str)
        out[a] = o
    return out


def selections(oofs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    sel = {}
    r = OR.consensus(oofs, 90.0, models=OR.BOOST, keep=True)
    sel["3モデル 90以上"] = r["rows"] if r.get("n") else pd.DataFrame()
    r = OR.consensus(oofs, 95.0, models=("lgbm",), keep=True)
    sel["lgbm 単体 95以上"] = r["rows"] if r.get("n") else pd.DataFrame()
    o = oofs["lgbm"]
    pct = o.groupby("fold")["score"].rank(pct=True)
    sel["lgbm 窓の中の上位10%"] = o[pct >= 0.9].reset_index(drop=True)
    return sel


def exclude_mask(s: pd.DataFrame, ref: pd.DataFrame, col: str, side: str, q: float) -> np.ndarray:
    """
    選んだ行のうち、列 col が「それより前の窓の母集団」の百分位 q 以上（top）/ 以下（bottom）の行を True に。
    しきい値に先の値は使わない。参照が 500行未満の窓（最初の窓）は外さない。値が無い行は外さない。
    """
    out = np.zeros(len(s), dtype=bool)
    x = pd.to_numeric(s[col], errors="coerce").to_numpy(dtype=float)
    fo = s["fold"].to_numpy()
    for k in np.unique(fo):
        prev = pd.to_numeric(ref.loc[ref["fold"] < k, col], errors="coerce").dropna()
        if len(prev) < 500:
            continue
        m = fo == k
        thr = np.percentile(prev, q)
        out[m] = (x[m] >= thr) if side == "top" else (x[m] <= thr)
    out &= np.isfinite(x)
    return out


def evaluate(sel: pd.DataFrame, ref: pd.DataFrame, col: str = COL, rules=RULES) -> pd.DataFrame:
    base = pd.to_numeric(sel[OUTCOME], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sel["label"], errors="coerce").to_numpy(dtype=float)
    fo = sel["fold"].to_numpy()
    rows = []
    for name, side, q in rules:
        m = exclude_mask(sel, ref, col, side, q)
        won = nf = 0
        diffs = []
        for w in np.unique(fo):
            ww = fo == w
            if (ww & ~m).sum() and np.isfinite(base[ww]).any():
                d = np.nanmean(base[ww & ~m]) - np.nanmean(base[ww])
                nf += 1
                won += int(d > 1e-12)
                diffs.append(d)
        keep = ~m
        rows.append({
            "rule": name, "n": int(len(sel)), "n_excl": int(m.sum()),
            "before": np.nanmean(base) * 100, "after": np.nanmean(base[keep]) * 100,
            "diff": (np.nanmean(base[keep]) - np.nanmean(base)) * 100,
            "diff_fold_mean": float(np.mean(diffs)) * 100 if diffs else np.nan,
            "won": won, "nf": nf,
            "excl_mean": np.nanmean(base[m]) * 100 if m.any() else np.nan,
            "bad_before": np.nanmean(base <= BIG) * 100, "bad_after": np.nanmean(base[keep] <= BIG) * 100,
            "pos_before": np.nanmean(y) * 100, "pos_after": np.nanmean(y[keep]) * 100,
        })
    return pd.DataFrame(rows)


def shape(sel: pd.DataFrame, col: str = COL) -> pd.DataFrame:
    """選んだ行の中で、列の四分位ごとの平均収益（記述。しきい値は選定の中の分位なので先読みを含む）。"""
    x = pd.to_numeric(sel[col], errors="coerce")
    q = pd.qcut(x.rank(method="first"), 4, labels=["Q1 低", "Q2", "Q3", "Q4 高"])
    r = pd.to_numeric(sel[OUTCOME], errors="coerce")
    g = pd.DataFrame({"q": q, "r": r, "y": sel["label"]}).dropna(subset=["q"]).groupby("q", observed=True)
    return pd.DataFrame({"n": g.size(), "ret": g["r"].mean() * 100, "bad": g["r"].apply(lambda v: (v <= BIG).mean() * 100),
                         "pos": g["y"].mean() * 100})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験68: nl_tra1_120 の除外規則")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    os.makedirs(OOF_DIR, exist_ok=True)

    df = E66.load_frame()
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    base = F.columns(F.DEFAULT_PRESET)
    print("=" * 78)
    print(f"実験68 {COL} の除外規則（実験66 の腕 T のスコア・種{len(seeds)}つ・ずらし {shifts}か月）: {len(df):,}件")
    print(f"  列: {COL} = {E66.NL.NL_DESC[COL]}")
    print("=" * 78)
    feat = df[["Code", "Date", COL, "vol_20d"]]
    all_rows = []
    for sh in shifts:
        oofs = load_oofs(df, base, sh, seeds)
        ref = oofs["lgbm"][["Code", "Date", "fold"]].merge(feat, on=["Code", "Date"], how="left")
        sels = selections(oofs)
        print(f"\n■ ずらし{sh}か月（母集団 {len(ref):,}件 / 窓 {ref['fold'].nunique()}）")
        for name in SELECTIONS:
            s = sels[name]
            if not len(s):
                print(f"  {name}: 選定なし")
                continue
            s = s.merge(feat, on=["Code", "Date"], how="left")
            res = evaluate(s, ref)
            res.insert(0, "selection", name)
            res.insert(0, "shift", sh)
            all_rows.append(res)
            r0 = res.iloc[0]
            print(f"  ◆ {name}: {len(s):,}件 / 平均 {r0['before']:+.2f}% / −10%以下 {r0['bad_before']:.1f}% / "
                  f"正例率 {r0['pos_before']:.1f}% / {COL} 欠測 {s[COL].isna().mean()*100:.1f}%")
            print(f"    {'外す条件':<22}{'外す':>6}{'残りの平均':>10}{'差pt':>8}{'勝ち窓':>8}{'外した側の平均':>14}"
                  f"{'大負け':>14}{'正例率':>14}")
            for _, r in res.iterrows():
                print(f"    {r['rule']:<22}{int(r['n_excl']):>6}{r['after']:>+9.2f}%{r['diff']:>+8.2f}"
                      f"{int(r['won']):>5}/{int(r['nf']):<3}{r['excl_mean']:>+13.2f}%"
                      f"{r['bad_before']:>6.1f}→{r['bad_after']:<5.1f}%{r['pos_before']:>6.1f}→{r['pos_after']:<5.1f}%")
            if sh == shifts[0]:
                sp = shape(s)
                print(f"    （記述）選んだ行の中の {COL} 四分位: "
                      + " / ".join(f"{i} {int(r['n'])}件 {r['ret']:+.2f}% 大負け {r['bad']:.1f}% 正例 {r['pos']:.0f}%"
                                   for i, r in sp.iterrows()))
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out.to_csv(os.path.join(OOF_DIR, "e68_exclusion.csv"), index=False)
    if len(out) and len(shifts) > 1:
        print(f"\n■ 切り方{len(shifts)}通りをまとめて（差pt は切り方ごと、勝ち窓は合計）")
        print(f"  {'選定':<22}{'外す条件':<22}" + "".join(f"{'ずらし' + str(s):>12}" for s in shifts)
              + f"{'平均':>9}{'勝ち窓':>9}{'外す件数':>9}")
        for name in SELECTIONS:
            for rule, _, _ in RULES:
                g = out[(out["selection"] == name) & (out["rule"] == rule)].sort_values("shift")
                if not len(g):
                    continue
                cells = "".join(f"{v:>+12.2f}" for v in g["diff"])
                print(f"  {name:<22}{rule:<22}{cells}{g['diff'].mean():>+9.2f}"
                      f"{int(g['won'].sum()):>5}/{int(g['nf'].sum()):<3}{int(g['n_excl'].sum()):>9}")
    log(f"記録: {OOF_DIR}/e68_exclusion.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
