#!/usr/bin/env python3
"""
実験70: σ_dfa のラベル（実験69）を、実験55 と同じ作りで確かめる。

運用者の依頼（2026-10-09）「σ_dfa のラベルを実験55 と同じ作りで確かめる。そのラベルで LightGBM を探索し直し、
XGBoost・CatBoost と運用規則の取引まで」。

作りは実験55（research/exp/e55_label_sigma_full.py）そのもので、腕だけを差し替える:
  L0  1.2 × σ20（現行）
  L5  1.2 × σ_dfa（σ_dfa = σ20^w × σ120^(1−w)、w = clip((α − 0.5) / 0.3, 0, 1)、α = nl_dfa_abs120。実験69 と同じ式）
学習（本番の239列）: lgbm B1（本番の木の形）/ lgbm B2（そのラベルで探索し直し: 50試行 × 5分割 year_cap_date、
ホールドアウトより前）/ xgb・cat（本番の木の形）。評価は実験55 と同じ（閾値ルール・発火数そろえ・窓の中の上位10%・
日付内・ボラの帯・運用規則 ops_rule と live_track の取引、3モデル合議の百分位 90 / 95）。3切り方 × 種3つ。

実装は実験55 の関数を流用する。差し替えるのは4点だけ（configure）:
  1. E54.sigmas_from_bars に σ_dfa を足す（α は research/_data/nonlinear_features.parquet から）
  2. 腕の表（ARMS / LABELS）を L0 / L5 にする
  3. 保存名を e70_*（out-of-fold・探索結果・集計 CSV）にする
  4. out-of-fold の読み書きを e70_ の名前にする

  python3 research/exp/e70_label_sigma_dfa_full.py [--trials 50] [--shifts 0,2,4] [--seeds 3] [--consensus 90,95]
  結果は research/_data/oof/e70_*。本番の設定には書かない。

的中率の表（学習しない。上の実行で保存した out-of-fold を読むだけ）:
  python3 research/exp/e70_label_sigma_dfa_full.py --precision [--shifts 0,2,4] [--seeds 3]
  運用の選定（lgbm 単体 95 / 3モデル 90 / 3モデル 95）で選ばれた行の、正例率（自分のラベル / 現行ラベル）・
  勝率（ret_o1_20 > 0）・+10% 以上・−10% 未満を、腕 × 木の形 × 切り方で出す。運用者の目的（選ぶ件数は少なくてよいので
  選んだものが当たること）に合わせた物差し。結果は e70_precision.csv。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import nonlinear_features as NL  # noqa: E402
import ops_rule as OR  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e54_label_sigma as E54  # noqa: E402
import e55_label_sigma_full as E55  # noqa: E402
import e69_label_sigma_dfa as E69  # noqa: E402
from e25_auc_noise import average  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PREFIX = "e70"
ARMS = {"L0": "sigma20", "L5": "sigma_dfa"}
LABELS = {"L0": "L0 1.2σ20（現行）", "L5": "L5 1.2σ_dfa"}
SEEDS = (42, 7, 123)  # e27_timing_multi.SEEDS3 と同じ（実験55 の --seeds 3）
#: 的中率の表で見る選定: 運用の規則 2つ（ops_rule.RULES）+ 3モデル合議 95
SELECTIONS = (("lgbm 単体 95以上", ("lgbm",), 95.0),
              ("3モデル 90以上", OR.BOOST, 90.0),
              ("3モデル 95以上", OR.BOOST, 95.0))
BIG = 0.10  # 「+10% 以上」「−10% 未満」の境（ret_o1_20、比率）

_orig_sigmas = E54.sigmas_from_bars


def sigmas_with_dfa(bars: pd.DataFrame) -> pd.DataFrame:
    """実験54 の σ の表に σ_dfa を足す（α は nonlinear_features.parquet の nl_dfa_abs120）。"""
    sg = _orig_sigmas(bars)
    if not os.path.exists(NL.OUT_PATH):
        raise SystemExit(f"{NL.OUT_PATH} がありません。python3 research/nonlinear_features.py で作ってください")
    nl = pd.read_parquet(NL.OUT_PATH, columns=["Code", "Date", E69.ALPHA_COL])
    nl["Code"] = nl["Code"].astype(str)
    nl["Date"] = pd.to_datetime(nl["Date"])
    sg = sg.merge(nl, on=["Code", "Date"], how="left")
    sg["sigma_dfa"] = E69.sigma_dfa(sg["sigma20"], sg["sigma120"], sg[E69.ALPHA_COL]).to_numpy()
    return sg.drop(columns=[E69.ALPHA_COL])


def oof_path(arm: str, fit: str, algo: str, shift: int, seed: int) -> str:
    return os.path.join(OOF_DIR, f"{PREFIX}_{arm}_{fit}_{algo}_sh{shift}_s{seed}.parquet")


def oof70(df, cols, arm, fit, algo, shift, seeds, params):
    """E55.oof と同じ。保存名だけ e70_。"""
    folds = E41.folds_for(df["Date"], shift)
    parts = []
    for sd in seeds:
        path = oof_path(arm, fit, algo, shift, sd)
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        o = E41.oof_folds(algo, df, cols, params, sd, folds)
        o.to_parquet(path, index=False)
        parts.append(o)
        E55.log(f"  腕{arm} {fit} {algo} ずらし{shift}か月 種{sd}: {len(o):,}件 {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def configure() -> None:
    """実験55 の部品を実験70 の設定に差し替える（σ_dfa・腕 L0 / L5・保存名 e70_*）。"""
    E54.sigmas_from_bars = sigmas_with_dfa
    E55.ARMS.clear()
    E55.ARMS.update(ARMS)
    E55.LABELS.clear()
    E55.LABELS.update(LABELS)
    E55.PARAMS_PATH = os.path.join(OOF_DIR, f"{PREFIX}_params.json")
    E55.CSV_PREFIX = PREFIX
    E55.oof = oof70


# --------------------------------------------------------------------------- #
# 的中率の表（保存済みの out-of-fold から。学習しない）
# --------------------------------------------------------------------------- #
def load_oof(arm: str, fit: str, algo: str, shift: int, seeds: Sequence[int] = SEEDS) -> pd.DataFrame:
    """保存済みの out-of-fold を種ごとに読んで score を平均する。無ければ FileNotFoundError。"""
    parts = []
    for sd in seeds:
        p = oof_path(arm, fit, algo, shift, sd)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        parts.append(pd.read_parquet(p))
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def band_stats(rows: pd.DataFrame, big: float = BIG) -> dict:
    """選んだ行の的中率と実収益の散らばり。label = 自分のラベル、y_L0 = 現行ラベル（列があれば）。
    勝率・+10% 以上・−10% 未満は ret_o1_20 が欠測でない行の中での割合。"""
    r = pd.to_numeric(rows["ret_o1_20"], errors="coerce").dropna()
    n = int(len(rows))
    return {"n": n,
            "hit_own": float(rows["label"].mean()) if n else np.nan,
            "hit_l0": float(rows["y_L0"].mean()) if n and "y_L0" in rows else np.nan,
            "ret20": float(r.mean() * 100) if len(r) else np.nan,
            "win": float((r > 0).mean()) if len(r) else np.nan,
            "big_up": float((r >= big).mean()) if len(r) else np.nan,
            "big_down": float((r < -big).mean()) if len(r) else np.nan}


def precision_table(oofs_by_arm: Dict[str, Dict[str, pd.DataFrame]], base_arm: str = "L0",
                    selections=SELECTIONS) -> pd.DataFrame:
    """腕ごと・選定ごとに、選ばれた行の的中率（自分のラベル / 現行ラベル）と実収益の散らばりを出す。
    oofs_by_arm: {腕: {"lgbm": oof, "xgb": oof, "cat": oof}}（label は自分のラベル）。
    現行ラベルは base_arm の lgbm の label 列（全腕が同じ行の集まりなので、そのまま y_L0）。
    選び方は ops_rule.consensus（前の窓の分布の百分位。前の窓が 500 行未満の窓は飛ばす）。"""
    y0 = oofs_by_arm[base_arm]["lgbm"][["Code", "Date", "label"]].rename(columns={"label": "y_L0"})
    recs = []
    for arm, oofs in oofs_by_arm.items():
        for name, models, pct in selections:
            if any(m not in oofs for m in models):
                continue
            c = OR.consensus(oofs, pct, models=models, keep=True)
            if not c.get("n"):
                recs.append({"arm": arm, "rule": name, "n": 0})
                continue
            rows = c["rows"].merge(y0, on=["Code", "Date"], how="left")
            st = band_stats(rows)
            st.update({"arm": arm, "rule": name, "rate": float(c["rate"]),
                       "lift20": float(c["ret_o1_20"]["fold_mean"]),
                       "won": int(c["ret_o1_20"]["won"]), "folds": int(c["ret_o1_20"]["n_folds"])})
            recs.append(st)
    cols = ["arm", "rule", "n", "rate", "hit_own", "hit_l0", "ret20", "win", "big_up", "big_down",
            "lift20", "won", "folds"]
    t = pd.DataFrame(recs)
    return t.reindex(columns=[c for c in cols if c in t.columns])


def summarize_precision(t: pd.DataFrame) -> pd.DataFrame:
    """切り方をまとめる（件数で重み付けした割合と平均。窓の勝ちは合計）。"""
    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["n"].to_numpy(dtype=float)
        tot = w.sum()
        out = {"n": int(tot), "shifts": int(len(g))}
        for k in ("hit_own", "hit_l0", "ret20", "win", "big_up", "big_down"):
            v = g[k].to_numpy(dtype=float)
            ok = np.isfinite(v) & (w > 0)
            out[k] = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        out["won"] = int(g["won"].fillna(0).sum())
        out["folds"] = int(g["folds"].fillna(0).sum())
        return pd.Series(out)
    keys = [k for k in ("fit", "arm", "rule") if k in t.columns]
    recs = []
    for key, g in t.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(keys, key))
        rec.update(agg(g).to_dict())
        recs.append(rec)
    return pd.DataFrame(recs)


HEADER = (f"  {'腕':<18}{'規則':<16}{'件数':>7}{'選定率':>7}{'正例率(自分)':>12}{'正例率(現行)':>12}"
          f"{'ret20':>8}{'勝率':>7}{'+10%以上':>9}{'-10%未満':>9}{'窓平均超過':>10}{'勝ち窓':>8}")


def fmt_row(r: pd.Series, label: str) -> str:
    if not r.get("n"):
        return f"  {label:<18}{r['rule']:<16}（選定なし）"
    rate = f"{r['rate']*100:>6.1f}%" if "rate" in r and np.isfinite(r.get("rate", np.nan)) else f"{'':>7}"
    lift = f"{r['lift20']:>+9.2f}pt" if "lift20" in r and np.isfinite(r.get("lift20", np.nan)) else f"{'':>11}"
    return (f"  {label:<18}{r['rule']:<16}{int(r['n']):>7,}{rate}{r['hit_own']*100:>11.1f}%{r['hit_l0']*100:>11.1f}%"
            f"{r['ret20']:>+7.2f}%{r['win']*100:>6.0f}%{r['big_up']*100:>8.1f}%{r['big_down']*100:>8.1f}%"
            f"{lift}{int(r['won']):>5}/{int(r['folds']):<3}")


def precision_main(shifts: Sequence[int], seeds: Sequence[int] = SEEDS, fits=("B1", "B2")) -> pd.DataFrame:
    """保存済みの e70_* から、運用の選定の的中率の表を出す。"""
    print("=" * 78)
    print(f"実験70 的中率の表（保存済みの out-of-fold。ずらし {list(shifts)} / 種 {list(seeds)} / 選定 {[s[0] for s in SELECTIONS]}）")
    print("  正例率(自分) = その腕のラベルでの正例率、正例率(現行) = 現行ラベル L0 での正例率。勝率は ret_o1_20 > 0 の割合")
    print("=" * 78)
    parts = []
    for sh in shifts:
        for fit in fits:
            by_arm = {a: {"lgbm": load_oof(a, fit, "lgbm", sh, seeds),
                          "xgb": load_oof(a, "B1", "xgb", sh, seeds),
                          "cat": load_oof(a, "B1", "cat", sh, seeds)} for a in ARMS}
            t = precision_table(by_arm)
            t.insert(0, "fit", fit)
            t.insert(0, "shift", sh)
            parts.append(t)
            print(f"\n■ ずらし{sh}か月 / lgbm {fit}（{E55.FIT_JA[fit]}）+ xgb + cat")
            print(HEADER)
            for _, r in t.iterrows():
                print(fmt_row(r, LABELS[r["arm"]]))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(os.path.join(OOF_DIR, f"{PREFIX}_precision.csv"), index=False)
    s = summarize_precision(out)
    print(f"\n■ 切り方{len(shifts)}通りをまとめて（件数で重み付け。勝ち窓は合計）")
    for fit in fits:
        print(f"  ◇ lgbm {fit}（{E55.FIT_JA[fit]}）+ xgb + cat")
        print(HEADER)
        for _, r in s[s["fit"] == fit].iterrows():
            print(fmt_row(r, LABELS[r["arm"]]))
    E55.log(f"記録: {OOF_DIR}/{PREFIX}_precision.csv")
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--precision" in argv:
        ap = argparse.ArgumentParser(description="実験70 的中率の表（保存済みの out-of-fold から）")
        ap.add_argument("--precision", action="store_true")
        ap.add_argument("--shifts", default="0,2,4")
        ap.add_argument("--seeds", type=int, default=3)
        a = ap.parse_args(argv)
        precision_main([int(s) for s in a.shifts.split(",")], SEEDS[:a.seeds])
        return 0
    configure()
    if not any(a.startswith("--arms") for a in argv):
        argv += ["--arms", "L0,L5"]
    if not any(a.startswith("--consensus") for a in argv):
        argv += ["--consensus", "90,95"]
    print("=" * 78)
    print(f"実験70 σ_dfa のラベルを実験55 と同じ作りで確かめる（腕 {list(ARMS)}、保存名 {PREFIX}_*）")
    print(f"  σ_dfa = σ20^w × σ120^(1−w)、w = clip((α − {E69.A_LO}) / {E69.A_SPAN}, 0, 1)、α = {E69.ALPHA_COL}")
    print("=" * 78)
    return E55.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
