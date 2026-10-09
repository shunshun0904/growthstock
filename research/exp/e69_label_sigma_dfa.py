#!/usr/bin/env python3
"""
実験69: ラベルの σ を |r| の持続性（nl_dfa_abs120）で補正する。

運用者の依頼（2026-10-09）「ラベル側の σ を nl_dfa_abs120 で補正する案」。確認（AskUserQuestion）で
「σ20 と σ120 の幾何混合、重みは α」「検証は実験54 と同じ」を選択。

考え方（docs/MODEL_ADOPTION_RULES.md §13 の補足）
  ラベルの到達しきい値は 1.2 × σ20 × √20（直近20日の日次リターンの標準偏差）。ボラは平均回帰するので、
  ブレイク後20営業日の実現ボラ ÷ σ20 は低ボラの帯で中央値 1.38、高ボラの帯で 0.72。前の20日の σ でしきい値を
  作ると、高ボラ銘柄には実態より厳しく、低ボラ銘柄には緩い上昇を要求している。
  |r| の DFA 指数 α（nl_dfa_abs120）はボラの塊の持続性。α が低い銘柄はボラが長い窓の水準に戻りやすく、
  α が高い銘柄は直近の水準が続きやすい。そこで

    σ_dfa = σ20^w × σ120^(1−w)、 w = clip((α − 0.5) / 0.3, 0, 1)

  α ≥ 0.8 なら σ20 のまま、α ≤ 0.5 なら σ120、間は幾何平均で混ぜる。定数（0.5 / 0.3）はデータから決めず固定
  （α の中央値 0.64、p5〜p95 0.44〜0.88）。α が無い行は w = 0.5、σ120 が無い行は σ20。

腕（ラベルだけ違う。学習は本番の分類器・本番の239列・本番の木の形・LightGBM。実験54 と同じ）
  L0   1.2 × σ20（現行）
  L5   1.2 × σ_dfa
  L5m  k × σ_dfa、k は正例率が L0 と同じになるように決める

前提の確認（しきい値を作る前に）
  ブレイク後20営業日の実現ボラ σ_fwd（t+1〜t+20 の日次リターンの標準偏差）を、σ20 / σ120 / σ_dfa の
  どれがよく当てるか: log の相関、|log(σ_fwd/σ)| の中央値、ボラの帯ごとの σ_fwd/σ の中央値。
  σ_dfa が σ20 より当たらなければ、ラベルを変える根拠が無い。

評価は実験54 と同じ（E54.evaluate / summarize）: 閾値ルール（前の窓の上位5% / 10%）、発火数を L0 に
そろえる、窓の中の上位10%、日付内、上位10% のボラの帯。3切り方 × 種3つ。ラベルが違うと正例の集合が
違うので、PR-AUC は参考にとどめ、採否は実収益で読む（§16・§18 と同じ）。

  python3 research/exp/e69_label_sigma_dfa.py [--shifts 0,2,4] [--seeds 3] [--arms L0,L5,L5m]
  結果は research/_data/oof/e69_*。本番の設定には書かない。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import nonlinear_features as NL  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e52_return_objective as E52  # noqa: E402
import e54_label_sigma as E54  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PREFIX = "e69"
ALPHA_COL = "nl_dfa_abs120"
#: w = clip((α − A_LO) / A_SPAN, 0, 1)。結果を見る前に決めた定数
A_LO, A_SPAN = 0.5, 0.3
ARMS: Dict[str, tuple] = {"L0": ("sigma20", False), "L5": ("sigma_dfa", False), "L5m": ("sigma_dfa", True)}
LABELS = {"L0": "L0 1.2σ20（現行）", "L5": "L5 1.2σ_dfa", "L5m": "L5m kσ_dfa（正例率そろえ）"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def weight(alpha) -> pd.Series:
    """α → σ20 の重み w。α が無ければ 0.5。"""
    a = pd.to_numeric(pd.Series(alpha), errors="coerce")
    w = ((a - A_LO) / A_SPAN).clip(0.0, 1.0)
    return w.fillna(0.5)


def sigma_dfa(sigma20, sigma120, alpha) -> pd.Series:
    """σ20^w × σ120^(1−w)。σ120 が無ければ σ20。σ20 が無ければ欠測。"""
    s20 = pd.to_numeric(pd.Series(sigma20), errors="coerce")
    s120 = pd.to_numeric(pd.Series(sigma120), errors="coerce")
    w = weight(alpha)
    w.index = s20.index
    s120 = s120.where(s120 > 0)
    out = np.exp(w * np.log(s20.where(s20 > 0)) + (1 - w) * np.log(s120))
    out = out.where(s120.notna(), s20)
    return out.where(s20.notna())


def forward_sigma(sg: pd.DataFrame, h: int = B.RISE_HORIZON) -> pd.Series:
    """t+1〜t+h の日次リターンの標準偏差 = t+h 時点の σ20（h=20 のとき）。bars の並び（Code, Date）で h 行先。"""
    assert h == 20, "σ_fwd は 20日の rolling std を h 行先に取る作り（h=20 のみ）"
    return sg.groupby("Code", sort=False)["sigma20"].shift(-h)


def premise(df: pd.DataFrame) -> pd.DataFrame:
    """σ_fwd をどの σ がよく当てるか。"""
    rows = []
    fwd = np.log(pd.to_numeric(df["sigma_fwd"], errors="coerce"))
    band = pd.cut(df["vol_20d"], E52.VOL_EDGES, labels=E52.VOL_NAMES, include_lowest=True)
    for c in ("sigma20", "sigma60", "sigma120", "sigma_dfa"):
        x = np.log(pd.to_numeric(df[c], errors="coerce"))
        ok = fwd.notna() & x.notna()
        ratio = np.exp(fwd - x)
        rec = {"sigma": c, "n": int(ok.sum()), "corr_log": float(np.corrcoef(fwd[ok], x[ok])[0, 1]),
               "mad_log": float((fwd - x).abs()[ok].median()),
               "ratio_med": float(ratio[ok].median())}
        for name, g in ratio[ok].groupby(band[ok], observed=False):
            rec[f"ratio_{name}"] = float(g.median())
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験69: ラベルの σ を nl_dfa_abs120 で補正する")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    arms = [a for a in args.arms.split(",") if a]
    if "L0" not in arms or any(a not in ARMS for a in arms):
        raise SystemExit(f"腕は {list(ARMS)} から。L0 は必須")

    cols = F.columns(F.DEFAULT_PRESET)
    df = lab.frame()
    df = df[df["label"].notna()].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    os.makedirs(OOF_DIR, exist_ok=True)
    print("=" * 78)
    print(f"実験69 ラベルの σ を |r| の持続性で補正（{len(df):,}件 / {F.DEFAULT_PRESET} {len(cols)}列 / "
          f"種{len(seeds)}つ / ずらし {shifts}か月）")
    print(f"  σ_dfa = σ20^w × σ120^(1−w)、w = clip((α − {A_LO}) / {A_SPAN}, 0, 1)、α = {ALPHA_COL}")
    print("=" * 78)

    t0 = time.time()
    sg = E54.sigmas_from_bars(E54.load_bars())
    sg["sigma_fwd"] = forward_sigma(sg)
    df = df.merge(sg, on=["Code", "Date"], how="left")
    if not os.path.exists(NL.OUT_PATH):
        raise SystemExit(f"{NL.OUT_PATH} がありません。python3 research/nonlinear_features.py で作ってください")
    nl = pd.read_parquet(NL.OUT_PATH, columns=["Code", "Date", ALPHA_COL])
    nl["Code"] = nl["Code"].astype(str)
    nl["Date"] = pd.to_datetime(nl["Date"])
    df = df.merge(nl, on=["Code", "Date"], how="left")
    df["w_dfa"] = weight(df[ALPHA_COL]).to_numpy()
    df["sigma_dfa"] = sigma_dfa(df["sigma20"], df["sigma120"], df[ALPHA_COL]).to_numpy()
    n_mismatch = int(((df["sigma20"] - df["vol_20d"]).abs() > 1e-6).sum())
    log(f"σ {time.time()-t0:.0f}秒 / σ20 と vol_20d の差 > 1e-6: {n_mismatch}件 / σ120 欠測 "
        f"{df['sigma120'].isna().mean()*100:.2f}% / α 欠測 {df[ALPHA_COL].isna().mean()*100:.2f}% / "
        f"σ_fwd 欠測 {df['sigma_fwd'].isna().mean()*100:.2f}%")
    if n_mismatch:
        raise SystemExit("bars から作った σ20 がデータセットの vol_20d と一致しません")
    y0 = E54.relabel(df, E54.need_from_sigma(df["sigma20"], E54.K0))
    diff = int((y0 != df["label"]).sum())
    if diff:
        raise SystemExit(f"引き直した L0 が本番のラベルと {diff}件違います")
    log(f"L0 の引き直しは本番のラベルと全 {len(df):,}件一致")

    print(f"\n■ 重み w（σ20 側）の分布: 平均 {df['w_dfa'].mean():.2f} / w=0（σ120 のまま）{(df['w_dfa'] == 0).mean()*100:.1f}% / "
          f"w=1（σ20 のまま）{(df['w_dfa'] == 1).mean()*100:.1f}% / 中間 {((df['w_dfa'] > 0) & (df['w_dfa'] < 1)).mean()*100:.1f}%")
    r = df["sigma_dfa"] / df["sigma20"]
    print(f"  σ_dfa ÷ σ20: 中央値 {r.median():.3f} / p10 {r.quantile(0.1):.3f} / p90 {r.quantile(0.9):.3f}")

    pr = premise(df)
    pr.to_csv(os.path.join(OOF_DIR, f"{PREFIX}_premise.csv"), index=False)
    print("\n■ 前提: ブレイク後20営業日の実現ボラ σ_fwd をどの σ が当てるか（log の相関 / |log 比| の中央値 / "
          "σ_fwd÷σ の中央値、ボラの帯（vol_20d 低→高）ごと）")
    print(f"  {'σ':<10}{'n':>7}{'相関':>7}{'|log比|':>8}{'比の中央値':>10}  " + "  ".join(f"{n:>7}" for n in E52.VOL_NAMES))
    for _, q in pr.iterrows():
        print(f"  {q['sigma']:<10}{int(q['n']):>7}{q['corr_log']:>7.3f}{q['mad_log']:>8.3f}{q['ratio_med']:>10.3f}  "
              + "  ".join(f"{q[f'ratio_{n}']:>7.3f}" for n in E52.VOL_NAMES))
    print("  比が 1 に近く帯の間で平らなほど、しきい値の σ が先の実態に合っている")

    # 実験54 の作りでラベルを引き直して評価する（腕の定義だけ差し替える）
    E54.ARMS.update(ARMS)
    E54.LABELS.update(LABELS)
    df, tab = E54.build_labels(df, arms)
    df = E54.report_labels(df, tab, arms, labels=LABELS, prefix=PREFIX)
    ed, rows = E54.evaluate(df, arms, cols, shifts, seeds, labels=LABELS, prefix=PREFIX)
    E54.summarize(ed, rows, arms, shifts, labels=LABELS)
    log(f"記録: {OOF_DIR}/{PREFIX}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
