#!/usr/bin/env python3
"""
実験54: ラベルの σ（到達しきい値 = k × σ × √20）を見直す。

実験50 の補足（docs/MODEL_ADOPTION_RULES.md §13）: ブレイク後20営業日の実現ボラ ÷ 直前の
vol_20d は、最も低いボラの帯で中央値 1.38、最も高い帯で 0.72。ボラは平均回帰するので、
「前の20日の σ」でしきい値を作ると、高ボラ銘柄には実態より厳しい上昇を、低ボラ銘柄には
緩い上昇を要求している。運用者の承認（2026-09-26）で、σ の作り方を変えたラベルを比べる。

腕（ラベルだけ違う。学習は本番の分類器・本番の206列・本番の木の形・LightGBM）
  L0   現行: 1.2 × σ20（直近20日の日次リターン標準偏差）
  L1   1.2 × σ60（直近60日）         … 平均回帰の影響を薄める（窓を長く）
  L1m  k × σ60、k は正例率が L0 と同じになるように決める
  L2   1.2 × √(σ20 × その日の全銘柄の σ20 の中央値） … 半分だけ横断面に寄せる（縮約）
  L2m  k × 同上、k は正例率をそろえる
  L3m  固定 +X%、X は正例率をそろえる … ボラ依存を完全に外した極端側
終盤の条件（t+20 の5日平均が しきい値の 0.5倍 以上）と MA5>=MA20 は現行のまま。
しきい値の比例関係（END_RATIO / RISE_THRESHOLD）も現行のまま。

ラベルはデータセットに残っている future_rise / end_level / uptrend_end から引き直す
（build_dataset.attach_rise_label と同じ合成）。L0 を引き直した結果が本番のラベルと全行一致する
ことを最初に検査し、違えば止まる。σ60 と日付ごとの中央値は bars_*.parquet から price_panel と
同じ式（AdjC を C で埋めた終値の日次リターン）で作る。

評価（実収益。ラベルが違うと正例の集合が違うので、PR-AUC は参考にとどめる）
  1. 閾値ルール（前の窓のスコア分布の上位5% / 10%）: 件数・平均収益・全体との差・窓ごと・−10% 未満
  2. 発火数を L0 にそろえる（窓ごとに L0 が買った件数を、その腕のスコアの上から取る）
  3. 窓の中の上位10%（件数をそろえる）と L0 との差（窓ごと）
  4. 日付内: 1位・上位2件の平均収益、順位相関
  5. 上位10% の中身（ボラの帯）と、ラベルそのものの帯ごとの正例率（実験50 の 9.3% 対 24.3% が縮むか）
どれも本番と同じ窓（36/6/6か月・エンバーゴ20営業日）を 0/2/4か月ずらした3通り × 種3つ。
全腕でラベルが確定している行だけで比べる（σ60 は上場後45営業日から）。

  python3 research/exp/e54_label_sigma.py [--shifts 0,2,4] [--seeds 3] [--arms L0,L1,L1m,L2,L2m,L3m]
  結果は research/_data/oof/e54_*。本番の設定には書かない。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e52_return_objective as E52  # noqa: E402
import e53_objective_tuning as E53  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
OUTCOME = lab.OUTCOME
HORIZON = B.RISE_HORIZON
K0 = float(B.VOL_NORM_K)
#: 終盤の必要水準はしきい値の何倍か（本番: 0.10 / 0.20）
END_MULT = float(B.END_RATIO / B.RISE_THRESHOLD)
#: 腕 → (σ の列, k を正例率でそろえるか)。σ の列が None なら固定%
ARMS: Dict[str, Tuple[str | None, bool]] = {
    "L0": ("sigma20", False), "L1": ("sigma60", False), "L1m": ("sigma60", True),
    "L2": ("sigma_shrink", False), "L2m": ("sigma_shrink", True), "L3m": (None, True)}
LABELS = {"L0": "L0 1.2σ20（現行）", "L1": "L1 1.2σ60", "L1m": "L1m kσ60（正例率そろえ）",
          "L2": "L2 1.2√(σ20·中央値)", "L2m": "L2m k√(σ20·中央値)（そろえ）", "L3m": "L3m 固定%（そろえ）"}
VOL_WINDOWS = (20, 60, 120)
#: 窓 n のσは n×0.75 本から作る（price_panel の 20/15 と同じ比率）。120日は実験55 が使う
MIN_PERIODS = {20: 15, 60: 45, 120: 90}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# σ の候補
# --------------------------------------------------------------------------- #

def sigmas_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """
    (Code, Date) ごとの σ20 / σ60（%）と、その日の全銘柄の σ20 の中央値。
    price_panel と同じ式（AdjC を C で埋めた終値、銘柄ごとの日次リターン、rolling std × 100）。
    """
    df = bars[["Code", "Date", "C"] + (["AdjC"] if "AdjC" in bars.columns else [])].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)
    close = df["AdjC"].fillna(df["C"]) if "AdjC" in df.columns else df["C"]
    close = pd.to_numeric(close, errors="coerce")
    g = close.groupby(df["Code"], sort=False)
    ret1 = close / g.shift(1) - 1.0
    out = df[["Code", "Date"]].copy()
    for n in VOL_WINDOWS:
        out[f"sigma{n}"] = ret1.groupby(df["Code"], sort=False).transform(
            lambda s, n=n: s.rolling(n, min_periods=MIN_PERIODS[n]).std()) * 100.0
    med = out.groupby("Date")["sigma20"].median().rename("sigma20_med")
    out = out.merge(med, left_on="Date", right_index=True, how="left")
    out["sigma_shrink"] = np.sqrt(out["sigma20"] * out["sigma20_med"])
    return out


def load_bars() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


# --------------------------------------------------------------------------- #
# ラベルの引き直し
# --------------------------------------------------------------------------- #

def relabel(df: pd.DataFrame, need: pd.Series) -> pd.Series:
    """
    到達しきい値 need（行ごと。比率）でラベルを引き直す。attach_rise_label の合成と同じ:
    到達 & 終盤（END_MULT 倍）& MA5>=MA20。判定できない行（元のラベルが未確定、need が欠測）は NaN。
    """
    need = pd.to_numeric(need, errors="coerce")
    ok = ((df["future_rise"] >= need) & (df["end_level"] >= END_MULT * need)
          & (df["uptrend_end"] == 1.0))
    determined = df["label"].notna() & need.notna()
    return ok.astype(float).where(determined)


def need_from_sigma(sigma: pd.Series, k: float) -> pd.Series:
    """k × σ(%) × √horizon（build_dataset.rise_thresholds と同じ式）。"""
    return k * pd.to_numeric(sigma, errors="coerce") / 100.0 * np.sqrt(HORIZON)


def match_rate(df: pd.DataFrame, sigma: pd.Series | None, target: float,
               lo: float = 0.02, hi: float = 6.0, tol: float = 1e-4) -> float:
    """
    正例率が target になる k（σ あり）または固定しきい値 X（σ なし）を二分探索で求める。
    正例率は k / X に単調減少。
    """
    def rate(v: float) -> float:
        need = need_from_sigma(sigma, v) if sigma is not None else pd.Series(v, index=df.index)
        y = relabel(df, need)
        return float(y.mean())

    if not (rate(hi) <= target <= rate(lo)):
        raise SystemExit(f"正例率 {target:.4f} を [{lo}, {hi}] の範囲で作れない "
                         f"（{rate(lo):.4f} 〜 {rate(hi):.4f}）")
    for _ in range(60):
        mid = (lo + hi) / 2
        if rate(mid) > target:
            lo = mid
        else:
            hi = mid
        if abs(rate(mid) - target) < tol:
            break
    return (lo + hi) / 2


def build_labels(df: pd.DataFrame, arms: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """各腕のラベル列 y_<腕> を付け、腕ごとの (k または X, 正例率, 確定件数) を返す。"""
    target = float(df["label"].mean())
    rows = []
    for a in arms:
        col, matched = ARMS[a]
        if col is None:
            v = match_rate(df, None, target)
            need = pd.Series(v, index=df.index)
        else:
            v = match_rate(df, df[col], target) if matched else K0
            need = need_from_sigma(df[col], v)
        df[f"y_{a}"] = relabel(df, need)
        df[f"need_{a}"] = need
        rows.append({"arm": a, "k_or_x": v, "pos_rate": float(df[f"y_{a}"].mean()),
                     "n": int(df[f"y_{a}"].notna().sum()),
                     "need_median": float(need[df[f"y_{a}"].notna()].median())})
    return df, pd.DataFrame(rows)


def pos_by_vol(y: pd.Series, vol: pd.Series) -> pd.Series:
    band = pd.cut(vol, E52.VOL_EDGES, labels=E52.VOL_NAMES, include_lowest=True)
    return y.groupby(band, observed=False).mean()


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験54: ラベルの σ を見直す")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"知らない腕: {bad}（使えるもの {list(ARMS)}）")
    if "L0" not in arms:
        raise SystemExit("L0（現行）は比べる基準なので外せません")

    cols = F.columns(F.DEFAULT_PRESET)
    df = lab.frame()
    miss = [c for c in cols + ["future_rise", "end_level", "uptrend_end", "vol_20d"] if c not in df.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss[:6]}。research/build_dataset.py を回し直してください")
    df = df[df["label"].notna()].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    os.makedirs(OOF_DIR, exist_ok=True)
    print("=" * 78)
    print(f"実験54 ラベルの σ を見直す（{len(df):,}件 / {F.DEFAULT_PRESET} {len(cols)}列 / "
          f"種{len(seeds)}つ / ずらし {shifts}か月）")
    print(f"  現行: {B.DEFAULT_RISE.name} / 終盤はしきい値の {END_MULT:.2f}倍 / 正例率 {df['label'].mean()*100:.2f}%")
    print("=" * 78)

    # σ の候補を bars から作り、σ20 がデータセットの vol_20d と一致することを確かめる
    t0 = time.time()
    sg = sigmas_from_bars(load_bars())
    df = df.merge(sg, on=["Code", "Date"], how="left")
    d20 = (df["sigma20"] - df["vol_20d"]).abs()
    n_mismatch = int((d20 > 1e-6).sum())
    log(f"σ の候補 {time.time()-t0:.0f}秒 / σ20 と vol_20d の差 > 1e-6: {n_mismatch}件 / "
        f"σ60 欠測 {df['sigma60'].isna().mean()*100:.2f}% / 中央値の欠測 {df['sigma20_med'].isna().mean()*100:.2f}%")
    if n_mismatch:
        raise SystemExit("bars から作った σ20 がデータセットの vol_20d と一致しません。式がずれています")

    # L0 を引き直して本番のラベルと一致することを検査（合成の式が同じであることの確認）
    y0 = relabel(df, need_from_sigma(df["sigma20"], K0))
    diff = int((y0 != df["label"]).sum())
    if diff:
        raise SystemExit(f"引き直した L0 が本番のラベルと {diff}件違います。合成の式がずれています")
    log(f"L0 の引き直しは本番のラベルと全 {len(df):,}件一致")

    df, tab = build_labels(df, arms)
    ycols = [f"y_{a}" for a in arms]
    keep = df[ycols].notna().all(axis=1)
    df = df[keep].reset_index(drop=True)
    print(f"\n■ ラベル（全腕で確定している {len(df):,}件。正例率の目標 {tab.loc[0, 'pos_rate']*100:.2f}%）")
    print(f"  {'腕':<30}{'k / X':>8}{'正例率':>8}{'しきい値の中央値':>12}  ボラの帯ごとの正例率（低→高）  L0 と違う行")
    for _, r in tab.iterrows():
        a = r["arm"]
        pv = pos_by_vol(df[f"y_{a}"], df["vol_20d"])
        ch = float((df[f"y_{a}"] != df["y_L0"]).mean())
        print(f"  {LABELS[a]:<30}{r['k_or_x']:>8.3f}{df[f'y_{a}'].mean()*100:>7.2f}%"
              f"{df[f'need_{a}'].median()*100:>+11.1f}%  "
              + " / ".join(f"{v*100:.1f}%" for v in pv) + f"  {ch*100:>5.1f}%")
    tab.to_csv(os.path.join(OOF_DIR, "e54_labels.csv"), index=False)

    # 各腕の正例の実収益（中央値）: しきい値がボラに比例しなくなると、正例の中身がどう変わるか
    print("  正例の実収益（ret_o1_20）の中央値 / 平均: " + " / ".join(
        f"{a} {df.loc[df[f'y_{a}'] == 1, OUTCOME].median()*100:+.1f}% "
        f"{df.loc[df[f'y_{a}'] == 1, OUTCOME].mean()*100:+.1f}%" for a in arms))

    edges, rows = [], []
    for sh in shifts:
        res = {}
        for a in arms:
            d = df.copy()
            d["label"] = d[f"y_{a}"]
            o = E52.oof_arm(d, cols, "C", sh, seeds, "", name=a, prefix="e54")
            # 評価の正例率は現行のラベル（L0）でそろえて見る
            o = o.drop(columns=["label"]).merge(df[["Code", "Date", "y_L0"]], on=["Code", "Date"],
                                                 how="left").rename(columns={"y_L0": "label"})
            res[a] = o
        print(f"\n■ ずらし{sh}か月")
        for pct in (95, 90):
            print(f"  ◆ 上位{100-pct}%（前の窓の分布の閾値）: 件数 / 平均 / 全体との差 / "
                  f"窓ごとの差 ± SE（上の窓 / 最悪） / −10%未満")
            for a in arms:
                e = lab.threshold_edge(res[a], pct=pct, outcome=OUTCOME)
                pf = E52._per_fold(res[a], pct, within=False)
                bad = float((pf["bad"] * pf["n"]).sum() / pf["n"].sum()) if pf["n"].sum() else np.nan
                se = e["thr_fold_sd"] / np.sqrt(max(1, e["thr_folds"]))
                print(f"    {LABELS[a]:<30}{e['thr_n']:>6}件 {e['thr_end']:>+7.2f}% {e['thr_lift']:>+7.2f}pt "
                      f" {e['thr_fold_mean']:>+6.2f} ± {se:.2f}（{e['thr_folds_won']}/{e['thr_folds']} / "
                      f"{e['thr_worst']:+.2f}） {bad*100:>5.1f}%")
                edges.append({"shift": sh, "arm": a, "kind": f"thr{pct}", "bad": bad, **e})
            counts = E53.threshold_counts(res["L0"], pct)
            base = E53.matched_picks(res["L0"], counts)
            print(f"  ◆ 発火数を L0 にそろえる（L0 の上位{100-pct}% と同じ件数を各窓で上から取る）: "
                  f"件数 / 平均収益 / −10%未満 / 正例率(L0) ｜ L0 との差（窓ごと、pt）")
            for a in arms:
                cur = E53.matched_picks(res[a], counts)
                n = int(cur["n"].sum())
                ret = float((cur["ret"] * cur["n"]).sum() / n) if n else np.nan
                bd = float((cur["bad"] * cur["n"]).sum() / n) if n else np.nan
                pos = float((cur["pos"] * cur["n"]).sum() / n) if n else np.nan
                line = f"    {LABELS[a]:<30}{n:>6}件 {ret*100:>+7.2f}% {bd*100:>5.1f}% {pos*100:>4.0f}%"
                if a != "L0":
                    line += E53._diff_line(base, cur)
                print(line)
                edges.append({"shift": sh, "arm": a, "kind": f"match{pct}", "bad": bd,
                              "thr_n": n, "thr_end": ret * 100, "thr_fold_mean": cur["ret"].mean() * 100})
        print("  ◆ 窓の中の上位10%（件数をそろえる）: 平均収益 / −10%未満 / 正例率(L0) ｜ L0 との差（窓ごと、pt）")
        base = E52._per_fold(res["L0"], 90, within=True)
        for a in arms:
            cur = E52._per_fold(res[a], 90, within=True)
            line = (f"    {LABELS[a]:<30}{cur['ret'].mean()*100:>+6.2f}% {cur['bad'].mean()*100:>5.1f}% "
                    f"{cur['pos'].mean()*100:>4.0f}%")
            if a != "L0":
                line += E53._diff_line(base, cur)
            print(line)
            edges.append({"shift": sh, "arm": a, "kind": "top10_within", "bad": cur["bad"].mean(),
                          "thr_fold_mean": cur["ret"].mean() * 100, "thr_n": int(cur["n"].sum())})
        print(f"  ◆ 日付内（発火{E52.MIN_BREAKS}件以上の日）: 1位の平均収益 / 上位2件の平均 / 順位相関 / 日数"
              " ｜ 参考 PR-AUC / ROC-AUC（現行ラベル L0 で測る）")
        for a in arms:
            w = E52.within_date_metrics(res[a])
            m = E52.label_metrics(res[a])
            print(f"    {LABELS[a]:<30}{w['top1']:>+7.2f}% {w['top2']:>+7.2f}% {w['rho']:>+6.3f} "
                  f"{w['days']:>5}日 ｜ {m['pr']:.4f} {m['roc']:.4f}")
            rows.append({"shift": sh, "arm": a, **w, **m})
        print("  ◆ 上位10% の中身（ボラの帯ごと: 割合 / 平均収益 / 正例率(L0)）")
        for a in arms:
            t = E52.picks_by_vol(res[a], 90)
            if len(t):
                print(f"    {LABELS[a]:<30}" + " | ".join(
                    f"{i} {r['share']*100:.0f}% {r['ret']*100:+.1f}% {r['pos']*100:.0f}%"
                    for i, r in t.iterrows()))

    ed = pd.DataFrame(edges)
    ed.to_csv(os.path.join(OOF_DIR, "e54_edges.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e54_within_date.csv"), index=False)
    if len(shifts) > 1 and len(ed):
        print(f"\n■ 切り方{len(shifts)}通りをまとめて（切り方ごと / 平均）")
        for kind, ja in (("thr95", "上位5%（閾値）全体との差 pt"), ("thr90", "上位10%（閾値）全体との差 pt")):
            print(f"  {ja} / −10%未満")
            for a in arms:
                g = ed[(ed["arm"] == a) & (ed["kind"] == kind)]
                print(f"    {LABELS[a]:<30}" + " / ".join(f"{v:+.2f}" for v in g["thr_lift"])
                      + f"（平均 {g['thr_lift'].mean():+.2f}）/ {g['bad'].mean()*100:.1f}%")
        for kind, ja in (("match95", "発火数を L0 の上位5% にそろえた平均収益"),
                         ("match90", "発火数を L0 の上位10% にそろえた平均収益"),
                         ("top10_within", "窓の中の上位10%（件数をそろえる）の平均収益")):
            g = ed[ed["kind"] == kind]
            print(f"  {ja} / −10%未満")
            for a in arms:
                h = g[g["arm"] == a]
                print(f"    {LABELS[a]:<30}" + " / ".join(f"{v:+.2f}%" for v in h["thr_fold_mean"])
                      + f"（平均 {h['thr_fold_mean'].mean():+.2f}%）/ {h['bad'].mean()*100:.1f}%")
        r = pd.DataFrame(rows)
        print("  日付内の1位の平均収益（切り方の平均）: " + " / ".join(
            f"{LABELS[a]} {r[r['arm'] == a]['top1'].mean():+.2f}%" for a in arms))
    log(f"記録: {OOF_DIR}/e54_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
