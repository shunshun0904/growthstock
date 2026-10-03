#!/usr/bin/env python3
"""
実験63: 「これは絶対外さない」を把握する。

運用者の依頼（2026-10-03）「発火条件を満たす（真陽性）のデータがどんな特徴量なのか知りたい。条件を満たすすべてではなく、
（OOF の全期間を通して）スコア最上位の5サンプル程度（もちろん真陽性）でよい」。

作り
  1. 本番と同じ OOF（5モデル・239列・本番のパラメータと前処理・36/6/6か月の窓・ずらし0・種42。実験62 の base_oof を再利用）
  2. 画面と同じ「過去分布の百分位」（その日より前の OOF 分布。実験59 の pct_expanding）を5モデルに付け、
     発火 = 全5モデル 95以上（運用者の規則）
  3. 発火した行を「5モデルの最小百分位（p_min）→ LightGBM のスコア」の順に並べる（画面の並びと同じ）。
     上位 k 件の正例率を出す（本当に「外さない」か）
  4. ラベル1（真陽性）の上位 N 件について、全239列の値と「同じ日の候補の中での百分位」を出し、
     N 件の多くが同じ向きに極端な列を並べる（= この型の共通点）
出力
  公開ログ: 件数・正例率・列名と百分位の集計だけ（銘柄・日付・生の値は出さない）
  research/_data/oof/e63_top_samples.csv: 上位の行（銘柄・日付・スコア・全列の値と百分位。artifact だけ）
  research/_data/oof/e63_profile.csv: 列ごとの集計

使い方
    python3 research/exp/e63_sure_things.py [--top 5] [--context 10]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import feature_dict as FD  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e59_unit_sim as E59  # noqa: E402
import e62_stacking as E62  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
ALGOS = tuple(L.ALL5)
RULE_PCT = 95.0
SEED = 42
TOP_N = 5
CONTEXT_N = 10
HIGH, LOW = 0.8, 0.2        # 同じ日の候補の中での百分位がこれ以上 / 以下なら「極端」
KS = (5, 10, 20, 50, 100)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- #
# 規則と並び
# ---------------------------------------------------------------- #

def add_rule(frame: pd.DataFrame, pct: float = RULE_PCT, min_hist: int = E59.MIN_HIST) -> pd.DataFrame:
    """p_<algo>（過去分布の百分位）、p_min、passed を足す。"""
    out = frame.copy()
    for a in ALGOS:
        out[f"p_{a}"] = E59.pct_expanding(out["Date"], out[f"s_{a}"].to_numpy(dtype=float), min_hist)
    out["p_min"] = out[[f"p_{a}" for a in ALGOS]].min(axis=1, skipna=False)
    out["passed"] = (out["p_min"] >= pct).fillna(False).astype(bool)
    return out


def fired_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """発火した行を p_min → LightGBM のスコアの順に並べ、rank（1 が最上位）を付ける。"""
    f = frame[frame["passed"]].sort_values(["p_min", "s_lgbm"], ascending=[False, False],
                                           kind="mergesort").reset_index(drop=True)
    f["rank"] = np.arange(1, len(f) + 1)
    return f


def precision_at(fired: pd.DataFrame, ks: Sequence[int] = KS) -> pd.DataFrame:
    rows = []
    for k in list(ks) + [len(fired)]:
        if k <= 0 or k > len(fired):
            continue
        y = fired["label"].to_numpy(dtype=float)[:k]
        rows.append({"k": int(k), "positives": int(y.sum()), "precision": float(y.mean() * 100)})
    return pd.DataFrame(rows).drop_duplicates("k")


# ---------------------------------------------------------------- #
# 特徴量の百分位と型
# ---------------------------------------------------------------- #

def within_date_pct(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """全候補（df の行）について、同じ日の候補の中での百分位（(順位 − 0.5) / 件数）。欠損は NaN。"""
    g = df.groupby("Date")
    out = {}
    n = g["Code"].transform("size").to_numpy(dtype=float)
    for c in cols:
        r = g[c].rank(method="average")
        cnt = g[c].transform("count").to_numpy(dtype=float)
        out[c] = ((r.to_numpy(dtype=float) - 0.5) / np.where(cnt > 0, cnt, np.nan))
    del n
    return pd.DataFrame(out, index=df.index)


def profile(pct_rows: pd.DataFrame, cols: Sequence[str], *, high: float = HIGH, low: float = LOW) -> pd.DataFrame:
    """
    上位 N 件の百分位から、列ごとに「高い側に極端な件数 / 低い側に極端な件数 / 中央値 / 件数」を出し、
    同じ向きにそろっている列から順に並べる。
    """
    rows = []
    for c in cols:
        p = pct_rows[c].to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        if len(p) == 0:
            continue
        nh, nl = int((p >= high).sum()), int((p <= low).sum())
        rows.append({"col": c, "n": int(len(p)), "n_high": nh, "n_low": nl, "median": float(np.median(p)),
                     "agree": max(nh, nl), "side": "高" if nh >= nl else "低"})
    out = pd.DataFrame(rows)
    out["dist"] = (out["median"] - 0.5).abs()
    return out.sort_values(["agree", "dist"], ascending=[False, False]).reset_index(drop=True)


def describe(col: str) -> str:
    try:
        d = FD.describe(col)
    except Exception:  # noqa: BLE001
        d = ""
    return (d or "").split("。")[0][:40]


def company_names(data_dir: str) -> Dict[str, str]:
    p = os.path.join(data_dir, "master.parquet")
    if not os.path.exists(p):
        return {}
    m = pd.read_parquet(p, columns=["Code", "CoName"])
    m["Code"] = m["Code"].astype(str)
    return m.drop_duplicates("Code", keep="last").set_index("Code")["CoName"].to_dict()


# ---------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験63: 全5モデル 95以上の真陽性のうち最上位の特徴量")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--context", type=int, default=CONTEXT_N)
    ap.add_argument("--data-dir", default=lab.DATA_DIR)
    args = ap.parse_args(argv)
    os.makedirs(OOF_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    cols = F.columns(F.DEFAULT_PRESET)
    print("=" * 78)
    print("実験63 「絶対外さない」の型: 全5モデル 95以上を満たした真陽性の最上位の特徴量")
    print(f"  列: {F.DEFAULT_PRESET} {len(cols)}列 / OOF: 本番と同じ作り（ずらし0・種{SEED}）/ 規則: 全5モデル {RULE_PCT:.0f}以上")
    print("=" * 78)

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    log(f"データ {len(df):,}件 / 正例率 {df['label'].mean() * 100:.2f}% / {df['Date'].min().date()} 〜 {df['Date'].max().date()}")

    folds = E41.folds_for(df["Date"], 0)
    oofs = {a: E62.base_oof(a, df, cols, 0, SEED, folds) for a in ALGOS}
    frame = add_rule(E62.merge_base(oofs))
    fired = fired_ranking(frame)
    n_tp = int(fired["label"].sum())
    print(f"\n■ 1. 発火（全5モデル {RULE_PCT:.0f}以上）: {len(fired):,}件 / OOF {len(frame):,}件 / 真陽性 {n_tp}件 "
          f"（正例率 {fired['label'].mean() * 100:.1f}%、母集団 {frame['label'].mean() * 100:.1f}%）")
    print("  並び: 5モデルの最小百分位 → LightGBM のスコア（画面と同じ）。上位 k 件の正例率:")
    pk = precision_at(fired)
    for _, r in pk.iterrows():
        print(f"    上位 {int(r['k']):>4}件: 正例 {int(r['positives']):>3} / 正例率 {r['precision']:>5.1f}%")
    # 上位 k 件の中で「外した」行の順位（件数と順位だけ）
    miss = fired[fired["label"] == 0]["rank"].head(10).tolist()
    print(f"  外した行の順位（上位から10件）: {miss}")

    tp = fired[fired["label"] == 1].head(max(args.top, args.context)).copy()
    tp["tp_rank"] = np.arange(1, len(tp) + 1)
    top = tp.head(args.top)
    # 全候補の列ごとの日内百分位 → 上位の行だけ取り出す
    feat = df.set_index(["Code", "Date"])
    pct_all = within_date_pct(df, cols)
    pct_all.index = feat.index
    key = list(zip(top["Code"], top["Date"]))
    pct_top = pct_all.loc[key]
    raw_top = feat.loc[key, cols]
    key_ctx = list(zip(tp["Code"], tp["Date"]))
    pct_ctx = pct_all.loc[key_ctx]
    # 比較用: 発火した全行、発火した真陽性全部
    key_f = list(zip(fired["Code"], fired["Date"]))
    pct_fired = pct_all.loc[key_f]
    pct_fired_tp = pct_all.loc[[k for k, y in zip(key_f, fired["label"]) if y == 1]]

    prof = profile(pct_top, cols)
    prof["median_context"] = prof["col"].map(pct_ctx.median())
    prof["median_fired"] = prof["col"].map(pct_fired.median())
    prof["median_fired_tp"] = prof["col"].map(pct_fired_tp.median())
    prof["desc"] = prof["col"].map(describe)
    prof.to_csv(os.path.join(OOF_DIR, "e63_profile.csv"), index=False)

    print(f"\n■ 2. 真陽性の上位 {len(top)}件が同じ向きにそろっている列（同じ日の候補の中での百分位。0.5 = 真ん中）")
    print(f"  {'列':<28}{'向き':>4}{'そろい':>7}{'上位の中央値':>12}{'上位10の中央値':>14}{'発火全体':>9}{'発火の正例':>10}  意味")
    shown = prof[prof["agree"] >= max(3, int(np.ceil(len(top) * 0.8)))].head(40)
    for _, r in shown.iterrows():
        print(f"  {r['col']:<28}{r['side']:>4}{r['agree']:>4}/{r['n']:<2}{r['median']:>12.2f}{r['median_context']:>14.2f}"
              f"{r['median_fired']:>9.2f}{r['median_fired_tp']:>10.2f}  {r['desc']}")
    print("  （「そろい」は上位の件数のうち百分位が 0.8 以上（高）または 0.2 以下（低）だった件数）")

    # 私的な出力: 行ごとの中身（銘柄・日付・スコア・百分位・全列）
    names = company_names(args.data_dir)
    out = tp[["tp_rank", "rank", "Code", "Date", "label", "ret_o1_20", "p_min"] + [f"p_{a}" for a in ALGOS]
             + [f"s_{a}" for a in ALGOS]].copy()
    out["name"] = out["Code"].map(names).fillna("")
    raw_ctx = feat.loc[key_ctx, cols].reset_index(drop=True)
    pct_ctx_r = pct_ctx.reset_index(drop=True).add_suffix("__pct")
    out = pd.concat([out.reset_index(drop=True), raw_ctx, pct_ctx_r], axis=1)
    out.to_csv(os.path.join(OOF_DIR, "e63_top_samples.csv"), index=False)
    print(f"\n  上位 {len(tp)}件の中身（銘柄・日付・全列の値と百分位）は e63_top_samples.csv（artifact）に書いた。公開ログには出さない")
    log(f"記録: {OOF_DIR}/e63_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
