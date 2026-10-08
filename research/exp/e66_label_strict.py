#!/usr/bin/env python3
"""
実験66: ラベルを「40営業日で +30%」のように厳しく・長くしたら、選定の実収益は上がるか。

運用者の問い（2026-10-08）
  「40営業日後に +30% 到達（今よりラベルをシビアにする）とかでモデルを構築した方がよさそうでしょうか？」

何を測るか
  1. ラベルそのものの性質。正例率（全体・年ごと・窓ごとの正例数）と、ボラの帯ごとの正例率。
     固定 +X% のしきい値は、日次ボラが高い銘柄ほど届きやすい（実験50 §13 の裏返し）。
  2. 今の本番モデル（Release の5モデル OOF）が、新しいラベルをどれだけ当てているか。
     学習し直さなくても、今の上位が +30% 銘柄を含んでいるなら、ラベルを変える利得は小さい。
  3. 学習し直し。LightGBM（本番の木の形 = research/model/meta.json の params、種3つの平均）を、
     現行ラベルと新ラベルで **同じ行・同じ窓** で学習し、out-of-fold を同じ物差しで比べる。
       - 閾値ルール（前の窓のスコア分布の上位 5% / 10% なら買う。lab.threshold_edge）の
         ret_o1_20 / ret_o1_40 の平均と全体との差、窓ごとの勝ち負け、−10% 未満の割合
       - 窓の中の上位10%（件数をそろえる）の ret_o1_20 / ret_o1_40、+30%/40日 の到達率、ボラの帯
       - 日付内1位（その日の候補が5件以上の日）の ret_o1_20 / ret_o1_40
       - 参考: 自分のラベル・現行ラベル・+30%/40 ラベルそれぞれに対する ROC-AUC
     窓はどのラベルも確定している必要があるので、エンバーゴは最長の 40営業日にそろえる
     （現行ラベル側にやや不利 = 保守側。実験18 と同じ細工）。

腕（ラベル）。到達はどれも **翌営業日の寄り（AdjO[t+1]）** を分母に、終値の最大で測る（現行と同じ）。
  L0     現行: 1ヶ月内+1.2σ / 終盤+0.50倍 / MA5>=MA20 / 翌営業日寄り基準（データセットの label）
  L20_40 40営業日以内に終値が買値の +20% 以上（終盤・トレンドの条件なし）
  L30_40 40営業日以内に +30% 以上（運用者の案）
  L30_60 60営業日以内に +30% 以上（性質の表と、今のモデルの当て方だけ。学習はしない）

使い方
    python3 research/exp/e66_label_strict.py [--seeds 42,7,123] [--arms L0,L20_40,L30_40]
結果は research/_data/oof/e66_*。**本番の設定には書かない。**
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import walkforward as WF  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e59_unit_sim as E59  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
K = 60
LABELS = {"L0": "L0 現行（1.2σ・20日・終盤・トレンド）", "L20_40": "L20_40 40日で+20%",
          "L30_40": "L30_40 40日で+30%", "L30_60": "L30_60 60日で+30%"}
SPEC = {"L20_40": (0.20, 40), "L30_40": (0.30, 40), "L30_60": (0.30, 60)}
TRAIN_ARMS = ("L0", "L20_40", "L30_40")
EMBARGO = 40                     # 学習する腕の中で最長のラベル確定日数
SEEDS = (42, 7, 123)
OUTCOMES = ("ret_o1_20", "ret_o1_40")
PCTS = (95.0, 90.0)
VOL_BANDS = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- #
# ラベル
# ---------------------------------------------------------------------- #

def strict_labels(df: pd.DataFrame, bars: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """
    (Code, Date) ごとに、翌営業日の寄りを分母にした「n 日以内の終値の最大 ÷ 買値 − 1」から
    固定しきい値のラベルを作る。先の足が n 日ぶん無い行は NaN（未確定）。
    """
    keys = df[["Code", "Date"]].copy()
    keys["Code"] = keys["Code"].astype(str)
    P = E41.forward(bars, keys, k=k)
    e = P["entry"]
    out = keys.copy()
    out["entry_fwd"] = e
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, (x, n) in SPEC.items():
            have = P["ok"][:, n - 1] & np.isfinite(e)
            mx = np.nanmax(np.where(P["ok"][:, :n], P["C"][:, :n], np.nan), axis=1)
            rise = mx / e - 1.0
            y = np.where(have, (rise >= x).astype(float), np.nan)
            out[f"y_{name}"] = y
            out[f"rise_{n}"] = np.where(have, rise, np.nan)
    return out


def vol_band(v: pd.Series, n: int = VOL_BANDS) -> pd.Series:
    r = v.rank(pct=True, method="first")
    return np.minimum((r * n).fillna(0).astype(int), n - 1).where(v.notna())


# ---------------------------------------------------------------------- #
# 学習（本番の木の形・種平均）
# ---------------------------------------------------------------------- #

def folds_common(dates: pd.Series, embargo: int = EMBARGO):
    return WF.make_folds(pd.to_datetime(dates), min_train_months=lab.MIN_TRAIN_MONTHS,
                         test_months=lab.TEST_MONTHS, step_months=lab.STEP_MONTHS,
                         embargo_days=embargo)


def oof_for(df: pd.DataFrame, ycol: str, cols: List[str], params: Dict, seeds: Sequence[int],
            folds, cache: str) -> pd.DataFrame:
    """ラベル ycol で LightGBM を窓ごとに学習し、種平均の out-of-fold を返す（保存済みなら読む）。"""
    import lightgbm as lgb
    import tuning

    if os.path.exists(cache):
        return pd.read_parquet(cache)
    d = pd.to_datetime(df["Date"])
    keep = ["Code", "Date", "fold_eval"]
    parts = []
    for f in folds:
        tr = df[(d <= pd.Timestamp(f.train_end)) & df[ycol].notna()]
        te = df[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))]
        if len(te) < lab.MIN_TEST or len(tr) < lab.MIN_TRAIN:
            continue
        Xtr = tr[cols].to_numpy(dtype=float)
        ytr = tr[ycol].to_numpy(dtype=int)
        Xte = te[cols].to_numpy(dtype=float)
        sc = np.zeros(len(te))
        for sd in seeds:
            m = lgb.LGBMClassifier(**{**params, "random_state": sd},
                                   scale_pos_weight=tuning.scale_pos_weight(ytr))
            m.fit(Xtr, ytr)
            sc += m.predict_proba(Xte)[:, 1] / len(seeds)
        part = te[keep].copy()
        part["score"] = sc
        part["fold"] = f.index
        part["n_train"] = len(tr)
        part["pos_train"] = int(ytr.sum())
        parts.append(part)
        log(f"    窓{f.index} {f.test_start}〜{f.test_end}: 訓練 {len(tr):,}（正例 {int(ytr.sum()):,}）/ 検証 {len(te):,}")
    o = pd.concat(parts, ignore_index=True)
    o.to_parquet(cache, index=False)
    return o


# ---------------------------------------------------------------------- #
# 評価
# ---------------------------------------------------------------------- #

def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok].astype(int), s[ok]))


def top_in_fold(o: pd.DataFrame, share: float = 0.10) -> pd.DataFrame:
    """窓の中の上位 share（件数をそろえる）。"""
    parts = []
    for f, g in o.groupby("fold"):
        n = max(1, int(round(len(g) * share)))
        parts.append(g.nlargest(n, "score"))
    return pd.concat(parts)


def pick1(o: pd.DataFrame, min_day: int = 5) -> pd.DataFrame:
    g = o.groupby("Date")
    o = o[g["score"].transform("size") >= min_day]
    return o.sort_values(["Date", "score"], ascending=[True, False]).groupby("Date").head(1)


def summarize(o: pd.DataFrame, name: str) -> dict:
    out = {"arm": name, "n": int(len(o))}
    for oc in OUTCOMES:
        r = pd.to_numeric(o[oc], errors="coerce")
        out[f"{oc}_mean"] = float(r.mean() * 100)
        out[f"{oc}_lt_m10"] = float((r < -0.10).mean() * 100)
    out["hit_L0"] = float(o["y_L0"].mean() * 100)
    out["hit_L30_40"] = float(o["y_L30_40"].mean() * 100)
    out["hit_L20_40"] = float(o["y_L20_40"].mean() * 100)
    out["vol_top_band"] = float((o["vol_band"] == VOL_BANDS - 1).mean() * 100)
    out["vol_low_band"] = float((o["vol_band"] == 0).mean() * 100)
    return out


def fold_diff(a: pd.DataFrame, b: pd.DataFrame, oc: str) -> dict:
    """窓ごとの平均の差（a − b）。a, b は同じ窓の集合を持つ選定。"""
    ma = a.groupby("fold")[oc].mean()
    mb = b.groupby("fold")[oc].mean()
    d = (ma - mb).dropna() * 100
    return {"mean": float(d.mean()) if len(d) else float("nan"),
            "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "won": int((d > 0).sum()), "n": int(len(d)), "worst": float(d.min()) if len(d) else float("nan")}


def pc(x, w=8, d=2, sign=True) -> str:
    if x is None or not np.isfinite(x):
        return f"{'-':>{w}}"
    return f"{x:{'+' if sign else ''}{w}.{d}f}"


# ---------------------------------------------------------------------- #
# 本体
# ---------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験66: ラベルを厳しく・長くしたら選定の実収益は上がるか")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--arms", default=",".join(TRAIN_ARMS))
    ap.add_argument("--oof-dir", default=L.DATA_DIR)
    ap.add_argument("--model-dir", default=L.MODEL_DIR)
    ap.add_argument("--prefix", default="e66")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in TRAIN_ARMS]
    if bad:
        raise SystemExit(f"学習する腕は {TRAIN_ARMS} から: {bad}")
    os.makedirs(OOF_DIR, exist_ok=True)

    meta = json.load(open(os.path.join(args.model_dir, "meta.json"), encoding="utf-8"))
    params = {k: v for k, v in meta["params"].items() if not k.startswith("_")}
    cols = F.columns(meta["preset"])
    log(f"本番の木の形（{meta['preset']}・{len(cols)}列）: " + ", ".join(f"{k}={v}" for k, v in params.items()
                                                                  if k in ("learning_rate", "num_leaves", "min_child_samples", "n_estimators")))

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df.rename(columns={"label": "y_L0"})
    log(f"データセット {len(df):,}行 {df['Date'].min().date()}〜{df['Date'].max().date()}")
    bars = E41.load_bars(set(df["Code"]))
    lab_new = strict_labels(df, bars, k=K)
    df = df.merge(lab_new, on=["Code", "Date"], how="left")
    gap = (df["entry_fwd"] / df["entry_price"] - 1.0).abs()
    log(f"買値の一致（forward の1日目の始値 対 データセットの entry_price）: 最大ずれ {gap.max():.2e}")
    df["vol_band"] = vol_band(df["vol_20d"])
    df["year"] = df["Date"].dt.year

    print("=" * 120)
    print("実験66 ラベルを「40営業日で +30%」のように厳しく・長くしたら、選定の実収益は上がるか")
    print("=" * 120)

    # ---- 1. ラベルの性質 ---- #
    print("\n■ 1. ラベルの性質（全行。到達は翌営業日の寄りを分母にした終値の最大）")
    print(f"  {'ラベル':<36}{'確定':>8}{'正例':>8}{'正例率':>8}" + "".join(f"{y:>8}" for y in sorted(df['year'].unique()))
          + "".join(f"{'ボラ' + str(b):>7}" for b in range(VOL_BANDS)))
    props = []
    for a, nm in LABELS.items():
        y = df[f"y_{a}"]
        ok = y.notna()
        row = {"label": a, "n": int(ok.sum()), "pos": int(y.sum()), "rate": float(y.mean() * 100)}
        cells = []
        for yy in sorted(df["year"].unique()):
            m = ok & (df["year"] == yy)
            row[f"y{yy}"] = float(y[m].mean() * 100) if m.any() else float("nan")
            cells.append(f"{row[f'y{yy}']:>7.1f}%")
        vb = []
        for b in range(VOL_BANDS):
            m = ok & (df["vol_band"] == b)
            row[f"vol{b}"] = float(y[m].mean() * 100) if m.any() else float("nan")
            vb.append(f"{row[f'vol{b}']:>6.1f}%")
        props.append(row)
        print(f"  {nm:<36}{row['n']:>8,}{row['pos']:>8,}{row['rate']:>7.2f}%" + "".join(cells) + "".join(vb))
    print("  （ボラ0 = 日次ボラの最も低い帯、ボラ4 = 最も高い帯。固定しきい値は高い帯ほど届きやすい）")
    r40 = df["rise_40"]
    print(f"  40日以内の終値の最大上昇の分位（全行）: 中央値 {r40.median()*100:+.1f}% / 75% {r40.quantile(0.75)*100:+.1f}% / "
          f"90% {r40.quantile(0.9)*100:+.1f}% / 95% {r40.quantile(0.95)*100:+.1f}%")

    # ---- 2. 今の本番モデルは新ラベルをどれだけ当てているか ---- #
    print("\n■ 2. 今の本番モデル（Release の5モデル OOF・過去分布の百分位）が各ラベルをどれだけ当てているか")
    oofs = E59.load_oofs(args.oof_dir, args.model_dir)
    base = E59.rows_with_pct(oofs, "past", E59.MIN_HIST)
    base = base.merge(df[["Code", "Date"] + [f"y_{a}" for a in LABELS] + ["ret_o1_40", "vol_band"]],
                      on=["Code", "Date"], how="left")
    cur = {}
    for nm, models, pct, tk in (("全5モデル 95以上・1日1件", L.ALL5, 95.0, 1),
                                ("全5モデル 95以上（全件）", L.ALL5, 95.0, 10 ** 6),
                                ("GBDT3 90以上（全件）", L.BOOST, 90.0, 10 ** 6)):
        _, _, picks = L.decide(base, agree=pct, min_break=0, top_k=tk, models=models)
        cur[nm] = picks
    print(f"  {'ラベル':<36}{'母集団':>8}{'全5-95・1件':>12}{'全5-95 全件':>12}{'GBDT3-90':>10}{'ROC(lgbm)':>10}{'ROC(5平均)':>10}")
    s5 = base[[f"s_{a}" for a in L.ALL5]].rank(pct=True).mean(axis=1)
    cur_rows = []
    for a, nm in LABELS.items():
        y = base[f"y_{a}"]
        ok = y.notna()
        row = {"label": a, "base": float(y[ok].mean() * 100)}
        for k, p in cur.items():
            yy = p[f"y_{a}"]
            row[k] = float(yy.mean() * 100) if yy.notna().any() else float("nan")
        row["roc_lgbm"] = auc(y, base["s_lgbm"])
        row["roc_mean5"] = auc(y, s5)
        cur_rows.append(row)
        print(f"  {nm:<36}{row['base']:>7.1f}%{row['全5モデル 95以上・1日1件']:>11.1f}%{row['全5モデル 95以上（全件）']:>11.1f}%"
              f"{row['GBDT3 90以上（全件）']:>9.1f}%{row['roc_lgbm']:>10.3f}{row['roc_mean5']:>10.3f}")
    print(f"  （件数: 全5-95・1件 {len(cur['全5モデル 95以上・1日1件']):,} / 全5-95 全件 {len(cur['全5モデル 95以上（全件）']):,} / "
          f"GBDT3-90 {len(cur['GBDT3 90以上（全件）']):,}。ROC は OOF 全行（{int(base['y_L30_40'].notna().sum()):,}行）で、"
          f"lgbm のスコアと5モデルの順位平均）")

    # ---- 3. 学習し直し ---- #
    print(f"\n■ 3. 学習し直し（LightGBM・本番の木の形・種 {seeds} の平均・窓は 36/6/6か月・エンバーゴ {EMBARGO}営業日）")
    need = df[[f"y_{a}" for a in arms]].notna().all(axis=1) & df["ret_o1_40"].notna()
    d = df[need].sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    folds = folds_common(d["Date"])
    d["fold_eval"] = 0
    for f in folds:
        m = (d["Date"] >= pd.Timestamp(f.test_start)) & (d["Date"] <= pd.Timestamp(f.test_end))
        d.loc[m, "fold_eval"] = f.index
    print(f"  全腕でラベルが確定している {len(d):,}行（{d['Date'].min().date()}〜{d['Date'].max().date()}）、窓 {len(folds)}")
    print(f"  {'ラベル':<36}{'正例率':>8}" + "".join(f"{'窓' + str(f.index):>7}" for f in folds) + "   （窓ごとの検証側の正例数）")
    for a in arms:
        cells = [f"{int(d.loc[d['fold_eval'] == f.index, f'y_{a}'].sum()):>7,}" for f in folds]
        print(f"  {LABELS[a]:<36}{d[f'y_{a}'].mean()*100:>7.2f}%" + "".join(cells))

    oof = {}
    for a in arms:
        log(f"  腕 {a} を学習")
        cache = os.path.join(OOF_DIR, f"{args.prefix}_{a}_oof_s{'-'.join(str(s) for s in seeds)}.parquet")
        o = oof_for(d, f"y_{a}", cols, params, seeds, folds, cache)
        o = o.merge(d[["Code", "Date", "y_L0", "y_L20_40", "y_L30_40", "vol_band"] + list(OUTCOMES)],
                    on=["Code", "Date"], how="left")
        o["label"] = o[f"y_{a}"]
        oof[a] = o

    # 分離力（参考）
    print("\n  --- 分離力（参考。自分のラベル / 現行ラベル L0 / L30_40 に対する ROC-AUC。日内AUC は自分のラベル）---")
    print(f"  {'腕':<36}{'件数':>8}{'ROC 自分':>10}{'ROC L0':>10}{'ROC L30_40':>12}{'日内AUC':>9}")
    sep_rows = []
    for a in arms:
        o = oof[a]
        row = {"arm": a, "n": len(o), "roc_own": auc(o[f"y_{a}"], o["score"]),
               "roc_L0": auc(o["y_L0"], o["score"]), "roc_L30_40": auc(o["y_L30_40"], o["score"]),
               "auc_in_day": lab.auc_in_day(o.assign(label=o[f"y_{a}"].astype(int)))}
        sep_rows.append(row)
        print(f"  {LABELS[a]:<36}{row['n']:>8,}{row['roc_own']:>10.3f}{row['roc_L0']:>10.3f}{row['roc_L30_40']:>12.3f}"
              f"{row['auc_in_day']:>9.3f}")

    # 閾値ルール（運用の形）
    thr_rows = []
    for oc in OUTCOMES:
        for pct in PCTS:
            print(f"\n  --- 閾値ルール: 前の窓のスコア分布の上位 {100 - pct:.0f}% なら買う（物差し {oc}）---")
            print(f"  {'腕':<36}{'件数':>7}{'平均':>8}{'全体との差':>10}{'窓平均差':>9}{'SE':>6}{'勝ち窓':>7}{'最悪窓':>8}"
                  f"{'勝率':>6}{'−10%未満':>9}{'L30_40到達':>11}{'高ボラ帯':>9}")
            for a in arms:
                o = oof[a]
                t = lab.threshold_edge(o, pct=pct, outcome=oc)
                # 選ばれた行そのもの（−10% 未満・到達率・ボラの帯のため）
                fs = sorted(o["fold"].unique())
                picks = []
                for f in fs:
                    ref = o.loc[o["fold"] < f, "score"].to_numpy()
                    if len(ref) < 500:
                        continue
                    thr = float(np.percentile(ref, pct))
                    cur_ = o[o["fold"] == f]
                    picks.append(cur_[cur_["score"] > thr])
                p = pd.concat(picks) if picks else o.iloc[:0]
                r = pd.to_numeric(p[oc], errors="coerce")
                row = {"outcome": oc, "pct": pct, "arm": a, **{k: v for k, v in t.items()},
                       "lt_m10": float((r < -0.10).mean() * 100) if len(p) else float("nan"),
                       "hit_L30_40": float(p["y_L30_40"].mean() * 100) if len(p) else float("nan"),
                       "vol_top": float((p["vol_band"] == VOL_BANDS - 1).mean() * 100) if len(p) else float("nan")}
                thr_rows.append(row)
                print(f"  {LABELS[a]:<36}{t['thr_n']:>7,}{pc(t['thr_end'])}{pc(t['thr_lift'], 10)}{pc(t['thr_fold_mean'], 9)}"
                      f"{pc(t['thr_fold_sd'] / math.sqrt(t['thr_folds']) if t['thr_folds'] > 1 else float('nan'), 6, 2, False)}"
                      f"{t['thr_folds_won']:>4}/{t['thr_folds']:<2}{pc(t['thr_worst'], 8)}{t['thr_win']*100:>5.0f}%"
                      f"{row['lt_m10']:>8.1f}%{row['hit_L30_40']:>10.1f}%{row['vol_top']:>8.1f}%")
    print("  （全体との差 = 選ばれた行の平均 − その窓全体の平均。窓平均差 = 窓ごとの差の平均。高ボラ帯 = 選ばれた行のうち"
          "日次ボラが最も高い帯の割合。母集団では 20%）")

    # 窓の中の上位10%（件数そろえ）と 日付内1位
    print("\n  --- 窓の中の上位10%（件数をそろえる）と、日付内1位（候補5件以上の日）---")
    print(f"  {'腕':<36}{'上位10% 件数':>12}{'ret20':>8}{'ret40':>8}{'−10%未満':>9}{'L0到達':>8}{'L30_40到達':>11}{'高ボラ帯':>9}"
          f"{'1位 件数':>9}{'ret20':>8}{'ret40':>8}{'L30_40到達':>11}")
    sel_rows = []
    tops = {a: top_in_fold(oof[a]) for a in arms}
    p1s = {a: pick1(oof[a]) for a in arms}
    for a in arms:
        t, p1 = summarize(tops[a], a), summarize(p1s[a], a)
        sel_rows.append({"kind": "top10", **t})
        sel_rows.append({"kind": "pick1", **p1})
        print(f"  {LABELS[a]:<36}{t['n']:>12,}{pc(t['ret_o1_20_mean'])}{pc(t['ret_o1_40_mean'])}{t['ret_o1_20_lt_m10']:>8.1f}%"
              f"{t['hit_L0']:>7.1f}%{t['hit_L30_40']:>10.1f}%{t['vol_top_band']:>8.1f}%"
              f"{p1['n']:>9,}{pc(p1['ret_o1_20_mean'])}{pc(p1['ret_o1_40_mean'])}{p1['hit_L30_40']:>10.1f}%")
    print("\n  --- L0 との窓ごとの差（上位10%・件数そろえ）---")
    print(f"  {'腕':<36}{'ret20 差':>9}{'SE':>6}{'勝ち窓':>7}{'最悪窓':>8}{'ret40 差':>9}{'SE':>6}{'勝ち窓':>7}{'最悪窓':>8}")
    diff_rows = []
    for a in arms:
        if a == "L0":
            continue
        cells = ""
        row = {"arm": a}
        for oc in OUTCOMES:
            dd = fold_diff(tops[a], tops["L0"], oc)
            row.update({f"{oc}_{k}": v for k, v in dd.items()})
            cells += f"{pc(dd['mean'], 9)}{pc(dd['se'], 6, 2, False)}{dd['won']:>4}/{dd['n']:<2}{pc(dd['worst'], 8)}"
        diff_rows.append(row)
        print(f"  {LABELS[a]:<36}{cells}")

    # 年ごと（上位10%・件数そろえ）
    print("\n  --- 年ごと（上位10%・件数そろえ。件数 / ret20 / ret40）---")
    yrs = sorted(tops["L0"]["Date"].dt.year.unique())
    print(f"  {'腕':<36}" + "".join(f"{y:>22}" for y in yrs))
    for a in arms:
        cells = []
        for y in yrs:
            g = tops[a][tops[a]["Date"].dt.year == y]
            cells.append(f"{len(g):>4} {g['ret_o1_20'].mean()*100:>+7.2f} {g['ret_o1_40'].mean()*100:>+7.2f}" if len(g) else f"{'-':>22}")
        print(f"  {LABELS[a]:<36}" + "".join(f"{c:>22}" for c in cells))

    pd.DataFrame(props).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_labels.csv"), index=False)
    pd.DataFrame(cur_rows).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_current_model.csv"), index=False)
    pd.DataFrame(sep_rows).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_separation.csv"), index=False)
    pd.DataFrame(thr_rows).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_threshold.csv"), index=False)
    pd.DataFrame(sel_rows).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_selection.csv"), index=False)
    pd.DataFrame(diff_rows).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_diff.csv"), index=False)
    log(f"書いた: {OOF_DIR}/{args.prefix}_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
