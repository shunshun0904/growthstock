#!/usr/bin/env python3
"""
実験20: 年単位の業績の軌道（悪化 → 立て直し → 新高値）に検出力はあるか。

なぜ測るか
--------
母集団は78週（約1年半）ぶりの高値更新日。「業績が悪化し、立て直しを経て
新高値を付ける」という経路が典型なら、**年単位の業績の変化率や軌道**は
説明変数として効くはずだ、という仮説。

既存の決算特徴量は四半期の刻み（q0..q3、chg1..3）で、見ているのは直近1年
ほどの動き。年単位で2〜3期の軌道を見る特徴量は持っていない。

EDINET DB の取得を待たずに測れる。J-Quants の通期開示（CurPerType == "FY"）
から、売上・営業利益・純利益・EPS の年次パネルを**開示日ベースで時点整合**に
作れる。EDINET 側で足すのは、J-Quants に無い明細（販管費・設備投資・
有利子負債・のれん・浮動株）の軌道であって、軌道そのものの検出力は
ここで先に白黒つける。

測り方
-----
1. 年次パネル: 各 (Code, 年度末) について、その年度の通期開示のうち
   **最初の開示**の値を使う（訂正後の値で過去を上書きしない。時点整合の
   保守側）。前期・前々期は同じ銘柄の直前の年度末の行（年度末の間隔が
   300〜430日 / 600〜860日のときだけ連続とみなす。決算期変更をまたがない）
2. 各サンプル (Code, Date) に、Date 以前に開示された最新の通期を
   merge_asof で付ける。前期・前々期はそれより前に開示済みなので先読みは無い
3. 軌道の特徴量（下の FEATURES）
4. 検出力: 特徴量ごとに
     - 併合 AUC（ラベル）
     - 窓ごとの AUC（本番の out-of-fold と同じ10窓）の平均 ± SE
     - **上位10%の実収益の超過**（窓ごとに、特徴量が上位10%の行の
       ret_o1_20 平均 − 窓全体の平均）の平均 ± SE と z
       これがモデルの評価と同じ物差し。ノイズ床 0.143pt（実験11）
   |z| で並べる。負の z は「低いほど良い」で、それも検出力。

足切り
-----
特徴量は約40本あるので、偶然でも数本は |z|>1 を超える。z>2 に加えて
「窓の何本で同じ符号か」を見る。
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import walkforward as WF  # noqa: E402
from train_production import (  # noqa: E402
    OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS)

ITEMS = ["Sales", "OP", "NP", "EPS"]
#: 何期前まで持つか（1 = 前期, 2 = 2年前, 3 = 3年前）
LAGS = (1, 2, 3)
OUTCOME = "ret_o1_20"
TOP_PCT = 90
#: 最新の通期がこれより古ければ「開示が止まっている」とみなして捨てる
STALE_DAYS = 400
OUT = os.path.join(lab.DATA_DIR, "oof", "e20_screen.csv")


def sym(a: pd.Series, b: pd.Series) -> pd.Series:
    """対称変化率 (a-b)/((|a|+|b|)/2)。符号をまたいでも定義でき、[-2, 2] に収まる。"""
    den = (a.abs() + b.abs()) / 2.0
    return ((a - b) / den).where(den > 0)


def annual_panel(data_dir: str) -> pd.DataFrame:
    cols = ["Code", "DiscDate", "CurPerType", "CurFYEn", "CurPerEn"] + ITEMS
    paths = sorted(glob.glob(os.path.join(data_dir, "fins_*.parquet")))
    if not paths:
        raise SystemExit("fins_*.parquet がありません")
    f = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    f = f[f["CurPerType"] == "FY"].copy()
    f["DiscDate"] = pd.to_datetime(f["DiscDate"])
    f["fy_end"] = pd.to_datetime(f["CurFYEn"], errors="coerce").fillna(
        pd.to_datetime(f["CurPerEn"], errors="coerce"))
    f = f.dropna(subset=["fy_end"])
    for c in ITEMS:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    # 値の無い通期開示（配当予想の修正など）は年度の代表にしない
    f = f[f[ITEMS].notna().any(axis=1)]
    f = (f.sort_values(["Code", "fy_end", "DiscDate"])
          .drop_duplicates(["Code", "fy_end"], keep="first")
          .sort_values(["Code", "fy_end"]).reset_index(drop=True))
    g = f.groupby("Code")
    for k in LAGS:
        for c in ITEMS:
            f[f"{c}_y{k}"] = g[c].shift(k)
        f[f"fy_end_y{k}"] = g["fy_end"].shift(k)
    # 年度末の間隔が k 年 ± 約2か月のときだけ連続とみなす（決算期変更をまたがない）
    for k in LAGS:
        gap = (f["fy_end"] - f[f"fy_end_y{k}"]).dt.days
        ok = gap.between(365 * k - 65, 365 * k + 65)
        for c in ITEMS:
            f.loc[~ok, f"{c}_y{k}"] = np.nan
    keep = ["Code", "DiscDate", "fy_end"] + ITEMS + \
        [f"{c}_y{k}" for c in ITEMS for k in LAGS]
    return f[keep]


def features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for c in ITEMS:
        y0, y1, y2, y3 = df[c], df[f"{c}_y1"], df[f"{c}_y2"], df[f"{c}_y3"]
        yoy1, yoy2, yoy3 = sym(y0, y1), sym(y1, y2), sym(y2, y3)
        out[f"{c}_yoy1"] = yoy1                       # 直近の前年比
        out[f"{c}_yoy2"] = yoy2                       # 1期前の前年比
        out[f"{c}_yoy3"] = yoy3                       # 2期前の前年比
        out[f"{c}_2y"] = sym(y0, y2)                  # 2年での変化
        out[f"{c}_3y"] = sym(y0, y3)                  # 3年での変化
        out[f"{c}_accel"] = yoy1 - yoy2               # 加速
        peak = pd.concat([y1, y2], axis=1).max(axis=1)
        out[f"{c}_recovery"] = sym(y0, peak)          # 過去2期の高いほうを超えたか
        out[f"{c}_dip"] = sym(y1, y2).where(y1 < y2, 0.0).where(y1.notna() & y2.notna())
        both = y0.notna() & y1.notna() & y2.notna()
        out[f"{c}_vshape"] = ((y1 < y2) & (y0 > y1)).astype(float).where(both)
        out[f"{c}_3y_high"] = ((y0 > y1) & (y0 > y2)).astype(float).where(both)
        # 3年前まで見た版（悪化 → 立て直し が2期以上かかる経路を拾う）
        peak3 = pd.concat([y1, y2, y3], axis=1).max(axis=1)
        prior3 = pd.concat([y2, y3], axis=1).max(axis=1)
        all4 = both & y3.notna()
        out[f"{c}_recovery3"] = sym(y0, peak3).where(all4)   # 過去3期の最高を超えたか
        out[f"{c}_dip3"] = sym(y1, prior3).clip(upper=0.0).where(all4)  # 前期が過去の山からどれだけ落ちたか
        out[f"{c}_vshape3"] = ((y1 < prior3) & (y0 > y1)).astype(float).where(all4)
        out[f"{c}_4y_high"] = (y0 > peak3).astype(float).where(all4)
        if c != "Sales":
            out[f"{c}_turn"] = ((y1 <= 0) & (y0 > 0)).astype(float).where(y0.notna() & y1.notna())
    m0 = (df["OP"] / df["Sales"]).where(df["Sales"] > 0)
    m1 = (df["OP_y1"] / df["Sales_y1"]).where(df["Sales_y1"] > 0)
    m2 = (df["OP_y2"] / df["Sales_y2"]).where(df["Sales_y2"] > 0)
    out["opm_chg1"] = m0 - m1
    out["opm_chg2"] = m1 - m2
    out["opm_2y"] = m0 - m2
    return out


def screen(frame: pd.DataFrame, feats: List[str], windows: List[tuple]) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    y = frame["label"].to_numpy(dtype=float)
    r = frame[OUTCOME].to_numpy(dtype=float)
    d = frame["Date"].to_numpy()
    rows = []
    for f in feats:
        x = frame[f].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
        cov = ok.mean()
        if ok.sum() < 500 or len(np.unique(y[ok])) < 2 or np.nanstd(x[ok]) == 0:
            rows.append({"feature": f, "coverage": cov, "n": int(ok.sum())})
            continue
        pooled = roc_auc_score(y[ok], x[ok])
        aucs, edges = [], []
        for (s, e) in windows:
            w = ok & (d >= s) & (d <= e)
            if w.sum() < 100 or len(np.unique(y[w])) < 2:
                continue
            xw, yw, rw = x[w], y[w], r[w]
            aucs.append(roc_auc_score(yw, xw))
            thr = np.nanpercentile(xw, TOP_PCT)
            top = xw >= thr
            if top.sum() >= 5:
                edges.append((rw[top].mean() - rw.mean()) * 100)
        aucs, edges = np.array(aucs), np.array(edges)
        se_e = edges.std(ddof=1) / np.sqrt(len(edges)) if len(edges) > 1 else np.nan
        rows.append({
            "feature": f, "coverage": cov, "n": int(ok.sum()),
            "auc_pooled": pooled,
            "auc_win": aucs.mean() if len(aucs) else np.nan,
            "auc_se": aucs.std(ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else np.nan,
            "auc_win_gt05": int((aucs > 0.5).sum()), "n_win": len(aucs),
            "edge_pt": edges.mean() if len(edges) else np.nan,
            "edge_se": se_e,
            "edge_z": edges.mean() / se_e if len(edges) > 1 and se_e > 0 else np.nan,
            "edge_win_pos": int((edges > 0).sum()),
        })
    out = pd.DataFrame(rows)
    out["abs_z"] = out["edge_z"].abs()
    return out.sort_values("abs_z", ascending=False).reset_index(drop=True)


def main() -> int:
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    print(f"母集団 {len(frame):,}件 / 正例率 {frame['label'].mean()*100:.2f}% / "
          f"{frame['Date'].min().date()} 〜 {frame['Date'].max().date()}")

    panel = annual_panel(lab.DATA_DIR)
    print(f"年次パネル {len(panel):,}行（Code×年度）/ 銘柄 {panel['Code'].nunique():,} / "
          f"年度末 {panel['fy_end'].min().date()} 〜 {panel['fy_end'].max().date()}")
    for k in LAGS:
        print(f"  {k}期前が連続で取れている行: "
              f"{panel[f'Sales_y{k}'].notna().mean()*100:.1f}%")

    # 各サンプルに「Date 以前に開示された最新の通期」を付ける
    frame = frame.sort_values("Date")
    panel = panel.sort_values("DiscDate")
    m = pd.merge_asof(frame, panel, left_on="Date", right_on="DiscDate", by="Code",
                      direction="backward", allow_exact_matches=True)
    stale = (m["Date"] - m["DiscDate"]).dt.days > STALE_DAYS
    for c in [c for c in m.columns if c in ITEMS or c.endswith(("_y1", "_y2", "_y3"))]:
        m.loc[stale | m["DiscDate"].isna(), c] = np.nan
    print(f"通期が付いた行 {m['Sales'].notna().mean()*100:.1f}% / "
          f"前期まで {m['Sales_y1'].notna().mean()*100:.1f}% / "
          f"2年前まで {m['Sales_y2'].notna().mean()*100:.1f}% / "
          f"3年前まで {m['Sales_y3'].notna().mean()*100:.1f}%")

    feats = features(m)
    for k, v in feats.items():
        m[k] = v
    names = list(feats)

    folds = WF.make_folds(m["Date"], min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    windows = [(np.datetime64(f.test_start), np.datetime64(f.test_end)) for f in folds]
    print(f"評価窓 {len(windows)}本（本番の out-of-fold と同じ刻み）/ 物差し {OUTCOME}\n")

    res = screen(m, names, windows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== 検出力（|z| の順。z は上位10%の実収益の超過 / 窓SE）===")
    print(f"  {'特徴量':<16}{'充足':>6}{'AUC併合':>9}{'AUC窓':>8}{'>0.5':>6}"
          f"{'超過pt':>8}{'SE':>6}{'z':>7}{'正の窓':>7}")
    for _, r in res.iterrows():
        if np.isnan(r.get("edge_z", np.nan)):
            print(f"  {r['feature']:<16}{r['coverage']*100:>5.0f}%   （件数不足）")
            continue
        print(f"  {r['feature']:<16}{r['coverage']*100:>5.0f}%{r['auc_pooled']:>9.3f}"
              f"{r['auc_win']:>8.3f}{r['auc_win_gt05']:>3}/{r['n_win']:<2}"
              f"{r['edge_pt']:>+8.2f}{r['edge_se']:>6.2f}{r['edge_z']:>+7.2f}"
              f"{r['edge_win_pos']:>4}/{r['n_win']:<2}")
    hits = res[res["abs_z"] > 2]
    print(f"\n|z|>2: {len(hits)}本 / {len(res)}本（偶然でも 5% ≈ {len(res)*0.05:.1f}本は超える）")
    print(f"ノイズ床（実験11、モデルの種を振ったときの窓平均レンジ）: 0.143pt")
    print(f"記録: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
