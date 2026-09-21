#!/usr/bin/env python3
"""
実験23: どの「次元」に検出力があるか —— EDINET の特徴量を考案するための基礎分析。

問い
----
EDINET で足せるのは、J-Quants に無い明細（売上総利益・販管費・研究開発費・
設備投資・減価償却・棚卸資産・売上債権・有利子負債・のれん・従業員・
浮動株・自己株・政策保有株・役員持株）。どの明細を優先するかを、
今あるデータで測れる「同じ次元の粗い代理」の検出力から決める。

  A. 本番の特徴量 151本を1本ずつ、実験20 と同じ方法で測る（上位10% と
     下位10% の両側）。「どの次元に差があるか」の地図
  B. J-Quants の通期（FY）行から作れる、EDINET の明細の粗い代理:
       発行済株式数の変化（増資/消却）、自己株比率とその変化（自社株買い）、
       総資産・自己資本の成長、現金比率とその変化、投資CF/総資産（設備投資の
       代理）、財務CF/総資産（借入返済・還元）、営業CFマージンの変化、
       アクルーアルの変化、配当の変化、決算月、開示からの経過日数
     を同じ方法で測る

測り方は実験20 と同じ（本番の out-of-fold と同じ 11窓、`ret_o1_20` の
上位/下位10% の窓平均超過 / 窓SE = z）。ノイズ床 0.143pt（実験11）。
特徴量が多いので偶然の |z|>2 が数本は出る。同じ次元で符号が揃うかで読む。
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import walkforward as WF  # noqa: E402
from e20_annual_trajectory import OUTCOME, sym  # noqa: E402
from train_production import (  # noqa: E402
    OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS)

OUT = os.path.join(lab.DATA_DIR, "oof", "e23_screen.csv")
STALE_DAYS = 400
FY_ITEMS = ["Sales", "OP", "NP", "TA", "Eq", "CashEq", "CFO", "CFI", "CFF",
            "ShOutFY", "TrShFY", "DivAnn"]


def screen_both(frame: pd.DataFrame, feats: List[str], windows: List[tuple],
                pct: int = 10) -> pd.DataFrame:
    """上位 pct% と下位 pct% の両側で、窓ごとの実収益の超過を測る。"""
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
        aucs, top_e, bot_e = [], [], []
        for (s, e) in windows:
            w = ok & (d >= s) & (d <= e)
            if w.sum() < 100 or len(np.unique(y[w])) < 2:
                continue
            xw, yw, rw = x[w], y[w], r[w]
            aucs.append(roc_auc_score(yw, xw))
            hi, lo = np.nanpercentile(xw, 100 - pct), np.nanpercentile(xw, pct)
            top, bot = xw >= hi, xw <= lo
            if top.sum() >= 5:
                top_e.append((rw[top].mean() - rw.mean()) * 100)
            if bot.sum() >= 5:
                bot_e.append((rw[bot].mean() - rw.mean()) * 100)
        rec = {"feature": f, "coverage": cov, "n": int(ok.sum()), "auc_pooled": pooled,
               "auc_win": float(np.mean(aucs)) if aucs else np.nan, "n_win": len(aucs)}
        for name, arr in (("top", np.array(top_e)), ("bot", np.array(bot_e))):
            if len(arr) > 1 and arr.std(ddof=1) > 0:
                se = arr.std(ddof=1) / np.sqrt(len(arr))
                rec[f"{name}_pt"] = arr.mean()
                rec[f"{name}_se"] = se
                rec[f"{name}_z"] = arr.mean() / se
                rec[f"{name}_pos"] = int((arr > 0).sum())
            else:
                rec[f"{name}_pt"] = rec[f"{name}_se"] = rec[f"{name}_z"] = np.nan
                rec[f"{name}_pos"] = 0
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["max_abs_z"] = out[["top_z", "bot_z"]].abs().max(axis=1)
    return out.sort_values("max_abs_z", ascending=False).reset_index(drop=True)


def fy_panel(data_dir: str) -> pd.DataFrame:
    cols = ["Code", "DiscDate", "CurPerType", "CurFYEn", "CurPerEn"] + FY_ITEMS
    paths = sorted(glob.glob(os.path.join(data_dir, "fins_*.parquet")))
    f = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    f = f[f["CurPerType"] == "FY"].copy()
    f["DiscDate"] = pd.to_datetime(f["DiscDate"])
    f["fy_end"] = pd.to_datetime(f["CurFYEn"], errors="coerce").fillna(
        pd.to_datetime(f["CurPerEn"], errors="coerce"))
    f = f.dropna(subset=["fy_end"])
    for c in FY_ITEMS:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f[f[["Sales", "NP", "TA"]].notna().any(axis=1)]
    f = (f.sort_values(["Code", "fy_end", "DiscDate"])
          .drop_duplicates(["Code", "fy_end"], keep="first")
          .sort_values(["Code", "fy_end"]).reset_index(drop=True))
    g = f.groupby("Code")
    for k in (1, 2):
        for c in FY_ITEMS:
            f[f"{c}_y{k}"] = g[c].shift(k)
        f[f"fy_end_y{k}"] = g["fy_end"].shift(k)
        gap = (f["fy_end"] - f[f"fy_end_y{k}"]).dt.days
        ok = gap.between(365 * k - 65, 365 * k + 65)
        for c in FY_ITEMS:
            f.loc[~ok, f"{c}_y{k}"] = np.nan
    return f


def proxy_features(m: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    ta, ta1 = m["TA"], m["TA_y1"]
    # 資本政策
    out["jq_shares_chg1"] = sym(m["ShOutFY"], m["ShOutFY_y1"])            # 発行済株式数の変化
    tsy, tsy1 = (m["TrShFY"] / m["ShOutFY"]).where(m["ShOutFY"] > 0), \
                (m["TrShFY_y1"] / m["ShOutFY_y1"]).where(m["ShOutFY_y1"] > 0)
    out["jq_tsy_r"] = tsy                                                   # 自己株比率
    out["jq_tsy_chg1"] = tsy - tsy1                                         # 自社株買い（比率の増分）
    out["jq_div_chg1"] = sym(m["DivAnn"], m["DivAnn_y1"])                   # 配当の変化
    out["jq_div_cut"] = (m["DivAnn"] < m["DivAnn_y1"]).astype(float).where(
        m["DivAnn"].notna() & m["DivAnn_y1"].notna())
    out["jq_cff_ta"] = (m["CFF"] / ta).where(ta > 0)                        # 財務CF（負 = 返済・還元）
    # 投資
    out["jq_ta_g1"] = sym(ta, ta1)                                          # 総資産成長
    out["jq_ta_g2"] = sym(ta, m["TA_y2"])
    out["jq_eq_g1"] = sym(m["Eq"], m["Eq_y1"])
    out["jq_cfi_ta"] = (m["CFI"] / ta).where(ta > 0)                        # 投資CF（負 = 投資）
    out["jq_cfi_chg1"] = out["jq_cfi_ta"] - (m["CFI_y1"] / ta1).where(ta1 > 0)
    # 財務体質・収益の質
    cash_r, cash_r1 = (m["CashEq"] / ta).where(ta > 0), (m["CashEq_y1"] / ta1).where(ta1 > 0)
    out["jq_cash_r"] = cash_r
    out["jq_cash_chg1"] = cash_r - cash_r1
    out["jq_liab_r"] = ((ta - m["Eq"]) / ta).where(ta > 0)                  # 負債比率
    out["jq_liab_chg1"] = out["jq_liab_r"] - ((ta1 - m["Eq_y1"]) / ta1).where(ta1 > 0)
    cfo_m, cfo_m1 = (m["CFO"] / m["Sales"]).where(m["Sales"] > 0), \
                    (m["CFO_y1"] / m["Sales_y1"]).where(m["Sales_y1"] > 0)
    out["jq_cfo_m"] = cfo_m
    out["jq_cfo_m_chg1"] = cfo_m - cfo_m1
    acc, acc1 = ((m["NP"] - m["CFO"]) / ta).where(ta > 0), \
                ((m["NP_y1"] - m["CFO_y1"]) / ta1).where(ta1 > 0)
    out["jq_accr"] = acc
    out["jq_accr_chg1"] = acc - acc1
    out["jq_fcf_ta"] = ((m["CFO"] + m["CFI"]) / ta).where(ta > 0)            # ざっくり FCF
    # 属性・タイミング
    out["jq_fyend_month"] = m["fy_end"].dt.month.astype(float)
    out["jq_days_since_fy"] = (m["Date"] - m["DiscDate"]).dt.days.astype(float)
    return out


def main() -> int:
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    folds = WF.make_folds(frame["Date"], min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    windows = [(np.datetime64(f.test_start), np.datetime64(f.test_end)) for f in folds]
    print(f"母集団 {len(frame):,}件 / 評価窓 {len(windows)}本 / 物差し {OUTCOME}")

    # ---- A. 本番の特徴量 ----
    base = [c for c in F.columns("all") if c in frame.columns]
    res_a = screen_both(frame, base, windows)
    res_a["part"] = "A"

    # ---- B. J-Quants の通期から作る代理 ----
    panel = fy_panel(lab.DATA_DIR).sort_values("DiscDate")
    fr = frame.sort_values("Date")
    m = pd.merge_asof(fr, panel, left_on="Date", right_on="DiscDate", by="Code",
                      direction="backward", allow_exact_matches=True)
    stale = (m["Date"] - m["DiscDate"]).dt.days > STALE_DAYS
    lag_cols = [c for c in m.columns if c in FY_ITEMS or c.endswith(("_y1", "_y2"))]
    m.loc[stale | m["DiscDate"].isna(), lag_cols] = np.nan
    feats = proxy_features(m)
    for k, v in feats.items():
        m[k] = v
    # 四半期を含む「直近の開示からの日数」
    q = pd.concat([pd.read_parquet(p, columns=["Code", "DiscDate", "Sales", "NP"])
                   for p in sorted(glob.glob(os.path.join(lab.DATA_DIR, "fins_*.parquet")))],
                  ignore_index=True)
    q = q[q[["Sales", "NP"]].notna().any(axis=1)][["Code", "DiscDate"]].copy()
    q["DiscDate"] = pd.to_datetime(q["DiscDate"])
    q = q.sort_values("DiscDate").rename(columns={"DiscDate": "AnyDisc"})
    m = pd.merge_asof(m.sort_values("Date"), q, left_on="Date", right_on="AnyDisc",
                      by="Code", direction="backward", allow_exact_matches=True)
    m["jq_days_since_disc"] = (m["Date"] - m["AnyDisc"]).dt.days.astype(float).clip(upper=400)
    names = list(feats) + ["jq_days_since_disc"]
    print(f"代理特徴量 {len(names)}本 / 充足 平均 {m[names].notna().mean().mean()*100:.1f}%")
    res_b = screen_both(m, names, windows)
    res_b["part"] = "B"

    res = pd.concat([res_a, res_b], ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_csv(OUT, index=False)

    def show(df: pd.DataFrame, title: str, k: int) -> None:
        print(f"\n=== {title}（|z| 上位 {k}本。top = 上位10%の超過、bot = 下位10%の超過）===")
        print(f"  {'特徴量':<26}{'充足':>6}{'AUC窓':>7}{'top pt':>8}{'z':>7}{'窓':>6}"
              f"{'bot pt':>8}{'z':>7}{'窓':>6}")
        for _, r in df.head(k).iterrows():
            if np.isnan(r.get("max_abs_z", np.nan)):
                continue
            print(f"  {r['feature']:<26}{r['coverage']*100:>5.0f}%{r['auc_win']:>7.3f}"
                  f"{r['top_pt']:>+8.2f}{r['top_z']:>+7.2f}{int(r['top_pos']):>3}/{int(r['n_win']):<2}"
                  f"{r['bot_pt']:>+8.2f}{r['bot_z']:>+7.2f}{int(r['bot_pos']):>3}/{int(r['n_win']):<2}")

    show(res_a, "A. 本番の特徴量", 40)
    show(res_b, "B. J-Quants の通期から作る EDINET の代理", 40)
    for part, df in (("A", res_a), ("B", res_b)):
        n = int((df["max_abs_z"] > 2).sum())
        print(f"\n{part}: |z|>2 は {n}本 / {len(df)}本（偶然でも ≈ {len(df)*0.05*2:.1f}本）")
    print(f"記録: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
