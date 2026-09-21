#!/usr/bin/env python3
"""
EDINET DB の年次財務（research/_data/edinet_fin.parquet）から、年単位の
業績の軌道を特徴量にする。

なぜ年単位か
----------
母集団は78週（約1年半）ぶりの高値更新。「業績が悪化し、立て直しを経て
新高値」という経路が典型なら、直近の値だけでなく **2年前・3年前の値と、
年ごとの変化率** に情報があるはず（実験20 で J-Quants の通期データから
検出力を確認済み。docs/MODEL_ANNUAL_TRAJECTORY.md）。
EDINET で足せるのは J-Quants に無い明細（売上総利益・販管費・研究開発費・
設備投資・減価償却・棚卸資産・売上債権・有利子負債・のれん・従業員数・
浮動株・自己株）。

時点整合
-------
各行（Code, Date）には「提出日の翌日 ≤ Date」を満たす最新の有価証券報告書
（y0）を付ける。提出時刻が場中・引け後どちらでも、翌日から知りえたとする
（保守側）。前期・2年前・3年前（y1/y2/y3）は y0 より前の年度の書類なので、
y0 が読める時点で既知。`submit_date` が無い古い年度（実測で 2015年以前）は
y0 にはなれないが、y1〜y3 としては使う。

y0 が Date から STALE_DAYS より古ければ、開示が止まっている（上場廃止・
決算期変更など）とみなして全部 NaN にする。

比較できない年度
--------------
同じ会社でも、会計基準（JP → IFRS）や連結/単体（`basis`）が変わった年度を
またぐ比較は意味が無いので、y0 と yk で違えば yk を NaN にする。
EPS は `adjusted_eps`（分割調整済み）を使う。株数は比率（浮動株比率・
自己株比率）にして分割の影響を消す。

値の無い項目
----------
応答から省かれた項目は NaN のまま。ただし
  - 有利子負債（`ibd_*`）と、のれん（`goodwill`）は「無い＝ゼロ」の会社が多い
    （任天堂には借入ものれんも無い。docs/DATA_EDINETDB.md）ので、貸借対照表
    自体がある行では 0 とみなす
  - `ibd_*` が無い行では借入金・社債・CP の合計で代用する

出力の列名は全部 `ed_` で始まる（既存の特徴量と衝突させない）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
FIN = os.path.join(DATA_DIR, "edinet_fin.parquet")

KEY = "jq_code"
LAGS = (1, 2, 3)
#: y0 が Date からこれより古ければ捨てる（年1回の開示 + 猶予）
STALE_DAYS = 400

#: 水準を持つ項目（金額・人数）。この軌道（前年比など）を特徴量にする
LEVELS: List[str] = [
    "revenue", "operating_income", "net_income", "adjusted_eps",
    "gross_profit", "sga", "rnd_expenses", "cf_operating", "capex", "depreciation",
    "num_employees", "total_assets", "net_assets", "cash", "inventories",
    "trade_receivables", "debt", "goodwill",
]
#: 依頼の中核（売上・営業利益・純利益・EPS）
CORE = ["revenue", "operating_income", "net_income", "adjusted_eps"]
#: 比率（水準と、その年ごとの差を特徴量にする）
RATIOS: List[str] = [
    "opm", "gpm", "sga_r", "rnd_r", "ocf_m", "roe", "roa", "eq_r", "debt_r",
    "cash_r", "gw_r", "inv_r", "rec_r", "capex_dep", "rev_per_emp",
    "float_r", "tsy_r", "accrual",
]
LEVEL_KINDS = ("yoy1", "yoy2", "yoy3", "chg2y", "chg3y",
               "accel", "recovery3", "dip3", "vshape3", "high4")
RATIO_KINDS = ("chg1", "chg2y", "chg3y")

RAW_NUMERIC = [
    "revenue", "operating_income", "net_income", "adjusted_eps", "gross_profit",
    "sga", "rnd_expenses", "cf_operating", "capex", "depreciation", "num_employees",
    "total_assets", "total_liabilities", "net_assets", "cash", "inventories",
    "trade_receivables", "goodwill", "ibd_current", "ibd_noncurrent",
    "short_term_loans", "current_portion_lt_loans", "long_term_loans",
    "bonds_payable", "current_portion_bonds_payable", "commercial_papers",
    "float_shares", "treasury_shares_count", "shares_issued",
]
GUARDS = ["accounting_standard", "basis"]


def sym(a: pd.Series, b: pd.Series) -> pd.Series:
    """対称変化率 (a-b)/((|a|+|b|)/2)。符号をまたいでも定義でき、[-2, 2] に収まる。"""
    den = (a.abs() + b.abs()) / 2.0
    return ((a - b) / den).where(den > 0)


def _num(s: Optional[pd.Series], n: int) -> pd.Series:
    if s is None:
        return pd.Series(np.nan, index=range(n), dtype=float)
    return pd.to_numeric(s, errors="coerce").astype(float)


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).where(b > 0)


def load_fin(path: str = FIN) -> pd.DataFrame:
    return pd.read_parquet(path)


def annual_panel(fin: pd.DataFrame) -> pd.DataFrame:
    """
    1行 = (Code, 年度) の年次パネル。y0 の値・派生比率と、y1〜y3 の同じ列。

    同じ年度に複数の書類（訂正報告書など）があれば **最初の提出** を使う
    （訂正後の値で過去を上書きしない。時点整合の保守側）。
    """
    f = fin.copy()
    n = len(f)
    f[KEY] = f[KEY].astype(str)
    f["fiscal_year"] = pd.to_numeric(f["fiscal_year"], errors="coerce")
    f["submit_date"] = pd.to_datetime(f.get("submit_date"), errors="coerce")
    f = f.dropna(subset=["fiscal_year"])
    f["fiscal_year"] = f["fiscal_year"].astype(int)
    for c in GUARDS:
        f[c] = f[c].astype("string").fillna("").astype(str) if c in f.columns else ""
    for c in RAW_NUMERIC:
        f[c] = _num(f.get(c), len(f)).to_numpy() if c in f.columns else np.nan
    f = (f.sort_values([KEY, "fiscal_year", "submit_date"], na_position="last")
          .drop_duplicates([KEY, "fiscal_year"], keep="first")
          .reset_index(drop=True))

    # --- 派生の水準 ---
    has_bs = f["total_liabilities"].notna() | f["total_assets"].notna()
    ibd = f[["ibd_current", "ibd_noncurrent"]]
    loans = f[["short_term_loans", "current_portion_lt_loans", "long_term_loans",
               "bonds_payable", "current_portion_bonds_payable", "commercial_papers"]]
    debt = ibd.sum(axis=1, min_count=1)
    debt = debt.where(ibd.notna().any(axis=1), loans.sum(axis=1, min_count=1))
    f["debt"] = debt.fillna(0.0).where(has_bs)
    f["goodwill"] = f["goodwill"].fillna(0.0).where(has_bs)

    # --- 比率 ---
    rev, ta = f["revenue"], f["total_assets"]
    f["opm"] = _ratio(f["operating_income"], rev)
    f["gpm"] = _ratio(f["gross_profit"], rev)
    f["sga_r"] = _ratio(f["sga"], rev)
    f["rnd_r"] = _ratio(f["rnd_expenses"], rev)
    f["ocf_m"] = _ratio(f["cf_operating"], rev)
    f["roe"] = _ratio(f["net_income"], f["net_assets"])
    f["roa"] = _ratio(f["net_income"], ta)
    f["eq_r"] = _ratio(f["net_assets"], ta)
    f["debt_r"] = _ratio(f["debt"], ta)
    f["cash_r"] = _ratio(f["cash"], ta)
    f["gw_r"] = _ratio(f["goodwill"], ta)
    f["inv_r"] = _ratio(f["inventories"], rev)
    f["rec_r"] = _ratio(f["trade_receivables"], rev)
    f["capex_dep"] = _ratio(f["capex"], f["depreciation"])
    f["rev_per_emp"] = _ratio(rev, f["num_employees"])
    f["float_r"] = _ratio(f["float_shares"], f["shares_issued"])
    f["tsy_r"] = _ratio(f["treasury_shares_count"], f["shares_issued"])
    f["accrual"] = ((f["net_income"] - f["cf_operating"]) / ta).where(ta > 0)

    vals = LEVELS + RATIOS
    base = f[[KEY, "fiscal_year"] + GUARDS + vals]
    out = f[[KEY, "fiscal_year", "submit_date"] + GUARDS + vals].copy()
    for k in LAGS:
        lag = base.copy()
        lag["fiscal_year"] = lag["fiscal_year"] + k
        lag = lag.rename(columns={c: f"{c}_y{k}" for c in GUARDS + vals})
        out = out.merge(lag, on=[KEY, "fiscal_year"], how="left")
        # 会計基準・連結/単体が違う年度とは比べない
        bad = pd.Series(False, index=out.index)
        for g in GUARDS:
            bad |= out[f"{g}_y{k}"].notna() & (out[f"{g}_y{k}"] != out[g])
        out.loc[bad, [f"{c}_y{k}" for c in vals]] = np.nan
        out = out.drop(columns=[f"{g}_y{k}" for g in GUARDS])
    # 提出日の翌日から知りえたとする
    out["avail_date"] = out["submit_date"].dt.normalize() + pd.Timedelta(days=1)
    return out


def feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """年次パネルの各行に、ed_* の特徴量を付けて返す。"""
    out: Dict[str, pd.Series] = {}
    for c in LEVELS:
        y0, y1, y2, y3 = panel[c], panel[f"{c}_y1"], panel[f"{c}_y2"], panel[f"{c}_y3"]
        yoy1, yoy2, yoy3 = sym(y0, y1), sym(y1, y2), sym(y2, y3)
        out[f"ed_{c}_yoy1"] = yoy1
        out[f"ed_{c}_yoy2"] = yoy2
        out[f"ed_{c}_yoy3"] = yoy3
        out[f"ed_{c}_chg2y"] = sym(y0, y2)
        out[f"ed_{c}_chg3y"] = sym(y0, y3)
        out[f"ed_{c}_accel"] = yoy1 - yoy2
        peak3 = pd.concat([y1, y2, y3], axis=1).max(axis=1)
        prior3 = pd.concat([y2, y3], axis=1).max(axis=1)
        all4 = y0.notna() & y1.notna() & y2.notna() & y3.notna()
        out[f"ed_{c}_recovery3"] = sym(y0, peak3).where(all4)
        out[f"ed_{c}_dip3"] = sym(y1, prior3).clip(upper=0.0).where(all4)
        out[f"ed_{c}_vshape3"] = ((y1 < prior3) & (y0 > y1)).astype(float).where(all4)
        out[f"ed_{c}_high4"] = (y0 > peak3).astype(float).where(all4)
    for r in RATIOS:
        y0 = panel[r]
        out[f"ed_{r}"] = y0
        out[f"ed_{r}_chg1"] = y0 - panel[f"{r}_y1"]
        out[f"ed_{r}_chg2y"] = y0 - panel[f"{r}_y2"]
        out[f"ed_{r}_chg3y"] = y0 - panel[f"{r}_y3"]
    feats = pd.DataFrame(out, index=panel.index)
    keep = panel[[KEY, "fiscal_year", "avail_date"]]
    return pd.concat([keep, feats], axis=1)


def columns(group: str = "all") -> List[str]:
    """
    グループ名から列名の一覧。
      core        売上・営業利益・純利益・EPS の軌道（40列）
      detail      それ以外の水準項目の軌道（140列）
      ratio       比率の水準（18列）
      ratio_chg   比率の年ごとの差（54列）
      all         全部（252列）
    """
    groups = {
        "core": [f"ed_{c}_{k}" for c in CORE for k in LEVEL_KINDS],
        "detail": [f"ed_{c}_{k}" for c in LEVELS if c not in CORE for k in LEVEL_KINDS],
        "ratio": [f"ed_{r}" for r in RATIOS],
        "ratio_chg": [f"ed_{r}_{k}" for r in RATIOS for k in RATIO_KINDS],
    }
    if group == "all":
        return sum(groups.values(), [])
    if group not in groups:
        raise KeyError(f"未知のグループ: {group}. 利用可能: {sorted(groups)} / all")
    return groups[group]


def attach(frame: pd.DataFrame, feats: pd.DataFrame, *, code_col: str = "Code",
           date_col: str = "Date", stale_days: int = STALE_DAYS) -> pd.DataFrame:
    """
    各行に「avail_date ≤ Date」の最新の年度の特徴量を付ける。
    y0 が stale_days より古い行は全部 NaN（開示が止まっている）。
    行の順序は元のまま返す。
    """
    cols = [c for c in feats.columns if c.startswith("ed_")]
    right = feats.dropna(subset=["avail_date"]).sort_values("avail_date")
    right = right.rename(columns={KEY: code_col})
    right[code_col] = right[code_col].astype(str)
    left = frame[[code_col, date_col]].copy()
    left["_i"] = np.arange(len(left))
    left[date_col] = pd.to_datetime(left[date_col])
    left[code_col] = left[code_col].astype(str)
    left = left.sort_values(date_col)
    m = pd.merge_asof(left, right[[code_col, "avail_date", "fiscal_year"] + cols],
                      left_on=date_col, right_on="avail_date", by=code_col,
                      direction="backward", allow_exact_matches=True)
    stale = (m[date_col] - m["avail_date"]).dt.days > stale_days
    m.loc[stale | m["avail_date"].isna(), cols + ["fiscal_year"]] = np.nan
    m = m.sort_values("_i").set_index(frame.index)
    add = m[cols + ["fiscal_year"]].astype(float).rename(columns={"fiscal_year": "ed_fiscal_year"})
    return pd.concat([frame.drop(columns=[c for c in add.columns if c in frame.columns]), add],
                     axis=1)


def build(frame: pd.DataFrame, path: str = FIN, **kw) -> pd.DataFrame:
    """edinet_fin.parquet を読んで frame に ed_* を付ける（無ければ NaN の列だけ足す）。"""
    if not os.path.exists(path):
        out = frame.copy()
        for c in columns("all"):
            out[c] = np.nan
        out["ed_fiscal_year"] = np.nan
        return out
    return attach(frame, feature_frame(annual_panel(load_fin(path))), **kw)
