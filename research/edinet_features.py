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
EPS は `adjusted_eps`（分割調整済み）を使う。株数は `split_adjustment_factor`
で分割調整するか、比率（自己株比率）にして分割の影響を消す。
`float_shares` は EDINET DB では「発行済 − 自己株」（取得済み 582行で 100% 一致）
であって浮動株ではない。`outstanding_adj`（自己株控除後の株数）として使う。

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

#: 依頼の中核（売上・営業利益・純利益・EPS）
CORE = ["revenue", "operating_income", "net_income", "adjusted_eps"]
#: J-Quants に無い明細（損益・貸借・CF）
DETAIL = [
    "gross_profit", "sga", "rnd_expenses", "cf_operating", "capex", "depreciation",
    "total_assets", "net_assets", "cash", "inventories", "trade_receivables",
    "trade_payables", "debt", "goodwill", "ppe", "intangible_assets",
    "investment_securities", "retained_earnings", "interest_expenses",
    "extraordinary_net", "impairment_loss",
]
#: 資本政策（株数は分割調整済み。`*_adj`）
CAPITAL = [
    "shares_adj", "treasury_adj", "outstanding_adj", "cash_dividends_paid",
    "cross_shareholding_total_book_value", "directors_shares_adj",
]
#: 人的資本（有報の非財務情報）。EDINET DB にしか無い。
#: J-Quants は従業員数も平均年収も平均年齢も返さない
PEOPLE = ["num_employees", "temp_employees", "avg_annual_salary",
          "avg_age", "avg_tenure_years"]
#: 水準を持つ項目。この軌道（前年比など）を特徴量にする
LEVELS: List[str] = CORE + DETAIL + CAPITAL + PEOPLE

#: 比率（水準と、その年ごとの差を特徴量にする）
RATIOS_QUALITY = [
    "opm", "gpm", "sga_r", "rnd_r", "ocf_m", "roe", "roa", "accrual",
    "int_r", "extra_r", "tax_r", "impair_r",
]
RATIOS_BALANCE = [
    "eq_r", "debt_r", "cash_r", "gw_r", "inv_r", "rec_r", "pay_r", "capex_dep",
    "ppe_r", "intang_r", "secu_r", "re_r",
]
RATIOS_CAPITAL = ["tsy_r", "cancel_r", "div_ni", "xhold_r", "dir_r"]
RATIOS_PEOPLE = ["rev_per_emp", "temp_r", "salary_rev"]
#: 2026-09-22 に足した固有項目（運用者の判断「edinet DB固有のデータが
#: あるはずなので、代替のとか以前に追加はします」）。
#: 充足率は取得済み817行での実測。
#:   tsr            78.8%  株主総利回り。EDINET DB だけが持つ
#:   dbo_r          69.4%  退職給付債務 ÷ 総資産
#:   bad_debt_r     57.8%  貸倒引当金 ÷ 売上債権
#:   dta_r          51.8%  繰延税金資産 ÷ 総資産
#:   dir_pay_r      52.3%  役員報酬総額 ÷ 売上
#:   dir_pay_head   51.3%  1人あたり役員報酬（円）
#:   fem_dir_r      47.7%  女性役員比率
#: ghg_*（1.5〜4.9%）と gender_pay_gap_*（17.1%）は充足率が低すぎるので
#: 入れない。いま入れても全欠測に近く、判定できない
RATIOS_EXTRA = ["tsr", "dbo_r", "bad_debt_r", "dta_r",
                "dir_pay_r", "dir_pay_head", "fem_dir_r"]
RATIOS: List[str] = (RATIOS_QUALITY + RATIOS_BALANCE + RATIOS_CAPITAL
                     + RATIOS_PEOPLE + RATIOS_EXTRA)

#: 時価総額と組み合わせる（attach のとき。frame の market_cap は億円）
MCAP = ["fcf_yield", "netcash_mcap", "ev_ebitda", "xhold_mcap", "div_yield",
        "buyback_yield", "re_mcap"]
MCAP_UNIT = 1e8

LEVEL_KINDS = ("yoy1", "yoy2", "yoy3", "chg2y", "chg3y",
               "accel", "recovery3", "dip3", "vshape3", "high4")
RATIO_KINDS = ("chg1", "chg2y", "chg3y")

RAW_NUMERIC = [
    "revenue", "operating_income", "net_income", "adjusted_eps", "gross_profit",
    "sga", "rnd_expenses", "cf_operating", "capex", "depreciation", "num_employees",
    "total_assets", "total_liabilities", "net_assets", "cash", "inventories",
    "trade_receivables", "trade_payables", "goodwill", "ibd_current", "ibd_noncurrent",
    "short_term_loans", "current_portion_lt_loans", "long_term_loans",
    "bonds_payable", "current_portion_bonds_payable", "commercial_papers",
    "float_shares", "treasury_shares_count", "shares_issued", "split_adjustment_factor",
    "treasury_cancelled_count", "treasury_stock", "cash_dividends_paid",
    "cross_shareholding_total_book_value", "directors_shares_held",
    "directors_ownership_ratio", "temp_employees", "avg_annual_salary",
    "ppe", "intangible_assets", "investment_securities", "short_term_securities",
    "retained_earnings", "interest_expenses", "extraordinary_income",
    "extraordinary_loss", "impairment_loss", "effective_tax_rate",
    # 2026-09-22 に足した固有項目
    "total_shareholder_return", "avg_age", "avg_tenure_years",
    "net_defined_benefit_liability", "deferred_tax_assets",
    "allowance_for_doubtful_accounts", "director_remuneration_total",
    "director_remuneration_headcount", "female_director_ratio",
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
    f["impairment_loss"] = f["impairment_loss"].fillna(0.0).where(has_bs)
    # 特別損益は日本基準にしか無い。日本基準で省かれていれば「無かった」= 0
    jp_pl = (f["accounting_standard"] == "JP") & f["net_income"].notna()
    ex_in = f["extraordinary_income"].where(~jp_pl, f["extraordinary_income"].fillna(0.0))
    ex_lo = f["extraordinary_loss"].where(~jp_pl, f["extraordinary_loss"].fillna(0.0))
    f["extraordinary_net"] = ex_in - ex_lo
    # 株数は分割調整（adjusted_eps = eps / factor なので、株数は × factor）
    fac = f["split_adjustment_factor"].where(f["split_adjustment_factor"] > 0, 1.0).fillna(1.0)
    f["shares_adj"] = f["shares_issued"] * fac
    f["treasury_adj"] = f["treasury_shares_count"] * fac
    f["outstanding_adj"] = f["float_shares"] * fac      # 発行済 − 自己株（分割調整）
    f["directors_shares_adj"] = f["directors_shares_held"] * fac
    f["treasury_stock"] = f["treasury_stock"].abs()

    # --- 比率 ---
    rev, ta, ni = f["revenue"], f["total_assets"], f["net_income"]
    f["opm"] = _ratio(f["operating_income"], rev)
    f["gpm"] = _ratio(f["gross_profit"], rev)
    f["sga_r"] = _ratio(f["sga"], rev)
    f["rnd_r"] = _ratio(f["rnd_expenses"], rev)
    f["ocf_m"] = _ratio(f["cf_operating"], rev)
    f["roe"] = _ratio(ni, f["net_assets"])
    f["roa"] = _ratio(ni, ta)
    f["accrual"] = ((ni - f["cf_operating"]) / ta).where(ta > 0)
    f["int_r"] = _ratio(f["interest_expenses"], rev)
    f["extra_r"] = (f["extraordinary_net"] / ta).where(ta > 0)
    f["tax_r"] = f["effective_tax_rate"]
    f["impair_r"] = _ratio(f["impairment_loss"], ta)
    f["eq_r"] = _ratio(f["net_assets"], ta)
    f["debt_r"] = _ratio(f["debt"], ta)
    f["cash_r"] = _ratio(f["cash"], ta)
    f["gw_r"] = _ratio(f["goodwill"], ta)
    f["inv_r"] = _ratio(f["inventories"], rev)
    f["rec_r"] = _ratio(f["trade_receivables"], rev)
    f["pay_r"] = _ratio(f["trade_payables"], rev)
    f["capex_dep"] = _ratio(f["capex"], f["depreciation"])
    f["ppe_r"] = _ratio(f["ppe"], ta)
    f["intang_r"] = _ratio(f["intangible_assets"], ta)
    f["secu_r"] = _ratio(f["investment_securities"], ta)
    f["re_r"] = (f["retained_earnings"] / f["net_assets"]).where(f["net_assets"] > 0)
    f["tsy_r"] = _ratio(f["treasury_shares_count"], f["shares_issued"])
    f["cancel_r"] = _ratio(f["treasury_cancelled_count"], f["shares_issued"])
    f["div_ni"] = (f["cash_dividends_paid"] / ni).where(ni > 0)
    f["xhold_r"] = _ratio(f["cross_shareholding_total_book_value"], ta)
    f["dir_r"] = f["directors_ownership_ratio"]
    f["rev_per_emp"] = _ratio(rev, f["num_employees"])
    f["temp_r"] = _ratio(f["temp_employees"], f["num_employees"])
    f["salary_rev"] = ((f["avg_annual_salary"] * f["num_employees"]) / rev).where(rev > 0)
    # --- 2026-09-22 に足した固有項目 --- #
    # 株主総利回り。そのまま比率なので割らない
    f["tsr"] = f["total_shareholder_return"]
    # 金額は会社の大きさで決まってしまうので、必ず何かで割る。
    # 貸倒引当金は総資産ではなく**売上債権**で割る。引当が厚いかは
    # 債権の質の話であって、会社の規模の話ではない。
    #
    # **符号に注意。** 貸倒引当金は控除項目なので負で入っている
    # （実測 472行のうち 99.2% が負、中央値 -6,600万円）。そのまま割ると
    # 「引当が厚いほど小さい値」になって読み違える。絶対値にして
    # 「厚いほど大きい」に揃える
    f["dbo_r"] = _ratio(f["net_defined_benefit_liability"], ta)
    f["dta_r"] = _ratio(f["deferred_tax_assets"], ta)
    f["bad_debt_r"] = _ratio(f["allowance_for_doubtful_accounts"].abs(),
                             f["trade_receivables"])
    f["dir_pay_r"] = _ratio(f["director_remuneration_total"], rev)
    # 1人あたり役員報酬は、割ったあとは規模に依存しない水準になる
    f["dir_pay_head"] = _ratio(f["director_remuneration_total"],
                               f["director_remuneration_headcount"])
    f["fem_dir_r"] = f["female_director_ratio"]
    # 時価総額と組み合わせる元の値（attach で割る）
    f["_fcf"] = f["cf_operating"] - f["capex"]
    f["_netcash"] = f["cash"] + f["short_term_securities"].fillna(0.0) - f["debt"]
    f["_ebitda"] = f["operating_income"] + f["depreciation"]
    f["_tsy_value_chg"] = f["treasury_stock"] - f.groupby(KEY)["treasury_stock"].shift(1)

    vals = LEVELS + RATIOS
    bases = ["_fcf", "_netcash", "_ebitda", "_tsy_value_chg", "cross_shareholding_total_book_value",
             "cash_dividends_paid", "retained_earnings"]
    base = f[[KEY, "fiscal_year"] + GUARDS + vals]
    out = f[[KEY, "fiscal_year", "submit_date"] + GUARDS + vals
            + [b for b in bases if b not in vals]].copy()
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
    keep = panel[[KEY, "fiscal_year", "avail_date", "_fcf", "_netcash", "_ebitda",
                  "_tsy_value_chg", "cross_shareholding_total_book_value",
                  "cash_dividends_paid", "retained_earnings"]]
    return pd.concat([keep, feats], axis=1)


def columns(group: str = "all") -> List[str]:
    """
    グループ名から列名の一覧。
      core        売上・営業利益・純利益・EPS の軌道
      detail      J-Quants に無い明細（損益・貸借・CF）の軌道
      capital     資本政策（株数・自己株・浮動株・配当・政策保有株・役員持株）の軌道
      people      人的資本（従業員・臨時従業員・平均給与）の軌道
      ratio       比率の水準
      ratio_chg   比率の年ごとの差
      mcap        時価総額と組み合わせた水準（attach のときだけ計算）
      all         全部
    """
    groups = {
        "core": [f"ed_{c}_{k}" for c in CORE for k in LEVEL_KINDS],
        "detail": [f"ed_{c}_{k}" for c in DETAIL for k in LEVEL_KINDS],
        "capital": [f"ed_{c}_{k}" for c in CAPITAL for k in LEVEL_KINDS],
        "people": [f"ed_{c}_{k}" for c in PEOPLE for k in LEVEL_KINDS],
        "ratio": [f"ed_{r}" for r in RATIOS],
        "ratio_chg": [f"ed_{r}_{k}" for r in RATIOS for k in RATIO_KINDS],
        "mcap": [f"ed_{r}" for r in MCAP],
    }
    if group == "all":
        return sum(groups.values(), [])
    if group not in groups:
        raise KeyError(f"未知のグループ: {group}. 利用可能: {sorted(groups)} / all")
    return groups[group]


BASES = ["_fcf", "_netcash", "_ebitda", "_tsy_value_chg", "cross_shareholding_total_book_value",
         "cash_dividends_paid", "retained_earnings"]


def attach(frame: pd.DataFrame, feats: pd.DataFrame, *, code_col: str = "Code",
           date_col: str = "Date", stale_days: int = STALE_DAYS,
           mcap_col: str = "market_cap") -> pd.DataFrame:
    """
    各行に「avail_date ≤ Date」の最新の年度の特徴量を付ける。
    y0 が stale_days より古い行は全部 NaN（開示が止まっている）。
    frame に時価総額（億円）があれば、それと組み合わせた MCAP も付ける。
    行の順序は元のまま返す。
    """
    cols = [c for c in feats.columns if c.startswith("ed_")]
    bases = [b for b in BASES if b in feats.columns]
    right = feats.dropna(subset=["avail_date"]).sort_values("avail_date")
    right = right.rename(columns={KEY: code_col})
    right[code_col] = right[code_col].astype(str)
    left = frame[[code_col, date_col] + ([mcap_col] if mcap_col in frame.columns else [])].copy()
    left["_i"] = np.arange(len(left))
    left[date_col] = pd.to_datetime(left[date_col])
    left[code_col] = left[code_col].astype(str)
    left = left.sort_values(date_col)
    m = pd.merge_asof(left, right[[code_col, "avail_date", "fiscal_year"] + cols + bases],
                      left_on=date_col, right_on="avail_date", by=code_col,
                      direction="backward", allow_exact_matches=True)
    stale = (m[date_col] - m["avail_date"]).dt.days > stale_days
    m.loc[stale | m["avail_date"].isna(), cols + bases + ["fiscal_year"]] = np.nan
    if mcap_col in m.columns:
        mc = pd.to_numeric(m[mcap_col], errors="coerce").astype(float) * MCAP_UNIT
        mc = mc.where(mc > 0)
        m["ed_fcf_yield"] = m["_fcf"] / mc
        m["ed_netcash_mcap"] = m["_netcash"] / mc
        ev = mc + (m["_netcash"] * -1.0)          # 時価総額 + 有利子負債 − 現金
        m["ed_ev_ebitda"] = (ev / m["_ebitda"]).where(m["_ebitda"] > 0)
        m["ed_xhold_mcap"] = m["cross_shareholding_total_book_value"] / mc
        m["ed_div_yield"] = m["cash_dividends_paid"] / mc
        m["ed_buyback_yield"] = m["_tsy_value_chg"] / mc
        m["ed_re_mcap"] = m["retained_earnings"] / mc
    else:
        for r in MCAP:
            m[f"ed_{r}"] = np.nan
    cols = cols + [f"ed_{r}" for r in MCAP]
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
