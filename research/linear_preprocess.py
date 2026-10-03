#!/usr/bin/env python3
"""
線形（ロジスティック回帰）・MLP 用の前処理 v2（2026-10-03、docs/MODEL_LINEAR_PREPROCESSING.md）。

木は閾値分割なので単調変換に不変だが、線形・MLP は列の分布にそのまま影響される。
いまの前処理（tuning_multi.preprocess = 中央値補完 + 欠損指示子 + 標準化 + one-hot）は
裾の重い列（239列中115列）をそのまま標準化していて、外れ値が |z| で最大 148 に達する。
ここでは列の**型**ごとに変換を分ける。

  符号つき・裾が重い   0.5/99.5%点で切る → asinh(x / (四分位範囲/2)) → 標準化
  符号つき             0.1/99.9%点で切る → 標準化
  正の比率・裾が重い   99.5%点で切る → log1p → 標準化
  正の比率（有界）     0.1/99.9%点で切る → 標準化
  対称変化率（_sym）   0.5/99.5%点で切る → 標準化
  件数・日数           0.1/99.9%点で切る → 標準化（lvs_n_* は log1p）
  小さい整数・旗・順位 そのまま → 標準化
  カテゴリ             one-hot（v1 と同じ）

欠損は意味で分ける。「無い＝0」の列は 0、「ずっと無い」の日数列は上限（400 / 5 / 訓練の最大）、
旗は最頻値、それ以外は中央値。欠損のあった列には指示子を残す（v1 と同じ）。

**統計はすべて fit で渡された訓練側だけから決める**（分位点・四分位範囲・中央値・最大）。
ColumnTransformer / Pipeline の中なので fold ごとに訓練側だけで決まり、検証側の分布は漏れない。

列の型は**名前で決めて固定する**（この表）。データで決めると fold ごとに型が変わりうる。
新しい列は既定（符号つき・切るだけ・中央値）に落ちるが、tests/test_linear_preprocess.py が
本番の列に既定落ちが無いことを固定しているので、列を足したらここに型を書き足す。
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

CATEGORICAL = ("s33_code", "s17_code", "mkt_code", "scalecat_code")

# --- 列の型（docs/MODEL_LINEAR_PREPROCESSING.md の表から） --------------------------------- #

SIGNED_HEAVY = (
    "ROA_accel", "ROA_chg", "ROA_chg1", "ROA_chg2",
    "ROA_chg3", "ROA_chg_3q", "ROA_q0", "ROA_q1",
    "ROA_q2", "ROA_q3", "ROE_accel", "ROE_chg",
    "ROE_chg1", "ROE_chg2", "ROE_chg3", "ROE_chg_3q",
    "ROE_q0", "ROE_q1", "ROE_q2", "ROE_q3",
    "accruals", "bps_yoy", "bps_yoy_p1", "break_margin",
    "cff_mcap", "cfi_mcap", "cfo_to_op", "cfo_yield",
    "earnings_yield", "eps_growth_accel", "eps_growth_chg", "eps_growth_chg1",
    "eps_growth_chg2", "eps_growth_chg3", "eps_growth_chg_3q", "eps_growth_q0",
    "eps_growth_q1", "eps_growth_q2", "eps_growth_q3", "equity_ratio_accel",
    "equity_ratio_chg", "equity_ratio_chg1", "equity_ratio_chg2", "equity_ratio_chg3",
    "equity_ratio_chg_3q", "equity_turnover_chg1", "equity_turnover_chg2", "fcf_yield",
    "growth250_ret_120", "guidance_op_growth", "guidance_revision", "inv_busco",
    "jq_fwd_earnings_yield", "jq_fwdeps", "jq_fwdroe", "jq_roe_gap",
    "lvs_ratio_chg", "net_margin", "op_margin_accel", "op_margin_chg",
    "op_margin_chg1", "op_margin_chg2", "op_margin_chg3", "op_margin_chg_3q",
    "op_margin_q0", "op_margin_q1", "op_margin_q2", "op_margin_q3",
    "ordinary_margin", "rel_sector_20", "ret_20d", "rev_pct",
    "sales_growth_accel", "sales_growth_chg", "sales_growth_chg1", "sales_growth_chg2",
    "sales_growth_chg3", "sales_growth_chg_3q", "sales_growth_q0", "sales_growth_q1",
    "sales_growth_q2", "sales_growth_q3", "shares_yoy", "shares_yoy_p1",
    "sustainable_growth", "sustainable_growth_chg1", "sustainable_growth_chg2", "xh_net",
    "xh_net_mc",
)

SIGNED = (
    "gold_ret_120", "gold_ret_20", "growth250_ret_20", "inv_busco_4w",
    "inv_foreign", "inv_foreign_4w", "inv_indiv", "inv_indiv_4w",
    "inv_prop", "inv_prop_4w", "inv_trust", "inv_trust_4w",
    "nk225_ret_120", "nk225_ret_20", "risk_off_20", "sector_ret_120",
    "sector_ret_20", "sector_vs_topix_20", "topix_ma200_gap", "topix_ret_120",
    "topix_ret_20",
)

POSITIVE_HEAVY = (
    "alert_slratio", "asset_turnover", "cash_mcap", "credit_ratio",
    "div_yield", "equity_turnover", "jq_fwdper", "jq_per_gap",
    "payout_ratio", "pbr", "peg", "per",
    "psr", "sales_yield", "topix_vol_20", "vol_20d",
    "vol_gap_ratio", "vol_rel_mkt", "vol_rel_sector", "vol_updown",
    "volume_trend", "xh_bookval", "xh_bookval_mc", "xh_dec_amt",
    "xh_inc_cost", "xh_iss",
)

POSITIVE = (
    "alert_longoutratio", "alert_shrtoutratio", "book_yield", "close_position",
    "equity_ratio_q0", "equity_ratio_q1", "equity_ratio_q2", "equity_ratio_q3",
    "log_market_cap", "log_trading_value", "lvs_ratio", "mjr_conc",
    "mjr_indiv", "mjr_top1", "mjr_top10", "mjr_trust",
    "r_high", "r_high_3m", "r_high_6m", "short_ratio",
    "short_ratio_20", "vol_rel_long", "vol_rel_short",
)

COUNT_DAYS = (
    "alert_days", "base_length", "days_since_disc", "days_since_divrev",
    "days_since_fy", "days_since_rev", "days_to_earn", "listing_years",
    "lvs_days", "lvs_n_20", "lvs_n_250", "lvs_n_60",
    "mjr_days", "xh_days",
)

SMALL_INT = (
    "ROA_pos_ratio", "ROA_up_streak", "ROE_pos_ratio", "ROE_up_streak",
    "cap_band", "eps_growth_pos_ratio", "eps_growth_sym_pos_ratio", "eps_growth_sym_up_streak",
    "eps_growth_up_streak", "equity_ratio_pos_ratio", "equity_ratio_up_streak", "mjr_n",
    "op_margin_pos_ratio", "op_margin_up_streak", "rev_dn_n_250", "rev_n_250",
    "rev_n_60", "rev_up_n_250", "sales_growth_pos_ratio", "sales_growth_sym_pos_ratio",
    "sales_growth_sym_up_streak", "sales_growth_up_streak",
)

FLAGS = (
    "acct_ifrs", "div_up", "eps_growth_turn", "fund_complete",
    "has_dividend", "rev_up", "sales_growth_turn",
)

RANK = (
    "progress_pct",
)

#: 欠測が「無い＝0」を意味する列（信用規制の比率、大量保有の比率、政策保有の件数・金額、業績予想の修正）
FILL_ZERO = frozenset({
    "alert_longoutratio", "alert_shrtoutratio", "alert_slratio", "lvs_ratio", "lvs_ratio_chg",
    "xh_iss", "xh_bookval", "xh_dec_amt", "xh_inc_cost", "xh_net", "xh_bookval_mc", "xh_net_mc",
    "guidance_revision", "rev_pct", "rev_up",
})
#: 欠測が「まだ無い・ずっと無い」を意味する日数列。値は埋める上限（"max" は訓練側の最大）
FILL_CAP: Dict[str, object] = {
    "days_since_rev": 400.0, "days_since_divrev": 400.0, "listing_years": 5.0,
    "alert_days": "max", "lvs_days": "max",
}
#: 件数のうち右に裾を引くもの（log1p）
COUNT_LOG_PREFIX = ("lvs_n_",)

EPS = 1e-9


def kind_of(col: str) -> str:
    """列の型。既定（名前がどの表にも無い）は "default"（符号つき・切るだけ）。"""
    if col in CATEGORICAL:
        return "cat"
    if col in FLAGS:
        return "flag"
    if col in SMALL_INT:
        return "small_int"
    if col in RANK:
        return "rank"
    if "_sym" in col:
        return "sym"
    if col in COUNT_DAYS:
        return "count_log" if col.startswith(COUNT_LOG_PREFIX) else "count"
    if col in POSITIVE_HEAVY:
        return "pos_heavy"
    if col in POSITIVE:
        return "pos"
    if col in SIGNED_HEAVY:
        return "signed_heavy"
    if col in SIGNED:
        return "signed"
    return "default"


def fill_of(col: str, kind: str):
    """欠損の埋め方: "median" / "mode" / "max" / 数値。"""
    if col in FILL_ZERO:
        return 0.0
    if col in FILL_CAP:
        return FILL_CAP[col]
    if kind == "flag":
        return "mode"
    return "median"


def unclassified(cols: Sequence[str]) -> List[str]:
    """型の表に無い列（既定に落ちる列）。本番の列では空のはず。"""
    return [c for c in cols if kind_of(c) == "default"]


def plan(cols: Sequence[str]) -> List[Tuple[str, str, object]]:
    """(列, 型, 欠損の埋め方) の一覧。docs とテスト用。"""
    out = []
    for c in cols:
        k = kind_of(c)
        out.append((c, k, fill_of(c, k)))
    return out


# --- 変換器（NaN は素通し。統計は fit の訓練側だけ） ---------------------------------------- #

def _as2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


class ClipQuantile(BaseEstimator, TransformerMixin):
    """訓練側の分位点で切る。lo / hi は 0〜1（None なら切らない）。"""

    def __init__(self, lo: Optional[float] = 0.001, hi: Optional[float] = 0.999):
        self.lo = lo
        self.hi = hi

    def fit(self, X, y=None):
        X = _as2d(X)
        # 全部欠測の列（提供が後から始まる列の初期の窓）は nanquantile が警告を出す。切らない扱いにする
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.lo_ = (np.nanquantile(X, self.lo, axis=0) if self.lo is not None
                        else np.full(X.shape[1], -np.inf))
            self.hi_ = (np.nanquantile(X, self.hi, axis=0) if self.hi is not None
                        else np.full(X.shape[1], np.inf))
        # 全部欠測の列は切らない
        self.lo_ = np.where(np.isnan(self.lo_), -np.inf, self.lo_)
        self.hi_ = np.where(np.isnan(self.hi_), np.inf, self.hi_)
        return self

    def transform(self, X):
        X = _as2d(X)
        return np.clip(X, self.lo_, self.hi_)





class Asinh(BaseEstimator, TransformerMixin):
    """asinh(x / s)。s は訓練側の四分位範囲の半分（0 なら EPS）。中央はほぼ線形、裾は対数で縮む。"""

    def fit(self, X, y=None):
        X = _as2d(X)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q1 = np.nanquantile(X, 0.25, axis=0)
            q3 = np.nanquantile(X, 0.75, axis=0)
        s = (q3 - q1) / 2.0
        s = np.where(np.isfinite(s) & (s > EPS), s, 1.0)   # 広がりが無い列はそのまま asinh
        self.scale_ = s
        return self

    def transform(self, X):
        return np.arcsinh(_as2d(X) / self.scale_)





class Log1p(BaseEstimator, TransformerMixin):
    """log1p(max(x, 0))。負の値（本来無い）は 0 に寄せる。NaN は素通し。"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _as2d(X)
        return np.log1p(np.where(np.isnan(X), np.nan, np.clip(X, 0.0, None)))





class Fill(BaseEstimator, TransformerMixin):
    """
    列ごとに決めた値で欠損を埋め、訓練側で欠損のあった列には指示子（0/1）を右に足す。

    fills は列ごとの "median" / "mode" / "max" / 数値。SimpleImputer(add_indicator=True) と同じく、
    指示子は訓練側に欠損があった列だけ（検証側にだけ欠損が出た列は埋めるだけ）。
    """

    def __init__(self, fills: Sequence[object]):
        self.fills = fills

    def fit(self, X, y=None):
        X = _as2d(X)
        fills = list(self.fills)
        if X.shape[1] != len(fills):
            raise ValueError(f"列数 {X.shape[1]} と埋め方の数 {len(fills)} が違う")
        vals = np.zeros(X.shape[1])
        with np.errstate(all="ignore"):
            for j, f in enumerate(fills):
                col = X[:, j]
                ok = col[~np.isnan(col)]
                if isinstance(f, str):
                    if f == "median":
                        v = np.median(ok) if len(ok) else 0.0
                    elif f == "max":
                        v = ok.max() if len(ok) else 0.0
                    elif f == "mode":
                        if len(ok):
                            u, n = np.unique(ok, return_counts=True)
                            v = u[np.argmax(n)]
                        else:
                            v = 0.0
                    else:
                        raise ValueError(f"知らない埋め方: {f}")
                else:
                    v = float(f)
                vals[j] = v
        self.values_ = vals
        self.indicator_cols_ = np.where(np.isnan(X).any(axis=0))[0]
        return self

    def transform(self, X):
        X = _as2d(X)
        miss = np.isnan(X)
        out = np.where(miss, self.values_, X)
        if len(self.indicator_cols_):
            out = np.hstack([out, miss[:, self.indicator_cols_].astype(float)])
        return out





#: 型 -> 変換の列（Fill と標準化の前に当てるもの）
STEPS = {
    "signed_heavy": lambda: [ClipQuantile(0.005, 0.995), Asinh()],
    "signed": lambda: [ClipQuantile(0.001, 0.999)],
    "default": lambda: [ClipQuantile(0.001, 0.999)],
    "pos_heavy": lambda: [ClipQuantile(None, 0.995), Log1p()],
    "pos": lambda: [ClipQuantile(0.001, 0.999)],
    "sym": lambda: [ClipQuantile(0.005, 0.995)],
    "count": lambda: [ClipQuantile(0.001, 0.999)],
    "count_log": lambda: [Log1p()],
    "flag": lambda: [],
    "small_int": lambda: [],
    "rank": lambda: [],
}


def preprocess_v2(cols: Sequence[str]):
    """
    線形・MLP 用の前処理 v2 を ColumnTransformer で組む（tuning_multi.build から使う）。

    カテゴリは v1 と同じ one-hot。数値は型ごとの枝に分け、各枝は
    [切る / asinh / log1p] → Fill（欠損を意味で埋め、指示子を足す）→ StandardScaler。
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cols = list(cols)
    branches = []
    cat = [i for i, c in enumerate(cols) if c in CATEGORICAL]
    if cat:
        branches.append(("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat))
    by_kind: Dict[str, List[int]] = {}
    for i, c in enumerate(cols):
        k = kind_of(c)
        if k == "cat":
            continue
        by_kind.setdefault(k, []).append(i)
    for k in STEPS:                                  # 枝の順番を固定する（列の並びが再現できるように）
        idx = by_kind.get(k)
        if not idx:
            continue
        fills = [fill_of(cols[i], k) for i in idx]
        steps = STEPS[k]() + [Fill(fills), StandardScaler()]
        branches.append((k, make_pipeline(*steps), idx))
    return ColumnTransformer(branches)
