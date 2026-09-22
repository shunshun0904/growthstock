#!/usr/bin/env python3
"""
EDINET DB（有価証券報告書）の明細を、決算開示イベントに as-of で結び付ける。

## なぜ要るのか

J-Quants の `/fins/details` は本契約では 403 で、減価償却費・運転資本・
売上原価といった明細が取れない（docs/DATA_FIELDS.md）。そのため
要件にあった「税引前利益 → 減価償却費 → 運転資本 → 営業CF」という粒度の
グラフが組めなかった。EDINET DB はそれを持っている（docs/DATA_EDINETDB.md）。

## 粒度が違うものを混ぜるときの約束

EDINET は **年1回（有価証券報告書）** しか出ない。四半期のグラフに
そのまま混ぜると、四半期ごとに動く値と年1回しか動かない値が同じ顔で並ぶ。
そこで

  1. 年次のフローは4で割って「1四半期あたり」に直す（`span` = 4）
  2. 「その値が何年前の書類か」を `age_years` としてノード特徴量に持たせる
  3. 基準日から `STALE_DAYS` より古い書類しか無ければ、全部欠測にする

としてある。粒度の違いをモデルから見える形にするのが目的で、
隠して均すのが目的ではない。

## 時点整合

research/edinet_features.py と同じ規則に揃える。

  - 同じ事業年度に複数の書類（訂正報告書）があれば **最初の提出** を使う。
    訂正後の値で過去を上書きしない
  - `avail_date` = 提出日の翌日。提出が場中でも引け後でも、翌日から
    知りえたとする（保守側）
  - 会計基準（JP / IFRS）や連結・単体が変わった年度をまたぐ比較はしない
    ——ここでは水準しか使わないので、基準が変わった書類でも値は使うが、
    変わったこと自体を `basis_changed` として残す

なぜ edinet_features.annual_panel をそのまま呼ばないか: あちらは
別の研究ライン（新高値ブレイク）向けに項目と派生列が固定されていて、
税引前利益・売上原価を持っていない。規則だけ揃えて項目は独自に持つ。
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import schema

#: 基準日からこれより古い有価証券報告書しか無ければ欠測にする。
#: 有報は年1回なので、12か月＋猶予。research/edinet_features.py と同じ値
STALE_DAYS = 400

#: 生データの置き場所（他の parquet と同じ）
FIN_NAME = "edinet_fin.parquet"

#: 使う項目。schema が参照しているものだけを読む
FLOW_FIELDS: Tuple[str, ...] = schema.EDINET_FLOW_FIELDS
STOCK_FIELDS: Tuple[str, ...] = schema.EDINET_STOCK_FIELDS
ALL_FIELDS: Tuple[str, ...] = FLOW_FIELDS + STOCK_FIELDS

#: 会計基準・連結単体が変わったかを見る列
GUARDS = ("accounting_standard", "basis")


def find_parquet(data_dir: str) -> Optional[str]:
    p = os.path.join(data_dir, FIN_NAME)
    return p if os.path.exists(p) else None


def load(data_dir: str) -> Optional[pd.DataFrame]:
    """EDINET の年次財務を読む。無ければ None（グラフは粗いまま成立する）。"""
    p = find_parquet(data_dir)
    if not p:
        print(f"[edinet] {FIN_NAME} が無いので明細ノードは全部欠測になる")
        return None
    df = pd.read_parquet(p)
    print(f"[edinet] {len(df):,}行 / {df['jq_code'].nunique():,}社"
          if "jq_code" in df.columns else f"[edinet] {len(df):,}行")
    return df


def annual_panel(fin: pd.DataFrame) -> pd.DataFrame:
    """
    1行 = (銘柄コード, 事業年度)。使う項目と `avail_date` だけを持つ。

    同じ年度に複数の書類があれば最初の提出を採る。訂正報告書で過去を
    上書きすると、訂正が出る前の時点に未来の値が混ざる。
    """
    f = fin.copy()
    if "jq_code" not in f.columns:
        raise ValueError("edinet_fin.parquet に jq_code がありません")
    f["Code"] = f["jq_code"].astype(str).str.strip()
    f["fiscal_year"] = pd.to_numeric(f.get("fiscal_year"), errors="coerce")
    f["submit_date"] = pd.to_datetime(f.get("submit_date"), errors="coerce")
    f = f.dropna(subset=["fiscal_year", "submit_date"])
    f["fiscal_year"] = f["fiscal_year"].astype(int)

    for c in ALL_FIELDS:
        f[c] = (pd.to_numeric(f[c], errors="coerce") if c in f.columns
                else pd.Series(np.nan, index=f.index))
    for c in GUARDS:
        f[c] = (f[c].astype("string").fillna("").astype(str) if c in f.columns
                else "")

    f = (f.sort_values(["Code", "fiscal_year", "submit_date"])
          .drop_duplicates(["Code", "fiscal_year"], keep="first")
          .reset_index(drop=True))

    # 提出日の翌日から知りえたとする
    f["avail_date"] = f["submit_date"].dt.normalize() + pd.Timedelta(days=1)

    # 前年度の書類と基準が変わったか（水準は使うが、変わったことは残す）
    f = f.sort_values(["Code", "fiscal_year"]).reset_index(drop=True)
    g = f.groupby("Code", sort=False)
    changed = np.zeros(len(f), dtype=bool)
    for c in GUARDS:
        prev = g[c].shift(1)
        changed |= prev.notna().to_numpy() & (prev.to_numpy() != f[c].to_numpy())
    f["basis_changed"] = changed.astype(float)

    cols = ["Code", "fiscal_year", "avail_date", "basis_changed"] + list(ALL_FIELDS)
    out = f[cols].sort_values("avail_date").reset_index(drop=True)
    print(f"[edinet] 年次パネル {len(out):,}行 / {out['Code'].nunique():,}社 "
          f"{out['fiscal_year'].min()}〜{out['fiscal_year'].max()}期")
    return out


def asof_matrices(anchor_df: pd.DataFrame, disc_date_days: np.ndarray,
                  panel: Optional[pd.DataFrame],
                  stale_days: int = STALE_DAYS) -> Dict[str, np.ndarray]:
    """
    [アンカー, ラグ] の各位置に、その四半期の開示時点で読めた有報を当てる。

    disc_date_days は panel.build_asof_matrices が返す「その四半期の開示日」
    （1970-01-01 からの日数、欠測は NaN）。アンカー自身の日付ではなく
    各ラグの開示日を使うのは、過去の期のグラフに未来の有報を混ぜないため。

    返す辞書:
      e_<FIELD>   1四半期あたりに直した金額（ストックはそのまま）
      e_age_years その値が何年前の書類か
      e_basis_changed 会計基準が変わった書類か
    """
    n_anchors, n_lags = disc_date_days.shape
    out: Dict[str, np.ndarray] = {
        f"e_{f}": np.full((n_anchors, n_lags), np.nan) for f in ALL_FIELDS
    }
    out["e_age_years"] = np.full((n_anchors, n_lags), np.nan)
    out["e_basis_changed"] = np.full((n_anchors, n_lags), np.nan)
    if panel is None or panel.empty:
        return out

    codes = anchor_df["Code"].astype(str).to_numpy()
    left = pd.DataFrame({
        "Code": np.repeat(codes, n_lags),
        "lag": np.tile(np.arange(n_lags), n_anchors),
        "row": np.repeat(np.arange(n_anchors), n_lags),
        "ref_days": disc_date_days.reshape(-1),
    })
    known = left["ref_days"].notna()
    todo = left[known].copy()
    if todo.empty:
        return out
    todo["ref_date"] = pd.to_datetime(todo["ref_days"] * 86_400_000_000_000.0)
    todo = todo.sort_values("ref_date")

    right = panel.copy()
    right["avail_date"] = pd.to_datetime(right["avail_date"])
    merged = pd.merge_asof(todo, right, left_on="ref_date", right_on="avail_date",
                           by="Code", direction="backward",
                           allow_exact_matches=True)

    age_days = (merged["ref_date"] - merged["avail_date"]).dt.days
    fresh = merged["avail_date"].notna() & (age_days <= stale_days)

    rows = merged["row"].to_numpy()
    lags = merged["lag"].to_numpy()
    ok = fresh.to_numpy()

    for f in ALL_FIELDS:
        v = merged[f].to_numpy(dtype="float64")
        if f in FLOW_FIELDS:
            # 年次のフローを1四半期あたりに直す。ストックは期末残高なので割らない
            v = v / schema.ANNUAL_TO_QUARTER
        arr = out[f"e_{f}"]
        arr[rows[ok], lags[ok]] = v[ok]
    out["e_age_years"][rows[ok], lags[ok]] = (
        age_days.to_numpy(dtype="float64")[ok] / 365.25)
    out["e_basis_changed"][rows[ok], lags[ok]] = (
        merged["basis_changed"].to_numpy(dtype="float64")[ok])

    cov = float(np.isfinite(out["e_profit_before_tax"][:, 0]).mean() * 100)
    print(f"[edinet] 当該四半期に税引前利益が引けたサンプル {cov:.1f}%")
    return out


def span_matrix(shape: Tuple[int, int], field: str) -> np.ndarray:
    """EDINET 由来の値が何四半期ぶんか。フローは1年＝4期、ストックは1期。"""
    v = schema.ANNUAL_TO_QUARTER if field in FLOW_FIELDS else 1.0
    return np.full(shape, v)
