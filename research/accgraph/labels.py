#!/usr/bin/env python3
"""
決算発表後の超過リターンと3クラスラベル。

## 定義

  エントリー : 決算発表日の **翌営業日の始値**
  エグジット : そこから 5 / 10 / 20 営業日後の終値
  超過リターン: 銘柄のリターン - ベンチマークのリターン（同じ区間）
  3クラス     : +2%超 = 上昇 / ±2%以内 = 中立 / -2%未満 = 下落

発表時刻（DiscTime）は使わず、一律で翌営業日始値を起点にする。
場中に開示された決算は翌日までに一部が織り込まれるが、
「開示より前の価格でエントリーしてしまう」ことだけは絶対に起きない。
リーク方向に間違えないほうを選んでいる。

## ベンチマーク

  TOPIX   … /indices/bars/daily の 0000
  業種指数… research/build_dataset.py の S33_TO_INDEX（実測で同定済み）

どちらも銘柄と同じ区間（翌営業日始値 → N営業日後終値）で測る。
終値どうしで測ると、銘柄側だけが始値起点になり区間がずれる。

## 株式分割

リターンは調整後価格（Adj*）で計算する。未調整だと分割日に -50% のような
偽のリターンが立つ。実際に買える価格ではないが、比率としては正しい。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# build_dataset の業種→指数コード対応をそのまま使う（実測で同定したもの）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: TOPIX の指数コード。/indices/bars/daily はコードしか返さない。
TOPIX_CODE = "0000"


@dataclass(frozen=True)
class LabelConfig:
    #: 保有する営業日数
    horizons: Tuple[int, ...] = (5, 10, 20)
    #: 主ラベルに使う保有日数（フェーズ1は20営業日）
    primary_horizon: int = 20
    #: 上昇とみなす超過リターン
    up_threshold: float = 0.02
    #: 下落とみなす超過リターン
    down_threshold: float = -0.02
    #: これを下回る流動性の銘柄は母集団から外す（20日平均売買代金・億円）
    min_turnover_oku: float = 1.0


DEFAULT_LABEL = LabelConfig()

#: クラスの意味。0 = 下落 / 1 = 中立 / 2 = 上昇
CLASS_NAMES = ("下落", "中立", "上昇")


def sector_index_map() -> Dict[str, str]:
    """33業種コード -> 指数コード。build_dataset の実測結果を使う。"""
    import build_dataset  # noqa: E402  （重い import なので関数内に置く）
    return dict(build_dataset.S33_TO_INDEX)


# --------------------------------------------------------------------------- #
# 価格の行列化
# --------------------------------------------------------------------------- #

def _adjusted(bars: pd.DataFrame) -> pd.DataFrame:
    """調整後の始値・終値を用意する。Adj が欠測なら素の値で埋める。"""
    df = bars.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str).str.strip()
    o = pd.to_numeric(df.get("AdjO"), errors="coerce")
    c = pd.to_numeric(df.get("AdjC"), errors="coerce")
    df["adj_open"] = o.fillna(pd.to_numeric(df["O"], errors="coerce"))
    df["adj_close"] = c.fillna(pd.to_numeric(df["C"], errors="coerce"))
    # 売買代金（億円）。流動性フィルタに使うので素の価格・出来高で測る
    if "Va" in df.columns:
        df["turnover_oku"] = pd.to_numeric(df["Va"], errors="coerce") / 1e8
    else:
        df["turnover_oku"] = (pd.to_numeric(df["C"], errors="coerce")
                              * pd.to_numeric(df["Vo"], errors="coerce") / 1e8)
    return df[["Date", "Code", "adj_open", "adj_close", "turnover_oku"]]


@dataclass
class PriceGrid:
    """[営業日 × 銘柄] の価格行列。位置で引けるようにしておく。"""
    dates: pd.DatetimeIndex
    codes: pd.Index
    adj_open: np.ndarray
    adj_close: np.ndarray
    turnover_ma20: np.ndarray

    def code_pos(self, codes: pd.Series) -> np.ndarray:
        return self.codes.get_indexer(codes.astype(str).str.strip())

    def date_pos_after(self, dates: pd.Series) -> np.ndarray:
        """その日より後の最初の営業日の位置。無ければ -1。"""
        pos = self.dates.searchsorted(pd.to_datetime(dates), side="right")
        pos = np.asarray(pos, dtype=np.int64)
        return np.where(pos >= len(self.dates), -1, pos)


def price_grid(bars: pd.DataFrame) -> PriceGrid:
    adj = _adjusted(bars)
    adj = adj.drop_duplicates(["Date", "Code"], keep="last")
    op = adj.pivot(index="Date", columns="Code", values="adj_open").sort_index()
    cl = adj.pivot(index="Date", columns="Code", values="adj_close").reindex(
        index=op.index, columns=op.columns)
    tv = adj.pivot(index="Date", columns="Code", values="turnover_oku").reindex(
        index=op.index, columns=op.columns)
    # 20営業日平均の売買代金。当日を含む（エントリー判断は前日引け後なので
    # 当日を含めるとわずかに先読みになるが、流動性は母集団を決めるだけで
    # ラベルには効かない。念のため当日を除いた平均にする）
    tv_ma = tv.shift(1).rolling(20, min_periods=10).mean()
    print(f"[price] {len(op):,}営業日 × {op.shape[1]:,}銘柄 "
          f"{op.index.min().date()}〜{op.index.max().date()}")
    return PriceGrid(
        dates=op.index, codes=op.columns,
        adj_open=op.to_numpy(dtype=np.float64),
        adj_close=cl.to_numpy(dtype=np.float64),
        turnover_ma20=tv_ma.to_numpy(dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# 指数
# --------------------------------------------------------------------------- #

def index_grid(indices: pd.DataFrame, dates: pd.DatetimeIndex
               ) -> Tuple[pd.Index, np.ndarray, np.ndarray]:
    """指数の始値・終値を営業日カレンダーに合わせた行列にする。"""
    df = indices.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str).str.strip()
    for c in ("O", "C"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(["Date", "Code"], keep="last")
    op = df.pivot(index="Date", columns="Code", values="O").reindex(dates)
    cl = df.pivot(index="Date", columns="Code", values="C").reindex(
        index=dates, columns=op.columns)
    print(f"[index] {op.shape[1]}指数 / 営業日カレンダーに整列 "
          f"（欠測 {float(op.isna().mean().mean()) * 100:.1f}%）")
    return op.columns, op.to_numpy(dtype=np.float64), cl.to_numpy(dtype=np.float64)


def check_topix(index_codes: pd.Index, idx_close: np.ndarray,
                dates: pd.DatetimeIndex, topix: Optional[pd.DataFrame]) -> None:
    """
    指数コード 0000 が本当に TOPIX かを、専用エンドポイントの系列と突き合わせる。

    /indices/bars/daily は名称を返さないので、0000 = TOPIX は
    「そう呼ばれている」だけの前提になる。ベンチマークを取り違えると
    超過リターンの符号ごと変わるので、必ず確かめる。
    """
    if topix is None or topix.empty:
        print("[check] topix の系列が無いので 0000 の突き合わせは省略")
        return
    if TOPIX_CODE not in index_codes:
        print(f"[warn] 指数 {TOPIX_CODE} が indices に無い")
        return
    t = topix.copy()
    t["Date"] = pd.to_datetime(t["Date"])
    col = "topix" if "topix" in t.columns else t.columns[-1]
    s = t.set_index("Date")[col].astype(float).reindex(dates)
    mine = pd.Series(idx_close[:, index_codes.get_loc(TOPIX_CODE)], index=dates)
    both = pd.concat([s, mine], axis=1).dropna()
    if both.empty:
        print("[warn] 0000 と topix の重なる日が無い")
        return
    dev = float(((both.iloc[:, 1] - both.iloc[:, 0]).abs()
                 / both.iloc[:, 0]).max())
    verdict = "一致" if dev < 1e-3 else "不一致"
    print(f"[check] 指数 {TOPIX_CODE} vs /indices/bars/daily/topix: "
          f"{len(both):,}日で最大乖離 {dev * 100:.4f}% → {verdict}")
    if dev >= 1e-3:
        print("[warn] 0000 を TOPIX として使うのは危うい。"
              "ベンチマークの取り違えを疑うこと")


# --------------------------------------------------------------------------- #
# 業種
# --------------------------------------------------------------------------- #

def attach_sector(anchor_df: pd.DataFrame,
                  master_hist: Optional[pd.DataFrame]) -> pd.Series:
    """
    各アンカーに、その開示時点で有効だった33業種コードを当てる。

    最新の業種を過去に当てると業種変更をまたいだところで別の業種が付く。
    月次スナップショットを merge_asof で時点に合わせる。
    """
    if master_hist is None or master_hist.empty or "S33" not in master_hist.columns:
        print("[warn] master_hist に S33 が無いので業種ベンチマークは作れない")
        return pd.Series(np.nan, index=anchor_df.index, dtype=object)
    mh = master_hist.copy()
    mh["Date"] = pd.to_datetime(mh["Date"])
    mh["Code"] = mh["Code"].astype(str).str.strip()
    mh["S33"] = mh["S33"].astype(str).str.strip()
    mh = (mh[["Date", "Code", "S33"]].dropna()
          .sort_values("Date").drop_duplicates(["Date", "Code"], keep="last"))

    left = anchor_df[["disc_date", "Code"]].copy()
    left["Code"] = left["Code"].astype(str).str.strip()
    left["_order"] = np.arange(len(left))
    merged = pd.merge_asof(left.sort_values("disc_date"), mh,
                           left_on="disc_date", right_on="Date", by="Code",
                           direction="backward")
    merged = merged.sort_values("_order")
    s33 = merged["S33"].to_numpy()
    miss = float(pd.isna(s33).mean() * 100)
    print(f"[sector] 33業種を時点別に付与（欠測 {miss:.1f}%）")
    return pd.Series(s33, index=anchor_df.index, dtype=object)


# --------------------------------------------------------------------------- #
# ラベル本体
# --------------------------------------------------------------------------- #

def _returns(price_open: np.ndarray, price_close: np.ndarray,
             row: np.ndarray, col: np.ndarray, horizon: int) -> np.ndarray:
    """始値でエントリーし horizon 営業日後の終値で降りたときのリターン。"""
    n_rows = price_open.shape[0]
    out = np.full(len(row), np.nan)
    ok_entry = (row >= 0) & (col >= 0)
    exit_row = row + horizon
    ok = ok_entry & (exit_row < n_rows)
    if not ok.any():
        return out
    r, c, e = row[ok], col[ok], exit_row[ok]
    entry = price_open[r, c]
    exit_ = price_close[e, c]
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(entry > 0, exit_ / entry - 1.0, np.nan)
    out[ok] = vals
    return out


def _index_returns(idx_open: np.ndarray, idx_close: np.ndarray,
                   row: np.ndarray, col: np.ndarray, horizon: int) -> np.ndarray:
    return _returns(idx_open, idx_close, row, col, horizon)


def classify(excess: np.ndarray, cfg: LabelConfig = DEFAULT_LABEL) -> np.ndarray:
    """超過リターンを3クラスに割る。欠測は -1。"""
    out = np.full(len(excess), -1, dtype=np.int8)
    known = ~np.isnan(excess)
    out[known & (excess > cfg.up_threshold)] = 2
    out[known & (excess < cfg.down_threshold)] = 0
    out[known & (excess >= cfg.down_threshold) & (excess <= cfg.up_threshold)] = 1
    return out


def build_labels(anchor_df: pd.DataFrame, bars: pd.DataFrame,
                 indices: Optional[pd.DataFrame],
                 topix: Optional[pd.DataFrame],
                 master_hist: Optional[pd.DataFrame],
                 cfg: LabelConfig = DEFAULT_LABEL) -> pd.DataFrame:
    """
    アンカー（決算開示イベント）ごとにエントリー日・超過リターン・
    3クラスラベルを付ける。母集団から落ちた件数は理由別に必ず出す。
    """
    grid = price_grid(bars)
    entry_row = grid.date_pos_after(anchor_df["disc_date"])
    code_col = grid.code_pos(anchor_df["Code"])

    out = anchor_df.copy()
    out["entry_date"] = pd.Series(
        np.where(entry_row >= 0, grid.dates.to_numpy()[np.clip(entry_row, 0, None)],
                 np.datetime64("NaT")), index=out.index)
    out["entry_price"] = np.where(
        (entry_row >= 0) & (code_col >= 0),
        grid.adj_open[np.clip(entry_row, 0, None), np.clip(code_col, 0, None)],
        np.nan)
    out["turnover_ma20"] = np.where(
        (entry_row >= 0) & (code_col >= 0),
        grid.turnover_ma20[np.clip(entry_row, 0, None), np.clip(code_col, 0, None)],
        np.nan)

    # --- ベンチマークの列位置 --- #
    idx_codes, idx_open, idx_close = (pd.Index([]), None, None)
    topix_col = np.full(len(out), -1, dtype=np.int64)
    sector_col = np.full(len(out), -1, dtype=np.int64)
    if indices is not None and not indices.empty:
        idx_codes, idx_open, idx_close = index_grid(indices, grid.dates)
        check_topix(idx_codes, idx_close, grid.dates, topix)
        if TOPIX_CODE in idx_codes:
            topix_col[:] = idx_codes.get_loc(TOPIX_CODE)
        s33 = attach_sector(anchor_df, master_hist)
        out["s33"] = s33.to_numpy()
        smap = sector_index_map()
        target = s33.map(lambda v: smap.get(v) if isinstance(v, str) else None)
        sector_col = idx_codes.get_indexer(pd.Index(target.fillna("")))
    else:
        print("[warn] indices が無いのでベンチマーク控除ができない")
        out["s33"] = np.nan

    # --- 各保有期間のリターン --- #
    for h in cfg.horizons:
        raw = _returns(grid.adj_open, grid.adj_close, entry_row, code_col, h)
        out[f"ret_{h}d"] = raw
        if idx_open is not None:
            bt = _index_returns(idx_open, idx_close, entry_row, topix_col, h)
            bs = _index_returns(idx_open, idx_close, entry_row, sector_col, h)
        else:
            bt = np.full(len(out), np.nan)
            bs = np.full(len(out), np.nan)
        out[f"topix_{h}d"] = bt
        out[f"sector_{h}d"] = bs
        out[f"excess_topix_{h}d"] = raw - bt
        out[f"excess_sector_{h}d"] = raw - bs
        out[f"y_topix_{h}d"] = classify(out[f"excess_topix_{h}d"].to_numpy(), cfg)
        out[f"y_sector_{h}d"] = classify(out[f"excess_sector_{h}d"].to_numpy(), cfg)

    # --- ラベルが確定する日（Purge / Embargo に使う） --- #
    hmax = max(cfg.horizons)
    label_row = np.where(entry_row >= 0, entry_row + hmax, -1)
    valid = (label_row >= 0) & (label_row < len(grid.dates))
    out["label_ready_date"] = pd.Series(
        np.where(valid, grid.dates.to_numpy()[np.clip(label_row, 0, len(grid.dates) - 1)],
                 np.datetime64("NaT")), index=out.index)

    report_universe(out, cfg)
    return out


def report_universe(labeled: pd.DataFrame, cfg: LabelConfig = DEFAULT_LABEL) -> None:
    """
    母集団から落ちる理由と件数を出す。

    生存者バイアスはここでしか見えない。上場廃止・売買停止でエグジット価格が
    取れなかった開示は、黙って消えると「無かったこと」になる。
    """
    n = len(labeled)
    h = cfg.primary_horizon
    reasons = {
        "エントリー日が無い（データ末尾）": labeled["entry_date"].isna(),
        "エントリー価格が欠測": labeled["entry_date"].notna() & labeled["entry_price"].isna(),
        f"{h}営業日後の終値が取れない": (labeled["entry_price"].notna()
                                        & labeled[f"ret_{h}d"].isna()),
        "TOPIXを控除できない": (labeled[f"ret_{h}d"].notna()
                                & labeled[f"excess_topix_{h}d"].isna()),
        "業種指数を控除できない": (labeled[f"ret_{h}d"].notna()
                                  & labeled[f"excess_sector_{h}d"].isna()),
        f"流動性が {cfg.min_turnover_oku} 億円未満": (
            labeled["turnover_ma20"].notna()
            & (labeled["turnover_ma20"] < cfg.min_turnover_oku)),
    }
    print(f"[universe] アンカー {n:,}件のうち")
    for name, mask in reasons.items():
        c = int(mask.sum())
        print(f"  {name:<28} {c:>7,}件 ({c / max(n, 1) * 100:5.1f}%)")
    usable = labeled[f"y_topix_{h}d"] >= 0
    print(f"  {'TOPIX控除ラベルが付いた':<28} {int(usable.sum()):>7,}件 "
          f"({float(usable.mean()) * 100:5.1f}%)")
    if usable.any():
        dist = labeled.loc[usable, f"y_topix_{h}d"].value_counts().sort_index()
        total = int(dist.sum())
        parts = [f"{CLASS_NAMES[int(k)]} {int(v):,} ({v / total * 100:.1f}%)"
                 for k, v in dist.items()]
        print(f"  クラス分布({h}営業日/TOPIX控除): " + " / ".join(parts))
