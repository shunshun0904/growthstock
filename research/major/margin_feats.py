"""
本ブレイク予測モデル: 信用残から作る特徴量（実験用。本番の205列には入れていない）。

運用者の依頼（2026-09-23）
  「「上下におおきく動く銘柄」を拾う。下落が先に来てしまうものも拾ってしまうということですよね？
   それをファンダメンタルズや、信用倍率等で拾いたいのですが。」

本番の205列にある信用の列は credit_ratio（信用倍率 = 信用買残 ÷ 信用売残）と、日々公表・規制の
銘柄にだけ付く alert_* の4本。ここでは残高を出来高と比べた「重さ」と、買残の増え方を足す。

  m_long_days   信用買残 ÷ 直近20営業日の平均出来高（買残が出来高の何日分あるか）
  m_short_days  信用売残 ÷ 同じ平均出来高
  m_long_chg4   信用買残の4週前からの変化率

使う時点
  信用残は週次（金曜時点）の記録で、公表は翌週になる。公表前の値を使わないよう、記録日から
  LAG_DAYS 日（暦日）たってから使えるものとする。本番の credit_ratio は記録日でつないでいる
  （公表前の数日ぶん先の値を使う）ので、それとは揃えていない
株数の単位
  信用残・出来高とも、分割を調整した株数にそろえる（生の株数に C ÷ AdjC を掛ける）
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

LAG_DAYS = 6          # 記録日（金曜）から何日たてば使えるとするか
VOL_DAYS = 20         # 平均出来高の日数
STALE_DAYS = 21       # 使える日からこれより古い記録は使わない
COLUMNS = ("m_long_days", "m_short_days", "m_long_chg4")


def load_margin(data_dir: str) -> pd.DataFrame:
    """週次の信用残（margin_YYYY.parquet。marginalert_* は別物なので読まない）。"""
    paths = sorted(glob.glob(os.path.join(data_dir, "margin_[0-9]*.parquet")))
    if not paths:
        raise SystemExit("margin_*.parquet がありません")
    m = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "LongVol", "ShrtVol"]) for p in paths],
                  ignore_index=True)
    m["Date"] = pd.to_datetime(m["Date"])
    m["Code"] = m["Code"].astype(str)
    return m


def build(keys: pd.DataFrame, bars: pd.DataFrame, margin: pd.DataFrame, lag_days: int = LAG_DAYS) -> pd.DataFrame:
    """
    keys（Code, Date）の各行に COLUMNS を返す（keys と同じ順・同じ行数）。
    bars は Code, Date, C, AdjC, AdjVo を持つ日次の足。margin は Code, Date（記録日）, LongVol, ShrtVol。
    """
    b = bars[["Code", "Date", "C", "AdjC", "AdjVo"]].copy()
    b["Code"] = b["Code"].astype(str)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values(["Code", "Date"]).reset_index(drop=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b["fac"] = np.where(b["AdjC"] > 0, b["C"] / b["AdjC"], np.nan)    # 生の株数 → 分割調整後の株数
    b["vo20"] = (b.groupby("Code", sort=False)["AdjVo"]
                 .rolling(VOL_DAYS, min_periods=15).mean().reset_index(level=0, drop=True))

    m = margin[["Code", "Date", "LongVol", "ShrtVol"]].copy()
    m["Code"] = m["Code"].astype(str)
    m["Date"] = pd.to_datetime(m["Date"])
    m = m.drop_duplicates(["Code", "Date"], keep="last").sort_values("Date")
    # 記録日の分割係数（記録日が休みならその前の足）
    m = pd.merge_asof(m, b[["Code", "Date", "fac"]].sort_values("Date"), on="Date", by="Code",
                      direction="backward", tolerance=pd.Timedelta("10D"))
    m = m.sort_values(["Code", "Date"]).reset_index(drop=True)
    m["long_adj"] = m["LongVol"] * m["fac"]
    m["short_adj"] = m["ShrtVol"] * m["fac"]
    g = m.groupby("Code", sort=False)
    prev = g["long_adj"].shift(4)
    gap = (m["Date"] - g["Date"].shift(4)).dt.days
    with np.errstate(invalid="ignore", divide="ignore"):
        m["m_long_chg4"] = np.where(gap.between(21, 35) & (prev > 0), m["long_adj"] / prev - 1.0, np.nan)
    m["usable"] = m["Date"] + pd.Timedelta(days=lag_days)

    k = keys[["Code", "Date"]].copy()
    k["Code"] = k["Code"].astype(str)
    k["Date"] = pd.to_datetime(k["Date"])
    k["_i"] = np.arange(len(k))
    r = pd.merge_asof(k.sort_values("Date"),
                      m[["Code", "usable", "long_adj", "short_adj", "m_long_chg4"]].sort_values("usable"),
                      left_on="Date", right_on="usable", by="Code", direction="backward",
                      tolerance=pd.Timedelta(days=STALE_DAYS))
    r = r.merge(b[["Code", "Date", "vo20"]], on=["Code", "Date"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        r["m_long_days"] = np.where(r["vo20"] > 0, r["long_adj"] / r["vo20"], np.nan)
        r["m_short_days"] = np.where(r["vo20"] > 0, r["short_adj"] / r["vo20"], np.nan)
    return r.sort_values("_i")[list(COLUMNS)].reset_index(drop=True)
