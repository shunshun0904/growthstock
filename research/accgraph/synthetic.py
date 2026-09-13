#!/usr/bin/env python3
"""
決定的な合成データ。CI と単体テストで使う。

実データは日々変わるので、パイプラインの正しさを回帰で押さえられない。
J-Quants の実際の癖（累計開示、CFは半期のみ、訂正開示、株式分割、上場廃止）を
そのまま持たせた合成データを作り、これに対して結果を固定する。

  Code 9001..  非金融の通常銘柄
  分割銘柄     期中に1:2の株式分割（調整後価格でのみ連続する）
  訂正銘柄     過去の四半期を後から訂正して再開示する
  廃止銘柄     期の途中で株価が途切れる（エグジット価格が取れない）
  CF           2Q と FY だけが開示する（1Q/3Q は欠測）
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List

import numpy as np
import pandas as pd

#: 33業種コード（build_dataset の S33_TO_INDEX に実在するものを使う）
SECTORS = ["3650", "5250", "6100", "3600"]
#: 対応する指数コード
SECTOR_INDEX_CODES = {"3650": "004F", "5250": "0058", "6100": "005A", "3600": "004E"}
TOPIX_CODE = "0000"


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """土日を除いた平日。合成データなので祝日は考えない。"""
    return pd.bdate_range(start, end)


def _fy_periods(start_year: int, n_years: int) -> List[Dict]:
    """4月始まりの会計年度を四半期に割る。"""
    out = []
    for y in range(start_year, start_year + n_years):
        fy_start = dt.date(y, 4, 1)
        fy_end = dt.date(y + 1, 3, 31)
        ends = [dt.date(y, 6, 30), dt.date(y, 9, 30),
                dt.date(y, 12, 31), fy_end]
        for q, pe in enumerate(ends, 1):
            out.append({
                "quarter": q,
                "per_type": "FY" if q == 4 else f"{q}Q",
                "per_end": pe,
                "fy_start": fy_start,
                "fy_end": fy_end,
                # 開示は期末から45日後（実務の目安）。土日に当たったら翌月曜
                "disc_date": _next_weekday(pe + dt.timedelta(days=45)),
            })
    return out


def _next_weekday(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def make_fins(codes: List[str], start_year: int, n_years: int,
              seed: int = 7) -> pd.DataFrame:
    """
    /fins/summary の形をした決算データ。値は累計。

    CF（CFO/CFI/CFF/CashEq）は 2Q と FY だけに入れる。実データの開示率
    （1Q 9.9% / 2Q 76.0% / 3Q 8.2% / FY 88.8%、docs/DATA_FIELDS.md）を
    「四半期では原則出ない」という形で単純化したもの。
    """
    rng = np.random.default_rng(seed)
    periods = _fy_periods(start_year, n_years)
    rows = []
    for i, code in enumerate(codes):
        base_sales = float(10_000 * (i + 3)) * 1e6      # 百万円 -> 円
        margin = 0.05 + 0.02 * (i % 4)
        assets = base_sales * 1.4
        equity = assets * (0.35 + 0.05 * (i % 3))
        growth = 1.0 + 0.01 * (i % 5)
        for p in periods:
            q = p["quarter"]
            if q == 1:
                # 会計年度が変わるところで規模を1段伸ばす
                base_sales *= growth
                assets *= growth
                equity *= growth
            shock = float(rng.normal(0.0, 0.08))
            q_sales = base_sales / 4.0 * (1.0 + shock)
            q_op = q_sales * margin * (1.0 + float(rng.normal(0.0, 0.2)))
            q_odp = q_op * 1.05
            q_np = q_odp * 0.68
            rows.append({
                "Code": code, "quarter": q, "per": p,
                "q_sales": q_sales, "q_op": q_op, "q_odp": q_odp, "q_np": q_np,
                "assets": assets, "equity": equity,
            })

    df = pd.DataFrame(rows)
    # 単期の値を年度内で累計に直す
    df["fy_start"] = df["per"].map(lambda p: p["fy_start"])
    df = df.sort_values(["Code", "fy_start", "quarter"]).reset_index(drop=True)
    g = df.groupby(["Code", "fy_start"], sort=False)
    for src, dst in [("q_sales", "Sales"), ("q_op", "OP"),
                     ("q_odp", "OdP"), ("q_np", "NP")]:
        df[dst] = g[src].cumsum()

    out = []
    for _, r in df.iterrows():
        p = r["per"]
        q = int(r["quarter"])
        has_cf = q in (2, 4)
        cfo = float(r["NP"]) * 1.3 if has_cf else np.nan
        cfi = -float(r["Sales"]) * 0.05 if has_cf else np.nan
        cff = -float(r["NP"]) * 0.2 if has_cf else np.nan
        cash = float(r["assets"]) * 0.12 if has_cf else np.nan
        # 通期の会社予想（1Q〜3Q で出す）
        fy_mult = 4.0 / q
        forecast = q < 4
        out.append({
            "Code": r["Code"],
            "DiscDate": p["disc_date"].isoformat(),
            "DiscTime": "15:30",
            "DocType": "FYFinancialStatements" if q == 4 else "QuarterlyFinancialStatements",
            "CurPerType": p["per_type"],
            "CurPerSt": p["fy_start"].isoformat(),
            "CurPerEn": p["per_end"].isoformat(),
            "CurFYSt": p["fy_start"].isoformat(),
            "CurFYEnd": p["fy_end"].isoformat(),
            "Sales": float(r["Sales"]), "OP": float(r["OP"]),
            "OdP": float(r["OdP"]), "NP": float(r["NP"]),
            "TA": float(r["assets"]), "Eq": float(r["equity"]),
            "ShEq": float(r["equity"]) * 0.95,
            "CashEq": cash, "CFO": cfo, "CFI": cfi, "CFF": cff,
            "FSales": float(r["Sales"]) * fy_mult * 1.02 if forecast else np.nan,
            "FOP": float(r["OP"]) * fy_mult * 1.02 if forecast else np.nan,
            "FOdP": float(r["OdP"]) * fy_mult * 1.02 if forecast else np.nan,
            "FNP": float(r["NP"]) * fy_mult * 1.02 if forecast else np.nan,
        })
    fins = pd.DataFrame(out)

    # --- 訂正開示 --- #
    # 最初の銘柄の、ある1Qを1年後に訂正して再開示する。
    # as-of が効いていなければ、訂正前の期のグラフに訂正後の値が混ざる
    target = fins[(fins["Code"] == codes[0]) & (fins["CurPerType"] == "1Q")]
    if len(target) > 2:
        fix = target.iloc[1].copy()
        fix["DiscDate"] = (dt.date.fromisoformat(fix["DiscDate"])
                           + dt.timedelta(days=365)).isoformat()
        fix["Sales"] = float(fix["Sales"]) * 1.5
        fix["OP"] = float(fix["OP"]) * 1.5
        fins = pd.concat([fins, pd.DataFrame([fix])], ignore_index=True)

    return fins.sort_values(["DiscDate", "Code"]).reset_index(drop=True)


def make_bars(codes: List[str], days: pd.DatetimeIndex, seed: int = 11,
              split_code: str | None = None,
              delist_code: str | None = None) -> pd.DataFrame:
    """株価。調整後（Adj*）と未調整（O/H/L/C）の両方を持たせる。"""
    rng = np.random.default_rng(seed)
    frames = []
    for i, code in enumerate(codes):
        n = len(days)
        drift = 0.0002 + 0.00005 * (i % 5)
        ret = rng.normal(drift, 0.018, n)
        adj_close = 1000.0 * (1 + 0.1 * i) * np.exp(np.cumsum(ret))
        adj_open = adj_close * (1.0 + rng.normal(0.0, 0.004, n))
        factor = np.ones(n)
        if split_code is not None and code == split_code:
            # 期の途中で1:2分割。未調整の価格は半分になるが、調整後は連続する
            k = n // 2
            factor[k:] = 2.0
        df = pd.DataFrame({
            "Date": days,
            "Code": code,
            "AdjO": adj_open, "AdjC": adj_close,
            "AdjH": np.maximum(adj_open, adj_close) * 1.006,
            "AdjL": np.minimum(adj_open, adj_close) * 0.994,
            "AdjVo": rng.integers(50_000, 900_000, n).astype(float),
        })
        df["O"] = df["AdjO"] * factor
        df["C"] = df["AdjC"] * factor
        df["H"] = df["AdjH"] * factor
        df["L"] = df["AdjL"] * factor
        df["Vo"] = df["AdjVo"] / factor
        df["Va"] = df["C"] * df["Vo"]
        if delist_code is not None and code == delist_code:
            df = df.iloc[: int(n * 0.6)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_indices(days: pd.DatetimeIndex, seed: int = 13) -> pd.DataFrame:
    """TOPIX と業種指数。/indices/bars/daily と同じ列（名称は返らない）。"""
    rng = np.random.default_rng(seed)
    frames = []
    codes = [TOPIX_CODE] + sorted(SECTOR_INDEX_CODES.values())
    for i, code in enumerate(codes):
        n = len(days)
        ret = rng.normal(0.0001, 0.010 + 0.002 * i, n)
        close = 2000.0 * np.exp(np.cumsum(ret))
        op = close * (1.0 + rng.normal(0.0, 0.003, n))
        frames.append(pd.DataFrame({
            "Date": days, "Code": code, "O": op, "C": close,
            "H": np.maximum(op, close) * 1.004,
            "L": np.minimum(op, close) * 0.996,
        }))
    return pd.concat(frames, ignore_index=True)


def make_topix_series(indices: pd.DataFrame) -> pd.DataFrame:
    """/indices/bars/daily/topix 相当（Date と終値のみ）。"""
    t = indices[indices["Code"] == TOPIX_CODE][["Date", "C"]].copy()
    return t.rename(columns={"C": "topix"}).reset_index(drop=True)


def make_master_hist(codes: List[str], days: pd.DatetimeIndex) -> pd.DataFrame:
    """月末スナップショットの銘柄マスタ（業種・市場区分）。"""
    month_ends = pd.Series(days).groupby(
        [days.year, days.month]).max().sort_values()
    rows = []
    for d in month_ends:
        for i, code in enumerate(codes):
            rows.append({"Date": d, "Code": code,
                         "S33": SECTORS[i % len(SECTORS)],
                         "S17": "10", "Mkt": "0111"})
    return pd.DataFrame(rows)


def write_all(data_dir: str, n_codes: int = 12, start_year: int = 2017,
              n_years: int = 7) -> Dict[str, int]:
    """合成データ一式を parquet として書き出す（年別、jq_bulk と同じ形）。"""
    os.makedirs(data_dir, exist_ok=True)
    codes = [f"{9000 + i}0" for i in range(n_codes)]
    days = trading_days(f"{start_year}-01-01", f"{start_year + n_years}-12-31")

    fins = make_fins(codes, start_year, n_years)
    bars = make_bars(codes, days, split_code=codes[1], delist_code=codes[-1])
    indices = make_indices(days)
    topix = make_topix_series(indices)
    master_hist = make_master_hist(codes, days)

    counts = {}
    for prefix, df, date_col in [
        ("fins", fins, "DiscDate"), ("bars", bars, "Date"),
        ("indices", indices, "Date"), ("topix", topix, "Date"),
        ("master_hist", master_hist, "Date"),
    ]:
        d = df.copy()
        d[date_col] = pd.to_datetime(d[date_col])
        for year, part in d.groupby(d[date_col].dt.year):
            part.to_parquet(
                os.path.join(data_dir, f"{prefix}_{int(year)}.parquet"),
                index=False, compression="zstd")
        counts[prefix] = len(d)
    return counts


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/accgraph_synth"
    print(write_all(out))
