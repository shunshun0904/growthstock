#!/usr/bin/env python3
"""
J-Quants API V2 から全銘柄・長期間のデータを一括取得するヘルパー。

銘柄ごとにループすると 4,441銘柄 × 4エンドポイント = 約1.8万リクエストになるが、
V2 は `date` パラメータで **1リクエスト = その日の全銘柄** を返す。
実測で全4,441行が 4.19秒。営業日ベースで回すのが唯一現実的な方法。

出力は Parquet（列指向・圧縮）。10年分の日次バーは約1,080万行になるため、
JSON や素の CSV では扱えない。
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from typing import Callable, Iterable, List, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from jquants_data_fetcher import JQuantsClient, JQuantsError, resolve_api_key  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "_data")

# 契約がカバーする最古の日付（research/probe_boundary.py で実測）
EARLIEST_DATE = dt.date(2016, 10, 1)

# 保持する列（全列を持つとサイズが数倍になるため、必要なものだけ）
BAR_COLS = ["Date", "Code", "O", "H", "L", "C", "Vo", "Va", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo"]
FIN_COLS = [
    "DiscDate", "DiscTime", "Code", "DocType", "CurPerType", "CurPerEn", "CurFYSt",
    "Sales", "OP", "NP", "EPS", "Eq", "TA", "ROE",
    "FSales", "FOP", "FNP", "FEPS", "ShOutFY", "TrShFY",
]
MARGIN_COLS = ["Date", "Code", "LongVol", "ShrtVol"]


# --------------------------------------------------------------------------- #
# 営業日
# --------------------------------------------------------------------------- #

def trading_days(client: JQuantsClient, start: dt.date, end: dt.date) -> List[dt.date]:
    """
    /markets/calendar から実際の営業日だけを取り出す。
    土日祝を自前で判定すると祝日でリクエストを無駄撃ちするため、API に従う。
    """
    rows = client.get_paginated(
        "/markets/calendar", {"from": start.isoformat(), "to": end.isoformat()}
    )
    # V2 の列名は HolDiv（V1 は HolidayDivision）。実レスポンスで確認済み。
    # 値: "0" = 非営業日, "1" = 営業日, "2" = 東証半日立会
    div_keys = ("HolDiv", "HolidayDivision", "HolidayDiv")
    days = []
    for r in rows:
        div = next((str(r[k]) for k in div_keys if k in r and r[k] is not None), "")
        d = r.get("Date")
        if not d:
            continue
        if div in ("1", "2"):
            days.append(dt.date.fromisoformat(d))
    if not days:
        raise JQuantsError(
            f"/markets/calendar が営業日を返しませんでした ({start}〜{end})。"
            f"応答例: {rows[:1]}"
        )
    return sorted(days)


# --------------------------------------------------------------------------- #
# 日次ループでの一括取得
# --------------------------------------------------------------------------- #

def _fetch_by_day(
    client: JQuantsClient,
    path: str,
    days: Iterable[dt.date],
    columns: List[str],
    label: str,
    progress_every: int = 50,
) -> pd.DataFrame:
    """日付を1日ずつ指定して全銘柄ぶんを集める。"""
    frames: List[pd.DataFrame] = []
    days = list(days)
    total = len(days)
    t0 = time.time()
    failed: List[str] = []

    for i, d in enumerate(days, 1):
        try:
            rows = client.get_paginated(path, {"date": d.isoformat()})
        except JQuantsError as exc:
            failed.append(f"{d}: {str(exc)[:120]}")
            continue
        if not rows:
            continue
        df = pd.DataFrame.from_records(rows)
        keep = [c for c in columns if c in df.columns]
        frames.append(df[keep])

        if i % progress_every == 0 or i == total:
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (total - i) / rate if rate else 0
            got = sum(len(f) for f in frames)
            print(f"  [{label}] {i}/{total}日  {got:,}行  "
                  f"{el/60:.1f}分経過  残り約{eta/60:.1f}分", flush=True)

    if failed:
        print(f"  [{label}] 取得に失敗した日: {len(failed)}件", file=sys.stderr)
        for f in failed[:5]:
            print(f"    {f}", file=sys.stderr)

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def fetch_bars(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """株価四本値（全銘柄）。"""
    df = _fetch_by_day(client, "/equities/bars/daily", days, BAR_COLS, "bars")
    return _numify(df, [c for c in BAR_COLS if c not in ("Date", "Code")])


def fetch_fins(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """財務情報（その日に開示されたもの）。"""
    df = _fetch_by_day(client, "/fins/summary", days, FIN_COLS, "fins")
    num = ["Sales", "OP", "NP", "EPS", "Eq", "TA", "ROE",
           "FSales", "FOP", "FNP", "FEPS", "ShOutFY", "TrShFY"]
    return _numify(df, num)


def fetch_margin(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """
    信用取引週末残高。週次データなので毎営業日叩く必要はない。
    公表は週1回なので、各週の全営業日を試すのではなく週次で間引く。
    """
    weekly = sorted({d for d in days if d.weekday() == 4})  # 金曜だけ試す
    # 金曜が休場の週を拾えないので、その週の他の日も候補に入れる
    covered = {(d.isocalendar().year, d.isocalendar().week) for d in weekly}
    for d in days:
        key = (d.isocalendar().year, d.isocalendar().week)
        if key not in covered:
            weekly.append(d)
            covered.add(key)
    df = _fetch_by_day(client, "/markets/margin-interest", sorted(weekly), MARGIN_COLS, "margin")
    return _numify(df, ["LongVol", "ShrtVol"])


def fetch_topix(client: JQuantsClient, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    TOPIX の日次終値。市場環境の特徴量に使う。
    銘柄固有の力と「地合いが良かっただけ」を分離するために必須。
    """
    rows = client.get_paginated(
        "/indices/bars/daily/topix", {"from": start.isoformat(), "to": end.isoformat()}
    )
    if not rows:
        raise JQuantsError("TOPIX のデータを取得できませんでした")
    df = pd.DataFrame.from_records(rows)
    close_col = next((c for c in ("C", "Close", "AdjC") if c in df.columns), None)
    if close_col is None:
        raise JQuantsError(f"TOPIX の終値列が見つかりません。列: {list(df.columns)}")
    out = df[["Date", close_col]].rename(columns={close_col: "topix"})
    out["topix"] = pd.to_numeric(out["topix"], errors="coerce")
    return out.dropna().sort_values("Date").reset_index(drop=True)


def _numify(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="J-Quants V2 から全銘柄データを一括取得する")
    ap.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out-dir", default=DATA_DIR)
    ap.add_argument("--pause", type=float, default=0.05,
                    help="リクエスト間隔(秒)。既定は控えめ（1日1リクエストのため）")
    ap.add_argument("--what", nargs="*", default=["bars", "fins", "margin", "topix"],
                    choices=["bars", "fins", "margin", "topix"])
    args = ap.parse_args(argv)

    start = dt.date.fromisoformat(args.date_from)
    end = dt.date.fromisoformat(args.date_to)
    if start < EARLIEST_DATE:
        print(f"[warn] {start} は契約範囲外です。{EARLIEST_DATE} に切り上げます", file=sys.stderr)
        start = EARLIEST_DATE

    os.makedirs(args.out_dir, exist_ok=True)
    client = JQuantsClient(resolve_api_key(), pause=args.pause)

    print(f"[calendar] 営業日を取得 ({start} 〜 {end})")
    days = trading_days(client, start, end)
    print(f"[calendar] {len(days)}営業日")

    tag = f"{start.isoformat()}_{end.isoformat()}"
    jobs: List[tuple] = []
    if "bars" in args.what:
        jobs.append(("bars", lambda: fetch_bars(client, days)))
    if "fins" in args.what:
        jobs.append(("fins", lambda: fetch_fins(client, days)))
    if "margin" in args.what:
        jobs.append(("margin", lambda: fetch_margin(client, days)))
    if "topix" in args.what:
        jobs.append(("topix", lambda: fetch_topix(client, start, end)))

    for name, fn in jobs:
        print(f"\n[{name}] 取得開始")
        t0 = time.time()
        df = fn()
        path = os.path.join(args.out_dir, f"{name}_{tag}.parquet")
        df.to_parquet(path, index=False, compression="zstd")
        size = os.path.getsize(path) / 1e6
        print(f"[{name}] {len(df):,}行 -> {path} ({size:.1f}MB, {time.time()-t0:.0f}秒)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
