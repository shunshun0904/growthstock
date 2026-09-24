#!/usr/bin/env python3
"""
取り込みの重複除去で、別々の行を1行に潰していないかを実測する。

年別ファイルへのマージ（data_store.merge_into_years）は「同じ日・同じ銘柄」を
重複とみなして後の1行だけ残す。1日1銘柄1行のデータ（日足など）ならそれで
よいが、同じ日に同じ銘柄の行が複数あるデータ（大量保有報告書の複数の提出者、
空売り残高報告の複数の報告者、業種別の空売り比率の33業種）では、別々の行を
黙って捨てることになる。業種別の空売り比率は銘柄の列が無いので「同じ日」
だけで潰れ、保存データは1日1行しか無い（API は1日34行。docs/DATA_FIELDS.md）。

確かめること（種別ごと・日ごと）
  1. API が返す行数と、今の重複の判定（日付列・銘柄）で残る行数 = 捨てていた行
  2. 候補のキーごとに一意になるか（本当の行の見分け方）
  3. 業種別の空売り比率: S33=9999 の行が「全業種の合計」か「その他」か
     （9999 の行の値 ÷ ほかの行の合計。1 に近ければ合計）

出すのは件数と比だけ。値そのものは出さない（Actions のログは公開）。

  $ JQUANTS_API=... python3 research/probe_dedupe_keys.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from jquants_data_fetcher import (  # noqa: E402
    JQuantsClient, JQuantsError, resolve_api_key)

ALL = "全列"
#: 種別 -> (パス, 日付の引数, 日付列, 候補のキー)
KINDS = {
    "fins":        ("/fins/summary", "date", "DiscDate",
                    [["DiscDate", "Code"], ["DiscDate", "Code", "DocType"],
                     ["DiscDate", "Code", "DiscTime", "DocType"], ["DiscNo"], ALL]),
    "margin":      ("/markets/margin-interest", "date", "Date", [["Date", "Code"], ALL]),
    "valuation":   ("/equities/valuation", "date", "Date", [["Date", "Code"], ALL]),
    "shortratio":  ("/markets/short-ratio", "date", "Date", [["Date"], ["Date", "S33"], ALL]),
    "marginalert": ("/markets/margin-alert", "date", "PubDate",
                    [["PubDate", "Code"], ["PubDate", "Code", "AppDate"],
                     ["PubDate", "Code", "PubReason"], ALL]),
    "earndate":    ("/fins/earnings-date", "date", "PubDate",
                    [["PubDate", "Code"], ["PubDate", "Code", "SchDate"],
                     ["PubDate", "Code", "FQName"],
                     ["PubDate", "Code", "FQName", "SchDate"], ALL]),
    "lvshld":      ("/edinet/large-volume-shareholders", "date", "SubDate",
                    [["SubDate", "Code"], ["DocId"], ALL]),
    "mjrshld":     ("/edinet/major-shareholders", "date", "SubDate",
                    [["SubDate", "Code"], ["DocId"], ALL]),
    "xhold":       ("/edinet/cross-shareholdings", "date", "SubDate",
                    [["SubDate", "Code"], ["DocId"], ALL]),
    "shortsale":   ("/markets/short-sale-report", "disc_date", "DiscDate",
                    [["DiscDate", "Code"], ["DiscDate", "Code", "SSName"],
                     ["DiscDate", "Code", "SSName", "FundName", "DICName"],
                     ["DiscDate", "CalcDate", "Code", "SSName", "FundName", "DICName"],
                     ALL]),
}
#: 決算の集中日・有報の集中日（6月末）・ふつうの日を混ぜる（すべて営業日）
DAYS = ["2019-06-27", "2021-06-29", "2022-11-14", "2023-06-29",
        "2024-05-14", "2025-06-27", "2026-02-13", "2026-09-18"]
SR_VALUES = ["SellExShortVa", "ShrtWithResVa", "ShrtNoResVa"]


def hashable(df: pd.DataFrame) -> pd.DataFrame:
    """入れ子の列（list / dict）は文字列にして一意の判定に使えるようにする。"""
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda v: isinstance(v, (list, dict, tuple))).any():
            out[c] = out[c].map(repr)
    return out


def merge_key(df: pd.DataFrame, date_col: str) -> list:
    """今の data_store.merge_into_years の判定と同じ（日付列と、あれば銘柄）。"""
    return [c for c in (date_col, "Code") if c in df.columns]


def main() -> int:
    key = resolve_api_key()
    if not key:
        print("[stop] JQUANTS_API が無い")
        return 1
    client = JQuantsClient(key, pause=0.2)
    total = {}
    for name, (path, param, date_col, cands) in KINDS.items():
        print(f"\n=== {name} ({path}, {param}=) ===")
        agg = {"rows": 0, "kept_now": 0}
        for d in DAYS:
            try:
                rows = client.get_paginated(path, {param: d})
            except JQuantsError as exc:
                print(f"  {d} err: {str(exc)[:80]}")
                continue
            if not rows:
                print(f"  {d}      0行")
                continue
            df = hashable(pd.DataFrame.from_records(rows))
            mk = merge_key(df, date_col)
            kept = len(df.drop_duplicates(subset=mk)) if mk else len(df)
            agg["rows"] += len(df)
            agg["kept_now"] += kept
            parts = []
            for k in cands:
                cols = list(df.columns) if k == ALL else [c for c in k if c in df.columns]
                if k != ALL and len(cols) < len(k):
                    parts.append(f"{'+'.join(k)}=列なし")
                    continue
                u = len(df.drop_duplicates(subset=cols))
                label = ALL if k == ALL else "+".join(k)
                parts.append(f"{label}={u}")
                agg[label] = agg.get(label, 0) + u
            print(f"  {d} {len(df):>6}行  今の判定({'+'.join(mk)})で残る {kept:>6}  | "
                  + "  ".join(parts))
            if name == "shortsale":
                near = ["DiscDate", "CalcDate", "Code", "SSName", "FundName", "DICName"]
                near = [c for c in near if c in df.columns]
                g = df[df.duplicated(subset=near, keep=False)]
                if len(g):
                    diff = sorted({c for _, x in g.groupby(near, dropna=False)
                                   for c in df.columns if x[c].nunique(dropna=False) > 1})
                    print(f"           近いキーで重なる {len(g)}行。違う列: {diff}")
                r = pd.to_numeric(df.get("ShrtPosToSO"), errors="coerce").dropna()
                if len(r):
                    print(f"           ShrtPosToSO の分位（単位の確認）: 最小 {r.min():.4f} / "
                          f"10% {r.quantile(.1):.4f} / 中央 {r.median():.4f} / 最大 {r.max():.4f}"
                          f" / 0.5 未満 {int((r < 0.5).sum())}行 / 0.005 未満 {int((r < 0.005).sum())}行")
            if name == "shortratio" and "S33" in df.columns:
                s33 = pd.to_numeric(df["S33"], errors="coerce")
                tail = df[s33 == 9999]
                rest = df[s33 != 9999]
                ratios = []
                for c in SR_VALUES:
                    if c in df.columns and len(tail) == 1:
                        a = float(pd.to_numeric(tail[c], errors="coerce").iloc[0])
                        b = float(pd.to_numeric(rest[c], errors="coerce").sum())
                        ratios.append(f"{c} {a / b:.3f}" if b else f"{c} -")
                print(f"           S33 の種類 {s33.nunique()}（9999 を含む: {bool((s33 == 9999).any())}）"
                      f"  9999 の行 ÷ ほかの行の合計: {', '.join(ratios) or '-'}")
                if d == DAYS[-1]:
                    print(f"           S33: {sorted(s33.dropna().astype(int).unique().tolist())}")
        total[name] = agg
        lost = agg["rows"] - agg["kept_now"]
        share = lost / agg["rows"] * 100 if agg["rows"] else 0.0
        print(f"  計 {agg['rows']}行 / 今の判定で捨てる {lost}行（{share:.1f}%）")

    print("\n=== まとめ（8日ぶん） ===")
    print(f"  {'種別':<12}{'API の行':>9}{'今残る':>9}{'捨てる':>8}{'割合':>8}  一意になる候補")
    for name, agg in total.items():
        lost = agg["rows"] - agg["kept_now"]
        share = lost / agg["rows"] * 100 if agg["rows"] else 0.0
        ok = [k for k, v in agg.items() if k not in ("rows", "kept_now") and v == agg["rows"]]
        print(f"  {name:<12}{agg['rows']:>9}{agg['kept_now']:>9}{lost:>8}{share:>7.1f}%  "
              f"{', '.join(ok) or '（候補では一意にならない）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
