#!/usr/bin/env python3
"""
空売り残高報告（/markets/short-sale-report）を特徴量に使えるかの下調べ。

確かめること（取り込みを書く前に、推測せずに叩いて測る）
  1. どの引数で引けるか（disc_date / calc_date / date / code / 期間）。
     エンドポイント一覧の実測（docs/DATA_FIELDS.md）では code だけ確かめてある
  2. 過去がどこまであるか（公表日 DiscDate の最古）
  3. 1日に何件あるか（取り込みの負荷と、日次で引けるか）
  4. 列（比率・株数・前回比・提出者）と、公表日と計算日の差（知りえた日を決めるため）

出すのは件数・列名・日付の範囲・比率の分布だけ。鍵は出さない。

  $ JQUANTS_API=... python3 research/probe_shortsale.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from jquants_data_fetcher import (  # noqa: E402
    JQuantsClient, JQuantsError, resolve_api_key)

PATH = "/markets/short-sale-report"
#: 引数の形。上から順に試す（日付は祝日でない平日）
SHAPES = [
    ("disc_date", {"disc_date": "2024-05-15"}),
    ("calc_date", {"calc_date": "2024-05-15"}),
    ("date", {"date": "2024-05-15"}),
    ("disc_date_from/to", {"disc_date_from": "2024-05-13", "disc_date_to": "2024-05-17"}),
    ("from/to", {"from": "2024-05-13", "to": "2024-05-17"}),
    ("code", {"code": "72030"}),
    ("code+disc_date", {"code": "72030", "disc_date": "2024-05-15"}),
]
#: 過去の深さを見る日（平日）。取れた引数でこの日を引く
DEPTH = ["2016-10-03", "2017-04-03", "2018-04-02", "2019-04-01", "2020-04-01",
         "2021-04-01", "2022-04-01", "2023-04-03", "2024-04-01", "2025-04-01",
         "2026-04-01", "2026-09-18"]


def summarize(rows):
    cols = sorted({k for r in rows for k in r})
    disc = sorted({str(r.get("DiscDate", ""))[:10] for r in rows if r.get("DiscDate")})
    calc = sorted({str(r.get("CalcDate", ""))[:10] for r in rows if r.get("CalcDate")})
    return cols, disc, calc


def main() -> int:
    key = resolve_api_key()
    if not key:
        print("[stop] JQUANTS_API が無い")
        return 1
    client = JQuantsClient(key, pause=0.3)

    print("=== 1. 引数の形 ===")
    works = {}
    for name, params in SHAPES:
        try:
            rows = client.get_paginated(PATH, params)
        except JQuantsError as exc:
            print(f"  {name:<18} err: {str(exc)[:90]}")
            continue
        cols, disc, calc = summarize(rows)
        works[name] = params
        print(f"  {name:<18} {len(rows):>6}行  DiscDate {disc[:1]}〜{disc[-1:]}  "
              f"CalcDate {calc[:1]}〜{calc[-1:]}")
        if rows and "cols" not in works:
            works["cols"] = cols
    if "cols" in works:
        print(f"  列: {works['cols']}")

    day_param = next((n for n in ("disc_date", "date", "calc_date") if n in works), None)
    print(f"\n=== 2. 過去の深さ（{day_param or '日付で引けない'}） ===")
    if day_param:
        for d in DEPTH:
            try:
                rows = client.get_paginated(PATH, {day_param: d})
            except JQuantsError as exc:
                print(f"  {d} err: {str(exc)[:80]}")
                continue
            _, disc, calc = summarize(rows)
            lag = Counter()
            for r in rows:
                try:
                    a = dt.date.fromisoformat(str(r["CalcDate"])[:10])
                    b = dt.date.fromisoformat(str(r["DiscDate"])[:10])
                    lag[(b - a).days] += 1
                except (KeyError, ValueError):
                    pass
            codes = len({r.get("Code") for r in rows})
            print(f"  {d}: {len(rows):>5}行 / 銘柄 {codes:>4} / 公表日−計算日（暦日）"
                  f"{dict(sorted(lag.items())[:5])}")

    print("\n=== 3. 銘柄で全期間を引いたとき（code） ===")
    for code in ("72030", "99840", "65010", "40630"):
        try:
            rows = client.get_paginated(PATH, {"code": code})
        except JQuantsError as exc:
            print(f"  {code} err: {str(exc)[:80]}")
            continue
        _, disc, _ = summarize(rows)
        print(f"  {code}: {len(rows):>5}行  DiscDate {disc[:1]}〜{disc[-1:]}")

    # 値そのものは出さない（リポジトリも Actions のログも公開。J-Quants のデータを
    # 転載しない）。列ごとの型と、埋まっている割合だけ
    print("\n=== 4. 列の型と充足（code=72030 の全行）===")
    try:
        rows = client.get_paginated(PATH, {"code": "72030"})
        for k in sorted({k for r in rows for k in r}):
            vals = [r.get(k) for r in rows]
            filled = sum(v not in (None, "", "-") for v in vals)
            kinds = sorted({type(v).__name__ for v in vals if v not in (None, "", "-")})
            print(f"  {k:<16} 型 {','.join(kinds) or '-':<12} 充足 {filled}/{len(rows)}")
    except JQuantsError as exc:
        print(f"  err: {str(exc)[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
