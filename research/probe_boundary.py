#!/usr/bin/env python3
"""
学習期間の下限（データ提供開始日）と、一括取得のスループットを実測する。

  * 2016〜2018 を月単位で探索して株価データの開始境界を特定
  * 全銘柄1日分の取得にかかる時間を計測 -> 全期間構築の所要時間を見積もる
"""
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from jquants_data_fetcher import JQuantsClient, JQuantsError, resolve_api_key  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_boundary.json")


def has_data(client, ym: str) -> tuple:
    """その年月にトヨタの株価があるか。(bool, 詳細)"""
    y, m = ym.split("-")
    last = 28
    try:
        rows = client.get_paginated(
            "/equities/bars/daily",
            {"code": "72030", "from": f"{y}-{m}-01", "to": f"{y}-{m}-{last}"},
        )
        return (len(rows) > 0, f"{len(rows)}営業日")
    except JQuantsError as exc:
        return (False, str(exc)[:160])


def main():
    client = JQuantsClient(resolve_api_key(), pause=0.3)
    result = {"probedAt": dt.datetime.now(dt.timezone.utc).isoformat()}

    print("[1] 株価データの開始境界を探索")
    scan = {}
    for ym in ["2015-04", "2016-04", "2016-10", "2017-01", "2017-04",
               "2017-07", "2017-10", "2018-01", "2018-04"]:
        ok, detail = has_data(client, ym)
        scan[ym] = {"ok": ok, "detail": detail}
        print(f"    {ym}: {'OK ' if ok else 'NG '} {detail}")
    result["boundary_scan"] = scan
    earliest = next((ym for ym, v in scan.items() if v["ok"]), None)
    result["earliest_available"] = earliest
    print(f"    -> 取得可能な最古: {earliest}")

    print("\n[2] 全銘柄1日分の取得スループット (3日分を計測)")
    timings = []
    d = dt.date.today() - dt.timedelta(days=3)
    measured = 0
    while measured < 3:
        if d.weekday() < 5:
            t0 = time.time()
            try:
                rows = client.get_paginated("/equities/bars/daily", {"date": d.isoformat()})
                el = time.time() - t0
                timings.append({"date": d.isoformat(), "rows": len(rows), "seconds": round(el, 2)})
                print(f"    {d}: {len(rows)}行 / {el:.2f}秒")
                measured += 1
            except JQuantsError as exc:
                print(f"    {d}: 取得不可 {str(exc)[:80]}")
        d -= dt.timedelta(days=1)
    result["throughput"] = timings

    if timings and earliest:
        avg = sum(t["seconds"] for t in timings) / len(timings)
        y, m = map(int, earliest.split("-"))
        years = (dt.date.today() - dt.date(y, m, 1)).days / 365.25
        days = years * 245  # 年間営業日
        print(f"\n[見積] 平均 {avg:.2f}秒/日 × 約{days:.0f}営業日 "
              f"= 約{avg * days / 60:.0f}分 (株価のみ・{years:.1f}年分)")
        result["estimate"] = {
            "avg_seconds_per_day": round(avg, 2),
            "trading_days": round(days),
            "years": round(years, 1),
            "estimated_minutes": round(avg * days / 60),
        }

    print("\n[3] 財務の日付一括取得を1ヶ月ぶん試す (決算集中期)")
    fins_total = 0
    for day in range(1, 16):
        try:
            rows = client.get_paginated("/fins/summary", {"date": f"2026-05-{day:02d}"})
            fins_total += len(rows)
        except JQuantsError:
            pass
    print(f"    2026-05-01〜15 の開示件数: {fins_total}件")
    result["fins_may_first_half"] = fins_total

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n[done] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
