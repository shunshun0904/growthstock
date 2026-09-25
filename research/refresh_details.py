#!/usr/bin/env python3
"""
画面の銘柄データ（public/data/stocks.json）を、予測をせずに作り直すときの補助。
.github/workflows/refresh-details.yml（Refresh Stock Details）から使う。

stocks.json はふだん日次予測（Predict Breakouts）の最後の取得が作る。取得スクリプト
（scripts/jquants_data_fetcher.py）の作りを変えたときに、次の日次予測を待たずに画面へ
出すために使う（2026-09-26、株価推移を直近1年 -> 目的変数と同じ78週にしたとき。
運用者「株価推移を直近78週（サンプル母集団の定義）で表示してほしいです」）。

  python3 research/refresh_details.py codes
      公開中の predictions.json の候補から、日次予測と同じ規則（predict_daily.detail_codes、
      上限 predict_daily.TOP_CODES）で銘柄コードを選び、空白区切りで出す。直近の日次予測が
      top_codes.txt に書いたのと同じ銘柄・同じ順になる。候補が無ければ失敗にする
      （空のまま取得スクリプトを回すと、画面の銘柄が消える）

  python3 research/refresh_details.py check
      作り直した stocks.json の株価推移を数える。出すのは銘柄数・点の数・週数・日付の範囲
      だけで、株価の値は出さない（Actions のログは公開される）。次のどちらかなら失敗にして、
      コミットさせない
        - 株価推移の最後の点が、その銘柄の基準日（最新の足）でない銘柄がある
        - 78週（取得スクリプトの CHART_BARS 本の日足）そろった銘柄が1つも無い
          （78週にしていない取得スクリプトで作られた、API から日足が足りるだけ返らなかった）
      上場から78週たっていない銘柄は短くてよい（その銘柄の取れる全期間になる）。
      そろったかは日足の本数（stocks.json の historyBars。無ければ点の数）で見る。暦の週数では
      見ない: 368営業日は約550暦日 = 78.6週で、祝日の数でも変わる（祝日の無い暦なら約73週）。
      画面の見出しの週数も、予測モデルと同じ換算（245営業日 = 52週）で日足の本数から出す
      （src/lib/pricechart.js の barsToWeeks。ここの bars_to_weeks と同じ）
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import jquants_data_fetcher as JF  # noqa: E402
import predict_daily as P  # noqa: E402

PREDICTIONS_JSON = os.path.join(P.PUBLIC_DIR, "predictions.json")
STOCKS_JSON = os.path.join(P.PUBLIC_DIR, "stocks.json")

EVENT_NAMES = {"earnings": "決算発表", "breakout": "78週高値の更新", "volume_spike": "出来高急増"}

#: 予測モデルが窓を「78週」と呼ぶときの換算（research/build_dataset.py の
#: round(HIGH_WINDOW / 245 * 52)。368営業日 -> 78週）。画面の barsToWeeks と同じ
TRADING_DAYS_PER_YEAR = 245


def pick_codes(predictions_path: str = PREDICTIONS_JSON,
               limit: int = P.TOP_CODES) -> List[str]:
    with open(predictions_path, encoding="utf-8") as fh:
        rows = json.load(fh).get("candidates") or []
    return P.detail_codes(rows, limit)


def full_points() -> int:
    """78週ぶん（CHART_BARS 本）の日足がある銘柄の株価推移の点の数（取得スクリプトの間引きで数える）。"""
    bars = [{"Date": f"bar{i:04d}", "C": 1.0} for i in range(JF.CHART_BARS)]
    return len(JF.chart_history(bars))


def _weeks(first: str, last: str) -> float:
    return (dt.date.fromisoformat(last) - dt.date.fromisoformat(first)).days / 7


def _round(x: float) -> int:
    """JS の Math.round と同じ（.5 は上へ）。Python の round は偶数へ丸める。"""
    return int(math.floor(x + 0.5))


def bars_to_weeks(bars: int) -> int:
    """日足の本数を、予測モデルと同じ換算で週に直す（CHART_BARS = 369本 -> 78週）。"""
    return _round((bars - 1) / TRADING_DAYS_PER_YEAR * 52)


def _bars(stock: Dict) -> Optional[int]:
    b = stock.get("historyBars")
    return b if isinstance(b, int) and not isinstance(b, bool) else None


def _few(items: Sequence[str], n: int = 8) -> str:
    head = ", ".join(items[:n])
    return head if len(items) <= n else f"{head} ほか {len(items) - n}"


def check(payload: Dict) -> Dict:
    """株価推移の期間を数える。problems が空でなければ失敗。"""
    stocks = payload.get("stocks") or []
    spans: List[Dict] = []
    no_history: List[str] = []
    stale_last: List[str] = []
    problems: List[str] = []
    events: Dict[str, int] = {}
    outside = 0
    full = full_points()
    for s in stocks:
        h = [p for p in (s.get("history") or []) if p.get("date")]
        if not h:
            no_history.append(str(s.get("code")))
            continue
        first, last = h[0]["date"], h[-1]["date"]
        bars = _bars(s)
        weeks = _weeks(first, last)
        spans.append({
            "code": s.get("code"), "points": len(h), "first": first, "last": last,
            "weeks": weeks, "bars": bars,
            # 78週そろったか。日足の本数で見る（無ければ点の数）
            "complete": bars >= JF.CHART_BARS if bars is not None else len(h) >= full,
            # 画面の見出しの週数（src/lib/pricechart.js の buildPriceSeries と同じ出し方）
            "screen_weeks": bars_to_weeks(bars) if bars is not None and bars > 1 else _round(weeks),
        })
        if s.get("asOf") and last != s["asOf"]:
            stale_last.append(str(s.get("code")))
        for m in s.get("milestones") or []:
            events[m.get("type")] = events.get(m.get("type"), 0) + 1
            if not (first <= str(m.get("date")) <= last):
                outside += 1
    if stale_last:
        problems.append(f"株価推移の最後の点が基準日（最新の足）でない銘柄が {len(stale_last)}"
                        f"（{_few(stale_last)}）")
    longest = max(spans, key=lambda x: (x["bars"] or 0, x["points"], x["weeks"])) if spans else None
    if spans and not any(x["complete"] for x in spans):
        got = f"{longest['bars']}本・" if longest["bars"] is not None else ""
        problems.append(f"78週（{JF.CHART_BARS}本の日足）そろった銘柄が無い（いちばん長くて "
                        f"{got}{longest['points']}点・暦で {longest['weeks']:.1f}週。"
                        f"そろえば {JF.CHART_BARS}本・{full}点）")
    if stocks and not spans:
        problems.append("株価推移のある銘柄が無い")
    return {"stocks": len(stocks), "failures": len(payload.get("failures") or []),
            "spans": spans, "no_history": no_history, "stale_last": stale_last,
            "full_points": full, "longest": longest,
            "events": events, "outside": outside, "problems": problems}


def report(r: Dict) -> str:
    spans = r["spans"]
    short = [x for x in spans if not x["complete"]]
    sparse = [x for x in spans if x["complete"] and x["points"] < r["full_points"]]
    lines = ["## 画面の銘柄データ（stocks.json）", "",
             f"- 銘柄 {r['stocks']}（株価推移あり {len(spans)}・なし {len(r['no_history'])}）"
             f" / 取得に失敗 {r['failures']}"]
    if spans:
        lines.append(f"- 株価推移: 78週そろった銘柄 {len(spans) - len(short)} / "
                     f"78週に満たない銘柄 {len(short)}")
        heads = Counter(x["screen_weeks"] for x in spans)
        lines.append("- 画面の見出し: " + "・".join(
            f"「直近{w}週」{n}銘柄" for w, n in sorted(heads.items(), key=lambda kv: -kv[0])))
        lg = r["longest"]
        got = f"{lg['bars']}本・" if lg["bars"] is not None else ""
        lines.append(f"- いちばん長い銘柄の期間: {lg['first']} 〜 {lg['last']}"
                     f"（{got}{lg['points']}点・暦で {lg['weeks']:.1f}週）")
        lasts = sorted({x["last"] for x in spans})
        lines.append(f"- 最後の点の日付: {', '.join(lasts)}")
    if r["events"]:
        ev = " / ".join(f"{EVENT_NAMES.get(k, k)} {v}" for k, v in sorted(r["events"].items()))
        lines.append(f"- 出来事: {ev}（株価推移の期間の外 {r['outside']}）")
    if short:
        lines.append("- 78週に満たない銘柄（上場から日が浅いなど）: " + _few(
            [f"{x['code']} {x['screen_weeks']}週" + (f"（{x['bars']}本）" if x["bars"] else "")
             for x in short]))
    if sparse:
        lines.append(f"- 78週そろっているが点が {r['full_points']} より少ない銘柄（値の付かない日がある）: "
                     + _few([f"{x['code']} {x['points']}点" for x in sparse]))
    for p in r["problems"]:
        lines.append(f"- ⚠ {p}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("codes", help="日次予測と同じ規則で銘柄コードを選ぶ")
    c.add_argument("--predictions", default=PREDICTIONS_JSON)
    k = sub.add_parser("check", help="作り直した stocks.json の株価推移の期間を数える")
    k.add_argument("--stocks", default=STOCKS_JSON)
    args = ap.parse_args(argv)

    if args.cmd == "codes":
        codes = pick_codes(args.predictions)
        if not codes:
            print("predictions.json に候補が無いので、取り直す銘柄がありません", file=sys.stderr)
            return 1
        print(" ".join(codes))
        return 0

    with open(args.stocks, encoding="utf-8") as fh:
        r = check(json.load(fh))
    print(report(r))
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
