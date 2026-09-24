#!/usr/bin/env python3
"""
今日が非営業日で、かつデータが揃っているなら、日次予測を飛ばす。

なぜ要るか
--------
鮮度チェック（check_freshness.py）は遅れを**平日**で数える。祝日カレンダーを
持ち込んでいないのは意図的で、「更新が止まったのに祝日だからと自分で
言い訳する」余地を作らないため。

その代わり、連休では区別がつかなくなる。日次予測の許容は1営業日なので、
3連休（例 2026-09-21〜23）だと火・水は「遅れ2〜3営業日」で止まる。
止まること自体は正しいが、**正常な休場と本物の取り込み障害が同じ赤い×
として出る**。毎回の連休で赤を見ることになり、本物を見逃す訓練になる。

そこで、営業日かどうかを**推測せずに受け取る**。営業日カレンダーを引いて
いるのは取り込み（jq_bulk.py）だけなので、そこが残したものを読む。
自分でカレンダーを判断しない。

読む先は2つあり、**カレンダーのほうを先に見る**。

  1. `research/_data/calendar.parquet`
     取り込みが `/markets/calendar` の応答をそのまま保存したもの。
     取引所が公表した事実で、**いつ保存したかに依らない**
  2. `manifest.json` の `calendar` ブロック（従来の経路）
     取り込みが走った日の判定。1 が対象日を覆っていないときの控え

1 を足した理由: 2 は「manifest が12時間以内」を条件にしていたため、
連休2日目のように**その日の取り込みがまだ走っていない**と使えなかった。
2026-09-22 はこれで落ちた（manifest が18時間前）。カレンダーは将来ぶんまで
入っているので、取り込みが走っていなくても今日のことが分かる。

飛ばす条件
--------
**カレンダー経路**（2つとも満たすこと）
1. カレンダーが今日を覆っていて、かつ今日が **営業日ではない**
2. 保存データの最終バー日が、カレンダー上の**直近の営業日に届いている**
   **ここが肝**。取り込みが壊れていれば届いていないので、休場日であっても
   飛ばさず、鮮度チェックで落とす。

**manifest 経路**（カレンダーが今日を覆っていないときだけ。3つとも）
1. `calendar.isTradingDay` が **明示的に False**
2. 保存データの最終バー日 == `calendar.lastTradingDay`
3. manifest が十分に新しい（既定12時間以内）

どれか1つでも欠ければ「通常どおり進む」。**飛ばす側に倒さない。**

予測済みの日を二度と予測しない（2026-09-24 に足した第一の規則）
------------------------------------------------------------
上の判定は「今日が営業日か」を壁時計の日付で見ていた。これでは、
休場日の取り込みが日付をまたいで終わり、連鎖した予測が翌営業日の
**未明**（まだ当日の日足が無い時刻）に走ると「営業日だから進む」になり、
既に予測・公開した日（asOf）を**新しいモデルで予測し直して記録に混ぜる**。
実際に 2026-09-24 03:24 JST の自動実行（1f18614）で、9/18 の追跡記録に
9/20 学習のモデルが選んだ銘柄（日本ナレッジ）が入った。

問うべきは「今日は営業日か」ではなく「**まだ予測していない日足があるか**」。

  最終バー日 > 公開済みの asOf   → 進む（未予測の日がある。休場日でも）
  最終バー日 <= 公開済みの asOf  → 飛ばす（予測するものが無い）

飛ばす日でも、**その時刻に揃っているはずの日足が無ければ** stale=true を
出す（ワークフローはそれで赤にする）。揃っているはずの日は、営業日の
18:00 JST（取り込みの待ちの締切）以降ならその日、それより前なら直前の
営業日。取り込みが壊れているのに「予測済みだから」と黙って緑にしない。

意図して予測し直すときは FORCE_PREDICT=true（workflow_dispatch の force）。
最終バー日か asOf が読めないときは、従来の判定に落ちる。

止めない
------
判断できない事情（manifest が読めない、バーが無い、時刻が壊れている）は
すべて「通常どおり進む」に倒し、**必ず exit 0 で返す**。ここが新しい
失敗の入口になってはいけない。止めるかどうかは鮮度チェックの仕事。

  $ python3 research/trading_day_gate.py
  -> 標準出力に判定、$GITHUB_OUTPUT に run=true|false
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trading_calendar  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
#: manifest がこれより古ければ「今日の判断ではない」とみなす（manifest 経路だけ）
STALE_HOURS = 12
#: 「今日」は東京の日付で決める。ランナーは UTC なので明示する
JST = dt.timezone(dt.timedelta(hours=9))
#: 公開済みの予測。asOf がそこまで予測した日
PRED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "public", "data", "predictions.json")
#: 営業日の当日の日足が揃っているはずの時刻（JST）。取り込みは 18:00 まで待つ
#: （update-data.yml の wait_for_data --deadline 18:00）
READY_JST = dt.time(18, 0)


def last_bar_date(data_dir: str) -> Optional[dt.date]:
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        return None
    d = pd.read_parquet(paths[-1], columns=["Date"])
    if not len(d):
        return None
    return pd.Timestamp(pd.to_datetime(d["Date"]).max()).date()


def published_as_of(path: str = PRED_PATH) -> Optional[dt.date]:
    """公開済みの予測がどの日まで予測しているか。読めなければ None。"""
    try:
        with open(path, encoding="utf-8") as fh:
            v = json.load(fh).get("asOf")
        return dt.date.fromisoformat(str(v)[:10]) if v else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def expected_bar_date(cal, now_jst: dt.datetime) -> Optional[dt.date]:
    """
    この時刻に保存データに揃っているはずの最新の日足の日。分からなければ None。

    営業日の 18:00 JST 以降ならその日、それより前（未明を含む）や休場日なら
    直前の営業日。
    """
    if cal is None or not cal:
        return None
    today = now_jst.date()
    if not cal.covers(today):
        return None
    if cal.is_trading_day(today) and now_jst.time() >= READY_JST:
        return today
    return cal.last_trading_day(today - dt.timedelta(days=1))


def decide_new_data(last_bar: Optional[dt.date], as_of: Optional[dt.date],
                    expected: Optional[dt.date]
                    ) -> Optional[Tuple[bool, str, bool, str]]:
    """
    (進むか, 理由, 日足が欠けているか, その理由)。判断できなければ None。

    **壁時計の日付では決めない。** 未予測の日足があるかだけで決める。
    """
    if last_bar is None or as_of is None:
        return None
    stale = expected is not None and last_bar < expected
    stale_why = (f"{expected} の日足が保存データに無い（最終バー {last_bar}）。"
                 "取り込みを疑う" if stale else "")
    if last_bar > as_of:
        return True, f"未予測の日足がある（最終バー {last_bar} > 予測済み {as_of}）", \
            stale, stale_why
    return False, (f"新しい日足が無い（最終バー {last_bar} は予測済み {as_of}）。"
                   "予測済みの日を予測し直さない"), stale, stale_why


def stale_excused(manifest: dict, expected: Optional[dt.date]) -> Optional[str]:
    """
    日足が欠けていても「取り込みを疑う」べきでない理由。無ければ None。

    直前の取り込みが、揃っているはずの日より**前までしか要求していない**とき
    （過去日の取り直しなど、to を指定した手動の取り込み）。その取り込みが終わると
    日次予測が起動するが、今日の日足が無いのは取り込みの故障ではない。
    """
    note = (manifest or {}).get("calendar") or {}
    try:
        req = dt.date.fromisoformat(str(note.get("requestedTo"))[:10])
    except (TypeError, ValueError):
        return None
    if expected is not None and req < expected:
        return f"直前の取り込みは {req} までしか要求していない（過去日の取り直し）"
    return None


def decide_by_calendar(cal, last_bar: Optional[dt.date], today: dt.date
                       ) -> Optional[Tuple[bool, str]]:
    """
    保存済みカレンダーだけで判定する。判定できなければ None（次の経路へ）。

    **保存した時刻は見ない。** カレンダーは将来ぶんまで入っているので、
    今日の取り込みがまだ走っていなくても今日のことが分かる。従来の
    manifest 経路が連休2日目で使えなかったのは、そこを時刻で縛ったため。
    """
    if cal is None or not cal:
        return None
    trading = cal.is_trading_day(today)
    if trading is None:
        return None                                   # 範囲外。分からない
    if trading:
        return True, f"{today} はカレンダー上の営業日"
    ltd = cal.last_trading_day(today)
    if ltd is None:
        return None
    if last_bar is None:
        return True, "保存データの最終バー日が分からない"
    if last_bar < ltd:
        return True, (f"保存データの最終バー {last_bar} が"
                      f"直近の営業日 {ltd} に届いていない（取り込みを疑う）")
    return False, f"{today} はカレンダー上の非営業日で、直近の営業日 {ltd} まで揃っている"


def decide(manifest: dict, last_bar: Optional[dt.date],
           now: dt.datetime, stale_hours: int = STALE_HOURS,
           cal=None, today: Optional[dt.date] = None) -> Tuple[bool, str]:
    """
    (予測を走らせるか, 理由) を返す。判断できなければ走らせる側に倒す。

    カレンダー（cal）が今日を覆っていればそれで決める。覆っていなければ
    manifest の判定に落ちる。
    """
    today = today or now.astimezone(JST).date()
    by_cal = decide_by_calendar(cal, last_bar, today)
    if by_cal is not None:
        return by_cal

    cal_note = manifest.get("calendar")
    if not isinstance(cal_note, dict):
        return True, "カレンダーが今日を覆っておらず、manifest にも記録が無い"
    cal = cal_note

    if cal.get("isTradingDay") is not False:
        v = cal.get("isTradingDay")
        if v is True:
            return True, f"{cal.get('asOfJst')} は営業日"
        return True, "今日が営業日かを取り込み側が判断していない"

    updated = manifest.get("updatedAt")
    if not updated:
        return True, "manifest に updatedAt が無い"
    try:
        ts = dt.datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
    except ValueError:
        return True, f"updatedAt を読めない（{updated}）"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    age_h = (now - ts).total_seconds() / 3600
    if age_h > stale_hours:
        return True, (f"取り込みが {age_h:.1f}時間前で古い"
                      f"（{stale_hours}時間以内でないと非営業日と判断しない）")

    ltd = cal.get("lastTradingDay")
    if not ltd:
        return True, "直近の営業日が記録されていない"
    if last_bar is None:
        return True, "保存データの最終バー日が分からない"
    if last_bar.isoformat() != ltd:
        return True, (f"保存データの最終バー {last_bar} が"
                      f"直近の営業日 {ltd} に届いていない（取り込みを疑う）")

    return False, (f"{cal.get('asOfJst')} は非営業日で、"
                   f"直近の営業日 {ltd} まで揃っている")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="非営業日なら日次予測を飛ばす（判断できなければ進む）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--stale-hours", type=int, default=STALE_HOURS)
    ap.add_argument("--predictions", default=PRED_PATH,
                    help="公開済みの予測（asOf を読む）")
    args = ap.parse_args(argv)

    run, why = True, "判定できなかったので通常どおり進む"
    stale, stale_why = False, ""
    force = os.environ.get("FORCE_PREDICT", "").strip().lower() in ("1", "true", "yes")
    try:
        path = os.path.join(args.data_dir, "manifest.json")
        manifest = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        now = dt.datetime.now(dt.timezone.utc)
        cal = trading_calendar.load(args.data_dir)
        last_bar = last_bar_date(args.data_dir)
        nd = None
        if force:
            print("[gate] FORCE_PREDICT が指定されたので、予測済みの日でも予測し直す")
        else:
            expected = expected_bar_date(cal, now.astimezone(JST))
            nd = decide_new_data(last_bar, published_as_of(args.predictions), expected)
            if nd is not None and nd[2]:
                excuse = stale_excused(manifest, expected)
                if excuse:
                    print(f"[gate] 日足が欠けているが、取り込みの故障ではない — {excuse}")
                    nd = (nd[0], nd[1], False, "")
        if nd is not None:
            run, why, stale, stale_why = nd
        else:
            run, why = decide(manifest, last_bar, now, args.stale_hours, cal=cal)
    except Exception as exc:                      # noqa: BLE001
        # 判断できない理由が何であれ、止めずに通常どおり進む。
        # ここを失敗にすると、連休対応のための部品が新しい障害になる
        print(f"[warn] 判定に失敗したので通常どおり進む: "
              f"{type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)

    if stale:
        print(f"[gate] 日足が欠けている — {stale_why}")
    if run:
        print(f"[gate] 予測を実行する — {why}")
    else:
        print(f"[gate] 予測を飛ばす — {why}")
        print("       新しい日足が無いので新しい候補は出ない。画面は予測済みの日のまま。")

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        # 理由はそのまま workflow の shell に埋め込まれる。改行は
        # $GITHUB_OUTPUT の形式を壊し、二重引用符は shell を壊すので落とす
        safe = " ".join(str(why).split()).replace('"', "'")
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"run={'true' if run else 'false'}\n")
            fh.write(f"reason={safe}\n")
            fh.write(f"stale={'true' if stale else 'false'}\n")
            fh.write("stale_reason=" + " ".join(str(stale_why).split()).replace('"', "'")
                     + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
