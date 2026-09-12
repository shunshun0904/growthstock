#!/usr/bin/env python3
"""
株価分足 `/equities/bars/minute` が、この契約で実際に何を返すかを実測する。

なぜ必要か
----------
エントリータイミングの設計（docs/ENTRY_TIMING_DESIGN.md）は、
「翌日の押し目を待つか、寄りで成行するか」を分足の到達順序で判定する前提で
組み立ててある。その前提が成り立つかは、叩いてみるまで分からない。

分足は 2026-01-19 に追加されたアドオン（Lightプラン以上・月額5,500円税込・
格納2年）だと **公式に発表されている**が、
  * この契約でアドオンが有効か
  * 返る列が何か（時刻の列名すら分からない）
  * 昼休み・立会外・寄り前の行が入るか
  * 1銘柄1日で何行か / code+date 1回に何秒かかるか
  * 実際に何年遡れるか
は発表資料からは決まらない。ここで確かめるのはその5つ。

未契約なら 401/403 が返る。**それも実測結果**として記録する
（未契約を確かめるのがこのプローブの目的の一つ）。
その場合は後段を丸ごと飛ばし、終了コード 0 で終わる。
「叩いたが権限が無かった」は失敗ではなく測定結果である。

キーが悪いのかアドオンが無いのかを取り違えないため、
先に日足 `/equities/bars/daily` を叩いて「キー自体は通る」ことを確かめる。
これをやらないと、403 を見て「キーが切れた」と誤診する。

出力
----
  research/probe_minute_bars.json  生の実測値
  docs/MINUTE_DATA.md              読み物としての要約

依存は標準ライブラリのみ（他の probe_*.py と同じ。CI に pip install が無い）。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from jquants_data_fetcher import (  # noqa: E402
    API_BASE, JQuantsClient, JQuantsError, resolve_api_key,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_JSON = os.path.join(HERE, "probe_minute_bars.json")
OUT_MD = os.path.join(ROOT, "docs", "MINUTE_DATA.md")

PATH = "/equities/bars/minute"
DAILY_PATH = "/equities/bars/daily"

#: 測定に使う銘柄。板が厚く、昼休みや寄り前の扱いが読み取りやすい。
#: 4桁だと優先株と紛れる可能性があるので5桁で指定する。
LIQUID_CODE = "72030"   # トヨタ自動車
#: 流動性の低い側も見る。分足が「値が付かない分」をどう表現するか
#: （行が無い / 出来高0の行がある）は、押し目指値の約定判定を左右する。
THIN_CODE = "90820"     # 大和自動車交通（直近の候補に実際に出た銘柄）

#: 分足の遡及境界を探す候補月。公表では格納2年なので、その前後を挟む。
BOUNDARY_MONTHS = [
    "2023-09", "2024-01", "2024-03", "2024-06", "2024-09",
    "2024-12", "2025-03", "2025-06",
]

#: date 指定（全銘柄1日分）で辿るページ数の上限。
#: 4,400銘柄 × 1日300本 ≒ 130万行あり、全部引くと測定にならない。
#: 1ページあたりの行数と所要秒が分かれば全体は外挿できる。
MAX_PAGES = 3

#: code+date のレイテンシを測る回数。全期間の取得見積もりの根拠になる。
LATENCY_N = 8

PAUSE = 0.2  # API への礼儀。レイテンシ測定には含めない


def _pause() -> None:
    time.sleep(PAUSE)


def call(client: JQuantsClient, path: str, params: dict) -> Tuple[bool, Any, float]:
    """1回だけ叩いて (成否, 中身またはエラー文, 秒) を返す。"""
    t0 = time.time()
    try:
        data = client.get(path, params)
        return True, data, time.time() - t0
    except JQuantsError as exc:
        return False, str(exc)[:400], time.time() - t0


def rows_of(data: Any) -> List[dict]:
    """V2 はどのエンドポイントもデータ配列を "data" キーで返す。"""
    if isinstance(data, dict):
        got = data.get("data")
        if isinstance(got, list):
            return got
    return []


def looks_like_time(value: Any) -> bool:
    """値が時刻・日時らしいか。列名を当てにせず中身で判定する。"""
    if not isinstance(value, str):
        return False
    v = value.strip()
    return (":" in v and any(ch.isdigit() for ch in v)) or ("T" in v and "-" in v)


def describe_columns(rows: List[dict]) -> List[dict]:
    """返ってきた列の名前・例・欠測率を、列名を仮定せずに記録する。"""
    if not rows:
        return []
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out = []
    for k in keys:
        vals = [r.get(k) for r in rows]
        nonnull = [v for v in vals if v is not None and v != ""]
        example = nonnull[0] if nonnull else None
        kind = "欠測のみ"
        if nonnull:
            if isinstance(example, bool):
                kind = "真偽"
            elif isinstance(example, (int, float)):
                kind = "数値"
            elif looks_like_time(example):
                kind = "時刻らしい"
            else:
                kind = "文字列"
        out.append({
            "column": k,
            "kind": kind,
            "example": example,
            "nullShare": round(100.0 * (len(vals) - len(nonnull)) / len(vals), 1),
        })
    return out


def time_column(rows: List[dict]) -> Optional[str]:
    """時刻らしい値が入っている列を1つ選ぶ。"""
    for c in describe_columns(rows):
        if c["kind"] == "時刻らしい":
            return c["column"]
    return None


def session_shape(rows: List[dict]) -> Dict[str, Any]:
    """
    1銘柄1日の「立会の形」を読む。

    昼休み（11:30〜12:30）に行があるか、寄り前・引け後の行があるか、
    そして時刻が何分刻みか。押し目指値の約定判定は
    「安値と高値のどちらが先か」を分足の並びで決めるので、
    どの時間帯に行があるかが判定の前提そのものになる。
    """
    shape: Dict[str, Any] = {"rows": len(rows)}
    tcol = time_column(rows)
    shape["timeColumn"] = tcol
    if not tcol:
        return shape

    stamps = sorted({str(r.get(tcol)) for r in rows if r.get(tcol)})
    shape["first"] = stamps[0] if stamps else None
    shape["last"] = stamps[-1] if stamps else None
    shape["distinct"] = len(stamps)

    def hhmm(s: str) -> Optional[str]:
        if ":" not in s:
            return None
        head = s.split("T")[-1] if "T" in s else s
        head = head.split("+")[0].split("Z")[0]
        parts = head.strip().split(":")
        if len(parts) < 2:
            return None
        return f"{parts[0][-2:].zfill(2)}:{parts[1][:2]}"

    minutes = [m for m in (hhmm(s) for s in stamps) if m]
    shape["distinctMinutes"] = len(set(minutes))
    # 時間帯ごとの本数。昼休みと立会外の有無はここに出る
    buckets: Dict[str, int] = {}
    for m in minutes:
        buckets[m[:2] + "時台"] = buckets.get(m[:2] + "時台", 0) + 1
    shape["byHour"] = dict(sorted(buckets.items()))
    # 立会時間は時期によって変わる（2024-11-05 に大引けが 15:00 -> 15:30）。
    # 「立会外かどうか」を決め打ちで判定せず、時間帯の実測だけを残す。
    shape["rowsInLunch"] = sum(1 for m in minutes if "11:3" <= m < "12:30")
    shape["rowsBefore0900"] = sum(1 for m in minutes if m < "09:00")
    shape["rowsAfter1530"] = sum(1 for m in minutes if m > "15:30")
    return shape


# --------------------------------------------------------------------------- #
# 各段
# --------------------------------------------------------------------------- #

def recent_trading_day(client: JQuantsClient) -> Tuple[Optional[str], List[dict]]:
    """
    直近で日足が付いている日を探す。分足の測定日はここに合わせる。
    暦日で数えると休場に当たり、「データが無い」と「休場」を取り違える。
    """
    tried = []
    d = dt.date.today()
    for _ in range(12):
        d -= dt.timedelta(days=1)
        if d.weekday() >= 5:
            continue
        ok, data, sec = call(client, DAILY_PATH, {"code": LIQUID_CODE, "date": d.isoformat()})
        _pause()
        n = len(rows_of(data)) if ok else 0
        tried.append({"date": d.isoformat(), "ok": ok, "rows": n,
                      "detail": None if ok else data})
        if ok and n:
            return d.isoformat(), tried
    return None, tried


def stage_key_works(client: JQuantsClient) -> Dict[str, Any]:
    """
    キー自体が通るかを日足で確かめる。
    ここが通って分足だけが 403 なら、原因はアドオン未契約であって
    キーの失効ではない。取り違えると設定をいじって半日溶かす。
    """
    ok, data, sec = call(client, DAILY_PATH, {"code": LIQUID_CODE,
                                              "date": "2026-01-06"})
    _pause()
    return {"ok": ok, "rows": len(rows_of(data)) if ok else 0,
            "seconds": round(sec, 2), "detail": None if ok else data}


def stage_availability(client: JQuantsClient, date: str) -> Dict[str, Any]:
    """分足が引けるか。このプローブの本題。"""
    ok, data, sec = call(client, PATH, {"code": LIQUID_CODE, "date": date})
    _pause()
    rows = rows_of(data) if ok else []
    return {"ok": ok, "date": date, "rows": len(rows),
            "seconds": round(sec, 2),
            "detail": None if ok else data,
            "sampleRow": rows[0] if rows else None}


def stage_param_shapes(client: JQuantsClient, date: str) -> List[dict]:
    """
    どのパラメータの組み合わせを受け付けるか。
    エラー文には「何が要るか」が書いてあることが多く、仕様書より早い。

    date 指定（全銘柄1日分）は行数が大きいので、ここでは1ページだけ見る。
    """
    shapes = [
        {},
        {"code": LIQUID_CODE},
        {"code": LIQUID_CODE, "date": date},
        {"code": LIQUID_CODE, "from": date, "to": date},
        {"date": date},
    ]
    out = []
    for params in shapes:
        ok, data, sec = call(client, PATH, params)
        _pause()
        rows = rows_of(data) if ok else []
        out.append({
            "params": params,
            "ok": ok,
            "rows": len(rows),
            "seconds": round(sec, 2),
            "hasNextPage": bool(isinstance(data, dict) and data.get("pagination_key")) if ok else None,
            "detail": None if ok else data,
        })
        print(f"    {params} -> {'OK' if ok else 'NG'} "
              f"{len(rows)}行 {sec:.2f}秒", flush=True)
    return out


def stage_session(client: JQuantsClient, date: str) -> Dict[str, Any]:
    """立会の形（行数・時刻の刻み・昼休み・立会外）を、2銘柄で読む。"""
    out: Dict[str, Any] = {"date": date, "byCode": {}}
    for code, note in ((LIQUID_CODE, "板が厚い"), (THIN_CODE, "板が薄い")):
        rows: List[dict] = []
        cursor = None
        pages = 0
        t0 = time.time()
        while pages < 20:
            params: Dict[str, Any] = {"code": code, "date": date}
            if cursor:
                params["pagination_key"] = cursor
            ok, data, _ = call(client, PATH, params)
            _pause()
            pages += 1
            if not ok:
                out["byCode"][code] = {"note": note, "ok": False, "detail": data}
                break
            rows.extend(rows_of(data))
            cursor = data.get("pagination_key") if isinstance(data, dict) else None
            if not cursor:
                break
        if code in out["byCode"]:
            continue
        shape = session_shape(rows)
        shape.update({"note": note, "ok": True, "pages": pages,
                      "seconds": round(time.time() - t0, 2),
                      "columns": describe_columns(rows[:200])})
        out["byCode"][code] = shape
        print(f"    {code}({note}): {shape['rows']}行 "
              f"{shape.get('first')}〜{shape.get('last')} "
              f"{pages}ページ", flush=True)
    return out


def stage_boundary(client: JQuantsClient) -> Dict[str, Any]:
    """
    何年遡れるか。公表は「過去2年」だが、境界は月単位で実測して確かめる。
    学習に使える件数が決まるので、ここは推定で済ませない。
    """
    scan = {}
    for ym in BOUNDARY_MONTHS:
        y, m = ym.split("-")
        # 月初の1週間を見る。1日だけだと休場に当たって取り違える
        ok, data, _ = call(client, PATH, {
            "code": LIQUID_CODE, "from": f"{y}-{m}-01", "to": f"{y}-{m}-07"})
        _pause()
        n = len(rows_of(data)) if ok else 0
        scan[ym] = {"ok": bool(ok and n), "rows": n,
                    "detail": None if ok else data}
        print(f"    {ym}: {'OK' if ok and n else 'NG'} {n}行", flush=True)
    earliest = next((ym for ym in BOUNDARY_MONTHS if scan[ym]["ok"]), None)
    return {"scan": scan, "earliestMonth": earliest}


def stage_bulk_by_date(client: JQuantsClient, date: str) -> Dict[str, Any]:
    """
    date 指定（全銘柄1日分）の重さを測る。
    1ページの行数と秒が分かれば、全銘柄を保存する設計が現実的かが決まる。
    全ページは引かない（130万行規模の想定）。
    """
    pages = []
    cursor = None
    for i in range(MAX_PAGES):
        params: Dict[str, Any] = {"date": date}
        if cursor:
            params["pagination_key"] = cursor
        ok, data, sec = call(client, PATH, params)
        _pause()
        if not ok:
            pages.append({"page": i + 1, "ok": False, "detail": data})
            break
        rows = rows_of(data)
        cursor = data.get("pagination_key") if isinstance(data, dict) else None
        codes = {str(r.get("Code")) for r in rows if r.get("Code") is not None}
        pages.append({"page": i + 1, "ok": True, "rows": len(rows),
                      "seconds": round(sec, 2), "distinctCodes": len(codes),
                      "hasNextPage": bool(cursor)})
        print(f"    ページ{i+1}: {len(rows)}行 {sec:.2f}秒 "
              f"銘柄{len(codes)}件 続き{'あり' if cursor else 'なし'}", flush=True)
        if not cursor:
            break
    return {"date": date, "pages": pages, "maxPagesProbed": MAX_PAGES}


def stage_latency(client: JQuantsClient, date: str) -> Dict[str, Any]:
    """
    code+date 1回の所要時間。
    運用で使うのはこの形（候補銘柄の、その日だけ）なので、
    全期間の取得見積もりはこの数字から立てる。
    """
    samples = []
    d = dt.date.fromisoformat(date)
    got = 0
    back = 0
    while got < LATENCY_N and back < 30:
        back += 1
        day = d - dt.timedelta(days=back)
        if day.weekday() >= 5:
            continue
        ok, data, sec = call(client, PATH, {"code": LIQUID_CODE,
                                            "date": day.isoformat()})
        _pause()
        if not ok:
            samples.append({"date": day.isoformat(), "ok": False, "detail": data})
            continue
        rows = rows_of(data)
        if not rows:
            continue
        got += 1
        samples.append({"date": day.isoformat(), "ok": True,
                        "rows": len(rows), "seconds": round(sec, 3)})
    secs = sorted(s["seconds"] for s in samples if s.get("ok"))
    stat = {}
    if secs:
        stat = {"n": len(secs), "min": secs[0], "max": secs[-1],
                "median": secs[len(secs) // 2],
                "mean": round(sum(secs) / len(secs), 3)}
    return {"samples": samples, "stat": stat}


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #

def _save(result: dict) -> None:
    """段ごとに保存する。後段で落ちても前段の実測が残る。"""
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


def _md_table(header: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def write_md(result: dict) -> None:
    L: List[str] = []
    L.append("# 株価分足の取得可否（実測）")
    L.append("")
    L.append("`research/probe_minute_bars.py` の出力。**API を実際に叩いた結果のみ**を記載する。")
    L.append("")
    L.append(f"- 実測日時: {result.get('probedAt')}")
    L.append(f"- エンドポイント: `{PATH}`（`{API_BASE}{PATH}`）")
    L.append(f"- 測定に使った銘柄: `{LIQUID_CODE}` / `{THIN_CODE}`")
    L.append("")

    key = result.get("keyWorks", {})
    L.append("## 1. キー自体は通るか（日足で確認）")
    L.append("")
    if key.get("ok"):
        L.append(f"`{DAILY_PATH}` は **OK**（{key.get('rows')}行 / {key.get('seconds')}秒）。")
        L.append("以降で分足が失敗した場合、原因はキーではなくアドオンの契約範囲である。")
    else:
        L.append(f"`{DAILY_PATH}` が **NG**。キーまたは契約そのものに問題がある。")
        L.append("")
        L.append("```")
        L.append(str(key.get("detail"))[:400])
        L.append("```")
    L.append("")

    av = result.get("availability", {})
    L.append("## 2. 分足は引けるか")
    L.append("")
    if av.get("ok"):
        L.append(f"**引ける**。`code={LIQUID_CODE}` / `date={av.get('date')}` で "
                 f"{av.get('rows')}行 / {av.get('seconds')}秒。")
        if av.get("sampleRow"):
            L.append("")
            L.append("返ってきた1行（そのまま）:")
            L.append("")
            L.append("```json")
            L.append(json.dumps(av["sampleRow"], ensure_ascii=False, indent=2))
            L.append("```")
    else:
        L.append("**引けない**。返ってきたエラーをそのまま載せる。")
        L.append("")
        L.append("```")
        L.append(str(av.get("detail"))[:600])
        L.append("```")
        L.append("")
        if key.get("ok"):
            L.append("日足は通っているので、キーの失効ではない。"
                     "分足・Tick アドオンが契約に含まれていない状態だと読める。")
        L.append("")
        L.append("この結果は失敗ではなく測定結果である。"
                 "契約すれば何が変わるかは、契約後に本プローブを再実行すれば分かる。")
        L.append("")
        L.append("---")
        L.append("")
        L.append("以降の測定（列・立会の形・遡及範囲・スループット）は、"
                 "分足が引けないため実行していない。")
        with open(OUT_MD, "w", encoding="utf-8") as fh:
            fh.write("\n".join(L) + "\n")
        return
    L.append("")

    shapes = result.get("paramShapes") or []
    if shapes:
        L.append("## 3. 受け付けるパラメータ")
        L.append("")
        rows = []
        for s in shapes:
            detail = s.get("detail")
            note = "—" if s["ok"] else str(detail)[:120].replace("|", "\\|")
            rows.append([f"`{s['params']}`", "OK" if s["ok"] else "NG",
                         s.get("rows", "—"), f"{s.get('seconds')}秒", note])
        L += _md_table(["パラメータ", "結果", "件数", "所要", "備考"], rows)
        L.append("")

    sess = result.get("session") or {}
    if sess.get("byCode"):
        L.append(f"## 4. 立会の形（{sess.get('date')}）")
        L.append("")
        rows = []
        for code, s in sess["byCode"].items():
            if not s.get("ok"):
                rows.append([f"`{code}`", s.get("note", ""), "NG", "—", "—", "—", "—", "—"])
                continue
            rows.append([
                f"`{code}`", s.get("note", ""), s.get("rows"),
                f"{s.get('first')} 〜 {s.get('last')}",
                s.get("distinctMinutes"),
                f"{s.get('rowsInLunch', 0)}本",
                f"{s.get('rowsBefore0900', 0)}本",
                f"{s.get('rowsAfter1530', 0)}本",
            ])
        L += _md_table(["銘柄", "板", "行数", "時刻の範囲", "異なる分",
                        "11:30〜12:30", "09:00より前", "15:30より後"], rows)
        L.append("")
        for code, s in sess["byCode"].items():
            if not s.get("ok") or not s.get("columns"):
                continue
            L.append(f"### `{code}` が返した列")
            L.append("")
            L += _md_table(
                ["列", "種別", "例", "欠測率"],
                [[f"`{c['column']}`", c["kind"], str(c["example"])[:40], f"{c['nullShare']}%"]
                 for c in s["columns"]])
            L.append("")
            hours = s.get("byHour") or {}
            if hours:
                L.append("時間帯ごとの本数: "
                         + " / ".join(f"{k} {v}本" for k, v in hours.items()))
                L.append("")
            break

    b = result.get("boundary") or {}
    if b.get("scan"):
        L.append("## 5. どこまで遡れるか")
        L.append("")
        L += _md_table(["年月", "結果", "件数"],
                       [[ym, "OK" if v["ok"] else "NG", v.get("rows", 0)]
                        for ym, v in b["scan"].items()])
        L.append("")
        L.append(f"取得できた最古の月: **{b.get('earliestMonth') or '見つからず'}**")
        L.append("")

    bulk = result.get("bulkByDate") or {}
    if bulk.get("pages"):
        L.append("## 6. `date` 指定（全銘柄1日分）の重さ")
        L.append("")
        L += _md_table(["ページ", "行数", "所要", "含まれる銘柄数", "続き"],
                       [[p.get("page"),
                         p.get("rows", "—"),
                         f"{p.get('seconds')}秒" if p.get("ok") else "NG",
                         p.get("distinctCodes", "—"),
                         ("あり" if p.get("hasNextPage") else "なし") if p.get("ok") else "—"]
                        for p in bulk["pages"]])
        L.append("")
        L.append(f"上限 {bulk.get('maxPagesProbed')} ページまでで打ち切っている"
                 "（全銘柄1日分を最後まで引くと測定にならないため）。")
        L.append("")

    lat = result.get("latency") or {}
    if lat.get("stat"):
        s = lat["stat"]
        L.append("## 7. `code`+`date` 1回の所要時間")
        L.append("")
        L.append(f"{s['n']}回の実測: 中央値 **{s['median']}秒** "
                 f"（最小 {s['min']} / 最大 {s['max']} / 平均 {s['mean']}）。")
        L.append("")
        L.append("運用で使うのはこの形（候補銘柄の、その日だけ）なので、"
                 "必要な (銘柄, 日) の組数にこの秒数を掛ければ所要時間が出る。")
        L.append("")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main() -> int:
    client = JQuantsClient(resolve_api_key(), pause=0.0)
    result: Dict[str, Any] = {
        "probedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": PATH,
        "codes": {"liquid": LIQUID_CODE, "thin": THIN_CODE},
    }

    print("[0] キー自体が通るか（日足で確認）")
    result["keyWorks"] = stage_key_works(client)
    print(f"    {DAILY_PATH}: {'OK' if result['keyWorks']['ok'] else 'NG'}")
    _save(result)

    print("\n[1] 測定日を決める（直近で日足が付いている営業日）")
    date, tried = recent_trading_day(client)
    result["measureDate"] = {"date": date, "tried": tried}
    print(f"    -> {date}")
    _save(result)
    if not date:
        print("    直近の営業日を特定できなかった。以降を中止する。", file=sys.stderr)
        write_md(result)
        return 0

    print("\n[2] 分足が引けるか")
    result["availability"] = stage_availability(client, date)
    _save(result)
    if not result["availability"]["ok"]:
        print(f"    引けない: {str(result['availability']['detail'])[:200]}")
        print("    未契約またはプラン外と読める。以降の測定は行わない。")
        write_md(result)
        print(f"\n[write] {OUT_JSON}\n[write] {OUT_MD}")
        return 0
    print(f"    引ける: {result['availability']['rows']}行")

    print("\n[3] 受け付けるパラメータ")
    result["paramShapes"] = stage_param_shapes(client, date)
    _save(result)

    print("\n[4] 立会の形")
    result["session"] = stage_session(client, date)
    _save(result)

    print("\n[5] 遡及範囲")
    result["boundary"] = stage_boundary(client)
    _save(result)

    print("\n[6] date 指定の重さ")
    result["bulkByDate"] = stage_bulk_by_date(client, date)
    _save(result)

    print("\n[7] code+date のレイテンシ")
    result["latency"] = stage_latency(client, date)
    _save(result)

    write_md(result)
    print(f"\n[write] {OUT_JSON}\n[write] {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
