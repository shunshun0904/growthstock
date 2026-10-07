#!/usr/bin/env python3
"""
保有している銘柄の損益の推移を、スプレッドシートの別タブ「保有の推移」に書く。

保有の印は、予測ログ（export_sheets.SHEET_TITLE）の**行を赤く塗る**こと（運用者の決め方。
2026-10-08）。毎晩の予測（Predict Breakouts）が終わると、ワークフロー Update Holdings Sheet が
このスクリプトを走らせ、タブを**まるごと作り直す**（このタブに手で書いた内容は消える）。
予測ログ側は読むだけで、書かない。

計算の決まり
  - 買い: 予測日の次の営業日の寄り付き（始値。分割調整なし）。予測ログの「建値」に値があればそれ
  - 株数: 「株数」に値があればそれ、無ければ 100（1単元）
  - 売り: 「手仕舞い日」に日付があれば、その日で確定する（「手仕舞い値」があればそれ、無ければ
    その日の終値）。それ以降は確定した損益のまま合計に残る
  - 日々の値: その日の終値。買ったあとに分割・併合があっても数が合うよう、調整後の終値（AdjC）を
    買った日の調整の比（AdjO / O）で割り、買った日の株の単位に戻して使う
  - 合計損益% = 合計損益 / それまでに買った建値の総額

赤の見分け方: 予測日とコードの両方のセルの背景が赤（R が 0.8 以上で G・B が 0.4 以下。
シートの色の「赤」「濃い赤 1」「明るい赤 1」が当たる）。行の一部だけの色は数えない。

公開ログには件数と最新日だけを出す（銘柄名・株価・損益の値は出さない）。

  GOOGLE_SERVICE_ACCOUNT_JSON='{...}' GSHEET_ID=... python3 research/holdings_sheet.py
  python3 research/holdings_sheet.py --dry-run   # シートを読んで計算だけする（書かない）
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_sheets as ES  # noqa: E402
import trading_calendar as TC  # noqa: E402

TAB = "保有の推移"
DEFAULT_SHARES = 100
CHART_TITLE = "保有の損益の推移（円）"
#: 合計の線。保有が1銘柄の間は銘柄の線と同じ値で重なるので、青の明るい段の太い帯にして下に敷く
#: （銘柄の 2px の線が上に重なって見える）
TOTAL_STYLE = ("#86b6ef", 6)
#: 保有銘柄の線の色。買った順に、dataviz の参照パレットの2番目から（1番目の青は合計の帯と同じ
#: 色相なので使わない）。先頭の青・橙・緑は色覚の差の検査に全組み合わせで通る（2026-10-08 に
#: validate_palette で確認）
SERIES_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SUMMARY_HEADER = ["銘柄", "コード", "予測日", "買った日", "建値", "株数",
                  "最新日", "最新終値", "損益(円)", "損益%", "状態"]


@dataclass
class Position:
    row: int                       # 予測ログの行番号（1始まり）
    pick_date: str                 # 予測日（YYYY-MM-DD）
    code: str                      # 4桁（英字を含むことがある）
    name: str
    entry_price: Optional[float]   # 建値（手入力）。無ければ翌営業日の始値
    shares: float
    exit_date: Optional[pd.Timestamp]
    exit_price: Optional[float]

    @property
    def jq_code(self) -> str:
        """J-Quants の5桁コード（日足の Code）。"""
        return self.code + "0" if len(self.code) == 4 else self.code

    @property
    def label(self) -> str:
        return f"{self.name}({self.code})" if self.name else self.code


# --------------------------------------------------------------------------- #
# 赤い行
# --------------------------------------------------------------------------- #

def is_red(color: Optional[Dict]) -> bool:
    """背景色が赤か。Sheets API の色は 0〜1 の red/green/blue で、0 のキーは省かれる。"""
    if not color:
        return False
    r = float(color.get("red", 0) or 0)
    g = float(color.get("green", 0) or 0)
    b = float(color.get("blue", 0) or 0)
    return r >= 0.8 and g <= 0.4 and b <= 0.4


def red_rows(grid: Dict, key_cols: Sequence[int]) -> List[int]:
    """
    fetch_sheet_metadata(includeGridData) の応答から、背景が赤い行の番号（1始まり）。

    key_cols（0始まりの列番号。予測日とコード）のセルが**すべて**赤い行だけを返す。
    """
    out: List[int] = []
    for sheet in grid.get("sheets", []):
        for data in sheet.get("data", []):
            r0 = int(data.get("startRow", 0) or 0)
            c0 = int(data.get("startColumn", 0) or 0)
            for i, row in enumerate(data.get("rowData", []) or []):
                values = row.get("values", []) or []
                ok = True
                for c in key_cols:
                    j = c - c0
                    cell = values[j] if 0 <= j < len(values) else {}
                    fmt = (cell or {}).get("effectiveFormat", {}) or {}
                    if not is_red(fmt.get("backgroundColor")):
                        ok = False
                        break
                if ok and key_cols:
                    out.append(r0 + i + 1)
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# 予測ログの行 -> 保有
# --------------------------------------------------------------------------- #

def _num(s) -> Optional[float]:
    if s is None:
        return None
    t = str(s).replace(",", "").replace("¥", "").replace("円", "").strip()
    if not t:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _date(s) -> Optional[pd.Timestamp]:
    t = str(s or "").strip()
    if not t:
        return None
    d = pd.to_datetime(t.replace("/", "-"), errors="coerce")
    return None if pd.isna(d) else pd.Timestamp(d).normalize()


def _code(s) -> str:
    t = str(s or "").strip()
    if t.endswith(".0"):                  # 数値として入った 6912.0
        t = t[:-2]
    return t.upper()


def positions_from_values(values: List[List[str]], rows: Sequence[int]) -> List[Position]:
    """予測ログの値（get_all_values）と赤い行の番号から、保有を作る。列は見出しの名前で探す。"""
    if not values:
        return []
    header = [str(h).strip() for h in values[0]]
    pos = {h: i for i, h in enumerate(header)}

    def cell(line: List[str], name: str) -> str:
        i = pos.get(name)
        return line[i] if i is not None and i < len(line) else ""

    out: List[Position] = []
    for r in rows:
        if r < 2 or r > len(values):      # 見出し行・範囲外は数えない
            continue
        line = values[r - 1]
        pick = _date(cell(line, "予測日"))
        code = _code(cell(line, "コード"))
        if pick is None or not code:
            continue
        shares = _num(cell(line, "株数"))
        out.append(Position(
            row=r, pick_date=pick.strftime("%Y-%m-%d"), code=code,
            name=str(cell(line, "銘柄名")).strip(),
            entry_price=_num(cell(line, "建値")),
            shares=shares if shares and shares > 0 else DEFAULT_SHARES,
            exit_date=_date(cell(line, "手仕舞い日")),
            exit_price=_num(cell(line, "手仕舞い値"))))
    # 買った順（予測日・コード）。色はこの順で割り当てる
    return sorted(out, key=lambda p: (p.pick_date, p.code, p.row))


# --------------------------------------------------------------------------- #
# 損益の計算
# --------------------------------------------------------------------------- #

def load_bars(data_dir: str, jq_codes: Sequence[str]) -> pd.DataFrame:
    """対象銘柄の直近2年ぶんの日足（Date, Code, O, C, AdjO, AdjC）。"""
    cols = ["Date", "Code", "O", "C", "AdjO", "AdjC"]
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths or not jq_codes:
        return pd.DataFrame(columns=cols)
    want = set(jq_codes)
    parts = []
    for p in paths[-2:]:
        try:
            df = pd.read_parquet(p, columns=cols)
        except Exception:                 # 調整後の列が無い古いファイル
            df = pd.read_parquet(p, columns=["Date", "Code", "O", "C"])
        df["Code"] = df["Code"].astype(str)
        parts.append(df[df["Code"].isin(want)])
    bars = pd.concat(parts, ignore_index=True)
    for c in ("AdjO", "AdjC"):
        if c not in bars.columns:
            bars[c] = np.nan
    bars["Date"] = pd.to_datetime(bars["Date"]).dt.normalize()
    return bars.sort_values(["Code", "Date"]).reset_index(drop=True)


def position_path(p: Position, bars: pd.DataFrame, cal=None) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    その保有の日々の損益。返り値は (状態, 表)。表は Date, close, pnl_yen, pnl_pct, held。

    状態: "保有中" / "手仕舞い" / "買い待ち"（次の営業日の日足がまだ無い）/
          "寄り付かず"（次の営業日に始値が無い）/ "株価なし"
    """
    b = bars[bars["Code"] == p.jq_code].sort_values("Date")
    if b.empty:
        return "株価なし", None
    pick = pd.Timestamp(p.pick_date)
    after = b[b["Date"] > pick]
    if after.empty:
        return "買い待ち", None
    first = after.iloc[0]
    entry_day = pd.Timestamp(first["Date"])
    if cal is not None and cal:
        k = cal.count_between(pick.date(), entry_day.date())
        if k is not None and k != 1:
            return "買い待ち", None       # 間の営業日が保存データに無い（欠けを埋めずに待つ）
    raw_open = float(first["O"]) if pd.notna(first["O"]) else float("nan")
    if not (math.isfinite(raw_open) and raw_open > 0):
        return "寄り付かず", None
    adj_open = float(first["AdjO"]) if pd.notna(first["AdjO"]) else raw_open
    ratio = adj_open / raw_open if raw_open > 0 and adj_open > 0 else 1.0
    entry = p.entry_price if p.entry_price else raw_open

    path = after[["Date"]].copy()
    adj_close = after["AdjC"].where(after["AdjC"].notna(), after["C"] * ratio)
    # 買った日の株の単位に戻した終値。売買の無い日は前の日の値のまま
    path["close"] = (adj_close / ratio).ffill()
    path = path[path["close"].notna()]
    if path.empty:
        return "寄り付かず", None
    state = "保有中"
    if p.exit_date is not None:
        done = path["Date"] <= p.exit_date
        if done.any():
            on_exit = float(path.loc[done, "close"].iloc[-1])
            exit_value = p.exit_price if p.exit_price else on_exit
            last_day = path.loc[done, "Date"].iloc[-1]
            path.loc[path["Date"] >= last_day, "close"] = exit_value
            state = "手仕舞い"
        path["held"] = done
    else:
        path["held"] = True
    path["pnl_yen"] = (path["close"] - entry) * p.shares
    path["pnl_pct"] = (path["close"] / entry - 1.0) * 100.0
    path["entry"] = entry
    path["entry_day"] = entry_day
    return state, path.reset_index(drop=True)


def daily_table(positions: List[Position], paths: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """日付 × 保有の表。列は 日付・合計損益(円)・合計損益%・保有数・各銘柄の損益(円)・各銘柄の損益%。"""
    live = [p for p in positions if p.row in paths]
    if not live:
        return pd.DataFrame(columns=["日付", "合計損益(円)", "合計損益%", "保有数"])
    dates = sorted(set().union(*[set(paths[p.row]["Date"]) for p in live]))
    out = pd.DataFrame({"日付": dates})
    total = pd.Series(0.0, index=out.index)
    cost = pd.Series(0.0, index=out.index)
    held = pd.Series(0, index=out.index)
    yen_cols, pct_cols = [], []
    for p in live:
        s = paths[p.row].set_index("Date")
        yen = out["日付"].map(s["pnl_yen"])
        pct = out["日付"].map(s["pnl_pct"])
        h = out["日付"].map(s["held"])
        entered = yen.notna()
        total = total + yen.fillna(0.0)
        cost = cost + entered * float(s["entry"].iloc[0]) * p.shares
        held = held + h.eq(True).astype(int)
        yen_cols.append((f"{p.label} 損益(円)", yen.round(0)))
        pct_cols.append((f"{p.label} 損益%", pct.round(2)))
    out["合計損益(円)"] = total.round(0)
    out["合計損益%"] = (total / cost.where(cost > 0) * 100.0).round(2)
    out["保有数"] = held
    for name, col in yen_cols + pct_cols:
        out[name] = col
    return out


def summary_rows(positions: List[Position], states: Dict[int, str],
                 paths: Dict[int, pd.DataFrame]) -> List[List]:
    """銘柄ごと1行と合計の行。"""
    rows: List[List] = []
    tot_yen = 0.0
    tot_cost = 0.0
    for p in positions:
        st = states.get(p.row, "")
        path = paths.get(p.row)
        if path is None or path.empty:
            rows.append([p.name, p.code, p.pick_date, "", p.entry_price or "", p.shares,
                         "", "", "", "", st])
            continue
        last = path.iloc[-1]
        entry = float(last["entry"])
        tot_yen += float(last["pnl_yen"])
        tot_cost += entry * p.shares
        if st == "手仕舞い" and p.exit_date is not None:
            st = f"手仕舞い {p.exit_date:%Y-%m-%d}"
        rows.append([p.name, p.code, p.pick_date, f"{pd.Timestamp(last['entry_day']):%Y-%m-%d}",
                     round(entry, 1), p.shares, f"{pd.Timestamp(last['Date']):%Y-%m-%d}",
                     round(float(last["close"]), 1), round(float(last["pnl_yen"])),
                     round(float(last["pnl_pct"]), 2), st])
    pct = round(tot_yen / tot_cost * 100.0, 2) if tot_cost > 0 else ""
    rows.append(["合計", "", "", "", "", "", "", "", round(tot_yen), pct, ""])
    return rows


# --------------------------------------------------------------------------- #
# タブの中身とグラフ
# --------------------------------------------------------------------------- #

@dataclass
class Layout:
    values: List[List]
    daily_header_row: int          # 0始まり
    daily_rows: int                # 見出しを除く行数
    daily_cols: int
    yen_series_cols: List[int]     # グラフにする列（0始まり）。先頭が合計
    summary_rows: int              # 合計の行を含む
    width: int


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float) and not math.isfinite(v):
        return ""
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return f"{pd.Timestamp(v):%Y-%m-%d}"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if math.isfinite(f) else ""
    return v


def build_layout(as_of: str, summary: List[List], daily: pd.DataFrame,
                 n_red: int) -> Layout:
    """
    タブ全体の値。1行目に説明、3行目から銘柄ごとの表、その下に日付ごとの表。
    """
    note = ("予測ログで赤く塗った行（保有の印）の損益。予測日の次の営業日の寄り付きで買ったとして"
            "計算し、毎晩の予測のあとに作り直します（このタブに手で書いた内容は消えます）。"
            "建値・株数・手仕舞い日・手仕舞い値を予測ログに書けば、それを使います。株数の既定は100株")
    values: List[List] = [[f"保有の推移（{as_of} まで）"], [note]]
    if n_red == 0:
        values.append(["予測ログに赤く塗った行がありません"])
        return Layout(values, daily_header_row=-1, daily_rows=0, daily_cols=0,
                      yen_series_cols=[], summary_rows=0, width=1)
    values.append([])
    values.append(list(SUMMARY_HEADER))
    values.extend([[_cell(v) for v in r] for r in summary])
    values.append([])
    header_row = len(values)
    cols = list(daily.columns)
    values.append(cols)
    for rec in daily.itertuples(index=False):
        values.append([_cell(v) for v in rec])
    yen_cols = [i for i, c in enumerate(cols) if c.endswith("損益(円)")]
    width = max(len(SUMMARY_HEADER), len(cols))
    return Layout(values, daily_header_row=header_row, daily_rows=len(daily),
                  daily_cols=len(cols), yen_series_cols=yen_cols,
                  summary_rows=len(summary), width=width)


def _rgb(hexcolor: str) -> Dict[str, float]:
    h = hexcolor.lstrip("#")
    return {"red": int(h[0:2], 16) / 255, "green": int(h[2:4], 16) / 255,
            "blue": int(h[4:6], 16) / 255}


def chart_request(sheet_id: int, lay: Layout) -> Optional[Dict]:
    """日付ごとの表の「損益(円)」の列を折れ線にする（軸は1本、単位は円）。"""
    if lay.daily_header_row < 0 or lay.daily_rows == 0 or not lay.yen_series_cols:
        return None
    r0 = lay.daily_header_row
    r1 = r0 + 1 + lay.daily_rows

    def src(c):
        return {"sourceRange": {"sources": [{
            "sheetId": sheet_id, "startRowIndex": r0, "endRowIndex": r1,
            "startColumnIndex": c, "endColumnIndex": c + 1}]}}

    series = []
    for k, c in enumerate(lay.yen_series_cols):
        # 先頭が合計（帯）。あとは買った順の銘柄。系列は後ろほど上に描かれる
        color, width = (TOTAL_STYLE if k == 0
                        else (SERIES_COLORS[(k - 1) % len(SERIES_COLORS)], 2))
        series.append({"series": src(c), "targetAxis": "LEFT_AXIS",
                       "colorStyle": {"rgbColor": _rgb(color)},
                       "lineStyle": {"width": width}})
    return {"addChart": {"chart": {
        "spec": {
            "title": CHART_TITLE,
            "basicChart": {
                "chartType": "LINE",
                "legendPosition": "BOTTOM_LEGEND",
                "headerCount": 1,
                "axis": [{"position": "BOTTOM_AXIS", "title": "日付"},
                         {"position": "LEFT_AXIS", "title": "損益（円）"}],
                "domains": [{"domain": src(0)}],
                "series": series,
            }},
        "position": {"overlayPosition": {
            "anchorCell": {"sheetId": sheet_id, "rowIndex": 2,
                           "columnIndex": lay.width + 1},
            "widthPixels": 760, "heightPixels": 420}},
    }}}


def format_requests(sheet_id: int, lay: Layout) -> List[Dict]:
    """円は桁区切りと符号、% は小数2桁と符号。見出しは太字。"""
    reqs: List[Dict] = []

    def num(r0, r1, c0, c1, pattern):
        reqs.append({"repeatCell": {
            "range": {"sheetId": sheet_id, "startRowIndex": r0, "endRowIndex": r1,
                      "startColumnIndex": c0, "endColumnIndex": c1},
            "cell": {"userEnteredFormat": {"numberFormat": {"type": "NUMBER",
                                                            "pattern": pattern}}},
            "fields": "userEnteredFormat.numberFormat"}})

    def bold(r):
        reqs.append({"repeatCell": {
            "range": {"sheetId": sheet_id, "startRowIndex": r, "endRowIndex": r + 1},
            "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
            "fields": "userEnteredFormat.textFormat.bold"}})

    bold(0)
    if lay.daily_header_row < 0:
        return reqs
    yen = "+#,##0;-#,##0;0"
    pct = "+0.00;-0.00;0.00"
    s0 = 3                                    # 銘柄ごとの表の見出し
    bold(s0)
    s1 = s0 + 1 + lay.summary_rows
    num(s0 + 1, s1, 4, 5, "#,##0.0")          # 建値
    num(s0 + 1, s1, 7, 8, "#,##0.0")          # 最新終値
    num(s0 + 1, s1, 8, 9, yen)
    num(s0 + 1, s1, 9, 10, pct)
    bold(s1 - 1)                              # 合計の行
    h = lay.daily_header_row
    bold(h)
    d0, d1 = h + 1, h + 1 + lay.daily_rows
    header = lay.values[h]
    for i, name in enumerate(header):
        if name.endswith("(円)"):
            num(d0, d1, i, i + 1, yen)
        elif name.endswith("%"):
            num(d0, d1, i, i + 1, pct)
    return reqs


# --------------------------------------------------------------------------- #
# シートとのやりとり
# --------------------------------------------------------------------------- #

def read_log(book, title: str) -> Tuple[List[List[str]], List[int]]:
    """予測ログの値と、赤い行の番号。"""
    ws = book.worksheet(title)
    values = ws.get_all_values()
    if not values:
        return [], []
    header = [str(h).strip() for h in values[0]]
    keys = [header.index(k) for k in ("予測日", "コード") if k in header]
    if len(keys) < 2:
        raise SystemExit("予測ログの見出しに 予測日・コード がありません")
    grid = book.fetch_sheet_metadata(params={
        "includeGridData": "true",
        "ranges": f"'{title}'",
        "fields": ("sheets(data(startRow,startColumn,"
                   "rowData(values(effectiveFormat(backgroundColor)))))")})
    return values, red_rows(grid, keys)


def write_tab(book, lay: Layout) -> int:
    """タブを作り直す（値・書式・グラフ）。グラフの系列数を返す。"""
    try:
        ws = book.worksheet(TAB)
    except Exception:
        ws = book.add_worksheet(title=TAB, rows=max(200, len(lay.values) + 50),
                                cols=max(26, lay.width + 12))
    sid = ws.id
    need_rows = len(lay.values) + 50
    need_cols = lay.width + 12
    if ws.row_count < need_rows or ws.col_count < need_cols:
        ws.resize(rows=max(ws.row_count, need_rows), cols=max(ws.col_count, need_cols))
    meta = book.fetch_sheet_metadata(params={
        "fields": "sheets(properties(sheetId),charts(chartId))"})
    old = [c["chartId"] for s in meta.get("sheets", [])
           if s.get("properties", {}).get("sheetId") == sid
           for c in s.get("charts", []) or []]
    reqs: List[Dict] = [{"deleteEmbeddedObject": {"objectId": cid}} for cid in old]
    # 値と書式をまとめて消す（前の日の書式が別の中身に残らないように）
    reqs.append({"updateCells": {"range": {"sheetId": sid},
                                 "fields": "userEnteredValue,userEnteredFormat"}})
    book.batch_update({"requests": reqs})
    ws.update(lay.values, "A1", value_input_option="USER_ENTERED")
    after: List[Dict] = format_requests(sid, lay)
    chart = chart_request(sid, lay)
    if chart:
        after.append(chart)
    if after:
        book.batch_update({"requests": after})
    return len(lay.yen_series_cols) if chart else 0


def report(n_red: int, states: Dict[int, str], n_days: int, as_of: str) -> List[str]:
    """公開ログに出す行。件数と最新日だけ（銘柄名・株価・損益は出さない）。"""
    counts: Dict[str, int] = {}
    for st in states.values():
        counts[st] = counts.get(st, 0) + 1
    parts = " / ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    return [f"[holdings] 赤い行 {n_red}" + (f"（{parts}）" if parts else ""),
            f"[holdings] 日付ごとの表 {n_days}日（最新 {as_of}）"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="保有の損益の推移を別タブに書く")
    ap.add_argument("--sheet-id", default=os.environ.get("GSHEET_ID", ""))
    ap.add_argument("--log-title", default=ES.SHEET_TITLE)
    ap.add_argument("--data-dir", default=ES.DATA_DIR)
    ap.add_argument("--dry-run", action="store_true", help="シートを読んで計算だけする（書かない）")
    args = ap.parse_args(argv)
    if not args.sheet_id:
        raise SystemExit("GSHEET_ID が設定されていません（シートのURLの /d/ と /edit の間）")

    book = ES.open_sheet(args.sheet_id)
    values, rows = read_log(book, args.log_title)
    positions = positions_from_values(values, rows)
    bars = load_bars(args.data_dir, [p.jq_code for p in positions])
    try:
        cal = TC.load(args.data_dir)
    except Exception:                     # カレンダーが無くても日足の日付で進める
        cal = None
    states: Dict[int, str] = {}
    paths: Dict[int, pd.DataFrame] = {}
    for p in positions:
        st, path = position_path(p, bars, cal)
        states[p.row] = st
        if path is not None:
            paths[p.row] = path
    daily = daily_table(positions, paths)
    as_of = (f"{pd.Timestamp(daily['日付'].max()):%Y-%m-%d}" if len(daily)
             else f"{pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d}")
    summary = summary_rows(positions, states, paths)
    lay = build_layout(as_of, summary, daily, n_red=len(positions))
    for line in report(len(positions), states, len(daily), as_of):
        print(line)
    if args.dry_run:
        print("[holdings] --dry-run なので書かない")
        return 0
    n = write_tab(book, lay)
    print(f"[holdings] タブ「{TAB}」を作り直した（{len(lay.values)}行・グラフ {n}系列）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
