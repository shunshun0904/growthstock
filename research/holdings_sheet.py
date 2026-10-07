#!/usr/bin/env python3
"""
保有の損益の推移を、スプレッドシートの別タブ「保有の推移」に書き、予測ログの行を色で塗る。

取引の記録（運用者の依頼。2026-10-08 から）
  運用者はチャットで「買った（ことにした）」「売った（ことにした）」を報告し、セッションが
  research/trades_inbox.py で暗号化して受け皿（research/trades_inbox.jsonl）に足す。ワークフロー
  Update Holdings Sheet がこのスクリプトで復号し、シートの「取引の記録」タブに写す（id で重複を
  除いて足すだけ。タブの行を手で直したら、その値を使う）。保有はこのタブから作る。
  取り消しは、取り消す行の id（先頭 8 文字以上）を「取消の対象」に書いた「取消」の行を足す
  （行そのものは消さない。消しても、受け皿に残っているので次の実行でまた足される）。

売りのルール（運用者の決定。2026-10-08）
  - 買った日（寄り付きで買う）を1日目として 20 営業日以内に、日中の高値が 買値×1.10 に届いたら、
    その日に 買値×1.10 で売る（指値を置いておく想定）
  - 届かなければ 20 日目の終値で売る。ただし終値が買値を下回っていたら売らずに持ち越し、
    売りの報告があるまで持つ
  - 売りの報告があれば、ルールより報告（その日・その値段。値段が無ければその日の終値）を優先する
  - ルールで売ったものも売却済みとして確定する
  営業日は取引所のカレンダーで数える（無ければその銘柄の日足の日付）。

色: 予測ログで、保有中（持ち越し・買い待ちを含む）の行を赤、売却済みの行を黄色に塗る。取り消した
買いの行は塗りを外す。取引の記録にある行だけで、ほかの行の色には触らない。あわせて、その行の
「建値」「株数」「手仕舞い日」「手仕舞い値」「損益」を毎晩書き直す（取り消した行は空にする）。

計算の決まり
  - 買い: 報告の日の寄り付き（始値。分割調整なし）。日付が無ければ予測日の次の営業日。値段の報告があればそれ
  - 株数: 報告に無ければ 100（1単元）。売りはその保有をまるごと売る
  - 日々の値は終値（売るまで）。買ったあとに分割・併合があっても、買った日の株の単位に戻して数える
    （調整後の値 ÷ 買った日の調整の比 AdjO / O）
  - 合計損益 = 売ったものの確定した損益 + 持っているものの含み損益
  - 合計損益% = 合計損益 / それまでに買った建値の総額

グラフ: 合計の線と、保有中・売ってから 20 営業日以内の銘柄の線（売った日で線が終わる）。色は
買った順に、期間の重なる銘柄どうしで違う色にする（一度決まった色は変わらない）。

公開ログには件数と最新日だけを出す（銘柄名・株価・損益の値は出さない）。

  GOOGLE_SERVICE_ACCOUNT_JSON='{...}' GSHEET_ID=... python3 research/holdings_sheet.py
  python3 research/holdings_sheet.py --dry-run   # シートを読んで計算だけする（書かない）
"""
from __future__ import annotations

import argparse
import glob
import itertools
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_sheets as ES  # noqa: E402
import trading_calendar as TC  # noqa: E402
import trades_inbox as TI  # noqa: E402

TAB = "保有の推移"
LOG_TAB = "取引の記録"
LOG_HEADER = ["id", "受付", "種類", "予測日", "コード", "銘柄名", "日付", "価格", "株数",
              "取消の対象", "メモ"]
KIND_JA = {"buy": "買い", "sell": "売り", "cancel": "取消"}
KIND_EN = {**{v: k for k, v in KIND_JA.items()}, **{k: k for k in KIND_JA}}
DEFAULT_SHARES = 100
TARGET = 1.10                     # 買値の +10% で売る
HORIZON = 20                      # 買った日を1日目として 20 営業日
RED = {"red": 1.0, "green": 0.0, "blue": 0.0}
YELLOW = {"red": 1.0, "green": 1.0, "blue": 0.0}
SOLD = "売却済み"
#: 予測ログで、取引の記録がある行に書く列（利用者の記入欄だったもの。記録がある行だけ書く）
USER_WRITE_COLS = ["建値", "株数", "手仕舞い日", "手仕舞い値", "損益"]
CHART_TITLE = "保有の損益の推移（円）"
#: 合計の線。保有が1銘柄の間は銘柄の線と同じ値で重なるので、青の明るい段の太い帯にして下に敷く
#: （銘柄の 2px の線が上に重なって見える）
TOTAL_STYLE = ("#86b6ef", 6)
#: 保有銘柄の線の色。dataviz の参照パレットの2番目から（1番目の青は合計の帯と同じ色相なので
#: 使わない）。先頭の青・橙・緑は色覚の差の検査に全組み合わせで通る（2026-10-08 に
#: validate_palette で確認）
SERIES_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SUMMARY_HEADER = ["銘柄", "コード", "予測日", "買った日", "建値", "株数", "20日目", "売った日",
                  "売値", "売りの理由", "最新日", "最新終値", "損益(円)", "損益%", "状態"]


@dataclass
class Position:
    pick_date: Optional[str]       # 予測日（YYYY-MM-DD）。予測ログの行を見つけるのに使う
    code: str                      # 4桁（英字を含むことがある）
    name: str
    row: Optional[int]             # 予測ログの行番号（1始まり）。無ければ None
    buy_date: Optional[pd.Timestamp]
    entry_price: Optional[float]   # 報告の買値。無ければ買った日の始値
    shares: float
    sell_date: Optional[pd.Timestamp] = None    # 報告の売り
    sell_price: Optional[float] = None
    # 計算の結果（simulate が埋める）
    state: str = ""
    reason: str = ""
    issue: str = ""                # 報告と日足が合わなかったとき（公開ログには種類だけ出す）
    entry_day: Optional[pd.Timestamp] = None
    entry: Optional[float] = None
    day20: Optional[pd.Timestamp] = None
    exit_day: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None

    @property
    def jq_code(self) -> str:
        """J-Quants の5桁コード（日足の Code）。"""
        return self.code + "0" if len(self.code) == 4 else self.code

    @property
    def label(self) -> str:
        return f"{self.name}({self.code})" if self.name else self.code


# --------------------------------------------------------------------------- #
# 値の読み取り
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


def _ymd(d) -> str:
    return "" if d is None or pd.isna(d) else f"{pd.Timestamp(d):%Y-%m-%d}"


# --------------------------------------------------------------------------- #
# 取引の記録（タブ）
# --------------------------------------------------------------------------- #

def log_rows(values: List[List[str]]) -> List[Dict[str, str]]:
    """取引の記録タブの値を、見出しの名前で辞書にする。"""
    if not values:
        return []
    header = [str(h).strip() for h in values[0]]
    out = []
    for line in values[1:]:
        rec = {h: (line[i] if i < len(line) else "") for i, h in enumerate(header)}
        if any(str(v).strip() for v in rec.values()):
            out.append(rec)
    return out


def event_record(e: Dict, names: Dict[str, str]) -> Dict[str, object]:
    """受け皿の取引1件を、取引の記録タブの見出しの名前 → 値に。"""
    code = e.get("code") or ""
    return {"id": e["id"], "受付": e.get("at", ""),
            "種類": KIND_JA.get(e.get("action", ""), e.get("action", "")),
            "予測日": e.get("pick_date") or "", "コード": code, "銘柄名": names.get(code, ""),
            "日付": e.get("date") or "",
            "価格": "" if e.get("price") is None else e["price"],
            "株数": "" if e.get("shares") is None else e["shares"],
            "取消の対象": e.get("ref") or "", "メモ": e.get("note") or ""}


def new_events(rows: List[Dict], events: List[Dict]) -> List[Dict]:
    """取引の記録にまだ無い取引（id で見分ける）。受付の順（同じ時刻なら受け皿の順）。"""
    have = {str(r.get("id", "")).strip() for r in rows}
    return sorted([e for e in events if e.get("id") and e["id"] not in have],
                  key=lambda e: e.get("at", ""))


def _kind(rec: Dict) -> str:
    return KIND_EN.get(str(rec.get("種類", "")).strip(), "")


def positions_from_log(rows: List[Dict], pred_values: List[List[str]]
                       ) -> Tuple[List[Position], List[str], Set[int]]:
    """
    取引の記録から保有を作る。

    返り値は (保有, 合わなかった行の説明, 取り消した買いの予測ログの行番号)。
    売りは、同じコードで、売った日までに買っていて、まだ売りの無い保有のうち最も新しいものに付ける
    （予測日があればそれも合わせる）。
    """
    header = [str(h).strip() for h in pred_values[0]] if pred_values else []
    idx = {h: i for i, h in enumerate(header)}

    def cell(line, name):
        i = idx.get(name)
        return line[i] if i is not None and i < len(line) else ""

    where: Dict[Tuple[str, str], Tuple[int, str]] = {}
    for r, line in enumerate(pred_values[1:], start=2):
        d = _date(cell(line, "予測日"))
        c = _code(cell(line, "コード"))
        if d is not None and c:
            where.setdefault((_ymd(d), c), (r, str(cell(line, "銘柄名")).strip()))

    issues: List[str] = []
    refs = []
    for rec in rows:
        if _kind(rec) == "cancel":
            ref = str(rec.get("取消の対象", "")).strip().lower()
            if len(ref) >= TI.MIN_REF:
                refs.append(ref)
            else:
                issues.append("取消の対象が短い（id の先頭 8 文字以上）")

    def cancelled(rec) -> bool:
        rid = str(rec.get("id", "")).strip().lower()
        return bool(rid) and any(rid.startswith(r) for r in refs)

    positions: List[Position] = []
    cleared: Set[int] = set()
    seen = set()
    for rec in rows:
        if _kind(rec) not in ("buy", "sell", "cancel"):
            issues.append("種類が 買い・売り・取消 のどれでもない")
            continue
        if _kind(rec) != "buy":
            continue
        code = _code(rec.get("コード"))
        pick = _date(rec.get("予測日"))
        day = _date(rec.get("日付"))
        row, name = where.get((_ymd(pick), code), (None, ""))
        if cancelled(rec):
            if row is not None:
                cleared.add(row)
            continue
        if not code or (pick is None and day is None):
            issues.append("買いの行に コード・予測日（か日付）が無い")
            continue
        key = (_ymd(pick), code, _ymd(day))
        if key in seen:
            issues.append("同じ買いが2回ある")
            continue
        seen.add(key)
        if pick is not None and row is None:
            issues.append("買いの予測日とコードに合う予測ログの行が無い")
        shares = _num(rec.get("株数"))
        positions.append(Position(
            pick_date=_ymd(pick) or None, code=code,
            name=name or str(rec.get("銘柄名", "")).strip(), row=row,
            buy_date=day, entry_price=_num(rec.get("価格")),
            shares=shares if shares and shares > 0 else DEFAULT_SHARES))

    def start_of(p: Position) -> pd.Timestamp:
        return p.buy_date if p.buy_date is not None else pd.Timestamp(p.pick_date)

    for rec in rows:
        if _kind(rec) != "sell" or cancelled(rec):
            continue
        code = _code(rec.get("コード"))
        pick = _date(rec.get("予測日"))
        day = _date(rec.get("日付"))
        cands = [p for p in positions if p.code == code and p.sell_date is None
                 and (pick is None or p.pick_date == _ymd(pick))
                 and day is not None and start_of(p) <= day]
        if not cands:
            issues.append("売りに合う買いが無い")
            continue
        p = max(cands, key=start_of)
        p.sell_date = day
        p.sell_price = _num(rec.get("価格"))
    active = {p.row for p in positions if p.row is not None}
    positions.sort(key=lambda p: (p.pick_date or _ymd(p.buy_date), p.code))
    return positions, issues, cleared - active


# --------------------------------------------------------------------------- #
# 損益の計算（売りのルール）
# --------------------------------------------------------------------------- #

def load_bars(data_dir: str, jq_codes: Sequence[str], since_year: Optional[int] = None
              ) -> pd.DataFrame:
    """対象銘柄の日足（Date, Code, O, H, C, AdjO, AdjH, AdjC）。since_year 以降の年のファイル
    （無ければ直近2年）を読む。"""
    cols = ["Date", "Code", "O", "H", "C", "AdjO", "AdjH", "AdjC"]
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths or not jq_codes:
        return pd.DataFrame(columns=cols)

    def year(p):
        m = re.search(r"bars_(\d{4})", os.path.basename(p))
        return int(m.group(1)) if m else None

    picked = [p for p in paths if since_year is not None and (year(p) or 0) >= since_year]
    want = set(jq_codes)
    parts = []
    for p in picked or paths[-2:]:
        try:
            df = pd.read_parquet(p, columns=cols)
        except Exception:                 # 調整後の列が無い古いファイル
            df = pd.read_parquet(p, columns=["Date", "Code", "O", "H", "C"])
        df["Code"] = df["Code"].astype(str)
        parts.append(df[df["Code"].isin(want)])
    bars = pd.concat(parts, ignore_index=True)
    for c in ("AdjO", "AdjH", "AdjC"):
        if c not in bars.columns:
            bars[c] = np.nan
    bars["Date"] = pd.to_datetime(bars["Date"]).dt.normalize()
    return bars.sort_values(["Code", "Date"]).reset_index(drop=True)


def nth_trading_day(start: pd.Timestamp, n: int, days: Sequence[pd.Timestamp], cal=None
                    ) -> Optional[pd.Timestamp]:
    """start を1日目として n 日目の営業日。取引所のカレンダーで数え、無ければ日足の日付で数える。"""
    if cal is not None and cal and cal.covers(start.date()):
        fut = sorted(d for d in cal.days if d >= start.date())
        if len(fut) >= n:
            return pd.Timestamp(fut[n - 1])
    have = sorted(pd.Timestamp(d) for d in days if pd.Timestamp(d) >= start)
    return have[n - 1] if len(have) >= n else None


def simulate(p: Position, bars: pd.DataFrame, cal=None) -> Optional[pd.DataFrame]:
    """
    売りのルールで日々の損益を出す。p の state / reason / issue / entry / exit を埋める。

    返り値は Date, close, pnl_yen, pnl_pct の表（買った日から、売った日まで。持っていれば最新日まで）。
    買えていなければ None。
    state: "保有中" / "持ち越し中" / "売却済み" / "買い待ち" / "寄り付かず" / "株価なし"
    """
    p.state = p.reason = p.issue = ""
    p.entry_day = p.entry = p.day20 = p.exit_day = p.exit_price = None
    b = bars[bars["Code"] == p.jq_code].sort_values("Date").reset_index(drop=True)
    if b.empty:
        p.state = "株価なし"
        return None
    start = (p.buy_date if p.buy_date is not None
             else pd.Timestamp(p.pick_date) + pd.Timedelta(days=1))
    after = b[b["Date"] >= start].reset_index(drop=True)
    if not after.empty and cal is not None and cal:
        gap = cal.count_between(start.date(), pd.Timestamp(after.loc[0, "Date"]).date())
        if gap:                           # 買う日の日足が保存データに無い（欠けを埋めずに待つ）
            after = after.iloc[0:0]
            p.issue = "買う日の日足が無い"
    if after.empty:
        p.state = "買い待ち"
        return None
    first = after.iloc[0]
    raw_open = float(first["O"]) if pd.notna(first["O"]) else float("nan")
    if not (math.isfinite(raw_open) and raw_open > 0) and not p.entry_price:
        p.state = "寄り付かず"
        return None
    adj_open = float(first["AdjO"]) if pd.notna(first["AdjO"]) else raw_open
    ratio = (adj_open / raw_open
             if math.isfinite(raw_open) and raw_open > 0 and adj_open > 0 else 1.0)
    entry = float(p.entry_price) if p.entry_price else raw_open
    p.entry_day = pd.Timestamp(first["Date"])
    p.entry = entry
    p.day20 = nth_trading_day(p.entry_day, HORIZON, list(b["Date"]), cal)
    last_bar = pd.Timestamp(b["Date"].max())

    adj_close = after["AdjC"].where(after["AdjC"].notna(), after["C"] * ratio)
    adj_high = after["AdjH"].where(after["AdjH"].notna(), after["H"] * ratio)
    close = (adj_close / ratio).ffill()
    high = adj_high / ratio
    exit_i: Optional[int] = None
    exit_px: Optional[float] = None
    reason = ""
    target = entry * TARGET
    window = after.index if p.day20 is None else after.index[after["Date"] <= p.day20]
    for i in window:
        h = high.iloc[i]
        if pd.notna(h) and h >= target - 1e-9:
            exit_i, exit_px, reason = int(i), target, "+10%に到達"
            break
    carried = False
    if exit_i is None and p.day20 is not None and last_bar >= p.day20:
        j = int(window[-1])               # 20日目（その日に日足が無ければ直前の日）
        c20 = close.iloc[j]
        if pd.notna(c20) and float(c20) >= entry - 1e-9:
            exit_i, exit_px, reason = j, float(c20), "20日目の終値"
        elif pd.notna(c20):
            carried = True

    extra: Optional[pd.Timestamp] = None  # まだ日足の無い日に売った（値段の報告あり）
    sell = p.sell_date
    if sell is not None and cal is not None and cal and cal.is_trading_day(sell.date()) is False:
        prev = cal.last_trading_day(sell.date())
        p.issue = "売りの日付が営業日ではない（その前の営業日にした）"
        sell = pd.Timestamp(prev) if prev is not None else sell
    if sell is not None and sell < p.entry_day:
        p.issue = "売りの日付が買った日より前"
    elif sell is not None:                # 報告の売りを優先する
        if sell <= last_bar:
            j = int(after.index[after["Date"] <= sell][-1])
            exit_i, reason, carried = j, "報告", False
            exit_px = float(p.sell_price) if p.sell_price else float(close.iloc[j])
        elif p.sell_price:
            exit_i, extra, exit_px, reason, carried = None, sell, float(p.sell_price), "報告", False
        else:                             # 値段の報告が無く、その日の日足もまだ無い
            exit_i, exit_px, reason = None, None, "売りの報告あり（その日の終値待ち）"

    path = pd.DataFrame({"Date": after["Date"], "close": close})
    if exit_i is not None:
        path = path.iloc[:exit_i + 1].copy()
        path.iloc[-1, path.columns.get_loc("close")] = exit_px
    if extra is not None:
        path = pd.concat([path, pd.DataFrame({"Date": [extra], "close": [exit_px]})],
                         ignore_index=True)
    path = path[path["close"].notna()].reset_index(drop=True)
    if path.empty:
        p.state = "寄り付かず"
        return None
    if exit_i is not None or extra is not None:
        p.state, p.exit_day, p.exit_price = SOLD, pd.Timestamp(path["Date"].iloc[-1]), exit_px
    else:
        p.state = "持ち越し中" if carried else "保有中"
    p.reason = reason or ("20日目に含み損で持ち越し" if carried else "")
    path["pnl_yen"] = (path["close"] - entry) * p.shares
    path["pnl_pct"] = (path["close"] / entry - 1.0) * 100.0
    return path


def trading_days(bars: pd.DataFrame, paths: Dict[int, pd.DataFrame]) -> List[pd.Timestamp]:
    """日付ごとの表の日付（対象銘柄の日足の日付と、日足より先の売りの日）。"""
    days = set(pd.to_datetime(bars["Date"])) if len(bars) else set()
    for path in paths.values():
        days |= set(path["Date"])
    return sorted(pd.Timestamp(d) for d in days)


def column_labels(positions: List[Position], paths: Dict[int, pd.DataFrame]) -> Dict[int, str]:
    """日付ごとの表の列の名前。同じ銘柄を2回買ったら買った日で見分ける。"""
    count: Dict[str, int] = {}
    for k in paths:
        count[positions[k].code] = count.get(positions[k].code, 0) + 1
    out = {}
    for k in paths:
        p = positions[k]
        out[k] = (p.label if count[p.code] == 1
                  else f"{p.label} {p.entry_day.month}/{p.entry_day.day}買い")
    return out


def daily_table(positions: List[Position], paths: Dict[int, pd.DataFrame],
                days: Sequence[pd.Timestamp]) -> pd.DataFrame:
    """
    日付 × 保有の表。列は 日付・合計損益(円)・合計損益%・保有数・各銘柄の損益(円)・各銘柄の損益%。

    銘柄の列は持っている間だけ（売った日まで）。合計には売ったあとも確定した損益を入れる。
    """
    live = [(k, p) for k, p in enumerate(positions) if k in paths]
    if not live:
        return pd.DataFrame(columns=["日付", "合計損益(円)", "合計損益%", "保有数"])
    first = min(p.entry_day for _, p in live)
    out = pd.DataFrame({"日付": [pd.Timestamp(d) for d in days if pd.Timestamp(d) >= first]})
    total = pd.Series(0.0, index=out.index)
    cost = pd.Series(0.0, index=out.index)
    held = pd.Series(0, index=out.index)
    labels = column_labels(positions, paths)
    yen_cols, pct_cols = [], []
    for k, p in live:
        s = paths[k].set_index("Date")
        # 日足の無い日（売買停止など）と最後の日足のあとは、直前の値のまま
        yen = out["日付"].map(s["pnl_yen"]).ffill()
        pct = out["日付"].map(s["pnl_pct"]).ffill()
        total = total + yen.fillna(0.0)
        cost = cost + yen.notna() * float(p.entry) * p.shares
        end = p.exit_day if p.state == SOLD else None
        holding = (out["日付"] >= p.entry_day) & (out["日付"] <= end if end is not None else True)
        held = held + holding.astype(int)
        if end is not None:
            yen = yen.where(out["日付"] <= end)
            pct = pct.where(out["日付"] <= end)
        yen_cols.append((f"{labels[k]} 損益(円)", yen.round(0)))
        pct_cols.append((f"{labels[k]} 損益%", pct.round(2)))
    out["合計損益(円)"] = total.round(0)
    out["合計損益%"] = (total / cost.where(cost > 0) * 100.0).round(2)
    out["保有数"] = held
    for name, col in yen_cols + pct_cols:
        out[name] = col
    return out


def chart_series(positions: List[Position], paths: Dict[int, pd.DataFrame],
                 days: Sequence[pd.Timestamp]) -> List[Tuple[str, str, int]]:
    """
    グラフの系列 (日付ごとの表の列の名前, 色, 線の太さ)。先頭が合計の帯。

    銘柄は、保有中のものと、売ってから HORIZON 営業日以内のもの。色は買った順に、グラフに出る期間
    （買った日〜売ってから HORIZON 営業日。持っていれば先まで）が重なる先の銘柄が使っていない最初の
    色にする。あとから買った銘柄で先の銘柄の色が変わらず、同時に出る銘柄どうしは必ず違う色になる。
    """
    series = [("合計損益(円)", TOTAL_STYLE[0], TOTAL_STYLE[1])]
    if not paths or not days:
        return series
    days = sorted(pd.Timestamp(d) for d in days)
    as_of = days[-1]

    def shown_until(p: Position) -> pd.Timestamp:
        if p.state != SOLD or p.exit_day is None:
            return pd.Timestamp.max
        later = [d for d in days if d > p.exit_day]
        return later[HORIZON - 1] if len(later) >= HORIZON else pd.Timestamp.max

    order = sorted(paths, key=lambda k: (positions[k].entry_day, k))
    until = {k: shown_until(positions[k]) for k in order}
    color: Dict[int, int] = {}
    for k in order:
        used = {color[j] for j in color
                if positions[j].entry_day <= until[k] and positions[k].entry_day <= until[j]}
        color[k] = next(i for i in itertools.count() if i not in used)
    labels = column_labels(positions, paths)
    for k in order:
        if until[k] >= as_of:
            series.append((f"{labels[k]} 損益(円)",
                           SERIES_COLORS[color[k] % len(SERIES_COLORS)], 2))
    return series


def summary_rows(positions: List[Position], paths: Dict[int, pd.DataFrame]) -> List[List]:
    """銘柄ごと1行と、合計・うち確定・うち含みの3行。"""
    rows: List[List] = []
    tot = {"all": 0.0, "sold": 0.0, "open": 0.0}
    cost = {"all": 0.0, "sold": 0.0, "open": 0.0}
    for k, p in enumerate(positions):
        path = paths.get(k)
        if path is None or path.empty:
            rows.append([p.name, p.code, p.pick_date or "", _ymd(p.buy_date),
                         p.entry_price or "", p.shares, "", "", "", p.reason,
                         "", "", "", "", p.state])
            continue
        last = path.iloc[-1]
        yen = float(last["pnl_yen"])
        part = "sold" if p.state == SOLD else "open"
        for key in ("all", part):
            tot[key] += yen
            cost[key] += float(p.entry) * p.shares
        rows.append([p.name, p.code, p.pick_date or "", _ymd(p.entry_day), round(float(p.entry), 1),
                     p.shares, _ymd(p.day20), _ymd(p.exit_day),
                     "" if p.exit_price is None else round(float(p.exit_price), 1), p.reason,
                     _ymd(last["Date"]), "" if p.state == SOLD else round(float(last["close"]), 1),
                     round(yen), round(float(last["pnl_pct"]), 2), p.state])

    def pct(key):
        return round(tot[key] / cost[key] * 100.0, 2) if cost[key] > 0 else ""

    yen_i = SUMMARY_HEADER.index("損益(円)")
    for label, key in (("合計", "all"), ("うち確定（売却済み）", "sold"), ("うち含み（保有中）", "open")):
        line = [""] * len(SUMMARY_HEADER)
        line[0], line[yen_i], line[yen_i + 1] = label, round(tot[key]), pct(key)
        rows.append(line)
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
    series: List[Tuple[int, str, int]]   # グラフの系列 (列 0始まり, 色, 太さ)。先頭が合計
    summary_rows: int              # 合計などの行を含む
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


def build_layout(as_of: str, summary: List[List], daily: pd.DataFrame, n_positions: int,
                 series: Sequence[Tuple[str, str, int]] = ()) -> Layout:
    """タブ全体の値。1〜2行目に説明、4行目から銘柄ごとの表、その下に日付ごとの表。"""
    note = ("チャットで報告した取引（このシートの「取引の記録」タブ）の損益。売りのルール: 買った日を1日目と"
            "して20営業日以内に日中の高値が買値の+10%に届いたら、その値で売り。届かなければ20日目の終値で"
            "売り（含み損なら売らずに持ち越し、売りの報告まで持つ）。報告の売りはルールより優先。グラフは"
            "合計と、保有中・売ってから20営業日以内の銘柄。毎晩の予測のあとに作り直します（このタブに手で"
            "書いた内容は消えます）")
    values: List[List] = [[f"保有の推移（{as_of} まで）"], [note]]
    if n_positions == 0:
        values.append(["取引の記録がありません"])
        return Layout(values, daily_header_row=-1, daily_rows=0, daily_cols=0,
                      series=[], summary_rows=0, width=1)
    values.append([])
    values.append(list(SUMMARY_HEADER))
    values.extend([[_cell(v) for v in r] for r in summary])
    values.append([])
    header_row = len(values)
    cols = list(daily.columns)
    values.append(cols)
    for rec in daily.itertuples(index=False):
        values.append([_cell(v) for v in rec])
    where = {c: i for i, c in enumerate(cols)}
    plotted = [(where[name], color, width) for name, color, width in series if name in where]
    width = max(len(SUMMARY_HEADER), len(cols))
    return Layout(values, daily_header_row=header_row, daily_rows=len(daily),
                  daily_cols=len(cols), series=plotted,
                  summary_rows=len(summary), width=width)


def _rgb(hexcolor: str) -> Dict[str, float]:
    h = hexcolor.lstrip("#")
    return {"red": int(h[0:2], 16) / 255, "green": int(h[2:4], 16) / 255,
            "blue": int(h[4:6], 16) / 255}


def chart_request(sheet_id: int, lay: Layout) -> Optional[Dict]:
    """日付ごとの表の「損益(円)」の列を折れ線にする（軸は1本、単位は円）。"""
    if lay.daily_header_row < 0 or lay.daily_rows == 0 or not lay.series:
        return None
    r0 = lay.daily_header_row
    r1 = r0 + 1 + lay.daily_rows

    def src(c):
        return {"sourceRange": {"sources": [{
            "sheetId": sheet_id, "startRowIndex": r0, "endRowIndex": r1,
            "startColumnIndex": c, "endColumnIndex": c + 1}]}}

    # 先頭が合計（帯）。系列は後ろほど上に描かれるので、銘柄の線が帯の上に乗る
    series = [{"series": src(c), "targetAxis": "LEFT_AXIS",
               "colorStyle": {"rgbColor": _rgb(color)}, "lineStyle": {"width": width}}
              for c, color, width in lay.series]
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
            "anchorCell": {"sheetId": sheet_id, "rowIndex": 3,
                           "columnIndex": lay.width + 1},
            "widthPixels": 760, "heightPixels": 420}},
    }}}


def format_requests(sheet_id: int, lay: Layout) -> List[Dict]:
    """円は桁区切りと符号、% は小数2桁と符号、値段は小数1桁。見出しは太字。"""
    reqs: List[Dict] = []

    def num(r0, r1, c, pattern):
        reqs.append({"repeatCell": {
            "range": {"sheetId": sheet_id, "startRowIndex": r0, "endRowIndex": r1,
                      "startColumnIndex": c, "endColumnIndex": c + 1},
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
    price = "#,##0.0"
    s0 = 3                                    # 銘柄ごとの表の見出し
    bold(s0)
    s1 = s0 + 1 + lay.summary_rows
    for name, pattern in (("建値", price), ("売値", price), ("最新終値", price),
                          ("損益(円)", yen), ("損益%", pct)):
        num(s0 + 1, s1, SUMMARY_HEADER.index(name), pattern)
    for r in range(s1 - 3, s1):               # 合計・うち確定・うち含み
        bold(r)
    h = lay.daily_header_row
    bold(h)
    d0, d1 = h + 1, h + 1 + lay.daily_rows
    for i, name in enumerate(lay.values[h]):
        if name.endswith("(円)"):
            num(d0, d1, i, yen)
        elif name.endswith("%"):
            num(d0, d1, i, pct)
    return reqs


def color_requests(sheet_id: int, positions: List[Position], n_cols: int,
                   cleared: Sequence[int] = ()) -> List[Dict]:
    """予測ログの、取引の記録がある行を塗る（保有中は赤、売却済みは黄色。取り消した行は塗りを外す）。"""
    def paint(row, fmt):
        return {"repeatCell": {
            "range": {"sheetId": sheet_id, "startRowIndex": row - 1, "endRowIndex": row,
                      "startColumnIndex": 0, "endColumnIndex": n_cols},
            "cell": {"userEnteredFormat": fmt},
            "fields": "userEnteredFormat.backgroundColor"}}

    reqs = [paint(p.row, {"backgroundColor": YELLOW if p.state == SOLD else RED})
            for p in positions if p.row is not None]
    reqs += [paint(r, {}) for r in sorted(cleared)]    # 項目だけ指定して値を渡さないと既定に戻る
    return reqs


def user_cell_updates(header: List[str], positions: List[Position],
                      paths: Dict[int, pd.DataFrame], cleared: Sequence[int] = ()) -> List[Dict]:
    """予測ログの、取引の記録がある行の 建値・株数・手仕舞い日・手仕舞い値・損益（取り消した行は空に）。"""
    idx = {h: i for i, h in enumerate(header)}
    out = []

    def put(row, vals):
        for name in USER_WRITE_COLS:
            if name in idx:
                out.append({"range": ES.a1(idx[name] + 1, row), "values": [[vals.get(name, "")]]})

    for k, p in enumerate(positions):
        path = paths.get(k)
        if p.row is None or path is None or path.empty:
            continue
        put(p.row, {"建値": round(float(p.entry), 1), "株数": p.shares,
                    "手仕舞い日": _ymd(p.exit_day),
                    "手仕舞い値": "" if p.exit_price is None else round(float(p.exit_price), 1),
                    "損益": round(float(path.iloc[-1]["pnl_yen"]))})
    for r in sorted(cleared):
        put(r, {})
    return out


# --------------------------------------------------------------------------- #
# シートとのやりとり
# --------------------------------------------------------------------------- #

def sync_log_tab(book, events: List[Dict], names: Dict[str, str],
                 dry_run: bool = False) -> Tuple[List[Dict], int]:
    """受け皿の取引のうち、取引の記録タブに無いものを足す。返り値は (タブの全行, 足した数)。"""
    try:
        ws = book.worksheet(LOG_TAB)
        values = ws.get_all_values()
    except Exception:
        ws, values = None, []
    rows = log_rows(values)
    new = new_events(rows, events)
    header = [str(h).strip() for h in values[0]] if values and any(values[0]) else list(LOG_HEADER)
    header += [h for h in LOG_HEADER if h not in header]     # 手で消された見出しは右端に戻す
    recs = [event_record(e, names) for e in new]
    if new and not dry_run:
        if ws is None:
            ws = book.add_worksheet(title=LOG_TAB, rows=500, cols=len(LOG_HEADER) + 2)
        if not values or header != [str(h).strip() for h in values[0]]:
            ws.update([header], "A1", value_input_option="RAW")
        # 文字のまま書く（id や日付を数や日付として読まれないように）。列は見出しの名前で合わせる
        ws.append_rows([[r.get(h, "") for h in header] for r in recs], value_input_option="RAW",
                       table_range="A1")
    rows = rows + [{h: ("" if r.get(h) is None else str(r.get(h, ""))) for h in header}
                   for r in recs]
    return rows, len(new)


def write_tab(book, lay: Layout) -> int:
    """保有の推移タブを作り直す（値・書式・グラフ）。グラフの系列数を返す。"""
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
    return len(lay.series) if chart else 0


def mark_log(book, pred_ws, header: List[str], positions: List[Position],
             paths: Dict[int, pd.DataFrame], cleared: Sequence[int] = ()) -> int:
    """予測ログの行を塗り、建値などを書く。塗った（外した）行の数を返す。"""
    reqs = color_requests(pred_ws.id, positions, len(header), cleared)
    if reqs:
        book.batch_update({"requests": reqs})
    cells = user_cell_updates(header, positions, paths, cleared)
    if cells:
        pred_ws.batch_update(cells, value_input_option="USER_ENTERED")
    return len(reqs)


def report(n_events: int, n_failed: int, n_added: int, positions: List[Position],
           issues: List[str], n_days: int, as_of: str) -> List[str]:
    """公開ログに出す行。件数と最新日だけ（銘柄名・株価・損益は出さない）。"""
    counts: Dict[str, int] = {}
    for p in positions:
        counts[p.state] = counts.get(p.state, 0) + 1
    parts = " / ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    lines = [f"[holdings] 受け皿 {n_events}件（復号できない {n_failed}）・取引の記録に足した {n_added}件",
             f"[holdings] 保有 {len(positions)}件" + (f"（{parts}）" if parts else ""),
             f"[holdings] 日付ごとの表 {n_days}日（最新 {as_of}）"]
    found = list(issues) + [p.issue for p in positions if p.issue]
    if found:
        lines.append(f"[holdings] 取引の記録で合わなかった行 {len(found)}: "
                     + "; ".join(sorted(set(found))))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="保有の損益の推移を別タブに書き、予測ログを塗る")
    ap.add_argument("--sheet-id", default=os.environ.get("GSHEET_ID", ""))
    ap.add_argument("--log-title", default=ES.SHEET_TITLE)
    ap.add_argument("--data-dir", default=ES.DATA_DIR)
    ap.add_argument("--inbox", default=TI.INBOX)
    ap.add_argument("--dry-run", action="store_true", help="シートを読んで計算だけする（書かない）")
    args = ap.parse_args(argv)
    if not args.sheet_id:
        raise SystemExit("GSHEET_ID が設定されていません（シートのURLの /d/ と /edit の間）")

    key = TI.private_key_from_env()
    events, failed = TI.read_inbox(args.inbox, key) if key is not None else ([], 0)
    book = ES.open_sheet(args.sheet_id)
    pred_ws = book.worksheet(args.log_title)
    pred_values = pred_ws.get_all_values()
    header = [str(h).strip() for h in pred_values[0]] if pred_values else []
    names: Dict[str, str] = {}
    if "コード" in header and "銘柄名" in header:
        ci, ni = header.index("コード"), header.index("銘柄名")
        for line in pred_values[1:]:
            if ci < len(line) and ni < len(line):
                names.setdefault(_code(line[ci]), str(line[ni]).strip())
    rows, added = sync_log_tab(book, events, names, dry_run=args.dry_run)
    positions, issues, cleared = positions_from_log(rows, pred_values)
    starts = [d for p in positions
              for d in (p.buy_date, _date(p.pick_date)) if d is not None]
    bars = load_bars(args.data_dir, [p.jq_code for p in positions],
                     since_year=min(starts).year if starts else None)
    try:
        cal = TC.load(args.data_dir)
    except Exception:                     # カレンダーが無くても日足の日付で進める
        cal = None
    paths: Dict[int, pd.DataFrame] = {}
    for k, p in enumerate(positions):
        path = simulate(p, bars, cal)
        if path is not None:
            paths[k] = path
    days = trading_days(bars, paths)
    daily = daily_table(positions, paths, days)
    as_of = (f"{pd.Timestamp(daily['日付'].max()):%Y-%m-%d}" if len(daily)
             else f"{pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d}")
    lay = build_layout(as_of, summary_rows(positions, paths), daily, len(positions),
                       chart_series(positions, paths, days))
    for line in report(len(events), failed, added, positions, issues, len(daily), as_of):
        print(line)
    if args.dry_run:
        print("[holdings] --dry-run なので書かない")
        return 0
    n = write_tab(book, lay)
    marked = mark_log(book, pred_ws, header, positions, paths, sorted(cleared))
    print(f"[holdings] タブ「{TAB}」を作り直した（{len(lay.values)}行・グラフ {n}系列）・"
          f"予測ログの {marked}行を塗った（取り消しで外した {len(cleared)}行を含む）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
