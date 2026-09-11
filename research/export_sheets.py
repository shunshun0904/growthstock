#!/usr/bin/env python3
"""
予測の全候補を Google スプレッドシートに追記し、その後の値動きを更新する。

なぜリポジトリではなくスプレッドシートか
----------------------------------------
運用が手売買なので、モデルの推奨と「実際にいくらで入って、いくらで出たか」を
突き合わせられないと良し悪しが判断できない。同じ行に自分で書き足せることが要る。

リポジトリ側（public/data/*.json）は画面用で、
  ・predictions.json は毎日まるごと上書き（過去は git 履歴の中）
  ・prediction_history.json は上位5件だけ・400日で削除
なので、長期の台帳にはならない。こちらを正本にする。

列の扱い（ここが壊れると利用者の記入が消える）
--------------------------------------------
・列は**見出しの名前で探す**。位置では探さない。
  利用者が途中に列を挿しても壊れないようにするため。
・このスクリプトが書くのは OWNED_COLS と TRACK_COLS だけ。
  それ以外の列（建値・手仕舞い・メモなど）は読みも書きもしない。
・行は追記のみ。既存行を作り直さない。
  作り直すと、その行に書かれた手入力が消える。

  GOOGLE_SERVICE_ACCOUNT_JSON='{...}' GSHEET_ID=... python3 research/export_sheets.py
  python3 research/export_sheets.py --dry-run   # 通信せず、書く内容だけ出す
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
PUBLIC_DIR = os.path.join(os.path.dirname(HERE), "public", "data")

SHEET_TITLE = "予測ログ"

#: このスクリプトが書き込む列（追記時に一度だけ埋める）。
#: 順番は初回にシートを作るときの並びで、以降は見出し名で探すので動かしてよい。
OWNED_COLS = [
    "予測日", "コード", "銘柄名", "業種", "順位", "候補数",
    "スコア", "帯", "較正確率%", "帯の正例率%", "帯の実収益%", "帯の勝率%",
    "必要上昇率%", "予測時株価", "時価総額(億)", "売買代金(億/日)",
    "日次ボラ%", "20日リターン%", "地合い寄与", "銘柄固有寄与", "PER", "PBR",
]

#: 毎回更新する列。実行のたびに最新の株価で書き直す。
TRACK_COLS = ["現在値", "騰落率%", "経過営業日"]

#: 見出しだけ作って中身は一切触らない列。利用者の記入欄。
USER_COLS = ["建値", "株数", "手仕舞い日", "手仕舞い値", "損益", "メモ"]

KEY_COLS = ("予測日", "コード")


def rows_from_predictions(pred: Dict) -> List[Dict]:
    """全候補を1行ずつにする。上位だけに絞らない（選ばなかった側も検証したいので）。"""
    out = []
    for c in pred["candidates"]:
        ct = c.get("contrib") or {}
        out.append({
            "予測日": c["date"], "コード": c["code"], "銘柄名": c.get("name") or "",
            "業種": c.get("sector") or "",
            "順位": c["rankInDay"], "候補数": c["nInDay"],
            "スコア": c["score"], "帯": c["band"],
            "較正確率%": c.get("calibProb"),
            "帯の正例率%": c.get("bandPositiveRate"),
            "帯の実収益%": c.get("bandEndMedian"),
            "帯の勝率%": c.get("bandWinRate"),
            "必要上昇率%": c.get("needPct"),
            "予測時株価": c.get("close"),
            "時価総額(億)": c.get("marketCap"),
            "売買代金(億/日)": c.get("tradingValue"),
            "日次ボラ%": c.get("vol20d"),
            "20日リターン%": c.get("ret20d"),
            "地合い寄与": ct.get("marketContrib"),
            "銘柄固有寄与": ct.get("stockContrib"),
            "PER": c.get("per"), "PBR": c.get("pbr"),
            "_jqCode": c["jqCode"],
        })
    return out


def latest_closes(data_dir: str) -> "pd.Series":
    """銘柄ごとの最新終値。追跡列の更新に使う。"""
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        return pd.Series(dtype=float)
    bars = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "C"])
                      for p in paths[-2:]], ignore_index=True)
    bars["Date"] = pd.to_datetime(bars["Date"])
    return bars.sort_values("Date").groupby("Code")["C"].last()


def track_values(jq_code: str, price_at_pick, pick_date: str,
                 closes: "pd.Series", as_of: Optional[pd.Timestamp]) -> Dict:
    """その行の『いま』。値が取れないところは空にする（0 で埋めない）。"""
    cur = closes.get(jq_code)
    if cur is None or not np.isfinite(cur) or not price_at_pick:
        return {}
    days = (int(np.busday_count(np.datetime64(pick_date), np.datetime64(as_of.date())))
            if as_of is not None else "")
    return {"現在値": round(float(cur), 1),
            "騰落率%": round(float(cur) / float(price_at_pick) * 100 - 100, 2),
            "経過営業日": days}


def open_sheet(sheet_id: str):
    """サービスアカウントでシートを開く。認証情報は環境変数から読むだけで、出力しない。"""
    import gspread
    from google.oauth2.service_account import Credentials

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise SystemExit(
            "GOOGLE_SERVICE_ACCOUNT_JSON が設定されていません。"
            "サービスアカウントの JSON をそのまま入れてください")
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"GOOGLE_SERVICE_ACCOUNT_JSON が JSON として読めません: {e}")
    creds = Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    return gspread.authorize(creds).open_by_key(sheet_id)


def ensure_worksheet(book, title: str):
    """無ければ作り、見出し行を置く。既にあれば触らない。"""
    try:
        ws = book.worksheet(title)
        return ws, False
    except Exception:
        pass
    header = OWNED_COLS + TRACK_COLS + USER_COLS
    ws = book.add_worksheet(title=title, rows=2000, cols=max(30, len(header) + 5))
    ws.update([header], "A1")
    ws.freeze(rows=1)
    return ws, True


def a1(col_idx: int, row_idx: int) -> str:
    """1始まりの列番号を A1 記法に。列が26を超えるので自前で桁上げする。"""
    s = ""
    n = col_idx
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return f"{s}{row_idx}"


def sync(ws, rows: List[Dict], closes, as_of, dry_run: bool = False) -> Dict[str, int]:
    values = ws.get_all_values()
    if not values:
        raise SystemExit("シートが空です。見出し行が作られていません")
    header = values[0]
    pos = {name: i for i, name in enumerate(header)}
    missing = [c for c in KEY_COLS if c not in pos]
    if missing:
        raise SystemExit(f"見出しに必要な列がありません: {missing}")

    # 既存行の位置。キーは (予測日, コード)
    seen = {}
    for r, line in enumerate(values[1:], start=2):
        if len(line) <= max(pos[KEY_COLS[0]], pos[KEY_COLS[1]]):
            continue
        seen[(line[pos[KEY_COLS[0]]], line[pos[KEY_COLS[1]]])] = r

    # --- 追記する行 --- #
    new = [x for x in rows if (x["予測日"], x["コード"]) not in seen]
    appended = []
    for x in new:
        line = [""] * len(header)
        merged = dict(x)
        merged.update(track_values(x["_jqCode"], x.get("予測時株価"),
                                   x["予測日"], closes, as_of))
        for name, v in merged.items():
            if name.startswith("_") or name not in pos:
                continue
            line[pos[name]] = "" if v is None else v
        appended.append(line)

    # --- 既存行の追跡列だけ更新する --- #
    updates = []
    by_key = {(x["予測日"], x["コード"]): x for x in rows}
    for (d, code), r in seen.items():
        x = by_key.get((d, code))
        price = None
        jq = None
        if x:
            price, jq = x.get("予測時株価"), x["_jqCode"]
        else:
            # シートにあってこの日の予測に無い行（過去分）。
            # 予測時株価は行から読む。コードは5桁に直す
            line = values[r - 1]
            if "予測時株価" in pos and len(line) > pos["予測時株価"]:
                try:
                    price = float(line[pos["予測時株価"]])
                except (TypeError, ValueError):
                    price = None
            jq = f"{code}0" if len(str(code)) == 4 else str(code)
        tv = track_values(jq, price, d, closes, as_of)
        for name, v in tv.items():
            if name in pos:
                updates.append({"range": a1(pos[name] + 1, r), "values": [[v]]})

    if dry_run:
        print(f"[dry-run] 追記 {len(appended)}行 / 追跡列の更新 {len(updates)}セル")
        for line in appended[:3]:
            print("  " + " | ".join(str(v) for v in line[:10]) + " …")
        return {"appended": len(appended), "updated": len(updates)}

    if appended:
        ws.append_rows(appended, value_input_option="RAW")
    if updates:
        # まとめて投げる。1セルずつだと候補数×日数ぶん API を叩いて上限に当たる
        for i in range(0, len(updates), 500):
            ws.batch_update(updates[i:i + 500], value_input_option="RAW")
    return {"appended": len(appended), "updated": len(updates)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="予測をスプレッドシートに追記する")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--predictions",
                    default=os.path.join(PUBLIC_DIR, "predictions.json"))
    ap.add_argument("--sheet-id", default=os.environ.get("GSHEET_ID", ""))
    ap.add_argument("--title", default=SHEET_TITLE)
    ap.add_argument("--dry-run", action="store_true",
                    help="通信せず、書き込む内容だけ出す")
    args = ap.parse_args(argv)

    pred = json.load(open(args.predictions, encoding="utf-8"))
    rows = rows_from_predictions(pred)
    closes = latest_closes(args.data_dir)
    as_of = None
    paths = sorted(glob.glob(os.path.join(args.data_dir, "bars_*.parquet")))
    if paths:
        d = pd.read_parquet(paths[-1], columns=["Date"])
        as_of = pd.Timestamp(pd.to_datetime(d["Date"]).max())
    print(f"[load] 候補 {len(rows)}行（{pred['dates'][0]} 〜 {pred['asOf']}） "
          f"/ 最新終値 {len(closes):,}銘柄（{as_of.date() if as_of is not None else '—'}）")

    if args.dry_run:
        # 通信しないので、見出しは初期構成を仮定して整合だけ見る
        header = OWNED_COLS + TRACK_COLS + USER_COLS
        print(f"[dry-run] 列 {len(header)}個: {' / '.join(header)}")
        unknown = sorted({k for x in rows for k in x
                          if not k.startswith('_') and k not in header})
        if unknown:
            raise SystemExit(f"見出しに無い項目を書こうとしています: {unknown}")
        for x in rows[:3]:
            tv = track_values(x["_jqCode"], x.get("予測時株価"), x["予測日"],
                              closes, as_of)
            print(f"  {x['予測日']} {x['コード']} {x['銘柄名']} "
                  f"{x['順位']}/{x['候補数']}位 帯{x['帯']} → {tv}")
        print("[dry-run] 通信していません")
        return 0

    if not args.sheet_id:
        raise SystemExit("GSHEET_ID が設定されていません（シートのURLの /d/ と /edit の間）")
    book = open_sheet(args.sheet_id)
    ws, created = ensure_worksheet(book, args.title)
    if created:
        print(f"[sheet] ワークシート「{args.title}」を作成し、見出しを置きました")
    res = sync(ws, rows, closes, as_of)
    print(f"[done] 追記 {res['appended']}行 / 追跡列の更新 {res['updated']}セル")
    print(f"[done] https://docs.google.com/spreadsheets/d/{args.sheet_id}/edit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
