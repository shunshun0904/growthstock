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
・このスクリプトが書く列が増えたときは、**見出し行の右端に足す**。
  途中に挿し込むと既存セルがずれて、利用者の記入が別の列に移る。
  列の探索は名前なので、右端にあっても正しく書ける。位置が気になるときは
  利用者が手でドラッグして動かしてよい（名前を変えないこと）。

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M  # noqa: E402

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

# 注: 順位・帯・スコア・較正確率はすべて基準モデル（LightGBM）のもの。
# モデル別の列（MODEL_COLS）は隣に並べる参考値で、順位には効かない。

#: モデル別の列。画面に並べている5モデルと同じ順・同じ記号。
#:
#: 入れるのは**そのモデル自身の過去スコア分布での位置**（0〜100）だけ。
#: 生スコアは学習器ごとにスケールも意味も違うので、台帳に並べても
#: 足したり比べたりできない。位置なら同じ物差しになる。
#:
#: 「一致」は上位10%と見ているモデルの数。独立した判定の数え上げで、
#: スコアを混ぜた値ではない（アンサンブルはしない方針）。
MODEL_COLS = [f"{M.SHORT.get(a, a[:3].upper())}%" for a in M.ALGOS]
AGREE_COL = "一致(上位10%)"

#: 毎回更新する列。実行のたびに最新の株価で書き直す。
TRACK_COLS = ["現在値", "騰落率%", "経過営業日"]

#: 見出しだけ作って中身は一切触らない列。利用者の記入欄。
USER_COLS = ["建値", "株数", "手仕舞い日", "手仕舞い値", "損益", "メモ"]

KEY_COLS = ("予測日", "コード")


def rows_from_predictions(pred: Dict) -> List[Dict]:
    """
    全候補を1行ずつにする。上位だけに絞らない（選ばなかった側も検証したいので）。

    モデル別の列は、その予測ファイルが持っているモデルだけ埋める。
    5モデルを学習する前の予測ファイル（byModel が無い）でも落ちない。
    """
    # 予測ファイルが持っているモデルの順。無ければ models.ALGOS の順
    algos = [m["algo"] for m in (pred.get("models") or [])] or list(M.ALGOS)
    short = {m["algo"]: m.get("short") for m in (pred.get("models") or [])}
    out = []
    for c in pred["candidates"]:
        ct = c.get("contrib") or {}
        per = c.get("byModel") or {}
        # そのモデル自身の過去分布での位置。生スコアは入れない
        # （学習器ごとにスケールが違い、台帳で比べられないため）
        by_model = {
            f"{short.get(a) or M.SHORT.get(a, a[:3].upper())}%":
                (per.get(a) or {}).get("pctHistorical")
            for a in algos if a in per
        }
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
            **by_model,
            AGREE_COL: (f"{c['agree90']}/{c['nModels']}"
                        if c.get("nModels") else None),
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


def _mask(v: Optional[str]) -> str:
    """先頭だけ残して伏せる。公開ログに完全な値を残さないため。"""
    if not v:
        return "無し"
    v = str(v)
    if "@" in v:
        local, _, domain = v.partition("@")
        suffix = ".".join(domain.rsplit(".", 3)[-3:]) if "." in domain else domain
        return f"{local[:4]}…@…{suffix}"
    return f"{v[:4]}…（{len(v)}文字）"


def describe_credentials(raw: str) -> str:
    """
    認証情報の「形」だけを説明する（値そのものは絶対に出力しない）。

    ここで一番多い取り違えは、サービスアカウントの鍵(JSON)ではなく
    APIキー（AIza... の文字列）を登録してしまうこと。APIキーは
    公開データの読み取り用で、非公開シートへの書き込みには使えない。
    エラーだけ見ても原因が分からないので、形の段階で言い当てる。
    """
    v = (raw or "").strip()
    if not v:
        return "未設定"
    out = [f"長さ {len(v)} 文字"]
    if v.startswith("AIza"):
        out.append(
            "APIキーの形式 (AIza...)"
            "\n      → これは公開データ読み取り用のキーで、非公開シートへの"
            "\n        書き込みには使えません。"
            "\n        サービスアカウントの『鍵(JSON)』の中身を貼ってください"
            "\n        （{ \"type\": \"service_account\", ... } で始まる文字列）")
        return " / ".join(out)
    if not v.startswith("{"):
        out.append("JSON ではない（{ で始まっていない）")
        if len(v) == 40 and all(c in "0123456789abcdefABCDEF" for c in v):
            out.append(
                "40桁の16進 → サービスアカウントの『鍵ID』の可能性が高いです"
                "\n      → 鍵IDは識別子であって認証情報ではありません。"
                "\n        鍵を作成したときにダウンロードされた JSON ファイルの"
                "\n        中身（{ \"type\": \"service_account\", ... }）を貼ってください。"
                "\n        ダウンロードし損ねた場合は、鍵を作り直せば再取得できます")
        elif "@" in v and v.endswith("gserviceaccount.com"):
            out.append(
                "サービスアカウントの『メールアドレス』のようです"
                "\n      → これはシートの共有先に使うもので、認証情報ではありません。"
                "\n        鍵(JSON)の中身を貼ってください")
        else:
            out.append("→ 鍵ファイルの中身をそのまま貼り付けてください"
                       "\n        （{ \"type\": \"service_account\", ... } で始まります）")
        return " / ".join(out)
    try:
        info = json.loads(v)
    except json.JSONDecodeError as e:
        return " / ".join(out + [f"JSON として読めない: {e}"])
    out.append(f"type={info.get('type')}")
    # client_email と project_id は認証情報ではないが、公開リポジトリの
    # Actions ログに残るので伏せる。診断に要るのは「入っているか」までで、
    # 完全な値は利用者が GCP の画面で見られる。private_key は当然出さない
    out.append(f"client_email={_mask(info.get('client_email'))}")
    out.append(f"project_id={_mask(info.get('project_id'))}")
    out.append("private_key=" + ("あり" if info.get("private_key") else "無し"))
    if info.get("type") != "service_account":
        out.append("→ type が service_account ではありません")
    return " / ".join(out)


def check(sheet_id: str, title: str) -> int:
    """
    疎通テスト。認証・シートを開く・書き込み権限、の3つを順に確かめる。

    書き込みは一時ワークシートを作って1セル書き、読み戻してから消す。
    既存のシートには触らない。読めるだけでは足りない（追記できないと
    毎晩ここで失敗する）ので、実際に書いて確かめる。
    """
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    print(f"[1/4] 認証情報の形: {describe_credentials(raw)}")
    if not raw.strip():
        print("      GOOGLE_SERVICE_ACCOUNT_JSON が未設定です")
        return 1
    if not sheet_id:
        print("[fatal] GSHEET_ID が未設定です（シートURLの /d/ と /edit の間）")
        return 1
    print(f"[2/4] シートID: 長さ {len(sheet_id)} 文字")

    try:
        book = open_sheet(sheet_id)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[fatal] シートを開けませんでした: {type(e).__name__}: {e}")
        print("      よくある原因:")
        print("        ・シートをサービスアカウントに共有していない")
        print("          → GCP の [IAMと管理 » サービスアカウント] で")
        print("            該当アカウントのメールアドレスを確認し、")
        print("            シートの共有に『編集者』で追加してください")
        print("        ・GSHEET_ID が違う（URLの /d/ と /edit の間だけ）")
        print("        ・Google Sheets API が有効になっていない")
        return 1
    print(f"[3/4] シートを開けました: 「{book.title}」 / "
          f"既存のワークシート: {[w.title for w in book.worksheets()]}")

    tmp = None
    try:
        tmp = book.add_worksheet(title="_疎通テスト", rows=2, cols=2)
        stamp = pd.Timestamp.utcnow().isoformat()
        tmp.update([["疎通テスト", stamp]], "A1")
        back = tmp.get("A1:B1")
        ok = bool(back) and back[0][0] == "疎通テスト"
        print(f"[4/4] 書き込み: {'成功' if ok else '書けたが読み戻せない'}（読み戻し: {back}）")
    except Exception as e:
        print(f"[fatal] 書き込めませんでした: {type(e).__name__}: {e}")
        print("      共有の権限が『閲覧者』になっていないか確認してください"
              "（『編集者』が必要です）")
        return 1
    finally:
        if tmp is not None:
            try:
                book.del_worksheet(tmp)
                print("      一時ワークシートは削除しました")
            except Exception as e:
                print(f"[warn] 一時ワークシート「_疎通テスト」を消せませんでした: {e}"
                      "\n      手で削除してください")
    print("\n[OK] 認証・シート・書き込み権限ともに問題ありません")
    return 0


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
    header = OWNED_COLS + MODEL_COLS + [AGREE_COL] + TRACK_COLS + USER_COLS
    ws = book.add_worksheet(title=title, rows=2000, cols=max(30, len(header) + 5))
    ws.update([header], "A1")
    ws.freeze(rows=1)
    return ws, True


def ensure_columns(ws, header: List[str], dry_run: bool = False) -> List[str]:
    """
    このスクリプトが書く列のうち、見出しに無いものを**右端に足す**。

    列が増えたときに黙って書き落とさないため。sync は名前で列を探すので、
    見出しに無い列の値は捨てられる（このセッションで実際にモデル別の列で
    起きかけた）。ここで見出しを伸ばしておけば、次の追記から入る。

    途中に挿し込まないのが肝。挿すと既存セルがずれて、利用者が書いた
    建値やメモが別の列に移る。右端への追加は空セルへの書き込みなので、
    既存の中身に触らない。並び順が気になるときは利用者が手で動かしてよい
    （名前を変えなければ、そのまま正しく書き込まれる）。
    """
    want = OWNED_COLS + MODEL_COLS + [AGREE_COL] + TRACK_COLS
    missing = [c for c in want if c not in header]
    if not missing:
        return header
    print(f"[header] 見出しに無い列を右端に足す: {missing}")
    if dry_run:
        return header + missing
    start = len(header) + 1
    ws.update([missing], f"{a1(start, 1)}:{a1(start + len(missing) - 1, 1)}")
    return header + missing


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
    header = ensure_columns(ws, values[0], dry_run=dry_run)
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

    # --- 既存行を更新する（追跡列と、空のままのモデル別列だけ） --- #
    updates = []
    # 列が増えた直後は、直近5営業日ぶんの既存行にモデル別の値が入っていない。
    # 空のセルにだけ入れる。既に値があるセルは触らない（利用者が手で
    # 上書きしている可能性がある。この台帳は手書きと同居する前提）
    backfill = [c for c in MODEL_COLS + [AGREE_COL] if c in pos]
    by_key = {(x["予測日"], x["コード"]): x for x in rows}
    for (d, code), r in seen.items():
        x = by_key.get((d, code))
        price = None
        jq = None
        if x:
            price, jq = x.get("予測時株価"), x["_jqCode"]
            line = values[r - 1]
            for name in backfill:
                v = x.get(name)
                if v is None:
                    continue
                cur = line[pos[name]] if len(line) > pos[name] else ""
                if str(cur).strip():
                    continue
                updates.append({"range": a1(pos[name] + 1, r),
                                "values": [[v]]})
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
    ap.add_argument("--check", action="store_true",
                    help="疎通テストのみ。予測データは書かない")
    args = ap.parse_args(argv)

    if args.check:
        return check(args.sheet_id, args.title)

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
        header = OWNED_COLS + MODEL_COLS + [AGREE_COL] + TRACK_COLS + USER_COLS
        print(f"[dry-run] 列 {len(header)}個: {' / '.join(header)}")
        unknown = sorted({k for x in rows for k in x
                          if not k.startswith('_') and k not in header})
        if unknown:
            raise SystemExit(f"見出しに無い項目を書こうとしています: {unknown}")
        mc = [c for c in MODEL_COLS + [AGREE_COL]]
        print(f"[dry-run] モデル別: {' / '.join(mc)}")
        for x in rows[:5]:
            print("  " + " ".join(
                f"{c}={x.get(c) if x.get(c) is not None else '—'}"
                for c in [ "予測日", "コード"] + mc))
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
