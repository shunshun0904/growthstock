#!/usr/bin/env python3
"""
J-Quants API V2 から全銘柄・長期間のデータを一括取得するヘルパー。

銘柄ごとにループすると 4,441銘柄 × 4エンドポイント = 約1.8万リクエストになるが、
V2 は `date` パラメータで **1リクエスト = その日の全銘柄** を返す。
実測で全4,441行が 4.19秒。営業日ベースで回すのが唯一現実的な方法。

出力は Parquet（列指向・圧縮）。10年分の日次バーは約1,080万行になるため、
JSON や素の CSV では扱えない。

--incremental を付けると、manifest（data_store.py）を見て
**まだ取得していない営業日だけ**を取得し、年別ファイルにマージする。
全期間の取得は約2.5時間かかるため、日次更新でこれを繰り返すのは現実的でない。
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from typing import Callable, Iterable, List, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jquants_data_fetcher import JQuantsClient, JQuantsError, resolve_api_key  # noqa: E402
import data_store  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "_data")

# 契約がカバーする最古の日付（research/probe_boundary.py で実測）
EARLIEST_DATE = dt.date(2016, 10, 1)

# 保持する列（全列を持つとサイズが数倍になるため、必要なものだけ）
BAR_COLS = ["Date", "Code", "O", "H", "L", "C", "Vo", "Va", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo"]
# 決算は全項目を保持する（None = 絞らない）。
#
# 以前はホワイトリストで絞っており、書き漏らした項目が取得時点で捨てられていた。
# 実際 BPS は API が返しているのに列挙しておらず、PBR を作れなかった。
# しかも「捨てた」という記録が残らないので、後から気づけない。
#
# 決算は全期間でも十数万行しかなく、株価（数千万行）と違って
# 全項目を持ってもサイズが問題にならない。列を選ぶ理由が無い。
# 何が返ってくるかは docs/DATA_FIELDS.md（probe_fins_fields.py の実測）を参照。
FIN_COLS = None
# 決算のうち数値化しない列。これ以外はすべて数値として扱う
FIN_TEXT_COLS = {
    "DiscDate", "DiscTime", "Code", "DocType", "CurPerType", "CurPerEn",
    "CurPerSt", "CurFYSt", "CurFYEnd", "NxFYSt", "NxFYEnd", "NxPerType",
    "ChgFYEnd", "RetroRestate", "Sig",
}
MARGIN_COLS = ["Date", "Code", "LongVol", "ShrtVol"]
# 銘柄マスタ。銘柄名・業種・市場区分はダッシュボード表示に必須で、
# 株価データ側には入っていない（V2 の /equities/master にしかない）
# 列は選ばない。
# 銘柄マスタは1日4,400行ほどで、月次スナップショットを全期間集めても
# 数十万行にしかならない。サイズを理由に列を絞る必要が無い。
#
# 以前は上の FIN_COLS と同じく手書きの白リストにしていて、
# 実在しない "ScaleCat" を書いたために規模区分が100%欠測の空列になり、
# しかも白リストが黙って落とすので気づけなかった。同じ事故を繰り返さない。
MASTER_COLS = None


# --------------------------------------------------------------------------- #
# 営業日
# --------------------------------------------------------------------------- #

def trading_days(client: JQuantsClient, start: dt.date, end: dt.date) -> List[dt.date]:
    """
    /markets/calendar から実際の営業日だけを取り出す。
    土日祝を自前で判定すると祝日でリクエストを無駄撃ちするため、API に従う。
    """
    rows = client.get_paginated(
        "/markets/calendar", {"from": start.isoformat(), "to": end.isoformat()}
    )
    # V2 の列名は HolDiv（V1 は HolidayDivision）。実レスポンスで確認済み。
    # 値: "0" = 非営業日, "1" = 営業日, "2" = 東証半日立会
    div_keys = ("HolDiv", "HolidayDivision", "HolidayDiv")
    days = []
    for r in rows:
        div = next((str(r[k]) for k in div_keys if k in r and r[k] is not None), "")
        d = r.get("Date")
        if not d:
            continue
        if div in ("1", "2"):
            days.append(dt.date.fromisoformat(d))
    if not days:
        raise JQuantsError(
            f"/markets/calendar が営業日を返しませんでした ({start}〜{end})。"
            f"応答例: {rows[:1]}"
        )
    return sorted(days)


# --------------------------------------------------------------------------- #
# 日次ループでの一括取得
# --------------------------------------------------------------------------- #

def _fetch_by_day(
    client: JQuantsClient,
    path: str,
    days: Iterable[dt.date],
    columns: List[str],
    label: str,
    progress_every: int = 50,
) -> pd.DataFrame:
    """日付を1日ずつ指定して全銘柄ぶんを集める。"""
    frames: List[pd.DataFrame] = []
    days = list(days)
    total = len(days)
    t0 = time.time()
    failed: List[str] = []

    for i, d in enumerate(days, 1):
        try:
            rows = client.get_paginated(path, {"date": d.isoformat()})
        except JQuantsError as exc:
            failed.append(f"{d}: {str(exc)[:120]}")
            continue
        if not rows:
            continue
        df = pd.DataFrame.from_records(rows)
        if columns is not None:
            df = df[[c for c in columns if c in df.columns]]
        frames.append(df)

        if i % progress_every == 0 or i == total:
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (total - i) / rate if rate else 0
            got = sum(len(f) for f in frames)
            print(f"  [{label}] {i}/{total}日  {got:,}行  "
                  f"{el/60:.1f}分経過  残り約{eta/60:.1f}分", flush=True)

    if failed:
        print(f"  [{label}] 取得に失敗した日: {len(failed)}件", file=sys.stderr)
        for f in failed[:5]:
            print(f"    {f}", file=sys.stderr)

    if not frames:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(frames, ignore_index=True)


def fetch_bars(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """株価四本値（全銘柄）。"""
    df = _fetch_by_day(client, "/equities/bars/daily", days, BAR_COLS, "bars")
    return _numify(df, [c for c in BAR_COLS if c not in ("Date", "Code")])


def fetch_fins(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """財務情報（その日に開示されたもの）。"""
    df = _fetch_by_day(client, "/fins/summary", days, FIN_COLS, "fins")
    # 全項目を保持しているので、数値化はテキスト列以外すべてに掛ける
    num = [c for c in df.columns if c not in FIN_TEXT_COLS]
    return _numify(df, num)


def fetch_margin(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """
    信用取引週末残高。週次データなので毎営業日叩く必要はない。
    公表は週1回なので、各週の全営業日を試すのではなく週次で間引く。
    """
    weekly = sorted({d for d in days if d.weekday() == 4})  # 金曜だけ試す
    # 金曜が休場の週を拾えないので、その週の他の日も候補に入れる
    covered = {(d.isocalendar().year, d.isocalendar().week) for d in weekly}
    for d in days:
        key = (d.isocalendar().year, d.isocalendar().week)
        if key not in covered:
            weekly.append(d)
            covered.add(key)
    df = _fetch_by_day(client, "/markets/margin-interest", sorted(weekly), MARGIN_COLS, "margin")
    return _numify(df, ["LongVol", "ShrtVol"])


def fetch_master_history(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """
    銘柄マスタを時点別に取る（業種・市場区分の point-in-time 用）。

    最新のマスタを過去のサンプルに当てると先読みになる。
    とくに市場区分は2022年4月の東証再編で全銘柄が変わっているため、
    2018年のサンプルに現在の区分を付けるのは誤り。

    毎営業日は要らない（業種はめったに変わらない）ので月次で取る。
    """
    monthly = sorted({d for d in days if d == max(
        x for x in days if (x.year, x.month) == (d.year, d.month))})
    print(f"[master_hist] {len(monthly)}時点（月末）を取得")
    df = _fetch_by_day(client, "/equities/master", monthly, MASTER_COLS,
                       "master_hist")
    return df


def fetch_master(client: JQuantsClient, as_of: dt.date) -> pd.DataFrame:
    """
    銘柄マスタ（全銘柄の名称・業種・市場区分）。1リクエストで全銘柄が返る。

    株価・財務データには銘柄名が入っていないため、これが無いと
    画面に「(銘柄名なし)」としか出せない。
    """
    rows = client.get_paginated("/equities/master", {"date": as_of.isoformat()})
    if not rows:
        raise JQuantsError(f"/equities/master が空を返しました (date={as_of})")
    df = pd.DataFrame.from_records(rows)
    if MASTER_COLS is None:
        return df
    return df[[c for c in MASTER_COLS if c in df.columns]]


def fetch_topix(client: JQuantsClient, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    TOPIX の日次終値。市場環境の特徴量に使う。
    銘柄固有の力と「地合いが良かっただけ」を分離するために必須。
    """
    rows = client.get_paginated(
        "/indices/bars/daily/topix", {"from": start.isoformat(), "to": end.isoformat()}
    )
    if not rows:
        raise JQuantsError("TOPIX のデータを取得できませんでした")
    df = pd.DataFrame.from_records(rows)
    close_col = next((c for c in ("C", "Close", "AdjC") if c in df.columns), None)
    if close_col is None:
        raise JQuantsError(f"TOPIX の終値列が見つかりません。列: {list(df.columns)}")
    out = df[["Date", close_col]].rename(columns={close_col: "topix"})
    out["topix"] = pd.to_numeric(out["topix"], errors="coerce")
    return out.dropna().sort_values("Date").reset_index(drop=True)


def _numify(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="J-Quants V2 から全銘柄データを一括取得する")
    ap.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out-dir", default=DATA_DIR)
    ap.add_argument("--pause", type=float, default=0.05,
                    help="リクエスト間隔(秒)。既定は控えめ（1日1リクエストのため）")
    ap.add_argument("--what", nargs="*",
                    default=["bars", "fins", "margin", "topix", "master", "master_hist"],
                    choices=["bars", "fins", "margin", "topix", "master",
                             "master_hist"])
    ap.add_argument("--incremental", action="store_true",
                    help="manifest を見て、まだ取得していない営業日だけを取得する")
    ap.add_argument("--reset", nargs="*", default=[],
                    choices=["bars", "fins", "margin", "topix", "master", "master_hist"],
                    help="指定した種別の保存済みデータと取得記録を消してから取得する。"
                         "取得する列を増やしたときに使う（既存 parquet には新しい列が"
                         "入っていないが、manifest 上は取得済みなので取り直されない）")
    args = ap.parse_args(argv)

    start = dt.date.fromisoformat(args.date_from)
    end = dt.date.fromisoformat(args.date_to)
    if start < EARLIEST_DATE:
        print(f"[warn] {start} は契約範囲外です。{EARLIEST_DATE} に切り上げます", file=sys.stderr)
        start = EARLIEST_DATE

    os.makedirs(args.out_dir, exist_ok=True)
    client = JQuantsClient(resolve_api_key(), pause=args.pause)

    print(f"[calendar] 営業日を取得 ({start} 〜 {end})")
    days = trading_days(client, start, end)
    print(f"[calendar] {len(days)}営業日")

    if args.incremental:
        return _run_incremental(client, days, start, end, args)

    tag = f"{start.isoformat()}_{end.isoformat()}"
    jobs: List[tuple] = []
    if "bars" in args.what:
        jobs.append(("bars", lambda: fetch_bars(client, days)))
    if "fins" in args.what:
        jobs.append(("fins", lambda: fetch_fins(client, days)))
    if "margin" in args.what:
        jobs.append(("margin", lambda: fetch_margin(client, days)))
    if "topix" in args.what:
        jobs.append(("topix", lambda: fetch_topix(client, start, end)))
    if "master" in args.what:
        jobs.append(("master", lambda: fetch_master(client, days[-1])))

    for name, fn in jobs:
        print(f"\n[{name}] 取得開始")
        t0 = time.time()
        df = fn()
        path = os.path.join(args.out_dir, f"{name}_{tag}.parquet")
        df.to_parquet(path, index=False, compression="zstd")
        size = os.path.getsize(path) / 1e6
        print(f"[{name}] {len(df):,}行 -> {path} ({size:.1f}MB, {time.time()-t0:.0f}秒)")

    return 0


def _run_incremental(client: JQuantsClient, days: List[dt.date],
                     start: dt.date, end: dt.date, args) -> int:
    """
    manifest を見て、まだ取得していない営業日だけを取得し、年別ファイルにマージする。

    「その日を取得しに行ったか」で判定する（行数ではない）。
    財務のようにその日の開示が0件でも取得済みとして扱わないと、毎回叩き直してしまう。
    """
    manifest = data_store.load_manifest(args.out_dir)

    for kind in args.reset:
        removed = data_store.reset_kind(args.out_dir, manifest, kind)
        print(f"[reset] {kind}: parquet {len(removed)}件と取得記録を削除 "
              f"-> 全期間を取り直す")
        for r in removed:
            print(f"         {r}")
    if args.reset:
        data_store.save_manifest(args.out_dir, manifest)

    print("\n[manifest] 取得済み:")
    print(data_store.summarize(manifest))

    total_new = 0
    for name in args.what:
        if name == "topix":
            continue  # topix は期間指定で一括取得するので後段でまとめて扱う
        if name == "master_hist":
            # 月次スナップショット。日付ループの共通処理には乗せず個別に扱う
            todo = data_store.missing_days(manifest, "master_hist", days)
            print(f"\n[master_hist] 候補 {len(days)}日 / 未取得 {len(todo)}日")
            if todo:
                t0 = time.time()
                mh = fetch_master_history(client, todo)
                written = data_store.merge_into_years(args.out_dir, "master_hist",
                                                      mh, "Date")
                data_store.mark_fetched(manifest, "master_hist", todo)
                total_new += len(mh)
                print(f"[master_hist] {len(mh):,}行を追加 / 更新ファイル "
                      f"{len(written)}件 ({time.time()-t0:.0f}秒)")
            else:
                print("[master_hist] 取得済み。スキップします")
            continue
        if name == "master":
            # 銘柄マスタは日付ごとの蓄積ではなく毎回最新に上書きする。
            # 後段で別に取得するので、この日付ループでは扱わない。
            #
            # ここを通していたため、下の else（catch-all）に落ちて
            # /markets/margin-interest を叩き、信用残のデータを
            # master_YYYY.parquet に書き込んでいた。
            # 2,425日ぶん・約38分を毎回無駄にしていた。
            continue
        if name == "margin":
            # 信用残は週次公表。週に1日だけ候補にする
            cand = sorted({d for d in days if d.weekday() == 4})
            covered = {(d.isocalendar().year, d.isocalendar().week) for d in cand}
            for d in days:
                key = (d.isocalendar().year, d.isocalendar().week)
                if key not in covered:
                    cand.append(d); covered.add(key)
            cand = sorted(cand)
        else:
            cand = days

        todo = data_store.missing_days(manifest, name, cand)
        print(f"\n[{name}] 候補 {len(cand)}日 / 未取得 {len(todo)}日")
        if not todo:
            print(f"[{name}] 取得済み。スキップします")
            continue

        t0 = time.time()
        # catch-all の else にしない。
        # 種別を1つ増やしたときに、黙って別のエンドポイントを叩いてしまう。
        if name == "bars":
            df = fetch_bars(client, todo)
        elif name == "fins":
            df = fetch_fins(client, todo)
        elif name == "margin":
            # todo は上で既に週次に間引いてある。
            # fetch_margin() は同じ間引きを内部でも行うので、ここでは使わない
            # （二重に掛けても結果は同じだが、それに依存したくない）
            df = _fetch_by_day(client, "/markets/margin-interest", todo,
                               MARGIN_COLS, "margin")
            df = _numify(df, ["LongVol", "ShrtVol"])
        else:
            raise SystemExit(f"日付ループで扱えない種別です: {name}")

        date_col = "DiscDate" if name == "fins" else "Date"
        written = data_store.merge_into_years(args.out_dir, name, df, date_col)
        data_store.mark_fetched(manifest, name, todo)
        total_new += len(df)
        print(f"[{name}] {len(df):,}行を追加 / 更新ファイル {len(written)}件 "
              f"({time.time()-t0:.0f}秒)")
        for w in written:
            print(f"    {os.path.basename(w)} ({os.path.getsize(w)/1e6:.1f}MB)")

    # TOPIX は軽いので取得済み範囲の外側だけ取り直す
    if "topix" in args.what:
        have = data_store.fetched_days(manifest, "topix")
        todo_tp = [d for d in days if d.isoformat() not in have]
        if todo_tp:
            print(f"\n[topix] 未取得 {len(todo_tp)}日 -> {todo_tp[0]} 〜 {todo_tp[-1]}")
            df = fetch_topix(client, todo_tp[0], todo_tp[-1])
            written = data_store.merge_into_years(args.out_dir, "topix", df, "Date")
            data_store.mark_fetched(manifest, "topix", todo_tp)
            print(f"[topix] {len(df):,}行を追加 / 更新ファイル {len(written)}件")
        else:
            print("\n[topix] 取得済み。スキップします")

    # 銘柄マスタは1リクエストで全銘柄が返る（実測4.2秒）ので毎回取り直す。
    # 新規上場・社名変更・市場区分変更を取りこぼさないため、差分にしない。
    if "master" in args.what:
        print("\n[master] 銘柄マスタを取得（毎回最新に更新）")
        try:
            df = fetch_master(client, days[-1])
            path = os.path.join(args.out_dir, "master.parquet")
            df.to_parquet(path, index=False, compression="zstd")
            print(f"[master] {len(df):,}銘柄 -> {os.path.basename(path)} "
                  f"({os.path.getsize(path)/1e6:.2f}MB)")
            manifest["master"] = {"as_of": days[-1].isoformat(), "count": int(len(df))}
        except JQuantsError as exc:
            print(f"[master] 取得できませんでした: {exc}", file=sys.stderr)

    data_store.save_manifest(args.out_dir, manifest)
    print("\n[manifest] 更新後:")
    print(data_store.summarize(manifest))
    print(f"\n[done] 新規取得 {total_new:,}行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
