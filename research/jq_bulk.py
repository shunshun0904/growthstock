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
import trading_calendar  # noqa: E402
from jquants_data_fetcher import JQuantsClient, JQuantsError, resolve_api_key  # noqa: E402
import data_store  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "_data")

# 契約がカバーする最古の日付（research/probe_boundary.py で実測）
EARLIEST_DATE = dt.date(2016, 10, 1)

# 保持する列（全列を持つとサイズが数倍になるため、必要なものだけ）
BAR_COLS = ["Date", "Code", "O", "H", "L", "C", "Vo", "Va", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo"]

#: 営業日カレンダーを何日ぶん先まで取っておくか。
#: 下流（鮮度チェック・営業日ゲート）は「今日」がカレンダーの範囲に
#: 入っていないと使えない。取り込みが当日まだ走っていない時刻に予測が
#: 動くことがあるので、余裕を持たせる。年末年始の休みより長く取る。
CALENDAR_LOOKAHEAD_DAYS = 45
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
# 2026-09-22 に足した種別
#
# エンドポイント一覧を辿って測った結果（docs/DATA_FIELDS.md）、スタンダードで
# 使えるのに取り込んでいないものが10本あった。運用者の判断で、1本ずつ検証
# するのではなく**全部入れてから3モデル×5分割で一括評価**する。
#
# 列は絞らない（None）。FIN_COLS / MASTER_COLS で白リストを書いて取りこぼした
# 事故を2回起こしているので、同じことをしない。行数はどれも小さい。
# --------------------------------------------------------------------------- #

#: 日付を1日ずつ指定して取る種別 -> (パス, 日付列, 表示名)
#:
#: 日付列は「その行をいつ知りえたか」を表すものを選ぶ。EDINET は提出日
#: （SubDate）、信用規制は公表日（PubDate）。ここを取り違えると、
#: 未来の情報で学習することになる。
DAILY_KINDS = {
    "valuation":   ("/equities/valuation", "Date",
                    "バリュエーション（BPS/EPS/FwdEPS/PER/PBR/ROE/FwdPER/FwdROE）"),
    "lvshld":      ("/edinet/large-volume-shareholders", "SubDate",
                    "大量保有報告書"),
    "mjrshld":     ("/edinet/major-shareholders", "SubDate", "大株主"),
    "xhold":       ("/edinet/cross-shareholdings", "SubDate", "政策保有株"),
    "shortratio":  ("/markets/short-ratio", "Date", "業種別の空売り比率"),
    "marginalert": ("/markets/margin-alert", "PubDate", "信用取引の規制・残高警報"),
    # **SchDate ではなく PubDate。** 実測（2026-09-01〜18 を取得）で
    # ?date=X は PubDate で絞っており、SchDate は中央値64日先を指していた
    # （範囲内 PubDate 100% / SchDate 0%）。SchDate で記録すると
    # 取得済みの管理がずれるうえ、**未来の情報で学習する**ことになる。
    "earndate":    ("/fins/earnings-date", "PubDate", "決算発表予定日"),
    # 0.5% 以上の空売りの持ち高（報告者ごと）。1日300〜900行、2016-10 から。
    # 引数は disc_date（公表日）。date は HTTP 400（実測 2026-09-24、
    # research/probe_shortsale.py）。同じ日・同じ銘柄に報告者が何人もいるので、
    # 行のキーは data_store.ROW_KEYS で全列にしてある
    "shortsale":   ("/markets/short-sale-report", "DiscDate", "空売り残高報告"),
}

#: 日付の引数名が date でない種別。書いていない種別は date
DAY_PARAM = {
    "shortsale": "disc_date",
}

#: 日付を指定せず一度に全部返る種別 -> (パス, 日付列, 表示名)
#: 投資部門別は週次で、全期間でも2,400行ほど。1リクエストで足りる。
#: 日付列は公表日（PubDate）。集計期間の末日（EnDate）はその約6日前で、
#: そちらを「知りえた日」として扱うと公表前の値を使うことになる
#: （2026-09-24 に特徴量側で見つかった。research/availability.py）
BULK_KINDS = {
    "investor": ("/equities/investor-types", "PubDate", "投資部門別売買"),
}


# --------------------------------------------------------------------------- #
# 営業日
# --------------------------------------------------------------------------- #

def trading_days(client: JQuantsClient, start: dt.date, end: dt.date,
                 out_dir: Optional[str] = None) -> List[dt.date]:
    """
    /markets/calendar から実際の営業日だけを取り出す。
    土日祝を自前で判定すると祝日でリクエストを無駄撃ちするため、API に従う。

    応答は **保存する**（research/_data/calendar.parquet）。下流の鮮度
    チェックと営業日ゲートが、平日で数える代わりにこれを読む。
    叩いているのに捨てていたせいで、2026-09-22 の連休で日次予測が落ちた。

    取りに行く範囲は end より **CALENDAR_LOOKAHEAD_DAYS 日ぶん先**まで。
    保存したカレンダーが「今日」を覆っていないと、下流は平日で数える側に
    倒れる。取り込みが当日まだ走っていない時刻に予測が動くことがあるので、
    先まで持っておく（取引所は先の予定まで公表している）。
    日次ループに返すのは end までの営業日だけ。先の日を取りに行かせない。
    """
    fetch_to = end + dt.timedelta(days=CALENDAR_LOOKAHEAD_DAYS)
    rows = client.get_paginated(
        "/markets/calendar", {"from": start.isoformat(), "to": fetch_to.isoformat()}
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
            day = dt.date.fromisoformat(d)
            if day <= end:                     # 先の日は保存だけして、取りに行かない
                days.append(day)
    if not days:
        raise JQuantsError(
            f"/markets/calendar が営業日を返しませんでした ({start}〜{end})。"
            f"応答例: {rows[:1]}"
        )
    # 保存に失敗しても取得は続ける。ここが新しい障害の入口にならないように
    try:
        path = trading_calendar.save(rows, out_dir or DATA_DIR)
        if path:
            print(f"[calendar] {len(rows)}日ぶん（{fetch_to} まで）を保存 -> {path}")
    except Exception as exc:                                 # noqa: BLE001
        print(f"[warn] カレンダーを保存できませんでした（続行する）: "
              f"{type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
    return sorted(days)


# --------------------------------------------------------------------------- #
# 日次ループでの一括取得
# --------------------------------------------------------------------------- #

#: 直前の _fetch_by_day で**問い合わせ自体が失敗した日**。
#:
#: 失敗した日を「取得済み」として記録すると、その日は二度と取りに行かない
#: （missing_days は「候補 − 記録」なので候補から消える）。一過性の 503 で
#: 1日ぶんが永久に失われることになるので、記録から外すために持ち回す。
#: 呼ぶたびに入れ替わるので、_fetch_by_day の直後に読むこと。
LAST_FAILED_DAYS: List[dt.date] = []


def _fetch_by_day(
    client: JQuantsClient,
    path: str,
    days: Iterable[dt.date],
    columns: List[str],
    label: str,
    progress_every: int = 50,
    param: str = "date",
) -> pd.DataFrame:
    """日付を1日ずつ指定して全銘柄ぶんを集める。param は日付の引数名。"""
    global LAST_FAILED_DAYS
    frames: List[pd.DataFrame] = []
    days = list(days)
    total = len(days)
    t0 = time.time()
    failed: List[str] = []
    LAST_FAILED_DAYS = []

    for i, d in enumerate(days, 1):
        try:
            rows = client.get_paginated(path, {param: d.isoformat()})
        except JQuantsError as exc:
            failed.append(f"{d}: {str(exc)[:120]}")
            LAST_FAILED_DAYS.append(d)
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


def margin_candidates(days: List[dt.date]) -> List[dt.date]:
    """
    信用残を叩く日。週次公表なので、週に1日だけ試す。

    残高は**その週の最終営業日**時点のもの。ふつうは金曜だが、金曜が
    休場なら木曜（以前）になる。

    以前は「金曜を全部足し、金曜が無い週だけその週の適当な日を足す」と
    していた。この「適当な日」が週の**先頭**（多くは月曜）になるため、
    金曜が祝日の週は月曜を叩いて0件のまま取得済みにしていた。
    保存データを数えると、10年ぶんで23週がこれで空いていた
    （2026-03-19、2020-03-19 など。いずれも金曜が祝日の週）。
    """
    last: dict = {}
    for d in days:
        k = (d.isocalendar().year, d.isocalendar().week)
        if k not in last or d > last[k]:
            last[k] = d
    return sorted(last.values())


def fetch_margin(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """
    信用取引週末残高。週次データなので毎営業日叩く必要はない。
    公表は週1回なので、各週の全営業日を試すのではなく週次で間引く。
    """
    df = _fetch_by_day(client, "/markets/margin-interest",
                       margin_candidates(days), MARGIN_COLS, "margin")
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


#: 指数を1リクエストで列挙できる件数（実測79件）。
#: 未取得日がこれ以下なら日付指定のほうが少ないリクエストで済む
INDEX_ENUM_SIZE = 79


def fetch_indices_by_date(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """
    日付指定で全指数を取る。1リクエスト = その日の全指数（実測79件）。

    日次更新はこちらが速い（1日1リクエスト）。
    """
    frames = []
    for i, d in enumerate(days, 1):
        rows = client.get_paginated("/indices/bars/daily", {"date": d.isoformat()})
        if rows:
            frames.append(pd.DataFrame.from_records(rows))
        if i % 200 == 0 or i == len(days):
            print(f"  [indices] {i}/{len(days)}日")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_indices_by_code(client: JQuantsClient, start: dt.date, end: dt.date,
                          as_of: dt.date) -> pd.DataFrame:
    """
    コード指定で全期間をまとめて取る。1リクエスト = 1指数の全期間。

    初回の取り込みはこちらが速い。2,428営業日を日付指定で回すと
    2,428リクエストになるが、コード指定なら79リクエストで済む。

    どの指数が存在するかは日付指定で1回引いて列挙する
    （一覧エンドポイント /indices と /indices/master は403。
      docs/MARKET_DATA.md の実測）。
    """
    codes = sorted({str(r.get("Code"))
                    for r in client.get_paginated(
                        "/indices/bars/daily", {"date": as_of.isoformat()})})
    if not codes:
        raise JQuantsError(f"{as_of} の指数を列挙できませんでした")
    print(f"  [indices] {as_of} 時点の指数: {len(codes)}件")
    frames = []
    for i, code in enumerate(codes, 1):
        rows = client.get_paginated(
            "/indices/bars/daily",
            {"code": code, "from": start.isoformat(), "to": end.isoformat()})
        if rows:
            frames.append(pd.DataFrame.from_records(rows))
        if i % 20 == 0 or i == len(codes):
            print(f"  [indices] {i}/{len(codes)}件")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_indices(client: JQuantsClient, days: List[dt.date]) -> pd.DataFrame:
    """
    指数の日次バー。業種指数を特徴量に使うために要る。

    どのコードがどの業種かは research/identify_indices.py で実測した
    （docs/INDEX_MAPPING.md）。ここでは全件そのまま保存し、
    どれを使うかは build_dataset.py 側で決める。

    未取得日の数で取り方を切り替える。日付指定は1日1リクエスト、
    コード指定は1指数1リクエスト。少ないほうを選ぶ。
    """
    if not days:
        return pd.DataFrame()
    if len(days) <= INDEX_ENUM_SIZE:
        df = fetch_indices_by_date(client, days)
    else:
        df = fetch_indices_by_code(client, days[0], days[-1], days[-1])
    if df.empty:
        raise JQuantsError("指数のデータを取得できませんでした")
    keep = [c for c in ("Date", "Code", "O", "H", "L", "C") if c in df.columns]
    df = df[keep]
    return _numify(df, [c for c in ("O", "H", "L", "C") if c in df.columns])


#: API が「値なし」を表すのに使う文字。実測で marginalert の ShrtOutChg に
#: '-' が混ざっており、数値と同じ列に入るため parquet の書き出しが落ちた
#:   ArrowInvalid: Could not convert '-' with type str: tried to convert to double
#: "None" / "null" は入れない。本物の None は既に欠測として扱われており、
#: 文字列として現れる保証も無い。入れると入れ子の列（list / dict）を
#: 欠測と誤判定する（実測で xhold.Largest がそうなった）。
NULL_MARKERS = {"-", "－", "—", "ー", "N/A", "n/a", ""}


def _sanitize(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    欠測記号を NaN にし、数値になる列は数値にする。

    **列は落とさない。** 数値に直せない列は文字列のまま残す（型が混ざって
    いると parquet が書けないので、そこだけ揃える）。白リストで列を絞って
    取りこぼす事故を2回起こしているので、ここでも捨てない。
    """
    if not len(df):
        return df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != object:
            continue
        v = out[c]
        # **入れ子（list / dict）の列は触らない。**
        # 大株主一覧（Hldrs）や政策保有の明細（Report / Largest）がこれで、
        # 文字列に潰すと構造が壊れる。parquet は入れ子のまま書ける。
        sample = v.dropna()
        if len(sample) and isinstance(sample.iloc[0], (list, dict, tuple)):
            continue
        # 欠測記号は**文字列のときだけ**見る。他の型を str 化して
        # 判定すると、入れ子や None を巻き込む
        mark = v.map(lambda x: isinstance(x, str) and x.strip() in NULL_MARKERS)
        col = v.where(~mark)
        num = pd.to_numeric(col, errors="coerce")
        # 元が非NaNだったところが全部数値になったなら数値列とみなす
        if not (col.notna() & num.isna()).any():
            out[c] = num
        else:
            out[c] = col.astype("string")
    return out


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
                    default=(["bars", "fins", "margin", "topix", "indices",
                              "master", "master_hist"]
                             + list(DAILY_KINDS) + list(BULK_KINDS)),
                    choices=(["bars", "fins", "margin", "topix", "indices",
                              "master", "master_hist"]
                             + list(DAILY_KINDS) + list(BULK_KINDS)))
    ap.add_argument("--incremental", action="store_true",
                    help="manifest を見て、まだ取得していない営業日だけを取得する")
    ap.add_argument("--reset", nargs="*", default=[],
                    choices=(["bars", "fins", "margin", "topix", "indices",
                              "master", "master_hist"]
                             + list(DAILY_KINDS) + list(BULK_KINDS)),
                    help="指定した種別の保存済みデータと取得記録を消してから取得する。"
                         "取得する列を増やしたときに使う（既存 parquet には新しい列が"
                         "入っていないが、manifest 上は取得済みなので取り直されない）")
    ap.add_argument("--forget", nargs="*", default=[],
                    choices=(["bars", "fins", "margin", "topix", "indices",
                              "master_hist"]
                             + list(DAILY_KINDS) + list(BULK_KINDS)),
                    help="指定した種別の**取得記録だけ**を消す（parquet は消さない）。"
                         "公表前に叩いて0件のまま取得済みになった日を取り直すとき用。"
                         "--forget-from で範囲を絞れる")
    ap.add_argument("--forget-from", default=None, metavar="YYYY-MM-DD",
                    help="--forget で記録を消す範囲の開始日。省略すると全期間")
    args = ap.parse_args(argv)

    start = dt.date.fromisoformat(args.date_from)
    end = dt.date.fromisoformat(args.date_to)
    if start < EARLIEST_DATE:
        print(f"[warn] {start} は契約範囲外です。{EARLIEST_DATE} に切り上げます", file=sys.stderr)
        start = EARLIEST_DATE

    os.makedirs(args.out_dir, exist_ok=True)
    client = JQuantsClient(resolve_api_key(), pause=args.pause)

    print(f"[calendar] 営業日を取得 ({start} 〜 {end})")
    days = trading_days(client, start, end, args.out_dir)
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
    if "indices" in args.what:
        jobs.append(("indices", lambda: fetch_indices(client, days)))
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


JST = dt.timezone(dt.timedelta(hours=9))


def _today_jst() -> dt.date:
    """公表ラグの判定に使う「今日」。Actions のランナーは UTC なので明示する。"""
    return dt.datetime.now(JST).date()


def _record_fetched(manifest: dict, name: str, todo: List[dt.date],
                    df: pd.DataFrame, date_col: str,
                    today: Optional[dt.date] = None,
                    failed: Optional[List[dt.date]] = None) -> None:
    """
    取得済みの記録を付ける。ただし**公表前だったかもしれない日は外す**。

    記録は「叩きに行ったか」で付ける設計（開示0件の日を毎回叩き直さない
    ため）。それだけだと、公表前に叩いた日を取得済みにしてしまい、
    その日のデータは二度と取りに行かなくなる。取り込みを 16:05 JST に
    前倒ししたので、指数(16:30)・財務(18:00)が毎日これに当たる。

    問い合わせ自体が失敗した日も外す。一過性の 503 で1日ぶんを永久に
    失わないため。既定で LAST_FAILED_DAYS（直前の _fetch_by_day の結果）を
    見る。渡し忘れたときに「余計に取り直す」側へ倒れるようにしてあり、
    逆（黙って失う側）にはしていない。_fetch_by_day を通さない経路
    （topix / indices）は failed=[] を明示すること。

    外した日は次回の候補に戻るだけなので、遅れて入るが失われない。
    判定そのものは data_store.confirmed_days が持つ（公表ラグの表も）。
    """
    today = today or _today_jst()
    bad = set(LAST_FAILED_DAYS if failed is None else failed)
    got = set()
    if len(df) and date_col in df.columns:
        got = set(pd.to_datetime(df[date_col], errors="coerce").dropna().dt.date)
    ok = [d for d in data_store.confirmed_days(name, todo, got, today)
          if d not in bad]
    data_store.mark_fetched(manifest, name, ok)
    held = [d for d in todo if d not in set(ok)]
    if held:
        shown = ", ".join(d.isoformat() for d in held[:5])
        more = f" ほか{len(held)-5}日" if len(held) > 5 else ""
        why = ("問い合わせに失敗したか、0件でまだ公表前かもしれない"
               if bad else "0件で、まだ公表前かもしれない")
        print(f"[{name}] {why}{len(held)}日は"
              f"取得済みにしない（次回また取りに行く）: {shown}{more}")


def calendar_note(days: List[dt.date], end: dt.date,
                  today: Optional[dt.date] = None) -> dict:
    """
    取得時点の営業日カレンダーを manifest に残すための1行。

    なぜ要るか
    --------
    日次予測は「新しいデータが無い」を2通りに取り違えうる。

      (a) 今日は非営業日なので、そもそも新しいデータが無い（正常）
      (b) 取り込みが壊れていて新しいデータが入っていない（異常）

    鮮度チェック（check_freshness.py）は平日で数えるので、この2つを
    区別できない。連休は (a) なのに (b) として落ちる。

    営業日カレンダーを引いているのは取り込みのここだけなので、ここで
    事実として書き残し、下流（research/trading_day_gate.py）に渡す。
    予測側にカレンダーを持ち込むのではなく、既に知っている側から渡す形に
    してある。予測側が自分で「祝日だから」と言い訳できる余地を作らない。

    「今日が営業日か」は、要求した期間の終わりが今日以降のときしか
    分からない（過去日を指定して取り直した場合、days に今日は入らない）。
    分からないときは None を入れる。下流は None を「判断材料なし」として
    扱い、通常どおり鮮度チェックへ進む。
    """
    today = today or _today_jst()
    known = bool(days) and end >= today
    return {
        "asOfJst": today.isoformat(),
        "requestedTo": end.isoformat(),
        "isTradingDay": (today in set(days)) if known else None,
        "lastTradingDay": days[-1].isoformat() if days else None,
    }


def _run_incremental(client: JQuantsClient, days: List[dt.date],
                     start: dt.date, end: dt.date, args) -> int:
    """
    manifest を見て、まだ取得していない営業日だけを取得し、年別ファイルにマージする。

    「その日を取得しに行ったか」で判定する（行数ではない）。
    財務のようにその日の開示が0件でも取得済みとして扱わないと、毎回叩き直してしまう。

    ただし**公表前に叩いた日は例外**。0件が返った日のうち公表ラグの中に
    あるものは記録せず、次回の候補に戻す（_record_fetched）。
    そうしないと、まだ出ていないだけの日を「取得済み」にして永久に失う。
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

    # --forget は記録だけを外す。保存済みの parquet はそのまま残るので、
    # 取り直せなければ現状維持で済む（reset のように穴が開かない）
    since = (dt.date.fromisoformat(args.forget_from)
             if args.forget_from else None)
    for kind in args.forget:
        dropped = data_store.forget_days(manifest, kind, since=since)
        rng = f"{dropped[0]} 〜 {dropped[-1]}" if dropped else "なし"
        print(f"[forget] {kind}: 取得記録 {len(dropped)}日を外す（{rng}）"
              f" -> この差分取得で取り直す。parquet は消していない")
    if args.forget:
        data_store.save_manifest(args.out_dir, manifest)

    print("\n[manifest] 取得済み:")
    print(data_store.summarize(manifest))

    total_new = 0
    for name in args.what:
        if name in ("topix", "indices") or name in BULK_KINDS:
            continue  # 期間指定・引数なしで一括取得するので後段で扱う
        if name == "master_hist":
            # 月次スナップショット。日付ループの共通処理には乗せず個別に扱う
            todo = data_store.missing_days(manifest, "master_hist", days)
            print(f"\n[master_hist] 候補 {len(days)}日 / 未取得 {len(todo)}日")
            if todo:
                t0 = time.time()
                mh = fetch_master_history(client, todo)
                written = data_store.merge_into_years(args.out_dir, "master_hist",
                                                      mh, "Date")
                _record_fetched(manifest, "master_hist", todo, mh, "Date")
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
            # 信用残は週次公表。週の最終営業日だけを候補にする
            # （fetch_margin と同じ関数を呼ぶ。二重に持つとずれる）
            cand = margin_candidates(days)
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
        elif name in DAILY_KINDS:
            path_, _, _ = DAILY_KINDS[name]
            df = _sanitize(_fetch_by_day(client, path_, todo, None, name,
                                         param=DAY_PARAM.get(name, "date")), name)
        else:
            raise SystemExit(f"日付ループで扱えない種別です: {name}")

        if name == "fins":
            date_col = "DiscDate"
        elif name in DAILY_KINDS:
            # 応答に無ければ Date に倒す。無い列で merge すると全部落ちる
            want = DAILY_KINDS[name][1]
            date_col = want if want in df.columns else "Date"
            if want not in df.columns and len(df):
                print(f"[warn] {name}: 期待した日付列 {want} が無い。"
                      f"Date で記録する（応答の列: {sorted(df.columns)[:12]}）")
        else:
            date_col = "Date"
        try:
            written = data_store.merge_into_years(args.out_dir, name, df, date_col)
        except Exception as exc:                             # noqa: BLE001
            # **新しい種別の失敗で、既存の取り込み全体を巻き込まない。**
            # 取得記録も付けないので、次回また取りに行く。
            # （実測: marginalert の '-' で parquet が書けず、後続の
            #   earndate / investor が丸ごと走らなかった）
            if name in DAILY_KINDS:
                print(f"[warn] {name} を保存できませんでした（続行する）: "
                      f"{type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
                continue
            raise
        _record_fetched(manifest, name, todo, df, date_col)
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
            # TOPIX は期間指定の1リクエスト。失敗すれば例外で落ちるので
            # 「黙って失敗した日」は無い。直前の _fetch_by_day の記録が
            # 残っているだけなので、明示的に空を渡す
            _record_fetched(manifest, "topix", todo_tp, df, "Date",
                            failed=[])
            print(f"[topix] {len(df):,}行を追加 / 更新ファイル {len(written)}件")
        else:
            print("\n[topix] 取得済み。スキップします")

    # 指数も TOPIX と同じく、取得済みでない日だけ取り直す。
    # 未取得日が多ければコード指定、少なければ日付指定に切り替わる
    if "indices" in args.what:
        have = data_store.fetched_days(manifest, "indices")
        todo_ix = [d for d in days if d.isoformat() not in have]
        if todo_ix:
            print(f"\n[indices] 未取得 {len(todo_ix)}日 -> {todo_ix[0]} 〜 {todo_ix[-1]}")
            df = fetch_indices(client, todo_ix)
            written = data_store.merge_into_years(args.out_dir, "indices", df, "Date")
            # 指数も同じ（fetch_indices_* は例外を握りつぶさない）
            _record_fetched(manifest, "indices", todo_ix, df, "Date",
                            failed=[])
            print(f"[indices] {len(df):,}行を追加 / 更新ファイル {len(written)}件")
        else:
            print("\n[indices] 取得済み。スキップします")

    # 引数なしで全期間が返る種別（いまは投資部門別売買だけ）。
    # 週次で全期間でも2,400行ほどなので、差分にせず毎回まるごと取り直す。
    # 差分にすると「どこまで取ったか」を持つ必要があり、1リクエストで済む
    # ものにその仕組みを足す理由が無い。
    for name, (path_, date_col, ja) in BULK_KINDS.items():
        if name not in args.what:
            continue
        print(f"\n[{name}] {ja}を取得（毎回まるごと）")
        try:
            rows = client.get_paginated(path_, {})
            df = _sanitize(pd.DataFrame.from_records(rows), name) if rows \
                else pd.DataFrame()
            if not len(df):
                print(f"[{name}] 0件。保存しない")
                continue
            out = os.path.join(args.out_dir, f"{name}.parquet")
            df.to_parquet(out, index=False, compression="zstd")
            manifest[name] = {"rows": int(len(df)),
                              "as_of": _today_jst().isoformat()}
            rng = ""
            if date_col in df.columns:
                rng = f" / {df[date_col].min()} 〜 {df[date_col].max()}"
            print(f"[{name}] {len(df):,}行 -> {os.path.basename(out)}{rng}")
        except JQuantsError as exc:
            # 1種別の失敗で取り込み全体を落とさない。次回また取りに行く
            print(f"[warn] {name} を取得できませんでした（続行する）: "
                  f"{str(exc)[:160]}", file=sys.stderr)

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

    # 営業日カレンダーを事実として書き残す（理由は calendar_note）
    cal = calendar_note(days, end)
    manifest["calendar"] = cal
    if cal["isTradingDay"] is False:
        print(f"\n[calendar] {cal['asOfJst']} は非営業日。"
              f"直近の営業日は {cal['lastTradingDay']}")
    elif cal["isTradingDay"] is True:
        print(f"\n[calendar] {cal['asOfJst']} は営業日")
    else:
        print(f"\n[calendar] {cal['asOfJst']} が営業日かは判断しない"
              f"（要求の終わりが {cal['requestedTo']}）")

    data_store.save_manifest(args.out_dir, manifest)
    print("\n[manifest] 更新後:")
    print(data_store.summarize(manifest))
    print(f"\n[done] 新規取得 {total_new:,}行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
