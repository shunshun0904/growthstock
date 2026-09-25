#!/usr/bin/env python3
"""
日本証券金融（日証金、taisyaku.jp）の需給データを取り込む（docs/DATA_JSF.md）。

  daily    DATA ページの CSV を取り、申込日ごとに積む
             /data/zandaka.csv        銘柄別の融資・貸株残高（前営業日の申込日、確報）
             /data/shina.csv          品貸料率（逆日歩）
             /data/seigenichiran.csv  制限措置等の一覧（その日の写し）
  history  銘柄ごとの過去（3年）を、銘柄のページ → 期間を入れて検索 → CSV の順に取る。
           1銘柄3リクエスト。--max-codes で1回の数を区切り、manifest で続きから取る

保存（research/_data/。Release data-raw に上げるのはワークフロー側）
  jsf_balance.parquet    残高（日次）      キー: 申込日・code・取引所区分名
  jsf_lending.parquet    品貸料率（日次）  キー: 貸借申込日・code・取引所区分
  jsf_restrict.parquet   制限措置等の写し  キー: snap_date・code・実施措置・通知日・実施日
  jsf_hist.parquet       銘柄ごとの過去    キー: 申込日・code（取引所区分は既定の東証）
  jsf_manifest.json      history の済んだ銘柄と、daily の取得記録

利用の条件: 掲載内容は私的利用の範囲を超えて使えない（公開・第三者への提供の禁止）。
運用者の決定（2026-09-25）でこのリポジトリの Release に置く（のちに非公開にする予定）。
Actions のログは公開なので、**データの値は出さない**（件数・日付の範囲だけ）。

  $ python3 research/jsf_fetch.py daily
  $ python3 research/jsf_fetch.py history --max-codes 400
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import sys
import urllib.parse
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from jquants_data_fetcher import normalize_code  # noqa: E402
import probe_jsf as PJ  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
TARGETS = os.path.join(HERE, "edinet_targets.txt")   # 母集団に多く出る順（J-Quants の5桁コード）
MANIFEST = "jsf_manifest.json"
BASE = PJ.BASE
JST = PJ.JST

#: 保存するファイル・キー・見出しの目印
SPECS = {
    "balance": {"url": f"{BASE}/data/zandaka.csv", "file": "jsf_balance.parquet",
                "keys": ("申込日", "銘柄コード", "融資残高株数"),
                "code": "銘柄コード", "dates": ("申込日", "決済日"),
                "key": ["申込日", "code", "取引所区分名"], "date_col": "申込日"},
    "lending": {"url": f"{BASE}/data/shina.csv", "file": "jsf_lending.parquet",
                "keys": ("貸借申込日", "コード", "当日品貸料率（円）"),
                "code": "コード", "dates": ("貸借申込日", "決済日"),
                "key": ["貸借申込日", "code", "取引所区分"], "date_col": "貸借申込日"},
    "restrict": {"url": f"{BASE}/data/seigenichiran.csv", "file": "jsf_restrict.parquet",
                 "keys": ("銘柄コード", "実施措置"),
                 "code": "銘柄コード", "dates": ("通知日・実施日",),
                 "key": ["snap_date", "code", "実施措置", "通知日・実施日"], "date_col": "snap_date"},
}
HIST = {"file": "jsf_hist.parquet", "keys": ("申込日", "銘柄コード", "融資残高（株）"),
        "code": "銘柄コード", "dates": ("申込日", "直後基準日"), "key": ["申込日", "code"]}

#: 数値として読む列（見出しに含まれる語）。「－」などは欠測にする
NUMERIC_HINT = re.compile(r"株数|（株）|金額|（円）|料率|日数|権利落額|更新差金")
#: 数値にしない列（見出しに数値の語を含んでも文字のもの）
TEXT_COLS = {"応札ランク", "応札倍率ランク", "銘柄名", "市場区分", "貸借区分", "取引所区分",
             "取引所区分名", "上場区分", "速報／確報", "決算事由", "決算等", "備考", "制限",
             "制限措置", "臨時措置", "特別措置", "直近制限措置", "直近臨時措置", "直近特別措置",
             "新株引受・権利入札", "実施措置", "実施内容", "後場停止", "直近発表"}
#: history で続けて失敗したら止める数（サイトが止めている・落ちているとき叩き続けない）
MAX_CONSECUTIVE_FAIL = 5


# ---------------------------------------------------------------------- #
# 読み取り
# ---------------------------------------------------------------------- #

def to_date(s: pd.Series) -> pd.Series:
    """2026/09/24・2026-9-4・20260924 を日付に。読めないものは欠測。"""
    def one(x) -> pd.Timestamp:
        t = str(x).strip()
        m = (re.fullmatch(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", t)
             or re.fullmatch(r"(\d{4})(\d{2})(\d{2})", t))
        if not m:
            return pd.NaT
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return pd.NaT
    return pd.to_datetime(s.map(one))


def to_number(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(t, errors="coerce")


def read_table(text: str, keys: Sequence[str]) -> pd.DataFrame:
    """
    題・注記の行を飛ばし、keys をすべて含む行を見出しとして表にする。
    空の見出しと重なった見出しには名前を付ける（DATA の対象銘柄一覧に「－」の列がある）。
    """
    rows = list(csv.reader(io.StringIO(text)))
    hi = next((i for i, r in enumerate(rows)
               if all(k in [c.strip() for c in r] for k in keys)), None)
    if hi is None:
        raise ValueError(f"見出しの行が見つからない: {list(keys)}")
    cols: List[str] = []
    for j, c in enumerate(rows[hi]):
        name = c.strip() or f"_col{j}"
        cols.append(name if name not in cols else f"{name}_{j}")
    data = []
    for r in rows[hi + 1:]:
        if not any(x.strip() for x in r):
            continue
        r = (r + [""] * len(cols))[:len(cols)]
        data.append([x.strip() for x in r])
    return pd.DataFrame(data, columns=cols)


def normalize(df: pd.DataFrame, code_col: str, dates: Iterable[str]) -> pd.DataFrame:
    """銘柄コードを J-Quants の5桁に、日付と数値の列を型に直す。"""
    out = df.copy()
    out["code"] = out[code_col].map(lambda c: normalize_code(c) if str(c).strip() else None)
    out = out[out["code"].notna() & out["code"].astype(str).str.fullmatch(r"[0-9][0-9A-Z]{3}0?")]
    for c in dates:
        if c in out.columns:
            out[c] = to_date(out[c])
    for c in out.columns:
        if c in TEXT_COLS or c in dates or c in ("code", code_col):
            continue
        if NUMERIC_HINT.search(c):
            out[c] = to_number(out[c])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------- #
# 保存
# ---------------------------------------------------------------------- #

def merge_store(path: str, new: pd.DataFrame, key: List[str]) -> Tuple[pd.DataFrame, int]:
    """保存済みと合わせ、キーが重なれば新しい方を残す。(全体, 増えた行数) を返す。"""
    old = pd.read_parquet(path) if os.path.exists(path) else None
    n_old = 0 if old is None else len(old)
    allf = new if old is None else pd.concat([old, new], ignore_index=True, sort=False)
    allf = allf.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    for c in allf.columns:
        # 文字と欠測が混ざる列は parquet が型を決められないので文字に寄せる
        if allf[c].dtype == object:
            allf[c] = allf[c].astype("string")
    allf.to_parquet(path, index=False, compression="zstd")
    return allf, len(allf) - n_old


def load_manifest(data_dir: str) -> dict:
    p = os.path.join(data_dir, MANIFEST)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
    else:
        m = {}
    m.setdefault("daily", {})
    m.setdefault("hist", {})
    return m


def save_manifest(data_dir: str, m: dict) -> None:
    m["updatedAt"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with open(os.path.join(data_dir, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=1, sort_keys=True)


def span(s: pd.Series) -> str:
    s = pd.to_datetime(s).dropna()
    return f"{s.min().date()} 〜 {s.max().date()}" if len(s) else "-"


# ---------------------------------------------------------------------- #
# daily
# ---------------------------------------------------------------------- #

def fetch_daily(f: PJ.Fetcher, data_dir: str, m: dict,
                now: Optional[dt.datetime] = None) -> int:
    """DATA ページの3本を取り、積む。取れなかった本数を返す。"""
    now = now or dt.datetime.now(JST)
    failed = 0
    for kind, sp in SPECS.items():
        st, headers, body, final = f.get(sp["url"])
        if st != 200 or not body:
            print(f"  [{kind}] 取れない（HTTP {st}）")
            failed += 1
            continue
        text, enc = PJ.decode(body)
        try:
            df = normalize(read_table(text, sp["keys"]), sp["code"], sp["dates"])
        except ValueError as e:
            print(f"  [{kind}] 形が変わった: {e}")
            failed += 1
            continue
        if kind == "restrict":
            df["snap_date"] = pd.Timestamp(now.date())
        df["_fetched_at"] = now.isoformat()
        df["_last_modified"] = headers.get("Last-Modified", "")
        path = os.path.join(data_dir, sp["file"])
        allf, added = merge_store(path, df, sp["key"])
        d = df[sp["date_col"]]
        print(f"  [{kind}] {len(df):,}行・{df['code'].nunique():,}銘柄（{span(d)}、文字コード {enc}、"
              f"更新 {headers.get('Last-Modified', '?')}）/ 保存 {len(allf):,}行（+{added:,}）"
              f"・{span(allf[sp['date_col']])}")
        m["daily"][kind] = {"last": span(d).split(" 〜 ")[-1], "rows": int(len(df)),
                            "at": now.isoformat()}
    return failed


# ---------------------------------------------------------------------- #
# history
# ---------------------------------------------------------------------- #

def read_targets(path: str, jsf_codes: Iterable[str]) -> List[str]:
    """取る順番: 母集団に多く出る順（edinet_targets.txt）で日証金にある銘柄 → 残り。"""
    have = set(jsf_codes)
    order: List[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                c = ln.strip()
                if c and not c.startswith("#") and c in have and c not in order:
                    order.append(c)
    order += sorted(have - set(order))
    return order


def display4(code5: str) -> str:
    return code5[:4] if len(code5) == 5 and code5.endswith("0") else code5


def fetch_one_history(f: PJ.Fetcher, code5: str, start: dt.date, end: dt.date
                      ) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    1銘柄の過去を取る。('ok' | 'no_data' | 'error:...', 表) を返す。
    銘柄のページ（合言葉と cookie）→ 期間を入れて検索 → CSV。
    """
    c4 = display4(code5)
    detail = f"{BASE}/app/stock/detail/{c4}-01"
    st, _, body, final = f.get(detail)
    if st == 404:
        return "no_data", None                  # 東証（-01）の銘柄のページが無い
    if st != 200 or not body:
        return f"error:detail {st}", None
    page = PJ.parse_html(body)
    form = next((x for x in page.forms if x["action"].rstrip("/").endswith(f"/detail/{c4}/search")),
                None)
    if form is None:
        return "no_data", None
    payload = PJ.form_payload(form)
    payload["mkYmdFrom"] = [start.strftime("%Y/%m/%d")]
    payload["mkYmdTo"] = [end.strftime("%Y/%m/%d")]
    # 全日（実測: 空で731営業日、"7" で43行、"2" で12行。2026-09-25 run 36125941082）。
    # 画面の既定に頼らず明示する
    payload["kjnYmdDays"] = [""]
    action = urllib.parse.urljoin(final, form["action"])
    st, _, _, final2 = f.get(action, data=payload, referer=final)
    if st != 200:
        return f"error:search {st}", None
    st, _, body3, _ = f.get(f"{BASE}/app/stock/detail/{c4}/csv", referer=final2)
    if st != 200 or not body3:
        return f"error:csv {st}", None
    if body3.lstrip().startswith(b"<"):
        return "error:csv html", None
    text, _ = PJ.decode(body3)
    try:
        df = normalize(read_table(text, HIST["keys"]), HIST["code"], HIST["dates"])
    except ValueError:
        return "error:shape", None
    df = df[df["code"] == code5]
    if df.empty:
        return "no_data", None
    return "ok", df


def _flush(data_dir: str, m: dict, frames: List[pd.DataFrame], now: dt.datetime) -> int:
    """取れた分を保存し、manifest も書く（途中で打ち切られても、ここまでの分は残る）。"""
    if not frames:
        save_manifest(data_dir, m)
        return 0
    new = pd.concat(frames, ignore_index=True, sort=False)
    new["_fetched_at"] = now.isoformat()
    allf, added = merge_store(os.path.join(data_dir, HIST["file"]), new, HIST["key"])
    save_manifest(data_dir, m)
    print(f"  [hist] 保存 {len(frames):,}銘柄・{len(new):,}行（{span(new['申込日'])}）/ 累計 "
          f"{len(allf):,}行・{allf['code'].nunique():,}銘柄（+{added:,}行）")
    frames.clear()
    return added


def fetch_history(f: PJ.Fetcher, data_dir: str, m: dict, codes: List[str], max_codes: int,
                  years: float, now: Optional[dt.datetime] = None,
                  time_budget: Optional[float] = None, checkpoint: int = 50,
                  clock: Callable[[], float] = None) -> Tuple[int, int]:
    """
    (取りに行った銘柄数, 増えた行数)。済んだ銘柄は飛ばす。失敗は次回また試す。

    checkpoint 銘柄ごとに保存する。1銘柄に実測約8.4秒（3リクエスト）かかるので、
    ステップの打ち切りに掛かっても、それまでの分は失わない。time_budget（秒）を過ぎたら
    次の銘柄に進まない。
    """
    import time as _time
    clock = clock or _time.monotonic
    t0 = clock()
    now = now or dt.datetime.now(JST)
    end = now.date()
    start = end - dt.timedelta(days=int(365.25 * years))
    todo = [c for c in codes if m["hist"].get(c, {}).get("status") not in ("ok", "no_data")]
    print(f"  取る銘柄 残り {len(todo):,} / {len(codes):,}（この実行の上限 {max_codes}"
          f"{f'・{time_budget / 60:.0f}分' if time_budget else ''}） / 期間 {start} 〜 {end}")
    frames: List[pd.DataFrame] = []
    done = fails = added = 0
    for code in todo[:max_codes]:
        if f.used + 3 > f.max:
            print(f"  [stop] リクエストの上限 {f.max} に近い")
            break
        if time_budget is not None and clock() - t0 > time_budget:
            print(f"  [stop] 時間の上限 {time_budget / 60:.0f}分に達した（{done}銘柄）")
            break
        status, df = fetch_one_history(f, code, start, end)
        done += 1
        rec = m["hist"].get(code, {})
        if status == "ok":
            frames.append(df)
            m["hist"][code] = {"status": "ok", "rows": int(len(df)), "at": now.isoformat(),
                               "from": str(df["申込日"].min().date()), "to": str(df["申込日"].max().date())}
            fails = 0
        elif status == "no_data":
            m["hist"][code] = {"status": "no_data", "at": now.isoformat()}
            fails = 0
        else:
            m["hist"][code] = {"status": "error", "why": status, "at": now.isoformat(),
                               "tries": int(rec.get("tries", 0)) + 1}
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAIL:
                print(f"  [stop] {fails}銘柄続けて失敗（最後: {status}）。ここで止める")
                break
        if done % checkpoint == 0:
            added += _flush(data_dir, m, frames, now)
    added += _flush(data_dir, m, frames, now)
    st = pd.Series([v.get("status") for v in m["hist"].values()]).value_counts().to_dict()
    print(f"  [hist] 済み {st.get('ok', 0):,} / データ無し {st.get('no_data', 0):,} / "
          f"失敗（次回また試す）{st.get('error', 0):,} / 取る銘柄 {len(codes):,}")
    return done, added


def jsf_codes(data_dir: str) -> List[str]:
    """日証金の残高に出てくる銘柄（東証の行）。"""
    p = os.path.join(data_dir, SPECS["balance"]["file"])
    if not os.path.exists(p):
        return []
    b = pd.read_parquet(p, columns=["code", "取引所区分名"])
    tse = b[b["取引所区分名"].astype(str).str.contains("東証", na=False)]
    return sorted(set((tse if len(tse) else b)["code"].astype(str)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="日証金（taisyaku.jp）の需給データを取り込む")
    ap.add_argument("mode", choices=["daily", "history", "both"])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--targets", default=TARGETS)
    ap.add_argument("--max-codes", type=int, default=400, help="history で1回に取る銘柄数")
    ap.add_argument("--years", type=float, default=3.0, help="history で求める年数（公開は3年まで）")
    ap.add_argument("--pause", type=float, default=2.0, help="リクエストの間の秒数")
    ap.add_argument("--max-requests", type=int, default=1300)
    ap.add_argument("--time-budget", type=float, default=None,
                    help="history でこの秒数を過ぎたら次の銘柄に進まない（ステップの打ち切り対策）")
    args = ap.parse_args(argv)

    os.makedirs(args.data_dir, exist_ok=True)
    print(f"[jsf] {PJ.now_jst()} / {args.mode} / 間隔 {args.pause}秒 / 上限 {args.max_requests}回")
    f = PJ.Fetcher(args.max_requests, args.pause)
    m = load_manifest(args.data_dir)
    rc = 0
    if args.mode in ("daily", "both"):
        print("\n=== daily ===")
        failed = fetch_daily(f, args.data_dir, m)
        save_manifest(args.data_dir, m)
        if failed == len(SPECS):
            print("[error] DATA の CSV が1本も取れなかった")
            rc = 1
    if args.mode in ("history", "both"):
        print("\n=== history ===")
        codes = read_targets(args.targets, jsf_codes(args.data_dir))
        if not codes:
            print("[stop] 取る銘柄が分からない（先に daily を回して残高の銘柄を得る）")
            return rc or 1
        fetch_history(f, args.data_dir, m, codes, args.max_codes, args.years,
                      time_budget=args.time_budget)
        save_manifest(args.data_dir, m)
    print(f"\n[done] 使ったリクエスト {f.used}（{PJ.now_jst()}）")
    return rc


if __name__ == "__main__":
    sys.exit(main())
