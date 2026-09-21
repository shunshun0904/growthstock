#!/usr/bin/env python3
"""
EDINET DB (edinetdb.jp) から有価証券報告書ベースの年次財務を差分で取り込む。

何を取るか
--------
- 会社一覧（EDINET コード ↔ 証券コード ↔ 業種）: `/companies` を 200件×20ページ。
  J-Quants の Code と結合するための対応表。月1回で足りる
- 各社の年次財務: `/companies/{EDINETコード}/financials?years=30`。
  1社1リクエストで全期（実測 2012年〜）が返る

取れる項目と制約は docs/DATA_EDINETDB.md（実測）。値の無い項目は応答から
省かれるので、列は「見た会社の和集合」になる。欠測は欠測のまま保存する。

利用枠
-----
フリープランは 100/日・900/月。ここが一番の制約なので、
  - 1回の実行で使う数を --fetch で固定し
  - 応答ヘッダの残数（x-ratelimit-remaining）が予備を割ったら止め
  - 月の使用数を manifest に記録して、月の予算を超えない
ようにする。429 が返ったら記録せずに止める（翌日また取る）。

取る順番は research/edinet_targets.txt（母集団に多く出る銘柄が先）。
200 で取れた会社と 404（データ無し）の会社は記録し、REFRESH_AFTER_DAYS
（60日）が過ぎるまで叩かない。過ぎたら、未取得の会社を全部片づけた後に
古い順に取り直す（有報は年1回増える。新規上場の 404 もいずれデータが付く）。
それ以外の失敗は記録して次回また試す。

保存
----
  research/_data/edinet_companies.parquet   対応表
  research/_data/edinet_fin.parquet         年次財務（全社ぶん1ファイル）
  research/_data/edinet_manifest.json       取得状況・月ごとの使用数
GitHub Release（data-raw）に他の生データと同じく置く（ワークフロー側）。

鍵は EDINET_API_KEY。出力には出さない（probe_edinetdb.Probe が伏せる）。

  $ EDINET_API_KEY=... python3 research/edinet_fetch.py --mapping auto --fetch 85
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from jquants_data_fetcher import normalize_code  # noqa: E402
from probe_edinetdb import Probe, as_rows  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
TARGETS = os.path.join(HERE, "edinet_targets.txt")
MANIFEST = "edinet_manifest.json"
COMPANIES = "edinet_companies.parquet"
FIN = "edinet_fin.parquet"

DAILY_LIMIT = 100
MONTHLY_LIMIT = 900
#: 月の予算。上限いっぱい使うと probe や手動確認の余地が無くなる
MONTHLY_BUDGET = 850
#: 日次の残数がこれを割ったら止める（同じ日の他の用途に残す）
DAILY_RESERVE = 10
#: 対応表を取り直す間隔
MAPPING_MAX_AGE_DAYS = 30
#: 取得済み（200）・データ無し（404）の会社を取り直す間隔。有報は年1回なので、
#: 全社を一巡（約3,800社 ÷ 85社/日 ≈ 45日）してから古い順に更新する
REFRESH_AFTER_DAYS = 60
PAGE_SIZE = 200
YEARS = 30
KEY_COLS = ["edinet_code", "fiscal_year", "doc_id"]


# ---------------------------------------------------------------------- #
# 状態
# ---------------------------------------------------------------------- #

def load_manifest(data_dir: str) -> dict:
    p = os.path.join(data_dir, MANIFEST)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
    else:
        m = {}
    m.setdefault("mapping_at", None)
    m.setdefault("requests", {})     # "YYYY-MM" -> 使った数
    m.setdefault("companies", {})    # EDINET コード -> {status, ...}
    return m


def save_manifest(data_dir: str, m: dict) -> None:
    os.makedirs(data_dir, exist_ok=True)
    m["updatedAt"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with open(os.path.join(data_dir, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=1)


def month_key(now: dt.datetime) -> str:
    return now.strftime("%Y-%m")


def read_targets(path: str) -> List[str]:
    out: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(normalize_code(s))
    return out


def header_remaining(p: Probe, name: str = "x-ratelimit-remaining") -> Optional[int]:
    """直前の応答の利用枠ヘッダ（既定は日次の残数）。無ければ None。"""
    for k, v in p.last_headers.items():
        if k.lower() == name:
            try:
                return int(v)
            except ValueError:
                return None
    return None


def _age_days(rec: dict, now: dt.datetime) -> Optional[float]:
    try:
        at = dt.datetime.fromisoformat(rec["at"])
    except (KeyError, TypeError, ValueError):
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=dt.timezone.utc)
    return (now - at).total_seconds() / 86400.0


def fetch_order(targets: List[str], code_to_edinet: Dict[str, str], comp: dict,
                now: dt.datetime) -> List[str]:
    """
    この実行で問い合わせる順。
      1. まだ取れていない会社（未取得・前回失敗）を取得順リストの順に
      2. 取得済み・データ無しのうち REFRESH_AFTER_DAYS を過ぎた会社を古い順に
    対応表に無い会社は含めない。
    """
    fresh: List[str] = []
    stale: List[tuple] = []
    for jq in targets:
        e = code_to_edinet.get(jq)
        if not e:
            continue
        rec = comp.get(e) or {}
        if rec.get("status") not in ("ok", "no_data"):
            fresh.append(jq)
            continue
        age = _age_days(rec, now)
        if age is None or age >= REFRESH_AFTER_DAYS:
            stale.append((-(age if age is not None else 1e9), jq))
    return fresh + [jq for _, jq in sorted(stale)]


def clamp_budget(p: Probe, reserve: int) -> None:
    """
    応答ヘッダの残数に合わせて、この実行の上限を締める。
    「今日はもう N 本しか無い」を、送ってから 429 で知るのではなく先に知る。
    """
    rem = header_remaining(p)
    if rem is None:
        return
    allowed = p.used + max(0, rem - reserve)
    if allowed < p.budget:
        p.budget = allowed


# ---------------------------------------------------------------------- #
# 会社一覧（対応表）
# ---------------------------------------------------------------------- #

def refresh_mapping(p: Probe, data_dir: str, m: dict, now: dt.datetime,
                    reserve: int) -> Optional[pd.DataFrame]:
    rows: List[dict] = []
    seen: set = set()
    page = 1
    while True:
        st, body = p.get("/companies", {"per_page": PAGE_SIZE, "page": page})
        clamp_budget(p, reserve)
        if st != 200:
            p.say(f"  [mapping] ページ {page} が取れない（{st}）。対応表は更新しない")
            return None
        rs = as_rows(body)
        codes = {r.get("edinet_code") for r in rs}
        if not rs or codes <= seen:
            # page 引数が効いていない（同じページが返る）か、終端
            break
        seen |= codes
        rows.extend(rs)
        total_pages = 1
        if isinstance(body, dict):
            total_pages = int((body.get("meta") or {}).get("pagination", {})
                              .get("total_pages", 1) or 1)
        if page >= total_pages:
            break
        page += 1
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["code"] = df.get("sec_code", pd.Series([None] * len(df))).map(
        lambda v: normalize_code(v) if v not in (None, "") else None)
    df = df.drop_duplicates("edinet_code").reset_index(drop=True)
    df.to_parquet(os.path.join(data_dir, COMPANIES), index=False, compression="zstd")
    m["mapping_at"] = now.isoformat()
    p.say(f"  [mapping] {len(df):,}社 / 証券コード付き {df['code'].notna().sum():,}社 "
          f"/ {page}ページ")
    return df


def load_mapping(data_dir: str) -> Optional[pd.DataFrame]:
    p = os.path.join(data_dir, COMPANIES)
    return pd.read_parquet(p) if os.path.exists(p) else None


def mapping_is_fresh(m: dict, now: dt.datetime) -> bool:
    if not m.get("mapping_at"):
        return False
    try:
        at = dt.datetime.fromisoformat(m["mapping_at"])
    except ValueError:
        return False
    if at.tzinfo is None:
        at = at.replace(tzinfo=dt.timezone.utc)
    return (now - at).days < MAPPING_MAX_AGE_DAYS


# ---------------------------------------------------------------------- #
# 年次財務
# ---------------------------------------------------------------------- #

def write_fin(data_dir: str, df: pd.DataFrame) -> str:
    """
    列は会社ごとに違う（値の無い項目は省かれる）ので、和集合で持つ。
    bool と欠測が混ざる列は pyarrow が推論できないので nullable 型に寄せる。
    それでも書けない列は文字列にして落とさない。
    """
    path = os.path.join(data_dir, FIN)
    out = df.convert_dtypes()
    try:
        out.to_parquet(path, index=False, compression="zstd")
    except (TypeError, ValueError, OverflowError):
        bad = [c for c in out.columns if str(out[c].dtype) in ("object", "string")]
        for c in bad:
            out[c] = out[c].astype("string")
        out.to_parquet(path, index=False, compression="zstd")
    return path


def fetch_financials(p: Probe, data_dir: str, m: dict, mapping: pd.DataFrame,
                     targets: List[str], n: int, now: dt.datetime,
                     reserve: int) -> Tuple[int, int, int]:
    """
    (取りに行った社数, 新しく入った行数, 対応表に無かった社数) を返す。
    """
    code_to_edinet: Dict[str, str] = {
        c: e for c, e in zip(mapping["code"], mapping["edinet_code"])
        if isinstance(c, str) and c}
    existing_path = os.path.join(data_dir, FIN)
    existing = pd.read_parquet(existing_path) if os.path.exists(existing_path) else None
    frames: List[pd.DataFrame] = []
    done = new_rows = 0
    unmapped = sum(1 for jq in targets if not code_to_edinet.get(jq))
    comp = m["companies"]
    for jq in fetch_order(targets, code_to_edinet, comp, now):
        if done >= n:
            break
        e = code_to_edinet[jq]
        rec = comp.get(e) or {}
        rem = header_remaining(p)
        if rem is not None and rem <= reserve:
            p.say(f"  [stop] 日次の残数 {rem} が予備 {reserve} 以下。ここで止める")
            break
        st, body = p.get(f"/companies/{e}/financials", {"years": YEARS})
        clamp_budget(p, reserve)
        if st is None:
            break                                    # 予算切れ・接続不能
        if st == 429:
            p.say("  [stop] 429（利用枠）。記録せずに止める")
            break
        done += 1
        if st == 404:
            comp[e] = {"status": "no_data", "at": now.isoformat(), "jq": jq}
            continue
        if st != 200:
            comp[e] = {"status": "error", "http": st, "at": now.isoformat(),
                       "jq": jq, "tries": int(rec.get("tries", 0)) + 1}
            continue
        rows = as_rows(body)
        df = pd.DataFrame(rows)
        if df.empty:
            comp[e] = {"status": "no_data", "at": now.isoformat(), "jq": jq}
            continue
        df["edinet_code"] = e
        df["jq_code"] = jq
        frames.append(df)
        new_rows += len(df)
        fy = pd.to_numeric(df.get("fiscal_year"), errors="coerce")
        comp[e] = {"status": "ok", "rows": int(len(df)), "at": now.isoformat(),
                   "jq": jq, "last_fy": int(fy.max()) if fy.notna().any() else None}

    if frames:
        allf = pd.concat(([existing] if existing is not None else []) + frames,
                         ignore_index=True, sort=False)
        for c in KEY_COLS:
            if c not in allf.columns:
                allf[c] = None
        allf = allf.drop_duplicates(KEY_COLS, keep="last").reset_index(drop=True)
        write_fin(data_dir, allf)
        p.say(f"  [fin] 保存 {len(allf):,}行 / {allf['edinet_code'].nunique():,}社")
    return done, new_rows, unmapped


# ---------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EDINET DB から年次財務を差分で取り込む")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--targets", default=TARGETS)
    ap.add_argument("--mapping", choices=["auto", "yes", "no"], default="auto",
                    help=f"対応表を取り直すか。auto は無いか {MAPPING_MAX_AGE_DAYS} 日より古いとき")
    ap.add_argument("--fetch", type=int, default=85, help="この実行で財務を取る社数の上限")
    ap.add_argument("--reserve", type=int, default=DAILY_RESERVE,
                    help="日次の残数がこれ以下になったら止める")
    ap.add_argument("--monthly-budget", type=int, default=MONTHLY_BUDGET)
    args = ap.parse_args(argv)

    key = os.environ.get("EDINET_API_KEY", "").strip() or None
    if not key:
        print("[stop] EDINET_API_KEY が無い")
        return 1
    now = dt.datetime.now(dt.timezone.utc)
    os.makedirs(args.data_dir, exist_ok=True)
    m = load_manifest(args.data_dir)
    spent = int(m["requests"].get(month_key(now), 0))
    monthly_left = max(0, args.monthly_budget - spent)
    if monthly_left == 0:
        print(f"[stop] 今月の予算 {args.monthly_budget} を使い切っている（{spent}）")
        return 0
    # この実行の硬い上限。応答ヘッダの残数でさらに締める
    budget = min(DAILY_LIMIT - args.reserve, monthly_left)
    p = Probe(key, budget)
    p.say(f"今月の使用 {spent} / 予算 {args.monthly_budget} -> この実行の上限 {budget}")

    mapping = load_mapping(args.data_dir)
    need = (args.mapping == "yes"
            or (args.mapping == "auto" and (mapping is None or not mapping_is_fresh(m, now))))
    if need:
        p.say("\n=== 対応表を取り直す ===")
        got = refresh_mapping(p, args.data_dir, m, now, args.reserve)
        if got is not None:
            mapping = got
    if mapping is None or "code" not in mapping.columns:
        m["requests"][month_key(now)] = spent + p.used
        save_manifest(args.data_dir, m)
        p.say("[stop] 対応表が無いので財務は取れない")
        return 1

    targets = read_targets(args.targets)
    p.say(f"\n=== 年次財務（取得順リスト {len(targets):,}社 / 上限 {args.fetch}社）===")
    done, new_rows, unmapped = fetch_financials(
        p, args.data_dir, m, mapping, targets, args.fetch, now, args.reserve)

    m["requests"][month_key(now)] = spent + p.used
    save_manifest(args.data_dir, m)
    comp = m["companies"]
    n_ok = sum(1 for v in comp.values() if v.get("status") == "ok")
    n_nd = sum(1 for v in comp.values() if v.get("status") == "no_data")
    n_er = sum(1 for v in comp.values() if v.get("status") == "error")
    p.say(f"\n[done] 今回 {done}社に問い合わせ / 新規 {new_rows:,}行 / 対応表に無い {unmapped}社")
    p.say(f"[state] 取得済み {n_ok:,}社 / データ無し {n_nd:,}社 / 失敗（再試行）{n_er:,}社 "
          f"/ 残り {sum(1 for t in targets if t not in {v.get('jq') for v in comp.values()}):,}社")
    p.say(f"[quota] この実行 {p.used} / 今月 {spent + p.used} / 予算 {args.monthly_budget}"
          f"（応答ヘッダ: 日次の残数 {header_remaining(p)} / "
          f"月次の残数 {header_remaining(p, 'x-ratelimit-monthly-remaining')}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
