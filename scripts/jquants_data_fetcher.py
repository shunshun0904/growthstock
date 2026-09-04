#!/usr/bin/env python3
"""
jquants_data_fetcher.py
=======================
GrowthStockAnalyzer - Focus のデータ取得・指標算出パイプライン (仕様書 §3)。

J-Quants API から日次株価・財務諸表・信用残を取得し、8軸スコアリングに必要な
指標へ前処理したうえで ``public/data/stocks.json`` を出力する。
標準ライブラリのみで動作する (CI での依存インストール不要)。

認証
----
環境変数 ``JQUANTS_API`` (GitHub Actions の Repository secret) を使用する。
中身は以下のいずれの形式でも自動判別する:

  * リフレッシュトークン (既定の想定 / 有効期限 1週間)
  * IDトークン (有効期限 24時間)
  * ``{"mailaddress": "...", "password": "..."}`` の JSON
  * ``mail@example.com:password`` のコロン区切り

``JQUANTS_MAIL`` / ``JQUANTS_PASSWORD`` が設定されている場合はそちらを優先し、
リフレッシュトークンを毎回取得し直す (無人運用向け)。

出力方針
--------
取得できなかった指標は「0」ではなく ``null`` として出力し、``sources`` に
``unavailable`` を記録する。実測値と欠測を混同させないため、値の捏造・推定補完は行わない。

使い方
------
  python3 scripts/jquants_data_fetcher.py                     # watchlist 全銘柄
  python3 scripts/jquants_data_fetcher.py --codes 7203 6758   # 銘柄指定
  python3 scripts/jquants_data_fetcher.py --check-auth        # 認証疎通のみ
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence

API_BASE = "https://api.jquants.com/v1"
USER_AGENT = "GrowthStockAnalyzer-Focus/1.0 (+https://github.com/shunshun0904/growthstock)"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WATCHLIST = os.path.join(ROOT, "scripts", "watchlist.json")
DEFAULT_OUTPUT = os.path.join(ROOT, "public", "data", "stocks.json")

PRICE_LOOKBACK_DAYS = 640   # 52週高値を「6ヶ月前時点」でも算出するため 365 + 183 + 余裕
MARGIN_LOOKBACK_DAYS = 400

# 取得できなかった理由の分類
SRC_API = "jquants"
SRC_NA = "unavailable"


class JQuantsError(RuntimeError):
    """J-Quants API 呼び出しに関する回復不能なエラー。"""


class AuthError(JQuantsError):
    """認証に失敗した (トークン期限切れ・認証情報不正など)。"""


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def _request(
    method: str,
    url: str,
    *,
    body: Optional[dict] = None,
    headers: Optional[dict] = None,
    retries: int = 4,
) -> dict:
    """JSON を返す HTTP リクエスト。429/5xx は指数バックオフで再試行する。"""
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if payload is not None:
        hdrs["Content-Type"] = "application/json"
    if headers:
        hdrs.update(headers)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=payload, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # noqa: BLE001 - 詳細が読めなくても本体のエラーを優先
                pass
            if exc.code in (401, 403):
                raise AuthError(f"HTTP {exc.code} {url} : {detail}") from exc
            if exc.code == 429 or 500 <= exc.code < 600:
                last_err = JQuantsError(f"HTTP {exc.code} {url} : {detail}")
                time.sleep(2 ** attempt)
                continue
            raise JQuantsError(f"HTTP {exc.code} {url} : {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise JQuantsError(f"リクエストが {retries} 回失敗しました: {url} ({last_err})")


# --------------------------------------------------------------------------- #
# 認証
# --------------------------------------------------------------------------- #

def _looks_like_jwt(value: str) -> bool:
    return bool(re.fullmatch(r"[\w-]+\.[\w-]+\.[\w-]+", value.strip()))


def _jwt_expiry(token: str) -> Optional[dt.datetime]:
    """JWT の exp クレームを読む (署名検証はしない / 期限切れ判定の情報提供のみ)。"""
    import base64

    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            return dt.datetime.fromtimestamp(exp, tz=dt.timezone.utc)
    except Exception:  # noqa: BLE001 - 解析できなければ情報なしとして扱う
        return None
    return None


def id_token_from_refresh(refresh_token: str) -> str:
    url = f"{API_BASE}/token/auth_refresh?" + urllib.parse.urlencode(
        {"refreshtoken": refresh_token}
    )
    data = _request("POST", url)
    token = data.get("idToken")
    if not token:
        raise AuthError(f"auth_refresh が idToken を返しませんでした: {data}")
    return token


def refresh_token_from_login(mail: str, password: str) -> str:
    data = _request(
        "POST",
        f"{API_BASE}/token/auth_user",
        body={"mailaddress": mail, "password": password},
    )
    token = data.get("refreshToken")
    if not token:
        raise AuthError(f"auth_user が refreshToken を返しませんでした: {data}")
    return token


def resolve_id_token() -> str:
    """環境変数から ID トークンを解決する。secret の形式は自動判別。"""
    mail = os.environ.get("JQUANTS_MAIL", "").strip()
    password = os.environ.get("JQUANTS_PASSWORD", "").strip()
    if mail and password:
        print("[auth] JQUANTS_MAIL / JQUANTS_PASSWORD からリフレッシュトークンを取得します")
        return id_token_from_refresh(refresh_token_from_login(mail, password))

    secret = os.environ.get("JQUANTS_API", "").strip()
    if not secret:
        raise AuthError(
            "環境変数 JQUANTS_API が未設定です。"
            "GitHub の Settings > Secrets and variables > Actions に登録してください。"
        )

    # 1) JSON 形式 {"mailaddress": ..., "password": ...} / {"refreshToken": ...}
    if secret.startswith("{"):
        try:
            obj = json.loads(secret)
        except json.JSONDecodeError as exc:
            raise AuthError("JQUANTS_API が JSON らしき文字列ですが解析できません") from exc
        if obj.get("mailaddress") and obj.get("password"):
            print("[auth] JQUANTS_API (JSON: mailaddress/password) でログインします")
            return id_token_from_refresh(
                refresh_token_from_login(obj["mailaddress"], obj["password"])
            )
        for key in ("idToken", "id_token"):
            if obj.get(key):
                print("[auth] JQUANTS_API (JSON: idToken) をそのまま使用します")
                return obj[key]
        for key in ("refreshToken", "refresh_token"):
            if obj.get(key):
                print("[auth] JQUANTS_API (JSON: refreshToken) から idToken を取得します")
                return id_token_from_refresh(obj[key])
        raise AuthError("JQUANTS_API の JSON に利用可能なキーがありません")

    # 2) mail:password 形式
    if "@" in secret and ":" in secret and not _looks_like_jwt(secret):
        mail_part, _, pw_part = secret.partition(":")
        print("[auth] JQUANTS_API (mail:password) でログインします")
        return id_token_from_refresh(refresh_token_from_login(mail_part.strip(), pw_part))

    # 3) 生のトークン。ID トークン (24h) かリフレッシュトークン (1週間) かを判別する。
    exp = _jwt_expiry(secret)
    if exp is not None:
        remaining = exp - dt.datetime.now(tz=dt.timezone.utc)
        if remaining.total_seconds() <= 0:
            raise AuthError(
                f"JQUANTS_API のトークンは {exp:%Y-%m-%d %H:%M UTC} に失効しています。"
                "J-Quants でリフレッシュトークンを再発行し、secret を更新してください。"
            )
        print(f"[auth] トークン残り有効期間: {remaining}")

    try:
        print("[auth] JQUANTS_API をリフレッシュトークンとして idToken を取得します")
        return id_token_from_refresh(secret)
    except AuthError as exc:
        # リフレッシュトークンとして通らなければ ID トークンの可能性を試す
        print(f"[auth] auth_refresh に失敗 ({exc}) — idToken として直接検証します")
        probe = _request(
            "GET",
            f"{API_BASE}/listed/info?code=72030",
            headers={"Authorization": f"Bearer {secret}"},
        )
        if "info" in probe:
            print("[auth] JQUANTS_API は idToken として有効でした")
            return secret
        raise AuthError(
            "JQUANTS_API を認証情報として解決できませんでした。"
            "リフレッシュトークン (有効期限1週間) が失効している可能性があります。"
        ) from exc


# --------------------------------------------------------------------------- #
# API クライアント
# --------------------------------------------------------------------------- #

class JQuantsClient:
    def __init__(self, id_token: str, *, pause: float = 0.25) -> None:
        self._headers = {"Authorization": f"Bearer {id_token}"}
        self._pause = pause
        #: プラン制約などで利用できなかったエンドポイント -> 理由
        self.unavailable_endpoints: Dict[str, str] = {}

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{API_BASE}{path}"
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        time.sleep(self._pause)
        return _request("GET", url, headers=self._headers)

    def get_paginated(self, path: str, params: dict, key: str) -> List[dict]:
        """pagination_key を辿って全件取得する。"""
        out: List[dict] = []
        cursor: Optional[str] = None
        for _ in range(50):  # 無限ループ防止
            page_params = dict(params)
            if cursor:
                page_params["pagination_key"] = cursor
            data = self.get(path, page_params)
            out.extend(data.get(key, []) or [])
            cursor = data.get("pagination_key")
            if not cursor:
                break
        return out

    # --- 個別エンドポイント ------------------------------------------------ #

    def listed_info(self, code: str) -> Optional[dict]:
        try:
            rows = self.get("/listed/info", {"code": code}).get("info") or []
            return rows[-1] if rows else None
        except JQuantsError as exc:
            self.unavailable_endpoints["/listed/info"] = str(exc)
            return None

    def daily_quotes(self, code: str, date_from: str, date_to: str) -> List[dict]:
        try:
            rows = self.get_paginated(
                "/prices/daily_quotes",
                {"code": code, "from": date_from, "to": date_to},
                "daily_quotes",
            )
        except JQuantsError as exc:
            self.unavailable_endpoints["/prices/daily_quotes"] = str(exc)
            return []
        return sorted(rows, key=lambda r: r.get("Date", ""))

    def statements(self, code: str) -> List[dict]:
        try:
            rows = self.get_paginated("/fins/statements", {"code": code}, "statements")
        except JQuantsError as exc:
            self.unavailable_endpoints["/fins/statements"] = str(exc)
            return []
        return sorted(rows, key=lambda r: (r.get("DisclosedDate", ""), r.get("DisclosedTime", "")))

    def weekly_margin_interest(self, code: str, date_from: str, date_to: str) -> List[dict]:
        try:
            rows = self.get_paginated(
                "/markets/weekly_margin_interest",
                {"code": code, "from": date_from, "to": date_to},
                "weekly_margin_interest",
            )
        except JQuantsError as exc:
            # Light / Free プランでは利用不可。需給軸は「データなし」として扱う。
            self.unavailable_endpoints["/markets/weekly_margin_interest"] = str(exc)
            return []
        return sorted(rows, key=lambda r: r.get("Date", ""))


# --------------------------------------------------------------------------- #
# ヘルパー
# --------------------------------------------------------------------------- #

def normalize_code(code: str) -> str:
    """仕様書 §3.1: 4桁コードは末尾に '0' を付加して5桁化する (6928 -> 69280)。"""
    c = str(code).strip().upper()
    return c + "0" if len(c) == 4 else c


def display_code(code: str) -> str:
    """5桁コードの末尾 '0' を落として表示用の4桁コードに戻す。"""
    c = str(code).strip().upper()
    return c[:4] if len(c) == 5 and c.endswith("0") else c


def fnum(value: Any) -> Optional[float]:
    """API の文字列/空文字/None を float | None に正規化する。"""
    if value is None or value == "" or value == "-":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None  # NaN / inf を除外


def pick(row: dict, *keys: str) -> Optional[float]:
    """先に見つかった有効な数値を返す (調整後値 → 素の値 のフォールバック用)。"""
    for k in keys:
        v = fnum(row.get(k))
        if v is not None:
            return v
    return None


def pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """
    前年同期比 (%). 前年値が 0 以下の場合は成長率が定義できないため None を返す。
    (赤字からの黒字転換を「+1000%」等と表現すると誤解を招くため、値を作らない)
    """
    if current is None or previous is None or previous <= 0:
        return None
    return (current - previous) / previous * 100.0


def rows_upto(rows: Sequence[dict], key: str, as_of: str) -> List[dict]:
    return [r for r in rows if (r.get(key) or "") <= as_of]


# --------------------------------------------------------------------------- #
# 指標算出 : 株価系 (仕様書 §3.2)
# --------------------------------------------------------------------------- #

def price_metrics(quotes: Sequence[dict], as_of: Optional[str] = None) -> Dict[str, Any]:
    """
    指定日時点 (as_of, 省略時は最新) の株価系指標を算出する。
    未来のバーは一切参照しない (タイムマシーン・モードの point-in-time 保証)。
    """
    series = rows_upto(quotes, "Date", as_of) if as_of else list(quotes)
    out: Dict[str, Any] = {
        "date": None, "price": None, "high52w": None, "highRatio": None,
        "tradingValue": None, "turnoverValue": None, "volumeTrend": None,
        "volume": None, "ma20Volume": None,
    }
    if not series:
        return out

    latest = series[-1]
    out["date"] = latest.get("Date")

    price = pick(latest, "AdjustmentClose", "Close")
    out["price"] = price

    # 52週高値: 直近営業日から遡って 365日ぶんのバーの最高値
    end = dt.date.fromisoformat(latest["Date"])
    start = end - dt.timedelta(days=365)
    window = [r for r in series if dt.date.fromisoformat(r["Date"]) >= start]
    highs = [h for h in (pick(r, "AdjustmentHigh", "High") for r in window) if h is not None]
    if highs:
        out["high52w"] = max(highs)
        if price is not None and out["high52w"] > 0:
            out["highRatio"] = price / out["high52w"] * 100.0

    # 売買代金 (億円): 仕様書 §3.2-4 の定義 P_current × Volume / 1e8
    raw_close = pick(latest, "Close", "AdjustmentClose")
    raw_volume = pick(latest, "Volume", "AdjustmentVolume")
    out["volume"] = raw_volume
    if raw_close is not None and raw_volume is not None:
        out["tradingValue"] = raw_close * raw_volume / 1e8
    # API が返す実際の売買代金 (参考値)
    turnover = fnum(latest.get("TurnoverValue"))
    if turnover is not None:
        out["turnoverValue"] = turnover / 1e8

    # 出来高モメンタム (%): 直近出来高 / 直前20日平均出来高
    prior = [v for v in (pick(r, "AdjustmentVolume", "Volume") for r in series[-21:-1]) if v is not None]
    if len(prior) >= 5 and raw_volume is not None:
        ma20 = sum(prior) / len(prior)
        out["ma20Volume"] = ma20
        if ma20 > 0:
            latest_vol = pick(latest, "AdjustmentVolume", "Volume")
            if latest_vol is not None:
                out["volumeTrend"] = latest_vol / ma20 * 100.0

    return out


# --------------------------------------------------------------------------- #
# 指標算出 : 財務系
# --------------------------------------------------------------------------- #

QUARTER_MAP = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4}

# 実績決算のみを対象とする (業績予想の修正等は除外)
FINANCIAL_DOC_RE = re.compile(r"FinancialStatements")


def _is_financial_statement(row: dict) -> bool:
    doc = row.get("TypeOfDocument") or ""
    return bool(FINANCIAL_DOC_RE.search(doc)) and row.get("TypeOfCurrentPeriod") in QUARTER_MAP


def quarterize(statements: Sequence[dict]) -> List[dict]:
    """
    累計ベースの決算短信データを「単一四半期」の値へ差分展開する。

    J-Quants の fins/statements は会計年度内の累計値を返すため、
    2Q の値から 1Q の値を引く等の処理を行わないと前年同期比が正しく出せない。
    """
    rows = [r for r in statements if _is_financial_statement(r)]
    by_fy: Dict[str, List[dict]] = {}
    for r in rows:
        by_fy.setdefault(r.get("CurrentFiscalYearStartDate") or "", []).append(r)

    result: List[dict] = []
    for fy_start, group in by_fy.items():
        group = sorted(group, key=lambda r: QUARTER_MAP[r["TypeOfCurrentPeriod"]])
        # 同一四半期の重複開示 (訂正) は最後の開示を採用
        dedup: Dict[int, dict] = {}
        for r in group:
            dedup[QUARTER_MAP[r["TypeOfCurrentPeriod"]]] = r
        ordered = [dedup[q] for q in sorted(dedup)]

        for idx, r in enumerate(ordered):
            q = QUARTER_MAP[r["TypeOfCurrentPeriod"]]
            prev = ordered[idx - 1] if idx > 0 else None
            prev_q = QUARTER_MAP[prev["TypeOfCurrentPeriod"]] if prev else 0

            def diff(field: str) -> Optional[float]:
                cur = fnum(r.get(field))
                if cur is None:
                    return None
                if prev is None or prev_q != q - 1:
                    # 直前四半期が欠けている場合、1Q 以外は差分を作れない
                    return cur if q == 1 else None
                pv = fnum(prev.get(field))
                return None if pv is None else cur - pv

            result.append({
                "fiscalYearStart": fy_start,
                "quarter": q,
                "period": r.get("TypeOfCurrentPeriod"),
                "disclosedDate": r.get("DisclosedDate"),
                "periodEnd": r.get("CurrentPeriodEndDate"),
                # 単一四半期値
                "qNetSales": diff("NetSales"),
                "qOperatingProfit": diff("OperatingProfit"),
                "qProfit": diff("Profit"),
                "qEps": diff("EarningsPerShare"),
                # 累計値 (進捗率算出に使用)
                "cumNetSales": fnum(r.get("NetSales")),
                "cumOperatingProfit": fnum(r.get("OperatingProfit")),
                "cumProfit": fnum(r.get("Profit")),
                "cumEps": fnum(r.get("EarningsPerShare")),
                # 会社予想 (通期)
                "forecastNetSales": fnum(r.get("ForecastNetSales")),
                "forecastOperatingProfit": fnum(r.get("ForecastOperatingProfit")),
                "forecastProfit": fnum(r.get("ForecastProfit")),
                "forecastEps": fnum(r.get("ForecastEarningsPerShare")),
                # 財政状態
                "equity": fnum(r.get("Equity")),
                "totalAssets": fnum(r.get("TotalAssets")),
                "sharesIssued": fnum(
                    r.get("NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock")
                ),
                "treasuryShares": fnum(r.get("NumberOfTreasuryStockAtTheEndOfFiscalYear")),
            })

    return sorted(result, key=lambda r: (r.get("disclosedDate") or "", r["fiscalYearStart"], r["quarter"]))


def fundamental_metrics(quarters: Sequence[dict], as_of: Optional[str] = None) -> Dict[str, Any]:
    """指定日時点で「開示済み」の決算のみを使って財務指標を算出する。"""
    hist = rows_upto(quarters, "disclosedDate", as_of) if as_of else list(quarters)
    out: Dict[str, Any] = {
        "epsGrowth": None, "salesGrowth": None, "roe": None, "opMargin": None,
        "progressRate": None, "quarter": None, "sharesOutstanding": None,
        "fiscalPeriod": None, "disclosedDate": None, "opMarginBasis": None,
    }
    if not hist:
        return out

    latest = hist[-1]
    out["quarter"] = latest["quarter"]
    out["fiscalPeriod"] = latest["period"]
    out["disclosedDate"] = latest["disclosedDate"]

    if latest.get("sharesIssued") is not None:
        shares = latest["sharesIssued"] - (latest.get("treasuryShares") or 0)
        out["sharesOutstanding"] = shares if shares > 0 else None

    # --- 前年同期比 (同一四半期どうしを比較) ---
    prior_year = next(
        (r for r in reversed(hist[:-1])
         if r["quarter"] == latest["quarter"] and r["fiscalYearStart"] < latest["fiscalYearStart"]),
        None,
    )
    if prior_year:
        out["epsGrowth"] = pct_change(latest.get("qEps"), prior_year.get("qEps"))
        out["salesGrowth"] = pct_change(latest.get("qNetSales"), prior_year.get("qNetSales"))

    # --- ROE: 直近4四半期の純利益合計 / 自己資本 ---
    ttm = [r.get("qProfit") for r in hist[-4:]]
    equity = latest.get("equity")
    if len(ttm) == 4 and all(v is not None for v in ttm) and equity and equity > 0:
        out["roe"] = sum(ttm) / equity * 100.0

    # --- 営業利益率: 直近4四半期 (TTM) を優先、不可なら当期累計 ---
    ttm_op = [r.get("qOperatingProfit") for r in hist[-4:]]
    ttm_sales = [r.get("qNetSales") for r in hist[-4:]]
    if (len(ttm_op) == 4 and all(v is not None for v in ttm_op)
            and all(v is not None for v in ttm_sales) and sum(ttm_sales) > 0):
        out["opMargin"] = sum(ttm_op) / sum(ttm_sales) * 100.0
        out["opMarginBasis"] = "TTM (直近4四半期)"
    elif latest.get("cumOperatingProfit") is not None and (latest.get("cumNetSales") or 0) > 0:
        out["opMargin"] = latest["cumOperatingProfit"] / latest["cumNetSales"] * 100.0
        out["opMarginBasis"] = f"当期累計 ({latest['period']})"

    # --- 決算進捗率: 当期累計営業利益 / 通期会社予想営業利益 ---
    forecast = latest.get("forecastOperatingProfit")
    cum_op = latest.get("cumOperatingProfit")
    if forecast and forecast > 0 and cum_op is not None:
        out["progressRate"] = cum_op / forecast * 100.0

    return out


# --------------------------------------------------------------------------- #
# 指標算出 : 需給 (信用倍率)
# --------------------------------------------------------------------------- #

def credit_metrics(margin_rows: Sequence[dict], as_of: Optional[str] = None) -> Dict[str, Any]:
    hist = rows_upto(margin_rows, "Date", as_of) if as_of else list(margin_rows)
    out: Dict[str, Any] = {
        "creditRatio": None, "marginLong": None, "marginShort": None, "marginDate": None,
    }
    if not hist:
        return out
    latest = hist[-1]
    long_v = fnum(latest.get("LongMarginTradeVolume"))
    short_v = fnum(latest.get("ShortMarginTradeVolume"))
    out["marginDate"] = latest.get("Date")
    out["marginLong"] = long_v
    out["marginShort"] = short_v
    if long_v is not None and short_v is not None and short_v > 0:
        out["creditRatio"] = long_v / short_v
    return out


# --------------------------------------------------------------------------- #
# マイルストーン (タイムマシーン・モードのストーリータイムライン)
# --------------------------------------------------------------------------- #

def build_milestones(
    quotes: Sequence[dict], quarters: Sequence[dict], months: int = 12
) -> List[dict]:
    """
    実データから検出できるイベントのみを時系列で列挙する。
    (定性的なストーリーを創作しない — 検出条件は各イベントの detail に明記する)
    """
    events: List[dict] = []
    if not quotes:
        return events

    last_date = dt.date.fromisoformat(quotes[-1]["Date"])
    since = last_date - dt.timedelta(days=30 * months)
    since_s = since.isoformat()

    # 1) 決算発表
    for q in quarters:
        d = q.get("disclosedDate")
        if not d or d < since_s:
            continue
        parts = []
        if q.get("qNetSales") is not None:
            parts.append(f"売上 {q['qNetSales'] / 1e8:,.0f}億円")
        if q.get("qOperatingProfit") is not None:
            parts.append(f"営業利益 {q['qOperatingProfit'] / 1e8:,.0f}億円")
        events.append({
            "date": d,
            "type": "earnings",
            "title": f"{q['period']} 決算発表",
            "detail": "（単一四半期換算）" + " / ".join(parts) if parts else "決算短信を開示",
        })

    # 2) 52週高値更新 (BREAKOUT)
    # 3) 出来高急増 (20日平均の2倍以上 かつ 陽線) = 機関投資家の参入痕跡候補
    prev_breakout: Optional[str] = None
    for i, row in enumerate(quotes):
        d = row.get("Date")
        if not d or d < since_s or i < 21:
            continue
        close = pick(row, "AdjustmentClose", "Close")
        open_ = pick(row, "AdjustmentOpen", "Open")
        vol = pick(row, "AdjustmentVolume", "Volume")
        if close is None or vol is None:
            continue

        start = dt.date.fromisoformat(d) - dt.timedelta(days=365)
        window = [r for r in quotes[:i] if dt.date.fromisoformat(r["Date"]) >= start]
        highs = [h for h in (pick(r, "AdjustmentHigh", "High") for r in window) if h is not None]
        if highs and close > max(highs):
            # 直近30日以内に同種イベントがある場合はまとめる (毎日出さない)
            if prev_breakout is None or (
                dt.date.fromisoformat(d) - dt.date.fromisoformat(prev_breakout)
            ).days > 30:
                events.append({
                    "date": d,
                    "type": "breakout",
                    "title": "52週高値を更新",
                    "detail": f"終値 {close:,.0f}円 が過去52週の最高値 {max(highs):,.0f}円 を上抜け",
                })
                prev_breakout = d

        prior = [v for v in (pick(r, "AdjustmentVolume", "Volume") for r in quotes[i - 20:i]) if v is not None]
        if len(prior) >= 15:
            ma20 = sum(prior) / len(prior)
            if ma20 > 0 and vol >= ma20 * 2.0 and open_ is not None and close > open_:
                events.append({
                    "date": d,
                    "type": "volume_spike",
                    "title": "出来高急増（大口買い痕跡）",
                    "detail": f"出来高 {vol:,.0f}株 = 20日平均の {vol / ma20:.1f}倍・陽線",
                })

    # 出来高急増は多くなりがちなので、直近のものから最大8件に絞る
    spikes = [e for e in events if e["type"] == "volume_spike"]
    others = [e for e in events if e["type"] != "volume_spike"]
    spikes = sorted(spikes, key=lambda e: e["date"])[-8:]
    return sorted(others + spikes, key=lambda e: e["date"])


# --------------------------------------------------------------------------- #
# 銘柄1件分の組み立て
# --------------------------------------------------------------------------- #

SNAPSHOT_OFFSETS = [("now", 0), ("m3", 91), ("m6", 183)]
SNAPSHOT_LABELS = {"now": "現在", "m3": "3ヶ月前", "m6": "6ヶ月前"}


def build_stock(client: JQuantsClient, raw_code: str, note: str = "") -> Dict[str, Any]:
    code = normalize_code(raw_code)
    today = dt.date.today()
    date_from = (today - dt.timedelta(days=PRICE_LOOKBACK_DAYS)).isoformat()
    date_to = today.isoformat()

    print(f"[fetch] {code} : listed/info")
    info = client.listed_info(code) or {}
    print(f"[fetch] {code} : prices/daily_quotes ({date_from} 〜 {date_to})")
    quotes = client.daily_quotes(code, date_from, date_to)
    print(f"[fetch] {code} : fins/statements")
    statements = client.statements(code)
    print(f"[fetch] {code} : markets/weekly_margin_interest")
    margins = client.weekly_margin_interest(
        code, (today - dt.timedelta(days=MARGIN_LOOKBACK_DAYS)).isoformat(), date_to
    )

    quarters = quarterize(statements)

    if not quotes:
        return {
            "code": display_code(code),
            "jqCode": code,
            "name": info.get("CompanyName") or display_code(code),
            "error": "日次株価データを取得できませんでした",
            "metrics": {}, "snapshots": {}, "milestones": [], "sources": {},
        }

    latest_date = quotes[-1]["Date"]

    def snapshot(as_of: Optional[str]) -> Dict[str, Any]:
        pm = price_metrics(quotes, as_of)
        fm = fundamental_metrics(quarters, as_of)
        cm = credit_metrics(margins, as_of)
        merged: Dict[str, Any] = {**pm, **fm, **cm}
        if pm.get("price") is not None and fm.get("sharesOutstanding"):
            merged["marketCap"] = pm["price"] * fm["sharesOutstanding"] / 1e8
        else:
            merged["marketCap"] = None
        return merged

    snapshots: Dict[str, Any] = {}
    base = dt.date.fromisoformat(latest_date)
    for key, offset in SNAPSHOT_OFFSETS:
        as_of = None if offset == 0 else (base - dt.timedelta(days=offset)).isoformat()
        snap = snapshot(as_of)
        snap["label"] = SNAPSHOT_LABELS[key]
        snap["asOf"] = as_of or latest_date
        snapshots[key] = snap

    metrics = snapshots["now"]

    # 各指標のデータ由来を記録する (UI で「実データ / 未取得」を区別するため)
    source_fields = [
        "price", "high52w", "highRatio", "tradingValue", "volumeTrend",
        "epsGrowth", "salesGrowth", "roe", "opMargin", "progressRate",
        "creditRatio", "marketCap",
    ]
    sources = {f: (SRC_API if metrics.get(f) is not None else SRC_NA) for f in source_fields}

    # 直近1年の終値推移 (スパークライン用に週次へ間引き)
    year_start = (base - dt.timedelta(days=365)).isoformat()
    history = [
        {"date": r["Date"], "close": pick(r, "AdjustmentClose", "Close")}
        for r in quotes if r["Date"] >= year_start
    ]
    history = [h for h in history[::3] if h["close"] is not None]

    return {
        "code": display_code(code),
        "jqCode": code,
        "name": info.get("CompanyName") or display_code(code),
        "nameEn": info.get("CompanyNameEnglish"),
        "sector": info.get("Sector33CodeName") or info.get("Sector17CodeName"),
        "market": info.get("MarketCodeName"),
        "scale": info.get("ScaleCategory"),
        "note": note,
        "asOf": latest_date,
        "metrics": metrics,
        "snapshots": snapshots,
        "milestones": build_milestones(quotes, quarters),
        "quarters": quarters[-9:],
        "history": history,
        "sources": sources,
        "origin": "jquants",
    }


# --------------------------------------------------------------------------- #
# エントリポイント
# --------------------------------------------------------------------------- #

def load_watchlist(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"ウォッチリストが見つかりません: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    items = data.get("stocks", data) if isinstance(data, dict) else data
    out = []
    for item in items:
        if isinstance(item, str):
            out.append({"code": item, "note": ""})
        else:
            out.append({"code": str(item.get("code")), "note": item.get("note", "")})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="J-Quants から成長株分析用データを取得する")
    parser.add_argument("--codes", nargs="*", help="銘柄コード (指定時はウォッチリストを無視)")
    parser.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--check-auth", action="store_true", help="認証疎通のみ確認して終了")
    parser.add_argument("--pause", type=float, default=0.25, help="API 呼び出し間隔 (秒)")
    args = parser.parse_args(argv)

    try:
        token = resolve_id_token()
    except AuthError as exc:
        print(f"\n[FATAL] 認証に失敗しました: {exc}\n", file=sys.stderr)
        print(
            "対処: J-Quants にログインして新しいリフレッシュトークンを発行し、\n"
            "      GitHub の Settings > Secrets and variables > Actions で\n"
            "      JQUANTS_API を更新してください。\n"
            "      （無人運用したい場合は JQUANTS_MAIL / JQUANTS_PASSWORD を登録すると\n"
            "        毎回自動でトークンを取得します）",
            file=sys.stderr,
        )
        return 2

    client = JQuantsClient(token, pause=args.pause)

    if args.check_auth:
        probe = client.listed_info("72030")
        print(f"[check-auth] OK : {probe.get('CompanyName') if probe else '(listed/info が空)'}")
        return 0

    targets = (
        [{"code": c, "note": ""} for c in args.codes]
        if args.codes else load_watchlist(args.watchlist)
    )
    print(f"[run] 対象 {len(targets)} 銘柄")

    stocks: List[dict] = []
    failures: List[dict] = []
    for item in targets:
        try:
            stocks.append(build_stock(client, item["code"], item.get("note", "")))
        except AuthError:
            raise
        except (JQuantsError, KeyError, ValueError) as exc:
            print(f"[warn] {item['code']} をスキップ: {exc}", file=sys.stderr)
            failures.append({"code": item["code"], "reason": str(exc)})

    if not stocks:
        print("[FATAL] 1銘柄も取得できませんでした", file=sys.stderr)
        return 1

    payload = {
        "generatedAt": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "source": "J-Quants API (https://jpx-jquants.com/)",
        "stocks": stocks,
        "failures": failures,
        "unavailableEndpoints": client.unavailable_endpoints,
        "notes": {
            "codeNormalization": "4桁コードは末尾に'0'を付加して5桁化 (仕様書 §3.1)",
            "tradingValue": "売買代金(億円) = 終値 × 出来高 / 1e8 (仕様書 §3.2-4)。API の TurnoverValue も turnoverValue に併記",
            "missingData": "取得できなかった指標は null。0 で埋めることはしない",
            "pointInTime": "snapshots の各時点は、その日までに開示済みのデータのみで算出 (先読みなし)",
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"\n[done] {len(stocks)} 銘柄を {args.output} に出力しました")
    if client.unavailable_endpoints:
        print("[info] 利用できなかったエンドポイント (プラン制約の可能性):")
        for ep, reason in client.unavailable_endpoints.items():
            print(f"       - {ep}: {reason[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
