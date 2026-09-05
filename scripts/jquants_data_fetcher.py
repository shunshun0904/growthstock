#!/usr/bin/env python3
"""
jquants_data_fetcher.py
=======================
GrowthStockAnalyzer - Focus のデータ取得・指標算出パイプライン (仕様書 §3)。

J-Quants API **V2** から日次株価・財務諸表・信用残を取得し、8軸スコアリングに必要な
指標へ前処理したうえで ``public/data/stocks.json`` を出力する。
標準ライブラリのみで動作する (CI での依存インストール不要)。

認証 (V2)
---------
V2 は APIキー方式である。V1 の ``auth_user`` → ``auth_refresh`` →
``Authorization: Bearer`` という3段階のトークン交換は廃止された。

  * ベースURL : ``https://api.jquants.com/v2``
  * 認証      : リクエストヘッダー ``x-api-key: <APIキー>``
  * APIキー   : J-Quants ダッシュボードの [設定 » API キー] で発行

環境変数 ``JQUANTS_API`` (GitHub Actions の Repository secret) を使用する。
公式クライアントと同じ ``JQUANTS_API_KEY`` も受け付ける。

V1 → V2 のエンドポイント対応 (本スクリプトで使用するもの)
--------------------------------------------------------
  /v1/listed/info                      -> /v2/equities/master
  /v1/prices/daily_quotes              -> /v2/equities/bars/daily
  /v1/fins/statements                  -> /v2/fins/summary
  /v1/markets/weekly_margin_interest   -> /v2/markets/margin-interest

V2 ではレスポンスの配列キーが一律 ``data`` になり、列名が短縮された
(``Close`` -> ``C``、``NetSales`` -> ``Sales`` 等)。対応は FIELD MAP 節を参照。

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

API_BASE = "https://api.jquants.com/v2"
USER_AGENT = "GrowthStockAnalyzer-Focus/1.0 (+https://github.com/shunshun0904/growthstock)"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WATCHLIST = os.path.join(ROOT, "scripts", "watchlist.json")
DEFAULT_OUTPUT = os.path.join(ROOT, "public", "data", "stocks.json")

PRICE_LOOKBACK_DAYS = 640   # 52週高値を「6ヶ月前時点」でも算出するため 365 + 183 + 余裕
MARGIN_LOOKBACK_DAYS = 400

# 取得できなかった理由の分類
SRC_API = "jquants"
SRC_NA = "unavailable"

# --------------------------------------------------------------------------- #
# FIELD MAP : V1 の列名 -> V2 の列名 (公式クライアント jquantsapi/constants.py 準拠)
# --------------------------------------------------------------------------- #
#
#  /equities/master   CompanyName->CoName  CompanyNameEnglish->CoNameEn
#                     Sector33CodeName->S33Nm  Sector17CodeName->S17Nm
#                     MarketCodeName->MktNm    ScaleCategory->ScaleCat
#
#  /equities/bars/daily
#                     Open->O  High->H  Low->L  Close->C
#                     Volume->Vo   TurnoverValue->Va
#                     AdjustmentOpen->AdjO   AdjustmentHigh->AdjH
#                     AdjustmentLow->AdjL    AdjustmentClose->AdjC
#                     AdjustmentVolume->AdjVo
#
#  /fins/summary      DisclosedDate->DiscDate  DisclosedTime->DiscTime
#                     TypeOfDocument->DocType  TypeOfCurrentPeriod->CurPerType
#                     CurrentPeriodEndDate->CurPerEn
#                     CurrentFiscalYearStartDate->CurFYSt
#                     NetSales->Sales  OperatingProfit->OP  Profit->NP
#                     EarningsPerShare->EPS  Equity->Eq  TotalAssets->TA
#                     ForecastNetSales->FSales  ForecastOperatingProfit->FOP
#                     ForecastProfit->FNP       ForecastEarningsPerShare->FEPS
#                     NumberOfIssuedAndOutstandingShares...->ShOutFY
#                     NumberOfTreasuryStock...->TrShFY
#                     (V2 で新設) ROE ... 自己資本利益率が直接提供される
#
#  /markets/margin-interest
#                     LongMarginTradeVolume->LongVol
#                     ShortMarginTradeVolume->ShrtVol


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
# 認証 (V2: APIキー方式)
# --------------------------------------------------------------------------- #

def _looks_like_jwt(value: str) -> bool:
    """ドット区切り3パート = JWT (V1 のトークン形式) かどうか。"""
    return bool(re.fullmatch(r"[\w-]+\.[\w-]+\.[\w-]+", value.strip()))


def describe_secret(value: str) -> str:
    """
    secret の「形」だけを説明する文字列を返す（値そのものは絶対に出力しない）。

    V1 のトークン (Cognito の JWT / ドット区切り3パート・800文字超) が
    そのまま残っていると V2 では必ず 401/403 になるため、
    JWT らしき値を検出したら移行漏れとして警告する。
    """
    v = value.strip()
    parts = [f"長さ {len(v)} 文字"]
    if _looks_like_jwt(v):
        parts.append(
            "JWT形式 (ドット区切り3パート)"
            "\n      → これは V1 のリフレッシュトークン/IDトークンの形式です。"
            "\n        V2 は APIキー方式に変更されました。J-Quants ダッシュボードの"
            "\n        [設定 » API キー] で発行した APIキーに差し替えてください。"
        )
    else:
        parts.append("JWTではない (APIキーとして送信します)")
    return " / ".join(parts)


def resolve_api_key() -> str:
    """
    環境変数から V2 の APIキーを取得する。

    優先順位: JQUANTS_API_KEY (公式クライアントと同じ名前) -> JQUANTS_API
    """
    for name in ("JQUANTS_API_KEY", "JQUANTS_API"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue

        # JSON で渡された場合はキーらしきフィールドを拾う
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AuthError(f"{name} が JSON らしき文字列ですが解析できません") from exc
            for key in ("apiKey", "api_key", "apikey", "key"):
                if obj.get(key):
                    print(f"[auth] {name} (JSON: {key}) を APIキーとして使用します")
                    return str(obj[key])
            raise AuthError(f"{name} の JSON に APIキーらしきフィールドがありません")

        print(f"[auth] {name} を APIキーとして使用します ({describe_secret(raw)})")
        return raw

    raise AuthError(
        "環境変数 JQUANTS_API (または JQUANTS_API_KEY) が未設定です。\n"
        "      J-Quants ダッシュボードの [設定 » API キー] で APIキーを発行し、\n"
        "      GitHub の Settings > Secrets and variables > Actions に登録してください。"
    )


# --------------------------------------------------------------------------- #
# API クライアント (V2)
# --------------------------------------------------------------------------- #

class JQuantsClient:
    """J-Quants API V2 クライアント (認証は x-api-key ヘッダー)。"""

    def __init__(self, api_key: str, *, pause: float = 0.25) -> None:
        self._headers = {"x-api-key": api_key}
        self._pause = pause
        #: プラン制約などで利用できなかったエンドポイント -> 理由
        self.unavailable_endpoints: Dict[str, str] = {}

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{API_BASE}{path}"
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        time.sleep(self._pause)
        return _request("GET", url, headers=self._headers)

    def get_paginated(self, path: str, params: dict) -> List[dict]:
        """
        pagination_key を辿って全件取得する。
        V2 はどのエンドポイントもデータ配列を "data" キーで返す。
        """
        out: List[dict] = []
        cursor: Optional[str] = None
        for _ in range(50):  # 無限ループ防止
            page_params = dict(params)
            if cursor:
                page_params["pagination_key"] = cursor
            data = self.get(path, page_params)
            batch = data.get("data")
            if isinstance(batch, list):
                out.extend(batch)
            cursor = data.get("pagination_key")
            if not cursor:
                break
        return out

    # --- 個別エンドポイント (V2) ------------------------------------------ #

    def equities_master(self, code: str) -> Optional[dict]:
        """V2 /equities/master (V1: /listed/info)"""
        try:
            rows = self.get_paginated("/equities/master", {"code": code})
            return rows[-1] if rows else None
        except JQuantsError as exc:
            self.unavailable_endpoints["/equities/master"] = str(exc)
            return None

    def daily_bars(self, code: str, date_from: str, date_to: str) -> List[dict]:
        """V2 /equities/bars/daily (V1: /prices/daily_quotes)"""
        try:
            rows = self.get_paginated(
                "/equities/bars/daily", {"code": code, "from": date_from, "to": date_to}
            )
        except JQuantsError as exc:
            self.unavailable_endpoints["/equities/bars/daily"] = str(exc)
            return []
        return sorted(rows, key=lambda r: r.get("Date", ""))

    def fin_summary(self, code: str) -> List[dict]:
        """V2 /fins/summary (V1: /fins/statements)"""
        try:
            rows = self.get_paginated("/fins/summary", {"code": code})
        except JQuantsError as exc:
            self.unavailable_endpoints["/fins/summary"] = str(exc)
            return []
        return sorted(rows, key=lambda r: (r.get("DiscDate", ""), r.get("DiscTime", "")))

    def margin_interest(self, code: str, date_from: str, date_to: str) -> List[dict]:
        """V2 /markets/margin-interest (V1: /markets/weekly_margin_interest)"""
        try:
            rows = self.get_paginated(
                "/markets/margin-interest",
                {"code": code, "from": date_from, "to": date_to},
            )
        except JQuantsError as exc:
            # 契約プランによっては利用不可。需給軸は「データなし」として扱う。
            self.unavailable_endpoints["/markets/margin-interest"] = str(exc)
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

    price = pick(latest, "AdjC", "C")
    out["price"] = price

    # 52週高値: 直近営業日から遡って 365日ぶんのバーの最高値
    end = dt.date.fromisoformat(latest["Date"])
    start = end - dt.timedelta(days=365)
    window = [r for r in series if dt.date.fromisoformat(r["Date"]) >= start]
    highs = [h for h in (pick(r, "AdjH", "H") for r in window) if h is not None]
    if highs:
        out["high52w"] = max(highs)
        if price is not None and out["high52w"] > 0:
            out["highRatio"] = price / out["high52w"] * 100.0

    # 売買代金 (億円): 仕様書 §3.2-4 の定義 P_current × Volume / 1e8
    raw_close = pick(latest, "C", "AdjC")
    raw_volume = pick(latest, "Vo", "AdjVo")
    out["volume"] = raw_volume
    if raw_close is not None and raw_volume is not None:
        out["tradingValue"] = raw_close * raw_volume / 1e8
    # API が返す実際の売買代金 (参考値)
    turnover = fnum(latest.get("Va"))
    if turnover is not None:
        out["turnoverValue"] = turnover / 1e8

    # 出来高モメンタム (%): 直近出来高 / 直前20日平均出来高
    prior = [v for v in (pick(r, "AdjVo", "Vo") for r in series[-21:-1]) if v is not None]
    if len(prior) >= 5 and raw_volume is not None:
        ma20 = sum(prior) / len(prior)
        out["ma20Volume"] = ma20
        if ma20 > 0:
            latest_vol = pick(latest, "AdjVo", "Vo")
            if latest_vol is not None:
                out["volumeTrend"] = latest_vol / ma20 * 100.0

    return out


# --------------------------------------------------------------------------- #
# 指標算出 : 財務系
# --------------------------------------------------------------------------- #

QUARTER_MAP = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4}

# 実績値を持つ列 (いずれか1つでも埋まっていれば実績開示とみなす)
ACTUAL_FIELDS = ("Sales", "OP", "NP", "EPS")


def _is_financial_statement(row: dict) -> bool:
    """
    実績決算の開示行かどうかを判定する。

    V1 では TypeOfDocument に "FinancialStatements" が含まれるかで判定していたが、
    V2 の DocType の列挙値は公式クライアントのソースからは確認できなかったため、
    文字列一致には依存しない判定にしている:

      * CurPerType が 1Q/2Q/3Q/4Q/FY のいずれかであること
      * かつ 売上・営業利益・純利益・EPS のいずれかに実績値があること

    業績予想の修正のみを開示した行は実績値を持たないため、この条件で除外される。
    """
    if row.get("CurPerType") not in QUARTER_MAP:
        return False
    return any(fnum(row.get(f)) is not None for f in ACTUAL_FIELDS)


def quarterize(statements: Sequence[dict]) -> List[dict]:
    """
    累計ベースの決算短信データを「単一四半期」の値へ差分展開する。

    J-Quants の fins/statements は会計年度内の累計値を返すため、
    2Q の値から 1Q の値を引く等の処理を行わないと前年同期比が正しく出せない。
    """
    rows = [r for r in statements if _is_financial_statement(r)]
    by_fy: Dict[str, List[dict]] = {}
    for r in rows:
        by_fy.setdefault(r.get("CurFYSt") or "", []).append(r)

    result: List[dict] = []
    for fy_start, group in by_fy.items():
        group = sorted(group, key=lambda r: QUARTER_MAP[r["CurPerType"]])
        # 同一四半期の重複開示 (訂正) は最後の開示を採用
        dedup: Dict[int, dict] = {}
        for r in group:
            dedup[QUARTER_MAP[r["CurPerType"]]] = r
        ordered = [dedup[q] for q in sorted(dedup)]

        for idx, r in enumerate(ordered):
            q = QUARTER_MAP[r["CurPerType"]]
            prev = ordered[idx - 1] if idx > 0 else None
            prev_q = QUARTER_MAP[prev["CurPerType"]] if prev else 0

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
                "period": r.get("CurPerType"),
                "disclosedDate": r.get("DiscDate"),
                "periodEnd": r.get("CurPerEn"),
                # 単一四半期値
                "qNetSales": diff("Sales"),
                "qOperatingProfit": diff("OP"),
                "qProfit": diff("NP"),
                "qEps": diff("EPS"),
                # 累計値 (進捗率算出に使用)
                "cumNetSales": fnum(r.get("Sales")),
                "cumOperatingProfit": fnum(r.get("OP")),
                "cumProfit": fnum(r.get("NP")),
                "cumEps": fnum(r.get("EPS")),
                # 会社予想 (通期)
                "forecastNetSales": fnum(r.get("FSales")),
                "forecastOperatingProfit": fnum(r.get("FOP")),
                "forecastProfit": fnum(r.get("FNP")),
                "forecastEps": fnum(r.get("FEPS")),
                # 財政状態
                "equity": fnum(r.get("Eq")),
                "totalAssets": fnum(r.get("TA")),
                "sharesIssued": fnum(r.get("ShOutFY")),
                "treasuryShares": fnum(r.get("TrShFY")),
                # V2 で新設: 自己資本利益率が API から直接提供される
                "reportedRoe": fnum(r.get("ROE")),
            })

    return sorted(result, key=lambda r: (r.get("disclosedDate") or "", r["fiscalYearStart"], r["quarter"]))


def fundamental_metrics(quarters: Sequence[dict], as_of: Optional[str] = None) -> Dict[str, Any]:
    """指定日時点で「開示済み」の決算のみを使って財務指標を算出する。"""
    hist = rows_upto(quarters, "disclosedDate", as_of) if as_of else list(quarters)
    out: Dict[str, Any] = {
        "epsGrowth": None, "salesGrowth": None, "roe": None, "opMargin": None,
        "progressRate": None, "quarter": None, "sharesOutstanding": None,
        "fiscalPeriod": None, "disclosedDate": None, "opMarginBasis": None,
        "roeBasis": None,
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

    # --- ROE ---
    # V2 の /fins/summary は ROE を直接提供する。提供値があればそれを使い、
    # 無い場合のみ「直近4四半期の純利益合計 / 自己資本」で算出する。
    if latest.get("reportedRoe") is not None:
        out["roe"] = latest["reportedRoe"]
        out["roeBasis"] = "J-Quants 提供値 (ROE)"
    else:
        ttm = [r.get("qProfit") for r in hist[-4:]]
        equity = latest.get("equity")
        if len(ttm) == 4 and all(v is not None for v in ttm) and equity and equity > 0:
            out["roe"] = sum(ttm) / equity * 100.0
            out["roeBasis"] = "TTM純利益 / 自己資本 で算出"

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
    long_v = fnum(latest.get("LongVol"))
    short_v = fnum(latest.get("ShrtVol"))
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
        close = pick(row, "AdjC", "C")
        open_ = pick(row, "AdjO", "O")
        vol = pick(row, "AdjVo", "Vo")
        if close is None or vol is None:
            continue

        start = dt.date.fromisoformat(d) - dt.timedelta(days=365)
        window = [r for r in quotes[:i] if dt.date.fromisoformat(r["Date"]) >= start]
        highs = [h for h in (pick(r, "AdjH", "H") for r in window) if h is not None]
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

        prior = [v for v in (pick(r, "AdjVo", "Vo") for r in quotes[i - 20:i]) if v is not None]
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

    print(f"[fetch] {code} : /equities/master")
    info = client.equities_master(code) or {}
    print(f"[fetch] {code} : /equities/bars/daily ({date_from} 〜 {date_to})")
    quotes = client.daily_bars(code, date_from, date_to)
    print(f"[fetch] {code} : /fins/summary")
    statements = client.fin_summary(code)
    print(f"[fetch] {code} : /markets/margin-interest")
    margins = client.margin_interest(
        code, (today - dt.timedelta(days=MARGIN_LOOKBACK_DAYS)).isoformat(), date_to
    )

    quarters = quarterize(statements)

    if not quotes:
        return {
            "code": display_code(code),
            "jqCode": code,
            "name": info.get("CoName") or display_code(code),
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
        {"date": r["Date"], "close": pick(r, "AdjC", "C")}
        for r in quotes if r["Date"] >= year_start
    ]
    history = [h for h in history[::3] if h["close"] is not None]

    return {
        "code": display_code(code),
        "jqCode": code,
        "name": info.get("CoName") or display_code(code),
        "nameEn": info.get("CoNameEn"),
        "sector": info.get("S33Nm") or info.get("S17Nm"),
        "market": info.get("MktNm"),
        "scale": info.get("ScaleCat"),
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
        api_key = resolve_api_key()
    except AuthError as exc:
        print(f"\n[FATAL] 認証情報を解決できませんでした: {exc}\n", file=sys.stderr)
        return 2

    client = JQuantsClient(api_key, pause=args.pause)

    if args.check_auth:
        # 疎通確認は /equities/master を 1 銘柄だけ叩く
        try:
            rows = client.get_paginated("/equities/master", {"code": "72030"})
        except AuthError as exc:
            print(f"\n[FATAL] APIキーが拒否されました: {exc}\n", file=sys.stderr)
            print(
                "対処: J-Quants ダッシュボードの [設定 » API キー] で APIキーを確認し、\n"
                "      GitHub の Settings > Secrets and variables > Actions で\n"
                "      JQUANTS_API を更新してください。\n"
                "      (V2 は APIキー方式です。V1 のリフレッシュトークンは使えません)",
                file=sys.stderr,
            )
            return 2
        except JQuantsError as exc:
            print(f"\n[FATAL] J-Quants API に接続できませんでした: {exc}\n", file=sys.stderr)
            print("      ネットワーク到達性を確認してください "
                  "(api.jquants.com への HTTPS 接続が必要です)", file=sys.stderr)
            return 3
        if not rows:
            print("[check-auth] 認証は通りましたが /equities/master が空を返しました",
                  file=sys.stderr)
            return 1
        info = rows[-1]
        print(f"[check-auth] OK : {info.get('Code')} {info.get('CoName')} "
              f"({info.get('MktNm')} / {info.get('S33Nm')})")
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
        "source": "J-Quants API V2 (https://jpx-jquants.com/)",
        "apiVersion": "v2",
        "stocks": stocks,
        "failures": failures,
        "unavailableEndpoints": client.unavailable_endpoints,
        "notes": {
            "codeNormalization": "4桁コードは末尾に'0'を付加して5桁化 (仕様書 §3.1)",
            "tradingValue": "売買代金(億円) = 終値 × 出来高 / 1e8 (仕様書 §3.2-4)。API の TurnoverValue も turnoverValue に併記",
            "missingData": "取得できなかった指標は null。0 で埋めることはしない",
            "pointInTime": "snapshots の各時点は、その日までに開示済みのデータのみで算出 (先読みなし)",
            "apiVersion": "J-Quants API V2 (x-api-key 認証 / エンドポイントと列名は V1 から変更)",
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
