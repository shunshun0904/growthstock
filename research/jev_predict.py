#!/usr/bin/env python3
"""
Jev（TypeSafe AI の判断モデル）に、その日の候補について「翌営業日の寄りで買い、
20営業日以内に +10% に届くか」の確率を問い、予測ファイルと控えに残す。

Jev とは
--------
TypeSafe AI が 2026-09-15 に公開した判断特化のモデル（System One）。文章を生成せず、
与えた状態（テキストか JSON）と名前付きの問いに対して、型付きの値と確率だけを返す。
問いは3種類: noul（真偽の確率 0〜1）/ choice（選択肢ごとの確率）/ score（段階の期待値）。
ここでは noul を1つだけ使う（運用者の決定 2026-10-10。docs/MODEL_JEV.md）。

HTTP の形は公式 Python SDK（typesafe-sdk 0.7.4。api.typesafe.ai/openapi.json から生成された
_schemas/models.py）から写した:

  POST {TYPESAFE_BASE_URL}/v1/systemone
  Authorization: Bearer <TYPESAFE_API_KEY>
  {"state": <text|object|array>, "model": "jev-latest",
   "questions": {"rise10": {"type": "noul", "instructions": "...",
                            "criteria": {"true": "...", "false": "..."}}}}
  -> {"model": "jev-1.13.0",
      "answers": {"rise10": {"type": "noul", "noul": 0.37}},
      "usage": {"input_tokens": 900, "output_tokens": 1}}

SDK を入れずに標準ライブラリ（urllib）で呼ぶ。依存（httpx2 / pydantic / tenacity）を増やさず、
通信を差し替えて単体テストできるようにするため。再試行（429 / 5xx / 接続失敗）は自前。
鍵は環境変数からだけ読み、ログにも例外にも出さない。

扱いの原則
----------
1. 鍵（TYPESAFE_API_KEY）が無ければ何もしない。日次予測はそのまま動く（画面は「—」）
2. 答えは (予測日, 銘柄コード, 問いの版) ごとに控え（research/_data/jev_answers.parquet。
   Release data-raw に置く）、**最初の答えで凍結する**。同じ日を予測し直しても問い直さない
   （記録（prediction_history / 台帳）と同じ考え方。課金もしない）
3. 1銘柄の失敗で他を止めない。失敗した銘柄は null のまま（0 で埋めない）。
   続けて失敗したら、その実行では残りを問わない（障害の日に何十分も待たない）
4. 1回の実行で問う数に上限（JEV_MAX_PER_RUN、既定 200）。新しい日・上位から順に問う
5. 選定の規則（src/lib/strategy.js の strategySignal）には入れない。画面に並べ、台帳に残す
   だけ。確率は較正されていない（過去分は実験73 research/exp/e73_jev_oof.py で測る）

  python3 research/jev_predict.py --dry-run   # 公開中の predictions.json の1件目で state と問いを出す（通信しない）
  python3 research/jev_predict.py --check     # 鍵と疎通（GET /v1/models）
  python3 research/jev_predict.py --sample    # 1件だけ実際に問う（課金あり。控えには書かない）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
PUBLIC_DIR = os.path.join(os.path.dirname(HERE), "public", "data")

API_KEY_ENV = "TYPESAFE_API_KEY"
BASE_URL_ENV = "TYPESAFE_BASE_URL"
#: モデル名の上書き。既定は別名 jev-latest（応答に実際の版が入るので、それを控えに残す）
MODEL_ENV = "JEV_MODEL"
MAX_PER_RUN_ENV = "JEV_MAX_PER_RUN"

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
SYSTEM_ONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"

DEFAULT_MAX_PER_RUN = 200
#: 1リクエストの待ち（秒）。SDK の既定 10秒より長め（state が長い）
TIMEOUT = 30.0
#: 再試行の回数と最初の待ち（秒）。2 → 4 → 8
TRIES = 4
BACKOFF = 2.0
#: 続けてこの回数失敗したら、その実行では残りを問わない
MAX_CONSECUTIVE_FAILURES = 3

#: 問い。+10% は実験70 の物差し（+10% の指値）と同じ。20営業日は運用の保有期間
TARGET_PCT = 10
HOLD_DAYS = 20
QUESTION_KEY = "rise10"
#: 問いの文言を変えたら上げる。控えと予測ファイルに残し、版の違う答えを混ぜない
QUESTION_VERSION = "rise10_v1"
QUESTIONS: Dict[str, Dict[str, Any]] = {
    QUESTION_KEY: {
        "type": "noul",
        "instructions": (
            "この日本株を翌営業日の寄り付き（始値）で買ったとき、買った日を1日目として"
            f"{HOLD_DAYS}営業日以内に、株価の高値が買値の +{TARGET_PCT}% 以上に達する。"),
        "criteria": {
            "true": (f"{HOLD_DAYS}営業日以内のいずれかの日の高値が、"
                     f"買値の {1 + TARGET_PCT / 100:.2f}倍以上になる"),
            "false": (f"{HOLD_DAYS}営業日のあいだ、高値が一度も"
                      f"買値の {1 + TARGET_PCT / 100:.2f}倍に届かない"),
        },
    },
}

#: state に入れる「効いた特徴量」の数（predict_daily の寄与の上位から）
TOP_FEATURES = 8

ANSWERS_FILE = "jev_answers.parquet"
#: 控えを残す日数。live_features と同じ
KEEP_DAYS = 400
ANSWER_COLS = ["date", "code", "question", "model", "prob",
               "input_tokens", "output_tokens", "asked_at", "state_sha"]


class JevError(RuntimeError):
    """Jev の呼び出しに失敗した（再試行しても）。鍵は含めない。"""


def enabled() -> bool:
    return bool(os.environ.get(API_KEY_ENV, "").strip())


# ---------------------------------------------------------------------- #
# state（Jev に渡す中身）
# ---------------------------------------------------------------------- #

def _num(v, d: int = 2) -> Optional[float]:
    """数値を JSON に出せる float に。欠測・NaN・非数は None（0 で埋めない）。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, d)


def _int(v) -> Optional[int]:
    f = _num(v, 0)
    return None if f is None else int(f)


def _text(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    s = str(v).strip()
    return s or None


def prune(obj):
    """None と空の入れ物を再帰的に落とす（Jev に「無い」と伝えるより、書かないほうが短い）。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pv = prune(v)
            if pv is None or pv == {} or pv == []:
                continue
            out[k] = pv
        return out
    if isinstance(obj, list):
        out = [prune(v) for v in obj]
        return [v for v in out if v is not None and v != {} and v != []]
    return obj


def market_tone(rows: List[Dict]) -> Optional[str]:
    """
    その日の地合い。候補ごとの地合い寄与（contrib.marketContrib）の中央値で決める。
    画面（src/lib/predictions.js marketTone）と同じしきい値 ±0.05。
    """
    v = sorted(x for x in ((r.get("contrib") or {}).get("marketContrib") for r in rows)
               if isinstance(x, (int, float)) and np.isfinite(x))
    if not v:
        return None
    med = v[len(v) // 2]
    return "追い風" if med > 0.05 else "向かい風" if med < -0.05 else "中立"


def build_state(c: Dict, tone: Optional[str] = None, anonymous: bool = False) -> Dict:
    """
    候補1件（predictions.json の candidates の1要素）を Jev に渡す形にする。

    入れるもの（運用者の決定 2026-10-10）: 画面に出している実測値 ＋ 4モデルの百分位 ＋ 基準モデルの
    寄与（区分ごとと上位の特徴量）。日本語の鍵で、欠測は書かない。
    anonymous=True は実験73 の腕で、銘柄コード・名前・業種・市場・日付を伏せる（Jev が過去の
    日付の銘柄について「その後」を知っている疑いを測るため）。本番では使わない。
    """
    per = c.get("byModel") or {}
    ct = c.get("contrib") or {}

    def pct(algo: str) -> Optional[float]:
        return _num((per.get(algo) or {}).get("pctHistorical"), 1)

    state = {
        "課題": ("日本株。78週ぶりの高値を更新した日の終値後に見ている銘柄。"
                 f"翌営業日の寄り付きで買い、{HOLD_DAYS}営業日以内に買値から +{TARGET_PCT}% に"
                 "届くかを判定する"),
        "日付": None if anonymous else _text(c.get("date")),
        "銘柄": None if anonymous else {
            "コード": _text(c.get("code")), "名前": _text(c.get("name")),
            "業種": _text(c.get("sector")), "市場": _text(c.get("market")),
            "規模区分": _text(c.get("scale")),
        },
        "株価": {
            "終値(円)": _num(c.get("close"), 1),
            "78週高値(円)": _num(c.get("high52w"), 1),
            "高値に対する終値の位置(%)": _num(c.get("rHigh")),
            "高値を抜けた幅(%)": _num(c.get("breakMargin")),
            "ベースの長さ(前の高値からの営業日数)": _int(c.get("baseLength")),
            "20営業日リターン(%)": _num(c.get("ret20d")),
            "日次ボラティリティ20日(%)": _num(c.get("vol20d")),
            "出来高トレンド(%。100が平常)": _num(c.get("volumeTrend"), 1),
        },
        "需給・規模": {
            "時価総額(億円)": _num(c.get("marketCap"), 0),
            "売買代金20日平均(億円/日)": _num(c.get("tradingValue")),
            "信用倍率(倍)": _num(c.get("creditRatio")),
        },
        "バリュエーション": {
            "PER(倍)": _num(c.get("per"), 1), "PBR(倍)": _num(c.get("pbr")),
            "配当利回り(%)": _num(c.get("divYield")),
        },
        "決算": {
            "直近四半期のEPS成長率(前年同期比%)": _num(c.get("epsGrowth"), 1),
            "直近四半期の売上成長率(前年同期比%)": _num(c.get("salesGrowth"), 1),
            "ROE(%)": _num(c.get("roe")),
            "営業利益率(%)": _num(c.get("opMargin")),
            "通期計画に対する進捗率(%)": _num(c.get("progressRate"), 1),
            "四半期": _int(c.get("quarter")),
            "進捗率の基準(例年並みの%)": _num(c.get("progressBenchmark"), 1),
        },
        "その日": {
            "78週高値を更新した銘柄数(発火数)": _int(c.get("nInDay")),
            "基準モデルでの順位(その日の候補の中)": _int(c.get("rankInDay")),
            "地合い(候補全体の市場環境の寄与)": _text(tone),
        },
        "機械学習モデルの見立て": {
            "説明": ("学習器の違う4モデルが独立に付けたスコアの、各モデル自身の過去スコア分布での"
                     "百分位（0〜100。高いほど上がる見込み）。母集団（高値更新日の銘柄）のうち"
                     "モデルの基準で上がった割合は約18%"),
            "LightGBM": pct("lgbm"), "XGBoost": pct("xgb"), "CatBoost": pct("cat"),
            "ロジスティック回帰": pct("logit"),
            "必要上昇率(モデルの正例の基準。銘柄のボラの1.2σ、%)": _num(c.get("needPct"), 1),
        },
        "基準モデル(LightGBM)の寄与": {
            "説明": "TreeSHAP による対数オッズの寄与。正は押し上げ、負は押し下げ",
            "区分ごとの寄与": {str(k): _num(v, 3) for k, v in (ct.get("groups") or {}).items()},
            "効いた特徴量": [
                {"特徴量": _text(t.get("col")), "意味": _text(t.get("ja")),
                 "寄与": _num(t.get("contrib"), 3)}
                for t in (ct.get("top") or [])[:TOP_FEATURES]
                if isinstance(t, dict)
            ],
        },
    }
    return prune(state)


def state_sha(state: Dict) -> str:
    """state の指紋。作り方を変えたときに、控えの答えがどの作りのものか分かるように残す。"""
    s = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------- #
# HTTP
# ---------------------------------------------------------------------- #

class Client:
    """
    TypeSafe AI の System One を標準ライブラリで叩く。

    opener / sleep は単体テストで差し替える（通信しない）。鍵はヘッダーに入れるだけで、
    repr にも例外にも出さない。
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = TIMEOUT, tries: int = TRIES,
                 backoff: float = BACKOFF, opener: Optional[Callable] = None,
                 sleep: Callable[[float], None] = time.sleep):
        key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")).strip()
        if not key:
            raise JevError(f"鍵がありません（環境変数 {API_KEY_ENV}）")
        if not key.isascii() or not key.isprintable() or " " in key:
            raise JevError("鍵に使えない文字が入っています（空白・制御文字・非 ASCII）")
        self._key = key
        self.base_url = ((base_url if base_url is not None else os.environ.get(BASE_URL_ENV, ""))
                         .strip() or DEFAULT_BASE_URL).rstrip("/")
        self.model = ((model if model is not None else os.environ.get(MODEL_ENV, "")).strip()
                      or DEFAULT_MODEL)
        self.timeout = float(timeout)
        self.tries = max(1, int(tries))
        self.backoff = float(backoff)
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep

    def __repr__(self) -> str:        # 鍵を出さない
        return f"Client(base_url={self.base_url!r}, model={self.model!r})"

    def _headers(self, body: bool) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {self._key}", "Accept": "application/json",
             "User-Agent": "growthstock-jev/1 (urllib)"}
        if body:
            h["Content-Type"] = "application/json"
        return h

    def _request(self, method: str, path: str, body: Optional[Dict] = None) -> Dict:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        last = "不明な失敗"
        for i in range(self.tries):
            req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                         headers=self._headers(data is not None))
            retry_after: Optional[float] = None
            try:
                with self._open(req, timeout=self.timeout) as r:
                    raw = r.read()
                try:
                    out = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise JevError(f"応答が JSON ではない: {type(exc).__name__}") from None
                if not isinstance(out, dict):
                    raise JevError("応答が JSON オブジェクトではない")
                return out
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                try:
                    snippet = exc.read().decode("utf-8", "replace")[:200]
                except Exception:              # noqa: BLE001
                    snippet = ""
                last = f"HTTP {status} {snippet}".strip()
                if status in (408, 409, 425, 429) or status >= 500:
                    ra = exc.headers.get("retry-after") if exc.headers else None
                    try:
                        retry_after = float(ra) if ra else None
                    except ValueError:
                        retry_after = None
                else:
                    # 400 / 401 / 403 / 404 / 422 は繰り返しても同じ。すぐ止める
                    raise JevError(last) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
            if i + 1 < self.tries:
                wait = self.backoff * (2 ** i)
                if retry_after is not None:
                    wait = max(wait, min(retry_after, 60.0))
                self._sleep(wait)
        raise JevError(f"{self.tries}回失敗: {last}")

    def system_one(self, state, questions: Dict[str, Dict] = QUESTIONS) -> Dict:
        if state is None:
            raise JevError("state が無い")
        if not questions:
            raise JevError("問いが無い")
        return self._request("POST", SYSTEM_ONE_PATH,
                             {"state": state, "model": self.model, "questions": questions})

    def models(self) -> List[Dict]:
        out = self._request("GET", MODELS_PATH)
        models = out.get("models")
        return models if isinstance(models, list) else []


def parse_noul(resp: Dict, key: str = QUESTION_KEY) -> Dict:
    """
    応答から noul の確率を取り出す。確率は % に直す（画面の較正確率と同じ単位）。
    形が違えば JevError（黙って 0 や None にしない）。
    """
    answers = resp.get("answers") if isinstance(resp, dict) else None
    ans = answers.get(key) if isinstance(answers, dict) else None
    if not isinstance(ans, dict) or ans.get("type") != "noul":
        raise JevError(f"答え {key!r} が noul ではない")
    try:
        p = float(ans.get("noul"))
    except (TypeError, ValueError):
        raise JevError(f"答え {key!r} の確率が数ではない") from None
    if not np.isfinite(p) or p < 0 or p > 1:
        raise JevError(f"答え {key!r} の確率が 0〜1 の外: {p!r}")
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    return {
        "prob": round(p * 100.0, 1),
        "model": _text(resp.get("model")) or "",
        "input_tokens": _int(usage.get("input_tokens")) or 0,
        "output_tokens": _int(usage.get("output_tokens")) or 0,
    }


# ---------------------------------------------------------------------- #
# 控え
# ---------------------------------------------------------------------- #

def load_answers(path: str, log: Callable[[str], None] = print) -> pd.DataFrame:
    """控えを読む。無い・読めないときは空（読めないときはその旨を出して作り直す）。"""
    empty = pd.DataFrame(columns=ANSWER_COLS)
    if not os.path.exists(path):
        return empty
    try:
        df = pd.read_parquet(path)
    except Exception as exc:                  # noqa: BLE001
        log(f"[jev] 控えを読めないので作り直す: {type(exc).__name__}")
        return empty
    for c in ANSWER_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[ANSWER_COLS].copy()
    df["date"] = df["date"].astype(str)
    df["code"] = df["code"].astype(str)
    df["question"] = df["question"].astype(str)
    return df


def save_answers(df: pd.DataFrame, path: str, keep_days: int = KEEP_DAYS) -> pd.DataFrame:
    """古い行を落として書く。date は 'YYYY-MM-DD' の文字列。"""
    out = df[ANSWER_COLS].copy()
    if len(out):
        d = pd.to_datetime(out["date"], errors="coerce")
        cutoff = d.max() - pd.Timedelta(days=keep_days)
        out = out[d.isna() | (d >= cutoff)]
    out = out.drop_duplicates(["date", "code", "question"], keep="first").reset_index(drop=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_parquet(path, index=False, compression="zstd")
    return out


def _from_record(rec: Dict, cached: bool) -> Dict:
    """控えの1行 → predictions.json の candidates[].jev。"""
    return {
        "prob": _num(rec.get("prob"), 1),
        "model": _text(rec.get("model")) or None,
        "question": _text(rec.get("question")) or QUESTION_VERSION,
        "askedAt": _text(rec.get("asked_at")),
        "cached": bool(cached),
    }


def _max_per_run(value: Optional[int]) -> int:
    if value is not None:
        return max(0, int(value))
    raw = os.environ.get(MAX_PER_RUN_ENV, "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_PER_RUN
    except ValueError:
        return DEFAULT_MAX_PER_RUN


def summary_skeleton(model: Optional[str] = None) -> Dict:
    """predictions.json の payload.jev。画面の「Jev（判断モデル）」の欄がそのまま読む。"""
    return {
        "enabled": False,
        "model": model,
        "resolvedModel": None,
        "question": QUESTIONS[QUESTION_KEY],
        "questionVersion": QUESTION_VERSION,
        "target": {"pct": TARGET_PCT, "days": HOLD_DAYS},
        "asked": 0, "cached": 0, "failed": 0, "skipped": 0,
        "inputTokens": 0, "outputTokens": 0,
        "note": "",
    }


def annotate(rows: List[Dict], data_dir: str = DATA_DIR, client: Optional[Client] = None,
             max_per_run: Optional[int] = None, now: Optional[str] = None,
             log: Callable[[str], None] = print) -> Dict:
    """
    各候補（predictions.json の candidates。byModel / contrib 付き）に row["jev"] を足す。

    控えにある (日付, コード, 問いの版) は問い直さない（凍結）。新しい日・上位から順に問い、
    上限（max_per_run）と連続失敗の打ち切りの先は None のまま。戻り値は payload.jev。
    """
    summary = summary_skeleton()
    for r in rows:
        r.setdefault("jev", None)
    if client is None:
        if not enabled():
            summary["note"] = (f"鍵（{API_KEY_ENV}）が未設定のため問うていない。"
                               "設定した日から値が入る（過去分は遡らない）")
            return summary
        client = Client()
    summary["enabled"] = True
    summary["model"] = client.model
    limit = _max_per_run(max_per_run)
    asked_at = now or pd.Timestamp.now(tz="UTC").isoformat()

    path = os.path.join(data_dir, ANSWERS_FILE)
    cache = load_answers(path, log)
    known: Dict[tuple, Dict] = {
        (str(x["date"]), str(x["code"]), str(x["question"])): x
        for x in cache.to_dict("records")
    }
    tones = {}
    for r in rows:
        tones.setdefault(r.get("date"), []).append(r)
    tones = {d: market_tone(v) for d, v in tones.items()}

    # 新しい日から、その日の上位から。上限で切れるのが古くて順位の低いものになるように
    order = sorted(rows, key=lambda r: (str(r.get("date") or ""), -int(r.get("rankInDay") or 0)),
                   reverse=True)
    new_records: List[Dict] = []
    attempts = 0
    consecutive = 0
    halted = False
    resolved: Dict[str, int] = {}
    for r in order:
        key = (str(r.get("date")), str(r.get("jqCode")), QUESTION_VERSION)
        if key in known:
            r["jev"] = _from_record(known[key], cached=True)
            summary["cached"] += 1
            continue
        if halted or attempts >= limit:
            summary["skipped"] += 1
            continue
        state = build_state(r, tones.get(r.get("date")))
        attempts += 1
        try:
            ans = parse_noul(client.system_one(state))
        except JevError as exc:
            summary["failed"] += 1
            consecutive += 1
            log(f"  [jev] {r.get('date')} {r.get('code')} 失敗: {exc}")
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                halted = True
                log(f"  [jev] {consecutive}回続けて失敗したので、この実行では残りを問わない")
            continue
        consecutive = 0
        rec = {
            "date": key[0], "code": key[1], "question": key[2],
            "model": ans["model"] or client.model, "prob": ans["prob"],
            "input_tokens": ans["input_tokens"], "output_tokens": ans["output_tokens"],
            "asked_at": asked_at, "state_sha": state_sha(state),
        }
        new_records.append(rec)
        known[key] = rec
        r["jev"] = _from_record(rec, cached=False)
        summary["asked"] += 1
        summary["inputTokens"] += int(ans["input_tokens"])
        summary["outputTokens"] += int(ans["output_tokens"])
        resolved[rec["model"]] = resolved.get(rec["model"], 0) + 1

    if new_records:
        new_df = pd.DataFrame(new_records, columns=ANSWER_COLS)
        merged = new_df if cache.empty else pd.concat([cache, new_df], ignore_index=True)
        kept = save_answers(merged, path)
        log(f"[jev] 控えを書いた: {path}（{len(new_records)}件追加 / 累計 {len(kept):,}件）")
    if resolved:
        summary["resolvedModel"] = max(resolved.items(), key=lambda kv: kv[1])[0]
    elif len(cache):
        last = cache.sort_values("asked_at").iloc[-1]
        summary["resolvedModel"] = _text(last.get("model"))
    notes = []
    if summary["skipped"] and halted:
        notes.append(f"続けて失敗したので {summary['skipped']}件は問うていない")
    elif summary["skipped"]:
        notes.append(f"1回の上限 {limit}件に当たり {summary['skipped']}件は問うていない")
    if summary["failed"]:
        notes.append(f"{summary['failed']}件は失敗（値は空のまま）")
    summary["note"] = "。".join(notes)
    return summary


def describe(summary: Dict) -> str:
    """ログ1行。"""
    if not summary.get("enabled"):
        return f"Jev: 問うていない（{summary.get('note') or '無効'}）"
    s = (f"Jev: 問うた {summary['asked']}件 / 控えから {summary['cached']}件 / "
         f"失敗 {summary['failed']}件 / 問わず {summary['skipped']}件 / "
         f"入力トークン {summary['inputTokens']:,}")
    if summary.get("resolvedModel"):
        s += f" / モデル {summary['resolvedModel']}"
    if summary.get("note"):
        s += f"（{summary['note']}）"
    return s


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def _first_candidate(path: str) -> tuple:
    """予測ファイルの最新日の1位と、その日の全候補（地合いを出すため）。"""
    with open(path, encoding="utf-8") as fh:
        pred = json.load(fh)
    cands = pred.get("candidates") or []
    if not cands:
        raise SystemExit(f"候補が無い: {path}")
    latest = max(c.get("date", "") for c in cands)
    today = [c for c in cands if c.get("date") == latest]
    today.sort(key=lambda c: c.get("rankInDay") or 0)
    return today[0], today


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Jev に候補の +10% 到達の確率を問う")
    ap.add_argument("--predictions", default=os.path.join(PUBLIC_DIR, "predictions.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="最新日の1位の候補で state と問いを出す（通信しない）")
    ap.add_argument("--check", action="store_true", help="鍵と疎通（GET /v1/models）")
    ap.add_argument("--sample", action="store_true",
                    help="最新日の1位の候補を1件だけ実際に問う（課金あり。控えには書かない）")
    args = ap.parse_args(argv)

    if args.dry_run:
        c, today = _first_candidate(args.predictions)
        state = build_state(c, market_tone(today))
        body = {"state": state, "model": os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL,
                "questions": QUESTIONS}
        text = json.dumps(body, ensure_ascii=False, indent=1)
        print(text)
        print(f"\n[dry-run] {len(text):,}文字 / {len(text.encode('utf-8')):,}バイト "
              f"/ 指紋 {state_sha(state)} / 通信していません")
        return 0

    if args.check:
        if not enabled():
            print(f"[check] {API_KEY_ENV} が未設定です")
            return 1
        client = Client()
        print(f"[check] {client.base_url} / 既定のモデル {client.model}")
        try:
            models = client.models()
        except JevError as exc:
            print(f"[check] 疎通に失敗: {exc}")
            return 1
        for m in models:
            print(f"  {m.get('name')}  {m.get('release_date', '')}  {m.get('description', '')}")
        print(f"[check] OK（{len(models)}モデル）")
        return 0

    if args.sample:
        if not enabled():
            print(f"[sample] {API_KEY_ENV} が未設定です")
            return 1
        c, today = _first_candidate(args.predictions)
        client = Client()
        state = build_state(c, market_tone(today))
        try:
            resp = client.system_one(state)
        except JevError as exc:
            print(f"[sample] 失敗: {exc}")
            return 1
        ans = parse_noul(resp)
        print(f"[sample] {c.get('date')} {c.get('code')} {c.get('name')}: "
              f"+{TARGET_PCT}% 到達 {ans['prob']:.1f}%（モデル {ans['model']} / "
              f"入力 {ans['input_tokens']:,} トークン・出力 {ans['output_tokens']:,}）")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
