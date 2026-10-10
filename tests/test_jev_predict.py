#!/usr/bin/env python3
"""
Jev の呼び出し（research/jev_predict.py）の単体テスト。通信はしない（opener を差し替える）。

守りたいこと
  - state は画面の値・4モデルの百分位・寄与を日本語の鍵で持ち、欠測は書かない（0 で埋めない）
  - 問いは noul 1つ。HTTP の形は公式 SDK（typesafe-sdk 0.7.4 の OpenAPI の写し）と同じ
  - 答えは (日付, コード, 問いの版) で凍結: 控えにあれば問い直さない（課金しない）
  - 1銘柄の失敗で他を止めない。続けて失敗したら残りを問わない。上限を超えたら問わない
  - 鍵が無ければ何もしない（行は jev=None、要約は enabled=False）
  - 鍵をログ・例外・repr に出さない

  python3 tests/test_jev_predict.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402

import jev_predict as JV  # noqa: E402

KEY = "test-key-0123456789abcdef"


def cand(code="1234", date="2026-10-09", rank=1, **over):
    c = {
        "code": code, "jqCode": f"{code}0", "name": f"銘柄{code}", "sector": "情報・通信業",
        "market": "プライム", "scale": "TOPIX Small 1", "date": date,
        "score": 0.6, "rankInDay": rank, "nInDay": 12, "pctHistorical": 95.2,
        "close": 7230.0, "high52w": 7297.0, "rHigh": 99.08, "breakMargin": 0.17,
        "baseLength": 46.0, "ret20d": 8.56, "vol20d": 1.28, "needPct": 6.9,
        "marketCap": 128364.29, "tradingValue": 278.04, "creditRatio": 12.7,
        "per": 20.63, "pbr": 2.92, "divYield": 3.76,
        "epsGrowth": 44.54, "salesGrowth": 17.03, "roe": 13.97, "opMargin": 32.05,
        "volumeTrend": 219.31, "progressRate": 63.98, "quarter": 2,
        "progressBenchmark": 55.35, "progressBasis": "seasonal",
        "contrib": {
            "base": -0.45, "marketContrib": 0.195, "marketShare": 10.0, "stockContrib": 1.0,
            "top": [{"col": "days_to_earn", "ja": "次の決算まで", "contrib": 0.2684},
                    {"col": "vol_20d", "ja": "", "contrib": 0.1496}],
            "groups": {"ボラの分解": 0.3335, "地合い（市場環境）": 0.1957, "決算（変化）": -0.0636},
        },
        "byModel": {"lgbm": {"score": 0.61, "pctHistorical": 95.2},
                    "xgb": {"score": 0.66, "pctHistorical": 97.8},
                    "cat": {"score": 0.59, "pctHistorical": 95.2},
                    "logit": {"score": 0.55, "pctHistorical": 80.6}},
    }
    c.update(over)
    return c


class FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body, ensure_ascii=False).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """urllib.request.urlopen の代わり。送った要求を残し、決めた応答を順に返す。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        if not self.replies:
            raise AssertionError("想定より多く呼ばれた")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return FakeResponse(r)


def http_error(code, body=b"", headers=None):
    import email.message
    h = email.message.Message()
    for k, v in (headers or {}).items():
        h[k] = v
    return urllib.error.HTTPError("https://api.typesafe.ai/v1/systemone", code, "err", h, io.BytesIO(body))


def ok(prob=0.375, model="jev-1.13.0", tokens=900):
    return {"model": model, "answers": {JV.QUESTION_KEY: {"type": "noul", "noul": prob}},
            "usage": {"input_tokens": tokens, "output_tokens": 1}}


def client(replies, **kw):
    sleeps = []
    c = JV.Client(api_key=KEY, base_url="https://api.typesafe.ai", model="jev-latest",
                  opener=FakeOpener(replies), sleep=sleeps.append, **kw)
    return c, c._open, sleeps


class TestState(unittest.TestCase):
    def test_state_has_values_models_and_contributions(self):
        s = JV.build_state(cand(), tone="追い風")
        self.assertEqual(s["銘柄"]["コード"], "1234")
        self.assertEqual(s["株価"]["終値(円)"], 7230.0)
        self.assertEqual(s["決算"]["四半期"], 2)
        self.assertEqual(s["その日"]["78週高値を更新した銘柄数(発火数)"], 12)
        self.assertEqual(s["その日"]["地合い(候補全体の市場環境の寄与)"], "追い風")
        m = s["機械学習モデルの見立て"]
        self.assertEqual((m["LightGBM"], m["XGBoost"], m["CatBoost"], m["ロジスティック回帰"]),
                         (95.2, 97.8, 95.2, 80.6))
        g = s["基準モデル(LightGBM)の寄与"]
        self.assertEqual(g["区分ごとの寄与"]["ボラの分解"], 0.334)
        self.assertEqual(g["効いた特徴量"][0]["特徴量"], "days_to_earn")
        self.assertNotIn("意味", g["効いた特徴量"][1])          # 空の説明は書かない
        json.dumps(s, ensure_ascii=False)                       # JSON に出せる

    def test_missing_values_are_omitted_not_zero(self):
        c = cand(per=None, pbr=float("nan"), creditRatio=np.nan, quarter=None,
                 progressRate=None, progressBenchmark=None, byModel=None, contrib=None)
        s = JV.build_state(c)
        self.assertNotIn("PER(倍)", s["バリュエーション"])
        self.assertNotIn("PBR(倍)", s["バリュエーション"])
        self.assertNotIn("信用倍率(倍)", s["需給・規模"])
        self.assertNotIn("四半期", s["決算"])
        self.assertNotIn("LightGBM", s["機械学習モデルの見立て"])
        self.assertNotIn("区分ごとの寄与", s["基準モデル(LightGBM)の寄与"])
        self.assertNotIn("地合い(候補全体の市場環境の寄与)", s["その日"])
        text = json.dumps(s, ensure_ascii=False)
        self.assertNotIn("NaN", text)
        self.assertNotIn("null", text)

    def test_anonymous_hides_identity_and_date(self):
        s = JV.build_state(cand(), anonymous=True)
        self.assertNotIn("銘柄", s)
        self.assertNotIn("日付", s)
        text = json.dumps(s, ensure_ascii=False)
        self.assertNotIn("1234", text)
        self.assertNotIn("銘柄1234", text)
        self.assertNotIn("2026-10-09", text)
        self.assertIn("終値(円)", text)                           # 数字は残す

    def test_numpy_values_are_plain_floats(self):
        s = JV.build_state(cand(close=np.float64(100.5), nInDay=np.int64(7), baseLength=np.float32(3)))
        self.assertIsInstance(s["株価"]["終値(円)"], float)
        self.assertIsInstance(s["その日"]["78週高値を更新した銘柄数(発火数)"], int)
        self.assertEqual(s["株価"]["ベースの長さ(前の高値からの営業日数)"], 3)

    def test_market_tone_uses_median_like_the_screen(self):
        rows = [cand(contrib={"marketContrib": v}) for v in (0.3, -0.2, 0.1)]
        self.assertEqual(JV.market_tone(rows), "追い風")        # 中央値 0.1 > 0.05
        rows = [cand(contrib={"marketContrib": v}) for v in (-0.3, 0.04, 0.1)]
        self.assertEqual(JV.market_tone(rows), "中立")
        self.assertIsNone(JV.market_tone([cand(contrib={})]))

    def test_state_sha_is_stable(self):
        a = JV.state_sha(JV.build_state(cand()))
        b = JV.state_sha(JV.build_state(cand()))
        c = JV.state_sha(JV.build_state(cand(close=1.0)))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 12)


class TestQuestion(unittest.TestCase):
    def test_single_noul_question(self):
        self.assertEqual(list(JV.QUESTIONS), [JV.QUESTION_KEY])
        q = JV.QUESTIONS[JV.QUESTION_KEY]
        self.assertEqual(q["type"], "noul")
        self.assertIn("+10%", q["instructions"])
        self.assertIn("20営業日", q["instructions"])
        self.assertIn("1.10倍", q["criteria"]["true"])
        self.assertIn("1.10倍", q["criteria"]["false"])
        self.assertEqual((JV.TARGET_PCT, JV.HOLD_DAYS), (10, 20))


class TestClient(unittest.TestCase):
    def test_request_shape_matches_the_sdk(self):
        c, opener, _ = client([ok()])
        state = JV.build_state(cand())
        resp = c.system_one(state)
        self.assertEqual(resp["answers"][JV.QUESTION_KEY]["noul"], 0.375)
        req = opener.requests[0]
        self.assertEqual(req.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), f"Bearer {KEY}")
        self.assertEqual(req.get_header("Content-type"), "application/json")
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(sorted(body), ["model", "questions", "state"])
        self.assertEqual(body["model"], "jev-latest")
        self.assertEqual(body["state"], state)
        self.assertEqual(body["questions"], JV.QUESTIONS)

    def test_parse_noul_to_percent(self):
        a = JV.parse_noul(ok(0.375, "jev-1.13.0", 900))
        self.assertEqual(a, {"prob": 37.5, "model": "jev-1.13.0", "input_tokens": 900, "output_tokens": 1})
        for bad in ({}, {"answers": {}}, {"answers": {JV.QUESTION_KEY: {"type": "choice", "choice": "a"}}},
                    {"answers": {JV.QUESTION_KEY: {"type": "noul", "noul": 1.5}}},
                    {"answers": {JV.QUESTION_KEY: {"type": "noul", "noul": "x"}}}):
            with self.assertRaises(JV.JevError):
                JV.parse_noul(bad)

    def test_retries_on_429_and_5xx_then_succeeds(self):
        c, opener, sleeps = client([http_error(429, headers={"retry-after": "7"}),
                                    http_error(503), ok(0.5)])
        self.assertEqual(c.system_one({"x": 1})["answers"][JV.QUESTION_KEY]["noul"], 0.5)
        self.assertEqual(len(opener.requests), 3)
        self.assertEqual(sleeps, [7.0, 4.0])      # retry-after を優先、次は 2×2

    def test_gives_up_after_tries(self):
        c, opener, sleeps = client([http_error(500, b"boom")] * 4)
        with self.assertRaises(JV.JevError) as cm:
            c.system_one({"x": 1})
        self.assertIn("HTTP 500", str(cm.exception))
        self.assertEqual(len(opener.requests), 4)
        self.assertEqual(sleeps, [2.0, 4.0, 8.0])

    def test_client_errors_are_not_retried(self):
        c, opener, _ = client([http_error(401, b"bad key")])
        with self.assertRaises(JV.JevError):
            c.system_one({"x": 1})
        self.assertEqual(len(opener.requests), 1)

    def test_connection_errors_are_retried(self):
        c, opener, _ = client([urllib.error.URLError("dns"), TimeoutError(), ok(0.2)])
        self.assertEqual(JV.parse_noul(c.system_one({"x": 1}))["prob"], 20.0)
        self.assertEqual(len(opener.requests), 3)

    def test_key_never_leaks(self):
        c, _, _ = client([http_error(500, b"x")] * 4)
        self.assertNotIn(KEY, repr(c))
        try:
            c.system_one({"x": 1})
        except JV.JevError as exc:
            self.assertNotIn(KEY, str(exc))
        with self.assertRaises(JV.JevError):
            JV.Client(api_key="", opener=lambda *a, **k: None)
        with self.assertRaises(JV.JevError):
            JV.Client(api_key="bad key", opener=lambda *a, **k: None)

    def test_models_endpoint(self):
        c, opener, _ = client([{"models": [{"name": "jev-latest", "description": "d", "release_date": "2026-09-15"}]}])
        self.assertEqual([m["name"] for m in c.models()], ["jev-latest"])
        self.assertEqual(opener.requests[0].full_url, "https://api.typesafe.ai/v1/models")
        self.assertEqual(opener.requests[0].get_method(), "GET")


class TestAnnotate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.logs = []

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def log(self, m):
        self.logs.append(m)

    def test_disabled_without_key(self):
        os.environ.pop(JV.API_KEY_ENV, None)
        rows = [cand()]
        s = JV.annotate(rows, self.dir, log=self.log)
        self.assertFalse(s["enabled"])
        self.assertIsNone(rows[0]["jev"])
        self.assertIn(JV.API_KEY_ENV, s["note"])
        self.assertFalse(os.path.exists(os.path.join(self.dir, JV.ANSWERS_FILE)))
        self.assertEqual(s["question"], JV.QUESTIONS[JV.QUESTION_KEY])

    def test_asks_newest_first_and_writes_cache(self):
        rows = [cand("1111", "2026-10-08", 1), cand("2222", "2026-10-09", 2), cand("3333", "2026-10-09", 1)]
        c, opener, _ = client([ok(0.3, "jev-1.13.0", 800), ok(0.6, "jev-1.13.0", 850), ok(0.1, "jev-1.13.0", 700)])
        s = JV.annotate(rows, self.dir, client=c, now="2026-10-09T10:00:00+00:00", log=self.log)
        self.assertTrue(s["enabled"])
        self.assertEqual((s["asked"], s["cached"], s["failed"], s["skipped"]), (3, 0, 0, 0))
        self.assertEqual(s["inputTokens"], 2350)
        self.assertEqual(s["resolvedModel"], "jev-1.13.0")
        # 新しい日の1位から問う: 3333（10-09 1位）→ 2222（10-09 2位）→ 1111（10-08）
        asked = [json.loads(r.data)["state"]["銘柄"]["コード"] for r in opener.requests]
        self.assertEqual(asked, ["3333", "2222", "1111"])
        by = {r["code"]: r["jev"] for r in rows}
        self.assertEqual(by["3333"]["prob"], 30.0)
        self.assertEqual(by["2222"]["prob"], 60.0)
        self.assertEqual(by["1111"]["prob"], 10.0)
        self.assertEqual(by["3333"]["model"], "jev-1.13.0")
        self.assertEqual(by["3333"]["question"], JV.QUESTION_VERSION)
        self.assertFalse(by["3333"]["cached"])
        df = JV.load_answers(os.path.join(self.dir, JV.ANSWERS_FILE))
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["code"]), {"11110", "22220", "33330"})
        self.assertEqual(list(df.columns), JV.ANSWER_COLS)

    def test_cached_answers_are_frozen_and_not_asked_again(self):
        rows = [cand("1111", "2026-10-09", 1)]
        c, opener, _ = client([ok(0.3)])
        JV.annotate(rows, self.dir, client=c, log=self.log)
        # 2回目: 値が変わっていても（state が違っても）問い直さない
        rows2 = [cand("1111", "2026-10-09", 1, close=1.0), cand("4444", "2026-10-09", 2)]
        c2, opener2, _ = client([ok(0.9)])
        s = JV.annotate(rows2, self.dir, client=c2, log=self.log)
        self.assertEqual((s["asked"], s["cached"]), (1, 1))
        self.assertEqual(len(opener2.requests), 1)
        self.assertEqual(json.loads(opener2.requests[0].data)["state"]["銘柄"]["コード"], "4444")
        self.assertEqual(rows2[0]["jev"]["prob"], 30.0)
        self.assertTrue(rows2[0]["jev"]["cached"])
        self.assertEqual(rows2[1]["jev"]["prob"], 90.0)
        df = JV.load_answers(os.path.join(self.dir, JV.ANSWERS_FILE))
        self.assertEqual(len(df), 2)

    def test_one_failure_does_not_stop_the_others(self):
        rows = [cand("1111", "2026-10-09", 1), cand("2222", "2026-10-09", 2)]
        c, opener, _ = client([http_error(400, b"bad request"), ok(0.4)])
        s = JV.annotate(rows, self.dir, client=c, log=self.log)
        self.assertEqual((s["asked"], s["failed"]), (1, 1))
        self.assertIsNone(rows[0]["jev"])
        self.assertEqual(rows[1]["jev"]["prob"], 40.0)
        self.assertIn("失敗", s["note"])

    def test_consecutive_failures_halt_the_run(self):
        rows = [cand(f"{i}{i}{i}{i}", "2026-10-09", i) for i in range(1, 7)]
        c, opener, _ = client([http_error(422, b"x")] * JV.MAX_CONSECUTIVE_FAILURES)
        s = JV.annotate(rows, self.dir, client=c, log=self.log)
        self.assertEqual(s["failed"], JV.MAX_CONSECUTIVE_FAILURES)
        self.assertEqual(s["skipped"], 6 - JV.MAX_CONSECUTIVE_FAILURES)
        self.assertEqual(len(opener.requests), JV.MAX_CONSECUTIVE_FAILURES)
        self.assertTrue(all(r["jev"] is None for r in rows))
        self.assertIn("続けて失敗", s["note"])

    def test_max_per_run_caps_requests(self):
        rows = [cand(f"{i}{i}{i}{i}", "2026-10-09", i) for i in range(1, 5)]
        c, opener, _ = client([ok(0.5), ok(0.5)])
        s = JV.annotate(rows, self.dir, client=c, max_per_run=2, log=self.log)
        self.assertEqual((s["asked"], s["skipped"]), (2, 2))
        self.assertEqual(len(opener.requests), 2)
        self.assertIsNotNone(rows[0]["jev"])
        self.assertIsNone(rows[3]["jev"])
        self.assertIn("上限", s["note"])

    def test_unreadable_cache_is_rebuilt(self):
        path = os.path.join(self.dir, JV.ANSWERS_FILE)
        with open(path, "wb") as fh:
            fh.write(b"not parquet")
        rows = [cand("1111", "2026-10-09", 1)]
        c, _, _ = client([ok(0.3)])
        s = JV.annotate(rows, self.dir, client=c, log=self.log)
        self.assertEqual(s["asked"], 1)
        self.assertEqual(len(JV.load_answers(path)), 1)
        self.assertTrue(any("控えを読めない" in m for m in self.logs))

    def test_old_rows_are_dropped_from_the_cache(self):
        import pandas as pd
        old = pd.DataFrame([{"date": "2024-01-01", "code": "99990", "question": JV.QUESTION_VERSION,
                             "model": "m", "prob": 1.0, "input_tokens": 1, "output_tokens": 1,
                             "asked_at": "x", "state_sha": "y"}])
        path = os.path.join(self.dir, JV.ANSWERS_FILE)
        JV.save_answers(old, path)
        rows = [cand("1111", "2026-10-09", 1)]
        c, _, _ = client([ok(0.3)])
        JV.annotate(rows, self.dir, client=c, log=self.log)
        df = JV.load_answers(path)
        self.assertEqual(list(df["code"]), ["11110"])      # 400日より古い行は落ちる

    def test_describe_mentions_counts(self):
        s = JV.summary_skeleton()
        self.assertIn("問うていない", JV.describe(s))
        s.update({"enabled": True, "asked": 3, "cached": 2, "failed": 0, "skipped": 0,
                  "inputTokens": 2500, "resolvedModel": "jev-1.13.0"})
        self.assertIn("問うた 3件", JV.describe(s))
        self.assertIn("jev-1.13.0", JV.describe(s))


class TestPredictDailyWiring(unittest.TestCase):
    def test_predict_daily_calls_annotate_after_bymodel_and_records_prob(self):
        import inspect
        import predict_daily as PD
        src = inspect.getsource(PD.main)
        self.assertIn("JV.annotate(rows, args.data_dir)", src)
        self.assertLess(src.index('x["nModels"] = len(pcts)'), src.index("JV.annotate("))
        self.assertLess(src.index("JV.annotate("), src.index("rows.sort(key="))
        self.assertIn('"jev": jev', src)
        self.assertIn('"jevProb": (x.get("jev") or {}).get("prob")', inspect.getsource(PD.update_history))

    def test_predict_workflow_passes_the_key_and_uploads_the_cache(self):
        with open(os.path.join(ROOT, ".github", "workflows", "predict.yml"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("TYPESAFE_API_KEY: ${{ secrets.TYPESAFE_API_KEY }}", text)
        self.assertIn("research/_data/jev_answers.parquet", text)
        self.assertIn('bash scripts/gh_release_upload.sh data-raw "$f"', text)
        with open(os.path.join(ROOT, ".github", "workflows", "run-experiment.yml"), encoding="utf-8") as fh:
            self.assertIn("TYPESAFE_API_KEY: ${{ secrets.TYPESAFE_API_KEY }}", fh.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
