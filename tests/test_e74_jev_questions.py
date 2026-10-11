#!/usr/bin/env python3
"""
実験74（research/exp/e74_jev_questions.py）: 通信と重い計算を回さない部分。

- 問いは4つの noul（A +10%指値 / B +20%指値 / C モデルの正例 / R 実験73 と同じ）。R は本番の問いと同一
- 答え合わせの列: A ↔ tp10 > 0 / B ↔ tp20 > 0 / C ↔ 正例 / R ↔ hit10
- 応答の4つの答えをまとめて %。1つ欠けていれば JevError（部分的に控えない）
- 控えの鍵は腕と版を含む。偽の client で問うと、1候補1リクエストで4つの確率が控えに入る
- 集計が合成データで最後まで通る

  python3 tests/test_e74_jev_questions.py
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e74_jev_questions as E  # noqa: E402
import jev_predict as JV  # noqa: E402


class TestQuestions(unittest.TestCase):
    def test_four_nouls(self):
        self.assertEqual(tuple(E.QUESTIONS), E.KEYS)
        for k, q in E.QUESTIONS.items():
            self.assertEqual(q["type"], "noul", k)
            self.assertTrue(q["instructions"], k)
            self.assertIn("true", q["criteria"])
            self.assertIn("false", q["criteria"])
        self.assertIn("+10%", E.QUESTIONS["a_tp10"]["instructions"])
        self.assertIn("+20%", E.QUESTIONS["b_tp20"]["instructions"])
        self.assertIn("必要上昇率", E.QUESTIONS["c_label"]["instructions"])
        self.assertIn("5日移動平均", E.QUESTIONS["c_label"]["instructions"])
        self.assertEqual(E.QUESTIONS["r_rise10"], JV.QUESTIONS[JV.QUESTION_KEY])   # 本番の問いと同一
        self.assertEqual(E.TRUTH_OF, {"a_tp10": "y_a", "b_tp20": "y_b", "c_label": "y_c", "r_rise10": "y_r"})
        self.assertEqual(E.PRIMARY, "y_c")

    def test_truth_columns(self):
        d = pd.DataFrame({"tp10": [10.0, -2.0, 0.0], "tp20": [20.0, 1.0, -1.0], "label": [1, 0, np.nan],
                          "hit10": [True, False, False]})
        out = E.add_truths(d)
        self.assertEqual(list(out["y_a"]), [True, False, False])
        self.assertEqual(list(out["y_b"]), [True, True, False])
        self.assertEqual(list(out["y_c"]), [True, False, False])
        self.assertEqual(list(out["y_r"]), [True, False, False])
        with self.assertRaises(SystemExit):
            E.add_truths(d.drop(columns=["tp20"]))

    def test_parse_answers(self):
        resp = {"model": "jev-1.13.0", "usage": {"input_tokens": 2500, "output_tokens": 4},
                "answers": {k: {"type": "noul", "noul": p} for k, p in
                            zip(E.KEYS, (0.6, 0.55, 0.3, 0.7))}}
        a = E.parse_answers(resp)
        self.assertEqual((a["a_tp10"], a["b_tp20"], a["c_label"], a["r_rise10"]), (60.0, 55.0, 30.0, 70.0))
        self.assertEqual((a["model"], a["input_tokens"]), ("jev-1.13.0", 2500))
        del resp["answers"]["c_label"]
        with self.assertRaises(JV.JevError):
            E.parse_answers(resp)

    def test_cache_key_has_arm_and_qset(self):
        self.assertNotEqual(E.cache_key("prod", "1", "d"), E.cache_key("anon", "1", "d"))
        self.assertEqual(E.cache_key("prod", "1", "d")[3], E.QSET)


class TestAskAndReport(unittest.TestCase):
    ALGOS = ("lgbm", "xgb", "cat", "logit")

    def make_d(self, n=600, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.Timestamp("2025-01-06") + pd.to_timedelta(rng.integers(0, 250, n), unit="D")
        d = pd.DataFrame({"Code": [f"{1000 + i % 80}0" for i in range(n)], "Date": dates,
                          "r": rng.normal(1, 8, n), "n_break": 10, "rank_in_day": 1})
        hit10 = rng.random(n) < 0.35
        hit20 = hit10 & (rng.random(n) < 0.3)
        d["hit10"] = hit10
        d["tp10"] = np.where(hit10, 10.0, d["r"])
        d["tp20"] = np.where(hit20, 20.0, d["r"])
        d["label"] = (rng.random(n) < 0.2).astype(int)
        for a in self.ALGOS:
            d[f"hp_{a}"] = rng.uniform(0, 100, n)
        d.loc[:70, ["hp_lgbm", "hp_xgb", "hp_cat"]] = 96.0
        d["vol_20d"] = rng.uniform(0.5, 8, n)
        d["ret_20d"] = rng.normal(5, 10, n)
        d = E.add_truths(d)
        d["hp_min"] = d[["hp_lgbm", "hp_xgb", "hp_cat"]].min(axis=1)
        for arm in ("prod", "anon"):
            for q in E.KEYS:
                d[f"{arm}_{q}"] = np.clip(rng.normal(50, 20, n), 0, 100)
        d.loc[:3, "anon_a_tp10"] = np.nan
        return d

    def test_report_runs(self):
        d = self.make_d()
        usage = {a: {"asked": 10, "cached": 590, "failed": 0, "input_tokens": 25000, "output_tokens": 40,
                     "models": {"jev-1.13.0": 600}} for a in ("prod", "anon")}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = E.report(d, ["prod", "anon"], usage)
        out = buf.getvalue()
        for head in ("=== 1. 較正", "=== 2. 分離力", "=== 3. モデルが陽性と見た候補", "=== 4. 問い同士の相関",
                     "=== 5. prod と anon", "=== 6. 費用"):
            self.assertIn(head, out)
        self.assertEqual(set(res["calibration"]["prod"]), set(E.KEYS))
        self.assertIn("母集団", res["auc"])
        self.assertIn("prod_c_label", res["auc"]["母集団"])
        p = res["precision"]["prod"]["母集団"]
        self.assertEqual(set(p["gap"]), set(E.KEYS))
        self.assertIn("primary", p["gap"]["c_label"])
        self.assertEqual(len(p["terciles"]["b_tp20"]), 3)
        self.assertIn("50", p["gates"]["b_tp20"])
        self.assertEqual(set(res["arms"]), set(E.KEYS))
        import json
        json.dumps(res, ensure_ascii=False,
                   default=lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x))

    def test_ask_arm_asks_once_per_candidate_and_caches_four_answers(self):
        class FakeClient:
            model = "jev-latest"

            def __init__(self):
                self.calls = []

            def system_one(self, state, questions=None):
                self.calls.append((state, questions))
                return {"model": "jev-1.13.0",
                        "answers": {k: {"type": "noul", "noul": 0.1 * (i + 1)} for i, k in enumerate(E.KEYS)},
                        "usage": {"input_tokens": 2500, "output_tokens": 4}}

        cands = [{"code": f"{1000 + i}", "jqCode": f"{1000 + i}0", "name": f"銘柄{i}", "date": "2025-10-09",
                  "rankInDay": i + 1, "nInDay": 2, "close": 100.0, "vol20d": 1.0,
                  "byModel": {"lgbm": {"score": 0.5, "pctHistorical": 90.0}},
                  "contrib": {"groups": {}, "top": [], "marketContrib": 0.0}} for i in range(2)]
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "e74.parquet")
            client = FakeClient()
            with contextlib.redirect_stdout(io.StringIO()):
                probs, usage, cache = E.ask_arm("prod", cands, {"2025-10-09": "中立"}, client,
                                                pd.DataFrame(columns=E.CACHE_COLS), workers=2, path=path)
            self.assertEqual(len(client.calls), 2)                       # 1候補1リクエスト
            self.assertIs(client.calls[0][1], E.QUESTIONS)               # 4つの問いをまとめて渡す
            self.assertEqual(usage["asked"], 2)
            self.assertEqual(probs[("10000", "2025-10-09")],
                             {"a_tp10": 10.0, "b_tp20": 20.0, "c_label": 30.0, "r_rise10": 40.0})
            self.assertEqual(len(cache), 2)
            self.assertEqual(set(cache["qset"]), {E.QSET})
            # 2回目は控えから（問わない）
            client2 = FakeClient()
            with contextlib.redirect_stdout(io.StringIO()):
                probs2, usage2, _ = E.ask_arm("prod", cands, {"2025-10-09": "中立"}, client2,
                                              E.load_cache(path), workers=1, path=path)
            self.assertEqual(len(client2.calls), 0)
            self.assertEqual((usage2["asked"], usage2["cached"]), (0, 2))
            self.assertEqual(probs2, probs)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
