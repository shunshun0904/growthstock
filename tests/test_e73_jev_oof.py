#!/usr/bin/env python3
"""
実験73（research/exp/e73_jev_oof.py）: 重い計算と通信を回さない部分。

- 較正の帯: 帯ごとの件数・到達率・+10%指値。Brier は「母集団の到達率を常に答える」基準と比べられる
- ROC-AUC: 順位で数える。同点は 0.5。片方の群しか無ければ NaN。日付内の AUC は両方の群がある日だけ
- 腕: anon は銘柄・日付を伏せ、prod は本番と同じ。控えの鍵は腕を含む
- 候補の形: OOF の行を predictions.json の候補と同じ鍵にする（jev_predict.build_state が読める）

  python3 tests/test_e73_jev_oof.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e73_jev_oof as E  # noqa: E402
import jev_predict as JV  # noqa: E402


class TestCalibration(unittest.TestCase):
    def test_bins_count_hits_and_returns(self):
        prob = [5, 15, 15, 55, 95, 100]
        hit = [False, True, False, True, True, True]
        tp10 = [-2.0, 10.0, -1.0, 10.0, 10.0, 10.0]
        hold = [-2.0, 3.0, -1.0, 4.0, 5.0, 6.0]
        codes = ["a", "b", "c", "d", "e", "f"]
        tab = E.calib_table(prob, hit, tp10, hold, codes)
        self.assertEqual(len(tab), len(E.BINS) - 1)
        by = {(r["lo"], r["hi"]): r for r in tab}
        self.assertEqual(by[(10.0, 20.0)]["n"], 2)
        self.assertEqual(by[(10.0, 20.0)]["hit"], 50.0)
        self.assertEqual(by[(10.0, 20.0)]["tp10"], 4.5)
        self.assertEqual(by[(90.0, 100.0)]["n"], 2)          # 100 は最後の帯に入る
        self.assertEqual(by[(20.0, 30.0)]["n"], 0)
        self.assertNotIn("hit", by[(20.0, 30.0)])

    def test_brier(self):
        self.assertAlmostEqual(E.brier([1.0, 0.0], [True, False]), 0.0)
        self.assertAlmostEqual(E.brier([0.5, 0.5], [True, False]), 0.25)
        self.assertAlmostEqual(E.brier([0.2, 0.2], [True, False]), (0.64 + 0.04) / 2)


class TestAuc(unittest.TestCase):
    def test_roc_auc(self):
        self.assertEqual(E.roc_auc([True, False], [0.9, 0.1]), 1.0)
        self.assertEqual(E.roc_auc([True, False], [0.1, 0.9]), 0.0)
        self.assertEqual(E.roc_auc([True, False, True, False], [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertTrue(np.isnan(E.roc_auc([True, True], [0.1, 0.9])))
        self.assertAlmostEqual(E.roc_auc([True, False, False], [0.9, 0.5, np.nan]), 1.0)  # NaN は数えない

    def test_auc_in_day_weights_by_rows_and_skips_one_class_days(self):
        dates = ["d1"] * 2 + ["d2"] * 4 + ["d3"] * 2
        y = [True, False, True, True, False, False, True, True]
        s = [0.9, 0.1, 0.1, 0.2, 0.8, 0.9, 0.5, 0.6]
        a, days = E.auc_in_day(dates, y, s)
        self.assertEqual(days, 2)                                 # d3 は正例だけ
        self.assertAlmostEqual(a, (1.0 * 2 + 0.0 * 4) / 6)


class TestArmsAndCache(unittest.TestCase):
    def cand(self):
        return {"code": "1234", "jqCode": "12340", "name": "銘柄", "sector": "業種", "market": "プライム",
                "date": "2025-10-09", "rankInDay": 1, "nInDay": 9, "close": 100.0, "vol20d": 1.0,
                "byModel": {"lgbm": {"score": 0.5, "pctHistorical": 95.0}},
                "contrib": {"groups": {"地合い（市場環境）": 0.1}, "top": [], "marketContrib": 0.1}}

    def test_prod_matches_production_state_and_anon_hides_identity(self):
        c = self.cand()
        prod = E.arm_state(c, "追い風", "prod")
        self.assertEqual(prod, JV.build_state(c, "追い風"))
        anon = E.arm_state(c, "追い風", "anon")
        self.assertNotIn("銘柄", anon)
        self.assertNotIn("日付", anon)
        self.assertEqual(anon["株価"], prod["株価"])
        with self.assertRaises(ValueError):
            E.arm_state(c, None, "other")

    def test_cache_key_includes_arm_and_question(self):
        self.assertNotEqual(E.cache_key("prod", "12340", "2025-10-09"), E.cache_key("anon", "12340", "2025-10-09"))
        self.assertEqual(E.cache_key("prod", "12340", "2025-10-09")[3], JV.QUESTION_VERSION)

    def test_split_by_line(self):
        d = pd.DataFrame({"Code": list("abcdefgh"), "jev": [10, 20, 30, 40, 60, 70, 80, 90],
                          "tp10": [1, 2, 3, 4, 5, 6, 7, 8], "r": [1, 2, 3, 4, 5, 6, 7, 8],
                          "hit10": [False] * 4 + [True] * 4, "label": [0] * 4 + [1] * 4})
        sp = E.split_by_line(d, "jev", 50)
        self.assertEqual((sp["hi"]["n"], sp["lo"]["n"]), (4, 4))
        self.assertAlmostEqual(sp["diff_tp10"]["d"], 6.5 - 2.5)


class TestCandidateRows(unittest.TestCase):
    def test_rows_take_the_shape_of_predictions_json(self):
        d = pd.DataFrame({
            "Code": ["12340", "56780"], "Date": pd.to_datetime(["2025-10-09", "2025-10-09"]),
            "s_lgbm": [0.6, 0.3], "hp_lgbm": [95.0, 50.0], "s_xgb": [0.5, 0.2], "hp_xgb": [96.0, 40.0],
            "rank_in_day": [1, 2], "n_break": [2, 2], "vol_20d": [1.0, 2.0],
            "close_raw": [100.0, 200.0], "high52w": [101.0, np.nan], "per": [10.0, None],
        })
        contrib = {("12340", pd.Timestamp("2025-10-09")): {"groups": {"x": 1.0}, "top": [], "marketContrib": 0.0}}
        names = {"12340": {"CoName": "テスト", "S33Nm": "業種", "MktNm": "プライム", "ScaleCat": None}}
        prog = {("12340", "2025-10-09"): {"progressRate": 55.0, "quarter": 2, "progressBenchmark": 50.0,
                                         "progressBasis": "seasonal"}}
        rows = E.candidate_rows(d, ["lgbm", "xgb"], contrib, names, prog)
        self.assertEqual(len(rows), 2)
        a, b = rows
        self.assertEqual((a["code"], a["jqCode"], a["name"], a["date"]), ("1234", "12340", "テスト", "2025-10-09"))
        self.assertEqual(a["byModel"]["xgb"]["pctHistorical"], 96.0)
        self.assertEqual(a["contrib"]["groups"]["x"], 1.0)
        self.assertEqual(a["progressRate"], 55.0)
        self.assertEqual(a["close"], 100.0)
        self.assertAlmostEqual(a["needPct"], 1.2 * 1.0 * np.sqrt(20))   # 銘柄のボラの 1.2σ（20営業日）
        self.assertIsNone(b["name"])
        self.assertIsNone(b["contrib"])
        state = JV.build_state(a, "中立")
        self.assertEqual(state["銘柄"]["名前"], "テスト")
        self.assertEqual(state["機械学習モデルの見立て"]["XGBoost"], 96.0)
        self.assertNotIn("PER(倍)", JV.build_state(b).get("バリュエーション", {}))   # 全部欠測なら塊ごと無い


if __name__ == "__main__":
    unittest.main(verbosity=2)
