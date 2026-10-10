#!/usr/bin/env python3
"""
実験69（research/exp/e69_forward_check.py）: 先に決めた仮説を、決めたあとのデータだけで確かめ直す。

- 仮説（列・向き・行・窓・判定の線）が決めたとおりであること（あとから動かしたら落ちる）
- 判定: 決めた向きでの片側 t 検定、隣どうしの相関で SE を広げる、窓が足りなければ「測れない」
- 窓: 日証金は 2026-11 〜 2027-10 の暦月12本。結果がそろっていない月は使わない
- EDINET: 2026-10-10 に取れていた 1,670社と数が合わなければ止める。新しい銘柄の行だけを使う
- 実データの代わりに作った表で、EDINET・日証金とも最後まで回ること（12月・2027年に初歩的な誤りで落ちない）

  python3 tests/test_e69_forward_check.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import edinet_features as EF  # noqa: E402
import jsf_features as JF  # noqa: E402
import e69_forward_check as X  # noqa: E402
from scipy import stats  # noqa: E402


class TestHypotheses(unittest.TestCase):
    def test_fixed_as_decided(self):
        self.assertEqual(X.DECIDED, "2026-10-10")
        self.assertEqual(X.ALPHA, 0.05)
        ed, js = X.HYPOTHESES["edinet"], X.HYPOTHESES["jsf"]
        self.assertEqual(ed["features"], {"ed_avg_annual_salary_yoy1": +1, "ed_sga_chg2y": -1})
        self.assertEqual(ed["reference"], {})
        self.assertEqual(ed["new_since"], "2026-10-11T00:00:00+09:00")
        self.assertEqual(ed["old_companies"], 1670)
        self.assertEqual((ed["min_windows"], ed["run_after"]), (8, "2026-12-11"))
        self.assertEqual(js["features"], {"jsf_loan_chg20_v": +1, "jsf_fee_days20": -1})
        self.assertEqual(js["reference"], {"jsf_fee_max20": -1})
        self.assertEqual((js["start"], js["end"], js["min_windows"], js["run_after"]),
                         ("2026-11-01", "2027-10-31", 9, "2027-12-06"))

    def test_columns_exist_and_origin_signs_match(self):
        ed, js = X.HYPOTHESES["edinet"], X.HYPOTHESES["jsf"]
        self.assertTrue(set(ed["features"]) <= set(EF.columns("all")))
        self.assertTrue(set(js["features"]) | set(js["reference"]) <= set(JF.columns("all")))
        for h in (ed, js):
            for f, sign in {**h["features"], **h["reference"]}.items():
                edge, z, _ = h["origin"][f]
                self.assertEqual(np.sign(edge), sign)                 # 決めた向き = 10/10 に見た向き
                self.assertEqual(np.sign(z), sign)


class TestVerdict(unittest.TestCase):
    def test_pass_weak_reverse_short(self):
        rng = np.random.default_rng(0)
        strong = 1.0 + rng.normal(0, 0.5, 12)
        self.assertEqual(X.verdict(strong, +1, 9)["label"], "pass")
        self.assertEqual(X.verdict(-strong, -1, 9)["label"], "pass")         # 低いほど良い
        self.assertEqual(X.verdict(-strong, +1, 9)["label"], "reverse")
        weak = np.array([0.3, -0.2, 0.5, -0.4, 0.2, 0.1, -0.3, 0.4, 0.0, 0.2, -0.1, 0.1])
        v = X.verdict(weak, +1, 9)
        self.assertGreater(v["mean"], 0)
        self.assertEqual(v["label"], "weak")
        self.assertEqual(X.verdict(strong[:8], +1, 9)["label"], "short")      # 窓が足りない
        self.assertEqual(X.verdict([np.nan, 1.0], +1, 1)["label"], "short")   # NaN は数えない
        self.assertEqual(X.verdict([], +1, 1)["n"], 0)

    def test_t_and_p_match_scipy_when_no_autocorrelation(self):
        e = np.array([0.8, -0.1, 0.5, 0.9, -0.4, 0.6, 0.2, 1.1, -0.2, 0.4, 0.7])
        self.assertLessEqual(X.lag1_autocorr(e), 0)                     # 負なら広げない
        v = X.verdict(e, +1, 8)
        t, p2 = stats.ttest_1samp(e, 0.0)
        self.assertAlmostEqual(v["t"], t, places=10)
        self.assertAlmostEqual(v["p"], p2 / 2, places=10)                 # 片側
        self.assertEqual(v["agree"], 8)

    def test_positive_autocorrelation_widens_se(self):
        e = np.array([0.2, 0.4, 0.7, 0.9, 1.0, 0.8, 0.5, 0.3, 0.4, 0.6, 0.9, 1.1])
        rho = X.lag1_autocorr(e)
        self.assertGreater(rho, 0.3)
        v = X.verdict(e, +1, 9)
        self.assertGreater(v["se_adj"], v["se"] * 1.3)
        t_plain = stats.ttest_1samp(e, 0.0)[0]
        self.assertLess(v["t"], t_plain)

    def test_se_inflation(self):
        self.assertEqual(X.se_inflation(0.0, 12), 1.0)
        self.assertEqual(X.se_inflation(-0.5, 12), 1.0)                   # 負は 0 に
        self.assertAlmostEqual(X.se_inflation(0.5, 2), 1 + 2 * 0.5 * 0.5)
        n, r = 12, 0.3
        self.assertAlmostEqual(X.se_inflation(r, n), 1 + 2 * sum((1 - k / n) * r ** k for k in range(1, n)))
        self.assertAlmostEqual(X.se_inflation(5.0, 12), X.se_inflation(0.9, 12))   # 0.9 で止める


class TestWindows(unittest.TestCase):
    def test_twelve_calendar_months(self):
        w = X.month_windows("2026-11-01", "2027-10-31")
        self.assertEqual(len(w), 12)
        self.assertEqual((pd.Timestamp(w[0][0]), pd.Timestamp(w[0][1])),
                         (pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-30")))
        self.assertEqual((pd.Timestamp(w[-1][0]), pd.Timestamp(w[-1][1])),
                         (pd.Timestamp("2027-10-01"), pd.Timestamp("2027-10-31")))
        for (s0, e0), (s1, _) in zip(w, w[1:]):                            # 隙間も重なりも無い
            self.assertEqual(pd.Timestamp(s1) - pd.Timestamp(e0), pd.Timedelta(days=1))
        self.assertEqual(pd.Timestamp(w[3][1]), pd.Timestamp("2027-02-28"))

    def test_complete_only_when_later_outcomes_exist(self):
        w = X.month_windows("2026-11-01", "2027-10-31")
        self.assertEqual(sum(X.complete_windows(w, pd.Timestamp("2027-10-29"))), 11)
        self.assertEqual(sum(X.complete_windows(w, pd.Timestamp("2027-11-01"))), 12)
        self.assertEqual(sum(X.complete_windows(w, pd.Timestamp("2027-05-31"))), 6)
        self.assertEqual(X.complete_windows(w, None), [False] * 12)

    def test_last_labeled(self):
        f = pd.DataFrame({"Date": pd.to_datetime(["2027-01-04", "2027-01-05", "2027-01-06"]),
                          "label": [1.0, 0.0, np.nan], X.OUTCOME: [0.1, np.nan, 0.2]})
        self.assertEqual(X.last_labeled(f), pd.Timestamp("2027-01-04"))


def synthetic(dates, codes, feats, seed=0, effect=None):
    """日付×銘柄の表。effect = {列: 係数} の列ほど ret_o1_20 が上がる（負なら下がる）。"""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([dates, codes], names=["Date", "Code"]).to_frame(index=False)
    n = len(idx)
    for f in feats:
        idx[f] = rng.normal(0, 1, n)
    r = rng.normal(0, 0.08, n)
    for f, b in (effect or {}).items():
        r = r + b * idx[f].to_numpy()
    idx[X.OUTCOME] = r
    idx["label"] = (r > 0.05).astype(float)
    return idx


class TestMeasure(unittest.TestCase):
    def test_planted_signal_passes_and_noise_does_not(self):
        dates = pd.bdate_range("2026-11-02", "2027-10-29")
        codes = [f"{i:04d}0" for i in range(25)]
        df = synthetic(dates, codes, ["a", "b", "c"], effect={"a": 0.02, "b": -0.02})
        res = X.measure(df, {"a": +1, "b": -1, "c": +1}, X.month_windows("2026-11-01", "2027-10-31"), 9)
        self.assertEqual(res["a"]["label"], "pass")
        self.assertEqual(res["b"]["label"], "pass")
        self.assertIn(res["c"]["label"], ("weak", "reverse"))
        self.assertEqual(res["a"]["n"], 12)
        self.assertEqual(len(res["a"]["windows"]), 12)
        flipped = X.measure(df, {"a": -1}, X.month_windows("2026-11-01", "2027-10-31"), 9)
        self.assertEqual(flipped["a"]["label"], "reverse")

    def test_too_few_rows_is_short(self):
        df = synthetic(pd.bdate_range("2026-11-02", periods=10), ["10000", "20000"], ["a"])
        res = X.measure(df, {"a": +1}, X.month_windows("2026-11-01", "2026-11-30"), 1)
        self.assertEqual(res["a"]["label"], "short")
        self.assertEqual(res["a"]["windows"], [])


class TestEdinetRows(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"Code": ["10000", "20000", "30000", "30000"],
                                "ed_fiscal_year": [2024.0, 2024.0, 2024.0, np.nan]})

    def test_only_new_codes_with_y0(self):
        rows = X.edinet_rows(self.df, {"30000"}, {"10000", "20000"}, 2)
        self.assertEqual(list(rows), [False, False, True, False])

    def test_stops_when_old_count_differs_or_sets_overlap(self):
        with self.assertRaises(SystemExit):
            X.edinet_rows(self.df, {"30000"}, {"10000"}, 2)
        with self.assertRaises(SystemExit):
            X.edinet_rows(self.df, {"30000", "10000"}, {"10000", "20000"}, 2)


class TestEndToEnd(unittest.TestCase):
    """実データの代わりに作った表で最後まで回す（表示・記録まで）。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved_out = X.OUT_DIR
        X.OUT_DIR = self.dir

    def tearDown(self):
        X.OUT_DIR = self.saved_out
        shutil.rmtree(self.dir, ignore_errors=True)

    def _quiet(self):
        return mock.patch("builtins.print")

    def test_jsf_interim_then_final(self):
        h = X.HYPOTHESES["jsf"]
        feats = list(h["features"]) + list(h["reference"]) + ["jsf_ratio"]
        codes = [f"{i:04d}0" for i in range(25)]
        for end, final in (("2027-05-31", False), ("2027-11-30", True)):
            frame = synthetic(pd.bdate_range("2026-09-01", end), codes, feats, seed=1,
                              effect={"jsf_loan_chg20_v": 0.02})
            with mock.patch.object(X.lab, "frame", return_value=frame[["Date", "Code", "label", X.OUTCOME]].copy()), \
                    mock.patch.object(JF, "build", return_value=frame), self._quiet():
                X.main(["--source", "jsf"])
            with open(os.path.join(self.dir, "e69_jsf.json"), encoding="utf-8") as fh:
                out = json.load(fh)
            self.assertEqual(out["final"], final)
            self.assertEqual(out["windows_complete"], 12 if final else 6)
            # 9/1〜10/31 の行は使わない（決めた期間の外）
            self.assertEqual(out["rows"], int(((frame["Date"] >= "2026-11-01")
                                               & (frame["Date"] <= "2027-10-31")).sum()))
            if final:
                self.assertEqual(out["result"]["jsf_loan_chg20_v"]["label"], "pass")

    def _edinet_env(self, n_old, effect):
        codes_old = [f"{i:04d}0" for i in range(n_old)]
        codes_new = [f"{i:04d}0" for i in range(5000, 5030)]
        frame = synthetic(pd.bdate_range("2018-01-04", "2026-11-13"), codes_old[:5] + codes_new,
                          list(X.HYPOTHESES["edinet"]["features"]), seed=2, effect=effect)
        frame["ed_fiscal_year"] = 2024.0
        comp = {f"E{i}": {"status": "ok", "jq": c, "at": "2026-09-21T06:00:00+00:00"}
                for i, c in enumerate(codes_old)}
        comp.update({f"N{i}": {"status": "ok", "jq": c, "at": "2026-11-01T01:00:00+00:00",
                               "first_ok_at": "2026-11-01T01:00:00+00:00"} for i, c in enumerate(codes_new)})
        fin = os.path.join(self.dir, "edinet_fin.parquet")
        pd.DataFrame({"x": [1]}).to_parquet(fin)
        with open(os.path.join(self.dir, "edinet_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"companies": comp}, fh)
        return frame, fin, set(codes_new)

    def test_edinet_runs_on_new_companies_only(self):
        frame, fin, new = self._edinet_env(1670, {"ed_avg_annual_salary_yoy1": 0.02})
        patches = [mock.patch.object(X.lab, "frame", return_value=frame[["Date", "Code", "label", X.OUTCOME]].copy()),
                   mock.patch.object(EF, "load_fin"), mock.patch.object(EF, "annual_panel"),
                   mock.patch.object(EF, "feature_frame"), mock.patch.object(EF, "attach", return_value=frame)]
        for p in patches:
            p.start()
        try:
            with self._quiet():
                X.main(["--source", "edinet", "--fin", fin, "--no-family"])
        finally:
            for p in patches:
                p.stop()
        with open(os.path.join(self.dir, "e69_edinet.json"), encoding="utf-8") as fh:
            out = json.load(fh)
        self.assertEqual((out["old_companies"], out["new_companies"]), (1670, 30))
        self.assertEqual(out["rows"], int(frame["Code"].isin(new).sum()))
        self.assertEqual(out["result"]["ed_avg_annual_salary_yoy1"]["label"], "pass")
        self.assertGreaterEqual(out["result"]["ed_avg_annual_salary_yoy1"]["n"], 8)

    def test_edinet_stops_when_the_record_is_short(self):
        frame, fin, _ = self._edinet_env(1600, {})
        with mock.patch.object(X.lab, "frame", return_value=frame), \
                mock.patch.object(EF, "load_fin"), mock.patch.object(EF, "annual_panel"), \
                mock.patch.object(EF, "feature_frame"), mock.patch.object(EF, "attach", return_value=frame), \
                self._quiet():
            with self.assertRaises(SystemExit):
                X.main(["--source", "edinet", "--fin", fin, "--no-family"])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "e69_edinet.json")))


if __name__ == "__main__":
    unittest.main()
