#!/usr/bin/env python3
"""
進捗の比率の順位（build_dataset.progress_percentile）と、版を残した母集団のテスト。

2026-10-01 の学習と予測の一致チェックで progress_pct がずれた。原因は、同じ期の出し直し
（訂正）が来ると元の版が順位の母集団から消え、それより後に開示された行の順位が変わること
（docs/DATA_TIMING.md）。2026-10-02 から、母集団は「その日までに出ていた各期の最新の版」で持つ。

- keys を渡さなければ以前と同じ値（素朴な O(n²) の計算と一致）
- keys を渡すと、後の版はその日に母集団の値を差し替える。比率の無い版が来たら外す
- 同じ日の行は「その日より前」の母集団で順位を付け、同じ日の開示どうしは数えない
- quarterize_panel を通しても、後から出し直しが来た期より前に開示された行の順位は変わらない

  python3 tests/test_progress_percentile.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_dataset as B  # noqa: E402


def naive(dates, quarter, basis, ratio, min_history, keys=None):
    """素朴な計算。各行について、その日より前の「期ごとの最新の版」を集めて順位を付ける。"""
    n = len(ratio)
    dates = pd.to_datetime(pd.Series(dates)).to_numpy()
    ratio = np.asarray(ratio, dtype=float)
    quarter = np.asarray(quarter)
    table = np.where(np.isin(np.asarray(basis, dtype=object), ["seasonal", "floor"]),
                     "seasonal", "linear")
    keys = list(range(n)) if keys is None else list(keys)
    out = np.full(n, np.nan)
    order = np.argsort(dates, kind="mergesort")
    for r in range(n):
        if not (np.isfinite(ratio[r]) and quarter[r] in (1, 2, 3)):
            continue
        latest = {}
        for s in order:                              # 日付順に、その日より前の版を鍵ごとに差し替える
            if dates[s] >= dates[r]:
                break
            latest[keys[s]] = s
        pool = [ratio[s] for s in latest.values()
                if np.isfinite(ratio[s]) and quarter[s] == quarter[r] and table[s] == table[r]]
        if len(pool) < min_history:
            continue
        less = sum(1 for v in pool if v < ratio[r])
        eq = sum(1 for v in pool if v == ratio[r])
        out[r] = (less + 0.5 * eq) / len(pool) * 100.0
    return out


def random_case(seed, n=300, n_keys=120):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2026-01-05") + pd.to_timedelta(rng.integers(0, 60, n), unit="D")
    quarter = rng.choice([1, 2, 3, 4], n, p=[0.35, 0.3, 0.3, 0.05])
    basis = rng.choice(["seasonal", "floor", "linear"], n)
    ratio = rng.choice([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, np.nan], n, p=[.15, .2, .2, .2, .1, .1, .05])
    keys = rng.integers(0, n_keys, n)
    return dates, quarter, basis, ratio, keys


class TestAgainstNaive(unittest.TestCase):
    def test_without_keys_matches_the_naive_count(self):
        for seed in range(4):
            dates, quarter, basis, ratio, _ = random_case(seed)
            got = B.progress_percentile(dates, quarter, basis, ratio, min_history=5)
            want = naive(dates, quarter, basis, ratio, 5)
            np.testing.assert_allclose(got, want, equal_nan=True, err_msg=f"seed {seed}")

    def test_with_keys_matches_the_naive_replacement(self):
        for seed in range(4):
            dates, quarter, basis, ratio, keys = random_case(seed)
            got = B.progress_percentile(dates, quarter, basis, ratio, min_history=5, keys=keys)
            want = naive(dates, quarter, basis, ratio, 5, keys=keys)
            np.testing.assert_allclose(got, want, equal_nan=True, err_msg=f"seed {seed}")

    def test_keys_change_something(self):
        """差し替えが実際に起きる例で、keys の有無で値が変わること（テストが空振りでない）。"""
        dates, quarter, basis, ratio, keys = random_case(1)
        a = B.progress_percentile(dates, quarter, basis, ratio, min_history=5)
        b = B.progress_percentile(dates, quarter, basis, ratio, min_history=5, keys=keys)
        self.assertFalse(np.allclose(a, b, equal_nan=True))

    def test_wrong_key_length_is_an_error(self):
        with self.assertRaises(ValueError):
            B.progress_percentile(["2026-01-05"], [1], ["linear"], [1.0], keys=[1, 2])


class TestReplacement(unittest.TestCase):
    """手で追える例。母集団は 1Q・Q×25%（linear）の1つだけ、min_history は 1。"""

    def pct(self, rows, keys=None):
        d = pd.DataFrame(rows, columns=["date", "ratio", "key"])
        return B.progress_percentile(d["date"], [1] * len(d), ["linear"] * len(d), d["ratio"],
                                     min_history=1, keys=keys if keys is None else d["key"])

    def test_a_later_version_replaces_the_earlier_one_from_its_day(self):
        rows = [("2026-01-05", 1.0, "a"), ("2026-01-05", 2.0, "b"), ("2026-01-05", 3.0, "c"),
                ("2026-01-05", 2.0, "k"),          # 期 k の最初の版
                ("2026-01-06", 2.5, "y"),          # 母集団 {1, 2, 3, 2} の中で 2.5 -> 75
                ("2026-01-06", 10.0, "k"),         # 期 k の出し直し（同じ日の y には効かない）
                ("2026-01-07", 2.5, "x")]          # 母集団 {1, 2, 3, 10} の中で 2.5 -> 50
        with_keys = self.pct(rows, keys=True)
        self.assertAlmostEqual(with_keys[4], 75.0)
        self.assertAlmostEqual(with_keys[6], 50.0)
        # keys が無ければ k の2つの版は別々の行として数える:
        # {1, 2, 3, 2, 2.5, 10} の中で 2.5 -> 下に3つ・同じ値1つ -> 3.5/6
        without = self.pct(rows)
        self.assertAlmostEqual(without[4], 75.0)
        self.assertAlmostEqual(without[6], 3.5 / 6 * 100.0)

    def test_a_later_version_without_a_ratio_removes_the_period(self):
        rows = [("2026-01-05", 1.0, "a"), ("2026-01-05", 2.0, "b"), ("2026-01-05", 3.0, "c"),
                ("2026-01-05", 2.0, "k"),
                ("2026-01-06", np.nan, "k"),       # 比率の無い出し直し -> 母集団から外す
                ("2026-01-07", 2.5, "x")]          # 母集団 {1, 2, 3} の中で 2.5 -> 2/3
        got = self.pct(rows, keys=True)
        self.assertAlmostEqual(got[5], 2 / 3 * 100.0)
        self.assertTrue(np.isnan(got[4]))

    def test_same_day_disclosures_do_not_count_each_other(self):
        rows = [("2026-01-05", 1.0, "a"), ("2026-01-06", 2.0, "b"), ("2026-01-06", 3.0, "c")]
        got = self.pct(rows, keys=True)
        self.assertAlmostEqual(got[1], 100.0)      # 1/5 の {1} の中で 2 -> 100
        self.assertAlmostEqual(got[2], 100.0)
        self.assertTrue(np.isnan(got[0]))          # それより前が無い


def fins_row(code, fy, period, disc, op, fop=None):
    """quarterize_panel に渡せる決算の行（tests/test_progress.py の statement と同じ形）。"""
    return {"Code": code, "DiscDate": disc, "DiscTime": "15:00", "CurPerType": period,
            "CurFYSt": fy, "CurPerEn": disc, "DocType": "FinancialStatements",
            "Sales": 100.0, "OP": float(op), "NP": 1.0, "EPS": 1.0,
            "Eq": 1000.0, "TA": 2000.0, "ROE": None, "FSales": None, "FNP": None, "FEPS": None,
            "FOP": None if fop is None else float(fop), "ShOutFY": 1_000_000, "TrShFY": 0}


class TestThroughThePanel(unittest.TestCase):
    """
    quarterize_panel を通して。前年が無い会社ばかりなので物差しは Q×25%（linear）の1つ。
    min_history（200）を 8/20 より前に超えるだけの 1Q を 8月に並べ（400社、1日16社）、
    途中の1社が 9/14 に出し直す。
    """

    @classmethod
    def setUpClass(cls):
        rows = []
        n = B.PROGRESS_PCT_MIN_HISTORY * 2
        for i in range(n):
            code = f"{10000 + i}"
            day = pd.Timestamp("2026-08-01") + pd.Timedelta(days=i % 25)   # 8/1〜8/25
            rows.append(fins_row(code, "2026-04-01", "1Q", day.strftime("%Y-%m-%d"),
                                 op=10 + (i * 7) % 50, fop=100))
        cls.fixed = "10003"                                                 # 8/4 に開示
        cls.before = [r["Code"] for r in rows if r["DiscDate"] == "2026-08-20"][0]
        cls.after_code = "99990"
        rows.append(fins_row(cls.after_code, "2026-04-01", "1Q", "2026-09-20", op=30, fop=100))
        cls.without = B.quarterize_panel(pd.DataFrame(rows))
        # 出し直しで比率が 1.24（31/25）から 0.80（20/25）へ。9/20 の行（1.20）をまたぐので順位が動く
        corrected = rows + [fins_row(cls.fixed, "2026-04-01", "1Q", "2026-09-14", op=20, fop=100)]
        cls.with_fix = B.quarterize_panel(pd.DataFrame(corrected))

    def one(self, q, code):
        x = q[(q["Code"] == code) & (q["quarter"] == 1)]
        self.assertEqual(len(x), 1)
        return x.iloc[0]

    def test_rows_before_the_correction_keep_their_rank(self):
        """出し直しより前に開示された行の順位は、出し直しが来ても変わらない。"""
        a = self.one(self.without, self.before)["progress_pct"]
        b = self.one(self.with_fix, self.before)["progress_pct"]
        self.assertTrue(np.isfinite(a))
        self.assertAlmostEqual(a, b)

    def test_rows_after_the_correction_see_the_new_value(self):
        """出し直しより後に開示された行は、差し替え後の値を母集団に持つ。"""
        a = self.one(self.without, self.after_code)["progress_pct"]
        b = self.one(self.with_fix, self.after_code)["progress_pct"]
        self.assertTrue(np.isfinite(a) and np.isfinite(b))
        self.assertNotAlmostEqual(a, b)

    def test_the_corrected_period_keeps_one_row_with_the_last_version(self):
        r = self.one(self.with_fix, self.fixed)
        self.assertEqual(pd.Timestamp(r["DiscDate"]), pd.Timestamp("2026-09-14"))
        self.assertAlmostEqual(r["progress_ratio"], 20 / 25.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
