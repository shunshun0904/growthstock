#!/usr/bin/env python3
"""
実験110（research/exp/e110_us_overnight.py）。

- 米国の日足: 2つの JSON の日付がそろわない・重複・正でない終値・別の銘柄は止める。前日比は終値から、日付は昇順
- 前夜の対応: D ≤ U ≤ B−1日 の最後の米国の日。週末・米国の休場（印なし）・東証の連休（2日。累積も）・買う日が無い
- 印: 小数2桁に丸めて線ちょうどは当たる。NaN は当たらない。片方の線だけも
- 日単位の並べ替え: 同じ日の行は同じ印（食い違えば止める）。差を仕込めば片側 p が小さく、仕込まなければ小さくない
- 合成データで報告が全部の段を出し、判定の件数が規則の行と合い、銘柄コードを出さない

  python3 tests/test_e110_us_overnight.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import e79_model_clusters as E79  # noqa: E402
import e110_us_overnight as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


def write_us(tmp, name, sym, dates, prices):
    """FMP の形（新しい日が先）で書く。"""
    rows = [{"symbol": sym, "date": d, "price": p, "volume": 1} for d, p in zip(dates, prices)][::-1]
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return path


def us_frame(dates, sp, nq):
    return pd.DataFrame({"us_date": pd.to_datetime(dates), "sp": sp, "nq": nq})


class TestLoadUs(unittest.TestCase):
    def test_returns_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]
            a = write_us(tmp, "sp.json", "^GSPC", d, [100.0, 101.0, 99.99, 102.0])
            b = write_us(tmp, "nq.json", "^IXIC", d, [200.0, 196.0, 196.0, 200.0])
            us = E.load_us(a, b)
        self.assertEqual(list(us["us_date"].dt.strftime("%Y-%m-%d")), d[1:])
        np.testing.assert_allclose(us["sp"], [1.0, -1.0, (102.0 / 99.99 - 1) * 100])
        np.testing.assert_allclose(us["nq"], [-2.0, 0.0, (200.0 / 196.0 - 1) * 100])

    def test_rejects_bad_files(self):
        d = ["2026-10-19", "2026-10-20"]
        with tempfile.TemporaryDirectory() as tmp:
            a = write_us(tmp, "sp.json", "^GSPC", d, [100.0, 101.0])
            cases = {
                "日付": write_us(tmp, "nq1.json", "^IXIC", ["2026-10-19", "2026-10-21"], [1.0, 2.0]),
                "銘柄": write_us(tmp, "nq2.json", "^NDX", d, [1.0, 2.0]),
                "終値": write_us(tmp, "nq3.json", "^IXIC", d, [1.0, 0.0]),
                "重複": write_us(tmp, "nq4.json", "^IXIC", ["2026-10-19", "2026-10-19"], [1.0, 2.0]),
            }
            for why, b in cases.items():
                with self.subTest(why), self.assertRaises(SystemExit):
                    E.load_us(a, b)


class TestOvernight(unittest.TestCase):
    # 米国: 11/26（木）は感謝祭で休み。東証: 11/23（月）は勤労感謝の日で休み
    US = us_frame(["2026-11-20", "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30"],
                  [-1.0, -2.0, 0.5, -0.3, 1.0, -1.6], [-3.0, 0.1, -2.5, 0.2, -0.4, 0.0])
    JP = pd.to_datetime(["2026-11-20", "2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27", "2026-11-30"])

    def test_next_day(self):
        b = E.next_day(pd.to_datetime(["2026-11-20", "2026-11-26", "2026-11-30"]), self.JP)
        self.assertEqual(str(pd.Timestamp(b[0]).date()), "2026-11-24")     # 月曜は東証の休み
        self.assertEqual(str(pd.Timestamp(b[1]).date()), "2026-11-27")
        self.assertTrue(np.isnat(b[2]))                                       # 次の日足が無い

    def test_cases(self):
        sig = pd.to_datetime(["2026-11-24", "2026-11-26", "2026-11-25", "2026-11-20", "2026-11-27", "2026-11-30"])
        buy = pd.Series(pd.to_datetime(["2026-11-25", "2026-11-27", "2026-11-26", "2026-11-24", "2026-11-30", None]))
        o = E.overnight(sig, buy, self.US)
        # 平日: 前夜は D と同じ日付の米国
        self.assertEqual(str(o.loc[0, "us_date"].date()), "2026-11-24")
        self.assertEqual((o.loc[0, "sp"], o.loc[0, "nq"], o.loc[0, "n_us"]), (0.5, -2.5, 1))
        # 米国の休場（11/26）: 前夜なし
        self.assertEqual(o.loc[1, "n_us"], 0)
        self.assertTrue(pd.isna(o.loc[1, "us_date"]) and np.isnan(o.loc[1, "sp"]))
        # 翌日（東証は開くが米国は休み）の買いは 11/25 の米国
        self.assertEqual(str(o.loc[2, "us_date"].date()), "2026-11-25")
        # 東証の連休: 金曜と月曜の米国の2日。前夜は月曜、累積は2日分
        self.assertEqual(str(o.loc[3, "us_date"].date()), "2026-11-23")
        self.assertEqual((o.loc[3, "n_us"], o.loc[3, "sp"]), (2, -2.0))
        self.assertAlmostEqual(o.loc[3, "sp_cum"], (0.99 * 0.98 - 1) * 100)
        self.assertAlmostEqual(o.loc[3, "nq_cum"], (0.97 * 1.001 - 1) * 100)
        # 週末: 金曜の米国だけ
        self.assertEqual((str(o.loc[4, "us_date"].date()), o.loc[4, "n_us"]), ("2026-11-27", 1))
        # 買う日が無い
        self.assertEqual(o.loc[5, "n_us"], 0)

    def test_attach_us(self):
        ft = pd.DataFrame({"Code": ["10010", "10020"], "Date": pd.to_datetime(["2026-11-20", "2026-11-26"])})
        out = E.attach_us(ft, self.JP, self.US)
        for c in E.US_COLS:
            self.assertIn(c, out.columns)
        self.assertEqual(list(out["n_us"]), [2, 0])
        self.assertEqual(list(E.flag(out["sp"], out["nq"])), [True, False])


class TestFlag(unittest.TestCase):
    def test_lines(self):
        f = E.flag([-1.5, -1.49, np.nan, 0.0, -1.4951, -1.4949], [0.0, -1.99, -2.0, np.nan, 0.0, 0.0])
        self.assertEqual(list(f), [True, False, True, False, True, False])

    def test_one_line_only(self):
        self.assertFalse(E.flag([-5.0], [0.0], sp_line=None)[0])
        self.assertTrue(E.flag([-5.0], [0.0], nq_line=None)[0])
        self.assertTrue(E.flag([0.0], [-3.0], sp_line=None, nq_line=-3.0)[0])


def day_frame(n_days=60, per_day=3, windows=(3, 4, 5), shift=0.0, share=0.15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in windows:
        days = pd.bdate_range(f"{2018 + w}-01-05", periods=n_days)
        fd = rng.random(n_days) < share
        for d, f in zip(days, fd):
            r = rng.normal(2.0, 4.0, size=per_day) + (shift if f else 0.0)
            rows.append(pd.DataFrame({"buy": d, "window": w, "f": f, "v": r}))
    return pd.concat(rows, ignore_index=True)


class TestDayPerm(unittest.TestCase):
    def test_d_and_days(self):
        d = pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-07"])
        t = E.day_perm([1.0, 2.0, 3.0, 4.0], [True, True, False, False], d, np.zeros(4), n_perm=50)
        self.assertAlmostEqual(t["D"], 1.5 - 3.5)
        self.assertEqual((t["n_yes"], t["n_no"], t["days_yes"], t["days_no"]), (2, 2, 1, 2))
        self.assertTrue(0 < t["p_less"] <= 1 and 0 < t["p_two"] <= 1)

    def test_mixed_day_raises(self):
        d = pd.to_datetime(["2026-01-05", "2026-01-05"])
        with self.assertRaises(ValueError):
            E.day_perm([1.0, 2.0], [True, False], d, np.zeros(2), n_perm=5)

    def test_nan_and_nat_dropped(self):
        d = pd.Series(pd.to_datetime(["2026-01-05", "2026-01-06", None, "2026-01-07"]))
        t = E.day_perm([1.0, np.nan, 5.0, 3.0], [True, True, False, False], d, np.zeros(4), n_perm=20)
        self.assertEqual((t["n_yes"], t["n_no"]), (1, 1))
        self.assertAlmostEqual(t["D"], 1.0 - 3.0)

    def test_planted_and_null(self):
        g = day_frame(shift=-5.0)
        t = E.day_perm(g["v"], g["f"], g["buy"], g["window"], n_perm=400)
        self.assertLess(t["D"], 0)
        self.assertLess(t["p_less"], 0.05)
        g0 = day_frame(shift=0.0, seed=7)
        t0 = E.day_perm(g0["v"], g0["f"], g0["buy"], g0["window"], n_perm=400)
        self.assertGreater(t0["p_less"], 0.05)

    def test_whole_days_move_together(self):
        # 日ごとの平均だけが違う（日の中は同じ値）。日単位の並べ替えの D は「日の平均の差」の重み付きに一致する
        g = day_frame(n_days=30, per_day=4, shift=-3.0, seed=3)
        g["v"] = g.groupby(["window", "buy"])["v"].transform("mean")
        t = E.day_perm(g["v"], g["f"], g["buy"], g["window"], n_perm=50)
        self.assertAlmostEqual(t["D"], g.loc[g["f"], "v"].mean() - g.loc[~g["f"], "v"].mean())
        self.assertEqual(t["days_yes"] + t["days_no"], g.groupby(["window", "buy"]).ngroups)

    def test_one_side_empty(self):
        d = pd.to_datetime(["2026-01-05", "2026-01-06"])
        t = E.day_perm([1.0, 2.0], [False, False], d, np.zeros(2), n_perm=10)
        self.assertTrue(np.isnan(t["D"]))
        self.assertIn("0件", E.fmt_p(t))


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T83.synth_with_exits()
        split = E79.window_split(cls.df, folds)
        ft = E.E87.build(cls.df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-12-30"))          # どの行にも次の営業日がある
        rng = np.random.default_rng(11)
        ud = pd.bdate_range("2021-03-01", "2024-09-30")
        cls.us = us_frame(ud, rng.normal(0.0, 1.2, len(ud)), rng.normal(0.0, 1.6, len(ud)))
        cls.ft = E.attach_us(ft, cls.bar_days, cls.us)
        cls.ft["gap"] = rng.normal(0.0, 2.0, len(cls.ft))

    def test_report_sections_and_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, self.bar_days, self.us, n_perm=60)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "■ 7.", "■ 8.", "■ まとめ", "判定"):
            self.assertIn(tag, out)
        for k in ("data", "main", "skip", "bands", "alt", "full", "gap", "per_window", "trade"):
            self.assertIn(k, s)
        k = E.E88.rule_mask(self.ft, "min_K15", 95.0)
        n_rule = int(pd.to_numeric(self.ft.loc[k, "tp10c"], errors="coerce").notna().sum())
        self.assertGreater(s["main"]["test"]["n_yes"], 0)
        self.assertEqual(s["main"]["test"]["n_yes"] + s["main"]["test"]["n_no"], n_rule)
        self.assertEqual(s["main"]["yes"]["n"] + s["main"]["no"]["n"], n_rule)
        n_bands = sum(v["n"] for v in s["bands"].values())
        self.assertEqual(n_bands, n_rule)                           # 帯と休場で規則の行を割り切る
        self.assertEqual(s["skip"]["dropped_n"], s["main"]["yes"]["n"])
        self.assertLessEqual(s["trade"]["skip"]["n"], s["main"]["no"]["n"])     # 買わない形の取引は目安でない日の行だけ
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)

    def test_needs_attach_us(self):
        with self.assertRaises(ValueError):
            E.report(self.ft.drop(columns=["n_us"]), self.bar_days, self.us, n_perm=5)


if __name__ == "__main__":
    unittest.main()
