#!/usr/bin/env python3
"""
実験103（research/exp/e103_tp15_vs_tp10.py）の、重い計算を回さない部分のテスト。

- 回す前に決めたもの（規則・買い方・売りの2水準・判定の3条件・確かめの数字）
- 符号の入れ替えの p（全列挙）と判定（仕込んだ差は支持、無差・負・過半に足りない差は支持しない）
- 差の分解の合計が平均の差と一致する
- 3枠の回し方: その日の上位2件まで・同時に3件を超えない・実験87 の run_one と同じ取引
- 月の損益（1件に 1/3・取引の無い月は 0）と窓ごとの平均
- 合成の前向きの行で報告が最後まで出て、要約が保存され、銘柄コードを出さない

  python3 tests/test_e103_tp15_vs_tp10.py
"""
import contextlib
import io
import json
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

import e87_monthly_return_search as E87  # noqa: E402
import e103_tp15_vs_tp10 as E  # noqa: E402

CODES = [f"{1300 + i}0" for i in range(6)]


def synth_ft(seed: int = 0, per_day: int = 6) -> tuple[pd.DataFrame, list]:
    """前向きの行の形（窓3〜5・営業日ごとに per_day 行）。高値の最大 u から到達・到達日・出口を矛盾なく作る。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-05-02", "2023-10-31")
    rows = []
    for d in days:
        w = 3 + ((d.year - 2022) * 12 + d.month - 5) // 6
        for i in range(per_day):
            u = rng.gamma(2.0, 5.0) - 2.0
            c20 = min(u - rng.uniform(0, 6), rng.normal(0, 8))
            h0 = int(rng.integers(1, 21))
            h15 = min(20, h0 + int(rng.integers(0, 5)))
            h20 = min(20, h15 + int(rng.integers(0, 5)))
            rec = {"Code": CODES[i], "Date": d, "window": w, "score": float(rng.uniform()),
                   f"min_{E.RULE_CODE}": round(float(rng.uniform(85, 100)), 1), "n_break": per_day,
                   "c20": c20, "need15": float(rng.uniform(5, 25))}
            for t, hd in ((10.0, h0), (15.0, h15), (20.0, h20)):
                k = f"{int(t)}"
                hit = u >= t
                rec[f"hit{k}"] = hit
                rec[f"hitday{k}"] = float(hd) if hit else np.nan
                rec[f"tp{k}c"] = t if hit else c20
            rows.append(rec)
    ft = pd.DataFrame(rows)
    bar_days = list(pd.bdate_range("2022-04-01", "2024-01-31"))
    return ft, bar_days


class TestLineup(unittest.TestCase):
    def test_decided_before_running(self):
        self.assertEqual((E.RULE_CODE, E.RULE_LINE, E.MIN_BREAK, E.TOP_K, E.SLOTS, E.HOLD), ("K15", 95.0, 1, 2, 3, 20))
        self.assertEqual((E.TP_A, E.TP_B, E.ALPHA), (10.0, 15.0, 0.05))
        self.assertEqual(E.LEVELS, (10.0, 15.0, 20.0, None))
        self.assertEqual((E.H1, E.H2), (E87.H1, E87.H2))
        self.assertEqual(E.CHECK_A, {"n": 128, "trade_mean": 2.19, "avg": 1.76})     # §47（実験88）
        self.assertEqual(E.RULE_ROWS, 559)
        self.assertEqual([b[2] for b in E.VOL_BANDS], ["1.5σ が 10% 未満", "1.5σ が 10〜15%", "1.5σ が 15% 以上"])
        self.assertEqual(E.level_cols(15.0), ("tp15c", "hit15", "hitday15"))
        self.assertEqual(E.level_cols(None), ("c20", None, None))


class TestJudge(unittest.TestCase):
    def test_sign_flip_p(self):
        self.assertAlmostEqual(E.sign_flip_p([1.0] * 9), 1 / 512)
        self.assertAlmostEqual(E.sign_flip_p([0.0] * 5), 1.0)
        self.assertAlmostEqual(E.sign_flip_p([1.0, -1.0]), 0.75)
        self.assertTrue(np.isnan(E.sign_flip_p([])))
        with self.assertRaises(ValueError):
            E.sign_flip_p(np.ones(21))

    def test_judge(self):
        self.assertTrue(E.judge(np.linspace(0.5, 1.5, 9))["pass"])
        flat = E.judge(np.array([0.3, -0.3] * 4 + [0.3]))
        self.assertFalse(flat["pass"])
        self.assertFalse(E.judge(-np.linspace(0.5, 1.5, 9))["pass"])
        few = E.judge(np.array([5.0, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1]))   # 平均は正だが上の窓 4/9
        self.assertTrue(few["positive"])
        self.assertFalse(few["majority"])
        self.assertFalse(few["pass"])
        self.assertEqual(few["need"], 5)
        self.assertFalse(E.judge([])["pass"])


class TestPieces(unittest.TestCase):
    def setUp(self):
        self.ft, self.bar_days = synth_ft()

    def test_decompose(self):
        rule = self.ft[self.ft[f"min_{E.RULE_CODE}"] >= E.RULE_LINE]
        d = E.decompose(rule)
        self.assertGreater(d["n"], 50)
        self.assertAlmostEqual(d["gain"] + d["loss"] + d["rest"], d["diff"], places=9)
        self.assertAlmostEqual(d["share_top"] + d["share_mid"] + d["share_low"], 100.0, places=9)
        self.assertAlmostEqual(d["rest"], 0.0, places=12)
        self.assertEqual(d["order_bad"], 0)
        self.assertEqual(d["per_top"], 5.0)
        self.assertAlmostEqual(d["gain"], 5.0 * d["share_top"] / 100, places=9)
        self.assertLess(d["mid_c20"], 15.0)

    def test_screen_trades(self):
        idx = {pd.Timestamp(x): k for k, x in enumerate(self.bar_days)}
        for t in E.LEVELS:
            tr = E.screen_trades(self.ft, self.bar_days, t)
            self.assertGreater(len(tr), 10)
            self.assertTrue((tr["rank"] <= E.TOP_K).all())
            self.assertTrue((tr[f"min_{E.RULE_CODE}"] >= E.RULE_LINE).all())
            buy = np.array([idx[pd.Timestamp(x)] + 1 for x in tr["Date"]], dtype=float)
            rel = buy + E.held_days(tr, t)
            for j in range(len(tr)):                        # 買う時点で持っているのは2件以下（3件目が入る）
                self.assertLessEqual(int((rel[:j] > buy[j]).sum()), E.SLOTS - 1)
            if t is not None:
                r1 = E87.run_one(self.ft, self.bar_days, E.RULE_CODE, E.RULE_LINE, E.MIN_BREAK, E.TOP_K, E.SLOTS, 0, t)
                self.assertEqual(list(zip(r1["Code"], r1["Date"])), list(zip(tr["Code"], tr["Date"])))

    def test_monthly_and_windows(self):
        allm, pos, month_win = E87.months_index(self.ft)
        tr = pd.DataFrame({"Date": pd.to_datetime(["2022-05-10", "2022-05-20", "2022-07-01"]),
                           "tp10c": [10.0, -3.0, 10.0], "hit10": [True, False, True]})
        mon = E.monthly_pnl(tr, 10.0, pos, len(allm))
        self.assertAlmostEqual(mon[pos[pd.Period("2022-05", "M")]], 7.0 / 3)
        self.assertAlmostEqual(mon[pos[pd.Period("2022-06", "M")]], 0.0)
        self.assertAlmostEqual(mon[pos[pd.Period("2022-07", "M")]], 10.0 / 3)
        self.assertEqual(int(E.trade_counts(tr, pos, len(allm)).sum()), 3)
        wm = E.window_means(mon, month_win, [3, 4])
        self.assertAlmostEqual(wm[0], (7.0 + 10.0) / 3 / 6)
        self.assertAlmostEqual(wm[1], 0.0)
        self.assertEqual(len(E.monthly_pnl(tr.iloc[0:0], 10.0, pos, len(allm))), len(allm))
        self.assertAlmostEqual(E.max_drawdown(np.array([1.0, -2.0, -1.0, 4.0])), 3.0)

    def test_trade_stats(self):
        g = pd.DataFrame({"tp15c": [15.0, -4.0, 2.0], "hit15": [True, False, False], "hitday15": [5.0, np.nan, np.nan]})
        s = E.trade_stats(g, 15.0)
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], 13.0 / 3)
        self.assertAlmostEqual(s["days"], (5 + 20 + 20) / 3)
        self.assertAlmostEqual(s["per_day"], 13.0 / 45)
        self.assertAlmostEqual(s["hit"], 100 / 3)


class TestReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = E.OUT_DIR
        E.OUT_DIR = self.dir

    def tearDown(self):
        E.OUT_DIR = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_report_prints_every_section_and_saves(self):
        ft, bar_days = synth_ft(1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(ft, bar_days, {"order_violations": 0}, check_a=None, rule_rows=None)
        text = buf.getvalue()
        for head in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "■ 7.", "■ まとめ"):
            self.assertIn(head, text)
        self.assertIn(s["verdict"], ("+15% のほうが妥当（指値を変える案を運用者に出す）", "+10% のまま"))
        self.assertEqual(s["judge"]["n"], 3)
        self.assertTrue(s["checks"]["same_as_e87_10"] and s["checks"]["same_as_e87_15"])
        self.assertEqual(len(s["per_window"]), 3)
        self.assertEqual(set(s["screen"]), {E.level_name(t) for t in E.LEVELS})
        path = E.save(s)
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertIn("judge", saved)
        for code in CODES:
            self.assertNotIn(code, text)


if __name__ == "__main__":
    unittest.main()
