#!/usr/bin/env python3
"""
実験65（research/exp/e65_exit_horizon.py）の出口の規則（exit_rule）のテスト。重い計算は回さない。

約定の仮定を固定する（docs/MODEL_EXIT_HORIZON.md と e65 の冒頭に書いたもの）:
  - 指値は 1〜tp_days 日目に高値が届けば、ちょうど +tp（飛び越えても）
  - decide 日目の終値が買値以上なら plus の扱い（sell / hold）、未満なら minus の扱い（sell / carry）
  - 持ち越した玉は decide+1 日目から、高値が建値に届けばちょうど be（既定 0%）で売る
  - cap 日目の終値で必ず売る。終値が無い日は、それまでの最後の終値
  - 同じ日に複数当たれば 指値 → プラ転 → 終値 の順
  - 窓の割り当ては本番の OOF と同じ切り方（36/6/6か月・エンバーゴ20営業日）

  python3 tests/test_e65_exit_horizon.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e65_exit_horizon as E  # noqa: E402

NAN = float("nan")


def paths(rows):
    """rows: 行ごとに [(O, H, L, C), ...]（1日目から）。買値は1日目の O。足りない日は NaN で埋める。"""
    n = max(len(r) for r in rows)
    P = {c: np.full((len(rows), n), NAN) for c in "OHLC"}
    for i, r in enumerate(rows):
        for j, (o, h, lo, c) in enumerate(r):
            P["O"][i, j], P["H"][i, j], P["L"][i, j], P["C"][i, j] = o, h, lo, c
    P["entry"] = P["O"][:, 0].copy()
    P["ok"] = np.isfinite(P["C"])
    return P


def flat(n, px=100.0):
    return [(px, px, px, px)] * n


class ExitRule(unittest.TestCase):
    def test_limit_hits_exactly_at_tp(self):
        # 3日目に高値が +10% を飛び越える → ちょうど +10% で、3日目
        rows = [flat(2) + [(100, 115, 99, 112)] + flat(20, 112)]
        r, d, w = E.exit_rule(paths(rows), cap=20, tp=0.10)
        self.assertAlmostEqual(r[0], 0.10)
        self.assertEqual((d[0], w[0]), (3, 1))

    def test_limit_only_within_tp_days(self):
        # 25日目に +10% に届いても、指値は 20日目まで。20日目の終値（+2%）で売る
        rows = [flat(19) + [(101, 102, 100, 102)] + flat(4) + [(100, 115, 100, 110)] + flat(20)]
        r, d, w = E.exit_rule(paths(rows), cap=40, tp=0.10, plus="hold", minus="carry")
        # plus=hold なので 20日目には売らず、指値は切れているので 40日目の終値（100）で売る
        self.assertAlmostEqual(r[0], 0.0)
        self.assertEqual((d[0], w[0]), (40, 4))
        # tp_days=40 なら 25日目に指値で売れる
        r, d, w = E.exit_rule(paths(rows), cap=40, tp=0.10, tp_days=40, decide=40)
        self.assertAlmostEqual(r[0], 0.10)
        self.assertEqual((d[0], w[0]), (25, 1))

    def test_decide_plus_sells_at_close(self):
        rows = [flat(19) + [(100, 104, 100, 103)] + flat(30, 150)]
        r, d, w = E.exit_rule(paths(rows), cap=60, tp=0.10, minus="carry")
        self.assertAlmostEqual(r[0], 0.03)
        self.assertEqual((d[0], w[0]), (20, 2))

    def test_decide_minus_sell_or_carry(self):
        rows = [flat(19) + [(100, 100, 95, 97)] + flat(5, 97) + [(97, 100.5, 97, 100)] + flat(40, 90)]
        # minus=sell: 20日目の終値 −3%
        r, d, w = E.exit_rule(paths(rows), cap=20, tp=0.10)
        self.assertAlmostEqual(r[0], -0.03)
        self.assertEqual((d[0], w[0]), (20, 6))
        # minus=carry: 26日目に高値が建値に届いてプラ転 0%
        r, d, w = E.exit_rule(paths(rows), cap=60, tp=0.10, minus="carry")
        self.assertAlmostEqual(r[0], 0.0)
        self.assertEqual((d[0], w[0]), (26, 3))
        # be=0.02 なら 建値+2% には届かず、上限の日（60日目）の終値 90 → −10%
        r, d, w = E.exit_rule(paths(rows), cap=60, tp=0.10, minus="carry", be=0.02)
        self.assertAlmostEqual(r[0], -0.10)
        self.assertEqual((d[0], w[0]), (60, 4))

    def test_breakeven_does_not_apply_to_held_plus(self):
        # plus=hold の玉は建値の指値の対象にならない（21日目に高値が建値以上でも売らない）
        rows = [flat(19) + [(100, 104, 100, 103)] + [(103, 106, 102, 105)] + flat(18, 105) + [(105, 108, 104, 108)]]
        r, d, w = E.exit_rule(paths(rows), cap=40, tp=0.10, plus="hold", minus="carry")
        self.assertAlmostEqual(r[0], 0.08)
        self.assertEqual((d[0], w[0]), (40, 4))

    def test_cap_uses_last_close_when_missing(self):
        rows = [flat(19) + [(100, 100, 96, 96)] + flat(19, 96) + [(NAN, NAN, NAN, NAN)]]
        r, d, w = E.exit_rule(paths(rows), cap=40, minus="carry")
        self.assertAlmostEqual(r[0], -0.04)
        self.assertEqual((d[0], w[0]), (40, 4))

    def test_same_day_priority_limit_before_close(self):
        # 20日目に高値が +10% に届き、終値は +1% → 指値が先
        rows = [flat(19) + [(100, 111, 100, 101)]]
        r, d, w = E.exit_rule(paths(rows), cap=20, tp=0.10)
        self.assertAlmostEqual(r[0], 0.10)
        self.assertEqual(w[0], 1)

    def test_no_entry_is_nan(self):
        rows = [[(NAN, NAN, NAN, NAN)] + flat(30)]
        r, d, w = E.exit_rule(paths(rows), cap=20)
        self.assertTrue(np.isnan(r[0]) and np.isnan(d[0]) and w[0] == 0)

    def test_hold_to_cap_matches_close(self):
        rows = [flat(39) + [(100, 100, 100, 107)]]
        # decide = cap のときは「decide 日目にプラスで売った」(2) として数える（上限 (4) ではない）
        r, d, w = E.exit_rule(paths(rows), cap=40, decide=40)
        self.assertAlmostEqual(r[0], 0.07)
        self.assertEqual((d[0], w[0]), (40, 2))
        # plus=hold なら上限の日の終値として売る (4)
        r, d, w = E.exit_rule(paths(rows), cap=40, decide=40, plus="hold")
        self.assertAlmostEqual(r[0], 0.07)
        self.assertEqual((d[0], w[0]), (40, 4))

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            E.exit_rule(paths([flat(20)]), cap=20, plus="keep")
        with self.assertRaises(ValueError):
            E.exit_rule(paths([flat(20)]), cap=20, decide=30)


class Folds(unittest.TestCase):
    def test_fold_assignment_follows_production_windows(self):
        f = E.fold_of(pd.Series(pd.to_datetime(["2021-05-06", "2021-11-30", "2022-01-15", "2018-06-01"])),
                      "2018-04-03", "2026-09-01")
        self.assertEqual(f[0], 1)          # 最初の窓（訓練36か月 + エンバーゴ）
        self.assertEqual(f[3], 0)          # 窓より前
        self.assertGreater(f[2], f[1] - 1) # 後の日ほど後の窓
        self.assertTrue(f[2] >= f[1])


class Rules(unittest.TestCase):
    def test_rule_names_unique_and_base_first(self):
        names = [n for n, _ in E.rules()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], E.BASE)
        self.assertIn(E.CURRENT, names)


if __name__ == "__main__":
    unittest.main()
