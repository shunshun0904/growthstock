#!/usr/bin/env python3
"""
実験80（research/exp/e80_k15_rule_forward.py）。

- 前向きの行: 合意・d15・d18・lgbm の百分位、帯の印（閾値は訓練の分布から）、地図の値・k 近傍の値が付く
- 規則（d15 > 0）の窓ごとの差と並べ替え検定: 仕込んだ効果は拾い、効果が無ければ p は小さくならない
- 地図・k 近傍・lgbm で同じ件数を選ぶ比較と、月に2銘柄
- 報告に銘柄コードが出ない

  python3 tests/test_e80_k15_rule_forward.py
"""
import contextlib
import io
import os
import sys
import types
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e79_model_clusters as E79  # noqa: E402
import e80_k15_rule_forward as E  # noqa: E402


def synth(n_windows=6, per=1500, seed=0, planted=0.05):
    """合意 c と「1.5σ の4本だけ高い」ずれ q。上位（c > 1）で q > 0 の行の実収益を planted だけ上げる。"""
    rng = np.random.default_rng(seed)
    folds, rows = [], []
    start = pd.Timestamp("2021-05-01")
    for w in range(1, n_windows + 1):
        ts = start + pd.DateOffset(months=6 * (w - 1))
        te = start + pd.DateOffset(months=6 * w) - pd.Timedelta(days=1)
        folds.append(types.SimpleNamespace(index=w, test_start=str(ts.date()), test_end=str(te.date())))
        d = rng.choice(pd.bdate_range(ts, te), size=per)
        c = rng.normal(size=per)
        q = rng.normal(size=per)
        ret = 0.01 * c + planted * ((q > 0) & (c > 1.0)) + rng.normal(0, 0.05, size=per)
        rec = {"Code": [f"{8000 + i % 900}0" for i in range(per)], "Date": d, "window": w, E79.OUTCOME: ret,
               "future_rise": np.abs(ret) + 0.05, "ret_o1_40": ret * 1.5, "label_k12": (ret > 0.04).astype(float),
               "label_k15": (ret > 0.07).astype(float), "label_k18": (ret > 0.10).astype(float)}
        for a in E79.MODELS:
            tilt = 0.8 if a in E79.K15 else 0.0
            rec[f"score_{a}"] = c + tilt * q + rng.normal(0, 0.3, size=per)
        rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True).drop_duplicates(["Code", "Date"])
    return df.sort_values(["Date", "Code"]).reset_index(drop=True), folds


class TestForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.folds = synth()
        cls.split = E79.window_split(cls.df, cls.folds)
        cls.ft, cls.info = E.forward_frame(cls.df, cls.split)
        cls.df0, folds0 = synth(planted=0.0, seed=11)
        cls.ft0, _ = E.forward_frame(cls.df0, E79.window_split(cls.df0, folds0))

    def test_frame_columns_and_bands(self):
        for c in ("cons", "lgbm", "d15", "d18", "map", "knn", "in_0.1", "in_0.025", "d15hi_0.1"):
            self.assertIn(c, self.ft.columns)
        self.assertEqual(sorted(self.info), [3, 4, 5, 6])
        share10 = self.ft["in_0.1"].mean()
        share25 = self.ft["in_0.025"].mean()
        self.assertTrue(0.05 < share10 < 0.2, share10)                       # 訓練の閾値でもおおよそ10%
        self.assertTrue(0.005 < share25 < 0.06, share25)
        self.assertTrue((self.ft["in_0.025"] <= self.ft["in_0.1"]).all())     # 2.5% は 10% の中
        self.assertGreater(np.isfinite(self.ft["map"]).mean(), 0.3)            # 過去の件数が少ない窓ではマス目が埋まらない
        self.assertTrue(np.isfinite(self.ft["knn"]).all())

    def test_rule_finds_planted_effect(self):
        res = E.rule_test(self.ft, 0.10, "d15", n_perm=300)
        self.assertGreater(res["summary"]["mean"], 1.0)
        self.assertLess(res["p"], 0.05)
        self.assertTrue(E.judge(res)["pass"])
        self.assertGreater(res["yes"]["ret"], res["no"]["ret"])

    def test_rule_without_effect_is_not_significant(self):
        res = E.rule_test(self.ft0, 0.10, "d15", n_perm=300)
        self.assertGreater(res["p"], 0.01)
        self.assertFalse(E.judge(res)["pass"])

    def test_selection_and_monthly(self):
        st = E.selection_test(self.ft, 0.10)
        for k in ("map", "knn", "lgbm"):
            self.assertEqual(st[k]["vs_cons"]["n"], 4)
            self.assertTrue(0.0 <= st[k]["overlap"] <= 1.0)
        mo = E.monthly(self.ft)
        self.assertEqual(list(mo), [m[0] for m in E.MONTHLY])
        self.assertIsNone(mo["合意の上位2件"]["vs_cons"])
        self.assertEqual(mo["合意の上位2件"]["n"], 2 * mo["合意の上位2件"]["months"])
        self.assertIsNotNone(mo["d15 > 0 の中で合意の上位2件"]["vs_cons"])

    def test_report_prints_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, self.info, n_perm=50)
        out = buf.getvalue()
        self.assertIn("■ 2.", out)
        self.assertIn("rule", s)
        for code in self.df["Code"].unique()[:60]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
