#!/usr/bin/env python3
"""
実験81（research/exp/e81_model_anomaly.py）。

- 珍しさ（近傍距離・孤立森・地図の疎ら・形の珍しさ）は訓練だけで作り、大きいほど珍しい
- 形の珍しさは合意の高さを除く（全部が同じだけ高い行は珍しくない）
- 帯ごとの三分位（切り目は訓練の帯の中から）と窓ごとの 珍しい − 普通
- 仕込んだ「並び方が珍しい行は収益が低い」を形の珍しさで拾う。報告に銘柄コードが出ない

  python3 tests/test_e81_model_anomaly.py
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
import e81_model_anomaly as E  # noqa: E402


def synth(n_windows=6, per=1500, seed=0, planted=-0.10, odd_share=0.08):
    """odd の行はロジスティック回帰の3本だけ大きく低い（並び方が珍しい）。その行の実収益を planted だけ動かす。"""
    rng = np.random.default_rng(seed)
    folds, rows = [], []
    start = pd.Timestamp("2021-05-01")
    for w in range(1, n_windows + 1):
        ts = start + pd.DateOffset(months=6 * (w - 1))
        te = start + pd.DateOffset(months=6 * w) - pd.Timedelta(days=1)
        folds.append(types.SimpleNamespace(index=w, test_start=str(ts.date()), test_end=str(te.date())))
        d = rng.choice(pd.bdate_range(ts, te), size=per)
        c = rng.normal(size=per)
        odd = rng.random(per) < odd_share
        ret = 0.01 * c + planted * odd + rng.normal(0, 0.05, size=per)
        rec = {"Code": [f"{7000 + i % 900}0" for i in range(per)], "Date": d, "window": w, E79.OUTCOME: ret,
               "future_rise": np.abs(ret) + 0.05, "ret_o1_40": ret * 1.5, "label_k12": (ret > 0.04).astype(float)}
        for a in E79.MODELS:
            shift = np.where(odd, -2.5, 0.0) if a in E79.LOGITS else 0.0
            rec[f"score_{a}"] = c + shift + rng.normal(0, 0.3, size=per)
        rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True).drop_duplicates(["Code", "Date"])
    return df.sort_values(["Date", "Code"]).reset_index(drop=True), folds


class TestScores(unittest.TestCase):
    def test_shape_ignores_level(self):
        rng = np.random.default_rng(0)
        Ptr = np.clip(rng.normal(50, 20, size=(500, 12)), 0, 100)
        Pte = np.vstack([np.full(12, 95.0), np.r_[np.full(9, 95.0), np.full(3, 20.0)]])
        tr, te = E.scores_for_window(Ptr, Pte)
        for k in ("knn", "iso", "sparse", "shape"):
            self.assertEqual(len(tr[k]), 500)
            self.assertEqual(len(te[k]), 2)
        self.assertLess(te["shape"][0], te["shape"][1])                 # そろって高い行より、3本だけ低い行が珍しい
        self.assertLess(te["shape"][0], np.median(tr["shape"]) * 1.5)    # そろって高いだけでは形は珍しくない


class TestForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = synth()
        cls.af = E.anomaly_frame(cls.df, E79.window_split(cls.df, folds))
        cls.df0, folds0 = synth(planted=0.0, seed=5)
        cls.af0 = E.anomaly_frame(cls.df0, E79.window_split(cls.df0, folds0))

    def test_frame(self):
        for c in ("cons", "knn", "iso", "sparse", "shape", "in_1.0", "in_0.1", "t_shape_0.1"):
            self.assertIn(c, self.af.columns)
        self.assertTrue(self.af["in_1.0"].all())
        t = self.af["t_shape_1.0"]
        self.assertTrue(set(t.dropna().unique()) <= {0.0, 1.0, 2.0})
        self.assertTrue(self.af.loc[~self.af["in_0.1"], "t_shape_0.1"].isna().all())

    def test_planted_unusual_rows_are_found(self):
        x = E.tercile_report(self.af, 1.0, "shape")
        # 珍しい三分位の収益が低い（合意が低い行は百分位の床に当たって形が崩れないので、拾えるのは一部）
        self.assertLess(x["hi_lo"]["mean"], -0.5)
        self.assertLess(x["terciles"][2]["ret"], x["terciles"][0]["ret"])
        x0 = E.tercile_report(self.af0, 1.0, "shape")
        self.assertLess(abs(x0["hi_lo"]["mean"]), 1.0)                  # 仕込まなければ差は小さい

    def test_report_prints_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.af)
        out = buf.getvalue()
        self.assertIn("■ まとめ", out)
        self.assertEqual(set(s["bands"]), {"1.0", "0.1", "0.025"})
        for code in self.df["Code"].unique()[:60]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
