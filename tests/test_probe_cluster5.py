#!/usr/bin/env python3
"""
5変数のクラスタごとの正例率（research/exp/probe_cluster5.py）。合成データで:

- Wilson の区間・百分位（同点は中央の順位）の値
- 組の件数と正例の数が全体と合い、信用倍率なし・欠測で外した行の数が合う。番号は正例率の高い順
- 前向き: 窓が本番の out-of-fold と同じ作りで、本番のスコアが行に正しく付く（ラベルから作ったスコアの AUC が高い）
- 本番の上位が組のどこにいるか: 割合の合計が 1・上位はおよそ 5%
- 報告に銘柄コードが出ない

  python3 tests/test_probe_cluster5.py
"""
import contextlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import probe_cluster5 as P  # noqa: E402


def synth(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """データセットの形の作りもの（ボラが低いほど正例になりやすい。信用倍率の約18%は欠測、20日上昇率の5行は欠測）。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-04-02", "2026-09-15")
    vol = rng.lognormal(0.8, 0.5, n)
    z = (np.log(vol) - 0.8) / 0.5
    p = 1.0 / (1.0 + np.exp(-(-1.6 - 0.8 * z)))
    credit = rng.lognormal(1.5, 1.5, n)
    credit[rng.random(n) < 0.18] = np.nan
    ret20 = rng.normal(15, 15, n)
    ret20[rng.choice(n, 5, replace=False)] = np.nan
    return pd.DataFrame({
        "Code": [str(10000 + i) for i in range(n)],
        "Date": pd.to_datetime(rng.choice(days, n)),
        "label": (rng.random(n) < p).astype(float),
        "future_rise": rng.normal(0.05, 0.1, n),
        "credit_ratio": credit,
        "volume_trend": rng.lognormal(5.3, 0.6, n),
        "ret_20d": ret20,
        "vol_20d": vol,
        "tx_ma_ratio_25_75": rng.normal(5, 6, n),
    })


def synth_oof(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """本番の out-of-fold の形の作りもの（窓の中の行だけ。スコアはラベルにノイズを足したもの）。"""
    rng = np.random.default_rng(seed)
    folds = P.make_windows(df["Date"])
    start, end = pd.Timestamp(folds[0].test_start), pd.Timestamp(folds[-1].test_end)
    o = df[(df["Date"] >= start) & (df["Date"] <= end)][["Code", "Date", "label"]].copy()
    o["score"] = o["label"] + rng.normal(0, 0.6, len(o))
    return o


class TestProbeCluster5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synth()
        cls.oof = synth_oof(cls.df)

    def test_wilson(self):
        lo, hi = P.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        lo, hi = P.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        self.assertTrue(all(np.isnan(P.wilson(0, 0))))

    def test_pct_midrank(self):
        ref = np.array([1.0, 2.0, 2.0, 3.0])
        out = P.pct_apply(ref, np.array([2.0, 0.0, 5.0, np.nan]))
        self.assertAlmostEqual(out[0], 0.5)          # 同点 2 つの真ん中
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[2], 1.0)
        self.assertTrue(np.isnan(out[3]))

    def test_groups_add_up(self):
        g, X = P.cluster_all(self.df, 4)
        g = P.renumber_by_rate(self.df, g)
        d = P.describe(self.df, g, X)
        others_ok = self.df[P.COLS[1:]].notna().all(axis=1)
        self.assertEqual(d["dropped"], int((~others_ok).sum()))
        self.assertEqual(d["rows"], int(others_ok.sum()))
        no_credit = [r for r in d["groups"] if r["id"] == P.NO_CREDIT][0]
        self.assertEqual(no_credit["n"], int((self.df["credit_ratio"].isna() & others_ok).sum()))
        self.assertEqual(no_credit["level"]["credit_ratio"], "-")
        self.assertTrue(np.isnan(no_credit["median"]["credit_ratio"]))
        rates = [r["rate"] for r in d["groups"] if r["id"] > 0]
        self.assertEqual(rates, sorted(rates, reverse=True))        # C1 が一番高い
        for r in d["groups"]:
            self.assertLessEqual(r["lo"], r["rate"])
            self.assertLessEqual(r["rate"], r["hi"])

    def test_forward_aligns_base_scores(self):
        fw = P.forward(self.df, self.oof, 4, with_gbm=True)
        s = fw["summary"]
        self.assertGreater(s["n_windows"], 3)
        # 本番のスコアはラベル + ノイズなので、行が正しく付いていれば AUC は高い（ずれていれば 0.5 前後）
        self.assertGreater(s["auc"]["base"]["mean"], 0.7)
        for name in ("cluster", "gbm5"):
            self.assertIn(name, s["auc"])
        for w in fw["windows"]:
            self.assertLessEqual(w["n"], w["n_test"])
        self.assertEqual(s["rows"], sum(w["n"] for w in fw["windows"]))

    def test_model_top_shares(self):
        g, _ = P.cluster_all(self.df, 4)
        g = P.renumber_by_rate(self.df, g)
        mt = P.model_top(self.df, g, self.oof)
        self.assertAlmostEqual(sum(r["share_all"] for r in mt["groups"]), 1.0, places=6)
        self.assertAlmostEqual(sum(r["share_top"] for r in mt["groups"]), 1.0, places=6)
        self.assertEqual(sum(r["n_top"] for r in mt["groups"]), mt["top_rows"])
        self.assertLess(abs(mt["top_rows"] / mt["rows"] - P.TOP_SHARE), 0.02)

    def test_report_has_no_codes(self):
        g, X = P.cluster_all(self.df, 4)
        g = P.renumber_by_rate(self.df, g)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.print_describe(P.describe(self.df, g, X), 4)
            P.print_forward(P.forward(self.df, self.oof, 4, with_gbm=False))
            P.print_model_top(P.model_top(self.df, g, self.oof))
        out = buf.getvalue()
        self.assertIn("組ごと", out)
        self.assertFalse(any(c in out for c in self.df["Code"]))


if __name__ == "__main__":
    unittest.main()
