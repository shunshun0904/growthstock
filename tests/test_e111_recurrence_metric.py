#!/usr/bin/env python3
"""
実験111（research/exp/e111_recurrence_metric.py）の部品。データには触らない（作りものの小さな例で確かめる）。

- 白色化: 縮小推定の共分散 Σ̂ について W Σ̂ W = I。白色化した座標の距離の2乗 = (x−y)ᵀ Σ̂⁻¹ (x−y)（マハラノビス）
- E（今のユークリッド）の近い順は、本番の recurrence_net.knn_order（今日の行）と同じ。タイラー窓の中の日は選ばない・同じ距離なら古い日
- 重なった列: ほぼ同じ5列の向きだけで違う日と、1列だけで違う日では、E は前者を遠く、M は前者を近く数える（5列ぶんを1つ分にする）
- 同じ銘柄の中だけの AUC: 銘柄どうしの差は効かない。同じ列なら差は 0
- 報告: M が銘柄の中で当てる作りものでは「マハラノビスの方が当てる」、当てなければ「ユークリッドのまま」。銘柄コードを出さない
- 実験107 の行との突き合わせ: 同じなら一致 100%、値を変えれば下がる

  python3 tests/test_e111_recurrence_metric.py
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import recurrence_net as RN  # noqa: E402
import e111_recurrence_metric as E  # noqa: E402


def block_data(n=600, seed=0):
    """9列: 0〜4 はほぼ同じ1つの向き（相関 0.9 程度）、5〜8 は互いに独立。標準化してある。"""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 1))
    trend = f + 0.3 * rng.normal(size=(n, 5))
    other = rng.normal(size=(n, 4))
    Z = np.hstack([trend, other])
    return (Z - Z.mean(0)) / Z.std(0)


class Whitener(unittest.TestCase):
    def test_identity_and_mahalanobis(self):
        Z = block_data()
        W, shrink = E.whitener(Z)
        S, s2 = ledoit_wolf(Z)
        self.assertAlmostEqual(shrink, s2)
        self.assertTrue(0.0 <= shrink <= 1.0)
        np.testing.assert_allclose(W @ S @ W, np.eye(9), atol=1e-8)
        np.testing.assert_allclose(W, W.T, atol=1e-12)                 # 対称
        x, y = Z[3], Z[17]
        d_w = (((x - y) @ W) ** 2).sum()
        d_m = (x - y) @ np.linalg.inv(S) @ (x - y)
        self.assertAlmostEqual(d_w, d_m, places=8)

    def test_states_are_capped_integers(self):
        Z = block_data(n=300, seed=1)
        zs, _ = E.arm_states(Z)
        self.assertEqual(set(zs), {"E", "M"})
        np.testing.assert_array_equal(zs["E"], RN.quantize(Z))
        for a in ("E", "M"):
            self.assertEqual(zs[a].dtype, np.int64)
            self.assertLessEqual(int(np.abs(zs[a]).max()), int(RN.Z_CAP * RN.Q))


class Nearest(unittest.TestCase):
    def test_euclid_matches_production_knn(self):
        rng = np.random.default_rng(2)
        for n in (60, 150, 400):
            zq = RN.quantize(rng.normal(size=(n, 9)))
            order, cnt = RN.knn_order(RN.sq_dist(zq), k_max=10)
            want = order[-1][:cnt[-1]]
            np.testing.assert_array_equal(E.nearest(zq.astype(np.int64), kmax=10), want)

    def test_theiler_and_ties(self):
        zq = np.zeros((30, 2), dtype=np.int64)
        zq[:, 0] = 50
        zq[-1] = [0, 0]                       # 今日
        zq[25] = [0, 0]                       # タイラー窓（20ノード）の中: 選ばない
        zq[3] = [1, 0]                        # 距離1
        zq[7] = [0, 1]                        # 距離1（同じ距離なら古い日が先）
        zq[5] = [0, 2]                        # 距離4
        got = E.nearest(zq, kmax=3)
        self.assertEqual(got.tolist(), [3, 7, 5])
        self.assertNotIn(25, E.nearest(zq, kmax=8).tolist())
        self.assertTrue((E.nearest(zq, kmax=30) < 30 - 1 - E.TH).all())

    def test_duplicated_columns_count_once_in_m(self):
        # 今日（最後）と、5列の重なった向きだけで 0.5 ずつ違う日 A、独立な1列だけで 1.0 違う日 B
        Z = block_data(n=600, seed=3)
        Z = np.vstack([Z, np.zeros((1, 9))])
        a, b = 100, 200
        Z[a] = 0.0
        Z[a, :5] = 0.5                         # E: 5 × 0.25 = 1.25
        Z[b] = 0.0
        Z[b, 6] = 1.0                          # E: 1.0
        zs, _ = E.arm_states(Z)
        e_rank = E.nearest(zs["E"], kmax=len(Z)).tolist()
        m_rank = E.nearest(zs["M"], kmax=len(Z)).tolist()
        self.assertLess(e_rank.index(b), e_rank.index(a))           # E は 1列の違い（B）の方を近いとみる
        self.assertLess(m_rank.index(a), m_rank.index(b))           # M は重なった5列を1つ分に数え、A の方を近いとみる


class WithinAuc(unittest.TestCase):
    def test_only_pairs_within_code(self):
        df = pd.DataFrame({"code": ["A", "A", "B", "B"], "y": [1.0, 0.0, 1.0, 0.0],
                           "s1": [0.9, 0.8, 0.2, 0.1], "s2": [0.1, 0.2, 0.8, 0.9]})
        num, den = E.within_parts(df, ("s1", "s2"))
        np.testing.assert_allclose(num.sum(0) / den.sum(), [1.0, 0.0])   # 銘柄の中では s1 はどちらも並びどおり・s2 はどちらも逆
        self.assertAlmostEqual(E.auc(df["y"].to_numpy(), df["s1"].to_numpy()), 0.75)   # 全日まとめると銘柄の差が混ざる
        self.assertAlmostEqual(E.auc(df["y"].to_numpy(), df["s2"].to_numpy()), 0.25)

    def test_no_pairs_is_nan(self):
        df = pd.DataFrame({"code": ["A", "A"], "y": [1.0, 1.0], "p": [0.1, 0.2], "q": [0.3, 0.4]})
        self.assertTrue(all(np.isnan(v) for v in E.boot_within(df, "p", "q", n_boot=5)))

    def test_same_column_zero_diff(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"code": np.repeat([f"C{i}" for i in range(30)], 10), "y": rng.integers(0, 2, 300).astype(float),
                           "p": rng.random(300)})
        df["q"] = df["p"]
        a, b, d, lo, hi = E.boot_within(df, "p", "q", n_boot=50)
        self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(a, b)


def synth_rows(m_skill=True, n_codes=120, per=12, seed=5):
    """銘柄ごとに到達しやすさが違う作りもの。M は銘柄の中で y と関係し（m_skill）、E は関係しない。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_codes):
        base = rng.uniform(0.1, 0.6)
        for j in range(per):
            y = float(rng.random() < base)
            pe = rng.choice([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            pm = np.clip(0.5 * y + rng.normal(0.25, 0.2), 0, 1) if m_skill else rng.choice([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            rows.append({"code": f"9{i:03d}0", "date": pd.Timestamp("2019-01-04") + pd.Timedelta(days=150 * j + i),
                         "y": y, "base": base, "p5_E": pe, "p10_E": pe, "p5_M": pm, "p10_M": pm,
                         "per_E": 2, "per_M": 3, "ov5": 0.4, "shrink": 0.05, "n_nodes": 500})
    return pd.DataFrame(rows)


class Report(unittest.TestCase):
    def test_verdict_and_no_codes(self):
        for skill, want in ((True, "マハラノビスの方が当てる"), (False, "ユークリッドのまま")):
            df = synth_rows(m_skill=skill)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = E.report(df, n_boot=200)
            text = buf.getvalue()
            self.assertIn(want, text)
            self.assertEqual(out["main"]["better"], skill)
            for k in ("main", "within10", "pooled", "overlap5", "shrink_median"):
                self.assertIn(k, out)
            for code in df["code"].unique()[:50]:
                self.assertNotIn(code, text)

    def test_check_e107(self):
        df = synth_rows(n_codes=20)
        old = df.rename(columns={"base": "base_9", "p5_E": "p5_9", "p10_E": "p10_9"})[
            ["code", "date", "y", "base_9", "p5_9", "p10_9"]]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "e107_rows.parquet")
            old.to_parquet(p, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                res = E.check_e107(df, p)
            self.assertEqual((res["common"], res["p5_E"], res["y"]), (len(df), 1.0, 1.0))
            old.loc[:9, "p5_9"] = old.loc[:9, "p5_9"] + 0.5
            old.to_parquet(p, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                res = E.check_e107(df, p)
            self.assertAlmostEqual(res["p5_E"], 1 - 10 / len(df))


if __name__ == "__main__":
    unittest.main()
