#!/usr/bin/env python3
"""
実験62（research/exp/e62_stacking.py）のテスト。重い計算は回さない。

- 窓 k の meta は「前の窓の行」かつ「窓 k の訓練終了日以前」の行だけで学習する（先読みなし）
- 学習行が足りない窓は NaN、足りる窓は 0〜1 のスコア
- 日内百分位は 0〜1 で日の平均が 0.5、順位平均はその5モデルの平均
- logit は 0 / 1 でも有限
- 件数そろえは窓ごとに指定の件数だけ取る

  python3 tests/test_e62_stacking.py
"""
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e62_stacking as E  # noqa: E402

warnings.filterwarnings("ignore")


def toy(n_days=120, per_day=20, seed=0):
    """6窓 × 20日 × 20件。5モデルのスコアは、隠れた z にノイズを足したもの。"""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    for i, d in enumerate(dates):
        fold = i // 20 + 1
        z = rng.standard_normal(per_day)
        y = (z + 0.8 * rng.standard_normal(per_day) > 0.9).astype(int)
        for j in range(per_day):
            s = {f"s_{a}": 1 / (1 + np.exp(-(z[j] + rng.standard_normal() * (0.5 + 0.2 * k))))
                 for k, a in enumerate(E.ALGOS)}
            rows.append({"Code": f"{1000 + j}", "Date": d, "fold": fold, "label": int(y[j]),
                         "ret_o1_20": float(z[j] + rng.standard_normal()), "ret_o1_40": 0.0, **s})
    fr = pd.DataFrame(rows)
    # 窓 k の訓練終了日 = 窓 k の最初の日の 5営業日前（エンバーゴの代わり）
    first = fr.groupby("fold")["Date"].min()
    train_end = {int(k): d - pd.tseries.offsets.BDay(5) for k, d in first.items()}
    return fr, train_end


class Stacking(unittest.TestCase):
    def setUp(self):
        self.fr, self.train_end = toy()

    def test_train_rows_never_see_the_window_or_later(self):
        for k in sorted(self.train_end):
            tr = E.train_rows(self.fr, k, self.train_end[k])
            picked = self.fr[tr]
            self.assertTrue((picked["fold"] < k).all())
            self.assertTrue((picked["Date"] <= self.train_end[k]).all())
            # エンバーゴ: 窓 k−1 の末尾（訓練終了日より後）は入らない
            tail = self.fr[(self.fr["fold"] == k - 1) & (self.fr["Date"] > self.train_end[k])]
            self.assertFalse(tr[tail.index].any())

    def test_stack_scores_nan_until_enough_history(self):
        X, names = E.features(self.fr, "logit")
        sc, n_train, w = E.stack_scores(self.fr, X, names, "lr", self.train_end, min_train=400)
        f = self.fr["fold"].to_numpy()
        # 窓1 は学習行 0、窓2 は約 300行 → NaN。窓3 以降（≥ 400行）は 0〜1
        self.assertTrue(np.isnan(sc[f == 1]).all())
        self.assertTrue(np.isnan(sc[f == 2]).all())
        for k in (3, 4, 5, 6):
            self.assertTrue(np.isfinite(sc[f == k]).all())
            self.assertTrue(((sc[f == k] >= 0) & (sc[f == k] <= 1)).all())
            self.assertIn(k, w)
            self.assertEqual(set(w[k]), set(E.ALGOS))
        self.assertEqual(n_train[1], 0)
        self.assertLess(n_train[2], 400)

    def test_evaluated_folds_and_add_stacks(self):
        fr, n_train, weights = E.add_stacks(self.fr, self.train_end, min_train=400)
        self.assertEqual(E.evaluated_folds(n_train, 400), [3, 4, 5, 6])
        for key, _, _, learner in E.METHODS:
            self.assertIn(f"m_{key}", fr.columns)
            if learner:
                self.assertTrue(np.isfinite(fr.loc[fr["fold"] >= 3, f"m_{key}"]).all())
        # 順位平均は学習なしなので全窓にある
        self.assertTrue(np.isfinite(fr["m_rank_avg"]).all())
        # 学習した meta は、信号のある toy では LightGBM 単体より ROC が高い（窓3以降）
        from sklearn.metrics import roc_auc_score
        ev = fr[fr["fold"] >= 3]
        self.assertGreater(roc_auc_score(ev["label"], ev["m_lr_logit"]), roc_auc_score(ev["label"], ev["s_lgbm"]) - 0.02)

    def test_features(self):
        X, names = E.features(self.fr, "rank")
        self.assertEqual(names, list(E.ALGOS))
        self.assertTrue(((X > 0) & (X < 1)).all())
        day_mean = pd.DataFrame(X).groupby(self.fr["Date"].to_numpy()).mean()
        np.testing.assert_allclose(day_mean.to_numpy(), 0.5, atol=1e-9)
        np.testing.assert_allclose(E.rank_average(self.fr), X.mean(axis=1))
        X3, names3 = E.features(self.fr, "logit3")
        self.assertEqual(names3, list(E.BOOST))
        self.assertEqual(X3.shape[1], 3)
        self.assertTrue(np.isfinite(E.logit(np.array([0.0, 1.0, 0.5]))).all())
        self.assertAlmostEqual(float(E.logit(np.array([0.5]))[0]), 0.0)
        Xp, _ = E.features(self.fr, "pct")
        self.assertTrue(np.isnan(Xp[0]).all())              # 過去の行が無い最初の日は NaN
        self.assertTrue(np.isfinite(Xp[-1]).all())

    def test_matched_takes_the_given_count_per_fold(self):
        base = self.fr.copy()
        base["score"] = base["s_lgbm"]
        base["p_x"] = base["s_xgb"] * 100
        out = E.matched(base, "p_x", {3: 4, 4: 2})
        self.assertEqual(out.groupby("fold").size().to_dict(), {3: 4, 4: 2})
        top3 = base[base["fold"] == 3].nlargest(4, "p_x")["Code"].tolist()
        self.assertEqual(sorted(out[out["fold"] == 3]["Code"]), sorted(top3))

    def test_average_frames_checks_rows(self):
        fr2 = self.fr.copy()
        fr2["s_lgbm"] = fr2["s_lgbm"] + 0.1
        avg = E.average_frames([self.fr, fr2])
        np.testing.assert_allclose(avg["s_lgbm"], self.fr["s_lgbm"] + 0.05)
        bad = self.fr.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(SystemExit):
            E.average_frames([self.fr, bad])


if __name__ == "__main__":
    unittest.main(verbosity=2)
