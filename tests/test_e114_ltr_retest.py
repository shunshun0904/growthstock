#!/usr/bin/env python3
"""
実験114（research/exp/e114_ltr_retest.py）。合成データで:

- 今の売り方の収益: 指値に届いた行は +10%、届かない行は20営業日目の終値、20営業日に満たない行は空（手で計算した値と一致）。
  L が学ぶ20営業日目の終値は指値に関係なく20営業日目の終値
- 段階（同じ日の中の5分位）: 同じ収益は同じ段階・1件だけの日は真ん中
- NDCG@5: 手で計算した値・理想の並びで 1・候補の少ない日は数えない
- LambdaRank: 20営業日目の終値の順を学ぶ（でたらめな順より NDCG が高い）・同じ種なら同じスコア・入力の行の並びに左右されない
- out-of-fold: 窓は基準モデルの out-of-fold と同じ作り・両腕で同じ行を採点・保存したものを読み直すと同じ
- 件数をそろえた選定（窓の中の上位10%・基準モデルの95以上と同じ件数）と判定（すべてのずらし方で収益が上かつ大負けが少ない）

  python3 tests/test_e114_ltr_retest.py
"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e41_stop_loss as E41  # noqa: E402
import e114_ltr_retest as E  # noqa: E402

#: 単体テスト用の小さな木（探索の結果の形に合わせる）
SMALL = {"boosting_type": "gbdt", "n_estimators": 40, "learning_rate": 0.1, "num_leaves": 7,
         "min_child_samples": 10, "subsample": 1.0, "subsample_freq": 1, "colsample_bytree": 1.0,
         "reg_alpha": 1e-6, "reg_lambda": 1e-6, "n_jobs": 1, "verbose": -1}
FEATS = ["x0", "x1", "x2", "x3"]


def synth(start="2018-04-02", end="2023-06-30", per_day=3, seed=0) -> pd.DataFrame:
    """データセットの形の作りもの。20営業日目の終値は x0 が大きいほど高い（ノイズあり）。今の売り方はそれを +10% で頭打ち。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    rows = []
    for i, d in enumerate(days):
        for j in range(per_day):
            rows.append({"Code": f"{1000 + j + per_day * (i % 7)}", "Date": d})
    df = pd.DataFrame(rows).drop_duplicates(["Code", "Date"]).reset_index(drop=True)
    n = len(df)
    X = rng.normal(size=(n, len(FEATS)))
    for k, c in enumerate(FEATS):
        df[c] = X[:, k]
    ret = 0.04 * X[:, 0] + rng.normal(0, 0.03, n)
    df[E.RET] = np.minimum(ret, 0.10)
    df[E.R20] = ret
    df[E.HIT] = ret >= 0.10
    df["label"] = (ret > 0.02).astype(float)
    df[E.OUTCOME] = ret
    df["vol_20d"] = rng.lognormal(0.8, 0.5, n)
    return df.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)


class TestExit(unittest.TestCase):
    def test_hand_computed_exit(self):
        days = pd.bdate_range("2024-01-01", periods=30)
        o = np.full(30, 100.0)
        h = np.full(30, 101.0)
        c = np.full(30, 100.0)
        h[5] = 111.0          # D0 の候補（買いは D1）は5営業日目に +10% に届く
        c[27] = 95.0          # D7 の候補（買いは D8）は届かず、20営業日目（D27）の終値で −5%
        bars = pd.DataFrame({"Date": days, "Code": "1001", "AdjO": o, "AdjH": h, "AdjC": c})
        df = pd.DataFrame({"Code": ["1001"] * 3, "Date": [days[0], days[7], days[20]],
                           E.OUTCOME: [0.0, 0.0, np.nan]})
        res, chk = E.attach_exit(df, bars)
        self.assertAlmostEqual(res.loc[0, E.RET], 0.10)
        self.assertTrue(bool(res.loc[0, E.HIT]))
        # 指値に届いた行でも、L が学ぶのは20営業日目（D20）の終値 100 ÷ 買値 100 − 1 = 0
        self.assertAlmostEqual(res.loc[0, E.R20], 0.0)
        self.assertAlmostEqual(res.loc[1, E.RET], -0.05)
        self.assertAlmostEqual(res.loc[1, E.R20], -0.05)
        self.assertFalse(bool(res.loc[1, E.HIT]))
        # D20 の候補は20営業日目が日足に無い（保有中）ので収益は空
        self.assertTrue(math.isnan(res.loc[2, E.RET]))
        self.assertTrue(math.isnan(res.loc[2, E.R20]))
        self.assertEqual(len(res), 3)
        self.assertEqual(chk["rows"], 3)


class TestGradesNdcg(unittest.TestCase):
    def test_grades(self):
        dates = ["a"] * 5 + ["b"]
        r = [0.10, 0.10, -0.2, 0.05, 0.0, 0.03]
        g = E.grades(dates, r)
        self.assertEqual(list(g), [4, 4, 0, 2, 1, 2])
        self.assertTrue(set(np.unique(g)) <= set(range(E.N_GRADES)))

    def test_ndcg_hand(self):
        gains = np.array([4, 4, 2, 1, 0], dtype=float)
        dates = ["d"] * 5
        self.assertAlmostEqual(E.ndcg_at_k(dates, gains, gains), 1.0)
        # スコアで 0 の段階を先頭に置いた並び: 0, 4, 4, 2, 1
        score = np.array([4, 3, 2, 1, 5], dtype=float)
        disc = 1 / np.log2(np.arange(2, 7))
        dcg = (np.array([0, 4, 4, 2, 1]) * disc).sum()
        idcg = (np.array([4, 4, 2, 1, 0]) * disc).sum()
        self.assertAlmostEqual(E.ndcg_at_k(dates, score, gains), dcg / idcg)
        self.assertLess(E.ndcg_at_k(dates, -gains, gains), 1.0)

    def test_ndcg_skips_small_days(self):
        self.assertTrue(math.isnan(E.ndcg_at_k(["d"] * 3, [1, 2, 3], [0, 1, 2])))
        # 段階がすべて 0 の日（理想の DCG が 0）は数えない
        v = E.ndcg_at_k(["a"] * 5 + ["b"] * 5, [0, 1, 2, 3, 4, 9, 8, 7, 6, 5], [0] * 5 + [4, 3, 2, 1, 0])
        self.assertAlmostEqual(v, 1.0)


class TestFit(unittest.TestCase):
    def setUp(self):
        self.df = synth("2020-01-01", "2021-06-30", per_day=8)
        d = self.df["Date"]
        cut = d.unique()[250]
        self.tr = self.df[d <= cut]
        self.te = self.df[d > cut]

    def test_ranker_learns_return_order(self):
        s = E.fit_score("L", SMALL, self.tr, self.te, FEATS, seed=42)
        g = E.grades(self.te["Date"], self.te[E.R20])
        nd = E.ndcg_at_k(self.te["Date"], s, g)
        rnd = E.ndcg_at_k(self.te["Date"], np.random.default_rng(0).random(len(self.te)), g)
        self.assertGreater(nd, rnd + 0.1)
        rho = E.E53.date_rho(self.te["Date"], s, self.te[E.R20].to_numpy(dtype=float), min_n=5)
        self.assertGreater(rho, 0.3)

    def test_ranker_uses_r20_not_capped_return(self):
        # 学ぶ並びを入れ替えるとスコアが変わる（今の売り方の収益ではなく20営業日目の終値で学んでいる）
        tr2 = self.tr.copy()
        tr2[E.R20] = -tr2[E.R20]
        a = E.fit_score("L", SMALL, self.tr, self.te, FEATS, seed=42)
        b = E.fit_score("L", SMALL, tr2, self.te, FEATS, seed=42)
        self.assertLess(np.corrcoef(a, b)[0, 1], 0)

    def test_ranker_deterministic_and_order_free(self):
        a = E.fit_score("L", SMALL, self.tr, self.te, FEATS, seed=7)
        b = E.fit_score("L", SMALL, self.tr, self.te, FEATS, seed=7)
        shuffled = self.tr.sample(frac=1.0, random_state=3)
        c = E.fit_score("L", SMALL, shuffled, self.te, FEATS, seed=7)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, c)

    def test_classifier_scores_are_probabilities(self):
        s = E.fit_score("C", {**SMALL, "objective": "binary", "random_state": 0}, self.tr, self.te, FEATS, seed=42)
        self.assertEqual(len(s), len(self.te))
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_tree_only(self):
        p = E.tree_only({"objective": "binary", "random_state": 0, "scale_pos_weight": 3.0, "_x": 1,
                         "num_leaves": 7})
        self.assertEqual(p, {"num_leaves": 7})


class TestOof(unittest.TestCase):
    def test_rows_windows_and_cache(self):
        df = synth()
        with tempfile.TemporaryDirectory() as tmp:
            oc = E.oof_arm(df, FEATS, "C", {**SMALL, "objective": "binary"}, 0, (42,), out_dir=tmp)
            ol = E.oof_arm(df, FEATS, "L", SMALL, 0, (42,), out_dir=tmp)
            np.testing.assert_array_equal(oc[["Code", "Date"]].astype(str).to_numpy(),
                                          ol[["Code", "Date"]].astype(str).to_numpy())
            folds = E41.folds_for(df["Date"], 0)
            used = sorted(oc["fold"].unique())
            self.assertGreaterEqual(len(used), 2)
            for f in folds:
                if f.index in used:
                    g = oc[oc["fold"] == f.index]
                    self.assertGreaterEqual(g["Date"].min(), pd.Timestamp(f.test_start))
                    self.assertLessEqual(g["Date"].max(), pd.Timestamp(f.test_end))
            again = E.oof_arm(df, FEATS, "L", SMALL, 0, (42,), out_dir=tmp)
            np.testing.assert_array_equal(again["score"].to_numpy(), ol["score"].to_numpy())
            # 木の形が変われば別の保存名になる（古いものを読まない）
            self.assertNotEqual(E.run_tag("L", SMALL, FEATS, df), E.run_tag("L", {**SMALL, "num_leaves": 15}, FEATS, df))


def toy_oof(seed=0, n_fold=100, folds=(1, 2, 3)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for f in folds:
        for i in range(n_fold):
            rows.append({"fold": f, "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(f * 50 + i // 10)),
                         "score": rng.random(), E.RET: rng.normal(0.01, 0.05), E.HIT: False, "label": 0.0})
    return pd.DataFrame(rows)


class TestSelection(unittest.TestCase):
    def test_top_within_counts_and_picks(self):
        a, b = toy_oof(0), toy_oof(1)
        ta, tb = E.top_within(a, 0.10), E.top_within(b, 0.10)
        self.assertEqual(list(ta["n"]), [10, 10, 10])
        self.assertEqual(list(ta["n"]), list(tb["n"]))
        g = a[a["fold"] == 2].sort_values("score", ascending=False).head(10)
        self.assertAlmostEqual(float(ta.loc[ta["fold"] == 2, "ret"].iloc[0]), float(g[E.RET].mean()))
        self.assertAlmostEqual(float(ta.loc[ta["fold"] == 2, "bad"].iloc[0]), float((g[E.RET] < E.BAD).mean()))

    def test_forward_counts_and_matched(self):
        a = toy_oof(0)
        cnt = E.forward_counts(a, 95.0, min_ref=50)
        self.assertEqual(sorted(cnt), [2, 3])
        ref = a.loc[a["fold"] < 3, "score"].to_numpy()
        self.assertEqual(cnt[3], int((a.loc[a["fold"] == 3, "score"] > np.percentile(ref, 95)).sum()))
        m = E.matched(toy_oof(1), cnt)
        self.assertEqual(dict(zip(m["fold"], m["n"])), {f: n for f, n in cnt.items() if n > 0})

    def test_within_date_top(self):
        o = pd.DataFrame({"Date": [pd.Timestamp("2024-01-05")] * 8, "score": np.arange(8, dtype=float),
                          E.RET: [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.10], E.HIT: False, "label": 0.0})
        w = E.within_date(o)
        self.assertAlmostEqual(w["top1"], 0.10)
        self.assertAlmostEqual(w["top2"], 0.08)
        self.assertEqual(w["days"], 1)
        self.assertGreater(w["rho"], 0.99)

    def test_fold_diff(self):
        base = pd.DataFrame({"fold": [1, 2, 3], "ret": [0.01, 0.02, 0.03]})
        cur = pd.DataFrame({"fold": [1, 2, 3], "ret": [0.02, 0.01, 0.05]})
        d = E.fold_diff(base, cur)
        self.assertAlmostEqual(d["mean"], (0.01 - 0.01 + 0.02) / 3)
        self.assertEqual((d["won"], d["n"]), (2, 3))


class TestVerdict(unittest.TestCase):
    @staticmethod
    def frame(ret, bad):
        return pd.DataFrame({"fold": range(len(ret)), "ret": ret, "bad": bad})

    def test_all_shifts_needed(self):
        good = {"C": self.frame([0.01, 0.02], [0.10, 0.10]), "L": self.frame([0.02, 0.03], [0.05, 0.05])}
        self.assertTrue(E.verdict({0: good, 2: good, 4: good})["pass"])
        worse_ret = {"C": self.frame([0.02, 0.03], [0.10, 0.10]), "L": self.frame([0.01, 0.02], [0.05, 0.05])}
        v = E.verdict({0: good, 2: worse_ret, 4: good})
        self.assertFalse(v["pass"])
        self.assertEqual([r["ok"] for r in v["rows"]], [True, False, True])

    def test_bad_must_be_lower(self):
        same_bad = {"C": self.frame([0.01], [0.10]), "L": self.frame([0.02], [0.10])}
        self.assertFalse(E.verdict({0: same_bad})["pass"])
        self.assertFalse(E.verdict({})["pass"])


if __name__ == "__main__":
    unittest.main()
