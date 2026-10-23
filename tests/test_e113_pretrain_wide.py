#!/usr/bin/env python3
"""
実験113（research/exp/e113_pretrain_wide.py）の部品。データには触らない（手で確かめられる小さな合成の例）。

- 学習器: 重みも始めの点数も無ければ、本番の組み方（LightGBM は e112.full_lgbm、XGBoost・CatBoost は models.fit）と同じ点数。
  始めの点数（生の値）を渡すと、予測にもその値が足される（定数 ±3 で始めても、平均の確率がふつうの学習とそろう）
- 窓ごとの組み立て: A は T だけ、W・P は T と「学習の終わりまでの S」を混ぜ、S の重みの合計 = T の行数。F は S だけで学んだ点数を
  T の学習・テストの行に付けてから T で学ぶ。採点はどれも T のテストの行だけ
- 広いデータセット: T の行がすべてあって値が同じなら S を返し（T の行は除く）、足りない・食い違うなら止める
- 広いデータセットを作る間だけ「更新日すべて」を候補にし、終われば（失敗しても）元に戻す
- 対照のラベルの入れ替えは同じ日付の中だけ（日付ごとの正例の数は変わらない）

  python3 tests/test_e113_pretrain_wide.py
"""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import e112_chronos2_cols as E112  # noqa: E402
import e113_pretrain_wide as E  # noqa: E402
import models as M  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
from e44_shortsale import permuted  # noqa: E402

PARAMS = {"lgbm": {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10},
          "xgb": {"max_depth": 3, "learning_rate": 0.1},
          "cat": {"depth": 4, "learning_rate": 0.1}}


def data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + rng.normal(size=n) > 1.0).astype(int)
    return X, y


class FitPredict(unittest.TestCase):
    def setUp(self):
        old = TM.N_ESTIMATORS
        TM.N_ESTIMATORS = 30            # 速くする（本番の組み方の固定の値は同じ道を通る）
        self.addCleanup(setattr, TM, "N_ESTIMATORS", old)
        self.X, self.y = data()
        self.Xte, _ = data(400, seed=1)

    def test_plain_matches_production_build(self):
        import lightgbm as lgb

        p = {**PARAMS["lgbm"], "n_estimators": 30}
        m = lgb.LGBMClassifier(**{**E112.full_lgbm(p), "random_state": 7}, scale_pos_weight=tuning.scale_pos_weight(self.y))
        m.fit(self.X, self.y)
        np.testing.assert_allclose(E.sigmoid(E.fit_predict("lgbm", p, 7, self.X, self.y, self.Xte)),
                                   m.predict_proba(self.Xte)[:, 1], rtol=1e-12, atol=1e-12)
        for a in ("xgb", "cat"):
            old = TM.SEED
            try:
                TM.SEED = 7
                want = M.predict(M.fit(a, self.X, self.y, [f"c{i}" for i in range(6)], params=PARAMS[a]), self.Xte)
            finally:
                TM.SEED = old
            np.testing.assert_allclose(E.sigmoid(E.fit_predict(a, PARAMS[a], 7, self.X, self.y, self.Xte)), want,
                                       rtol=1e-6, atol=1e-6, err_msg=a)
            self.assertEqual(TM.SEED, old)                                   # 種は元に戻る

    def test_start_score_is_added_back(self):
        for a, p in PARAMS.items():
            plain = E.sigmoid(E.fit_predict(a, p, 42, self.X, self.y, self.Xte)).mean()
            for c in (3.0, -3.0):
                got = E.sigmoid(E.fit_predict(a, p, 42, self.X, self.y, self.Xte, init_tr=np.full(len(self.X), c),
                                              init_te=np.full(len(self.Xte), c))).mean()
                self.assertLess(abs(got - plain), 0.03, msg=f"{a} {c}")      # 足し忘れ・二重に足すと 0.02 や 0.9 になる


class FoldAssembly(unittest.TestCase):
    """fit_predict を差し替えて、窓ごとに何を学習に渡したかを見る。"""

    def setUp(self):
        days = pd.bdate_range("2019-01-01", "2023-12-29")
        self.cols = ["f0", "f1"]
        t = pd.DataFrame({"Code": np.tile(["1110", "2220"], len(days)), "Date": np.repeat(days, 2)})
        s = pd.DataFrame({"Code": np.tile(["3330", "4440", "5550"], len(days)), "Date": np.repeat(days, 3)})
        rng = np.random.default_rng(3)
        for d in (t, s):
            d["Date"] = d["Date"].astype("datetime64[ns]")
            d["label"] = rng.integers(0, 2, len(d))
            for c in self.cols:
                d[c] = rng.normal(size=len(d))
        for c in ("ret_o1_20", "ret_o1_40", "hit10", "tp10c"):
            t[c] = 0.0
        self.t, self.s = t, s
        self.folds = [SimpleNamespace(index=1, train_end="2022-06-30", test_start="2022-08-01", test_end="2023-01-31"),
                      SimpleNamespace(index=2, train_end="2022-12-31", test_start="2023-02-01", test_end="2023-07-31")]
        self.calls = []
        orig = E.fit_predict

        def stub(algo, params, seed, Xtr, ytr, Xte, weight=None, init_tr=None, init_te=None):
            self.calls.append({"n": len(Xtr), "y": ytr, "nte": len(Xte), "w": weight, "init_tr": init_tr, "init_te": init_te})
            return np.zeros(len(Xte))

        E.fit_predict = stub
        self.addCleanup(setattr, E, "fit_predict", orig)

    def counts(self, f):
        nt = int((self.t["Date"] <= pd.Timestamp(f.train_end)).sum())
        ns = int((self.s["Date"] <= pd.Timestamp(f.train_end)).sum())
        te = (self.t["Date"] >= pd.Timestamp(f.test_start)) & (self.t["Date"] <= pd.Timestamp(f.test_end))
        return nt, ns, int(te.sum())

    def test_a_uses_target_only(self):
        o = E.fold_oof("lgbm", "A", self.t, self.s, self.cols, {}, 42, self.folds, self.s["label"].to_numpy())
        for c, f in zip(self.calls, self.folds):
            nt, _, nte = self.counts(f)
            self.assertEqual((c["n"], c["nte"]), (nt, nte))
            self.assertIsNone(c["w"])
        self.assertEqual(sorted(o["fold"].unique()), [1, 2])
        self.assertTrue(o.set_index(["Code", "Date"]).index.isin(self.t.set_index(["Code", "Date"]).index).all())

    def test_w_mixes_with_balanced_weight(self):
        E.fold_oof("lgbm", "W", self.t, self.s, self.cols, {}, 42, self.folds, self.s["label"].to_numpy())
        for c, f in zip(self.calls, self.folds):
            nt, ns, nte = self.counts(f)
            self.assertEqual((c["n"], c["nte"]), (nt + ns, nte))      # S は学習の終わりまでの行だけ
            np.testing.assert_allclose(c["w"][:nt], 1.0)
            self.assertAlmostEqual(float(c["w"][nt:].sum()), float(nt))  # S の重みの合計 = T の行数

    def test_p_uses_the_given_source_labels(self):
        yp = 1 - self.s["label"].to_numpy()
        E.fold_oof("lgbm", "P", self.t, self.s, self.cols, {}, 42, self.folds[:1], yp)
        nt, ns, _ = self.counts(self.folds[0])
        np.testing.assert_array_equal(self.calls[0]["y"][nt:], yp[:ns])     # S は日付の順に並んでいる

    def test_f_starts_from_source_scores(self):
        o = E.fold_oof("lgbm", "F", self.t, self.s, self.cols, {}, 42, self.folds[:1], self.s["label"].to_numpy())
        nt, ns, nte = self.counts(self.folds[0])
        first, second = self.calls
        self.assertEqual((first["n"], first["nte"]), (ns, nt + nte))   # S だけで学び、T の学習・テストの行に点数を付ける
        self.assertIsNone(first["init_tr"])
        self.assertEqual((second["n"], len(second["init_tr"]), len(second["init_te"])), (nt, nt, nte))
        np.testing.assert_allclose(o["score_S"], 0.5)


class Wide(unittest.TestCase):
    def frames(self):
        t = pd.DataFrame({"Code": ["1110", "2220"], "Date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
                          "label": [1.0, 0.0], "f0": [0.5, 1.5]})
        w = pd.DataFrame({"Code": ["1110", "1110", "2220"], "Date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-05"]),
                          "label": [1.0, 0.0, 0.0], "f0": [0.5, 9.0, 1.5]})
        return t, w

    def test_split_returns_source_without_target(self):
        t, w = self.frames()
        s, chk = E.split_wide(w, t, ["f0"])
        self.assertEqual(list(zip(s["Code"], s["Date"].dt.strftime("%m-%d"))), [("1110", "01-05")])
        self.assertEqual((chk["found_in_wide"], chk["source_rows"], chk["bad_cells"]), (2, 1, 0))

    def test_split_stops_on_mismatch_or_missing(self):
        t, w = self.frames()
        w2 = w.copy()
        w2.loc[2, "f0"] = 1.6
        with self.assertRaises(SystemExit):
            E.split_wide(w2, t, ["f0"])
        w3 = w.copy()
        w3.loc[0, "label"] = 0.0
        with self.assertRaises(SystemExit):
            E.split_wide(w3, t, ["f0"])
        with self.assertRaises(SystemExit):
            E.split_wide(w.iloc[1:], t, ["f0"])

    def test_build_wide_takes_every_high_and_restores(self):
        df = pd.DataFrame({"Code": ["1110"] * 6, "high": [1, 2, 3, 3, 4, 5.0],
                           "high52w_prior": [np.nan, 1, 2, 3, 3, 4.0]})
        seen = {}
        orig_build, orig_mark, orig_cd = B.build, B.mark_new_highs, B.BREAKOUT_COOLDOWN

        def fake_build(data_dir, out_path):
            out = B.mark_new_highs(df.copy())
            seen["fresh"] = out["is_fresh_break"].tolist()
            seen["new"] = out["is_new_high"].tolist()
            seen["cd"] = B.BREAKOUT_COOLDOWN

        B.build = fake_build
        try:
            E.build_wide("x", "y.parquet")
            self.assertEqual(seen["fresh"], seen["new"])
            self.assertEqual(seen["new"], [False, True, True, False, True, True])
            self.assertEqual(seen["cd"], 0)
            self.assertIs(B.mark_new_highs, orig_mark)
            self.assertEqual(B.BREAKOUT_COOLDOWN, orig_cd)

            def broken(data_dir, out_path):
                raise RuntimeError("x")

            B.build = broken
            with self.assertRaises(RuntimeError):
                E.build_wide("x", "y.parquet")
            self.assertIs(B.mark_new_highs, orig_mark)                    # 失敗しても戻る
            self.assertEqual(B.BREAKOUT_COOLDOWN, orig_cd)
        finally:
            B.build = orig_build
        # 今のコードでは cooldown 0 は通らない（だから差し替える）
        with self.assertRaises(ValueError):
            orig_mark(df.copy(), cooldown=0)


class Control(unittest.TestCase):
    def test_permutation_keeps_daily_positives(self):
        d = pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"] * 4 + ["2024-01-05"] * 3), "label": [1, 0, 0, 1, 0, 0, 1]})
        p = permuted(d, ["label"], seed=E.PERM_SEED)
        self.assertEqual(p.groupby("Date")["label"].sum().tolist(), d.groupby("Date")["label"].sum().tolist())

    def test_with_s(self):
        o = pd.DataFrame({"Code": ["a"], "score": [0.2], "score_S": [0.7]})
        self.assertEqual(E.with_s(o)["score"].tolist(), [0.7])


if __name__ == "__main__":
    unittest.main()
