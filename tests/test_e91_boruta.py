#!/usr/bin/env python3
"""
実験91（research/exp/e91_boruta.py）の、重い計算を回さない部分のテスト。

- 回す前に決めたもの（モデル・種・ラベル・Boruta の設定）と、モデルの名前の受け方・パラメータの読み方
- 日付単位の列の見つけ方・影の単位（1行の日は月、それでも1行なら年でまとめる）・入れ替えが単位の中に収まる・
  日付ごとの入れ替え
- Boruta の判定（BorutaPy の二段の多重比較と同じ数）と本体（効く列は confirmed、でたらめな列は rejected、
  影との比べ方は列の種類ごと）。小さい LightGBM でも同じ
- ずらした窓に使う選定は訓練の終わりがそれ以前の窓（先読みにならない）・腕ごとの列（S は rejected を外す、
  R は同じ本数をでたらめに外す、窓ごとに決まる）
- 入れ替え重要度の向き（効く列は正、使っていない列は 0）
- 合成の結果で、表（選定・本番と同じ OOF・窓ごと・判定・窓ごとの値・診断）が最後まで出て、要約が保存される
- 時間の上限を過ぎたら新しい計算を始めない

  python3 tests/test_e91_boruta.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import features as F  # noqa: E402
import tuning_multi as TM  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e91_boruta as E  # noqa: E402


class TestLineup(unittest.TestCase):
    def test_decided_before_running(self):
        self.assertEqual(E.ALGOS, ("lgbm_k15", "xgb_k15", "cat_k15"))
        self.assertEqual(E.SELECT_ALGO, "lgbm_k15")
        self.assertEqual(E.SEEDS, (42, 7, 123))
        self.assertEqual(E.K, 1.5)
        self.assertEqual(E.ARMS, ("A", "S", "R"))
        # BorutaPy 0.4.3 の既定と同じ（max_iter=100・alpha=0.05・perc=100）
        self.assertEqual((E.MAX_ITER, E.ALPHA, E.PERC), (100, 0.05, 100))
        self.assertEqual((E.BORUTA_SEED, E.R_SEED, E.PI_SEED), (20261014, 20261015, 20261016))
        self.assertEqual(E.MIN_DATE_ROWS, 2)
        self.assertTrue(all(TM.label_k(a) == 1.5 for a in E.ALGOS))
        # 影の作りは date が回す前に決めたもの（既定）。month・global は試運転のあとに足した選択肢
        self.assertEqual(E.SHADOW_MODES, ("date", "month", "global"))
        self.assertEqual(E.SHADOW, "date")

    def test_perc_option(self):
        saved = (E.PERC_IN_USE, E.SHADOW)
        try:
            self.assertEqual(E.variant(), "date")                            # 登録した版はファイル名を変えない
            E.PERC_IN_USE = 75.0
            self.assertEqual(E.variant(), "date_p75")
            E.SHADOW = "global"
            self.assertEqual(E.variant(), "global_p75")
        finally:
            E.PERC_IN_USE, E.SHADOW = saved
        for bad in ("50", "40", "101"):
            with self.assertRaises(SystemExit):
                E.main(["--perc", bad, "--dry"])

    def test_algo_names(self):
        self.assertEqual([E.algo_name(a) for a in ("lgbm", "xgb", "cat", "cat_k15")],
                         ["lgbm_k15", "xgb_k15", "cat_k15", "cat_k15"])
        self.assertEqual(E.short_name("xgb_k15"), "xgb")
        for bad in ("logit", "mlp", "lgbm_k18"):
            with self.assertRaises(SystemExit):
                E.algo_name(bad)

    def test_prod_params_reads_the_239_column_k15_search_only(self):
        base = F.columns(F.DEFAULT_PRESET)
        d = tempfile.mkdtemp()
        saved = TM.PARAMS_PATH
        try:
            p = os.path.join(d, "multi_params.json")
            good = {"params": {"num_leaves": 31}, "_cv": {"features_sig": F.signature(base), "label_k": 1.5}}
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"lgbm_k15": good,
                           "xgb_k15": {**good, "_cv": {"features_sig": "000000000000", "label_k": 1.5}},
                           "cat_k15": {**good, "_cv": {"features_sig": F.signature(base), "label_k": 1.2}}}, fh)
            TM.PARAMS_PATH = p
            self.assertEqual(E.prod_params("lgbm_k15"), {"num_leaves": 31})
            for a in ("xgb_k15", "cat_k15"):                                  # 列か k が違う探索は使わない
                with self.assertRaises(SystemExit):
                    E.prod_params(a)
        finally:
            TM.PARAMS_PATH = saved
            shutil.rmtree(d, ignore_errors=True)


class TestShadows(unittest.TestCase):
    def test_date_level_columns(self):
        dates = pd.to_datetime(["2024-01-04"] * 3 + ["2024-01-05"] * 2 + ["2024-01-09"])
        df = pd.DataFrame({"Date": dates,
                           "mkt": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],          # その日は同じ・日で違う
                           "stk": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],          # 日の中で違う
                           "const": [7.0] * 6,                              # 全期間で同じ（日付単位とは言わない）
                           "mkt_nan": [np.nan] * 3 + [0.5, 0.5, 0.7],       # 欠測の日も同じ値として数える
                           "mixed": [np.nan, 1.0, 1.0, 2.0, 2.0, 3.0]})     # 日の中で欠測と値が混ざる
        self.assertEqual(E.date_level_columns(df, ["stk", "mkt", "const", "mkt_nan", "mixed"]), ["mkt", "mkt_nan"])
        self.assertEqual(E.date_level_columns(df.iloc[:0], ["mkt"]), [])

    def test_shadow_groups_pool_single_row_days(self):
        dates = pd.to_datetime(["2024-01-04"] * 3                 # 3行の日: その日
                               + ["2024-01-05", "2024-01-09"]      # 1行の日が同じ月に2つ: 月でまとめる
                               + ["2024-02-01", "2024-03-01"]      # 1行の日が月に1つずつ: 年でまとめる
                               + ["2024-04-01"] * 2)               # 2行の日: その日
        g = E.shadow_groups(dates)
        self.assertEqual(len(g), 9)
        self.assertEqual(len(set(g[:3])), 1)
        self.assertEqual(g[3], g[4])
        self.assertEqual(g[5], g[6])
        self.assertEqual(g[7], g[8])
        self.assertEqual(len({g[0], g[3], g[5], g[7]}), 4)
        self.assertEqual(len(E.shadow_groups([])), 0)
        # month: 同じ月どうし（1行しか無い月は同じ年の1行の月どうし）。global: 全行で1つ
        gm = E.shadow_groups(dates, "month")
        self.assertEqual(len(set(gm[:5])), 1)                               # 1月の5行
        self.assertEqual(gm[5], gm[6])                                       # 2月・3月は1行ずつなので年でまとめる
        self.assertNotEqual(gm[0], gm[5])
        self.assertEqual(len(set(gm[7:])), 1)                                # 4月の2行
        np.testing.assert_array_equal(E.shadow_groups(dates, "global"), np.zeros(9))

    def setUp(self):
        rng = np.random.default_rng(0)
        self.dates = pd.to_datetime(np.repeat(pd.bdate_range("2024-01-04", periods=40), rng.integers(1, 12, 40)))
        self.n = len(self.dates)
        self.sh = E.Shuffler(self.dates)

    def test_within_stays_in_its_unit_and_is_a_permutation(self):
        rng = np.random.default_rng(1)
        moved = 0
        for _ in range(20):
            perm = self.sh.within(rng)
            np.testing.assert_array_equal(np.sort(perm), np.arange(self.n))
            np.testing.assert_array_equal(self.sh.groups[perm], self.sh.groups)
            moved += int((perm != np.arange(self.n)).sum())
        self.assertGreater(moved, self.n)                                   # 自分のままの行ばかりではない

    def test_by_date_keeps_one_value_per_day_and_permutes_days(self):
        rng = np.random.default_rng(2)
        d = pd.Series(self.dates)
        day_value = {k: float(i) for i, k in enumerate(sorted(d.unique()))}
        v = d.map(day_value).to_numpy()
        for _ in range(5):
            out = v[self.sh.by_date(rng)]
            per_day = pd.Series(out).groupby(d.to_numpy()).nunique()
            self.assertTrue((per_day == 1).all())                           # その日は全行同じ値のまま
            got = pd.Series(out).groupby(d.to_numpy()).first().sort_values().to_numpy()
            np.testing.assert_array_equal(got, np.sort(np.array(list(day_value.values()))))  # 日の値の並べ替え

    def test_shadows_by_kind(self):
        rng = np.random.default_rng(3)
        d = pd.Series(self.dates)
        X = np.column_stack([rng.standard_normal(self.n), d.map(d.value_counts()).to_numpy(dtype=float)
                             + d.dt.day.to_numpy()])
        S = E.shadows(X, self.sh, np.array([False, True]), rng)
        g = self.sh.groups
        for k in np.unique(g):                                              # 銘柄ごとの列: 単位の中の値の集まりは同じ
            np.testing.assert_array_equal(np.sort(S[g == k, 0]), np.sort(X[g == k, 0]))
        self.assertTrue((pd.Series(S[:, 1]).groupby(d.to_numpy()).nunique() == 1).all())
        self.assertFalse(np.array_equal(S[:, 0], X[:, 0]))


class TestBorutaDecisions(unittest.TestCase):
    def test_fdr_is_benjamini_hochberg_step_up(self):
        np.testing.assert_array_equal(E.fdr(np.array([0.01, 0.04, 0.03, 0.2]), 0.05), [True, False, False, False])
        np.testing.assert_array_equal(E.fdr(np.array([0.012, 0.024, 0.036, 0.9]), 0.05), [True, True, True, False])
        # 0.013 は単独では 0.0125 を超えるが、後ろの 0.036 ≤ 0.0375 が通るので一緒に棄却（step-up）
        np.testing.assert_array_equal(E.fdr(np.array([0.02, 0.013, 0.036, 0.9]), 0.05), [True, True, True, False])
        self.assertEqual(len(E.fdr(np.array([]), 0.05)), 0)

    def test_decide_matches_two_step_counts(self):
        dec = E.decide(np.zeros(3, dtype=int), np.array([10, 0, 5]), 10)
        self.assertEqual(dec.tolist(), [1, -1, 0])
        # 20回で勝ち4回: 二項の下側 0.0059 > 0.05/20 なので Bonferroni で止まる。勝ち3回（0.0013）は rejected
        self.assertEqual(E.decide(np.zeros(1, dtype=int), np.array([4]), 20).tolist(), [0])
        self.assertEqual(E.decide(np.zeros(1, dtype=int), np.array([3]), 20).tolist(), [-1])
        # 決まった列は変えない（rejected は検定に入れない）
        self.assertEqual(E.decide(np.array([1, -1, 0]), np.array([0, 20, 20]), 20).tolist(), [1, -1, 1])


def corr_importance(dates):
    """偽の重要度: |y との相関|。日付単位の列（その日は同じ値）には +100（影も同じ）。"""
    d = pd.Series(pd.to_datetime(dates)).to_numpy()

    def fit_imp(X, y, it):
        out = []
        for j in range(X.shape[1]):
            x = X[:, j]
            c = abs(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 else 0.0
            date_level = (pd.Series(x).groupby(d).nunique() <= 1).all()
            out.append(c + (100.0 if date_level else 0.0))
        return np.array(out)
    return fit_imp


class TestBorutaLoop(unittest.TestCase):
    def frame(self, n_days=200, per_day=10, seed=0, n_mkt=8):
        """効く列2本・銘柄ごとのでたらめな列4本・日付単位のでたらめな列 n_mkt 本。"""
        rng = np.random.default_rng(seed)
        dates = np.repeat(pd.bdate_range("2023-01-04", periods=n_days), per_day)
        n = len(dates)
        s1, s2 = rng.standard_normal(n), rng.standard_normal(n)
        noise = rng.standard_normal((n, 4))
        mkt = np.repeat(rng.standard_normal((n_days, n_mkt)), per_day, axis=0)
        y = (1.2 * s1 + 0.8 * s2 + rng.standard_normal(n) > 0.8).astype(int)
        X = np.column_stack([s1, s2, noise, mkt])
        return X, y, dates, np.array([False] * 6 + [True] * n_mkt)

    def test_signal_confirmed_noise_rejected(self):
        X, y, dates, is_date = self.frame()
        res = E.boruta(X, y, E.Shuffler(dates), is_date, corr_importance(dates), max_iter=40)
        dec = res["decision"]
        self.assertEqual(dec[:2].tolist(), [1, 1])
        self.assertGreaterEqual(int((dec[2:6] == -1).sum()), 3)
        # 日付単位のでたらめな列は、日付単位の影8本の最大と比べるので、確定するのは偶然の少数だけ
        self.assertLessEqual(int((dec[6:] == 1).sum()), 2)
        self.assertLessEqual(res["n_iter"], 40)
        self.assertEqual(len(res["thr"]), res["n_iter"])

    def test_thresholds_are_per_kind(self):
        """日付単位の影は +100 なので、混ぜて比べると銘柄ごとの列は勝てない。種類ごとなら効く列は確定する。"""
        X, y, dates, is_date = self.frame(seed=1)
        res = E.boruta(X, y, E.Shuffler(dates), is_date, corr_importance(dates), max_iter=20)
        self.assertEqual(res["decision"][:2].tolist(), [1, 1])
        self.assertGreater(res["thr"][0]["date"], 50)
        self.assertLess(res["thr"][0]["stock"], 1)
        # 影の最大になった列（診断）は、その種類の列を指す
        self.assertTrue(all(0 <= t["stock_from"] < 6 for t in res["thr"]))
        self.assertTrue(all(6 <= t["date_from"] < X.shape[1] for t in res["thr"]))

    def test_perc_relaxes_the_bar(self):
        """影の最大（100）だと弱い列も外れる。75 百分位なら弱い列は残り、でたらめと同じ列は外れる。"""
        rng0 = np.random.default_rng(5)
        dates = np.repeat(pd.bdate_range("2023-01-04", periods=50), 4)
        X = rng0.standard_normal((len(dates), 8))

        def fit_imp(Xs, y, it):
            rng = np.random.default_rng(100 + it)
            p = Xs.shape[1] // 2
            real = np.array([10.0, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5])[:p]   # 強い・弱い2本・でたらめと同じ5本
            return np.concatenate([real, rng.random(p)])                       # 影は一様乱数（最大は 1 に近い）

        y = np.zeros(len(dates), dtype=int)
        isd = np.zeros(8, dtype=bool)
        strict = E.boruta(X, y, E.Shuffler(dates), isd, fit_imp, max_iter=60, perc=100)["decision"]
        mild = E.boruta(X, y, E.Shuffler(dates), isd, fit_imp, max_iter=60, perc=75)["decision"]
        self.assertEqual(strict[0], 1)
        self.assertEqual(strict[1:3].tolist(), [-1, -1])                     # 影の最大には勝てない
        self.assertTrue((mild[1:3] >= 0).all())                              # 75 百分位なら外れない
        self.assertTrue((mild[3:] == -1).all())                              # でたらめと同じ列は外れる

    def test_with_a_small_lightgbm(self):
        X, y, dates, is_date = self.frame(n_days=150, per_day=12, seed=2, n_mkt=1)
        saved = TM.N_ESTIMATORS
        TM.N_ESTIMATORS = 40
        try:
            fit = E.gain_fitter({"num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 20,
                                 "colsample_bytree": 1.0, "subsample": 1.0})
            res = E.boruta(X, y, E.Shuffler(dates), is_date, fit, max_iter=60)
        finally:
            TM.N_ESTIMATORS = saved
        dec = res["decision"]
        self.assertEqual(dec[:2].tolist(), [1, 1])
        self.assertGreaterEqual(int((dec[2:6] == -1).sum()), 3)
        self.assertEqual(TM.SEED, 0)                                        # 種を元に戻す


class TestWindows(unittest.TestCase):
    def test_shifted_windows_use_an_earlier_selection(self):
        dates = pd.Series(pd.bdate_range("2018-04-02", "2026-09-30"))
        f0 = E41.folds_for(dates, 0)
        self.assertGreaterEqual(len(f0), 10)
        for sh in (0, 2, 4):
            for f in E41.folds_for(dates, sh):
                s = E.selection_for(f, f0)
                self.assertIsNotNone(s)
                self.assertLessEqual(pd.Timestamp(s.train_end), pd.Timestamp(f.train_end))      # 先読みにならない
                later = [g for g in f0 if pd.Timestamp(s.train_end) < pd.Timestamp(g.train_end)
                         <= pd.Timestamp(f.train_end)]
                self.assertEqual(later, [])                                                      # その中で最も後
                self.assertEqual(s.index, f.index)          # 窓 i の訓練の終わりはずらし0の窓 i の 2・4か月後ろ
        early = SimpleNamespace(index=1, train_end="2017-01-01")
        self.assertIsNone(E.selection_for(early, f0))

    def test_colmaps(self):
        base = [f"c{i}" for i in range(10)]
        f0 = [SimpleNamespace(index=1, train_end="2021-04-03"), SimpleNamespace(index=2, train_end="2021-10-03"),
              SimpleNamespace(index=3, train_end="2022-04-03")]
        f2 = [SimpleNamespace(index=1, train_end="2021-06-03"), SimpleNamespace(index=2, train_end="2021-12-03"),
              SimpleNamespace(index=3, train_end="2022-06-03")]
        sels = {1: {"rejected": ["c3", "c7"]}, 2: {"rejected": []}}         # 窓3の選定はまだ
        cms = E.colmaps(base, {0: f0, 2: f2}, f0, sels)
        for sh in (0, 2):
            self.assertEqual(sorted(cms[("A", sh)]), [1, 2])                # 選定の無い窓はどの腕にも入れない
            self.assertEqual(cms[("A", sh)][1], base)
            self.assertEqual(cms[("S", sh)][1], [c for c in base if c not in ("c3", "c7")])
            self.assertEqual(len(cms[("R", sh)][1]), 8)
            self.assertEqual(cms[("R", sh)][1], [c for c in base if c in set(cms[("R", sh)][1])])  # 並びは元のまま
            self.assertEqual(cms[("S", sh)][2], base)
            self.assertEqual(cms[("R", sh)][2], base)
        again = E.colmaps(base, {0: f0, 2: f2}, f0, sels)
        self.assertEqual(again[("R", 0)][1], cms[("R", 0)][1])              # 窓ごとに決まる
        self.assertEqual(E.colmap_sig(again[("R", 2)]), E.colmap_sig(cms[("R", 2)]))
        self.assertNotEqual(E.colmap_sig(cms[("A", 0)]), E.colmap_sig(cms[("S", 0)]))
        self.assertNotEqual(E.selection_sig(sels), E.selection_sig({1: {"rejected": ["c3"]}, 2: {"rejected": []}}))


class FakeModel:
    """1列目だけを使うモデル。"""

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-2 * X[:, 0]))
        return np.column_stack([1 - p, p])


class TestPermImportance(unittest.TestCase):
    def test_signs(self):
        rng = np.random.default_rng(4)
        dates = np.repeat(pd.bdate_range("2024-01-04", periods=60), 25)
        n = len(dates)
        x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
        mkt = np.repeat(rng.standard_normal(60), 25)
        X = np.column_stack([x0, x1, mkt])
        y = (x0 + 0.7 * rng.standard_normal(n) > 0.5).astype(int)
        m = FakeModel()
        base = m.predict_proba(X)[:, 1]
        imp = E.perm_importance(m, X, y, dates, ["x0", "x1", "mkt"], base,
                                {"g_used": ["x0", "x1"], "g_unused": ["x1", "mkt"], "g_empty": ["zz"]}, {"mkt"}, 42, 3)
        got = imp.set_index(["kind", "name"])
        self.assertGreater(got.loc[("col", "x0"), "d_pr"], 0.05)
        self.assertGreater(got.loc[("col", "x0"), "d_roc"], 0.05)
        for nm in ("x1", "mkt"):
            self.assertEqual(got.loc[("col", nm), "d_pr"], 0.0)              # 使っていない列は動かない
        self.assertGreater(got.loc[("group", "g_used"), "d_pr"], 0.05)
        self.assertEqual(got.loc[("group", "g_unused"), "d_pr"], 0.0)
        self.assertNotIn(("group", "g_empty"), got.index)
        self.assertTrue((imp["fold"] == 3).all())


def synthetic_oof(seed, n_folds=3, per_day=10, days=60, strength=0.4):
    """fold ごとに日付が分かれた合成の out-of-fold（実験89 のテストと同じ形）。"""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2021-06-01", periods=n_folds * days)
    for f in range(n_folds):
        for d in dates[f * days:(f + 1) * days]:
            for c in range(per_day):
                y = int(rng.random() < 0.25)
                rows.append({"Code": f"{1000 + c}0", "Date": d, "label": y, "fold": f + 1,
                             "score": y * strength + rng.random(), "ret_o1_20": rng.normal(1 + 3 * y, 5),
                             "ret_o1_40": rng.normal(1 + 4 * y, 7)})
    return pd.DataFrame(rows)


class TestReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E.OOF_DIR, E.RUN)
        E.OOF_DIR, E.RUN = self.dir, "e91test_k15"

    def tearDown(self):
        E.OOF_DIR, E.RUN = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_report_prints_every_section_and_saves_the_summary(self):
        base = [f"c{i}" for i in range(12)]
        groups = {"g1": base[:4], "g2": base[4:10], "mkt": base[10:]}
        date_cols = base[10:]
        sels = {k: {"fold": k, "train_end": f"202{k}-04-03", "n_train": 5000 + 1000 * k, "pos_rate": 0.13,
                    "n_iter": 100, "seconds": 300, "confirmed": base[:2], "tentative": base[2:6] + base[8:10],
                    "rejected": base[6:8] + ([base[11]] if k == 3 else []),
                    "max_shadow_from": {"stock": {"c1": 60, "c4": 40}, "date": {"c10": 100}}} for k in (1, 2, 3)}
        f0 = [SimpleNamespace(index=k, train_end=f"202{k}-04-03") for k in (1, 2, 3)]
        cms = E.colmaps(base, {0: f0, 2: f0}, f0, sels)
        results = {}
        for i, a in enumerate(("lgbm_k15", "xgb_k15")):
            prod, wins, imps = {}, {}, {}
            for j, arm in enumerate(E.ARMS):
                st = {"A": 0.4, "S": 0.45, "R": 0.38}[arm]
                prod[arm] = synthetic_oof(10 * i + j, strength=st)
                for sh in (0, 2):
                    wins[(sh, arm)] = E.windows_on(synthetic_oof(100 * i + 10 * j + sh, strength=st), None)
            rng = np.random.default_rng(i)
            for sh in (0, 2):
                rows = [{"fold": f, "kind": "col", "name": c, "d_pr": float(rng.normal(0.002, 0.003)),
                         "d_roc": float(rng.normal(0.001, 0.002))} for f in (1, 2, 3) for c in base]
                rows += [{"fold": f, "kind": "group", "name": g, "d_pr": float(rng.normal(0.004, 0.003)),
                          "d_roc": 0.002} for f in (1, 2, 3) for g in groups]
                rows += [{"fold": f, "kind": "col", "name": "unused", "d_pr": 0.0, "d_roc": 0.0} for f in (1, 2, 3)]
                imps[sh] = pd.DataFrame(rows)
            results[a] = {"params": {}, "prod": prod, "wins": wins, "imps": imps}
        stamp = {"built_utc": "2026-10-14 05:00 UTC", "rows": 1800, "date_min": "2021-06-01", "date_max": "2022-02-15"}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(results, [0, 2], (42, 7, 123), stamp, cms, sels, base, groups, date_cols)
        text = buf.getvalue()
        for head in ("■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6."):
            self.assertIn(head, text)
        self.assertIn("そろってから判定", text)                               # cat がまだ・ずらし4が無い
        self.assertIn("lgbm・ずらし2か月（3窓）", text)
        self.assertIn("2021-06-01〜", text)
        self.assertIn("最後の窓（訓練 〜2023-04-03", text)
        self.assertIn("影の最大になった列（銘柄ごと・3窓の 300回のうち）: c1 180回（60%）, c4 120回（40%）", text)
        self.assertIn("偶然の見込み", text)
        self.assertNotIn("nan", text.lower())
        self.assertEqual(len(s["per_window"]), 2 * 2 * 3)
        self.assertFalse(s["judge7"]["verdict"]["complete"])
        self.assertIn("lgbm", s["judge7"])
        self.assertEqual(s["selection"]["rejected_count"]["c6"], 3)
        self.assertEqual(s["selection"]["rejected_count"]["c11"], 1)
        p = os.path.join(self.dir, "e91test_k15_summary_date.json")
        self.assertTrue(os.path.exists(p))
        with open(p, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(saved["label_k"], 1.5)
        self.assertEqual(saved["boruta"]["max_iter"], 100)
        self.assertEqual(saved["selection_sig"], E.selection_sig(sels))
        self.assertEqual(saved["shadow"], "date")

    def test_verdict(self):
        ok, ng = {"pass": True}, {"pass": False}
        self.assertEqual(E.verdict({"a": ok, "b": ok, "c": ng}, True)["passed"], 2)
        self.assertIn("本番に入れる案", E.verdict({"a": ok, "b": ok, "c": ng}, True)["text"])
        self.assertIn("239列のまま", E.verdict({"a": ok, "b": ng, "c": ng}, True)["text"])
        self.assertIn("そろってから", E.verdict({"a": ok, "b": ok}, False)["text"])


class TestTimeBudget(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT)
        E.OOF_DIR, E.RUN = self.dir, "e91test_k15"

    def tearDown(self):
        E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_nothing_new_after_the_deadline(self):
        df = pd.DataFrame({"Code": ["10000"] * 3, "Date": pd.bdate_range("2024-01-04", periods=3),
                           "label": [1, 0, 1], "x": [1.0, 2.0, 3.0]})
        stamp = {"built_utc": "2026-10-14 00:00 UTC"}
        E.DEADLINE, E.TIMED_OUT = time.time() - 1, False
        cms = {(arm, 0): {1: ["x"]} for arm in E.ARMS}
        self.assertEqual(E.oof("lgbm_k15", "A", df, cms, {}, 0, (42,), stamp, None), (None, None))
        fold = SimpleNamespace(index=1, train_end="2024-01-08")
        self.assertIsNone(E.select_window(df, ["x"], fold, {}, stamp, [], 5))
        self.assertTrue(E.TIMED_OUT)
        self.assertEqual(os.listdir(self.dir), [])
        self.assertTrue(E.path("summary.json").endswith("e91test_k15_summary.json"))
        self.assertTrue(E.fpath("frame.parquet").endswith(f"{E.TAG}_frame.parquet"))


if __name__ == "__main__":
    unittest.main()
