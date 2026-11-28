#!/usr/bin/env python3
"""
実験79（12モデルの予測値の傾向分析。research/exp/e79_model_clusters.py）。

- 百分位は画面の pctHistorical と同じ（同点は数えない）
- 12本の out-of-fold を (Code, Date) で突き合わせ、ラベルは 1.2σ / 1.5σ / 1.8σ の3つを持つ。足りないファイルは止める
- テスト窓の割り当ては本番の out-of-fold と同じ窓
- 地図のマス目は MIN_CELL 件未満を出さない
- 前向き: 訓練はテスト窓より前、かつ結果の出ていない直前 RISE_HORIZON 営業日を外す。百分位は訓練の分布で付ける
- 仕込んだ「ずれの型」（ロジスティックだけ高い行の収益が高い）を前向きの手順で拾える
- 報告に銘柄コードが出ない（公開ログ）

  python3 tests/test_e79_model_clusters.py
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import e79_model_clusters as E  # noqa: E402
import models as M  # noqa: E402

LOGIT_SET = set(E.LOGITS)


def synth(n_windows=6, per=900, seed=0, planted=0.06):
    """
    窓ごとに per 行。合意 c と「ロジスティックだけ高い」ずれ p を持つ12本のスコアを作り、p が大きい行の実収益を
    planted だけ上げる。窓は 2021-05-01 から6か月ずつ。
    """
    rng = np.random.default_rng(seed)
    folds, rows = [], []
    start = pd.Timestamp("2021-05-01")
    for w in range(1, n_windows + 1):
        ts = start + pd.DateOffset(months=6 * (w - 1))
        te = start + pd.DateOffset(months=6 * w) - pd.Timedelta(days=1)
        folds.append(types.SimpleNamespace(index=w, test_start=str(ts.date()), test_end=str(te.date())))
        days = pd.bdate_range(ts, te)
        d = rng.choice(days, size=per)
        c = rng.normal(size=per)
        p = rng.normal(size=per)
        ret = 0.01 * c + planted * (p > 1.0) + rng.normal(0, 0.06, size=per)
        rec = {"Code": [f"{9000 + i % 700}0" for i in range(per)], "Date": d, "window": w,
               E.OUTCOME: ret, "future_rise": np.abs(ret) + 0.05, "ret_o1_40": ret * 1.5,
               "label_k12": (ret > 0.04).astype(float), "label_k15": (ret > 0.07).astype(float),
               "label_k18": (ret > 0.10).astype(float)}
        for a in E.MODELS:
            tilt = 1.0 if a in LOGIT_SET else -0.25
            rec[f"score_{a}"] = c + 0.9 * p * tilt + rng.normal(0, 0.3, size=per)
        rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True)
    # 同じ (Code, Date) を避ける
    df = df.drop_duplicates(["Code", "Date"]).sort_values(["Date", "Code"]).reset_index(drop=True)
    return df, folds


class TestBasics(unittest.TestCase):
    def test_pct_within_matches_pct_historical(self):
        ref = np.sort(np.array([1.0, 2.0, 3.0, 4.0]))
        got = E.pct_within(ref, np.array([0.0, 2.5, 4.0, 5.0, 2.0]))
        self.assertEqual(got.tolist(), [0.0, 50.0, 75.0, 100.0, 25.0])
        for x in (0.0, 2.5, 4.0, 5.0, 2.0):
            self.assertAlmostEqual(E.pct_within(ref, np.array([x]))[0], M.pct_historical(x, ref), places=6)
        self.assertTrue(np.isnan(E.pct_within(np.array([]), np.array([1.0]))[0]))

    def test_models_are_the_twelve_in_lineup_order(self):
        # 実験79〜88 の12本（1.2σ・1.5σ・1.8σ × 4学習器）。2026-10-24 に 1.8σ の4本を、2026-11-02 の予測からロジスティック
        # 回帰を毎日の採点・画面（M.LINEUP）から外したが、実験は記録どおり12本のまま（M.VARIANTS と E.ALGOS4 に残してある）
        self.assertEqual(E.ALGOS4, ("lgbm", "xgb", "cat", "logit"))
        self.assertEqual(E.MODELS, E.ALGOS4 + tuple(M.VARIANTS))
        self.assertEqual(len(E.MODELS), 12)
        self.assertEqual(E.BASE4, E.ALGOS4)
        self.assertTrue(set(M.LINEUP) <= set(E.MODELS))                      # 今の6本は12本の中にある
        self.assertEqual(set(E.TREES) | set(E.LOGITS), set(E.MODELS))
        self.assertEqual(E.LOGITS, ("logit", "logit_k15", "logit_k18"))
        self.assertEqual(E.oof_path("lgbm", "/x"), "/x/oof.parquet")              # 基準モデルの Release 名
        self.assertEqual(E.oof_path("cat_k18", "/x"), "/x/cat_k18_oof.parquet")

    def test_order_map_and_terciles(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        key = np.array([90, 80, 10, 20, 50, 60.0])
        remap = E.order_map(labels, key, 3)
        self.assertEqual(remap[labels].tolist(), [2, 2, 0, 0, 1, 1])             # 合意の低い順
        t = E.terciles(np.arange(9.0))
        self.assertEqual(t.tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_window_diffs_and_summary(self):
        df = pd.DataFrame({"window": [1] * 4 + [2] * 4, E.OUTCOME: [0.1, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]})
        m = np.array([True, True, False, False, True, False, False, False])
        d = E.window_diffs(df, m, min_rows=1)
        self.assertEqual(np.round(d, 6).tolist(), [5.0, 15.0])                  # 窓1: 10 − 5、窓2: 20 − 5
        s = E.summarize(d)
        self.assertEqual((s["n"], s["wins"]), (2, 2))
        self.assertAlmostEqual(s["mean"], 10.0)
        self.assertEqual(E.window_diffs(df, m, min_rows=3).tolist(), [])        # 件数が足りない窓は落とす

    def test_grid_cells_hide_small_cells(self):
        n = 60
        E2 = np.column_stack([np.r_[np.zeros(50), np.ones(10)], np.zeros(n)])
        df = pd.DataFrame({E.OUTCOME: np.r_[np.full(50, 0.02), np.full(10, 0.5)],
                           "future_rise": 0.1, "label_k12": 1.0})
        ex = np.array([-0.5, 0.5, 1.5])
        ey = np.array([-0.5, 0.5])
        cells, info = E.grid_cells(E2, df, ex, ey, min_n=20)
        self.assertEqual(len(cells), 1)
        self.assertEqual(int(cells["n"].iloc[0]), 50)
        self.assertAlmostEqual(float(cells["ret"].iloc[0]), 2.0)
        self.assertEqual((info["hidden_cells"], info["hidden_rows"]), (1, 10))   # 10件のマス目は出さない

    def test_assign_windows_matches_production_windows(self):
        dates = pd.Series(pd.bdate_range("2018-04-03", "2026-09-08"))
        folds = E.make_windows(dates)
        self.assertGreaterEqual(len(folds), 10)
        w = E.assign_windows(dates, folds)
        first = pd.Timestamp(folds[0].test_start)
        self.assertTrue((w[dates < first] == 0).all())
        self.assertTrue((w[dates >= first] > 0).all())
        for f in folds:
            m = w == f.index
            self.assertEqual(pd.Timestamp(dates[m].min()), pd.bdate_range(f.test_start, f.test_end)[0])


class TestLoad(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        rng = np.random.default_rng(1)
        self.base = pd.DataFrame({"Code": [str(1000 + i) for i in range(30)],
                                  "Date": pd.bdate_range("2022-01-03", periods=30)})
        for a in E.MODELS:
            o = self.base.copy()
            o["score"] = rng.random(30)
            o["label"] = (rng.random(30) > 0.8).astype(float)
            o[E.OUTCOME] = rng.normal(0, 0.05, 30)
            if a == "xgb":                                   # 重複を1行
                o = pd.concat([o, o.iloc[[0]]], ignore_index=True)
            if a == "cat_k15":                               # 1行少ない
                o = o.iloc[1:]
            o.to_parquet(E.oof_path(a, self.d))

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_join(self):
        wide, info = E.load_oofs(self.d)
        self.assertEqual(info["dups"]["xgb"], 1)
        self.assertEqual(info["rows"]["cat_k15"], 29)
        self.assertEqual(info["kept"], 29)
        for c in ["label_k12", "label_k15", "label_k18", E.OUTCOME] + [f"score_{a}" for a in E.MODELS]:
            self.assertIn(c, wide.columns)
        self.assertFalse(wide.duplicated(["Code", "Date"]).any())

    def test_missing_file_stops(self):
        os.remove(E.oof_path("logit_k18", self.d))
        with self.assertRaises(SystemExit):
            E.load_oofs(self.d)


class TestForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.folds = synth()

    def test_split_uses_only_past_with_embargo(self):
        split = E.window_split(self.df, self.folds)
        self.assertEqual(sorted(split), [3, 4, 5, 6])                           # 訓練に窓2つ以上
        gap = pd.Timedelta(days=int(round(B.RISE_HORIZON * E.WF.TRADING_TO_CALENDAR)))
        for w, (tr, te, Ptr, Pte) in split.items():
            start = pd.Timestamp(self.folds[w - 1].test_start)
            self.assertTrue((self.df["window"].to_numpy()[tr] < w).all())
            self.assertTrue((pd.to_datetime(self.df["Date"].to_numpy()[tr]) <= start - gap).all())
            self.assertTrue((self.df["window"].to_numpy()[te] == w).all())
            self.assertEqual(Ptr.shape[1], 12)
            # テストの百分位は訓練の分布で付ける（訓練の最大より大きければ 100）
            ref = np.sort(self.df[f"score_{E.MODELS[0]}"].to_numpy()[tr])
            np.testing.assert_allclose(Pte[:, 0], E.pct_within(ref, self.df[f"score_{E.MODELS[0]}"].to_numpy()[te]))

    def test_planted_pattern_is_found(self):
        split = E.window_split(self.df, self.folds)
        wt = E.wf_types(self.df, split, n_placebo=5)
        self.assertGreater(wt["vs_top"]["mean"], 0.5)                          # 仕込んだ +6% の型を拾う
        self.assertGreaterEqual(wt["vs_top"]["wins"], 3)
        wc = E.wf_clusters(self.df, split, 6, n_placebo=5)
        self.assertEqual(len(wc["rows"]), 4)
        self.assertTrue(np.isfinite(wc["vs_cons"]["mean"]))
        self.assertIn("vs_cons_p", wc["placebo"])
        self.assertEqual(wc["pick_mask"].shape, (len(self.df),))

    def test_monthly_picks(self):
        d = pd.DataFrame({"Date": pd.to_datetime(["2022-01-04", "2022-01-05", "2022-01-06", "2022-02-01"]),
                          "x": [1.0, 3.0, 2.0, 5.0]})
        p = E.monthly_picks(d, "x", n=2)
        self.assertEqual(p["x"].tolist(), [3.0, 2.0, 5.0])


class TestReport(unittest.TestCase):
    def test_analyze_runs_and_prints_no_codes(self):
        df, folds = synth(n_windows=5, per=600, seed=3)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.analyze(df, folds=folds, k_main=4, ks=(3, 4), wf_ks=(4,), tsne=False, n_placebo=3)
        out = buf.getvalue()
        for key in ("decile", "clusters", "types", "contrasts", "wf", "monthly", "grid"):
            self.assertIn(key, s)
        self.assertEqual(len(s["clusters"]), 4)
        self.assertIn("@grid,pca,", out)
        self.assertIn("@centroid,4,", out)
        for code in df["Code"].unique()[:50]:
            self.assertNotIn(code, out)                                          # 銘柄コードは出さない
        # マス目は MIN_CELL 件以上だけ
        ns = [int(line.split(",")[4]) for line in out.splitlines() if line.startswith("@grid,")]   # @grid,地図,ix,iy,件数,...
        self.assertTrue(ns and min(ns) >= E.MIN_CELL)


if __name__ == "__main__":
    unittest.main()
