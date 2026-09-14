#!/usr/bin/env python3
"""
research/tabicl_model.py と research/exp/e17_tabicl.py の単体テスト。

TabICL 本体（torch と Hugging Face の重み）は要らない部分だけを固定する。
入力の作り方・文脈の切り方・種の平均・行の揃え方が壊れると、
出てくる数字が黙って別のものになるため。

  python3 tests/test_tabicl.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import lab  # noqa: E402
import tabicl_model as T  # noqa: E402
import tuning_multi as TM  # noqa: E402


def _oof(folds=(1, 2, 3), per_fold=400, seed=0, shift=0.0):
    """lab.metrics が読める最小限の out-of-fold を合成する。"""
    rng = np.random.default_rng(seed)
    rows = []
    for f in folds:
        for i in range(per_fold):
            rows.append({
                "Code": f"{1000 + i:04d}",
                "Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=f * 30 + i % 20),
                "label": int(rng.random() < 0.2),
                "ref_end": float(rng.normal(0.01, 0.1)),
                "ref_rise": float(rng.normal(0.02, 0.1)),
                "entry_gap": 0.0,
                "ret_flex": float(rng.normal(0.01, 0.1)),
                "score": float(rng.random() + shift),
                "fold": f,
            })
    df = pd.DataFrame(rows)
    for h in lab.HORIZONS:
        df[f"ret_o1_{h}"] = rng.normal(0.01, 0.1, len(df))
        df[f"ret_c0_{h}"] = rng.normal(0.01, 0.1, len(df))
    return df


class TestToFrame(unittest.TestCase):
    def setUp(self):
        self.cols = ["r_high", "per", "s33_code", "mkt_code"]
        self.df = pd.DataFrame({
            "r_high": [99.0, 95.0, np.nan, 90.0],
            "per": [10.0, np.inf, -np.inf, 12.0],
            "s33_code": [5250, 1050, 5250, np.nan],
            "mkt_code": [111, 112, 111, 111],
            "Date": pd.to_datetime(["2020-01-01"] * 4),
        })

    def test_categorical_columns_become_category(self):
        out = T.to_frame(self.df, self.cols)
        for c in self.cols:
            if c in TM.CATEGORICAL:
                self.assertEqual(str(out[c].dtype), "category", c)
            else:
                self.assertEqual(out[c].dtype, float, c)

    def test_infinite_becomes_missing(self):
        # 補完は平均（TabICL 内の SimpleImputer の既定）なので、inf が
        # 1つ混じると列ぜんぶが inf になる。欠損に倒しておく必要がある
        out = T.to_frame(self.df, self.cols)
        self.assertTrue(np.isfinite(out["per"].dropna()).all())
        self.assertEqual(int(out["per"].isna().sum()), 2)

    def test_column_order_preserved(self):
        out = T.to_frame(self.df, self.cols)
        self.assertEqual(list(out.columns), self.cols)

    def test_source_not_mutated(self):
        before = self.df["per"].tolist()
        T.to_frame(self.df, self.cols)
        self.assertEqual(self.df["per"].tolist(), before)


class TestCapContext(unittest.TestCase):
    def setUp(self):
        self.tr = pd.DataFrame({
            "Date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-05",
                                    "2020-01-02", "2020-01-04"]),
            "label": [0, 1, 0, 1, 0],
        })

    def test_keeps_latest_rows(self):
        out = T.cap_context(self.tr, 2)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(out["Date"].dt.day.tolist()), [4, 5])

    def test_no_cap_when_zero_or_larger(self):
        self.assertEqual(len(T.cap_context(self.tr, 0)), 5)
        self.assertEqual(len(T.cap_context(self.tr, 99)), 5)

    def test_rows_and_labels_stay_together(self):
        # 行を落とすので、X と y がずれないことは致命的に重要
        out = T.cap_context(self.tr, 3)
        for _, r in out.iterrows():
            src = self.tr.loc[r.name]
            self.assertEqual(src["label"], r["label"])
            self.assertEqual(src["Date"], r["Date"])


class TestAverageSeeds(unittest.TestCase):
    def test_scores_are_averaged(self):
        a = lab.Result("a", _oof(seed=1), {})
        b = lab.Result("b", _oof(seed=2), {})
        out = T.average_seeds([a, b])
        key = ["Code", "Date"]
        ja = a.oof.set_index(key)["score"]
        jb = b.oof.set_index(key)["score"]
        jo = out.oof.set_index(key)["score"]
        expect = (ja + jb) / 2
        np.testing.assert_allclose(jo.loc[expect.index].to_numpy(),
                                   expect.to_numpy(), rtol=1e-9)

    def test_only_common_folds_are_kept(self):
        # 予算切れで種ごとに回せた窓が違うことがある。揃っている窓だけで
        # 平均しないと、種によって見ている期間が違うまま平均することになる
        a = lab.Result("a", _oof(folds=(1, 2, 3), seed=1), {})
        b = lab.Result("b", _oof(folds=(1, 2), seed=2), {})
        out = T.average_seeds([a, b])
        self.assertEqual(sorted(out.oof["fold"].unique().tolist()), [1, 2])


class TestMemoryTracking(unittest.TestCase):
    """
    2026-09-14 に本走がランナーごとメモリ不足で落ちた（exit 143）。
    原因は2つとも「測れていなかった」こと。ここで両方を押さえる。
    """

    def test_peak_resets_and_tracks(self):
        # 最高水位はプロセスを通した高水位線なので、リセットできないと
        # 先に読んだデータフレームの 5.4GB に TabICL の確保ぶんが埋もれる
        if not T.reset_peak_rss():
            self.skipTest("この環境では最高水位をリセットできない")
        base = T.peak_rss_gb()
        x = np.ones((1 << 26,), dtype=np.float64)   # 512MB。実際に書き込む
        grown = T.peak_rss_gb()
        del x
        self.assertGreater(grown, base + 0.3, "確保したぶんが水位に出ていない")
        T.reset_peak_rss()
        self.assertLess(T.peak_rss_gb(), grown - 0.3, "リセットが効いていない")

    def test_limit_raises_instead_of_killing_the_runner(self):
        # 上限が無いとホストがランナーごと落とし、`if: always()` を付けた
        # 後片付けまで飛ばされてそこまでの計算が消える
        import multiprocessing as mp

        def child(q):
            import sys as _s
            _s.path.insert(0, os.path.join(ROOT, "research"))
            import numpy as _np
            import tabicl_model as _T
            q.put(("limited", _T.limit_memory(1.0)))
            try:
                _np.ones((1 << 29,), dtype=_np.float64)   # 4GB
                q.put(("result", "allocated"))
            except MemoryError:
                q.put(("result", "MemoryError"))

        q = mp.Queue()
        p = mp.Process(target=child, args=(q,))
        p.start()
        p.join(60)
        got = dict(q.get(timeout=5) for _ in range(2))
        if not got.get("limited"):
            self.skipTest("この環境では上限を掛けられない")
        self.assertEqual(got.get("result"), "MemoryError")


class TestEstimateTotal(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 6000
        d = pd.date_range("2018-01-01", periods=n // 4, freq="B")
        self.df = pd.DataFrame({
            "Date": np.repeat(d, 4)[:n],
            "Code": [f"{1000 + i % 400:04d}" for i in range(n)],
            "label": rng.integers(0, 2, n).astype(float),
        })
        self.probe = {"n_test_probe": 500, "n_estimators": 2,
                      "seconds_per_context_row": 0.01, "fixed_seconds": 5.0}

    def test_more_estimators_costs_more(self):
        a = T.estimate_total(self.df, self.probe, max_context=0, n_estimators=2)
        b = T.estimate_total(self.df, self.probe, max_context=0, n_estimators=8)
        self.assertGreater(b["seconds"], a["seconds"])
        self.assertAlmostEqual(b["seconds"] / a["seconds"], 4.0, places=5)

    def test_capping_context_costs_less(self):
        full = T.estimate_total(self.df, self.probe, max_context=0, n_estimators=8)
        cap = T.estimate_total(self.df, self.probe, max_context=500, n_estimators=8)
        self.assertLess(cap["seconds"], full["seconds"])

    def test_seeds_multiply(self):
        one = T.estimate_total(self.df, self.probe, max_context=0, n_estimators=8)
        three = T.estimate_total(self.df, self.probe, max_context=0,
                                 n_estimators=8, n_seeds=3)
        self.assertAlmostEqual(three["seconds"] / one["seconds"], 3.0, places=5)

    def test_no_slope_returns_empty(self):
        self.assertEqual(T.estimate_total(self.df, {}, max_context=0,
                                          n_estimators=8), {})


class TestAlign(unittest.TestCase):
    def test_trims_to_common_rows(self):
        import e17_tabicl as E17

        full = _oof(folds=(1, 2, 3), seed=1)
        short = _oof(folds=(1, 2), seed=2)
        res = {"tabicl": lab.Result("tabicl", short, lab.metrics(short)),
               "lgbm": lab.Result("lgbm", full, lab.metrics(full))}
        out = E17.align(res, log=lambda *a, **k: None)
        n = {k: len(v.oof) for k, v in out.items()}
        self.assertEqual(len(set(n.values())), 1, n)
        self.assertEqual(n["tabicl"], len(short))

    def test_equal_frames_pass_through(self):
        import e17_tabicl as E17

        a = _oof(seed=1)
        b = _oof(seed=2)
        res = {"tabicl": lab.Result("tabicl", a, lab.metrics(a)),
               "lgbm": lab.Result("lgbm", b, lab.metrics(b))}
        out = E17.align(res, log=lambda *a, **k: None)
        self.assertIs(out["tabicl"].oof, a)


class TestCheckpointSettings(unittest.TestCase):
    def test_missing_checkpoint_path_is_rejected(self):
        # 存在しないパスを渡すと TabICL は「無いので HF から落とす」に
        # 倒れる。渡したつもりのファイルが使われないまま別の重みで
        # 測っていた、という事故を防ぐためここで止める
        os.environ["TABICL_CKPT"] = "/nonexistent/does-not-exist.ckpt"
        try:
            with self.assertRaises(SystemExit):
                T.ckpt_path()
        finally:
            del os.environ["TABICL_CKPT"]

    def test_default_version_is_v2(self):
        os.environ.pop("TABICL_CKPT_VERSION", None)
        self.assertIn("v2", T.ckpt_version())


class TestFoldPath(unittest.TestCase):
    def test_settings_change_the_path(self):
        # 設定が違う結果を同じファイルに書くと、再開のときに
        # 「文脈8000で測った窓」と「全行で測った窓」が混ざる
        base = T._fold_path(42, 3, 0, 8)
        self.assertNotEqual(base, T._fold_path(42, 3, 8000, 8))
        self.assertNotEqual(base, T._fold_path(42, 3, 0, 4))
        self.assertNotEqual(base, T._fold_path(7, 3, 0, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
