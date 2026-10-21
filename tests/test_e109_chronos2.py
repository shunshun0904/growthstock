#!/usr/bin/env python3
"""
実験109（research/exp/e109_chronos2.py）の部品。データにも Chronos-2 の重みにも触らない（手で解ける小さな例で確かめる）。

- 毎日動く列: 既定の360列から 159列。決算・開示・更新日の行だけの列・時価総額・バリュエーション・区分は入らない。どの列も
  共変量を作る関数（covariates）の作れる範囲にある
- 上位30列: gain の大きい順・同じ値は列名の順・毎日動く列だけ・gain の無い列は 0 扱い
- 分位点 -> 確率: 分位点ちょうどでは 1 − 水準、間は直線、外は端で止める、交差は並べ直す、欠測は欠測
- 点数: p_end は20日目、p_reach は最大（p_end 以上）、med は中央値の上昇 ÷ log(1 + 必要上昇率)、外側で止めた印
- 文脈: t の行で終わり、長さは CONTEXT 以下、銘柄をまたがない。日足に無い行は止める
- 入力: 株価だけなら終値の対数の配列、共変量ありなら同じ日足の行の値を名前つきで渡す
- 判定の言い方と、日付で束ねたブートストラップ（同じ点数の腕どうしの差の区間は 0）
- 途中から続ける: 保存済みの分位点を読み、まだの行があれば None。束の名前は行と入力の値で決まる（行が増えた束だけ計算し直す）
- 回をまたぐ上位列: 保存してある回ごとの A の gain から窓ごとの候補（列の組が違う回は使わない）、保存済みの束が多いものを選ぶ
  （同じなら今回の列。同じ列の組は1回だけ数える）。前の回の列の束が残っていれば計算し直さない
- 基準モデルの学習: E41.oof_folds の LightGBM と同じ点数（小さな合成データ）

  python3 tests/test_e109_chronos2.py
"""
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e109_chronos2 as E  # noqa: E402
import features as F  # noqa: E402
import technicals as TX  # noqa: E402

LV = [0.1, 0.5, 0.9]


class Columns(unittest.TestCase):
    def setUp(self):
        self.cols = F.columns(F.DEFAULT_PRESET)
        self.el = E.eligible_columns(self.cols)

    def test_count_and_members(self):
        self.assertEqual(len(self.cols), 360)
        self.assertEqual(len(self.el), 159)
        for c in ("tx_ma_gap_5", "topix_ret_20", "gold_ret_120", "sector_ret_20", "rel_sector_20", "ret_1d",
                  "r_high_3m", "vol_rel_mkt", "volume_trend", "credit_ratio", "short_ratio", "inv_foreign_4w",
                  "break_margin", "close_position", "ret_20d", "vol_20d", "log_trading_value"):
            self.assertIn(c, self.el)
        for c in ("base_length", "broken_high_age", "prev_nh_ratio", "base_depth", "pullback_depth", "max_dd_120",
                  "log_market_cap", "per", "pbr", "div_yield", "jq_fwdper", "progress_pct", "eps_growth_q0",
                  "days_since_disc", "days_to_earn", "listing_years", "s33_code", "mkt_code", "alert_days",
                  "lvs_ratio", "rev_pct"):
            self.assertNotIn(c, self.el)

    def test_all_eligible_are_buildable(self):
        can = (set(E.PANEL_FEATURES) | {"log_trading_value", "vol_rel_mkt", "vol_rel_sector", "credit_ratio"}
               | set(F.GROUPS["market"]) | set(F.GROUPS["sector_index"]) | set(F.GROUPS["flow"]) | set(TX.COLUMNS))
        self.assertEqual(sorted(set(self.el) - can), [])

    def test_top_eligible(self):
        gain = pd.Series({"a": 5.0, "b": 9.0, "c": 5.0, "x": 100.0})
        self.assertEqual(E.top_eligible(gain, ["a", "b", "c", "d"], n=3), ["b", "a", "c"])
        self.assertEqual(E.top_eligible(gain, ["d", "a"], n=5), ["a", "d"])


class Probability(unittest.TestCase):
    def test_knots_between_and_outside(self):
        q = np.array([[0.0, 1.0, 2.0]] * 5)
        x = np.array([0.0, 1.0, 1.5, -5.0, 9.0])
        p = E.prob_above(q, LV, x)
        np.testing.assert_allclose(p, [0.9, 0.5, 0.3, 0.9, 0.1])

    def test_crossing_is_sorted_and_nan(self):
        q = np.array([[2.0, 1.0, 0.0], [0.0, np.nan, 2.0]])
        p = E.prob_above(q, LV, np.array([1.0, 1.0]))
        self.assertAlmostEqual(p[0], 0.5)
        self.assertTrue(np.isnan(p[1]))
        self.assertTrue(np.isnan(E.prob_above(np.array([[0.0, 1.0, 2.0]]), LV, np.array([np.nan]))[0]))

    def test_scores(self):
        h = 3
        base = np.log(100.0)
        need = np.array([0.1, 0.1])
        # 1行目: 先に行くほど上へ。2行目: 上がらない
        q = np.zeros((2, 3, h))
        for j in range(h):
            q[0, :, j] = base + np.array([-0.05, 0.02 * (j + 1), 0.2])
            q[1, :, j] = base + np.array([-0.1, -0.05, 0.0])
        s = E.scores(q, LV, np.array([base, base]), need)
        thr = np.log1p(0.1)
        want_end = E.prob_above(q[:, :, -1], LV, base + np.array([thr, thr]))
        np.testing.assert_allclose(s["p_end"].to_numpy(), want_end)
        self.assertTrue((s["p_reach"] >= s["p_end"] - 1e-12).all())
        self.assertAlmostEqual(s["med"].iloc[0], 0.06 / thr)
        self.assertAlmostEqual(s["med"].iloc[1], -0.05 / thr)
        self.assertEqual(list(s["clip_hi"]), [False, True])
        self.assertEqual(list(s["crossed"]), [False, False])


def toy_panel():
    rows = []
    for code, n in (("1111", 6), ("2222", 4)):
        for i in range(n):
            rows.append({"Code": code, "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                         "close": 100.0 + i + (50 if code == "2222" else 0)})
    return pd.DataFrame(rows)


class Context(unittest.TestCase):
    def test_bounds(self):
        p = toy_panel()
        r = pd.DataFrame({"Code": ["1111", "1111", "2222"],
                          "Date": pd.to_datetime(["2024-01-02", "2024-01-06", "2024-01-03"])})
        st, en = E.context_bounds(p, r, 3)
        self.assertEqual(list(en), [1, 5, 8])
        self.assertEqual(list(st), [0, 3, 6])          # 2行目は3行、1・3行目は銘柄の頭で止まる
        with self.assertRaises(SystemExit):
            E.context_bounds(p, pd.DataFrame({"Code": ["2222"], "Date": [pd.Timestamp("2024-01-09")]}), 3)

    def test_items(self):
        p = toy_panel()
        close = p["close"].to_numpy(dtype=float)
        st, en = np.array([3, 6]), np.array([5, 8])
        uni = E.make_items(close, st, en)
        np.testing.assert_allclose(uni[0], np.log([103.0, 104.0, 105.0]), rtol=1e-6)
        self.assertEqual(len(uni[1]), 3)
        cov = np.arange(20, dtype=np.float32).reshape(10, 2)       # 共変量の表の行 = 日足の行
        row = np.arange(10)
        it = E.make_items(close, st, en, cov, [1], ["b"], row)
        self.assertEqual(sorted(it[0]["past_covariates"]), ["b"])
        np.testing.assert_allclose(it[0]["past_covariates"]["b"], [7.0, 9.0, 11.0])
        bad = row.copy()
        bad[4] = -1
        with self.assertRaises(SystemExit):
            E.make_items(close, st, en, cov, [0], ["a"], bad)


class Judge(unittest.TestCase):
    def test_verdict(self):
        self.assertEqual(E.verdict(0.01, 0.001, 0.02, 7, 10), "上回る")
        self.assertEqual(E.verdict(-0.01, -0.02, -0.001, 3, 10), "下回る")
        self.assertEqual(E.verdict(0.01, -0.001, 0.02, 7, 10), "決められない")
        self.assertEqual(E.verdict(0.01, 0.001, 0.02, 5, 10), "決められない")

    def test_boot(self):
        rng = np.random.default_rng(0)
        n = 300
        o = pd.DataFrame({"Date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D"),
                          "label": rng.integers(0, 2, n), "s_A": rng.random(n)})
        o["s_Z0"] = o["s_A"]
        b1 = E.boot(o, ["A", "Z0"], n=50, seed=1)
        b2 = E.boot(o, ["A", "Z0"], n=50, seed=1)
        pd.testing.assert_frame_equal(b1, b2)
        self.assertEqual(E.ci(b1["pr_Z0"] - b1["pr_A"]), (0.0, 0.0))


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = (E.OOF_DIR, E.TAG, E.CHUNK_ROWS, E.forecast, E.DEADLINE)
        E.OOF_DIR, E.TAG, E.CHUNK_ROWS, E.DEADLINE = self.tmp, "t109", 2, None

    def tearDown(self):
        E.OOF_DIR, E.TAG, E.CHUNK_ROWS, E.forecast, E.DEADLINE = self.old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_and_resume(self):
        p = toy_panel()
        rows = pd.DataFrame({"Code": ["1111", "1111", "2222"],
                             "Date": pd.to_datetime(["2024-01-02", "2024-01-06", "2024-01-03"]), "fold": [0, 0, 1]})
        st, en = E.context_bounds(p, rows, 3)
        calls = []

        def fake(pipe, items, bs):
            calls.append(len(items))
            return np.stack([np.full((3, E.HORIZON), float(len(np.atleast_1d(it))), dtype=np.float32)
                             for it in items])

        E.forecast = fake
        pipe = SimpleNamespace(quantiles=LV)
        q1 = E.run_arm("Z0", pipe, "rev", rows, p, st, en, {}, None, [], None)
        self.assertEqual(q1.shape, (3, 3, E.HORIZON))
        self.assertEqual(list(q1[:, 0, 0]), [2.0, 3.0, 3.0])     # 文脈の長さ（行の並びのまま戻る）
        self.assertEqual(sum(calls), 3)
        q2 = E.run_arm("Z0", None, "rev", rows, p, st, en, {}, None, [], None)
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(sum(calls), 3)                          # 2回目は保存済みを読むだけ
        self.assertIsNone(E.run_arm("Z0", None, "other", rows, p, st, en, {}, None, [], None))
        # 日々の取り込みで後ろに行が増えても、前の束はそのまま読める（束の名前は行と入力の値で決まる）
        more = pd.concat([rows, pd.DataFrame({"Code": ["2222"], "Date": [pd.Timestamp("2024-01-04")], "fold": [1]})],
                         ignore_index=True)
        st2, en2 = E.context_bounds(p, more, 3)
        self.assertIsNone(E.run_arm("Z0", None, "rev", more, p, st2, en2, {}, None, [], None))
        q3 = E.run_arm("Z0", pipe, "rev", more, p, st2, en2, {}, None, [], None)
        self.assertEqual(sum(calls), 5)                          # 窓0 の束は読み、行が増えた窓1 の束（2行）だけ計算し直した
        np.testing.assert_array_equal(q3[:3], q1)

    def test_tops_across_runs(self):
        """前の回の A の上位列で計算した束が残っていれば、その窓はその列を選び、計算し直さない。"""
        p = toy_panel()
        rows = pd.DataFrame({"Code": ["1111", "1111", "2222"],
                             "Date": pd.to_datetime(["2024-01-02", "2024-01-06", "2024-01-03"]), "fold": [0, 0, 1]})
        st, en = E.context_bounds(p, rows, 3)
        close = p["close"].to_numpy(dtype=float)
        cov = np.arange(30, dtype=np.float32).reshape(10, 3)
        cov_cols, cov_row = ["a", "b", "c"], np.arange(10)
        calls = []

        def fake(pipe, items, bs):
            calls.append(len(items))
            return np.ones((len(items), 3, E.HORIZON), dtype=np.float32)

        E.forecast = fake
        pipe = SimpleNamespace(quantiles=LV)
        old = {0: ["a", "b"], 1: ["a", "b"]}
        q1 = E.run_arm("Z30", pipe, "rev", rows, p, st, en, old, cov, cov_cols, cov_row)
        self.assertEqual(sum(calls), 3)

        def saved(f, lst):
            return E.saved_rows("rev", rows, close, st, en, f, lst, cov, cov_cols, cov_row)

        self.assertEqual((saved(0, ["a", "b"]), saved(1, ["a", "b"]), saved(0, ["a", "c"])), (2, 1, 0))
        now = {0: ["a", "c"], 1: ["a", "c"]}                     # 学び直した A で30位前後が入れ替わった
        tops, why = E.choose_tops(now, {"old": old}, saved)
        self.assertEqual(tops, old)
        self.assertEqual(why, {0: ("old", 2), 1: ("old", 1)})
        q2 = E.run_arm("Z30", None, "rev", rows, p, st, en, tops, cov, cov_cols, cov_row)
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(sum(calls), 3)                          # 読むだけ
        self.assertIsNone(E.run_arm("Z30", None, "rev", rows, p, st, en, now, cov, cov_cols, cov_row))

    def test_choose_tops_rules(self):
        seen = []

        def saved(f, lst):
            seen.append((f, tuple(lst)))
            return {(1, ("a", "c")): 10, (1, ("a", "b")): 4, (2, ("b", "c")): 3}.get((f, tuple(lst)), 0)

        cur = {1: ["a", "b"], 2: ["a", "b"], 3: ["a", "b"]}
        cands = {"y": {1: ["a", "c"], 2: ["a", "b"]}, "x": {1: ["a", "b"], 2: ["b", "c"], 3: ["b", "c"]}}
        tops, why = E.choose_tops(cur, cands, saved)
        self.assertEqual(tops, {1: ["a", "c"], 2: ["b", "c"], 3: ["a", "b"]})
        self.assertEqual(why, {1: ("y", 10), 2: ("x", 3), 3: ("今回", 0)})   # 同じ行の数なら今回の列
        self.assertEqual(len(seen), len(set(seen)))                          # 同じ列の組は1回だけ数える

    def test_tops_candidates(self):
        elig = ["a", "b", "c"]
        g = pd.DataFrame({"col": ["a", "b", "c", "x"] * 2, "gain": [1.0, 3.0, 2.0, 9.0, 5.0, 1.0, 1.0, 0.0],
                          "fold": [1] * 4 + [2] * 4})
        g.to_parquet(E.apath("A_aaaa_bbbb_gain.parquet"), index=False)
        g[g["col"] != "c"].to_parquet(E.apath("A_cccc_dddd_gain.parquet"), index=False)   # 列の組が違う回は使わない
        got = E.tops_candidates(elig, n=2)
        self.assertEqual(got, {"aaaa_bbbb": {1: ["b", "c"], 2: ["a", "b"]}})

    def test_chunk_key(self):
        keys = pd.DataFrame({"Code": ["1111"], "Date": [pd.Timestamp("2024-01-02")]})
        a = [{"target": np.array([1.0, 2.0], dtype=np.float32),
              "past_covariates": {"x": np.array([0.5, np.nan], dtype=np.float32)}}]
        b = [{"target": np.array([1.0, 2.0], dtype=np.float32),
              "past_covariates": {"x": np.array([0.5, 0.7], dtype=np.float32)}}]
        meta = {"arm": "Z30"}
        self.assertEqual(E.chunk_key(meta, keys, a), E.chunk_key(meta, keys, [dict(a[0])]))
        self.assertNotEqual(E.chunk_key(meta, keys, a), E.chunk_key(meta, keys, b))
        self.assertNotEqual(E.chunk_key(meta, keys, a), E.chunk_key({"arm": "Z0"}, keys, a))
        self.assertNotEqual(E.chunk_key(meta, keys, a),
                            E.chunk_key(meta, keys.assign(Code="2222"), a))


class BaseModel(unittest.TestCase):
    def test_same_as_common_part(self):
        try:
            import e41_stop_loss as E41
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"e41_stop_loss を読めない: {exc}")
        rng = np.random.default_rng(3)
        n = 1600
        df = pd.DataFrame({"Code": [f"{i:04d}" for i in range(n)],
                           "Date": pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) // 2, unit="D"),
                           "a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
        df["label"] = (df["a"] + rng.normal(scale=1.0, size=n) > 0.8).astype(int)
        df["ret_o1_20"] = 0.0
        df["ret_o1_40"] = 0.0
        f = SimpleNamespace(index=0, train_end="2021-08-31", test_start="2021-09-01", test_end="2022-03-31")
        params = {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "verbose": -1}
        got = E.fold_fit(df, ["a", "b", "c"], params, 42, f)
        want = E41.oof_folds("lgbm", df, ["a", "b", "c"], params, 42, [f])
        np.testing.assert_allclose(got[1], want["score"].to_numpy(), rtol=0, atol=1e-12)
        self.assertEqual(list(got[2].index), ["a", "b", "c"])
        self.assertGreater(got[2]["a"], got[2]["b"])


if __name__ == "__main__":
    unittest.main()
