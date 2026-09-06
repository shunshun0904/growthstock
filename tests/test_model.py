#!/usr/bin/env python3
"""
research/train_model.py の評価ロジックの単体テスト。

特徴量を足すたびに「ベースラインをわずかに上回った」が出る。
その差が誤差かどうかを判定するのがブートストラップなので、
判定器そのものが壊れていないことを既知のケースで固定する。

  python3 tests/test_model.py
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import pandas as pd  # noqa: E402

from train_model import clean_score, evaluate, paired_bootstrap  # noqa: E402
from walkforward import (  # noqa: E402
    DEFAULT_PRESETS, REFERENCE, make_folds, run, sign_test, summarize,
    uncovered_presets,
)
from within_date_signal import conditional, marginal  # noqa: E402
from validate_metrics import check_identities, describe  # noqa: E402


def _synthetic(n=8000, rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    return y, rng


class TestCleanScore(unittest.TestCase):
    def test_missing_becomes_worst(self):
        out = clean_score(np.array([3.0, np.nan, 1.0, np.inf]))
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out.argmax(), 0)          # 3.0 が最良のまま
        self.assertLess(out[1], out[2])            # 欠測は最下位の 1.0 より下
        self.assertLess(out[3], out[2])

    def test_all_missing_does_not_crash(self):
        out = clean_score(np.array([np.nan, np.nan]))
        self.assertTrue(np.isfinite(out).all())

    def test_missing_never_ranks_above_a_real_score(self):
        """欠測を高評価にしてしまうと、予測できない銘柄を推奨してしまう。"""
        y = np.array([0, 1, 0, 0])
        r = evaluate("t", y, np.array([np.nan, 0.9, 0.1, 0.2]))
        self.assertEqual(r["precision@1%"], 1.0)   # 上位1件は本物の正例


class TestPairedBootstrap(unittest.TestCase):
    """既知の答えがあるケースで、判定が正しく出ることを固定する。"""

    def test_clearly_better_model_is_significant(self):
        y, rng = _synthetic(seed=1)
        ref = y * 0.5 + rng.normal(0, 1, len(y))
        strong = y * 2.0 + rng.normal(0, 1, len(y))
        out = paired_bootstrap(y, {"ref": ref, "strong": strong},
                               reference="ref", n_boot=200, seed=1)
        r = out[0]
        self.assertGreater(r["ci_low"], 0)
        self.assertGreater(r["p_better"], 0.99)

    def test_identical_scores_give_exactly_zero(self):
        """対応のある比較になっていれば、同じスコアの差は厳密に0になる。
        ここが0でなければリサンプルが対になっていない。"""
        y, rng = _synthetic(seed=2)
        ref = y * 0.5 + rng.normal(0, 1, len(y))
        out = paired_bootstrap(y, {"ref": ref, "copy": ref.copy()},
                               reference="ref", n_boot=100, seed=2)
        r = out[0]
        self.assertEqual(r["diff"], 0.0)
        self.assertEqual(r["ci_low"], 0.0)
        self.assertEqual(r["ci_high"], 0.0)

    def test_pure_noise_is_significantly_worse(self):
        y, rng = _synthetic(seed=3)
        ref = y * 0.5 + rng.normal(0, 1, len(y))
        out = paired_bootstrap(y, {"ref": ref, "noise": rng.normal(0, 1, len(y))},
                               reference="ref", n_boot=200, seed=3)
        self.assertLess(out[0]["ci_high"], 0)

    def test_tiny_difference_is_not_called_significant(self):
        """本命のケース。ほぼ同じ2つのモデルを『有意』と言わないこと。"""
        y, rng = _synthetic(seed=4)
        ref = y * 0.5 + rng.normal(0, 1, len(y))
        nudged = ref + rng.normal(0, 0.01, len(y))   # ごくわずかに違うだけ
        out = paired_bootstrap(y, {"ref": ref, "nudged": nudged},
                               reference="ref", n_boot=300, seed=4)
        r = out[0]
        self.assertLessEqual(r["ci_low"], 0)
        self.assertGreaterEqual(r["ci_high"], 0)

    def test_resamples_without_positives_are_redrawn(self):
        """正例が極端に少なくても NaN を返さない。"""
        rng = np.random.default_rng(5)
        y = np.zeros(500, dtype=int)
        y[:3] = 1
        ref = rng.normal(0, 1, 500)
        out = paired_bootstrap(y, {"ref": ref, "b": rng.normal(0, 1, 500)},
                               reference="ref", n_boot=50, seed=5)
        for key in ("diff", "ci_low", "ci_high", "p_better"):
            self.assertFalse(np.isnan(out[0][key]), key)


class TestSignTest(unittest.TestCase):
    """符号検定。フォールド数が少ないので、何勝すれば有意かを把握しておく。"""

    def test_unanimous_is_significant(self):
        self.assertAlmostEqual(sign_test(8, 0), 2 / 256)
        self.assertLess(sign_test(9, 0), 0.05)

    def test_even_split_is_not_significant(self):
        self.assertEqual(sign_test(4, 4), 1.0)

    def test_symmetric(self):
        self.assertEqual(sign_test(7, 2), sign_test(2, 7))

    def test_nine_folds_needs_eight_wins(self):
        """9フォールドでは 8勝1敗 でようやく有意。7勝2敗では足りない。
        レポートの判定を読むときにこの厳しさを踏まえる必要がある。"""
        self.assertLess(sign_test(8, 1), 0.05)
        self.assertGreater(sign_test(7, 2), 0.05)

    def test_no_folds_is_nan(self):
        self.assertTrue(np.isnan(sign_test(0, 0)))


class TestMakeFolds(unittest.TestCase):
    def _folds(self, first="2017-09-29", last="2025-11-28", **kw):
        kw.setdefault("min_train_months", 36)
        kw.setdefault("test_months", 6)
        kw.setdefault("step_months", 6)
        kw.setdefault("embargo_days", 180)
        return make_folds(pd.Series(pd.to_datetime([first, last])), **kw)

    def test_test_windows_never_overlap(self):
        """重なると同じサンプルが2フォールドに入り、独立でなくなる。"""
        folds = self._folds()
        self.assertGreater(len(folds), 1)
        for a, b in zip(folds, folds[1:]):
            self.assertLess(a.test_end, b.test_start)

    def test_embargo_is_respected(self):
        """訓練最終日のラベルがテスト期間に食い込まないこと。"""
        for f in self._folds():
            gap = (pd.Timestamp(f.test_start) - pd.Timestamp(f.train_end)).days
            self.assertGreaterEqual(gap, int(180 * 1.45) - 1)

    def test_training_window_expands(self):
        folds = self._folds()
        ends = [pd.Timestamp(f.train_end) for f in folds]
        self.assertEqual(ends, sorted(ends))
        self.assertEqual(len({f.train_start for f in folds}), 1)

    def test_never_runs_past_the_data(self):
        for f in self._folds():
            self.assertLessEqual(pd.Timestamp(f.test_end), pd.Timestamp("2025-11-28"))

    def test_too_short_a_period_yields_no_folds(self):
        self.assertEqual(self._folds(last="2019-01-01"), [])


class TestWalkForwardEndToEnd(unittest.TestCase):
    """合成データで最後まで通ること。列名や集計の取り違えを検出する。"""

    def _dataset(self, n_dates=60, n_codes=120, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2017-09-29", periods=n_dates, freq="ME")
        rows = []
        for d in dates:
            r_high = rng.uniform(0, 95, n_codes)
            # r_high が高いほど正例になりやすい合成ラベル
            p = 1 / (1 + np.exp(-(r_high - 70) / 10))
            frame = {
                "Date": d,
                "Code": [f"{i:04d}" for i in range(n_codes)],
                "r_high": r_high,
                "r_high_r": rng.uniform(0, 1, n_codes),
                "volume_trend": rng.normal(0, 1, n_codes),
                "label": (rng.random(n_codes) < p).astype(int),
            }
            # ベースラインの8軸スコアが必要とする列
            for c in ("ROE_q0", "credit_ratio", "eps_growth_q0", "market_cap",
                      "op_margin_q0", "progress_vs_base", "sales_growth_q0",
                      "tv_ma20"):
                frame[c] = rng.normal(0, 1, n_codes)
            rows.append(pd.DataFrame(frame))
        return pd.concat(rows, ignore_index=True)

    def test_runs_and_summarises(self):
        df = self._dataset()
        folds = make_folds(pd.to_datetime(df["Date"]), min_train_months=24,
                           test_months=6, step_months=6, embargo_days=20)
        self.assertGreaterEqual(len(folds), 2)

        # 基準（無情報）は baseline_scores が必ず入れるので、
        # 評価するセットは何でもよい
        import features as Fx
        Fx.GROUPS["_t"] = ["r_high", "volume_trend"]
        Fx.PRESETS["_t"] = ["_t"]
        try:
            res = run(df, ["_t"], folds)
        finally:
            del Fx.GROUPS["_t"], Fx.PRESETS["_t"]

        self.assertTrue(res["folds"])
        self.assertTrue(res["summary"])
        for s in res["summary"]:
            self.assertEqual(s["wins"] + s["losses"] <= s["n_folds"], True)
            self.assertGreaterEqual(s["best_diff"], s["worst_diff"])
        # 基準そのものは要約に出さない（自分との差は常に0で無意味）
        self.assertNotIn(REFERENCE, [s["name"] for s in res["summary"]])

    def test_reference_scores_exactly_the_base_rate(self):
        """
        基準は無情報（全件同じスコア）。全件同点の PR-AUC はその窓の
        正例率に一致する。だから「差」は正例率をどれだけ上回ったかになる。
        この性質が崩れると、差の解釈が変わってしまう。
        """
        from train_model import baseline_scores, evaluate, REFERENCE_MODEL
        df = self._dataset()
        y = df["label"].to_numpy(dtype=int)
        sc = baseline_scores(df)
        self.assertEqual(list(sc), [REFERENCE_MODEL])
        r = evaluate(REFERENCE_MODEL, y, sc[REFERENCE_MODEL])
        self.assertAlmostEqual(r["pr_auc"], float(y.mean()), places=6)
        self.assertAlmostEqual(r["roc_auc"], 0.5, places=6)

    def test_summarise_counts_wins_correctly(self):
        per_fold = [
            {"results": [{"name": REFERENCE, "pr_auc": 0.3, "diff_vs_ref": 0.0,
                          "lift@5%": 1.0},
                         {"name": "m", "pr_auc": 0.4, "diff_vs_ref": +0.1,
                          "lift@5%": 1.2}]},
            {"results": [{"name": REFERENCE, "pr_auc": 0.3, "diff_vs_ref": 0.0,
                          "lift@5%": 1.0},
                         {"name": "m", "pr_auc": 0.2, "diff_vs_ref": -0.1,
                          "lift@5%": 0.8}]},
        ]
        s = summarize(per_fold)[0]
        self.assertEqual((s["wins"], s["losses"], s["n_folds"]), (1, 1, 2))
        self.assertAlmostEqual(s["mean_diff"], 0.0)
        self.assertAlmostEqual(s["worst_diff"], -0.1)
        self.assertAlmostEqual(s["best_diff"], +0.1)


class TestWithinDateSignal(unittest.TestCase):
    """日付内 AUC の診断。LTR に見込みがあるかの判断材料になるので、
    「信号があるとき見つかる / 無いとき見つけない」の両方を固定する。"""

    def _frame(self, n_dates=40, n_codes=200, seed=0, signal=True):
        rng = np.random.default_rng(seed)
        rows = []
        for d in pd.date_range("2020-01-31", periods=n_dates, freq="ME"):
            r_high = rng.uniform(0, 95, n_codes)
            useful = rng.normal(0, 1, n_codes)
            noise = rng.normal(0, 1, n_codes)
            lin = (r_high - 60) / 15 + (useful * 1.2 if signal else 0.0)
            # 局面ごとに正例率を大きく動かす（日付内AUCが影響されないことの確認）
            lin += rng.normal(0, 2)
            p = 1 / (1 + np.exp(-lin))
            rows.append(pd.DataFrame({
                "Date": d, "r_high": r_high, "useful": useful, "noise": noise,
                "label": (rng.random(n_codes) < p).astype(int)}))
        return pd.concat(rows, ignore_index=True)

    def test_finds_a_real_within_date_signal(self):
        df = self._frame(signal=True)
        res = {r["feature"]: r for r in conditional(df, ["useful", "noise"])}
        self.assertGreater(res["useful"]["mean_auc"], 0.6)
        self.assertLess(res["useful"]["p_sign"], 0.05)

    def test_does_not_invent_signal_from_noise(self):
        df = self._frame(signal=True)
        res = {r["feature"]: r for r in conditional(df, ["useful", "noise"])}
        self.assertLess(res["noise"]["abs_edge"], 0.02)

    def test_reports_nothing_when_only_r_high_matters(self):
        """R_high だけが効く世界では、条件付けると全部 0.5 付近になるはず。
        ここが誤って『信号あり』と出ると、無駄な LTR 実装に進んでしまう。"""
        df = self._frame(signal=False)
        for r in conditional(df, ["useful", "noise"]):
            self.assertLess(r["abs_edge"], 0.02, r["feature"])

    def test_rank_transform_does_not_change_within_date_auc(self):
        """順位化は日付内の単調変換なので AUC は変わらない。
        この前提で順位列を測定対象から外している。"""
        import build_dataset as B
        df = self._frame(signal=True)
        df = B.add_cross_sectional_ranks(df, ["useful"])
        res = {r["feature"]: r for r in marginal(df, ["useful", "useful_r"])}
        self.assertAlmostEqual(res["useful"]["mean_auc"],
                               res["useful_r"]["mean_auc"], places=10)


class TestMetricValidation(unittest.TestCase):
    """
    PER/PBR/ROE/ROA は割り算で作るので分母が小さいと発散する。
    「列はあるが値が壊れている」状態を検出できることを固定する。
    """

    def _clean(self, n=800, seed=0):
        rng = np.random.default_rng(seed)
        close = rng.uniform(100, 5000, n)
        eps = rng.normal(50, 80, n)
        bps = rng.uniform(50, 3000, n)
        return pd.DataFrame({
            "close": close, "eps_ttm": eps, "BPS": bps,
            "earnings_yield": eps / close * 100,
            "per": np.where(eps > 0, close / eps, np.nan),
            "book_yield": bps / close,
            "pbr": np.where(bps > 0, close / bps, np.nan),
        })

    def test_identities_hold_on_correct_data(self):
        checks = {c["name"]: c for c in check_identities(self._clean())}
        self.assertLess(checks["per × earnings_yield == 100"]["max_rel_err"], 1e-9)
        self.assertLess(checks["pbr × book_yield == 1"]["max_rel_err"], 1e-9)

    def test_detects_a_broken_ratio(self):
        """片方の計算式がずれていれば恒等式で分かる。"""
        df = self._clean()
        df["per"] = df["close"] / (df["eps_ttm"] * 1.05)
        c = check_identities(df)[0]
        self.assertGreater(c["max_rel_err"], 1e-3)

    def test_detects_sign_mismatch(self):
        """株価は正なので per の符号は EPS の符号と一致するはず。"""
        df = self._clean()
        df["per"] = -df["per"]
        c = [x for x in check_identities(df) if "sign(per)" in x["name"]][0]
        self.assertGreater(c["mismatches"], 0)

    def test_detects_infinities(self):
        st = describe(pd.Series([1.0, np.inf, 2.0, -np.inf]), (0, 10))
        self.assertEqual(st["n_inf"], 2)

    def test_detects_an_all_zero_column(self):
        st = describe(pd.Series([0.0] * 50), (0, 10))
        self.assertEqual(st["zero_pct"], 100.0)

    def test_detects_an_all_missing_column(self):
        st = describe(pd.Series([np.nan] * 50), (0, 10))
        self.assertEqual(st["present_pct"], 0.0)

    def test_counts_values_outside_the_plausible_range(self):
        """目安外は「異常」ではなく件数として出す。
        ROE が100%を超えることは実在するので、切り捨ててはいけない。"""
        st = describe(pd.Series([10.0, 50.0, 250.0, 900.0]), (-500.0, 500.0))
        self.assertEqual(st["outside_plausible"], 1)   # 900 のみ
        self.assertEqual(st["n_negative"], 0)


class TestWalkForwardCoversNewPresets(unittest.TestCase):
    """
    DEFAULT_PRESETS は手で維持する一覧なので、
    プリセットを足しても評価対象に入れ忘れる事故が実際に起きた。
    """

    def test_valuation_presets_are_evaluated(self):
        """PER/PBR/ROA を足した効果を見るのが目的なので、既定に入れる。"""
        for name in ("fundamental_v2", "rank_fundamental_v2", "valuation_only"):
            self.assertIn(name, DEFAULT_PRESETS)

    def test_paired_comparisons_are_complete(self):
        """絶対値版と順位版は対で評価する。片方だけだと比較にならない。"""
        for base in ("price_only", "technical", "all", "fundamental",
                     "fundamental_v2"):
            self.assertIn(base, DEFAULT_PRESETS, base)
            self.assertIn(f"rank_{base}", DEFAULT_PRESETS, f"rank_{base}")

    def test_uncovered_presets_are_reported(self):
        """漏れを黙って通さないこと。"""
        import features as F
        self.assertEqual(uncovered_presets(list(F.PRESETS)), [])
        self.assertIn("price_only", uncovered_presets(["technical"]))

    def test_every_default_preset_exists(self):
        import features as F
        for name in DEFAULT_PRESETS:
            self.assertIn(name, F.PRESETS, name)


class TestTuningDoesNotSeeTestData(unittest.TestCase):
    """
    ハイパーパラメータ探索は「良さそうな設定を選ぶ」作業なので、
    評価に使う期間を一度でも見ればリークになり、以降の評価が全部無効になる。
    探索期間がすべてのテスト窓より前で打ち切られていることを固定する。
    """

    def _folds(self):
        from train_model import EMBARGO_DAYS
        return make_folds(pd.Series(pd.to_datetime(["2017-09-29", "2025-11-28"])),
                          min_train_months=36, test_months=6, step_months=6,
                          embargo_days=EMBARGO_DAYS), EMBARGO_DAYS

    def test_cutoff_precedes_every_test_window(self):
        folds, _ = self._folds()
        cutoff = pd.Timestamp(folds[0].train_end)
        for f in folds:
            self.assertLess(cutoff, pd.Timestamp(f.test_start), f.index)

    def test_cutoff_respects_the_embargo(self):
        """打ち切り日のラベルが最初のテスト窓に食い込まないこと。"""
        folds, embargo = self._folds()
        cutoff = pd.Timestamp(folds[0].train_end)
        gap = (pd.Timestamp(folds[0].test_start) - cutoff).days
        self.assertGreaterEqual(gap, int(embargo * 1.45) - 1)

    def test_cutoff_is_derived_from_folds_not_hardcoded(self):
        """
        フォールドの切り方を変えたら打ち切りも動くこと。
        固定値を書いていると、切り方を変えた瞬間に静かにリークする。
        """
        from train_model import EMBARGO_DAYS
        dates = pd.Series(pd.to_datetime(["2017-09-29", "2025-11-28"]))
        a = make_folds(dates, min_train_months=36, test_months=6,
                       step_months=6, embargo_days=EMBARGO_DAYS)[0].train_end
        b = make_folds(dates, min_train_months=48, test_months=6,
                       step_months=6, embargo_days=EMBARGO_DAYS)[0].train_end
        self.assertNotEqual(a, b)


class TestTuning(unittest.TestCase):
    """探索そのものの挙動。"""

    def _frame(self, n_dates=30, n=200, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for d in pd.date_range("2018-01-31", periods=n_dates, freq="ME"):
            x1 = rng.normal(0, 1, n)
            p = 1 / (1 + np.exp(-(1.5 * x1 - 1.0)))
            rows.append(pd.DataFrame({
                "Date": d, "x1": x1, "x2": rng.normal(0, 1, n),
                "label": (rng.random(n) < p).astype(int)}))
        return pd.concat(rows, ignore_index=True)

    def test_inner_split_is_chronological(self):
        """
        内側検証をランダムに取ると、同一銘柄の隣接月が両側に入って
        検証が簡単になりすぎ、必ず楽観的なパラメータが選ばれる。
        """
        import tuning
        tr, va = tuning.chronological_split(self._frame())
        self.assertLess(tr["Date"].max(), va["Date"].min())
        self.assertGreater(len(tr), 0)
        self.assertGreater(len(va), 0)

    def test_reported_score_matches_the_returned_params(self):
        """
        報告した CV スコアが、返したパラメータで再現できること。

        「良いスコアを報告しながら別のパラメータを返す」が起きると、
        結果を信じてよいかが分からなくなる。
        （既定値より必ず良くなることは保証しない。試行数が少なければ
        TPE が既定値より良い設定を見つけないことは普通にある）
        """
        import tuning
        df = self._frame()
        cols = ["x1", "x2"]
        best = tuning.tune(df, cols, n_trials=5, verbose=False, n_splits=3)
        folds = tuning.year_folds(df, n_splits=3)
        got = float(np.mean([
            tuning._fit_one(best, tr, va, cols, early_stopping=False)[0]
            for tr, va in folds]))
        self.assertAlmostEqual(got, tuning.LAST_CV["mean_pr_auc"], places=3)

    def test_tree_count_is_fixed_during_the_search(self):
        """
        本数を early stopping に決めさせると、試行ごとに別の大きさの
        モデルを比べることになる。固定して他のパラメータだけを比べる。
        """
        import tuning
        best = tuning.tune(self._frame(), ["x1", "x2"], n_trials=3, verbose=False,
                           n_splits=3)
        self.assertEqual(best["n_estimators"], tuning.SEARCH_N_ESTIMATORS)

    def test_falls_back_to_defaults_without_both_classes(self):
        import tuning
        df = self._frame()
        df["label"] = 0
        best = tuning.tune(df, ["x1", "x2"], n_trials=3, verbose=False)
        self.assertEqual(best["learning_rate"],
                         tuning.DEFAULT_PARAMS["learning_rate"])

    def test_params_for_returns_defaults_for_unknown_preset(self):
        import tuning
        p = tuning.params_for("__no_such_preset__", {})
        self.assertEqual(p["num_leaves"], tuning.DEFAULT_PARAMS["num_leaves"])

    def test_scale_pos_weight_handles_imbalance(self):
        import tuning
        y = np.array([0] * 90 + [1] * 10)
        self.assertAlmostEqual(tuning.scale_pos_weight(y), 9.0)


class TestTunedParamsArePersisted(unittest.TestCase):
    """
    探索結果は research/lgbm_params.json に残す。

    最初これを research/_data/ に置いていたが、そこは .gitignore されており、
    ワークフローのコンテナが終わると消えていた。
    結果、毎回6分かけて探索し直していたのに、値はどこにも残っていなかった。
    """

    def test_params_path_is_not_in_the_ignored_data_dir(self):
        import tuning
        self.assertNotIn(f"_data{os.sep}", tuning.PARAMS_PATH)
        self.assertTrue(tuning.PARAMS_PATH.endswith("lgbm_params.json"))

    def test_data_dir_is_gitignored(self):
        """前提の確認。ここが変わったら PARAMS_PATH の判断も変わる。"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ignore = open(os.path.join(root, ".gitignore"), encoding="utf-8").read()
        self.assertIn("research/_data/", ignore)

    def test_falls_back_to_defaults_when_file_is_absent(self):
        import tuning
        p = tuning.params_for("anything", {})
        self.assertEqual(p["num_leaves"], tuning.DEFAULT_PARAMS["num_leaves"])


class TestYearStratifiedFolds(unittest.TestCase):
    """
    探索の評価は年で層別した k 分割。

    時系列分割はこの規模では推定が安定しなかった
    （実測で分割ごとの PR-AUC が 0.036〜0.230 と6倍以上ばらついた）。
    各フォールドを同じ年構成にすれば、局面の当たり外れが相殺される。
    """

    def _df(self, spec=((2018, 60), (2019, 300), (2020, 900), (2021, 1200))):
        rng = np.random.default_rng(0)
        rows = []
        for year, n in spec:
            a = rng.normal(0, 1, n)
            rows.append(pd.DataFrame({
                "Date": pd.date_range(f"{year}-01-05", f"{year}-12-25", periods=n),
                "a": a,
                "label": (rng.random(n) < 1 / (1 + np.exp(-(a * 0.7 - 2.6)))).astype(int)}))
        return pd.concat(rows, ignore_index=True)

    def test_every_fold_has_the_same_year_mix(self):
        from tuning import year_folds
        df = self._df()
        folds = year_folds(df, n_splits=5, seed=0)
        self.assertEqual(len(folds), 5)
        mixes = []
        for _, va in folds:
            y = pd.to_datetime(va["Date"]).dt.year.value_counts(normalize=True)
            mixes.append(y.sort_index().round(2).to_dict())
        self.assertEqual(len(set(map(str, mixes))), 1, mixes)

    def test_positive_rate_is_balanced_across_folds(self):
        """年だけで層別すると正例数が偏る。層は 年の束 × ラベル にする。"""
        from tuning import year_folds
        rates = [v["label"].mean() for _, v in year_folds(self._df(), n_splits=5)]
        self.assertLess(float(np.std(rates)), 0.005)

    def test_small_years_are_merged(self):
        """
        StratifiedKFold は分割数未満の層で落ちる。
        正例の少ない年（初期は決算4期分の履歴が要るぶん少ない）は隣に寄せる。
        """
        from tuning import year_folds
        df = self._df(spec=((2017, 20), (2018, 60), (2019, 300), (2020, 900)))
        folds = year_folds(df, n_splits=5, seed=0)   # 落ちなければよい
        self.assertEqual(len(folds), 5)

    def test_search_fixes_the_number_of_trees(self):
        """
        本数を early stopping に決めさせると、試行ごとに別の大きさの
        モデルを比べることになる。固定して他のパラメータだけを比べる。
        """
        from tuning import tune, SEARCH_N_ESTIMATORS
        best = tune(self._df(), ["a"], n_trials=2, n_splits=5, scheme="year",
                    verbose=False)
        self.assertEqual(best["n_estimators"], SEARCH_N_ESTIMATORS)
        self.assertEqual(SEARCH_N_ESTIMATORS, 100)

    def test_unknown_scheme_stops(self):
        from tuning import tune
        with self.assertRaises(SystemExit):
            tune(self._df(), ["a"], n_trials=1, scheme="random", verbose=False)


class TestTuningFolds(unittest.TestCase):
    """
    時系列分割（既定ではないが残してある）。
    ランダム分割にすると同じ銘柄の隣接期間が訓練と検証の両方に入り、
    必ず楽観的なパラメータが選ばれる。
    """

    def _df(self, days=1000, per_day=3):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2018-07-01", periods=days, freq="D")
        n = days * per_day
        d = pd.DataFrame({"Date": np.repeat(dates, per_day),
                          "a": rng.normal(0, 1, n)})
        d["label"] = (rng.random(n) < 0.1).astype(int)
        return d

    def test_validation_always_follows_training(self):
        from tuning import time_series_folds
        folds = time_series_folds(self._df(), n_splits=5, embargo_days=60)
        self.assertEqual(len(folds), 5)
        for tr, va in folds:
            t, v = pd.to_datetime(tr["Date"]), pd.to_datetime(va["Date"])
            self.assertGreater(v.min(), t.max())

    def test_embargo_is_respected(self):
        """ラベルは先60営業日の情報を含む。隣接させると訓練が検証に食い込む。"""
        from tuning import time_series_folds
        folds = time_series_folds(self._df(), n_splits=5, embargo_days=60)
        for tr, va in folds:
            gap = (pd.to_datetime(va["Date"]).min()
                   - pd.to_datetime(tr["Date"]).max()).days
            self.assertGreaterEqual(gap, int(60 * 1.45) - 1)

    def test_training_window_expands(self):
        from tuning import time_series_folds
        folds = time_series_folds(self._df(), n_splits=5, embargo_days=60)
        sizes = [len(tr) for tr, _ in folds]
        self.assertEqual(sizes, sorted(sizes))
        starts = {pd.to_datetime(tr["Date"]).min() for tr, _ in folds}
        self.assertEqual(len(starts), 1)

    def test_record_of_the_search_is_not_passed_to_lightgbm(self):
        """_cv は探索の記録。LGBMClassifier に渡すと未知の引数で落ちる。"""
        from tuning import params_for
        store = {"x": {"learning_rate": 0.1, "_cv": {"mean_pr_auc": 0.2}}}
        got = params_for("x", store)
        self.assertAlmostEqual(got["learning_rate"], 0.1)
        self.assertFalse([k for k in got if k.startswith("_")])


class TestReportWithoutBaselines(unittest.TestCase):
    """
    単変量ベースラインを廃止したあと、レポートが基準の PR-AUC を
    baselines から引き続けていて IndexError で落ちていた。
    基準はモデルなので experiments から引く。
    """

    def test_reference_pr_auc_comes_from_experiments(self):
        from train_model import reference_pr_auc
        exps = [{"preset": "technical",
                 "results": {"test": [{"name": "LightGBM", "pr_auc": 0.13}]}}]
        self.assertAlmostEqual(
            reference_pr_auc(exps, {"test": []}, "LightGBM [technical]"), 0.13)
        self.assertIsNone(
            reference_pr_auc(exps, {"test": []}, "LightGBM [none]"))

    def test_report_survives_empty_baselines(self):
        import argparse
        from train_model import _report
        n = 40
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "Code": ["1"] * n,
            "label": ([0] * 35) + ([1] * 5)})
        parts = {"train": df.iloc[:20], "val": df.iloc[20:30], "test": df.iloc[30:]}
        exps = [{"preset": "technical", "n_features": 3,
                 "groups": ["price"],
                 "results": {"val": [{"name": "LightGBM", "pr_auc": 0.1,
                                      "roc_auc": 0.5, "precision@1%": 0.1,
                                      "precision@5%": 0.1, "lift@5%": 1.0,
                                      "base_rate": 0.1, "n": 10}],
                             "test": [{"name": "LightGBM", "pr_auc": 0.13,
                                       "roc_auc": 0.6, "precision@1%": 0.2,
                                       "precision@5%": 0.2, "lift@5%": 1.5,
                                       "base_rate": 0.1, "n": 10}]}}]
        boot = [{"name": "LightGBM [all]", "pr_auc": 0.12, "diff": -0.01,
                 "ci_low": -0.05, "ci_high": 0.03, "p_better": 0.3}]
        args = argparse.Namespace(val_start="2020-01-21", test_start="2020-01-31",
                                  n_boot=100)
        body = _report(df, parts, {"val": [], "test": []}, exps, args, boot,
                       "LightGBM [technical]")
        self.assertIn("0.1300", body)
        self.assertIn("LightGBM [technical]", body)


class TestStratifiedEvaluation(unittest.TestCase):
    """
    層内で比べる枠組みを固定する。

    母集団が高値更新日なので r_high は全件ほぼ100になり、層別の軸には使えない。
    軸は時価総額（+20%上昇の起きやすさが規模で大きく違う。実測で
    〜100億 43.9% に対し 3000億〜 18.7%）。
    ここは stratified_eval.STRATIFY_BY を直接使い、軸を変えたら
    テストも一緒に動くようにする。
    """

    def _frame(self, n_dates=12, n=400, seed=0):
        from stratified_eval import STRATIFY_BY
        rng = np.random.default_rng(seed)
        rows = []
        for d in pd.date_range("2020-01-31", periods=n_dates, freq="ME"):
            x = rng.uniform(10, 95, n)
            rows.append(pd.DataFrame({"Date": d, STRATIFY_BY: x,
                                      "label": (rng.random(n) < x / 200).astype(int)}))
        return pd.concat(rows, ignore_index=True)

    def test_strata_are_assigned_within_each_date(self):
        """局面で分布が動くので、日付をまたいで切ってはいけない。"""
        from stratified_eval import assign_strata, N_STRATA
        df = self._frame()
        st = assign_strata(df)
        for _, g in df.assign(_s=st).groupby("Date"):
            self.assertEqual(g["_s"].nunique(), N_STRATA)

    def test_axis_range_is_narrow_inside_a_stratum(self):
        """層内では軸の差がほとんど無いことを確認する。これが枠組みの前提。"""
        from stratified_eval import assign_strata, STRATIFY_BY
        df = self._frame().assign(_s=lambda d: assign_strata(d))
        overall = df[STRATIFY_BY].max() - df[STRATIFY_BY].min()
        for (_, _), g in df.groupby(["Date", "_s"]):
            self.assertLess(g[STRATIFY_BY].max() - g[STRATIFY_BY].min(), overall / 2)

    def test_evaluate_within_rejects_tiny_or_degenerate_groups(self):
        from stratified_eval import evaluate_within
        self.assertIsNone(evaluate_within(np.array([1, 0]), np.array([1.0, 0.0])))
        y = np.zeros(200, dtype=int)
        self.assertIsNone(evaluate_within(y, np.random.rand(200)))
        y[:20] = 1
        self.assertIsNotNone(evaluate_within(y, np.random.rand(200)))

    def test_lift_is_relative_to_the_stratum_base_rate(self):
        """層ごとに正例率が違うので、PR-AUC の生値では比べられない。"""
        from stratified_eval import evaluate_within
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.3).astype(int)
        res = evaluate_within(y, rng.random(500))
        self.assertAlmostEqual(res["base_rate"], y.mean())
        self.assertAlmostEqual(res["lift"], res["pr_auc"] / res["base_rate"])


class TestEmbargoFollowsTheLabel(unittest.TestCase):
    """
    母集団を高値更新日にしたことでラベルが変わり、
    確定に必要な将来日数が 180 -> 60 営業日になった。
    エンバーゴをラベルに追随させないとリークする。
    """

    def test_embargo_matches_the_rise_horizon(self):
        import build_dataset as B
        from train_model import EMBARGO_DAYS
        if B.POPULATION == "breakout":
            self.assertEqual(EMBARGO_DAYS, B.RISE_HORIZON)
        else:
            self.assertEqual(EMBARGO_DAYS, B.DEFAULT_LABEL.forward_needed)

    def test_embargo_is_not_hardcoded(self):
        """固定値だと、ラベルを変えた瞬間に静かにリークする。"""
        import inspect
        import train_model
        src = inspect.getsource(train_model)
        self.assertIn("EMBARGO_DAYS = (B.RISE_HORIZON", src)


class TestSmallFoldsAreSkipped(unittest.TestCase):
    """
    母集団を高値更新日に変えてサンプルが減り、
    実際に「テスト0件」のフォールドが出た。
    少数サンプルの PR-AUC は勝敗の符号がほぼ運で決まるので、評価に入れない。
    """

    def test_thresholds_are_set(self):
        import walkforward as W
        self.assertGreaterEqual(W.MIN_TEST_ROWS, 100)
        self.assertGreaterEqual(W.MIN_TEST_POSITIVES, 10)

    def test_empty_fold_produces_no_rows(self):
        import walkforward as W
        import features as Fx
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2020-01-31"] * 10),
            "Code": [f"{i}" for i in range(10)],
            "r_high": np.linspace(10, 90, 10),
            "volume_trend": np.ones(10),
            "label": [0] * 9 + [1],
        })
        for c in ("ROE_q0", "credit_ratio", "eps_growth_q0", "market_cap",
                  "op_margin_q0", "progress_vs_base", "sales_growth_q0",
                  "tv_ma20"):
            df[c] = 1.0
        fold = W.Fold(1, "2020-01-01", "2020-01-31", "2020-02-01", "2020-02-28")
        Fx.GROUPS["_t"] = ["r_high"]
        Fx.PRESETS["_t"] = ["_t"]
        try:
            res = W.run(df, ["_t"], [fold])
        finally:
            del Fx.GROUPS["_t"], Fx.PRESETS["_t"]
        self.assertEqual(res["folds"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
