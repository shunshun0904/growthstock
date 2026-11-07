#!/usr/bin/env python3
"""
実験128（research/exp/e128_double_half_year_tuned.py）。合成のデータで:

- 探索に使う行とホールドアウト: 境界は本番の train_model.holdout_bounds と同じ（エンバーゴは2倍の窓の125営業日を暦日に
  換算）。直近12か月の終わりは窓が最後まで揃っている最後の日で、それより後の行（売買が止まった銘柄だけ）はホールドアウトに
  入れない。探索の行はすべてエンバーゴより前
- 探索に渡すラベル: label の列を2倍のラベルに差し替える（+10% のラベルのままにしない）。片方のクラスしか無ければ止まる
- 読み方（C と A。2倍のラベル × +10% 用の木（B）は運用者の指示で比べない）: fund+val は小数1桁に丸めてすべての組で同じ向きの
  ときだけ決まる。値動きの荒さの5列の合計
- しきい値を上げたときの的中率: 上位はちょうど round(q×行数) 件（少なくとも1）で、スコアの高い順。銘柄ごとは期間ごとに
  しきい値を初めて越えた日の行だけ。期間ごとに上位を切ってまとめる。的中率の95%の幅（Wilson）。伸び（最高値の倍率 − 1）の
  中央値。最高値の倍率から作り直した「2倍に届いたか」が2倍のラベルとすべての行で合い、ずらせば見つかる
- 行方: 半年後の終値は窓の最後の日（t+h 行目）の終値、窓の途中で売買が止まった銘柄は最後の終値、先が足りず止まってもいない行は
  不明。+30%・+50% 以上・マイナスの数と中央値・平均。2倍に届かなかった銘柄だけの行方も出る
- 前の回の探索を使う: 列の指紋・ラベル・境界が同じときだけ読み、違えば止まる。前の回と件数・的中が同じかを数える。通しで2回
  回すと、2回目は探索をやり直さず、同じ銘柄が選ばれ（一致 20 / 20）、行方が出る
- 通しで回す（データの読み込みと本番の木は差し替え、探索は本物の tuning.tune を小さな木・2試行で）: 探索は本番と同じ分割
  （year_cap_date・5分割・PR-AUC）で2倍のラベルを学び、本番の research/lgbm_params.json は書かない（save_params を呼ばない・
  ファイルが変わらない）。表と2つの読み方・CV とホールドアウトの参考・しきい値を上げたときの的中率（ホールドアウトと前進分割、
  A・C）が出て、結果と探索したパラメータのファイルが書かれ、公開ログに銘柄コードを出さない

  python3 tests/test_e128_double_half_year_tuned.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F  # noqa: E402
import train_model  # noqa: E402
import tuning  # noqa: E402
import e127_triple_one_year as E  # noqa: E402
import e128_double_half_year_tuned as X  # noqa: E402
from test_e127_triple_one_year import bars_frame, synthetic_inputs  # noqa: E402


def frame(days, have_until, stopped_after=0):
    """1日1行。have_until までは窓が揃っている。その後ろに売買が止まった銘柄の行を stopped_after 行つなぐ。"""
    d = pd.bdate_range("2019-01-01", periods=days + stopped_after)
    have = np.arange(len(d)) < have_until
    x2 = (np.arange(len(d)) % 9 == 0).astype(int)
    return pd.DataFrame({"Date": d, "Code": "1", "x2": x2, "x2_have": have, "p10": 1 - x2, "label": 1 - x2})


class SplitForTuningTest(unittest.TestCase):
    def test_bounds_follow_production_and_leave_the_embargo_out(self):
        df = frame(900, 700)
        tune_rows, hold, b = X.split_for_tuning(df)
        last_full = df.loc[df["x2_have"], "Date"].max()
        train_end, test_start, dmax = train_model.holdout_bounds(df.loc[df["Date"] <= last_full, "Date"], 12, 125)
        self.assertEqual(b["last_full"], str(last_full.date()))
        self.assertEqual(b["train_end"], str(train_end.date()))
        self.assertEqual(b["test_start"], str(test_start.date()))
        self.assertEqual(dmax, last_full)
        self.assertEqual(b["embargo_calendar_days"], int(round(125 * train_model.TRADING_TO_CALENDAR)))
        self.assertLessEqual(tune_rows["Date"].max(), train_end)
        self.assertGreaterEqual(hold["Date"].min(), test_start)
        self.assertLessEqual(hold["Date"].max(), last_full)
        # 窓の揃った最後の日より後の行（ここでは200行）はホールドアウトに入れない
        self.assertEqual(b["after_last_full"], 200)
        self.assertEqual(len(tune_rows) + len(hold) + b["embargo_rows"] + b["after_last_full"], len(df))

    def test_rows_after_the_last_full_window_are_not_in_the_holdout_even_if_stopped(self):
        df = frame(900, 900, stopped_after=30)
        _, hold, b = X.split_for_tuning(df)
        self.assertEqual(b["after_last_full"], 30)
        self.assertEqual(hold["Date"].max(), df.loc[df["x2_have"], "Date"].max())

    def test_no_full_window_stops(self):
        df = frame(300, 0)
        with self.assertRaises(SystemExit):
            X.split_for_tuning(df)


class TuningFrameTest(unittest.TestCase):
    def test_label_is_replaced_by_the_two_times_label(self):
        df = frame(400, 400)
        t = X.tuning_frame(df)
        self.assertTrue((t["label"] == df["x2"]).all())
        self.assertFalse((t["label"] == df["p10"]).all())
        self.assertTrue((df["label"] == df["p10"]).all())          # 元の行は書き換えない

    def test_one_class_stops(self):
        df = frame(400, 400)
        df["x2"] = 0
        with self.assertRaises(SystemExit):
            X.tuning_frame(df)


class ReadingTest(unittest.TestCase):
    def test_fund_verdict_needs_every_pair_in_the_same_direction(self):
        self.assertTrue(X.fund_verdict([(30.0, 31.0), (32.0, 32.2)]).startswith("仮説どおり"))
        self.assertIn("決算・財務＋バリュエーションの割合が高い", X.fund_verdict([(30.0, 31.0)]))
        self.assertTrue(X.fund_verdict([(30.0, 29.0), (32.0, 31.8)]).startswith("仮説と逆"))
        self.assertEqual(X.fund_verdict([(30.0, 31.0), (32.0, 31.0)]), "はっきりしない")
        # 丸めると同じ値は「高い」に数えない
        self.assertEqual(X.fund_verdict([(30.0, 31.0), (32.01, 32.04)]), "はっきりしない")
        self.assertTrue(X.fund_verdict([]).startswith("比べられない"))

    def test_only_the_production_lgbm_and_the_tuned_tree_are_compared(self):
        self.assertEqual([a for a, _, _, _ in X.ARMS], ["A", "C"])
        self.assertEqual([a for a, _, _ in X.SCORE_ARMS], ["A", "C"])
        self.assertEqual(dict((a, (label, tree)) for a, label, tree, _ in X.ARMS), {"A": ("p10", "prod"), "C": ("x2", "tuned")})

    def test_vol_share_sums_the_five_columns(self):
        share = {c: 0.01 for c in X.VOL_COLS}
        share["tx_ma_gap_100"] = 0.5
        self.assertAlmostEqual(X.vol_share(share), 5.0)
        self.assertEqual(len(X.VOL_COLS), 5)


class PrecisionTest(unittest.TestCase):
    @staticmethod
    def frame():
        return pd.DataFrame({"Date": pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09",
                                                     "2025-01-10", "2025-01-14"]),
                             "Code": ["A1", "A1", "B2", "C3", "D4", "B2"],
                             "x2": [1, 1, 0, 1, 0, 0],
                             "peak_ratio": [2.5, 2.2, 1.1, 2.0, 1.3, 1.4],
                             "period": "hold"})

    SCORES = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.3])

    def test_top_picks_take_exactly_the_highest_scores(self):
        f = self.frame()
        self.assertEqual(list(X.top_picks(f, self.SCORES, 0.5).index), [0, 1, 3])
        self.assertEqual(len(X.top_picks(f, self.SCORES, 0.01)), 1)          # 少なくとも1件

    def test_first_cross_keeps_the_first_day_per_stock_within_each_period(self):
        f = self.frame()
        fc = X.first_cross(X.top_picks(f, self.SCORES, 1.0))
        self.assertEqual(sorted(fc["Code"]), ["A1", "B2", "C3", "D4"])
        self.assertEqual(fc.loc[fc["Code"] == "A1", "Date"].iloc[0], pd.Timestamp("2025-01-06"))
        self.assertEqual(fc.loc[fc["Code"] == "B2", "Date"].iloc[0], pd.Timestamp("2025-01-08"))
        both = pd.concat([f.assign(period="2024"), f.assign(period="2025")])
        self.assertEqual(len(X.first_cross(both)), 8)                         # 期間が違えば別に数える

    def test_pick_stats_and_wilson(self):
        st = X.pick_stats(self.frame().iloc[[0, 1, 3]])
        self.assertEqual((st["n"], st["hits"]), (3, 3))
        self.assertAlmostEqual(st["precision"], 1.0)
        self.assertAlmostEqual(st["median_gain"], 1.2)                        # 倍率 2.5・2.2・2.0 → 伸び 1.5・1.2・1.0
        lo, hi = X.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        lo, hi = X.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertTrue(np.isnan(X.wilson(0, 0)[0]))

    def test_precision_table_cuts_each_period_separately(self):
        f = self.frame()
        t = X.precision_table([(f.assign(period="2024"), self.SCORES), (f.assign(period="2025"), self.SCORES)], tops=(0.5,))
        self.assertEqual(t[0]["rows"]["n"], 6)                                # 期間ごとに3件
        self.assertEqual(t[0]["stocks"]["n"], 4)                              # 期間ごとに A1・C3
        self.assertEqual(t[0]["stocks"]["hits"], 4)

    def test_peak_ratio_rebuilds_the_two_times_label_exactly(self):
        ds, bars = synthetic_inputs()
        df, _ = E.build_frame(ds, bars, [X.TARGET], F.columns(F.DEFAULT_PRESET), log=lambda *a, **k: None)
        df["peak_ratio"] = X.peak_ratio(bars, df)
        self.assertEqual(X.ratio_mismatch(df), 0.0)
        self.assertGreater(int(df["x2"].sum()), 0)
        self.assertTrue((df.loc[df["x2"] == 1, "peak_ratio"] >= 2.0 - 1e-9).all())
        df.loc[df.index[0], "x2"] = 1 - int(df.loc[df.index[0], "x2"])
        self.assertGreater(X.ratio_mismatch(df), 0.0)


class OutcomeTest(unittest.TestCase):
    def test_end_is_the_close_on_the_last_day_of_the_window_and_stopped_names_use_their_last_close(self):
        closes = [100, 110, 150, 120, 130, 125, 140, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
        a = bars_frame("1111", closes, opens=[100.0] * len(closes))
        b = bars_frame("2222", [100, 120, 80, 90], opens=[100.0] * 4)        # 4日で売買が止まる（上場廃止の形）
        bars = pd.concat([a, b], ignore_index=True)
        rows = pd.DataFrame({"Code": ["1111", "1111", "2222", "1111"],
                             "Date": [a["Date"][0], a["Date"][5], b["Date"][1], a["Date"][18]]})
        out = X.forward_outcomes(bars, rows, horizon=3)
        # 初日: 買値は翌日の寄り 100、窓は2〜4日目の終値（110・150・120）→ 最高 1.5・窓の最後の日 120 → 1.2
        self.assertAlmostEqual(out["peak_ratio"][0], 1.5)
        self.assertAlmostEqual(out["end_ratio"][0], 1.2)
        # 6日目: 窓は 140・90・95 → 最高 1.4・最後の日 0.95（最高値から下がって終わる）
        self.assertAlmostEqual(out["peak_ratio"][1], 1.4)
        self.assertAlmostEqual(out["end_ratio"][1], 0.95)
        # 止まった銘柄の2日目: 窓は 80・90 しかない → 最高 0.9・最後の終値 0.9
        self.assertAlmostEqual(out["peak_ratio"][2], 0.9)
        self.assertAlmostEqual(out["end_ratio"][2], 0.9)
        # 先が足りず、売買も止まっていない → 半年後の終値は不明
        self.assertTrue(np.isnan(out["end_ratio"][3]))
        known = out["end_ratio"].notna()
        self.assertTrue((out.loc[known, "end_ratio"] <= out.loc[known, "peak_ratio"] + 1e-12).all())

    def test_outcome_stats_counts_levels_and_losses(self):
        rows = pd.DataFrame({"peak_ratio": [2.5, 1.6, 1.3, 1.1, 1.0], "end_ratio": [1.8, 1.2, 0.9, 1.3, 0.8]})
        st = X.outcome_stats(rows)
        self.assertEqual(st["n"], 5)
        self.assertEqual((st["peak_ge_30"], st["peak_ge_50"]), (3, 2))          # ちょうど +30% も数える
        self.assertEqual((st["end_ge_30"], st["end_ge_50"]), (2, 1))
        self.assertEqual(st["end_lt0"], 2)
        self.assertAlmostEqual(st["peak_median"], 0.3)
        self.assertAlmostEqual(st["end_median"], 0.2)
        self.assertAlmostEqual(st["end_mean"], 0.2)

    def test_pick_stats_splits_hits_and_misses(self):
        f = PrecisionTest.frame().assign(end_ratio=[1.9, 1.5, 0.9, 1.4, 1.2, 1.0])
        st = X.pick_stats(f)
        self.assertEqual((st["all"]["n"], st["miss"]["n"]), (6, 3))              # 2倍に届かなかったのは3行
        self.assertEqual(st["miss"]["end_lt0"], 1)
        self.assertEqual(st["miss"]["peak_ge_30"], 2)
        self.assertEqual(st["miss"]["end_ge_30"], 0)


class ParamsFromTest(unittest.TestCase):
    BOUNDS = {"train_end": "2024-11-01", "test_start": "2025-05-01", "last_full": "2026-05-01"}

    def write(self, tmp, **over):
        rec = {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 200,
               "_cv": {"scheme": "year_cap_date", "n_splits": 5, "n_trials": 50, "mean_pr_auc": 0.165},
               "_features_sig": "abc", "_label": {"key": "x2", "horizon": 125, "multiple": 2.0},
               "_bounds": dict(self.BOUNDS)}
        rec.update(over)
        path = os.path.join(tmp, "p.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({X.PARAMS_KEY: rec}, fh)
        return path

    def test_reads_params_only_when_columns_label_and_bounds_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            params, rec = X.load_params_from(self.write(tmp), "abc", self.BOUNDS)
            self.assertEqual(params, {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 200, "verbose": -1})
            self.assertEqual(rec["n_trials"], 50)
            with self.assertRaises(SystemExit):
                X.load_params_from(self.write(tmp), "xyz", self.BOUNDS)
            with self.assertRaises(SystemExit):
                X.load_params_from(self.write(tmp, _label={"key": "x2", "horizon": 250, "multiple": 3.0}), "abc", self.BOUNDS)
            with self.assertRaises(SystemExit):
                X.load_params_from(self.write(tmp), "abc", {**self.BOUNDS, "test_start": "2025-05-08"})
            with self.assertRaises(SystemExit):
                X.load_params_from(os.path.join(tmp, "none.json"), "abc", self.BOUNDS)

    def test_compare_prev_counts_identical_tables(self):
        t = {"holdout": {"C": [{"top": 0.01, "rows": {"n": 43, "hits": 13}, "stocks": {"n": 39, "hits": 13}}]}}
        self.assertEqual(X.compare_prev({"precision": t}, t), (1, 1))
        changed = json.loads(json.dumps(t))
        changed["holdout"]["C"][0]["stocks"]["hits"] = 12
        self.assertEqual(X.compare_prev({"precision": t}, changed), (0, 1))
        self.assertEqual(X.compare_prev(None, t), (0, 1))


class EndToEndTest(unittest.TestCase):
    PARAMS = {"objective": "binary", "boosting_type": "gbdt", "n_estimators": 15, "num_leaves": 7, "learning_rate": 0.1,
              "min_child_samples": 5, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
              "reg_alpha": 1e-3, "reg_lambda": 1e-3, "random_state": 0, "n_jobs": 1, "verbose": -1}

    def test_runs_through_with_the_production_split_and_keeps_production_params_untouched(self):
        ds, bars = synthetic_inputs()
        # 本番のデータの cap_band は時価総額の帯の番号（0〜4 の整数）。合成の乱数のままだと本番の層別（tuning._cap_bands）が読めない
        ds["cap_band"] = (np.arange(len(ds)) % 5).astype(float)
        seen = {}
        real_tune = tuning.tune

        def spy(df, cols, **kw):
            seen["label_rate"] = float(df["label"].mean())
            seen["x2_rate"] = float(df["x2"].mean())
            seen["p10_rate"] = float(df["p10"].mean())
            seen["kw"] = kw
            return real_tune(df, cols, **kw)

        with open(tuning.PARAMS_PATH, "rb") as fh:
            before = fh.read()
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            res, par = os.path.join(tmp, "res.json"), os.path.join(tmp, "par.json")
            with mock.patch.object(E, "load_inputs", return_value=(ds, bars)), \
                    mock.patch.object(E, "production_params", return_value=dict(self.PARAMS)), \
                    mock.patch.object(tuning, "SEARCH_N_ESTIMATORS", 8), \
                    mock.patch.dict(tuning.FIXED, {"n_jobs": 1}), \
                    mock.patch.object(tuning, "tune", side_effect=spy), \
                    mock.patch.object(tuning, "save_params", side_effect=AssertionError("本番のパラメータを書いた")), \
                    contextlib.redirect_stdout(out):
                rc = X.main(["--n-trials", "2", "--seeds", "0,1", "--ends", "2020-06-30,all",
                             "--out", res, "--params-out", par])
            with open(res, encoding="utf-8") as fh:
                saved = json.load(fh)
            with open(par, encoding="utf-8") as fh:
                params = json.load(fh)
            # 2回目: 探索はやり直さず、1回目のパラメータを読む。同じ銘柄が選ばれ、行方が出る
            out2 = io.StringIO()
            with mock.patch.object(E, "load_inputs", return_value=(ds, bars)), \
                    mock.patch.object(E, "production_params", return_value=dict(self.PARAMS)), \
                    mock.patch.object(tuning, "tune", side_effect=AssertionError("探索をやり直した")), \
                    contextlib.redirect_stdout(out2):
                rc2 = X.main(["--seeds", "0,1", "--ends", "2020-06-30,all", "--out", res, "--params-out", par,
                              "--params-from", par, "--prev", res])
            with open(res, encoding="utf-8") as fh:
                saved2 = json.load(fh)
        text2 = out2.getvalue()
        self.assertEqual(rc2, 0)
        self.assertIn("探索はやり直さず、前の回の探索したパラメータを使う", text2)
        self.assertIn("一致 20 / 20", text2)
        self.assertIn("行方（銘柄ごと。", text2)
        self.assertIn(" 外れ ", text2)
        self.assertIn("[check] 半年後の終値の倍率が無い行 0 / 最高値の倍率を超える行 0", text2)
        self.assertAlmostEqual(saved2["cv"]["arms"]["C"]["pr_auc"], saved["cv"]["arms"]["C"]["pr_auc"])
        for part in ("holdout", "oof"):
            for arm in ("A", "C"):
                for r1, r2 in zip(saved["precision"][part][arm], saved2["precision"][part][arm]):
                    self.assertEqual((r1["stocks"]["n"], r1["stocks"]["hits"]), (r2["stocks"]["n"], r2["stocks"]["hits"]))
                    st = r2["stocks"]
                    self.assertEqual(st["all"]["n"], st["n"])
                    self.assertEqual(st["miss"]["n"], st["n"] - st["hits"])
                    self.assertLessEqual(st["miss"]["peak_ge_50"], st["miss"]["peak_ge_30"])
                    self.assertLessEqual(st["miss"]["end_ge_30"], st["miss"]["peak_ge_30"])
        for code in ds["Code"].unique():
            self.assertNotIn(code, text2)
        with open(tuning.PARAMS_PATH, "rb") as fh:
            self.assertEqual(fh.read(), before)
        text = out.getvalue()
        self.assertEqual(rc, 0)

        # 探索は2倍のラベルを、本番と同じ分割・目的・試行数で
        self.assertAlmostEqual(seen["label_rate"], seen["x2_rate"])
        self.assertNotAlmostEqual(seen["label_rate"], seen["p10_rate"])
        self.assertEqual(seen["kw"]["scheme"], "year_cap_date")
        self.assertEqual(seen["kw"]["objective"], "pr_auc")
        self.assertEqual(seen["kw"]["n_splits"], 5)
        self.assertEqual(seen["kw"]["n_trials"], 2)
        rec = params[X.PARAMS_KEY]["_cv"]
        self.assertEqual(rec["scheme"], "year_cap_date")
        self.assertEqual(rec["n_trials"], 2)
        self.assertEqual(params[X.PARAMS_KEY]["n_estimators"], 8)
        self.assertEqual(params[X.PARAMS_KEY]["_label"], {"key": "x2", "horizon": 125, "multiple": 2.0})
        # 計算し直した CV の値が探索の記録と合う（同じ分割・同じ木）
        self.assertAlmostEqual(saved["cv"]["arms"]["C"]["pr_auc"], rec["mean_pr_auc"], places=4)

        # 境界: 探索の行はエンバーゴより前
        b = saved["bounds"]
        self.assertLess(b["train_end"], b["test_start"])
        self.assertEqual(b["embargo_calendar_days"], int(round(125 * train_model.TRADING_TO_CALENDAR)))

        # 表・読み方・参考
        self.assertIn("区分ごとの重要度の割合", text)
        self.assertIn("読み方（回す前に決めたもの。C と A）:", text)
        self.assertIn("1. 2倍用に探索した木では本番の lgbm より決算が効いてくるか（fund+val）: ", text)
        self.assertIn("2. 本番の lgbm よりテクニカルの重要度が下がるか（テクニカル）: ", text)
        self.assertNotIn("2倍（+10%の木）", text)
        self.assertIn("参考: 探索と同じ5分割", text)
        self.assertIn("参考: ホールドアウト", text)
        self.assertIn("[check] 半年以内の終値の最高値 ÷ 買値 から作り直した「2倍に届いたか」がラベルと合わない行 0.000%", text)
        self.assertIn("参考: しきい値を上げたときの的中率", text)
        self.assertIn("[ホールドアウト ", text)
        self.assertIn("[前進分割 ", text)
        self.assertEqual(len(saved["pairs"]["fund_val_CA"]), 4)          # 期間2つ × 種2つ
        self.assertEqual(sorted(saved["pairs"]), ["fund_val_CA", "tech_CA", "vol_CA"])
        self.assertEqual(sorted(saved["readings"]), ["fund", "tech"])
        self.assertEqual(sorted(saved["cv"]["arms"]), ["C"])
        self.assertEqual(len(saved["pairs"]["tech_CA"]), 4)
        self.assertEqual(sorted(saved["mean"]), ["A", "C"])
        for arm in ("A", "C"):
            self.assertAlmostEqual(sum(saved["mean"][arm]["category"].values()), 1.0, places=6)
        self.assertIsNotNone(saved["holdout"])
        self.assertEqual(sorted(saved["holdout"]["arms"]), ["A", "C"])
        prec = saved["precision"]
        self.assertEqual(prec["purge_days"], 193)
        for part in ("holdout", "oof"):
            self.assertEqual(sorted(prec[part]), ["A", "C"])
            for arm in ("A", "C"):
                rows = prec[part][arm]
                self.assertEqual([r["top"] for r in rows], list(X.PREC_TOPS))
                for r in rows:
                    self.assertLessEqual(r["stocks"]["n"], r["rows"]["n"])
                    self.assertLessEqual(r["rows"]["hits"], r["rows"]["n"])
                # 上位を絞るほど件数は減る（増えない）
                ns = [r["rows"]["n"] for r in rows]
                self.assertEqual(ns, sorted(ns, reverse=True))
        self.assertTrue(prec["oof_years"])
        self.assertTrue(all(y["year"] in X.OOF_YEARS for y in prec["oof_years"]))
        for code in ds["Code"].unique():
            self.assertNotIn(code, text)


if __name__ == "__main__":
    unittest.main()
