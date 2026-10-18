#!/usr/bin/env python3
"""
実験104（research/exp/e104_market_drift.py）の部品。データには触らない（小さな作りものの日足・表で確かめる）。

- ラベル: label_on は build_dataset.attach_rise_label そのもの（1.2σ は build() の作りと、1.5σ は rise_label_for_k と全行一致）
- 超過ラベル: TOPIX が一定（2 の累乗）なら今のラベルと全行一致、株価が TOPIX と同じ動きなら決まる行はすべて 0、
  TOPIX の最後の日より後の行は使わない。TOPIX の欠けた日・始値の欠測・topix との食い違いは止める
- 重み: 半減期で半分、窓ごとに平均1、入れ替えは同じ値の並べ替えで種が同じなら同じ。正例の重みは重みを掛けた負例 ÷ 正例
- out-of-fold: 重みなし・今のラベルなら E41.oof_folds（本番と同じ作り）と同じスコア。学習のラベルを替えても評価の
  label はそのまま。重みでスコアが変わる
- 窓の中の上位2%・報告の一通り（作りものの out-of-fold で最後まで通る）

  python3 tests/test_e104_market_drift.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import e104_market_drift as E  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import features as F  # noqa: E402
import tuning  # noqa: E402


def make_panel(n=90, seed=0, codes=("10010", "10020"), drift=0.006):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    parts = []
    for c in codes:
        close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))
        df = pd.DataFrame({"Code": c, "Date": dates, "close": close,
                           "open": close * np.exp(rng.normal(0, 0.01, n))})
        df["vol_20d"] = (df["close"] / df["close"].shift(1) - 1).rolling(20, min_periods=15).std() * 100
        parts.append(df)
    p = pd.concat(parts, ignore_index=True)
    p["Date"] = p["Date"].astype("datetime64[ns]")
    return p


def topix_of(dates, o=1024.0, c=1024.0):
    return pd.DataFrame({"Date": pd.to_datetime(pd.Series(dates)).astype("datetime64[ns]"), "O": o, "C": c})


def same(a, b) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))


class TestLabels(unittest.TestCase):
    def test_label_on_is_attach_rise_label(self):
        p = make_panel()
        full = B.attach_rise_label(p.copy(), replace(B.DEFAULT_RISE, vol_norm_k=B.VOL_NORM_K))
        y12 = E.label_on(p, B.VOL_NORM_K)
        self.assertTrue(same(y12["label"], full["label"]))
        self.assertTrue(set(full["label"].dropna().unique()) == {0.0, 1.0})      # 両方の印が出る（確かめに意味がある）
        # 1.5σ は rise_label_for_k（データセットから引き直す側）と全行一致
        self.assertTrue(same(E.label_on(p, 1.5)["label"], B.rise_label_for_k(full, 1.5)))
        self.assertEqual(list(y12.columns), ["Code", "Date", "label"])

    def test_flat_topix_keeps_the_label(self):
        p = make_panel()
        tp = topix_of(p["Date"].unique())
        rel = E.relative_panel(p, tp)
        self.assertTrue(same(E.label_on(rel, E.K)["label"], E.label_on(p, E.K)["label"]))
        self.assertTrue(same(rel["vol_20d"], p["vol_20d"]))                   # σ は銘柄自身のまま

    def test_stock_moving_with_market_is_never_positive(self):
        p = make_panel(codes=("10010",), drift=0.01)
        tp = pd.DataFrame({"Date": p["Date"], "O": p["open"] / 2, "C": p["close"] / 2})
        self.assertGreater(np.nansum(E.label_on(p, E.K)["label"]), 0)          # そのままなら正例がある
        y = E.label_on(E.relative_panel(p, tp), E.K)["label"]
        self.assertGreater(int(y.notna().sum()), 30)
        self.assertEqual(float(np.nansum(y)), 0.0)                             # 市場と同じ動きは超過では正例にならない

    def test_rows_after_topix_end_are_dropped(self):
        p = make_panel()
        dates = sorted(p["Date"].unique())
        tp = topix_of(dates[:-10])
        rel = E.relative_panel(p, tp)
        self.assertEqual(rel["Date"].max(), dates[-11])
        self.assertEqual(len(rel), len(p) - 20)
        full = E.label_on(p, E.K)
        cut = E.label_on(rel, E.K)
        m = full.merge(cut, on=["Code", "Date"], how="left", suffixes=("", "_cut"))
        # TOPIX の最後の日から20営業日より前の行は同じ、それより後は決まらない
        early = m["Date"] <= dates[-11 - B.RISE_HORIZON]
        self.assertTrue(same(m.loc[early, "label"], m.loc[early, "label_cut"]))
        self.assertTrue(m.loc[~early, "label_cut"].isna().all())

    def test_lagged(self):
        tp = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=6), "O": np.arange(6.0) + 1,
                           "C": np.arange(6.0) + 11})
        lg = E.lagged(tp, 2)
        self.assertTrue(lg[["O", "C"]].iloc[:2].isna().all().all())
        self.assertEqual(lg["O"].tolist()[2:], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(lg["C"].tolist()[2:], [11.0, 12.0, 13.0, 14.0])
        self.assertEqual(E.lagged(tp, 0)["O"].tolist(), tp["O"].tolist())

    def test_compare_labels(self):
        keys = pd.DataFrame({"Code": ["1", "1", "2"], "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"])})
        lbl = keys.assign(label=[1.0, np.nan, 0.0])
        c = E.compare_labels(keys, pd.Series([1.0, np.nan, 0.0]), lbl)
        self.assertEqual((c["same"], c["both"], c["nan_mismatch"]), (2, 2, 0))
        with redirect_stdout(io.StringIO()):
            E.check_or_stop("ok", c)
            with self.assertRaises(SystemExit):
                E.check_or_stop("x", E.compare_labels(keys, pd.Series([0.0, np.nan, 0.0]), lbl))
            with self.assertRaises(SystemExit):
                E.check_or_stop("x", E.compare_labels(keys, pd.Series([1.0, 1.0, 0.0]), lbl))


class TestTopix(unittest.TestCase):
    def indices(self, n=8):
        d = pd.bdate_range("2024-01-01", periods=n)
        t = pd.DataFrame({"Date": d, "Code": "0000", "O": 100.0 + np.arange(n), "C": 101.0 + np.arange(n)})
        other = t.assign(Code="0040", O=5.0, C=5.0)
        return pd.concat([other, t], ignore_index=True), pd.DataFrame({"Date": d, "topix": t["C"].to_numpy()})

    def test_series_and_checks(self):
        ix, ref = self.indices()
        tp, info = E.topix_series(ix, ref)
        self.assertEqual(list(tp.columns), ["Date", "O", "C"])
        self.assertEqual(len(tp), 8)
        self.assertEqual((info["topix_common_days"], info["topix_max_abs_diff"]), (8, 0.0))
        with self.assertRaises(SystemExit):
            E.topix_series(ix, ref.assign(topix=ref["topix"] + 1))           # topix の値と合わない
        bad = ix.copy()
        bad.loc[bad["Code"].eq("0000").to_numpy().nonzero()[0][3], "O"] = 0.0
        with self.assertRaises(SystemExit):
            E.topix_series(bad, ref)                                          # 始値が 0
        with self.assertRaises(SystemExit):
            E.topix_series(pd.concat([ix, ix.tail(1)], ignore_index=True), ref)    # 日付の重なり
        with self.assertRaises(SystemExit):
            E.topix_series(ix[ix["Code"] != "0000"], ref)                     # TOPIX が無い

    def test_missing_days(self):
        ix, _ = self.indices()
        tp, _ = E.topix_series(ix)
        days = list(tp["Date"])
        extra = pd.Timestamp("2024-01-06")                                    # 土曜（TOPIX に無い）
        panel_dates = pd.Series(days + [extra, days[-1] + pd.Timedelta(days=3)])
        self.assertEqual(E.missing_topix_days(panel_dates, tp, days[0]), [extra])
        self.assertEqual(E.missing_topix_days(panel_dates, tp, pd.Timestamp("2024-01-08")), [])

    def test_forward(self):
        ix, _ = self.indices()
        tp, _ = E.topix_series(ix)
        out = E.topix_forward(pd.Series([tp["Date"][0], tp["Date"][5], pd.Timestamp("2024-01-06")]), tp, horizon=2)
        self.assertAlmostEqual(out[0], (tp["C"][2] / tp["O"][1] - 1) * 100)
        self.assertAlmostEqual(out[1], (tp["C"][7] / tp["O"][6] - 1) * 100)
        self.assertTrue(np.isnan(out[2]))
        self.assertTrue(np.isnan(E.topix_forward(pd.Series([tp["Date"][6]]), tp, horizon=2)[0]))   # 先が足りない


class TestPanelFromBars(unittest.TestCase):
    def test_same_label_as_build(self):
        """日足ファイルから作る表とラベルが、build() と同じ作り（price_panel → mark_new_highs → attach_rise_label）と一致する。"""
        rng = np.random.default_rng(3)
        rows = []
        for c in ("10010", "10020", "10030"):
            n = 70
            close = 100 * np.exp(np.cumsum(rng.normal(0.006, 0.02, n)))
            o = close * np.exp(rng.normal(0, 0.01, n))
            rows.append(pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=n), "Code": c, "O": o,
                                      "H": np.maximum(o, close) * 1.01, "L": np.minimum(o, close) * 0.99, "C": close,
                                      "Vo": 1000.0}))
        bars = pd.concat(rows, ignore_index=True)
        for c in ("O", "H", "L", "C", "Vo"):
            bars[f"Adj{c}"] = bars[c]
        tmp = tempfile.mkdtemp()
        try:
            bars.to_parquet(os.path.join(tmp, "bars_2024.parquet"), index=False)
            with redirect_stdout(io.StringIO()):
                p = E.price_panel_for_labels(tmp, ["10010", "10030"])
        finally:
            shutil.rmtree(tmp)
        self.assertEqual(list(p.columns), E.LABEL_INPUTS)
        self.assertEqual(sorted(p["Code"].unique()), ["10010", "10030"])
        want = B.attach_rise_label(B.mark_new_highs(B.price_panel(bars)))
        want = want[want["Code"].astype(str).isin(["10010", "10030"])]
        got = E.label_on(p, B.VOL_NORM_K)
        self.assertTrue(same(got["label"], want["label"]))
        self.assertTrue(want["label"].notna().sum() > 50)


class TestWeights(unittest.TestCase):
    def test_decay(self):
        d = pd.to_datetime(["2020-01-01", "2022-01-01", "2024-01-01"])
        w = E.decay_weights(d, "2024-01-01", 2.0)
        self.assertAlmostEqual(w.mean(), 1.0)
        self.assertAlmostEqual(w[1] / w[2], 0.5 ** (730 / 365.25 / 2), places=12)
        self.assertAlmostEqual(w[0] / w[2], 0.5 ** (1461 / 365.25 / 2), places=12)
        self.assertTrue(w[0] < w[1] < w[2])
        with self.assertRaises(ValueError):
            E.decay_weights(pd.to_datetime(["2025-01-01"]), "2024-01-01", 2.0)

    def test_fold_weights(self):
        d = pd.Series(pd.bdate_range("2020-01-01", periods=300))
        self.assertIsNone(E.fold_weights(d, d.max(), None, 1))
        w = E.fold_weights(d, d.max(), ("decay", 2.0), 1)
        s1 = E.fold_weights(d, d.max(), ("shuffle", 2.0), 7)
        s2 = E.fold_weights(d, d.max(), ("shuffle", 2.0), 7)
        s3 = E.fold_weights(d, d.max(), ("shuffle", 2.0), 8)
        self.assertTrue(np.array_equal(np.sort(s1), np.sort(w)))              # 同じ値の並べ替え
        self.assertTrue(np.array_equal(s1, s2))                               # 種が同じなら同じ
        self.assertFalse(np.array_equal(s1, w))
        self.assertFalse(np.array_equal(s1, s3))
        with self.assertRaises(ValueError):
            E.fold_weights(d, d.max(), ("other", 2.0), 1)

    def test_spw(self):
        y = np.array([1, 0, 0, 0, 1, 0])
        self.assertEqual(E.weighted_spw(y, None), tuning.scale_pos_weight(y))
        w = np.array([2.0, 1.0, 1.0, 1.0, 0.5, 3.0])
        self.assertAlmostEqual(E.weighted_spw(y, w), 6.0 / 2.5)
        self.assertAlmostEqual(E.weighted_spw(y, np.ones(6)), tuning.scale_pos_weight(y))


PARAMS = {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 20, "verbose": -1, "n_jobs": 1}


def make_frame(seed=0, per_day=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", "2022-12-30")
    n = len(dates) * per_day
    df = pd.DataFrame({"Code": [f"{1000 + i % 50}" for i in range(n)], "Date": np.repeat(dates, per_day)})
    for j in range(4):
        df[f"f{j}"] = rng.normal(size=n)
    df["label"] = (df["f1"] + rng.normal(0, 1.0, n) > 1.0).astype(float)
    df["y_cur"] = df["label"]
    df["y_x"] = (df["f0"] + rng.normal(0, 0.3, n) > 0.8).astype(float)
    df["y_px"] = (df["f2"] + rng.normal(0, 1.0, n) > 1.0).astype(float)
    df["ret_o1_20"] = rng.normal(0.01, 0.1, n)
    df["ret_o1_40"] = rng.normal(0.01, 0.15, n)
    df["hit10"] = (rng.random(n) < 0.3).astype(float)
    df["tp10c"] = np.where(df["hit10"] > 0, 10.0, rng.normal(-1, 8, n))
    df["topix_fwd"] = rng.normal(0.5, 4, n)
    df["Date"] = df["Date"].astype("datetime64[ns]")
    return df


COLS = ["f0", "f1", "f2", "f3"]


class TestOOF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_frame()
        cls.folds = E41.folds_for(cls.df["Date"], 0)

    def test_same_as_production_oof(self):
        """重みなし・今のラベルなら、本番と同じ作りの E41.oof_folds と同じスコア。"""
        a = E.oof_arm(self.df, COLS, PARAMS, 42, self.folds)
        b = E41.oof_folds("lgbm", self.df, COLS, PARAMS, 42, self.folds)
        self.assertGreaterEqual(a["fold"].nunique(), 4)
        self.assertTrue(np.array_equal(a[["Code", "Date", "fold"]].to_numpy(), b[["Code", "Date", "fold"]].to_numpy()))
        self.assertTrue(np.allclose(a["score"].to_numpy(), b["score"].to_numpy(), rtol=0, atol=1e-12))
        for c in ("y_x", "hit10", "tp10c"):
            self.assertIn(c, a.columns)

    def test_train_label_and_eval_label(self):
        from sklearn.metrics import roc_auc_score

        x = E.oof_arm(self.df, COLS, PARAMS, 42, self.folds, train_col="y_x")
        a = E.oof_arm(self.df, COLS, PARAMS, 42, self.folds)
        self.assertTrue(np.array_equal(x["label"].to_numpy(), a["label"].to_numpy()))   # 評価のラベルはそのまま
        self.assertGreater(roc_auc_score(x["y_x"], x["score"]), 0.85)          # 超過のラベルで学んでいる
        self.assertLess(roc_auc_score(x["label"], x["score"]), 0.6)
        self.assertGreater(roc_auc_score(a["label"], a["score"]), 0.7)

    def test_weights_change_scores(self):
        a = E.oof_arm(self.df, COLS, PARAMS, 42, self.folds)
        w = E.oof_arm(self.df, COLS, PARAMS, 42, self.folds, weight=("decay", 2.0))
        p1 = E.oof_arm(self.df, COLS, PARAMS, 42, self.folds, weight=("shuffle", 2.0), perm_base=5)
        p2 = E.oof_arm(self.df, COLS, PARAMS, 42, self.folds, weight=("shuffle", 2.0), perm_base=5)
        p3 = E.oof_arm(self.df, COLS, PARAMS, 42, self.folds, weight=("shuffle", 2.0), perm_base=6)
        self.assertGreater(np.abs(w["score"] - a["score"]).max(), 1e-6)
        self.assertTrue(np.array_equal(p1["score"].to_numpy(), p2["score"].to_numpy()))
        self.assertGreater(np.abs(p1["score"] - p3["score"]).max(), 1e-6)

    def test_top_by_window(self):
        o = pd.DataFrame({"fold": [1] * 150 + [2] * 50, "score": np.r_[np.arange(150.0), np.arange(50.0)],
                          "label": np.r_[np.zeros(147), [1.0, 0.0, 1.0], np.ones(50)],
                          "hit10": np.r_[np.zeros(147), [1.0, 1.0, 0.0], np.zeros(50)],
                          "tp10c": np.r_[np.zeros(147), [10.0, 10.0, -4.0], np.zeros(50)]})
        t = E.top_by_window(o)
        self.assertEqual(t["fold"].tolist(), [1])                              # 100行未満の窓は落とす
        r = t.iloc[0]
        self.assertEqual((int(r["n"]), int(r["k"])), (150, 3))                # 2% を切り上げて3件
        self.assertAlmostEqual(r["prec"], 2 / 3)
        self.assertAlmostEqual(r["hit10"], 2 / 3)
        self.assertAlmostEqual(r["tp10c"], 16 / 3)


class TestRunAndReport(unittest.TestCase):
    def test_end_to_end(self):
        """保存済みの探索（A・XB）から、全部の腕の out-of-fold と報告が最後まで通る（種1つ・ずらし0 と 2）。"""
        df = make_frame(seed=1)
        df.attrs["built_utc"] = "test"
        stamp = {"built_utc": "test", "rows": len(df), "date_max": "2022-12-30"}
        cutoff = pd.Timestamp("2021-12-31")
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = tmp, (42,), "e104test", None
            for arm in ("A", "XB"):
                rec = {"params": PARAMS, "_cv": {"mean_pr_auc": 0.3, "std": 0.01, "mean_roc_auc": 0.6,
                                                 "fold_scores": [0.3, 0.31]},
                       "_n_trials": 2, "_cutoff": str(cutoff.date()), "_data": "test"}
                with open(E.path(f"params_{arm}.json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                lab_out = E.label_report(df)
                res = E.run(list(E.ARMS), df, COLS, cutoff, 2, stamp, [0, 2])
                summary = E.report(res, [0, 2], stamp, lab_out)
            out = buf.getvalue()
        finally:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = old
            shutil.rmtree(tmp)
        self.assertEqual(E.complete_arms(res, [0, 2]), list(E.ARMS))
        self.assertEqual(set(summary["judge7"]), {"A", "B"})
        self.assertEqual(summary["judge7"]["A"]["arm"], "X")
        self.assertEqual(summary["judge7"]["B"]["placebo"], "PW")
        self.assertEqual(set(summary["top"]), set(E.ARMS))
        self.assertIn("pr_X-A", summary["windows"])
        self.assertIn("pr_X-A", summary["windows_excess_label"])
        self.assertEqual(len(summary["by_year"]), df["Date"].dt.year.nunique() - 3)   # OOF は4年目から
        self.assertEqual(len(lab_out["by_year"]), df["Date"].dt.year.nunique() + 1)   # 年ごと + 全体
        self.assertIn("部A 目的変数を TOPIX 超過に", out)
        self.assertIn("部B 古い年の重みを下げる", out)
        # X と XB は同じパラメータ（作りもの）でも、学習のラベルが同じなので同じ out-of-fold になる
        self.assertTrue(np.allclose(res["prod"]["X"]["score"], res["prod"]["XB"]["score"]))
        # 重みの腕は A と違う
        self.assertGreater(np.abs(res["prod"]["W2"]["score"] - res["prod"]["A"]["score"]).max(), 1e-6)


class TestConstants(unittest.TestCase):
    def test_arms(self):
        self.assertEqual(E.K, 1.5)
        self.assertEqual(E.PLACEBO_LAG, 250)
        self.assertEqual(E.HALF_LIVES, {"W1": 1.0, "W2": 2.0, "W4": 4.0})
        self.assertEqual(set(E.PRIMARY) | set(E.REFERENCE), set(E.ARMS))
        self.assertFalse(set(E.PRIMARY) & set(E.REFERENCE))
        for d in (E.TRAIN_LABEL, E.PARAMS_OF, E.WEIGHT_OF, E.LABELS):
            self.assertEqual(set(d), set(E.ARMS))
        self.assertEqual(E.JUDGES["A"][:2], ("X", "PX"))
        self.assertEqual(E.JUDGES["B"][:2], ("W2", "PW"))
        self.assertEqual(E.WEIGHT_OF["PW"], ("shuffle", 2.0))
        self.assertEqual({a for a in E.ARMS if E.PARAMS_OF[a] != "A"}, {"XB"})
        self.assertEqual(E.SEEDS, (42, 7, 123))

    def test_columns(self):
        cols = F.columns(E.BASE_PRESET)                 # 回した時点の既定（249列）
        self.assertEqual(len(cols), 249)
        self.assertFalse(set(cols) & set(E.KEEP))
        self.assertFalse(set(cols) & set(B.FUTURE_COLS))


if __name__ == "__main__":
    unittest.main()
