#!/usr/bin/env python3
"""
実験130（research/exp/e130_jsf_v2_ab.py）の部品。データには触らない（作りものの表で確かめる）。

- 列: 比べる土台は回した時点の既定 373列（名前で固定）。足すのは research/jsf_features_v2.py の20列で、土台と重ならない
- ラベル: データセットの label が本番の +10% の定義（label_specs）と全行一致しなければ止める
- 付けた20列の確かめ: 範囲の外・無限があれば止める
- 評価の行: 全行と、日証金の付いた行だけ（jsf が True の行。欠測は付いていない側）。窓は行が100件以上で正例・負例がそろうもの
- 写しの対応の引数は必須（none で使わない）。保存の頭に入る
- 報告の一通り: 保存済みの探索（A・B）から、4つの腕の out-of-fold・寄与・報告が最後まで通る（種1つ・ずらし0 と 2）

  python3 -m unittest tests.test_e130_jsf_v2_ab
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e130_jsf_v2_ab as E  # noqa: E402
import features as F  # noqa: E402
import jsf_features_v2 as V2  # noqa: E402
import label_specs as LS  # noqa: E402


class TestColumns(unittest.TestCase):
    def test_sets(self):
        base = F.columns(E.BASE_PRESET)
        self.assertEqual(len(base), 373)
        self.assertEqual(E.BASE_PRESET, F.DEFAULT_PRESET)              # 回す時点の既定（変わったら見直す）
        self.assertEqual(list(E.NEW_COLS), V2.columns("all"))
        self.assertEqual(len(E.NEW_COLS), 20)
        self.assertFalse(set(base) & set(E.NEW_COLS))
        self.assertNotIn("jsf2_lag", E.NEW_COLS)                       # 確認用の列は学習に使わない
        for d in (E.COLS_OF, E.PARAMS_OF, E.LABELS):
            self.assertEqual(set(d), set(E.ARMS))
        self.assertEqual(E.PRIMARY, ("A", "C", "P"))
        self.assertEqual({a for a in E.ARMS if E.PARAMS_OF[a] != "A"}, {"B"})
        self.assertEqual(E.SEEDS, (42, 7, 123))
        self.assertEqual(E.ROWSETS, ("all", "jsf"))

    def test_shift_args(self):
        self.assertIsNone(E.parse_shift("none"))
        self.assertEqual(E.parse_shift("1"), 1)
        self.assertEqual(E.parse_shift("-1"), -1)
        self.assertEqual(E.shift_tag(None), "rsN")
        self.assertEqual(E.shift_tag(1), "rsp1")
        self.assertEqual(E.shift_tag(-1), "rsm1")
        self.assertEqual(E.shift_tag(0), "rsp0")
        old = (E.RUN, E.TAG, E.DEADLINE)
        try:
            with mock.patch.object(E, "prepare", side_effect=RuntimeError("stop")):
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    E.main(["--restrict-shift", "1", "--arms", "A,C,P,B", "--tag", "e130t"])
            self.assertEqual(E.RUN, "e130t_rsp1")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                E.main(["--arms", "A"])                                  # 写しの対応は必須
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.main(["--restrict-shift", "none", "--arms", "C,P"])   # 基準 A は外せない
        finally:
            E.RUN, E.TAG, E.DEADLINE = old


def label_frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"future_rise": rng.normal(0.08, 0.1, n), "end_level": rng.normal(0.05, 0.08, n),
                       "uptrend_end": (rng.random(n) < 0.7).astype(float), "entry_price": 100.0})
    df["label"] = LS.labels(df, "lgbm")
    df.loc[[2, 9], ["future_rise", "label"]] = np.nan                  # 未確定の行
    return df


class TestLabel(unittest.TestCase):
    def test_same(self):
        df = label_frame()
        with redirect_stdout(io.StringIO()):
            out = E.check_label(df)
        self.assertEqual(out["same"], out["rows"])
        self.assertEqual(out["defined"], 298)
        self.assertEqual(out["spec"], LS.for_model("lgbm"))

    def test_stops_on_mismatch(self):
        df = label_frame()
        i = int(np.flatnonzero(df["label"].to_numpy() == 0)[0])
        bad = df.copy()
        bad.loc[i, "label"] = 1.0
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.check_label(bad)
        bad = df.copy()
        bad.loc[2, "label"] = 0.0                                       # 決まらない行に値がある
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.check_label(bad)


def jsf_frame(n=50):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=n)})
    for c in V2.columns("all"):
        df[c] = rng.uniform(0, 1, n)
    df["jsf2_fee_days20"] = rng.integers(0, 21, n).astype(float)
    df["jsf2_restrict"] = rng.integers(0, 4, n).astype(float)
    df["jsf2_lendable"] = rng.integers(0, 2, n).astype(float)
    df["jsf2_lag"] = rng.integers(1, 5, n).astype(float)
    df.loc[:4, "jsf2_lag"] = np.nan
    return df


class TestJsfChecks(unittest.TestCase):
    def test_ok_and_stops(self):
        df = jsf_frame()
        with redirect_stdout(io.StringIO()):
            out = E.jsf_checks(df)
        self.assertEqual(out["attached"], 45)
        self.assertEqual(out["bad"], {})
        for col, val in (("jsf2_loan_pos60", 1.5), ("jsf2_fee", -0.1), ("jsf2_ratio", np.inf), ("jsf2_lag", 9.0)):
            bad = df.copy()
            bad.loc[10, col] = val
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.jsf_checks(bad)


class TestRows(unittest.TestCase):
    def test_rows_and_windows(self):
        rng = np.random.default_rng(2)
        n = 600
        o = pd.DataFrame({"Date": np.repeat(pd.bdate_range("2024-01-01", periods=n // 3), 3),
                          "fold": np.repeat([0, 1, 2], n // 3), "label": (rng.random(n) < 0.3).astype(float),
                          "score": rng.random(n), "jsf": [True] * 250 + [np.nan] * 50 + [False] * 300})
        self.assertEqual(len(E.rows_of(o, "all")), n)
        self.assertEqual(len(E.rows_of(o, "jsf")), 250)                 # 欠測は付いていない側
        w = E.windows_on(o, "jsf")
        self.assertEqual(w["fold"].tolist(), [0])                      # 窓1は付いた行が50件、窓2は0件
        self.assertEqual(int(w["n"].iloc[0]), 200)
        self.assertEqual(len(E.windows_on(o, "all")), 3)
        o2 = o.copy()
        o2.loc[o2["fold"] == 0, "label"] = 0.0                          # 正例の無い窓は落とす
        self.assertEqual(E.windows_on(o2, "jsf")["fold"].tolist(), [])


PARAMS = {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 20, "verbose": -1, "n_jobs": 1}


def make_frame(seed=0, per_day=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", "2022-12-30")
    n = len(dates) * per_day
    cols = {"Code": [f"{1000 + i % 50}" for i in range(n)], "Date": np.repeat(dates, per_day)}
    cols.update({f"b{j}": rng.normal(size=n) for j in range(4)})
    cols.update({c: rng.normal(size=n) for c in E.NEW_COLS})
    df = pd.DataFrame(cols)
    df["jsf2_lag"] = np.where(df["Date"] >= "2019-01-01", 1.0, np.nan)
    for c in E.NEW_COLS:
        df.loc[df["jsf2_lag"].isna(), c] = np.nan                       # 日証金は途中の年から
    df["label"] = (df["b1"] + 0.5 * df["jsf2_loan_chg20_v"].fillna(0) + rng.normal(0, 1.0, n) > 1.0).astype(float)
    df["ret_o1_20"] = rng.normal(0.01, 0.1, n)
    df["ret_o1_40"] = rng.normal(0.01, 0.15, n)
    df["hit10"] = (rng.random(n) < 0.3).astype(float)
    df["tp10c"] = np.where(df["hit10"] > 0, 10.0, rng.normal(-1, 8, n))
    df["Date"] = df["Date"].astype("datetime64[ns]")
    return df


class TestRunAndReport(unittest.TestCase):
    def test_end_to_end(self):
        df = make_frame(1)
        df.attrs["built_utc"] = "test"
        base = ["b0", "b1", "b2", "b3"]
        new = list(E.NEW_COLS)
        perm = E.permuted(df, new, seed=E.PERM_SEED)
        perm.attrs["built_utc"] = "test"
        colsets = {"base": base, "new": base + new, "perm": base + new}
        frames = {"base": df, "new": df, "perm": perm}
        stamp = {"built_utc": "test", "rows": len(df), "date_max": "2022-12-30"}
        cutoff = pd.Timestamp("2021-12-31")
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = tmp, (42,), "e130test", None
            for arm, cols in (("A", base), ("B", base + new)):
                rec = {"params": PARAMS, "_cv": {"mean_pr_auc": 0.3, "std": 0.01, "mean_roc_auc": 0.6,
                                                 "fold_scores": [0.3, 0.31]},
                       "_n_trials": 2, "_cutoff": str(cutoff.date()), "_data": "test", "_sig": F.signature(cols)}
                with open(E.path(f"params_{arm}.json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                extra = {"coverage": E.coverage_report(df, base)}
                res = E.run(list(E.ARMS), frames, colsets, cutoff, 2, stamp, [0, 2])
                extra["gain"] = E.gain_report(df, base, PARAMS, cutoff)
                summary = E.report(res, [0, 2], stamp, extra)
            out = buf.getvalue()
        finally:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = old
            shutil.rmtree(tmp)
        self.assertEqual(E.complete_arms(res, [0, 2]), list(E.ARMS))
        self.assertEqual(set(summary["judge7"]), {"all", "jsf"})
        self.assertIn("pass", summary["judge7"]["all"])
        self.assertEqual(set(summary["top"]), set(E.ARMS))
        self.assertIn("all_pr_C-A", summary["windows"])
        self.assertIn("jsf_pr_C-A", summary["windows"])
        self.assertIn("jsf_C", summary["production_oof"])
        self.assertEqual(set(extra["gain"]["share"]), {"既存の373列", "日証金の20列"})
        self.assertAlmostEqual(sum(extra["gain"]["share"].values()), 1.0, places=6)
        self.assertEqual(len(extra["coverage"]["columns"]), 20)
        self.assertIn("§7", out)
        self.assertIn("日証金の付いた行だけ", out)
        # 付いた行だけの窓は、日証金が始まる前の窓を含まない
        jsf_windows = pd.DataFrame(summary["per_window"]["jsf"])
        self.assertTrue((pd.to_datetime(jsf_windows["start"]) >= pd.Timestamp("2019-01-01")).all())
        # C と B は同じパラメータ（作りもの）・同じ列なので同じ out-of-fold、P は違う
        self.assertTrue(np.allclose(res["prod"]["C"]["score"], res["prod"]["B"]["score"]))
        self.assertGreater(np.abs(res["prod"]["P"]["score"] - res["prod"]["C"]["score"]).max(), 1e-6)
        # 対照 P は20列だけを入れ替え、jsf2_lag（付いたかどうか）はそのまま
        self.assertTrue(perm["jsf2_lag"].equals(df["jsf2_lag"]))

    def test_tune_cache_needs_same_columns(self):
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.RUN, E.DEADLINE = tmp, "e130cache", None
            cutoff = pd.Timestamp("2021-12-31")
            rec = {"params": PARAMS, "_cv": {}, "_n_trials": 2, "_cutoff": str(cutoff.date()), "_data": "test",
                   "_sig": F.signature(["b0"])}
            with open(E.path("params_A.json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            got = E.tune("A", make_frame(2), ["b0"], cutoff, 2, {"built_utc": "test"})
            self.assertEqual(got["params"], PARAMS)                     # 同じ列なら保存済みを使う
            with mock.patch.object(E.tuning, "tune", side_effect=RuntimeError("searched")):
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    E.tune("A", make_frame(2), ["b0", "b1"], cutoff, 2, {"built_utc": "test"})   # 列が違えば探し直す
        finally:
            E.OOF_DIR, E.RUN, E.DEADLINE = old
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
