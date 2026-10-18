#!/usr/bin/env python3
"""
実験105（research/exp/e105_technical_ab.py）の部品。データには触らない（作りものの日足・表で確かめる）。

- 列: 比べる土台は回した時点の既定 249列（名前で固定）。足すのは research/technicals.py の111列で、土台と重ならない。
  系統ごとの参考の腕は5つ（列は 249 + その系統）
- 日足の突き合わせ: 学習データの行の順のまま111列を付け、日足に無い行は欠測。close_position が合わなければ止める
- 報告の一通り: 保存済みの探索（A・B）から、全部の腕の out-of-fold・寄与・報告が最後まで通る（種1つ・ずらし0 と 2）
- 1.2σ の回: --k 1.2 で保存の頭が e105_k12 になり、1.2 と 1.5 以外は受け付けない。1.2σ に引き直したラベルがデータセットの
  label と食い違えば（値・決まる決まらない）止める。1.5σ では確かめない

  python3 tests/test_e105_technical_ab.py
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

import build_dataset as B  # noqa: E402
import e105_technical_ab as E  # noqa: E402
import features as F  # noqa: E402
import technicals as TX  # noqa: E402


def bars(code, n, seed):
    rng = np.random.default_rng(seed)
    close = 500 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    o = close * np.exp(rng.normal(0, 0.01, n))
    return pd.DataFrame({"Code": code, "Date": pd.bdate_range("2023-01-02", periods=n), "open": o,
                         "high": np.maximum(o, close) * 1.01, "low": np.minimum(o, close) * 0.99, "close": close,
                         "vol": rng.integers(1000, 50000, n).astype(float), "trading_value": close * 1000 / 1e8})


class TestColumns(unittest.TestCase):
    def test_sets(self):
        base = F.columns(E.BASE_PRESET)
        self.assertEqual(len(base), 249)
        self.assertEqual(list(E.NEW_COLS), TX.COLUMNS)
        self.assertEqual(len(E.NEW_COLS), 111)
        self.assertFalse(set(base) & set(E.NEW_COLS))
        self.assertEqual(set(E.GROUP_ARMS.values()), set(TX.GROUPS))
        for d in (E.COLS_OF, E.PARAMS_OF, E.LABELS):
            self.assertEqual(set(d), set(E.ARMS))
        self.assertEqual(E.PRIMARY, ("A", "C", "P"))
        self.assertEqual({a for a in E.ARMS if E.PARAMS_OF[a] != "A"}, {"B"})
        self.assertEqual(E.K, 1.5)
        self.assertEqual(E.K_DEFAULT, 1.5)
        self.assertEqual(E.SEEDS, (42, 7, 123))


class TestPanel(unittest.TestCase):
    def setUp(self):
        self.p = pd.concat([bars("10010", 260, 0), bars("10020", 120, 1)], ignore_index=True)
        self.p["Date"] = self.p["Date"].astype("datetime64[ns]")

    def test_attach_keeps_order(self):
        keys = pd.DataFrame({"Code": ["10020", "10010", "99990", "10010"],
                             "Date": pd.to_datetime(["2023-06-15", "2023-12-29", "2023-12-29", "2023-02-01"])})
        got = E.attach_technicals(keys, self.p)
        self.assertEqual(list(got.columns), TX.COLUMNS)
        full = TX.compute(self.p)
        for i, (c, d) in enumerate(zip(keys["Code"], keys["Date"])):
            m = (self.p["Code"] == c) & (self.p["Date"] == d)
            if m.any():
                want = full[m.to_numpy()].iloc[0].to_numpy(dtype=float)
                have = got.iloc[i].to_numpy(dtype=float)
                self.assertTrue(np.array_equal(np.isnan(want), np.isnan(have)))
                self.assertTrue(np.allclose(want[~np.isnan(want)], have[~np.isnan(have)]))
            else:
                self.assertTrue(got.iloc[i].isna().all())                       # 日足に無い行は欠測

    def test_alignment_check(self):
        keys = self.p.iloc[[5, 100, 300]][["Code", "Date"]].reset_index(drop=True)
        rng = self.p["high"] - self.p["low"]
        cp = ((self.p["close"] - self.p["low"]) / rng * 100).iloc[[5, 100, 300]].to_numpy()
        with redirect_stdout(io.StringIO()):
            E.check_alignment(keys.assign(close_position=cp), self.p)
            with self.assertRaises(SystemExit):
                E.check_alignment(keys.assign(close_position=cp + 1), self.p)   # 値が違う
            with self.assertRaises(SystemExit):
                bad = keys.copy()
                bad.loc[0, "Date"] = pd.Timestamp("2030-01-01")                  # 日足に無い行
                E.check_alignment(bad.assign(close_position=cp), self.p)


class TestCompareColumns(unittest.TestCase):
    def test_counts(self):
        rng = np.random.default_rng(5)
        a = pd.DataFrame({c: rng.normal(size=20) for c in TX.COLUMNS})
        b = a.copy()
        self.assertEqual(sum(E.compare_columns(a, b).values()), 0)
        b.loc[0, "tx_rsi_14"] *= 1 + 1e-9                                       # 丸めの差は同じとみなす
        self.assertEqual(sum(E.compare_columns(a, b).values()), 0)
        b.loc[1, "tx_rsi_14"] += 0.01                                              # 値が違う
        b.loc[2, "tx_macd"] = np.nan                                               # 欠測の位置が違う
        got = E.compare_columns(a, b)
        self.assertEqual(got["tx_rsi_14"], 1)
        self.assertEqual(got["tx_macd"], 1)
        self.assertEqual(sum(got.values()), 2)


class TestLabelK12(unittest.TestCase):
    def frame(self):
        rng = np.random.default_rng(3)
        n = 400
        df = pd.DataFrame({"vol_20d": rng.uniform(1, 5, n), "future_rise": rng.normal(0.08, 0.1, n),
                           "end_level": rng.normal(0.05, 0.08, n), "uptrend_end": (rng.random(n) < 0.7).astype(float),
                           "label": 0.0})
        df["label"] = B.rise_label_for_k(df, 1.2)
        df.loc[[3, 7], "label"] = np.nan                                         # 未確定の行
        return df

    def test_same(self):
        df = self.frame()
        with redirect_stdout(io.StringIO()):
            out = E.check_label(df, 1.2)
        self.assertEqual(out["same"], out["both"])
        self.assertEqual(out["both"], 398)
        self.assertEqual(out["defined_mismatch"], 0)
        self.assertGreater(df["label"].sum(), 0)
        self.assertIsNone(E.check_label(df, 1.5))                                 # 1.5σ では確かめない

    def test_stops_on_mismatch(self):
        df = self.frame()
        i = int(np.flatnonzero(df["label"].to_numpy() == 0)[0])
        bad = df.copy()
        bad.loc[i, "label"] = 1.0                                                # 値が違う
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.check_label(bad, 1.2)
        bad = df.copy()
        bad.loc[10, "vol_20d"] = np.nan                                          # 引き直すと決まらない行
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.check_label(bad, 1.2)

    def test_k_option(self):
        old = (E.K, E.RUN, E.TAG, E.DEADLINE)
        try:
            with mock.patch.object(E, "prepare", side_effect=RuntimeError("stop")):
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    E.main(["--k", "1.2", "--arms", "A,C,P,B", "--tag", "e105t"])
            self.assertEqual(E.K, 1.2)
            self.assertEqual(E.RUN, "e105t_k12")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                E.main(["--k", "1.8"])                                           # 回す予定の無い値は受け付けない
        finally:
            E.K, E.RUN, E.TAG, E.DEADLINE = old


PARAMS = {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 20, "verbose": -1, "n_jobs": 1}


def make_frame(seed=0, per_day=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", "2022-12-30")
    n = len(dates) * per_day
    cols = {"Code": [f"{1000 + i % 50}" for i in range(n)], "Date": np.repeat(dates, per_day)}
    cols.update({f"b{j}": rng.normal(size=n) for j in range(4)})
    cols.update({c: rng.normal(size=n) for c in TX.COLUMNS})
    df = pd.DataFrame(cols)
    df["label"] = (df["b1"] + 0.5 * df["tx_rsi_14"] + rng.normal(0, 1.0, n) > 1.0).astype(float)
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
        colsets = {"base": base, "new": base + new, "perm": base + new,
                   **{a: base + list(TX.GROUPS[g]) for a, g in E.GROUP_ARMS.items()}}
        frames = {"base": df, "new": df, "perm": perm, **{a: df for a in E.GROUP_ARMS}}
        stamp = {"built_utc": "test", "rows": len(df), "date_max": "2022-12-30"}
        cutoff = pd.Timestamp("2021-12-31")
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = tmp, (42,), "e105test", None
            for arm in ("A", "B"):
                rec = {"params": PARAMS, "_cv": {"mean_pr_auc": 0.3, "std": 0.01, "mean_roc_auc": 0.6,
                                                 "fold_scores": [0.3, 0.31]},
                       "_n_trials": 2, "_cutoff": str(cutoff.date()), "_data": "test"}
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
        self.assertIn("pass", summary["judge7"])
        self.assertEqual(set(summary["top"]), set(E.ARMS))
        self.assertIn("pr_C-A", summary["windows"])
        self.assertIn("pr_G_volume-A", summary["windows"])
        self.assertEqual(set(extra["gain"]["share"]), {"既存の249列"} | set(TX.GROUP_JA.values()))
        self.assertAlmostEqual(sum(extra["gain"]["share"].values()), 1.0, places=6)
        self.assertEqual(len(extra["coverage"]["columns"]), 111)
        self.assertIn("§7", out)
        # C と B は同じパラメータ（作りもの）・同じ列なので同じ out-of-fold、P は違う
        self.assertTrue(np.allclose(res["prod"]["C"]["score"], res["prod"]["B"]["score"]))
        self.assertGreater(np.abs(res["prod"]["P"]["score"] - res["prod"]["C"]["score"]).max(), 1e-6)
        # 系統の腕はその系統の列だけを足す
        self.assertEqual(len(colsets["G_candle"]), 4 + 12)


if __name__ == "__main__":
    unittest.main()
