#!/usr/bin/env python3
"""
実験132（research/exp/e132_minute_vp_ab.py）の部品。データには触らない（作りものの表で確かめる）。

- 列: 比べる土台は回した時点の既定 373列（名前で固定）。表には research/minute_features.py の33列を付け、学習に足すのは
  VWAP の3列を除いた30列（運用者の選択「VWAPの3列を外した30列」）。土台と重ならない
- 腕は A・B だけ（運用者の選択「AとBだけ」）。A は必ず含める
- ラベル: データセットの label が本番の +10% の定義（label_specs）と全行一致しなければ止める
- 付けた33列の確かめ: 割合が 0〜1 の外・無限・終値との差が −100% 以下・期間の中で欠けのある行・時間帯の割合の合計が1でない行があれば止める
- 評価の行: 全行と、分足の付いた行だけ（30列のどれかがある行）。窓は行が100件以上で正例・負例がそろうもの
- 表に付いている VWAP の3列は、B の学習・寄与・付き方の報告に入らない
- 報告の一通り: 保存済みの探索（A・B）から、out-of-fold・寄与・報告が最後まで通る（種1つ・ずらし0 と 2）。§7 の判定は出さない

  python3 -m unittest tests.test_e132_minute_vp_ab
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

import e132_minute_vp_ab as E  # noqa: E402
import features as F  # noqa: E402
import label_specs as LS  # noqa: E402
import minute_features as MF  # noqa: E402
import volume_profile as VP  # noqa: E402


class TestColumns(unittest.TestCase):
    def test_sets(self):
        base = F.columns(E.BASE_PRESET)
        self.assertEqual(len(base), 373)
        self.assertEqual(E.BASE_PRESET, F.DEFAULT_PRESET)              # 回す時点の既定（変わったら見直す）
        self.assertEqual(list(E.BUILT_COLS), MF.COLUMNS)
        self.assertEqual(E.DROPPED_COLS, ("vp_minute_vwap_gap_1d", "vp_minute_vwap_gap_5d", "vp_minute_vwap_gap_20d"))
        self.assertEqual(list(E.NEW_COLS), [c for c in MF.COLUMNS if "_vwap_gap_" not in c])   # 並びは minute_features のまま
        self.assertEqual(len(E.NEW_COLS), 30)
        self.assertFalse(set(base) & set(E.BUILT_COLS))
        self.assertEqual(E.ARMS, ("A", "B"))
        for d in (E.COLS_OF, E.LABELS):
            self.assertEqual(set(d), set(E.ARMS))
        self.assertEqual(E.SEEDS, (42, 7, 123))
        self.assertEqual(E.ROWSETS, ("all", "minute"))

    def test_args(self):
        old = (E.RUN, E.TAG, E.DEADLINE)
        try:
            with mock.patch.object(E, "prepare", side_effect=RuntimeError("stop")):
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    E.main(["--tag", "e132t"])
            self.assertEqual(E.RUN, "e132t")
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.main(["--arms", "B"])                                  # 基準 A は外せない
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.main(["--arms", "A,C"])                                # 腕は A・B だけ
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
    def test_same_and_stops(self):
        df = label_frame()
        with redirect_stdout(io.StringIO()):
            out = E.check_label(df)
        self.assertEqual(out["same"], out["rows"])
        bad = df.copy()
        bad.loc[int(np.flatnonzero(df["label"].to_numpy() == 0)[0]), "label"] = 1.0
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.check_label(bad)


def minute_frame(n=60, start_attached=10):
    """作りものの33列。最初の start_attached 行は分足の前（欠測）。"""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"Date": pd.bdate_range("2024-10-15", periods=n)})
    for w in MF.WINDOWS:
        for m in MF.PROFILE_METRICS:
            df[MF.column(m, w)] = rng.uniform(0, 0.9, n) if m.endswith("share") else rng.normal(0, 0.05, n)
        parts = rng.dirichlet(np.ones(4), n)
        for j, b in enumerate(VP.BUCKETS):
            df[MF.column(f"{b}_share", w)] = parts[:, j]
    df.loc[:start_attached - 1, list(MF.COLUMNS)] = np.nan
    return df


REPORT = {"minute_from": "2024-10-29", "minute_to": "2025-01-06", "minute_days": 50, "status": {"1d/ok": 50}}


class TestMinuteChecks(unittest.TestCase):
    def test_ok_and_stops(self):
        df = minute_frame()
        with redirect_stdout(io.StringIO()):
            out = E.minute_checks(df, REPORT)
        self.assertEqual(out["bad"], {})
        self.assertEqual(sum(out["torn"].values()), 0)
        self.assertAlmostEqual(out["attached_share_since_minute"], 1.0)
        for col, val in ((MF.column("poc_share", 5), 1.2), (MF.column("vwap_gap", 1), -1.0),
                         (MF.column("close_share", 20), np.inf)):
            bad = df.copy()
            bad.loc[20, col] = val
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.minute_checks(bad, REPORT)
        torn = df.copy()
        torn.loc[20, MF.column("poc_gap", 5)] = np.nan                   # 期間の中で1指標だけ欠ける
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.minute_checks(torn, REPORT)
        off = df.copy()
        off.loc[20, MF.column("open_share", 1)] += 0.1                   # 時間帯の割合の合計が1でない
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.minute_checks(off, REPORT)


class TestRows(unittest.TestCase):
    def test_rows_and_windows(self):
        rng = np.random.default_rng(2)
        n = 600
        o = pd.DataFrame({"Date": np.repeat(pd.bdate_range("2024-01-01", periods=n // 3), 3),
                          "fold": np.repeat([0, 1, 2], n // 3), "label": (rng.random(n) < 0.3).astype(float),
                          "score": rng.random(n), "minute": [True] * 250 + [np.nan] * 50 + [False] * 300})
        self.assertEqual(len(E.rows_of(o, "all")), n)
        self.assertEqual(len(E.rows_of(o, "minute")), 250)              # 欠測は付いていない側
        w = E.windows_on(o, "minute")
        self.assertEqual(w["fold"].tolist(), [0])
        self.assertEqual(len(E.windows_on(o, "all")), 3)


PARAMS = {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 20, "verbose": -1, "n_jobs": 1}


def make_frame(seed=0, per_day=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", "2022-12-30")
    n = len(dates) * per_day
    cols = {"Code": [f"{1000 + i % 50}" for i in range(n)], "Date": np.repeat(dates, per_day)}
    cols.update({f"b{j}": rng.normal(size=n) for j in range(4)})
    cols.update({c: rng.normal(size=n) for c in E.BUILT_COLS})          # 表には33列が付いている
    df = pd.DataFrame(cols)
    before = df["Date"] < "2019-01-01"
    df.loc[before, list(E.BUILT_COLS)] = np.nan                         # 分足は途中の年から
    df["label"] = (df["b1"] + 0.5 * df[MF.column("poc_gap", 20)].fillna(0)
                   + 0.8 * df[MF.column("vwap_gap", 20)].fillna(0) + rng.normal(0, 1.0, n) > 1.0).astype(float)
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
        colsets = {"base": base, "new": base + new}
        frames = {"base": df, "new": df}
        stamp = {"built_utc": "test", "rows": len(df), "date_max": "2022-12-30"}
        cutoff = pd.Timestamp("2021-12-31")
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = tmp, (42,), "e132test", None
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
            names = sorted(os.listdir(tmp))
        finally:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = old
            shutil.rmtree(tmp)
        self.assertEqual(E.complete_arms(res, [0, 2]), list(E.ARMS))
        self.assertEqual(set(summary["top"]), set(E.ARMS))
        self.assertIn("all_pr_B-A", summary["windows"])
        self.assertIn("minute_pr_B-A", summary["windows"])
        self.assertIn("minute_B", summary["production_oof"])
        self.assertNotIn("judge7", summary)
        self.assertEqual(set(extra["gain"]["share"]), {"既存の373列", "分足の30列"})
        self.assertAlmostEqual(sum(extra["gain"]["share"].values()), 1.0, places=6)
        self.assertEqual(set(extra["gain"]["new"]), set(E.NEW_COLS))     # 表にある VWAP の3列は学習に入らない
        self.assertEqual(set(extra["coverage"]["columns"]), set(E.NEW_COLS))
        self.assertNotIn("vp_minute_vwap_gap_20d↔", out)
        self.assertIn(f"lgbm_cnew{F.signature(base + new)}_", "\n".join(names))     # out-of-fold の名前に列の指紋
        self.assertIn("§7 の判定は出さない", out)
        self.assertIn("分足の付いた行だけ", out)
        # 付いた行だけの窓は、分足が始まる前の窓を含まない
        windows = pd.DataFrame(summary["per_window"]["minute"])
        self.assertTrue((pd.to_datetime(windows["start"]) >= pd.Timestamp("2019-01-01")).all())
        # A と B は列が違うので out-of-fold も違う
        self.assertGreater(np.abs(res["prod"]["A"]["score"].to_numpy() - res["prod"]["B"]["score"].to_numpy()).max(), 1e-9)

    def test_tune_cache_needs_same_columns(self):
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.RUN, E.DEADLINE = tmp, "e132cache", None
            cutoff = pd.Timestamp("2021-12-31")
            rec = {"params": PARAMS, "_cv": {}, "_n_trials": 2, "_cutoff": str(cutoff.date()), "_data": "test",
                   "_sig": F.signature(["b0"])}
            with open(E.path("params_A.json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            got = E.tune("A", make_frame(2), ["b0"], cutoff, 2, {"built_utc": "test"})
            self.assertEqual(got["params"], PARAMS)
            with mock.patch.object(E.tuning, "tune", side_effect=RuntimeError("searched")):
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    E.tune("A", make_frame(2), ["b0", "b1"], cutoff, 2, {"built_utc": "test"})
        finally:
            E.OOF_DIR, E.RUN, E.DEADLINE = old
            shutil.rmtree(tmp)

    def test_prepare_needs_the_minute_store(self):
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.MINUTE_DIR, E.RUN)
        try:
            E.OOF_DIR, E.MINUTE_DIR, E.RUN = tmp, os.path.join(tmp, "minute"), "e132prep"
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                E.prepare(["b0"])
            self.assertIn("分足の取得記録が無い", str(ctx.exception))
        finally:
            E.OOF_DIR, E.MINUTE_DIR, E.RUN = old
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
