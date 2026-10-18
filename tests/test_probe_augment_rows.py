#!/usr/bin/env python3
"""
データ拡張の案0の probe（research/exp/probe_augment_rows.py）。データには触らない（小さな合成の日足で確かめる）。

- 更新日すべてを残す差し替え: 中では新規のブレイクの印が更新日の印と同じになり、本番の印（直前20営業日に更新が無い日）・
  銘柄の中の行の位置・日足の日を控える。抜けたら mark_new_highs と SWEEP_OVERRIDES が元に戻る
- base_length（1つ前の更新日からの営業日数）で新規かどうかが分かる（21以上か欠測なら新規）。合成の日足で本番の印と全行一致
- 終値の母集団の書き換えも、抜けたら元に戻る
- 上昇局面（新規のブレイクで始まり、次の新規のブレイクの前で終わる）・局面の中の位置・大きさ
- 平均の一意性（判定期間が重なる行の数で割った平均。重ならなければ 1。手で数えた値と一致）
- KS 統計量・近さ（直前・直後 N 営業日）・本番の既定が違うと止まる
- 作った表を次の回に使う（同じデータなら build を呼ばない）・--rebuild とデータが変わったときは作り直す（古い表は消す）・
  足りない表だけ作る・案2 は省ける
- 合成の表で報告が全部の段を出し、確かめの数が 0 になり、銘柄コードを出さない。要約が保存される

  python3 tests/test_probe_augment_rows.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import probe_augment_rows as E  # noqa: E402

CODES = ("10010", "10020", "10030", "10040")


def synthetic_bars(codes=CODES, n=560, seed=3) -> pd.DataFrame:
    """78週の窓（368営業日）のあとに、上げ・押し・上げを入れて連続した更新日を作る。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n)
    out = []
    for c in codes:
        r = rng.normal(0.0003, 0.02, n)
        r[400:425] += 0.012
        r[425:445] -= 0.015
        r[445:470] += 0.015
        r[500:520] += 0.01
        close = 1000 * np.exp(np.cumsum(r))
        high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
        op = close * (1 + rng.normal(0, 0.005, n))
        vol = rng.integers(10_000, 50_000, n).astype(float)
        out.append(pd.DataFrame({"Date": days, "Code": c, "O": op, "H": high, "L": low, "C": close, "Vo": vol,
                                 "AdjO": op, "AdjH": high, "AdjL": low, "AdjC": close, "AdjVo": vol}))
    return pd.concat(out, ignore_index=True)


#: 合成の表に残す列（ラベルの引き直しに要る列と、分布を並べる列）
KEEP = ["Code", "Date", "label", "future_rise", "end_level", "uptrend_end", "vol_20d", "base_length",
        "broken_high_age", "pullback_depth", "break_margin", "ret_1d", "ret_5d", "ret_20d", "volume_trend",
        "r_high_3m", "close_position"]


def fake_build_rows(bars: pd.DataFrame) -> pd.DataFrame:
    """build の代わり: そのときの設定（差し替え・書き換え）のまま本番の関数で印とラベルを付け、新規のブレイクの印の行を返す。"""
    p = B.mark_new_highs(B.price_panel(bars))
    p = B.add_breakout_context(B.attach_rise_label(p))
    return E.norm(p[p["is_fresh_break"] & p["label"].notna() & p["high52w"].notna()][KEEP])


def synthetic_tables(bars: pd.DataFrame) -> dict:
    """本番の関数（price_panel・mark_new_highs・attach_rise_label・add_breakout_context）で表を作る（build の代わり）。"""
    cap: dict = {}
    with E.all_update_days(cap):
        al = fake_build_rows(bars)
    ds = al[E.fresh_by_base_length(al["base_length"])].reset_index(drop=True)
    with E.close_population():
        cl = fake_build_rows(bars)
    return {"ds": ds, "all": al, "updates": cap["updates"], "days": cap["days"], "close": cl}


class TestPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = B.price_panel(synthetic_bars())

    def test_marks_all_update_days_and_restores(self):
        orig = B.mark_new_highs
        ref = orig(self.panel.copy())
        cap: dict = {}
        with E.all_update_days(cap):
            self.assertIsNot(B.mark_new_highs, orig)
            self.assertEqual(B.SWEEP_OVERRIDES, {"ALL_UPDATE_DAYS": True})      # meta.json に残る目印
            p = B.mark_new_highs(self.panel.copy())
        self.assertIs(B.mark_new_highs, orig)
        self.assertEqual(B.SWEEP_OVERRIDES, {})
        self.assertTrue((p["is_fresh_break"] == p["is_new_high"]).all())
        up = cap["updates"]
        nh = ref[ref["is_new_high"]]
        self.assertEqual(len(up), len(nh))
        self.assertGreater(len(up), int(up["fresh"].sum()))                      # 連続した更新日がある
        np.testing.assert_array_equal(up["fresh"].to_numpy(), nh["is_fresh_break"].to_numpy())
        pos = ref.groupby("Code", sort=False).cumcount()[ref["is_new_high"]]
        np.testing.assert_array_equal(up["pos"].to_numpy(), pos.to_numpy())
        self.assertEqual(len(cap["days"]), self.panel["Date"].nunique())

    def test_restores_even_on_error(self):
        orig = B.mark_new_highs
        with self.assertRaises(RuntimeError):
            with E.all_update_days({}):
                raise RuntimeError("途中で落ちる")
        self.assertIs(B.mark_new_highs, orig)
        self.assertEqual(B.SWEEP_OVERRIDES, {})

    def test_base_length_tells_fresh(self):
        cap: dict = {}
        with E.all_update_days(cap):
            p = B.mark_new_highs(self.panel.copy())
        p = B.add_breakout_context(p)
        nh = p[p["is_new_high"]]
        np.testing.assert_array_equal(E.fresh_by_base_length(nh["base_length"]), cap["updates"]["fresh"].to_numpy())

    def test_close_population_restores(self):
        with E.close_population():
            self.assertEqual((B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH), ("close", False))
            self.assertEqual(B.SWEEP_OVERRIDES, {"BREAKOUT_REF": "close", "BREAKOUT_ON_HIGH": False})
        self.assertEqual((B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH, B.SWEEP_OVERRIDES), ("high", True, {}))

    def test_stops_when_defaults_differ(self):
        with E.close_population():
            with self.assertRaises(SystemExit):
                E.build_tables(tempfile.gettempdir())


class TestParts(unittest.TestCase):
    def test_fresh_by_base_length(self):
        bl = pd.Series([np.nan, 21, 20, 1, 400])
        np.testing.assert_array_equal(E.fresh_by_base_length(bl), [True, True, False, False, True])

    def test_clusters(self):
        up = pd.DataFrame({"Code": ["A"] * 5 + ["B"] * 2, "Date": pd.date_range("2024-01-01", periods=7),
                           "pos": [10, 12, 15, 40, 41, 5, 30],
                           "fresh": [True, False, False, True, False, False, True]})
        c = E.assign_clusters(up)
        a, b = c[c["Code"] == "A"], c[c["Code"] == "B"]
        self.assertEqual(a["cluster"].nunique(), 2)
        self.assertEqual(list(a["offset"]), [0, 2, 5, 0, 1])
        self.assertEqual(list(a["size"]), [3, 3, 3, 2, 2])
        self.assertEqual(int(b["cluster"].iloc[0]), -1)                      # 新規のブレイクより前の更新日
        self.assertTrue(np.isnan(b["offset"].iloc[0]))
        self.assertEqual(float(b["offset"].iloc[1]), 0.0)
        self.assertEqual(len(set(a["cluster"]) & set(b["cluster"])), 0)      # 銘柄をまたいで同じ番号にしない

    def test_uniqueness_by_hand(self):
        # 5営業日ずれた2行: 判定期間20日のうち15日が重なる → (5×1 + 15×1/2) / 20 = 0.625
        np.testing.assert_allclose(E.avg_uniqueness(["a", "a"], [0, 5]), [0.625, 0.625])
        np.testing.assert_allclose(E.avg_uniqueness(["a", "b"], [0, 5]), [1.0, 1.0])     # 銘柄が違えば重ならない
        np.testing.assert_allclose(E.avg_uniqueness(["a", "a"], [0, 20]), [1.0, 1.0])    # 20日ずれたら重ならない
        # 3行（0・1・2）: 行0 の判定期間 1〜20 → 1日目 1、2日目 1/2、3〜20日目 1/3、、、
        u = E.avg_uniqueness(["a", "a", "a"], [0, 1, 2])
        self.assertAlmostEqual(u[0], (1 + 1 / 2 + 18 / 3) / 20)
        self.assertAlmostEqual(u[2], (18 / 3 + 1 / 2 + 1) / 20)
        self.assertAlmostEqual(float(u.sum()), 22 / 20)                      # 合計 = 覆う日数（1〜22日目）÷ 20
        self.assertEqual(len(E.avg_uniqueness([], [])), 0)

    def test_ks(self):
        self.assertEqual(E.ks_stat([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertEqual(E.ks_stat([1, 2], [5, 6]), 1.0)
        self.assertAlmostEqual(E.ks_stat([1, 2, np.nan, 3, 4], [3, 4, 5, 6]), 0.5)
        self.assertTrue(np.isnan(E.ks_stat([np.nan], [1.0])))

    def test_nearby(self):
        only = pd.DataFrame({"Code": ["A", "A", "B"], "mpos": [100, 200, 100]})
        other = pd.DataFrame({"Code": ["A", "A", "B"], "mpos": [90, 215, 130]})
        nb = E.nearby(only, other, days=20)
        self.assertEqual(list(nb["before"]), [True, False, False])
        self.assertEqual(list(nb["after"]), [False, True, False])

    def test_market_pos(self):
        days = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-09"]).to_numpy()
        np.testing.assert_array_equal(E.market_pos(pd.Series(pd.to_datetime(["2024-01-05", "2024-01-09"])), days), [1, 2])


class TestBuildTables(unittest.TestCase):
    """作った表を次の回に使う・作り直す条件（build は合成の日足で表を作る差し替え）。"""

    @classmethod
    def setUpClass(cls):
        cls.bars = synthetic_bars()
        cls.ds = synthetic_tables(cls.bars)["ds"]

    def setUp(self):
        self.calls = []
        self.orig = B.build

        def fake_build(data_dir, out_path):
            self.calls.append(os.path.basename(out_path))
            fake_build_rows(self.bars).to_parquet(out_path, index=False)

        B.build = fake_build
        self.dir = tempfile.TemporaryDirectory()
        self.d = self.dir.name
        self.ds.to_parquet(os.path.join(self.d, "dataset.parquet"), index=False)

    def tearDown(self):
        B.build = self.orig
        self.dir.cleanup()

    def test_builds_once_then_reuses(self):
        t = E.build_tables(self.d)
        self.assertEqual(self.calls, ["pa0_all.parquet", "pa0_close.parquet"])
        self.assertGreater(len(t["all"]), len(t["ds"]))
        self.assertGreaterEqual(len(t["updates"]), len(t["all"]))           # 控えた更新日は除外の前
        self.assertEqual((B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH, B.SWEEP_OVERRIDES), ("high", True, {}))
        self.calls.clear()
        E.build_tables(self.d)
        self.assertEqual(self.calls, [])                                     # 同じデータなら作らない
        E.build_tables(self.d, rebuild=True)
        self.assertEqual(self.calls, ["pa0_all.parquet", "pa0_close.parquet"])

    def test_rebuilds_when_dataset_changes(self):
        E.build_tables(self.d)
        self.ds.iloc[1:].to_parquet(os.path.join(self.d, "dataset.parquet"), index=False)
        self.calls.clear()
        E.build_tables(self.d, with_close=False)
        self.assertEqual(self.calls, ["pa0_all.parquet"])
        self.assertFalse(os.path.exists(E.fpath(self.d, "close.parquet")))   # 古いデータの表は消す
        self.calls.clear()
        t = E.build_tables(self.d)
        self.assertEqual(self.calls, ["pa0_close.parquet"])                  # 足りない表だけ作る
        self.assertIsNotNone(t["close"])

    def test_close_is_optional(self):
        t = E.build_tables(self.d, with_close=False)
        self.assertEqual(self.calls, ["pa0_all.parquet"])
        self.assertIsNone(t["close"])


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = synthetic_tables(synthetic_bars())

    def test_report_runs_and_checks_are_zero(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summ = E.report(self.t)
        out = buf.getvalue()
        for sec in range(7):
            self.assertIn(f"■ {sec}.", out)
        ck = summ["check"]
        self.assertEqual((ck["fresh_only_table"], ck["dataset_only"], ck["flag_disagree"], ck["unmatched"]), (0, 0, 0, 0))
        self.assertEqual(ck["fresh_in_dataset"], len(self.t["ds"]))
        self.assertGreater(summ["plan1"]["added"], 0)
        self.assertAlmostEqual(summ["uniqueness"]["fresh_mean"], 1.0)        # いまの行どうしは判定期間が重ならない
        self.assertLess(summ["uniqueness"]["mean_added_in_union"], 1.0)
        self.assertLessEqual(summ["uniqueness"]["effective_union"], summ["uniqueness"]["union_rows"])
        self.assertIn("plan2", summ)
        for c in CODES:
            self.assertNotIn(c, out)                                         # 銘柄コードを出さない

    def test_main_saves_summary(self):
        orig = E.build_tables
        E.build_tables = lambda data_dir, with_close=True, rebuild=False: dict(self.t, close=self.t["close"] if with_close
                                                                               else None)
        try:
            with tempfile.TemporaryDirectory() as d:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = E.main(["--data-dir", d, "--no-close"])
                with open(E.fpath(d, "summary.json"), encoding="utf-8") as fh:
                    summ = json.load(fh)
        finally:
            E.build_tables = orig
        self.assertEqual(rc, 0)
        self.assertIn("作っていない（--no-close）", buf.getvalue())
        self.assertNotIn("plan2", summ)
        self.assertIn("plan1", summ)


if __name__ == "__main__":
    unittest.main()
