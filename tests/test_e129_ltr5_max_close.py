#!/usr/bin/env python3
"""
実験129（research/exp/e129_ltr5_max_close.py）。合成の日足と候補で:

- RMAX: 判断日の翌営業日の寄りで買い、買った日〜20営業日目の終値の最高値で売った収益。窓の最後の日の終値から作り直した収益が
  R20 と合う（合わなければ数える）、RMAX ≥ R20、20営業日に満たない行・寄り付かない行は欠測、R20 が欠ける行は RMAX も欠測
- 当てはめ・探索: 学ぶ並び（段階）も探索の NDCG の利得も、腕の目的変数（R20 / RMAX）の10分位
- 読み方: 基準は G10T。M10T がすべてのずらしで (a)(b) とも上 / 下 / それ以外
- 通しで回す（データの読み込み・印・探索は差し替え、当てはめは小さな木で本物）: 探索は2つの腕で1回ずつ（2回目は読む）、RMAX の
  確かめが記録に残り、公開ログに銘柄コードを出さない

  python3 tests/test_e129_ltr5_max_close.py
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
sys.path.insert(0, os.path.join(ROOT, "tests"))

import e129_ltr5_max_close as X  # noqa: E402
import test_e115_weekly_ltr as T115  # noqa: E402
import test_e117_rolling5_ltr as T117  # noqa: E402
import test_e118_ltr5_grades as T118  # noqa: E402

COLS, COLS_W = T118.COLS, T118.COLS_W


def bars_for(code, closes, opens):
    d = pd.bdate_range("2021-01-04", periods=len(closes))
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Code": code, "Date": d, "AdjO": np.asarray(opens, dtype=float), "AdjH": c, "AdjC": c})


class TestRmax(unittest.TestCase):
    def setUp(self):
        closes = [100.0] * 30
        closes[5] = 140.0                              # 窓の途中の最高値
        closes[20] = 110.0                             # 窓の最後の日（判断日 0 の i+20。live_track.forward と同じ数え方）
        self.a = bars_for("1111", closes, [100.0] * 30)
        self.b = bars_for("2222", [50.0] * 25, [50.0, np.nan] + [50.0] * 23)     # 判断日 0 の翌日は寄り付かない

    def rl(self, rows, r20):
        return pd.DataFrame({"Code": [c for c, _ in rows], "Date": [d for _, d in rows], X.R20: r20})

    def test_max_close_in_the_window_and_rebuilt_r20(self):
        bars = pd.concat([self.a, self.b], ignore_index=True)
        d = self.a["Date"]
        rows = [("1111", d[0]), ("1111", d[15]), ("2222", self.b["Date"][0]), ("2222", self.b["Date"][1])]
        rl = self.rl(rows, [0.10, np.nan, np.nan, 0.0])
        out, info = X.attach_rmax(rl, bars)
        self.assertAlmostEqual(out[X.RMAX][0], 0.40)                      # 140 / 100 − 1
        self.assertTrue(np.isnan(out[X.RMAX][1]))                         # 20営業日に満たない
        self.assertTrue(np.isnan(out[X.RMAX][2]))                         # 翌日の寄りが無い
        self.assertAlmostEqual(out[X.RMAX][3], 0.0)
        self.assertEqual(info["rebuild_mismatch"], 0)
        self.assertEqual(info["below_r20"], 0)
        self.assertEqual(info["only_one"], 0)
        bad, info2 = X.attach_rmax(self.rl(rows, [0.20, np.nan, np.nan, 0.0]), bars)
        self.assertEqual(info2["rebuild_mismatch"], 1)                    # R20 がずれていれば数える

    def test_rmax_is_dropped_when_r20_is_missing(self):
        a = self.a.copy()
        a.loc[20, "AdjC"] = np.nan                                        # 窓の最後の日の終値が無い
        out, info = X.attach_rmax(self.rl([("1111", a["Date"][0])], [np.nan]), a)
        self.assertTrue(np.isnan(out[X.RMAX][0]))
        self.assertEqual(info["only_one"], 0)
        self.assertEqual(info["rebuild_mismatch"], 0)


class TestTarget(unittest.TestCase):
    def frame(self):
        f = T118.pool_rows()
        f[X.RMAX] = f[X.R20].abs() + 0.01 * np.arange(len(f)) % 7
        return f

    def test_fit_rank_learns_the_target_grades(self):
        f = self.frame()
        seen = []
        real = X.grades

        def spy(dates, r, n):
            seen.append(np.asarray(r, dtype=float))
            return real(dates, r, n)

        with mock.patch.object(X, "grades", side_effect=spy):
            X.fit_rank(dict(T115.SMALL_RANK), f, f.head(20), COLS_W, 10, target=X.RMAX)
        d = f.sort_values(["Date", "Code"], kind="mergesort")
        np.testing.assert_array_equal(seen[0], d[X.RMAX].to_numpy(dtype=float))

    def test_tune_uses_the_target_for_the_gain(self):
        f = self.frame()
        d = sorted(f["Date"].unique())
        folds = [(f[f["Date"] <= d[19]], f[f["Date"] > d[19]])]
        with mock.patch.object(X, "fit_rank", side_effect=lambda p, tr, va, c, n, s, t: va[t].to_numpy()):
            rec = X.tune_g(f, COLS_W, folds, 10, n_trials=1, target=X.RMAX)
        self.assertEqual(rec["_cv"]["target"], X.RMAX)
        self.assertIn(X.RMAX, rec["_cv"]["objective"])
        self.assertEqual(rec["_cv"]["ndcg"], 1.0)                          # 目的変数の順に並べれば満点


class TestReading(unittest.TestCase):
    def row(self, sh, ret, rho):
        """ret・rho は (G10T, M10T)。"""
        return {"shift": sh, "top1": {a: {"ret_w": r} for a, r in zip(X.ARMS, ret)},
                "rank": {a: {"rho": p} for a, p in zip(X.ARMS, rho)}}

    def test_better_worse_none_against_g10t(self):
        self.assertEqual(X.ARMS, ("G10T", "M10T"))
        self.assertEqual(X.BASE, "G10T")
        up = [self.row(s, (0.010, 0.012), (0.050, 0.060)) for s in (0, 2, 4)]
        self.assertEqual(X.reading(up)["M10T"]["read"], "better")
        down = [self.row(s, (0.010, 0.008), (0.050, 0.040)) for s in (0, 2, 4)]
        self.assertEqual(X.reading(down)["M10T"]["read"], "worse")
        mixed = up[:2] + [self.row(4, (0.010, 0.012), (0.050, 0.040))]
        self.assertEqual(X.reading(mixed)["M10T"]["read"], "none")
        self.assertEqual(X.READ_JA["better"], "最高値のほうが良い")


class TestMain(unittest.TestCase):
    def test_main(self):
        full, bars, codes = T117.main_data()
        cutoff = pd.Timestamp("2019-12-31")
        no_pct_until = pd.Timestamp("2019-03-01")
        targets = []

        def fake_tune(sub, cols_w, folds, n, n_trials, target=X.R20):
            self.assertEqual(list(cols_w), COLS_W)
            self.assertEqual(n, 10)
            self.assertTrue(sub[target].notna().all())
            targets.append(target)
            return {"params": dict(T115.SMALL_RANK, num_leaves=6 if target == X.R20 else 7),
                    "_cv": {"objective": "x", "grades": 10, "ndcg": 0.5, "best_trial": 0, "minutes": 0.0,
                            "target": target}}

        def fake_rule(daily, models=None):
            line = 0.9 if tuple(models) == X.R0_TREES else 0.5
            r = daily[["Code", "Date"]].assign(r15=(daily["f1"] > line).astype(float), pmin=np.round(daily["f1"] * 10, 1))
            r.loc[r["Date"] < no_pct_until, ["r15", "pmin"]] = np.nan
            return r, {"found": True, "joined": len(r), "rows": len(r), "with_pct": int(r["r15"].notna().sum()),
                       "r15": int(r["r15"].sum()), "matched": int(r["r15"].notna().sum())}

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as st:
            chk = {"hold5_match": 0, "hold5_both": 0}
            rec5 = {"params": dict(T115.SMALL_RANK, num_leaves=5), "_cv": {"n_rows": 1, "train_to": "2019-01-01"},
                    "_source": "テスト"}
            for p in [mock.patch.object(X, "OOF_DIR", tmp),
                      mock.patch.object(X, "PARAMS_PATH", os.path.join(tmp, "e129_params.json")),
                      mock.patch.object(X, "SUMMARY_PATH", os.path.join(tmp, "e129_summary.json")),
                      mock.patch.object(X.F, "columns", return_value=list(COLS)),
                      mock.patch.object(X.W, "load_all", return_value=(full, bars, cutoff, chk)),
                      mock.patch.object(X.W, "daily_ltr_params",
                                        return_value={"params": dict(T115.SMALL_RANK, num_leaves=9), "_cutoff": "x"}),
                      mock.patch.object(X.L5, "load_params", return_value=rec5),
                      mock.patch.object(X.LTR, "params_mismatch", return_value=None),
                      mock.patch.object(X.R, "daily_rule", side_effect=fake_rule),
                      mock.patch.object(X, "tune_g", side_effect=fake_tune)]:
                st.enter_context(p)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(X.main(["--trials", "2", "--shifts", "0,2"]), 0)
            out = buf.getvalue()
            with open(os.path.join(tmp, "e129_summary.json"), encoding="utf-8") as fh:
                sm = json.load(fh)
            self.assertEqual(sorted(targets), sorted([X.R20, X.RMAX]))         # 2つの腕で1回ずつ探索
            self.assertEqual(sm["rmax_info"]["rebuild_mismatch"], 0)
            self.assertEqual(sm["rmax_info"]["below_r20"], 0)
            self.assertEqual(sm["params"]["G10T"]["num_leaves"], 6)
            self.assertEqual(sm["params"]["M10T"]["num_leaves"], 7)
            self.assertIn(sm["reading"]["M10T"]["read"], ("better", "worse", "none"))
            s0 = sm["shifts"]["0"]
            self.assertEqual(s0["top1"]["G10T"]["n"], s0["days"])
            self.assertTrue(np.isfinite(s0["top1"]["M10T"]["rmax"]))
            self.assertTrue(np.isfinite(s0["rank"]["M10T"]["rho_max"]))
            self.assertIn("RMAX（終値の最高値）", out)
            self.assertIn("読み方", out)
            for code in codes:
                self.assertNotIn(code, out)
            with contextlib.redirect_stdout(io.StringIO()):                    # 2回目は探索を読む
                self.assertEqual(X.main(["--trials", "2", "--shifts", "0,2"]), 0)
            self.assertEqual(len(targets), 2)


if __name__ == "__main__":
    unittest.main()
