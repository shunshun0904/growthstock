#!/usr/bin/env python3
"""
実験101（research/exp/e101_limit_up_recency_ab.py）の、重い計算を回さない部分のテスト。実験100 のテストと同じ形で、7列と腕 D を確かめる。

- 回す前に決めたもの（学習器・腕 A/B/C/P/D・種・ラベル・対照の種・高スコアの割合・7列と実験100 の4列）と、7列が本番の列に無いこと
- 列を付ける: 合成の日足から 7列（減衰つきの和・60日と120日の回数を含む）と ul_fwd が付き、結果の列は特徴量に入らない
- 対照 P: 4列を同じ日付の行どうしで入れ替えても、日ごとの値の多重集合は変わらない
- 窓ごとの AUC に期間・件数・正例率が付く。上位 q の集計（件数・正例率・収益・ストップ高）
- 合成の結果で、表（探索の CV・本番と同じ OOF・窓ごと・判定1・判定2・窓ごとの値の一覧）が最後まで出て、要約が保存される
- 時間の上限を過ぎたら新しい計算を始めない・知らない --k は止める

  python3 tests/test_e101_limit_up_recency_ab.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import features as F  # noqa: E402
import e101_limit_up_recency_ab as E  # noqa: E402


class TestLineup(unittest.TestCase):
    def test_decided_before_running(self):
        self.assertEqual(E.ALGO, "lgbm")
        self.assertEqual(E.SEEDS, (42, 7, 123))
        self.assertEqual(E.ARMS, ("A", "B", "C", "P", "D"))
        self.assertEqual(E.COLS_OF["D"], "prev")
        self.assertEqual(E.PARAMS_OF["D"], "A")
        self.assertEqual(E.KS, (1.5, 1.2))
        self.assertEqual(E.PERM_SEED, 20261017)
        self.assertEqual(E.TOP_Q, 0.05)
        self.assertEqual(E.FEATURES, ("ul_t", "ul_past20_n", "ul_past60_n", "ul_past120_n", "ul_past250_n", "ul_days_since", "ul_decay"))
        self.assertEqual(E.PREV_FEATURES, ("ul_t", "ul_past20_n", "ul_past250_n", "ul_days_since"))
        self.assertTrue(set(E.PREV_FEATURES) < set(E.FEATURES))
        self.assertIn(("D", "C"), E.PAIRS)
        self.assertEqual(E.OUTCOME_COL, "ul_fwd")
        self.assertNotIn(E.OUTCOME_COL, E.FEATURES)
        self.assertEqual((E.k_tag(1.5), E.k_tag(1.2)), ("k15", "k12"))

    def test_new_columns_are_not_production_columns(self):
        base = F.columns(F.DEFAULT_PRESET)
        self.assertEqual(len(base), 239)
        self.assertFalse(set(base) & set(E.FEATURES))
        self.assertFalse(any(c.startswith("ul_") for c in base))         # 本番の設定には書いていない

    def test_unknown_k_stops(self):
        with self.assertRaises(SystemExit):
            E.main(["--k", "1.8", "--dry"])


class TestAttach(unittest.TestCase):
    def test_features_and_outcome(self):
        n = 320
        closes = np.full(n, 1000.0)
        highs = closes.copy()
        t = 280
        for k in (t - 100, t - 5, t, t + 3):
            highs[k] = 1300                                   # 基準 1,000 → 幅 300
        bars = pd.DataFrame({"Date": pd.bdate_range("2024-01-04", periods=n), "Code": "10010", "H": highs, "L": closes,
                             "C": closes, "AdjC": closes})
        keys = pd.DataFrame({"Code": ["10010", "10010", "99990"], "Date": [bars["Date"].iloc[t], bars["Date"].iloc[t - 50], bars["Date"].iloc[t]]})
        out, info = E.attach_features(keys, bars, hold=20, past=250)
        self.assertEqual(list(out.columns), ["Code", "Date"] + list(E.FEATURES) + [E.OUTCOME_COL])
        r = out.iloc[0]
        self.assertEqual((r["ul_t"], r["ul_past20_n"], r["ul_past250_n"], r["ul_days_since"], r["ul_fwd"]), (1.0, 1.0, 2.0, 5.0, 1.0))
        self.assertEqual((r["ul_past60_n"], r["ul_past120_n"]), (1.0, 2.0))
        self.assertAlmostEqual(r["ul_decay"], 0.98 ** 5 + 0.98 ** 100)
        r2 = out.iloc[1]                                      # t−50: 前250日に t−100 だけ。あとの20日には無い
        self.assertEqual((r2["ul_t"], r2["ul_past250_n"], r2["ul_days_since"], r2["ul_fwd"]), (0.0, 1.0, 50.0, 0.0))
        self.assertAlmostEqual(r2["ul_decay"], 0.98 ** 50)
        self.assertTrue(np.isnan(out.iloc[2]["ul_past250_n"]))
        self.assertEqual(info["attached"], 2)
        self.assertEqual(info["past_full"], 1)                # t−50 は日足の始まりから 230日しか無い

    def test_permuted_keeps_values_within_date(self):
        from e44_shortsale import permuted
        d = pd.to_datetime(["2024-01-04"] * 4 + ["2024-01-05"] * 3)
        df = pd.DataFrame({"Date": d, "Code": [str(i) for i in range(7)], "ul_t": [1, 0, 0, 0, 1, 1, 0],
                           "ul_decay": [0.5, 0, 1.2, 2, 0.1, 0, 0], "x": range(7)})
        p = permuted(df, ["ul_t", "ul_decay"], seed=E.PERM_SEED)
        for day, g in df.groupby("Date"):
            gp = p[p["Date"] == day]
            for c in ("ul_t", "ul_decay"):
                self.assertEqual(sorted(g[c]), sorted(gp[c]))
        self.assertEqual(list(p["x"]), list(df["x"]))         # ほかの列と行の並びはそのまま


def synthetic_oof(seed: int, strength: float = 0.4, folds=(3, 4, 5), n_per=600) -> pd.DataFrame:
    """窓ごとに n_per 行。lab.threshold_edge は参照の分布に 500行以上を要るので、窓は 600行にしてある。"""
    rng = np.random.default_rng(seed)
    parts = []
    for f in folds:
        y = (rng.uniform(size=n_per) < 0.2).astype(int)
        s = np.clip(0.2 + strength * y + rng.normal(0, 0.15, size=n_per), 0, 1)
        d = pd.bdate_range(f"{2018 + f}-06-01", periods=n_per)
        parts.append(pd.DataFrame({"Code": [f"{1000 + i}0" for i in range(n_per)], "Date": d, "label": y, "score": s,
                                   "fold": f, "ret_o1_20": rng.normal(0.01, 0.05, size=n_per) + 0.02 * y,
                                   "ret_o1_40": rng.normal(0.01, 0.06, size=n_per),
                                   "ul_fwd": (rng.uniform(size=n_per) < 0.05 + 0.1 * y).astype(float)}))
    return pd.concat(parts, ignore_index=True)


class TestWindows(unittest.TestCase):
    def test_windows_and_tops(self):
        o = synthetic_oof(0)
        w = E.windows_on(o)
        self.assertEqual(list(w.columns), ["fold", "pr", "roc", "start", "end", "n", "pos"])
        self.assertEqual(len(w), 3)
        self.assertTrue((w["n"] == 600).all())
        t = E.top_by_window(o, 0.10)
        self.assertEqual(list(t.columns), ["fold", "n_top", "pos_top", "ret_top", "ul_top"])
        self.assertEqual(len(t), 3)
        self.assertTrue((t["n_top"] >= 60).all())
        self.assertTrue((t["pos_top"] > o["label"].mean()).all())       # 上位はスコアが効くぶん正例率が高い
        self.assertTrue(np.isfinite(t["ret_top"]).all() and np.isfinite(t["ul_top"]).all())
        small = E.top_by_window(o[o["fold"] == 3].head(50), 0.10)
        self.assertEqual(len(small), 0)                                  # MIN_WINDOW_ROWS 未満の窓は落ちる

    def test_with_outcome(self):
        o = synthetic_oof(1).drop(columns=["ul_fwd"])
        frame = synthetic_oof(1)[["Code", "Date", "ul_fwd"]]
        got = E.with_outcome(o, frame)
        self.assertIn("ul_fwd", got.columns)
        self.assertEqual(len(got), len(o))


class TestReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E.OOF_DIR, E.RUN)
        E.OOF_DIR, E.RUN = self.dir, "e101test_k15"

    def tearDown(self):
        E.OOF_DIR, E.RUN = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def results(self, shifts=(0, 2)):
        cv = {"mean_pr_auc": 0.33, "std": 0.02, "mean_roc_auc": 0.64, "fold_scores": [0.31, 0.32, 0.33, 0.34, 0.35]}
        cvb = {**cv, "mean_pr_auc": 0.335, "fold_scores": [0.32, 0.32, 0.34, 0.33, 0.36]}
        recs = {"A": {"params": {"learning_rate": 0.05, "num_leaves": 31}, "_cv": cv, "_seconds": 10},
                "B": {"params": {"learning_rate": 0.04, "num_leaves": 15}, "_cv": cvb, "_seconds": 12}}
        strength = {"A": 0.4, "B": 0.45, "C": 0.5, "P": 0.38, "D": 0.44}
        prod, wins, tops = {}, {}, {}
        for j, arm in enumerate(E.ARMS):
            prod[arm] = synthetic_oof(10 + j, strength=strength[arm])
            for sh in shifts:
                o2 = synthetic_oof(100 + 10 * j + sh, strength=strength[arm])
                wins[(sh, arm)] = E.windows_on(o2)
                tops[(sh, arm)] = {q: E.top_by_window(o2, q) for q in E.TOP_QS}
        return {"recs": recs, "params": {arm: recs[E.PARAMS_OF[arm]]["params"] for arm in E.ARMS},
                "prod": prod, "wins": wins, "tops": tops}

    def test_report_prints_every_section_and_saves_the_summary(self):
        res = self.results()
        stamp = {"built_utc": "2026-10-16 05:00 UTC", "rows": 450, "date_min": "2021-06-01", "date_max": "2023-12-29",
                 "rows_by_year": {"2021": 150, "2022": 150, "2023": 150}, "ul_t_by_year": {"2021": 10}, "past_any_by_year": {"2021": 30},
                 "flags": {"rows": 1000, "known_share": 96.0, "ul": 30, "exact_among_ul": 93.0, "attached": 450, "past_full": 440}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(res, [0, 2], stamp, {"label_k": 1.5}, 1.5)
        text = buf.getvalue()
        for head in ("■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "■ まとめ", "判定1", "判定2"):
            self.assertIn(head, text)
        self.assertIn("ずらし2か月（3窓）", text)
        self.assertIn("2021-06-01〜", text)
        self.assertEqual(len(s["per_window"]), 6)                      # 2通り × 3窓
        self.assertIn("pass", s["judge1"])
        self.assertIn("q05", s["judge2"])
        self.assertIn("q10", s["judge2"])
        self.assertIn("pr_C-D", s["windows"])                            # 参考 C−D（重み3列の上積み）
        self.assertTrue(os.path.exists(os.path.join(self.dir, "e101test_k15_summary.json")))
        with open(os.path.join(self.dir, "e101test_k15_summary.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(saved["label_k"], 1.5)
        self.assertEqual(saved["features"], list(E.FEATURES))
        for code in ("10000", "10010"):
            self.assertNotIn(code, text)

    def test_coverage_report(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({"Code": [f"{1000 + i}0" for i in range(n)], "Date": pd.bdate_range("2022-01-04", periods=n),
                           "label": rng.integers(0, 2, size=n).astype(float), "ul_t": rng.integers(0, 2, size=n).astype(float),
                           "ul_past20_n": rng.integers(0, 3, size=n).astype(float), "ul_past250_n": rng.integers(0, 6, size=n).astype(float),
                           "ul_past60_n": rng.integers(0, 4, size=n).astype(float), "ul_past120_n": rng.integers(0, 5, size=n).astype(float),
                           "ul_decay": rng.uniform(0, 2, size=n),
                           "ul_days_since": np.where(rng.uniform(size=n) < 0.5, rng.integers(1, 250, size=n), np.nan),
                           "ul_fwd": rng.integers(0, 2, size=n).astype(float)})
        stamp = {"built_utc": "x", "rows": n, "rows_by_year": {"2022": n}, "ul_t_by_year": {"2022": 5}, "past_any_by_year": {"2022": 50},
                 "flags": {}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            c = E.coverage_report(df, stamp, 1.5)
        self.assertIn("■ 0.", buf.getvalue())
        self.assertEqual(c["全行"]["n"], n)
        self.assertAlmostEqual(c["missing_days_since"], float(df["ul_days_since"].isna().mean()))


class TestTimeBudget(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT)
        E.OOF_DIR, E.RUN = self.dir, "e101test_k15"

    def tearDown(self):
        E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_nothing_new_after_the_deadline(self):
        frame = pd.DataFrame({"Code": ["10000"] * 3, "Date": pd.bdate_range("2024-01-04", periods=3),
                              "label": [1, 0, 1], "x": [1.0, 2.0, 3.0]})
        frame.attrs["built_utc"] = "2026-10-16 00:00 UTC"
        E.DEADLINE, E.TIMED_OUT = time.time() - 1, False
        self.assertIsNone(E.oof("new", frame, ["x"], {"n_estimators": 1}, 0, (42,)))
        self.assertIsNone(E.tune("A", frame, ["x"], pd.Timestamp("2024-01-10"), 2, {"built_utc": "2026-10-16 00:00 UTC"}))
        self.assertTrue(E.TIMED_OUT)
        self.assertEqual(os.listdir(self.dir), [])
        self.assertTrue(E.path("summary.json").endswith("e101test_k15_summary.json"))
        self.assertTrue(E.fpath("frame.parquet").endswith(f"{E.TAG}_frame.parquet"))


if __name__ == "__main__":
    unittest.main()
