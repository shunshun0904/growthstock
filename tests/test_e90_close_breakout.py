#!/usr/bin/env python3
"""
実験90（research/exp/e90_close_breakout.py）の、重い計算を回さない部分のテスト。

- 腕・母集団の定義・票・種・判定の行・規則の線（回す前に決めたもの）
- build_dataset の既定（high / 日中の高値）は変えていない。実験が母集団を作るときだけ書き換え、終わったら戻す
- ラベルの引き直し、共通の行、対照 P の無作為の抜き方（件数が合う・種で決まる）
- 前向きの百分位（前の窓が2つ未満の窓は付かない・画面と同じ丸め）、規則の表（発火・出口・共通の印）、
  規則の成績・窓ごとの差・取引の形
- 合成の結果で、表（母集団・探索の CV・OOF・窓ごと・§7・窓ごとの値・規則の成績・判定）が最後まで出て、要約が保存される
- e41.oof_folds の train_mask（学習の行だけ絞り、予測は全行）
- 時間の上限を過ぎたら新しい計算を始めない・知らない --k は止める

  python3 tests/test_e90_close_breakout.py
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

import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import live_track as LT  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e90_close_breakout as E  # noqa: E402


class TestLineup(unittest.TestCase):
    def test_decided_before_running(self):
        self.assertEqual(E.ALGOS, ("lgbm", "xgb", "cat", "logit"))
        self.assertEqual(E.JUDGE_ALGOS, ("lgbm", "xgb", "cat"))
        self.assertEqual(E.ARMS, ("A", "B", "C", "P"))
        self.assertEqual(E.POP_OF, {"A": "high", "B": "close", "C": "close", "P": "high"})
        self.assertEqual(E.PARAMS_OF, {"A": "A", "B": "B", "C": "A", "P": "A"})
        self.assertEqual(E.POP_DEF, {"high": ("high", True), "close": ("close", False)})
        self.assertEqual(E.PRIMARY_ROWS, "common")
        self.assertEqual(E.SUBSET_SEED, 20261015)
        self.assertEqual(E.KS, (1.5, 1.2))
        self.assertEqual((E.RULE_LINE, E.TP, E.MIN_BREAK), (95.0, 10.0, 1))      # 画面の規則と同じ
        self.assertEqual((E.MIN_TRAIN_WINDOWS, E.MIN_GROUP), (2, 5))
        self.assertEqual((E.k_tag(1.5), E.k_tag(1.2)), ("k15", "k12"))
        self.assertEqual(len(F.columns(F.DEFAULT_PRESET)), 249)    # 2026-10-18 から（実験90 の時点は239列）

    def test_production_default_is_untouched(self):
        self.assertEqual((B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH), E.POP_DEF["high"])
        self.assertEqual(B.SWEEP_OVERRIDES, {})

    def test_unknown_k_stops(self):
        with self.assertRaises(SystemExit):
            E.main(["--k", "1.8", "--dry"])


class TestBuildPopulation(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved_build = E.B.build

    def tearDown(self):
        E.B.build = self.saved_build
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_switches_while_building_and_restores_after(self):
        seen = {}

        def fake_build(data_dir, out_path):
            seen["ref"], seen["on_high"] = B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH
            seen["overrides"] = dict(B.SWEEP_OVERRIDES)
            pd.DataFrame({"Code": ["10000"], "Date": ["2024-01-04"], "label": [1.0]}).to_parquet(out_path, index=False)

        E.B.build = fake_build
        out = os.path.join(self.dir, "close.parquet")
        df = E.build_population("close", out)
        self.assertEqual((seen["ref"], seen["on_high"]), ("close", False))
        self.assertEqual(seen["overrides"], {"BREAKOUT_REF": "close", "BREAKOUT_ON_HIGH": False})
        self.assertEqual(len(df), 1)
        self.assertEqual((B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH), ("high", True))     # 戻っている
        self.assertEqual(B.SWEEP_OVERRIDES, {})
        # 保存済みなら作り直さない
        E.B.build = lambda *a: (_ for _ in ()).throw(AssertionError("作り直した"))
        self.assertEqual(len(E.build_population("close", out)), 1)

    def test_restores_even_when_building_fails(self):
        def boom(data_dir, out_path):
            raise RuntimeError("x")

        E.B.build = boom
        with self.assertRaises(RuntimeError):
            E.build_population("close", os.path.join(self.dir, "none.parquet"))
        self.assertEqual((B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH), ("high", True))
        self.assertEqual(B.SWEEP_OVERRIDES, {})


def label_frame(n=400, seed=0):
    """ラベルの材料（vol_20d・future_rise・end_level・uptrend_end）を持つ合成の表。"""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(1.5, 4.0, n)
    need12, end12 = B.rise_thresholds(pd.Series(vol), B.DEFAULT_RISE)
    rise = need12.to_numpy() * rng.uniform(0.5, 1.8, n)
    end = end12.to_numpy() * rng.uniform(0.5, 2.5, n)
    up = rng.integers(0, 2, n).astype(float)
    lab12 = ((rise >= need12.to_numpy()) & (end >= end12.to_numpy()) & (up == 1)).astype(float)
    lab12[:5] = np.nan
    return pd.DataFrame({"Code": [f"{1000 + i % 20}0" for i in range(n)],
                         "Date": pd.bdate_range("2023-01-04", periods=n),
                         "vol_20d": vol, "future_rise": rise, "end_level": end, "uptrend_end": up,
                         "label": lab12})


class TestLabelsAndKeys(unittest.TestCase):
    def test_k15_is_stricter_and_k12_keeps_the_label(self):
        df = label_frame()
        o12, o15 = E.with_label(df, 1.2), E.with_label(df, 1.5)
        self.assertEqual((len(o12), len(o15)), (len(df) - 5, len(df) - 5))
        self.assertLess(o15["label"].sum(), o12["label"].sum())
        np.testing.assert_array_equal(o15["label"].to_numpy(), B.rise_label_for_k(df, 1.5).dropna().to_numpy())

    def test_common_keys_and_in_keys(self):
        fa = pd.DataFrame({"Code": ["10000", "20000", "30000"], "Date": pd.to_datetime(["2024-01-04"] * 3)})
        fb = pd.DataFrame({"Code": ["20000", "30000", "40000"], "Date": pd.to_datetime(["2024-01-04"] * 3)})
        ck = E.common_keys(fa, fb)
        self.assertEqual(len(ck), 2)
        np.testing.assert_array_equal(E.in_keys(fa, ck), [False, True, True])
        np.testing.assert_array_equal(E.in_keys(fb, ck), [True, True, False])

    def test_nearby_marks_the_other_populations_breakouts(self):
        d = pd.to_datetime
        only = pd.DataFrame({"Code": ["10000", "10000", "20000", "30000"],
                             "Date": d(["2024-03-01", "2024-06-03", "2024-03-01", "2024-03-01"]),
                             "label": [1.0, 0.0, 1.0, 0.0]})
        other = pd.DataFrame({"Code": ["10000", "10000", "20000", "30000"],
                              "Date": d(["2024-02-20", "2024-06-20", "2024-03-01", "2024-01-05"])})
        out = E.nearby(only, other, days=20)
        # 10000 の 3/1: 2/20 が直前（10暦日）。10000 の 6/3: 6/20 が直後。20000: 同じ日は数えない。30000: 55日前は窓の外
        self.assertEqual(out["before"].tolist(), [True, False, False, False])
        self.assertEqual(out["after"].tolist(), [False, True, False, False])
        self.assertEqual(list(out.columns), ["Code", "Date", "label", "before", "after"])

    def test_subset_mask_is_exact_and_seeded(self):
        fa = pd.DataFrame({"x": np.arange(100)})
        m1, m2 = E.subset_mask(fa, 37), E.subset_mask(fa, 37)
        self.assertEqual(int(m1.sum()), 37)
        np.testing.assert_array_equal(m1, m2)
        self.assertFalse(np.array_equal(m1, E.subset_mask(fa, 37, seed=1)))
        self.assertEqual(int(E.subset_mask(fa, 500).sum()), 100)                # 母集団より多くは抜けない


def synthetic_oof(seed, n_folds=4, per_day=5, days=30, strength=0.4, first_fold=1):
    """fold ごとに日付が分かれた合成の out-of-fold（Code・Date・label・score・fold・ret_o1_20/40）。"""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2021-06-01", periods=n_folds * days)
    for f in range(n_folds):
        for d in dates[f * days:(f + 1) * days]:
            for c in range(per_day):
                y = int(rng.random() < 0.25)
                rows.append({"Code": f"{1000 + c}0", "Date": d, "label": y, "fold": f + first_fold,
                             "score": y * strength + rng.random(), "ret_o1_20": rng.normal(0.01 + 0.03 * y, 0.05),
                             "ret_o1_40": rng.normal(0.01 + 0.04 * y, 0.07)})
    return pd.DataFrame(rows)


def synthetic_bars(codes, start="2021-05-01", periods=180, seed=3):
    """合成の日足（分割調整後の始値・高値・終値）。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=periods)
    parts = []
    for i, c in enumerate(codes):
        px = 1000 * (1 + 0.1 * i) * np.cumprod(1 + rng.normal(0.0005, 0.02, periods))
        parts.append(pd.DataFrame({"Date": days, "Code": c, "AdjO": px * (1 + rng.normal(0, 0.003, periods)),
                                   "AdjH": px * (1 + np.abs(rng.normal(0.01, 0.01, periods))), "AdjC": px}))
    return pd.concat(parts, ignore_index=True)


class TestForwardPct(unittest.TestCase):
    def test_needs_two_prior_windows_and_rounds_like_the_screen(self):
        o = synthetic_oof(1)
        p = E.forward_pct(o, ["score"])
        self.assertEqual(list(p.columns), ["pct_score"])
        by = p["pct_score"].notna().groupby(o["fold"]).mean()
        self.assertEqual(by.loc[1], 0.0)                                        # 前の窓が無い
        self.assertEqual(by.loc[2], 0.0)                                        # 前の窓が1つ
        self.assertEqual(by.loc[3], 1.0)
        self.assertEqual(by.loc[4], 1.0)
        v = p["pct_score"].dropna()
        self.assertTrue(((v >= 0) & (v <= 100)).all())
        self.assertTrue(np.allclose(v, v.round(1)))                             # 小数1桁
        # 前の窓のどのスコアより高ければ 100、低ければ 0
        o2 = o.copy()
        top = o2.index[o2["fold"] == 4][0]
        o2.loc[top, "score"] = 10.0
        o2.loc[top + 1, "score"] = -10.0
        p2 = E.forward_pct(o2, ["score"])
        self.assertEqual((p2.loc[top, "pct_score"], p2.loc[top + 1, "pct_score"]), (100.0, 0.0))


class TestRuleFrame(unittest.TestCase):
    def setUp(self):
        self.codes = [f"{1000 + c}0" for c in range(5)]
        self.bars = synthetic_bars(self.codes)
        self.bar_days = list(pd.to_datetime(np.sort(self.bars["Date"].unique())))
        base = synthetic_oof(7)
        self.prods = {}
        for i, a in enumerate(E.JUDGE_ALGOS):
            o = base.copy()
            o["score"] = o["label"] * 0.5 + np.random.default_rng(i).random(len(o))
            self.prods[a] = o
        # 母集団の表: OOF の行 + 1日に1行余計（発火は OOF の行数より多い日がある）
        extra = base[base["Date"] == base["Date"].min()].iloc[:1].assign(Code="99990")
        self.pop = pd.concat([base[["Code", "Date"]], extra[["Code", "Date"]]], ignore_index=True)
        self.ck = E.keys_of(base[base["Code"] != "10000"])                        # 1銘柄だけ共通でない

    def test_rule_frame_columns_and_windows(self):
        ft = E.rule_frame(self.prods, self.pop, self.bars, self.ck)
        self.assertEqual(sorted(ft["window"].unique()), [3, 4])
        for c in ("min_K15", "score", "n_break", "tp10c", "hit10", "hitday10", "common", "window"):
            self.assertIn(c, ft.columns)
        self.assertTrue(ft["min_K15"].notna().all())
        self.assertTrue((ft["n_break"] >= 5).all())
        self.assertEqual(int(ft["common"].sum()), int((ft["Code"] != "10000").sum()))
        self.assertTrue(np.allclose(ft["score"], ft["score_lgbm"]))
        self.assertTrue(pd.to_numeric(ft["tp10c"], errors="coerce").notna().mean() > 0.9)

    def test_rule_frame_with_lgbm_only(self):
        """運用者の指示（lgbm だけ）: 規則の最小は lgbm の百分位そのもの。無いモデルは要らない。"""
        ft3 = E.rule_frame(self.prods, self.pop, self.bars, self.ck)
        ft1 = E.rule_frame({"lgbm": self.prods["lgbm"]}, self.pop, self.bars, self.ck)
        self.assertEqual(len(ft1), len(ft3))
        self.assertTrue(np.allclose(ft1["min_K15"], ft1["pct_score_lgbm"]) if "pct_score_lgbm" in ft1.columns
                        else (ft1["min_K15"] >= ft3["min_K15"]).all())
        self.assertTrue((ft1["min_K15"].to_numpy() >= ft3["min_K15"].to_numpy() - 1e-9).all())   # 3本の最小 ≤ 1本
        with self.assertRaises(ValueError):
            E.rule_frame({}, self.pop, self.bars, self.ck)

    def test_stats_pairs_and_trade_form(self):
        ft = E.rule_frame(self.prods, self.pop, self.bars, self.ck)
        months = E.E82.n_months(ft)
        rule = ft[ft["min_K15"] >= 50.0]                                         # 合成なので線を下げて行を残す
        s = E.rule_stats(rule, months)
        for k in ("n", "mean", "med", "win", "hit", "bad10", "se", "per_month", "month_median", "by_window"):
            self.assertIn(k, s)
        self.assertEqual(s["n"], len(rule))
        self.assertAlmostEqual(s["per_month"], len(rule) / months)
        self.assertEqual(sorted(s["by_window"]), [3, 4])
        other = E.rule_stats(ft[ft["min_K15"] >= 70.0], months)
        d = E.paired_windows(s, other)
        self.assertEqual(d["n"], 2)
        self.assertEqual(E.paired_windows(s, E.rule_stats(rule.iloc[:0], months))["n"], 0)
        t = E.trade_form(ft.assign(min_K15=ft["min_K15"].clip(lower=96.0)), self.bar_days, months)
        self.assertGreater(t["n"], 0)
        self.assertLessEqual(t["n"], len(ft))
        self.assertTrue(np.isfinite(t["capital_pm"]))


def big_oof(seed, strength=0.4):
    """1窓 600行（上位10% の超過は前の窓に 500件以上要る: lab.threshold_edge）。"""
    return synthetic_oof(seed, n_folds=3, per_day=10, days=60, strength=strength, first_fold=0)


def fake_results(algos=("lgbm", "logit"), shifts=(0, 2), keysets=None):
    base = big_oof(0)
    keysets = keysets if keysets is not None else {"common": E.keys_of(base[base["Code"] != "10000"])}
    out = {}
    for i, a in enumerate(algos):
        cv = {"mean_pr_auc": 0.33, "std": 0.02, "mean_roc_auc": 0.64, "fold_scores": [0.31, 0.32, 0.33, 0.34, 0.35]}
        cvb = {**cv, "mean_pr_auc": 0.335, "fold_scores": [0.32, 0.32, 0.34, 0.33, 0.36]}
        recs = {"A": {"params": {"learning_rate": 0.05, "num_leaves": 31}, "_cv": cv, "_seconds": 10},
                "B": {"params": {"learning_rate": 0.04, "num_leaves": 15}, "_cv": cvb, "_seconds": 12}}
        prod, prod_full = {rs: {} for rs in keysets}, {}
        wins = {rs: {} for rs in keysets}
        for j, arm in enumerate(E.ARMS):
            o = big_oof(10 * i + j, strength={"A": 0.4, "B": 0.45, "C": 0.5, "P": 0.38}[arm])
            prod_full[arm] = o
            for rs, keys in keysets.items():
                prod[rs][arm] = o if keys is None else E.on_rows(o, keys)
            for sh in shifts:
                o2 = big_oof(100 * i + 10 * j + sh, strength={"A": 0.4, "B": 0.45, "C": 0.5, "P": 0.38}[arm])
                for rs, keys in keysets.items():
                    wins[rs][(sh, arm)] = E.windows_on(o2, keys)
        out[a] = {"recs": recs, "params": {arm: recs[E.PARAMS_OF[arm]]["params"] for arm in E.ARMS},
                  "prod": prod, "prod_full": prod_full, "wins": wins}
    return out, keysets


STAMP = {"built_utc": "2026-10-15 12:00 UTC", "common": 480, "common_by_year": {"2021": 480},
         "high": {"rows": 600, "labeled": 600, "pos_rate_k12": 0.25, "date_min": "2021-06-01", "date_max": "2021-11-12",
                  "rows_by_year": {"2021": 600}, "pos_by_year": {"2021": 0.25}, "codes": 5},
         "close": {"rows": 500, "labeled": 500, "pos_rate_k12": 0.26, "date_min": "2021-06-01", "date_max": "2021-11-12",
                   "rows_by_year": {"2021": 500}, "pos_by_year": {"2021": 0.26}, "codes": 5}}


class TestReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E.OOF_DIR, E.RUN)
        E.OOF_DIR, E.RUN = self.dir, "e90test_k15"

    def tearDown(self):
        E.OOF_DIR, E.RUN = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_population_report(self):
        base = big_oof(0)
        fa = base.copy()
        fb = base[base["Code"] != "10000"].copy()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pop = E.population_report({"high": fa, "close": fb}, STAMP, 1.5)
        text = buf.getvalue()
        self.assertIn("■ 0. 母集団の違い", text)
        self.assertIn("共通の行", text)
        self.assertEqual((pop["common"], pop["only_high"], pop["only_close"]), (len(fb), len(fa) - len(fb), 0))
        self.assertIn("high だけの行", text)
        self.assertIn("片方だけの行の内訳", text)
        self.assertEqual(sum(v["n"] for v in pop["nearby"]["high"].values()), len(fa) - len(fb))
        self.assertNotIn("close", pop["nearby"])                                 # close だけの行は無い

    def test_report_prints_every_section_and_saves_the_summary(self):
        res, keysets = fake_results()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(res, [0, 2], STAMP, keysets, {"label_k": 1.5}, 1.5, None)
        text = buf.getvalue()
        for head in ("■ 1.", "■ 2a.", "■ 3a.", "■ 4a.", "■ 5a.", "■ 6."):
            self.assertIn(head, text)
        self.assertIn("判定に使う行（回す前に決めた）", text)
        self.assertIn("木3モデルがそろってから判定", text)
        self.assertIn("木モデル（lgbm, xgb, cat のどれか）の結果がそろった回で出す", text)
        self.assertIn("lgbm・ずらし2か月（3窓）", text)
        self.assertNotIn("nan", text.lower().replace("nan（", ""))
        self.assertEqual(len(s["per_window"]["common"]), 12)                     # 2モデル × 2通り × 3窓
        self.assertIn("lgbm_common", s["judge7"])
        self.assertIsNone(s["strategy"])
        with open(os.path.join(self.dir, "e90test_k15_summary.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual((saved["label_k"], saved["primary_rows"], saved["subset_seed"]), (1.5, "common", 20261015))

    def test_strategy_report_reads_the_rules(self):
        codes = [f"{1000 + c}0" for c in range(5)]
        bars = synthetic_bars(codes)
        bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
        base = synthetic_oof(7)
        prods = {a: base.assign(score=base["label"] * 0.5 + np.random.default_rng(i).random(len(base)))
                 for i, a in enumerate(E.JUDGE_ALGOS)}
        ck = E.keys_of(base[base["Code"] != "10000"])
        ft = E.rule_frame(prods, base[["Code", "Date"]], bars, ck)
        months = E.E82.n_months(ft)
        st = {"models": ["lgbm"]}
        for arm, line in zip(E.ARMS, (60.0, 55.0, 65.0, 70.0)):
            rule = ft[ft["min_K15"] >= line]
            st[arm] = {"pop": E.POP_OF[arm], "rows": len(ft), "windows": [3, 4], "months": months,
                       "rule": E.rule_stats(rule, months), "rule_common": E.rule_stats(rule[rule["common"]], months),
                       "rule_other": E.rule_stats(rule[~rule["common"]], months),
                       "trade": E.trade_form(ft.assign(min_K15=ft["min_K15"].clip(lower=96.0)), bar_days, months)}
        st["pairs"] = {f"{y}-{x}": E.paired_windows(st[x]["rule"], st[y]["rule"]) for x, y in E.PAIRS}
        st["pairs"]["A_common-A"] = E.paired_windows(st["A"]["rule"], st["A"]["rule_common"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = E.strategy_report(st, 1.5)
        text = buf.getvalue()
        self.assertIn("■ 6. 規則の成績", text)
        self.assertIn("■ 7. 判定", text)
        self.assertIn("(ii) 規則", text)
        self.assertIn("(iii) 軽い代わり", text)
        self.assertIn("行ってこい", text)
        self.assertIn("lgbm の1本（運用者の指示で絞った", text)
        for k in ("models", "rule_ok", "count_ok", "filter_equiv", "arms", "pairs"):
            self.assertIn(k, out)
        self.assertIsInstance(out["rule_ok"], bool)


class TestTrainMask(unittest.TestCase):
    def test_mask_limits_training_rows_only(self):
        import walkforward as WF
        rng = np.random.default_rng(0)
        n = 2400                                                                 # 学習 2,100行（半分に絞っても 1,000 以上）
        dates = pd.bdate_range("2015-01-05", periods=n)
        x = rng.normal(size=(n, 3))
        y = (x[:, 0] + 0.5 * rng.normal(size=n) > 0.3).astype(float)
        df = pd.DataFrame({"Code": "10000", "Date": dates, "label": y, "ret_o1_20": rng.normal(size=n),
                           "ret_o1_40": rng.normal(size=n), "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]})
        folds = [WF.Fold(1, str(dates[0].date()), str(dates[2099].date()), str(dates[2100].date()), str(dates[-1].date()))]
        params = {"n_estimators": 5, "num_leaves": 4, "verbose": -1}
        full = E41.oof_folds("lgbm", df, ["x1", "x2", "x3"], params, 42, folds)
        same = E41.oof_folds("lgbm", df, ["x1", "x2", "x3"], params, 42, folds, train_mask=np.ones(n, dtype=bool))
        half = E41.oof_folds("lgbm", df, ["x1", "x2", "x3"], params, 42, folds, train_mask=(np.arange(n) % 2 == 0))
        self.assertEqual((len(full), len(same), len(half)), (300, 300, 300))     # 予測は窓の全行
        np.testing.assert_allclose(full["score"], same["score"])
        self.assertFalse(np.allclose(full["score"], half["score"]))
        with self.assertRaises(ValueError):
            E41.oof_folds("lgbm", df, ["x1"], params, 42, folds, train_mask=np.ones(3, dtype=bool))


class TestTimeBudget(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT)
        E.OOF_DIR, E.RUN = self.dir, "e90test_k15"

    def tearDown(self):
        E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_nothing_new_after_the_deadline(self):
        frame = pd.DataFrame({"Code": ["10000"] * 3, "Date": pd.bdate_range("2024-01-04", periods=3),
                              "label": [1, 0, 1], "x": [1.0, 2.0, 3.0]})
        frame.attrs["built_utc"] = "2026-10-15 00:00 UTC"
        E.DEADLINE, E.TIMED_OUT = time.time() - 1, False
        self.assertIsNone(E.oof("lgbm", "close", frame, ["x"], {"n_estimators": 1}, 0, (42,)))
        self.assertIsNone(E.tune("lgbm", "A", frame, ["x"], pd.Timestamp("2024-01-10"), 2,
                                 {"built_utc": "2026-10-15 00:00 UTC"}))
        self.assertTrue(E.TIMED_OUT)
        self.assertEqual(os.listdir(self.dir), [])
        self.assertTrue(E.path("summary.json").endswith("e90test_k15_summary.json"))
        self.assertTrue(E.fpath("frame_close.parquet").endswith(f"{E.TAG}_frame_close.parquet"))


if __name__ == "__main__":
    unittest.main()
