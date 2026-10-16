"""
実験100（本番モデルの線が OOF より緩い原因の切り分け）の部品のテスト。

  地合いの列の検出（日内で一定）・印と分布の要約・反事実の置き換え・時期ごとの表・五分位の表・
  TreeSHAP の加法性（差の合計 = 生スコアの差）・判定・報告の形（銘柄の値を出さない）
"""
import contextlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
import e100_line_drift as E  # noqa: E402
import e100b_nk225_fix as EB  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402


def synth_oof(n_days=120, per_day=8, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-05-03", periods=n_days)
    df = pd.DataFrame([{"Code": f"{1000 + j}", "Date": d} for d in dates for j in range(per_day)])
    for a in E.RULE_MODELS:
        df[f"s_{a}"] = rng.random(len(df))
        df[f"p_{a}"] = LT.pct_of(df[f"s_{a}"].to_numpy(), df[f"s_{a}"].to_numpy())
    df["label_k15"] = (rng.random(len(df)) < 0.2).astype(float)
    df["label"] = (rng.random(len(df)) < 0.17).astype(float)
    df[lab.OUTCOME] = rng.normal(0, 0.05, len(df))
    df["window"] = np.where(df["Date"] < dates[60], 1, 2)
    df["mkt"] = np.repeat(rng.normal(size=n_days), per_day)
    return df


class TestPieces(unittest.TestCase):
    def test_market_wide(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        df = pd.DataFrame({"Date": np.repeat(dates, 4), "mkt": np.repeat(np.arange(30.0), 4),
                           "stock": np.arange(120.0), "sparse": np.nan})
        df.loc[df.index[:8], "sparse"] = 1.0           # 値のある日が2日だけ → 数えない
        self.assertEqual(E.market_wide(df, ["mkt", "stock", "sparse", "missing"], min_days=5), ["mkt"])

    def test_pass_mask_and_summary(self):
        p = {"lgbm_k15": np.array([95.0, 96.0, 50.0, np.nan]), "xgb_k15": np.array([95.0, 94.9, 99.0, 99.0]),
             "cat_k15": np.array([99.0, 99.0, 99.0, 99.0])}
        self.assertEqual(E.pass_mask(p).tolist(), [True, False, False, False])
        s = E.summary_of(p)
        self.assertEqual(s["n"], 3)                     # NaN は分布から外す
        self.assertAlmostEqual(s["pass"], 25.0)         # 印は4行のうち1行
        self.assertAlmostEqual(s["ge95"], 200 / 3)
        self.assertAlmostEqual(s["ge95_by_model"]["cat_k15"], 100.0)
        self.assertEqual(E.dist(np.array([np.nan]))["n"], 0)

    def test_replace_and_medians(self):
        hist = pd.DataFrame({"Date": pd.to_datetime(["2023-10-02", "2023-10-02", "2023-10-25", "2023-11-01"]),
                             "mkt": [1.0, 1.0, 5.0, 9.0], "x": [1.0, 2.0, 3.0, 4.0]})
        self.assertEqual(E.daily_medians(hist, ["mkt"]), {"mkt": 5.0})                       # 日ごとに1つ → 1, 5, 9
        self.assertEqual(E.daily_medians(hist, ["mkt"], month=10, day_hi=20), {"mkt": 1.0})   # 10/20 まで
        self.assertEqual(E.daily_medians(hist, ["mkt", "nope"], month=12), {})
        live = pd.DataFrame({"mkt": [7.0, 8.0], "x": [0.0, 0.0]})
        out = E.replace_cols(live, {"mkt": 5.0, "nope": 1.0})
        self.assertEqual(out["mkt"].tolist(), [5.0, 5.0])
        self.assertEqual(out["x"].tolist(), [0.0, 0.0])
        self.assertEqual(live["mkt"].tolist(), [7.0, 8.0])                                   # 元は変えない
        self.assertAlmostEqual(E.value_pct(np.array([1.0, 2.0, 3.0, 4.0]), 3.5), 75.0)
        self.assertTrue(np.isnan(E.value_pct(np.array([]), 1.0)))

    def test_period_tables(self):
        oof = synth_oof()
        w = E.window_table(oof)
        self.assertEqual(w["window"].tolist(), [1, 2])
        self.assertEqual(int(w["n"].sum()), len(oof))
        for a in E.RULE_MODELS:                        # 自分の分布に対する百分位なので 95以上は約5%
            self.assertLess(abs(float((oof[f"p_{a}"] >= 95).mean() * 100) - 5.0), 1.0)
        y = E.year_table(oof)
        self.assertEqual(y["year"].tolist(), [2021])
        m = E.month_table(oof)
        self.assertEqual(int(m["n"].sum()), len(oof))
        self.assertIn("ret_pass", w.columns)
        row = E.period_row(oof)
        self.assertEqual(row["n_pass"], int(E.pass_mask({a: oof[f"p_{a}"].to_numpy() for a in E.RULE_MODELS}).sum()))

    def test_quintile_table(self):
        oof = synth_oof()
        q = E.quintile_table(oof, "mkt")
        self.assertEqual(q["q"].tolist(), [1, 2, 3, 4, 5])
        self.assertTrue((q["lo"] <= q["hi"]).all())
        self.assertEqual(int(q["n"].sum()), len(oof))
        self.assertTrue(E.quintile_table(oof.head(30), "mkt").empty)

    def test_weekly_table(self):
        dates = pd.to_datetime(["2026-09-09", "2026-09-10", "2026-09-14", "2026-09-15"])
        rows = pd.DataFrame({"Date": np.repeat(dates, 2)})
        p = {a: np.array([96.0, 10, 96, 20, 50, 60, 96, 96]) for a in E.RULE_MODELS}
        w = E.weekly_table(rows, p)
        self.assertEqual(len(w), 2)
        self.assertEqual(w["days"].tolist(), [2, 2])
        self.assertAlmostEqual(w["pass"].iloc[0], 50.0)

    def test_judge(self):
        live = {"median": 86.0, "le10": 3.0}
        lw = {"ge95_lgbm_k15": 12.0}
        v = dict((k, (a, b)) for k, a, b in E.judge(live, lw, {"median": 60.0}, {"median": 52.0},
                                                      {"pos15": 24.0}, 17.0, 0.4))
        self.assertTrue(v["形"][0].startswith("水準の上方シフト"))
        self.assertTrue(v["OOF の直近の窓"][0].startswith("直近は高い"))
        self.assertTrue(v["地合いの特徴量"][0].startswith("主因"))
        self.assertTrue(v["決算の暦"][0].startswith("寄与あり"))
        self.assertTrue(v["地合いの元"][0].startswith("OOF でも"))
        self.assertEqual(v["学習と予測のずれ"][0], "原因ではない")
        v2 = dict((k, (a, b)) for k, a, b in E.judge({"median": 55.0, "le10": 9.0}, {"ge95_lgbm_k15": 5.0},
                                                       {"median": 54.0}, {"median": 54.0}, {"pos15": 17.5}, 17.0, 3.0))
        self.assertTrue(v2["形"][0].startswith("上方シフトとは言えない"))
        self.assertTrue(v2["OOF の直近の窓"][0].startswith("直近も定義どおり"))
        self.assertTrue(v2["地合いの特徴量"][0].startswith("主因ではない"))
        self.assertTrue(v2["地合いの元"][0].startswith("地合いの過信"))
        self.assertTrue(v2["学習と予測のずれ"][0].startswith("ずれがある"))
        v3 = dict((k, (a, b)) for k, a, b in E.judge(live, lw, {"median": 60.0}, {"median": 52.0}, None, 17.0, np.nan))
        self.assertEqual(v3["地合いの元"][0], "比べられない")


class TestContrib(unittest.TestCase):
    def test_decompose_additivity(self):
        import lightgbm as lgb

        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 6))
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=500) > 0).astype(int)
        m = lgb.LGBMClassifier(n_estimators=30, num_leaves=7, learning_rate=0.1, verbose=-1).fit(X, y)
        cols = [f"f{i}" for i in range(6)]
        live, hist = X[:40] + np.array([1.0, 0, 0, 0, 0, 0]), X[40:]
        C = E.contrib_matrix(m, live)
        self.assertEqual(C.shape, (40, 7))
        tbl, info = E.decompose(m, cols, live, hist)
        self.assertEqual(list(tbl.columns), ["col", "live", "hist", "diff", "abs"])
        self.assertAlmostEqual(float(tbl["diff"].sum()), info["raw_live"] - info["raw_hist"], places=9)
        self.assertEqual(tbl["col"].iloc[0], "f0")   # ずらした列の差が最大
        # 生スコアは predict_proba の対数オッズと一致
        p = m.predict_proba(live)[:, 1]
        self.assertTrue(np.allclose(C.sum(axis=1), np.log(p / (1 - p)), atol=1e-6))
        g = E.group_sums(tbl, ["f1"], ["f2"])
        self.assertAlmostEqual(sum(g.values()), float(tbl["diff"].sum()))


class TestSplitFix(unittest.TestCase):
    def test_corrected_series(self):
        dates = pd.bdate_range("2026-09-01", periods=40)
        price = np.full(40, 40000.0)
        price[25:] = 420.0                               # 1:100 の分割（+5% の値動き込み）
        s = pd.Series(price, index=dates)
        fixed, splits = EB.corrected_series(s)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0][0], dates[25])
        self.assertAlmostEqual(splits[0][1], 40000 / 420, places=6)
        self.assertAlmostEqual(float(fixed.iloc[0]), 400.0)   # 100 に丸めて割る
        self.assertAlmostEqual(float(fixed.iloc[-1]), 420.0)
        self.assertAlmostEqual(float(fixed.pct_change(20).iloc[-1]), 0.05)
        self.assertEqual(EB.find_splits(pd.Series([1.0, 1.1, 0.9], index=dates[:3])), [])
        self.assertEqual(EB.nice_factor(95.2), 100.0)
        self.assertEqual(EB.nice_factor(2.1), 2.0)
        self.assertAlmostEqual(EB.nice_factor(0.098), 0.1)        # 併合 10:1

    def test_with_fixed_and_daily_pass(self):
        dates = pd.to_datetime(["2026-10-13", "2026-10-13", "2026-10-14"])
        rows = pd.DataFrame({"Code": ["10000", "20000", "10000"], "Date": dates,
                             "nk225_ret_20": [-99.0] * 3, "nk225_ret_120": [-98.8] * 3, "x": [1.0, 2.0, 3.0]})
        after = pd.DataFrame({"Date": pd.to_datetime(["2026-10-13", "2026-10-14"]),
                              "nk225_ret_20": [2.0, 3.0], "nk225_ret_120": [25.0, 26.0]})
        out = EB.with_fixed(rows, after, ["nk225_ret_20", "nk225_ret_120", "x"])
        self.assertEqual(out["nk225_ret_20"].tolist(), [2.0, 2.0, 3.0])
        self.assertEqual(out["nk225_ret_120"].tolist(), [25.0, 25.0, 26.0])
        self.assertEqual(out["x"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(rows["nk225_ret_20"].tolist(), [-99.0] * 3)           # 元は変えない
        p = {a: np.array([96.0, 50.0, 97.0]) for a in E.RULE_MODELS}
        d = EB.daily_pass(rows, p)
        self.assertEqual(d["n"].tolist(), [2, 1])
        self.assertEqual(d["passed"].tolist(), [1, 1])

    def test_unadjusted_splits(self):
        dates = pd.bdate_range("2026-03-01", periods=5)
        b = pd.DataFrame({"Date": list(dates) * 2, "Code": ["10000"] * 5 + ["20000"] * 5,
                          "C": [100, 100, 50, 50, 50, 100, 100, 100, 100, 100],
                          "AdjC": [100, 100, 50, 50, 50, 100, 100, 50, 50, 50]})
        b["C"] = b["C"].astype(float)
        b["AdjC"] = b["AdjC"].astype(float)
        un = EB.unadjusted_splits(b)
        self.assertEqual(list(un.keys()), ["10000"])                            # C も動いた銘柄だけ
        self.assertEqual(un["10000"], dates[2])


class TestReport(unittest.TestCase):
    def test_report_smoke(self):
        oof = synth_oof()
        pc = {a: oof[f"p_{a}"].to_numpy()[:64] for a in E.RULE_MODELS}
        s = E.summary_of(pc)
        q = E.quintile_table(oof, "mkt")
        res = {"live_from": "2026-10-13", "live_to": "2026-10-16", "n_live": 64,
               "screen": {a: E.dist(pc[a]) for a in E.RULE_MODELS}, "rescore_maxdiff": 0.0,
               "windows": E.window_table(oof), "years": E.year_table(oof), "months": E.month_table(oof),
               "dec_info": {"raw_live": -0.5, "raw_hist": -1.2, "base_live": -1.0, "base_hist": -1.0},
               "groups": {"地合い（日内で一定）": 0.5, "決算の暦": 0.1, "銘柄固有（それ以外）": 0.1},
               "mkt_cols": ["mkt"],
               "top": [{"col": "mkt", "name": "地合い", "diff": 0.5, "live": 0.3, "hist": -0.2, "v_live": 1.5, "v_hist": 0.1,
                        "v_pct": 97.0, "mkt": True, "cal": False}],
               "cf": {"そのまま": s, "A": s, "B": s, "C": s}, "quintiles": {"mkt": q},
               "live_vals": {"mkt": 1.5}, "live_pct": {"mkt": 97.0}, "live_q": {"mkt": 5}, "overall_pos": 20.0,
               "train_to": "2026-09-08",
               "weekly": E.weekly_table(pd.DataFrame({"Date": pd.to_datetime(["2026-09-09"] * 64)}), pc),
               "skew": {"n": 64, "median": 0.0, "abs_median": 0.0, "abs_max": 0.1},
               "refs": {"全体（10行）": s}, "judge": E.judge(s, {"ge95_lgbm_k15": 5.0}, s, s, None, 20.0, 0.0)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.report(res)
        out = buf.getvalue()
        for key in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "反事実", "←いま"):
            self.assertIn(key, out)
        self.assertNotIn("1000", out)   # 銘柄の値を出さない


if __name__ == "__main__":
    unittest.main()
