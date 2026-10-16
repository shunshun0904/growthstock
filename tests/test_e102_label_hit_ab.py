#!/usr/bin/env python3
"""
実験102（research/exp/e102_label_hit_ab.py）の、重い計算を回さない部分のテスト。

- 回す前に決めたもの（腕・基準・判定する腕・ラベルの定義・判定の3条件・物差し）
- ラベルの引き直し: CUR12 は本番のラベルと同じ、HIT は CUR の正例を含む（終盤・トレンドを外したぶん増える）、TP10 は hit10 そのもの、
  現行のラベルが決まらない行は NaN
- 選び方（窓の中の上位・前の窓の閾値・月2銘柄）の成績と、窓ごとの差・判定（仕込んだ差は支持、無差は支持しない）
- 合成の結果で報告が最後まで出て、要約が保存され、銘柄コードを出さない

  python3 tests/test_e102_label_hit_ab.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import e68_edinet_ab as E68  # noqa: E402
import e102_label_hit_ab as E  # noqa: E402


def label_frame(n=600, seed=0):
    """ラベルの材料（vol_20d・future_rise・end_level・uptrend_end）と出口（hit10・c20・tp10c）を持つ合成の表。"""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(1.0, 4.0, size=n)
    rise = rng.normal(0.05, 0.15, size=n)
    end = rise - rng.uniform(0.0, 0.15, size=n)
    up = (rng.uniform(size=n) < 0.6).astype(float)
    df = pd.DataFrame({"Code": [f"{1000 + i % 40}0" for i in range(n)], "Date": pd.bdate_range("2021-01-04", periods=n),
                       "vol_20d": vol, "future_rise": rise, "end_level": end, "uptrend_end": up,
                       "ret_o1_20": end + rng.normal(0, 0.02, size=n), "hit10": rise >= 0.10, "c20": end * 100,
                       "hitday10": np.where(rise >= 0.10, rng.integers(1, 21, size=n), np.nan),
                       "ul_fwd": (rng.uniform(size=n) < 0.06).astype(float)})
    df["tp10c"] = np.where(df["hit10"], 10.0, df["c20"])
    y12 = B.rise_label_for_k(df.assign(label=0.0), 1.2)
    df["label"] = y12
    df.loc[df.index[:10], "label"] = np.nan                 # 決まらない行
    return df


class TestLineup(unittest.TestCase):
    def test_decided_before_running(self):
        self.assertEqual(E.ARMS, ("CUR15", "CUR12", "HIT15", "HIT12", "TP10"))
        self.assertEqual(E.BASE, "CUR15")
        self.assertEqual(E.JUDGED, ("HIT15", "HIT12", "TP10"))
        self.assertEqual(E.LABEL_DEFS["HIT15"], {"kind": "sigma", "k": 1.5, "end": False, "trend": False})
        self.assertEqual(E.LABEL_DEFS["TP10"], {"kind": "tp", "target": 10.0})
        self.assertEqual((E.OUTCOME, E.MAIN_FORM, E.MAIN_Q, E.Z_MIN), ("tp10c", "top05", 0.05, 1.64))
        self.assertEqual(E.ALGO, "lgbm")
        self.assertEqual([k for k, _, _ in E.FORMS], ["top05", "top10", "thr95", "month2"])


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.df = label_frame()

    def test_relabel(self):
        df = self.df
        y12 = E.relabel(df, E.LABEL_DEFS["CUR12"])
        ok = df["label"].notna()
        self.assertTrue((y12[ok] == df.loc[ok, "label"]).all())      # 1.2σ の3条件は本番のラベルそのもの
        self.assertTrue(y12[~ok].isna().all())
        y15 = E.relabel(df, E.LABEL_DEFS["CUR15"])
        h15 = E.relabel(df, E.LABEL_DEFS["HIT15"])
        self.assertTrue(((y15 == 1) <= (h15 == 1)).all())            # 到達のみは3条件の正例を含む
        self.assertGreater(h15.sum(), y15.sum())
        self.assertLessEqual(y15.sum(), y12.sum())                     # 1.5σ は 1.2σ より厳しい
        tp = E.relabel(df, E.LABEL_DEFS["TP10"])
        self.assertTrue((tp[ok] == df.loc[ok, "hit10"].astype(float)).all())
        self.assertTrue(tp[~ok].isna().all())
        with self.assertRaises(KeyError):
            E.relabel(df, {"kind": "x"})

    def test_build_and_report(self):
        df, tab = E.build_labels(self.df.copy())
        self.assertEqual([r["arm"] for r in tab], list(E.ARMS))
        for a in E.ARMS:
            self.assertIn(f"y_{a}", df.columns)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = E.label_report(df)
        self.assertIn("■ 1.", buf.getvalue())
        self.assertAlmostEqual(out["CUR15"]["covers_base_pos"], 1.0)
        self.assertAlmostEqual(out["HIT15"]["covers_base_pos"], 1.0)


def synthetic_oof(seed: int, strength: float = 0.4, folds=(3, 4, 5), n_per=600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for f in folds:
        y = (rng.uniform(size=n_per) < 0.2).astype(int)
        s = np.clip(0.2 + strength * y + rng.normal(0, 0.15, size=n_per), 0, 1)
        d = pd.bdate_range(f"{2018 + f}-06-01", periods=n_per)
        tp = np.where(rng.uniform(size=n_per) < 0.3 + 0.3 * y, 10.0, rng.normal(0, 6, size=n_per))
        parts.append(pd.DataFrame({"Code": [f"{1000 + i}0" for i in range(n_per)], "Date": d, "label": y, "score": s,
                                   "fold": f, "ret_o1_20": tp / 100 * 0.8, "ret_o1_40": rng.normal(0.01, 0.06, size=n_per),
                                   "tp10c": tp, "hit10": tp >= 10, "ul_fwd": (rng.uniform(size=n_per) < 0.05).astype(float),
                                   "vol_20d": rng.uniform(1, 4, size=n_per),
                                   **{f"y_{a}": y for a in E.ARMS}}))
    return pd.concat(parts, ignore_index=True)


class TestForms(unittest.TestCase):
    def test_top_threshold_monthly(self):
        o = synthetic_oof(0)
        t = E.top_within(o, 0.05)
        self.assertEqual(list(t.columns), ["fold", "n", "tp", "ret20", "hit", "bad", "ul", "pos_base"])
        self.assertEqual(len(t), 3)
        self.assertTrue((t["n"] >= 30).all())
        self.assertGreater(t["tp"].mean(), o["tp10c"].mean())          # 上位はスコアが効くぶん収益が高い（窓の平均で）
        th = E.threshold_prior(o, 95)
        self.assertEqual(list(th["fold"]), [4, 5])                         # 窓3 は参照が無い
        m = E.monthly(o, 2)
        self.assertEqual(len(m), 3)
        self.assertTrue((m["n"] <= 2 * 30).all())
        self.assertEqual(len(E.top_within(o[o["fold"] == 3].head(50), 0.05)), 0)

    def test_paired_and_judge(self):
        base = pd.DataFrame({"shift": [0] * 10 + [2] * 10, "fold": list(range(10)) * 2, "tp": np.linspace(1, 3, 20)})
        up = base.assign(tp=base["tp"] + np.linspace(0.5, 1.5, 20))
        d = E.paired(base, up, "tp")
        self.assertEqual(d["n"], 20)
        self.assertEqual(d["wins"], 20)
        j = E.judge(d)
        self.assertTrue(j["pass"])
        flat = base.assign(tp=base["tp"] + np.where(np.arange(20) % 2 == 0, 0.3, -0.3))
        j0 = E.judge(E.paired(base, flat, "tp"))
        self.assertFalse(j0["pass"])
        self.assertEqual(E.paired(base, base.iloc[0:0], "tp")["n"], 0)
        self.assertFalse(E.judge({"n": 0})["pass"])
        self.assertIn("z", E.fmt_pair(d))


class TestReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E68.OOF_DIR, E68.RUN, E68.TAG)
        E68.OOF_DIR, E68.RUN, E68.TAG = self.dir, "e102test_lgbm", "e102test"

    def tearDown(self):
        E68.OOF_DIR, E68.RUN, E68.TAG = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_report_prints_every_section_and_saves(self):
        df = label_frame()
        df, _ = E.build_labels(df)
        cv = {"mean_pr_auc": 0.33, "std": 0.02, "mean_roc_auc": 0.64, "fold_scores": [0.31, 0.32, 0.33, 0.34, 0.35]}
        strength = {"CUR15": 0.4, "CUR12": 0.4, "HIT15": 0.5, "HIT12": 0.45, "TP10": 0.55}
        res = {"recs": {a: {"params": {"learning_rate": 0.05}, "_cv": cv, "_seconds": 5} for a in E.ARMS},
               "params": {a: {"learning_rate": 0.05} for a in E.ARMS}, "prod": {}, "shifted": {}}
        for j, a in enumerate(E.ARMS):
            res["prod"][a] = synthetic_oof(10 + j, strength=strength[a])
            for sh in (0, 2):
                res["shifted"][(sh, a)] = synthetic_oof(100 + 10 * j + sh, strength=strength[a])
        stamp = {"built_utc": "2026-10-16 07:00 UTC", "rows": len(df), "date_min": "2021-01-04", "date_max": "2023-06-30"}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(res, df, [0, 2], stamp)
        text = buf.getvalue()
        for head in ("■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ まとめ", "判定"):
            self.assertIn(head, text)
        self.assertIn("ずらし2か月（3窓）", text)
        for a in E.JUDGED:
            self.assertIn(a, s["judge"])
            self.assertIn("pass", s["judge"][a])
        self.assertNotIn("CUR15", s["judge"])
        self.assertEqual(set(s["forms"]), {"top05", "top10", "thr95", "month2"})
        self.assertTrue(os.path.exists(os.path.join(self.dir, "e102test_lgbm_summary.json")))
        with open(os.path.join(self.dir, "e102test_lgbm_summary.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertIn("judge", saved)
        for code in ("10000", "10010", "15990"):
            self.assertNotIn(code, text)


if __name__ == "__main__":
    unittest.main()
