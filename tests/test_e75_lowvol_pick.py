#!/usr/bin/env python3
"""
実験75（research/exp/e75_lowvol_pick.py）: 通信も重い計算も無しで確かめる部分。

- 規則ごとの選び方（95優先の分岐・同点・候補の無い日・2銘柄・Jev の欠けた行）
- 2方向の SE（日付が全部違えば銘柄だけの SE と一致する。引数の順を入れ替えても同じ）
- 同じ日の対の差（同じ銘柄の日は差 0 として数に入れる。年ごとの符号）
- Jev の控えの読み込み（腕と版で絞る。Date が文字列でも付く。無ければ飛ばす）
- 集計が合成データで最後まで通り、JSON にできる

  python3 tests/test_e75_lowvol_pick.py
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

import e70_borderline as E70  # noqa: E402
import e74_jev_questions as E74  # noqa: E402
import e75_lowvol_pick as E  # noqa: E402

D1, D2, D3 = pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-04"), pd.Timestamp("2026-01-05")


def small() -> pd.DataFrame:
    """3日。1日目: 95以上が A・B、90〜95 が C。2日目: 90〜95 だけ D・E。3日目: 候補なし F。"""
    rows = [
        # Code, Date, hp_lgbm, hp_xgb, hp_cat, s_lgbm, vol_20d, tp20, jev_r
        ("A", D1, 99, 97, 96, 0.9, 3.0, 5.0, 60.0),
        ("B", D1, 96, 98, 95, 0.8, 1.0, 10.0, 50.0),
        ("C", D1, 92, 93, 91, 0.7, 2.0, -3.0, 40.0),
        ("D", D2, 93, 91, 94, 0.6, 4.0, 2.0, 30.0),
        ("E", D2, 90, 92, 95, 0.5, 0.5, 8.0, np.nan),
        ("F", D3, 80, 99, 99, 0.4, 1.0, 1.0, 20.0),
    ]
    d = pd.DataFrame(rows, columns=["Code", "Date", "hp_lgbm", "hp_xgb", "hp_cat", "s_lgbm", "vol_20d", "tp20", "jev_r"])
    d["tp10"] = d["tp20"].clip(upper=10)
    d["hit20"] = d["tp20"] >= 20
    d["r"] = d["tp20"] - 1
    d["label"] = (d["tp20"] > 4).astype(int)
    d["n_break"] = 10
    d["ret_20d"] = 1.0
    return E.prepare(d)


class TestPick(unittest.TestCase):
    def picks(self, d, rule):
        return sorted(d.loc[E.pick(d, rule), "Code"])

    def test_rules(self):
        d = small()
        self.assertEqual(list(d["cand90"]), [True, True, True, True, True, False])
        self.assertEqual(list(d["cand95"]), [True, True, False, False, False, False])
        self.assertEqual(self.picks(d, "all95"), ["A", "B"])
        self.assertEqual(self.picks(d, "all90"), ["A", "B", "C", "D", "E"])
        self.assertEqual(self.picks(d, "top1"), ["A", "D"])          # 3モデルの最小が最大
        self.assertEqual(self.picks(d, "top2"), ["A", "B", "D", "E"])
        self.assertEqual(self.picks(d, "lowvol1"), ["B", "E"])       # ボラ最小
        self.assertEqual(self.picks(d, "lowvol2"), ["B", "C", "D", "E"])
        self.assertEqual(self.picks(d, "pref95"), ["A", "E"])        # 95がある日は top1、無い日は lowvol1
        self.assertEqual(self.picks(d, "lowvol95"), ["B", "E"])      # 95がある日はその中の低ボラ
        self.assertEqual(self.picks(d, "lowjev1"), ["C", "D"])       # Jev 最小。E は答えが無いので D
        with self.assertRaises(ValueError):
            E.pick(d, "nope")

    def test_lowjev_without_column(self):
        d = small().drop(columns=["jev_r"])
        self.assertEqual(int(E.pick(d, "lowjev1").sum()), 0)

    def test_tie_breaks_by_hp_min_then_order(self):
        d = small()
        d.loc[d["Code"].isin(["B", "C"]), "vol_20d"] = 1.0     # 同点 → 3モデルの最小が高い B
        self.assertEqual(self.picks(d, "lowvol1"), ["B", "E"])

    def test_prepare_requires_columns(self):
        d = small().drop(columns=["vol_20d"])
        with self.assertRaises(SystemExit):
            E.prepare(d)


class TestStats(unittest.TestCase):
    def test_two_way_se_matches_one_way_when_other_is_unique(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 300)
        codes = rng.integers(0, 40, 300)
        dates = np.arange(300)                                  # 全部違う日
        self.assertAlmostEqual(E.two_way_se(x, codes, dates), E70.cluster_se(x, codes), places=10)
        self.assertAlmostEqual(E.two_way_se(x, codes, dates), E.two_way_se(x, dates, codes), places=10)
        self.assertTrue(np.isnan(E.two_way_se([1.0], ["a"], ["b"])))
        # 同じ銘柄が同じ日に 2行ずつ（相関を作る）→ 片方向より大きくならないことはない（符号だけ確かめる）
        x2 = np.repeat(rng.normal(0, 1, 50), 2)
        c2 = np.repeat(np.arange(50), 2)
        d2 = np.repeat(np.arange(50), 2)
        self.assertGreater(E.two_way_se(x2, c2, d2), 0)

    def test_summarize(self):
        d = small()
        s = E.summarize(d[E.pick(d, "all90")])
        self.assertEqual((s["n"], s["days"], s["codes"]), (5, 2, 5))
        self.assertAlmostEqual(s["tp20"], np.mean([5, 10, -3, 2, 8]))
        self.assertAlmostEqual(s["win20"], 80.0)
        self.assertEqual(E.summarize(d.iloc[0:0]), {"n": 0})

    def test_paired_diff(self):
        d = small()
        p = E.paired_diff(d, E.pick(d, "lowvol1"), E.pick(d, "top1"))
        self.assertEqual(p["n"], 2)
        self.assertAlmostEqual(p["d"], ((10 - 5) + (8 - 2)) / 2)
        self.assertEqual((p["same"], p["n_diff"]), (0.0, 2))
        self.assertEqual((p["years_pos"], p["years_n"]), (1, 1))
        q = E.paired_diff(d, E.pick(d, "lowvol95"), E.pick(d, "lowvol1"))   # 同じ銘柄を選ぶ
        self.assertEqual((q["n"], q["same"], q["n_diff"]), (2, 100.0, 0))
        self.assertEqual(q["d"], 0.0)
        self.assertTrue(np.isnan(q["d_diff"]))
        # 1銘柄 対 全部: 候補が1件の日だけ同じ集合
        r = E.paired_diff(d, E.pick(d, "lowvol1"), E.pick(d, "all90"))
        self.assertAlmostEqual(r["d"], ((10 - np.mean([5, 10, -3])) + (8 - np.mean([2, 8]))) / 2)
        self.assertEqual(E.paired_diff(d, E.pick(d, "all95") & False, E.pick(d, "top1")), {"n": 0})

    def test_terciles_and_measures(self):
        d = make_d(600)
        rows = E.terciles(d, "vol_20d")
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(r["n"] for r in rows), 600)
        self.assertLessEqual(rows[0]["hi"], rows[1]["lo"])
        m = E.measures_in_sets(d, E.MEASURES, [("母集団", pd.Series(True, index=d.index))])
        self.assertEqual(set(m["母集団"]["measures"]), set(E.MEASURES))
        self.assertIn("auc_c", m["母集団"]["measures"]["vol_20d"])
        self.assertEqual(E.terciles(d.head(5), "vol_20d"), [])


def make_d(n=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2023-01-02") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    d = pd.DataFrame({"Code": [f"{1000 + i % 120}0" for i in range(n)], "Date": dates})
    for a in ("lgbm", "xgb", "cat"):
        d[f"hp_{a}"] = rng.uniform(60, 100, n)
        d[f"s_{a}"] = d[f"hp_{a}"] / 100
    d["vol_20d"] = rng.uniform(0.5, 8, n)
    d["ret_20d"] = rng.normal(5, 10, n)
    d["r"] = rng.normal(1, 8, n)
    hit20 = rng.random(n) < 0.2
    d["hit20"] = hit20
    d["tp20"] = np.where(hit20, 20.0, d["r"])
    hit10 = hit20 | (rng.random(n) < 0.2)
    d["hit10"] = hit10
    d["tp10"] = np.where(hit10, 10.0, d["r"])
    d["label"] = (rng.random(n) < 0.25).astype(int)
    nb = {dt: int(v) for dt, v in zip(sorted(set(dates)), rng.integers(3, 30, len(set(dates))))}
    d["n_break"] = d["Date"].map(nb)
    d = E.prepare(d)
    d["jev_r"] = np.nan
    d["jev_c"] = np.nan
    late = d["Date"] >= d["Date"].max() - pd.Timedelta(days=200)
    d.loc[late, "jev_r"] = rng.uniform(10, 90, int(late.sum()))
    d.loc[late, "jev_c"] = rng.uniform(5, 70, int(late.sum()))
    return d


class TestReport(unittest.TestCase):
    def test_report_runs_and_is_json(self):
        d = make_d()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = E.report(d, {"rows": int(d["jev_r"].notna().sum())})
        out = buf.getvalue()
        for head in ("=== 0. 対象", "=== 1. 規則ごとの成績", "=== 2. 規則同士の差", "=== 3. 発火数",
                     "=== 4. 選定の集合の中で並べる物差し", "=== 5. 直近1年", "=== 6. 採否"):
            self.assertIn(head, out)
        self.assertEqual(set(res["rules"]), set(E.RULES) - {"lowjev1"})
        self.assertEqual(set(res["pairs"]), {f"{a}-{b}" for a, b in E.PAIRS})
        self.assertEqual(set(res["jev"]["rules"]), set(E.JEV_RULES))
        self.assertEqual(set(res["verdict"]), set(E.MAIN_RULES))
        for v in res["verdict"].values():
            self.assertEqual(len(v["conds"]), 4)
        s = json.dumps(res, ensure_ascii=False, default=E._js)
        self.assertIn("lowvol1", s)

    def test_dry_run_stops_early(self):
        d = make_d(300)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = E.report(d, {"rows": 0}, dry_run=True)
        self.assertIn("dry-run", buf.getvalue())
        self.assertIn("dry_run", res)
        self.assertNotIn("rules", res)

    def test_report_without_jev(self):
        d = make_d(400)
        d["jev_r"] = np.nan
        d["jev_c"] = np.nan
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = E.report(d, {"rows": 0})
        self.assertIn("飛ばす", buf.getvalue())
        self.assertEqual(res["jev"]["rules"], {})
        self.assertFalse(res["verdict"]["lowvol1"]["conds"][3][0])     # (4) は Jev が無いと満たせない


class TestLoadJev(unittest.TestCase):
    def test_merge_by_code_and_date_string(self):
        d = small()
        d = d.drop(columns=["jev_r"])
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "e74.parquet")
            c = pd.DataFrame({
                "arm": ["prod", "prod", "anon", "prod"],
                "Code": ["A", "B", "A", "C"],
                "Date": ["2025-03-03", "2025-03-03", "2025-03-03", "2025-03-03"],
                "qset": [E74.QSET, E74.QSET, E74.QSET, "other"],
                "state_sha": ["x"] * 4, "model": ["jev-1.13.0"] * 4,
                "a_tp10": [1.0] * 4, "b_tp20": [2.0] * 4, "c_label": [30.0, 35.0, 99.0, 99.0],
                "r_rise10": [60.0, 50.0, 99.0, 99.0],
                "input_tokens": [1] * 4, "output_tokens": [1] * 4, "asked_at": ["t"] * 4})
            c.to_parquet(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                out, info = E.load_jev(d, path=path, arm="prod")
            self.assertEqual(info["rows"], 2)
            got = out.set_index("Code")
            self.assertEqual((got.loc["A", "jev_r"], got.loc["B", "jev_c"]), (60.0, 35.0))
            self.assertTrue(np.isnan(got.loc["C", "jev_r"]))          # 版が違う行は使わない
            self.assertTrue(np.isnan(got.loc["D", "jev_r"]))
            with contextlib.redirect_stdout(io.StringIO()):
                out2, info2 = E.load_jev(small().drop(columns=["jev_r"]), path=os.path.join(tmp, "none.parquet"))
            self.assertEqual(info2["rows"], 0)
            self.assertTrue(out2["jev_r"].isna().all())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
