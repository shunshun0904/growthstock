#!/usr/bin/env python3
"""
実験114c（research/exp/e114c_ltr_retune.py。ランキング学習を毎週探索し直してよいかの確かめ。§78-10）。合成データで:

- 週ごとの打ち切り: k 週前のデータの最終日と、そこからの打ち切り（run_tuning.py と同じ式）。実験114 の最終日 2026-09-24 →
  2025-08-26。データの最終日で切っても行の順は変わらない
- 探索の置き場: 同じ条件なら読み、週が違えば別の名前。実験114 の探索そのもの（e114_search）も置き場に残る
- 本番の探索（tune_week → ltr.tune）と実験114 の探索そのもの（e114_search → E.tune_l）が同じデータで同じ結果を出す
- 再現の確かめ（木の形・最良の試行・分割ごとの NDCG@5・試行ごとの値）、実験114b の数との突き合わせ、判定
- 今の規則の形での比べ（rule_eval）が実験114b と同じ作りの表を作る
- main: 置き場の L と違えば同じデータで実験114 の探索を回し、同じなら続け、違えば止める。データの最終日で切る
- 実験114 の --retune からの呼び出し（後ろの引数を渡す）

  python3 tests/test_e114c_ltr_retune.py
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ltr  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e114b_ltr_rule as RB  # noqa: E402
import e114c_ltr_retune as RC  # noqa: E402
import test_e114b_ltr_rule as T114B  # noqa: E402

FEATS = ["x0", "x1", "x2", "x3"]


def synth(start="2018-04-02", end="2020-12-31", per_day=6, seed=0):
    """ltr と実験114 の両方の列を持つ合成データ（(Date, Code) の順。ラベルは全行にある）。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    df = pd.DataFrame([{"Code": f"{1000 + j}", "Date": d} for d in days for j in range(per_day)])
    n = len(df)
    for c in FEATS:
        df[c] = rng.normal(size=n)
    df[ltr.R20] = 0.04 * df["x0"] + rng.normal(0, 0.03, n)
    df.loc[df.index[::97], ltr.R20] = np.nan
    df[E.RET] = np.minimum(df[ltr.R20], 0.10)
    df["label"] = (df["x0"] + rng.normal(0, 1, n) > 0.8).astype(float)
    return df


def tune_rec(params, best=3, ndcg=0.61, folds=(0.6, 0.62), trials=None, minutes=0.1):
    return {"params": dict(params), "_cv": {"best_trial": best, "ndcg": ndcg, "ndcg_folds": list(folds),
                                            "minutes": minutes},
            "_trials": trials if trials is not None else [{"n": 0, "ndcg": 0.5}, {"n": 1, "ndcg": ndcg}]}


class TestWeeks(unittest.TestCase):
    def test_ends_and_cutoffs(self):
        d = pd.Series(pd.bdate_range("2020-01-01", "2026-09-24"))
        ends = RC.week_ends(d)
        self.assertEqual({k: str(v.date()) for k, v in ends.items()},
                         {0: "2026-09-24", 1: "2026-09-17", 2: "2026-09-10", 3: "2026-09-03", 4: "2026-08-27"})
        cuts = RC.week_cutoffs(d)
        self.assertEqual(cuts[0], pd.Timestamp("2025-08-26"))       # 実験114 と同じ打ち切り
        for k, end in ends.items():
            self.assertEqual(cuts[k], ltr.search_cutoff(d[d <= end]))
        self.assertTrue(all(cuts[k] < cuts[k - 1] for k in range(1, 5)))
        with self.assertRaises(SystemExit):
            RC.week_ends(pd.Series(pd.bdate_range("2026-09-01", "2026-09-24")), (4,))

    def test_until_keeps_order(self):
        df = synth(end="2018-06-29").sample(frac=1.0, random_state=1)
        cut = RC.until(df, pd.Timestamp("2018-05-31"))
        self.assertTrue((cut["Date"] <= "2018-05-31").all())
        self.assertEqual(len(cut), int((df["Date"] <= "2018-05-31").sum()))
        np.testing.assert_array_equal(cut["x0"].to_numpy(), df.loc[df["Date"] <= "2018-05-31", "x0"].to_numpy())
        self.assertIs(RC.until(df, None), df)
        self.assertEqual(RC.DATA_END, "2026-09-24")


class TestCache(unittest.TestCase):
    def test_tune_week_reads_cache(self):
        df = synth(end="2019-12-31")
        calls = []

        def fake(sub, cols, n_trials):
            calls.append((len(sub), n_trials))
            return tune_rec({"learning_rate": 0.1})

        cut = ltr.search_cutoff(df["Date"])
        with tempfile.TemporaryDirectory() as tmp:
            a = RC.tune_week(df, FEATS, 0, cut, 3, out_dir=tmp, tune_fn=fake)
            b = RC.tune_week(df, FEATS, 0, cut, 3, out_dir=tmp, tune_fn=fake)
            c = RC.tune_week(df, FEATS, 1, cut - pd.Timedelta(days=7), 3, out_dir=tmp, tune_fn=fake)
            names = sorted(os.listdir(tmp))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], (len(ltr.search_rows(df, cut)), 3))
        self.assertFalse(a["_cached"])
        self.assertTrue(b["_cached"])
        self.assertEqual((a["_week"], a["_cutoff"]), (0, str(cut.date())))
        self.assertEqual(b["params"], a["params"])
        self.assertEqual(c["_week"], 1)
        self.assertEqual(len(names), 2)
        self.assertTrue(names[0].startswith("e114c_tune_w0_") and names[1].startswith("e114c_tune_w1_"))

    def test_e114_search_rows_and_shape(self):
        df = synth(end="2019-12-31")
        seen = {}

        def fake(sub_l, cols, n_trials, folds):
            seen.update(rows=len(sub_l), last=sub_l["Date"].max(), folds=len(folds), n=n_trials)
            return {"params": {"learning_rate": 0.2}, "_cv": {"best_trial": 1, "ndcg": 0.6, "ndcg_folds": [0.6],
                                                              "trials": [{"n": 0, "ndcg": 0.6}]}}

        with tempfile.TemporaryDirectory() as tmp:
            er = RC.e114_search(df, FEATS, 4, out_dir=tmp, tune_fn=fake)
            again = RC.e114_search(df, FEATS, 4, out_dir=tmp, tune_fn=fake)
            self.assertTrue(os.listdir(tmp)[0].startswith("e114c_tune_e114_"))
        cut = ltr.search_cutoff(df["Date"])
        self.assertEqual(seen["rows"], len(ltr.search_rows(df, cut)))     # 本番の探索と同じ行
        self.assertLessEqual(seen["last"], cut)
        self.assertEqual((seen["folds"], seen["n"]), (E.N_SPLITS, 4))
        self.assertEqual(er["_trials"], [{"n": 0, "ndcg": 0.6}])
        self.assertNotIn("trials", er["_cv"])
        self.assertEqual((er["_cutoff"], er["_rows"]), (str(cut.date()), seen["rows"]))
        self.assertFalse(er["_cached"])
        self.assertTrue(again["_cached"])
        bad = df.copy()
        bad.loc[bad.index[5], "label"] = np.nan
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            RC.e114_search(bad, FEATS, 4, out_dir=tmp, tune_fn=fake)


class TestSameSearch(unittest.TestCase):
    """本番の探索（ltr.tune）と実験114 の探索そのもの（E.tune_l）が、同じデータで同じ結果を出す（P0 の作りの確かめ）。"""

    def test_production_equals_experiment(self):
        df = synth(end="2020-12-31")
        cut = RC.week_cutoffs(df["Date"], ())[0]
        with tempfile.TemporaryDirectory() as tmp:
            mine = RC.tune_week(df, FEATS, 0, cut, 3, out_dir=tmp)
            theirs = RC.e114_search(df, FEATS, 3, out_dir=tmp)
        same = RC.same_as_114(mine, theirs, theirs["_trials"])
        self.assertTrue(same["ok"], same)
        self.assertTrue(same["same_trials"])
        self.assertEqual(theirs["_cutoff"], mine["_cutoff"])


class TestChecks(unittest.TestCase):
    def test_same_as_114(self):
        p = {"learning_rate": 0.1, "num_leaves": 31}
        ref = tune_rec(p)
        ok = RC.same_as_114(tune_rec(p), ref, ref["_trials"])
        self.assertTrue(ok["ok"] and ok["same_trials"])
        d = RC.same_as_114(tune_rec({**p, "num_leaves": 30}), ref, ref["_trials"])
        self.assertEqual(d["diff_keys"], ["num_leaves"])
        self.assertFalse(d["ok"])
        self.assertFalse(RC.same_as_114(tune_rec(p, best=4), ref)["ok"])
        self.assertFalse(RC.same_as_114(tune_rec(p, folds=(0.6, 0.63)), ref)["ok"])
        t = RC.same_as_114(tune_rec(p, trials=[{"n": 0, "ndcg": 0.4}]), ref, ref["_trials"])
        self.assertFalse(t["ok"])
        self.assertFalse(t["same_trials"])
        n = RC.same_as_114(tune_rec(p), ref, None)
        self.assertTrue(n["ok"])
        self.assertIsNone(n["same_trials"])

    def test_decide(self):
        ok, ng = {"pass": True}, {"pass": False}
        self.assertEqual(RC.decide(False, {0: ok, 1: ok}), "stop")
        self.assertEqual(RC.decide(True, {0: ok, 1: ok, 2: ok, 3: ok, 4: ok}), "adopt")
        self.assertEqual(RC.decide(True, {0: ng, 1: ok, 2: ok, 3: ok, 4: ok}), "adopt")   # P0 は参考
        self.assertEqual(RC.decide(True, {0: ok, 1: ok, 2: ng, 3: ok, 4: ok}), "ask")
        self.assertEqual(RC.decide(True, {0: ok}), "ask")
        self.assertEqual(set(RC.DECISION_JA), {"stop", "adopt", "ask"})

    def test_check_114b(self):
        st = {k: {"n": 10, "ret_w": 0.012, "bad_w": 0.1} for k in RC.CHECK_KEYS}
        prev = {"shifts": {"0": {"stats": {k: dict(v) for k, v in st.items()}}}}
        self.assertIsNone(RC.check_114b(st, None, 0))
        self.assertIsNone(RC.check_114b(st, prev, 2))
        self.assertTrue(RC.check_114b(st, prev, 0))
        prev["shifts"]["0"]["stats"]["R0D"]["n"] = 11
        self.assertFalse(RC.check_114b(st, prev, 0))
        prev["shifts"]["0"]["stats"]["R0D"]["n"] = 10
        prev["shifts"]["0"]["stats"]["LN"]["ret_w"] = 0.013
        self.assertFalse(RC.check_114b(st, prev, 0))


class TestRuleEval(unittest.TestCase):
    def test_same_table_as_114b(self):
        w, folds = T114B.synth_wide()
        base = w[["Code", "Date", "window"]].rename(columns={"window": "fold"})
        trees = {a: base.assign(score=w[f"score_{a}"]) for a in RB.TREES}
        df = w.drop(columns=[f"score_{a}" for a in RB.MODELS4] + ["window"])
        lo = base.assign(score=w[f"score_{RB.LTR}"], **{E.R20: w[E.R20]})
        got = []

        def fake_oof(d, cols, arm, params, shift, seeds):
            got.append((arm, shift, tuple(seeds)))
            return lo

        with mock.patch.object(RC.E41, "folds_for", lambda dates, shift: folds):
            res = RC.rule_eval(df, FEATS, trees, {"learning_rate": 0.1}, 0, oof_fn=fake_oof)
        self.assertEqual(got, [("L", 0, (RB.L_SEED,))])
        ft = RB.forward_frame(RB.wide({**trees, RB.LTR: lo}, df), folds)
        masks = RB.rule_masks(ft)
        for k in RC.CHECK_KEYS:
            self.assertEqual(res["stats"][k]["n"], int(masks[k].sum()))
        self.assertEqual(len(res["picks"]), int(masks["R0D"].sum()))
        self.assertEqual(res["stats"]["R0D"]["diff"], RB.window_diff(ft, masks["R0"], masks["R0D"]))
        self.assertTrue(np.isfinite(res["ndcg_oof"]))
        self.assertEqual(res["rows"], len(ft))
        v = RC.verdict_row({0: res["stats"], 2: res["stats"]})
        self.assertEqual([r["shift"] for r in v["rows"]], [0, 2])


class TestMain(unittest.TestCase):
    """main の流れ: 置き場の L と違えば同じデータで実験114 の探索を回し、同じなら続け、違えば止める。"""

    P = {"learning_rate": 0.05, "num_leaves": 31}

    def run_main(self, p0, e114, argv=()):
        df = synth(start="2025-01-01", end="2026-10-30", per_day=2)
        seen = {"tune": [], "e114": 0}
        ref = {"params": dict(self.P), "_cv": tune_rec(self.P)["_cv"], "_cutoff": "2025-08-26"}

        def fake_tune(ds, cols, k, cutoff, n_trials, out_dir=None, tune_fn=None):
            seen["tune"].append((k, ds["Date"].max(), cutoff))
            rec = tune_rec(p0 if k == 0 else {**self.P, "num_leaves": 20 + k}, minutes=0.1)
            rec["_cutoff"] = str(cutoff.date())
            return rec

        def fake_e114(full, cols, n_trials, out_dir=None, tune_fn=None):
            seen["e114"] += 1
            seen["e114_last"] = full["Date"].max()
            return {**tune_rec(e114), "_cutoff": "2025-08-26", "_rows": 10}

        def fake_eval(d, cols, trees, params, shift, seed=42, oof_fn=None):
            seen.setdefault("eval_last", d["Date"].max())
            s0 = {"n": 10, "ret_w": 0.01, "bad_w": 0.12}
            sd = {"n": 7, "ret_w": 0.02, "bad_w": 0.10, "diff": {"mean": 0.01, "se": 0.005, "won": 3, "n": 4},
                  "diff_bad": {"mean": -0.02, "se": 0.01, "won": 1, "n": 4}}
            return {"stats": {"R0": s0, "R0D": sd, "LN": dict(s0)}, "picks": {("1000", "2025-03-03")},
                    "ndcg_oof": 0.6, "windows": 4, "rows": 100}

        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "e114_params.json")
            with open(store, "w", encoding="utf-8") as fh:
                json.dump({"L": {"params": dict(self.P), "_cv": {"trials": tune_rec(self.P)["_trials"]}}}, fh)
            patches = [
                mock.patch.object(RC.F, "columns", lambda preset: list(FEATS)),
                mock.patch.object(RC.ltr, "load_params", lambda *a, **k: ref),
                mock.patch.object(RC.ltr, "params_mismatch", lambda rec, cols: None),
                mock.patch.object(RC.ltr, "load_frame", lambda *a, **k: df.copy()),
                mock.patch.object(RC.E, "PARAMS_PATH", store),
                mock.patch.object(RC.E, "load_frame", lambda cols: (df.copy(), None, None)),
                mock.patch.object(RC, "tune_week", fake_tune),
                mock.patch.object(RC, "e114_search", fake_e114),
                mock.patch.object(RC, "rule_eval", fake_eval),
                mock.patch.object(RC.B, "prod_params", lambda cols: {a: {} for a in RC.B.TREES}),
                mock.patch.object(RC.B, "oof_tree", lambda *a, **k: None),
                mock.patch.object(RC.B, "SUMMARY_PATH", os.path.join(tmp, "none.json")),
                mock.patch.object(RC, "SUMMARY_PATH", os.path.join(tmp, "e114c_summary.json")),
                mock.patch.object(RC, "OOF_DIR", tmp),
            ]
            for p in patches:
                p.start()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = RC.main(["--weeks", "1,2", "--data-end", "2026-09-24", *argv])
            finally:
                for p in reversed(patches):
                    p.stop()
            with open(os.path.join(tmp, "e114c_summary.json"), encoding="utf-8") as fh:
                summary = json.load(fh)
        return code, summary, seen

    def test_same_as_store_goes_on(self):
        code, summary, seen = self.run_main(self.P, self.P)
        self.assertEqual(code, 0)
        self.assertEqual(seen["e114"], 0)                                  # 置き場の L と同じなら回さない
        self.assertTrue(summary["repro_ok"])
        self.assertEqual(summary["decision"], "adopt")
        self.assertEqual([k for k, _, _ in seen["tune"]], [0, 1, 2])
        self.assertTrue(all(last <= pd.Timestamp("2026-09-24") for _, last, _ in seen["tune"]))
        self.assertEqual(seen["tune"][0][2], pd.Timestamp("2025-08-26"))
        self.assertLessEqual(seen["eval_last"], pd.Timestamp("2026-09-24"))
        self.assertEqual(summary["data_end"], "2026-09-24")

    def test_differs_from_store_but_same_as_experiment(self):
        other = {**self.P, "num_leaves": 40}
        code, summary, seen = self.run_main(other, other)
        self.assertEqual(code, 0)
        self.assertEqual(seen["e114"], 1)
        self.assertLessEqual(seen["e114_last"], pd.Timestamp("2026-09-24"))
        self.assertFalse(summary["repro"]["ok"])
        self.assertTrue(summary["repro_same_data"]["ok"])
        self.assertTrue(summary["repro_ok"])
        self.assertIn(summary["decision"], ("adopt", "ask"))

    def test_differs_from_both_stops(self):
        code, summary, seen = self.run_main({**self.P, "num_leaves": 40}, self.P)
        self.assertEqual(code, 1)
        self.assertEqual(summary["decision"], "stop")
        self.assertFalse(summary["repro_ok"])
        self.assertEqual([k for k, _, _ in seen["tune"]], [0])            # P1 から先は探索しない
        self.assertNotIn("eval_last", seen)

    def test_budget_stops_before_compare(self):
        code, summary, seen = self.run_main(self.P, self.P, ["--budget-min", "-1"])
        self.assertEqual(code, 0)
        self.assertTrue(summary["unfinished"])
        self.assertNotIn("decision", summary)
        self.assertEqual(seen["tune"], [])


class TestEntry(unittest.TestCase):
    def test_retune_from_114(self):
        calls = []
        with mock.patch.object(RC, "main", lambda argv=None: calls.append(list(argv)) or 3):
            self.assertEqual(E.main(["--retune"]), 3)
            self.assertEqual(E.main(["--retune", "--weeks", "1,2", "--data-end", "all"]), 3)
        self.assertEqual(calls, [[], ["--weeks", "1,2", "--data-end", "all"]])
        with self.assertRaises(SystemExit):
            E.main(["--weeks", "1"])                                        # --retune 無しの知らない引数は止める


if __name__ == "__main__":
    unittest.main()
