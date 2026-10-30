#!/usr/bin/env python3
"""
ランキング学習を本番の脇に置く部分（research/ltr.py・research/train_ltr.py）。合成データで:

- その日の中の百分位と段階が実験52 の within_date_pct / within_date_grade と同じ（同点・1件だけの日も）
- 当てはめが実験114 の fit_score（L）と同じスコアを出す。out-of-fold が実験114 の oof_arm（L・ずらし0・種42）と
  同じ行・同じ窓・同じスコア（本番の窓 = 実験114 のずらし0の窓）
- 20営業日目の終値の収益が live_track.forward の ret_close と同じ（手で計算した例）
- 木の形の確かめ（列の指紋が違えば学習しない）・LightGBM に渡す鍵
- 毎日の候補の値（その日の中の百分位・順位・上半分の印・過去の分布での位置）
- 学習の台本が置き場に model.joblib / meta.json / oof.parquet を作り、読み直せる
- 日曜の再学習に木の形の探索（Tune ranking model）が学習の前に入り、ジョブの上限に近ければ飛ばし、木の形をコミットする

  python3 tests/test_ltr.py
"""
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

import features as F  # noqa: E402
import ltr  # noqa: E402
import models as M  # noqa: E402
import tuning  # noqa: E402
import e52_return_objective as E52  # noqa: E402
import e114_ltr_retest as E  # noqa: E402


def tuned_params_problems(rec: dict) -> list:
    """
    週次の探索（tune_ltr.py・tune_ltr5.py）が書いた木の形として正しいかを見る（合わないところの一覧。空なら正しい）。値そのものは
    日曜の探索で毎週変わるので見ない。固定の値・探索空間（ltr.search_space）の範囲・探索の記録の形だけ。
    """
    p, cv = rec.get("params", {}), rec.get("_cv", {})
    fixed = {"boosting_type": "gbdt", "n_estimators": tuning.SEARCH_N_ESTIMATORS, "subsample_freq": 1, "n_jobs": -1,
             "verbose": -1}
    space = {"learning_rate": tuning.lr_range(tuning.SEARCH_N_ESTIMATORS), "num_leaves": (7, 127),
             "min_child_samples": (10, 300), "subsample": (0.5, 1.0), "colsample_bytree": (0.3, 1.0),
             "reg_alpha": (1e-8, 10.0), "reg_lambda": (1e-8, 10.0)}
    bad = []
    if sorted(rec) != ["_cutoff", "_cv", "_features_sig", "_n_features", "_source", "params"]:
        bad.append(f"鍵 {sorted(rec)}")
    if sorted(p) != sorted([*fixed, *space]):
        bad.append(f"木の形の鍵 {sorted(p)}")
    bad += [f"{k} = {p.get(k)!r}" for k, v in fixed.items() if p.get(k) != v]
    bad += [f"{k} = {p.get(k)!r} が {lo}〜{hi} の外" for k, (lo, hi) in space.items()
            if not isinstance(p.get(k), (int, float)) or not lo <= p[k] <= hi]
    bad += [f"{k} = {p.get(k)!r} が整数でない" for k in ("num_leaves", "min_child_samples") if not isinstance(p.get(k), int)]
    if (cv.get("n_trials"), cv.get("n_splits")) != (ltr.TUNE_TRIALS, ltr.TUNE_SPLITS):
        bad.append(f"試行・分割 {cv.get('n_trials')} / {cv.get('n_splits')}")
    if not 0 < cv.get("ndcg", 0) < 1:
        bad.append(f"NDCG@5 {cv.get('ndcg')!r}")
    return bad


FEATS = ["x0", "x1", "x2", "x3"]
SMALL = {"boosting_type": "gbdt", "n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7,
         "min_child_samples": 10, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.9,
         "reg_alpha": 0.01, "reg_lambda": 0.01, "n_jobs": 1, "verbose": -1}


def synth(start="2018-04-02", end="2023-06-30", per_day=3, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    df = pd.DataFrame([{"Code": f"{1000 + j}", "Date": d} for d in days for j in range(per_day)])
    n = len(df)
    for c in FEATS:
        df[c] = rng.normal(size=n)
    df[ltr.R20] = 0.04 * df["x0"] + rng.normal(0, 0.03, n)
    df.loc[df.index[::97], ltr.R20] = np.nan
    df[E.R20] = df[ltr.R20]
    df[E.RET] = np.minimum(df[ltr.R20], 0.10)
    df[E.HIT] = df[E.RET] >= 0.10
    df["label"] = (df["x0"] + rng.normal(0, 1, n) > 0.8).astype(float)
    df[E.OUTCOME] = df[E.RET]
    df["vol_20d"] = 1.5
    return df


class TestGrades(unittest.TestCase):
    def test_same_as_experiment(self):
        rng = np.random.default_rng(1)
        dates = np.repeat(pd.bdate_range("2024-01-01", periods=30), rng.integers(1, 9, 30))
        r = np.round(rng.normal(size=len(dates)), 1)          # 丸めて同点を作る
        np.testing.assert_array_equal(ltr.within_date_pct(pd.Series(dates), r),
                                      E52.within_date_pct(pd.Series(dates), r))
        np.testing.assert_array_equal(ltr.grades(dates, r), E.grades(dates, r))
        one = ltr.within_date_pct(pd.Series(pd.to_datetime(["2024-01-01"])), [0.3])
        self.assertEqual(float(one[0]), 0.5)


class TestFit(unittest.TestCase):
    def test_same_scores_as_experiment_fit(self):
        df = synth(end="2019-12-31")
        tr = df[df[ltr.R20].notna() & df["label"].notna() & (df["Date"] < "2019-07-01")]
        te = df[df[ltr.R20].notna() & (df["Date"] >= "2019-07-01")]
        m = ltr.fit(tr, FEATS, SMALL)
        mine = ltr.predict(m, te[FEATS].to_numpy(dtype=float))
        theirs = E.fit_score("L", SMALL, tr, te, FEATS, ltr.SEED)
        np.testing.assert_array_equal(mine, theirs)
        # 行の順に左右されない（中で (Date, Code) に並べ直す）
        m2 = ltr.fit(tr.sample(frac=1.0, random_state=3), FEATS, SMALL)
        np.testing.assert_array_equal(ltr.predict(m2, te[FEATS].to_numpy(dtype=float)), mine)

    def test_oof_same_as_experiment_shift0(self):
        df = synth()
        with tempfile.TemporaryDirectory() as tmp:
            theirs = E.oof_arm(df, FEATS, "L", SMALL, 0, (ltr.SEED,), out_dir=tmp)
        mine = ltr.oof_scores(df, FEATS, SMALL)
        a = mine.sort_values(["Date", "Code"]).reset_index(drop=True)
        b = theirs.sort_values(["Date", "Code"]).reset_index(drop=True)
        np.testing.assert_array_equal(a[["Code", "Date", "fold"]].astype(str).to_numpy(),
                                      b[["Code", "Date", "fold"]].astype(str).to_numpy())
        np.testing.assert_array_equal(a["score"].to_numpy(), b["score"].to_numpy())
        self.assertTrue(a[ltr.R20].notna().all())

    def test_ranker_params(self):
        p = ltr.ranker_params({**SMALL, "objective": "binary", "scale_pos_weight": 3.0, "random_state": 0,
                               "_note": "x"}, seed=7)
        self.assertEqual(p["random_state"], 7)
        for k in ("objective", "scale_pos_weight", "_note"):
            self.assertNotIn(k, p)
        self.assertEqual(p["n_estimators"], 30)


class TestParams(unittest.TestCase):
    def test_mismatch(self):
        cols = ["a", "b", "c"]
        good = {"params": {"learning_rate": 0.1}, "_features_sig": F.signature(cols), "_n_features": 3}
        self.assertIsNone(ltr.params_mismatch(good, cols))
        self.assertIsNotNone(ltr.params_mismatch({**good, "_features_sig": "other"}, cols))
        self.assertIsNotNone(ltr.params_mismatch({}, cols))

    def test_committed_params_file(self):
        # 日曜の「Tune ranking model」が毎週書き換えるので、値（最初は実験114 の結果）は見ない。学習する列で探索した、
        # 探索空間の中の木の形であることだけ
        rec = ltr.load_params()
        cols = F.columns(F.DEFAULT_PRESET)
        self.assertIsNone(ltr.params_mismatch(rec, cols))
        self.assertEqual(tuned_params_problems(rec), [])

    def test_tuned_params_problems(self):
        rec = {"_source": "s", "_features_sig": "x", "_n_features": 4, "_cutoff": "2025-09-01",
               "_cv": {"n_trials": 50, "n_splits": 5, "ndcg": 0.58},
               "params": {**SMALL, "n_estimators": 200, "n_jobs": -1}}
        self.assertEqual(tuned_params_problems(rec), [])
        for bad in ({"num_leaves": 128}, {"learning_rate": 0.5}, {"n_estimators": 300}, {"min_child_samples": 10.5}):
            self.assertEqual(len(tuned_params_problems({**rec, "params": {**rec["params"], **bad}})), 1, bad)
        self.assertEqual(len(tuned_params_problems({**rec, "_cv": {**rec["_cv"], "n_trials": 3}})), 1)
        self.assertEqual(len(tuned_params_problems({**rec, "_trials": []})), 1)


class TestR20(unittest.TestCase):
    def test_hand_computed(self):
        days = pd.bdate_range("2024-01-01", periods=30)
        bars = pd.DataFrame({"Code": "1001", "Date": days, "AdjO": 100.0, "AdjH": 101.0, "AdjL": 99.0,
                             "AdjC": np.linspace(100, 129, 30)})
        ds = pd.DataFrame({"Code": ["1001", "1001"], "Date": [days[0], days[20]], "label": [1.0, 0.0]})
        out = ltr.attach_r20(ds, bars)
        # 選定日 days[0]: 買値は days[1] の寄り 100、20営業日目（days[20]）の終値 100 + 20 = 120 → +20%
        self.assertAlmostEqual(out[ltr.R20].iloc[0], 0.20)
        # 選定日 days[20]: 20営業日目（days[40]）が無いので欠測
        self.assertTrue(np.isnan(out[ltr.R20].iloc[1]))


class TestDayFields(unittest.TestCase):
    def test_fields(self):
        d = pd.to_datetime(["2024-03-01"] * 4 + ["2024-03-04"])
        s = np.array([0.5, -1.0, 2.0, 0.0, 0.3])
        hist = np.array([-2.0, -1.0, 0.0, 1.0])
        f = ltr.day_fields(d, s, hist)
        self.assertEqual([x["dayRank"] for x in f], [2, 4, 1, 3, 1])
        self.assertEqual([x["dayPct"] for x in f], [66.7, 0.0, 100.0, 33.3, 50.0])
        self.assertEqual([x["upper"] for x in f], [True, False, True, False, True])
        self.assertEqual([x["nDay"] for x in f], [4, 4, 4, 4, 1])
        self.assertEqual(f[0]["pctHistorical"], M.pct_historical(0.5, hist))
        self.assertIsNone(ltr.day_fields(d[:1], s[:1], None)[0]["pctHistorical"])


class TestTune(unittest.TestCase):
    """木の形の探索（ltr.tune）が実験114 の L の探索（tune_l）と同じ（運用者の選択「毎週探索し直す」§78-10）。"""

    def test_settings_same_as_experiment(self):
        import optuna
        import e53_objective_tuning as E53

        self.assertEqual((ltr.TUNE_SPLITS, ltr.TUNE_SEED, ltr.NDCG_K, ltr.MIN_DAY),
                         (E.N_SPLITS, E.SEED, E.NDCG_K, E.MIN_DAY))
        self.assertEqual(ltr.TUNE_TRIALS, 50)
        fixed = {"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 40, "subsample": 0.7,
                 "colsample_bytree": 0.5, "reg_alpha": 1e-3, "reg_lambda": 0.2}
        self.assertEqual(ltr.search_space(optuna.trial.FixedTrial(fixed)),
                         E53.search_params(optuna.trial.FixedTrial(fixed)))

    def test_same_search_as_experiment(self):
        import tuning
        from train_model import EMBARGO_DAYS

        # NDCG@5 は候補5件以上の日だけで測るので、1日6件にする
        df = synth(end="2020-12-31", per_day=6).sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
        sub = ltr.search_rows(df, df["Date"].max())
        self.assertTrue(sub[ltr.R20].notna().all() and sub["label"].notna().all())
        self.assertEqual(len(sub), int(ltr.trainable(df).sum()))
        # 実験114 の main と同じ分割を作って tune_l に渡す。TPE が値を見て引き始める11試行目より先まで比べる
        folds = tuning.cv_folds(sub, tuning.PRODUCTION_CV, n_splits=E.N_SPLITS, seed=E.SEED, embargo_days=EMBARGO_DAYS)
        theirs = E.tune_l(sub, FEATS, 12, folds)
        mine = ltr.tune(sub, FEATS, n_trials=12)
        self.assertEqual(mine["params"], theirs["params"])
        self.assertEqual(mine["_trials"], theirs["_cv"]["trials"])
        for k in ("ndcg", "ndcg_folds", "best_trial", "n_splits", "n_trials", "objective"):
            self.assertEqual(mine["_cv"][k], theirs["_cv"][k], k)
        self.assertEqual(mine["_cv"]["n_rows"], len(sub))
        self.assertEqual(mine["_cv"]["train_to"], str(sub["Date"].max().date()))
        self.assertEqual(sorted(mine["params"]), sorted(ltr.load_params()["params"]))

    def test_cutoff_same_as_run_tuning(self):
        from train_model import EMBARGO_DAYS, HOLDOUT_MONTHS, holdout_bounds

        d = pd.Series(pd.bdate_range("2020-01-01", "2026-09-24"))
        self.assertEqual(ltr.search_cutoff(d), pd.Timestamp(holdout_bounds(d, HOLDOUT_MONTHS, EMBARGO_DAYS)[0]))
        # 実験114 の探索の打ち切り（データの最終日 2026-09-24 → 2025-08-26）
        self.assertEqual(ltr.search_cutoff(d), pd.Timestamp("2025-08-26"))


class TestTrainScript(unittest.TestCase):
    def test_writes_and_loads(self):
        import train_ltr

        df = synth()
        cols = FEATS
        days = pd.bdate_range("2018-03-01", "2023-08-31")
        bars = []
        rng = np.random.default_rng(5)
        for j in range(3):
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
            bars.append(pd.DataFrame({"Code": f"{1000 + j}", "Date": days, "AdjO": c, "AdjH": c * 1.01,
                                      "AdjL": c * 0.99, "AdjC": c}))
        bars = pd.concat(bars, ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir)
            bars.to_parquet(os.path.join(data_dir, "bars_all.parquet"), index=False)
            ds_path = os.path.join(data_dir, "dataset.parquet")
            df.drop(columns=[ltr.R20, E.R20, E.RET, E.HIT]).to_parquet(ds_path, index=False)
            pp = os.path.join(tmp, "p.json")
            with open(pp, "w", encoding="utf-8") as fh:
                json.dump({"params": SMALL, "_features_sig": F.signature(cols), "_n_features": len(cols),
                           "_source": "テスト"}, fh)
            orig_cols, orig_load = F.columns, ltr.LT.load_bars
            try:
                F.columns = lambda preset=None: list(cols)
                ltr.LT.load_bars = lambda data_dir, start=None, extra=(): bars
                rc = train_ltr.main(["--data-dir", data_dir, "--dataset", ds_path, "--out-dir", tmp,
                                     "--params", pp])
                self.assertEqual(rc, 0)
                model, meta = ltr.load(tmp)
                self.assertIsNotNone(model)
                self.assertEqual(meta["features"], cols)
                self.assertEqual(meta["algo"], "ltr")
                oof = pd.read_parquet(os.path.join(M.sub_dir("ltr", tmp), "oof.parquet"))
                self.assertEqual(list(oof.columns), ["Code", "Date", "score", "fold"])
                self.assertEqual(meta["nOof"], len(oof))
                # 列の指紋が違う木の形なら学習しない
                with open(pp, "w", encoding="utf-8") as fh:
                    json.dump({"params": SMALL, "_features_sig": "other", "_n_features": 360}, fh)
                self.assertEqual(train_ltr.main(["--data-dir", data_dir, "--dataset", ds_path, "--out-dir", tmp,
                                                 "--params", pp]), 1)
            finally:
                F.columns, ltr.LT.load_bars = orig_cols, orig_load
        self.assertEqual(ltr.load(os.path.join(tmp, "none")), (None, None))

    def test_tune_then_train(self):
        """tune_ltr.py が ltr_params.json と同じ形で書き、train_ltr.py がその木の形で学習する。"""
        import train_ltr
        import tune_ltr

        df = synth(per_day=6)
        cols = FEATS
        days = pd.bdate_range("2018-03-01", "2023-08-31")
        rng = np.random.default_rng(6)
        bars = []
        for j in range(6):
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
            bars.append(pd.DataFrame({"Code": f"{1000 + j}", "Date": days, "AdjO": c, "AdjH": c * 1.01,
                                      "AdjL": c * 0.99, "AdjC": c}))
        bars = pd.concat(bars, ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir)
            ds_path = os.path.join(data_dir, "dataset.parquet")
            df.drop(columns=[ltr.R20, E.R20, E.RET, E.HIT]).to_parquet(ds_path, index=False)
            pp = os.path.join(tmp, "ltr_params.json")
            orig_cols, orig_load = F.columns, ltr.LT.load_bars
            try:
                F.columns = lambda preset=None: list(cols)
                ltr.LT.load_bars = lambda data_dir, start=None, extra=(): bars
                self.assertEqual(tune_ltr.main(["--data-dir", data_dir, "--dataset", ds_path, "--out", pp,
                                                "--trials", "2"]), 0)
                with open(pp, encoding="utf-8") as fh:
                    rec = json.load(fh)
                self.assertEqual(sorted(rec), ["_cutoff", "_cv", "_features_sig", "_n_features", "_source", "params"])
                self.assertEqual(sorted(rec["params"]), sorted(ltr.load_params()["params"]))
                self.assertIsNone(ltr.params_mismatch(rec, cols))
                ds = ltr.load_frame(ds_path, data_dir, cols)
                self.assertEqual(rec["_cutoff"], str(ltr.search_cutoff(ds["Date"]).date()))
                self.assertEqual(rec["_cv"]["n_rows"], len(ltr.search_rows(ds, ltr.search_cutoff(ds["Date"]))))
                self.assertEqual(rec["_cv"]["n_trials"], 2)
                self.assertTrue(rec["_source"].startswith("週次の探索（"))
                self.assertFalse(os.path.exists(pp + ".tmp"))
                self.assertEqual(train_ltr.main(["--data-dir", data_dir, "--dataset", ds_path, "--out-dir", tmp,
                                                 "--params", pp]), 0)
                _, meta = ltr.load(tmp)
                self.assertEqual(meta["params"], rec["params"])
                self.assertEqual(meta["tuning"], rec["_cv"])
            finally:
                F.columns, ltr.LT.load_bars = orig_cols, orig_load

    def test_source_note(self):
        import tune_ltr

        cv = {"n_rows": 18038, "n_trials": 50, "n_splits": 5}
        env = {"GITHUB_RUN_ID": "123", "GITHUB_WORKFLOW": "Retrain Weekly", "GITHUB_SHA": "abcdef0123"}
        self.assertEqual(tune_ltr.source_note(cv, "2025-08-26", env),
                         "週次の探索（Retrain Weekly run 123・abcdef0。2025-08-26 までの 18,038件・50試行・5分割・"
                         "目的は検証分割の日付内 NDCG@5）")
        self.assertIn("手元", tune_ltr.source_note(cv, "2025-08-26", {}))


class TestWeeklyWorkflow(unittest.TestCase):
    """日曜の再学習（retrain-weekly.yml）にランキング学習の探索が入っている（運用者の指示 2026-10-29。§78-10）。"""

    def setUp(self):
        import yaml

        with open(os.path.join(ROOT, ".github", "workflows", "retrain-weekly.yml"), encoding="utf-8") as fh:
            self.wf = yaml.safe_load(fh)
        self.steps = self.wf["jobs"]["retrain"]["steps"]
        self.names = [s.get("name") for s in self.steps]

    def step(self, name):
        return self.steps[self.names.index(name)]

    def test_tune_before_train(self):
        self.assertLess(self.names.index("Tune extra models (opt-in)"), self.names.index("Tune ranking model"))
        self.assertLess(self.names.index("Tune ranking model"), self.names.index("Train ranking model"))
        st = self.step("Tune ranking model")
        self.assertEqual(st["if"], "inputs.tune == 'yes' && steps.scope.outputs.ltr == 'yes'")
        self.assertIs(st["continue-on-error"], True)
        self.assertEqual(st["timeout-minutes"], 45)
        self.assertIn("python3 research/tune_ltr.py", st["run"])
        # ジョブの上限（330分）までの残りを、ジョブの最初のステップで控えた時刻から測る
        self.assertEqual(self.wf["jobs"]["retrain"]["timeout-minutes"], 330)
        self.assertIn("330 * 60", st["run"])
        self.assertEqual(self.names[0], "Start time")
        self.assertEqual(self.step("Start time")["id"], "start")
        self.assertEqual(st["env"]["JOB_START"], "${{ steps.start.outputs.t }}")

    def test_guard_skips_near_the_limit(self):
        import subprocess
        import time

        run = self.step("Tune ranking model")["run"].replace("python3 research/tune_ltr.py", "echo RUN")
        with tempfile.TemporaryDirectory() as tmp:
            for ago_min, want, not_want in ((300, "探索しない", "RUN"), (60, "RUN", "探索しない")):
                env = dict(os.environ, JOB_START=str(int(time.time()) - ago_min * 60))
                out = subprocess.run(["bash", "-eo", "pipefail", "-c", run], cwd=tmp, env=env, capture_output=True,
                                     text=True)
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertIn(want, out.stdout)
                self.assertNotIn(not_want, out.stdout)

    def test_commit_includes_params(self):
        run = self.step("Commit model metadata")["run"]
        self.assertIn("research/ltr_params.json", run)
        self.assertIn("git diff -- research/ltr_params.json", run)


if __name__ == "__main__":
    unittest.main()
