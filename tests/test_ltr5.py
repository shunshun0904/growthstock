#!/usr/bin/env python3
"""
直近5営業日の組のランキング学習（research/ltr5.py・train_ltr5.py・tune_ltr5.py）。合成データで:

- 組の行（判断日・候補の日・値動き4列）が実験117 の rolling_frame と同じ。判断日を絞っても、全部作ってから絞ったものと同じ
- 組の収益（R20・今の売り方の収益・到達の印）が実験115 の attach_week_exit と同じ
- 探索の分割が実験117 の search_folds と同じ。探索（少ない試行）が実験114 の tune_l（実験117 の W5 の探索）と同じ木の形を選ぶ
- 組の中の順位が実験117 の w5_rank と同じ（同じスコアは銘柄コードの順）
- 値動きの列の確かめ・判断日に入った候補の R20 の確かめが合う（ずらすと合わない）
- 3条件・4条件の印（どれかが決まらなければ決めない）
- 木の形の初期値が実験117 のログの全桁と同じで、学習する列（360列＋値動き4列）で探索したものとして通る
- 探索の台本が ltr5_params.json と同じ形で書き、学習の台本がその木の形で model.joblib / meta.json / oof.parquet を作る
- 毎日の予測（predict_daily.attach_ltr5）: 予測の日の候補だけに組の中の順位・組の行の数を付け、組を作り直して採点した順位と
  同じ。3条件・4条件の印（日次の1位・5日の1位・1.5σ / 1.2σ の木3本）。モデルが無い・候補が5営業日に足りないときは何も足さない
- 日曜の再学習（retrain-weekly.yml）: 「Tune 5-day ranking model」→「Train 5-day ranking model」がランキング学習の後に入り、
  ジョブの上限に近ければ探索を飛ばし、only に ltr5 と書けばこれだけ、木の形をコミットする

  python3 tests/test_ltr5.py
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
import ltr5  # noqa: E402
import models as M  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e115_weekly_ltr as W  # noqa: E402
import e117_rolling5_ltr as R  # noqa: E402

FEATS = ["x0", "x1", "x2"]
SMALL = {"boosting_type": "gbdt", "n_estimators": 20, "learning_rate": 0.1, "num_leaves": 7,
         "min_child_samples": 10, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.9,
         "reg_alpha": 0.01, "reg_lambda": 0.01, "n_jobs": 1, "verbose": -1}


def market(n_codes=8, start="2018-03-01", end="2023-09-29", seed=0):
    """日足（AdjO・AdjH・AdjL・AdjC）。銘柄ごとに欠けた日も作る。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    out = []
    for j in range(n_codes):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(days))))
        o = c * np.exp(rng.normal(0, 0.004, len(days)))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, len(days))))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, len(days))))
        b = pd.DataFrame({"Code": f"{1000 + j}", "Date": days, "AdjO": o, "AdjH": h, "AdjL": lo, "AdjC": c})
        if j == 3:
            b = b.drop(b.index[200:204])                       # 売買の無い日
        out.append(b)
    return pd.concat(out, ignore_index=True), days


def candidates(days, n_codes=8, rate=0.25, seed=1, start="2018-04-02", end="2023-06-30"):
    """候補の行（日次）。同じ銘柄が続けて候補になる日もある（組の中で最後の候補の日だけ残るかを見る）。"""
    rng = np.random.default_rng(seed)
    ds = [d for d in days if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    rows = [{"Code": f"{1000 + j}", "Date": d} for d in ds for j in range(n_codes) if rng.random() < rate]
    df = pd.DataFrame(rows)
    for c in FEATS:
        df[c] = rng.normal(size=len(df))
    df["label"] = (df["x0"] + rng.normal(0, 1, len(df)) > 0.8).astype(float)
    return df


class TestPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars, cls.days = market()
        cls.cand = candidates(cls.days)
        cls.cal = ltr5.calendar(cls.bars["Date"].unique())

    def experiment_pool(self, k=ltr5.K):
        full = self.cand.copy()
        for c in (R.RET, R.HIT, R.R20):
            full[c] = 0.0
        rl, _ = R.rolling_frame(full, self.bars, self.cal, FEATS, k=k)
        return rl

    def test_same_as_experiment(self):
        mine = ltr5.pool_frame(self.cand, self.bars, self.cal, FEATS, keep=("label",))
        theirs = self.experiment_pool()
        cols = ["Code", "Date", "cand_date", "r5_days", "label"] + FEATS + list(ltr5.MOVE_COLS)
        pd.testing.assert_frame_equal(mine[cols].reset_index(drop=True), theirs[cols].reset_index(drop=True))
        self.assertTrue((mine["r5_days"] > 0).any())
        # 同じ銘柄が組に2回入る判断日があり、最後の候補の日だけ残っている
        twice = self.cand.assign(_t=1).groupby("Code")["Date"].apply(lambda s: s.diff().dt.days.min())
        self.assertTrue((twice <= 4).any())

    def test_decision_days_subset(self):
        full = ltr5.pool_frame(self.cand, self.bars, self.cal, FEATS)
        pick = [full["Date"].iloc[0], full["Date"].iloc[len(full) // 2], full["Date"].iloc[-1]]
        sub = ltr5.pool_frame(self.cand, self.bars, self.cal, FEATS, decision_days=pick)
        want = full[full["Date"].isin(pick)].reset_index(drop=True)
        pd.testing.assert_frame_equal(sub.reset_index(drop=True), want)
        empty = ltr5.pool_frame(self.cand.iloc[:0], self.bars, self.cal, FEATS)
        self.assertEqual(len(empty), 0)
        for c in ltr5.MOVE_COLS:
            self.assertIn(c, empty.columns)
        with self.assertRaises(SystemExit):
            ltr5.pool_frame(self.cand, self.bars, self.cal, FEATS, decision_days=[pd.Timestamp("2019-01-05")])

    def test_rejects_clash_and_off_calendar(self):
        bad = self.cand.assign(r5_ret=0.0)
        with self.assertRaises(SystemExit):
            ltr5.pool_frame(bad, self.bars, self.cal, FEATS)
        off = self.cand.copy()
        off.loc[off.index[0], "Date"] = pd.Timestamp("2019-01-05")          # 土曜
        with self.assertRaises(SystemExit):
            ltr5.pool_frame(off, self.bars, self.cal, FEATS)

    def test_exit_same_as_experiment(self):
        pool = ltr5.pool_frame(self.cand, self.bars, self.cal, FEATS)
        mine = ltr5.attach_exit(pool, self.bars)
        theirs, _ = W.attach_week_exit(pool, self.bars, self.cal)
        np.testing.assert_array_equal(mine[ltr5.R20].to_numpy(), theirs[W.R20].to_numpy())
        np.testing.assert_array_equal(mine[ltr5.RET].to_numpy(), theirs[W.RET].to_numpy())
        np.testing.assert_array_equal(mine[ltr5.HIT].to_numpy(), theirs[W.HIT].to_numpy())
        self.assertTrue(mine[ltr5.R20].notna().mean() > 0.9)

    def test_checks(self):
        pool = ltr5.pool_frame(self.cand, self.bars, self.cal, FEATS)
        cm = ltr5.check_move_cols(pool, self.bars, self.cal, n=200)
        self.assertEqual(cm["same"], cm["rows"])
        self.assertEqual(cm["rows"], 200)
        bad = pool.copy()
        bad["r5_ret"] = bad["r5_ret"] + 1e-6
        self.assertLess(ltr5.check_move_cols(bad, self.bars, self.cal, n=200)["same"], 200)
        pool = ltr5.attach_exit(pool, self.bars)
        daily = ltr.attach_r20(self.cand, self.bars)
        sd = ltr5.check_same_day(pool, daily)
        self.assertEqual(sd["mismatch"], 0)
        self.assertEqual(sd["rows"], int((pool["r5_days"] == 0).sum()))
        daily2 = daily.copy()
        daily2[ltr.R20] = daily2[ltr.R20] + 0.01
        self.assertGreater(ltr5.check_same_day(pool, daily2)["mismatch"], 0)


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars, cls.days = market(n_codes=10, seed=3)
        cls.cand = candidates(cls.days, n_codes=10, rate=0.35, seed=4)
        cls.cal = ltr5.calendar(cls.bars["Date"].unique())
        pool = ltr5.pool_frame(cls.cand, cls.bars, cls.cal, FEATS, keep=("label",))
        cls.pool = ltr5.attach_exit(pool, cls.bars)

    def test_rows_and_folds_same_as_experiment(self):
        cutoff = pd.Timestamp("2022-06-30")
        sub, no_label = ltr5.search_rows(self.pool, cutoff)
        rl = self.pool.rename(columns={ltr5.R20: R.R20, ltr5.RET: R.RET})
        theirs, no_label2 = R.search_frame(rl, cutoff)
        self.assertEqual(no_label, no_label2)
        np.testing.assert_array_equal(sub[["Code", "Date"]].astype(str).to_numpy(),
                                      theirs[["Code", "Date"]].astype(str).to_numpy())
        mine_f = ltr5.search_folds(sub, self.cal)
        their_f = R.search_folds(theirs, self.cal)
        self.assertEqual(len(mine_f), 5)
        for (a_tr, a_va), (b_tr, b_va) in zip(mine_f, their_f):
            np.testing.assert_array_equal(a_tr.index.to_numpy(), b_tr.index.to_numpy())
            np.testing.assert_array_equal(a_va.index.to_numpy(), b_va.index.to_numpy())
            # 検証側の判断日の前後4営業日の判断日は訓練に入らない
            pos = pd.Series(np.arange(len(self.cal)), index=self.cal)
            vp = set(pos[pd.to_datetime(a_va["Date"])].to_numpy())
            tp = pos[pd.to_datetime(a_tr["Date"])].to_numpy()
            self.assertFalse(any(any(abs(int(t) - v) <= ltr5.PURGE for v in vp) for t in set(tp)))

    def test_tune_same_as_experiment(self):
        cutoff = pd.Timestamp("2022-06-30")
        sub, _ = ltr5.search_rows(self.pool, cutoff)
        cols_w = FEATS + list(ltr5.MOVE_COLS)
        folds = ltr5.search_folds(sub, self.cal)
        mine = ltr5.tune(sub, cols_w, folds, n_trials=3)
        e_folds = [(tr.rename(columns={ltr5.R20: E.R20}), va.rename(columns={ltr5.R20: E.R20})) for tr, va in folds]
        theirs = E.tune_l(sub.rename(columns={ltr5.R20: E.R20}), cols_w, 3, e_folds)
        self.assertEqual(mine["params"], theirs["params"])
        self.assertEqual(mine["_cv"]["ndcg"], theirs["_cv"]["ndcg"])
        self.assertEqual(mine["_cv"]["n_trials"], 3)
        self.assertEqual(mine["_cv"]["n_rows"], len(sub))

    def test_oof_windows_same_as_experiment(self):
        cols_w = FEATS + list(ltr5.MOVE_COLS)
        oof = ltr5.oof_scores(self.pool, cols_w, SMALL)
        ok = self.pool[ltr5.trainable(self.pool)]
        folds = [f for f in R.E41.folds_for(ok["Date"], 0)]
        d = pd.to_datetime(ok["Date"])
        used = [f for f in folds
                if ((d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))).sum() >= 200
                and (d <= pd.Timestamp(f.train_end)).sum() >= 1000]
        self.assertEqual(sorted(oof["fold"].unique().tolist()), [int(f.index) for f in used])
        # 1つの窓を手で当てはめ直すと同じスコア
        f = used[0]
        tr = ok[d <= pd.Timestamp(f.train_end)]
        te = ok[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))]
        want = W.fit_rank(SMALL, tr.rename(columns={ltr5.R20: W.R20}), te, cols_w)
        got = oof[oof["fold"] == int(f.index)]["score"].to_numpy()
        np.testing.assert_array_equal(got, want)


class TestRanksAndConds(unittest.TestCase):
    def test_ranks_same_as_experiment(self):
        rng = np.random.default_rng(2)
        days = pd.bdate_range("2024-01-01", periods=6)
        o = pd.DataFrame([{"Date": d, "Code": f"{1000 + j}", "r5_days": 0} for d in days for j in range(7)])
        o["score_W5"] = np.round(rng.normal(size=len(o)), 1)          # 同点を作る
        o["score_D"] = rng.normal(size=len(o))
        o["r15"] = 1.0
        o["cand_date"] = o["Date"]
        o[R.RET] = 0.0
        o[R.HIT] = False
        o[R.HITDAY] = np.nan
        x = o.copy()
        s = x.sort_values(["Date", "score_W5", "Code"], ascending=[True, False, True], kind="mergesort")
        x["w5_rank"] = 0
        x.loc[s.index, "w5_rank"] = (s.groupby("Date", sort=False).cumcount() + 1).to_numpy()
        got = ltr5.pool_ranks(o["Date"], o["Code"], o["score_W5"])
        np.testing.assert_array_equal(got, x["w5_rank"].to_numpy())
        self.assertEqual(sorted(got[o["Date"] == days[0]].tolist()), list(range(1, 8)))

    def test_conds(self):
        per = {"lgbm_k15": {"pctHistorical": 96.0}, "xgb_k15": {"pctHistorical": 95.0},
               "cat_k15": {"pctHistorical": 99.1}, "lgbm": {"pctHistorical": 94.9},
               "xgb": {"pctHistorical": 99.0}, "cat": {"pctHistorical": 98.0}}
        self.assertTrue(ltr5.all_at_least(per, ltr5.R15_TREES))
        self.assertFalse(ltr5.all_at_least(per, ltr5.R0_TREES))
        self.assertIsNone(ltr5.all_at_least({**per, "cat": {}}, ltr5.R0_TREES))
        self.assertIsNone(ltr5.all_at_least({k: v for k, v in per.items() if k != "xgb_k15"}, ltr5.R15_TREES))
        self.assertEqual(ltr5.conds(True, True, True, True), {"d1": True, "w1": True, "r15": True, "r0": True,
                                                             "c3": True, "c4": True})
        self.assertEqual(ltr5.conds(True, True, True, False)["c4"], False)
        self.assertEqual(ltr5.conds(True, False, True, True)["c3"], False)
        self.assertIsNone(ltr5.conds(True, None, True, True)["c3"])
        self.assertIsNone(ltr5.conds(True, None, True, True)["c4"])
        self.assertIsNone(ltr5.conds(True, True, True, None)["c4"])
        self.assertTrue(ltr5.conds(True, True, True, None)["c3"])


class TestParamsFile(unittest.TestCase):
    def test_committed_params_file(self):
        rec = ltr5.load_params()
        cols_w = list(F.columns(F.DEFAULT_PRESET)) + list(ltr5.MOVE_COLS)
        self.assertIsNone(ltr.params_mismatch(rec, cols_w))
        self.assertIsNotNone(ltr.params_mismatch(rec, list(F.columns(F.DEFAULT_PRESET))))   # 値動きの列なしでは通らない
        # 実験117 の run 41007055654 のログの「W5 の木の形（全桁）」と同じ
        self.assertEqual(rec["params"], {"boosting_type": "gbdt", "colsample_bytree": 0.8804170156336326,
                                         "learning_rate": 0.024033018135350435, "min_child_samples": 17,
                                         "n_estimators": 200, "n_jobs": -1, "num_leaves": 7,
                                         "reg_alpha": 9.2845440412956, "reg_lambda": 0.13416677080302547,
                                         "subsample": 0.7155999229063561, "subsample_freq": 1, "verbose": -1})
        self.assertEqual(rec["_cutoff"], "2025-09-01")
        self.assertEqual(rec["_cv"]["ndcg"], 0.5896)

    def test_settings_same_as_experiment(self):
        self.assertEqual(ltr5.K, R.K)
        self.assertEqual(ltr5.MOVE_COLS, R.MOVE_COLS)
        self.assertEqual(ltr5.SEED, R.SEED)
        self.assertEqual(ltr5.PURGE, R.K - 1)
        self.assertEqual((ltr5.TUNE_TRIALS, ltr5.TUNE_SPLITS, ltr5.TUNE_SEED), (50, E.N_SPLITS, E.SEED))
        self.assertEqual((ltr5.NDCG_K, ltr5.MIN_POOL), (E.NDCG_K, E.MIN_DAY))
        self.assertEqual(ltr5.R15_TREES, tuple(R.RULE_TREES))
        self.assertEqual(ltr5.R0_TREES, tuple(R.R0_TREES))
        self.assertEqual(ltr5.LINE, R.LINE)
        self.assertNotIn(ltr5.NAME, M.LINEUP)


class TestScripts(unittest.TestCase):
    def test_tune_then_train(self):
        import train_ltr5
        import tune_ltr5

        bars, days = market(n_codes=10, seed=5)
        cand = candidates(days, n_codes=10, rate=0.35, seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir)
            ds_path = os.path.join(data_dir, "dataset.parquet")
            cand.to_parquet(ds_path, index=False)
            pp = os.path.join(tmp, "ltr5_params.json")
            orig_cols, orig_load = F.columns, ltr5.LT.load_bars
            try:
                F.columns = lambda preset=None: list(FEATS)
                ltr5.LT.load_bars = lambda data_dir, start=None, extra=(): bars
                self.assertEqual(tune_ltr5.main(["--data-dir", data_dir, "--dataset", ds_path, "--out", pp,
                                                 "--trials", "2"]), 0)
                with open(pp, encoding="utf-8") as fh:
                    rec = json.load(fh)
                self.assertEqual(sorted(rec), ["_cutoff", "_cv", "_features_sig", "_n_features", "_source", "params"])
                cols_w = FEATS + list(ltr5.MOVE_COLS)
                self.assertIsNone(ltr.params_mismatch(rec, cols_w))
                self.assertEqual(rec["_n_features"], len(cols_w))
                self.assertEqual(rec["_cv"]["n_trials"], 2)
                self.assertTrue(rec["_source"].startswith("週次の探索（"))
                self.assertFalse(os.path.exists(pp + ".tmp"))
                self.assertEqual(train_ltr5.main(["--data-dir", data_dir, "--dataset", ds_path, "--out-dir", tmp,
                                                  "--params", pp]), 0)
                model, meta = ltr5.load(tmp)
                self.assertIsNotNone(model)
                self.assertEqual(meta["algo"], "ltr5")
                self.assertEqual(meta["features"], cols_w)
                self.assertEqual(meta["params"], rec["params"])
                self.assertEqual(meta["tuning"], rec["_cv"])
                self.assertEqual(meta["checks"]["sameDay"]["mismatch"], 0)
                self.assertEqual(meta["checks"]["moveCols"]["same"], meta["checks"]["moveCols"]["rows"])
                oof = pd.read_parquet(os.path.join(M.sub_dir("ltr5", tmp), "oof.parquet"))
                self.assertEqual(list(oof.columns), ["Code", "Date", "cand_date", "score", "fold"])
                self.assertEqual(meta["nOof"], len(oof))
                # 値動きの列を含まない木の形なら学習しない
                with open(pp, "w", encoding="utf-8") as fh:
                    json.dump({"params": SMALL, "_features_sig": F.signature(FEATS), "_n_features": len(FEATS)}, fh)
                self.assertEqual(train_ltr5.main(["--data-dir", data_dir, "--dataset", ds_path, "--out-dir", tmp,
                                                  "--params", pp]), 1)
            finally:
                F.columns, ltr5.LT.load_bars = orig_cols, orig_load
        self.assertEqual(ltr5.load(os.path.join(tmp, "none")), (None, None))

    def test_source_note(self):
        import tune_ltr5

        cv = {"n_rows": 90188, "n_trials": 50, "n_splits": 5}
        env = {"GITHUB_RUN_ID": "123", "GITHUB_WORKFLOW": "Retrain Weekly", "GITHUB_SHA": "abcdef0123"}
        self.assertEqual(tune_ltr5.source_note(cv, "2025-09-01", env),
                         "週次の探索（Retrain Weekly run 123・abcdef0。2025-09-01 までの組の行 90,188件・50試行・5分割・"
                         "目的は検証分割の組の中の NDCG@5）")


class TestAttach(unittest.TestCase):
    """毎日の予測で、予測の日の組を採点して順位と3条件・4条件の印を付ける（predict_daily.attach_ltr5）。"""

    @classmethod
    def setUpClass(cls):
        import live_track as LT

        cls.LT = LT
        cls.bars, cls.days = market(n_codes=12, seed=7)
        cls.cal = ltr5.calendar(cls.bars["Date"].unique())
        hist = candidates(cls.days, n_codes=12, rate=0.35, seed=8)
        pool = ltr5.attach_exit(ltr5.pool_frame(hist, cls.bars, cls.cal, FEATS, keep=("label",)), cls.bars)
        cls.cols_w = FEATS + list(ltr5.MOVE_COLS)
        cls.model = ltr.fit(pool[ltr5.trainable(pool)], cls.cols_w, SMALL, ltr5.SEED)
        # 予測の日（日足の最後の日）までの直近7営業日の候補。予測の日は8銘柄、その前の日は12銘柄から一部（予測の日に
        # 入った銘柄の前の候補は、組では予測の日の行だけが残る）
        last = cls.cal[-7:]
        rng = np.random.default_rng(9)
        rows = []
        for i, d in enumerate(last):
            for j in range(12):
                if (i == len(last) - 1 and j < 8) or (i < len(last) - 1 and rng.random() < 0.3):
                    rows.append({"Code": f"{1000 + j}", "Date": d})
        cand = pd.DataFrame(rows)
        for c in FEATS:
            cand[c] = rng.normal(size=len(cand))
        cand["extra"] = 1.0                                           # 予測の候補にはほかの列もある
        cls.cand = cand
        cls.day = last[-1]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        meta = {"algo": "ltr5", "features": self.cols_w, "trainedAt": "2026-11-01T00:00:00", "nOof": 0}
        M.save("ltr5", self.model, meta, pd.DataFrame({"Code": [], "Date": [], "score": []}), root=self.tmp.name)
        orig = self.LT.load_bars
        self.LT.load_bars = lambda data_dir, start=None, extra=(): self.bars
        self.addCleanup(setattr, self.LT, "load_bars", orig)

    def rows(self, cand):
        """予測の行（cand と同じ順）。日次のランキング学習の順位と、6本の百分位（既定はどれも低い）。"""
        out = []
        for d, g in cand.groupby("Date", sort=True):
            g = g.sort_values("Code")
            for r, (_, x) in enumerate(g.iterrows(), start=1):
                out.append({"date": pd.Timestamp(d).date().isoformat(), "jqCode": x["Code"], "code": x["Code"][:4],
                            "ltr": {"score": float(-r), "dayRank": r, "nDay": len(g)},
                            "byModel": {a: {"pctHistorical": 50.0} for a in ltr5.R15_TREES + ltr5.R0_TREES}})
        return out

    def test_ranks_on_latest_day(self):
        import predict_daily as PD

        rows = self.rows(self.cand)
        info = PD.attach_ltr5(rows, self.cand, "unused", self.tmp.name)
        self.assertIsNotNone(info)
        pool = ltr5.pool_frame(self.cand, self.bars, self.cal, FEATS, decision_days=[self.day])
        sc = self.model.predict(pool[self.cols_w].to_numpy(dtype=float))
        want = dict(zip(pool["Code"], ltr5.pool_ranks(pool["Date"], pool["Code"], sc)))
        iso = self.day.date().isoformat()
        latest = [x for x in rows if x["date"] == iso]
        self.assertEqual(len(latest), 8)
        for x in rows:
            if x["date"] != iso:
                self.assertNotIn("ltr5", x)                           # 前の日の行には付けない
                self.assertNotIn("conds", x)
                continue
            self.assertEqual(x["ltr5"]["rank"], int(want[x["jqCode"]]))
            self.assertEqual(x["ltr5"]["nPool"], len(pool))
        self.assertGreater(len(pool), 8)                               # 前の4営業日の候補も組に入る
        self.assertTrue((pool["r5_days"] > 0).any())
        self.assertFalse(pool[pool["r5_days"] > 0]["Code"].isin([f"{1000 + j}" for j in range(8)]).any())
        self.assertEqual((info["day"], info["nPool"], info["poolDays"]), (iso, len(pool), 5))
        self.assertEqual(sorted(x["ltr5"]["rank"] for x in latest)[0], int(min(want[x["jqCode"]] for x in latest)))

    def test_conds(self):
        import predict_daily as PD

        # まず順位を見て、予測の日の候補のうち組の中で1位のもの（W1）に日次の1位と6本の95以上を付ける
        rows = self.rows(self.cand)
        PD.attach_ltr5(rows, self.cand, "unused", self.tmp.name)
        iso = self.day.date().isoformat()
        latest = [x for x in rows if x["date"] == iso]
        w1 = [x for x in latest if x["ltr5"]["rank"] == 1]
        rows = self.rows(self.cand)
        if not w1:                                                     # 組の1位が前の日の候補なら、W1 の候補は無い
            PD.attach_ltr5(rows, self.cand, "unused", self.tmp.name)
            self.assertTrue(all(x["conds"]["w1"] is False and x["conds"]["c3"] is False
                                for x in rows if x["date"] == iso))
            return
        code = w1[0]["jqCode"]
        for x in rows:
            if x["date"] == iso:
                x["ltr"]["dayRank"] = 1 if x["jqCode"] == code else x["ltr"]["dayRank"] + 1
                if x["jqCode"] == code:
                    x["byModel"] = {a: {"pctHistorical": 96.0} for a in ltr5.R15_TREES + ltr5.R0_TREES}
        info = PD.attach_ltr5(rows, self.cand, "unused", self.tmp.name)
        got = {x["jqCode"]: x["conds"] for x in rows if x["date"] == iso}
        self.assertEqual(got[code], {"d1": True, "w1": True, "r15": True, "r0": True, "c3": True, "c4": True})
        self.assertTrue(all(not c["c3"] and not c["c4"] for k, c in got.items() if k != code))
        self.assertEqual((info["n3"], info["n4"]), (1, 1))
        # 1.2σ の1本が95未満なら3条件だけ
        rows2 = [dict(x) for x in rows]
        for x in rows2:
            x.pop("ltr5", None)
            x.pop("conds", None)
            if x["jqCode"] == code and x["date"] == iso:
                x["byModel"] = {**x["byModel"], "cat": {"pctHistorical": 94.9}}
        PD.attach_ltr5(rows2, self.cand, "unused", self.tmp.name)
        c = [x for x in rows2 if x["jqCode"] == code and x["date"] == iso][0]["conds"]
        self.assertEqual((c["c3"], c["c4"], c["r0"]), (True, False, False))
        # 日次の1位が2件（同点）なら銘柄コードの小さいほうが D1（実験117 と同じ）
        rows3 = self.rows(self.cand)
        for x in rows3:
            if x["date"] == iso:
                x["ltr"]["dayRank"] = 1
        PD.attach_ltr5(rows3, self.cand, "unused", self.tmp.name)
        d1 = [x["jqCode"] for x in rows3 if x["date"] == iso and x["conds"]["d1"]]
        self.assertEqual(d1, [min(x["jqCode"] for x in rows3 if x["date"] == iso)])

    def test_skips(self):
        import predict_daily as PD

        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        rows = self.rows(self.cand)
        self.assertIsNone(PD.attach_ltr5(rows, self.cand, "unused", empty.name))      # モデルが無い
        self.assertFalse(any("ltr5" in x for x in rows))
        short = self.cand[self.cand["Date"] == self.day]
        rows = self.rows(short)
        self.assertIsNone(PD.attach_ltr5(rows, short, "unused", self.tmp.name))        # 5営業日に足りない
        self.assertFalse(any("ltr5" in x or "conds" in x for x in rows))
        rows = self.rows(self.cand)
        self.assertIsNone(PD.attach_ltr5(rows, self.cand.drop(columns=["x1"]), "unused", self.tmp.name))   # 列が無い
        self.assertFalse(any("ltr5" in x for x in rows))
        # 日次のランキング学習の順位が無ければ D1 と3条件・4条件は決めない（W1・R15・R0 は付く）
        rows = self.rows(self.cand)
        for x in rows:
            x.pop("ltr")
        PD.attach_ltr5(rows, self.cand, "unused", self.tmp.name)
        iso = self.day.date().isoformat()
        c = [x["conds"] for x in rows if x["date"] == iso]
        self.assertTrue(all(v["d1"] is None and v["c3"] is None and v["c4"] is None for v in c))
        self.assertTrue(all(v["r15"] is False for v in c))


class TestWorkflow(unittest.TestCase):
    """日曜の再学習（retrain-weekly.yml）に直近5営業日の組のランキング学習の探索と学習が入っている（§82-6）。"""

    def setUp(self):
        import yaml

        with open(os.path.join(ROOT, ".github", "workflows", "retrain-weekly.yml"), encoding="utf-8") as fh:
            self.wf = yaml.safe_load(fh)
        self.steps = self.wf["jobs"]["retrain"]["steps"]
        self.names = [s.get("name") for s in self.steps]

    def step(self, name):
        return self.steps[self.names.index(name)]

    def test_order_and_conditions(self):
        i = self.names.index
        self.assertLess(i("Train ranking model"), i("Tune 5-day ranking model"))
        self.assertLess(i("Tune 5-day ranking model"), i("Train 5-day ranking model"))
        self.assertLess(i("Train 5-day ranking model"), i("Upload model to release"))
        tune = self.step("Tune 5-day ranking model")
        self.assertEqual(tune["if"], "inputs.tune == 'yes' && steps.scope.outputs.ltr5 == 'yes'")
        self.assertIs(tune["continue-on-error"], True)
        self.assertEqual(tune["timeout-minutes"], 60)
        self.assertIn("python3 research/tune_ltr5.py", tune["run"])
        self.assertEqual(tune["env"]["JOB_START"], "${{ steps.start.outputs.t }}")
        train = self.step("Train 5-day ranking model")
        self.assertEqual(train["if"], "steps.scope.outputs.ltr5 == 'yes'")
        self.assertIs(train["continue-on-error"], True)
        self.assertIn("python3 research/train_ltr5.py", train["run"])

    def test_guard_skips_near_the_limit(self):
        import subprocess
        import time

        run = self.step("Tune 5-day ranking model")["run"].replace("python3 research/tune_ltr5.py", "echo RUN")
        with tempfile.TemporaryDirectory() as tmp:
            for ago_min, want, not_want in ((240, "探索しない", "RUN"), (60, "RUN", "探索しない")):
                env = dict(os.environ, JOB_START=str(int(time.time()) - ago_min * 60))
                out = subprocess.run(["bash", "-eo", "pipefail", "-c", run], cwd=tmp, env=env, capture_output=True,
                                     text=True)
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertIn(want, out.stdout)
                self.assertNotIn(not_want, out.stdout)

    def test_scope(self):
        import subprocess

        run = self.step("Resolve scope")["run"]
        cases = {"": ("", "yes", "yes"), "ltr5": ("", "no", "yes"), "ltr": ("", "yes", "no"),
                 "lgbm_k15,xgb_k15,cat_k15": ("lgbm_k15,xgb_k15,cat_k15", "no", "no"),
                 "ltr,ltr5": ("", "yes", "yes"), "xgb,ltr5,cat": ("xgb,cat", "no", "yes")}
        with tempfile.TemporaryDirectory() as tmp:
            for only, (models, ltr_, ltr5_) in cases.items():
                outp = os.path.join(tmp, "out.txt")
                open(outp, "w").close()
                env = dict(os.environ, ONLY=only, GITHUB_OUTPUT=outp)
                r = subprocess.run(["bash", "-eo", "pipefail", "-c", run], cwd=tmp, env=env, capture_output=True,
                                   text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                with open(outp, encoding="utf-8") as fh:
                    got = dict(line.split("=", 1) for line in fh.read().splitlines() if "=" in line)
                self.assertEqual((got["models"], got["ltr"], got["ltr5"]), (models, ltr_, ltr5_), only)

    def test_commit_includes_params(self):
        run = self.step("Commit model metadata")["run"]
        self.assertIn("research/ltr5_params.json", run)
        self.assertIn("git diff -- research/ltr5_params.json", run)


if __name__ == "__main__":
    unittest.main()
