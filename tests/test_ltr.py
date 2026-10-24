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
import e52_return_objective as E52  # noqa: E402
import e114_ltr_retest as E  # noqa: E402

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
        rec = ltr.load_params()
        cols = F.columns(F.DEFAULT_PRESET)
        self.assertIsNone(ltr.params_mismatch(rec, cols))
        p = rec["params"]
        self.assertEqual(p["n_estimators"], 200)
        self.assertEqual((p["num_leaves"], p["min_child_samples"]), (69, 29))
        # 実験114 のログ（4桁に丸めた値）と同じ
        for k, v in {"learning_rate": 0.01215, "subsample": 0.5001, "colsample_bytree": 0.6665,
                     "reg_alpha": 2.165, "reg_lambda": 0.1016}.items():
            self.assertEqual(float(f"{p[k]:.4g}"), v, k)


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
            orig_cols, orig_load = F.columns, train_ltr.LT.load_bars
            try:
                F.columns = lambda preset=None: list(cols)
                train_ltr.LT.load_bars = lambda data_dir, start=None, extra=(): bars
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
                F.columns, train_ltr.LT.load_bars = orig_cols, orig_load
        self.assertEqual(ltr.load(os.path.join(tmp, "none")), (None, None))


if __name__ == "__main__":
    unittest.main()
