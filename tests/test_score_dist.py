#!/usr/bin/env python3
"""
research/score_dist.py（画面のタブ「スコアと収益の分布」の集計）。合成の日足・out-of-fold で:

- 収益: R20 = 判断日の翌営業日の寄りで買い20営業日目の終値、RMAX = 同じ窓の終値の最高値。live_track.forward の ret_close と一致、
  20営業日に満たない・寄り付かない行は欠測
- 度数: 端の外側は最初・最後の区間にまとめる
- 組の中の順位・百分位・10分位
- 通しで回す: 学習の記録と SHA-256 が違う out-of-fold では止まる、出力に銘柄コードが入らない、重なりの件数が手計算と合う

  python3 tests/test_score_dist.py
"""
import contextlib
import hashlib
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

import live_track as LT  # noqa: E402
import score_dist as SD  # noqa: E402


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def make_bars(codes, n=80, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2024-01-04", periods=n)
    rows = []
    for k, c in enumerate(codes):
        px = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        o = px * (1 + rng.normal(0, 0.005, n))
        rows.append(pd.DataFrame({"Code": c, "Date": d, "AdjO": o, "AdjH": np.maximum(o, px) * 1.01, "AdjC": px}))
    return pd.concat(rows, ignore_index=True)


class TestForward(unittest.TestCase):
    def test_r20_rmax_and_missing(self):
        d = pd.bdate_range("2024-01-04", periods=30)
        c = np.full(30, 100.0)
        c[5] = 150.0
        c[20] = 110.0
        o = np.full(30, 100.0)
        o[11] = np.nan                                         # 判断日 10 の翌日は寄り付かない
        bars = pd.DataFrame({"Code": "1111", "Date": d, "AdjO": o, "AdjH": c, "AdjC": c})
        keys = pd.DataFrame({"Code": ["1111"] * 3, "Date": [d[0], d[10], d[15]]})
        r = SD.forward_returns(keys, bars).set_index("Date")
        self.assertAlmostEqual(r.loc[d[0], "R20"], 0.10)
        self.assertAlmostEqual(r.loc[d[0], "RMAX"], 0.50)
        self.assertTrue(np.isnan(r.loc[d[10], "R20"]))         # 寄りが無い
        self.assertTrue(np.isnan(r.loc[d[15], "R20"]))         # 20営業日に満たない

    def test_matches_live_track(self):
        bars = make_bars(["1111", "2222"])
        d = sorted(bars["Date"].unique())
        keys = pd.DataFrame({"Code": ["1111", "2222", "1111"], "Date": [d[3], d[10], d[40]]})
        r = SD.forward_returns(keys, bars)
        fw = LT.forward(bars, keys)
        m = r.merge(fw[["Code", "Date", "ret_close"]], on=["Code", "Date"])
        np.testing.assert_allclose(m["R20"] * 100, m["ret_close"], atol=1e-9)
        self.assertTrue((r["RMAX"] >= r["R20"] - 1e-12).all())


class TestParts(unittest.TestCase):
    def test_hist_clips_ends(self):
        self.assertEqual(SD.hist([-5, 0.05, 0.5, 9], [0.0, 0.1, 0.2, 1.0]), [2, 0, 2])

    def test_pool_rank(self):
        lt = pd.DataFrame({"Date": ["a"] * 10 + ["b"] * 2, "score": list(range(10)) + [1, 2]})
        r = SD.pool_rank(lt)
        a = r[r["Date"] == "a"].sort_values("score")
        self.assertEqual(a["rk"].tolist(), list(range(10, 0, -1)))
        self.assertEqual(a["dec"].tolist(), list(range(1, 11)))
        self.assertAlmostEqual(a["pp"].iloc[-1], 1.0)
        b = r[r["Date"] == "b"].sort_values("score")
        self.assertEqual(b["dec"].tolist(), [5, 10])


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = os.path.join(self.tmp, "data")
        self.model = os.path.join(self.tmp, "model")
        os.makedirs(self.data)
        os.makedirs(os.path.join(self.model, "models", "ltr5"))
        codes = [f"{1000 + i}" for i in range(12)]
        self.codes = codes
        bars = make_bars(codes, n=90, seed=1)
        bars.to_parquet(os.path.join(self.data, "bars_2024.parquet"), index=False)
        d = sorted(bars["Date"].unique())
        rng = np.random.default_rng(2)
        # lgbm: 候補（Code, 更新日）ごとのスコア
        cand = pd.DataFrame([(c, d[i]) for i in range(5, 50, 3) for c in codes[(i % 4)::4]], columns=["Code", "Date"])
        cand["score"] = rng.uniform(0.05, 0.75, len(cand))
        cand["label"] = (rng.uniform(size=len(cand)) < cand["score"] * 0.5).astype(int)
        cand["ret_o1_20"] = rng.normal(0, 0.1, len(cand))
        self.lg = cand
        # ltr5: 判断日ごとに直近5営業日の候補を組にする
        rows = []
        for j in range(10, 55):
            pool = cand[(cand["Date"] <= d[j]) & (cand["Date"] > d[j - 5])]
            for _, r in pool.iterrows():
                rows.append((r["Code"], d[j], r["Date"], rng.normal(), 1))
        self.lt = pd.DataFrame(rows, columns=["Code", "Date", "cand_date", "score", "fold"])
        lg_p = os.path.join(self.data, "fixed_return_v1_oof.parquet")
        lt_p = os.path.join(self.data, "ltr5_oof.parquet")
        cand.to_parquet(lg_p, index=False)
        self.lt.to_parquet(lt_p, index=False)
        import label_specs as LS
        self.lg_meta = {"trainedAt": "2026-11-08T00:00:00", "label": "x", "labelSpec": LS.for_model("lgbm"),
                        "oofMetrics": {"prAuc": 0.3, "rocAuc": 0.6},
                        "artifactHashes": {"oof.parquet": sha(lg_p), "model.txt": "x"}}
        self.lt_meta = {"trainedAt": "2026-11-08T01:00:00", "nGrades": 10, "oofMetrics": {"ndcg5": 0.5},
                        "artifactHashes": {"oof.parquet": sha(lt_p)}}
        self.write_meta()

    def write_meta(self):
        with open(os.path.join(self.model, "meta.json"), "w") as fh:
            json.dump(self.lg_meta, fh)
        with open(os.path.join(self.model, "models", "ltr5", "meta.json"), "w") as fh:
            json.dump(self.lt_meta, fh)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_main(self):
        out = os.path.join(self.tmp, "out.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            SD.main(["--data-dir", self.data, "--model-dir", self.model, "--out", out])
        with open(out, encoding="utf-8") as fh:
            raw = fh.read()
        return json.loads(raw), buf.getvalue(), raw

    def test_end_to_end(self):
        d, log, raw = self.run_main()
        for c in self.codes:
            self.assertNotIn(c, raw)
            self.assertNotIn(c, log)
        L, R, O = d["lgbm"], d["ltr5"], d["overlap"]
        self.assertEqual(L["n"], len(self.lg))
        self.assertEqual(sum(L["hist"]["pos"]) + sum(L["hist"]["neg"]), len(self.lg))
        self.assertEqual(sum(L["hist"]["pos"]), int(self.lg["label"].sum()))
        self.assertEqual(sum(b["n"] for b in L["bands"]), len(self.lg))
        self.assertEqual(R["nRows"], len(self.lt))
        self.assertEqual(sum(R["r20Hist"]["all"]), len(self.lt))
        self.assertEqual(sum(R["r20Hist"]["top1"]), R["nDays"])
        self.assertEqual(sum(x["n"] for x in R["deciles"]), len(self.lt))
        self.assertEqual(d["checks"]["r20"]["same"], d["checks"]["r20"]["rows"])
        # 重なりを手で数える
        thr = self.lg["score"].quantile(0.9)
        lt = SD.pool_rank(self.lt)
        m = lt.merge(self.lg[["Code", "Date", "score"]].rename(columns={"Date": "cand_date", "score": "lg"}),
                     on=["Code", "cand_date"])
        both = int(((m["lg"] >= thr) & (m["pp"] > 0.9)).sum())
        self.assertEqual(O["both"], both)
        self.assertEqual(sum(g["n"] for g in O["groups"]), len(m))
        self.assertEqual(O["lgbmPick"]["n"], m["Date"].nunique())

    def test_stops_on_stale_ltr5(self):
        self.lt_meta["artifactHashes"]["oof.parquet"] = "0" * 64
        self.write_meta()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            SD.main(["--data-dir", self.data, "--model-dir", self.model, "--out", os.path.join(self.tmp, "o.json")])

    def test_stops_on_stale_lgbm(self):
        self.lg_meta["artifactHashes"]["oof.parquet"] = "0" * 64
        self.write_meta()
        with self.assertRaises((SystemExit, ValueError)), contextlib.redirect_stdout(io.StringIO()):
            SD.main(["--data-dir", self.data, "--model-dir", self.model, "--out", os.path.join(self.tmp, "o.json")])


if __name__ == "__main__":
    unittest.main()
