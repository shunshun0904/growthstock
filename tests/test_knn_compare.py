#!/usr/bin/env python3
"""
候補の比較（research/knn_compare.py。画面の「候補の比較」タブに渡す近傍の集計）。

- 分布・W1・尺度は実験91（research/exp/e91_wasserstein_knn_pick.py）の関数と同じ値
- 母集団の百分位は画面の物差し（models.pct_historical / live_track.pct_of）と同じ
- 母集団は、予測日より前に結果が出そろった行だけ（先読みしない）。候補の行は自分の母集団に入らない
- 候補の見積もり（25%点・中央値・−10% 割れ）は、実験93 の neighbor_stats で別に計算した値と一致する。雲の度数の合計は近傍の数
- 1.5σ の3本のどれかが欠けた候補には付けない。日足の足りない候補にも付けない
- 文脈の2次元の度数は MIN_CELL 未満のマス目を消す。出力に過去の事例の銘柄コード・日付が含まれない
- 過去の日は前の knn のまま（引き継ぎ）。out-of-fold が無ければ例外を出さず「出せない」として返す
- JSON に書ける

  python3 tests/test_knn_compare.py
"""
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

import knn_compare as KC  # noqa: E402
import live_track as LT  # noqa: E402
import models as M  # noqa: E402
import e91_wasserstein_knn_pick as E91  # noqa: E402
import e93_knn_quantile_pick as E93  # noqa: E402

DAY = "2025-07-01"
N_CODES, PER_CODE = 60, 60


def make_bars(codes, days, spiky, seed=0) -> pd.DataFrame:
    """spiky の銘柄は右に歪んだリターンと荒い出来高、ほかは正規のリターンと穏やかな出来高（実験91 のテストと同じ作り）。"""
    rng = np.random.default_rng(seed)
    n = len(days)
    parts = []
    for c in codes:
        if spiky[c]:
            r = rng.exponential(0.02, size=n) - 0.02
            v = np.exp(rng.normal(10.0, 1.0, size=n))
        else:
            r = rng.normal(0.0, 0.02, size=n)
            v = np.exp(rng.normal(10.0, 0.2, size=n))
        close = 1000.0 * np.cumprod(1.0 + r)
        o = close * (1.0 + rng.normal(0.0, 0.004, size=n))
        h = np.maximum(o, close) * (1.0 + np.abs(rng.normal(0.0, 0.01, size=n)))
        parts.append(pd.DataFrame({"Date": days, "Code": c, "AdjO": o, "AdjH": h, "AdjC": close, "C": close,
                                   "AdjVo": v, "Vo": v}))
    return pd.concat(parts, ignore_index=True)


def make_case(seed=0):
    """
    合成の out-of-fold（規則の3本。共通の潜在値 + 雑音で、最小が85以上の行が1割ほど）と日足を一時ディレクトリに置く。
    戻り値: (data_dir, model_dir, bars, oof_wide, codes)。日足は年ごとの bars_<年>.parquet（本番と同じ置き方）。
    """
    rng = np.random.default_rng(seed)
    codes = [f"{10000 + 10 * i}" for i in range(N_CODES)]
    spiky = {c: bool(rng.random() < 0.5) for c in codes}
    days = pd.bdate_range("2023-06-01", "2025-08-29")
    bars = make_bars(codes, days, spiky, seed=seed + 3)
    event_days = pd.bdate_range("2024-01-04", "2025-06-20")
    recs = []
    for c in codes:
        for d in rng.choice(event_days, size=PER_CODE, replace=False):
            u = rng.random()
            recs.append({"Code": c, "Date": pd.Timestamp(d), "u": u})
    oof = pd.DataFrame(recs)
    # 先読みの確かめ用: 判断日の直前の行（20営業日目が判断日より後）を必ず入れる
    oof = pd.concat([oof, pd.DataFrame([{"Code": codes[0], "Date": pd.Timestamp("2025-06-25"), "u": 0.999}])],
                    ignore_index=True)
    for a in KC.RULE_MODELS:
        oof[f"score_{a}"] = oof["u"] + rng.normal(0.0, 0.08, size=len(oof))
    data_dir = tempfile.mkdtemp()
    model_dir = tempfile.mkdtemp()
    for y, g in bars.groupby(bars["Date"].dt.year):
        g.to_parquet(os.path.join(data_dir, f"bars_{y}.parquet"), index=False)
    for a in KC.RULE_MODELS:
        d = os.path.join(model_dir, "models", a)
        os.makedirs(d)
        o = oof[["Code", "Date"]].assign(score=oof[f"score_{a}"])
        # 画面の物差しは NaN を含む全行で付ける（models.hist_scores と同じ）。1行だけ NaN を混ぜる（先頭の行。末尾は先読みの確かめ用）
        if a == KC.RULE_MODELS[0]:
            o.loc[o.index[0], "score"] = np.nan
        o.to_parquet(os.path.join(d, "oof.parquet"), index=False)
    return data_dir, model_dir, bars, oof, codes


def cand(jq, date, pcts, score=0.5):
    per = {a: {"score": score, "pctHistorical": p} for a, p in zip(KC.RULE_MODELS, pcts) if p is not None}
    return {"jqCode": jq, "code": jq[:4], "date": date, "score": score, "byModel": per}


class TestHelpersMatchExperiment(unittest.TestCase):
    def test_quantile_and_w1(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            a, b = rng.standard_t(3, 80), rng.normal(0.3, 2.0, 60)
            np.testing.assert_array_equal(KC.quantile_vector(a), E91.quantile_vector(a))
            self.assertEqual(float(KC.w1(KC.quantile_vector(a), KC.quantile_vector(b)[None, :])[0]),
                             float(E91.w1(E91.quantile_vector(a), E91.quantile_vector(b)[None, :])[0]))

    def test_distributions_and_scale(self):
        rng = np.random.default_rng(1)
        codes = ["10010", "10020", "10030"]
        days = pd.bdate_range("2024-01-04", periods=260)
        bars = make_bars(codes, days, {c: i % 2 == 0 for i, c in enumerate(codes)}, seed=2)
        keys = pd.DataFrame({"Code": rng.choice(codes, 40), "Date": rng.choice(days[130:240], 40)})
        a, b = KC.event_distributions(bars, keys), E91.event_distributions(bars, keys)
        np.testing.assert_array_equal(a["QR"], b["QR"])
        np.testing.assert_array_equal(a["QV"], b["QV"])
        np.testing.assert_array_equal(a["done"], b["done"])
        self.assertEqual(KC.pair_median(a["QR"], 2000, 42), E91.pair_median(b["QR"], 2000, 42))
        self.assertEqual(KC.done_before(np.sort(a["done"]), days[200]), E91.done_before(np.sort(b["done"]), days[200]))
        self.assertEqual((KC.K, KC.POOL_LINE, KC.CONTEXT_LINE, KC.TARGET, KC.HOLD, KC.WIN, KC.LEVELS),
                         (E91.K_MAIN, E91.POOL_LINE, E91.LINE, E91.TARGET, E91.HOLD, E91.WIN, E91.LEVELS))
        self.assertEqual(KC.RULE_MODELS, tuple(E91.RULE_MODELS))


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir, cls.model_dir, cls.bars, cls.oof, cls.codes = make_case()
        cls.cands = [cand(cls.codes[1] , DAY, [97.0, 98.5, 96.2], 0.9),
                     cand(cls.codes[2], DAY, [99.0, 95.0, 95.5], 0.8),
                     cand(cls.codes[3], DAY, [88.0, 93.0, 91.0], 0.7),
                     cand(cls.codes[4], DAY, [30.0, 70.0, 40.0], 0.6),
                     cand(cls.codes[5], DAY, [97.0, None, 96.0], 0.5),     # 1本欠け → 付けない
                     cand("99990", DAY, [99.0, 99.0, 99.0], 0.4)]          # 日足が無い → 付けない
        cls.knn = KC.build(cls.cands, ["2025-06-30", DAY], cls.data_dir, cls.model_dir, prev=None, frozen=[])
        cls.blk = cls.knn["byDay"][DAY]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)
        shutil.rmtree(cls.model_dir, ignore_errors=True)

    def direct_pool(self):
        """台本と別に、母集団（最小85以上・結果が出そろった行）と分布を作る。"""
        base = None
        for a in KC.RULE_MODELS:
            o = pd.read_parquet(os.path.join(self.model_dir, "models", a, "oof.parquet"))
            hist = o["score"].to_numpy(dtype=float)
            o = o[np.isfinite(hist)].copy()
            o[f"p_{a}"] = [M.pct_historical(float(s), hist) for s in o["score"]]
            o = o.drop(columns=["score"])
            base = o if base is None else base.merge(o, on=["Code", "Date"])
        base["min15"] = base[[f"p_{a}" for a in KC.RULE_MODELS]].min(axis=1)
        pool = base[base["min15"] >= 85].reset_index(drop=True)
        pool["Code"] = pool["Code"].astype(str)
        pool["Date"] = pd.to_datetime(pool["Date"])
        fw = LT.forward(self.bars, pool[["Code", "Date"]], target=10.0, hold=20, fallback="close")
        pool = pool.merge(fw[["Code", "Date", "ret", "status"]], on=["Code", "Date"])
        dist = E91.event_distributions(self.bars, pool[["Code", "Date"]])
        ok = (np.isfinite(pool["ret"].to_numpy(dtype=float)) & np.isfinite(dist["QR"]).all(axis=1)
              & np.isfinite(dist["QV"]).all(axis=1) & ~np.isnat(dist["done"]))
        pool = pool[ok].reset_index(drop=True)
        return pool, dist["QR"][ok], dist["QV"][ok], dist["done"][ok]

    def test_pool_excludes_unfinished_rows(self):
        pool, QR, QV, done = self.direct_pool()
        before = done < np.datetime64(pd.Timestamp(DAY), "ns")
        self.assertEqual(self.blk["poolN"], int(before.sum()))
        self.assertLess(int(before.sum()), len(pool))                       # 判断日の直前の行は入らない
        late = pool[pool["Date"] == pd.Timestamp("2025-06-25")]
        self.assertEqual(len(late), 1)
        self.assertFalse(before[late.index[0]])
        self.assertGreaterEqual(self.blk["poolN"], KC.MIN_POOL)
        self.assertEqual(self.blk["status"], "ok")

    def test_estimates_match_direct_recomputation(self):
        pool, QR, QV, done = self.direct_pool()
        order = np.argsort(done, kind="mergesort")
        pool, QR, QV, done = pool.iloc[order].reset_index(drop=True), QR[order], QV[order], done[order]
        n0 = E91.done_before(done, pd.Timestamp("2025-06-30"))               # 尺度は最初の採点日より前の母集団
        sR = E91.pair_median(QR[:n0], E93.N_PAIRS, E93.SEED)
        sV = E91.pair_median(QV[:n0], E93.N_PAIRS, E93.SEED + 100)
        self.assertEqual(self.knn["scale"], {"R": round(sR, 4), "V": round(sV, 4)})
        n = E91.done_before(done, pd.Timestamp(DAY))
        tp = pool["tp"].to_numpy(dtype=float) if "tp" in pool else pool["ret"].to_numpy(dtype=float)
        hit = pool["status"].eq(LT.HIT).to_numpy()
        keys = pd.DataFrame({"Code": [c["jqCode"] for c in self.cands], "Date": [pd.Timestamp(DAY)] * len(self.cands)})
        cd = E91.event_distributions(self.bars, keys)
        checked = 0
        for j, c in enumerate(self.cands[:4]):
            rec = self.blk["candidates"][c["jqCode"]]
            D = E91.w1(cd["QR"][j], QR[:n]) / sR + E91.w1(cd["QV"][j], QV[:n]) / sV
            st = E93.neighbor_stats(D, tp[:n], KC.K)
            self.assertAlmostEqual(rec["est"]["p25"], round(st["p25"], 2), places=2)
            self.assertAlmostEqual(rec["est"]["p50"], round(st["p50"], 2), places=2)
            self.assertAlmostEqual(rec["est"]["bad"], round(st["bad"], 1), places=1)
            nn = np.argpartition(D, KC.K - 1)[:KC.K]
            self.assertAlmostEqual(rec["est"]["hit"], round(float(hit[:n][nn].mean() * 100), 1), places=1)
            self.assertEqual(rec["est"]["n"], KC.K)
            self.assertEqual(rec["min15"], round(min(c["byModel"][a]["pctHistorical"] for a in KC.RULE_MODELS), 1))
            self.assertLessEqual(rec["est"]["p10"], rec["est"]["p25"])
            self.assertLessEqual(rec["est"]["p25"], rec["est"]["p50"])
            self.assertLessEqual(rec["est"]["p75"], rec["est"]["p90"])
            checked += 1
        self.assertEqual(checked, 4)

    def test_cloud_and_skipped_candidates(self):
        c = self.blk["candidates"]
        self.assertEqual(set(c), {x["jqCode"] for x in self.cands[:4]})      # 1本欠け・日足なしは付けない
        self.assertEqual(self.blk["n"], 4)
        for rec in c.values():
            counts = np.array(rec["cloud"])
            self.assertEqual(counts.shape, (len(KC.D_BINS) - 1, len(KC.BANDS)))
            self.assertEqual(int(counts.sum()), rec["est"]["n"])
            self.assertGreater(rec["dNn"]["R"] + rec["dNn"]["V"], 0.0)
        self.assertEqual(self.knn["bands"], ["-10 未満", "-10〜0", "0〜+5", "+5〜+10 未満", "+10 到達"])
        self.assertEqual(self.knn["dBins"], [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
        self.assertEqual(self.knn["models"], ["lgbm_k15", "xgb_k15", "cat_k15"])
        self.assertEqual(self.knn["byDay"]["2025-06-30"]["status"], "no_candidates")

    def test_context_and_no_identifiers(self):
        cx = self.knn["context"]
        self.assertIsNotNone(cx)
        self.assertEqual(len(cx["p25Deciles"]), 11)
        self.assertEqual(cx["p25Deciles"], sorted(cx["p25Deciles"]))
        counts = np.array(cx["density"]["counts"])
        self.assertTrue(((counts == 0) | (counts >= KC.MIN_CELL)).all())
        self.assertEqual(int(counts.sum()) + cx["density"]["suppressedRows"], cx["n"])
        self.assertEqual(cx["density"]["yEdges"], KC.Y_EDGES)
        js = json.dumps(self.knn, ensure_ascii=False)
        pool_only = set(self.codes) - {c["jqCode"] for c in self.cands}
        for code in list(pool_only)[:40]:
            self.assertNotIn(code, js)                                       # 過去の事例の銘柄コードは出ない
        self.assertNotIn("2025-06-25", js)                                   # 過去の事例の日付も出ない
        self.assertNotIn("2024-", js)

    def test_summary_line(self):
        s = KC.summary_line(self.knn, ["2025-06-30", DAY])
        self.assertIn("2025-06-30 no_candidates", s)
        self.assertIn(f"{DAY} 4件", s)
        self.assertIn("文脈", s)
        self.assertEqual(KC.summary_line(None, [DAY]), "近傍の集計なし")


class TestCarryOverAndFailure(unittest.TestCase):
    PREV = {"byDay": {"2025-06-27": {"status": "ok", "poolN": 5, "n": 1, "candidates": {"10010": {"min15": 96.0}}},
                      "2025-06-30": {"status": "ok", "poolN": 5, "n": 0, "candidates": {}}},
            "context": {"n": 9, "p25Deciles": list(range(11)), "density": {}}}

    def test_frozen_days_keep_previous_and_scored_day_is_marked_unavailable_without_oof(self):
        empty_model = tempfile.mkdtemp()
        empty_data = tempfile.mkdtemp()
        try:
            out = KC.build([cand("10010", DAY, [97.0, 98.0, 96.0])], ["2025-06-27", "2025-06-30", DAY],
                           empty_data, empty_model, prev=self.PREV, frozen=["2025-06-27", "2025-06-30"])
        finally:
            shutil.rmtree(empty_model, ignore_errors=True)
            shutil.rmtree(empty_data, ignore_errors=True)
        self.assertEqual(out["byDay"]["2025-06-27"], self.PREV["byDay"]["2025-06-27"])
        self.assertEqual(out["byDay"]["2025-06-30"], self.PREV["byDay"]["2025-06-30"])
        self.assertEqual(out["byDay"][DAY]["status"], "unavailable")
        self.assertIn("out-of-fold", out["byDay"][DAY]["reason"])
        self.assertEqual(out["context"], self.PREV["context"])              # 作れなければ前の文脈のまま
        self.assertEqual(out["k"], 50)
        json.dumps(out, ensure_ascii=False)

    def test_all_days_frozen_only_carries_over(self):
        out = KC.build([], ["2025-06-27"], "/nonexistent", "/nonexistent", prev=self.PREV, frozen=["2025-06-27"])
        self.assertEqual(list(out["byDay"]), ["2025-06-27"])
        self.assertEqual(out["byDay"]["2025-06-27"]["n"], 1)

    def test_frozen_day_missing_in_previous_file_is_left_out(self):
        out = KC.build([], ["2025-06-26", "2025-06-27"], "/nonexistent", "/nonexistent", prev=self.PREV,
                       frozen=["2025-06-26", "2025-06-27"])
        self.assertNotIn("2025-06-26", out["byDay"])
        self.assertIn("2025-06-27", out["byDay"])

    def test_candidate_min15(self):
        self.assertEqual(KC.candidate_min15(cand("1", DAY, [97.0, 98.5, 96.2])), 96.2)
        self.assertIsNone(KC.candidate_min15(cand("1", DAY, [97.0, None, 96.2])))
        self.assertIsNone(KC.candidate_min15({"jqCode": "1"}))


if __name__ == "__main__":
    unittest.main()
