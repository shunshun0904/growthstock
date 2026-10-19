#!/usr/bin/env python3
"""
候補ごとのリカレンス・ネットワーク（research/recurrence_net.py。画面の「リカレンス・ネットワーク」タブに渡す）。データには触らない
（作りものの日足）。

- 状態: 列の型（linear_preprocess）どおりに変換して、その銘柄のノード全体で平均0・標準偏差1。|z| は Z_CAP で切る。丸めは 1/Q σ
- 距離の2乗は整数で、素朴に数えた値と一致する。ε² は結べる組（タイラー窓より離れた組）の距離の2乗の m 番目
  （m = 密度 × 組の数 を四捨五入。0.5 は切り上げ）。結べる組が無ければ辺なし
- 辺はタイラー窓の中を結ばない・対称・ε² 以下は同じ値も全部入る
- 要約: 今日の次数・中位の百分位・近傍の結果（分かった日だけで数える）・近い更新日の並び（ε の外も近い順）が、小さな例で手で数えた値と一致
- 先読みしない: 予測日より後の日足があってもなくても、同じファイルになる。目的変数は最後の20営業日が未確定、+10% の到達は予測日までの
  高値だけで決まる
- 結果は live_track.forward・build_dataset の目的変数（1.2σ・1.5σ）・78週高値の更新日と同じ
- 作れない候補は理由を残して飛ばす（日足が無い・ノードが少ない）
- 作るのは購入条件（1.5σ の3本の百分位がすべて95以上）を満たした銘柄だけ。百分位が1本でも欠けた銘柄は通さない
  （news_links.rule_codes・画面の strategy.js と同じ）。並びは最小の降順・同点はスコアの降順。日の項目に高値更新の件数（nBreak）
- 前の archive にある日は引き継ぐ（候補の組が変わった日は作り直す）。archive を書いて読み戻せる（突き合わせ用の丸めない値は書かない）。
  出力フォルダの自分のファイル以外は消さない
- フィクスチャ（tests/fixtures/recurrence/ と recurrence-expected.json）は今の式で数え直した値と一致する（画面の
  src/lib/recurrence.js も tests/recurrence.test.js で同じフィクスチャを確かめる）

  python3 tests/test_recurrence_net.py
  python3 tests/test_recurrence_net.py --write-fixtures   # 式を変えたらフィクスチャを作り直す
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

import build_dataset as B  # noqa: E402
import linear_preprocess as LP  # noqa: E402
import live_track as LT  # noqa: E402
import recurrence_net as RN  # noqa: E402

FIX_DIR = os.path.join(ROOT, "tests", "fixtures", "recurrence")
FIX_EXPECTED = os.path.join(ROOT, "tests", "fixtures", "recurrence-expected.json")
PRED_FIXTURE = os.path.join(ROOT, "tests", "fixtures", "synthetic-predictions.json")
FIX_DAY = "2026-09-11"     # tests/fixtures/synthetic-predictions.json と同じ日・同じ銘柄コード（日足は作りもの）
#: 合成の予測ファイルの5件には購入条件を満たす銘柄が無いので、3件の 1.5σ の3本（RULE_MODELS の順）の百分位を上書きする
#: （tests/smoke.mjs の PASS と同じ値）。9082 は日足を作らない（購入条件は満たすが「日足が無い」で作れない例）
FIX_PASS = {"9324": [98.1, 97.1, 97.4], "9434": [97.1, 96.5, 95.2], "9082": [96.0, 100.0, 95.0]}
FIX_CANDS = [{"code": "9324", "jqCode": "93240", "name": "安田倉庫", "date": FIX_DAY, "n": 520, "seed": 11},
             {"code": "9434", "jqCode": "94340", "name": "ソフトバンク", "date": FIX_DAY, "n": 700, "seed": 12}]


MAX_AFTER = 60


def make_bars(code: str, n: int, seed: int, end: str = FIX_DAY, extra_after: int = 0) -> pd.DataFrame:
    """
    静かな時期と荒い時期が交互に来る作りものの日足（end で終わる n 営業日 + その後 extra_after 営業日）。
    乱数はいつも先の MAX_AFTER 営業日まで引いてから切る（extra_after を変えても end までの値は同じ）。
    """
    assert extra_after <= MAX_AFTER
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=end, periods=n).append(pd.bdate_range(pd.Timestamp(end) + pd.Timedelta(days=1),
                                                                     periods=MAX_AFTER))
    m = len(days)
    vol = np.where((np.arange(m) // 90) % 2 == 0, 0.012, 0.03)
    r = rng.normal(0.0006, vol)
    close = 1000.0 * np.exp(np.cumsum(r))
    o = close * np.exp(rng.normal(0.0, 0.004, m))
    h = np.maximum(o, close) * (1.0 + np.abs(rng.normal(0.0, 0.008, m)))
    lo = np.minimum(o, close) * (1.0 - np.abs(rng.normal(0.0, 0.008, m)))
    v = np.exp(rng.normal(11.0, 0.5, m))
    df = pd.DataFrame({"Date": days, "Code": code, "O": o, "H": h, "L": lo, "C": close, "Vo": v,
                       "AdjO": o, "AdjH": h, "AdjL": lo, "AdjC": close, "AdjVo": v})
    return df.iloc[:n + extra_after].reset_index(drop=True)


def fixture_bars(extra_after: int = 0) -> pd.DataFrame:
    parts = [make_bars(c["jqCode"], c["n"], c["seed"], extra_after=extra_after) for c in FIX_CANDS]
    return pd.concat(parts, ignore_index=True).sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)


def with_pcts(c: dict, pcts) -> dict:
    """候補に規則の3本の百分位（RULE_MODELS の順）を付けた写し。pcts の None はその1本が欠けた候補。"""
    return dict(c, byModel={a: ({} if v is None else {"pctHistorical": v}) for a, v in zip(RN.RULE_MODELS, pcts)})


def fixture_pred() -> dict:
    """合成の予測ファイル（2026-09-11 の5件）に FIX_PASS を上書きしたもの（tests/smoke.mjs の predPass と同じ）。"""
    with open(PRED_FIXTURE, encoding="utf-8") as fh:
        pred = json.load(fh)
    for c in pred["candidates"]:
        for a, v in zip(RN.RULE_MODELS, FIX_PASS.get(c["code"], [])):
            c["byModel"][a]["pctHistorical"] = v
    return pred


def write_bars(bars: pd.DataFrame, data_dir: str) -> None:
    """日足を年ごとの bars_<年>.parquet に書く（load_bars が読む形）。"""
    os.makedirs(data_dir, exist_ok=True)
    for y, g in bars.groupby(bars["Date"].dt.year):
        g.to_parquet(os.path.join(data_dir, f"bars_{y}.parquet"), index=False)


def fixture_build():
    """フィクスチャと同じ作り: 合成の予測ファイル（fixture_pred）と作りものの日足（9324・9434 だけ）で build を通す。"""
    tmp = tempfile.mkdtemp()
    try:
        write_bars(fixture_bars(), tmp)
        index, files = RN.build(fixture_pred(), tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    index["generatedAt"] = "2026-09-11T10:00:00+00:00"
    for d in index["days"]:
        d["builtAt"] = "2026-09-11T10:00:00+00:00"
    return index, files


def decode(p: dict):
    """候補のファイル → (zq, ret, labels, fresh, dates)。画面の recurrence.js の decode と同じ読み方。"""
    n = p["n"]
    zq = np.asarray(p["z"], dtype=np.int64).reshape(n, len(p["features"]))
    ret = np.array([np.nan if v is None else float(v) for v in p["ret"]], dtype=float)
    labels = {key: np.array([np.nan if v is None else float(v) for v in p[key]], dtype=float) for key, _ in RN.LABEL_KS}
    d = [pd.Timestamp(p["start"])]
    for g in p["gaps"]:
        d.append(d[-1] + pd.Timedelta(days=int(g)))
    dates = [x.date().isoformat() for x in d]
    return zq, ret, labels, p["fresh"], dates


def make_fixtures(out_dir: str = FIX_DIR, expected_path: str = FIX_EXPECTED) -> None:
    """フィクスチャ（index.json・銘柄ごとのファイル・密度ごとの要約）を作り直す。"""
    index, files = fixture_build()
    RN.write(index, files, out_dir, None)
    expected = {}
    for f, p in files.items():
        zq, ret, labels, fresh, dates = decode({k: v for k, v in p.items() if not k.startswith("_")})
        expected[f] = {str(bp): RN.round_summary(RN.summarize(zq, ret, labels, fresh, dates, rr_bp=bp))
                       for bp in RN.RR_CHOICES_BP}
    with open(expected_path, "w", encoding="utf-8") as fh:
        json.dump(expected, fh, ensure_ascii=False, indent=1)


class TestStates(unittest.TestCase):
    def test_transform_by_kind(self):
        rng = np.random.default_rng(0)
        n = 800
        X = np.column_stack([
            np.exp(rng.normal(0.5, 0.6, n)),          # tx_vol_60（正で裾が重い）
            rng.uniform(5, 45, n),                     # tx_mdi_14（正・有界）
            rng.normal(1.0, 0.15, n),                  # tx_vortex_pos_14
            np.exp(rng.normal(0.0, 0.3, n)),           # tx_atr_ratio_14_50
            rng.standard_t(2, n) * 3,                  # tx_ichi_kijun_gap（符号つきで裾が重い）
            np.exp(rng.normal(2.5, 0.5, n)),           # tx_bb_width
            rng.normal(0, 20, n),                      # tx_tsi（符号つき）
            rng.uniform(0, 100, n),                    # tx_stoch_k
            np.exp(rng.normal(0.0, 0.4, n)),           # tx_vol_ratio_60_250
        ])
        Z, params = RN.transform_states(X)
        self.assertEqual([p["col"] for p in params], list(RN.FEATURES))
        self.assertEqual([p["kind"] for p in params], [LP.kind_of(c) for c in RN.FEATURES])
        self.assertLessEqual(np.abs(Z).max(), RN.Z_CAP)
        # 手で同じ変換をした値と一致する
        for j, col in enumerate(RN.FEATURES):
            x = X[:, j]
            kind = LP.kind_of(col)
            if kind == "pos_heavy":
                t = np.log1p(np.clip(x, None, np.quantile(x, 0.995)))
            elif kind == "signed_heavy":
                lo, hi = np.quantile(x, [0.005, 0.995])
                c = np.clip(x, lo, hi)
                q1, q3 = np.quantile(c, [0.25, 0.75])
                t = np.arcsinh(c / ((q3 - q1) / 2))
            else:
                lo, hi = np.quantile(x, [0.001, 0.999])
                t = np.clip(x, lo, hi)
            want = np.clip((t - t.mean()) / t.std(), -RN.Z_CAP, RN.Z_CAP)
            np.testing.assert_allclose(Z[:, j], want, rtol=1e-10, atol=1e-10, err_msg=col)
            self.assertAlmostEqual(params[j]["mean"], float(t.mean()), places=10)
            self.assertAlmostEqual(params[j]["std"], float(t.std()), places=10)

    def test_constant_column_does_not_break(self):
        X = np.random.default_rng(1).normal(size=(200, len(RN.FEATURES)))
        X[:, 1] = 7.0
        Z, params = RN.transform_states(X)
        self.assertTrue(np.all(Z[:, 1] == 0))
        self.assertEqual(params[1]["std"], 1.0)

    def test_quantize(self):
        z = np.array([[0.0, 0.024, 0.026, -0.026, 6.0, -6.0, 1.0, -1.0, 0.075]])
        self.assertEqual(RN.quantize(z).tolist(), [[0, 0, 1, -1, 120, -120, 20, -20, 2]])


class TestEdges(unittest.TestCase):
    def test_sq_dist_matches_bruteforce(self):
        a = np.random.default_rng(2).integers(-120, 121, size=(40, 9))
        s = RN.sq_dist(a)
        for i in range(40):
            for j in range(40):
                self.assertEqual(int(s[i, j]), int(((a[i] - a[j]) ** 2).sum()))

    def test_threshold_and_adjacency(self):
        rng = np.random.default_rng(3)
        a = rng.integers(-40, 41, size=(30, 9))
        s = RN.sq_dist(a)
        th = RN.threshold(s, theiler=3, rr_bp=1000)
        vals = sorted(int(s[i, j]) for i in range(30) for j in range(i + 4, 30))
        self.assertEqual(th["pairs"], len(vals))
        self.assertEqual(th["m"], (1000 * len(vals) + 5000) // 10000)
        self.assertEqual(th["t"], vals[th["m"] - 1])
        A = RN.adjacency(s, th["t"], theiler=3)
        self.assertTrue(np.array_equal(A, A.T))
        self.assertFalse(A.diagonal().any())
        for i in range(30):
            for j in range(30):
                self.assertEqual(bool(A[i, j]), abs(i - j) > 3 and s[i, j] <= th["t"])
        self.assertEqual(int(A.sum() // 2), sum(v <= th["t"] for v in vals))   # 同じ値は全部入る

    def test_threshold_rounds_half_up(self):
        # 組の数 300（n = 27、タイラー窓 2: (27−3)(27−2)/2 = 300）、密度 50bp → 1.5 → 2
        s = RN.sq_dist(np.random.default_rng(4).integers(-5, 6, size=(27, 9)))
        th = RN.threshold(s, theiler=2, rr_bp=50)
        self.assertEqual((th["pairs"], th["m"]), (300, 2))

    def test_too_few_nodes(self):
        s = RN.sq_dist(np.zeros((5, 9), dtype=int))
        th = RN.threshold(s, theiler=20)
        self.assertEqual(th, {"t": -1, "pairs": 0, "m": 0})
        self.assertFalse(RN.adjacency(s, th["t"], theiler=20).any())


class TestSummary(unittest.TestCase):
    def test_hand_example(self):
        # 1列目だけが動く8ノード（ほかの列は 0）。タイラー窓 1・密度 3333bp（結べる組 21 → m = (3333×21 + 5000) // 10000 = 7）
        x = np.array([0, 10, 1, 30, 2, 31, 9, 0])
        zq = np.zeros((8, 9), dtype=int)
        zq[:, 0] = x
        ret = np.array([10.0, -3.0, np.nan, 2.5, 10.0, -12.0, 0.0, np.nan])
        labels = {"l12": np.array([1, 0, np.nan, 0, 1, 0, 0, np.nan], dtype=float),
                  "l15": np.array([1, 0, np.nan, 0, 0, 0, 0, np.nan], dtype=float)}
        fresh = [0, 4, 5, 6]
        dates = [f"2026-01-{d:02d}" for d in range(5, 13)]
        s = RN.summarize(zq, ret, labels, fresh, dates, theiler=1, rr_bp=3333, q=1)
        # 結べる組（|i−j| > 1）の距離の2乗を小さい順: 0（0-7）, 1（0-2・1-6・2-4・2-7・3-5）, 4（0-4・4-7）, 49, ... → 7 番目 = 4
        self.assertEqual((s["pairs"], s["m"], s["eps2"]), (21, 7, 4))
        self.assertAlmostEqual(s["eps"], 2.0)
        # 辺（4 以下）: 0-7・0-2・1-6・2-4・2-7・3-5・0-4・4-7
        self.assertEqual(s["edges"], 8)
        # 今日 = 7（x = 0）: 0（0）・2（1）・4（4）。5 は 31² = 961 で遠い。6 はタイラー窓の中
        self.assertEqual(s["deg"], 3)
        deg = np.array([3, 1, 3, 1, 3, 1, 1, 3])
        self.assertAlmostEqual(s["degPct"], 100.0 * (np.sum(deg < 3) + 0.5 * np.sum(deg == 3)) / 8)
        self.assertEqual(s["nbr"]["n"], 3)
        self.assertEqual(s["nbr"]["known"], 2)                  # 2 は結果が分からない
        self.assertAlmostEqual(s["nbr"]["hit"], 1.0)            # 0 と 4 はどちらも +10
        self.assertAlmostEqual(s["nbr"]["retMed"], 10.0)
        self.assertAlmostEqual(s["nbr"]["l15"], 0.5)
        self.assertEqual(s["nbr"]["l15N"], 2)
        self.assertEqual(s["base"]["known"], 6)
        self.assertAlmostEqual(s["base"]["hit"], 2 / 6)
        self.assertAlmostEqual(s["base"]["retMed"], (0.0 + 2.5) / 2)
        self.assertEqual(s["nbrFresh"]["n"], 2)                 # 0 と 4
        self.assertEqual(s["baseFresh"]["n"], 4)
        # 近い更新日: 今日（7）より 1 日より前の更新日 0, 4, 5 を近い順（6 はタイラー窓の中）
        self.assertEqual([r["i"] for r in s["nearestFresh"]], [0, 4, 5])
        self.assertEqual([r["inEps"] for r in s["nearestFresh"]], [True, True, False])
        self.assertEqual(s["nearestFresh"][0]["date"], "2026-01-05")
        self.assertEqual(s["nearestFresh"][2]["ret"], -12.0)
        self.assertEqual(s["nearestFresh"][1]["l12"], 1)

    def test_isolated_today(self):
        zq = np.zeros((60, 9), dtype=int)
        zq[:, 0] = np.arange(60) % 7
        zq[-1, 0] = 100                                          # 今日だけ遠い
        s = RN.summarize(zq, np.full(60, np.nan), {"l12": np.full(60, np.nan), "l15": np.full(60, np.nan)},
                         [], [f"d{i}" for i in range(60)], theiler=5, rr_bp=200)
        self.assertEqual(s["deg"], 0)
        self.assertEqual(s["nbr"], {"n": 0, "known": 0, "hit": None, "retMed": None,
                                    "l12": None, "l12N": 0, "l15": None, "l15N": 0})
        self.assertEqual(s["nearestFresh"], [])


class TestRule(unittest.TestCase):
    """対象は購入条件（1.5σ の3本の百分位がすべて95以上）を満たした銘柄だけ（運用者の指定 2026-10-19）。"""

    def test_rule_min(self):
        c = FIX_CANDS[0]
        self.assertEqual(RN.rule_min(with_pcts(c, [98.1, 97.1, 97.4])), 97.1)
        self.assertEqual(RN.rule_min(with_pcts(c, [95, 95, 95])), 95.0)
        self.assertIsNone(RN.rule_min(with_pcts(c, [99, None, 99])))          # 1本でも欠ければ出さない
        self.assertIsNone(RN.rule_min(with_pcts(c, [99, float("nan"), 99])))
        self.assertIsNone(RN.rule_min(with_pcts(c, [99, True, 99])))          # 真偽値は数として扱わない
        self.assertIsNone(RN.rule_min(with_pcts(c, [99, "99", 99])))
        self.assertIsNone(RN.rule_min(dict(c)))                                # byModel が無い
        self.assertIsNone(RN.rule_min(dict(c, byModel={"lgbm_k15": 99})))      # 形が違う

    def test_rule_candidates_order_and_line(self):
        rows = [with_pcts(dict(FIX_CANDS[0], code=k, score=sc), p) for k, sc, p in [
            ("A", 0.4, [96, 97, 98]), ("B", 0.9, [99, 94.99, 99]), ("C", 0.5, [96, 99, 99]),
            ("D", 0.7, [96, 99, 99]), ("E", 0.1, [95, 95, 95]), ("F", 0.8, [None, 99, 99])]]
        got = RN.rule_candidates(rows)
        # 最小の降順（C・D・A が 96、E が 95）。同点はスコアの降順（D 0.7 > C 0.5 > A 0.4）
        self.assertEqual([r["code"] for r in got], ["D", "C", "A", "E"])
        self.assertEqual([r["code"] for r in RN.rule_candidates(rows, line=96.5)], [])
        self.assertEqual(RN.rule_candidates([]), [])

    def test_same_as_news_links(self):
        # 候補銘柄のニュース（news_links.rule_codes）と同じ銘柄・同じ日を選ぶ（合成の予測ファイルに上書きしたもの）
        import news_links as NL
        pred = fixture_pred()
        want = NL.rule_codes(pred)
        got: dict = {}
        for d in pred["dates"]:
            for c in RN.rule_candidates([c for c in pred["candidates"] if c["date"] == d]):
                got.setdefault(c["code"], []).append(d)
        self.assertEqual(got, want)
        self.assertEqual(sorted(got), sorted(FIX_PASS))
        self.assertEqual([c["code"] for c in RN.rule_candidates(pred["candidates"])], ["9324", "9434", "9082"])


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, bars, day=FIX_DAY, cands=None):
        return RN.build_day(day, [dict(c) for c in (cands or FIX_CANDS)], bars, None)

    def test_no_lookahead(self):
        blk1, f1 = self.build(fixture_bars())
        blk2, f2 = self.build(fixture_bars(extra_after=40))     # 予測日より後の40営業日（上げても下げても）
        self.assertEqual(sorted(f1), sorted(f2))
        for f in f1:
            self.assertEqual(json.dumps(f1[f], sort_keys=True), json.dumps(f2[f], sort_keys=True), f)
        self.assertEqual([c["summary"] for c in blk1["candidates"]], [c["summary"] for c in blk2["candidates"]])

    def test_outcomes_match_sources(self):
        bars = fixture_bars(extra_after=30)
        cut = bars[bars["Date"] <= FIX_DAY]
        _, files = self.build(bars)
        panel = RN.build_panel(cut, None)
        for c in FIX_CANDS:
            p = files[RN.file_name(FIX_DAY, c["jqCode"])]
            zq, ret, labels, fresh, dates = decode(p)
            self.assertEqual(dates[-1], FIX_DAY)
            self.assertEqual(len(dates), p["n"])
            g = panel[panel["Code"] == c["jqCode"]].copy()
            g = g[np.isfinite(g[list(RN.FEATURES)].to_numpy(dtype=float)).all(axis=1)]
            self.assertEqual([d.date().isoformat() for d in g["Date"]], dates)
            fw = LT.forward(cut[cut["Code"] == c["jqCode"]], pd.DataFrame({"Code": c["jqCode"], "Date": g["Date"].to_numpy()}))
            want = np.round(pd.to_numeric(fw["ret"], errors="coerce").to_numpy(dtype=float), 1)
            np.testing.assert_array_equal(np.isnan(want), np.isnan(ret))
            np.testing.assert_allclose(want[~np.isnan(want)], ret[~np.isnan(ret)])
            self.assertTrue(np.isnan(ret[-1]))                      # 今日は翌営業日の寄りがまだ無い
            for key, k in RN.LABEL_KS:
                lab = B.rise_label_for_k(g, k).to_numpy(dtype=float)
                np.testing.assert_array_equal(np.isnan(lab), np.isnan(labels[key]))
                np.testing.assert_array_equal(lab[~np.isnan(lab)], labels[key][~np.isnan(labels[key])])
                self.assertTrue(np.isnan(labels[key][-B.RISE_HORIZON:]).all())   # 最後の20営業日は未確定
            self.assertEqual(fresh, np.flatnonzero(g["is_fresh_break"].fillna(False).to_numpy(dtype=bool)).tolist())
            np.testing.assert_allclose(p["_raw"], g[list(RN.FEATURES)].to_numpy(dtype=float)[-1])
            # 1.2σ の目的変数は本番のラベル（attach_rise_label の label）と同じ
            base = g["label"].to_numpy(dtype=float)
            np.testing.assert_array_equal(np.isnan(base), np.isnan(labels["l12"]))

    def test_skips_with_reason(self):
        bars = fixture_bars()
        short = make_bars("88880", 300, 5)                         # 9列がそろうのは約50日だけ
        cands = FIX_CANDS + [{"code": "8888", "jqCode": "88880", "name": "短い", "date": FIX_DAY},
                             {"code": "7777", "jqCode": "77770", "name": "日足なし", "date": FIX_DAY}]
        blk, files = self.build(pd.concat([bars, short], ignore_index=True), cands=cands)
        self.assertEqual(blk["status"], "partial")
        self.assertEqual(len(blk["candidates"]), 2)
        why = {s["jqCode"]: s["reason"] for s in blk["skipped"]}
        self.assertIn("ノードが", why["88880"])
        self.assertIn("日足が無い", why["77770"])

    def test_build_carry_and_archive(self):
        data_dir = os.path.join(self.tmp, "data")
        write_bars(fixture_bars(), data_dir)
        prev_day = "2026-09-04"
        ok1, ok2 = with_pcts(FIX_CANDS[0], [98, 97, 96]), with_pcts(FIX_CANDS[1], [95, 99, 99])
        below = with_pcts(FIX_CANDS[1], [99, 99, 94.9])                 # 1本が95未満 → 作らない（日足はある）
        pred = {"asOf": FIX_DAY, "dates": [prev_day, FIX_DAY],
                "candidates": [ok1, ok2, dict(below, date=prev_day), dict(ok1, date=prev_day)]}
        arch = os.path.join(self.tmp, "recurrence.tar.gz")
        out = os.path.join(self.tmp, "out")
        os.makedirs(out)
        with open(os.path.join(out, "keep.txt"), "w") as fh:
            fh.write("x")
        with open(os.path.join(out, "2020-01-01_11110.json"), "w") as fh:
            fh.write("{}")
        index, files = RN.build(pred, data_dir, prev_archive=arch)
        self.assertEqual([d["date"] for d in index["days"]], [prev_day, FIX_DAY])
        self.assertEqual([len(d["candidates"]) for d in index["days"]], [1, 2])
        self.assertEqual([d["nBreak"] for d in index["days"]], [2, 2])               # 満たさなかった銘柄も数える
        self.assertEqual(index["days"][0]["skipped"], [])                           # 満たさない銘柄は作れない扱いにもしない
        self.assertEqual(index["method"]["rule"], {"models": list(LT.RULE_MODELS), "agreePct": LT.AGREE_PCT})
        RN.write(index, files, out, arch)
        self.assertTrue(os.path.exists(os.path.join(out, "keep.txt")))              # 自分のファイル以外は残す
        self.assertFalse(os.path.exists(os.path.join(out, "2020-01-01_11110.json")))  # 前の自分のファイルは消す
        self.assertEqual(sorted(os.listdir(out)), sorted(["keep.txt", "index.json", *files]))
        idx2, files2 = RN.read_archive(arch)
        self.assertEqual(idx2["days"], json.loads(json.dumps(index["days"])))
        for f, p in files.items():
            self.assertNotIn("_raw", files2[f])
            self.assertEqual(files2[f], json.loads(json.dumps({k: v for k, v in p.items() if not k.startswith("_")})))
        with open(os.path.join(out, next(iter(files))), encoding="utf-8") as fh:
            self.assertNotIn("_raw", json.load(fh))
        # 2回目: 前の archive から両日を引き継ぐ（中身は同じ）
        index3, files3 = RN.build(pred, data_dir, prev_archive=arch)
        self.assertEqual([d.get("carried") for d in index3["days"]], [True, True])
        self.assertEqual(files3, files2)
        # 最新の日に購入条件を満たす銘柄の組が変わったら、その日だけ作り直す（満たさない銘柄が増えても作り直さない）
        pred2 = dict(pred, candidates=[ok1, dict(ok2, byModel={}), dict(below, date=prev_day), dict(ok1, date=prev_day),
                                       dict(below, date=prev_day, code="9999", jqCode="99990")])
        index4, files4 = RN.build(pred2, data_dir, prev_archive=arch)
        self.assertEqual([d.get("carried") for d in index4["days"]], [True, None])
        self.assertEqual([d["nBreak"] for d in index4["days"]], [3, 2])
        self.assertEqual(len(index4["days"][1]["candidates"]), 1)
        # 購入条件を満たす銘柄が無い日は「no_candidates」（高値更新の件数は残す）
        pred3 = dict(pred, dates=[prev_day], candidates=[dict(below, date=prev_day)])
        index5, files5 = RN.build(pred3, data_dir)
        self.assertEqual((index5["days"][0]["status"], index5["days"][0]["nBreak"], files5), ("no_candidates", 1, {}))
        # 版が違う archive は使わない
        bad = os.path.join(self.tmp, "bad.tar.gz")
        RN.write(dict(index, v=RN.VERSION + 1), files, None, bad)
        self.assertEqual(RN.read_archive(bad), (None, {}))
        self.assertEqual(RN.read_archive(os.path.join(self.tmp, "none.tar.gz")), (None, {}))

    def test_dataset_check(self):
        _, files = self.build(fixture_bars())
        ds = []
        for f, p in files.items():
            ds.append({"Code": p["jqCode"], "Date": pd.Timestamp(p["date"]), **dict(zip(RN.FEATURES, p["_raw"]))})
        df = pd.DataFrame(ds)
        path = os.path.join(self.tmp, "dataset_predict.parquet")
        df.to_parquet(path, index=False)
        self.assertEqual(RN.check_dataset(files, path), {"checked": 2, "mismatch": 0, "examples": [], "columns": []})
        df.loc[0, "tx_tsi"] += 1.0
        df.to_parquet(path, index=False)
        chk = RN.check_dataset(files, path)
        self.assertEqual((chk["checked"], chk["mismatch"]), (2, 1))
        self.assertIn("tx_tsi", chk["examples"][0])
        self.assertEqual(chk["columns"], ["tx_tsi"])                                   # ログに出すのは列だけ（銘柄コードは出さない）
        self.assertIsNone(RN.check_dataset(files, os.path.join(self.tmp, "none.parquet")))


class TestFixtures(unittest.TestCase):
    """画面（src/lib/recurrence.js）と同じフィクスチャ。式を変えたら --write-fixtures で作り直す。"""

    def test_fixture_summaries(self):
        with open(os.path.join(FIX_DIR, "index.json"), encoding="utf-8") as fh:
            index = json.load(fh)
        with open(FIX_EXPECTED, encoding="utf-8") as fh:
            expected = json.load(fh)
        files = [c["file"] for d in index["days"] for c in d["candidates"]]
        self.assertEqual(sorted(files), sorted(expected))
        self.assertGreaterEqual(len(files), 2)
        for f in files:
            with open(os.path.join(FIX_DIR, f), encoding="utf-8") as fh:
                p = json.load(fh)
            zq, ret, labels, fresh, dates = decode(p)
            self.assertEqual(dates[-1], p["date"])
            for bp, want in expected[f].items():
                got = RN.round_summary(RN.summarize(zq, ret, labels, fresh, dates, rr_bp=int(bp)))
                self.assertEqual(got, want, f"{f} {bp}bp")
            self.assertEqual(p["summary"], expected[f][str(RN.RR_BP)])

    def test_fixture_format(self):
        # フィクスチャの項目は今の作り（build_day）が書く項目と同じ。値そのものは比べない（作りものの日足の exp・log は
        # CPU の命令で最後の桁が変わりうるので、丸めた状態の整数が境目で1つずれることがある）
        built, files = fixture_build()
        with open(os.path.join(FIX_DIR, "index.json"), encoding="utf-8") as fh:
            index = json.load(fh)
        self.assertEqual(index["v"], RN.VERSION)
        self.assertEqual(index["method"], json.loads(json.dumps(RN.method_block())))
        blk = built["days"][0]
        self.assertEqual(sorted(index["days"][0]), sorted(blk))
        # 購入条件を満たした3件のうち、日足のある2件を作り、9082 は「日足が無い」（予測ファイルの5件は nBreak）
        for got in (index["days"][0], blk):
            self.assertEqual([c["jqCode"] for c in got["candidates"]], ["93240", "94340"])
            self.assertEqual([(x["jqCode"], x["reason"]) for x in got["skipped"]], [("90820", "日足が無い")])
            self.assertEqual((got["status"], got["nBreak"]), ("partial", 5))
        for f, p in files.items():
            with open(os.path.join(FIX_DIR, f), encoding="utf-8") as fh:
                q = json.load(fh)
            self.assertEqual(sorted(q), sorted(k for k in p if not k.startswith("_")), f)
            self.assertEqual(sorted(q["summary"]), sorted(p["summary"]), f)


if __name__ == "__main__":
    if "--write-fixtures" in sys.argv:
        make_fixtures()
        print(f"書いた: {FIX_DIR} / {FIX_EXPECTED}")
        sys.exit(0)
    unittest.main()
