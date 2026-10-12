#!/usr/bin/env python3
"""
実験89（research/exp/e89_split_trail.py）。

後半1単位のトレーリングの約定の仮定をここで固定する（実験41 の逆指値と同じ保守側）。
  - k日目の線は「k−1日目までの高値の最高値（1日目は買値）×（1 − W%）」。k日目の高値で線を上げるのは翌日から
    （同じ日に高値を更新してから線まで下げた分は数えない）
  - 寄りで線を割って始まったら寄値、場中に安値が線に触れたら線ちょうど
  - 値が欠測の日は判定しない（高値の最高値も変えない）
  - 上限（40営業日）まで掛からなければ上限の日の終値。上限の日の足が無い行は NaN（途中）
  - 幅が狭いほど早く（同じ日か前に）売る
- 分割の収益は前半 tp10c と後半の平均。3枠の模擬に使う売った日の列
- 同じ行の並べ替え検定: 差が無ければ p は小さくならず、仕込んだ差は拾う（銘柄単位でも）
- 3枠の模擬: 1単位の出口は live_track.simulate と同じ取引を買う。分割は1銘柄で2単位を使い、資金6単位なら3単位より
  多く買える。月の収益と複利の資金の数え方
- 学習し直しの配線（学習器は差し替え）: 6本を横に並べ、1.5σ のモデルにはラベルを引き直して渡す。保存済みは読む
- 報告に銘柄コードが出ない

  python3 tests/test_e89_split_trail.py
"""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e89_split_trail as E  # noqa: E402
import test_e82_and_rule_variants as T82  # noqa: E402

NAN = float("nan")
QUIET = (100.0, 101.0, 99.0, 100.0)
HAVE_PARQUET = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
HAVE_GBM = importlib.util.find_spec("lightgbm") is not None


def trail(path, w=10.0, entry=None):
    a = np.array(path, dtype=float)
    o, h, lo, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    return E.trail_one(o, h, lo, c, float(o[0] if entry is None else entry), w)


class 約定の仮定(unittest.TestCase):

    def test_寄りで線を割ったら寄値(self):
        r, day, fired, gap = trail([QUIET, (85.0, 86.0, 84.0, 85.5), QUIET])
        self.assertAlmostEqual(r, -15.0)                 # 線は 101 × 0.9 = 90.9 だが、寄りの 85 で約定
        self.assertEqual((day, fired, gap), (2, True, True))

    def test_場中に触れたら線ちょうど(self):
        r, day, fired, gap = trail([QUIET, (100.0, 100.5, 90.0, 95.0), QUIET])
        self.assertAlmostEqual(r, -9.1)                  # 101 × 0.9 = 90.9
        self.assertEqual((day, fired, gap), (2, True, False))

    def test_高値の最高値で線が上がる(self):
        r, day, fired, _ = trail([(100.0, 120.0, 99.0, 118.0), (118.0, 119.0, 107.5, 110.0), QUIET])
        self.assertAlmostEqual(r, 8.0)                   # 120 × 0.9 = 108
        self.assertEqual((day, fired), (2, True))

    def test_同じ日の高値更新は翌日から(self):
        # 1日目に 120 まで上げて 95 まで下げても、1日目の線は買値から（90）なので売らない。2日目は線 108 を寄りで割る
        r, day, fired, gap = trail([(100.0, 120.0, 95.0, 100.0), (100.0, 101.0, 99.5, 100.0), QUIET])
        self.assertAlmostEqual(r, 0.0)
        self.assertEqual((day, fired, gap), (2, True, True))

    def test_欠測の日は判定しない(self):
        r, day, fired, _ = trail([QUIET, (NAN, NAN, NAN, NAN), (100.0, 130.0, 99.0, 128.0), (128.0, 129.0, 116.0, 120.0)])
        self.assertAlmostEqual(r, 17.0)                  # 130 × 0.9 = 117
        self.assertEqual((day, fired), (4, True))

    def test_上限まで掛からなければ上限の日の終値(self):
        path = [QUIET] * 39 + [(100.0, 101.0, 99.0, 103.0)]
        for w in E.WIDTHS:
            r, day, fired, gap = trail(path, w)
            self.assertAlmostEqual(r, 3.0)
            self.assertEqual((day, fired, gap), (40, False, False))
        r, _, _, _ = trail([QUIET] * 39 + [(100.0, 101.0, 99.0, NAN)])
        self.assertTrue(np.isnan(r))

    def test_幅が狭いほど早く売る(self):
        rng = np.random.default_rng(5)
        for _ in range(200):
            c = 100 * np.cumprod(1 + rng.normal(0.003, 0.03, size=40))
            o = np.r_[100.0, c[:-1] * (1 + rng.normal(0, 0.01, size=39))]
            h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, size=40)))
            lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, size=40)))
            days = [E.trail_one(o, h, lo, c, 100.0, w)[1] for w in E.WIDTHS]
            self.assertEqual(days, sorted(days))
            for w in E.WIDTHS:
                r, _, fired, gap = E.trail_one(o, h, lo, c, 100.0, w)
                if fired and not gap:
                    self.assertGreaterEqual(r, -w - 1e-9)


def bars_case(n=60):
    """3銘柄 × n営業日。選定日は 0日目（買うのは1日目の寄り = 100）。30000 は選定日が遅く、40日目の足が無い。"""
    days = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for k, d in enumerate(days):
        # 10000: 1〜10日目に 130 まで上げ、11日目に 118 まで押し、12日目に 116 で始まり、そのあと 112 前後
        if k == 0:
            o, h, lo, c = 99.0, 100.0, 98.0, 99.0
        elif k <= 10:
            o = 100.0 + 3 * (k - 1)
            h, lo, c = o + 3, o - 0.5, o + 2
        elif k == 11:
            o, h, lo, c = 125.0, 126.0, 118.0, 119.0
        elif k == 12:
            o, h, lo, c = 116.0, 117.0, 112.0, 113.0
        else:
            o, h, lo, c = 112.0, 113.0, 111.0, 112.0
        rows.append({"Date": d, "Code": "10000", "AdjO": o, "AdjH": h, "AdjL": lo, "AdjC": c})
        c2 = 102.0 if k == 40 else 100.0
        rows.append({"Date": d, "Code": "20000", "AdjO": 100.0, "AdjH": 101.0, "AdjL": 99.0, "AdjC": c2})
        rows.append({"Date": d, "Code": "30000", "AdjO": 100.0, "AdjH": 101.0, "AdjL": 99.0, "AdjC": 100.0})
    return pd.DataFrame(rows), days


class TestExits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bars, days = bars_case()
        rr = lab.realized_returns(bars)
        df = pd.DataFrame({"Code": ["10000", "20000", "30000"], "Date": [days[0], days[0], days[len(days) - 25]]})
        df = df.merge(rr[["Code", "Date", E79.OUTCOME]], on=["Code", "Date"], how="left")
        cls.out, cls.chk = E.attach_all(df, bars)
        cls.o = cls.out.set_index("Code")

    def test_trailing_by_width(self):
        o = self.o.loc["10000"]
        self.assertAlmostEqual(o["trail7"], 20.9)        # 11日目: 130 × 0.93 = 120.9 に場中で触れる
        self.assertEqual((o["tday7"], bool(o["tfire7"]), bool(o["tgap7"])), (11, True, False))
        self.assertAlmostEqual(o["trail10"], 16.0)       # 12日目: 線 117 を寄り 116 で割る
        self.assertEqual((o["tday10"], bool(o["tfire10"]), bool(o["tgap10"])), (12, True, True))
        self.assertAlmostEqual(o["trail15"], 12.0)       # 線 110.5 に掛からず 40日目の終値 112
        self.assertEqual((o["tday15"], bool(o["tfire15"])), (40, False))
        self.assertAlmostEqual(o["c40"], 12.0)

    def test_split_is_mean_of_units(self):
        o = self.o.loc["10000"]
        self.assertEqual((o["tp10c"], o["hitday10"], o["d_tp10c"]), (10.0, 4, 4))   # 4日目の高値 112 で +10%
        self.assertAlmostEqual(o["split7"], (10.0 + 20.9) / 2)
        self.assertAlmostEqual(o["split10"], 13.0)
        self.assertAlmostEqual(o["split15"], 11.0)
        q = self.o.loc["20000"]
        self.assertAlmostEqual(q["tp10c"], 0.0)
        self.assertEqual(q["d_tp10c"], 20)
        self.assertAlmostEqual(q["trail10"], 2.0)        # 掛からず 40日目の終値 102
        self.assertAlmostEqual(q["split10"], 1.0)
        self.assertEqual((q["d_c20"], q["d_c40"]), (20, 40))

    def test_unfinished_rows_are_nan(self):
        o = self.o.loc["30000"]
        for c in ("c40", "trail7", "trail10", "trail15", "split10"):
            self.assertTrue(np.isnan(o[c]), c)
        self.assertFalse(bool(o["tfire10"]))

    def test_checks(self):
        self.assertEqual(self.chk["trail_order_violations"], 0)
        self.assertEqual(self.chk["trail_below_line"], 0)
        self.assertEqual(self.chk["with_c40"], 2)


class TestPaired(unittest.TestCase):
    @staticmethod
    def frame(diff, n_win=6, per=40, seed=1):
        rng = np.random.default_rng(seed)
        rows = []
        for w in range(1, n_win + 1):
            b = rng.normal(0, 5, size=per)
            a = b + diff + rng.normal(0, 1, size=per)
            rows.append(pd.DataFrame({"window": w, "Code": [f"{1000 + i % 15}0" for i in range(per)], "a": a, "b": b}))
        return pd.concat(rows, ignore_index=True)

    def test_no_difference(self):
        g = self.frame(0.0)
        g["a"] = g["b"]
        r = E.paired_test(g, "a", "b", n_perm=200)
        self.assertAlmostEqual(r["summary"]["mean"], 0.0)
        self.assertEqual(r["p"], 1.0)
        r = E.paired_test(self.frame(0.0), "a", "b", n_perm=500)
        self.assertGreater(r["p"], 0.01)

    def test_planted_difference(self):
        r = E.paired_test(self.frame(1.0), "a", "b", n_perm=500)
        self.assertAlmostEqual(r["summary"]["mean"], 1.0, delta=0.3)
        self.assertLess(r["p"], 0.01)
        self.assertEqual(r["summary"]["wins"], 6)
        rc = E.paired_test(self.frame(1.0), "a", "b", n_perm=500, unit="code")
        self.assertEqual(rc["rows"], 6 * 15)
        self.assertLess(rc["p"], 0.01)
        self.assertTrue(E.judge(r)["pass"])
        self.assertTrue(E.same_way(r))


def picks_case(n=80, seed=3):
    """毎営業日1件の候補（順位1）。前半は +10% に届くか 20日目、後半は 1〜40日目のどこかで売る。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n)
    hit = rng.random(n) < 0.4
    hd = rng.integers(1, 21, size=n).astype(float)
    p = pd.DataFrame({"Date": days, "Code": [f"{2000 + i}0" for i in range(n)], "rank": 1, "window": 1 + np.arange(n) // 20,
                      "hit10": hit, "hitday10": np.where(hit, hd, np.nan)})
    p["tp10c"] = np.where(hit, 10.0, rng.normal(0, 6, size=n))
    p["d_tp10c"] = np.where(hit, hd, 20.0)
    p["tday10"] = rng.integers(1, 41, size=n).astype(float)
    p["trail10"] = rng.normal(2, 8, size=n)
    p["tfire10"] = p["tday10"] < 40
    p["c40"] = rng.normal(2, 9, size=n)
    p["d_c40"] = 40.0
    cal = list(pd.bdate_range("2023-01-02", periods=n + 60))
    return p, cal


class TestUnitSim(unittest.TestCase):
    def test_single_unit_matches_live_track(self):
        p, cal = picks_case()
        mine = E.unit_sim(p, cal, E.legs_of("tp10c"), 3, n_days=len(p))
        q = p.assign(ret=p["tp10c"], status=np.where(p["hit10"], LT.HIT, "満了"), hit_day=p["hitday10"])
        sim = LT.simulate(q, cal, slots=3, hold=LT.HOLD)
        taken = sim[sim["taken"] == LT.TAKEN]
        self.assertEqual(mine["n"], len(taken))
        self.assertAlmostEqual(mine["mean"], float(taken["ret"].mean()))
        st = LT.trade_stats(taken, len(p), slots=3)
        self.assertAlmostEqual(mine["monthly"], st["monthly"])
        self.assertAlmostEqual(mine["hit"], st["hit"])

    def test_split_uses_two_units(self):
        p, cal = picks_case()
        one = E.unit_sim(p, cal, E.legs_of("tp10c"), 3, n_days=len(p))
        two = E.unit_sim(p, cal, E.legs_of("split10"), 3, n_days=len(p))
        six = E.unit_sim(p, cal, E.legs_of("split10"), 6, n_days=len(p))
        self.assertEqual(two["need_units"], 2)
        self.assertAlmostEqual(two["max_weight"], 200 / 3)
        self.assertLessEqual(two["util"], 100.0)
        self.assertLessEqual(two["held"], 2.0)          # 2単位の玉は資金3単位に1つと、前半を売った残りの1単位まで
        self.assertGreaterEqual(six["n"], two["n"])
        self.assertGreaterEqual(two["skipped"], one["skipped"])
        self.assertAlmostEqual(six["max_weight"], 100 / 3)
        # 分割の1取引の収益は2単位の平均、月の収益は単位ごとの収益の合計 ÷ 単位数 ÷ 月数
        self.assertIn("fire", two)

    def test_split_frees_only_the_sold_unit(self):
        # 毎日1件の候補。前半は5日目、後半は30日目に売る。資金3単位: 0日目の候補が2単位を使い、前半が空く 6営業日目
        # （5日目の候補）まで2単位そろわない。そのあとは後半2つ（30日目まで）と前半1つで埋まる
        cal = list(pd.bdate_range("2023-01-02", periods=60))
        p = pd.DataFrame({"Date": cal[:10], "Code": [f"{6000 + i}0" for i in range(10)], "rank": 1, "window": 1,
                          "hit10": True, "hitday10": 5.0, "tp10c": 10.0, "d_tp10c": 5.0,
                          "trail10": 4.0, "tday10": 30.0, "tfire10": True})
        st = E.unit_sim(p, cal, E.legs_of("split10"), 3, n_days=10)
        self.assertEqual((st["n"], st["skipped"]), (2, 8))
        self.assertAlmostEqual(st["util"], 200 / 3)          # 1〜5日目 2単位・6〜10日目 3・11〜30日目 2・31〜35日目 1（35日）
        self.assertAlmostEqual(st["held"], 60 / 35)
        one = E.unit_sim(p, cal, E.legs_of("tp10c"), 3, n_days=10)
        self.assertEqual(one["n"], 6)                         # 1単位なら 0・1・2・5・6・7日目の候補が買える

    def test_compounding_two_trades(self):
        cal = list(pd.bdate_range("2023-01-02", periods=80))
        p = pd.DataFrame({"Date": [cal[0], cal[30]], "Code": ["10000", "20000"], "rank": [1, 1], "window": [1, 1],
                          "hit10": [True, False], "hitday10": [3.0, np.nan], "tp10c": [10.0, -10.0],
                          "d_tp10c": [3.0, 20.0]})
        st = E.unit_sim(p, cal, E.legs_of("tp10c"), 3, n_days=42, eq_span=(cal[0], cal[-1]))
        self.assertAlmostEqual(st["final"], (1 + 0.10 / 3) * (1 - 0.10 / 3))
        self.assertAlmostEqual(st["monthly"], 0.0)            # 単位ごとの収益の合計 0 ÷ 3 ÷ 2か月
        self.assertEqual(st["months"], 4)                     # 2023-01〜04（eq_span の月）
        self.assertAlmostEqual(st["mdd"], ((1 + 0.10 / 3) * (1 - 0.10 / 3) / (1 + 0.10 / 3) - 1) * 100)

    def test_compounding_and_monthly(self):
        cal = list(pd.bdate_range("2023-01-02", periods=60))
        p = pd.DataFrame({"Date": [cal[0]], "Code": ["10000"], "rank": [1], "window": [1], "hit10": [True],
                          "hitday10": [3.0], "tp10c": [10.0], "d_tp10c": [3.0], "trail10": [20.0], "tday10": [8.0],
                          "tfire10": [True]})
        one = E.unit_sim(p, cal, E.legs_of("tp10c"), 3, n_days=21)
        self.assertAlmostEqual(one["final"], 1 + 0.10 / 3)
        self.assertAlmostEqual(one["monthly"], 10.0 / 3)
        two = E.unit_sim(p, cal, E.legs_of("split10"), 3, n_days=21)
        self.assertAlmostEqual(two["mean"], 15.0)
        self.assertAlmostEqual(two["final"], 1 + 0.10 / 3 + 0.20 / 3)
        self.assertAlmostEqual(two["monthly"], 30.0 / 3)
        self.assertAlmostEqual(two["fire"], 100.0)
        nod = E.unit_sim(p.assign(trail10=np.nan), cal, E.legs_of("split10"), 3, n_days=21)
        self.assertEqual((nod["n"], nod["nodata"]), (0, 1))


def frame_case(n=120, seed=7):
    """学習し直しの配線を見る小さなデータセット（ラベルの引き直しに要る列と、列2本）。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n // 4)
    fr = pd.DataFrame({"Code": np.repeat([f"{3000 + i}0" for i in range(4)], n // 4), "Date": np.tile(days, 4)})
    fr["vol_20d"] = rng.uniform(1.5, 4.0, size=n)
    fr["future_rise"] = rng.uniform(0.0, 0.5, size=n)
    fr["end_level"] = fr["future_rise"] * rng.uniform(0.3, 1.0, size=n)
    fr["uptrend_end"] = (rng.random(n) < 0.7).astype(float)
    fr["label"] = B.rise_label_for_k(fr.assign(label=0.0), B.DEFAULT_RISE.vol_norm_k)
    fr[E79.OUTCOME] = rng.normal(0, 0.05, size=n)
    fr["f1"] = rng.normal(size=n)
    fr["f2"] = rng.normal(size=n)
    return fr


def frame_long(seed=11):
    """窓を切れる長さ（2016〜2020年・2銘柄）の小さなデータセット。f1 が上昇率に効く。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-01", "2020-12-31")
    codes = ["40000", "50000"]
    fr = pd.DataFrame({"Code": np.repeat(codes, len(days)), "Date": np.tile(days, len(codes))})
    n = len(fr)
    fr["f1"] = rng.normal(size=n)
    fr["f2"] = rng.normal(size=n)
    fr["vol_20d"] = rng.uniform(1.5, 4.0, size=n)
    fr["future_rise"] = np.clip(0.15 + 0.06 * fr["f1"] + rng.normal(0, 0.08, size=n), 0, None)
    fr["end_level"] = fr["future_rise"] * rng.uniform(0.4, 1.0, size=n)
    fr["uptrend_end"] = (rng.random(n) < 0.75).astype(float)
    fr["label"] = B.rise_label_for_k(fr.assign(label=0.0), B.DEFAULT_RISE.vol_norm_k)
    fr[E79.OUTCOME] = rng.normal(0, 0.05, size=n)
    return fr


class TestNotices(unittest.TestCase):
    @staticmethod
    def unesc(s):
        return s.replace("%0A", "\n").replace("%0D", "\r").replace("%25", "%")

    def test_roundtrip_under_limit(self):
        text = "".join(f"  分割 +10% / 高値−{i % 15}%  1取引 +{i % 7}.12%（中央 +1.00）  勝率 70.0%\n" for i in range(300))
        chunks = E.notice_chunks(text)
        self.assertLessEqual(len(chunks), E.NOTICE_MAX)
        self.assertTrue(all(len(E.esc_data(c).encode("utf-8")) <= E.NOTICE_BYTES for c in chunks))
        self.assertEqual("".join(chunks), text)
        buf = io.StringIO()
        n = E.emit_notices(text, out=buf)
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), n)
        got = ""
        for i, ln in enumerate(lines, 1):
            head = f"::notice title=e89 結果 ({i}/{n})::"
            self.assertTrue(ln.startswith(head))
            got += self.unesc(ln[len(head):])
        self.assertEqual(got, text)

    def test_long_line_and_cap(self):
        text = "あ" * 30000 + "\n"                          # 1行 90KB。10KB ずつ 9塊に切れる
        chunks = E.notice_chunks(text, limit=10000)
        self.assertEqual(len(chunks), 10)                     # 9塊 + 改行だけの塊（行の終わり）
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(E.esc_data(c).encode("utf-8")) <= 10000 for c in chunks))
        many = E.notice_chunks("x" * 100 + "\n" * 1, limit=10, max_n=3)
        self.assertEqual(len(many), 3)
        self.assertIn("省略", many[-1])


class TestPlanAndVerdict(unittest.TestCase):
    def test_plan(self):
        self.assertEqual(E.make_plan([0]), [("0", 0, False)])                       # 軽く回す: 学習し直さない
        self.assertEqual(E.make_plan([0, 2, 4]), [("0", 0, False), ("0r", 0, True), ("2", 2, True), ("4", 4, True)])
        self.assertEqual(E.make_plan([0, 2, 4], no_retrain0=True), [("0", 0, False), ("2", 2, True), ("4", 4, True)])
        self.assertEqual(E.make_plan([2, 4]), [("2", 2, True), ("4", 4, True)])

    @staticmethod
    def runs(spec):
        """spec: {窓の切り方: {幅: (平均, 上の窓, 窓の数, p)}}"""
        out = {}
        for t, by in spec.items():
            main = {}
            for k in E.WKEYS:
                m, w, n, p = by[k]
                res = {"summary": {"n": n, "mean": m, "se": 0.1, "wins": w}, "p": p}
                main[k] = {"row": res, "judge": E.judge(res), "same_way": E.same_way(res)}
            out[t] = {"main": main}
        return out

    def verdict(self, spec):
        with contextlib.redirect_stdout(io.StringIO()):
            return E.final_summary(self.runs(spec))

    def test_verdict_branches(self):
        good, bad = (1.0, 7, 9, 0.01), (-0.5, 3, 9, 0.8)
        allg = {k: good for k in E.WKEYS}
        v = self.verdict({"0": allg, "0r": allg, "2": allg, "4": allg})
        self.assertTrue(v["verdict"].startswith("幅によらず"))
        v = self.verdict({"0": allg, "0r": allg})                       # ずらした窓を回していない
        self.assertTrue(v["verdict"].startswith("ずらし0か月では3水準とも通る"))
        v = self.verdict({"0": allg, "2": {**allg, "15": bad}, "4": allg})
        self.assertTrue(v["verdict"].startswith("幅しだい（−7%・−10% だけ通る"))
        v = self.verdict({"0": {k: bad for k in E.WKEYS}, "2": allg})
        self.assertTrue(v["verdict"].startswith("通らない"))
        v = self.verdict({"2": allg, "4": allg})                        # 本番の窓を回していない
        self.assertTrue(v["verdict"].startswith("判定なし"))
        st = self.verdict({"0": {"7": good, "10": (0.2, 5, 9, 0.3), "15": good},
                           "2": {"7": good, "10": bad, "15": (0.4, 6, 9, 0.2)}, "4": allg})["stability"]
        self.assertEqual([k for k, _, _, _ in st], ["7", "15", "10"])   # 3/3・最悪 1.0 > 3/3・最悪 0.4 > 2/3


class TestCompleteRows(unittest.TestCase):
    def test_cutoff_and_missing(self):
        days = pd.bdate_range("2024-01-01", periods=6)
        ft = pd.DataFrame({"Date": days, "A15": [True, True, False, True, False, True]})
        for c in E.COMPARE_COLS:
            ft[c] = 1.0
        ft.loc[1, "trail10"] = np.nan                                   # 後半だけ欠けた行も全部の出口から外す
        ft.loc[2, "c40"] = np.nan
        keep, info = E.complete_rows(ft, days[3])
        self.assertEqual(list(keep["Date"]), [days[0], days[3]])        # cutoff の日は残し、その後は外す
        self.assertEqual((info["late"], info["missing"], info["missing_a15"], info["kept"]), (2, 2, 1, 2))
        self.assertTrue(np.isfinite(keep[list(E.COMPARE_COLS)].to_numpy(dtype=float)).all())


class TestRowsMatchE82(unittest.TestCase):
    def test_flags_same_as_e82(self):
        # 6本だけで作った前向きの百分位でも、規則の印と 1.2σ の3本の最小は実験82（12本で作る）と同じ
        import e82_and_rule_variants as E82
        df, folds = T82.synth(seed=4)
        s12 = E79.window_split(df, folds)
        s6 = E79.window_split(df, folds, models=E.SIX)
        a = E82.build_frame(df, [(te, P) for _, (tr, te, Ptr, P) in s12.items()])
        b = E.build_rows(df, [(te, P) for _, (tr, te, Ptr, P) in s6.items()])
        for c in ("A15", "R0", "pmin3"):
            np.testing.assert_array_equal(a[c].to_numpy(), b[c].to_numpy())
        self.assertGreater(int(b["A15"].sum()), 0)


class TestRetrainWiring(unittest.TestCase):
    def setUp(self):
        self.fr = frame_case()
        self.calls = []

        def fake_oof(algo, df, cols, params, seed, folds):
            self.calls.append((algo, df["label"].copy(), seed, folds))
            d = df[df["label"].notna()]
            rng = np.random.default_rng(len(self.calls))
            out = d[["Code", "Date", "label", E79.OUTCOME]].copy()
            out["score"] = rng.normal(size=len(out))
            out["fold"] = 1
            return out

        self.fake_oof = fake_oof
        self.fake_folds = lambda dates, shift: [types.SimpleNamespace(index=1, shift=shift)]

    def test_wide_and_relabel(self):
        wide, folds, info = E.shifted_wide(self.fr, ["f1", "f2"], 2, oof_fn=self.fake_oof, folds_fn=self.fake_folds,
                                           cache=False)
        self.assertEqual(folds[0].shift, 2)
        self.assertEqual([c[0] for c in self.calls], list(E.SIX))
        for a in E.SIX:
            self.assertIn(f"score_{a}", wide.columns)
        self.assertIn(E79.OUTCOME, wide.columns)
        k15 = B.rise_label_for_k(self.fr, 1.5)
        for algo, lab_, seed, _ in self.calls:
            want = k15 if algo.endswith("_k15") else self.fr["label"]
            pd.testing.assert_series_equal(lab_.reset_index(drop=True), want.reset_index(drop=True), check_names=False)
            self.assertEqual(seed, E.prod_params(algo)[1])
        self.assertFalse(k15.equals(self.fr["label"]))           # 1.5σ では正例が減る（引き直しが効いている）
        self.assertEqual(info["kept"], len(wide))

    @unittest.skipUnless(HAVE_GBM, "本物の学習器（lightgbm）が要る（CI は research/requirements.txt を入れる）")
    def test_real_oof_folds_for_k15(self):
        # 実験41 の oof_folds と folds_for をそのまま使い、1.5σ のモデル（lgbm_k15）も本番と同じ経路で学習できること
        fr = frame_long()
        wide, folds, info = E.shifted_wide(fr, ["f1", "f2"], 2, models=("lgbm", "lgbm_k15"), cache=False)
        self.assertGreaterEqual(len(folds), 2)
        self.assertGreater(len(wide), 200)
        for a in ("lgbm", "lgbm_k15"):
            s = wide[f"score_{a}"].to_numpy(dtype=float)
            self.assertTrue(((s >= 0) & (s <= 1)).all())
        self.assertIn(E79.OUTCOME, wide.columns)
        self.assertLess(pd.to_datetime(fr["Date"]).min() + pd.DateOffset(months=2),
                        pd.Timestamp(folds[0].test_start))   # 窓の境界が2か月うしろにずれている

    @unittest.skipUnless(HAVE_PARQUET, "parquet の読み書きに pyarrow が要る")
    def test_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            E.shifted_wide(self.fr, ["f1", "f2"], 4, oof_fn=self.fake_oof, folds_fn=self.fake_folds, out_dir=d)
            n = len(self.calls)
            w2, _, info = E.shifted_wide(self.fr, ["f1", "f2"], 4, oof_fn=self.fake_oof, folds_fn=self.fake_folds,
                                         out_dir=d)
            self.assertEqual(len(self.calls), n)                 # 2回目は学習しない
            self.assertEqual(sorted(info["cached"]), sorted(E.SIX))
            self.assertEqual(len(w2), info["kept"])


def synth_rows(seed=0):
    """実験82 の合成データに、出口の列を作りもので足す（実収益から）。"""
    df, folds = T82.synth(seed=seed)
    rng = np.random.default_rng(seed)
    r = df[E79.OUTCOME].to_numpy() * 100
    n = len(df)
    df["c20"] = r
    df["hold5"] = r
    df["c40"] = r * 1.3 + rng.normal(0, 2, size=n)
    for t in (10.0, 20.0):
        k = f"{int(t)}"
        hit = r >= t
        df[f"hit{k}"] = hit
        df[f"hitday{k}"] = np.where(hit, 5.0, np.nan)
        df[f"tp{k}c"] = np.where(hit, t, r)
    df["tp20a"] = df["tp20c"]
    df["d_tp10c"] = np.where(df["hit10"], 5.0, 20.0)
    df["d_tp20a"] = np.where(df["hit20"], 5.0, 20.0)
    df["d_c20"] = 20.0
    df["d_c40"] = 40.0
    for w, k in zip(E.WIDTHS, E.WKEYS):
        fire = rng.random(n) < 0.5
        df[f"trail{k}"] = np.where(fire, np.maximum(-w, df["c40"] - w), df["c40"])
        df[f"tday{k}"] = np.where(fire, rng.integers(2, 40, size=n), 40).astype(float)
        df[f"tfire{k}"] = fire
        df[f"tgap{k}"] = fire & (rng.random(n) < 0.1)
        df[f"split{k}"] = (df["tp10c"] + df[f"trail{k}"]) / 2
    return df, folds


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = synth_rows()
        split = E79.window_split(cls.df, folds, models=E.SIX)
        cls.ft = E.build_rows(cls.df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-12-31"))

    def test_rows_have_flags_and_exits(self):
        for c in ("A15", "R0", "pmin3", "pmin15", "tp10c", "split10", "trail15", "d_tp10c", "tday7", "tfire10"):
            self.assertIn(c, self.ft.columns)
        self.assertFalse((self.ft["A15"] & ~self.ft["R0"]).any())
        self.assertTrue((self.ft.loc[self.ft["A15"], "pmin15"] >= E.LINE).all())

    def test_report_prints_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s0 = E.run_report("0", self.ft, self.bar_days, n_perm=50, seed=1, full=True, ft20=self.ft)
            s2 = E.run_report("2", self.ft, self.bar_days, n_perm=50, seed=1, full=False)
            fin = E.final_summary({"0": s0, "2": s2})
        out = buf.getvalue()
        for tag in ("■ 0-1.", "■ 0-2.", "■ 0-3.", "■ 0-4.", "■ 0-5.", "■ 2-2.", "■ 2-4.", "■ まとめ", "判定:"):
            self.assertIn(tag, out)
        for k in E.WKEYS:
            self.assertIn(k, s0["main"])
            self.assertIn(f"split{k}_6u", s0["sim"]["A15"])
            self.assertIn(k, fin["by_width"])
        self.assertGreater(s0["sim"]["A15"]["tp10c"]["n"], 0)
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)
        # 注釈に出す短い要約: 10個 × 3.8KB に収まり、銘柄コードを含まない
        infos = {t: {"exit_check": {"compare_rows": {"late": 1, "missing": 2, "missing_a15": 0}}} for t in ("0", "2")}
        text = E.compact_report({"0": s0, "2": s2}, infos, fin, "2024-05-02")
        chunks = E.notice_chunks(text)
        self.assertEqual("".join(chunks), text)
        self.assertLessEqual(len(chunks), E.NOTICE_MAX)
        for tag in ("━0", "━2", "判定 split−tp10c", "3枠 A15", "参考差 A15", "判定:", "幅の安定"):
            self.assertIn(tag, text)
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, text)


if __name__ == "__main__":
    unittest.main()
