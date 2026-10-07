#!/usr/bin/env python3
"""
実験65（research/exp/e65_window_20_40.py）と、目的変数の窓の一般化（build_dataset.RiseConfig の
start / sigma_days）のテスト。重い計算は回さない。

- 既定（start=1, sigma_days=None）は今の定義そのまま（名前・しきい値・ラベルが1件も変わらない）
- start をずらすと、到達は t+start 〜 t+horizon の終値だけで測る。終盤とトレンドは t+horizon で見る
- sigma_days はしきい値の √日数だけを変える（horizon=40, sigma_days=20 の必要上昇 = horizon=20 と同じ）
- 確定には先 horizon 営業日ぶんの行が要る。start の範囲外と、start>1 の keep_days は止める
- 実験65 の新ラベルは運用者の選択どおり（21〜40営業日目・1.2σ・σは20営業日・翌営業日寄り基準）
- OOF の関数はエンバーゴとパラメータを受け取れ、渡さなければ今と同じ（B.RISE_HORIZON）
- 候補単位の数え方（両方 + 現行だけ = 現行）、同じ日をまとめた SE、規則どおりの収益

  python3 tests/test_e65_window_20_40.py
"""
import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import e59_unit_sim as U  # noqa: E402
import e65_window_20_40 as E  # noqa: E402


def panel(closes, code="1234", opens=None, vol=2.0):
    """close / open / vol_20d だけの日足（attach_rise_label の入力）。"""
    n = len(closes)
    return pd.DataFrame({
        "Code": code,
        "Date": pd.bdate_range("2021-01-04", periods=n),
        "close": [float(c) for c in closes],
        "open": [float(o) for o in (opens if opens is not None else closes)],
        "vol_20d": float(vol)})


def reach_only(**kw):
    """到達の軸だけ（固定しきい値・終盤とトレンドなし・当日終値基準）。"""
    base = dict(entry="close", threshold=0.20, keep_days=0, end_ratio=None,
                require_uptrend=False, vol_norm_k=None)
    base.update(kw)
    return B.RiseConfig(**base)


class TestDefaultUnchanged(unittest.TestCase):
    def test_defaults(self):
        cfg = B.DEFAULT_RISE
        self.assertEqual(cfg.start, 1)
        self.assertIsNone(cfg.sigma_days)
        self.assertEqual(cfg.sigma_horizon, cfg.horizon)
        self.assertEqual(cfg.name, "1ヶ月内+1.2σ / 終盤+0.50倍 / MA5>=MA20 / 翌営業日寄り基準")

    def test_labels_identical_to_explicit_start1(self):
        """既定と、start=1・sigma_days=horizon を明示した定義で、ラベルが全行同じ。"""
        rng = np.random.default_rng(0)
        frames = []
        for k, code in enumerate(("1111", "2222", "3333")):
            r = rng.normal(0.002, 0.03, 120)
            c = 100 * np.exp(np.cumsum(r))
            o = c * np.exp(rng.normal(0, 0.01, 120))
            frames.append(panel(c, code=code, opens=o, vol=1.5 + k))
        df = pd.concat(frames, ignore_index=True)
        a = B.attach_rise_label(df.copy(), B.DEFAULT_RISE)
        b = B.attach_rise_label(df.copy(), B.RiseConfig(start=1, sigma_days=B.DEFAULT_RISE.horizon))
        for c in ("label", "future_rise", "end_level", "uptrend_end", "rise_need", "end_need"):
            pd.testing.assert_series_equal(a[c].astype(float), b[c].astype(float), check_names=False)
        self.assertGreater(int(a["label"].notna().sum()), 0)


class TestShiftedWindow(unittest.TestCase):
    def test_rise_before_the_window_does_not_count(self):
        """t+2 で +25% でも、窓が t+3〜t+5 なら到達していない（今の窓 t+1〜t+5 なら到達）。"""
        closes = [100, 100, 125, 100, 100, 100, 100]
        early = B.attach_rise_label(panel(closes), reach_only(horizon=5))
        late = B.attach_rise_label(panel(closes), reach_only(horizon=5, start=3))
        self.assertTrue(bool(early["label"].iloc[0]))
        self.assertFalse(bool(late["label"].iloc[0]))
        self.assertAlmostEqual(late["future_rise"].iloc[0], 0.0)

    def test_rise_inside_the_window_counts(self):
        closes = [100, 100, 100, 100, 125, 100, 100]
        out = B.attach_rise_label(panel(closes), reach_only(horizon=5, start=3))
        self.assertTrue(bool(out["label"].iloc[0]))          # t+4 は窓 t+3〜t+5 の中
        self.assertAlmostEqual(out["future_rise"].iloc[0], 0.25)

    def test_window_end_is_inclusive_and_beyond_is_ignored(self):
        """t+horizon は窓に入り、t+horizon+1 は入らない。"""
        at_end = B.attach_rise_label(panel([100, 100, 100, 100, 100, 125, 100]),
                                     reach_only(horizon=5, start=3))
        after = B.attach_rise_label(panel([100, 100, 100, 100, 100, 100, 125]),
                                    reach_only(horizon=5, start=3))
        self.assertTrue(bool(at_end["label"].iloc[0]))
        self.assertFalse(bool(after["label"].iloc[0]))

    def test_needs_horizon_rows_ahead(self):
        """確定には先 horizon 行が要る（窓の長さ horizon−start+1 ではない）。"""
        out = B.attach_rise_label(panel([100] * 8), reach_only(horizon=5, start=3))
        det = out["label"].notna().to_numpy()
        self.assertTrue(det[:3].all())                       # 0〜2 は t+5 まである
        self.assertFalse(det[3:].any())                      # 3 以降は足りない

    def test_end_and_trend_are_read_at_the_window_end(self):
        """窓の中で届いても、t+horizon の水準が落ちていれば終盤条件で外れる。"""
        cfg = B.RiseConfig(entry="close", horizon=8, start=4, threshold=0.20, keep_days=0,
                           end_ratio=0.10, end_window=1, require_uptrend=False, vol_norm_k=None)
        hold = B.attach_rise_label(panel([100] * 4 + [125] * 6), cfg)
        fade = B.attach_rise_label(panel([100] * 4 + [125] * 4 + [100] * 2), cfg)
        self.assertTrue(bool(hold["label"].iloc[0]))
        self.assertAlmostEqual(fade["end_level"].iloc[0], 0.0)   # t+8 の終値 100
        self.assertFalse(bool(fade["label"].iloc[0]))

    def test_next_open_entry_is_kept(self):
        """基準は start によらず t+1 の寄り。"""
        closes = [100, 100, 100, 100, 130, 130, 130]
        opens = [100, 110, 100, 100, 100, 100, 100]
        cfg = reach_only(horizon=5, start=3, entry="next_open")
        out = B.attach_rise_label(panel(closes, opens=opens), cfg)
        self.assertAlmostEqual(out["entry_price"].iloc[0], 110.0)
        self.assertAlmostEqual(out["future_rise"].iloc[0], 130 / 110 - 1)

    def test_bad_start_and_keep_days_stop(self):
        with self.assertRaises(SystemExit):
            B.attach_rise_label(panel([100] * 10), reach_only(horizon=5, start=0))
        with self.assertRaises(SystemExit):
            B.attach_rise_label(panel([100] * 10), reach_only(horizon=5, start=6))
        with self.assertRaises(SystemExit):
            B.attach_rise_label(panel([100] * 10), reach_only(horizon=5, start=2, keep_days=2))


class TestSigmaDays(unittest.TestCase):
    def test_sigma_days_sets_the_sqrt(self):
        n20, e20 = B.rise_thresholds([2.0], B.RiseConfig(horizon=20))
        n40, e40 = B.rise_thresholds([2.0], B.RiseConfig(horizon=40))
        m, me = B.rise_thresholds([2.0], B.RiseConfig(horizon=40, start=21, sigma_days=20))
        self.assertAlmostEqual(m[0], n20[0])                  # 必要上昇は今と同じ（+10.7%）
        self.assertAlmostEqual(me[0], e20[0])
        self.assertAlmostEqual(n40[0] / n20[0], math.sqrt(2.0))
        self.assertAlmostEqual(m[0] * 100, 10.7, places=1)

    def test_name_shows_window_and_sigma(self):
        self.assertEqual(E.NEW_RISE.name,
                         "21〜40営業日目+1.2σ（σは20営業日） / 終盤+0.50倍 / MA5>=MA20 / 翌営業日寄り基準")
        self.assertNotEqual(E.NEW_RISE.name, B.DEFAULT_RISE.name)


class TestExperimentConfig(unittest.TestCase):
    def test_new_label_is_the_operator_choice(self):
        cfg = E.NEW_RISE
        self.assertEqual((cfg.start, cfg.horizon, cfg.sigma_horizon), (21, 40, 20))
        d = B.DEFAULT_RISE
        # 窓としきい値の日数以外は今と同じ
        for f in ("threshold", "keep_days", "end_ratio", "end_window", "trend_short", "trend_long",
                  "require_uptrend", "vol_norm_k", "entry"):
            self.assertEqual(getattr(cfg, f), getattr(d, f), f)
        # MA20 は t+21〜t+40 = 判定窓そのもの
        self.assertEqual(cfg.trend_long, cfg.horizon - cfg.start + 1)
        self.assertEqual(E.EMBARGO_NEW, 40)

    def test_patterns(self):
        names = [p[0] for p in E.PATTERNS]
        self.assertIn("両方 全5モデル95以上", names)
        both = [p for p in E.PATTERNS if p[0] == "両方 全5モデル95以上"][0]
        self.assertEqual(set(both[1]), set(E.CUR) | set(E.NEW))
        self.assertEqual(len(both[1]), 10)


class TestOofPlumbing(unittest.TestCase):
    """OOF の関数がエンバーゴを受け取り、渡さなければ B.RISE_HORIZON のまま。"""

    def _capture(self, call):
        import walkforward as WF
        seen = {}
        orig = WF.make_folds

        def fake(dates, **kw):
            seen.update(kw)
            return []
        WF.make_folds = fake
        try:
            with self.assertRaises(SystemExit):        # 窓が無い → 「作れません」で止まる
                call()
        finally:
            WF.make_folds = orig
        return seen["embargo_days"]

    def test_train_production(self):
        import train_production as TP
        ds = pd.DataFrame({"Date": pd.bdate_range("2021-01-04", periods=5), "label": 0})
        self.assertEqual(self._capture(lambda: TP.oof_scores(ds, [], {})), B.RISE_HORIZON)
        self.assertEqual(self._capture(lambda: TP.oof_scores(ds, [], {}, embargo_days=40)), 40)

    def test_train_multi(self):
        import train_multi as TMU
        ds = pd.DataFrame({"Date": pd.bdate_range("2021-01-04", periods=5), "label": 0})
        self.assertEqual(self._capture(lambda: TMU.oof_scores("xgb", ds, [])), B.RISE_HORIZON)
        self.assertEqual(self._capture(lambda: TMU.oof_scores("xgb", ds, [], params={}, embargo_days=40)), 40)


class TestShiftedFolds(unittest.TestCase):
    """頑健性の確認で窓の境界をずらす（実験41 の folds_for と同じ切り方）。"""

    def _folds(self, shift):
        import walkforward as WF
        dates = pd.Series(pd.bdate_range("2018-01-01", "2023-12-29"))
        with E.shifted_folds(shift):
            return WF.make_folds(dates, min_train_months=36, test_months=6, step_months=6,
                                 embargo_days=20)

    def test_shift_moves_every_boundary(self):
        f0, f2 = self._folds(0), self._folds(2)
        self.assertEqual(pd.Timestamp(f0[0].train_end), pd.Timestamp("2021-01-01"))
        self.assertEqual(pd.Timestamp(f2[0].train_end), pd.Timestamp("2021-03-01"))
        # 窓の頭が後ろへずれる（訓練は呼び出し側で期間の最初から使う）
        self.assertGreater(pd.Timestamp(f2[0].test_start), pd.Timestamp(f0[0].test_start))

    def test_make_folds_is_restored(self):
        import walkforward as WF
        orig = WF.make_folds
        with E.shifted_folds(4):
            self.assertIsNot(WF.make_folds, orig)
        self.assertIs(WF.make_folds, orig)
        with self.assertRaises(RuntimeError):
            with E.shifted_folds(4):
                raise RuntimeError("x")
        self.assertIs(WF.make_folds, orig)

    def test_oof_path(self):
        self.assertTrue(E.oof_path("d", "cur_lgbm").endswith("oof_cur_lgbm.parquet"))
        self.assertTrue(E.oof_path("d", "cur_lgbm", 2).endswith("oof_cur_lgbm_sh2.parquet"))


class TestNewLabelFrame(unittest.TestCase):
    def test_with_new_label(self):
        frame = pd.DataFrame({"Code": ["1", "2", "3", "4"],
                              "Date": pd.to_datetime(["2021-01-04"] * 2 + ["2021-01-05"] * 2),
                              "label": [1, 0, 1, 0], "x": [1.0, 2.0, 3.0, 4.0]})
        labels = pd.DataFrame({"Code": ["1", "2", "3", "4"], "Date": frame["Date"],
                               "label_new": [0.0, 1.0, np.nan, 1.0]})
        out = E.with_new_label(frame, labels)
        self.assertEqual(list(out["Code"]), ["1", "2", "4"])          # 未確定は落とす・順は保つ
        self.assertEqual(list(out["label"]), [0, 1, 1])
        self.assertEqual(list(out["label_cur"]), [1, 0, 0])
        self.assertEqual(out["label"].dtype, np.int64)


class TestCandidates(unittest.TestCase):
    def _cands(self):
        rows = []
        for i, (pc, pn) in enumerate([(99, 99), (99, 80), (96, 97), (50, 99), (99, np.nan)]):
            r = {"Code": str(i), "Date": pd.Timestamp("2021-01-04") + pd.Timedelta(days=i),
                 "label_cur": 1, "label_new": 0, "ret_20": float(i), "hit_20": 0.0}
            for k in E.CUR:
                r[f"p_{k}"] = pc
            for k in E.NEW:
                r[f"p_{k}"] = pn
            rows.append(r)
        return pd.DataFrame(rows)

    def test_groups_partition(self):
        c = self._cands()
        g = E.candidate_groups(c, 95.0)
        keys = list(g)
        cur, both, only = g[keys[1]], g[keys[2]], g[keys[3]]
        self.assertEqual(sorted(cur["Code"]), ["0", "1", "2", "4"])
        self.assertEqual(sorted(both["Code"]), ["0", "2"])
        self.assertEqual(sorted(only["Code"]), ["1", "4"])            # 新の百分位が無い行は「現行だけ」
        self.assertEqual(len(both) + len(only), len(cur))
        self.assertEqual(sorted(g[keys[5]]["Code"]), ["3"])            # 新だけ

    def test_cluster_se(self):
        x = pd.Series([1.0, 2.0, 4.0, 7.0])
        d = pd.Series(pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]))
        n = len(x)
        # 1日1件なら、偏差の二乗和の平方根 / n（ddof=0 の SE）
        self.assertAlmostEqual(E.cluster_se(x, d), math.sqrt(((x - x.mean()) ** 2).sum()) / n)
        # 同じ日に同じ向きの値が固まると大きくなる
        same = pd.Series(pd.to_datetime(["2021-01-04", "2021-01-04", "2021-01-06", "2021-01-06"]))
        self.assertGreater(E.cluster_se(x, same), E.cluster_se(x, d))

    def test_ranking_check(self):
        """1日1件の並べ方: 10モデルの最小と現行5モデルの最小で1位が変わる日を数える。"""
        d1, d2, d3 = (pd.Timestamp(x) for x in ("2021-01-04", "2021-01-05", "2021-01-06"))
        rows = []
        # d1: 2件。10モデルの最小は B（96）が上、現行5モデルの最小は A（99）が上 → 変わる
        for code, pc, pn, date in (("A", 99, 95, d1), ("B", 96, 98, d1),
                                   # d2: 2件。どちらで並べても C が上 → 変わらない
                                   ("C", 99, 99, d2), ("D", 96, 97, d2),
                                   # d3: 1件だけ
                                   ("E", 97, 97, d3)):
            r = {"Code": code, "Date": date, "score": 0.5, "p_min": min(pc, pn)}
            r.update({f"p_{k}": pc for k in E.CUR})
            r.update({f"p_{k}": pn for k in E.NEW})
            rows.append(r)
        rc = E.ranking_check(pd.DataFrame(rows))
        self.assertEqual(rc, {"days": 3, "multi": 2, "diff": 1})
        self.assertEqual(E.ranking_check(pd.DataFrame(rows).iloc[:0]), {"days": 0, "multi": 0, "diff": 0})

    def test_matched_threshold(self):
        """発火数をそろえた対照の線: p_min >= x が n 件以上になる最大の x。"""
        p = pd.Series([99.5, 98.0, 97.0, 97.0, 96.0, np.nan])
        self.assertEqual(E.matched_threshold(p, 1), 99.5)
        self.assertEqual(E.matched_threshold(p, 3), 97.0)
        self.assertEqual(int((p >= E.matched_threshold(p, 3)).sum()), 4)   # 同点で n を超える
        self.assertEqual(E.matched_threshold(p, 99), 96.0)                 # 件数が足りなければ最小
        self.assertTrue(np.isnan(E.matched_threshold(p, 0)))

    def test_pmins(self):
        c = self._cands()
        pc, pn = E.pmins(c)
        self.assertEqual(list(pc), [99, 99, 96, 50, 99])
        self.assertTrue(np.isnan(pn.iloc[4]))

    def test_zdiff(self):
        a = {"mean": 3.0, "se": 1.0}
        b = {"mean": 1.0, "se": 1.0}
        self.assertAlmostEqual(E.zdiff(a, b), 2.0 / math.sqrt(2.0))


class TestCandidateReturns(unittest.TestCase):
    def test_buys_next_open_and_follows_rule_exit(self):
        days = list(pd.bdate_range("2024-01-01", periods=50))
        o = [100.0] * 50
        h = [101.0] * 50
        c = [100.0] * 50
        h[3] = 121.0                       # 買った日（index 1）から数えて 3日目に +20% に届く
        bars = pd.DataFrame({"Date": days, "Code": "A", "O": o, "AdjO": o, "AdjH": h, "AdjC": c})
        flat = bars.assign(Code="B", AdjH=101.0)
        px = U.PriceGrid(pd.concat([bars, flat], ignore_index=True), days, ["A", "B"])
        base = pd.DataFrame({"Code": ["A", "B", "B"], "Date": [days[0], days[0], days[45]]})
        r = E.candidate_returns(base, px, holds=(20, 40))
        self.assertAlmostEqual(r["ret_20"].iloc[0], 20.0)
        self.assertEqual(r["hit_20"].iloc[0], 1.0)
        self.assertAlmostEqual(r["ret_40"].iloc[0], 20.0)
        self.assertAlmostEqual(r["ret_20"].iloc[1], 0.0)           # 届かず満期の終値 100
        self.assertEqual(r["hit_20"].iloc[1], 0.0)
        self.assertTrue(np.isnan(r["ret_20"].iloc[2]))             # 先の日足が足りない


if __name__ == "__main__":
    unittest.main()
