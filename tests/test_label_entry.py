#!/usr/bin/env python3
"""
目的変数の基準の価格を「翌営業日の寄り」（分割調整後の始値 AdjO[t+1]）にしたことのテスト
（2026-09-25、運用者の決定。build_dataset.LABEL_ENTRY）。

候補が分かるのは高値を更新した日の終値の後で、実際に買えるのは翌営業日の寄り。
収益の計算（lab.realized_returns の ret_o1_*、買い = AdjO[t+1]）と同じ基準にそろえる。
変わるのは上昇率の分母だけで、見る窓（t+1〜t+horizon の終値）・しきい値・トレンド条件は
変えない。翌営業日に寄りが付かない行は、収益の計算と同じく判定できない（未確定）。
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import build_dataset as B  # noqa: E402


def frame(closes, opens, code="1234"):
    return pd.DataFrame({
        "Code": code,
        "Date": pd.bdate_range("2021-01-04", periods=len(closes)),
        "close": [float(c) for c in closes],
        "open": [np.nan if o is None else float(o) for o in opens]})


def reach(**kw):
    """到達の軸だけを見る設定（固定しきい値）。"""
    base = dict(horizon=3, threshold=0.20, keep_days=0, end_ratio=None,
                require_uptrend=False, vol_norm_k=None)
    base.update(kw)
    return B.RiseConfig(**base)


class TestDefault(unittest.TestCase):
    def test_production_label_uses_the_next_open(self):
        self.assertEqual(B.LABEL_ENTRY, "next_open")
        self.assertEqual(B.DEFAULT_RISE.entry, "next_open")
        self.assertTrue(B.DEFAULT_RISE.name.endswith("/ 翌営業日寄り基準"))
        self.assertNotIn("翌営業日寄り", reach(entry="close").name)

    def test_sweep_yardstick_stays_on_the_close(self):
        """掃引の物差し（ref_rise / ref_end）は過去の掃引と比べるため終値基準のまま。"""
        import sweep_design as S
        self.assertEqual(S.REF_RISE.entry, "close")


class TestNextOpenBase(unittest.TestCase):
    def test_denominator_is_the_next_day_open(self):
        out = B.attach_rise_label(
            frame([100, 110, 125, 112, 108], [99, 104, 111, 120, 110]), reach())
        self.assertEqual(out["entry_price"].iloc[0], 104.0)          # open[t+1]
        self.assertAlmostEqual(out["future_rise"].iloc[0], 125 / 104 - 1)

    def test_a_gap_up_can_turn_a_close_based_positive_into_a_negative(self):
        """終値基準なら +25% で正例だが、翌朝 112 で寄ったので買値からは +11.6%。"""
        closes = [100, 118, 125, 121, 119]
        opens = [100, 112, 118, 124, 120]
        by_close = B.attach_rise_label(frame(closes, opens), reach(entry="close"))
        by_open = B.attach_rise_label(frame(closes, opens), reach())
        self.assertTrue(bool(by_close["label"].iloc[0]))
        self.assertFalse(bool(by_open["label"].iloc[0]))
        self.assertAlmostEqual(by_open["future_rise"].iloc[0], 125 / 112 - 1)

    def test_window_still_starts_at_t_plus_1(self):
        """見る窓は変えない（t+1 の終値も入る。買った日の引けで届けば到達）。"""
        out = B.attach_rise_label(frame([100, 130, 100, 100, 100], [100, 105, 100, 100, 100]),
                                  reach())
        self.assertAlmostEqual(out["future_max_close"].iloc[0], 130.0)
        self.assertTrue(bool(out["label"].iloc[0]))                  # 130/105-1 = 23.8%

    def test_end_level_uses_the_next_open(self):
        cfg = reach(threshold=0.0, end_ratio=0.10, end_window=1)
        out = B.attach_rise_label(frame([100, 110, 120, 130], [100, 105, 118, 128]), cfg)
        self.assertAlmostEqual(out["end_level"].iloc[0], 130 / 105 - 1)

    def test_missing_next_open_is_undetermined_not_false(self):
        """翌営業日に寄りが付かなければ買えない。負例に数えない。"""
        closes = [100, 130, 130, 130, 130]
        out = B.attach_rise_label(frame(closes, [100, None, 130, 130, 130]), reach())
        self.assertTrue(pd.isna(out["label"].iloc[0]))
        self.assertTrue(pd.isna(out["entry_price"].iloc[0]))
        # 翌日の寄りがある隣の行は決まる（130/130-1 = 0% で負例）
        self.assertFalse(bool(out["label"].iloc[1]))
        # 終値基準なら同じ行は決まる（前の定義との違いはここだけ）
        old = B.attach_rise_label(frame(closes, [100, None, 130, 130, 130]),
                                  reach(entry="close"))
        self.assertTrue(bool(old["label"].iloc[0]))

    def test_keep_days_counts_from_the_next_open(self):
        closes = [100, 105, 115, 116, 117, 100, 100]
        opens = [100, 106, 110, 115, 116, 110, 100]
        cfg = reach(horizon=5, threshold=0.10, keep_days=2)
        by_open = B.attach_rise_label(frame(closes, opens), cfg)
        by_close = B.attach_rise_label(frame(closes, opens), reach(horizon=5, threshold=0.10,
                                                                   keep_days=2, entry="close"))
        # 寄り 106 × 1.10 = 116.6 以上の終値は 117 だけ。終値 100 × 1.10 = 110 以上は3日
        self.assertEqual(by_open["keep_days_cnt"].iloc[0], 1)
        self.assertEqual(by_close["keep_days_cnt"].iloc[0], 3)
        self.assertFalse(bool(by_open["label"].iloc[0]))
        self.assertTrue(bool(by_close["label"].iloc[0]))

    def test_next_open_does_not_leak_across_codes(self):
        """銘柄の最後の行の「翌日の寄り」を、次の銘柄の最初の行から取らない。"""
        a = frame([100, 110, 120, 130], [100, 101, 111, 121], code="1111")
        b = frame([500, 510, 520, 530], [500, 501, 511, 521], code="2222")
        out = B.attach_rise_label(pd.concat([a, b], ignore_index=True), reach())
        self.assertTrue(pd.isna(out["entry_price"].iloc[3]))
        self.assertEqual(out["entry_price"].iloc[4], 501.0)

    def test_unknown_entry_is_rejected(self):
        with self.assertRaises(SystemExit):
            B.attach_rise_label(frame([100] * 5, [100] * 5), reach(entry="vwap"))

    def test_next_open_needs_the_open_column(self):
        df = frame([100] * 5, [100] * 5).drop(columns=["open"])
        with self.assertRaises(SystemExit):
            B.attach_rise_label(df, reach())
        # 終値基準は始値が無くても作れる（前の定義と同じ）
        B.attach_rise_label(df, reach(entry="close"))


def bars(n=12):
    """分割調整後の始値（AdjO）が素の始値（O）と違う日足（2:1 の分割をまたぐ想定）。"""
    d = pd.bdate_range("2021-01-04", periods=n)
    c = np.linspace(1000, 1300, n)
    o = c * 0.99
    df = pd.DataFrame({"Date": d, "Code": "7777",
                       "O": o, "H": c * 1.01, "L": c * 0.98, "C": c,
                       "Vo": 10000.0, "Va": c * 10000.0,
                       "AdjO": o / 2, "AdjH": c * 1.01 / 2, "AdjL": c * 0.98 / 2,
                       "AdjC": c / 2, "AdjVo": 20000.0})
    df.loc[5, "AdjO"] = np.nan          # 調整後が欠けた日は素の値で埋める（終値と同じ規則）
    return df


class TestPanelOpen(unittest.TestCase):
    def test_panel_open_is_the_adjusted_open(self):
        b = bars()
        panel = B.price_panel(b)
        self.assertIn("open", panel.columns)
        self.assertTrue(np.allclose(panel["open"].drop(index=5), b["AdjO"].drop(index=5)))
        self.assertAlmostEqual(panel["open"].iloc[5], b["O"].iloc[5])

    def test_same_entry_as_the_realized_returns(self):
        """目的変数の買値は、収益の計算（ret_o1）の買値と同じ（AdjO[t+1]）。"""
        import lab
        b = bars()
        lab_out = lab.realized_returns(b)
        entry_lab = (b.sort_values("Date")["AdjC"].to_numpy()
                     * (1.0 + lab_out.sort_values("Date")["entry_gap"].to_numpy()))
        panel = B.attach_rise_label(B.price_panel(b), reach())
        entry_label = panel.sort_values("Date")["entry_price"].to_numpy()
        both = ~np.isnan(entry_lab) & ~np.isnan(entry_label)
        self.assertGreater(both.sum(), 5)
        self.assertTrue(np.allclose(entry_lab[both], entry_label[both]))


class TestLeak(unittest.TestCase):
    def test_entry_price_is_a_future_column(self):
        self.assertIn("entry_price", B.FUTURE_COLS)

    def test_open_and_entry_price_are_never_features(self):
        import features as F
        cols = set(F.all_columns())
        self.assertNotIn("entry_price", cols)
        self.assertNotIn("open", cols)


if __name__ == "__main__":
    unittest.main(verbosity=2)
