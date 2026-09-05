#!/usr/bin/env python3
"""
research/build_dataset.py のラベル生成ロジックの単体テスト。

株価予測で最も起きやすい事故は「未来を見てしまう」ことなので、
ブレイクアウト判定とラベル窓の境界を合成データで固定して検証する。

  python3 tests/test_dataset.py
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from build_dataset import (  # noqa: E402
    HOLD_DAYS, HORIZON_END, HORIZON_START, HIGH_WINDOW, LabelConfig,
    attach_labels, breakout_flags, price_panel, quarterize_panel,
)


def make_bars(closes, vols=None, code="00010", start="2020-01-01"):
    """1銘柄ぶんの日次バーを作る（High=Close, カレンダーは連番の営業日とみなす）。"""
    n = len(closes)
    vols = vols if vols is not None else [100000] * n
    d0 = dt.date.fromisoformat(start)
    return pd.DataFrame({
        "Date": [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)],
        "Code": [code] * n,
        "O": closes, "H": closes, "L": closes, "C": closes,
        "Vo": vols, "Va": [c * v for c, v in zip(closes, vols)],
        "AdjO": closes, "AdjH": closes, "AdjL": closes, "AdjC": closes, "AdjVo": vols,
    })


def flat_then(base_len, base_price, tail_closes, tail_vols=None, base_vol=100000):
    closes = [base_price] * base_len + list(tail_closes)
    vols = [base_vol] * base_len + list(tail_vols if tail_vols else [base_vol] * len(tail_closes))
    return closes, vols


class TestBreakoutDetection(unittest.TestCase):
    def _flags(self, closes, vols):
        df = price_panel(make_bars(closes, vols))
        return breakout_flags(df)


    def test_detects_valid_breakout(self):
        """高値更新 + 出来高1.5倍以上 + 20日定着 の3条件が揃えば True。"""
        n = HIGH_WINDOW + 5
        closes, vols = flat_then(n, 1000, [1200] + [1200] * (HOLD_DAYS + 5),
                                 [300000] + [100000] * (HOLD_DAYS + 5))
        df = self._flags(closes, vols)
        bo = df.loc[df["is_breakout"] == True]  # noqa: E712
        self.assertEqual(len(bo), 1, "ブレイク日はちょうど1日であるべき")
        self.assertEqual(bo.iloc[0]["close"], 1200)

    def test_rejects_breakout_without_volume(self):
        """出来高が伴わない高値更新は除外する（だまし対策）。"""
        n = HIGH_WINDOW + 5
        closes, vols = flat_then(n, 1000, [1200] * (HOLD_DAYS + 6),
                                 [100000] * (HOLD_DAYS + 6))
        df = self._flags(closes, vols)
        self.assertEqual(int((df["is_breakout"] == True).sum()), 0)  # noqa: E712

    def test_rejects_breakout_that_collapses(self):
        """ブレイク後に-8%超下落したら正例にしない（定着条件）。"""
        n = HIGH_WINDOW + 5
        tail = [1200] + [1000] * (HOLD_DAYS + 5)   # 翌日に -16%
        tvol = [300000] + [100000] * (HOLD_DAYS + 5)
        closes, vols = flat_then(n, 1000, tail, tvol)
        df = self._flags(closes, vols)
        self.assertEqual(int((df["is_breakout"] == True).sum()), 0)  # noqa: E712

    def test_hold_boundary_is_exactly_8_percent(self):
        """-8%ちょうどは許容、-8%超は不可。"""
        n = HIGH_WINDOW + 5
        for drop, expected in [(0.92, 1), (0.919, 0)]:
            tail = [1200] + [1200 * drop] * (HOLD_DAYS + 5)
            tvol = [300000] + [100000] * (HOLD_DAYS + 5)
            closes, vols = flat_then(n, 1000, tail, tvol)
            df = self._flags(closes, vols)
            self.assertEqual(int((df["is_breakout"] == True).sum()), expected,  # noqa: E712
                             f"drop={drop} の判定が期待と違う")

    def test_undetermined_at_series_end_is_nan(self):
        """定着を評価できない末尾は False ではなく NaN。"""
        n = HIGH_WINDOW + 5
        closes, vols = flat_then(n, 1000, [1200], [300000])
        df = self._flags(closes, vols)
        self.assertTrue(pd.isna(df.iloc[-1]["is_breakout"]),
                        "将来データが無い日は判定不能(NaN)であるべき")


class TestSustainCondition(unittest.TestCase):
    """定着条件 (sustain): 一定期間後も水準を保っているか。"""

    def _flags(self, closes, vols, cfg):
        return breakout_flags(price_panel(make_bars(closes, vols), cfg), cfg)

    def test_sustain_rejects_fade(self):
        """ブレイク後にじりじり戻して水準を割ったら正例にしない。"""
        cfg = LabelConfig(sustain_days=60, sustain_ratio=1.0)
        n = HIGH_WINDOW + 5
        # ブレイク直後は下げないので hold は通るが、60日後には水準を割る
        tail = [1200] + [1180] * 30 + [1100] * 40
        tvol = [300000] + [100000] * 70
        closes, vols = flat_then(n, 1000, tail, tvol)
        df = self._flags(closes, vols, cfg)
        self.assertEqual(int((df["is_breakout"] == True).sum()), 0)  # noqa: E712

    def test_sustain_accepts_holding_level(self):
        """60日後も水準を保っていれば正例。"""
        cfg = LabelConfig(sustain_days=60, sustain_ratio=1.0)
        n = HIGH_WINDOW + 5
        tail = [1200] + [1250] * 70
        tvol = [300000] + [100000] * 70
        closes, vols = flat_then(n, 1000, tail, tvol)
        df = self._flags(closes, vols, cfg)
        self.assertEqual(int((df["is_breakout"] == True).sum()), 1)  # noqa: E712

    def test_sustain_is_stricter_than_hold_alone(self):
        """同じ系列で、定着条件ありのほうが正例が減る(増えることはない)。"""
        n = HIGH_WINDOW + 5
        tail = [1200] + [1190] * 30 + [1120] * 40
        tvol = [300000] + [100000] * 70
        closes, vols = flat_then(n, 1000, tail, tvol)
        base = self._flags(closes, vols, LabelConfig())
        strict = self._flags(closes, vols, LabelConfig(sustain_days=60, sustain_ratio=1.0))
        self.assertLessEqual(int((strict["is_breakout"] == True).sum()),  # noqa: E712
                             int((base["is_breakout"] == True).sum()))

    def test_undetermined_when_sustain_horizon_missing(self):
        """sustain を評価できない末尾は False ではなく NaN。"""
        cfg = LabelConfig(sustain_days=60, sustain_ratio=1.0)
        n = HIGH_WINDOW + 5
        closes, vols = flat_then(n, 1000, [1200] + [1250] * 10, [300000] + [100000] * 10)
        df = self._flags(closes, vols, cfg)
        self.assertTrue(pd.isna(df.iloc[-1]["is_breakout"]))

    def test_forward_needed_accounts_for_sustain(self):
        self.assertEqual(LabelConfig(horizon_end=60, hold_days=20).forward_needed, 80)
        self.assertEqual(
            LabelConfig(horizon_end=60, hold_days=20, sustain_days=60).forward_needed, 120)


class TestHighWindowParameter(unittest.TestCase):
    def test_78week_window_needs_longer_history(self):
        """78週(368日)窓では、368営業日そろうまで高値が未定義になる。"""
        cfg = LabelConfig(high_window=368)
        n = 400
        df = price_panel(make_bars(list(range(1000, 1000 + n)), [100000] * n), cfg)
        self.assertTrue(df["high52w"].iloc[:367].isna().all())
        self.assertTrue(df["high52w"].iloc[367:].notna().all())

    def test_wider_window_is_harder_to_break(self):
        """同じ系列なら、78週高値のほうが52週高値以上になる(超えにくい)。"""
        n = 400
        # 前半に高値、後半は低い水準から回復する形
        closes = list(range(1500, 1500 + 100)) + list(range(1400, 1400 + 300))
        a = price_panel(make_bars(closes, [100000] * n), LabelConfig(high_window=245))
        b = price_panel(make_bars(closes, [100000] * n), LabelConfig(high_window=368))
        both = a["high52w"].notna() & b["high52w"].notna()
        self.assertTrue((b.loc[both, "high52w"] >= a.loc[both, "high52w"]).all())


class TestLabelWindow(unittest.TestCase):
    def _label_at(self, breakout_offset):
        """基準日から breakout_offset 営業日後にブレイクを置き、基準日のラベルを返す。"""
        base = HIGH_WINDOW + 10           # 基準日の位置
        # ラベル確定には基準日から HORIZON_END + HOLD_DAYS 営業日ぶん必要
        n_tail = max(breakout_offset, HORIZON_END) + HOLD_DAYS + 30
        closes = [1000] * base
        vols = [100000] * base
        for i in range(n_tail):
            if i == breakout_offset:
                closes.append(1200); vols.append(300000)
            elif i > breakout_offset:
                closes.append(1200); vols.append(100000)
            else:
                closes.append(1000); vols.append(100000)
        df = price_panel(make_bars(closes, vols))
        df = breakout_flags(df)
        df = attach_labels(df)
        return df.iloc[base - 1]["label"]

    def test_breakout_inside_horizon_is_positive(self):
        """ホライズン内(t+20〜t+120)のブレイクは正例。"""
        self.assertEqual(self._label_at(HORIZON_START + 30), 1.0)

    def test_breakout_too_soon_is_not_counted(self):
        """t+20より手前のブレイクはホライズン外なので数えない。"""
        self.assertEqual(self._label_at(HORIZON_START - 5), 0.0)

    def test_breakout_at_horizon_start_is_positive(self):
        """境界 t+20 ちょうどは含む。"""
        self.assertEqual(self._label_at(HORIZON_START), 1.0)

    def test_breakout_beyond_horizon_is_negative(self):
        """t+120 を超えたブレイクは負例。"""
        self.assertEqual(self._label_at(HORIZON_END + 15), 0.0)

    def test_label_is_nan_when_future_data_insufficient(self):
        """
        将来データが HORIZON_END + HOLD_DAYS 営業日ぶん無い基準日は
        「未確定」として NaN。0（起きなかった）に丸めてはいけない。
        """
        n = HIGH_WINDOW + 60
        df = price_panel(make_bars([1000] * n, [100000] * n))
        df = attach_labels(breakout_flags(df))
        need = HORIZON_END + HOLD_DAYS
        self.assertTrue(df["label"].iloc[-need:].isna().all(),
                        "末尾 140営業日はラベル未確定であるべき")
        self.assertTrue(df["label"].iloc[:-need].notna().all(),
                        "それ以前はラベルが確定しているべき")


class TestNoLookahead(unittest.TestCase):
    def test_features_do_not_change_when_future_is_appended(self):
        """
        最重要のテスト。
        系列の後ろに未来のバーを足しても、基準日以前の特徴量は1つも変わらないこと。
        変わるならどこかで未来を見ている。
        """
        n = HIGH_WINDOW + 60
        rng = np.random.default_rng(42)
        closes = list(1000 + np.cumsum(rng.normal(0, 10, n)).round(2))
        vols = list(rng.integers(50000, 200000, n))

        short = price_panel(make_bars(closes, vols))
        long_closes = closes + list(1000 + np.cumsum(rng.normal(0, 10, 200)).round(2))
        long_vols = vols + list(rng.integers(50000, 200000, 200))
        long = price_panel(make_bars(long_closes, long_vols))

        cols = ["close", "high52w", "high52w_prior", "r_high",
                "vol_ma20", "volume_trend", "tv_ma20", "r_high_3m", "r_high_6m"]
        a = short[cols].reset_index(drop=True)
        b = long[cols].iloc[:len(short)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False,
                                      obj="未来のバーを足すと過去の特徴量が変わった(先読みの疑い)")

    def test_high52w_prior_excludes_current_day(self):
        """ブレイク判定の基準は『当日を含まない』52週高値であること。"""
        n = HIGH_WINDOW + 3
        closes = [1000] * n + [1500]
        df = price_panel(make_bars(closes, [100000] * (n + 1)))
        last = df.iloc[-1]
        self.assertEqual(last["high52w"], 1500, "当日込みの高値は当日値を含む")
        self.assertEqual(last["high52w_prior"], 1000, "当日を除いた高値は前日までの最大")


class TestQuarterizePanel(unittest.TestCase):
    def _fins(self):
        rows = []
        # FY2024: 累計 100 -> 220 -> 360 -> 520
        for q, (disc, sales, op, np_, eps) in enumerate(
            [("2024-08-05", 100, 10, 6, 6.0), ("2024-11-05", 220, 24, 15, 15.0),
             ("2025-02-05", 360, 42, 26, 26.0), ("2025-05-12", 520, 64, 40, 40.0)], start=1):
            rows.append({
                "Code": "00010", "DiscDate": disc, "DiscTime": "15:00",
                "CurPerType": ["1Q", "2Q", "3Q", "4Q"][q - 1], "CurFYSt": "2024-04-01",
                "Sales": sales, "OP": op, "NP": np_, "EPS": eps,
                "Eq": 1000, "TA": 2000, "ROE": 4.0, "FOP": 80,
                "FSales": None, "FNP": None, "FEPS": None,
                "ShOutFY": 1_000_000, "TrShFY": 50_000, "DocType": "x", "CurPerEn": disc,
            })
        # FY2025 1Q: 売上 140 (前年同期100 -> +40%)
        rows.append({
            "Code": "00010", "DiscDate": "2025-08-05", "DiscTime": "15:00",
            "CurPerType": "1Q", "CurFYSt": "2025-04-01",
            "Sales": 140, "OP": 16, "NP": 10, "EPS": 10.0,
            "Eq": 1050, "TA": 2100, "ROE": 5.0, "FOP": 90,
            "FSales": None, "FNP": None, "FEPS": None,
            "ShOutFY": 1_000_000, "TrShFY": 50_000, "DocType": "x", "CurPerEn": "2025-08-05",
        })
        return pd.DataFrame(rows)

    def test_cumulative_differencing(self):
        q = quarterize_panel(self._fins()).sort_values("DiscDate").reset_index(drop=True)
        latest = q.iloc[-1]
        self.assertAlmostEqual(latest["sales_growth_q0"], 40.0, places=6)
        self.assertAlmostEqual(latest["eps_growth_q0"], (10.0 - 6.0) / 6.0 * 100, places=6)

    def test_lags_are_previous_disclosures(self):
        q = quarterize_panel(self._fins()).sort_values("DiscDate").reset_index(drop=True)
        latest = q.iloc[-1]
        # 直近が FY2025-1Q なら q1 は FY2024-4Q、q2 は FY2024-3Q
        self.assertAlmostEqual(latest["ROE_q0"], 5.0)
        self.assertAlmostEqual(latest["ROE_q1"], 4.0)

    def test_progress_vs_base(self):
        q = quarterize_panel(self._fins()).sort_values("DiscDate").reset_index(drop=True)
        latest = q.iloc[-1]
        # 1Q 累計営業利益16 / 通期予想90 = 17.8%、基準 1×25 = 25 -> -7.2
        self.assertAlmostEqual(latest["progress_vs_base"], 16 / 90 * 100 - 25, places=6)

    def test_forecast_only_disclosure_is_dropped(self):
        fins = self._fins()
        noise = fins.iloc[[0]].copy()
        noise["DiscDate"] = "2025-09-01"
        noise[["Sales", "OP", "NP", "EPS"]] = np.nan
        q = quarterize_panel(pd.concat([fins, noise], ignore_index=True))
        self.assertNotIn("2025-09-01", set(q["DiscDate"].astype(str).str[:10]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
