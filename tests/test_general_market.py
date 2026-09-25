#!/usr/bin/env python3
"""
TOKYO PRO MARKET（TPM）から一般市場へ移った銘柄の「78週の履歴」と、上場からの年数
（運用者の選択 A と ①、2026-09-25）のテスト。

J-Quants は TPM の銘柄にも日足の行を持つが、値はほぼ空。移った銘柄は、その空の行も
368本の履歴に数えられ、実際には9か月ほどの高値が「78週高値」になっていた（5537）。
移った9銘柄のうち8銘柄は TPM に上場した日などに値のある行を持つので、「最初に値が
付いた日から数える」だけでは直らない。最後に TPM だった月末の後で、最初に値が付いた
日から数える。

モデル（research/build_dataset.py）と画面（scripts/jquants_data_fetcher.py）が
同じ日から数えること（general_market_start.json を経由して）もここで確かめる。

  python3 tests/test_general_market.py
"""
import datetime as dt
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import build_dataset as B  # noqa: E402
import jquants_data_fetcher as JF  # noqa: E402

W = B.HIGH_WINDOW                 # 368
DAYS = pd.bdate_range("2019-01-01", periods=1000)
MOVE = 400                        # 行 400 から一般市場（その前は TPM で値はほぼ空）


def bars_for(code: str, priced) -> pd.DataFrame:
    """priced[i] が True の行だけ値がある日足（高値 = 終値 = 100 + i）。"""
    px = np.where(priced, 100.0 + np.arange(len(DAYS)), np.nan)
    return pd.DataFrame({"Date": DAYS, "Code": code, "O": px, "H": px, "L": px, "C": px,
                         "Vo": np.where(priced, 1000.0, np.nan),
                         "AdjO": px, "AdjH": px, "AdjL": px, "AdjC": px,
                         "AdjVo": np.where(priced, 1000.0, np.nan)})


def mover_priced():
    p = np.zeros(len(DAYS), dtype=bool)
    p[0] = True                   # TPM に上場した日の約定（移った8銘柄にあった形）
    p[150] = True                 # TPM の時期のまれな約定
    p[MOVE:] = True
    return p


def segments() -> pd.DataFrame:
    """月末ごとの市場区分。TPM は移る前の月末まで、その後はグロース。"""
    ends = pd.date_range(DAYS[0], DAYS[-1], freq="BME")
    rows = []
    for d in ends:
        rows.append({"Date": d, "Code": "11110",
                     "MktNm": B.TPM_NAME if d < DAYS[MOVE] else "グロース"})
        rows.append({"Date": d, "Code": "22220", "MktNm": "プライム"})
        rows.append({"Date": d, "Code": "33330", "MktNm": B.TPM_NAME})      # まだ TPM
        if d < DAYS[600]:
            rows.append({"Date": d, "Code": "44440", "MktNm": B.TPM_NAME})  # TPM のまま消えた
    return pd.DataFrame(rows)


def all_bars() -> pd.DataFrame:
    vanished = np.zeros(len(DAYS), dtype=bool)
    vanished[[0, 610]] = True      # 最後の TPM 月末の後に約定があっても「移った」ではない
    return pd.concat([
        bars_for("11110", mover_priced()),
        bars_for("22220", np.ones(len(DAYS), dtype=bool)),
        bars_for("33330", np.eye(1, len(DAYS), 0, dtype=bool)[0]),
        bars_for("44440", vanished).iloc[:620],
    ], ignore_index=True)


class TestGeneralMarketStart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gm = B.general_market_start(all_bars(), segments()).set_index("Code")

    def test_mover_starts_at_the_first_price_after_the_last_tpm_month(self):
        """TPM に上場した日（行0）とまれな約定（行150）からは数えない。"""
        self.assertTrue(self.gm.loc["11110", "tpm"])
        self.assertEqual(self.gm.loc["11110", "start"], DAYS[MOVE])
        self.assertTrue(self.gm.loc["11110", "known"])

    def test_normal_stock_starts_at_its_first_row(self):
        self.assertFalse(self.gm.loc["22220", "tpm"])
        self.assertEqual(self.gm.loc["22220", "start"], DAYS[0])
        self.assertFalse(self.gm.loc["22220", "known"])     # データの初日から = 上場日は不明

    def test_still_tpm_and_vanished_never_start(self):
        for c in ("33330", "44440"):
            self.assertTrue(pd.isna(self.gm.loc[c, "start"]), c)
            self.assertFalse(self.gm.loc[c, "known"], c)

    def test_without_segments_nothing_is_tpm(self):
        gm = B.general_market_start(all_bars(), None).set_index("Code")
        self.assertFalse(gm["tpm"].any())
        self.assertEqual(gm.loc["11110", "start"], DAYS[0])


class TestWindowCountsOnlyGeneralMarketRows(unittest.TestCase):
    """モデル（price_panel）と画面（price_metrics / build_milestones）が同じ日から数える。"""

    @classmethod
    def setUpClass(cls):
        cls.bars = all_bars()
        gm = B.general_market_start(cls.bars, segments())
        cls.start = gm.loc[gm["tpm"]].set_index("Code")["start"]
        cls.off = B.price_panel(cls.bars[cls.bars["Code"].isin(["11110", "22220"])])
        cls.on = B.price_panel(cls.bars[cls.bars["Code"].isin(["11110", "22220"])],
                               start=cls.start)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.json_path = os.path.join(cls.tmp.name, "gm.json")
        B.write_general_market_start(gm, cls.json_path, enabled=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def first_high(self, panel, code):
        x = panel[(panel["Code"] == code) & panel["high52w"].notna()]
        return int(np.searchsorted(DAYS.values, x["Date"].min().to_datetime64()))

    def test_model_waits_for_368_rows_on_the_general_market(self):
        # 今まで: 行0から数え、窓の半分（184本）に値がそろった行 MOVE+183 で通っていた
        self.assertEqual(self.first_high(self.off, "11110"), MOVE + W // 2 - 1)
        # A: 移ってから 368本
        self.assertEqual(self.first_high(self.on, "11110"), MOVE + W - 1)

    def test_normal_stock_is_unchanged(self):
        a = self.off[self.off["Code"] == "22220"]["high52w"].to_numpy()
        b = self.on[self.on["Code"] == "22220"]["high52w"].to_numpy()
        self.assertTrue(np.array_equal(np.isnan(a), np.isnan(b)))
        self.assertTrue(np.allclose(a[~np.isnan(a)], b[~np.isnan(b)]))

    def quotes(self, code):
        b = self.bars[self.bars["Code"] == code]
        return [{"Date": d.date().isoformat(),
                 **{k: (None if pd.isna(v) else float(v)) for k, v in r.items()
                    if k not in ("Date", "Code")}}
                for d, (_, r) in zip(b["Date"], b.iterrows())]

    def test_screen_uses_the_same_day_through_the_json(self):
        gm_start = JF.load_general_market_start(self.json_path)
        self.assertEqual(gm_start["11110"], DAYS[MOVE].date().isoformat())
        self.assertIsNone(gm_start["33330"])
        count_from = JF.count_from_for("11110", gm_start)
        q = self.quotes("11110")
        k = MOVE + W - 1
        self.assertIsNone(JF.price_metrics(q[:k], count_from=count_from)["high52w"])
        self.assertIsNotNone(JF.price_metrics(q[:k + 1], count_from=count_from)["high52w"])
        # 今までどおりに数えると、もっと早く通ってしまう（直す前の画面）
        self.assertIsNotNone(JF.price_metrics(q[:MOVE + W // 2])["high52w"])
        # モデルの78週高値と同じ値
        on = self.on[(self.on["Code"] == "11110")].reset_index(drop=True)
        self.assertAlmostEqual(JF.price_metrics(q[:k + 1], count_from=count_from)["high52w"],
                               on.loc[k, "high52w"])

    def test_still_tpm_never_gets_a_high_on_the_screen(self):
        gm_start = JF.load_general_market_start(self.json_path)
        self.assertEqual(JF.count_from_for("33330", gm_start), JF.NEVER)
        self.assertIsNone(JF.count_from_for("22220", gm_start))      # 一覧に無い = 今までどおり
        q = self.quotes("22220")
        self.assertIsNone(JF.price_metrics(q, count_from=JF.NEVER)["high52w"])
        self.assertIsNotNone(JF.price_metrics(q)["high52w"])

    def test_milestones_wait_for_the_same_history(self):
        q = self.quotes("11110")
        count_from = DAYS[MOVE].date().isoformat()
        ev = JF.build_milestones(q, [], months=60, count_from=count_from)
        hi = [e["date"] for e in ev if e["type"] == "breakout"]
        self.assertTrue(hi)
        self.assertGreaterEqual(min(hi), DAYS[MOVE + W].date().isoformat())
        before = [e["date"] for e in JF.build_milestones(q, [], months=60) if e["type"] == "breakout"]
        self.assertLess(min(before), DAYS[MOVE + W].date().isoformat())


class TestSwitch(unittest.TestCase):
    def test_disabled_or_missing_file_keeps_the_old_behaviour(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "gm.json")
            gm = B.general_market_start(all_bars(), segments())
            B.write_general_market_start(gm, p, enabled=False)
            self.assertIsNone(JF.load_general_market_start(p))
            self.assertIsNone(JF.load_general_market_start(os.path.join(d, "無い.json")))
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("{壊れた")
            self.assertIsNone(JF.load_general_market_start(p))
        self.assertIsNone(JF.load_general_market_start(None))

    def test_default_is_off_until_experiment_47(self):
        """実験47で測る前に本番の母集団を変えない。"""
        if os.environ.get("SWEEP_GENERAL_MARKET_START") is None:
            self.assertFalse(B.GENERAL_MARKET_START)

    def test_json_only_lists_codes_that_were_on_tpm(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "gm.json")
            B.write_general_market_start(B.general_market_start(all_bars(), segments()), p, True)
            with open(p, encoding="utf-8") as fh:
                payload = json.load(fh)
        self.assertEqual(sorted(payload["start"]), ["11110", "33330", "44440"])
        self.assertTrue(payload["enabled"])


class TestListingYears(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        extra = bars_for("55550", np.r_[np.zeros(300, dtype=bool), np.ones(700, dtype=bool)])
        extra = extra.iloc[300:]                          # データの途中で新規上場
        early = bars_for("66660", np.ones(len(DAYS), dtype=bool)).iloc[100:]   # 3年を超える
        cls.gm = B.general_market_start(pd.concat([all_bars(), extra, early], ignore_index=True),
                                        segments())

    def years(self, code, i):
        return float(B.listing_years(pd.Series([code]), pd.Series([DAYS[i]]), self.gm)[0])

    def test_new_listing_counts_from_its_first_row_and_caps_at_three(self):
        self.assertAlmostEqual(self.years("55550", 300 + 261),
                               (DAYS[561] - DAYS[300]).days / 365.25)
        # 上場から3年を超えたら 3 で打ち止め（行100に上場、行999は約3.4年後）
        self.assertGreater((DAYS[999] - DAYS[100]).days / 365.25, 3.0)
        self.assertEqual(self.years("66660", 999), 3.0)
        self.assertLess(self.years("66660", 500), 3.0)

    def test_mover_counts_from_the_general_market(self):
        self.assertAlmostEqual(self.years("11110", 800), (DAYS[800] - DAYS[MOVE]).days / 365.25)

    def test_listing_before_the_data_is_unknown_until_three_years_pass(self):
        three = int(np.searchsorted(DAYS.values, (DAYS[0] + pd.DateOffset(years=3)).to_datetime64()))
        self.assertTrue(np.isnan(self.years("22220", three - 1)))
        self.assertEqual(self.years("22220", three), 3.0)

    def test_still_tpm_is_missing(self):
        self.assertTrue(np.isnan(self.years("33330", 999)))


if __name__ == "__main__":
    unittest.main()
