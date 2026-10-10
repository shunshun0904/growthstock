#!/usr/bin/env python3
"""
実験71（research/exp/e71_borderline_edinet.py）: 際どい候補を EDINET DB の3つの比率で分ける。重い計算を回さない部分。

- 比率: 粗利率・販管費率は edinet_features と同じ定義。営業外依存度は 経常 vs 営業 の対称変化率で、IFRS だけ
  税引前利益で代える（日本基準で経常利益が無ければ欠測）
- 年度の表: 同じ年度の2つ目の書類（訂正）は使わない。前年差は会計基準が変わった年度とは比べない。avail_date は提出日の翌日
- 業種相対: 月末の時点で各社の最新の有報だけから中央値（先の書類・古すぎる書類は使わない。社数が足りなければ欠測）
- 3分位: その月より前の行だけで線を引く
- main: 作った表で最後まで回り、3分位の件数が漏れなく重ならない

  python3 tests/test_e71_borderline_edinet.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e71_borderline_edinet as E  # noqa: E402


def fin_rows():
    """2社ぶんの年次財務（EDINET DB の応答の形）。"""
    base = dict(revenue=1000.0, gross_profit=400.0, sga=250.0, operating_income=150.0,
                ordinary_income=180.0, profit_before_tax=170.0, basis="consolidated")
    return pd.DataFrame([
        # A: 日本基準 2023・2024（2024 は訂正報告書が後から出た）・2025 は IFRS に変わった
        {**base, "jq_code": "10000", "fiscal_year": 2023, "submit_date": "2023-06-28 15:00", "accounting_standard": "JP"},
        {**base, "jq_code": "10000", "fiscal_year": 2024, "submit_date": "2024-06-26 10:00", "accounting_standard": "JP",
         "gross_profit": 450.0, "sga": 300.0, "ordinary_income": 120.0},
        {**base, "jq_code": "10000", "fiscal_year": 2024, "submit_date": "2024-07-10 10:00", "accounting_standard": "JP",
         "gross_profit": 999.0},                                                  # 訂正。使わない
        {**base, "jq_code": "10000", "fiscal_year": 2025, "submit_date": "2025-06-25 10:00", "accounting_standard": "IFRS",
         "ordinary_income": np.nan, "profit_before_tax": 200.0},
        # B: 日本基準だが経常利益が無い（税引前はある）
        {**base, "jq_code": "20000", "fiscal_year": 2024, "submit_date": "2024-06-20 10:00", "accounting_standard": "JP",
         "ordinary_income": np.nan},
    ])


class TestMeasures(unittest.TestCase):
    def test_ratios_and_lags(self):
        m = E.measures(fin_rows()).set_index(["jq_code", "fiscal_year"])
        self.assertEqual(len(m), 4)                                                # 訂正は数えない
        a24 = m.loc[("10000", 2024)]
        self.assertAlmostEqual(a24["ed_gpm_level"], 0.45)                          # 最初の提出の値
        self.assertAlmostEqual(a24["ed_sga_r_level"], 0.30)
        self.assertAlmostEqual(a24["ed_nonop_level"], (120 - 150) / ((120 + 150) / 2))
        self.assertAlmostEqual(a24["ed_gpm_chg1"], 0.45 - 0.40)
        self.assertAlmostEqual(a24["ed_sga_r_chg1"], 0.30 - 0.25)
        self.assertEqual(a24["avail_date"], pd.Timestamp("2024-06-27"))          # 提出日の翌日
        a25 = m.loc[("10000", 2025)]
        self.assertAlmostEqual(a25["ed_nonop_level"], (200 - 150) / ((200 + 150) / 2))   # IFRS は税引前で代える
        self.assertTrue(np.isnan(a25["ed_gpm_chg1"]))                              # 会計基準が変わったので比べない
        self.assertTrue(np.isnan(a25["ed_nonop_chg1"]))
        a23 = m.loc[("10000", 2023)]
        self.assertTrue(np.isnan(a23["ed_gpm_chg1"]))                              # 前年が無い
        b = m.loc[("20000", 2024)]
        self.assertTrue(np.isnan(b["ed_nonop_level"]))                             # 日本基準で経常が無い → 欠測
        self.assertAlmostEqual(b["ed_gpm_level"], 0.40)

    def test_nonop_bounds(self):
        f = pd.DataFrame({"operating_income": [0.0, 100.0, 100.0], "ordinary_income": [100.0, 100.0, -100.0],
                          "accounting_standard": ["JP"] * 3})
        v = E.nonop_dependence(f)
        self.assertAlmostEqual(v.iloc[0], 2.0)                                     # 営業 0・経常プラス → 全部が営業外
        self.assertAlmostEqual(v.iloc[1], 0.0)
        self.assertAlmostEqual(v.iloc[2], -2.0)


class TestIndustry(unittest.TestCase):
    def feats(self):
        rows = []
        for i in range(6):
            rows.append({"jq_code": f"{i+1}0000", "fiscal_year": 2024, "avail_date": pd.Timestamp("2024-06-27"),
                         "ed_gpm_level": 0.1 * (i + 1), "ed_sga_r_level": 0.05, "ed_nonop_level": 0.0,
                         "ed_gpm_chg1": 0.0, "ed_sga_r_chg1": 0.0, "ed_nonop_chg1": 0.0})
        # 1社は 2025 の有報が 2025-06-27 に出た（粗利率が跳ねた）。別の1社は古い書類しか無い
        rows.append({**rows[0], "fiscal_year": 2025, "avail_date": pd.Timestamp("2025-06-27"), "ed_gpm_level": 0.9})
        rows.append({**rows[1], "jq_code": "70000", "fiscal_year": 2022, "avail_date": pd.Timestamp("2022-06-27"),
                     "ed_gpm_level": 0.0})
        return pd.DataFrame(rows)

    def companies(self):
        return pd.DataFrame({"code": [f"{i+1}0000" for i in range(7)] + ["80000"],
                             "industry": ["foods"] * 7 + [None]})

    def test_reference_is_point_in_time(self):
        ref = E.industry_reference(self.feats(), self.companies(),
                                   pd.DatetimeIndex(["2024-05-31", "2024-06-30", "2025-06-30", "2025-07-31"]), min_n=5)
        r = ref.set_index("month_end")
        self.assertNotIn(pd.Timestamp("2024-05-31"), r.index)                      # まだ1社も無い
        self.assertEqual(r.loc["2024-06-30", "n"], 6)                              # 古すぎる 70000 は入らない
        self.assertAlmostEqual(r.loc["2024-06-30", "ref_gpm"], 0.35)              # 0.1〜0.6 の中央値
        self.assertAlmostEqual(r.loc["2025-06-30", "ref_gpm"], 0.45)              # 2025-06-27 の書類は 6/30 には入る（0.1 → 0.9）→ 中央値 (0.4+0.5)/2
        self.assertEqual(r.loc["2025-06-30", "n"], 6)
        # 2025-07-31: 2024-06-27 の書類は 400日以内（399日）なのでまだ有効
        self.assertEqual(r.loc["2025-07-31", "n"], 6)

    def test_reference_values(self):
        ref = E.industry_reference(self.feats(), self.companies(), pd.DatetimeIndex(["2025-06-30"]), min_n=5)
        self.assertAlmostEqual(float(ref["ref_gpm"].iloc[0]), 0.45)               # {0.9,0.2,...,0.6} → (0.4+0.5)/2
        few = E.industry_reference(self.feats(), self.companies(), pd.DatetimeIndex(["2025-06-30"]), min_n=7)
        self.assertTrue(np.isnan(few["ref_gpm"].iloc[0]))                          # 社数が足りない

    def test_attach_relative_uses_previous_month_end(self):
        ref = pd.DataFrame({"month_end": pd.to_datetime(["2024-06-30", "2024-07-31"]), "industry": ["foods", "foods"],
                            "n": [6, 6], "ref_gpm": [0.35, 0.40], "ref_sga_r": [0.05, 0.05], "ref_nonop": [0.0, 0.1]})
        d = pd.DataFrame({"Code": ["10000", "10000", "80000", "99000"],
                          "Date": pd.to_datetime(["2024-07-15", "2024-08-05", "2024-07-15", "2024-07-15"]),
                          "ed_gpm_level": [0.5, 0.5, 0.5, 0.5], "ed_sga_r_level": [0.1] * 4, "ed_nonop_level": [0.3] * 4})
        out = E.attach_relative(d, ref, self.companies())
        self.assertAlmostEqual(out["ed_gpm_rel"].iloc[0], 0.5 - 0.35)             # 7月の行は 6月末の中央値
        self.assertAlmostEqual(out["ed_gpm_rel"].iloc[1], 0.5 - 0.40)             # 8月の行は 7月末
        self.assertAlmostEqual(out["ed_nonop_rel"].iloc[1], 0.3 - 0.1)
        self.assertTrue(np.isnan(out["ed_gpm_rel"].iloc[2]))                       # 業種が無い
        self.assertTrue(np.isnan(out["ed_gpm_rel"].iloc[3]))                       # 対応表に無い
        self.assertEqual(len(out), 4)


class TestTertiles(unittest.TestCase):
    def test_lines_use_only_earlier_months(self):
        dates = pd.Series(pd.to_datetime(["2024-01-10"] * 6 + ["2024-02-10"] * 4))
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 3.5, 6.0, np.nan])
        t = E.tertiles(dates, x, min_rows=6)
        self.assertTrue(t.iloc[:6].isna().all())                                   # 1月は前の月が無い
        self.assertEqual(list(t.iloc[6:9]), ["lo", "mid", "hi"])
        self.assertIsNone(t.iloc[9])                                               # 値が無い


class TestEndToEnd(unittest.TestCase):
    def test_main_runs(self):
        rng = np.random.default_rng(3)
        n = 6000
        dates = pd.bdate_range("2023-01-04", "2026-08-31")
        d = pd.DataFrame({"Code": [f"{i:04d}0" for i in rng.integers(0, 800, n)], "Date": rng.choice(dates, n)})
        hp = rng.uniform(60, 100, (n, 4))
        for i, a in enumerate(("lgbm", "xgb", "cat", "logit")):
            d[f"hp_{a}"] = np.round(hp[:, i], 1)
        d["r"] = rng.normal(1, 10, n)
        d["hit10"] = rng.random(n) < 0.3
        d["tp10"] = np.where(d["hit10"], 10.0, d["r"])
        d["label"] = (d["r"] > 5).astype(int)
        d["n_break"] = rng.integers(1, 40, n)
        d["entry"] = 100.0
        for k, _, _ in E.MEASURES:
            for v, _ in E.VARIANTS:
                d[f"ed_{k}_{v}"] = np.where(rng.random(n) < 0.7, rng.normal(0, 1, n), np.nan)
        tmp = tempfile.mkdtemp()
        try:
            out = os.path.join(tmp, "e71.json")
            with mock.patch.object(E, "load", return_value=(d.sort_values("Date").reset_index(drop=True), True, True)), \
                    mock.patch.object(E, "OUT", out), mock.patch("builtins.print"):
                self.assertEqual(E.main([]), 0)
            with open(out, encoding="utf-8") as fh:
                res = json.load(fh)
            self.assertEqual(res["rows"], n)
            self.assertIn("gpm_level_border_lo", res["split"])
            self.assertIn("nonop_rel_all_diff_hold", res["split"])
            self.assertIn("level_border_2plus", res["nbad"])
            self.assertIn("chg1_weak_悪い印なし", res["nbad"])
            for k, _, _ in E.MEASURES:
                for v, _ in E.VARIANTS:
                    s = res["split"]
                    three = sum(s[f"{k}_{v}_all_{t}"]["n"] for t in ("lo", "mid", "hi"))
                    self.assertGreater(three, 0)
                    bad = {"gpm": "lo", "sga_r": "hi", "nonop": "hi"}[k]
                    self.assertEqual(three, s[f"{k}_{v}_all_{bad}"]["n"] + s[f"{k}_{v}_all_rest"]["n"])
                    self.assertLessEqual(three, int(d[f"ed_{k}_{v}"].notna().sum()))
            nb = res["nbad"]
            self.assertEqual(sum(nb[f"level_all_{i}"]["n"] for i in range(4)),
                             nb["level_all_0"]["n"] + nb["level_all_1"]["n"] + nb["level_all_2plus"]["n"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
