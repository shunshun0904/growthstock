#!/usr/bin/env python3
"""
8軸の「進捗期待」（2026-09-25 に運用者の選択 A で入れた定義）の、言語・経路をまたぐテスト。

  基準 = 前年同期の進捗（scripts/jquants_data_fetcher.py の progress_benchmark）
  点数 = 進捗率 ÷ 基準 の、過去の開示の中での順位（src/lib/scoring.js の PROGRESS_TABLE）

同じ定義が4か所に出てくる。ずれると、同じ銘柄の点数が経路によって変わる:
  ・データ取得（stocks.json）          scripts/jquants_data_fetcher.py
  ・予測タブから送った候補              research/predict_daily.progress_fields
  ・点数表の作成                        research/progress_percentiles.py
  ・点数の計算                          src/lib/scoring.js

  python3 tests/test_progress.py
"""
import json
import os
import re
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import jquants_data_fetcher as JF  # noqa: E402
import predict_daily as PD  # noqa: E402
import progress_percentiles as PP  # noqa: E402

SCORING_JS = os.path.join(ROOT, "src", "lib", "scoring.js")


def fins_row(code, fy, period, disc, op, fop=None, sales=100.0):
    """fins_*.parquet と同じ列名の決算の行（日付は Timestamp、数値は float）。"""
    return {"Code": code, "DocType": "FinancialStatements", "CurPerType": period,
            "CurFYSt": fy, "CurPerEn": disc, "DiscDate": pd.Timestamp(disc),
            "Sales": sales, "OP": float(op), "NP": 1.0, "EPS": 1.0,
            "FOP": None if fop is None else float(fop)}


def sample_fins():
    return pd.DataFrame([
        # A: 前年同期の進捗 38.5%（9081 の 2026年度1Q と同じ形）
        fins_row("11110", "2025-04-01", "1Q", "2025-08-05", 385, 1000),
        fins_row("11110", "2025-04-01", "FY", "2026-05-10", 1000),
        fins_row("11110", "2026-04-01", "1Q", "2026-08-05", 469, 1000),
        # A の 2Q（候補の日付より後の開示。見えてはいけない）
        fins_row("11110", "2026-04-01", "2Q", "2026-11-05", 900, 1000),
        # B: 前年が無い -> Q×25%
        fins_row("22220", "2026-04-01", "1Q", "2026-08-06", 30, 100),
        # C: 前年同期の進捗 5% -> 下限 12.5%
        fins_row("33330", "2025-04-01", "1Q", "2025-08-07", 50, 1000),
        fins_row("33330", "2025-04-01", "FY", "2026-05-11", 1000),
        fins_row("33330", "2026-04-01", "1Q", "2026-08-07", 200, 1000),
    ])


class TestPredictionCandidates(unittest.TestCase):
    """予測タブの候補にも、データ取得と同じ値が出ること。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        f = sample_fins()
        f[f["DiscDate"] < "2026-01-01"].to_parquet(os.path.join(cls.tmp.name, "fins_2025.parquet"))
        f[f["DiscDate"] >= "2026-01-01"].to_parquet(os.path.join(cls.tmp.name, "fins_2026.parquet"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_values_as_the_fetcher(self):
        got = PD.progress_fields(self.tmp.name, [("11110", "2026-09-24"), ("22220", "2026-09-24"),
                                                 ("33330", "2026-09-24")])
        self.assertAlmostEqual(got[("11110", "2026-09-24")]["progressRate"], 46.9)
        self.assertAlmostEqual(got[("11110", "2026-09-24")]["progressBenchmark"], 38.5)
        self.assertEqual(got[("11110", "2026-09-24")]["progressBasis"], "seasonal")
        self.assertEqual(got[("11110", "2026-09-24")]["quarter"], 1)
        self.assertEqual(got[("22220", "2026-09-24")]["progressBasis"], "linear")
        self.assertAlmostEqual(got[("22220", "2026-09-24")]["progressBenchmark"], 25.0)
        self.assertEqual(got[("33330", "2026-09-24")]["progressBasis"], "floor")
        self.assertAlmostEqual(got[("33330", "2026-09-24")]["progressBenchmark"], 12.5)

        # データ取得の関数に、API と同じ形（日付は文字列）で渡した結果と一致する
        f = sample_fins()
        rows = [{k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
                 for k, v in r.items()} for r in f[f["Code"] == "11110"].to_dict("records")]
        m = JF.fundamentals_as_of(rows, "2026-09-24")
        for k in ("progressRate", "quarter", "progressBenchmark", "progressBasis"):
            self.assertEqual(got[("11110", "2026-09-24")][k], m[k], k)

    def test_point_in_time(self):
        """候補の日付より後の開示（2026-11-05 の 2Q）は使わない。"""
        got = PD.progress_fields(self.tmp.name, [("11110", "2026-09-24"), ("11110", "2026-11-10")])
        self.assertEqual(got[("11110", "2026-09-24")]["quarter"], 1)
        self.assertEqual(got[("11110", "2026-11-10")]["quarter"], 2)
        self.assertAlmostEqual(got[("11110", "2026-11-10")]["progressRate"], 90.0)
        # 前年の 2Q が無いので、2Q は Q×25%
        self.assertEqual(got[("11110", "2026-11-10")]["progressBasis"], "linear")

    def test_unknown_code_and_empty(self):
        self.assertEqual(PD.progress_fields(self.tmp.name, [("99990", "2026-09-24")]), {})
        self.assertEqual(PD.progress_fields(self.tmp.name, []), {})
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(PD.progress_fields(empty, [("11110", "2026-09-24")]), {})

    def test_output_is_json_serialisable(self):
        got = PD.progress_fields(self.tmp.name, [("11110", "2026-09-24")])
        json.dumps(got[("11110", "2026-09-24")], allow_nan=False)


class TestTableBuilder(unittest.TestCase):
    """点数表は、データ取得と同じ関数で、各開示の日に見えていた値から作る。"""

    def test_events_use_the_fetcher_and_the_first_disclosure(self):
        f = sample_fins()
        # A の 1Q の訂正（後日）。点数表には最初の開示だけを数える
        f = pd.concat([f, pd.DataFrame([fins_row("11110", "2026-04-01", "1Q", "2026-10-01",
                                                 300, 1000)])], ignore_index=True)
        for c in ("DiscDate", "CurFYSt"):
            f[c] = pd.to_datetime(f[c]).dt.strftime("%Y-%m-%d")
        d = PP.events(f)
        a = d[(d["Code"] == "11110") & (d["CurFYSt"] == "2026-04-01") & (d["q"] == 1)]
        self.assertEqual(len(a), 1)
        self.assertEqual(a.iloc[0]["DiscDate"], "2026-08-05")
        self.assertAlmostEqual(a.iloc[0]["ratio"], 46.9 / 38.5)
        self.assertAlmostEqual(a.iloc[0]["ratio_linear"], 46.9 / 25.0)
        self.assertEqual(set(d["basis"]), {"seasonal", "linear", "floor"})
        # 本決算の行は数えない（通期予想に対する進捗が無い）
        self.assertTrue(set(d["q"]) <= {1, 2, 3})

    def test_floor_can_be_switched_off_for_the_comparison(self):
        f = sample_fins()
        for c in ("DiscDate", "CurFYSt"):
            f[c] = pd.to_datetime(f[c]).dt.strftime("%Y-%m-%d")
        raw = PP.events(f, floor=0.0)
        c = raw[raw["Code"] == "33330"].iloc[-1]
        self.assertEqual(c["basis"], "seasonal")
        self.assertAlmostEqual(c["bench"], 5.0)
        # 比べ終わったら元の下限に戻っている
        self.assertEqual(JF.PROGRESS_FLOOR, 0.5)


def read_js() -> str:
    with open(SCORING_JS, encoding="utf-8") as fh:
        return fh.read()


def js_array(name: str) -> list:
    src = read_js()
    m = re.search(rf"export const {name} = \[([^\]]*)\]", src)
    assert m, f"{name} が scoring.js に無い"
    return [float(x) for x in m.group(1).split(",") if x.strip()]


class TestJavaScriptAgreesWithPython(unittest.TestCase):
    """点数の計算（JS）と、基準・点数表を作る側（Python）の前提が一致していること。"""

    def test_percentile_breakpoints(self):
        self.assertEqual(js_array("PROGRESS_PCTS"), PP.PCTS)

    def test_every_basis_the_fetcher_emits_is_known_to_the_screen(self):
        src = read_js()
        m = re.search(r"const SEASONAL_BASES = new Set\(\[([^\]]*)\]\)", src)
        self.assertIsNotNone(m)
        seasonal_js = set(re.findall(r"'(\w+)'", m.group(1)))
        self.assertEqual(seasonal_js, set(PP.SEASONAL))
        labels = re.search(r"export const PROGRESS_BASIS_JA = \{([^}]*)\}", src).group(1)
        for basis in ("seasonal", "floor", "linear"):
            self.assertRegex(labels, rf"\b{basis}:", basis)

    def test_floor_matches_the_documented_value(self):
        """画面の説明（Q×12.5%）と、データ取得の下限が同じ。"""
        self.assertEqual(JF.PROGRESS_FLOOR * 25.0, 12.5)
        self.assertIn("Q×12.5%", read_js())


if __name__ == "__main__":
    unittest.main()
