#!/usr/bin/env python3
"""
8軸の「進捗期待」（2026-09-25 に運用者の選択 A で入れた定義）の、言語・経路をまたぐテスト。

  基準 = 前年同期の進捗（scripts/jquants_data_fetcher.py の progress_benchmark）
  点数 = 進捗率 ÷ 基準 の、過去の開示の中での順位（src/lib/scoring.js の PROGRESS_TABLE）

同じ定義が5か所に出てくる。ずれると、同じ銘柄の点数が経路によって変わる:
  ・データ取得（stocks.json）          scripts/jquants_data_fetcher.py
  ・予測タブから送った候補              research/predict_daily.progress_fields
  ・点数表の作成                        research/progress_percentiles.py
  ・点数の計算                          src/lib/scoring.js
  ・モデルの特徴量（実験46の候補）      research/build_dataset.seasonal_progress

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

import build_dataset as B  # noqa: E402
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


def statement(code, fy, period, disc, op, fop=None):
    """build_dataset.quarterize_panel と API の両方に渡せる決算の行（日付は文字列）。"""
    return {"Code": code, "DiscDate": disc, "DiscTime": "15:00", "CurPerType": period,
            "CurFYSt": fy, "CurPerEn": disc, "DocType": "FinancialStatements",
            "Sales": 100.0, "OP": float(op), "NP": 1.0, "EPS": 1.0,
            "Eq": 1000.0, "TA": 2000.0, "ROE": None, "FSales": None, "FNP": None, "FEPS": None,
            "FOP": None if fop is None else float(fop), "ShOutFY": 1_000_000, "TrShFY": 0}


def scenarios():
    rows = []
    # A: 前年同期の進捗 38.5%（9081 の形）
    rows += [statement("10010", "2025-04-01", "1Q", "2025-08-05", 385, 1000),
             statement("10010", "2025-04-01", "FY", "2026-05-10", 1000),
             statement("10010", "2026-04-01", "1Q", "2026-08-05", 469, 1000)]
    # B: 前年同期の進捗 5% -> 下限 12.5%
    rows += [statement("20020", "2025-04-01", "1Q", "2025-08-06", 50, 1000),
             statement("20020", "2025-04-01", "FY", "2026-05-11", 1000),
             statement("20020", "2026-04-01", "1Q", "2026-08-06", 200, 1000)]
    # C: 前年の通期が赤字 -> Q×25%
    rows += [statement("30030", "2025-04-01", "1Q", "2025-08-07", 50, 1000),
             statement("30030", "2025-04-01", "FY", "2026-05-12", -100),
             statement("30030", "2026-04-01", "1Q", "2026-08-07", 200, 1000)]
    # D: 決算期の変更（前年度が 2025-01-01 開始）-> Q×25%
    rows += [statement("40040", "2025-01-01", "1Q", "2025-05-08", 300, 1000),
             statement("40040", "2025-01-01", "FY", "2026-02-12", 1000),
             statement("40040", "2026-04-01", "1Q", "2026-08-08", 300, 1000)]
    # E: 前年の本決算を 1Q の後に訂正（1000 -> 800）。1Q の時点では訂正前で計算する。
    #    2Q の時点では訂正後が見えている
    rows += [statement("50050", "2025-04-01", "1Q", "2025-08-05", 385, 1000),
             statement("50050", "2025-04-01", "2Q", "2025-11-05", 450, 1000),
             statement("50050", "2025-04-01", "FY", "2026-05-10", 1000),
             statement("50050", "2026-04-01", "1Q", "2026-08-05", 469, 1000),
             statement("50050", "2025-04-01", "FY", "2026-10-01", 800),
             statement("50050", "2026-04-01", "2Q", "2026-11-05", 520, 1000)]
    return rows


class TestDatasetAgreesWithTheScreen(unittest.TestCase):
    """モデルの特徴量（build_dataset）が、画面のデータ取得と同じ基準・比率を出すこと。"""

    @classmethod
    def setUpClass(cls):
        cls.rows = scenarios()
        q = B.quarterize_panel(pd.DataFrame(cls.rows))
        cls.q = q.assign(D=pd.to_datetime(q["DiscDate"]).dt.strftime("%Y-%m-%d"))

    def row(self, code, day):
        x = self.q[(self.q["Code"] == code) & (self.q["D"] == day)]
        self.assertEqual(len(x), 1, (code, day))
        return x.iloc[0]

    def test_every_row_matches_the_fetcher(self):
        n = 0
        for r in self.q[self.q["progress_basis"].notna()].itertuples(index=False):
            m = JF.fundamentals_as_of([x for x in self.rows if x["Code"] == r.Code], r.D)
            self.assertEqual(m["progressBasis"], r.progress_basis, (r.Code, r.D))
            self.assertAlmostEqual(m["progressRate"] / m["progressBenchmark"],
                                   r.progress_ratio, places=12, msg=(r.Code, r.D))
            n += 1
        # 通期予想のある四半期の行: A・B・C・D は前年と今年の1Q（2行ずつ）、E は1Q・2Q が2年ぶん
        self.assertEqual(n, 12)

    def test_each_rule(self):
        self.assertEqual(self.row("10010", "2026-08-05")["progress_basis"], "seasonal")
        self.assertAlmostEqual(self.row("10010", "2026-08-05")["progress_ratio"], 46.9 / 38.5)
        self.assertEqual(self.row("20020", "2026-08-06")["progress_basis"], "floor")
        self.assertAlmostEqual(self.row("20020", "2026-08-06")["progress_ratio"], 20.0 / 12.5)
        self.assertEqual(self.row("30030", "2026-08-07")["progress_basis"], "linear")
        self.assertEqual(self.row("40040", "2026-08-08")["progress_basis"], "linear")
        self.assertAlmostEqual(self.row("40040", "2026-08-08")["progress_ratio"], 30.0 / 25.0)

    def test_a_later_correction_is_not_seen_earlier(self):
        """前年の本決算の訂正（2026-10-01）は、それより前の 1Q の行の基準に入らない。"""
        self.assertAlmostEqual(self.row("50050", "2026-08-05")["progress_ratio"], 46.9 / 38.5)
        # 2Q（2026-11-05）の時点では訂正後の通期 800 で割る
        self.assertAlmostEqual(self.row("50050", "2026-11-05")["progress_ratio"],
                               52.0 / (450 / 800 * 100))

    def test_full_year_rows_have_no_ratio(self):
        fy = self.q[self.q["quarter"] == 4]
        self.assertTrue(len(fy) > 0)
        self.assertTrue(fy["progress_ratio"].isna().all())
        self.assertTrue(fy["progress_basis"].isna().all())

    def test_floor_is_shared_with_the_fetcher(self):
        self.assertIs(B.PROGRESS_FLOOR, JF.PROGRESS_FLOOR)


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
