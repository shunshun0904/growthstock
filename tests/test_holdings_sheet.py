#!/usr/bin/env python3
"""
保有の推移のタブ（research/holdings_sheet.py）の単体テスト。通信しない。

固定すること
  - 赤の見分け方（赤・濃い赤 1・明るい赤 1 は赤、薄い赤や白は違う）。予測日とコードの両方が赤い行だけ
  - 予測ログの列は見出しの名前で探す。建値・株数・手仕舞い日・手仕舞い値を読む（無ければ既定）
  - 買いは予測日の次の営業日の寄り付き。分割をまたいでも損益が合う
  - 手仕舞い日のあとは確定した損益のまま
  - 日付ごとの表: 買う前は空、合計と元手に対する%、保有数
  - グラフは「損益(円)」の列だけ（軸は1本）。色は合計が1番目、保有は買った順
  - 公開ログに銘柄名・株価・損益の値を出さない

  python3 tests/test_holdings_sheet.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import holdings_sheet as H  # noqa: E402

T = pd.Timestamp
RED = {"red": 1}
WHITE = {"red": 1, "green": 1, "blue": 1}


def grid(rows, start_row=0):
    """rows: 各行の各セルの背景色（None は書式なし）。"""
    return {"sheets": [{"data": [{"startRow": start_row, "rowData": [
        {"values": [({"effectiveFormat": {"backgroundColor": c}} if c is not None else {})
                    for c in r]} for r in rows]}]}]}


def bars(code, days, opens, closes, adj=1.0, adj_from=None):
    """adj: 調整の比（AdjX = X * adj）。adj_from 以降は 1（その日に分割があった形）。"""
    df = pd.DataFrame({"Date": pd.to_datetime(days), "Code": code,
                       "O": opens, "C": closes})
    k = np.where(df["Date"] < T(adj_from), adj, 1.0) if adj_from else adj
    df["AdjO"] = df["O"] * k
    df["AdjC"] = df["C"] * k
    return df


class Red(unittest.TestCase):
    def test_colors(self):
        self.assertTrue(H.is_red({"red": 1}))                              # 赤
        self.assertTrue(H.is_red({"red": 0.8}))                            # 濃い赤 1 (#cc0000)
        self.assertTrue(H.is_red({"red": 0.878, "green": 0.4, "blue": 0.4}))  # 明るい赤 1
        self.assertFalse(H.is_red({"red": 0.918, "green": 0.6, "blue": 0.6}))  # 明るい赤 2
        self.assertFalse(H.is_red(WHITE))
        self.assertFalse(H.is_red({"red": 0.6}))                           # 濃い赤 2
        self.assertFalse(H.is_red(None))
        self.assertFalse(H.is_red({}))

    def test_only_rows_with_both_key_cells_red(self):
        g = grid([[WHITE, WHITE, WHITE],      # 1行目（見出し）
                  [RED, RED, RED],            # 2行目: 行ごと赤
                  [RED, None, None],          # 3行目: 予測日だけ赤 → 数えない
                  [None, None, None],
                  [RED, RED]])                # 5行目
        self.assertEqual(H.red_rows(g, [0, 1]), [2, 5])

    def test_start_row_and_key_columns_by_position(self):
        g = grid([[None, RED, RED]], start_row=9)       # 10行目
        self.assertEqual(H.red_rows(g, [1, 2]), [10])
        self.assertEqual(H.red_rows(g, [0, 1]), [])


class Positions(unittest.TestCase):
    HEADER = ["コード", "予測日", "銘柄名", "メモ", "建値", "株数", "手仕舞い日", "手仕舞い値"]

    def test_reads_by_header_name_and_defaults(self):
        values = [self.HEADER,
                  ["6912", "2026-09-24", "A社", "", "", "", "", ""],
                  ["173A", "2026/10/02", "B社", "x", "1,234", "300", "2026-10-06", "1,300"],
                  ["7713", "2026-10-02", "C社", "", "", "", "", ""]]
        ps = H.positions_from_values(values, [2, 3])
        self.assertEqual([p.code for p in ps], ["6912", "173A"])
        a, b = ps
        self.assertEqual(a.jq_code, "69120")
        self.assertEqual(a.shares, H.DEFAULT_SHARES)
        self.assertIsNone(a.entry_price)
        self.assertIsNone(a.exit_date)
        self.assertEqual(b.jq_code, "173A0")
        self.assertEqual(b.pick_date, "2026-10-02")
        self.assertEqual(b.entry_price, 1234.0)
        self.assertEqual(b.shares, 300.0)
        self.assertEqual(b.exit_date, T("2026-10-06"))
        self.assertEqual(b.exit_price, 1300.0)

    def test_header_row_and_unreadable_rows_are_skipped(self):
        values = [self.HEADER, ["", "メモの行", "", "", "", "", "", ""]]
        self.assertEqual(H.positions_from_values(values, [1, 2, 99]), [])

    def test_sorted_by_pick_date(self):
        values = [self.HEADER,
                  ["7713", "2026-10-02", "C", "", "", "", "", ""],
                  ["6912", "2026-09-24", "A", "", "", "", "", ""]]
        ps = H.positions_from_values(values, [2, 3])
        self.assertEqual([p.code for p in ps], ["6912", "7713"])


def pos(code="6912", pick="2026-09-24", entry=None, shares=100, exit_date=None, exit_price=None):
    return H.Position(row=2, pick_date=pick, code=code, name="A社", entry_price=entry,
                      shares=shares, exit_date=T(exit_date) if exit_date else None,
                      exit_price=exit_price)


class Path(unittest.TestCase):
    DAYS = ["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29"]

    def test_buys_at_next_open(self):
        b = bars("69120", self.DAYS, [990, 1000, 1010, 1020], [995, 1010, 990, 1100])
        st, path = H.position_path(pos(), b)
        self.assertEqual(st, "保有中")
        self.assertEqual(list(path["Date"]), [T(d) for d in self.DAYS[1:]])
        self.assertEqual(list(path["pnl_yen"]), [1000.0, -1000.0, 10000.0])
        self.assertAlmostEqual(path["pnl_pct"].iloc[-1], 10.0)
        self.assertEqual(path["entry_day"].iloc[0], T("2026-09-25"))

    def test_split_after_buying(self):
        """買った後に 1:2 の分割。生の終値は半分になるが、損益は株の単位をそろえて数える。"""
        b = bars("69120", self.DAYS, [990, 1000, 505, 560], [995, 1010, 500, 560],
                 adj=0.5, adj_from="2026-09-28")
        st, path = H.position_path(pos(), b)
        # 買った日の単位に戻した終値: 1010, 1000, 1120
        self.assertEqual(list(path["close"].round(6)), [1010.0, 1000.0, 1120.0])
        self.assertAlmostEqual(path["pnl_pct"].iloc[-1], 12.0)

    def test_manual_entry_and_shares(self):
        b = bars("69120", self.DAYS, [990, 1000, 1010, 1020], [995, 1010, 990, 1100])
        st, path = H.position_path(pos(entry=1050, shares=300), b)
        self.assertEqual(path["pnl_yen"].iloc[-1], (1100 - 1050) * 300)

    def test_exit_freezes_the_result(self):
        b = bars("69120", self.DAYS, [990, 1000, 1010, 1020], [995, 1010, 990, 1100])
        st, path = H.position_path(pos(exit_date="2026-09-28", exit_price=1050), b)
        self.assertEqual(st, "手仕舞い")
        self.assertEqual(list(path["pnl_yen"]), [1000.0, 5000.0, 5000.0])
        self.assertEqual(list(path["held"]), [True, True, False])

    def test_exit_without_price_uses_that_days_close(self):
        b = bars("69120", self.DAYS, [990, 1000, 1010, 1020], [995, 1010, 990, 1100])
        st, path = H.position_path(pos(exit_date="2026-09-28"), b)
        self.assertEqual(list(path["pnl_yen"]), [1000.0, -1000.0, -1000.0])

    def test_waiting_and_missing(self):
        b = bars("69120", self.DAYS[:1], [990], [995])
        self.assertEqual(H.position_path(pos(), b), ("買い待ち", None))
        self.assertEqual(H.position_path(pos(code="9999"), b)[0], "株価なし")
        b2 = bars("69120", self.DAYS, [990, np.nan, 1010, 1020], [995, np.nan, 990, 1100])
        self.assertEqual(H.position_path(pos(), b2)[0], "寄り付かず")

    def test_no_trade_day_keeps_the_previous_close(self):
        b = bars("69120", self.DAYS, [990, 1000, np.nan, 1020], [995, 1010, np.nan, 1100])
        st, path = H.position_path(pos(), b)
        self.assertEqual(list(path["close"]), [1010.0, 1010.0, 1100.0])


class Daily(unittest.TestCase):
    def setUp(self):
        days = ["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29"]
        b = pd.concat([
            bars("69120", days, [990, 1000, 1010, 1020], [995, 1010, 990, 1100]),
            bars("77130", days, [500, 505, 510, 515], [500, 506, 520, 530])])
        self.a = pos()
        self.b = H.Position(row=3, pick_date="2026-09-25", code="7713", name="C社",
                            entry_price=None, shares=100, exit_date=None, exit_price=None)
        self.paths = {}
        self.states = {}
        for p in (self.a, self.b):
            st, path = H.position_path(p, b)
            self.states[p.row] = st
            self.paths[p.row] = path
        self.daily = H.daily_table([self.a, self.b], self.paths)

    def test_columns_and_blank_before_buying(self):
        d = self.daily
        self.assertEqual(list(d.columns[:4]), ["日付", "合計損益(円)", "合計損益%", "保有数"])
        self.assertIn("A社(6912) 損益(円)", d.columns)
        self.assertIn("C社(7713) 損益%", d.columns)
        self.assertTrue(pd.isna(d.loc[0, "C社(7713) 損益(円)"]))     # 9/25 はまだ買っていない
        self.assertEqual(list(d["保有数"]), [1, 2, 2])

    def test_total_and_percent_of_cost(self):
        last = self.daily.iloc[-1]
        # A: (1100-1000)*100 = 10,000 / B: 9/28 の寄り 510 で買い (530-510)*100 = 2,000
        self.assertEqual(last["合計損益(円)"], 12000)
        self.assertAlmostEqual(last["合計損益%"], round(12000 / (1000 * 100 + 510 * 100) * 100, 2))

    def test_layout_and_chart(self):
        summary = H.summary_rows([self.a, self.b], self.states, self.paths)
        self.assertEqual(summary[-1][0], "合計")
        self.assertEqual(summary[-1][8], 12000)
        lay = H.build_layout("2026-09-29", summary, self.daily, n_red=2)
        self.assertEqual(lay.values[3], H.SUMMARY_HEADER)
        self.assertEqual(lay.values[lay.daily_header_row][0], "日付")
        self.assertEqual(lay.values[lay.daily_header_row + 1][0], "2026-09-25")
        self.assertEqual(lay.yen_series_cols, [1, 4, 5])
        req = H.chart_request(123, lay)["addChart"]["chart"]
        spec = req["spec"]["basicChart"]
        self.assertEqual(spec["chartType"], "LINE")
        self.assertEqual(len(spec["series"]), 3)                  # 合計 + 2銘柄
        self.assertEqual({s["targetAxis"] for s in spec["series"]}, {"LEFT_AXIS"})  # 軸は1本
        first = spec["series"][0]["colorStyle"]["rgbColor"]
        self.assertAlmostEqual(first["red"], 0x2a / 255)
        rng = spec["domains"][0]["domain"]["sourceRange"]["sources"][0]
        self.assertEqual(rng["startRowIndex"], lay.daily_header_row)
        self.assertEqual(rng["endRowIndex"], lay.daily_header_row + 1 + len(self.daily))
        fmts = H.format_requests(123, lay)
        self.assertTrue(any("numberFormat" in r["repeatCell"]["cell"]["userEnteredFormat"]
                            for r in fmts))

    def test_no_red_rows(self):
        lay = H.build_layout("2026-10-08", [], pd.DataFrame(), n_red=0)
        self.assertIn("赤く塗った行がありません", lay.values[-1][0])
        self.assertIsNone(H.chart_request(1, lay))

    def test_report_has_no_names_or_values(self):
        lines = H.report(2, self.states, len(self.daily), "2026-09-29")
        text = "\n".join(lines)
        self.assertIn("赤い行 2", text)
        for word in ("A社", "C社", "6912", "7713", "12000", "12,000", "1100", "530"):
            self.assertNotIn(word, text)


class FakeWorksheet:
    def __init__(self, title, values=None, sid=1, rows=100, cols=30):
        self.title, self.id = title, sid
        self.values = values or []
        self.row_count, self.col_count = rows, cols
        self.updates = []

    def get_all_values(self):
        return self.values

    def resize(self, rows=None, cols=None):
        self.row_count, self.col_count = rows or self.row_count, cols or self.col_count

    def update(self, values, range_name, value_input_option=None):
        self.updates.append((values, range_name, value_input_option))


class FakeBook:
    """予測ログと（あれば）保有の推移のタブを持つ偽のスプレッドシート。"""

    def __init__(self, log_values, red, has_tab=False, old_charts=(7,)):
        self.log = FakeWorksheet(H.ES.SHEET_TITLE, log_values, sid=0)
        self.tab = FakeWorksheet(H.TAB, sid=5) if has_tab else None
        self.red = red
        self.old_charts = list(old_charts)
        self.requests = []

    def worksheet(self, title):
        if title == self.log.title:
            return self.log
        if title == H.TAB and self.tab is not None:
            return self.tab
        raise KeyError(title)

    def add_worksheet(self, title, rows, cols):
        self.tab = FakeWorksheet(title, sid=5, rows=rows, cols=cols)
        return self.tab

    def fetch_sheet_metadata(self, params=None):
        if params.get("includeGridData") == "true":
            n = len(self.log.values)
            return grid([[RED, RED] if (i + 1) in self.red else [None, None] for i in range(n)])
        return {"sheets": [{"properties": {"sheetId": 5},
                            "charts": [{"chartId": c} for c in self.old_charts]}]}

    def batch_update(self, body):
        self.requests.extend(body["requests"])


class EndToEnd(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()
        days = ["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29"]
        pd.concat([bars("69120", days, [990, 1000, 1010, 1020], [995, 1010, 990, 1100]),
                   bars("77130", days, [500, 505, 510, 515], [500, 506, 520, 530])]
                  ).to_parquet(os.path.join(self.dir, "bars_2026.parquet"), index=False)
        self.values = [["予測日", "コード", "銘柄名", "建値", "株数", "手仕舞い日", "手仕舞い値"],
                       ["2026-09-24", "6912", "A社", "", "", "", ""],
                       ["2026-09-24", "1111", "B社", "", "", "", ""],
                       ["2026-09-25", "7713", "C社", "", "", "", ""]]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_main(self, book, *extra):
        orig = H.ES.open_sheet
        H.ES.open_sheet = lambda sheet_id: book
        try:
            return H.main(["--sheet-id", "x", "--data-dir", self.dir, *extra])
        finally:
            H.ES.open_sheet = orig

    def test_writes_the_tab_and_replaces_the_chart(self):
        book = FakeBook(self.values, red={2, 4}, has_tab=True, old_charts=(7, 8))
        self.assertEqual(self.run_main(book), 0)
        kinds = [next(iter(r)) for r in book.requests]
        self.assertEqual(kinds[:3], ["deleteEmbeddedObject", "deleteEmbeddedObject", "updateCells"])
        self.assertEqual(kinds[-1], "addChart")
        values, rng, opt = book.tab.updates[-1]
        self.assertEqual((rng, opt), ("A1", "USER_ENTERED"))
        names = [r[0] for r in values[4:6]]
        self.assertEqual(names, ["A社", "C社"])                  # 赤い2行だけ（B社は赤くない）
        series = book.requests[-1]["addChart"]["chart"]["spec"]["basicChart"]["series"]
        self.assertEqual(len(series), 3)

    def test_creates_the_tab_when_missing_and_dry_run_writes_nothing(self):
        book = FakeBook(self.values, red={2}, has_tab=False, old_charts=())
        self.assertEqual(self.run_main(book, "--dry-run"), 0)
        self.assertIsNone(book.tab)
        self.assertEqual(book.requests, [])
        self.assertEqual(self.run_main(book), 0)
        self.assertIsNotNone(book.tab)
        self.assertEqual(next(iter(book.requests[0])), "updateCells")


class Workflow(unittest.TestCase):
    def test_runs_after_the_daily_prediction(self):
        path = os.path.join(ROOT, ".github", "workflows", "holdings.yml")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("workflows: ['Predict Breakouts']", text)
        self.assertIn("workflow_dispatch", text)
        self.assertNotIn("schedule:", text)
        self.assertNotIn("git add", text)                  # コミットしない
        self.assertNotIn("gh_release_upload", text)        # Release に書かない
        self.assertIn("research/holdings_sheet.py", text)
        self.assertIn("'bars_*'", text)                    # 日足だけ取り出す
        # export_sheets -> models -> tuning_multi が scikit-learn を読み込むので、予測と同じ一式を入れる
        self.assertIn("pip install -r research/requirements.txt", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
