#!/usr/bin/env python3
"""
スプレッドシート追記（research/export_sheets.py）の単体テスト。

守りたいのは1つ。**利用者が手で書いた列を壊さないこと。**
この台帳は「モデルの推奨」と「実際にいくらで入って出たか」を同じ行に
並べるためのもので、建値・手仕舞い・メモは人間が書く。列が増えたときに
既存セルがずれると、その記入が別の列に移って気づけない。

もとは tests/test_model.py に相乗りしていたが、モデル別の列を足した際に
テストが増えたのでこちらに分けた（test_model.py は学習・評価の担当）。

    python3 tests/test_sheets.py
"""
from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "research"))
import export_sheets as ES  # noqa: E402


class FakeWorksheet:
    """gspread のワークシートのうち、export_sheets が使う分だけ真似る。"""

    def __init__(self, values):
        self.values = [list(r) for r in values]
        self.appended = []
        self.batches = []
        self.updates = []

    def get_all_values(self):
        return [list(r) for r in self.values]

    def update(self, values, range_name=None, **kw):
        self.updates.append((range_name, values))
        # 見出し行の右端追加だけ再現する
        if range_name and range_name.endswith("1") and values:
            col = 0
            n = 0
            for ch in range_name.split(":")[0]:
                if ch.isalpha():
                    n = n * 26 + (ord(ch) - 64)
            col = n - 1
            row = self.values[0]
            while len(row) < col:
                row.append("")
            for i, v in enumerate(values[0]):
                if len(row) <= col + i:
                    row.append(v)
                else:
                    row[col + i] = v

    def append_rows(self, rows, **kw):
        self.appended.extend([list(r) for r in rows])

    def batch_update(self, updates, **kw):
        self.batches.extend(updates)


def a1_to_rc(ref: str):
    """"AB12" -> (12, 28)。テスト側で書き込み先を確かめるため。"""
    col, row = 0, ""
    for ch in ref:
        if ch.isalpha():
            col = col * 26 + (ord(ch.upper()) - 64)
        else:
            row += ch
    return int(row), col


def row_dict(header, line):
    return {h: (line[i] if i < len(line) else "") for i, h in enumerate(header)}


PRED = {
    "asOf": "2026-09-10",
    "dates": ["2026-09-09", "2026-09-10"],
    "models": [{"algo": a, "short": s} for a, s in
               (("lgbm", "LGB"), ("xgb", "XGB"), ("cat", "CAT"),
                ("logit", "LR"), ("mlp", "NN"))],
    "candidates": [
        {"date": "2026-09-10", "code": "1234", "jqCode": "12340",
         "name": "テスト銘柄", "sector": "情報・通信業",
         "rankInDay": 1, "nInDay": 2, "score": 0.42, "band": 9,
         "close": 1000.0, "agree90": 3, "nModels": 5,
         "byModel": {"lgbm": {"score": .42, "pctHistorical": 95.1},
                     "xgb": {"score": .33, "pctHistorical": 91.0},
                     "cat": {"score": .51, "pctHistorical": 93.2},
                     "logit": {"score": .20, "pctHistorical": 40.5},
                     "mlp": {"score": .28, "pctHistorical": 55.0}}},
        {"date": "2026-09-10", "code": "5678", "jqCode": "56780",
         "name": "別の銘柄", "sector": "陸運業",
         "rankInDay": 2, "nInDay": 2, "score": 0.11, "band": 3,
         "close": 500.0, "agree90": 0, "nModels": 5,
         "byModel": {"lgbm": {"score": .11, "pctHistorical": 20.0},
                     "xgb": {"score": .09, "pctHistorical": 18.0},
                     "cat": {"score": .12, "pctHistorical": 22.0},
                     "logit": {"score": .05, "pctHistorical": 8.0},
                     "mlp": {"score": .07, "pctHistorical": 12.0}}},
    ],
}


class TestRows(unittest.TestCase):
    def test_model_columns_are_percentiles(self):
        rows = ES.rows_from_predictions(PRED)
        r = rows[0]
        self.assertEqual(r["LGB%"], 95.1)
        self.assertEqual(r["LR%"], 40.5)
        self.assertEqual(r[ES.AGREE_COL], "3/5")

    def test_raw_scores_are_not_written(self):
        """生スコアは入れない。学習器ごとにスケールが違い、台帳で比べられない。"""
        rows = ES.rows_from_predictions(PRED)
        for r in rows:
            for k in r:
                self.assertNotIn("score", str(k).lower())

    def test_old_payload_without_bymodel(self):
        """5モデルを学習する前の予測ファイルでも落ちない。"""
        pred = {**PRED, "models": [],
                "candidates": [{k: v for k, v in c.items() if k != "byModel"}
                               for c in PRED["candidates"]]}
        for c in pred["candidates"]:
            c.pop("agree90", None)
            c.pop("nModels", None)
        rows = ES.rows_from_predictions(pred)
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0][ES.AGREE_COL])
        self.assertNotIn("LGB%", rows[0])


class TestSync(unittest.TestCase):
    #: モデル別の列がまだ無い、運用中のシート。利用者の記入が入っている
    OLD_HEADER = ES.OWNED_COLS + ES.TRACK_COLS + ES.USER_COLS

    def _old_sheet(self):
        header = list(self.OLD_HEADER)
        line = [""] * len(header)
        p = {h: i for i, h in enumerate(header)}
        line[p["予測日"]] = "2026-09-10"
        line[p["コード"]] = "1234"
        line[p["予測時株価"]] = "1000"
        line[p["建値"]] = "1005"          # 利用者の記入
        line[p["株数"]] = "100"
        line[p["メモ"]] = "寄りで指値"
        return FakeWorksheet([header, line])

    def test_missing_columns_are_appended_to_the_right(self):
        ws = self._old_sheet()
        ES.sync(ws, ES.rows_from_predictions(PRED), {}, None)
        header = ws.values[0]
        # 記入欄は元の位置から動いていない
        self.assertEqual(header.index("メモ"),
                         self.OLD_HEADER.index("メモ"))
        # 新しい列は右端に付いている
        for c in ES.MODEL_COLS + [ES.AGREE_COL]:
            self.assertIn(c, header)
            self.assertGreater(header.index(c), header.index("メモ"))

    def test_user_entries_are_never_written(self):
        ws = self._old_sheet()
        ES.sync(ws, ES.rows_from_predictions(PRED), {}, None)
        header = ws.values[0]
        cols = {header.index(c) + 1 for c in ES.USER_COLS if c in header}
        for u in ws.batches:
            _, col = a1_to_rc(u["range"].split("!")[-1])
            self.assertNotIn(col, cols,
                             f"記入欄に書き込もうとした: {u['range']}")
        for line in ws.appended:
            for c in ES.USER_COLS:
                self.assertEqual(line[header.index(c)], "",
                                 f"追記行が記入欄を埋めた: {c}")

    def test_existing_row_gets_model_values_backfilled(self):
        """列が増えた直後の既存行に、空のモデル別セルだけ埋まる。"""
        ws = self._old_sheet()
        ES.sync(ws, ES.rows_from_predictions(PRED), {}, None)
        header = ws.values[0]
        wrote = {}
        for u in ws.batches:
            row, col = a1_to_rc(u["range"].split("!")[-1])
            if row == 2:
                wrote[header[col - 1]] = u["values"][0][0]
        self.assertEqual(wrote.get("LGB%"), 95.1)
        self.assertEqual(wrote.get(ES.AGREE_COL), "3/5")
        # 記入済みのセルは候補に入らない
        self.assertNotIn("建値", wrote)

    def test_filled_model_cell_is_not_overwritten(self):
        """既に値があるモデル別セルは触らない（手で直している可能性がある）。"""
        ws = self._old_sheet()
        ws.values[0] = self.OLD_HEADER + ES.MODEL_COLS + [ES.AGREE_COL]
        p = {h: i for i, h in enumerate(ws.values[0])}
        line = ws.values[1] + [""] * (len(ws.values[0]) - len(ws.values[1]))
        line[p["LGB%"]] = "99.9"          # 手で書いた値
        ws.values[1] = line
        ES.sync(ws, ES.rows_from_predictions(PRED), {}, None)
        header = ws.values[0]
        for u in ws.batches:
            row, col = a1_to_rc(u["range"].split("!")[-1])
            if row == 2 and header[col - 1] == "LGB%":
                self.fail("手書きのモデル別セルを上書きした")

    def test_new_row_carries_model_values_in_the_right_cells(self):
        ws = self._old_sheet()
        ES.sync(ws, ES.rows_from_predictions(PRED), {}, None)
        header = ws.values[0]
        self.assertEqual(len(ws.appended), 1)   # 1234 は既存、5678 が新規
        got = row_dict(header, ws.appended[0])
        self.assertEqual(got["コード"], "5678")
        self.assertEqual(got["LGB%"], 20.0)
        self.assertEqual(got["NN%"], 12.0)
        self.assertEqual(got[ES.AGREE_COL], "0/5")

    def test_user_reordered_columns_still_work(self):
        """利用者が列をドラッグして動かしても、名前で探すので正しく入る。"""
        header = (["メモ", "建値"] + ES.MODEL_COLS + [ES.AGREE_COL]
                  + ES.OWNED_COLS + ES.TRACK_COLS
                  + [c for c in ES.USER_COLS if c not in ("メモ", "建値")])
        ws = FakeWorksheet([header])
        ES.sync(ws, ES.rows_from_predictions(PRED), {}, None)
        self.assertEqual(len(ws.appended), 2)
        got = row_dict(header, ws.appended[0])
        self.assertEqual(got["コード"], "1234")
        self.assertEqual(got["LGB%"], 95.1)
        self.assertEqual(got["メモ"], "")

    def test_header_is_not_touched_when_nothing_is_missing(self):
        header = (ES.OWNED_COLS + ES.MODEL_COLS + [ES.AGREE_COL]
                  + ES.TRACK_COLS + ES.USER_COLS)
        ws = FakeWorksheet([header])
        ES.sync(ws, ES.rows_from_predictions(PRED), {}, None)
        self.assertEqual(ws.updates, [], "見出しを不要に書き換えた")


class TestSheetExport(unittest.TestCase):
    """
    スプレッドシートは利用者が手で書き込む台帳でもある。
    こちらの書き込みで利用者の記入が消えることが最大の事故なので、
    「触ってよい列しか触らない」を固定する。
    """

    def setUp(self):
        import export_sheets as E
        self.E = E
        self.header = E.OWNED_COLS + E.TRACK_COLS + E.USER_COLS

    @staticmethod
    def _row(date="2026-09-10", code="1234", price=1000.0):
        return {"予測日": date, "コード": code, "銘柄名": "テスト", "業種": "情報･通信業",
                "順位": 1, "候補数": 5, "スコア": 0.38, "帯": 6,
                "較正確率%": 23.2, "帯の正例率%": 22.8, "帯の実収益%": 0.59,
                "帯の勝率%": 52.3, "必要上昇率%": 33.9, "予測時株価": price,
                "時価総額(億)": 70.9, "売買代金(億/日)": 0.33, "日次ボラ%": 3.64,
                "20日リターン%": 27.6, "地合い寄与": -0.18, "銘柄固有寄与": 0.28,
                "PER": 138.1, "PBR": 7.21, "_jqCode": f"{code}0"}

    def test_a1_handles_columns_past_z(self):
        """列は31個あるので AA 以降に届く。桁上げを間違えると別の列を上書きする。"""
        E = self.E
        self.assertEqual(E.a1(1, 1), "A1")
        self.assertEqual(E.a1(26, 3), "Z3")
        self.assertEqual(E.a1(27, 3), "AA3")
        self.assertEqual(E.a1(28, 10), "AB10")
        self.assertEqual(E.a1(52, 2), "AZ2")

    def test_appends_only_new_keys(self):
        E = self.E
        existing = [""] * len(self.header)
        existing[self.header.index("予測日")] = "2026-09-10"
        existing[self.header.index("コード")] = "1234"
        ws = FakeWorksheet([self.header, existing])
        rows = [self._row(code="1234"), self._row(code="5678")]
        res = E.sync(ws, rows, pd.Series(dtype=float), None)
        self.assertEqual(res["appended"], 1)
        self.assertEqual(len(ws.appended), 1)
        got = ws.appended[0][self.header.index("コード")]
        self.assertEqual(got, "5678")

    def test_never_writes_user_columns(self):
        """建値・メモなどは見出しを作るだけ。値を書いたら利用者の記入を潰す。"""
        E = self.E
        ws = FakeWorksheet([self.header])
        E.sync(ws, [self._row()], pd.Series({"12340": 1100.0}),
               pd.Timestamp("2026-09-11"))
        line = ws.appended[0]
        for name in E.USER_COLS:
            self.assertEqual(line[self.header.index(name)], "",
                             f"{name} に書き込んでいる")

    def test_tracking_update_finds_columns_by_name(self):
        """
        利用者が途中に列を挿しても、見出し名で探していれば正しい列に書く。
        位置で決め打ちしていると、挿された瞬間に別の列を壊す。
        """
        E = self.E
        header = list(self.header)
        header.insert(3, "自分メモ")          # 利用者が挿した列
        line = [""] * len(header)
        line[header.index("予測日")] = "2026-09-10"
        line[header.index("コード")] = "1234"
        line[header.index("予測時株価")] = "1000"
        line[header.index("自分メモ")] = "消えてはいけない"
        ws = FakeWorksheet([header, line])
        E.sync(ws, [], pd.Series({"12340": 1100.0}), pd.Timestamp("2026-09-11"))
        # 騰落率の列に、正しい A1 で書かれていること
        want_col = E.a1(header.index("騰落率%") + 1, 2)
        ranges = {u["range"]: u["values"][0][0] for u in ws.batches}
        self.assertIn(want_col, ranges)
        self.assertAlmostEqual(ranges[want_col], 10.0)
        # 利用者の列には1つも書いていない
        memo_col = E.a1(header.index("自分メモ") + 1, 2)
        self.assertNotIn(memo_col, ranges)

    def test_track_values_leaves_gaps_empty(self):
        """株価が取れない行は空。0 で埋めると『実測でゼロ』と混ざる。"""
        E = self.E
        self.assertEqual(E.track_values("99999", 100.0, "2026-09-10",
                                        pd.Series(dtype=float), None), {})
        self.assertEqual(E.track_values("12340", None, "2026-09-10",
                                        pd.Series({"12340": 100.0}), None), {})

    def test_rows_keep_every_candidate(self):
        """上位だけに絞らない。選ばなかった側も後から検証したいので。"""
        E = self.E
        pred = {"candidates": [
            {"date": "2026-09-10", "code": "1234", "jqCode": "12340",
             "rankInDay": 1, "nInDay": 3, "score": 0.4, "band": 6},
            {"date": "2026-09-10", "code": "5678", "jqCode": "56780",
             "rankInDay": 3, "nInDay": 3, "score": 0.1, "band": 0},
        ]}
        self.assertEqual(len(E.rows_from_predictions(pred)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
