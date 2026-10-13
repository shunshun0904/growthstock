#!/usr/bin/env python3
"""
research/inst_features.py（機関投資家＝運用目的の保有者の直近の買いのフラグ。実験89 の候補）のテスト。

- 保有目的の区分が probe（運用者に見せた件数）と同じ正規表現・同じ順であること
- 買いの判定（増えた・新規5%超は買い、減った・同じ・今回の割合なしは買いでない）
- 訂正報告書を数えない・1報告書 = 1件・運用目的でない保有者の買いは数えない
- 窓の端（その日を含む過去 N 暦日）・提出日より前の行に効かない（先読みしない）・銘柄をまたがない
- 取り込みの始まりから窓がそろう前の行は NaN（0 で埋めない）
- parquet から読み戻した入れ子（ndarray）でも動き、ログに名前が出ない

  python3 tests/test_inst_features.py
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import inst_features as I  # noqa: E402
import probe_lvs_holders as P  # noqa: E402

#: 合成の名前（ログに出ないことを確かめる）
NAMES = ["合成アセット運用株式会社", "合成証券株式会社", "合成太郎", "合成発行体株式会社"]

INVEST_TEXT = "顧客の資産の運用（投資一任契約）"


def holder(purpose, now, last, name=NAMES[0]):
    return {"HldrName": name, "HldgPurp": purpose, "ShsRatio": now, "ShsRatioLast": last, "LargeHldrTypeCode": "2"}


def report(doc, code, sub, hldrs, doc_type="350"):
    return {"DocId": doc, "Code": code, "SubDate": sub, "DocTypeCode": doc_type, "IsrName": NAMES[3],
            "Hldrs": hldrs}


class TestSameAsProbe(unittest.TestCase):
    def test_purpose_classes_are_the_probe_ones(self):
        self.assertEqual([lb for lb, _ in I.PURPOSE_CLASSES], [lb for lb, _ in P.PURPOSE_CLASSES])
        self.assertEqual([p.pattern for _, p in I.PURPOSE_CLASSES], [p.pattern for _, p in P.PURPOSE_CLASSES])
        self.assertEqual(I.INVEST, P.PURPOSE_CLASSES[-1][0])
        for text in ("純投資", "証券業務に係る商品在庫として保有", "安定株主として保有", "重要提案行為等を行うこと",
                     INVEST_TEXT, "政策投資及び純投資", "", None, "特になし"):
            self.assertEqual(I.purpose_class(text), P.purpose_class(text), text)

    def test_precedence(self):
        self.assertEqual(I.purpose_class("純投資"), I.INVEST)
        self.assertEqual(I.purpose_class("信託財産の運用"), I.INVEST)
        # 先に当たった区分に入れる（運用の言葉があっても、貸借・経営が先）
        self.assertEqual(I.purpose_class("貸借取引に伴う担保・純投資"), "証券業務・貸借・担保")
        self.assertEqual(I.purpose_class("経営参加及び運用"), "経営・安定株主・政策投資")
        self.assertEqual(I.purpose_class("重要提案行為等及び純投資"), "重要提案")
        self.assertEqual(I.purpose_class("  "), "（空）")


class TestBuy(unittest.TestCase):
    def test_increase_and_new_are_buys(self):
        self.assertTrue(I.is_buy(holder(INVEST_TEXT, 0.07, 0.06)))
        self.assertTrue(I.is_buy(holder(INVEST_TEXT, 0.051, None)))            # 新たに5%超（前回なし）
        self.assertTrue(I.is_buy(holder(INVEST_TEXT, 0.051, 0.0)))             # 前回 0
        self.assertTrue(I.is_buy(holder(INVEST_TEXT, "0.07", "0.06")))         # 文字の数字
        self.assertTrue(I.is_buy(holder(INVEST_TEXT, 0.051, float("nan"))))

    def test_decrease_same_and_unknown_are_not_buys(self):
        self.assertFalse(I.is_buy(holder(INVEST_TEXT, 0.05, 0.06)))
        self.assertFalse(I.is_buy(holder(INVEST_TEXT, 0.06, 0.06)))
        self.assertFalse(I.is_buy(holder(INVEST_TEXT, None, 0.06)))             # 今回の割合なし
        self.assertFalse(I.is_buy(holder(INVEST_TEXT, "", None)))
        self.assertFalse(I.is_buy(holder(INVEST_TEXT, 0.0, None)))              # 前回なしでも今回 0 は買いでない
        self.assertFalse(I.is_buy({}))


class TestEvents(unittest.TestCase):
    def frame(self):
        rows = [
            report("D1", "11110", "2024-03-01", [holder(INVEST_TEXT, 0.06, None)]),                    # 数える
            report("D2", "11110", "2024-03-05", [holder(INVEST_TEXT, 0.05, 0.06)]),                    # 減った
            report("D3", "22220", "2024-03-05", [holder("証券業務に係る商品在庫", 0.07, 0.05, NAMES[1])]),  # 運用でない
            report("D4", "22220", "2024-04-01", [holder("安定株主", 0.05, None, NAMES[2]),
                                                 holder("純投資", 0.08, 0.07)]),                      # 共同保有者の2人目
            report("D5", "33330", "2024-04-02", [holder(INVEST_TEXT, 0.09, 0.05)], doc_type="360"),    # 訂正
            report("D6", "33330", "2024-04-03", [holder(INVEST_TEXT, 0.09, 0.05)], doc_type=360),      # 訂正（数値）
            report("D7", "44440", "2025-01-06", [holder(INVEST_TEXT, 0.06, None), holder("純投資", 0.07, 0.06)]),
            report("D7", "44440", "2025-01-06", [holder(INVEST_TEXT, 0.06, None)]),                    # 同じ書類の重複
            report("D8", "44440", "2025-01-07", []),                                                    # 一覧なし
        ]
        return pd.DataFrame(rows)

    def test_events(self):
        ev, st = I.buy_events(self.frame())
        self.assertEqual(st["reports"], 9)
        self.assertEqual(st["corrections"], 2)
        self.assertEqual(st["events"], 3)                                       # D1・D4・D7（1報告書 = 1件）
        self.assertEqual(sorted(zip(ev["Code"], ev["SubDate"].dt.strftime("%Y-%m-%d"))),
                         [("11110", "2024-03-01"), ("22220", "2024-04-01"), ("44440", "2025-01-06")])
        self.assertEqual(st["events_by_year"], {2024: 2, 2025: 1})

    def test_empty_and_missing_columns(self):
        ev, st = I.buy_events(pd.DataFrame())
        self.assertEqual((len(ev), st["events"]), (0, 0))
        ev, st = I.buy_events(pd.DataFrame({"Code": ["11110"], "SubDate": ["2024-01-04"]}))   # Hldrs が無い
        self.assertEqual(len(ev), 0)


class TestFlags(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame({"Code": ["11110", "22220"],
                                    "SubDate": pd.to_datetime(["2024-03-01", "2024-03-20"])})
        self.start = pd.Timestamp("2021-07-01")

    def flag(self, code, date, start=None):
        s = pd.DataFrame({"Code": [code], "Date": [pd.Timestamp(date)]})
        return I.flags(s, self.events, start or self.start).iloc[0].to_dict()

    def test_window_edges(self):
        # 提出日の当日から数えて、20日窓は 19日後まで、60日窓は 59日後まで
        self.assertEqual(self.flag("11110", "2024-03-01"), {"inst_buy_20": 1.0, "inst_buy_60": 1.0})
        self.assertEqual(self.flag("11110", "2024-03-20"), {"inst_buy_20": 1.0, "inst_buy_60": 1.0})   # +19日
        self.assertEqual(self.flag("11110", "2024-03-21"), {"inst_buy_20": 0.0, "inst_buy_60": 1.0})   # +20日
        self.assertEqual(self.flag("11110", "2024-04-29"), {"inst_buy_20": 0.0, "inst_buy_60": 1.0})   # +59日
        self.assertEqual(self.flag("11110", "2024-04-30"), {"inst_buy_20": 0.0, "inst_buy_60": 0.0})   # +60日

    def test_no_lookahead_and_no_cross_code(self):
        self.assertEqual(self.flag("11110", "2024-02-29"), {"inst_buy_20": 0.0, "inst_buy_60": 0.0})   # 前日
        self.assertEqual(self.flag("33330", "2024-03-01"), {"inst_buy_20": 0.0, "inst_buy_60": 0.0})   # 別の銘柄
        self.assertEqual(self.flag("22220", "2024-03-19"), {"inst_buy_20": 0.0, "inst_buy_60": 0.0})

    def test_nan_until_the_window_is_covered(self):
        start = pd.Timestamp("2024-02-15")
        # 20日窓は start+19日（3/5）から、60日窓は start+59日（4/14）から分かる
        f = self.flag("11110", "2024-03-04", start)
        self.assertTrue(np.isnan(f["inst_buy_20"]) and np.isnan(f["inst_buy_60"]))
        f = self.flag("11110", "2024-03-05", start)
        self.assertEqual(f["inst_buy_20"], 1.0)
        self.assertTrue(np.isnan(f["inst_buy_60"]))
        self.assertEqual(self.flag("11110", "2024-04-14", start), {"inst_buy_20": 0.0, "inst_buy_60": 1.0})
        s = pd.DataFrame({"Code": ["11110"], "Date": [pd.Timestamp("2024-03-01")]})
        self.assertTrue(I.flags(s, self.events, None).isna().all().all())       # 取り込みが無い

    def test_row_order_and_index_are_kept(self):
        s = pd.DataFrame({"Code": ["22220", "11110", "11110", "33330"],
                          "Date": pd.to_datetime(["2024-03-25", "2024-05-30", "2024-03-02", "2024-03-02"])},
                         index=[40, 10, 30, 20])
        f = I.flags(s, self.events, self.start)
        self.assertEqual(list(f.index), [40, 10, 30, 20])
        self.assertEqual(f["inst_buy_20"].tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(f["inst_buy_60"].tolist(), [1.0, 0.0, 1.0, 0.0])

    def test_no_events(self):
        s = pd.DataFrame({"Code": ["11110"], "Date": [pd.Timestamp("2024-03-01")]})
        f = I.flags(s, pd.DataFrame({"Code": pd.Series(dtype=str), "SubDate": pd.Series(dtype="datetime64[ns]")}),
                    self.start)
        self.assertEqual(f.iloc[0].tolist(), [0.0, 0.0])


class TestAttach(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_parquet_round_trip_and_log_has_no_names(self):
        rows = [
            report("D0", "99990", "2024-01-04", [holder("安定株主", 0.05, None, NAMES[2])]),       # 取り込みの始まり
            report("D1", "11110", "2024-03-01", [holder(INVEST_TEXT, 0.06, None)]),
            report("D2", "22220", "2024-03-05", [holder("証券業務に係る商品在庫", 0.07, 0.05, NAMES[1])]),
            report("D3", "22220", "2024-03-06", [holder(INVEST_TEXT, 0.09, 0.05)], doc_type="360"),
        ]
        pd.DataFrame(rows).to_parquet(os.path.join(self.dir, "lvshld.parquet"), index=False)
        s = pd.DataFrame({"Code": ["11110", "22220", "11110", "11110"],
                          "Date": pd.to_datetime(["2024-03-10", "2024-03-10", "2024-01-10", "2024-05-10"])})
        out, st = I.attach(s, self.dir)
        self.assertEqual(list(out.columns), I.COLUMNS)
        self.assertEqual(len(out), 4)
        self.assertEqual(out["inst_buy_20"].tolist()[:2], [1.0, 0.0])           # 訂正・運用でない買いは数えない
        self.assertTrue(np.isnan(out["inst_buy_20"].iloc[2]))                   # 始まり（1/4）+19日より前
        self.assertEqual(out["inst_buy_60"].iloc[3], 0.0)                       # 3/1 から 70日後
        self.assertEqual((st["start"], st["last"], st["events"], st["corrections"]),
                         ("2024-01-04", "2024-03-06", 1, 1))
        text = I.describe(st, len(s))
        self.assertIn("運用目的の保有者の買いを含む報告書 1件", text)
        for name in NAMES:
            self.assertNotIn(name, text)
        self.assertNotIn("11110", text)

    def test_no_data(self):
        s = pd.DataFrame({"Code": ["11110"], "Date": pd.to_datetime(["2024-03-10"])})
        out, st = I.attach(s, self.dir)
        self.assertTrue(out.isna().all().all())
        self.assertEqual(st["reports"], 0)


if __name__ == "__main__":
    unittest.main()
