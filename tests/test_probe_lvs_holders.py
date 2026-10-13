#!/usr/bin/env python3
"""
大量保有報告書の下調べ（research/exp/probe_lvs_holders.py）。

公開のログに出すのは項目名・件数だけ。合成データで、保有者・発行体の名前が出力に出ないこと、区分らしい項目の
値ごとの件数と増減・分類の件数が出ることを確かめる。

  python3 tests/test_probe_lvs_holders.py
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import pandas as pd  # noqa: E402

import probe_lvs_holders as P  # noqa: E402

NAMES = ["テスト・アセットマネジメント株式会社", "テスト証券株式会社", "山田太郎", "テスト商事株式会社",
         "テスト発行体株式会社", "Test Capital Partners LLC"]


def frame():
    rows = [
        {"SubDate": "2024-03-01", "Code": "11110", "IsrName": NAMES[4], "DocTitle": "大量保有報告書",
         "TotalShsRatio": 0.0512, "TotalShsRatioLast": None, "ChgRsn": "",
         "Hldrs": [{"HldrName": NAMES[0], "Purpose": "純投資", "ShsHeld": 1000},
                   {"HldrName": NAMES[5], "Purpose": "純投資", "ShsHeld": 10}]},
        {"SubDate": "2025-06-02", "Code": "22220", "IsrName": NAMES[4], "DocTitle": "変更報告書",
         "TotalShsRatio": 0.0710, "TotalShsRatioLast": 0.0601, "ChgRsn": "市場内取引による取得",
         "Hldrs": [{"HldrName": NAMES[1], "Purpose": "貸借", "ShsHeld": 5000}]},
        {"SubDate": "2025-07-03", "Code": "22220", "IsrName": NAMES[4], "DocTitle": "変更報告書",
         "TotalShsRatio": 0.0402, "TotalShsRatioLast": 0.0710, "ChgRsn": "市場内取引による処分",
         "Hldrs": [{"HldrName": NAMES[2], "Purpose": "政策投資", "ShsHeld": 7}]},
        {"SubDate": "2025-08-04", "Code": "33330", "IsrName": NAMES[4], "DocTitle": "変更報告書",
         "TotalShsRatio": 0.0800, "TotalShsRatioLast": 0.0700, "ChgRsn": None,
         "Hldrs": [{"HldrName": NAMES[3], "Purpose": "重要提案行為等", "ShsHeld": 9}]},
    ]
    return pd.DataFrame(rows)


class Classify(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(P.classify(NAMES[0]), "運用会社・投資顧問")
        self.assertEqual(P.classify(NAMES[1]), "証券会社")
        self.assertEqual(P.classify("野村證券株式会社"), "証券会社")
        self.assertEqual(P.classify(NAMES[2]), "個人")
        self.assertEqual(P.classify(NAMES[3]), "その他の法人")
        self.assertEqual(P.classify("テスト信託銀行株式会社"), "銀行・信託")
        self.assertEqual(P.classify(""), "名前なし")

    def test_ratio_move(self):
        self.assertEqual(P.ratio_move(0.05, None), "新規（前回なし）")
        self.assertEqual(P.ratio_move(0.07, 0.06), "増えた")
        self.assertEqual(P.ratio_move(0.04, 0.07), "減った")
        self.assertEqual(P.ratio_move(None, 0.07), "割合なし")

    def test_nested_values_are_not_listed(self):
        self.assertIsNone(P.level_counts([{"HldrName": "x"}, "a"]))
        self.assertEqual(P.level_counts(["a", "a", "b"])["a"], 2)


class Report(unittest.TestCase):
    def test_output_has_counts_but_no_names(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = P.report(frame())
        text = buf.getvalue()
        for name in NAMES:
            self.assertNotIn(name, text)
        self.assertIn("HldrName", text)                      # 項目名は出す
        self.assertIn("純投資: 2", text)                     # 区分らしい項目の値ごとの件数
        self.assertIn("「市場内」を含む: 2", text)            # 変更の理由は言葉ごとの件数だけ
        self.assertNotIn("市場内取引による取得", text)        # 自由な文の値そのものは出さない
        self.assertEqual(out["moves"], {"新規（前回なし）": 1, "増えた": 2, "減った": 1})
        self.assertEqual(out["first_class"]["運用会社・投資顧問"], 1)
        self.assertEqual(out["years"], {2024: 1, 2025: 3})

    def test_many_free_text_values_are_counted_by_words_only(self):
        f = frame()
        texts = [f"ポートフォリオ投資の一環として純投資（顧客の資産{i}）" for i in range(40)]
        rows = [{"SubDate": "2025-09-01", "Code": f"{i:04d}0", "IsrName": NAMES[4], "DocTitle": "変更報告書",
                 "TotalShsRatio": 0.06, "TotalShsRatioLast": 0.05, "ChgRsn": "",
                 "Hldrs": [{"HldrName": NAMES[0], "Purpose": t}]} for i, t in enumerate(texts)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.report(pd.concat([f, pd.DataFrame(rows)], ignore_index=True))
        text = buf.getvalue()
        self.assertNotIn("顧客の資産1", text)
        self.assertIn("Purpose: 文が入っている 45件", text)
        self.assertIn("純投資 42（93.3%）", text)
        for name in NAMES:
            self.assertNotIn(name, text)

    def test_reads_saved_parquet(self):
        tmp = tempfile.mkdtemp()
        try:
            frame().to_parquet(os.path.join(tmp, "lvshld_2025.parquet"), index=False)
            os.environ["DATA_DIR"] = tmp
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(P.main([]), 0)
            self.assertIn("[load] lvshld 4件", buf.getvalue())
            self.assertIn("Purpose", buf.getvalue())
        finally:
            os.environ.pop("DATA_DIR", None)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
