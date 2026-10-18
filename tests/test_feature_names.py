#!/usr/bin/env python3
"""
特徴量の日本語名（research/feature_dict.py の name_ja と public/data/feature_names.json）。

運用者の依頼（2026-10-13）「特徴量の重要度ランキングですが、実際のインプットに使用している名前ではなく、日本語に変換して
ほしいです」。画面の「効いた特徴量」は feature_names.json の名前で出す。

- 本番と並べているモデルが使う列は、すべて日本語名がある（元の列名のままの列が無い）。名前は重ならない
- 決算の軸 × 接尾辞の列は組み立てで決まる
- 画面が読む feature_names.json は、辞書から作り直したものと一致する（辞書を直したら書き出し直す）

  python3 tests/test_feature_names.py
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import feature_dict as FD  # noqa: E402


def model_features():
    cols = set()
    paths = [os.path.join(ROOT, "research", "model", "meta.json")]
    mdir = os.path.join(ROOT, "research", "model", "models")
    if os.path.isdir(mdir):
        paths += [os.path.join(mdir, d, "meta.json") for d in sorted(os.listdir(mdir))]
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                cols |= set(json.load(fh).get("features") or [])
    return sorted(cols)


class Names(unittest.TestCase):
    def test_every_model_feature_has_a_japanese_name(self):
        cols = model_features()
        self.assertGreater(len(cols), 200)
        raw = [c for c in cols if FD.name_ja(c) == c]
        self.assertEqual(raw, [])
        self.assertFalse([c for c in cols if "_" in FD.name_ja(c)])
        names = [FD.name_ja(c) for c in cols]
        dup = sorted({n for n in names if names.count(n) > 1})
        self.assertEqual(dup, [])

    def test_technical_columns_have_names(self):
        # 既定の360列（249列 + テクニカル指標111列。2026-10-18 の夕方から。§68）は、すべて日本語名があり重ならない
        sys.path.insert(0, os.path.join(ROOT, "research"))
        import features as F
        cols = F.columns("all_plus_prog_listing_vol_book_shape_dd_recent_tx")
        names = [FD.name_ja(c) for c in cols]
        self.assertEqual([c for c, n in zip(cols, names) if n == c or "_" in n], [])
        self.assertEqual(len(set(names)), len(names))
        self.assertTrue(all(FD.describe(c).startswith("テクニカル指標（") for c in F.GROUPS["tech_indicators"]))

    def test_axis_and_suffix(self):
        self.assertEqual(FD.name_ja("eps_growth_q1"), "EPS成長率（1期前）")
        self.assertEqual(FD.name_ja("eps_growth_sym_q0"), "EPS成長率・対称版（直近）")
        self.assertEqual(FD.name_ja("ROE_chg_3q"), "ROEの変化（3期前→今期）")
        self.assertEqual(FD.name_ja("op_margin_chg1"), "営業利益率の変化（前期→今期）")
        self.assertEqual(FD.name_ja("days_to_earn"), "次の決算発表までの日数")
        self.assertEqual(FD.name_ja("no_such_col"), "no_such_col")

    def test_json_matches_the_dictionary(self):
        with open(os.path.join(ROOT, "public", "data", "feature_names.json"), encoding="utf-8") as fh:
            got = json.load(fh)
        want = FD.names_payload(FD.known_columns())
        self.assertEqual(got, want, "辞書を直したら python3 research/feature_dict.py --names-json "
                                    "public/data/feature_names.json で書き出し直す")
        self.assertFalse([c for c, v in got.items() if "**" in v["desc"]])


if __name__ == "__main__":
    unittest.main()
