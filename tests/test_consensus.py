#!/usr/bin/env python3
"""
12本の合意（research/predict_daily.py の consensus_reference / consensus_fields。運用者の指示 2026-10-12）。

- 過去の物差し: 12本の out-of-fold を (Code, Date) で突き合わせ、各行の百分位（pctHistorical と同じ: 同点は数えない）
  の平均を昇順で持つ。基準モデルは model_dir/oof.parquet、ほかは models/<名前>/oof.parquet。無いモデルは外す
- 候補1件: 丸める前の百分位の平均（cons12）と、それが過去の合意の分布で下から何%か（consPct）。欠けたら None
- 実験79 の「合意」と同じ定義（e79_model_clusters.pct_matrix の平均）

  python3 tests/test_consensus.py
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

import models as M  # noqa: E402
import predict_daily as PD  # noqa: E402


def write_oofs(d, n=400, seed=0, skip=()):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({"Code": [str(10000 + i) for i in range(n)],
                         "Date": pd.bdate_range("2022-01-03", periods=n)})
    c = rng.normal(size=n)
    for a in M.LINEUP:
        if a in skip:
            continue
        o = base.copy()
        o["score"] = c + rng.normal(0, 0.5, size=n)
        o["label"] = 0.0
        p = (os.path.join(d, "oof.parquet") if a == M.BASELINE
             else os.path.join(M.sub_dir(a, d), "oof.parquet"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        o.to_parquet(p)
    return base


class TestReference(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_matches_e79_definition(self):
        write_oofs(self.d)
        ref, used = PD.consensus_reference(self.d, M.LINEUP)
        self.assertEqual(used, list(M.LINEUP))
        self.assertEqual(len(ref), 400)
        self.assertTrue(np.all(np.diff(ref) >= 0))                        # 昇順
        # 実験79 の合意（各モデルの自分の分布での百分位の平均）と同じ
        import e79_model_clusters as E79
        wide = None
        for a in M.LINEUP:
            o = PD.oof_scores_of(a, self.d).rename(columns={"score": f"score_{a}"})
            wide = o if wide is None else wide.merge(o, on=["Code", "Date"])
        cons = E79.pct_matrix(wide, M.LINEUP).mean(axis=1)
        np.testing.assert_allclose(ref, np.sort(cons))

    def test_missing_model_is_left_out(self):
        write_oofs(self.d, skip=("cat_k15",))
        ref, used = PD.consensus_reference(self.d, M.LINEUP)
        self.assertNotIn("cat_k15", used)
        self.assertEqual(len(used), len(M.LINEUP) - 1)                      # 8本（2026-10-24 に 1.8σ を外した）から1本欠け
        self.assertFalse(set(used) & set(M.RETIRED_VARIANTS))
        ref2, used2 = PD.consensus_reference(tempfile.mkdtemp(), M.LINEUP)
        self.assertIsNone(ref2)
        self.assertEqual(used2, [])


class TestFields(unittest.TestCase):
    def test_fields(self):
        ref = np.sort(np.arange(1000) / 10.0)                               # 0.0 〜 99.9
        used = list(M.LINEUP)
        pcts = {a: 90.0 for a in used}
        pcts["logit"] = 82.0                                                # 8本の平均 89.0
        f = PD.consensus_fields(pcts, ref, used)
        self.assertEqual(f["cons12"], 89.0)                                 # 名前は12本のころのまま
        self.assertEqual(f["consPct"], 89.0)                                # 0.0〜88.9 の 890件 / 1000
        self.assertEqual(f["nCons"], 8)
        # 1本でも欠けたら出さない（欠けたまま平均すると別の物差しになる）
        p2 = dict(pcts)
        p2["xgb_k15"] = None
        self.assertIsNone(PD.consensus_fields(p2, ref, used)["consPct"])
        self.assertIsNone(PD.consensus_fields(pcts, None, used)["cons12"])
        # 過去の最高より上なら 100、最低以下なら 0
        self.assertEqual(PD.consensus_fields({a: 100.0 for a in used}, ref, used)["consPct"], 100.0)
        self.assertEqual(PD.consensus_fields({a: 0.0 for a in used}, ref, used)["consPct"], 0.0)


if __name__ == "__main__":
    unittest.main()
