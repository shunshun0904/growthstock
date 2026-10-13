#!/usr/bin/env python3
"""
週次の再学習で加わったサンプルのチャート（research/new_samples.py）。

運用者の依頼（2026-10-13）「直近追加されたサンプルのチャート（週次のリトレインで新たに追加されたサンプルのチャート）、
正例、負例で5例ずつ程度で良いので、表示してほしい」。

- 加わったサンプル = 前回の学習期間の終わり（meta.json の履歴）より後・今回の終わりまで。前回は git の履歴から取る
- 正例・負例それぞれ最大 N 件。種を固定した無作為抽出なので、同じ週は何度回しても同じ
- 成否の内訳（到達・終盤・トレンド）から引き直したラベルが、データセットのラベルと一致する
- チャートの基準の価格・最大上昇率が、データセットの値と食い違わない（学習のラベルと同じ部品で引いている）

  python3 tests/test_new_samples.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_dataset as B  # noqa: E402
import new_samples as NS  # noqa: E402

N_DAYS = 470
BREAK_AT = 410


def path(kind: str, rng: np.random.Generator) -> np.ndarray:
    """
    日々の終値。410日目まで 1,000円前後で横ばい（日次ボラ 約1%）、410日目に高値を抜ける。
      up   … その後20営業日で +15%（到達・終盤・トレンドとも満たす → 正例）
      fall … 抜けた後 −10% まで下げる（未到達 → 負例）
      fade … +8% まで上げて元に戻る（到達しても終盤とトレンドで外れる → 負例）
    """
    base = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, N_DAYS)))
    base[:BREAK_AT] = 1000 * (1 + rng.normal(0, 0.01, BREAK_AT))
    peak = base[:BREAK_AT].max()
    base[BREAK_AT] = peak * 1.03
    after = np.arange(1, N_DAYS - BREAK_AT)
    if kind == "up":
        base[BREAK_AT + 1:] = base[BREAK_AT] * (1 + np.minimum(after, 20) * 0.0075)
    elif kind == "fall":
        base[BREAK_AT + 1:] = base[BREAK_AT] * (1 - np.minimum(after, 20) * 0.005)
    else:
        bump = np.where(after <= 8, after * 0.01, np.maximum(0.08 - (after - 8) * 0.012, -0.05))
        base[BREAK_AT + 1:] = base[BREAK_AT] * (1 + bump)
    return base


def synthetic_bars() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=N_DAYS)
    out = []
    for code, kind in [("11110", "up"), ("22220", "fall"), ("33330", "fade"), ("44440", "up")]:
        c = path(kind, rng)
        o = np.r_[c[0], c[:-1]]
        out.append(pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"), "Code": code,
            "O": o, "H": np.maximum(o, c) * 1.002, "L": np.minimum(o, c) * 0.998, "C": c, "Vo": 100000.0,
            "AdjO": o, "AdjH": np.maximum(o, c) * 1.002, "AdjL": np.minimum(o, c) * 0.998, "AdjC": c,
            "AdjVo": 100000.0}))
    return pd.concat(out, ignore_index=True)


def dataset_from(bars: pd.DataFrame) -> pd.DataFrame:
    """データセットの行を、学習と同じ部品で作る（更新日のうちラベルが確定した行）。"""
    p = B.attach_rise_label(B.mark_new_highs(B.price_panel(bars)))
    rows = p[p["is_fresh_break"].fillna(False) & p["label"].notna()].copy()
    rows["Code"] = rows["Code"].astype(str)
    return rows[NS.DS_COLS].reset_index(drop=True)


class Build(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = synthetic_bars()
        cls.ds = dataset_from(cls.bars)
        cls.break_day = pd.bdate_range("2024-01-01", periods=N_DAYS)[BREAK_AT].strftime("%Y-%m-%d")
        cls.prev_day = pd.bdate_range("2024-01-01", periods=N_DAYS)[BREAK_AT - 1].strftime("%Y-%m-%d")

    def test_synthetic_rows_cover_both_labels(self):
        on = self.ds[pd.to_datetime(self.ds["Date"]) == pd.Timestamp(self.break_day)]
        self.assertEqual(sorted(on["Code"]), ["11110", "22220", "33330", "44440"])
        self.assertEqual(dict(zip(on["Code"], on["label"].astype(int))),
                         {"11110": 1, "22220": 0, "33330": 0, "44440": 1})

    def test_build_picks_and_verdicts(self):
        meta = {"trainTo": self.break_day, "trainFrom": "2024-01-01", "trainedAt": "2026-10-11T00:00:00+00:00",
                "nTrain": 100, "label": B.DEFAULT_RISE.name, "riseHorizon": B.RISE_HORIZON}
        prev = {"trainTo": self.prev_day, "trainedAt": None, "nTrain": None}
        payload, log = NS.build(self.ds, meta, prev, "", n=1, bars=self.bars,
                                names={"11110": "テストA", "22220": "テストB"})
        self.assertEqual(payload["counts"], {"added": 4, "positive": 2, "negative": 2})
        self.assertEqual((payload["pick"]["positive"], payload["pick"]["negative"]), (1, 1))
        self.assertEqual(log, {"mismatch": 0, "noChart": 0, "entryMismatch": 0, "riseMismatch": 0})
        self.assertEqual(payload["window"], {"after": self.prev_day, "to": self.break_day})
        labels = sorted(c["label"] for c in payload["cases"])
        self.assertEqual(labels, [0, 1])
        for c in payload["cases"]:
            v = c["verdict"]
            self.assertEqual(c["label"] == 1, v["reached"] and v["endOk"] and v["uptrend"])
            self.assertEqual(len(c["dates"]), len(c["close"]))
            self.assertEqual(c["tPos"], NS.BEFORE)                       # 更新日の前は78週（368営業日）
            self.assertEqual(NS.BEFORE, B.HIGH_WINDOW)                    # 運用者の指示「表示期間は78週」
            self.assertEqual(len(c["dates"]), NS.BEFORE + 1 + NS.AFTER)   # 後ろは判定の20営業日＋5日
            self.assertEqual(c["horizon"], [NS.BEFORE + 1, NS.BEFORE + B.RISE_HORIZON])
            for k in ("maShort", "maLong", "high52w"):                    # 画面が描かない系列は持ち出さない
                self.assertNotIn(k, c)
            self.assertEqual(c["t"], self.break_day)
            self.assertGreater(c["target"], c["entryPrice"])
            self.assertEqual(len(c["code"]), 4)
            self.assertNotIn("volume", c)                                 # 出来高・厳しい案は持ち出さない
            self.assertNotIn("targetStrict", c)

    def test_same_week_gives_same_pick_and_caps_at_n(self):
        meta = {"trainTo": self.break_day}
        a, _ = NS.build(self.ds, meta, None, "", n=5, bars=self.bars, names={})
        b, _ = NS.build(self.ds, meta, None, "", n=5, bars=self.bars, names={})
        self.assertEqual([c["jqCode"] for c in a["cases"]], [c["jqCode"] for c in b["cases"]])
        self.assertLessEqual(a["pick"]["positive"], 5)
        self.assertEqual(a["pick"]["seed"], int(self.break_day.replace("-", "")))

    def test_no_added_rows(self):
        meta = {"trainTo": self.break_day}
        payload, log = NS.build(self.ds, meta, {"trainTo": self.break_day}, "", bars=self.bars, names={})
        self.assertEqual(payload["counts"]["added"], 0)
        self.assertEqual(payload["cases"], [])
        self.assertEqual(log["mismatch"], 0)


class Window(unittest.TestCase):
    def test_added_rows_boundaries(self):
        ds = pd.DataFrame({"Code": ["1", "2", "3", "4"], "label": [1, 0, None, 1],
                           "Date": ["2026-09-01", "2026-09-02", "2026-09-05", "2026-09-09"]})
        got = NS.added_rows(ds, "2026-09-01", "2026-09-08")
        self.assertEqual(list(got["Code"]), ["2"])                      # 前回の終わりは含まない・未確定は除く

    def test_prev_window_from_git_history(self):
        tmp = tempfile.mkdtemp()
        try:
            def git(*a):
                subprocess.check_call(["git", "-C", tmp, *a], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git("init", "-q")
            os.makedirs(os.path.join(tmp, "research", "model"))
            p = os.path.join(tmp, "research", "model", "meta.json")
            for to, n in [("2026-09-01", 21917), ("2026-09-08", 21937)]:
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump({"trainTo": to, "trainedAt": f"{to}T00:00:00", "nTrain": n}, fh)
                git("add", "-A")
                git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", to)
            prev = NS.prev_window("2026-09-08", root=tmp)
            self.assertEqual((prev["trainTo"], prev["nTrain"]), ("2026-09-01", 21917))
            self.assertIsNone(NS.prev_window("2026-09-01", root=tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_display_code(self):
        self.assertEqual(NS.display_code("72030"), "7203")
        self.assertEqual(NS.display_code("25935"), "25935")


if __name__ == "__main__":
    unittest.main()
