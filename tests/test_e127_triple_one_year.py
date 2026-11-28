#!/usr/bin/env python3
"""
実験127（research/exp/e127_triple_one_year.py）。合成の日足と候補で:

- 日足の取り方: 始値・終値は分割調整後を素の値で埋める（build_dataset.price_panel と同じ）。銘柄・日の順に並べる
- 新しいラベル（3倍・2倍は同じ作りで窓と倍率だけが違う。ここでは3倍で確かめる）: 買値は翌営業日の寄り。t+1〜t+h の終値の最大が 買値×3 以上なら正例（ちょうど3倍も正例）。窓の外で届いても負例。
  先が h 行ない行は不明。ただし売買が止まった銘柄は、止まるまでに届いたか（届かなければ負例）。買値が無い行は不明
- 窓の取り方が本番と同じ: 同じ関数で h=20 にした終値の最大／買値−1 が build_dataset.attach_rise_label の future_rise と一致
- 区分: 本番の373列すべてに区分が付く。決まっていないグループの列は止まる
- 割合・読み方: 合計1の割合。読み方は小数1桁に丸めてすべての組が同じ向きのときだけ決まる。--targets の読み方
- 通しで回す: 3倍（1年）と2倍（半年）を同じ回に並べ、すべてのラベルが分かる行だけを使う
- 通しで回す（データの読み込みと木のパラメータは差し替え、学習は小さな木で本物）: 確かめが通り、表と読み方が出て、結果の
  ファイルが書かれ、公開ログに銘柄コードを出さない。買値が本番と合わないデータは止まる

  python3 tests/test_e127_triple_one_year.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import label_specs as LS  # noqa: E402
import e127_triple_one_year as X  # noqa: E402


def bars_frame(code, closes, opens=None, start="2018-01-01"):
    days = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    return pd.DataFrame({"Code": code, "Date": days, "O": opens, "C": closes, "AdjO": opens, "AdjC": closes})


class PriceColumnsTest(unittest.TestCase):
    def test_adjusted_prices_are_filled_with_raw_ones_and_sorted(self):
        b = pd.DataFrame({"Code": ["2", "1", "1"], "Date": ["2020-01-02", "2020-01-03", "2020-01-02"],
                          "O": [10.0, 20.0, 30.0], "C": [11.0, 21.0, 31.0],
                          "AdjO": [np.nan, 2.0, 3.0], "AdjC": [1.1, np.nan, 3.1]})
        p = X.price_columns(b)
        self.assertEqual(list(p["Code"]), ["1", "1", "2"])
        self.assertEqual(list(p["Date"].dt.strftime("%m-%d")), ["01-02", "01-03", "01-02"])
        self.assertEqual(list(p["open"]), [3.0, 2.0, 10.0])
        self.assertEqual(list(p["close"]), [3.1, 21.0, 1.1])


class TripleLabelTest(unittest.TestCase):
    H = 10

    def label_for(self, frames, stop_gap=2):
        prices = X.price_columns(pd.concat(frames, ignore_index=True))
        fr = X.forward_frame(prices, self.H, stop_gap=stop_gap)
        fr["x3"] = X.multiple_label(fr, 3.0)
        return fr

    def test_reach_inside_the_window_is_positive_and_outside_is_negative(self):
        closes = [100.0] * 40
        closes[12] = 300.0          # t=2 の窓（t+1〜t+10 = 3〜12）の中。t=1 の窓（2〜11）の外
        fr = self.label_for([bars_frame("A", closes, [100.0] * 40)])     # 始値はずっと 100
        a = fr[fr["Code"] == "A"].reset_index(drop=True)
        self.assertEqual(a.loc[2, "entry"], 100.0)
        self.assertEqual(a.loc[2, "x3"], 1.0)
        self.assertEqual(a.loc[1, "x3"], 0.0)
        self.assertEqual(a.loc[11, "x3"], 1.0)    # 窓 12〜21 の先頭
        self.assertEqual(a.loc[12, "x3"], 0.0)    # その日自身は入らない（t+1 から）

    def test_exactly_three_times_counts_and_just_below_does_not(self):
        closes = [100.0] * 30
        closes[5] = 300.0
        closes[20] = 299.99
        fr = self.label_for([bars_frame("A", closes)])
        a = fr.reset_index(drop=True)
        self.assertEqual(a.loc[0, "x3"], 1.0)
        self.assertEqual(a.loc[15, "x3"], 0.0)

    def test_buy_price_is_the_next_open(self):
        closes = [100.0] * 30
        opens = [100.0] * 30
        opens[3] = 50.0             # t=2 の買値は 50。150 で3倍
        closes[8] = 150.0
        fr = self.label_for([bars_frame("A", closes, opens)])
        a = fr.reset_index(drop=True)
        self.assertEqual(a.loc[2, "entry"], 50.0)
        self.assertEqual(a.loc[2, "x3"], 1.0)
        self.assertEqual(a.loc[3, "x3"], 0.0)    # 買値 100 には 150 は3倍でない

    def test_rows_without_enough_future_are_unknown_while_listed(self):
        fr = self.label_for([bars_frame("A", [100.0] * 30)])
        a = fr.reset_index(drop=True)
        self.assertTrue(a.loc[19, "have_forward"])
        self.assertFalse(a.loc[20, "have_forward"])
        self.assertEqual(a.loc[19, "x3"], 0.0)
        self.assertTrue(np.isnan(a.loc[20, "x3"]))
        self.assertFalse(a["stopped"].any())

    def test_stopped_names_are_judged_by_what_happened_before_they_stopped(self):
        long = bars_frame("A", [100.0] * 40)
        closes = [100.0] * 25
        closes[22] = 310.0
        stopped = bars_frame("B", closes)
        fr = self.label_for([long, stopped])
        b = fr[fr["Code"] == "B"].reset_index(drop=True)
        self.assertTrue(b["stopped"].all())
        self.assertEqual(b.loc[20, "x3"], 1.0)        # 先は4行しかないが、止まるまでに届いた
        self.assertEqual(b.loc[22, "x3"], 0.0)        # 止まるまで届かなかった
        self.assertTrue(np.isnan(b.loc[24, "x3"]))    # 最後の行は買値が無い

    def test_recently_listed_names_are_not_treated_as_stopped(self):
        long = bars_frame("A", [100.0] * 40)
        recent = bars_frame("B", [100.0] * 39)       # 最後の日は全体の最後の1営業日前
        fr = self.label_for([long, recent])
        b = fr[fr["Code"] == "B"].reset_index(drop=True)
        self.assertFalse(b["stopped"].any())
        self.assertTrue(np.isnan(b.loc[35, "x3"]))

    def test_missing_next_open_is_unknown(self):
        opens = [100.0] * 30
        opens[3] = np.nan
        frame = bars_frame("A", [100.0] * 30, opens)
        frame.loc[3, "O"] = np.nan
        fr = self.label_for([frame])
        self.assertTrue(np.isnan(fr.reset_index(drop=True).loc[2, "x3"]))

    def test_twenty_day_window_matches_the_production_label_materials(self):
        rng = np.random.default_rng(3)
        frames = []
        for i, code in enumerate(["10010", "10020", "10030"]):
            r = rng.normal(0.003, 0.03, 120)
            closes = 100 * np.exp(np.cumsum(r))
            opens = closes * np.exp(rng.normal(0, 0.01, 120))
            frames.append(bars_frame(code, closes[: 120 - 7 * i], opens[: 120 - 7 * i]))
        bars = pd.concat(frames, ignore_index=True)
        prices = X.price_columns(bars)
        panel = prices.copy()
        prod = B.attach_rise_label(panel)
        mine = X.forward_frame(prices, LS.for_model("lgbm")["horizon"])
        ratio = (mine["fmax"] / mine["entry"] - 1.0).where(mine["have_forward"])
        self.assertEqual(X.mismatch_share(ratio, prod["future_rise"]), 0.0)
        self.assertEqual(X.mismatch_share(mine["entry"], prod["entry_price"]), 0.0)
        self.assertGreater(int(prod["future_rise"].notna().sum()), 100)


class CategoryTest(unittest.TestCase):
    def test_every_production_column_has_a_category(self):
        cols = F.columns(F.DEFAULT_PRESET)
        cmap = X.category_map(cols)
        self.assertEqual(len(cmap), len(cols))
        self.assertEqual(set(cmap.values()), {c for c, _ in X.CATEGORIES})
        tech_col = [c for c in F.GROUPS["tech_indicators"] if c in cmap][0]
        self.assertEqual(cmap[tech_col], "tech")
        self.assertEqual(cmap["break_margin"], "tech")
        self.assertEqual(cmap["eps_growth_q0"], "fund")
        self.assertEqual(cmap["per"], "val")
        self.assertEqual(cmap["tv_beta"], "market")
        self.assertEqual(cmap["tv_earnings_yield"], "val")

    def test_unknown_groups_stop(self):
        with mock.patch.object(F, "column_groups", return_value={"x": "brand_new_group"}):
            with self.assertRaises(SystemExit):
                X.category_map(["x"])


class ShareAndVerdictTest(unittest.TestCase):
    def test_shares_sum_to_one_and_group(self):
        s = X.shares(np.array([1.0, 3.0, 0.0]), ["a", "b", "c"])
        self.assertAlmostEqual(sum(s.values()), 1.0)
        self.assertEqual(X.by_key(s, {"a": "t", "b": "f", "c": "t"}), {"t": 0.25, "f": 0.75})
        self.assertEqual(X.shares(np.zeros(2), ["a", "b"]), {"a": 0.0, "b": 0.0})

    def test_verdict_needs_every_pair_in_the_same_direction(self):
        self.assertIn("仮説どおり", X.verdict([(40.0, 30.0), (42.0, 41.0)]))
        self.assertIn("仮説と逆", X.verdict([(30.0, 40.0), (30.0, 30.2)]))
        self.assertEqual(X.verdict([(40.0, 30.0), (30.0, 40.0)]), "はっきりしない")
        self.assertEqual(X.verdict([(40.0, 30.0), (40.04, 40.0)]), "はっきりしない")   # 丸めると同じ
        self.assertIn("比べられない", X.verdict([]))

    def test_direction_is_only_named_when_every_pair_agrees(self):
        self.assertIn("高い", X.direction([(1.0, 2.0), (3.0, 3.5)], "2倍"))
        self.assertIn("低い", X.direction([(2.0, 1.0)], "2倍"))
        self.assertEqual(X.direction([(1.0, 2.0), (3.0, 2.0)], "2倍"), "組によって違う")

    def test_targets_are_parsed_and_named(self):
        self.assertEqual(X.parse_targets("x3:250:3,x2:125:2"), [("x3", 250, 3.0), ("x2", 125, 2.0)])
        self.assertEqual(X.labels_for(X.TARGETS),
                         [("p10", "+10%（本番）"), ("x3", "3倍（1年以内）"), ("x2", "2倍（半年以内）")])
        self.assertEqual(X.target_name(60, 1.5), "1.5倍（60営業日以内）")
        for bad in ("x3:250", "X3:250:3", "x3:250:1", "x3:0:3", "p10:20:2", "x3:250:3,x3:125:2", "", "x-3:250:3"):
            with self.assertRaises(SystemExit, msg=bad):
                X.parse_targets(bad)
        self.assertEqual(X.purge_days_for(X.TARGETS), 380)

    def test_mismatch_share_counts_one_sided_gaps(self):
        a = pd.Series([1.0, 2.0, np.nan, 4.0])
        b = pd.Series([1.0, 2.0 + 1e-6, 3.0, np.nan])
        self.assertAlmostEqual(X.mismatch_share(a, b), 3 / 4)
        self.assertEqual(X.mismatch_share(a.iloc[:1], b.iloc[:1]), 0.0)


def synthetic_inputs(seed=0):
    """5年ぶんの日足（強く上がる銘柄・横ばいの銘柄）と、そこから本番の作り方で付けた +10% のラベルつきの候補。"""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(36):
        code = f"9{i:03d}0"
        drift = 0.006 if i % 3 == 0 else 0.0
        r = rng.normal(drift, 0.02, 1300)
        closes = 100 * np.exp(np.cumsum(r))
        opens = closes * np.exp(rng.normal(0, 0.005, 1300))
        n = 900 if i == 0 else 1300          # 1銘柄は途中で売買が止まる（上場廃止の形）
        frames.append(bars_frame(code, closes[:n], opens[:n], start="2018-01-01"))
    bars = pd.concat(frames, ignore_index=True)
    panel = X.price_columns(bars)
    panel = B.attach_rise_label(panel)
    ds = panel.iloc[::7].copy()
    ds = ds[ds["future_rise"].notna()]
    ds["label"] = LS.labels(ds, "lgbm")
    ds = ds[ds["label"].notna()].copy()
    cols = F.columns(F.DEFAULT_PRESET)
    noise = rng.normal(size=(len(ds), len(cols)))
    feat = pd.DataFrame(noise, columns=cols, index=ds.index)
    ds = pd.concat([ds, feat], axis=1)
    ds["Code"] = ds["Code"].astype(str)
    return ds.reset_index(drop=True), bars


class EndToEndTest(unittest.TestCase):
    PARAMS = {"n_estimators": 15, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1}

    def run_main(self, ds, bars, *extra):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "res.json")
            with mock.patch.object(X, "load_inputs", return_value=(ds, bars)), \
                    mock.patch.object(X, "production_params", return_value=dict(self.PARAMS)), \
                    contextlib.redirect_stdout(out):
                rc = X.main(["--seeds", "0,1", "--ends", "2020-06-30,all", "--out", path, *extra])
            saved = None
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    saved = json.load(fh)
        return rc, out.getvalue(), saved

    def test_runs_through_and_keeps_codes_out_of_the_log(self):
        ds, bars = synthetic_inputs()
        rc, text, saved = self.run_main(ds, bars)
        self.assertEqual(rc, 0)
        self.assertIn("[check] 買値", text)
        self.assertIn("区分ごとの重要度の割合", text)
        self.assertIn("読み方（回す前に決めたもの）:", text)
        self.assertIn("参考: 前進分割", text)
        self.assertIn("3倍（1年以内）: ", text)
        self.assertIn("2倍（半年以内）: ", text)
        for code in ds["Code"].unique():
            self.assertNotIn(code, text)
        self.assertIsNotNone(saved)
        self.assertEqual(sorted(saved["pairs_tech"]), ["x2", "x3"])
        self.assertEqual(len(saved["pairs_tech"]["x3"]), 4)          # 期間2つ × 種2つ
        self.assertEqual(len(saved["pairs_tech"]["x2"]), 4)
        self.assertEqual(sorted(saved["verdict"]), ["x2", "x3"])
        for key in ("p10", "x3", "x2"):
            self.assertAlmostEqual(sum(saved["mean"][key]["category"].values()), 1.0, places=6)
        info = saved["info"]
        self.assertGreater(info["known"], 0)
        # すべてのラベルが分かる行だけ: 2倍だけなら分かる直近の行は使わない
        self.assertGreater(info["targets"]["x2"]["known_alone"], info["known"])
        self.assertEqual(info["targets"]["x3"]["known_alone"], info["known"])
        self.assertGreater(info["targets"]["x3"]["stopped_known"], 0)
        self.assertEqual(sum(v["n"] for v in info["years"].values()), info["known"])

    def test_stops_when_the_buy_price_does_not_match_production(self):
        ds, bars = synthetic_inputs()
        ds.loc[: len(ds) // 2, "entry_price"] = ds.loc[: len(ds) // 2, "entry_price"] * 1.01
        with self.assertRaises(SystemExit):
            self.run_main(ds, bars)


if __name__ == "__main__":
    unittest.main()
