#!/usr/bin/env python3
"""
research/jsf_features_v2.py（日証金の需給の特徴量・作り直し版）の単体テスト。合成のデータで:

- 営業日の格子: 確報が欠けた日は行を足して欠測にする。その日の行はどの列も欠測
- 窓は営業日で数える: 欠けた日をまたぐ5・20日の変化は欠測、5日の新規の合計は5日がそろったときだけ、逆日歩の日数・最大は
  10日以上、60日の位置は40日以上
- 同じ申込日なら毎日の確報（daily）を残す。品貸料率の年率換算は daily の行に無ければ hist の同じ日の値。観測のある日で
  料率が無ければ 0
- 制限措置: hist の行があればその値、無ければ写し（写しの日から shift 営業日後の申込日）、写しの無い日は欠測。shift が
  決まっていなければ写しを使わない
- 貸借区分: J-Quants の信用区分（2 → 1、1・3 → 0）を申込日に後ろ向きで合わせる
- 取引日 T には T より前の最新の観測の値（8暦日以上前は欠測）
- 先読みが無い: T より前のデータだけで作り直しても、T の行の値は全部同じ
- 手で計算した値と一致する

  python3 -m unittest tests.test_jsf_features_v2
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import jsf_features_v2 as V2  # noqa: E402

DAYS = pd.bdate_range("2026-01-05", periods=95)
A, B, C = "11110", "22220", "33330"
HOLE = 50                         # A の確報が欠けた日（格子で足す）


def hist_rows(code, idx, base=1000.0, step=10.0, fee_on=(), restrict=None):
    rows = []
    for i in idx:
        loan = base + step * i
        stock = 300.0 + (i % 7) * 5.0
        rows.append({"app_date": DAYS[i], "Code": code, "loan_new": 20.0 + i % 3, "loan_ret": 10.0,
                     "loan_bal": loan, "stock_new": 5.0 + i % 2, "stock_ret": 4.0, "stock_bal": stock,
                     "net_bal": loan - stock, "price": 500.0, "fee_yen": np.nan, "fee_days": np.nan,
                     "fee_ann": (1.5 + i % 4) if i in fee_on else np.nan,
                     "restrict": (restrict(i) if restrict else 0.0), "lendable": 1.0, "source": "hist"})
    return pd.DataFrame(rows)


def make_inputs():
    ha = hist_rows(A, [i for i in range(0, 80) if i != HOLE], fee_on={3, 4, 40, 41, 42, 70},
                   restrict=lambda i: 1.0 if 30 <= i < 33 else 0.0)
    hb = hist_rows(B, range(10, 80), base=5000.0, step=-20.0)
    hist = pd.concat([ha, hb], ignore_index=True)
    d = hist_rows(A, range(75, 86))
    d["source"] = "daily"
    d["restrict"] = np.nan
    d["lendable"] = np.nan
    d.loc[d["app_date"] == DAYS[76], "loan_bal"] = 9999.0          # 同じ日は daily を残す
    d.loc[d["app_date"] == DAYS[79], "fee_ann"] = 7.0              # hist の最終日（速報）に料率が無い日
    snaps = pd.DataFrame({"snap_date": [DAYS[i] for i in range(78, 87)] + [DAYS[81]],
                          "code": ["99990"] * 9 + [A], "level": [3.0] * 9 + [1.0]})
    snaps = snaps.groupby(["snap_date", "code"], as_index=False)["level"].max()
    master = pd.DataFrame({"Date": [DAYS[0], DAYS[0], DAYS[60]], "Code": [A, B, A], "Mrgn": ["2", "1", "1"]})
    margin = V2.margin_from_master(master)
    vol = pd.DataFrame({"Date": list(DAYS) * 2, "Code": [A] * len(DAYS) + [B] * len(DAYS),
                        "avg_vol": [1000.0] * len(DAYS) + [2000.0] * len(DAYS)})
    return hist, d, snaps, margin, vol


def build(shift=1, cut=None):
    hist, daily, snaps, margin, vol = make_inputs()
    if cut is not None:                                             # T より前のデータだけ
        hist = hist[hist["app_date"] < cut]
        daily = daily[daily["app_date"] < cut]
        snaps = snaps[snaps["snap_date"] < cut]
        margin = margin[margin["Date"] < cut]
        vol = vol[vol["Date"] < cut]
    g = V2.panel_v2(hist, daily, DAYS, snaps, shift, margin)
    return g, V2.features_v2(g, vol)


def row(f, code, i):
    r = f[(f["Code"] == code) & (f["app_date"] == DAYS[i])]
    assert len(r) == 1, (code, i, len(r))
    return r.iloc[0]


class TestGridAndWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g, cls.f = build()

    def test_grid_has_the_hole_as_missing(self):
        a = self.g[self.g["Code"] == A]
        self.assertEqual(len(a), 86)                                 # 0〜85（daily で延びる）
        self.assertFalse(bool(row(self.g, A, HOLE)["observed"]))
        self.assertEqual(row(self.g, A, HOLE)["source"], "missing")
        for c in V2.BASE_COLS:
            self.assertTrue(np.isnan(row(self.f, A, HOLE)[c]), c)
        b = self.g[self.g["Code"] == B]
        self.assertEqual((b["app_date"].min(), b["app_date"].max()), (DAYS[10], DAYS[79]))

    def test_changes_across_the_hole_are_missing(self):
        self.assertTrue(np.isnan(row(self.f, A, 55)["jsf2_ratio_chg5"]))      # 5営業日前 = 欠けた日
        r56, r51 = row(self.f, A, 56), row(self.f, A, 51)
        self.assertAlmostEqual(r56["jsf2_ratio_chg5"], r56["jsf2_ratio"] - r51["jsf2_ratio"], places=12)
        self.assertTrue(np.isnan(row(self.f, A, 70)["jsf2_loan_chg20_v"]))    # 20営業日前 = 欠けた日
        loan71, loan51 = 1000.0 + 10 * 71, 1000.0 + 10 * 51
        self.assertAlmostEqual(row(self.f, A, 71)["jsf2_loan_chg20_v"], np.arcsinh((loan71 - loan51) / 1000.0),
                               places=12)

    def test_sums_need_all_five_days(self):
        for i in range(50, 55):
            self.assertTrue(np.isnan(row(self.f, A, i)["jsf2_long_new5_v"]), i)
        tot = sum(20.0 + k % 3 for k in range(51, 56))
        self.assertAlmostEqual(row(self.f, A, 55)["jsf2_long_new5_v"], np.arcsinh(tot / 1000.0), places=12)

    def test_fee_counts_and_max_use_available_days(self):
        r = row(self.f, A, 59)                       # 窓 40〜59（欠けた日 50 を除く19日）: 料率のある日 40・41・42
        self.assertEqual(r["jsf2_fee_days20"], 3.0)
        self.assertAlmostEqual(r["jsf2_fee_max20"], np.log1p(max(1.5 + k % 4 for k in (40, 41, 42))), places=12)
        self.assertAlmostEqual(row(self.f, A, 41)["jsf2_fee"], np.log1p(1.5 + 41 % 4), places=12)
        self.assertEqual(row(self.f, A, 43)["jsf2_fee"], 0.0)                  # 観測のある日で料率なし = 0
        self.assertTrue(np.isnan(row(self.f, A, 8)["jsf2_fee_days20"]))         # 10日に満たない

    def test_position_in_own_60_days(self):
        self.assertEqual(row(self.f, A, 70)["jsf2_loan_pos60"], 1.0)           # 増え続ける → 最も高い
        self.assertEqual(row(self.f, B, 70)["jsf2_loan_pos60"], 0.0)           # 減り続ける → 最も低い
        self.assertTrue(np.isnan(row(self.f, B, 48)["jsf2_loan_pos60"]))        # 39日しかない（10〜48）
        self.assertFalse(np.isnan(row(self.f, B, 49)["jsf2_loan_pos60"]))       # 40日
        # 手で: A の日 70 の貸株残高の位置（窓 11〜70、欠けた日 50 を除く59個）
        vals = np.array([300.0 + (k % 7) * 5.0 for k in range(11, 71) if k != HOLE])
        x = vals[-1]
        rank = (vals < x).sum() + ((vals == x).sum() + 1) / 2.0
        self.assertAlmostEqual(row(self.f, A, 70)["jsf2_stock_pos60"], (rank - 1) / (len(vals) - 1), places=12)

    def test_values_by_hand(self):
        r = row(self.f, A, 30)
        loan, stock = 1000.0 + 300.0, 300.0 + (30 % 7) * 5.0
        self.assertAlmostEqual(r["jsf2_ratio"], np.log((loan + 1) / (stock + 1)), places=12)
        self.assertAlmostEqual(r["jsf2_loan_v"], np.arcsinh(loan / 1000.0), places=12)
        self.assertAlmostEqual(r["jsf2_net_v"], np.arcsinh((loan - stock) / 1000.0), places=12)
        self.assertAlmostEqual(row(self.f, B, 30)["jsf2_loan_v"], np.arcsinh((5000.0 - 600.0) / 2000.0), places=12)


class TestSources(unittest.TestCase):
    def test_daily_wins_and_fills_fee(self):
        g, f = build()
        self.assertEqual(row(g, A, 76)["loan_bal"], 9999.0)
        self.assertEqual(row(g, A, 76)["source"], "daily")
        self.assertAlmostEqual(row(f, A, 79)["jsf2_fee"], np.log1p(7.0), places=12)

    def test_restrict_from_hist_then_snapshots(self):
        g, f = build(shift=1)
        self.assertEqual(row(f, A, 31)["jsf2_restrict"], 1.0)                  # hist
        self.assertEqual(row(g, A, 31)["restrict_src"], "hist")
        self.assertEqual(row(f, A, 79)["jsf2_restrict"], 0.0)                  # hist の最終日
        self.assertEqual(row(f, A, 82)["jsf2_restrict"], 1.0)                  # 写し 81 → 申込日 82
        self.assertEqual(row(g, A, 82)["restrict_src"], "snap")
        for i in (80, 81, 83, 84, 85):
            self.assertEqual(row(f, A, i)["jsf2_restrict"], 0.0, i)            # 写しにあって載っていない
        g0, f0 = build(shift=None)
        for i in (80, 82, 85):
            self.assertTrue(np.isnan(row(f0, A, i)["jsf2_restrict"]), i)       # 写しを使わない
        g2, f2 = build(shift=0)
        self.assertEqual(row(f2, A, 81)["jsf2_restrict"], 1.0)                 # shift 0: 写し 81 → 申込日 81

    def test_restrict_missing_without_snapshot(self):
        hist, daily, snaps, margin, vol = make_inputs()
        snaps = snaps[snaps["snap_date"] != DAYS[83]]                          # 写し 83（→ 申込日 84）が無い
        g = V2.panel_v2(hist, daily, DAYS, snaps, 1, margin)
        f = V2.features_v2(g, vol)
        self.assertTrue(np.isnan(row(f, A, 84)["jsf2_restrict"]))
        self.assertEqual(row(f, A, 85)["jsf2_restrict"], 0.0)

    def test_lendable_from_jquants_margin(self):
        g, f = build()
        self.assertEqual(row(f, A, 59)["jsf2_lendable"], 1.0)
        self.assertEqual(row(f, A, 60)["jsf2_lendable"], 0.0)                  # 区分が変わった日から
        self.assertEqual(row(f, B, 30)["jsf2_lendable"], 0.0)
        m = V2.margin_from_master(pd.DataFrame({"Date": [DAYS[0]] * 4, "Code": ["1", "2", "3", "4"],
                                                "Mrgn": ["2", "1", "3", None]}))
        self.assertEqual(m.set_index("Code")["lendable"].tolist()[:3], [1.0, 0.0, 0.0])
        self.assertTrue(np.isnan(m.set_index("Code").loc["4", "lendable"]))

    def test_off_calendar_dates_are_rejected(self):
        hist, daily, snaps, margin, vol = make_inputs()
        bad = hist.iloc[[0]].copy()
        bad["app_date"] = pd.Timestamp("2026-01-10")                          # 土曜
        with self.assertRaises(ValueError):
            V2.panel_v2(pd.concat([hist, bad], ignore_index=True), daily, DAYS, snaps, 1, margin)


class TestAttach(unittest.TestCase):
    def test_previous_observed_day_and_lag(self):
        g, f = build()
        frame = pd.DataFrame({"Code": [A, A, C, A], "Date": [DAYS[51], DAYS[86], DAYS[60], DAYS[0]]})
        out = V2.attach_v2(frame, f)
        self.assertEqual(out.loc[0, "jsf2_lag"], float((DAYS[51] - DAYS[49]).days))   # 50 は欠けた日 → 49
        self.assertAlmostEqual(out.loc[0, "jsf2_ratio"], row(f, A, 49)["jsf2_ratio"], places=12)
        self.assertAlmostEqual(out.loc[1, "jsf2_ratio"], row(f, A, 85)["jsf2_ratio"], places=12)
        self.assertTrue(out.loc[2, V2.columns("both")].isna().all())           # 日証金に無い銘柄
        self.assertTrue(out.loc[3, V2.columns("both")].isna().all())           # 最初の日より前

    def test_stale_values_are_not_used(self):
        g, f = build()
        frame = pd.DataFrame({"Code": [A], "Date": [DAYS[85] + pd.Timedelta(days=8)]})
        self.assertTrue(V2.attach_v2(frame, f)[V2.columns("all")].isna().all(axis=None))

    def test_no_lookahead(self):
        """T より前のデータだけで作り直しても、T の行の値は全部同じ（日足・銘柄一覧・写しも切る）。"""
        g, full = build()
        for i in (40, 51, 61, 80, 82, 85):
            t = DAYS[i]
            frame = pd.DataFrame({"Code": [A, B], "Date": [t, t]})
            want = V2.attach_v2(frame, full)
            _, part = build(cut=t)
            got = V2.attach_v2(frame, part)
            pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=0, atol=1e-12, obj=f"T={i}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
