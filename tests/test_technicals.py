#!/usr/bin/env python3
"""
テクニカル指標111列（research/technicals.py）の確かめ。データには触らない（作りものの日足で確かめる）。

- 一覧: 111列・5系統（41 / 26 / 14 / 18 / 12）・名前はすべて tx_ で始まり、日本語の名前がそろう
- 先を見ない: 各銘柄の後ろの行を消しても、残った行の値は111列すべて同じ
- 銘柄をまたがない: 1銘柄ずつ計算しても同じ。売買の無い行（終値の欠測）は欠測で、それを除いて計算しても同じ
- 定義どおり: 指標ごとに、定義をそのまま1行ずつ数える素朴な計算（下の ref_*）と一致する。値が出始める行も一致する
- 性質: 上がり続ける日足で RSI 100・ストキャスティクス 100・%R 0・アルーン・アップ 100、動かない日足で RSI 50・%B 50・CCI 0 など

  python3 tests/test_technicals.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import technicals as TX  # noqa: E402


def make_bars(n=420, seed=0, code="10010", start="2022-01-03"):
    rng = np.random.default_rng(seed)
    close = 1000 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    o = close * np.exp(rng.normal(0, 0.01, n))
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    # 同じ終値が続く日・値幅 0 の日も混ぜる（同じ値の順位・0 で割る所を通す）
    if n > 60:
        close[50] = close[49]
        o[60] = h[60] = lo[60] = close[60]
    v = rng.integers(1_000, 200_000, n).astype(float)
    return pd.DataFrame({"Code": code, "Date": pd.bdate_range(start, periods=n), "open": o, "high": h, "low": lo,
                         "close": close, "vol": v})


def panel3():
    return pd.concat([make_bars(420, 0, "10010"), make_bars(300, 1, "10020"), make_bars(40, 2, "10030")],
                     ignore_index=True)


def same(a, b, tol=1e-9) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    m = ~np.isnan(a)
    return bool(np.allclose(a[m], b[m], rtol=tol, atol=tol))


# --------------------------------------------------------------------------- #
# 素朴な計算（1銘柄。定義をそのまま1行ずつ）
# --------------------------------------------------------------------------- #
def ema(x, alpha):
    """pandas の ewm(adjust=False) と同じ: 最初の欠測でない値から始め、y = (1 − α) y + α x。"""
    out = np.full(len(x), np.nan)
    y = None
    for i, v in enumerate(x):
        if np.isnan(v):
            if y is not None:
                out[i] = y
            continue
        y = v if y is None else (1 - alpha) * y + alpha * v
        out[i] = y
    return out


def span(n):
    return 2.0 / (n + 1.0)


def sma(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        if not np.isnan(w).any():
            out[i] = w.mean()
    return out


def first_valid(x) -> int:
    ok = np.flatnonzero(~np.isnan(np.asarray(x, dtype=float)))
    return int(ok[0]) if len(ok) else -1


def ref_rsi(c, n):
    d = np.r_[np.nan, np.diff(c)]
    g = ema(np.where(np.isnan(d), np.nan, np.maximum(d, 0)), 1 / n)
    lo = ema(np.where(np.isnan(d), np.nan, np.maximum(-d, 0)), 1 / n)
    out = np.where(g + lo > 0, 100 * g / (g + lo), 50.0)
    out[:n] = np.nan
    return out


def ref_true_range(h, lo, c):
    tr = np.empty(len(c))
    tr[0] = h[0] - lo[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
    return tr


class TestList(unittest.TestCase):
    def test_columns(self):
        self.assertEqual(len(TX.COLUMNS), 111)
        self.assertEqual({g: len(c) for g, c in TX.GROUPS.items()},
                         {"trend": 41, "oscillator": 26, "volatility": 14, "volume": 18, "candle": 12})
        self.assertTrue(all(c.startswith("tx_") for c in TX.COLUMNS))
        self.assertEqual(set(TX.JA), set(TX.COLUMNS))
        self.assertEqual(set(TX.GROUP_JA), set(TX.GROUPS))

    def test_input_checks(self):
        p = panel3()
        with self.assertRaises(ValueError):
            TX.compute(p.drop(columns=["vol"]))
        with self.assertRaises(ValueError):
            TX.compute(p.iloc[::-1].reset_index(drop=True))                     # 日付が逆
        split = pd.concat([p[p["Code"] == "10010"].iloc[:100], p[p["Code"] == "10020"],
                           p[p["Code"] == "10010"].iloc[100:]], ignore_index=True)
        with self.assertRaises(ValueError):
            TX.compute(split)                                                   # 同じ銘柄が続いていない


class TestNoLeak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = panel3()
        cls.full = TX.compute(cls.p)

    def test_shape(self):
        self.assertEqual(list(self.full.columns), TX.COLUMNS)
        self.assertTrue(self.full.index.equals(self.p.index))
        last = self.full[self.p["Code"] == "10010"].iloc[-1]
        self.assertEqual([c for c in TX.COLUMNS if np.isnan(last[c])], [])     # 長い銘柄の最後の行は全列に値

    def test_no_future(self):
        cut = self.p["Date"] <= pd.Timestamp("2022-11-30")
        part = TX.compute(self.p[cut].reset_index(drop=True))
        for c in TX.COLUMNS:
            self.assertTrue(same(part[c].to_numpy(), self.full.loc[cut, c].to_numpy()), c)

    def test_one_code_at_a_time(self):
        for code in ("10010", "10020", "10030"):
            m = (self.p["Code"] == code).to_numpy()
            one = TX.compute(self.p[m].reset_index(drop=True))
            for c in TX.COLUMNS:
                self.assertTrue(same(one[c].to_numpy(), self.full.loc[m, c].to_numpy()), f"{code} {c}")

    def test_compute_at(self):
        """銘柄を分けて計算し keys の行だけ残しても、まとめて計算した値と同じ（keys の並びのまま・日足に無い行は欠測）。"""
        pick = self.p.iloc[[700, 3, 419, 420, 719, 250]][["Code", "Date"]]
        keys = pd.concat([pick, pd.DataFrame({"Code": ["99990"], "Date": [pd.Timestamp("2022-06-01")]})],
                         ignore_index=True)
        for chunk in (1, 2, 400):
            got = TX.compute_at(self.p, keys, codes_per_chunk=chunk)
            self.assertEqual(list(got.columns), TX.COLUMNS)
            self.assertEqual(len(got), len(keys))
            for i, j in enumerate([700, 3, 419, 420, 719, 250]):
                self.assertTrue(same(got.iloc[i].to_numpy(), self.full.iloc[j].to_numpy()), f"chunk {chunk} row {j}")
            self.assertTrue(got.iloc[-1].isna().all())

    def test_untraded_rows(self):
        p = self.p.copy()
        gone = [30, 31, 200, 500]
        p.loc[gone, ["open", "high", "low", "close"]] = np.nan
        p.loc[gone, "vol"] = 0.0
        out = TX.compute(p)
        self.assertTrue(out.loc[gone].isna().all().all())
        kept = TX.compute(p.drop(index=gone).reset_index(drop=True))
        rest = out.drop(index=gone).reset_index(drop=True)
        for c in TX.COLUMNS:
            self.assertTrue(same(rest[c].to_numpy(), kept[c].to_numpy()), c)


class TestDefinitions(unittest.TestCase):
    """1銘柄（420行）で、素朴な計算と一致するか。"""

    @classmethod
    def setUpClass(cls):
        cls.b = make_bars(420, 5)
        cls.o = TX.compute(cls.b)
        cls.C = cls.b["close"].to_numpy()
        cls.O = cls.b["open"].to_numpy()
        cls.H = np.maximum.reduce([cls.b["high"].to_numpy(), cls.O, cls.C])
        cls.L = np.minimum.reduce([cls.b["low"].to_numpy(), cls.O, cls.C])
        cls.V = cls.b["vol"].to_numpy()
        cls.n = len(cls.C)

    def check(self, col, want, start=None):
        got = self.o[col].to_numpy()
        self.assertTrue(same(got, want), f"{col}: 最初の値 {first_valid(got)} / {first_valid(want)}・"
                                         f"最大差 {np.nanmax(np.abs(got - want)) if (~np.isnan(got)).any() else 'nan'}")
        if start is not None:
            self.assertEqual(first_valid(got), start, col)

    # ---- トレンド系
    def test_moving_averages(self):
        C = self.C
        s = {k: sma(C, k) for k in (5, 25, 50, 75, 200)}
        self.check("tx_ma_gap_25", (C / s[25] - 1) * 100, 24)
        self.check("tx_ma_gap_200", (C / sma(C, 200) - 1) * 100, 199)
        slope = np.full(self.n, np.nan)
        slope[5:] = (s[25][5:] / s[25][:-5] - 1) * 100
        self.check("tx_ma_slope_25", slope, 29)
        self.check("tx_ma_ratio_50_200", (s[50] / s[200] - 1) * 100, 199)
        po = (s[5] > s[25]).astype(float) + (s[25] > s[75]) + (s[75] > s[200])
        po[np.isnan(s[200])] = np.nan
        self.check("tx_perfect_order", po, 199)
        # 25日と75日の交差からの日数（交差した日が 1。同じ並びが続く間は数え続ける）
        want = np.full(self.n, np.nan)
        prev, cnt = None, 0
        for i in range(self.n):
            if np.isnan(s[75][i]):
                continue
            sg = np.sign(s[25][i] - s[75][i])
            cnt = cnt + 1 if sg == prev else 1
            prev = sg
            want[i] = cnt * sg
        self.check("tx_ma_cross_days", want, 74)

    def test_macd(self):
        C = self.C
        macd = ema(C, span(12)) - ema(C, span(26))
        sig = ema(macd, span(9))
        w = macd / C * 100
        w[:25] = np.nan
        self.check("tx_macd", w, 25)
        h = (macd - sig) / C * 100
        h[:34] = np.nan
        self.check("tx_macd_hist", h, 34)

    def test_dmi(self):
        H, L, C = self.H, self.L, self.C
        tr = ref_true_range(H, L, C)
        pdm, mdm = np.full(self.n, np.nan), np.full(self.n, np.nan)
        for i in range(1, self.n):
            up, dn = H[i] - H[i - 1], L[i - 1] - L[i]
            pdm[i] = up if (up > dn and up > 0) else 0.0
            mdm[i] = dn if (dn > up and dn > 0) else 0.0
        atr = ema(tr, 1 / 14)
        pdi, mdi = 100 * ema(pdm, 1 / 14) / atr, 100 * ema(mdm, 1 / 14) / atr
        dx = np.where(pdi + mdi > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)
        dx[np.isnan(pdi)] = np.nan
        adx = ema(dx, 1 / 14)
        for a, k in ((pdi, 14), (mdi, 14), (adx, 27)):
            a[:k] = np.nan
        self.check("tx_pdi_14", pdi, 14)
        self.check("tx_mdi_14", mdi, 14)
        self.check("tx_adx_14", adx, 27)

    def test_aroon_donchian(self):
        H, L, C = self.H, self.L, self.C
        up, dn = np.full(self.n, np.nan), np.full(self.n, np.nan)
        for i in range(25, self.n):
            wh, wl = H[i - 25:i + 1], L[i - 25:i + 1]
            age_h = 25 - max(j for j in range(26) if wh[j] == wh.max())      # 一番新しい最大
            age_l = 25 - max(j for j in range(26) if wl[j] == wl.min())
            up[i], dn[i] = 100 * (25 - age_h) / 25, 100 * (25 - age_l) / 25
        self.check("tx_aroon_up_25", up, 25)
        self.check("tx_aroon_down_25", dn, 25)
        pos = np.full(self.n, np.nan)
        for i in range(19, self.n):
            hh, ll = H[i - 19:i + 1].max(), L[i - 19:i + 1].min()
            pos[i] = 100 * (C[i] - ll) / (hh - ll) if hh > ll else 50.0
        self.check("tx_donchian_pos_20", pos, 19)

    def test_ichimoku(self):
        H, L, C = self.H, self.L, self.C

        def mid(i, k):
            return (H[i - k + 1:i + 1].max() + L[i - k + 1:i + 1].min()) / 2 if i >= k - 1 else np.nan

        tk = np.array([mid(i, 9) for i in range(self.n)])
        kj = np.array([mid(i, 26) for i in range(self.n)])
        sb_raw = np.array([mid(i, 52) for i in range(self.n)])
        sa = np.r_[np.full(25, np.nan), ((tk + kj) / 2)[:-25]]
        sb = np.r_[np.full(25, np.nan), sb_raw[:-25]]
        top, bot = np.maximum(sa, sb), np.minimum(sa, sb)
        pos = np.where(C > top, (C - top) / C, np.where(C < bot, (C - bot) / C, 0.0)) * 100
        pos[np.isnan(top)] = np.nan
        self.check("tx_ichi_cloud_pos", pos, 76)
        self.check("tx_ichi_cloud_thick", (sa - sb) / C * 100, 76)
        self.check("tx_ichi_tk", (tk - kj) / C * 100, 25)
        c25 = np.r_[np.full(25, np.nan), C[:-25]]
        san = (tk > kj).astype(float) + (C > top) + (C > c25)
        san[np.isnan(top) | np.isnan(kj)] = np.nan
        self.check("tx_ichi_sanyaku", san, 76)
        self.check("tx_ichi_chikou", (C / c25 - 1) * 100, 25)

    def test_regression_er_trix_vortex(self):
        C, H, L = self.C, self.H, self.L
        y = np.log(C)
        sl, r2 = np.full(self.n, np.nan), np.full(self.n, np.nan)
        for i in range(59, self.n):
            t = np.arange(60.0)
            b = np.polyfit(t, y[i - 59:i + 1], 1)[0]
            sl[i] = b * 100
            r2[i] = np.corrcoef(t, y[i - 59:i + 1])[0, 1] ** 2
        self.check("tx_linreg_slope_60", sl, 59)
        self.check("tx_linreg_r2_60", r2, 59)
        er = np.full(self.n, np.nan)
        for i in range(20, self.n):
            path_ = np.abs(np.diff(C[i - 20:i + 1])).sum()
            er[i] = abs(C[i] - C[i - 20]) / path_ if path_ > 0 else 0.0
        self.check("tx_kaufman_er_20", er, 20)
        e3 = ema(ema(ema(C, span(15)), span(15)), span(15))
        trix = np.r_[np.nan, (e3[1:] / e3[:-1] - 1) * 100]
        trix[:44] = np.nan
        self.check("tx_trix_15", trix, 44)
        tr = ref_true_range(H, L, C)
        vp = np.full(self.n, np.nan)
        for i in range(14, self.n):
            vp[i] = sum(abs(H[j] - L[j - 1]) for j in range(i - 13, i + 1)) / tr[i - 13:i + 1].sum()
        self.check("tx_vortex_pos_14", vp, 14)

    def test_psar_and_supertrend_shape(self):
        # 上がり続ける日足: SAR は安値の下で、上昇のまま日数が伸びる。下がり続けると逆
        n = 60
        c = 100 * 1.01 ** np.arange(n)
        up = pd.DataFrame({"Code": "1", "Date": pd.bdate_range("2024-01-01", periods=n), "open": c * 0.995,
                           "high": c * 1.005, "low": c * 0.99, "close": c, "vol": 1000.0})
        o = TX.compute(up)
        self.assertTrue((o["tx_psar_gap"].dropna() > 0).all())
        self.assertEqual(o["tx_psar_days"].iloc[-1], n)
        self.assertTrue((o["tx_supertrend_gap"].dropna() > 0).all())
        dn = up.assign(open=up["open"].to_numpy()[::-1], high=up["high"].to_numpy()[::-1],
                       low=up["low"].to_numpy()[::-1], close=up["close"].to_numpy()[::-1])
        o2 = TX.compute(dn)
        self.assertTrue((o2["tx_psar_gap"].iloc[5:] < 0).all())
        self.assertTrue(o2["tx_psar_days"].iloc[-1] < 0)
        self.assertTrue((o2["tx_supertrend_gap"].iloc[20:] < 0).all())

    # ---- オシレーター
    def test_rsi_family(self):
        C = self.C
        r14 = ref_rsi(C, 14)
        self.check("tx_rsi_14", r14, 14)
        self.check("tx_rsi_9", ref_rsi(C, 9), 9)
        lo, hi = np.full(self.n, np.nan), np.full(self.n, np.nan)
        for i in range(27, self.n):
            lo[i], hi[i] = r14[i - 13:i + 1].min(), r14[i - 13:i + 1].max()
        srsi = np.where(hi > lo, 100 * (r14 - lo) / (hi - lo), 50.0)
        srsi[np.isnan(lo)] = np.nan
        self.check("tx_stoch_rsi_14", srsi, 27)
        cmo = np.full(self.n, np.nan)
        d = np.r_[np.nan, np.diff(C)]
        for i in range(14, self.n):
            w = d[i - 13:i + 1]
            u, dd = w[w > 0].sum(), -w[w < 0].sum()
            cmo[i] = 100 * (u - dd) / (u + dd) if u + dd > 0 else 0.0
        self.check("tx_cmo_14", cmo, 14)
        psy = np.full(self.n, np.nan)
        for i in range(12, self.n):
            psy[i] = 100 * (d[i - 11:i + 1] > 0).sum() / 12
        self.check("tx_psy_12", psy, 12)

    def test_connors(self):
        C = self.C
        d = np.r_[np.nan, np.diff(C)]
        streak = np.full(self.n, np.nan)
        for i in range(1, self.n):
            s = np.sign(d[i])
            prev = streak[i - 1] if i > 1 else np.nan
            if s == 0:
                streak[i] = 0.0
            elif not np.isnan(prev) and np.sign(prev) == s:
                streak[i] = prev + s
            else:
                streak[i] = s
        roc1 = np.r_[np.nan, (C[1:] / C[:-1] - 1) * 100]
        pr = np.full(self.n, np.nan)
        for i in range(101, self.n):
            pr[i] = (roc1[i - 100:i] < roc1[i]).sum()
        crsi = (ref_rsi(C, 3) + ref_rsi(streak, 2) + pr) / 3
        crsi[:101] = np.nan
        self.check("tx_connors_rsi", crsi, 101)

    def test_stochastics_williams_cci(self):
        H, L, C = self.H, self.L, self.C
        fk = np.full(self.n, np.nan)
        wr = np.full(self.n, np.nan)
        for i in range(13, self.n):
            hh, ll = H[i - 13:i + 1].max(), L[i - 13:i + 1].min()
            fk[i] = 100 * (C[i] - ll) / (hh - ll) if hh > ll else 50.0
            wr[i] = -100 * (hh - C[i]) / (hh - ll) if hh > ll else -50.0
        sk = sma(fk, 3)
        self.check("tx_stoch_k", sk, 15)
        self.check("tx_stoch_d", sma(sk, 3), 17)
        self.check("tx_williams_r_14", wr, 13)
        tp = (H + L + C) / 3
        cci = np.full(self.n, np.nan)
        for i in range(19, self.n):
            w = tp[i - 19:i + 1]
            mad = np.abs(w - w.mean()).mean()
            cci[i] = (tp[i] - w.mean()) / (0.015 * mad) if mad > 0 else 0.0
        self.check("tx_cci_20", cci, 19)

    def test_ultimate_tsi_dpo_kst_ao(self):
        H, L, C = self.H, self.L, self.C
        bp, tr = np.empty(self.n), np.empty(self.n)
        for i in range(self.n):
            lo = L[i] if i == 0 else min(L[i], C[i - 1])
            hi = H[i] if i == 0 else max(H[i], C[i - 1])
            bp[i], tr[i] = C[i] - lo, hi - lo
        uo = np.full(self.n, np.nan)
        for i in range(27, self.n):
            a = [bp[i - k + 1:i + 1].sum() / tr[i - k + 1:i + 1].sum() for k in (7, 14, 28)]
            uo[i] = 100 * (4 * a[0] + 2 * a[1] + a[2]) / 7
        self.check("tx_ultimate_osc", uo, 27)
        d = np.r_[np.nan, np.diff(C)]
        tsi = 100 * ema(ema(d, span(25)), span(13)) / ema(ema(np.abs(d), span(25)), span(13))
        tsi[:38] = np.nan
        self.check("tx_tsi", tsi, 38)
        dpo = np.full(self.n, np.nan)
        for i in range(19, self.n):
            dpo[i] = (C[i - 11] - C[i - 19:i + 1].mean()) / C[i] * 100
        self.check("tx_dpo_20", dpo, 19)

        def roc(k):
            return np.r_[np.full(k, np.nan), (C[k:] / C[:-k] - 1) * 100]

        kst = sma(roc(10), 10) + 2 * sma(roc(15), 10) + 3 * sma(roc(20), 10) + 4 * sma(roc(30), 15)
        self.check("tx_kst", kst, 44)
        mp = (H + L) / 2
        self.check("tx_awesome_osc", (sma(mp, 5) - sma(mp, 34)) / C * 100, 33)

    def test_rci(self):
        from scipy.stats import spearmanr

        C = self.C
        rci = np.full(self.n, np.nan)
        for i in range(8, self.n):
            rci[i] = 100 * spearmanr(np.arange(9), C[i - 8:i + 1]).correlation
        self.check("tx_rci_9", rci, 8)
        # 上がり続ければ 100、下がり続ければ −100
        n = 30
        c = np.arange(1.0, n + 1)
        bars = pd.DataFrame({"Code": "1", "Date": pd.bdate_range("2024-01-01", periods=n), "open": c, "high": c,
                             "low": c, "close": c, "vol": 1.0})
        self.assertAlmostEqual(TX.compute(bars)["tx_rci_26"].iloc[-1], 100.0)
        self.assertAlmostEqual(TX.compute(bars.assign(close=c[::-1], open=c[::-1], high=c[::-1], low=c[::-1]))
                               ["tx_rci_26"].iloc[-1], -100.0)

    def test_fisher_elder_bop(self):
        H, L, C, O = self.H, self.L, self.C, self.O
        mp = (H + L) / 2
        fish = np.full(self.n, np.nan)
        v = f = 0.0
        for i in range(9, self.n):
            hh, ll = mp[i - 9:i + 1].max(), mp[i - 9:i + 1].min()
            raw = 2 * ((mp[i] - ll) / (hh - ll) - 0.5) if hh > ll else 0.0
            v = 0.33 * raw + 0.67 * v
            v = 0.999 if v > 0.99 else (-0.999 if v < -0.99 else v)      # Ehlers: 0.99 を超えたら 0.999 に置き換える
            f = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * f
            fish[i] = f
        self.check("tx_fisher_10", fish, 9)
        e13 = ema(C, span(13))
        bull = (H - e13) / C * 100
        bull[:12] = np.nan
        self.check("tx_elder_bull", bull, 12)
        rng = H - L
        bop = sma(np.where(rng > 0, (C - O) / np.where(rng > 0, rng, 1), 0.0), 14)
        self.check("tx_bop_14", bop, 13)

    # ---- ボラティリティ
    def test_volatility(self):
        H, L, C, O = self.H, self.L, self.C, self.O
        tr = ref_true_range(H, L, C)
        a14, a50 = ema(tr, 1 / 14), ema(tr, 1 / 50)
        natr = a14 / C * 100
        natr[:13] = np.nan
        self.check("tx_natr_14", natr, 13)
        ratio = a14 / a50
        ratio[:49] = np.nan
        self.check("tx_atr_ratio_14_50", ratio, 49)
        pctb, width = np.full(self.n, np.nan), np.full(self.n, np.nan)
        for i in range(19, self.n):
            w = C[i - 19:i + 1]
            m, sd = w.mean(), w.std(ddof=0)
            pctb[i] = 100 * (C[i] - (m - 2 * sd)) / (4 * sd) if sd > 0 else 50.0
            width[i] = 4 * sd / m * 100
        self.check("tx_bb_pctb", pctb, 19)
        self.check("tx_bb_width", width, 19)
        rank = np.full(self.n, np.nan)
        for i in range(138, self.n):
            w = width[i - 119:i + 1]
            rank[i] = 100 * (np.sum(w < w[-1]) + (np.sum(w == w[-1]) + 1) / 2) / 120
        self.check("tx_bb_width_rank_120", rank, 138)
        lhl = np.log(H / L)
        park = np.sqrt(sma(lhl ** 2, 20) / (4 * np.log(2))) * 100
        self.check("tx_vol_parkinson_20", park, 19)
        gk = np.sqrt(np.clip(sma(0.5 * lhl ** 2 - (2 * np.log(2) - 1) * np.log(C / O) ** 2, 20), 0, None)) * 100
        self.check("tx_vol_gk_20", gk, 19)
        ui = np.full(self.n, np.nan)
        dd = np.full(self.n, np.nan)
        for i in range(13, self.n):
            mx = C[i - 13:i + 1].max()
            dd[i] = (C[i] - mx) / mx * 100
        for i in range(26, self.n):
            ui[i] = np.sqrt((dd[i - 13:i + 1] ** 2).mean())
        self.check("tx_ulcer_14", ui, 26)
        chop = np.full(self.n, np.nan)
        for i in range(14, self.n):
            rng = H[i - 13:i + 1].max() - L[i - 13:i + 1].min()
            chop[i] = 100 * np.log10(tr[i - 13:i + 1].sum() / rng) / np.log10(14)
        self.check("tx_chop_14", chop, 14)
        e9 = ema(H - L, span(9))
        mass = np.full(self.n, np.nan)
        ratio9 = e9 / ema(e9, span(9))
        for i in range(41, self.n):
            mass[i] = ratio9[i - 24:i + 1].sum()
        self.check("tx_mass_index_25", mass, 41)

    # ---- 出来高系
    def test_volume(self):
        H, L, C, V = self.H, self.L, self.C, self.V
        d = np.r_[np.nan, np.diff(C)]
        obv20 = np.full(self.n, np.nan)
        cmf = np.full(self.n, np.nan)
        upshare = np.full(self.n, np.nan)
        mfm = np.where(H > L, ((C - L) - (H - C)) / np.where(H > L, H - L, 1), 0.0)
        for i in range(20, self.n):
            w = slice(i - 19, i + 1)
            obv20[i] = 100 * (np.sign(d[w]) * V[w]).sum() / V[w].sum()
            upshare[i] = 100 * V[w][d[w] > 0].sum() / V[w].sum()
        for i in range(19, self.n):
            w = slice(i - 19, i + 1)
            cmf[i] = (mfm[w] * V[w]).sum() / V[w].sum()
        self.check("tx_obv_chg_20", obv20, 20)
        self.check("tx_up_vol_share_20", upshare, 20)
        self.check("tx_cmf_20", cmf, 19)
        tp = (H + L + C) / 3
        mfi = np.full(self.n, np.nan)
        for i in range(14, self.n):
            pos_ = sum(tp[j] * V[j] for j in range(i - 13, i + 1) if tp[j] > tp[j - 1])
            neg_ = sum(tp[j] * V[j] for j in range(i - 13, i + 1) if tp[j] < tp[j - 1])
            mfi[i] = 100 * pos_ / (pos_ + neg_) if pos_ + neg_ > 0 else 50.0
        self.check("tx_mfi_14", mfi, 14)
        vw = np.full(self.n, np.nan)
        for i in range(19, self.n):
            w = slice(i - 19, i + 1)
            vw[i] = (C[i] / ((tp[w] * V[w]).sum() / V[w].sum()) - 1) * 100
        self.check("tx_vwap_gap_20", vw, 19)
        vr = np.full(self.n, np.nan)
        for i in range(25, self.n):
            w = slice(i - 24, i + 1)
            u = V[w][d[w] > 0].sum()
            dn = V[w][d[w] < 0].sum()
            fl = V[w][d[w] == 0].sum()
            vr[i] = min(100 * (u + fl / 2) / (dn + fl / 2), 1000.0) if dn + fl > 0 else 1000.0
        self.check("tx_vr_25", vr, 25)
        nvi = np.full(self.n, np.nan)
        for i in range(60, self.n):
            g = 1.0
            for j in range(i - 59, i + 1):
                if V[j] < V[j - 1]:
                    g *= C[j] / C[j - 1]
            nvi[i] = (g - 1) * 100
        self.check("tx_nvi_chg_60", nvi, 60)
        hlc = H + L + C
        tr_k = np.r_[np.nan, np.where(hlc[1:] > hlc[:-1], 1.0, -1.0)]
        kv = tr_k * V
        kvo = (ema(kv, span(34)) - ema(kv, span(55))) / sma(V, 20)
        kvo[:55] = np.nan
        self.check("tx_klinger", kvo, 55)
        ad = np.cumsum(mfm * V)
        self.check("tx_ad_ma_gap", 100 * (ad - sma(ad, 20)) / np.r_[np.full(19, np.nan),
                                                                     np.convolve(V, np.ones(20), "valid")], 19)

    # ---- ローソク足
    def test_candles(self):
        H, L, C, O = self.H, self.L, self.C, self.O
        rng = H - L
        body = np.where(rng > 0, (C - O) / np.where(rng > 0, rng, 1), 0.0)
        self.check("tx_body_ratio", body, 0)
        cs = np.empty(self.n)
        for i in range(self.n):
            s = np.sign(C[i] - O[i])
            cs[i] = s if (i == 0 or s == 0 or np.sign(cs[i - 1]) != s) else cs[i - 1] + s
        self.check("tx_candle_streak", cs, 0)
        eng = np.full(self.n, np.nan)
        for i in range(1, self.n):
            bull = C[i - 1] < O[i - 1] and C[i] > O[i] and O[i] < C[i - 1] and C[i] > O[i - 1]
            bear = C[i - 1] > O[i - 1] and C[i] < O[i] and O[i] > C[i - 1] and C[i] < O[i - 1]
            eng[i] = 1.0 if bull else (-1.0 if bear else 0.0)
        self.check("tx_engulfing", eng, 1)
        nr7 = np.full(self.n, np.nan)
        for i in range(6, self.n):
            nr7[i] = 1.0 if rng[i] <= rng[i - 6:i + 1].min() else 0.0
        self.check("tx_nr7", nr7, 6)
        three = np.full(self.n, np.nan)
        for i in range(3, self.n):
            ok = all(C[j] > O[j] and C[j] > C[j - 1] and H[j] > H[j - 1] and L[j] > L[j - 1] for j in (i - 2, i - 1, i))
            three[i] = 1.0 if ok else 0.0
        self.check("tx_three_soldiers", three, 3)
        gaps = np.full(self.n, np.nan)
        for i in range(20, self.n):
            gaps[i] = sum(1 for j in range(i - 19, i + 1) if L[j] > H[j - 1])
        self.check("tx_gap_up_count_20", gaps, 20)


class TestRest(unittest.TestCase):
    """TestDefinitions で見ていない残りの列（同じ1銘柄で、素朴な計算と一致するか）。"""

    setUpClass = classmethod(TestDefinitions.setUpClass.__func__)
    check = TestDefinitions.check

    def test_rest_trend_osc(self):
        H, L, C = self.H, self.L, self.C
        macd = ema(C, span(12)) - ema(C, span(26))
        sig = ema(macd, span(9)) / C * 100
        sig[:34] = np.nan
        self.check("tx_macd_signal", sig, 34)
        kj = np.array([(H[i - 25:i + 1].max() + L[i - 25:i + 1].min()) / 2 if i >= 25 else np.nan for i in range(self.n)])
        self.check("tx_ichi_kijun_gap", (C / kj - 1) * 100, 25)
        tr = ref_true_range(H, L, C)
        vn = np.full(self.n, np.nan)
        for i in range(14, self.n):
            vn[i] = sum(abs(L[j] - H[j - 1]) for j in range(i - 13, i + 1)) / tr[i - 13:i + 1].sum()
        self.check("tx_vortex_neg_14", vn, 14)
        for k in (10, 60, 120, 250):
            self.check(f"tx_roc_{k}", np.r_[np.full(k, np.nan), (C[k:] / C[:-k] - 1) * 100], k)
        bear = (L - ema(C, span(13))) / C * 100
        bear[:12] = np.nan
        self.check("tx_elder_bear", bear, 12)
        y = np.log(C)
        sl = np.full(self.n, np.nan)
        for i in range(19, self.n):
            sl[i] = np.polyfit(np.arange(20.0), y[i - 19:i + 1], 1)[0] * 100
        self.check("tx_linreg_slope_20", sl, 19)

    def test_rest_volatility(self):
        H, L, C = self.H, self.L, self.C
        tr = ref_true_range(H, L, C)
        a10, a14, a20 = ema(tr, 1 / 10), ema(tr, 1 / 14), ema(tr, 1 / 20)
        mid = ema(C, span(20))
        kp = 100 * (C - (mid - 2 * a10)) / (4 * a10)
        kp[:19] = np.nan
        self.check("tx_keltner_pos", kp, 19)
        sq = np.full(self.n, np.nan)
        for i in range(19, self.n):
            sq[i] = 1.0 if 2 * C[i - 19:i + 1].std(ddof=0) < 1.5 * a20[i] else 0.0
        self.check("tx_squeeze", sq, 19)
        r = np.r_[np.nan, (C[1:] / C[:-1] - 1) * 100]
        v60 = np.full(self.n, np.nan)
        for i in range(60, self.n):
            v60[i] = r[i - 59:i + 1].std(ddof=1)
        self.check("tx_vol_60", v60, 60)
        e = ema(H - L, span(10))
        cv = np.r_[np.full(10, np.nan), (e[10:] / e[:-10] - 1) * 100]
        cv[:19] = np.nan
        self.check("tx_chaikin_vol_10", cv, 19)
        ra = (H - L) / a14
        ra[:13] = np.nan
        self.check("tx_range_atr", ra, 13)

    def test_rest_volume(self):
        H, L, C, V = self.H, self.L, self.C, self.V
        d = np.r_[np.nan, np.diff(C)]
        v20 = np.r_[np.full(19, np.nan), np.convolve(V, np.ones(20), "valid")]
        obv = np.cumsum(np.where(np.isnan(d), 0.0, np.sign(d)) * V)
        self.check("tx_obv_ma_gap", 100 * (obv - sma(obv, 20)) / v20, 19)
        rng = H - L
        mfm = np.where(rng > 0, ((C - L) - (H - C)) / np.where(rng > 0, rng, 1), 0.0)
        ad = np.cumsum(mfm * V)
        co = (ema(ad, span(3)) - ema(ad, span(10))) / sma(V, 10)
        self.check("tx_chaikin_osc", co, 9)
        fi = 100 * ema(d * V, span(13)) / (C * sma(V, 20))
        self.check("tx_force_13", fi, 19)
        mp = (H + L) / 2
        dmid = np.r_[np.nan, np.diff(mp)] / C * 100
        rel = V / sma(V, 20)
        x = np.where(rel > 0, dmid * (rng / C * 100) / np.where(rel > 0, rel, 1), 0.0)
        x[np.isnan(rel) | np.isnan(dmid)] = np.nan
        eom = sma(x, 14)
        eom[:33] = np.nan
        self.check("tx_eom_14", eom, 33)
        r = np.r_[0.0, (C[1:] / C[:-1] - 1) * 100]
        pvt = np.full(self.n, np.nan)
        for i in range(20, self.n):
            pvt[i] = (r[i - 19:i + 1] * V[i - 19:i + 1]).sum() / V[i - 19:i + 1].sum()
        self.check("tx_pvt_chg_20", pvt, 20)
        self.check("tx_vol_osc_5_20", (sma(V, 5) / sma(V, 20) - 1) * 100, 19)
        self.check("tx_vol_ratio_60_250", sma(V, 60) / sma(V, 250), 249)
        tv = C * V                                          # trading_value の無い日足は終値 × 出来高
        self.check("tx_tv_ratio_20_120", sma(tv, 20) / sma(tv, 120), 119)
        tpz = (H + L + C) / 3
        vw60 = np.full(self.n, np.nan)
        for i in range(59, self.n):
            w = slice(i - 59, i + 1)
            vw60[i] = (C[i] / ((tpz[w] * V[w]).sum() / V[w].sum()) - 1) * 100
        self.check("tx_vwap_gap_60", vw60, 59)

    def test_rest_candles(self):
        H, L, C, O = self.H, self.L, self.C, self.O
        rng = H - L
        safe = np.where(rng > 0, rng, 1)
        self.check("tx_upper_shadow", np.where(rng > 0, (H - np.maximum(O, C)) / safe, 0.0), 0)
        self.check("tx_lower_shadow", np.where(rng > 0, (np.minimum(O, C) - L) / safe, 0.0), 0)
        self.check("tx_bull_share_20", 100 * sma((C > O).astype(float), 20), 19)
        self.check("tx_body_mean_20", sma(np.where(rng > 0, np.abs(C - O) / safe, 0.0), 20), 19)


class TestProperties(unittest.TestCase):
    def bars(self, c):
        n = len(c)
        return pd.DataFrame({"Code": "1", "Date": pd.bdate_range("2024-01-01", periods=n), "open": c, "high": c,
                             "low": c, "close": c, "vol": 1000.0})

    def test_rising(self):
        o = TX.compute(self.bars(100 * 1.01 ** np.arange(80.0))).iloc[-1]
        self.assertAlmostEqual(o["tx_rsi_14"], 100.0)
        self.assertAlmostEqual(o["tx_stoch_k"], 100.0)
        self.assertAlmostEqual(o["tx_williams_r_14"], 0.0)
        self.assertAlmostEqual(o["tx_aroon_up_25"], 100.0)
        self.assertAlmostEqual(o["tx_aroon_down_25"], 0.0)
        self.assertAlmostEqual(o["tx_psy_12"], 100.0)
        self.assertAlmostEqual(o["tx_linreg_r2_60"], 1.0)
        self.assertAlmostEqual(o["tx_kaufman_er_20"], 1.0)
        self.assertEqual(o["tx_close_streak"], 79)

    def test_flat(self):
        o = TX.compute(self.bars(np.full(80, 500.0))).iloc[-1]
        self.assertAlmostEqual(o["tx_rsi_14"], 50.0)
        self.assertAlmostEqual(o["tx_bb_pctb"], 50.0)
        self.assertAlmostEqual(o["tx_cci_20"], 0.0)
        self.assertAlmostEqual(o["tx_ma_gap_25"], 0.0)
        self.assertAlmostEqual(o["tx_cmo_14"], 0.0)
        self.assertEqual(o["tx_close_streak"], 0)


if __name__ == "__main__":
    unittest.main()
