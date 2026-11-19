#!/usr/bin/env python3
"""
実験132（research/exp/e132_minute_vp_ab.py）の部品。データには触らない（作りものの表で確かめる）。

- 列: 比べる土台は回した時点の既定 373列（名前で固定）。表には research/minute_features.py の33列を付け、学習に足すのは
  VWAP の3列を除いた30列（運用者の選択「VWAPの3列を外した30列」）。土台と重ならない
- 腕は A・B・C（運用者の選択「AとBだけ」の後、結果を見て「C の腕で列の効果を切り分ける」）。A は必ず含める。C は A の
  パラメータのまま30列を足す（探索しない）。訓練に分足の無い窓では C は A とちょうど同じ
- --keep-frame: 保存済みの表を、分足の取得記録が変わっていてもそのまま使う（前の回の A・B と同じ表で C を比べる）
- --short: 短い期間の値動き（gap・r1・r3・r5。日足の並びで数える）。日付の中の順位相関（日付を1件として t）。今の列の予測を
  引いた偏相関で、予測と同じ情報の列は消え、別の情報の列は残る。30列まとめた信号は日付の前半・後半で作って測る
- --residual: A の out-of-fold で説明しきれない部分を列が説明するか。A が使っていない情報を持つ列は見つかり、雑音の列・
  A が使い切っている列は見つからない（並べ替えは同じ日付の中だけ）
- --importance: 保存済みの探索のパラメータで A・C・B を同じ行・種で学習し直し、gain の割合の上位の顔ぶれを比べる
  （分足の30列の割合と順位・上位 N 列の重なりと種だけ変えたときの重なり・順位相関・上位の表）。探索し直さない
- ラベル: データセットの label が本番の +10% の定義（label_specs）と全行一致しなければ止める
- 付けた33列の確かめ: 割合が 0〜1 の外・無限・終値との差が −100% 以下・期間の中で欠けのある行・時間帯の割合の合計が1でない行があれば止める
- 評価の行: 全行と、分足の付いた行だけ（30列のどれかがある行）。窓は行が100件以上で正例・負例がそろうもの
- 表に付いている VWAP の3列は、B の学習・寄与・付き方の報告に入らない
- 訓練に分足の付いた行が無い窓では30列が訓練で全行欠測で、A と B のパラメータが同じなら B = A。「訓練に分足の付いた行がある
  窓だけ」と「無い窓だけ」の集計に分ける
- 報告の一通り: 保存済みの探索（A・B）から、out-of-fold・寄与・報告が最後まで通る（種1つ・ずらし0 と 2）。§7 の判定は出さない

  python3 -m unittest tests.test_e132_minute_vp_ab
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e132_minute_vp_ab as E  # noqa: E402
import features as F  # noqa: E402
import label_specs as LS  # noqa: E402
import minute_features as MF  # noqa: E402
import volume_profile as VP  # noqa: E402


class TestColumns(unittest.TestCase):
    def test_sets(self):
        base = F.columns(E.BASE_PRESET)
        self.assertEqual(len(base), 373)
        self.assertEqual(E.BASE_PRESET, F.DEFAULT_PRESET)              # 回す時点の既定（変わったら見直す）
        self.assertEqual(list(E.BUILT_COLS), MF.COLUMNS)
        self.assertEqual(E.DROPPED_COLS, ("vp_minute_vwap_gap_1d", "vp_minute_vwap_gap_5d", "vp_minute_vwap_gap_20d"))
        self.assertEqual(list(E.NEW_COLS), [c for c in MF.COLUMNS if "_vwap_gap_" not in c])   # 並びは minute_features のまま
        self.assertEqual(len(E.NEW_COLS), 30)
        self.assertFalse(set(base) & set(E.BUILT_COLS))
        self.assertEqual(E.ARMS, ("A", "B", "C"))
        for d in (E.COLS_OF, E.LABELS, E.PARAMS_OF):
            self.assertEqual(set(d), set(E.ARMS))
        self.assertEqual(E.PARAMS_OF, {"A": "A", "B": "B", "C": "A"})
        self.assertEqual(E.COLS_OF["C"], E.COLS_OF["B"])
        self.assertEqual(E.SEEDS, (42, 7, 123))
        self.assertEqual(E.ROWSETS, ("all", "minute"))

    def test_args(self):
        old = (E.RUN, E.TAG, E.DEADLINE)
        try:
            with mock.patch.object(E, "prepare", side_effect=RuntimeError("stop")):
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    E.main(["--tag", "e132t"])
            self.assertEqual(E.RUN, "e132t")
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.main(["--arms", "B"])                                  # 基準 A は外せない
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.main(["--arms", "A,D"])                                # 腕は A・B・C だけ
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.main(["--rebuild", "--keep-frame"])                    # 一緒には使えない
        finally:
            E.RUN, E.TAG, E.DEADLINE = old


def label_frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"future_rise": rng.normal(0.08, 0.1, n), "end_level": rng.normal(0.05, 0.08, n),
                       "uptrend_end": (rng.random(n) < 0.7).astype(float), "entry_price": 100.0})
    df["label"] = LS.labels(df, "lgbm")
    df.loc[[2, 9], ["future_rise", "label"]] = np.nan                  # 未確定の行
    return df


class TestLabel(unittest.TestCase):
    def test_same_and_stops(self):
        df = label_frame()
        with redirect_stdout(io.StringIO()):
            out = E.check_label(df)
        self.assertEqual(out["same"], out["rows"])
        bad = df.copy()
        bad.loc[int(np.flatnonzero(df["label"].to_numpy() == 0)[0]), "label"] = 1.0
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.check_label(bad)


def minute_frame(n=60, start_attached=10):
    """作りものの33列。最初の start_attached 行は分足の前（欠測）。"""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"Date": pd.bdate_range("2024-10-15", periods=n)})
    for w in MF.WINDOWS:
        for m in MF.PROFILE_METRICS:
            df[MF.column(m, w)] = rng.uniform(0, 0.9, n) if m.endswith("share") else rng.normal(0, 0.05, n)
        parts = rng.dirichlet(np.ones(4), n)
        for j, b in enumerate(VP.BUCKETS):
            df[MF.column(f"{b}_share", w)] = parts[:, j]
    df.loc[:start_attached - 1, list(MF.COLUMNS)] = np.nan
    return df


REPORT = {"minute_from": "2024-10-29", "minute_to": "2025-01-06", "minute_days": 50, "status": {"1d/ok": 50}}


class TestMinuteChecks(unittest.TestCase):
    def test_ok_and_stops(self):
        df = minute_frame()
        with redirect_stdout(io.StringIO()):
            out = E.minute_checks(df, REPORT)
        self.assertEqual(out["bad"], {})
        self.assertEqual(sum(out["torn"].values()), 0)
        self.assertAlmostEqual(out["attached_share_since_minute"], 1.0)
        for col, val in ((MF.column("poc_share", 5), 1.2), (MF.column("vwap_gap", 1), -1.0),
                         (MF.column("close_share", 20), np.inf)):
            bad = df.copy()
            bad.loc[20, col] = val
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.minute_checks(bad, REPORT)
        torn = df.copy()
        torn.loc[20, MF.column("poc_gap", 5)] = np.nan                   # 期間の中で1指標だけ欠ける
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.minute_checks(torn, REPORT)
        off = df.copy()
        off.loc[20, MF.column("open_share", 1)] += 0.1                   # 時間帯の割合の合計が1でない
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.minute_checks(off, REPORT)


def residual_frame(seed=0, n_dates=120, per_day=30):
    """A の予測は signal だけを使う。ラベルは signal と info で決まる。noise は無関係。"""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2025-01-06", periods=n_dates), per_day)
    n = len(dates)
    signal, info, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + signal + 0.8 * info)))).astype(float)
    score = 1 / (1 + np.exp(-(-1.5 + signal)))
    return pd.DataFrame({"Code": [f"{1000 + i % per_day}" for i in range(n)], "Date": dates, "label": y,
                         "score": score, "n_scores": 9, "f_info": info, "f_noise": noise, "f_in_a": signal})


class TestResidual(unittest.TestCase):
    def test_finds_only_information_that_a_does_not_use(self):
        rows = residual_frame()
        with redirect_stdout(io.StringIO()) as buf:
            out = E.residual_report(rows, ["f_info", "f_noise", "f_in_a"], n_perm=199, n_perm_joint=99, seed=1)
        cols = out["columns"]
        self.assertLess(cols["f_info"]["pooled"]["p"], 0.01)
        self.assertLess(cols["f_info"]["pooled"]["q"], E.BH_Q)
        self.assertGreater(cols["f_info"]["pooled"]["coef"], 0)
        self.assertLess(cols["f_info"]["within"]["p"], 0.01)
        self.assertGreater(cols["f_noise"]["pooled"]["p"], 0.05)
        self.assertGreater(cols["f_in_a"]["pooled"]["p"], 0.05)                # A が使い切っている
        self.assertEqual(out["hits"]["pooled"], ["f_info"])
        q = cols["f_info"]["quintile_resid_pt"]
        self.assertLess(q[0], 0)
        self.assertGreater(q[-1], 0)                                         # 列が高いほど A の見込みより正例が多い
        self.assertLess(out["joint"]["p"], 0.05)
        self.assertTrue(all(v["auc_full"] > v["auc_a"] for v in out["joint"]["split"].values()))
        self.assertTrue(out["joint"]["reading"].startswith("情報が残っている"))
        self.assertIn("残差で確かめる", buf.getvalue())

    def test_nothing_left_when_columns_carry_no_new_information(self):
        rows = residual_frame(seed=3)
        with redirect_stdout(io.StringIO()):
            out = E.residual_report(rows, ["f_noise", "f_in_a"], n_perm=199, n_perm_joint=99, seed=2)
        self.assertGreater(out["joint"]["p"], 0.05)
        self.assertTrue(out["joint"]["reading"].startswith("残っていない"))
        self.assertEqual(out["hits"]["pooled"], [])

    def test_parts(self):
        np.testing.assert_allclose(E.bh_qvalues([0.01, 0.04, 0.03, 0.5]), [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])
        codes = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
        perm = E.date_permuter(codes, np.random.default_rng(0))
        for _ in range(20):
            idx = perm()
            self.assertEqual(sorted(idx), list(range(len(codes))))
            np.testing.assert_array_equal(codes[idx], codes)                  # 同じ日付の中だけで入れ替わる
        v = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 5.0])
        np.testing.assert_allclose(E.within_date(v, np.array([0, 0, 0, 1, 1, 2])), [-1, 0, 1, -5, 5, 0])
        X = np.column_stack([np.ones(200), np.linspace(-2, 2, 200)])
        y = (np.linspace(-2, 2, 200) + np.random.default_rng(1).normal(0, 1, 200) > 0).astype(float)
        b, ll = E.logit_fit(X, y)
        from sklearn.linear_model import LogisticRegression
        sk = LogisticRegression(C=1e9, max_iter=1000).fit(X[:, 1:], y)
        self.assertAlmostEqual(b[1], sk.coef_[0][0], places=3)


def short_frame(seed=0, n_dates=160, per_day=12):
    """r1 は今の列の予測（pred）と f_new の両方で決まる。f_same は pred と同じ情報。f_noise は無関係。"""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2025-01-06", periods=n_dates), per_day)
    n = len(dates)
    sig, new, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    r1 = 0.01 * (sig + 0.6 * new) + 0.02 * rng.normal(size=n)
    r3 = 0.03 * rng.normal(size=n)
    return pd.DataFrame({"Code": [f"{1000 + i % per_day}" for i in range(n)], "Date": dates, "r1": r1, "r3": r3,
                         "pred_r1": sig, "pred_r3": rng.normal(size=n), "f_new": new, "f_same": sig + 0.01 * noise,
                         "f_noise": noise})


class TestShort(unittest.TestCase):
    TARGETS = (("r1", "1日目"), ("r3", "3日目まで"))

    def test_raw_partial_and_joint(self):
        rows = short_frame()
        with redirect_stdout(io.StringIO()) as buf:
            out = E.short_report(rows, ["f_new", "f_same", "f_noise"], targets=self.TARGETS,
                                 pred_cols={"r1": "pred_r1", "r3": "pred_r3"})
        cols = out["columns"]
        self.assertGreater(cols["f_new"]["raw"]["r1"]["t"], 4)
        self.assertLess(cols["f_new"]["raw"]["r1"]["q"], E.BH_Q)
        self.assertGreater(cols["f_same"]["raw"]["r1"]["t"], 4)                 # 単独では効いて見える
        self.assertGreater(cols["f_new"]["partial"]["r1"]["t"], 4)              # 今の列を超えて残る
        self.assertLess(abs(cols["f_same"]["partial"]["r1"]["t"]), 3)           # 予測と同じ情報は消える
        self.assertGreater(cols["f_noise"]["raw"]["r1"]["q"], E.BH_Q)
        self.assertIn("r1", out["reading"]["works"])
        self.assertIn("r1", out["reading"]["beyond"])
        self.assertNotIn("r3", out["reading"]["works"])
        for r in out["joint"]["r1"].values():
            self.assertGreater(r["added"]["diff"], 0)
        self.assertIn("短い期間の値動き", buf.getvalue())

    def test_date_ic_counts_dates(self):
        rows = short_frame(seed=1, n_dates=40, per_day=6)
        codes = pd.factorize(rows["Date"])[0]
        r = E.date_ic(rows["f_noise"].to_numpy(), rows["r3"].to_numpy(), codes)
        self.assertEqual(r["dates"], 40)
        self.assertLess(abs(r["t"]), 3)
        few = E.date_ic(rows["f_noise"].to_numpy(), rows["r3"].to_numpy(), codes, min_rows=7)
        self.assertEqual(few["dates"], 0)                                     # 1日6行なので数えない
        self.assertTrue(np.isnan(few["ic"]))

    def test_short_rows_wiring(self):
        df = make_frame(6, per_day=2)
        df["entry_gap"] = np.nan
        tmp = tempfile.mkdtemp()
        try:
            rng = np.random.default_rng(0)
            days = pd.bdate_range("2016-12-01", "2023-01-31")
            bars = []
            for code in sorted(df["Code"].unique()):
                px = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
                bars.append(pd.DataFrame({"Date": days, "Code": code, "AdjO": px * (1 + rng.normal(0, 0.005, len(days))),
                                          "AdjH": px * 1.02, "AdjC": px}))
            b = pd.concat(bars, ignore_index=True)
            for y in sorted(b["Date"].dt.year.unique()):
                b[b["Date"].dt.year == y].to_parquet(os.path.join(tmp, f"bars_{y}.parquet"), index=False)
            with mock.patch.object(E.lab, "DATA_DIR", tmp), redirect_stdout(io.StringIO()):
                rows, pred_cols, check = E.short_rows(df, ["b0", "b1", "b2", "b3"], PARAMS)
        finally:
            shutil.rmtree(tmp)
        self.assertTrue(E.attached_mask(rows).all())
        self.assertEqual(set(pred_cols), {k for k, _ in E.SHORT_TARGETS})
        first_test = pd.Timestamp(E.E41.folds_for(df["Date"], 0)[0].test_start)
        late = rows["Date"] >= max(first_test, pd.Timestamp("2019-01-01"))
        for key in pred_cols:
            self.assertTrue(rows.loc[late, f"pred_{key}"].notna().all())
            self.assertTrue(rows[key].notna().mean() > 0.95)
        self.assertEqual(check["rows"], 0)                                   # entry_gap が無い作りもの

    def test_short_returns_from_bars(self):
        tmp = tempfile.mkdtemp()
        try:
            days = pd.bdate_range("2025-03-03", periods=10)
            rows = []
            for code, base_px in (("11110", 100.0), ("22220", 50.0)):
                for i, d in enumerate(days):
                    if code == "22220" and i == 3:
                        continue                                              # 売買の無い日（日足が無い）
                    rows.append({"Date": d, "Code": code, "AdjO": base_px + i, "AdjH": base_px + i + 2,
                                 "AdjC": base_px + i + 1})
            pd.DataFrame(rows).to_parquet(os.path.join(tmp, "bars_2025.parquet"), index=False)
            keys = pd.DataFrame({"Code": ["11110", "22220"], "Date": [days[1], days[1]]})
            got = E.short_returns(keys, tmp)
        finally:
            shutil.rmtree(tmp)
        a = got.iloc[0]
        self.assertAlmostEqual(a["gap"], 102 / 102 - 1)                       # D の終値 102 → D+1 の寄り 102
        self.assertAlmostEqual(a["r1"], 103 / 102 - 1)
        self.assertAlmostEqual(a["r3"], 105 / 102 - 1)
        self.assertAlmostEqual(a["r5"], 107 / 102 - 1)
        b = got.iloc[1]                                                       # 3日目が無いので日足の並びで数える
        self.assertAlmostEqual(b["gap"], 52 / 52 - 1)
        self.assertAlmostEqual(b["r1"], 53 / 52 - 1)
        self.assertAlmostEqual(b["r3"], 56 / 52 - 1)


class TestRows(unittest.TestCase):
    def test_rows_and_windows(self):
        rng = np.random.default_rng(2)
        n = 600
        o = pd.DataFrame({"Date": np.repeat(pd.bdate_range("2024-01-01", periods=n // 3), 3),
                          "fold": np.repeat([0, 1, 2], n // 3), "label": (rng.random(n) < 0.3).astype(float),
                          "score": rng.random(n), "minute": [True] * 250 + [np.nan] * 50 + [False] * 300})
        self.assertEqual(len(E.rows_of(o, "all")), n)
        self.assertEqual(len(E.rows_of(o, "minute")), 250)              # 欠測は付いていない側
        w = E.windows_on(o, "minute")
        self.assertEqual(w["fold"].tolist(), [0])
        self.assertEqual(len(E.windows_on(o, "all")), 3)


PARAMS = {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 20, "verbose": -1, "n_jobs": 1}
PARAMS_B = {**PARAMS, "num_leaves": 5, "learning_rate": 0.08}           # B は探索し直しで別のパラメータになる


def make_frame(seed=0, per_day=3, minute_from="2019-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", "2022-12-30")
    n = len(dates) * per_day
    cols = {"Code": [f"{1000 + i % 50}" for i in range(n)], "Date": np.repeat(dates, per_day)}
    cols.update({f"b{j}": rng.normal(size=n) for j in range(4)})
    cols.update({c: rng.normal(size=n) for c in E.BUILT_COLS})          # 表には33列が付いている
    df = pd.DataFrame(cols)
    before = df["Date"] < minute_from
    df.loc[before, list(E.BUILT_COLS)] = np.nan                         # 分足は途中の年から
    df["label"] = (df["b1"] + 0.5 * df[MF.column("poc_gap", 20)].fillna(0)
                   + 0.8 * df[MF.column("vwap_gap", 20)].fillna(0) + rng.normal(0, 1.0, n) > 1.0).astype(float)
    df["ret_o1_20"] = rng.normal(0.01, 0.1, n)
    df["ret_o1_40"] = rng.normal(0.01, 0.15, n)
    df["hit10"] = (rng.random(n) < 0.3).astype(float)
    df["tp10c"] = np.where(df["hit10"] > 0, 10.0, rng.normal(-1, 8, n))
    df["Date"] = df["Date"].astype("datetime64[ns]")
    return df


class TestRunAndReport(unittest.TestCase):
    def test_end_to_end(self):
        df = make_frame(1, minute_from="2020-03-02")                    # 最初の窓（訓練の終わり 2020-01-02）は訓練に分足が無い
        df.attrs["built_utc"] = "test"
        base = ["b0", "b1", "b2", "b3"]
        new = list(E.NEW_COLS)
        colsets = {"base": base, "new": base + new}
        frames = {"base": df, "new": df}
        stamp = {"built_utc": "test", "rows": len(df), "date_max": "2022-12-30"}
        cutoff = pd.Timestamp("2021-12-31")
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = tmp, (42,), "e132test", None
            for arm, cols, prm in (("A", base, PARAMS), ("B", base + new, PARAMS_B)):
                rec = {"params": prm, "_cv": {"mean_pr_auc": 0.3, "std": 0.01, "mean_roc_auc": 0.6,
                                                 "fold_scores": [0.3, 0.31]},
                       "_n_trials": 2, "_cutoff": str(cutoff.date()), "_data": "test", "_sig": F.signature(cols)}
                with open(E.path(f"params_{arm}.json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                extra = {"coverage": E.coverage_report(df, base)}
                res = E.run(list(E.ARMS), frames, colsets, cutoff, 2, stamp, [0, 2])
                extra["gain"] = E.gain_report(df, base, PARAMS_B, cutoff, "B")
                extra["gain_C"] = E.gain_report(df, base, res["params"]["C"], cutoff, "C")
                summary = E.report(res, [0, 2], stamp, extra)
            out = buf.getvalue()
            names = sorted(os.listdir(tmp))
            tmp2 = tempfile.mkdtemp()
            shutil.copytree(tmp, tmp2, dirs_exist_ok=True)
            self.addCleanup(shutil.rmtree, tmp2)
        finally:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = old
            shutil.rmtree(tmp)
        self.assertEqual(E.complete_arms(res, [0, 2]), list(E.ARMS))
        self.assertEqual(set(summary["top"]), set(E.ARMS))
        self.assertIn("all_pr_B-A", summary["windows"])
        self.assertIn("minute_pr_B-A", summary["windows"])
        self.assertIn("minute_B", summary["production_oof"])
        self.assertIn("all_pr_C-A", summary["windows"])
        self.assertIn("minute_C", summary["production_oof"])
        self.assertEqual(res["params"]["C"], PARAMS)                     # C は A のパラメータ
        self.assertEqual(res["params"]["B"], PARAMS_B)
        self.assertEqual(set(extra["gain_C"]["new"]), set(E.NEW_COLS))
        self.assertIn("C−A", out)
        # 残差: A の保存済みの out-of-fold（ずらし 0・2）を行ごとに平均し、分足の付いた行だけ
        old_run = (E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = tmp2, (42,), "e132test", None
            rows = E.residual_rows(df, base, PARAMS, [0, 2])
        finally:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = old_run
        self.assertTrue(E.attached_mask(rows).all())
        self.assertTrue(rows["score"].between(0, 1).all())
        self.assertTrue(set(rows["n_scores"].unique()) <= {1, 2})
        self.assertEqual(len(rows), len(rows.drop_duplicates(["Code", "Date"])))
        self.assertNotIn("judge7", summary)
        self.assertEqual(set(extra["gain"]["share"]), {"既存の373列", "分足の30列"})
        self.assertAlmostEqual(sum(extra["gain"]["share"].values()), 1.0, places=6)
        self.assertEqual(set(extra["gain"]["new"]), set(E.NEW_COLS))     # 表にある VWAP の3列は学習に入らない
        self.assertEqual(set(extra["coverage"]["columns"]), set(E.NEW_COLS))
        self.assertNotIn("vp_minute_vwap_gap_20d↔", out)
        self.assertIn(f"lgbm_cnew{F.signature(base + new)}_", "\n".join(names))     # out-of-fold の名前に列の指紋
        b_files = {n for n in names if n.startswith(f"e132test_lgbm_cnew{F.signature(base + new)}_")}
        self.assertEqual(len(b_files), 2 * 2)                            # B と C（パラメータが違う）× ずらし 0・2（本番はずらし0と同じファイル）
        self.assertIn("§7 の判定は出さない", out)
        self.assertIn("分足の付いた行だけ", out)
        # 付いた行だけの窓は、分足が始まる前の窓を含まない
        windows = pd.DataFrame(summary["per_window"]["minute"])
        self.assertTrue((pd.to_datetime(windows["start"]) >= pd.Timestamp("2020-03-02")).all())
        # 訓練に分足が無い窓は C = A（ちょうど同じ。A のパラメータで、30列は訓練で全行欠測）。B はパラメータが違うので違う
        every = pd.DataFrame(summary["per_window"]["all"])
        none = every[every["train_minute"] == 0]
        self.assertGreater(len(none), 0)
        np.testing.assert_array_equal(none["pr_A"].to_numpy(), none["pr_C"].to_numpy())
        np.testing.assert_array_equal(none["roc_A"].to_numpy(), none["roc_C"].to_numpy())
        self.assertGreater(np.abs(none["pr_A"].to_numpy() - none["pr_B"].to_numpy()).max(), 0)
        for arm in ("B", "C"):
            trained = summary["windows"][f"all_pr_{arm}-A_trained"]
            self.assertEqual(trained["n"], int((every["train_minute"] > 0).sum()))
            self.assertEqual(trained["n"] + len(none), summary["windows"][f"all_pr_{arm}-A"]["n"])
            self.assertEqual(summary["windows"][f"all_pr_{arm}-A_untrained"]["n"], len(none))
        self.assertEqual(summary["windows"]["all_pr_C-A_untrained"]["ties"], len(none))
        # 訓練に分足がある窓では、C は A と違う（30列を使う）
        some = every[every["train_minute"] > 0]
        self.assertGreater(np.abs(some["pr_A"].to_numpy() - some["pr_C"].to_numpy()).max(), 0)
        self.assertIn("うち訓練に分足の付いた行がある窓", out)
        # 訓練の分足の行数は、窓の訓練の終わりまでの付いた行（ラベルあり）の数
        folds = {f.index: f for f in E.E41.folds_for(df["Date"], 0)}
        att = E.attached_mask(df) & df["label"].notna()
        for _, r in every[every["shift"] == 0].iterrows():
            want = int((att & (df["Date"] <= pd.Timestamp(folds[int(r["fold"])].train_end))).sum())
            self.assertEqual(int(r["train_minute"]), want)
        # A と B は列が違うので out-of-fold も違う
        self.assertGreater(np.abs(res["prod"]["A"]["score"].to_numpy() - res["prod"]["B"]["score"].to_numpy()).max(), 1e-9)

    def test_importance_report(self):
        df = make_frame(4, minute_from="2020-03-02")
        base = ["b0", "b1", "b2", "b3"] + [f"x{j}" for j in range(40)]
        rng = np.random.default_rng(9)
        for c in base[4:]:
            df[c] = rng.normal(size=len(df))
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.RUN)
        try:
            E.OOF_DIR, E.RUN = tmp, "e132imp"
            buf = io.StringIO()
            with redirect_stdout(buf):
                out = E.importance_report(df, base, {"A": PARAMS, "B": PARAMS_B}, pd.Timestamp("2021-12-31"), seeds=(42, 7))
            log = buf.getvalue()
            saved = json.loads(open(E.path("importance.json"), encoding="utf-8").read())
        finally:
            E.OOF_DIR, E.RUN = old
            shutil.rmtree(tmp)
        self.assertEqual(set(out["periods"]), {"tuning", "all"})
        self.assertEqual(saved["seeds"], [42, 7])
        for key, rec in out["periods"].items():
            self.assertEqual(set(rec["arms"]), {"C", "B"})
            c = rec["arms"]["C"]
            self.assertEqual(set(c["minute_ranks"]), set(E.NEW_COLS))      # 分足の30列だけ（VWAP の3列は入らない）
            self.assertTrue(0 <= c["minute_gain_share"] <= 1)
            self.assertEqual(c["best_rank"], min(c["minute_ranks"].values()))
            for name, vals in rec["overlap"].items():
                for n, v in vals.items():
                    self.assertTrue(0 <= float(v) <= int(n))
            self.assertTrue(-1 <= rec["spearman"]["A_C_mean"] <= 1)
            self.assertEqual(len(rec["top"]["A"]), min(max(E.TOP_NS), len(base)))
            self.assertTrue(all(t["col"] in base for t in rec["top"]["A"]))     # A は既存の列だけ
            shares = [t["share"] for t in rec["top"]["C"]]
            self.assertEqual(shares, sorted(shares, reverse=True))
        self.assertGreater(out["periods"]["all"]["minute_rows"], out["periods"]["tuning"]["minute_rows"])
        # ラベルに効く分足の列（作りものでは poc_gap_20d）は、全期間の C で上位に入る
        self.assertLessEqual(out["periods"]["all"]["arms"]["C"]["minute_ranks"][MF.column("poc_gap", 20)], 10)
        for title in ("1. 分足の30列", "2. 上位 N 列の顔ぶれの重なり", "3. 既存の", "4. 上位20列", "種どうし"):
            self.assertIn(title, log)

    def test_importance_needs_saved_tuning(self):
        df = make_frame(5)
        with mock.patch.object(E, "prepare", return_value=(df, {"built_utc": "x", "rows": len(df)})), \
             mock.patch.object(E, "check_label", return_value={}), \
             mock.patch.object(E, "coverage_report", return_value={}), \
             mock.patch.object(E.F, "columns", return_value=["b0", "b1"]), \
             mock.patch.object(E, "load_tuned", return_value=None), \
             mock.patch.object(E.tuning, "tune", side_effect=RuntimeError("searched")):
            old = (E.RUN, E.TAG, E.DEADLINE)
            try:
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                    E.main(["--importance", "--keep-frame"])
            finally:
                E.RUN, E.TAG, E.DEADLINE = old
        self.assertIn("保存済みの探索の結果が無い", str(ctx.exception))

    def test_tune_cache_needs_same_columns(self):
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.RUN, E.DEADLINE = tmp, "e132cache", None
            cutoff = pd.Timestamp("2021-12-31")
            rec = {"params": PARAMS, "_cv": {}, "_n_trials": 2, "_cutoff": str(cutoff.date()), "_data": "test",
                   "_sig": F.signature(["b0"])}
            with open(E.path("params_A.json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            got = E.tune("A", make_frame(2), ["b0"], cutoff, 2, {"built_utc": "test"})
            self.assertEqual(got["params"], PARAMS)
            with mock.patch.object(E.tuning, "tune", side_effect=RuntimeError("searched")):
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    E.tune("A", make_frame(2), ["b0", "b1"], cutoff, 2, {"built_utc": "test"})
        finally:
            E.OOF_DIR, E.RUN, E.DEADLINE = old
            shutil.rmtree(tmp)

    def test_keep_frame_reuses_the_saved_table(self):
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.MINUTE_DIR, E.RUN)
        try:
            E.OOF_DIR, E.MINUTE_DIR, E.RUN = tmp, os.path.join(tmp, "minute"), "e132keep"
            os.makedirs(E.MINUTE_DIR)
            with open(os.path.join(E.MINUTE_DIR, "minute_manifest.json"), "w") as fh:
                fh.write('{"schema": 1, "days": {"2026-11-19": {}}}')       # 毎晩の取得で変わった記録
            saved = make_frame(3).head(50)
            saved.to_parquet(E.path("frame.parquet"), index=False)
            stamp = {"built_utc": "2026-11-19 12:58 UTC", "rows": 50, "date_max": "2017-01-17",
                     "sources": {**E.source_stamp(), "minute_manifest": "older"}}
            with open(E.path("stamp.json"), "w", encoding="utf-8") as fh:
                json.dump(stamp, fh)
            with mock.patch.object(E.lab, "frame", side_effect=RuntimeError("rebuilt")):
                with redirect_stdout(io.StringIO()):
                    df, got = E.prepare(["b0"], keep=True)
                self.assertEqual(got["built_utc"], stamp["built_utc"])   # 同じ表・同じ版（A・B の保存済みの結果が使える）
                self.assertEqual(len(df), 50)
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    E.prepare(["b0"])                                   # 既定では、取得記録が変われば作り直す
        finally:
            E.OOF_DIR, E.MINUTE_DIR, E.RUN = old
            shutil.rmtree(tmp)

    def test_prepare_needs_the_minute_store(self):
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.MINUTE_DIR, E.RUN)
        try:
            E.OOF_DIR, E.MINUTE_DIR, E.RUN = tmp, os.path.join(tmp, "minute"), "e132prep"
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                E.prepare(["b0"])
            self.assertIn("分足の取得記録が無い", str(ctx.exception))
        finally:
            E.OOF_DIR, E.MINUTE_DIR, E.RUN = old
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
