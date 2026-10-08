#!/usr/bin/env python3
"""
実験66: 非線形時系列解析の特徴量（nl_* 32列、research/nonlinear_features.py）の予備スクリーニングと、
本番の239列に足した A/B。

運用者の依頼（2026-10-08）「非線形時系列解析の観点での特徴量追加のアイデア」。確認（AskUserQuestion）で
①整理＋実データで予備スクリーニング ②対象は個別銘柄の株価と出来高・売買代金 ③網羅して一括で入れる、
を選んだ。候補の定義と根拠は docs/FEATURE_IDEAS_NONLINEAR.md。

段階
  screen  1. 充足（年ごと）と分布（中央値・5/95%点・外れ値）
          2. 既存列との冗長（vol_20d / vol_rel_long / ret_20d / base_length など）と、32列どうし
          3. 実験40 と同じ両側スクリーニング（11窓、上位/下位10% の ret_o1_20 の超過を窓ごとに取り、
             窓をまたいだ平均 / SE を z）。物差しの校正のため本番の列も数本並べる
          4. 本番 OOF（Release の oof.parquet、LightGBM）の上位10% の中で、当たり（label=1）と外れを
             分けるか（窓ごとの AUC と、列の上半分 − 下半分の ret_o1_20 の差）。
             docs/MODEL_ADOPTION_RULES.md §13 の「同じスコア帯の中では分ける列が無い」に対する直接の確認。
             高スコアの負例が集まる低ボラ帯（窓の中で vol_20d 下位20%）に限った版も出す
  ab      5. T 本番の239列 / V T + 32列（271列）/ P 対照（32列を同じ日の銘柄どうしで入れ替え）を
             ab_oof.compare で比べる（3モデル × 種3つ × ずらし 0/2/4か月、本番のパラメータを読むだけ）。
             採否は §7（2026-09-26 改訂）: 3モデル中2つ以上で V−T の PR-AUC が正、対照 P を上回り、
             上の窓が過半（32窓なら 17以上）

  python3 research/exp/e66_nonlinear_screen.py --stage screen
  python3 research/exp/e66_nonlinear_screen.py --stage ab --algos lgbm,xgb,cat --seeds 3 --shifts 0,2,4
  結果は research/_data/oof/e66_*。本番の設定には書かない。

32列は research/_data/nonlinear_features.parquet に置く（無ければここで計算する。4コアで数分）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import nonlinear_features as NL  # noqa: E402
import walkforward as WF  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
from e23_dimension_screen import screen_both  # noqa: E402
from e44_shortsale import permuted  # noqa: E402
from train_production import (  # noqa: E402
    OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS)

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
NEW_COLS = list(NL.NL_COLS)
#: 物差しの校正に並べる本番の列（実験40・50・60 で強かったもの）
REF_COLS = ["vol_20d", "vol_rel_long", "vol_rel_short", "vol_updown", "vol_gap_ratio",
            "ret_20d", "rel_sector_20", "base_length", "break_margin", "close_position",
            "volume_trend", "log_trading_value", "log_market_cap", "days_since_disc",
            "days_to_earn", "credit_ratio", "inv_busco_4w"]
LABELS = {"T": "T 本番の239列", "V": "V T + 非線形32列", "P": "P 対照（32列を日付内で入れ替え）"}
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261008
Z_CUT = 2.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_nl(df: pd.DataFrame) -> pd.DataFrame:
    """32列を読む。無ければ計算して保存する。"""
    path = NL.OUT_PATH
    if not os.path.exists(path):
        log("nonlinear_features.parquet が無いので計算する")
        bars = NL.load_bars(lab.DATA_DIR)
        nl = NL.attach_nonlinear(df[["Code", "Date"]], bars,
                                 workers=max(1, os.cpu_count() or 1), log=log)
        nl.to_parquet(path, index=False)
    nl = pd.read_parquet(path)
    nl["Code"] = nl["Code"].astype(str)
    nl["Date"] = pd.to_datetime(nl["Date"])
    return nl


def load_frame() -> pd.DataFrame:
    df = lab.frame()
    df = df[df["label"].notna()].reset_index(drop=True)
    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    nl = load_nl(df)
    df = df.merge(nl, on=["Code", "Date"], how="left")
    miss = [c for c in NEW_COLS if c not in df.columns]
    if miss:
        raise SystemExit(f"32列が結合できない: {miss}")
    return df


def windows_for(df: pd.DataFrame) -> List[tuple]:
    folds = WF.make_folds(df["Date"], min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    return [(np.datetime64(f.test_start), np.datetime64(f.test_end)) for f in folds]


# --------------------------------------------------------------------------- #
# 1〜2. 充足・分布・冗長
# --------------------------------------------------------------------------- #

def coverage_and_stats(df: pd.DataFrame) -> pd.DataFrame:
    y = df["Date"].dt.year
    rows = []
    for c in NEW_COLS:
        v = df[c]
        by = v.notna().groupby(y).mean() * 100
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        out = ((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).mean() * 100 if iqr > 0 else 0.0
        rows.append({"col": c, "充足%": v.notna().mean() * 100, "最悪の年": by.idxmin(),
                     "最悪年の充足%": by.min(), "中央値": v.median(), "p5": v.quantile(0.05),
                     "p95": v.quantile(0.95), "外れ値%": out})
    t = pd.DataFrame(rows)
    print("\n■ 1. 充足と分布（全行。初年度 2018 は履歴不足で欠けやすく、評価窓の外）")
    print(f"  {'列':<22}{'充足%':>7}{'最悪年':>8}{'その充足%':>10}{'中央値':>9}{'p5':>9}{'p95':>9}{'外れ値%':>8}")
    for _, r in t.iterrows():
        print(f"  {r['col']:<22}{r['充足%']:>6.1f}%{int(r['最悪の年']):>8}{r['最悪年の充足%']:>9.1f}%"
              f"{r['中央値']:>9.3f}{r['p5']:>9.3f}{r['p95']:>9.3f}{r['外れ値%']:>7.1f}%")
    return t


def redundancy(df: pd.DataFrame) -> pd.DataFrame:
    refs = [c for c in REF_COLS if c in df.columns]
    m = df[NEW_COLS + refs].corr(method="spearman")
    print("\n■ 2. 既存列との冗長（Spearman。|ρ| ≥ 0.5 のものだけ表示）")
    flagged = []
    for c in NEW_COLS:
        hits = [(r, m.loc[c, r]) for r in refs if abs(m.loc[c, r]) >= 0.5]
        if hits:
            flagged.append(c)
            print(f"  {c:<22}" + "  ".join(f"{r} {v:+.2f}" for r, v in hits))
    if not flagged:
        print("  既存列と |ρ| ≥ 0.5 の組は無い")
    print("\n  32列どうし（|ρ| ≥ 0.7）")
    pairs = [(a, b, m.loc[a, b]) for i, a in enumerate(NEW_COLS) for b in NEW_COLS[i + 1:]
             if abs(m.loc[a, b]) >= 0.7]
    print("  " + (" / ".join(f"{a}×{b} {v:+.2f}" for a, b, v in pairs) or "0.7 以上の組なし"))
    # vol_20d / vol_rel_long との相関は特に重要（§13: 大外れはボラの帯で決まる）
    print("\n  参考: vol_20d / vol_rel_long / ret_20d との Spearman（全32列）")
    print(f"  {'列':<22}{'vol_20d':>9}{'vol_rel_long':>13}{'ret_20d':>9}{'base_length':>12}{'log_tv':>8}")
    for c in NEW_COLS:
        print(f"  {c:<22}{m.loc[c, 'vol_20d']:>+9.2f}{m.loc[c, 'vol_rel_long']:>+13.2f}"
              f"{m.loc[c, 'ret_20d']:>+9.2f}{m.loc[c, 'base_length']:>+12.2f}"
              f"{m.loc[c, 'log_trading_value']:>+8.2f}")
    return m.loc[NEW_COLS, refs]


# --------------------------------------------------------------------------- #
# 3. 両側スクリーニング（実験40 と同じ）
# --------------------------------------------------------------------------- #

def show_screen(res: pd.DataFrame, title: str) -> None:
    print(f"\n■ {title}")
    print("  上位z = その列が高い銘柄の ret_o1_20 が同じ窓の平均より上か（11窓の平均÷SE）、下位z = 低い銘柄")
    print(f"  {'列':<22}{'充足':>6}{'窓AUC':>7}{'上位pt':>8}{'上位z':>7}{'勝窓':>6}{'下位pt':>8}{'下位z':>7}{'勝窓':>6}{'判定':>10}")
    for _, r in res.iterrows():
        if not np.isfinite(r.get("n_win", np.nan)) or r.get("n_win", 0) == 0:
            print(f"  {r['feature']:<22}{r['coverage']*100:>5.0f}%   測れない")
            continue
        verdict = []
        if r["top_z"] > Z_CUT:
            verdict.append("買う")
        if r["top_z"] < -Z_CUT:
            verdict.append("高いと悪い")
        if r["bot_z"] < -Z_CUT:
            verdict.append("見送る")
        if r["bot_z"] > Z_CUT:
            verdict.append("低いと良い")
        print(f"  {r['feature']:<22}{r['coverage']*100:>5.0f}%{r['auc_win']:>7.3f}"
              f"{r['top_pt']:>+8.2f}{r['top_z']:>+7.2f}{int(r['top_pos']):>3}/{int(r['n_win']):<2}"
              f"{r['bot_pt']:>+8.2f}{r['bot_z']:>+7.2f}{int(r['bot_pos']):>3}/{int(r['n_win']):<2}"
              f"{'・'.join(verdict) or '—':>10}")


def within_window_pct(df: pd.DataFrame, col: str, windows: Sequence[tuple]) -> np.ndarray:
    d = df["Date"].to_numpy()
    out = np.full(len(df), np.nan)
    v = df[col].to_numpy(dtype=float)
    for s, e in windows:
        w = (d >= s) & (d <= e) & np.isfinite(v)
        if w.sum():
            out[w] = pd.Series(v[w]).rank(pct=True).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# 4. 本番 OOF の上位帯の中で分けるか
# --------------------------------------------------------------------------- #

def band_analysis(df: pd.DataFrame, windows: Sequence[tuple], top_pct: float = 0.9,
                  lowvol_only: bool = False) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    path = os.path.join(lab.DATA_DIR, "oof.parquet")
    if not os.path.exists(path):
        print(f"\n■ 4. 本番 OOF（{path}）が無いので飛ばす")
        return pd.DataFrame()
    oof = pd.read_parquet(path)[["Code", "Date", "score"]]
    oof["Code"] = oof["Code"].astype(str)
    oof["Date"] = pd.to_datetime(oof["Date"])
    feats = NEW_COLS + [c for c in REF_COLS if c in df.columns]
    o = oof.merge(df[["Code", "Date", "label", lab.OUTCOME] + feats], on=["Code", "Date"], how="inner")
    if lowvol_only:
        o["_volpct"] = within_window_pct(o, "vol_20d", windows)
    o["_spct"] = within_window_pct(o, "score", windows)
    d = o["Date"].to_numpy()
    rows = []
    for c in feats:
        x = o[c].to_numpy(dtype=float)
        aucs, diffs = [], []
        for s, e in windows:
            w = (d >= s) & (d <= e) & (o["_spct"].to_numpy() >= top_pct) & np.isfinite(x)
            if lowvol_only:
                w &= o["_volpct"].to_numpy() <= 0.2
            if w.sum() < 30:
                continue
            y = o["label"].to_numpy(dtype=float)[w]
            r = o[lab.OUTCOME].to_numpy(dtype=float)[w]
            xx = x[w]
            if len(np.unique(y)) < 2 or np.nanstd(xx) == 0:
                continue
            aucs.append(roc_auc_score(y, xx))
            hi = xx >= np.median(xx)
            diffs.append((np.nanmean(r[hi]) - np.nanmean(r[~hi])) * 100)
        if len(aucs) >= 3:
            a, dd = np.array(aucs), np.array(diffs)
            rows.append({"feature": c, "n_win": len(a), "auc_mean": a.mean(),
                         "auc_z": (a.mean() - 0.5) / (a.std(ddof=1) / np.sqrt(len(a))) if a.std(ddof=1) > 0 else np.nan,
                         "diff_pt": dd.mean(), "diff_z": dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if dd.std(ddof=1) > 0 else np.nan,
                         "diff_pos": int((dd > 0).sum())})
    res = pd.DataFrame(rows)
    if not len(res):
        return res
    res["abs"] = (res["auc_mean"] - 0.5).abs()
    res = res.sort_values("abs", ascending=False).reset_index(drop=True)
    title = ("本番 OOF（LightGBM）の上位10% の中で、当たり（label=1）と外れを分けるか"
             + ("（低ボラ帯: 窓の中で vol_20d 下位20% に限る）" if lowvol_only else ""))
    print(f"\n■ 4{'b' if lowvol_only else 'a'}. {title}")
    print("  AUC は上位帯の中の label に対するもの（0.5 から遠いほど分ける）。差 = 列の上半分 − 下半分の ret_o1_20（pt）")
    print(f"  {'列':<22}{'窓':>4}{'AUC平均':>9}{'AUCのz':>8}{'差pt':>8}{'差のz':>8}{'正の窓':>7}")
    for _, r in res.iterrows():
        tag = "  ←本番" if r["feature"] in REF_COLS else ""
        print(f"  {r['feature']:<22}{int(r['n_win']):>4}{r['auc_mean']:>9.3f}{r['auc_z']:>+8.2f}"
              f"{r['diff_pt']:>+8.2f}{r['diff_z']:>+8.2f}{int(r['diff_pos']):>4}/{int(r['n_win']):<3}{tag}")
    return res


# --------------------------------------------------------------------------- #
# 5. A/B（ab_oof）
# --------------------------------------------------------------------------- #

def run_ab(df: pd.DataFrame, shifts: List[int], seeds, algos: List[str]) -> None:
    base = F.columns(F.DEFAULT_PRESET)
    v = base + NEW_COLS
    assert len(set(v)) == len(v)
    print("=" * 78)
    print(f"実験66 A/B（種{len(seeds)}つ・ずらし {shifts}か月・{algos}）: {len(df):,}件")
    for arm, cols in (("T", base), ("V", v), ("P", v)):
        print(f"  {LABELS[arm]:<30}{len(cols)}列  指紋 {F.signature(cols)}")
    print("=" * 78)
    fp = permuted(df, NEW_COLS, seed=PERM_SEED)
    arms = {"T": (df, base), "V": (df, v), "P": (fp, v)}
    AB.compare("e66", arms, "T", LABELS, shifts, seeds, algos)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験66: 非線形時系列解析の特徴量")
    ap.add_argument("--stage", default="screen", choices=("screen", "ab", "all"))
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default="lgbm,xgb,cat")
    args = ap.parse_args(argv)
    os.makedirs(OOF_DIR, exist_ok=True)

    df = load_frame()
    windows = windows_for(df)
    print("=" * 78)
    print(f"実験66 非線形時系列解析の特徴量（{len(NEW_COLS)}列）: {len(df):,}件 / 銘柄 {df['Code'].nunique():,} / "
          f"{df['Date'].min().date()}〜{df['Date'].max().date()} / 評価窓 {len(windows)}本 / 物差し {lab.OUTCOME}")
    print("=" * 78)

    if args.stage in ("screen", "all"):
        st = coverage_and_stats(df)
        st.to_csv(os.path.join(OOF_DIR, "e66_stats.csv"), index=False)
        red = redundancy(df)
        red.to_csv(os.path.join(OOF_DIR, "e66_redundancy.csv"))
        refs = [c for c in REF_COLS if c in df.columns]
        res = screen_both(df, NEW_COLS + refs, windows)
        res["ref"] = res["feature"].isin(refs)
        res.to_csv(os.path.join(OOF_DIR, "e66_screen.csv"), index=False)
        show_screen(res[~res["ref"]], "3a. 両側スクリーニング（全行、11窓）: 非線形32列")
        show_screen(res[res["ref"]], "3a'. 同じ物差しでの本番の列（校正用）")
        # 低ボラ帯（§13: 高スコアの負例が集まる帯）に限った版
        df["_volpct"] = within_window_pct(df, "vol_20d", windows)
        low = df[df["_volpct"] <= 0.2].reset_index(drop=True)
        res_low = screen_both(low, NEW_COLS + refs, windows)
        res_low["ref"] = res_low["feature"].isin(refs)
        res_low.to_csv(os.path.join(OOF_DIR, "e66_screen_lowvol.csv"), index=False)
        show_screen(res_low[~res_low["ref"]], f"3b. 低ボラ帯（窓の中で vol_20d 下位20%、{len(low):,}件）に限った両側スクリーニング: 非線形32列")
        show_screen(res_low[res_low["ref"]], "3b'. 同 本番の列")
        b1 = band_analysis(df, windows, lowvol_only=False)
        b1.to_csv(os.path.join(OOF_DIR, "e66_band_top10.csv"), index=False)
        b2 = band_analysis(df, windows, lowvol_only=True)
        b2.to_csv(os.path.join(OOF_DIR, "e66_band_top10_lowvol.csv"), index=False)
        n_hit = int((res[~res["ref"]]["max_abs_z"] > Z_CUT).sum())
        print(f"\n[screen] 足切り |z|>{Z_CUT} を超えた非線形の列: {n_hit}/{len(NEW_COLS)}"
              f"（32本測れば偶然でも数本出る。群の中で向きが揃うかで読む）")

    if args.stage in ("ab", "all"):
        shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
        seeds = E27.SEEDS3[:args.seeds]
        algos = [a for a in args.algos.split(",") if a]
        run_ab(df.drop(columns=[c for c in df.columns if c.startswith("_")]), shifts, seeds, algos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
