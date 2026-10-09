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
  python3 research/exp/e66_nonlinear_screen.py --stage ab --n-estimators 1000   # 木の本数だけ変えた A/B
  python3 research/exp/e66_nonlinear_screen.py --stage trees --n-estimators 1000  # 200本と1000本の比較表
  python3 research/exp/e66_nonlinear_screen.py --stage tune --n-estimators 1000   # 1000本で Optuna 探索し直し + OOF
  結果は research/_data/oof/e66_*（木の本数が 200 以外なら e66t<本数>_*）。本番の設定には書かない。

木の本数（運用者の依頼 2026-10-09「特徴量数やサンプル数に対して n_estimator が小さすぎるということはないか。
1000 にして試してほしい」）: --n-estimators N で3モデルとも木を N 本にする。LightGBM は本番のパラメータの
n_estimators を、XGBoost / CatBoost は tuning_multi.N_ESTIMATORS（build が読む）を差し替える。学習率などほかの
パラメータは本番のまま（探索し直さない。歩幅の合計 = 学習率 × 本数 が 5 倍になる）。T / V / P の3腕とも同じ本数。

探索し直し（運用者の依頼 2026-10-09「optuna でチューニングはしてほしい。過学習すると思う」。確認で LightGBM だけ・
学習率の範囲は本数に合わせる、を選択）: --stage tune で T（本番239列）と V（+32列）をそれぞれ Optuna で探索し直す。
作りは本番の週次（run_tuning.py / 実験49）と同じ: 50試行 × 5分割（年×時価総額帯で層別・日付単位）、ホールドアウトより
前のデータだけ、木の本数は固定、学習率の範囲は tuning.lr_range(本数)（1000本なら 0.002〜0.04。歩幅の合計を 200本と
同じ 2〜40 に保つ）。対照 P は V の探索結果を流用する。OOF は e66u<本数>_* に保存し、200本（e66）と探索なしの
同じ本数（e66t<本数>）の両方と窓ごとに突き合わせる。xgb / cat の探索し直しは未対応（tuning_multi の探索空間の学習率を
本数に合わせる作りが要る）。

32列は research/_data/nonlinear_features.parquet に置く（無ければここで計算する。4コアで数分）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import nonlinear_features as NL  # noqa: E402
import walkforward as WF  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e25_auc_noise import average  # noqa: E402
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

#: 本番の木の本数（lgbm_params.json / tuning.SEARCH_N_ESTIMATORS）。これ以外なら別のタグに保存する
PROD_TREES = 200


def tag_for(n_estimators: int) -> str:
    """out-of-fold の保存名。本数が本番と違えば別の名前にして、保存済みと混ざらないようにする。"""
    return "e66" if int(n_estimators) == PROD_TREES else f"e66t{int(n_estimators)}"


def with_trees(n_estimators: int):
    """
    3モデルの木の本数だけを n_estimators にする。元に戻す関数を返す。

    LightGBM は E27.prod_params が返す本番パラメータの n_estimators を書き換える（E41.oof_folds が
    **params で LGBMClassifier に渡す）。XGBoost / CatBoost は tuning_multi.build が
    モジュール変数 N_ESTIMATORS を読むので、それを差し替える。学習率などは触らない。
    """
    import tuning_multi as TM

    n = int(n_estimators)
    orig_prod, orig_n = E27.prod_params, TM.N_ESTIMATORS

    def prod_params(algo: str) -> dict:
        rec = orig_prod(algo)
        if algo == "lgbm":
            rec["params"] = {**rec["params"], "n_estimators": n}
        return rec

    E27.prod_params = prod_params
    TM.N_ESTIMATORS = n

    def restore() -> None:
        E27.prod_params = orig_prod
        TM.N_ESTIMATORS = orig_n

    return restore


def compare_tags(tag_a: str, tag_b: str, label_a: str, label_b: str) -> pd.DataFrame:
    """
    2つの A/B（タグ a と b）を同じ窓で突き合わせる。どちらも {tag}_auc_by_window.csv
    （shift / arm / algo / fold / pr_base / pr_arm / roc_base / roc_arm。pr_base = T）を読む。

    見るもの（PR-AUC / ROC-AUC、切り方3通りの全窓）
      T(b) − T(a)   設定 b にするだけで本番の239列の分離力が動くか
      腕(b) − 腕(a) 32列を足した腕（または対照）で同じこと
      腕 − T @b     設定 b での 32列の増分（採否はこちら。§7 の基準）
      腕 − T @a     参考: 設定 a での増分
    """
    fa = os.path.join(OOF_DIR, f"{tag_a}_auc_by_window.csv")
    fb = os.path.join(OOF_DIR, f"{tag_b}_auc_by_window.csv")
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print(f"\n[compare] 比べる CSV が揃っていない: {fa} / {fb}")
        return pd.DataFrame()
    da, db = pd.read_csv(fa), pd.read_csv(fb)
    keys = ["shift", "algo", "fold"]
    rows = []
    for arm in ("V", "P"):
        m = da[da["arm"] == arm].merge(db[db["arm"] == arm], on=keys, suffixes=("_a", "_b"))
        if not len(m):
            continue
        for algo, g in m.groupby("algo"):
            for metric in ("pr", "roc"):
                t_d = g[f"{metric}_base_b"] - g[f"{metric}_base_a"]
                v_d = g[f"{metric}_arm_b"] - g[f"{metric}_arm_a"]
                inc = g[f"{metric}_arm_b"] - g[f"{metric}_base_b"]
                inc0 = g[f"{metric}_arm_a"] - g[f"{metric}_base_a"]
                se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan  # noqa: E731
                rows.append({"arm": arm, "algo": algo, "metric": metric, "n_win": len(g),
                             "T_b-T_a": t_d.mean(), "T_se": se(t_d), "T_up": int((t_d > 0).sum()),
                             "arm_b-arm_a": v_d.mean(), "arm_se": se(v_d), "arm_up": int((v_d > 0).sum()),
                             "inc_b": inc.mean(), "inc_se": se(inc), "inc_up": int((inc > 0).sum()),
                             "inc_a": inc0.mean(), "inc_a_up": int((inc0 > 0).sum())})
    res = pd.DataFrame(rows)
    if not len(res):
        print(f"\n[compare] 共通の窓が無い: {tag_a} / {tag_b}")
        return res
    res.to_csv(os.path.join(OOF_DIR, f"{tag_b}_vs_{tag_a}.csv"), index=False)
    for arm in ("V", "P"):
        part = res[res["arm"] == arm]
        if not len(part):
            continue
        print(f"\n■ {label_a} → {label_b}（腕 {LABELS[arm]}、切り方3通りの全窓）")
        print(f"  {'':<14}{'窓':>4}{'T(b)−T(a)':>12}{'SE':>8}{'上':>7}{'腕(b)−腕(a)':>14}{'SE':>8}{'上':>7}"
              f"{'腕−T @b':>12}{'SE':>8}{'上':>7}{'腕−T @a':>12}{'上':>7}")
        for _, r in part.iterrows():
            nw = int(r["n_win"])
            print(f"  {r['algo'] + ' ' + r['metric'].upper() + '-AUC':<14}{nw:>4}"
                  f"{r['T_b-T_a']:>+12.4f}{r['T_se']:>8.4f}{int(r['T_up']):>4}/{nw:<3}"
                  f"{r['arm_b-arm_a']:>+14.4f}{r['arm_se']:>8.4f}{int(r['arm_up']):>4}/{nw:<3}"
                  f"{r['inc_b']:>+12.4f}{r['inc_se']:>8.4f}{int(r['inc_up']):>4}/{nw:<3}"
                  f"{r['inc_a']:>+12.4f}{int(r['inc_a_up']):>4}/{nw:<3}")
    return res


def trees_summary(n_estimators: int) -> pd.DataFrame:
    """木 PROD_TREES 本（e66）と n_estimators 本・探索なし（e66t<n>）を突き合わせる。"""
    return compare_tags(tag_for(PROD_TREES), tag_for(n_estimators),
                        f"木 {PROD_TREES}本（本番のパラメータ）", f"木 {int(n_estimators)}本（本番のパラメータのまま）")


# --------------------------------------------------------------------------- #
# 6. 探索し直し（Optuna）。木の本数を変えたときに、学習率などを本数に合わせて選び直す
# --------------------------------------------------------------------------- #

N_TRIALS = 50
N_SPLITS = 5
CV_SCHEME = "year_cap_date"          # 本番の retrain-weekly.yml / run_tuning.py と同じ


def tuned_tag(n_estimators: int) -> str:
    """探索し直した腕の保存名（u = tuned）。探索なしの e66t<n> とは別にする。"""
    return f"e66u{int(n_estimators)}"


def params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]


def cache_matches(rec: Optional[dict], want: dict) -> bool:
    """保存済みの探索結果が、いま求めている条件（列・試行数・本数・打ち切り日・学習率の範囲）と同じか。"""
    if not isinstance(rec, dict) or "params" not in rec:
        return False
    return all(rec.get(k) == v for k, v in want.items())


def tune_arm(algo: str, arm: str, sub: pd.DataFrame, cols: List[str], n_estimators: int,
             lr_bounds: Tuple[float, float], cutoff, n_trials: int = N_TRIALS,
             n_splits: int = N_SPLITS) -> dict:
    """
    run_tuning.py / 実験49 と同じ関数で探索する。本数と学習率の範囲だけを変える。
    sub はホールドアウトより前の行だけ。保存済み（条件が同じ）なら読む。
    """
    if algo != "lgbm":
        raise SystemExit(f"{algo} の探索し直しは未対応（tuning_multi の探索空間の学習率を本数に合わせる作りが要る）")
    n = int(n_estimators)
    path = os.path.join(OOF_DIR, f"{tuned_tag(n)}_params_{arm}_{algo}.json")
    want = {"_features_sig": F.signature(cols), "_n_trials": int(n_trials), "_n_estimators": n,
            "_cutoff": str(pd.Timestamp(cutoff).date()), "_lr_range": [float(lr_bounds[0]), float(lr_bounds[1])],
            "_n_splits": int(n_splits)}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if cache_matches(rec, want):
            log(f"  [{arm} {algo}] 探索済みを読む（{path}）")
            return rec
        log(f"  [{arm} {algo}] 保存済みの探索は条件が違うので探索し直す")
    log(f"  [{arm} {algo}] 探索: 木{n}本・学習率 {lr_bounds[0]:g}〜{lr_bounds[1]:g} / {n_trials}試行 × {n_splits}分割 "
        f"/ {len(cols)}列 / 〜{want['_cutoff']} {len(sub):,}件")
    t0 = time.time()
    params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=n_splits,
                         embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME, model="classifier",
                         n_estimators=n, lr_bounds=lr_bounds, verbose=False)
    rec = {"params": dict(params), "_cv": dict(tuning.LAST_CV), **want,
           "_n_features": len(cols), "_minutes": round((time.time() - t0) / 60, 1)}
    os.makedirs(OOF_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    log(f"  [{arm} {algo}] 探索 {rec['_minutes']}分 / CV PR-AUC {rec['_cv'].get('mean_pr_auc')}")
    return rec


def show_tuned(recs: Dict[str, dict], n_estimators: int) -> None:
    print(f"\n■ 探索し直し（木 {int(n_estimators)}本。層別{N_SPLITS}分割の CV は楽観側に出る。パラメータ選び用）")
    keys = ["learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree",
            "reg_alpha", "reg_lambda"]
    for arm, rec in recs.items():
        p, cv = rec["params"], rec.get("_cv", {})
        lo, hi = rec.get("_lr_range", [np.nan, np.nan])
        lr = float(p["learning_rate"])
        pos = (np.log(lr) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lo and hi else np.nan
        print(f"  {LABELS[arm]}: 所要 {rec.get('_minutes')}分 / CV PR-AUC {cv.get('mean_pr_auc')} ± {cv.get('std')} "
              f"/ ROC {cv.get('mean_roc_auc')} / 正例率 {cv.get('base_rate')}")
        print("    " + " / ".join(f"{k} {p[k]:.4g}" if isinstance(p.get(k), float) else f"{k} {p.get(k)}"
                                 for k in keys if k in p))
        print(f"    学習率 {lr:.5f}（範囲 {lo:g}〜{hi:g} の中の位置 {pos:.2f}。0 が下限、1 が上限）/ "
              f"歩幅の合計（学習率 × 本数）{lr * p['n_estimators']:.2f}")


def oof_arm_params(tag: str, df: pd.DataFrame, cols: List[str], arm: str, shift: int,
                   algo: str, seeds, params: dict) -> pd.DataFrame:
    """ab_oof.oof_arm と同じ作りで、パラメータを明示して out-of-fold を作る（種の平均。保存済みなら読む）。"""
    folds = E41.folds_for(df["Date"], shift)
    fp = AB.fingerprint(df, cols)
    ph = params_hash(params)
    parts = []
    for sd in seeds:
        path = os.path.join(OOF_DIR, f"{tag}_{arm}_{fp}_{algo}_{ph}_sh{shift}_s{sd}.parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        o = E41.oof_folds(algo, df, cols, params, sd, folds)
        o.to_parquet(path, index=False)
        parts.append(o)
        log(f"  腕{arm} {algo} ずらし{shift}か月 種{sd}: {len(o):,}件 {time.time()-t0:.0f}秒")
    out = average(parts)
    out["Date"] = pd.to_datetime(out["Date"])
    out["Code"] = out["Code"].astype(str)
    return out


def run_tuned(df: pd.DataFrame, shifts: List[int], seeds, algos: List[str], n_estimators: int,
              lr_bounds: Tuple[float, float], n_trials: int = N_TRIALS) -> pd.DataFrame:
    """T と V を探索し直して OOF を取り、V−T / P−T と、200本・探索なしの同じ本数との差を出す。"""
    n = int(n_estimators)
    tag = tuned_tag(n)
    base = F.columns(F.DEFAULT_PRESET)
    v = base + NEW_COLS
    d = pd.to_datetime(df["Date"])
    train_end, _, _ = T.holdout_bounds(d, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()]
    print("=" * 78)
    print(f"実験66 探索し直し（木 {n}本・学習率 {lr_bounds[0]:g}〜{lr_bounds[1]:g}・{n_trials}試行 × {N_SPLITS}分割・"
          f"種{len(seeds)}つ・ずらし {shifts}か月・{algos}・タグ {tag}）: {len(df):,}件 / 探索は 〜{train_end.date()} {len(sub):,}件")
    for arm, cols in (("T", base), ("V", v), ("P", v)):
        print(f"  {LABELS[arm]:<30}{len(cols)}列  指紋 {F.signature(cols)}")
    print("=" * 78)
    os.makedirs(OOF_DIR, exist_ok=True)
    fp_df = permuted(df, NEW_COLS, seed=PERM_SEED)
    summary = []
    for algo in algos:
        recs = {"T": tune_arm(algo, "T", sub, base, n, lr_bounds, train_end, n_trials),
                "V": tune_arm(algo, "V", sub, v, n, lr_bounds, train_end, n_trials)}
        show_tuned(recs, n)
        arms = {"T": (df, base, recs["T"]["params"]), "V": (df, v, recs["V"]["params"]),
                "P": (fp_df, v, recs["V"]["params"])}
        for sh in shifts:
            res = {arm: oof_arm_params(tag, fr, cols, arm, sh, algo, seeds, par)
                   for arm, (fr, cols, par) in arms.items()}
            for arm in ("V", "P"):
                print(f"\n■ 分離力（窓ごと。ずらし{sh}か月・木{n}本・探索し直し）: {LABELS[arm]} − {LABELS['T']}")
                print(f"  {'':<18}{'T':>9}{arm:>9}{'差':>10}{'SE':>9}{'上の窓':>9}")
                wa, wb = AB.auc_by_window(res["T"]), AB.auc_by_window(res[arm])
                m = wa.merge(wb, on="fold", suffixes=("_a", "_b"))
                print(AB.pair_line(f"{algo} PR-AUC", m["pr_a"].to_numpy(), m["pr_b"].to_numpy()))
                print(AB.pair_line(f"{algo} ROC-AUC", m["roc_a"].to_numpy(), m["roc_b"].to_numpy()))
                for _, r in m.iterrows():
                    summary.append({"shift": sh, "arm": arm, "algo": algo, "fold": int(r["fold"]),
                                    "pr_base": r["pr_a"], "pr_arm": r["pr_b"],
                                    "roc_base": r["roc_a"], "roc_arm": r["roc_b"]})
    out = pd.DataFrame(summary)
    out.to_csv(os.path.join(OOF_DIR, f"{tag}_auc_by_window.csv"), index=False)
    if len(out):
        print(f"\n■ 切り方3通りをまとめた窓ごとの差（PR-AUC、木{n}本・探索し直し）")
        print(f"  {'':<24}{'窓の数':>7}{'差の平均':>10}{'SE':>9}{'上の窓':>9}")
        for (arm, a), g in out.groupby(["arm", "algo"]):
            dd = (g["pr_arm"] - g["pr_base"]).to_numpy()
            se = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan
            print(f"  {LABELS[arm] + ' ' + a:<24}{len(dd):>7}{dd.mean():>+10.4f}{se:>9.4f}{(dd > 0).sum():>5}/{len(dd)}")
    compare_tags(tag_for(PROD_TREES), tag, f"木 {PROD_TREES}本（本番のパラメータ）", f"木 {n}本（探索し直し）")
    if os.path.exists(os.path.join(OOF_DIR, f"{tag_for(n)}_auc_by_window.csv")):
        compare_tags(tag_for(n), tag, f"木 {n}本（本番のパラメータのまま）", f"木 {n}本（探索し直し）")
    log(f"記録: {OOF_DIR}/{tag}_*")
    return out


def run_ab(df: pd.DataFrame, shifts: List[int], seeds, algos: List[str],
           n_estimators: int = PROD_TREES) -> None:
    base = F.columns(F.DEFAULT_PRESET)
    v = base + NEW_COLS
    assert len(set(v)) == len(v)
    tag = tag_for(n_estimators)
    print("=" * 78)
    print(f"実験66 A/B（種{len(seeds)}つ・ずらし {shifts}か月・{algos}・木 {int(n_estimators)}本・タグ {tag}）: {len(df):,}件")
    for arm, cols in (("T", base), ("V", v), ("P", v)):
        print(f"  {LABELS[arm]:<30}{len(cols)}列  指紋 {F.signature(cols)}")
    print("=" * 78)
    fp = permuted(df, NEW_COLS, seed=PERM_SEED)
    arms = {"T": (df, base), "V": (df, v), "P": (fp, v)}
    restore = with_trees(n_estimators)
    try:
        for a in algos:
            par = E27.prod_params(a)["params"]
            print(f"  [{a}] 木 {par.get('n_estimators', n_estimators)}本 / 学習率 {par.get('learning_rate')}")
        AB.compare(tag, arms, "T", LABELS, shifts, seeds, algos)
    finally:
        restore()
    if int(n_estimators) != PROD_TREES:
        trees_summary(n_estimators)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験66: 非線形時系列解析の特徴量")
    ap.add_argument("--stage", default="screen", choices=("screen", "ab", "all", "trees", "tune"))
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default="lgbm,xgb,cat")
    ap.add_argument("--n-estimators", type=int, default=PROD_TREES,
                    help=f"3モデルの木の本数（既定 {PROD_TREES} = 本番）。変えると保存名が e66t<本数> になる")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS, help="--stage tune の試行数（本番の週次は 50）")
    ap.add_argument("--lr", default="scaled", choices=("scaled", "fixed"),
                    help="--stage tune の学習率の範囲。scaled = tuning.lr_range(本数)（既定）、fixed = 0.01〜0.2")
    args = ap.parse_args(argv)
    os.makedirs(OOF_DIR, exist_ok=True)
    if args.stage == "trees":
        trees_summary(args.n_estimators)
        compare_tags(tag_for(PROD_TREES), tuned_tag(args.n_estimators),
                     f"木 {PROD_TREES}本（本番のパラメータ）", f"木 {args.n_estimators}本（探索し直し）")
        compare_tags(tag_for(args.n_estimators), tuned_tag(args.n_estimators),
                     f"木 {args.n_estimators}本（本番のパラメータのまま）", f"木 {args.n_estimators}本（探索し直し）")
        return 0

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
        run_ab(df.drop(columns=[c for c in df.columns if c.startswith("_")]), shifts, seeds, algos,
               n_estimators=args.n_estimators)

    if args.stage == "tune":
        shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
        seeds = E27.SEEDS3[:args.seeds]
        algos = [a for a in args.algos.split(",") if a]
        lr = tuning.lr_range(args.n_estimators) if args.lr == "scaled" else (0.01, 0.2)
        run_tuned(df.drop(columns=[c for c in df.columns if c.startswith("_")]), shifts, seeds, algos,
                  args.n_estimators, lr, n_trials=args.n_trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
