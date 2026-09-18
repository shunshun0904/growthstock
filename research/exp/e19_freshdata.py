#!/usr/bin/env python3
"""
実験19: 信用残を直したデータにすると、結果はどう動くか。

何が変わったか
------------
2026-09-18 に信用残（`credit_ratio` の元）を取り直した。

  復旧前  486週  2016-10-07 〜 2026-08-28
  復旧後  505週  2016-10-07 〜 2026-09-11

19週ぶんの穴が埋まった。内訳は「金曜が祝日の週に月曜を叩いていた」17週と、
「公表前に叩いて0件のまま取得済みにしていた」直近2週。詳細は
docs/OPERATIONS.md。`credit_ratio` は staleness の上限が無いので、
穴の週は前の週の値が前方に引き延ばされていた。

つまり**特徴量1本の値が一部の行で変わった**。それだけの変化なので、
効いても小さいはず。効かない（差なし）という結果も十分ありうる。
それでも測るのは、日曜の本番学習がこのデータで走るから。

何と何を比べるか
--------------
  A 旧データ・旧パラメータ   これまで
  B 新データ・旧パラメータ   A との差は**データだけ**（パラメータ固定）
  C 新データ・新パラメータ   日曜に実際に起きること

B を挟むのは、A と C の差が「データが良くなったから」なのか
「探索を引き直したから」なのかを分けるため。実験18b で、条件ごとに
探索し直しただけで順位が入れ替わるのを見ている。探索のばらつきは
データの差より大きいことがある。

公平にするための細工
------------------
1. **行を揃える。** 新データは四本値が1日多いぶん、ラベルが確定する行が
   増える。増えたこと自体は利点だが、それはデータの質とは別の話なので、
   両方に存在する (Code, Date) だけで比べる。
2. **探索の条件を本番と同じにする。** 50試行 / 5分割 / year_cap_date /
   ホールドアウトより手前で打ち切り。本番の run_tuning.py と同じ
   （研究用なので research/lgbm_params.json には書かない）。
3. **種を3つ平均する。** 本番は単一種だが、ここで見たい差は小さいので
   モデル側のばらつきを落とす。実験11 のノイズ床も種を振って測った。

判定の足切り
-----------
実験11 のノイズ床（同一設定・種5個）
  窓平均 レンジ 0.143pt  <- 最も解像度が高い
採否は z = 差 / √(SE_a² + SE_b²) > 2。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
from e18_horizon import edge  # noqa: E402
from train_production import (  # noqa: E402
    OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS)

OUTCOMES = ("ret_o1_20", "ret_o1_40")
SEEDS = (42, 7, 123)
N_TRIALS = 50
N_SPLITS = 5
CV_SCHEME = "year_cap_date"      # 本番の retrain-weekly.yml と同じ
PRESET = "all"                   # 本番が採点に使う鍵（train_production --params all）

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PARAMS_PATH = os.path.join(OOF_DIR, "e19_params.json")
OLD_DATASET = os.path.join(lab.DATA_DIR, "dataset_oldmargin.parquet")
NEW_DATASET = os.path.join(lab.DATA_DIR, "dataset.parquet")


def load_store() -> dict:
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_store(store: dict) -> None:
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)


def frames() -> tuple:
    """
    旧・新のデータセットに実収益を付けて、共通行だけに揃えて返す。

    実収益（ref_end / ret_o1_*）は四本値から作るので、両方に**同じ**
    （新しい）四本値を使う。物差しはデータ修正の対象ではないし、
    別々の物差しで測ったら比較にならない。
    """
    import glob

    import sweep_design as S

    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    panel = S.Panels(bars).get(B.HIGH_WINDOW)
    ref = S.reference_outcome(panel)
    rets = lab.realized_returns(bars)

    out = {}
    for name, path in (("旧", OLD_DATASET), ("新", NEW_DATASET)):
        if not os.path.exists(path):
            raise SystemExit(f"{path} がありません")
        ds = pd.read_parquet(path)
        ds["Date"] = pd.to_datetime(ds["Date"])
        ds = ds.merge(ref, on=["Code", "Date"], how="left")
        ds = ds.merge(rets, on=["Code", "Date"], how="left")
        print(f"[load] {name}データ {len(ds):,}件 / 正例率 "
              f"{ds['label'].mean()*100:.2f}% / "
              f"{ds['Date'].min().date()} 〜 {ds['Date'].max().date()}")
        out[name] = ds

    key = ["Code", "Date"]
    common = out["旧"][key].merge(out["新"][key], on=key, how="inner")
    print(f"[align] 共通行 {len(common):,}件 "
          f"（旧 -{len(out['旧'])-len(common):,} / "
          f"新 -{len(out['新'])-len(common):,}）")
    for name in ("旧", "新"):
        out[name] = (out[name].merge(common, on=key, how="inner")
                     .sort_values("Date").reset_index(drop=True))
    return out["旧"], out["新"]


def diff_report(old: pd.DataFrame, new: pd.DataFrame, cols: list) -> None:
    """どの列が、どれだけの行で変わったかを出す。想定外の差に気づくため。"""
    print("\n=== 共通行のうち値が変わった列 ===")
    a = old.sort_values(["Code", "Date"]).reset_index(drop=True)
    b = new.sort_values(["Code", "Date"]).reset_index(drop=True)
    any_diff = False
    for c in cols + ["label"]:
        if c not in a.columns or c not in b.columns:
            continue
        x, y = a[c], b[c]
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            same = np.isclose(x.to_numpy(dtype=float), y.to_numpy(dtype=float),
                              equal_nan=True)
        else:
            same = (x.to_numpy() == y.to_numpy()) | (x.isna() & y.isna()).to_numpy()
        n = int((~same).sum())
        if n:
            any_diff = True
            print(f"  {c:<24} {n:>7,}件 ({n/len(a)*100:5.2f}%)")
    if not any_diff:
        print("  （差なし）")


def tune_on(key: str, df: pd.DataFrame, cols: list) -> dict:
    """本番の run_tuning.py と同じ条件で探索する。保存先だけが違う。"""
    store = load_store()
    if key in store:
        cv = store[key].get("_cv", {})
        print(f"  [{key}] 探索済みを読む (CV PR-AUC {cv.get('mean_pr_auc')} / "
              f"ROC {cv.get('mean_roc_auc')})")
        return store[key]

    d = pd.to_datetime(df["Date"])
    cutoff, _, _ = T.holdout_bounds(d)
    tune_df = df[d <= cutoff]
    print(f"  [{key}] 探索 {len(tune_df):,}件 (〜{cutoff.date()} / "
          f"正例率 {tune_df['label'].mean()*100:.2f}% / "
          f"エンバーゴ {T.EMBARGO_DAYS}営業日)")
    t0 = time.time()
    params = tuning.tune(tune_df, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                         embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME,
                         verbose=False)
    rec = {**params, "_cv": dict(tuning.LAST_CV)}
    store[key] = rec
    save_store(store)
    cv = rec["_cv"]
    print(f"    {(time.time()-t0)/60:.1f}分 / 木{params['n_estimators']}本 "
          f"lr {params['learning_rate']:.4f} 葉{params['num_leaves']} / "
          f"CV PR-AUC {cv.get('mean_pr_auc')} ROC {cv.get('mean_roc_auc')}")
    return rec


def oof_for(df: pd.DataFrame, cols: list, params: dict) -> pd.DataFrame:
    """
    本番（train_production.oof_scores）と同じ窓で out-of-fold を作る。

    違いは2つだけ。パラメータを明示で渡すことと、種を平均すること。
    窓の定数は train_production から取り込んでいるので、本番とずれない。
    """
    import lightgbm as lgb
    import walkforward as WF

    folds = WF.make_folds(pd.to_datetime(df["Date"]),
                          min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS,
                          step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    d = pd.to_datetime(df["Date"])
    keep = ["Code", "Date", "label", "ref_end"] + list(OUTCOMES)
    parts = []
    for f in folds:
        tr = df[(d <= pd.Timestamp(f.train_end)) & df["label"].notna()]
        te = df[(d >= pd.Timestamp(f.test_start))
                & (d <= pd.Timestamp(f.test_end)) & df["label"].notna()]
        if len(te) < 200 or len(tr) < 1000:
            continue
        ytr = tr["label"].to_numpy(dtype=int)
        Xtr = tr[cols].to_numpy(dtype=float)
        Xte = te[cols].to_numpy(dtype=float)
        spw = tuning.scale_pos_weight(ytr)
        s = np.zeros(len(te), dtype=float)
        for seed in SEEDS:
            gbm = lgb.LGBMClassifier(**{**params, "random_state": seed},
                                     scale_pos_weight=spw)
            gbm.fit(Xtr, ytr)
            s += gbm.predict_proba(Xte)[:, 1]
        part = te[[c for c in keep if c in te.columns]].copy()
        part["score"] = s / len(SEEDS)
        part["fold"] = f.index
        parts.append(part)
    if not parts:
        raise SystemExit("out-of-fold を作れません")
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score

    os.makedirs(OOF_DIR, exist_ok=True)
    old, new = frames()
    cols = [c for c in F.columns(PRESET)
            if c in old.columns and c in new.columns]
    print(f"\n目的変数 {B.DEFAULT_RISE.name}")
    print(f"特徴量 {PRESET}（{len(cols)}列）/ エンバーゴ {B.RISE_HORIZON}営業日")
    print(f"探索 {N_TRIALS}試行 × {N_SPLITS}分割 / {CV_SCHEME}（本番と同条件）")
    print(f"種 {SEEDS} の確率平均 / 物差し {OUTCOMES}")

    diff_report(old, new, cols)

    print("\n=== パラメータ探索 ===")
    p_old = tune_on("old", old, cols)
    p_new = tune_on("new", new, cols)
    strip = lambda p: {k: v for k, v in p.items() if not k.startswith("_")}

    arms = [
        ("A 旧データ・旧パラメータ", old, strip(p_old)),
        ("B 新データ・旧パラメータ", new, strip(p_old)),
        ("C 新データ・新パラメータ", new, strip(p_new)),
    ]

    print("\n=== out-of-fold を作る ===")
    runs = {}
    for i, (name, df, pp) in enumerate(arms):
        p = os.path.join(OOF_DIR, f"e19_{i}.parquet")
        if os.path.exists(p):
            runs[name] = pd.read_parquet(p)
            print(f"  {name}: 保存済みを読む ({len(runs[name]):,}件)")
            continue
        t0 = time.time()
        o = oof_for(df, cols, pp)
        o.to_parquet(p, index=False)
        runs[name] = o
        print(f"  {name}: {len(o):,}件 / {time.time()-t0:.0f}秒")

    print("\n=== 分離力（out-of-fold）===")
    print(f"  {'条件':<26}{'件数':>8}{'正例率':>9}{'PR-AUC':>9}"
          f"{'PR/正例率':>11}{'ROC-AUC':>10}{'日内AUC':>10}")
    for name, o in runs.items():
        y = o["label"].to_numpy(dtype=int)
        s = o["score"].to_numpy(dtype=float)
        br = y.mean()
        pr = average_precision_score(y, s)
        print(f"  {name:<26}{len(o):>8,}{br*100:>8.2f}%{pr:>9.4f}"
              f"{pr/br:>10.2f}x{roc_auc_score(y, s):>10.4f}"
              f"{lab.auc_in_day(o):>10.4f}")

    names = [n for n, _, _ in arms]
    for outcome in OUTCOMES:
        print(f"\n=== 実収益 {outcome} ===")
        print(f"  {'条件':<26}{'窓平均':>10}{'標準誤差':>10}"
              f"{'勝ち窓':>9}{'最悪の窓':>11}{'取引数':>9}")
        ms = {}
        for name, o in runs.items():
            m = edge(o, outcome)
            ms[name] = m
            print(f"  {name:<26}{m['thr_fold_mean']:>+9.2f}pt{m['se']:>10.2f}"
                  f"{m['thr_folds_won']:>6}/{m['thr_folds']:<2}"
                  f"{m['thr_worst']:>+10.2f}pt{m['thr_n']:>9,}")

        print("  --- 差の検定（足切り z>2）---")
        for a, b, note in [(0, 1, "データだけを新しくした効果（パラメータは固定）"),
                           (1, 2, "その上で探索を引き直した効果"),
                           (0, 2, "合計（日曜に起きること）")]:
            ma, mb = ms[names[a]], ms[names[b]]
            dd = mb["thr_fold_mean"] - ma["thr_fold_mean"]
            se = np.sqrt(ma["se"] ** 2 + mb["se"] ** 2)
            z = dd / se if se else float("nan")
            v = "採用可" if z > 2 else ("要確認" if z > 1 else
                                     ("差なし" if z > -1 else "悪化"))
            print(f"    {note}")
            print(f"      差 {dd:+.2f}pt / 合成SE {se:.2f} / z {z:+.2f} → {v}")

    print("\n注意: 変わったのは特徴量1本（credit_ratio）の一部の行だけ。")
    print("      z が立たないのが自然で、立たなければ「直したが影響は小さい」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
