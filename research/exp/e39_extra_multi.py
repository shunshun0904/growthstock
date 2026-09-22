#!/usr/bin/env python3
"""
実験39: 足した列を全部入れたとき、3モデルの探索込みバックテストは
どう動くか。

足した中身は 52列 / 8つの塊。
  #47〜#50 の新しいエンドポイント7本（44列）
    fwd 6 / holders_lvs 6 / holders_major 7 / holders_cross 8 /
    margin_alert 4 / earn_ahead 1 / flow 12
  #45 の予想修正イベント（8列。fins から作るので新しい取得は無い）
    revision 8

運用者の指示（2026-09-22）
  「47から50は1個ずつ検証というより、すべて実装後に、3モデルの5cv
    パラメータチューニング&oofでも評価 というのを実行して下さい。
    なぜなら、もともと全部ぶち込む予定のものだったからです。」

腕（モデルごとに3つ）
  A   153列 / 本番のパラメータ（いまの本番そのもの）
  B1  205列 / A と同じパラメータ             —— 特徴量だけの効果
  B2  205列 / 205列で探索し直したパラメータ  —— 本番が実際にやること
      （year_cap_date・5分割・50試行・探索の種 0。実験27 と同じ作法）

モデルは LightGBM / XGBoost / CatBoost の3つ。運用の選定基準
（3モデルすべてが上位10%）がこの3つなので、ここを測れば足りる。

内訳の切り分け
  all_fwd / all_holders / all_flow / all_earn / all_revision を
  LightGBM だけで回し、205列の差がどの塊から来ているかを見る。全部入りで差が出ても、
  中身が1つの塊だけなら、そこだけ採る判断ができる。

判定は docs/MODEL_ADOPTION_RULES.md §7（分離力＝PR-AUC）。
実収益は運用の2基準（lgbm単体95 / 3モデル90）で記録する。

探索の DB は research/_data/oof/e39_optuna.db（本番と混ぜない）。
結果は research/_data/oof/e39_*。**本番の設定には書かない。**
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import ops_rule as OR  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import train_model as T  # noqa: E402
import e19_freshdata as E19  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402

MODELS = ("lgbm", "xgb", "cat")
SEEDS3 = E27.SEEDS3
N_TRIALS = E27.N_TRIALS
N_SPLITS = E27.N_SPLITS
AUC_RANGE = E27.AUC_RANGE
OOF_DIR = E27.OOF_DIR
STUDY_DB = os.path.join(OOF_DIR, "e39_optuna.db")

#: 内訳の切り分け。LightGBM だけで回す
SPLITS = ["all_fwd", "all_holders", "all_flow", "all_earn", "all_revision"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune_for(algo: str, sub: pd.DataFrame, cols: list, tag: str) -> dict:
    """205列で探索し直す。結果は残して、2回目以降は読むだけにする。"""
    path = os.path.join(OOF_DIR, f"e39_params_{algo}_{tag}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        log(f"  [{algo}] 探索済みを読む（CV PR-AUC "
            f"{rec.get('_cv', {}).get('mean_pr_auc')}）")
        return rec
    t0 = time.time()
    if algo == "lgbm":
        params = tuning.tune(sub, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                             embargo_days=T.EMBARGO_DAYS, scheme="year_cap_date",
                             verbose=False)
        rec = {"params": dict(params), "_cv": dict(tuning.LAST_CV)}
    else:
        TM.STUDY_DB = STUDY_DB
        rec = TM.tune(algo, sub, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                      verbose=False)
    rec["_cv"]["seconds"] = round(time.time() - t0)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    log(f"  [{algo}] 探索 {time.time()-t0:.0f}秒 / CV PR-AUC "
        f"{rec['_cv'].get('mean_pr_auc')}")
    return rec


def oof_arm(algo: str, tag: str, df: pd.DataFrame, cols: list,
            params: dict) -> pd.DataFrame:
    """種3つの確率平均。途中まででも保存しておき、再開できるようにする。

    params は素のハイパーパラメータでも、探索の記録
    （{"params": {...}, "_cv": {...}}）でも受ける。**ここで1回だけ
    ほどく。** 呼び出し側でほどく作りにしていたら実際に忘れて、
    LightGBM に `params` という名前の dict をそのまま渡し
    `TypeError: Unknown type of parameter:params, got:dict` で落ちた。
    学習器のパラメータに `params` という名前は無いので、この判定で曖昧さは無い。
    """
    if isinstance(params, dict) and isinstance(params.get("params"), dict):
        params = params["params"]
    oofs = []
    for s in SEEDS3:
        p = os.path.join(OOF_DIR, f"e39_{algo}_{tag}_s{s}.parquet")
        if os.path.exists(p):
            oofs.append(pd.read_parquet(p))
            continue
        t0 = time.time()
        if algo == "lgbm":
            E19.SEEDS = (s,)
            o = E19.oof_for(df, cols, params)
        else:
            o = E27.oof_multi(algo, df, cols, params, s)
        o.to_parquet(p, index=False)
        oofs.append(o)
        log(f"    {algo} {tag} 種 {s}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    return average(oofs)


# 列は e25_auc_noise.metrics() が実際に返す鍵に合わせる。
# metrics() は平らな辞書を返し、`auc` も `label_rate` も入れ子も持たない
#   pr_auc / roc_auc / day_auc / lift
#   ret_o1_20_mean / _won / _n / _worst（ret_o1_40 も同じ形）
HEAD = (f"  {'腕':<26}{'列':>5}{'PR-AUC':>9}{'リフト':>7}{'ROC':>8}{'日内':>8}"
        f"{'窓平均超過':>12}{'勝窓':>7}{'最悪':>10}")


def line(name: str, ncol: int, m: dict) -> str:
    return (f"  {name:<26}{ncol:>5}{m['pr_auc']:>9.4f}{m['lift']:>6.2f}x"
            f"{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
            f"{m['ret_o1_20_mean']:>+10.2f}pt"
            f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
            f"{m['ret_o1_20_worst']:>+8.2f}pt")


def main() -> int:
    os.makedirs(OOF_DIR, exist_ok=True)
    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    cols_a = [c for c in F.columns("all") if c in df.columns]
    cols_b = [c for c in F.columns("all_plus") if c in df.columns]
    missing = [c for c in F.columns("all_plus") if c not in df.columns]
    log(f"母集団 {len(df):,}件 / A {len(cols_a)}列 / B {len(cols_b)}列")
    if missing:
        log(f"  データセットに無い列 {len(missing)}本: {missing[:12]}")
    new = [c for c in cols_b if c not in cols_a]
    if not new:
        log("新しい列が1本もデータセットに入っていない。build_dataset を先に回す")
        return 1
    cov = df[new].notna().mean().sort_values()
    log(f"  新しい列 {len(new)}本 / 充足 最小 {cov.iloc[0]*100:.1f}%"
        f"（{cov.index[0]}）・中央値 {cov.median()*100:.1f}%"
        f"・最大 {cov.iloc[-1]*100:.1f}%（{cov.index[-1]}）")
    print(f"\n  {'列':<26}{'充足':>8}")
    for c, v in cov.items():
        print(f"  {c:<26}{v*100:>7.1f}%")

    # 探索はホールドアウトより手前だけで行う（本番の作法。実験27 と同じ）。
    # ここを全期間にすると、評価する窓を見て探索したことになる
    d = pd.to_datetime(df["Date"])
    train_end, _, _ = T.holdout_bounds(d, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()]
    log(f"  探索は 〜{train_end.date()} の {len(sub):,}件 / "
        f"{N_TRIALS}試行 × {N_SPLITS}分割 / 種 {SEEDS3}")
    rows, oofs_by_arm = [], {}
    print(f"\n=== 3モデル × 3腕（A={len(cols_a)}列 / B1={len(cols_b)}列・同じパラメータ / "
          f"B2={len(cols_b)}列・探索し直し）===")
    print(HEAD)
    for algo in MODELS:
        pa = E27.prod_params(algo)
        arms = [("A", cols_a, pa), ("B1", cols_b, pa)]
        try:
            pb = tune_for(algo, sub, cols_b, "all_plus")
            arms.append(("B2", cols_b, pb))
        except Exception as exc:                             # noqa: BLE001
            log(f"  [{algo}] 探索に失敗（A/B1 だけで続ける）: "
                f"{type(exc).__name__}: {str(exc)[:160]}")
        for tag, cols, par in arms:
            o = oof_arm(algo, tag, df, cols, par)
            m = metrics(o)
            oofs_by_arm.setdefault(tag, {})[algo] = o
            rows.append({"algo": algo, "arm": tag, "ncol": len(cols),
                         "label_rate": float(o["label"].mean()), **m})
            print(line(f"{algo} {tag}", len(cols), m))
        print()
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e39_arms.csv"), index=False)

    print(f"=== 判定（分離力。足切りは PR-AUC の差 > {2*AUC_RANGE:.4f}）===")
    print(f"  {'モデル':<10}{'A':>9}{'B1':>9}{'差':>9}{'B2':>9}{'差':>9}{'判定':>10}")
    for algo in MODELS:
        g = {r["arm"]: r for r in rows if r["algo"] == algo}
        if "A" not in g:
            continue
        a = g["A"]["pr_auc"]
        b1 = g.get("B1", {}).get("pr_auc", float("nan"))
        b2 = g.get("B2", {}).get("pr_auc", float("nan"))
        best = np.nanmax([b1, b2])
        ok = np.isfinite(best) and (best - a) > 2 * AUC_RANGE
        print(f"  {algo:<10}{a:>9.4f}{b1:>9.4f}{b1-a:>+9.4f}{b2:>9.4f}{b2-a:>+9.4f}"
              f"{'採用' if ok else '見送り':>10}")

    print(f"\n=== 運用の2基準での実収益（3モデルの合議）===")
    for tag in ("A", "B1", "B2"):
        by = oofs_by_arm.get(tag, {})
        if len(by) < 3:
            continue
        print(f"\n  --- 腕 {tag} ---")
        OR.report(by)

    print(f"\n=== 内訳の切り分け（LightGBM のみ。どの塊が効いたか）===")
    print(HEAD)
    pa = E27.prod_params("lgbm")
    for preset in ["all"] + SPLITS + ["all_plus"]:
        cols = [c for c in F.columns(preset) if c in df.columns]
        o = oof_arm("lgbm", f"p_{preset}", df, cols, pa)
        m = metrics(o)
        rows.append({"algo": "lgbm", "arm": f"preset:{preset}", "ncol": len(cols),
                     "label_rate": float(o["label"].mean()), **m})
        print(line(preset, len(cols), m))
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e39_arms.csv"), index=False)
    log(f"記録: {OOF_DIR}/e39_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
