#!/usr/bin/env python3
"""
実験15: 5モデルすべてを同じ条件で探索する（週次でも回る）。

    python3 research/exp/e15_tune_all.py            # 必要なものだけ探索
    python3 research/exp/e15_tune_all.py --force    # 全部やり直す
    python3 research/exp/e15_tune_all.py mlp        # モデルを指定

なぜ全部探索するか
-----------------
これまで lgbm だけが探索済みで、他は手で置いた既定値だった。
「lgbm が一貫している」という以前の読みは、探索労力の差を見ていただけ。
ダッシュボードに5モデルのスコアを並べるなら、全部に同じ手間をかける。

条件（lgbm と完全に同一）
  分割      年 × 時価総額帯で層別・日付単位で分割・5分割
  目的関数  分割平均の PR-AUC
  試行数    50
  木の本数  200 固定
  期間      ホールドアウト（直近12ヶ月）より手前のみ。実測 17,449件・
            2018-04-03 〜 2025-03-21

探索が終わったら、探索済みパラメータで out-of-fold を作り直して
運用指標（しきい値運用・翌営業日の寄り買い・40営業日後の5日平均終値売り）
で並べる（research/exp/e16_lineup.py）。

探索する列
----------
本番の列（features.DEFAULT_PRESET）で探索する。学習（train_multi.py）と
同じ列でなければ探索する意味が無い（運用者の指示 2026-09-23。153列 -> 205列 に
切り替えたとき、探索だけ153列のまま残っていた）。

いつ探索し直すか
--------------
訓練データの最終日（`_cv["train_to"]`）か、探索した列（`_cv["features_sig"]`）が
変わっていたら探索する。週次で新しい営業日が積まれると母集団が変わるので、
そのときは50試行やり直す。どちらも同じなら保存済みを使う（同じデータ・同じ列で
探索し直しても、乱数種が同じなので同じ答えになるだけ）。

実測の探索時間（50試行×5分割、Optuna の試行記録より）
  xgb 21分 / logit 51分 / cat 7分 / mlp 7分 = 合計86分
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402
import tuning_multi as TM  # noqa: E402
from train_model import EMBARGO_DAYS, HOLDOUT_MONTHS, holdout_bounds  # noqa: E402

N_TRIALS = 50


def why_retune(prev: dict, train_to: str, sig: str,
               trees: Optional[int] = None) -> Optional[str]:
    """
    保存済みの探索（prev = multi_params.json の _cv）を使えるか。
    使えるなら None、探索し直すなら理由を返す。

    訓練データの最終日と、探索した列の両方が同じときだけ使う。
    列の条件が無かったころは、153列 -> 205列 に切り替えても同じ週なら
    153列で探索したパラメータを使い続けるところだった。

    trees は木のモデルの本数（tuning_multi.N_ESTIMATORS）。木の無いモデルは None。
    本数が違う探索も使わない（学習は今の本数で組むので、別の本数で選んだ
    パラメータを当てはめることになる）。
    """
    if not prev:
        return "探索済みパラメータが無い"
    if prev.get("features_sig") != sig:
        return (f"探索した列が違う（{prev.get('n_features', '?')}列・指紋 "
                f"{prev.get('features_sig', 'なし')} -> {sig}）")
    if trees is not None and prev.get("n_estimators") != trees:
        return f"探索した木の本数が違う（{prev.get('n_estimators')}本 -> {trees}本）"
    if prev.get("train_to") != train_to:
        return f"訓練最終日が {prev.get('train_to')} から {train_to} に動いた"
    return None


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    algos = args or list(TM.ALGOS)
    df = lab.frame()
    preset = F.DEFAULT_PRESET
    cols = F.columns(preset)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{preset} の列がデータセットにありません（{len(missing)}本）: "
                         f"{missing[:5]}")
    sig = F.signature(cols)
    d = pd.to_datetime(df["Date"])
    train_end, _, _ = holdout_bounds(d, HOLDOUT_MONTHS, EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()]
    print(f"探索期間 {d.min().date()} 〜 {train_end.date()} / {len(sub):,}件 / "
          f"正例率 {sub['label'].mean()*100:.1f}%")
    print(f"対象: {', '.join(algos)} / 各{N_TRIALS}試行 / 列 {preset}（{len(cols)}列・指紋 {sig}）")
    print()

    store = TM.load()
    # 比較に使うのは「訓練データの最終取引日」。TM.tune が _cv に保存するのと
    # 同じ値でないと、いつまでも一致しない。
    #
    # ここで train_end（ホールドアウトの境界）を使っていたのが誤りだった。
    # 境界は暦日で決まるので週末に落ちることがあり、そのとき最終取引日とは
    # ずれる。しかも境界は毎週ほぼ7日ずつ進むので曜日が固定され、
    # 「毎週かならず再探索する」か「一生スキップし続ける」かの
    # どちらかに張り付く。実際 2026-09-13 の週次実行では前者になり、
    # 訓練データが1日も増えていないのに5モデルを2時間かけて探索し直した。
    train_to = str(pd.to_datetime(sub["Date"]).max().date())
    for algo in algos:
        prev = store.get(algo, {}).get("_cv", {})
        # 訓練データの最終取引日が同じなら探索し直さない。同じデータ・同じ種
        # なら同じ答えになるだけで、時間だけ掛かる。
        # 日付が動いていれば母集団が変わっているので50試行やり直す
        # （通常の週次実行では5営業日ぶん進むので、必ず探索が走る）
        why = why_retune(prev, train_to, sig,
                         TM.N_ESTIMATORS if algo in TM.TREE_ALGOS else None)
        if why is None and not force:
            print(f"  [{algo}] 探索済みを読む "
                  f"(PR-AUC {prev['mean_pr_auc']:.4f} / 訓練最終日 {train_to})")
            continue
        print(f"  [{algo}] 探索する: {why or '--force'}")
        t0 = time.time()
        store[algo] = TM.tune(algo, sub, cols, n_trials=N_TRIALS)
        store[algo]["_cv"]["preset"] = preset
        # 途中で落ちても結果を失わないよう、1つ終わるたびに保存する
        TM.save(store)
        print(f"  [{algo}] {time.time()-t0:.0f}秒")
        print()

    print("=" * 96)
    print("探索結果（内側検証の PR-AUC。運用指標ではない）")
    print("=" * 96)
    print(f"  {'モデル':<8}{'PR-AUC':>10}{'±':>9}{'ROC-AUC':>10}{'正例率':>9}  パラメータ")
    for algo in algos:
        cv = store[algo]["_cv"]
        p = store[algo]["params"]
        short = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in list(p.items())[:4])
        print(f"  {algo:<8}{cv['mean_pr_auc']:>10.4f}{cv['std']:>9.4f}"
              f"{cv['mean_roc_auc']:>10.4f}{cv['base_rate']*100:>8.1f}%  {short}")
    print()
    print(f"  保存先: {TM.PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
