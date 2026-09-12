#!/usr/bin/env python3
"""
運用に出すモデルを作る（週次・土曜に実行する想定）。

評価用の train_model.py とは役割が違う:
  train_model.py      … 直近1年を取り分けて「実力を測る」
  train_production.py … 全期間で学習して「明日から使う1本」を作る

出力（research/_data/model/）:
  model.txt   LightGBM のネイティブ形式。日次予測が読む
  meta.json   学習日・訓練期間・パラメータ・特徴量リスト・較正表・スコア帯統計

スコアの読み方を一緒に保存するのが肝
------------------------------------
生スコアは較正されていないので確率として読めない。そのままダッシュボードに
出すと「0.62 だから62%の確率」と誤読される。

そこでウォークフォワードの out-of-fold 予測（＝各行を、その行より前の
データだけで学習したモデルで採点したもの）を集め、スコア帯ごとに

  ・実際の正例率        … 較正確率として読める
  ・実収益の中央値と勝率 … ラベルではなく実際に何%上がったか

を数えて表にする。どちらも実測なので、嘘にならない。

注意: 最終モデルは全期間で学習するので、out-of-fold より訓練量が多く、
スコア分布はわずかにずれる。較正表は近似であって厳密な確率ではない。
meta.json にこの但し書きも入れて、画面から辿れるようにしてある。

  python3 research/train_production.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import sweep_design as S  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
#: モデルは research/_data ではなく research/model に置く。
#: _data は .gitignore されているので、そこに置くと週次の再学習が
#: コミットできず、日次の予測がモデルを見つけられない。
#: モデルは生データと違って「バージョン管理したい成果物」でもある。
MODEL_DIR = os.path.join(HERE, "model")

#: スコア帯の数。10分位。1帯あたり2,000件前後になり、正例率の推定が立つ
N_BANDS = 10

#: out-of-fold を作る窓。walkforward.py と同じ刻み
OOF_MIN_TRAIN_MONTHS = 36
OOF_TEST_MONTHS = 6
OOF_STEP_MONTHS = 6


def oof_scores(ds: pd.DataFrame, cols: List[str], params: Dict) -> pd.DataFrame:
    """
    各行を「その行より前のデータだけで学習したモデル」で採点する。

    較正表とスコア帯統計はこれで作る。全期間で学習したモデルの
    自己採点を使うと、訓練データを当てているだけの楽観的な表になる。
    """
    import lightgbm as lgb
    import walkforward as WF

    folds = WF.make_folds(pd.to_datetime(ds["Date"]),
                          min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    d = pd.to_datetime(ds["Date"])
    parts = []
    for f in folds:
        tr = ds[(d <= pd.Timestamp(f.train_end)) & ds["label"].notna()]
        te = ds[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))
                & ds["label"].notna()]
        if len(te) < 200 or len(tr) < 1000:
            print(f"  窓{f.index}: 件数不足で飛ばす")
            continue
        ytr = tr["label"].to_numpy(dtype=int)
        gbm = lgb.LGBMClassifier(**params,
                                 scale_pos_weight=tuning.scale_pos_weight(ytr))
        gbm.fit(tr[cols].to_numpy(dtype=float), ytr)
        part = te[["Code", "Date", "label", "ref_end", "ref_rise"]].copy()
        part["score"] = gbm.predict_proba(te[cols].to_numpy(dtype=float))[:, 1]
        parts.append(part)
        print(f"  窓{f.index} {f.test_start}〜{f.test_end}: {len(te):,}件を採点")
    if not parts:
        raise SystemExit("out-of-fold を作れません。期間が短すぎます")
    return pd.concat(parts, ignore_index=True)


def score_bands(oof: pd.DataFrame, n_bands: int = N_BANDS) -> Dict:
    """
    スコア帯ごとに「実際にどうだったか」を数える。

    ラベル側（正例率）と実収益側（60営業日後の上昇率）の両方を出す。
    画面ではこの表を引いて「このスコアなら過去はこうだった」と表示する。
    """
    q = pd.qcut(oof["score"], n_bands, labels=False, duplicates="drop")
    rows = []
    for b in sorted(pd.unique(q.dropna())):
        g = oof[q == b]
        end = pd.to_numeric(g["ref_end"], errors="coerce").dropna()
        rows.append({
            "band": int(b),
            "score_lo": round(float(g["score"].min()), 6),
            "score_hi": round(float(g["score"].max()), 6),
            "n": int(len(g)),
            "positive_rate": round(float(g["label"].mean()), 4),
            "end_median": (round(float(end.median()) * 100, 2) if len(end) else None),
            "end_mean": (round(float(end.mean()) * 100, 2) if len(end) else None),
            "win_rate": (round(float((end > 0).mean()), 4) if len(end) else None),
            "n_outcome": int(len(end)),
        })
    return {"n_bands": len(rows), "bands": rows,
            "base_positive_rate": round(float(oof["label"].mean()), 4),
            "base_end_median": round(float(
                pd.to_numeric(oof["ref_end"], errors="coerce").median()) * 100, 2),
            "base_win_rate": round(float(
                (pd.to_numeric(oof["ref_end"], errors="coerce") > 0).mean()), 4),
            "n": int(len(oof))}


def calibration(oof: pd.DataFrame) -> Dict:
    """
    生スコア -> 実際の正例率 の対応を単調回帰で作る。

    帯だけだと段差が出るので、画面で連続に引けるよう折れ線も持たせる。
    isotonic は単調性を保つので「スコアが高いのに確率が下がる」が起きない。
    """
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof["score"].to_numpy(dtype=float), oof["label"].to_numpy(dtype=float))
    xs = np.unique(np.quantile(oof["score"], np.linspace(0, 1, 51)))
    return {"x": [round(float(v), 6) for v in xs],
            "y": [round(float(v), 4) for v in iso.predict(xs)],
            "method": "isotonic",
            "note": ("out-of-fold 予測で作った実測の対応。"
                     "最終モデルは全期間で学習するぶん訓練量が多く、"
                     "スコア分布がわずかにずれるので近似として読むこと")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="運用に出すモデルを作る")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--features", default="all")
    ap.add_argument("--params", default="all", help="使うハイパーパラメータの鍵")
    ap.add_argument("--out-dir", default=MODEL_DIR)
    args = ap.parse_args(argv)

    import lightgbm as lgb

    cols = F.columns(args.features)
    params = tuning.params_for(args.params)
    ds = pd.read_parquet(args.dataset)
    ds["Date"] = pd.to_datetime(ds["Date"])
    # 予測用データセット（ラベル未確定を含む）で学習させない。
    # メタファイルの有無に頼らず、中身で判定する
    n_unlabeled = int(ds["label"].isna().sum())
    if n_unlabeled:
        raise SystemExit(
            f"ラベル未確定が {n_unlabeled:,}件あります。予測用データセットでは"
            "学習できません（--keep-unlabeled を付けずに build_dataset.py を回す）")
    print(f"[load] {len(ds):,}件 / 正例率 {ds['label'].mean()*100:.2f}% "
          f"/ {ds['Date'].min().date()} 〜 {ds['Date'].max().date()}")
    print(f"[setup] 特徴量 {args.features}（{len(cols)}列） / "
          f"パラメータ {args.params}（木{params['n_estimators']}本 "
          f"/ lr {params['learning_rate']:.4f} / 葉 {params['num_leaves']}）")

    # 実収益（ラベル非依存の物差し）。スコア帯統計に使う
    paths = sorted(glob.glob(os.path.join(args.data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    ds = ds.merge(S.reference_outcome(S.Panels(bars).get(B.HIGH_WINDOW)),
                  on=["Code", "Date"], how="left")

    print("\n[oof] スコアの読み方を作るため out-of-fold を計算")
    oof = oof_scores(ds, cols, params)
    print(f"[oof] {len(oof):,}件 / 正例率 {oof['label'].mean()*100:.2f}%")

    bands = score_bands(oof)
    calib = calibration(oof)
    print("\n[band] スコア帯ごとの実績（out-of-fold）")
    print(f"    {'帯':>3}{'件数':>8}{'スコア下限':>12}{'正例率':>9}"
          f"{'実収益の中央値':>14}{'勝率':>8}")
    for r in bands["bands"]:
        print(f"    {r['band']:>3}{r['n']:>8,}{r['score_lo']:>12.4f}"
              f"{r['positive_rate']*100:>8.1f}%{r['end_median']:>+13.2f}%"
              f"{r['win_rate']*100:>7.1f}%")
    print(f"    {'全体':>3}{bands['n']:>8,}{'':>12}"
          f"{bands['base_positive_rate']*100:>8.1f}%"
          f"{bands['base_end_median']:>+13.2f}%{bands['base_win_rate']*100:>7.1f}%")

    print("\n[fit] 全期間で最終モデルを学習")
    y = ds["label"].to_numpy(dtype=int)
    gbm = lgb.LGBMClassifier(**params, scale_pos_weight=tuning.scale_pos_weight(y))
    gbm.fit(ds[cols].to_numpy(dtype=float), y)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.txt")
    gbm.booster_.save_model(model_path)

    meta = {
        "trainedAt": pd.Timestamp.utcnow().isoformat(),
        "label": B.DEFAULT_RISE.name,
        "volNormK": B.VOL_NORM_K,
        "riseHorizon": B.RISE_HORIZON,
        "highWindow": B.HIGH_WINDOW,
        "cooldown": B.BREAKOUT_COOLDOWN,
        "minTradingValue": B.MIN_TRADING_VALUE,
        "preset": args.features,
        "features": cols,
        "params": {k: v for k, v in params.items() if not k.startswith("_")},
        "nTrain": int(len(ds)),
        "trainFrom": str(ds["Date"].min().date()),
        "trainTo": str(ds["Date"].max().date()),
        "positiveRate": round(float(ds["label"].mean()), 4),
        # 画面の「並べているモデル」が追加モデルと同じ列で並べるために使う。
        # 追加モデル（research/models.py）の meta と名前を揃えてある
        "nOof": int(len(oof)),
        "referenceHorizon": S.REF_HORIZON,
        "calibration": calib,
        "scoreBands": bands,
        "tuning": tuning.load_params().get(args.params, {}).get("_cv", {}),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    # out-of-fold そのものも残す。あとから帯の切り方を変えて数え直せる
    oof.to_parquet(os.path.join(args.out_dir, "oof.parquet"),
                   index=False, compression="zstd")
    print(f"\n[done] {model_path} "
          f"({os.path.getsize(model_path)/1e6:.1f}MB)")
    print(f"[done] {os.path.join(args.out_dir, 'meta.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
