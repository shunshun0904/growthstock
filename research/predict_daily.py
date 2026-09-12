#!/usr/bin/env python3
"""
当日の新規ブレイクを採点して、ダッシュボードが読む JSON を書く（日次・終値後）。

  python3 research/build_dataset.py --keep-unlabeled \
      --out research/_data/dataset_predict.parquet
  python3 research/predict_daily.py

出力 public/data/predictions.json
  ・その日の候補一覧（順位・スコア・較正確率・スコア帯の実績）
  ・銘柄ごとの寄与分解（何がスコアを押し上げ／押し下げたか）
  ・モデルの素性（学習日・訓練期間・パラメータ・CV スコア）

出力 public/data/prediction_history.json
  ・過去に出した上位銘柄と、その後の実際の値動き。
    運用でモデルを信じてよいかを確かめられる唯一の材料なので、
    毎回の実行で必ず追記し、既存分の実績を最新の株価で更新する。

スコアの読み方
--------------
生スコアは較正されていないので確率として読めない。train_production.py が
out-of-fold 予測から作った較正表とスコア帯統計を meta.json に持たせてあり、
ここではそれを引いて「このスコア帯は過去に何%が正例で、実際に何%上がったか」を
一緒に出す。数字の出どころを画面まで運ぶ。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import feature_dict as FD  # noqa: E402
import features as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
MODEL_DIR = os.path.join(HERE, "model")
PUBLIC_DIR = os.path.join(os.path.dirname(HERE), "public", "data")

#: 画面に出す寄与の数（正・負それぞれ）
N_CONTRIB = 8

#: 追跡に残す、その日の上位銘柄数
HISTORY_TOP = 5

#: 追跡の保持期間（暦日）。参照ホライズン60営業日 ≒ 87暦日 より長めに持つ
HISTORY_KEEP_DAYS = 400

#: 寄与をまとめる大区分。feature_dict.MACRO と同じ括り方を使う
#: （画面の用語を2箇所で別々に決めると必ずずれる）。
#:
#: ただし地合い（market）だけは独立させる。MACRO では需給とまとめているが、
#: 運用では「今日は全体が低いのか、この銘柄が低いのか」を切り分けたい。
#: 地合い11列はボラ正規化ラベルで効きが大きく（外すと CV PR-AUC
#: 0.4016 -> 0.3337）、混ぜると読めなくなる。
GROUP_JA = {g: ja for ja, groups, _ in FD.MACRO for g in groups}
GROUP_JA["market"] = "地合い（市場環境）"
MARKET_JA = GROUP_JA["market"]


def load_model(model_dir: str):
    import lightgbm as lgb
    path = os.path.join(model_dir, "model.txt")
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(path) or not os.path.exists(meta_path):
        raise SystemExit(
            f"モデルがありません（{path}）。"
            "先に research/train_production.py を実行してください")
    meta = json.load(open(meta_path, encoding="utf-8"))
    return lgb.Booster(model_file=path), meta


def calibrated(score: float, calib: Dict) -> float:
    """較正表（out-of-fold 実測）を線形に引く。範囲外は端に張り付ける。"""
    xs, ys = calib["x"], calib["y"]
    return float(np.interp(score, xs, ys, left=ys[0], right=ys[-1]))


def band_of(score: float, bands: Dict) -> Dict:
    """スコアがどの帯に入るか。帯は out-of-fold の10分位。"""
    for r in bands["bands"]:
        if score <= r["score_hi"]:
            return r
    return bands["bands"][-1]


def name_map(data_dir: str) -> pd.DataFrame:
    """銘柄名・業種・市場。最新の master を使う。"""
    path = os.path.join(data_dir, "master.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Code", "CoName", "S33Nm", "MktNm", "ScaleCat"])
    m = pd.read_parquet(path)
    keep = [c for c in ("Code", "CoName", "S33Nm", "MktNm", "ScaleCat")
            if c in m.columns]
    return m[keep].drop_duplicates("Code")


def contributions(booster, X: np.ndarray, cols: List[str]) -> List[Dict]:
    """
    TreeSHAP による寄与分解。LightGBM の pred_contrib で厳密に出る。

    値は対数オッズ空間の寄与。確率の差ではないので、画面では
    「押し上げ / 押し下げ」の相対の大きさとして読ませる。
    """
    raw = booster.predict(X, pred_contrib=True)
    col_group = F.column_groups()
    out = []
    for i in range(raw.shape[0]):
        c = raw[i, :-1]
        base = float(raw[i, -1])
        order = np.argsort(-np.abs(c))
        top = [{"col": cols[j], "ja": FD.COL_JA.get(cols[j], ""),
                "contrib": round(float(c[j]), 4)}
               for j in order[:N_CONTRIB * 2] if abs(c[j]) > 1e-9]
        groups: Dict[str, float] = {}
        for j, col in enumerate(cols):
            key = GROUP_JA.get(col_group.get(col, ""), "その他")
            groups[key] = groups.get(key, 0.0) + float(c[j])
        total = sum(abs(v) for v in groups.values()) or 1.0
        out.append({
            "base": round(base, 4),
            "top": top,
            "groups": {k: round(v, 4) for k, v in
                       sorted(groups.items(), key=lambda kv: -abs(kv[1]))},
            # 今日のスコアのうち地合いがどれだけを占めるか。
            # 「全体が低い日」と「この銘柄が弱い」を切り分けるため
            "marketContrib": round(groups.get(MARKET_JA, 0.0), 4),
            "marketShare": round(abs(groups.get(MARKET_JA, 0.0)) / total * 100, 1),
            "stockContrib": round(sum(v for k, v in groups.items()
                                      if k != MARKET_JA), 4),
        })
    return out


def ref_returns(panel: pd.DataFrame, picks: pd.DataFrame) -> pd.Series:
    """
    過去に出した銘柄の「その後どうなったか」を最新の終値で測る。

    まだ参照ホライズンに達していない銘柄も、途中経過として出す
    （達していないことは daysElapsed で分かるようにする）。
    """
    last = panel.groupby("Code")["close"].last()
    base = picks.set_index(["Code", "Date"]).index
    cur = picks["Code"].map(last)
    return (cur / picks["closeAtPick"] - 1.0) * 100.0


def score_others(cand: pd.DataFrame, cols: List[str],
                 model_dir: str) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    """
    基準モデル以外でも採点する。

    アンサンブルはしない。5つのスコアを混ぜて1つにはせず、画面に並べて
    人間が統合判断する。実測で lgbm と logit のスコア相関は 0.412、
    上位10%の重複は12%しかなく、ほぼ別の銘柄を選んでいる。

    モデル間で生スコアは比較できない（学習器が違えばスケールも意味も違う）。
    画面に出すのは各モデル自身の過去スコア分布での位置（pctHistorical）。

    1モデルの読込や採点が失敗しても他を止めない。週次学習で1つだけ
    転んだ日に、日次予測まで落とさないため。
    """
    import models as M

    X = cand[cols].to_numpy(dtype=float)
    scores: Dict[str, np.ndarray] = {}
    info: List[Dict] = []
    for algo in M.available(model_dir):
        try:
            model, meta = M.load(algo, model_dir)
            if model is None:
                continue
            want = meta.get("features") or cols
            if list(want) != list(cols):
                # 特徴量セットが違うモデルは、同じ X を食わせられない。
                # 黙って別の列で採点すると意味のないスコアが画面に出る
                print(f"  [{algo}] 特徴量が一致しないため飛ばす "
                      f"({len(want)}列 vs {len(cols)}列)")
                continue
            scores[algo] = M.predict(model, X)
            info.append({
                "algo": algo, "name": meta.get("name", algo),
                "note": meta.get("note", ""),
                "trainedAt": meta.get("trainedAt"),
                "nOof": meta.get("nOof"),
                "scoreBands": meta.get("scoreBands", {}),
                "cv": meta.get("tuning", {}),
            })
            print(f"  [{algo}] 採点 {len(cand):,}件")
        except Exception as exc:          # noqa: BLE001
            print(f"  [{algo}] 失敗: {type(exc).__name__}: {exc}")
    return scores, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="当日のブレイク候補を採点する")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset",
                    default=os.path.join(DATA_DIR, "dataset_predict.parquet"))
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--out-dir", default=PUBLIC_DIR)
    ap.add_argument("--days", type=int, default=5,
                    help="直近何営業日ぶんを出すか（画面で日を切り替えられる）")
    ap.add_argument("--top-codes", type=int, default=10,
                    help="詳細（スナップショット・株価履歴）を取りに行く上位何銘柄か")
    args = ap.parse_args(argv)

    booster, meta = load_model(args.model_dir)
    cols = meta["features"]
    ds = pd.read_parquet(args.dataset)
    ds["Date"] = pd.to_datetime(ds["Date"])
    missing = [c for c in cols if c not in ds.columns]
    if missing:
        raise SystemExit(f"データセットに無い特徴量: {missing[:10]}")

    days = sorted(ds["Date"].unique())[-args.days:]
    cand = ds[ds["Date"].isin(days)].copy()
    if cand.empty:
        raise SystemExit("候補がありません")
    print(f"[load] モデル学習日 {meta['trainedAt'][:10]} / "
          f"訓練 {meta['nTrain']:,}件（{meta['trainFrom']}〜{meta['trainTo']}）")
    print(f"[cand] 直近{len(days)}営業日 / 候補 {len(cand):,}件 "
          f"（{pd.Timestamp(days[0]).date()} 〜 {pd.Timestamp(days[-1]).date()}）")

    X = cand[cols].to_numpy(dtype=float)
    cand["score"] = booster.predict(X)
    contrib = contributions(booster, X, cols)

    # 到達に必要な上昇率（この銘柄が正例と呼ばれる水準）。
    # ラベル定義と同じ関数から作る
    need, _ = B.rise_thresholds(cand["vol_20d"], B.DEFAULT_RISE)
    cand["need_pct"] = need.to_numpy() * 100.0

    names = name_map(args.data_dir)
    cand = cand.merge(names, on="Code", how="left")

    # 日付内の順位と、訓練期間のスコア分布での位置
    cand["rank_in_day"] = cand.groupby("Date")["score"].rank(ascending=False,
                                                             method="min")
    cand["n_in_day"] = cand.groupby("Date")["score"].transform("size")
    oof_path = os.path.join(args.model_dir, "oof.parquet")
    hist_scores = (pd.read_parquet(oof_path, columns=["score"])["score"].to_numpy()
                   if os.path.exists(oof_path) else None)

    def r(v, d=2):
        return None if v is None or not np.isfinite(v) else round(float(v), d)

    rows = []
    for i, (_, s) in enumerate(cand.iterrows()):
        sc = float(s["score"])
        band = band_of(sc, meta["scoreBands"])
        jq = str(s["Code"])
        rows.append({
            "code": jq[:4], "jqCode": jq,
            "name": s.get("CoName") or None,
            "sector": s.get("S33Nm") or None,
            "market": s.get("MktNm") or None,
            "scale": (s.get("ScaleCat") if isinstance(s.get("ScaleCat"), str)
                      else None),
            "date": pd.Timestamp(s["Date"]).date().isoformat(),
            "score": round(sc, 6),
            "rankInDay": int(s["rank_in_day"]), "nInDay": int(s["n_in_day"]),
            "pctInDay": r((1 - (s["rank_in_day"] - 1) / max(1, s["n_in_day"])) * 100, 1),
            "pctHistorical": (r(float((hist_scores < sc).mean()) * 100, 1)
                              if hist_scores is not None else None),
            "band": band["band"],
            "calibProb": r(calibrated(sc, meta["calibration"]) * 100, 1),
            "bandPositiveRate": r(band["positive_rate"] * 100, 1),
            "bandEndMedian": band["end_median"],
            "bandWinRate": r((band["win_rate"] or 0) * 100, 1),
            # --- エントリー判断に使う素の値 --- #
            "close": r(s.get("close_raw"), 1),
            "marketCap": r(s.get("market_cap")),
            "tradingValue": r(s.get("tv_ma20")),
            "vol20d": r(s.get("vol_20d")),
            "needPct": r(s.get("need_pct"), 1),
            "ret20d": r(s.get("ret_20d")),
            "rHigh": r(s.get("r_high")),
            "breakMargin": r(s.get("break_margin")),
            "baseLength": r(s.get("base_length"), 0),
            "creditRatio": r(s.get("credit_ratio")),
            "per": r(s.get("per")), "pbr": r(s.get("pbr")),
            "divYield": r(s.get("div_yield")),
            "epsGrowth": r(s.get("eps_growth_q0")),
            "salesGrowth": r(s.get("sales_growth_q0")),
            "roe": r(s.get("ROE_q0")),
            "opMargin": r(s.get("op_margin_q0")),
            "volumeTrend": r(s.get("volume_trend")),
            "progressRate": r(s.get("progress_vs_base")),
            "high52w": r(s.get("high52w"), 1),
            # --- 内部挙動 --- #
            "contrib": contrib[i],
        })
    # --- 他モデルの採点を各候補に載せる --- #
    # アンサンブルはしない。並べるだけ。買うかの判断は人間が統合的に行う
    others, model_info = score_others(cand, cols, args.model_dir)
    hist_by_algo = {}
    if others:
        import models as M
        for algo in others:
            hist_by_algo[algo] = M.hist_scores(algo, args.model_dir)
        for i, x in enumerate(rows):
            per = {}
            for algo, sc in others.items():
                v = float(sc[i])
                per[algo] = {
                    "score": round(v, 6),
                    # モデルごとに別の過去分布で位置を出す。
                    # 共通の分布を使うとスケールの違いが混ざる
                    "pctHistorical": M.pct_historical(v, hist_by_algo[algo]),
                }
            x["byModel"] = per
            pcts = [d["pctHistorical"] for d in per.values()
                    if d["pctHistorical"] is not None]
            # 何個のモデルが「上位10%」と見ているか。一致度の目安
            x["agree90"] = int(sum(1 for p in pcts if p >= 90))
            x["nModels"] = len(pcts)
            # 表示順に使う平均。予測値ではない（アンサンブルではない）
            x["pctMean"] = round(float(np.mean(pcts)), 1) if pcts else None

    # 表示順。モデル別の平均パーセンタイルがあればそれで、無ければ基準モデル。
    # これは並べ方の都合で、統合された予測値という意味ではない
    def order_key(x):
        return x["pctMean"] if x.get("pctMean") is not None else x["score"] * 100
    rows.sort(key=lambda x: (x["date"], order_key(x)), reverse=True)

    payload = {
        "generatedAt": pd.Timestamp.utcnow().isoformat(),
        "asOf": pd.Timestamp(days[-1]).date().isoformat(),
        "dates": [pd.Timestamp(d).date().isoformat() for d in days],
        "candidates": rows,
        "model": {
            "trainedAt": meta["trainedAt"], "label": meta["label"],
            "volNormK": meta["volNormK"], "riseHorizon": meta["riseHorizon"],
            "highWindow": meta["highWindow"], "preset": meta["preset"],
            "nFeatures": len(cols), "nTrain": meta["nTrain"],
            "trainFrom": meta["trainFrom"], "trainTo": meta["trainTo"],
            "positiveRate": meta["positiveRate"],
            "minTradingValue": meta["minTradingValue"],
            "cv": meta.get("tuning", {}),
        },
        "scoreBands": meta["scoreBands"],
        "calibration": meta["calibration"],
        # 画面に並べるモデルの素性。基準モデル(lgbm)も含む
        "models": model_info,
        "notes": [
            "スコアは較正されていない生の出力。確率として読まず、"
            "同じ日の候補の中での順位と、スコア帯の過去実績で読むこと。",
            "スコア帯の実績は out-of-fold 予測（その行より前のデータだけで"
            "学習したモデルの採点）で数えた実測値。",
            "寄与は TreeSHAP による対数オッズ空間の分解。"
            "確率の差ではないので、押し上げ／押し下げの相対の大きさとして読む。",
            "モデル別の値は混ぜていない（アンサンブルではない）。"
            "学習器が違えばスコアのスケールも意味も違うので、"
            "各モデル自身の過去スコア分布での位置に揃えて並べてある。",
            "一致度は「上位10%と見ているモデルの数」。"
            "表示順の平均パーセンタイルは並べ方の都合であって、"
            "統合された予測値ではない。",
        ],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "predictions.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"[done] {out_path} ({os.path.getsize(out_path)/1e3:.0f}KB)")

    update_history(args, rows, days)

    # その日の上位コードを素のテキストで出す。
    # 取得スクリプト（scripts/jquants_data_fetcher.py --extra-codes）へ渡して、
    # スナップショットと株価履歴を作らせるため。これが無いと
    # タイムマシーン（過去比較）が予測候補で使えない。
    latest = pd.Timestamp(days[-1]).date().isoformat()
    top = sorted((x for x in rows if x["date"] == latest),
                 key=lambda x: x["rankInDay"])[:args.top_codes]
    codes_path = os.path.join(args.data_dir, "top_codes.txt")
    with open(codes_path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(x["code"] for x in top))
    print(f"[done] {codes_path} （{len(top)}銘柄: "
          f"{' '.join(x['code'] for x in top)}）")
    return 0


def update_history(args, rows: List[Dict], days) -> None:
    """
    その日の上位を追跡ファイルに足し、既存分の「その後」を最新株価で更新する。

    運用でモデルを信じてよいかを確かめられる唯一の材料。
    予測した時点の終値を残しておき、毎回の実行で現在値と比べ直す。
    """
    path = os.path.join(args.out_dir, "prediction_history.json")
    hist = {"entries": []}
    if os.path.exists(path):
        try:
            hist = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 追跡ファイルを読めないので作り直す: {e}")

    latest = pd.Timestamp(days[-1]).date().isoformat()
    existing = {(e["date"], e["jqCode"]) for e in hist.get("entries", [])}
    added = 0
    for x in rows:
        if x["date"] != latest or x["rankInDay"] > HISTORY_TOP:
            continue
        if (x["date"], x["jqCode"]) in existing:
            continue
        hist.setdefault("entries", []).append({
            "date": x["date"], "jqCode": x["jqCode"], "code": x["code"],
            "name": x["name"], "rank": x["rankInDay"], "nInDay": x["nInDay"],
            "score": x["score"], "band": x["band"],
            "calibProb": x["calibProb"], "needPct": x["needPct"],
            "closeAtPick": x["close"],
        })
        added += 1

    # 既存分の現在値を更新する
    paths = sorted(glob.glob(os.path.join(args.data_dir, "bars_*.parquet")))
    if paths and hist.get("entries"):
        bars = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "C"])
                          for p in paths[-2:]], ignore_index=True)
        bars["Date"] = pd.to_datetime(bars["Date"])
        last = bars.sort_values("Date").groupby("Code")["C"].last()
        asof = bars["Date"].max()
        for e in hist["entries"]:
            cur = last.get(e["jqCode"])
            if cur is None or not np.isfinite(cur) or not e.get("closeAtPick"):
                continue
            e["closeNow"] = round(float(cur), 1)
            e["returnPct"] = round(float(cur) / e["closeAtPick"] * 100 - 100, 2)
            e["daysElapsed"] = int(np.busday_count(
                np.datetime64(e["date"]), np.datetime64(asof.date())))
        hist["asOf"] = str(asof.date())

    # 古すぎる分は落とす（参照ホライズンを大きく超えたら追跡の意味が薄い）
    cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=HISTORY_KEEP_DAYS)).date()
    hist["entries"] = [e for e in hist.get("entries", [])
                       if pd.Timestamp(e["date"]).date() >= cutoff]
    hist["entries"].sort(key=lambda e: (e["date"], e["rank"]), reverse=True)
    hist["updatedAt"] = pd.Timestamp.utcnow().isoformat()
    hist["referenceHorizon"] = B.RISE_HORIZON
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(hist, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"[done] {path} （{added}件追加 / 累計 {len(hist['entries'])}件）")


if __name__ == "__main__":
    sys.exit(main())
