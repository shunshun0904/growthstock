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
  ・候補ごとの次の決算発表の予定（next_earnings。予測日までに公表済みのもの）と、前回の決算開示からの日数
  ・直近5営業日を画面で切り替えられる。**過去の日は、その日に出した値のまま**（freeze_past_days。
    運用者の指示 2026-10-12「過去の日は最初に出した値のまま固定してください」）。採点するのは新しい日だけで、
    日ごとにどのモデル（学習日）でいつ出した値かを dayInfo に残す

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import build_dataset as B  # noqa: E402
import feature_dict as FD  # noqa: E402
import features as F  # noqa: E402
import jquants_data_fetcher as JF  # noqa: E402
import trading_calendar as TC  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
MODEL_DIR = os.path.join(HERE, "model")
PUBLIC_DIR = os.path.join(os.path.dirname(HERE), "public", "data")

#: 画面に出す寄与の数（正・負それぞれ）
N_CONTRIB = 8

#: 追跡に残す、その日の上位銘柄数
HISTORY_TOP = 5

#: 詳細（スナップショット・株価履歴）を取りに行く銘柄数の上限（--top-codes の既定）。
#: 画面の銘柄データだけを作り直すとき（research/refresh_details.py）も同じ値で選ぶ
TOP_CODES = 120

#: 追跡の保持期間（暦日）。参照ホライズン（目的変数と同じ営業日数）より
#: 十分長く持つ。20営業日 ≒ 29暦日、60営業日 ≒ 87暦日
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


def next_earnings(keys, data_dir: str, cal=None) -> Dict[Tuple[str, str], Dict]:
    """
    候補ごとの次の決算発表予定（運用者の依頼 2026-10-13「予測が実質的に、次の決算の予測になっていないか確認するために、
    次の決算発表までの日数もどこかに表示してほしい」）。

    特徴量 days_to_earn（extra_features.earnings_ahead）と同じ規則: 予測日までに公表済み（PubDate ≤ 予測日）で、
    予測日以降（SchDate ≥ 予測日）の予定のうち一番早いもの。予定日は J-Quants の決算発表予定日（/fins/earnings-date）。
    keys: (jqCode, ISO の日付) の並び。戻り値: {(jqCode, 日付): {"date", "quarter", "days", "tradingDays"}}。
      days = 暦日、tradingDays = 予測日から発表日までの営業日（予測日の翌営業日に発表なら 1、予測日に発表なら 0。
      カレンダーが覆わない先は平日で数える）。予定が公表されていない銘柄は入れない。
    """
    import extra_features as XF

    d = XF._read("earndate", data_dir, columns=["PubDate", "SchDate", "FQName", "Code"])
    if d.empty or not {"PubDate", "SchDate", "Code"} <= set(d.columns):
        return {}
    d = d.copy()
    d["Code"] = d["Code"].astype(str)
    d["PubDate"] = pd.to_datetime(d["PubDate"], errors="coerce")
    d["SchDate"] = pd.to_datetime(d["SchDate"], errors="coerce")
    d = d.dropna(subset=["PubDate", "SchDate", "Code"])
    left = pd.DataFrame(list(dict.fromkeys((str(c), str(t)) for c, t in keys)), columns=["Code", "Date"])
    if left.empty or d.empty:
        return {}
    left["_d"] = pd.to_datetime(left["Date"])
    j = left.merge(d, on="Code", how="inner")
    j = j[(j["PubDate"] <= j["_d"]) & (j["SchDate"] >= j["_d"])]
    if j.empty:
        return {}
    j = j.sort_values(["Code", "_d", "SchDate"]).drop_duplicates(["Code", "Date"], keep="first")
    out = {}
    for code, day, dd, sd, q in zip(j["Code"], j["Date"], j["_d"], j["SchDate"], j.get("FQName", [None] * len(j))):
        d0, d1 = dd.date(), sd.date()
        out[(code, day)] = {"date": d1.isoformat(), "quarter": (str(q) if q is not None and not pd.isna(q) else None),
                            "days": int((d1 - d0).days), "tradingDays": int(TC.elapsed_days(d0, d1, cal))}
    return out


def feature_desc(col: str) -> str:
    """特徴量の説明（feature_dict.describe）。説明が無い列は空。"""
    d = FD.describe(col)
    return "" if d == "（説明未登録）" else d.replace("**", "")


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
        # name: 画面に出す短い日本語名、ja: 説明（どちらも research/feature_dict.py。決算の軸 × 接尾辞の列にも付く）
        top = [{"col": cols[j], "name": FD.name_ja(cols[j]), "ja": feature_desc(cols[j]),
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


def agreement(per: Dict[str, Dict], algos, pct: float = 90.0) -> Tuple[int, int]:
    """algos のうち百分位が付いたモデルの数と、そのうち pct 以上と見たモデルの数（独立した判定の数え上げ）。"""
    pcts = [per[a]["pctHistorical"] for a in algos
            if a in per and per[a].get("pctHistorical") is not None]
    return int(sum(1 for p in pcts if p >= pct)), len(pcts)


# --------------------------------------------------------------------------- #
# 12本の合意（運用者の指示 2026-10-12「ダッシュボードに12本の合意の百分位を表示」）
# --------------------------------------------------------------------------- #
#
# 合意 = 本番の4モデルとラベル違いの8本の百分位（pctHistorical と同じ「そのモデル自身の過去スコア分布での位置」）
# の平均。合意の百分位 = その平均が、過去の out-of-fold の行の合意の分布で下から何%か（97.5 以上 = 過去の上位2.5%、
# 90 以上 = 上位10%）。実験79〜81（docs/MODEL_ADOPTION_RULES.md §35〜37）で、12本の値の中で判断に効いたのは
# 「12本がそろって高いか」（合意の高さ）だけだった。順位・帯・較正・追跡ファイルは基準モデルのまま。
# 同じ日の運用者の指示（「12本の平均ではなく、12個全てを載せてほしい」）で、画面は12本の百分位をそのまま表で出す
# 形に変え、合意は画面に出さなくなった。値は確認・研究用に predictions.json に残す（§38）。2026-10-24 に 1.8σ の
# 4本を外したので、それからの合意は LINEUP の8本（1.2σ と 1.5σ の各4本）の平均。

def oof_scores_of(algo: str, model_dir: str) -> Optional[pd.DataFrame]:
    """そのモデルの out-of-fold（Code, Date, score）。基準モデルは model_dir/oof.parquet、ほかは models/<名前>/。"""
    import models as M
    p = (os.path.join(model_dir, "oof.parquet") if algo == M.BASELINE
         else os.path.join(M.sub_dir(algo, model_dir), "oof.parquet"))
    if not os.path.exists(p):
        return None
    o = pd.read_parquet(p, columns=["Code", "Date", "score"])
    o["Code"] = o["Code"].astype(str)
    o["Date"] = pd.to_datetime(o["Date"])
    return o


def consensus_reference(model_dir: str, algos) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    過去の out-of-fold の行（algos の全部で (Code, Date) がそろった行）の合意を昇順で返す。各行の百分位は
    そのモデルの out-of-fold 全行の分布で付ける（画面の pctHistorical と同じ: 同点は数えない）。
    out-of-fold が無いモデルは外し、使ったモデルを返す。
    """
    wide, used = None, []
    for a in algos:
        o = oof_scores_of(a, model_dir)
        if o is None or o["score"].notna().sum() == 0:
            continue
        o = o.dropna(subset=["score"])
        ref = np.sort(o["score"].to_numpy(dtype=float))
        o = o[["Code", "Date"]].assign(
            **{a: np.searchsorted(ref, o["score"].to_numpy(dtype=float), side="left") / len(ref) * 100.0})
        o = o.drop_duplicates(["Code", "Date"])
        wide = o if wide is None else wide.merge(o, on=["Code", "Date"], how="inner")
        used.append(a)
    if wide is None or wide.empty:
        return None, []
    return np.sort(wide[used].mean(axis=1).to_numpy(dtype=float)), used


def consensus_fields(pcts: Dict[str, Optional[float]], ref: Optional[np.ndarray], used: List[str]) -> Dict:
    """
    候補1件の合意（cons12。used の百分位の平均）と、それが過去の合意の分布で下から何%か（consPct）。
    pcts は丸める前の百分位。used の全部がそろわなければ None（欠けたまま平均すると別の物差しになる）。
    """
    n = len(used)
    vals = [pcts.get(a) for a in used]
    if ref is None or not len(ref) or not n or any(v is None or not np.isfinite(v) for v in vals):
        return {"cons12": None, "consPct": None, "nCons": n}
    c = float(np.mean(vals))
    return {"cons12": round(c, 1),
            "consPct": round(float(np.searchsorted(ref, c, side="left")) / len(ref) * 100.0, 1),
            "nCons": n}


def score_others(cand: pd.DataFrame, cols: List[str],
                 model_dir: str) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    """
    基準モデル以外でも採点する。

    アンサンブルはしない。各モデルのスコアを混ぜて1つにはせず、画面に並べて
    人間が統合判断する（models.ALGOS。2026-10-09 夜からは木3つ + ロジスティック回帰。MLP は外した）。

    モデル間で生スコアは比較できない（学習器が違えばスケールも意味も違う）。
    画面に出すのは各モデル自身の過去スコア分布での位置（pctHistorical）。

    1モデルの読込や採点が失敗しても他を止めない。週次学習で1つだけ
    転んだ日に、日次予測まで落とさないため。

    各モデルは**自分が学習した列**（meta の features。並びもそのとき）で採点する。列の組を切り替える週は
    土曜に 1.5σ、日曜に 1.2σ を学習し直すので、どちらかが失敗するとモデルごとに列が違う週ができる
    （2026-10-17、241列への切り替えで足した。1.8σ は 2026-10-24 に外した）。学習した列がデータに無いモデルだけ飛ばす。
    """
    import models as M

    scores: Dict[str, np.ndarray] = {}
    info: List[Dict] = []
    for algo in M.available(model_dir):
        if algo == M.BASELINE:
            # 基準モデルはここで採点しない。呼び出し側が本番モデル
            # （model.txt）のスコアをそのまま byModel に入れる。
            # 同じ LightGBM の別の当てはめを並べると、上位10%の重複が
            # 52.8% しかないため、画面に矛盾した2本が出る
            continue
        try:
            model, meta = M.load(algo, model_dir)
            if model is None:
                continue
            want = list(meta.get("features") or cols)
            miss = [c for c in want if c not in cand.columns]
            if miss:
                # 学習した列が無いまま別の列で採点すると、意味のないスコアが画面に出る
                print(f"  [{algo}] 学習した列のうち {len(miss)}列がデータに無いため飛ばす（例: {', '.join(miss[:3])}）")
                continue
            if want != list(cols):
                print(f"  [{algo}] 学習した {len(want)}列で採点（基準モデルは {len(cols)}列）")
            scores[algo] = M.predict(model, cand[want].to_numpy(dtype=float))
            info.append({
                "algo": algo, "name": meta.get("name", algo),
                "short": M.SHORT.get(algo, algo[:3].upper()),
                "note": meta.get("note", ""),
                "trainedAt": meta.get("trainedAt"),
                "nOof": meta.get("nOof"),
                "scoreBands": meta.get("scoreBands", {}),
                "cv": meta.get("tuning", {}),
                # ラベル違いのモデル（models.ACTIVE_VARIANTS。1.5σ の4本）の目印。画面は 1.2σ の一致（agree90）に数えず、並べて見せる
                "variant": M.label_k(algo) is not None,
                "labelK": meta.get("volNormK"),
                "label": meta.get("label"),
                "positiveRate": meta.get("positiveRate"),
            })
            print(f"  [{algo}] 採点 {len(cand):,}件")
        except Exception as exc:          # noqa: BLE001
            print(f"  [{algo}] 失敗: {type(exc).__name__}: {exc}")
    return scores, info


def attach_ltr(rows: List[Dict], cand: pd.DataFrame, model_dir: str) -> Optional[Dict]:
    """
    ランキング学習（research/ltr.py）で候補を採点し、各行に ltr（score・その日の中の百分位 dayPct・順位 dayRank・その日の
    候補数 nDay・上半分の印 upper・過去の分布での位置 pctHistorical）を足す。画面と実運用の追跡が「両方使う形」（今の規則
    かつ その日の中でランキング学習が上半分）を参考に並べるためのもの（運用者の選択「まず並べて見る」2026-10-24。
    docs/MODEL_ADOPTION_RULES.md §78-9）。**買いの規則には使わない。**

    rows は cand と同じ順（1行 = cand の1行）。その日の中の順位は、その日の候補すべての中で付ける（実験114b と同じ）。
    モデルが無い・学習した列がデータに無い・失敗したときは何も足さない（予測は止めない。score_others と同じ考え方）。
    """
    import ltr as LTR
    import models as M

    try:
        model, meta = LTR.load(model_dir)
        if model is None:
            print("[ltr] ランキング学習のモデルが無いので出さない")
            return None
        want = list(meta.get("features") or [])
        miss = [c for c in want if c not in cand.columns]
        if not want or miss:
            print(f"[ltr] 学習した列のうち {len(miss)}列がデータに無いので出さない")
            return None
        sc = LTR.predict(model, cand[want].to_numpy(dtype=float))
        fields = LTR.day_fields(cand["Date"], sc, M.hist_scores(LTR.NAME, model_dir))
        if len(fields) != len(rows):
            print(f"[ltr] 行の数が合わない（候補 {len(rows)} / 採点 {len(fields)}）ので出さない")
            return None
        for x, f in zip(rows, fields):
            x["ltr"] = f
        print(f"[ltr] 採点 {len(rows):,}件（学習日 {str(meta.get('trainedAt'))[:10]}）・その日の中で上半分 "
              f"{sum(1 for f in fields if f['upper']):,}件")
        return {"algo": LTR.NAME, "name": meta.get("name", LTR.JA), "short": meta.get("short", LTR.SHORT),
                "note": meta.get("note", LTR.NOTE), "trainedAt": meta.get("trainedAt"), "nOof": meta.get("nOof"),
                "dayUpper": round(float(meta.get("dayUpper", LTR.DAY_UPPER)) * 100, 1),
                "oofMetrics": meta.get("oofMetrics", {})}
    except Exception as exc:          # noqa: BLE001
        print(f"[ltr] 失敗（出さない）: {type(exc).__name__}: {exc}")
        return None


def progress_fields(data_dir: str, keys) -> Dict[Tuple[str, str], dict]:
    """
    候補（銘柄コード5桁, 'YYYY-MM-DD'）ごとに、画面の「進捗期待」に使う値を出す。

    画面のデータ取得（scripts/jquants_data_fetcher.py）と同じ関数に、その日までの
    開示だけを渡す。データセットの progress_vs_base は「進捗率 − Q×25」で、
    画面の進捗率とは別物（2026-09-25 まで progressRate としてそのまま渡していて、
    予測タブからオクタゴンへ送った銘柄の「進捗期待」の指標値が違う数字だった）。
    """
    keys = sorted({(str(c), str(d)[:10]) for c, d in keys})
    paths = sorted(glob.glob(os.path.join(data_dir, "fins_*.parquet")))
    if not keys or not paths:
        return {}
    codes = sorted({c for c, _ in keys})
    f = pd.concat([pd.read_parquet(p, filters=[("Code", "in", codes)]) for p in paths],
                  ignore_index=True)
    if f.empty:
        return {}
    for c in ("DiscDate", "CurFYSt", "CurPerEn"):
        if c in f.columns:
            f[c] = pd.to_datetime(f[c], errors="coerce").dt.strftime("%Y-%m-%d")
    f = f.astype(object).where(f.notna(), None)
    by_code = {c: g.to_dict("records") for c, g in f.groupby("Code", sort=False)}
    out = {}
    for code, day in keys:
        rows = by_code.get(code)
        if not rows:
            continue
        m = JF.fundamentals_as_of(rows, day)
        out[(code, day)] = {k: m.get(k) for k in
                            ("progressRate", "quarter", "progressBenchmark", "progressBasis")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="当日のブレイク候補を採点する")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset",
                    default=os.path.join(DATA_DIR, "dataset_predict.parquet"))
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--out-dir", default=PUBLIC_DIR)
    ap.add_argument("--days", type=int, default=5,
                    help="直近何営業日ぶんを出すか（画面で日を切り替えられる）")
    ap.add_argument("--top-codes", type=int, default=TOP_CODES,
                    help="詳細（スナップショット・株価履歴）を取りに行く銘柄数の"
                         "上限。画面に出ている候補を新しい日・上位から順に埋める。"
                         "1銘柄あたり4リクエストなので、120で約480リクエスト")
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

    prog = progress_fields(args.data_dir, zip(cand["Code"].astype(str),
                                              cand["Date"].dt.strftime("%Y-%m-%d")))

    rows = []
    for i, (_, s) in enumerate(cand.iterrows()):
        sc = float(s["score"])
        band = band_of(sc, meta["scoreBands"])
        jq = str(s["Code"])
        pf = prog.get((jq, pd.Timestamp(s["Date"]).date().isoformat()), {})
        rows.append({
            "code": JF.display_code(jq), "jqCode": jq,   # 普通株以外は5桁のまま（25935 など）
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
            # 帯の実収益。翌営業日の寄り買い・lab.OUTCOME の営業日数で測った値。
            # 何の数字かは scoreBands.outcome に入っている（画面で焼き込まない）。
            #
            # end_median を見に行くのは、物差しを変える前に学習したモデルが
            # Release に残っているあいだのため。日次予測はモデルを差し替えない
            # ので、週次の再学習が走るまでは古い meta（当日終値買い・60営業日）
            # のままになる。ここで拾わないと画面の列が数日ぶん空になる。
            # 再学習が一巡したら消してよい。
            "bandOutcome": band.get("outcome_median", band.get("end_median")),
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
            # 画面の「進捗期待」。データ取得と同じ関数で出す（progress_fields）
            "progressRate": r(pf.get("progressRate")),
            "quarter": pf.get("quarter"),
            "progressBenchmark": r(pf.get("progressBenchmark")),
            "progressBasis": pf.get("progressBasis"),
            "high52w": r(s.get("high52w"), 1),
            # 前回の決算開示からの日数（特徴量 days_since_disc。暦日・上限400）。次の決算の予定が未公表のときの手がかり
            "daysSinceDisc": r(s.get("days_since_disc"), 0),
            # --- 内部挙動 --- #
            "contrib": contrib[i],
        })
    # --- 他モデルの採点を各候補に載せる --- #
    # アンサンブルはしない。並べるだけ。買うかの判断は人間が統合的に行う
    others, model_info = score_others(cand, cols, args.model_dir)
    consensus_meta = None
    if others:
        import models as M
        hist_by_algo = {a: M.hist_scores(a, args.model_dir) for a in others}
        # 基準モデルを画面の並びの先頭に置く。採点し直さず、上で計算した
        # 本番モデルのスコアとパーセンタイルをそのまま使う。これで
        # 順位・帯・SHAP と、画面の LightGBM の棒が必ず一致する
        model_info.insert(0, {
            "algo": M.BASELINE, "name": M.JA.get(M.BASELINE, M.BASELINE),
            "short": M.SHORT.get(M.BASELINE, "LGB"),
            "note": M.NOTE.get(M.BASELINE, ""),
            "trainedAt": meta["trainedAt"],
            # 旧い meta.json には nOof が無い（追加したのは後）。
            # スコア帯の件数が同じものなので、そちらから拾う
            "nOof": meta.get("nOof") or meta.get("scoreBands", {}).get("n"),
            "scoreBands": meta.get("scoreBands", {}),
            "cv": meta.get("tuning", {}),
            "baseline": True,
            "variant": False,
            "labelK": meta.get("volNormK"),
            "label": meta.get("label"),
            "positiveRate": meta.get("positiveRate"),
        })
        # 合意の物差し（過去の out-of-fold の合意の分布）。並びは画面と同じ LINEUP の順（2026-10-24 から8本。それまでは12本）。
        # 作れなくても日次予測は止めない（合意を出さないだけ。1モデルの失敗で止めない score_others と同じ考え方）
        try:
            cons_ref, cons_used = consensus_reference(
                args.model_dir, [a for a in M.LINEUP if a == M.BASELINE or a in others])
        except Exception as exc:          # noqa: BLE001
            print(f"[consensus] 物差しを作れないので合意は出さない: {type(exc).__name__}: {exc}")
            cons_ref, cons_used = None, []
        sorted_hist = {a: (np.sort(h) if h is not None and len(h) else None)
                       for a, h in [(M.BASELINE, hist_scores)] + list(hist_by_algo.items())}

        def raw_pct(a, v):
            h = sorted_hist.get(a)
            if h is None or not np.isfinite(v):
                return None
            return float(np.searchsorted(h, v, side="left")) / len(h) * 100.0

        base_scores = cand["score"].to_numpy(dtype=float)
        for i, x in enumerate(rows):
            per = {M.BASELINE: {"score": x["score"],
                                "pctHistorical": x["pctHistorical"]}}
            for algo, sc in others.items():
                v = float(sc[i])
                per[algo] = {
                    "score": round(v, 6),
                    # モデルごとに別の過去分布で位置を出す。
                    # 共通の分布を使うとスケールの違いが混ざる
                    "pctHistorical": M.pct_historical(v, hist_by_algo[algo]),
                }
            x["byModel"] = per
            # 何個のモデルが「上位10%」と見ているか。独立した判定の数え上げ。
            # ラベル違いのモデル（1.5σ の4本。1.8σ は 2026-10-24 に外した）は 1.2σ の数（agree90 / nModels）に入れず、
            # 別に数える（agreeK90 / nK）
            x["agree90"], x["nModels"] = agreement(per, M.ALGOS)
            x["agreeK90"], x["nK"] = agreement(per, M.ACTIVE_VARIANTS)
            # 合意（cons12。名前は12本だったころのまま。2026-10-24 から8本の平均）と、その過去の分布での百分位（consPct）。
            # 丸める前の百分位で計算する。
            # 2026-10-12 に一度画面に出したが、同じ日の運用者の指示（「12本の平均ではなく、12個全てを載せてほしい」）で
            # 画面は12本の百分位をそのまま表で出す形に変え、合意は出さなくなった。確認・研究用に predictions.json には残す
            raw = {M.BASELINE: raw_pct(M.BASELINE, float(base_scores[i]))}
            raw.update({a: raw_pct(a, float(sc[i])) for a, sc in others.items()})
            x.update(consensus_fields(raw, cons_ref, cons_used))
        if cons_ref is not None:
            consensus_meta = {"models": cons_used, "nHist": int(len(cons_ref)),
                              "line90": round(float(np.quantile(cons_ref, 0.90)), 1),
                              "line975": round(float(np.quantile(cons_ref, 0.975)), 1)}
        print(f"[consensus] {len(cons_used)}本の合意 / 過去の行 {len(cons_ref) if cons_ref is not None else 0:,}件"
              + (f" / 上位10% の線 {np.quantile(cons_ref, 0.90):.1f}・上位2.5% の線 {np.quantile(cons_ref, 0.975):.1f}"
                 if cons_ref is not None else ""))

    # ランキング学習（参考。買いの規則には使わない）。その日の中の順位と上半分の印を足す（attach_ltr。§78-9）
    ltr_info = attach_ltr(rows, cand, args.model_dir)

    # ファイルの並び・rankInDay（予測ログの「順位」）はどちらも基準モデルのスコア（上の rank_in_day と同じ）。
    # 帯・較正・追跡ファイルも基準モデル基準なので、ここは基準モデルのまま。画面の一覧は 2026-10-13 から
    # 1.5σ の3本の百分位の最小の順に並べ直す（src/lib/strategy.js の orderRows。運用者の決定）
    # 次の決算発表の予定（公表済みのものだけ）。予測が実質的に次の決算の予測になっていないかを見るため
    earn = next_earnings([(x["jqCode"], x["date"]) for x in rows], args.data_dir, TC.load(args.data_dir))
    for x in rows:
        e = earn.get((x["jqCode"], x["date"])) or {}
        x["nextEarnDate"] = e.get("date")
        x["nextEarnQuarter"] = e.get("quarter")
        x["nextEarnDays"] = e.get("days")
        x["nextEarnTradingDays"] = e.get("tradingDays")
    latest_rows = [x for x in rows if x["date"] == pd.Timestamp(days[-1]).date().isoformat()]
    soon = sum(1 for x in latest_rows if x["nextEarnTradingDays"] is not None
               and x["nextEarnTradingDays"] <= B.RISE_HORIZON)
    print(f"[earn] 最新の日の候補 {len(latest_rows)}件: 決算の予定が公表済み "
          f"{sum(1 for x in latest_rows if x['nextEarnDate'])}件・{B.RISE_HORIZON}営業日以内に発表 {soon}件")

    # 過去の日は、前に出した値のまま（freeze_past_days）。採点し直したのは新しい日だけを使う
    generated_at = pd.Timestamp.utcnow().isoformat()
    out_path = os.path.join(args.out_dir, "predictions.json")
    day_list = [pd.Timestamp(d).date().isoformat() for d in days]
    prev_payload = load_previous(out_path)
    rows, day_info, fz = freeze_past_days(rows, day_list, prev_payload,
                                          {"trainedAt": meta["trainedAt"], "generatedAt": generated_at})
    print(f"[freeze] 過去の日 {len(fz['frozen'])}日（{fz['rows_frozen']}件）は前に出した値のまま"
          f"（今回の採点 {fz['rows_dropped']}件は使わない）・採点した日 {', '.join(fz['scored'])}"
          + ("（最新の日を予測し直した）" if fz["latest_overwritten"] else ""))

    # 候補の比較（似た過去の事例）。画面の「候補の比較」タブ用（research/knn_compare.py。運用者の指示 2026-10-15）。
    # 参考の表示で、選定には使わない。過去の日は前の値のまま（固定した日の knn.byDay を引き継ぐ）。
    # 作れなくても日次予測は止めない（build は例外を外に出さないが、念のためここでも受ける）
    knn = knn_block(rows, day_list, fz["frozen"], (prev_payload or {}).get("knn"), args.data_dir, args.model_dir)

    rows.sort(key=lambda x: (x["date"], x["score"]), reverse=True)

    payload = {
        "generatedAt": generated_at,
        "asOf": day_list[-1],
        "dates": day_list,
        # 日ごとに、どのモデル（学習日）でいつ出した値か。過去の日はその日に出した値のまま（freeze_past_days）
        "dayInfo": day_info,
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
        # 合意の物差し（consensus_reference）。線は過去の合意の分布の 90% / 97.5% 点（その日の LINEUP の平均の値）。
        # 画面には出さない（上の cons12 / consPct と同じ。確認・研究用）
        "consensus": consensus_meta,
        # 候補の比較（似た過去の事例）。画面の「候補の比較」タブが読む（research/knn_compare.py）
        "knn": knn,
        # ランキング学習の素性（attach_ltr。無ければ None）。各候補の ltr と合わせて、画面と実運用の追跡が「両方使う形」を
        # 参考に並べる（買いの規則には使わない。運用者の選択「まず並べて見る」2026-10-24）
        "ltr": ltr_info,
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
            "一致度は「上位10%と見ているモデルの数」。独立した判定の数え上げで、"
            "スコアを混ぜた値ではない。順位と帯は基準モデル（LightGBM）の"
            "スコアで付けてあり、他のモデルは並べて見るためのもの。",
            "過去の日は、その日に出した値のまま（後の再学習や、後から入ったデータで採点し直さない）。"
            "どのモデルで出した値かは dayInfo の学習日。",
            "L15 / X15 / C15 / LR15 は到達しきい値 1.5σ のラベルで学習したモデル（学習器は記号の元と同じ"
            " LightGBM / XGBoost / CatBoost / ロジスティック回帰）。1.2σ の4本の一致度（agree90）には数えず、"
            "agreeK90 / nK で別に数える。今日の戦略の規則は 1.2σ の木の3本（LGB / XGB / CAT）がすべて95以上"
            "（2026-10-23 から）。8本とも毎日の予測に使う。1.8σ の4本（L18 / X18 / C18 / LR18）は 2026-10-24 に"
            "外した（それより前の日の予測ファイルには値が残るが、画面には出さない）。",
            "ltr はランキング学習（LambdaRank。同じ日の候補の中で、翌営業日の寄りで買い20営業日目の終値で売った収益の"
            "順番を学んだもの）。dayPct はその日の候補の中の百分位（候補が1件の日は 50）、upper は 50 以上か。"
            "「両方使う形」（今の規則を満たし、かつ upper）を参考に並べるだけで、買いの規則には使わない"
            "（2026-10-24 から。それより前の日には無い）。",
        ],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"[done] {out_path} ({os.path.getsize(out_path)/1e3:.0f}KB)")

    update_history(args, rows, days)
    save_live_features(cand, cols, days, args.data_dir, meta.get("trainedAt"))

    # 詳細を取りに行く銘柄コードを素のテキストで出す。
    # 取得スクリプト（scripts/jquants_data_fetcher.py --extra-codes）へ渡して、
    # スナップショットと株価履歴を作らせるため。これが無いと
    # タイムマシーン（過去比較）が予測候補で使えない。
    #
    # **画面に出ている候補を全部対象にする。**
    #
    # 以前は「最新日の上位10件」だけだった。しかし画面は --days（既定5）
    # 営業日ぶんを切り替えられるので、実測（2026-09-18 時点）では
    # 候補46件のうち取得済みは8件しかなく、**83%でタイムマシーンが
    # 使えなかった**。利用者が選ぶのは上位10件とは限らない。
    #
    # 並びは「新しい日から、その日の上位から」。上限で切れるときに
    # 落ちるのが、いちばん古くて順位の低いものになるようにする。
    # 画面側は同じ銘柄コードなら厚いほう（スナップショット付き）を
    # 残すので（src/lib/store.js mergeStocks）、取れていれば必ず使われる。
    picked = detail_codes(rows, args.top_codes)
    codes_path = os.path.join(args.data_dir, "top_codes.txt")
    with open(codes_path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(picked))
    n_all = len({x["code"] for x in rows})
    cut = "" if len(picked) >= n_all else f"（上限 {args.top_codes} で打ち切り）"
    print(f"[done] {codes_path} （画面の候補 {n_all}銘柄 / "
          f"取りに行く {len(picked)}銘柄{cut}）")
    return 0


def knn_block(rows: List[Dict], day_list: List[str], frozen: List[str], prev_knn: Optional[Dict],
              data_dir: str, model_dir: str) -> Optional[Dict]:
    """
    候補の比較（似た過去の事例）の集計（research/knn_compare.py の build）。失敗しても日次予測は止めず、
    固定した日の分だけ前の値を引き継いで、採点した日は「出せない」として残す。
    """
    try:
        import knn_compare as KC
    except Exception as exc:                     # noqa: BLE001
        print(f"[knn] 近傍の集計のコードを読めないので出さない: {type(exc).__name__}: {exc}")
        return prev_knn
    try:
        knn = KC.build(rows, day_list, data_dir, model_dir, prev=prev_knn, frozen=frozen)
    except Exception as exc:                     # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        print(f"[knn] 近傍の集計を作れないので、採点した日は出せない: {reason}")
        knn = KC.carry_over(prev_knn, day_list, frozen, reason=reason[:300])
    print(f"[knn] {KC.summary_line(knn, day_list)}")
    return knn


def detail_codes(rows: List[Dict], limit: int) -> List[str]:
    """
    詳細（スナップショット・株価履歴）を取りに行く銘柄コードを選ぶ。

    **画面に出ている候補を全部対象にする。**

    以前は「最新日の上位10件」だけだった。画面は --days（既定5）営業日ぶんを
    切り替えられるので、実測（2026-09-18 時点）では候補46件のうち取得済みは
    8件しかなく、**83%でタイムマシーンが使えなかった**。利用者が選ぶのは
    上位10件とは限らない。

    並びは「新しい日から、その日の上位から」。上限で切れるときに落ちるのが、
    いちばん古くて順位の低いものになるようにする。同じ銘柄が複数日に
    出ていれば1回だけ数える。

    画面側は同じ銘柄コードなら厚いほう（スナップショット付き）を残すので
    （src/lib/store.js mergeStocks）、取れていれば必ず使われる。
    """
    if not rows:
        return []
    order = {d: i for i, d in enumerate(
        sorted({x["date"] for x in rows}, reverse=True))}
    picked: List[str] = []
    seen = set()
    for x in sorted(rows, key=lambda r: (order[r["date"]], r["rankInDay"])):
        if x["code"] in seen:
            continue
        seen.add(x["code"])
        picked.append(x["code"])
        if limit and len(picked) >= limit:
            break
    return picked


#: 予測に使った特徴量の控え。学習用に後から作り直した値と比べ、公表前の値や
#: 後からの書き換えで「学習だけが見ている値」が無いかを確かめる
#: （research/check_train_serve.py）。Release に置く（predict.yml が上げる）
LIVE_FEATURES = "live_features.parquet"
#: 控えを残す日数。比べるのは直近の数週間なので、1年あれば足りる
LIVE_KEEP_DAYS = 400


def save_live_features(cand: pd.DataFrame, cols: List[str], days, data_dir: str,
                       trained_at=None) -> None:
    """
    その日の候補に使った特徴量を控える。**最初の予測で凍結**する（予測の記録と
    同じ）: その日の控えが既にあれば書き換えない。

    行ごとに、控えた列の組を _features（カンマ区切り）に残す。モデルの列が変わると
    （例: 2026-09-27 の週次の再学習で151列 -> 206列）、前の行には新しい列が無い。ファイルは列の和で
    保存されるので、無い列は欠測に見え、「予測時は欠測・今は値あり」の食い違いと
    区別できなくなる（check_train_serve.compare はこの組で比べる行を絞る）。
    """
    path = os.path.join(data_dir, LIVE_FEATURES)
    latest = pd.Timestamp(days[-1])
    today = cand[pd.to_datetime(cand["Date"]) == latest]
    if today.empty:
        return
    keep = ["Code", "Date"] + [c for c in cols if c in today.columns]
    new = today[keep].copy()
    new["Code"] = new["Code"].astype(str)
    new["Date"] = pd.to_datetime(new["Date"])
    new["_saved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    new["_trained_at"] = str(trained_at or "")
    new["_features"] = ",".join(keep[2:])
    old = None
    if os.path.exists(path):
        try:
            old = pd.read_parquet(path)
            old["Date"] = pd.to_datetime(old["Date"])
        except Exception as exc:                 # noqa: BLE001
            print(f"[live] 控えを読めないので作り直す: {type(exc).__name__}")
    if old is not None and "_features" not in old.columns:
        # _features を残す前の控え。どの行も同じモデル（同じ列）で控えたので、
        # いまファイルにある列がそのまま控えた列の組
        old["_features"] = ",".join(c for c in old.columns
                                    if c not in ("Code", "Date") and not c.startswith("_"))
    if old is not None and (old["Date"] == latest).any():
        print(f"[live] {latest.date()} の特徴量は控え済み。最初の予測のまま凍結する")
        return
    out = new if old is None else pd.concat([old, new], ignore_index=True)
    cutoff = latest - pd.Timedelta(days=LIVE_KEEP_DAYS)
    out = out[out["Date"] >= cutoff]
    out.to_parquet(path, index=False, compression="zstd")
    print(f"[live] {latest.date()} の {len(new)}件 × {len(keep) - 2}列を控えた"
          f"（累計 {out['Date'].nunique()}日 / {len(out):,}件）")


def load_previous(path: str) -> Optional[Dict]:
    """前の predictions.json（無い・読めなければ None。固定せずに全部の日を今回の採点で出す）。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:                     # noqa: BLE001
        print(f"[freeze] 前の予測を読めないので、過去の日も今回の採点で出す: {type(exc).__name__}")
        return None


def freeze_past_days(rows: List[Dict], days: List[str], prev: Optional[Dict],
                     run_info: Dict) -> Tuple[List[Dict], Dict[str, Dict], Dict]:
    """
    過去の日は、前の predictions.json に出した値のまま残す（運用者の指示 2026-10-12「過去の日は最初に出した値のまま
    固定してください」）。それまでは毎回、直近5営業日ぶんを最新のモデル・その日に作り直した特徴量で採点し直して
    上書きしていたので、週次の再学習や後から入ったデータで、過去の日のスコア・順位・候補の数が当日に見たものと変わった。

    rows: 今回の採点（days のすべての日）。days: ISO の日付（古い順。最後が今回の新しい日）。
    prev: 前の predictions.json（None なら固定しない）。run_info: 今回の採点の {"trainedAt", "generatedAt"}。
    規則:
      - 最新の日は今回の採点（同じ日を予測し直すのは、workflow の FORCE_PREDICT で意図したときだけ）
      - それより前の日で、前のファイルの dates に入っている日は、前のファイルの行をそのまま使う（0件の日は0件のまま）
      - 前のファイルに無い過去の日（初めて出す日）は今回の採点
    戻り値: (行, dayInfo, 件数)。dayInfo[日] = {"trainedAt", "generatedAt"}: その日の値を出したモデルの学習日と採点した時刻。
      前のファイルの dayInfo を引き継ぎ、無い（固定を始める前の形の）ときは前のファイルの model.trainedAt と generatedAt。
    """
    latest = days[-1]
    prev_dates = set((prev or {}).get("dates") or [])
    prev_info = (prev or {}).get("dayInfo") or {}
    prev_model = ((prev or {}).get("model") or {}).get("trainedAt")
    prev_gen = (prev or {}).get("generatedAt")
    frozen = [d for d in days[:-1] if d in prev_dates]
    by_day: Dict[str, List[Dict]] = {}
    for x in (prev or {}).get("candidates") or []:
        if x.get("date") in frozen:
            by_day.setdefault(x["date"], []).append(x)
    out = [x for x in rows if x["date"] not in frozen]
    for d in frozen:
        out.extend(by_day.get(d, []))
    info = {}
    for d in days:
        if d in frozen:
            info[d] = dict(prev_info.get(d) or {"trainedAt": prev_model, "generatedAt": prev_gen})
        else:
            info[d] = {"trainedAt": run_info.get("trainedAt"), "generatedAt": run_info.get("generatedAt")}
    stats = {"frozen": frozen, "scored": [d for d in days if d not in frozen],
             "rows_frozen": sum(len(by_day.get(d, [])) for d in frozen),
             "rows_dropped": sum(1 for x in rows if x["date"] in frozen),
             "latest_overwritten": latest in prev_dates}
    return out, info, stats


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
            with open(path, encoding="utf-8") as fh:
                hist = json.load(fh)
        except Exception as e:
            print(f"[warn] 追跡ファイルを読めないので作り直す: {e}")

    latest = pd.Timestamp(days[-1]).date().isoformat()
    existing = {(e["date"], e["jqCode"]) for e in hist.get("entries", [])}
    # **記録は日付ごとに、最初の予測で凍結する。** 同じ日を後から予測し直すと
    # （別のモデルで）上位が入れ替わり、当時は選んでいない銘柄が記録に混ざる。
    # 2026-09-23・24 に2回起きた（9/18 に 日本ナレッジ が入った）。その日の
    # 記録が1件でもあれば足さない。追跡列（現在値など）の更新は続ける
    recorded = {e["date"] for e in hist.get("entries", [])}
    added = 0
    if latest in recorded:
        print(f"[history] {latest} は記録済み。最初の予測のまま凍結する（追加しない）")
        rows = []
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
        cal = TC.load(args.data_dir)
        for e in hist["entries"]:
            cur = last.get(e["jqCode"])
            if cur is None or not np.isfinite(cur) or not e.get("closeAtPick"):
                continue
            e["closeNow"] = round(float(cur), 1)
            e["returnPct"] = round(float(cur) / e["closeAtPick"] * 100 - 100, 2)
            # 平日ではなく取引所の営業日で数える（祝日を経過日に数えない）
            e["daysElapsed"] = TC.elapsed_days(
                pd.Timestamp(e["date"]).date(), asof.date(), cal)
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
