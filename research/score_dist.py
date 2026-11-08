#!/usr/bin/env python3
"""
スコアと収益の分布（画面のタブ「スコアと収益の分布」が読む public/data/score_dist.json）を書く。

運用者の依頼（2026-11-08）「lgbmの+10%モデルでよいので、スコアの分布がみたいです。oof含めてです」→「なんなら、
ダッシュボードにほしいくらいです」「ランキング学習による直近5営業日版のランキング予測とその収益（20営業日終値、20営業日内の
終値の最高値）の分布もみたいです」「最初のスコア分布の右側の銘柄とランキング学習モデルによる収益分布の右側の銘柄がどれくらい
重なっているのか」→ AskUserQuestion「3つとも載せる」。

出すもの（すべて集計だけ。銘柄ごとの値は出さない）
  lgbm   … +10% の LightGBM（固定 +10% のラベル）の out-of-fold のスコアの分布（正例・負例に分けた度数）、分位点（1%刻みも）、
             スコア帯ごとの件数・+10% 到達率・収益（ret_o1_20: 翌営業日の寄りで買い、20営業日後の5日平均終値で売った上昇率）
  ltr5   … 直近5営業日の組のランキング学習の out-of-fold。スコアの度数、組の行すべて・組の1位の収益の度数と要約
             （R20: 判断日の翌営業日の寄りで買い20営業日目の終値で売った収益。RMAX: 同じ窓の終値の最高値で売った収益）、
             組の中の順位の10分位ごとの平均収益
  overlap … lgbm の上位10%（out-of-fold のスコアの90%点以上）と、ltr5 の組の中の上位10% の重なり（組の行で数える。
             lgbm のスコアは組の行の候補（Code・cand_date）の out-of-fold を当てる）と、4つの組み合わせごとの収益

確かめ（合わなければ止めて書かない）
  - 2つの out-of-fold が学習の記録（meta.json の artifactHashes）と同じファイルであること（固定 +10% の lgbm は
    model_artifacts.flat_oof_path が確かめる）
  - R20 を live_track.forward の ret_close（同じ買い方・売り方）と 300 行で突き合わせ、すべて一致すること
  - RMAX ≥ R20（窓に20営業日目の終値が入っているので）

ログには件数・日付・割合だけを出す（銘柄は出さない）。予測の候補がこの分布のどこにいるか（運用者の依頼 2026-11-08「予測を
実施したときに、各サンプルが分布のどこに位置するのか、表示してほしいです」）は、画面が予測ファイルの候補と、ここの分位点・帯・
10分位・重なりの組み合わせを突き合わせて出す（src/lib/scoredist.js candidatePositions）。

  python3 research/score_dist.py                  # research/_data の out-of-fold と日足から
  python3 research/score_dist.py --data-dir DIR --out FILE
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import live_track as LT  # noqa: E402
import model_artifacts as MA  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
MODEL_DIR = os.path.join(HERE, "model")
OUT = os.path.join(ROOT, "public", "data", "score_dist.json")
LTR5_OOF = "ltr5_oof.parquet"
HOLD = 20
TOP = 0.10                   # 上位10%（lgbm は out-of-fold 全体の90%点、ltr5 は組の中の百分位）
CHECK_ROWS = 300
LGBM_EDGES = [round(x, 3) for x in np.arange(0.0, 0.8001, 0.025)]
LTR5_EDGES = [round(x, 2) for x in np.arange(-2.0, 2.0001, 0.1)]
R20_EDGES = [round(x, 1) for x in np.arange(-40.0, 60.001, 2.5)]
RMAX_EDGES = [round(x, 1) for x in np.arange(-15.0, 70.001, 2.5)]
LGBM_BANDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
QS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _r(v, nd=4):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def hist(values, edges: List[float]) -> List[int]:
    """度数。端の外側は最初・最後の区間にまとめる（画面は「端はまとめた」と書く）。"""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    e = np.asarray(edges, dtype=float)
    x = np.clip(x, e[0], np.nextafter(e[-1], -np.inf))
    return [int(c) for c in np.histogram(x, bins=e)[0]]


def quantiles(values) -> Dict[str, Optional[float]]:
    x = pd.Series(values, dtype=float).dropna()
    return {f"{q:g}": _r(x.quantile(q)) for q in QS} if len(x) else {}


def forward_returns(keys: pd.DataFrame, bars: pd.DataFrame, hold: int = HOLD) -> pd.DataFrame:
    """
    (Code, Date) ごとに R20・RMAX（比）。Date は判断日。買値は判断日の翌営業日の寄り（AdjO）、R20 は hold 営業日目の終値、
    RMAX は買った日〜hold 営業日目の終値（AdjC）の最高値で売った収益。live_track.forward と同じ日足の並べ方（銘柄の日足を
    日付の順に並べ、判断日の位置 i、買値は i+1、窓は i+1〜i+hold）。i+hold が日足に無い・寄りが無いなら欠測。
    """
    k = keys[["Code", "Date"]].drop_duplicates().copy()
    k["Code"] = k["Code"].astype(str)
    k["Date"] = pd.to_datetime(k["Date"])
    want = k.groupby("Code")["Date"].apply(list).to_dict()
    b = bars[bars["Code"].astype(str).isin(want)].copy()
    b["Code"] = b["Code"].astype(str)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values(["Code", "Date"])
    rows = []
    for code, g in b.groupby("Code", sort=False):
        d = g["Date"].to_numpy(dtype="datetime64[ns]")
        o = g["AdjO"].to_numpy(dtype=float)
        c = g["AdjC"].to_numpy(dtype=float)
        n = len(d)
        for x in want[code]:
            i = int(np.searchsorted(d, np.datetime64(x, "ns")))
            if i >= n or d[i] != np.datetime64(x, "ns") or i + hold >= n:
                continue
            e = o[i + 1]
            if not (np.isfinite(e) and e > 0):
                continue
            r20 = c[i + hold] / e - 1.0
            w = c[i + 1:i + hold + 1]
            mx = np.nanmax(w) / e - 1.0 if np.isfinite(w).any() else np.nan
            rows.append((code, pd.Timestamp(x), r20, mx if np.isfinite(r20) else np.nan))
    out = pd.DataFrame(rows, columns=["Code", "Date", "R20", "RMAX"])
    return k.merge(out, on=["Code", "Date"], how="left")


def check_r20(keys: pd.DataFrame, rets: pd.DataFrame, bars: pd.DataFrame, n: int = CHECK_ROWS) -> Dict:
    """R20 を live_track.forward（ret_close は %）と突き合わせる。"""
    have = rets.dropna(subset=["R20"])
    smp = have.sample(min(n, len(have)), random_state=0) if len(have) else have
    if not len(smp):
        return {"rows": 0, "same": 0}
    fw = LT.forward(bars, smp[["Code", "Date"]])
    m = smp.merge(fw[["Code", "Date", "ret_close"]], on=["Code", "Date"], how="left")
    same = int((np.abs(m["R20"] * 100.0 - m["ret_close"]) < 1e-9).sum())
    return {"rows": int(len(m)), "same": same}


def pool_rank(lt: pd.DataFrame) -> pd.DataFrame:
    """組（判断日）の中の順位（1 = 最上位）と上からの百分位 pp（1位 = 1、最下位 = 1/組の行の数）、10分位（10 = 上位）。"""
    lt = lt.copy()
    lt["rk"] = lt.groupby("Date")["score"].rank(ascending=False, method="first")
    lt["n_pool"] = lt.groupby("Date")["score"].transform("size")
    lt["pp"] = 1.0 - (lt["rk"] - 1.0) / lt["n_pool"]
    lt["dec"] = np.minimum(np.ceil(lt["pp"] * 10.0 - 1e-12).astype(int), 10)
    return lt


def ret_summary(x: pd.DataFrame) -> Dict:
    r20, mx = x["R20"].astype(float), x["RMAX"].astype(float)
    return {
        "n": int(len(x)),
        "r20Mean": _r(r20.mean()), "r20Median": _r(r20.median()),
        "rmaxMean": _r(mx.mean()), "rmaxMedian": _r(mx.median()),
        "rmaxHit10": _r((mx >= 0.10).mean()), "r20Below10": _r((r20 < -0.10).mean()),
    }


def lgbm_part(lg: pd.DataFrame, meta: Dict) -> Dict:
    s = lg["score"].astype(float)
    y = lg["label"].astype(int)
    ret = pd.to_numeric(lg.get("ret_o1_20"), errors="coerce")
    bands = []
    cut = pd.cut(s, LGBM_BANDS, include_lowest=True)
    for iv, g in lg.assign(_b=cut, _ret=ret).groupby("_b", observed=True):
        bands.append({"lo": float(iv.left) if iv.left > 0 else 0.0, "hi": float(iv.right), "n": int(len(g)),
                      "posRate": _r(g["label"].mean()), "retMean": _r(g["_ret"].mean()),
                      "retMedian": _r(g["_ret"].median())})
    return {
        "trainedAt": meta.get("trainedAt"), "label": meta.get("label"),
        "n": int(len(lg)), "from": str(pd.to_datetime(lg["Date"]).min().date()),
        "to": str(pd.to_datetime(lg["Date"]).max().date()),
        "positiveRate": _r(y.mean()), "prAuc": (meta.get("oofMetrics") or {}).get("prAuc"),
        "rocAuc": (meta.get("oofMetrics") or {}).get("rocAuc"),
        "quantiles": quantiles(s), "quantilesPos": quantiles(s[y == 1]), "quantilesNeg": quantiles(s[y == 0]),
        # 0〜100% の1%刻みの分位点。画面が予測の候補のスコアを out-of-fold の中の百分位に直す（線形に補間）
        "pctGrid": [_r(v) for v in s.quantile(np.linspace(0, 1, 101)).to_numpy()],
        "hist": {"edges": LGBM_EDGES, "pos": hist(s[y == 1], LGBM_EDGES), "neg": hist(s[y == 0], LGBM_EDGES)},
        "bands": bands,
    }


def ltr5_part(lt: pd.DataFrame, meta: Dict) -> Dict:
    t1 = lt[lt["rk"] == 1]
    dec = []
    for d, g in lt.groupby("dec"):
        dec.append({"dec": int(d), **ret_summary(g)})
    return {
        "trainedAt": meta.get("trainedAt"), "nGrades": meta.get("nGrades"),
        "ndcg5": (meta.get("oofMetrics") or {}).get("ndcg5"),
        "nRows": int(len(lt)), "nDays": int(lt["Date"].nunique()),
        "from": str(pd.to_datetime(lt["Date"]).min().date()), "to": str(pd.to_datetime(lt["Date"]).max().date()),
        "scoreHist": {"edges": LTR5_EDGES, "counts": hist(lt["score"], LTR5_EDGES)},
        "r20Hist": {"edges": R20_EDGES, "all": hist(lt["R20"] * 100, R20_EDGES), "top1": hist(t1["R20"] * 100, R20_EDGES)},
        "rmaxHist": {"edges": RMAX_EDGES, "all": hist(lt["RMAX"] * 100, RMAX_EDGES),
                     "top1": hist(t1["RMAX"] * 100, RMAX_EDGES)},
        "summary": {"all": ret_summary(lt), "top1": ret_summary(t1), "top10": ret_summary(lt[lt["pp"] > 1 - TOP]),
                    "bottom10": ret_summary(lt[lt["pp"] <= TOP])},
        "quantiles": {"r20All": quantiles(lt["R20"]), "r20Top1": quantiles(t1["R20"]),
                      "rmaxAll": quantiles(lt["RMAX"]), "rmaxTop1": quantiles(t1["RMAX"])},
        "deciles": dec,
    }


def overlap_part(lt: pd.DataFrame, lg: pd.DataFrame) -> Dict:
    thr = float(lg["score"].quantile(1 - TOP))
    s = lg[["Code", "Date", "score"]].rename(columns={"Date": "cand_date", "score": "lg"})
    s["Code"] = s["Code"].astype(str)
    m = lt.merge(s, on=["Code", "cand_date"], how="left")
    missing = int(m["lg"].isna().sum())
    m = m.dropna(subset=["lg"])
    a = m["lg"] >= thr
    b = m["pp"] > 1 - TOP
    groups = []
    for key, lab, sel in [("both", "両方の上位10%", a & b), ("lgbmOnly", "lgbm だけ上位10%", a & ~b),
                          ("ltr5Only", "ltr5 だけ上位10%", ~a & b), ("neither", "どちらも上位でない", ~a & ~b)]:
        groups.append({"key": key, "label": lab, **ret_summary(m[sel])})
    t1 = m[m["rk"] == 1]
    pick = m.loc[m.groupby("Date")["lg"].idxmax()] if len(m) else m
    corr = m.groupby("Date")[["lg", "score"]].apply(
        lambda g: g.corr(method="spearman").iloc[0, 1] if len(g) >= 5 else np.nan)
    return {
        "lgbmThreshold": _r(thr), "n": int(len(m)), "withoutLgbm": missing,
        "both": int((a & b).sum()), "expectedBoth": _r(a.mean() * b.mean() * len(m), 1),
        "ltr5TopInLgbmTop": _r(a[b].mean()), "lgbmTopInLtr5Top": _r(b[a].mean()),
        "rankCorrInPool": _r(corr.mean(), 3),
        "groups": groups,
        "ltr5Top1": {"n": int(len(t1)), "lgbmTop10": _r((t1["lg"] >= thr).mean()),
                     "lgbmTop1": _r((t1["lg"] >= float(lg["score"].quantile(0.99))).mean()),
                     "lgbmAboveMedian": _r((t1["lg"] >= float(lg["score"].median())).mean())},
        "lgbmPick": {**ret_summary(pick), "sameAsLtr5Top1": _r((pick["rk"] == 1).mean())},
    }


def load_meta(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build(data_dir: str = DATA_DIR, model_dir: str = MODEL_DIR) -> Dict:
    lg_meta = load_meta(os.path.join(model_dir, "meta.json"))
    lt_meta = load_meta(os.path.join(model_dir, "models", "ltr5", "meta.json"))
    lg_path = MA.flat_oof_path("lgbm", data_dir, model_dir)      # 学習の記録と SHA-256 が違えば止まる
    lt_path = os.path.join(data_dir, LTR5_OOF)
    want = (lt_meta.get("artifactHashes") or {}).get("oof.parquet")
    if not (os.path.isfile(lt_path) and want and _sha256(lt_path) == want):
        raise SystemExit("[stop] ltr5 の out-of-fold が学習の記録（ltr5/meta.json）と違う")
    lg = pd.read_parquet(lg_path)
    lg["Code"] = lg["Code"].astype(str)
    lt = pd.read_parquet(lt_path)
    lt["Code"] = lt["Code"].astype(str)
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("[stop] bars_*.parquet が無い")
    bars = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "AdjO", "AdjH", "AdjC"]) for p in paths],
                     ignore_index=True)
    bars["Code"] = bars["Code"].astype(str)
    bars["Date"] = pd.to_datetime(bars["Date"])
    print(f"[load] lgbm out-of-fold {len(lg):,}件 / ltr5 out-of-fold 組の行 {len(lt):,}件・判断日 {lt['Date'].nunique():,}日"
          f" / 日足 〜{bars['Date'].max().date()}")

    rets = forward_returns(lt[["Code", "Date"]], bars)
    chk = check_r20(lt, rets, bars)
    lt = lt.merge(rets, on=["Code", "Date"], how="left")
    no_ret = int(lt["R20"].isna().sum())
    below = int((lt["RMAX"] < lt["R20"] - 1e-12).sum())
    print(f"[check] R20 を live_track.forward と突き合わせ {chk['same']} / {chk['rows']} 行が同じ ｜ RMAX < R20 の行 {below}"
          f" ｜ 収益が付かない組の行 {no_ret}（20営業日たっていない・寄り付かない）")
    if chk["rows"] == 0 or chk["same"] != chk["rows"] or below:
        raise SystemExit("[stop] 収益の確かめが合わない")
    lt = pool_rank(lt.dropna(subset=["R20"]))

    out = {
        "generatedAt": pd.Timestamp.utcnow().isoformat(),
        "holdDays": HOLD, "top": TOP,
        "checks": {"r20": chk, "rmaxBelowR20": below, "ltr5RowsWithoutReturn": no_ret},
        "lgbm": lgbm_part(lg, lg_meta),
        "ltr5": ltr5_part(lt, lt_meta),
        "overlap": overlap_part(lt, lg),
    }
    L, R, O = out["lgbm"], out["ltr5"], out["overlap"]
    print(f"[lgbm] {L['n']:,}件 {L['from']}〜{L['to']} / 正例率 {L['positiveRate']*100:.2f}% / 中央 {L['quantiles']['0.5']}"
          f" / 90%点 {L['quantiles']['0.9']}")
    print(f"[ltr5] 組の行 {R['nRows']:,}件・判断日 {R['nDays']:,}日 / 組の1位 R20 平均 {R['summary']['top1']['r20Mean']*100:+.2f}%"
          f"・RMAX 平均 {R['summary']['top1']['rmaxMean']*100:+.2f}%")
    print(f"[overlap] 組の行 {O['n']:,}件（lgbm が付かない {O['withoutLgbm']}）/ 両方の上位10% {O['both']:,}件"
          f"（でたらめなら {O['expectedBoth']}）/ 組の中の順位相関 {O['rankCorrInPool']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    out = build(a.data_dir, a.model_dir)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"[done] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
