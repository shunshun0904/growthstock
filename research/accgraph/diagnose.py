#!/usr/bin/env python3
"""
EDINET の明細が揃った群で信号が消えた原因を切り分ける。

## 何が起きたか

明細が揃った群（2026-09-22 時点で106社・2,923件）だけで学習・評価すると、
J-Quants のノードだけのベースラインですら AUC が 0.47〜0.52 に落ちた。
全体（56,870件）では同じ特徴量で 0.554 出ている。
信号が無いところに明細を足しても、その増分は測れない。

## 原因の候補は2つ

  1. 件数の問題    訓練が 996〜2,554件しかなく、特徴量（322〜4,150次元）に
                   対して少なすぎて学習できていない
  2. 群の性質      この群はそもそも予測しにくい。EDINET の取得順は本流の
                   「78週高値を頻繁に更新する銘柄」から並んでいるので、
                   情報が早く織り込まれる大型・人気株に偏っている可能性がある

## 切り分け方

**全体で学習し、群に分けて評価する。** 学習に使うデータを全体に戻せば
1 は消える。それでも明細ありの群だけ AUC が落ちるなら 2 である。

大型株一般が予測しにくいだけなのかを分けるため、明細なしの群のうち
流動性（20日平均売買代金）を明細ありの群の分布に揃えたものも並べる。

## 判定基準（結果を見る前に決めておく）

  件数の問題  明細ありの群の AUC の95%区間が 0.5 を上回り、
              流動性を揃えた群との差の95%区間が 0 をまたぐ
  群の性質    明細ありの群の AUC の95%区間が 0.5 をまたぎ、
              流動性を揃えた群との差の95%区間が負に収まる
  それ以外    判定できない（件数が足りない）

区間は発表日単位のブートストラップで出す。同じ日の銘柄は地合いを
共有していて独立ではないので、行単位で引き直すと区間が狭く出すぎる。

  python3 research/accgraph/diagnose.py
  -> docs/ACCGRAPH_DIAGNOSE.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
ROOT = os.path.dirname(RESEARCH)
sys.path.insert(0, RESEARCH)

from accgraph import baselines, build, splits  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "ACCGRAPH_DIAGNOSE.md")
OUT_JSON = os.path.join(build.OUT_DIR, "diagnose.json")
CLASSES = np.array([0, 1, 2])

#: 判定に使うモデルと特徴量セット。MLP は小さい群で値が大きく揺れた
#: （0.472 -> 0.509）ので外す。`_jq` が主役で、素の版は参考に並べる
FEATURE_SETS = ("latest_jq", "latest")
MODELS = ("logit", "lgbm")

#: 流動性を揃えるときの幅。明細ありの群の分位点で切る
MATCH_Q = (0.10, 0.90)


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #

def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """3クラスの ovr macro AUC。どれかのクラスが欠けると定義できないので NaN。"""
    from sklearn.metrics import roc_auc_score
    if len(y) == 0 or len(np.unique(y)) < len(CLASSES):
        return float("nan")
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro",
                               labels=CLASSES))


def _date_blocks(dates: pd.Series) -> Tuple[np.ndarray, List[np.ndarray]]:
    """発表日ごとの行番号の束。ブートストラップはこの束ごとに引き直す。"""
    d = pd.to_datetime(dates).to_numpy()
    uniq, inv = np.unique(d, return_inverse=True)
    blocks = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    return uniq, blocks


def bootstrap_groups(y: np.ndarray, p: np.ndarray, dates: pd.Series,
                     groups: Dict[str, np.ndarray], pairs: List[Tuple[str, str]],
                     n_boot: int = 1000, seed: int = 0) -> Dict:
    """
    各群の AUC と、群どうしの差の区間を出す。

    差は**同じ引き直し**の上で両群の AUC を取って引く。群ごとに別々に
    引き直すと、同じ日の地合いが片方にだけ入って差の区間が広がりすぎる。
    """
    rng = np.random.default_rng(seed)
    _, blocks = _date_blocks(dates)
    point = {g: macro_auc(y[m], p[m]) for g, m in groups.items()}
    draws = {g: [] for g in groups}
    diffs = {f"{a}-{b}": [] for a, b in pairs}

    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        vals = {}
        for g, m in groups.items():
            sel = idx[m[idx]]
            vals[g] = macro_auc(y[sel], p[sel])
            if np.isfinite(vals[g]):
                draws[g].append(vals[g])
        for a, b in pairs:
            if np.isfinite(vals[a]) and np.isfinite(vals[b]):
                diffs[f"{a}-{b}"].append(vals[a] - vals[b])

    def ci(v):
        if len(v) < n_boot * 0.5:
            return [float("nan"), float("nan")]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    return {
        "auc": {g: {"point": point[g], "ci": ci(draws[g]),
                    "n_boot_ok": len(draws[g])} for g in groups},
        "diff": {k: {"point": (point[k.split("-")[0]] - point[k.split("-")[1]]),
                     "ci": ci(v)} for k, v in diffs.items()},
    }


def verdict(res: Dict) -> Tuple[str, str]:
    """事前に決めた基準で判定する。"""
    ed = res["auc"]["edinet"]["ci"]
    df = res["diff"]["edinet-matched"]["ci"]
    if any(np.isnan(ed)) or any(np.isnan(df)):
        return "判定できない", "区間を出せるだけの件数が無い"
    above = ed[0] > 0.5
    straddle = ed[0] <= 0.5 <= ed[1]
    diff_zero = df[0] <= 0.0 <= df[1]
    diff_neg = df[1] < 0.0
    if above and diff_zero:
        return "件数の問題", ("全体で学習すれば明細ありの群も予測でき、流動性を"
                            "揃えた群と差が無い。群だけで学習したときに信号が"
                            "消えたのは件数が足りなかったため")
    if straddle and diff_neg:
        return "群の性質", ("全体で学習しても明細ありの群は 0.5 と区別できず、"
                          "流動性を揃えた群より有意に低い。この群はそもそも"
                          "予測しにくい")
    return "判定できない", ("区間が広く、どちらの基準も満たさない。"
                         "件数が足りない")


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

def oof_full(meta: pd.DataFrame, X: np.ndarray, y: np.ndarray, model: str,
             folds_tr: List[np.ndarray], folds_te: List[np.ndarray],
             seed: int = 0) -> np.ndarray:
    """全体で学習し、テスト窓の各行に確率を付ける（テスト窓外は NaN）。"""
    out = np.full((len(y), len(CLASSES)), np.nan)
    fit = baselines.MODELS[model]
    for tr, te in zip(folds_tr, folds_te):
        m = fit(X[tr], y[tr], CLASSES, seed=seed)
        out[te] = baselines.align_proba(m, X[te], CLASSES)
    return out


def matched_mask(turnover: np.ndarray, edinet: np.ndarray,
                 base: np.ndarray, q: Tuple[float, float] = MATCH_Q) -> Tuple[np.ndarray, List[float]]:
    """
    明細なしの群のうち、流動性が明細ありの群の分位点の幅に入るもの。

    明細ありの群は大型株に偏っている。大型株一般が予測しにくいだけなら、
    流動性を揃えた群も同じだけ AUC が落ちるはず。
    """
    tv = np.log1p(np.nan_to_num(turnover, nan=0.0))
    ref = tv[edinet & base]
    if len(ref) == 0:
        return np.zeros_like(edinet), [float("nan"), float("nan")]
    lo, hi = float(np.quantile(ref, q[0])), float(np.quantile(ref, q[1]))
    m = base & ~edinet & (tv >= lo) & (tv <= hi)
    return m, [float(np.expm1(lo)), float(np.expm1(hi))]


def run(data_dir: str = build.OUT_DIR, *, benchmark: str = "topix",
        horizon: int = 20, max_horizon: int = 20,
        feature_sets=FEATURE_SETS, models=MODELS, n_boot: int = 1000,
        min_train_months: int = 48, test_months: int = 12,
        step_months: int = 12, min_test_rows: int = 200,
        seed: int = 0) -> Dict:
    meta, nf, ef, pm = build.load(data_dir, liquid_only=True)
    if "has_edinet" not in meta.columns:
        raise SystemExit("has_edinet がありません。build.py を回し直してください")
    y_col = f"y_{benchmark}_{horizon}d"
    usable = meta[y_col].to_numpy() >= 0
    meta = meta[usable].reset_index(drop=True)
    nf, ef, pm = nf[usable], ef[usable], pm[usable]
    y = meta[y_col].to_numpy().astype(int)
    edinet = meta["has_edinet"].to_numpy().astype(bool)
    print(f"[diag] 全体 {len(meta):,}件 / 明細あり {int(edinet.sum()):,}件 "
          f"({meta.loc[edinet, 'Code'].nunique():,}社)")

    folds, tr, te = splits.walk_forward(
        meta, min_train_months=min_train_months, test_months=test_months,
        step_months=step_months, embargo_trading_days=max_horizon,
        min_test_rows=min_test_rows)
    print(splits.describe(folds))
    in_test = np.zeros(len(meta), dtype=bool)
    for m in te:
        in_test |= m

    turnover = pd.to_numeric(meta["turnover_ma20"], errors="coerce").to_numpy()
    matched, band = matched_mask(turnover, edinet, in_test)
    groups = {
        "all": in_test,
        "edinet": in_test & edinet,
        "non_edinet": in_test & ~edinet,
        "matched": matched,
    }
    desc = {}
    for g, m in groups.items():
        desc[g] = {
            "n": int(m.sum()),
            "codes": int(meta.loc[m, "Code"].nunique()),
            "turnover_median": (float(np.nanmedian(turnover[m])) if m.any()
                                else None),
            "class_share": [float((y[m] == c).mean()) if m.any() else None
                            for c in CLASSES],
        }
    print("[diag] 評価する群: " + " / ".join(
        f"{g} {d['n']:,}件・{d['codes']:,}社" for g, d in desc.items()))

    results = {}
    pairs = [("edinet", "matched"), ("edinet", "non_edinet"),
             ("matched", "non_edinet")]
    for fs in feature_sets:
        X, _ = baselines.flatten(nf, ef, pm, meta, kind=fs)
        for mname in models:
            p = oof_full(meta, X, y, mname, tr, te, seed=seed)
            ok = ~np.isnan(p).any(axis=1)
            res = bootstrap_groups(
                y[ok], p[ok], meta.loc[ok, "entry_date"],
                {g: m[ok] for g, m in groups.items()}, pairs,
                n_boot=n_boot, seed=seed)
            v, why = verdict(res)
            res["verdict"], res["why"] = v, why
            results[f"{fs}/{mname}"] = res
            a = res["auc"]
            print(f"  {fs:<10} {mname:<6} all {a['all']['point']:.4f} / "
                  f"edinet {a['edinet']['point']:.4f} "
                  f"[{a['edinet']['ci'][0]:.3f}, {a['edinet']['ci'][1]:.3f}] / "
                  f"matched {a['matched']['point']:.4f} -> {v}")

    return {
        "benchmark": benchmark, "horizon": horizon, "n_boot": n_boot,
        "folds": [f.to_dict() for f in folds],
        "match_band_oku": band, "groups": desc, "results": results,
    }


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #

GROUP_JA = {
    "all": "全体（テスト窓）",
    "edinet": "EDINET 明細あり",
    "non_edinet": "明細なし",
    "matched": "明細なし・流動性を揃えたもの",
}


def _ci(c) -> str:
    if c is None or any(x is None or np.isnan(x) for x in c):
        return "—"
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def to_markdown(d: Dict) -> str:
    lines = [
        "# 会計フローグラフ — EDINET 明細ありの群で信号が消えた原因の切り分け",
        "",
        "`research/accgraph/diagnose.py` の出力。**実行した結果のみ**を載せる。",
        "",
        "## やったこと",
        "",
        "明細が揃った群だけで学習・評価すると、J-Quants のノードだけの",
        "ベースラインですら AUC が 0.5 前後に落ちた（`docs/ACCGRAPH_EDINET.md`）。",
        "原因が「群が小さくて学習できない」のか「群がそもそも予測しにくい」のかを",
        "分けるため、**全体で学習し、群に分けて評価**した。",
        "",
        f"- ベンチマーク: {'TOPIX' if d['benchmark'] == 'topix' else '業種指数'}控除 / "
        f"{d['horizon']}営業日",
        "- 学習: 流動性1億円以上の全サンプル（Purge / Embargo つきウォークフォワード）",
        f"- 区間: 発表日単位のブートストラップ {d['n_boot']:,}回の95%区間",
        f"- 流動性を揃えた群: 明細ありの群の20日平均売買代金の10〜90%点"
        f"（{d['match_band_oku'][0]:.1f}〜{d['match_band_oku'][1]:.1f}億円）"
        "に入る明細なしのサンプル",
        "",
        "## 判定基準（結果を見る前に決めたもの）",
        "",
        "| 判定 | 条件 |",
        "| --- | --- |",
        "| 件数の問題 | 明細ありの群の AUC の区間が 0.5 を上回り、流動性を揃えた群との差の区間が 0 をまたぐ |",
        "| 群の性質 | 明細ありの群の AUC の区間が 0.5 をまたぎ、流動性を揃えた群との差の区間が負に収まる |",
        "| 判定できない | どちらも満たさない |",
        "",
        "## 評価した群",
        "",
        "| 群 | 件数 | 社数 | 売買代金の中央値 | 下落 / 中立 / 上昇 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for g, x in d["groups"].items():
        cs = x["class_share"]
        share = ("—" if cs[0] is None else
                 f"{cs[0]*100:.1f} / {cs[1]*100:.1f} / {cs[2]*100:.1f}%")
        tv = "—" if x["turnover_median"] is None else f"{x['turnover_median']:.1f}億円"
        lines.append(f"| {GROUP_JA[g]} | {x['n']:,} | {x['codes']:,} | {tv} | {share} |")

    lines += [
        "",
        "## AUC（全体で学習したモデルを、群ごとに評価）",
        "",
        "| 特徴量 | モデル | 全体 | 明細あり | 明細なし・流動性を揃えた | 明細なし | 判定 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, r in d["results"].items():
        fs, m = key.split("/")
        a = r["auc"]
        cell = lambda g: f"{a[g]['point']:.4f} {_ci(a[g]['ci'])}"
        lines.append(f"| {fs} | {m} | {a['all']['point']:.4f} | {cell('edinet')} "
                     f"| {cell('matched')} | {cell('non_edinet')} | **{r['verdict']}** |")

    lines += [
        "",
        "## 群どうしの差（同じ引き直しの上で取った差）",
        "",
        "| 特徴量 | モデル | 明細あり − 揃えた群 | 明細あり − 明細なし | 揃えた群 − 明細なし |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, r in d["results"].items():
        fs, m = key.split("/")
        df = r["diff"]
        cell = lambda k: f"{df[k]['point']:+.4f} {_ci(df[k]['ci'])}"
        lines.append(f"| {fs} | {m} | {cell('edinet-matched')} "
                     f"| {cell('edinet-non_edinet')} | {cell('matched-non_edinet')} |")

    lines += ["", "## 判定の根拠", ""]
    for key, r in d["results"].items():
        lines.append(f"- `{key}`: **{r['verdict']}** — {r['why']}")
    lines += [
        "",
        "## 読み方",
        "",
        "- 「明細あり」の列を、群だけで学習したときの値（`docs/ACCGRAPH_EDINET.md`、"
        "0.47〜0.52）と比べる。全体で学習して上がるなら、群だけの学習では件数が足りなかった。",
        "- 「揃えた群」は大型株一般の予測しにくさを表す対照。"
        "明細ありの群がこれと同じなら、その群が特別なのではなく大型株だから。",
        "- `latest` は EDINET のノードを含むが、全体で学習すると明細は9割以上が欠測のまま"
        "学習される。明細の効果を測る比較ではない（それは件数が揃ってから群の中で行う）。",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="明細ありの群で信号が消えた原因を切り分ける")
    ap.add_argument("--data-dir", default=build.OUT_DIR)
    ap.add_argument("--benchmark", choices=["topix", "sector"], default="topix")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-train-months", type=int, default=48)
    ap.add_argument("--test-months", type=int, default=12)
    ap.add_argument("--step-months", type=int, default=12)
    ap.add_argument("--min-test-rows", type=int, default=200)
    ap.add_argument("--feature-sets", nargs="*", default=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--out-json", default=OUT_JSON)
    args = ap.parse_args(argv)

    d = run(args.data_dir, benchmark=args.benchmark, horizon=args.horizon,
            feature_sets=tuple(args.feature_sets), models=tuple(args.models),
            n_boot=args.n_boot, min_train_months=args.min_train_months,
            test_months=args.test_months, step_months=args.step_months,
            min_test_rows=args.min_test_rows)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(d))
    print(f"[diag] 書き出し {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
