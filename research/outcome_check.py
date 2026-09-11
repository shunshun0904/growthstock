#!/usr/bin/env python3
"""
本番モデルが「実際にいくら取れたか」を、ラベルに依存しない物差しで測る。

PR-AUC も ROC-AUC もラベルの関数なので、ラベル定義を変えた前後を
これらの数字だけで比べても「良くなった」ことにはならない。
固定の参照ホライズン（60営業日後の5日平均終値の上昇率）で、
上位k%が実際に何%取れたかを出す。この定義はラベルを変えても動かない。

同時に、モデルを使わず1列で並べただけの規則（大型順・低ボラ順など）と
**同じ行集合の上で対で**比べる。別々の区間の重なりでは判定できないため。

  python3 research/outcome_check.py --features all
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

DATA_DIR = S.DATA_DIR


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, cols: List[str],
                  preset: str) -> Dict[str, np.ndarray]:
    """train_model と同じ作り方で、同じモデルを当てる。"""
    models = T.fit_models(train, cols, verbose=True, preset=preset)
    X = test[cols].to_numpy(dtype=float)
    return {name: m.predict_proba(X)[:, 1] for name, m in models.items()}


#: 時価総額の帯。build_dataset.cap_band が付ける 0〜4（100/300/1000/3000億円で区切る）
BAND_COL = "cap_band"


def take_top(t: pd.DataFrame, score_col: str, k_pct: float) -> pd.DataFrame:
    n = max(1, int(len(t) * k_pct / 100))
    return t.nlargest(n, score_col)


def take_top_by_band(t: pd.DataFrame, score_col: str, k_pct: float,
                     band_col: str = BAND_COL) -> pd.DataFrame:
    """
    帯ごとに上位k%を選ぶ。

    こうすると選ばれた集合の帯構成が母集団と同じになるので、
    「大きい帯を多めに取った」効果が入らない。素朴に全体から上位k%を取ると、
    モデルが大型に寄っているだけで数字が良くなりうる。
    帯が欠測の行は帯を揃えられないので落とす（落とした件数は返り値に出す）。
    """
    picks = [take_top(g, score_col, k_pct)
             for _, g in t[t[band_col].notna()].groupby(band_col, sort=True)]
    return pd.concat(picks) if picks else t.iloc[:0]


def _date_groups(t: pd.DataFrame):
    """日付ブロックの行位置。同じ日の銘柄は地合いを共有していて独立ではない。"""
    d = t["Date"].to_numpy()
    return [np.flatnonzero(d == v) for v in pd.unique(d)]


def _boot_frames(t: pd.DataFrame, n_boot: int, seed: int):
    """日付単位でリサンプルした DataFrame を順に返す。"""
    rng = np.random.default_rng(seed)
    groups = _date_groups(t)
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        yield t.iloc[np.concatenate([groups[i] for i in pick])]


def _med(x) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(np.median(x)) * 100 if len(x) else float("nan")


def edge_stats(t: pd.DataFrame, score_col: str, k_pct: float, by_band: bool,
               n_boot: int = 1000, seed: int = 0) -> Dict:
    """「上位k% − 全件」の実収益に区間を付ける。by_band なら帯を揃えて選ぶ。"""
    pick = take_top_by_band if by_band else (
        lambda x, c, k: take_top(x, c, k))
    base = t[t[BAND_COL].notna()] if by_band else t
    top = pick(t, score_col, k_pct)
    diffs = []
    for bt in _boot_frames(t, n_boot, seed):
        b = bt[bt[BAND_COL].notna()] if by_band else bt
        if len(b) < 40:
            continue
        diffs.append(_med(pick(bt, score_col, k_pct)["ref_end"]) - _med(b["ref_end"]))
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
    return {"k_pct": k_pct, "by_band": by_band, "n_top": int(len(top)),
            "n_base": int(len(base)),
            "top": round(_med(top["ref_end"]), 2),
            "all": round(_med(base["ref_end"]), 2),
            "top_win": round(float((top["ref_end"] > 0).mean()), 4),
            "all_win": round(float((base["ref_end"] > 0).mean()), 4),
            "edge": round(_med(top["ref_end"]) - _med(base["ref_end"]), 2),
            "ci": [round(float(lo), 2), round(float(hi), 2)],
            "significant": bool(lo > 0)}


def paired_vs(t: pd.DataFrame, score_col: str, rule_col: str, desc: bool,
              k_pct: float, by_band: bool, n_boot: int = 1000,
              seed: int = 0) -> Dict:
    """
    モデルの上位k%と、1列で並べただけの上位k%を、同じ行集合の上で対で比べる。

    別々の区間の重なりでは判定できない（重なっていても差は有意でありうるし、
    その逆もある）。同じリサンプルの中で両方を選び直して差を取る。
    """
    u = t.copy()
    u["_rule"] = u[rule_col] if desc else -u[rule_col]
    u = u[u["_rule"].notna()]
    pick = take_top_by_band if by_band else (lambda x, c, k: take_top(x, c, k))
    obs = _med(pick(u, score_col, k_pct)["ref_end"]) - _med(pick(u, "_rule", k_pct)["ref_end"])
    diffs = []
    for bt in _boot_frames(u, n_boot, seed):
        if len(bt) < 40:
            continue
        diffs.append(_med(pick(bt, score_col, k_pct)["ref_end"])
                     - _med(pick(bt, "_rule", k_pct)["ref_end"]))
    if not diffs:
        return {}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(float(obs), 2),
            "ci": [round(float(lo), 2), round(float(hi), 2)],
            "p_positive": round(float(np.mean(np.array(diffs) > 0)), 3),
            "significant": bool(lo > 0)}


def pooled_scores(ds: pd.DataFrame, cols: List[str], preset: str,
                  folds) -> Dict[str, pd.DataFrame]:
    """
    ウォークフォワードの各窓で学習し、全窓のテスト行にスコアを付けてまとめる。

    1窓（直近1年）では上位5%が200件ほどしかなく、対大型順の差を
    誤差と区別できなかった。窓を増やして検出力を上げる。
    """
    d = pd.to_datetime(ds["Date"])
    out: Dict[str, List[pd.DataFrame]] = {}
    for f in folds:
        tr = ds[(d <= pd.Timestamp(f.train_end)) & ds["label"].notna()]
        te = ds[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))
                & ds["label"].notna()]
        if len(te) < 200 or te["label"].sum() < 20 or len(tr) < 1000:
            print(f"  窓{f.index} {f.test_start}〜{f.test_end}: 件数不足で飛ばす")
            continue
        print(f"  窓{f.index} {f.test_start}〜{f.test_end} "
              f"訓練{len(tr):,} / 評価{len(te):,}")
        for name, sc in fit_and_score(tr, te, cols, preset).items():
            part = te.copy()
            part["score"] = sc
            out.setdefault(name, []).append(part)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="本番モデルの実収益を測る")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--features", default="all")
    ap.add_argument("--k-list", default="1,5,10,20",
                    help="上位何%%で測るか。1つの k に依存した結論を避けるため複数")
    ap.add_argument("--walkforward", action="store_true",
                    help="直近1年だけでなく、10窓ぶんをまとめて測る")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "outcome_check.json"))
    ap.add_argument("--doc", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "MODEL_VS_SIZE.md"))
    args = ap.parse_args(argv)

    ks = [float(x) for x in args.k_list.split(",") if x.strip()]
    cols = F.columns(args.features)
    ds = pd.read_parquet(args.dataset)
    ds["Date"] = pd.to_datetime(ds["Date"])
    print(f"[load] {len(ds):,}件 / 正例率 {ds['label'].mean()*100:.2f}% "
          f"/ 特徴量 {args.features}（{len(cols)}列）")

    # 参照ホライズンの実リターンをパネルから作る。
    # データセットの列（future_rise / end_level）はラベル定義と一緒に動くので使わない
    paths = sorted(glob.glob(os.path.join(args.data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    panel = S.Panels(bars).get(B.HIGH_WINDOW)
    print(f"[ref] 参照ホライズン {S.REF_HORIZON}営業日（ラベル定義に依存しない物差し）")
    ds = ds.merge(S.reference_outcome(panel), on=["Code", "Date"], how="left")

    if args.walkforward:
        import walkforward as WF
        folds = WF.make_folds(pd.to_datetime(ds["Date"]), min_train_months=36,
                              test_months=6, step_months=6,
                              embargo_days=B.RISE_HORIZON)
        print(f"[wf] {len(folds)}窓で学習してスコアをまとめる")
        scored = pooled_scores(ds, cols, args.features, folds)
        span = (f"{folds[0].test_start} 〜 {folds[-1].test_end}", len(folds))
    else:
        parts = T.holdout_split(ds)
        train, test = parts["train"], parts["test"]
        scored = {}
        for name, sc in fit_and_score(train, test, cols, args.features).items():
            t = test.copy()
            t["score"] = sc
            scored[name] = t
        span = (f"{test['Date'].min().date()} 〜 {test['Date'].max().date()}", 1)

    rows = []
    for name, t in scored.items():
        n_band = int(t[BAND_COL].notna().sum())
        print(f"\n=== {name} ===  評価 {len(t):,}件"
              f"（帯あり {n_band:,}件 / 帯が欠測 {len(t)-n_band:,}件）")
        print(f"  {'上位':>6}{'選択':>10}{'上位の実収益':>13}{'全件':>9}{'勝率':>8}"
              f"{'差':>8}{'95%区間':>18}{'対 大型順':>10}{'上回った割合':>12}")
        for k in ks:
            for by_band in (False, True):
                e = edge_stats(t, "score", k, by_band, n_boot=args.n_boot)
                v = paired_vs(t, "score", "log_market_cap", True, k, by_band,
                              n_boot=args.n_boot)
                e["vs_cap"] = v
                e["model"] = name
                rows.append(e)
                tag = "帯を揃える" if by_band else "全体から"
                lo, hi = e["ci"]
                ci = "[{:+.2f},{:+.2f}]".format(lo, hi)
                print(f"  {k:>5.0f}%{tag:>10}{e['top']:>+12.2f}%{e['all']:>+8.2f}%"
                      f"{e['top_win']*100:>7.1f}%{e['edge']:>+8.2f}{ci:>18}"
                      f"{v.get('diff', float('nan')):>+9.2f}"
                      f"{v.get('p_positive', float('nan'))*100:>11.1f}%")
        # 参考: 他の単純な規則とも比べる（上位5%のみ）
        other = {}
        for nm, col, desc in S.NAIVE_RULES:
            if col == "log_market_cap" or col not in t.columns:
                continue
            other[nm] = paired_vs(t, "score", col, desc, 5.0, False,
                                  n_boot=args.n_boot)
        rows[-1]["vs_other"] = other

    payload = {"label": B.DEFAULT_RISE.name, "preset": args.features,
               "reference_horizon": S.REF_HORIZON,
               "walkforward": bool(args.walkforward),
               "span": span[0], "n_folds": span[1],
               "k_list": ks, "band_col": BAND_COL, "results": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out}")
    write_doc(args.doc, payload)
    print(f"[done] {args.doc}")
    return 0


def write_doc(path: str, payload: Dict) -> None:
    L = [
        "# モデルは「大型の高値更新を買うだけ」を超えているか",
        "",
        "`research/outcome_check.py --walkforward` が生成する。手で書き換えない。",
        "",
        "実収益（参照ホライズン"
        f"{payload['reference_horizon']}営業日後の5日平均終値の上昇率）で、",
        "モデルの上位k%と、`log_market_cap` で並べただけの上位k%を、",
        "**同じ行集合の上で対で**比べる。別々の95%区間の重なりでは判定できない。",
        "",
        "「帯を揃える」は時価総額の帯ごとに上位k%を取る選び方。選ばれた集合の",
        "帯構成が母集団と同じになるので、「大きい帯を多めに取った」効果が消える。",
        "ここで差が残れば、規模以外の情報を使えていることになる。",
        "",
        "## 条件",
        "",
        f"- ラベル: {payload['label']} / 特徴量: `{payload['preset']}`",
        f"- 評価: {payload['span']}"
        + (f"（ウォークフォワード{payload['n_folds']}窓をまとめたもの）"
           if payload["walkforward"] else "（単一分割）"),
        f"- 区間は日付単位のブロックブートストラップ",
        "",
        "## 結果",
        "",
        "| モデル | 上位 | 選び方 | 上位の実収益 | 全件 | 上位の勝率 | 差 | 95%区間 | 対 大型順 | 95%区間 | 上回った割合 |",
        "|---|---:|---|---:|---:|---:|---:|:---:|---:|:---:|---:|",
    ]
    for r in payload["results"]:
        v = r.get("vs_cap") or {}
        L.append(
            f"| {r['model']} | {r['k_pct']:.0f}% "
            f"| {'帯を揃える' if r['by_band'] else '全体から'} "
            f"| {r['top']:+.2f}% | {r['all']:+.2f}% | {r['top_win']*100:.1f}% "
            f"| {r['edge']:+.2f}pt | [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] "
            f"| {v.get('diff', float('nan')):+.2f}pt "
            f"| [{v.get('ci', [float('nan')]*2)[0]:+.2f},"
            f"{v.get('ci', [float('nan')]*2)[1]:+.2f}] "
            f"| {v.get('p_positive', float('nan'))*100:.1f}% |")
    other = next((r["vs_other"] for r in reversed(payload["results"])
                  if r.get("vs_other")), None)
    if other:
        L += ["", "## 参考: 他の単純な規則との対比較（上位5% / 全体から）", "",
              "| 規則 | 差 | 95%区間 | 上回った割合 |", "|---|---:|:---:|---:|"]
        for nm, v in other.items():
            if not v:
                continue
            L.append(f"| {nm} | {v['diff']:+.2f}pt "
                     f"| [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] "
                     f"| {v['p_positive']*100:.1f}% |")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
