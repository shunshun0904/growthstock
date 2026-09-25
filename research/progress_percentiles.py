#!/usr/bin/env python3
"""
8軸の「進捗期待」の点数表（四半期ごとの、進捗の比率の過去の分布）を作る。

進捗期待は 2026-09-25 から次の定義（運用者の選択 A、README §4.1 / §5）:

  進捗率   = その四半期までの累計営業利益 ÷ 通期の会社予想営業利益
  基準     = 前年の同じ四半期の進捗（前年のその四半期までの累計営業利益 ÷ 前年の通期実績）
             取れない（前年の数字が無い・0以下・決算期の変更）ときは 四半期 × 25%
  比率     = 進捗率 ÷ 基準         （1.0 = 例年どおりのペース）
  点数     = 同じ四半期・同じ物差しの、過去の開示の中での順位 × 10（例年並みが約5点）

比率のばらつきは四半期で大きく違う（第1四半期は第3四半期より何倍も広い）ので、
固定の幅で点数にすると第1四半期の多くが0点か10点に張り付く。順位にすれば
四半期どうしで同じ意味の点数になる。

基準の計算は画面のデータを作るフェッチャーの関数（scripts/jquants_data_fetcher.py の
fundamentals_as_of / progress_benchmark）をそのまま呼ぶ。ここで別に書くと、
点数表と画面の値が少しずつずれていく。各開示の日に、その日までの開示だけで計算する。

物差しは2つ:
  seasonal  前年同期の基準が取れた開示だけ。比率 = 進捗率 ÷ 前年同期の進捗
            （前年同期の進捗が Q×12.5% 未満なら Q×12.5% で割る。フェッチャーの 'floor'）
  linear    **全部の**開示。比率 = 進捗率 ÷ (Q×25%)
            基準が取れない銘柄と、手入力・シミュレーター（基準を知らない）に使う。
            取れない開示だけで作ると、前年赤字の会社と「データの最初の年」の会社の
            寄せ集めになり、真ん中（5点）が普通の会社の位置からずれる

出力は src/lib/scoring.js の PROGRESS_TABLE にそのまま貼れる形。
--auc を付けると、比率が「通期実績がその時点の会社予想を上回るか」をどれだけ
見分けるかを、季節性の基準（下限あり・なし）と Q×25% で比べる（標準エラー出力）。

  python3 research/progress_percentiles.py --data-dir research/_data [--auc]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import jquants_data_fetcher as F  # noqa: E402

#: 点数表の区切り（百分位）。src/lib/scoring.js の PROGRESS_PCTS と同じ
PCTS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
#: 前年同期の物差しで点数にする基準の種類（下限を当てた行も同じ表で順位を付ける）
SEASONAL = ("seasonal", "floor")
#: フェッチャーの進捗の計算が読む列（ほかの列は無くても None として扱われる）
COLS = ["Code", "DiscDate", "CurPerType", "CurFYSt", "Sales", "OP", "NP", "EPS", "FOP"]


def load(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "fins_*.parquet")))
    if not paths:
        raise SystemExit(f"決算データ（fins_*.parquet）がありません: {data_dir}")
    f = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    for c in ("DiscDate", "CurFYSt"):
        f[c] = pd.to_datetime(f[c], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ("Sales", "OP", "NP", "EPS", "FOP"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    return f.dropna(subset=["DiscDate"])


def events(f: pd.DataFrame, floor: float | None = None) -> pd.DataFrame:
    """
    四半期（1〜3）の開示ごとに、その日にフェッチャーが出す進捗率・基準・物差し。

    同じ四半期の訂正は数えない（最初の開示＝その時点で見えていた値だけ）。
    floor を渡すと、その間だけフェッチャーの下限（PROGRESS_FLOOR）を差し替える
    （--auc で下限なしと比べるため）。
    """
    saved = F.PROGRESS_FLOOR
    if floor is not None:
        F.PROGRESS_FLOOR = floor
    try:
        return _events(f)
    finally:
        F.PROGRESS_FLOOR = saved


def _events(f: pd.DataFrame) -> pd.DataFrame:
    out = []
    for code, g in f.groupby("Code", sort=False):
        rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()}
                for r in g.to_dict("records")]
        first = {}
        for r in rows:
            q = F.QUARTER_MAP.get(r.get("CurPerType"))
            if q not in (1, 2, 3) or not F._is_financial_statement(r):
                continue
            key = (r.get("CurFYSt"), q)
            if key not in first or r["DiscDate"] < first[key]["DiscDate"]:
                first[key] = r
        for (fy, q), r in first.items():
            m = F.fundamentals_as_of(rows, r["DiscDate"])
            # その日の最新の四半期がこの開示であること（古い期の出し直しの日を除く）
            if (m["progressRate"] is None or m["quarter"] != q
                    or m["disclosedDate"] != r["DiscDate"] or m["progressBasis"] is None):
                continue
            out.append({"Code": code, "DiscDate": r["DiscDate"], "CurFYSt": fy, "q": q,
                        "FOP": r.get("FOP"), "progress": m["progressRate"],
                        "bench": m["progressBenchmark"], "basis": m["progressBasis"]})
    d = pd.DataFrame(out)
    d["ratio"] = d["progress"] / d["bench"]
    d["ratio_linear"] = d["progress"] / (d["q"] * 25.0)
    return d


def table(d: pd.DataFrame) -> dict:
    out = {"seasonal": {}, "linear": {}}
    for q in (1, 2, 3):
        x = d.loc[d["basis"].isin(SEASONAL) & (d["q"] == q), "ratio"].dropna()
        out["seasonal"][q] = (len(x), [round(float(v), 3) for v in x.quantile(PCTS)])
        x = d.loc[d["q"] == q, "ratio_linear"].dropna()
        out["linear"][q] = (len(x), [round(float(v), 3) for v in x.quantile(PCTS)])
    return out


def auc_report(f: pd.DataFrame, d: pd.DataFrame) -> None:
    """
    通期実績 > その時点の会社予想 を、比率がどれだけ見分けるか（AUC）。
    本番の比率（下限あり）・下限なし・Q×25% を、同じ開示（前年同期の基準が取れたもの）で比べる。
    """
    from sklearn.metrics import roc_auc_score

    fy = f[f["CurPerType"] == "FY"].dropna(subset=["OP"])
    fy = (fy.sort_values("DiscDate").drop_duplicates(["Code", "CurFYSt"], keep="last")
          [["Code", "CurFYSt", "OP"]].rename(columns={"OP": "fy_op"}))
    raw = events(f, floor=0.0)[["Code", "DiscDate", "q", "ratio", "bench"]]
    raw = raw.rename(columns={"ratio": "ratio_raw", "bench": "bench_raw"})
    x = (d[d["basis"].isin(SEASONAL)]
         .merge(raw, on=["Code", "DiscDate", "q"], how="inner")
         .merge(fy, on=["Code", "CurFYSt"], how="inner"))
    x = x[x["FOP"] > 0]
    y = (x["fy_op"] > x["FOP"]).astype(int)
    defs = {"下限あり（本番）": x["ratio"], "下限なし": x["ratio_raw"],
            "Q×25%": x["ratio_linear"]}

    def auc(m, v):
        return roc_auc_score(y[m], v[m]) if y[m].nunique() == 2 else float("nan")

    small = x["bench_raw"] < x["q"] * 25.0 * F.PROGRESS_FLOOR
    years = sorted(x["DiscDate"].str[:4].unique())
    err = sys.stderr
    print(f"AUC（通期実績 > その時点の会社予想。前年同期の基準が取れた開示 {len(x):,}件）", file=err)
    print("  " + "".ljust(14) + "全体   Q1     Q2     Q3     基準が小さい開示"
          f"({int(small.sum()):,}件)", file=err)
    for k, v in defs.items():
        row = [auc(x.index == x.index, v)] + [auc(x["q"] == q, v) for q in (1, 2, 3)]
        print(f"  {k.ljust(12)}  " + "  ".join(f"{a:.3f}" for a in row)
              + f"  {auc(small, v):.3f}", file=err)
    print("  年ごと（" + " / ".join(years) + "）", file=err)
    for k, v in defs.items():
        yr = x["DiscDate"].str[:4]
        print(f"  {k.ljust(12)}  " + " ".join(f"{auc(yr == t, v):.3f}" for t in years), file=err)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="進捗期待の点数表を作る")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "_data"))
    ap.add_argument("--auc", action="store_true", help="見分ける力を標準エラー出力に出す")
    args = ap.parse_args(argv)
    f = load(args.data_dir)
    d = events(f)
    t = table(d)
    n_s = int(d["basis"].isin(SEASONAL).sum())
    print(f"// 作成: research/progress_percentiles.py（四半期の開示 {len(d):,}件、"
          f"うち前年同期の基準あり {n_s:,}件、{d['DiscDate'].min()}〜{d['DiscDate'].max()}）")
    print("export const PROGRESS_TABLE = {")
    for basis in ("seasonal", "linear"):
        print(f"  {basis}: {{")
        for q in (1, 2, 3):
            n, xs = t[basis][q]
            print(f"    {q}: [{', '.join(f'{v:.3f}' for v in xs)}],   // {n:,}件")
        print("  },")
    print("};")
    if args.auc:
        auc_report(f, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
