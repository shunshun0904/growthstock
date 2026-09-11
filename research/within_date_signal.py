#!/usr/bin/env python3
"""
日付内での分離力を直接測る。

なぜこれを先に測るか:
  ウォークフォワードで測れるのは「無情報を上回るか」までで、その優位が
  日付をまたいだ地合いの当て方から来ているのか、同じ日の銘柄を
  並べ替える力から来ているのかは分からない。実運用で見るのは
  「その日に高値更新した数銘柄のうちどれを買うか」なので、
  必要なのは後者だけである。
  次の候補は learning-to-rank（同一日内の順位を直接最適化する）だが、
  LTR が利用できるのは「同じ日の銘柄どうしを分ける情報」だけである。
  その情報が存在しないなら、LTR を作っても取り出せない。
  モデルを組む前に、情報があるかどうかを測る。

測り方:
  同一日内の（正例, 負例）ペアを全日から集め、正例のほうが値が大きい
  割合を出す。これは日付内 AUC と同じもので、0.5 が分離力ゼロ。

  以前は日付ごとに AUC を出して平均していたが、推定に1日30件を要求して
  いた。この母集団は平均7.2銘柄/日なので 1,739日のうち14日しか残らず、
  条件付きに至っては0セルだった。閾値が、母集団を月末×全銘柄から
  高値更新日に変えたときのまま追随していなかった。
  ペアなら1日2銘柄あれば1つ作れるので、全日を使える。

  CI は日付ブートストラップで出す。ペアは同じ日のもの同士で相関するので、
  ペア単位でリサンプルすると独立を仮定することになり CI が狭く出る。

測るもの:
  1. 単独の分離力
     全ペアでの一致率。

  2. R_high を与えた上での増分
     同じ日で r_high の差が RHIGH_TOL 以内のペアだけに絞る。
     「同じくらい高値付近で引けた銘柄どうしを、その特徴量が更に分けられるか」。
     LTR に見込みがあるかを決めるのはこちら。
     1 で高くても R_high と相関しているだけなら、2 で 0.5 に落ちる。

なお順位版 (*_r) は日付内での単調変換なので、日付内 AUC は元の列と一致する。
測る意味が無いので対象から外す（一致することはテストで確認している）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
from train_model import DATA_DIR  # noqa: E402
from walkforward import sign_test  # noqa: E402

#: 「同じ日で r_high が近い」と見なす差（ポイント）。
#: r_high は中央値99.08・p25 97.96 なので、1.0pt 以内なら
#: 「同じくらい高値付近で引けた者どうし」と言える。
RHIGH_TOL = 1.0

#: 日付ブートストラップの反復回数。ペアは日付内で相関するので、
#: ペアを直接リサンプルすると独立性を仮定してしまい CI が狭く出る。
#: 日付ごと引き直す。
N_BOOT = 1000

#: 対象にする最小ペア数。これを下回る特徴量は推定が不安定なので別掲。
MIN_PAIRS = 200


def _date_pairs(df: pd.DataFrame, on: Optional[str] = None,
                tol: float = RHIGH_TOL):
    """
    同一日内の (正例, 負例) ペアを全日から集める。

    日付ごとに AUC を出して平均する方式はここでは使えない。
    AUC の推定に1日30件を要求すると、この母集団（平均7.2銘柄/日）では
    1,739日のうち14日しか残らず、条件付きに至っては0セルになった。
    実際そうなっていた。ペアなら1日2銘柄あれば1つ作れる。

    on を指定すると、その列の差が tol 以内のペアだけに絞る。
    「高値からの距離が同程度の銘柄どうし」に限定するため。

    戻り値は (日付ID, 正例の行位置, 負例の行位置) の3配列。
    """
    y = df["label"].to_numpy(dtype=int)
    codes = pd.factorize(df["Date"], sort=True)[0]
    ref = df[on].to_numpy(dtype=float) if on else None

    d_ids, pi, ni = [], [], []
    order = np.argsort(codes, kind="stable")
    for _, idx in itertools.groupby(order, key=lambda i: codes[i]):
        idx = np.fromiter(idx, dtype=np.int64)
        p = idx[y[idx] == 1]
        n = idx[y[idx] == 0]
        if len(p) == 0 or len(n) == 0:
            continue
        a, b = np.meshgrid(p, n, indexing="ij")
        a, b = a.ravel(), b.ravel()
        if ref is not None:
            keep = np.abs(ref[a] - ref[b]) <= tol
            keep &= np.isfinite(ref[a]) & np.isfinite(ref[b])
            a, b = a[keep], b[keep]
            if len(a) == 0:
                continue
        d_ids.append(np.full(len(a), codes[idx[0]], dtype=np.int64))
        pi.append(a)
        ni.append(b)
    if not pi:
        return (np.empty(0, np.int64),) * 3
    return np.concatenate(d_ids), np.concatenate(pi), np.concatenate(ni)


def _concordance(x: np.ndarray, pi: np.ndarray, ni: np.ndarray) -> np.ndarray:
    """
    ペアごとに 1（正例のほうが大きい）/ 0.5（同点）/ 0 を返す。

    欠測を含むペアは NaN。AUC と同じく同点は 0.5 と数える。
    """
    xp, xn = x[pi], x[ni]
    ok = np.isfinite(xp) & np.isfinite(xn)
    out = np.where(xp > xn, 1.0, np.where(xp == xn, 0.5, 0.0))
    return np.where(ok, out, np.nan)


def _summarise(name: str, conc: np.ndarray, d_ids: np.ndarray,
               seed: int = 0) -> Optional[Dict]:
    """
    プールした一致率（= 日付内 AUC）と、日付ブートストラップの95%CI。

    ペアは同じ日のもの同士で相関するので、ペア単位でリサンプルすると
    独立を仮定することになり CI が狭く出る。日付ごと引き直す。
    """
    ok = np.isfinite(conc)
    conc, d_ids = conc[ok], d_ids[ok]
    if len(conc) < 1:
        return None
    auc = float(conc.mean())

    uniq = np.unique(d_ids)
    # 日付ごとの一致率（符号の一貫性を見るため）
    by_date = pd.Series(conc).groupby(pd.Series(d_ids)).mean()
    wins = int((by_date > 0.5).sum())
    losses = int((by_date < 0.5).sum())

    rng = np.random.default_rng(seed)
    idx_by_date = {d: np.flatnonzero(d_ids == d) for d in uniq}
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_date[d] for d in pick])
        boot[b] = conc[sel].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "feature": name,
        "n_pairs": int(len(conc)),
        "n_dates": int(len(uniq)),
        "mean_auc": auc,
        "ci_low": float(lo),
        "ci_high": float(hi),
        # 0.5 からどれだけ離れているか。向きは問わない
        "abs_edge": float(abs(auc - 0.5)),
        # CI が 0.5 をまたがなければ、日付間の相関を考慮しても差がある
        "significant": bool(lo > 0.5 or hi < 0.5),
        "wins": wins,
        "losses": losses,
        "p_sign": sign_test(wins, losses),
    }


def marginal(df: pd.DataFrame, cols: List[str]) -> List[Dict]:
    """同一日内のペアでの一致率（プール）。"""
    d_ids, pi, ni = _date_pairs(df)
    out = []
    for c in cols:
        s = _summarise(c, _concordance(df[c].to_numpy(dtype=float), pi, ni),
                       d_ids)
        if s:
            out.append(s)
    out.sort(key=lambda r: -r["abs_edge"])
    return out


def conditional(df: pd.DataFrame, cols: List[str], on: str = "r_high",
                tol: float = RHIGH_TOL) -> List[Dict]:
    """on が近い者どうしのペアに限った一致率。"""
    if on not in df.columns:
        return []
    d_ids, pi, ni = _date_pairs(df, on=on, tol=tol)
    out = []
    for c in cols:
        if c == on:
            continue
        s = _summarise(c, _concordance(df[c].to_numpy(dtype=float), pi, ni),
                       d_ids)
        if s:
            out.append(s)
    out.sort(key=lambda r: -r["abs_edge"])
    return out


def _table(rows: List[Dict], limit: int) -> List[str]:
    lines = ["| 特徴量 | ペア数 | 日付数 | 日付内AUC | 95%CI | 0.5からの差 | 判定 |",
             "| --- | ---: | ---: | ---: | :---: | ---: | --- |"]
    for r in rows[:limit]:
        mark = "**有意**" if r["significant"] else "誤差"
        lines.append(
            f"| `{r['feature']}` | {r['n_pairs']:,} | {r['n_dates']:,} | "
            f"{r['mean_auc']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
            f"{r['abs_edge']:.4f} | {mark} |")
    return lines


def _split_by_reliability(rows: List[Dict]):
    """ペア数が足りない特徴量を分ける。推定が不安定なので同列に並べない。"""
    ok = [r for r in rows if r["n_pairs"] >= MIN_PAIRS]
    thin = [r for r in rows if r["n_pairs"] < MIN_PAIRS]
    return ok, thin


def build_report(marg: List[Dict], cond: List[Dict], n_dates: int, n_cells: int,
                 n_feats: int) -> str:
    ref_m = next((r for r in marg if r["feature"] == "r_high"), None)
    lines = [
        "# 日付内での分離力",
        "",
        "`research/within_date_signal.py` の出力。**実測値のみ**を記載する。",
        "",
        "ウォークフォワード（[MODEL_WALKFORWARD.md](MODEL_WALKFORWARD.md)）が測るのは",
        "「無情報を上回るか」までで、その優位が地合いの当て方から来ているのか、",
        "同じ日の銘柄を並べ替える力から来ているのかは区別できない。",
        "実運用で必要なのは後者（その日の高値更新銘柄のどれを買うか）で、",
        "次の候補である learning-to-rank が利用できるのもそこだけなので、",
        "モデルを組む前にその情報が存在するかを測る。",
        "",
        "- 同一日内の（正例, 負例）ペアを全日から集め、正例のほうが値が大きい",
        "  割合を出す。これは日付内 AUC と同じもので、0.5 が分離力ゼロ。",
        "  日付内で完結するので正例率の局面差に影響されない",
        f"- 対象日付: {n_dates:,}（ペアが1つ以上作れた日）",
        f"- 条件付き: 同じ日で `r_high` の差が {RHIGH_TOL}pt 以内のペアに限る",
        f"- ペア数が {MIN_PAIRS} 未満の特徴量は別掲（推定が不安定なため）",
        f"- 95%CI は**日付ブートストラップ**（{N_BOOT}回）。ペアは同じ日のもの",
        "  同士で相関するので、ペア単位でリサンプルすると CI が狭く出る",
        "",
        "> **測り方を変えました。** 以前は日付ごとに AUC を出して平均していたが、",
        "> 推定に1日30件を要求していた。この母集団は平均7.2銘柄/日なので",
        f"> 1,739日のうち14日しか残らず、条件付きに至っては0セルだった。",
        "> 閾値が母集団を高値更新日に変える前のまま追随していなかった。",
        "> ペアなら1日2銘柄あれば1つ作れるので、全日を使える。",
        "",
        "## 1. 単独の分離力（日付内）",
        "",
    ]
    if ref_m:
        lines += [f"基準 `r_high` の日付内AUC は **{ref_m['mean_auc']:.4f}** "
                  f"（{ref_m['n_pairs']:,}ペア / "
                  f"CI [{ref_m['ci_low']:.4f}, {ref_m['ci_high']:.4f}]）。", ""]
    marg_ok, marg_thin = _split_by_reliability(marg)
    lines += _table(marg_ok, 20)
    if marg_thin:
        lines += ["", f"### ペア数が{MIN_PAIRS}未満（参考）", "",
                  "推定が不安定なので上の表とは分けて示す。", ""]
        lines += _table(marg_thin, 10)
    lines += [
        "",
        "## 2. R_high を与えた上での増分",
        "",
        f"同じ日で `r_high` の差が {RHIGH_TOL}pt 以内のペアだけに絞って測る。",
        "「同じくらい高値付近で引けた銘柄どうしを、その特徴量が更に分けられるか」。",
        "**LTR に見込みがあるかを決めるのはこちら。**",
        "1 で高くても `r_high` と相関しているだけの特徴量は、ここで 0.5 に落ちる。",
        "",
    ]
    cond_ok, cond_thin = _split_by_reliability(cond)
    lines += _table(cond_ok, 25)
    if cond_thin:
        lines += ["", f"### ペア数が{MIN_PAIRS}未満（参考）", "",
                  "推定が不安定なので上の表とは分けて示す。", ""]
        lines += _table(cond_thin, 10)
    lines += [
        "",
        "## 読み方",
        "",
        "- 2 の CI が軒並み 0.5 をまたぐなら、**日付内に取り出せる情報が無い**。",
        "  learning-to-rank を作っても結果は変わらない（存在しない情報は最適化できない）",
        "- 2 で CI が 0.5 をまたがない特徴量があれば、それが LTR で拾える候補",
        "- AUC が 0.5 未満（負の向き）でも情報はある。符号を反転すればよい",
        "- 判定は CI で見ること。ペアは日付内で相関するので、ペア数の多さは",
        "  そのまま信頼性にならない",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="日付内での分離力を測る")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "MODEL_WITHIN_DATE.md"))
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    # 順位版は日付内の単調変換なので AUC が元の列と一致する。測る意味が無い
    cols = [c for c in F.columns("all") if c in df.columns]
    n_dates = df["Date"].nunique()
    print(f"[load] {len(df):,}件 / {n_dates}日付 / 特徴量 {len(cols)}個")

    print("\n[1] 単独の分離力（同一日内ペアの一致率）")
    marg = marginal(df, cols)
    for r in marg[:8]:
        print(f"  {r['feature']:<24} AUC {r['mean_auc']:.4f} "
              f"({r['n_pairs']:,}ペア / {r['n_dates']:,}日)")

    print(f"\n[2] r_high の差 {RHIGH_TOL}pt 以内のペアに限定")
    cond = conditional(df, cols)
    for r in cond[:8]:
        print(f"  {r['feature']:<24} AUC {r['mean_auc']:.4f} "
              f"CI[{r['ci_low']:.4f},{r['ci_high']:.4f}] "
              f"{'有意' if r['significant'] else '誤差'}")

    n_cells = max((r["n_pairs"] for r in cond), default=0)
    body = build_report(marg, cond, n_dates, n_cells, len(cols))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"\n[done] {args.out}")

    with open(os.path.join(DATA_DIR, "within_date.json"), "w", encoding="utf-8") as fh:
        json.dump({"marginal": marg, "conditional": cond,
                   "n_dates": int(n_dates)}, fh, ensure_ascii=False, indent=2)

    # 判定は日付ブートストラップの CI で行う。ペアは日付内で相関するので、
    # ペア数を根拠にした p 値は独立を仮定してしまう。
    strong = [r for r in cond
              if r["n_pairs"] >= MIN_PAIRS and r["significant"]
              and r["abs_edge"] >= 0.02]
    print(f"\n[判定] r_high を揃えても分離力が残る特徴量 "
          f"(ペア>={MIN_PAIRS} / CI が0.5をまたがない / |AUC-0.5|>=0.02):")
    if strong:
        for r in strong[:15]:
            print(f"  {r['feature']:<24} AUC {r['mean_auc']:.4f} "
                  f"CI[{r['ci_low']:.4f},{r['ci_high']:.4f}] "
                  f"(差 {r['mean_auc'] - 0.5:+.4f})")
        print(f"  計 {len(strong)}個 -> learning-to-rank に見込みがある")
    else:
        print("  なし -> learning-to-rank を作っても結果は変わらない見込み")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
