#!/usr/bin/env python3
"""
実験40: 学習の前に、全特徴量（205列）を1本ずつ見る。

運用者の指示（2026-09-22）
  「学習や検証に入る前に改めて全特徴量のEDAをお願いします」

eda_stats.py が出すのは**列そのものの性質**（欠損・外れ値・分布・冗長）。
学習の前に知りたいのはそれだけではなく、**その列に検出力があるか**。
ここは実験20・23 と同じ方法で測る。

  本番の out-of-fold と同じ 11窓（36ヶ月訓練 / 6ヶ月テスト / 6ヶ月刻み、
  エンバーゴ 20営業日）で、上位10% と下位10% の実収益（ret_o1_20）の
  超過を窓ごとに取り、窓をまたいだ平均 / SE を z とする。

  足切りは |z| > 2。特徴量が205本もあるので偶然の |z|>2 は数本出る。
  同じ群の中で符号が揃うか、群ごとの本数で読む（1本だけの |z|>2 は
  採らない）。ノイズ床は 0.143pt（実験11）。

「上位で効く」と「下位で効く」は別物である。実験37 で分かったとおり、
ファンダメンタルズの多くは**下位を避ける**側にしか効かない。買う理由に
なる列と、見送る理由になる列を分けて読むために両側を出す。

  python3 research/exp/e40_feature_eda.py
  -> research/_data/oof/e40_feature_eda.csv

eda_stats.py の出力（research/eda.json）があれば、欠損・外れ値・年ごとの
欠損率も突き合わせて出す。無くても検出力だけは出る。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import walkforward as WF  # noqa: E402
from e20_annual_trajectory import OUTCOME  # noqa: E402
from e23_dimension_screen import screen_both  # noqa: E402
from train_production import (  # noqa: E402
    OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS)

OUT = os.path.join(lab.DATA_DIR, "oof", "e40_feature_eda.csv")
EDA_JSON = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "eda.json")

#: 足切り。|z| がこれを超えたら「効いている」と読む
Z_CUT = 2.0
#: 実験11 で測ったノイズ床（pt）
NOISE_FLOOR = 0.143
#: 取り込み待ちで空になりうる群。バックフィル前はここが全NaN になる
PENDING_GROUPS = ("fwd", "holders_lvs", "holders_major", "holders_cross",
                  "margin_alert", "earn_ahead", "flow")


def load_eda(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def year_spread(miss_by_year: Dict[str, Dict[str, float]], col: str,
                skip: Optional[str] = None) -> float:
    """年ごとの欠損率の幅。大きいほど、窓によって列の性格が変わる。

    skip に年を渡すとその年を外して測る。**初年度は必ず大きく欠ける。**
    4四半期ぶんの履歴や6ヶ月の高値が要る列は、母集団の開始直後には
    作れないため。実測（2026-09-22）では 40pt を超えて動く36列すべてで
    最悪の年が 2018 であり、2018 を外すと**1本も残らなかった**。
    しかも 2018 は最初のテスト窓（2021-05）より前なので、評価には
    一度も使われない。初年度ぶんは警告しないほうが正しい。
    """
    ys = miss_by_year.get(col) or {}
    if skip:
        ys = {y: v for y, v in ys.items() if y != skip}
    if len(ys) < 2:
        return float("nan")
    v = list(ys.values())
    return float(max(v) - min(v))


def summarise_groups(res: pd.DataFrame) -> pd.DataFrame:
    """群ごとにまとめる。1本だけの |z|>2 は偶然なので、本数で読む。"""
    rows = []
    for g, part in res.groupby("group", sort=False):
        live = part[part["n_win"].fillna(0) > 0]
        rows.append({
            "group": g,
            "n_col": len(part),
            "欠損中央": part["missing_pct"].median(),
            "測れた": len(live),
            "上位z>2": int((live["top_z"] > Z_CUT).sum()),
            "上位z<-2": int((live["top_z"] < -Z_CUT).sum()),
            "下位z>2": int((live["bot_z"] > Z_CUT).sum()),
            "下位z<-2": int((live["bot_z"] < -Z_CUT).sum()),
            "最大|z|": live["max_abs_z"].max() if len(live) else np.nan,
            "AUC中央": live["auc_win"].median() if len(live) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["効いた本数"] = (out["上位z>2"] + out["上位z<-2"]
                    + out["下位z>2"] + out["下位z<-2"])
    return out.sort_values("最大|z|", ascending=False, na_position="last")


def print_table(df: pd.DataFrame, cols: List[tuple], limit: Optional[int] = None) -> None:
    head = "  " + "".join(f"{ja:>{w}}" if i else f"{ja:<{w}}"
                          for i, (_, ja, w, _) in enumerate(cols))
    print(head)
    for _, r in (df.head(limit) if limit else df).iterrows():
        cells = []
        for i, (key, _, w, fmt) in enumerate(cols):
            v = r.get(key)
            if isinstance(v, str):
                s = v
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                s = "—"
            else:
                # 整数で出したい列も DataFrame からは float で返る
                s = format(int(v) if fmt == "d" else v, fmt)
            cells.append(f"{s:>{w}}" if i else f"{s:<{w}}")
        print("  " + "".join(cells))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="全特徴量の EDA（検出力つき）")
    ap.add_argument("--dataset", default=None,
                    help="省略時は lab.frame()（research/_data/dataset.parquet）")
    ap.add_argument("--eda", default=EDA_JSON, help="eda_stats.py の出力")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--top", type=int, default=30, help="表に出す本数")
    args = ap.parse_args(argv)

    if args.dataset:
        lab.DATASET = args.dataset
        lab.CACHE = args.dataset + ".labcache"
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    folds = WF.make_folds(frame["Date"], min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    windows = [(np.datetime64(f.test_start), np.datetime64(f.test_end))
               for f in folds]

    feats = [c for c in F.columns("all_plus") if c in frame.columns]
    absent = [c for c in F.columns("all_plus") if c not in frame.columns]
    prod = set(F.columns("all"))
    pending = {c for g in PENDING_GROUPS for c in F.GROUPS.get(g, ())}

    print(f"母集団 {len(frame):,}件 / 銘柄 {frame['Code'].nunique():,} / "
          f"{frame['Date'].min().date()}〜{frame['Date'].max().date()}")
    print(f"評価窓 {len(windows)}本（{OOF_MIN_TRAIN_MONTHS}ヶ月訓練 / "
          f"{OOF_TEST_MONTHS}ヶ月テスト / {OOF_STEP_MONTHS}ヶ月刻み）/ "
          f"物差し {OUTCOME}")
    print(f"特徴量 {len(feats)}列（本番 {len(prod)} + 追加 {len(feats)-len(prod)}）"
          + (f" / データセットに無い列 {len(absent)}本: {absent[:6]}" if absent else ""))
    print(f"足切り |z| > {Z_CUT} / ノイズ床 {NOISE_FLOOR}pt")

    print("\n[screen] 205列を1本ずつ、11窓の両側で測る…", flush=True)
    res = screen_both(frame, feats, windows)
    res["group"] = [F.group_of(c) for c in res["feature"]]
    res["本番"] = ["○" if c in prod else "" for c in res["feature"]]
    res["取込待ち"] = ["●" if c in pending else "" for c in res["feature"]]

    eda = load_eda(args.eda)
    st, my = eda.get("stats", {}), eda.get("missing_by_year", {})
    res["missing_pct"] = [st.get(c, {}).get("missing_pct", np.nan)
                          for c in res["feature"]]
    res["outlier_pct"] = [st.get(c, {}).get("outlier_pct", np.nan)
                          for c in res["feature"]]
    first_year = str(frame["Date"].dt.year.min())
    res["year_spread_all"] = [year_spread(my, c) for c in res["feature"]]
    res["year_spread"] = [year_spread(my, c, skip=first_year)
                          for c in res["feature"]]
    # eda.json が無い/古いときは screen_both の coverage から埋め直す
    fill = res["missing_pct"].isna()
    res.loc[fill, "missing_pct"] = (1 - res.loc[fill, "coverage"]) * 100

    live = res[res["n_win"].fillna(0) > 0]
    dead = res[res["n_win"].fillna(0) == 0]
    print(f"[screen] 測れた {len(live)}列 / 測れなかった {len(dead)}列"
          f"（うち取り込み待ち {int((dead['取込待ち'] == '●').sum())}列）")

    # ------------------------------------------------------------------ #
    print(f"\n=== 1. 群ごとの要約（{len(res['group'].unique())}群）===")
    print("  「上位z」= その列が高い銘柄の実収益が同じ窓の平均より上か")
    print("  「下位z」= その列が低い銘柄の実収益が同じ窓の平均より上か")
    g = summarise_groups(res)
    print_table(g, [
        ("group", "群", 16, "s"), ("n_col", "列", 4, "d"),
        ("欠損中央", "欠損", 8, ".0f"), ("測れた", "測れた", 7, "d"),
        ("上位z>2", "上位+", 6, "d"), ("上位z<-2", "上位−", 6, "d"),
        ("下位z>2", "下位+", 6, "d"), ("下位z<-2", "下位−", 6, "d"),
        ("効いた本数", "効いた", 7, "d"), ("最大|z|", "最大|z|", 8, ".2f"),
        ("AUC中央", "AUC", 7, ".3f"),
    ])

    # ------------------------------------------------------------------ #
    print(f"\n=== 2. 買う理由になる列（上位10%が勝つ。z > {Z_CUT}）===")
    buy = live[live["top_z"] > Z_CUT].sort_values("top_z", ascending=False)
    if len(buy):
        print_table(buy, [
            ("feature", "列", 26, "s"), ("本番", "本番", 5, "s"),
            ("group", "群", 15, "s"), ("missing_pct", "欠損", 7, ".0f"),
            ("top_pt", "上位超過", 9, "+.2f"), ("top_z", "z", 7, ".2f"),
            ("top_pos", "勝窓", 6, "d"), ("auc_win", "AUC", 7, ".3f"),
        ], args.top)
        print(f"  該当 {len(buy)}列")
    else:
        print("  該当なし")

    print(f"\n=== 3. 見送る理由になる列（下位10%が負ける。z < -{Z_CUT}）===")
    avoid = live[live["bot_z"] < -Z_CUT].sort_values("bot_z")
    if len(avoid):
        print_table(avoid, [
            ("feature", "列", 26, "s"), ("本番", "本番", 5, "s"),
            ("group", "群", 15, "s"), ("missing_pct", "欠損", 7, ".0f"),
            ("bot_pt", "下位超過", 9, "+.2f"), ("bot_z", "z", 7, ".2f"),
            ("bot_pos", "勝窓", 6, "d"), ("auc_win", "AUC", 7, ".3f"),
        ], args.top)
        print(f"  該当 {len(avoid)}列")
    else:
        print("  該当なし")

    # ------------------------------------------------------------------ #
    print(f"\n=== 4. 気をつける列 ===")
    n_first = int((res["year_spread_all"] > 40).sum())
    n_after = int((res["year_spread"] > 40).sum())
    print(f"  年ごとの欠損率の幅は、初年度（{first_year}年）を外して測る。")
    print(f"  4四半期の履歴や6ヶ月の高値が要る列は母集団の開始直後には作れず、"
          f"初年度は必ず大きく欠けるため。")
    print(f"  40pt超動く列: 全期間で {n_first}本 → {first_year}年を外すと {n_after}本"
          f"（{first_year}年は最初のテスト窓より前なので評価には使われない）")
    warn = live[(live["missing_pct"] > 40) | (live["outlier_pct"] > 8)
                | (live["year_spread"] > 40)].copy()
    warn["理由"] = [
        "・".join(filter(None, [
            f"欠損{r['missing_pct']:.0f}%" if r["missing_pct"] > 40 else "",
            f"外れ値{r['outlier_pct']:.0f}%" if r["outlier_pct"] > 8 else "",
            f"年で{r['year_spread']:.0f}pt動く" if r["year_spread"] > 40 else "",
        ])) for _, r in warn.iterrows()]
    warn = warn.sort_values("max_abs_z", ascending=False, na_position="last")
    if len(warn):
        print_table(warn, [
            ("feature", "列", 26, "s"), ("本番", "本番", 5, "s"),
            ("group", "群", 15, "s"), ("max_abs_z", "最大|z|", 8, ".2f"),
            ("理由", "理由", 34, "s"),
        ], args.top)
        print(f"  該当 {len(warn)}列")
    else:
        print("  該当なし")

    dup = eda.get("redundant", [])
    if dup:
        print(f"\n=== 5. 冗長な組（|相関| >= 0.95）===")
        zmap = dict(zip(res["feature"], res["max_abs_z"]))
        print(f"  {'A':<26}{'B':<26}{'相関':>8}{'A の|z|':>9}{'B の|z|':>9}")
        def z_(x):
            v = zmap.get(x, np.nan)
            return f"{v:.2f}" if np.isfinite(v) else "—"
        for d in dup[:20]:
            print(f"  {d['a']:<26}{d['b']:<26}{d['corr']:>+8.4f}"
                  f"{z_(d['a']):>9}{z_(d['b']):>9}")
        print(f"  該当 {len(dup)}組")

    if len(dead):
        print(f"\n=== 6. 測れなかった列 {len(dead)}本 ===")
        for grp, part in dead.groupby("group", sort=False):
            mark = "（取り込み待ち）" if part["取込待ち"].eq("●").all() else ""
            print(f"  {grp:<16}{len(part):>3}列 {mark} "
                  f"{', '.join(part['feature'].head(4))}"
                  f"{' …' if len(part) > 4 else ''}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\n記録: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
