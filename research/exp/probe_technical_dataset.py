#!/usr/bin/env python3
"""
probe: テクニカル指標111列の本番導入・第1段階（2026-10-18、運用者の決定「入れる（2段階）」。docs/MODEL_ADOPTION_RULES.md §68）を
全期間のデータで確かめる。

Run Experiment が先に作った学習用のデータセット（research/_data/dataset.parquet。週次の再学習と同じ
`python3 research/build_dataset.py`。build_dataset.attach_technicals が111列を付ける）と、日次の予測と同じ作り
（--keep-unlabeled）について:

  1. 一致     111列が実験105 の作り方（組み直した日足から technicals.compute_at）と一致するか。学習用は
              e105_technical_ab.prepare の中で全行を突き合わせる（食い違いがあれば止まる）。予測用はここで全行を突き合わせる
  2. 予測用   --keep-unlabeled で作り直したデータセットにも111列が付くか（ラベル未確定の行・いちばん新しい日の行の欠測）
  3. 付き方   列ごとの欠測の数と分位（0.1% / 1% / 5% / 50% / 95% / 99% / 99.9%）。線形の前処理の型（linear_preprocess）と
              並べる。裾の重さの目安: 上は (99.9% − 50%) ÷ (95% − 50%)、下は (50% − 0.1%) ÷ (50% − 5%)
  4. 列の組   249列の組の指紋は変わっていないか・360列の組 = 249列の後ろに111列（実験105 の腕 C と同じ並び・指紋）か・
              360列がすべてデータセットにあるか・いまの既定

出すのは件数・割合・分位・日付だけ（Actions のログは公開。銘柄ごとの値は出さない）。食い違いがあれば終了コード 1。

  exp=probe_technical_dataset.py
  exp=probe_technical_dataset.py args="--skip-predict"    （2 を飛ばす。データセットの作り直しが1回減る）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import e105_technical_ab as E105  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import linear_preprocess as LP  # noqa: E402
import technicals as TX  # noqa: E402

NEW = list(TX.COLUMNS)
BASE_PRESET = E105.BASE_PRESET
TX_PRESET = "all_plus_prog_listing_vol_book_shape_dd_recent_tx"
#: 249列の組の指紋（2026-10-18 からの既定）と、実験105 の腕 C（249 + 111列）の指紋
BASE_SIG = "5ffd7c53c80f"
TX_SIG = "a0638dfd0880"
QS = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)


def presets_report() -> Dict[str, object]:
    """列の組の確かめ。ok は 249列の指紋・360列の並び・指紋がそろったとき True。"""
    base = F.columns(BASE_PRESET)
    tx = F.columns(TX_PRESET)
    out = {"base_n": len(base), "base_sig": F.signature(base), "tx_n": len(tx), "tx_sig": F.signature(tx),
           "tx_is_base_plus_new": tx[:len(base)] == base and tx[len(base):] == NEW,
           "default": F.DEFAULT_PRESET, "default_sig": F.signature(F.columns(F.DEFAULT_PRESET))}
    out["ok"] = bool(out["base_sig"] == BASE_SIG and out["tx_sig"] == TX_SIG and out["tx_is_base_plus_new"])
    return out


def distribution(df: pd.DataFrame, cols: Sequence[str] = tuple(NEW)) -> pd.DataFrame:
    """列ごとの欠測の数・分位・裾の重さの目安と、線形の前処理の型。"""
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        q = v.quantile(list(QS)) if v.notna().any() else pd.Series(np.nan, index=list(QS))
        up = q[0.95] - q[0.5]
        dn = q[0.5] - q[0.05]
        rows.append({"col": c, "group": TX.GROUP_OF[c], "kind": LP.kind_of(c), "missing": int(v.isna().sum()),
                     **{f"q{k}": float(q[k]) for k in QS},
                     "tail_up": float((q[0.999] - q[0.5]) / up) if up > 0 else np.nan,
                     "tail_dn": float((q[0.5] - q[0.001]) / dn) if dn > 0 else np.nan})
    return pd.DataFrame(rows)


def print_distribution(d: pd.DataFrame, n_rows: int) -> None:
    """distribution の表を系統ごとに出す（列名に「.」があるので itertuples ではなく辞書で読む）。"""
    for g in TX.GROUPS:
        print(f"  ◆ {TX.GROUP_JA[g]}")
        for r in d[d["group"] == g].to_dict("records"):
            qs = " / ".join(f"{r[f'q{k}']:.4g}" for k in QS)
            print(f"    {r['col']:<24}{r['kind']:<14}欠測 {r['missing']:>4}  {qs}  裾 上 {r['tail_up']:5.1f} 下 {r['tail_dn']:5.1f}")
    print(f"  欠測のいちばん多い列 {d['missing'].max():,}行 / {n_rows:,}行。型の内訳: "
          + " / ".join(f"{k} {v}" for k, v in d["kind"].value_counts().items()))


def predict_table(out_path: str) -> pd.DataFrame:
    """日次の予測と同じ作り（--keep-unlabeled）でデータセットを作り直して読む。[tech] と [result] の行だけ見せる。"""
    cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "build_dataset.py"), "--keep-unlabeled", "--out", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith(("[tech]", "[result]", "[filter] ラベル未確定")):
            print("    " + line)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-3000:])
        raise SystemExit(f"--keep-unlabeled のデータセットが作れない（終了コード {r.returncode}）")
    df = pd.read_parquet(out_path)
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    df["Code"] = df["Code"].astype(str)
    return df


def check_predict(pred: pd.DataFrame) -> Dict[str, int]:
    """予測用のデータセットの111列を、実験105 の作り方で数え直したものと突き合わせる（全行）。"""
    miss = [c for c in NEW if c not in pred.columns]
    if miss:
        raise SystemExit(f"予測用のデータセットに111列が無い: {miss[:5]}")
    t0 = time.time()
    panel = E105.build_panel(lab.DATA_DIR, pred["Code"].unique())
    want = TX.compute_at(panel, pred[["Code", "Date"]])
    del panel
    bad = E105.compare_columns(pred[NEW].reset_index(drop=True), want)
    latest = pred["Date"].max()
    last = pred[pred["Date"] == latest]
    out = {"rows": int(len(pred)), "unlabeled": int(pred["label"].isna().sum()), "bad_cells": int(sum(bad.values())),
           "bad_columns": int(sum(v > 0 for v in bad.values())), "latest_rows": int(len(last)),
           "latest_all_nan": int(last[NEW].isna().all(axis=1).sum()),
           "latest_any_nan": int(last[NEW].isna().any(axis=1).sum()), "seconds": int(time.time() - t0)}
    out["latest"] = str(latest.date())
    return out


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description="probe: テクニカル指標111列の第1段階を全期間のデータで確かめる")
    ap.add_argument("--skip-predict", action="store_true", help="--keep-unlabeled の作り直し（2）を飛ばす")
    args = ap.parse_args(argv)
    ok = True
    print("=" * 100)
    print("probe: テクニカル指標111列の本番導入・第1段階（§68）を全期間のデータで確かめる")
    print("=" * 100)

    print("\n■ 4. 列の組")
    pr = presets_report()
    print(f"  249列の組 {BASE_PRESET}: {pr['base_n']}列 指紋 {pr['base_sig']}（期待 {BASE_SIG}）")
    print(f"  360列の組 {TX_PRESET}: {pr['tx_n']}列 指紋 {pr['tx_sig']}（期待 {TX_SIG}）・249列の後ろに111列: "
          f"{'はい' if pr['tx_is_base_plus_new'] else 'いいえ'}")
    print(f"  いまの既定: {pr['default']}（指紋 {pr['default_sig']}）")
    ok &= pr["ok"]

    print("\n■ 1. 学習用のデータセット（実験105 の作り方で数え直して全行を突き合わせる）")
    base = F.columns(BASE_PRESET)
    full, stamp = E105.prepare(base, rebuild=True)
    dc = stamp["checks"].get("dataset_columns")
    if dc is None:
        print("  データセットに111列が無い（build_dataset.attach_technicals が効いていない）")
        ok = False
    else:
        print(f"  {dc['rows']:,}行 × 111列 → 食い違い {dc['bad_cells']:,}セル（{dc['bad_columns']}列）")
        ok &= dc["bad_cells"] == 0
    have = [c for c in F.columns(TX_PRESET) if c not in full.columns]
    print(f"  360列の組のうちデータセットに無い列: {len(have)}" + (f"（{have[:5]}）" if have else ""))
    ok &= not have

    print("\n■ 3. 付き方（学習用。分位は 0.1% / 1% / 5% / 50% / 95% / 99% / 99.9%。裾は上・下の目安）")
    print_distribution(distribution(full), len(full))

    if not args.skip_predict:
        print("\n■ 2. 予測用（--keep-unlabeled。日次の予測と同じ作り）")
        with tempfile.TemporaryDirectory() as tmp:
            pred = predict_table(os.path.join(tmp, "dataset_predict.parquet"))
            r = check_predict(pred)
        print(f"  {r['rows']:,}行（ラベル未確定 {r['unlabeled']:,}行）× 111列 → 食い違い {r['bad_cells']:,}セル"
              f"（{r['bad_columns']}列）/ 数え直し {r['seconds']}秒")
        print(f"  いちばん新しい日 {r['latest']}: {r['latest_rows']}行・111列がすべて欠測 {r['latest_all_nan']}行・"
              f"どれかが欠測 {r['latest_any_nan']}行")
        ok &= r["bad_cells"] == 0 and r["latest_all_nan"] == 0

    print("\n" + ("すべて一致" if ok else "食い違いあり（上を見る）"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
