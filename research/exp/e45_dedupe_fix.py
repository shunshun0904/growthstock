#!/usr/bin/env python3
"""
実験45: 取り込みの重複除去の修正（2026-09-24）で、本番の205列と OOF がどう変わるか。

2026-09-24 まで、年別ファイルへのマージが「同じ日・同じ銘柄」を1行に潰していて、
決算の3%・決算発表予定の6%・大株主の8%・大量保有報告書の15%・業種別の空売り比率の
97% を捨てていた（research/probe_dedupe_keys.py の8日ぶんの実測）。キーを種別ごとに
直し（data_store.ROW_KEYS）、5種別を全期間取り直した。

腕（コードは同じ。生データだけが違う）
  A  取り直す前の生データ（Snapshot Raw Data で写したもの。--snapshot のタグ）
  B  取り直した後の生データ（いまの data-raw）
共通: 本番の205列（all_plus）、本番のパラメータ、ブースティング3モデル、種3つの平均、
      窓の切り方3通り（research/exp/ab_oof.py）

見るもの
  1. 行（サンプル）の出入りと、値が変わった列とその割合（どの特徴量が動いたか）
  2. 分離力の差（窓ごと）と、運用の規則での取引（ab_oof.compare）

  Actions: Run Experiment（exp=e45_dedupe_fix.py, snapshot=<タグ>, args=--snapshot <タグ>）
  結果は research/_data/oof/e45_*。本番の設定には書かない。
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402

LABELS = {"A": "A 取り直す前", "B": "B 取り直した後"}


def build_before(snapshot_dir: str, out_path: str) -> pd.DataFrame:
    """
    写しの種別だけ差し替えたデータの置き場を作り、同じコードでデータセットを作る。

    写しに入っている種別（ファイル名の接頭辞）は、いまのファイルを**全部外して**から
    写しを置く（取り直しで年のファイルが増えていても混ざらない）。
    """
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)
    snap = sorted(glob.glob(os.path.join(snapshot_dir, "*.parquet")))
    if not snap:
        raise SystemExit(f"写しが無い: {snapshot_dir}")
    kinds = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in snap})
    AB.log(f"写しの種別: {kinds}（{len(snap)}ファイル）")
    tmp = tempfile.mkdtemp(prefix="e45_before_")
    try:
        for p in glob.glob(os.path.join(lab.DATA_DIR, "*")):
            name = os.path.basename(p)
            if os.path.isdir(p) or any(name.startswith(f"{k}_") for k in kinds):
                continue
            os.symlink(p, os.path.join(tmp, name))
        for p in snap:
            os.symlink(p, os.path.join(tmp, os.path.basename(p)))
        B.build(tmp, out_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return pd.read_parquet(out_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験45: 取り込みの重複除去の修正の前後")
    ap.add_argument("--snapshot", required=True, help="写しのタグ（research/_data/snapshot/<タグ>）")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default=",".join(L.BOOST))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    algos = [a for a in args.algos.split(",") if a]

    cols = F.columns(F.DEFAULT_PRESET)
    fb = lab.frame()
    fb["Date"] = pd.to_datetime(fb["Date"])
    fb["Code"] = fb["Code"].astype(str)
    snap_dir = os.path.join(lab.DATA_DIR, "snapshot", args.snapshot)
    la = build_before(snap_dir, os.path.join(lab.DATA_DIR, f"dataset_{args.snapshot}.parquet"))
    la["Date"] = pd.to_datetime(la["Date"])
    la["Code"] = la["Code"].astype(str)

    key = ["Code", "Date"]
    fb = fb.dropna(subset=["label"])
    la = la.dropna(subset=["label"])
    both = la[key].merge(fb[key], on=key, how="inner")
    print("=" * 78)
    print(f"実験45 取り込みの重複除去の修正の前後（{F.DEFAULT_PRESET} {len(cols)}列・"
          f"種{len(seeds)}つ・ずらし {shifts}か月）")
    print("=" * 78)
    print(f"\n■ 1. 行の出入り: 前 {len(la):,} / 後 {len(fb):,} / 共通 {len(both):,}"
          f"（前だけ {len(la) - len(both):,} / 後だけ {len(fb) - len(both):,}）")
    extra = [c for c in fb.columns if c not in la.columns]
    fa = la.merge(fb[key + extra], on=key, how="inner", validate="one_to_one")
    fb2 = fb.merge(both, on=key, how="inner").set_index(key).loc[
        fa.set_index(key).index].reset_index()
    changed = []
    for c in cols:
        a, b = fa[c].to_numpy(dtype=float), fb2[c].to_numpy(dtype=float)
        share = float((~np.isclose(a, b, equal_nan=True)).mean())
        if share > 0:
            changed.append((c, share))
    changed.sort(key=lambda x: -x[1])
    print(f"  205列のうち値が変わった列: {len(changed)}本")
    for c, v in changed[:40]:
        print(f"    {c:<28}{v*100:>7.2f}%  （{F.group_of(c) or '-'}）")
    if len(changed) > 40:
        print(f"    ほか {len(changed) - 40}本")
    os.makedirs(AB.OOF_DIR, exist_ok=True)
    pd.DataFrame(changed, columns=["column", "share"]).to_csv(
        os.path.join(AB.OOF_DIR, "e45_changed_columns.csv"), index=False)

    AB.compare("e45", {"A": (fa, cols), "B": (fb2, cols)}, "A", LABELS, shifts, seeds, algos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
