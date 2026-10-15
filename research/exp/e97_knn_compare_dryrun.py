#!/usr/bin/env python3
"""
実験97（確かめだけ）: 候補の比較（research/knn_compare.py。画面の「候補の比較」タブに渡す近傍の集計）を、本番と同じデータ
（Release の out-of-fold と日足）で試しに作る。本番・画面は変えない。判定もしない。

日次予測（research/predict_daily.py）が毎日 knn_compare.build を呼ぶので、合成データのテスト（tests/test_knn_compare.py）だけで
なく、実データで作れること・かかる時間・JSON の大きさを、画面に載せる前に確かめる。候補は公開の public/data/predictions.json の
候補（直近5営業日）。過去の日の引き継ぎは使わず、全部の日を作る。

出すもの（集計だけ）: 日ごとの候補数・集計の付いた数・母集団の件数、尺度、文脈の件数・十分位の境・消したマス目、JSON の大きさ、
時間。候補ごとの値・過去の事例の銘柄・日付は出さない（日付は公開の予測日だけ）。

    exp=e97_knn_compare_dryrun.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import knn_compare as KC  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "research", "_data")
PRED = os.path.join(ROOT, "public", "data", "predictions.json")


def report(knn: dict, days: list, n_rows: int, size: int, secs: float) -> list[str]:
    """ログの行（集計だけ）。"""
    out = [f"候補 {n_rows}件・日 {len(days)}日（{days[0]} 〜 {days[-1]}）" if days else "候補なし",
           f"要約: {KC.summary_line(knn, days)}"]
    sc = knn.get("scale")
    if sc:
        out.append(f"尺度: リターンの形 {sc['R']}・出来高の比率 {sc['V']}・母集団（最初の日より前）{knn.get('poolN', 0):,}件")
    for d in days:
        b = (knn.get("byDay") or {}).get(d) or {}
        line = f"  {d}: {b.get('status', 'なし')}"
        if b.get("status") == "ok":
            line += f" 集計の付いた候補 {b.get('n', 0)}件・母集団 {b.get('poolN', 0):,}件"
        elif b.get("status") == "unavailable":
            line += f" 理由: {b.get('reason', '')}"
        elif b.get("status") == "pool_small":
            line += f" 母集団 {b.get('poolN', 0)}件"
        out.append(line)
    cx = knn.get("context")
    if cx:
        dec = cx["p25Deciles"]
        out.append(f"文脈: 過去の候補 {cx['n']}件・25%点の十分位の境 {dec[0]} / {dec[5]} / {dec[-1]}"
                   f"・消したマス目 {cx['density']['suppressedCells']}（{cx['density']['suppressedRows']}件）")
    else:
        out.append("文脈: 無し")
    out.append(f"JSON {size:,}文字 / {secs:.0f}秒")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験97: 候補の比較の集計を実データで試しに作る（確かめだけ）")
    ap.add_argument("--pred", default=PRED, help="候補を読む predictions.json")
    ap.add_argument("--data-dir", default=DATA_DIR, help="日足（bars_*.parquet）の置き場")
    ap.add_argument("--oof-dir", default=None, help="out-of-fold の置き場（既定は --data-dir。Release の <名前>_oof.parquet の形）")
    ap.add_argument("--model-dir", default=None, help="本番の形（research/model/models/<名前>/oof.parquet）で探す置き場（省略可）")
    args = ap.parse_args(argv)
    t0 = time.time()
    with open(args.pred, encoding="utf-8") as fh:
        pred = json.load(fh)
    days = list(pred.get("dates") or [])
    rows = list(pred.get("candidates") or [])
    model_dir = args.model_dir or os.path.join(args.data_dir, "_no_model_dir")
    knn = KC.build(rows, days, args.data_dir, model_dir, prev=None, frozen=[], oof_dir=args.oof_dir or args.data_dir)
    js = json.dumps(knn, ensure_ascii=False, separators=(",", ":"))
    print("\n■ 候補の比較の集計（実データ。集計だけ）")
    for line in report(knn, days, len(rows), len(js), time.time() - t0):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
