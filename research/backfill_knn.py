#!/usr/bin/env python3
"""
候補の比較（似た過去の事例）の集計を、公開済みの predictions.json に埋め戻す（運用者の指示 2026-10-15「せめて、今週分のモデルからで
良いので、ダッシュボードに出せないですか？」）。

日次予測（research/predict_daily.py）は新しい日だけに knn（research/knn_compare.py）を付け、過去の日は前の値を引き継ぐ。
タブを足した時点で画面に出ている日（直近5営業日）には knn が無いので、ここで一度だけ全部の日に付ける。
**候補の値（スコア・順位・百分位など）は一切変えない。** 足すのは knn だけ（過去の日の値を固定する約束はそのまま）。

  python3 research/backfill_knn.py                 # public/data/predictions.json を書き換える
  python3 research/backfill_knn.py --only-missing  # knn の付いていない日だけ
  python3 research/backfill_knn.py --dry-run       # 書かずに要約だけ

出すのは集計だけ（日ごとの件数・母集団・文脈の件数）。候補ごとの値・過去の事例の銘柄・日付は出さない。
.github/workflows/backfill-knn.yml が Release の生データを落としてこれを回し、公開ブランチに commit する。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knn_compare as KC  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
MODEL_DIR = os.path.join(HERE, "model")
PRED = os.path.join(os.path.dirname(HERE), "public", "data", "predictions.json")


def frozen_days(payload: Dict, only_missing: bool) -> List[str]:
    """引き継ぐ日。--only-missing のときは、前の knn で status が ok の日（それ以外は作り直す）。"""
    if not only_missing:
        return []
    by_day = ((payload.get("knn") or {}).get("byDay") or {})
    return [d for d in payload.get("dates") or [] if (by_day.get(d) or {}).get("status") == "ok"]


def backfill(payload: Dict, data_dir: str, model_dir: str, oof_dir: Optional[str] = None,
             only_missing: bool = False) -> Dict:
    """payload（predictions.json の中身）に knn を付けた新しい dict を返す。候補などほかの項目はそのまま。"""
    days = list(payload.get("dates") or [])
    rows = list(payload.get("candidates") or [])
    frozen = frozen_days(payload, only_missing)
    knn = KC.build(rows, days, data_dir, model_dir, prev=payload.get("knn"), frozen=frozen, oof_dir=oof_dir)
    out = dict(payload)
    out["knn"] = knn
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="候補の比較の集計を predictions.json に埋め戻す（候補の値は変えない）")
    ap.add_argument("--pred", default=PRED)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--oof-dir", default=None,
                    help="out-of-fold の置き場（Release の <名前>_oof.parquet の形。既定は --data-dir。無ければ --model-dir の本番の形を探す）")
    ap.add_argument("--only-missing", action="store_true", help="knn の付いていない日だけ作る（付いている日は前の値のまま）")
    ap.add_argument("--dry-run", action="store_true", help="書かずに要約だけ出す")
    args = ap.parse_args(argv)
    t0 = time.time()
    with open(args.pred, encoding="utf-8") as fh:
        payload = json.load(fh)
    days = list(payload.get("dates") or [])
    out = backfill(payload, args.data_dir, args.model_dir, args.oof_dir or args.data_dir, args.only_missing)
    knn = out["knn"]
    print(f"[backfill] {KC.summary_line(knn, days)}")
    by_day = knn.get("byDay") or {}
    unavailable = [d for d in days if (by_day.get(d) or {}).get("status") == "unavailable"]
    if unavailable and len(unavailable) == len([d for d in days if d not in frozen_days(payload, args.only_missing)]):
        print(f"[backfill] 作れなかった: {(by_day.get(unavailable[0]) or {}).get('reason', '')}")
        return 1
    if args.dry_run:
        print(f"[backfill] --dry-run なので書かない（{time.time() - t0:.0f}秒）")
        return 0
    with open(args.pred, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"[backfill] {args.pred} に knn を書いた（{os.path.getsize(args.pred) / 1e3:.0f}KB・{time.time() - t0:.0f}秒）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
