#!/usr/bin/env python3
"""
実運用の追跡（research/live_track.py）を run-experiment.yml から回す入口。

  Actions: Run Experiment（exp=live_vs_oof.py、ref=claude/jquants-browser-app-bwk19a）
    Release の日足と本番モデルの OOF（oof.parquet / xgb_oof.parquet /
    cat_oof.parquet）を research/_data に落とし、ラベル付きのデータセットを
    作ってから回る。checkout は浅いので、予測の履歴はスクリプトが取り寄せる。
  結果はログ（Summary）に出る。本番の設定には何も書かない。
  手順と読み方は docs/LIVE_TRACKING.md。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live_track  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(live_track.main())
