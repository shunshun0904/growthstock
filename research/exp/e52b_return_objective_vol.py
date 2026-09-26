#!/usr/bin/env python3
"""
実験52 を、実験51 の6列（ボラの分解）を足した212列（all_plus_prog_listing_vol）で回す。

運用者の指示（2026-09-26）「212列でも並行して回してください」。中身は e52_return_objective.py
そのもの（--preset を固定しただけ）。別名にしてあるのは、Actions の同時実行の組が実験名ごと
（run-experiment-<実験名>）なので、206列の実行と並行して回すため。
結果は research/_data/oof/e52_<列の指紋>_*。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e52_return_objective as E  # noqa: E402

PRESET = "all_plus_prog_listing_vol"

if __name__ == "__main__":
    raise SystemExit(E.main(["--preset", PRESET] + [a for a in sys.argv[1:] if not a.startswith("--preset")]))
