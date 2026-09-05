#!/usr/bin/env python3
"""
ラベル目視検証ビューアの HTML を生成する。

research/samples/label_samples.json を読み、データを埋め込んだ単一 HTML を出力する。
Artifact として公開してブラウザで確認するためのもの。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "research", "samples", "label_samples.json"))
    ap.add_argument("--template", default=os.path.join(HERE, "template.html"))
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "viewer", "label_review.html"))
    args = ap.parse_args(argv)

    if not os.path.exists(args.data):
        raise SystemExit(f"{args.data} がありません。先に export_label_samples.py を実行してください")

    with open(args.data, encoding="utf-8") as fh:
        data = json.load(fh)
    with open(args.template, encoding="utf-8") as fh:
        tpl = fh.read()

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # </script> がデータ内に現れても壊れないようにエスケープ
    payload = payload.replace("</", "<\\/")
    html = tpl.replace("__DATA__", payload)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[done] {args.out} ({os.path.getsize(args.out)/1e6:.2f}MB / {len(data['cases'])}ケース)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
