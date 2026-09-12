#!/usr/bin/env python3
"""
画面に並ぶモデルの一覧を表にする（ワークフローの実行要約用）。

    python3 research/model_lineup.py                  # 表を標準出力へ
    python3 research/model_lineup.py --markdown       # Markdown の表で

なぜ要約に出すか
---------------
週次の学習は5モデルを別々に回し、1つ失敗しても他を止めない
（train_multi.py の per-model try/except）。その代わり、静かに欠けたまま
運用が続くおそれがある。どのモデルがいつ学習されて、out-of-fold が
何件あるのかを毎週の実行要約に出しておけば、欠けに気づける。

learned/missing は research/model/models/<algo>/meta.json の有無で見る。
モデル本体（model.joblib）は Release にしか置かないので、コミットされた
リポジトリの中身では判定できない。学習直後のワークフロー内では両方ある。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M  # noqa: E402


def _w(s: str) -> int:
    """端末での表示幅。日本語のモデル名が入るので len() では桁が揃わない。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _w(s))


def row(algo: str, root: str) -> dict:
    d = M.sub_dir(algo, root)
    mp = os.path.join(d, "model.joblib")
    tp = os.path.join(d, "meta.json")
    r = {"algo": algo, "name": M.JA.get(algo, algo), "trained": "-",
         "n_oof": "-", "n_train": "-", "size": "-", "pr_auc": "-",
         "state": "未学習"}
    if not os.path.exists(tp):
        return r
    with open(tp, encoding="utf-8") as fh:
        meta = json.load(fh)
    r["state"] = "学習済み"
    r["trained"] = str(meta.get("trainedAt", ""))[:10]
    r["n_oof"] = f"{meta.get('nOof', 0):,}"
    r["n_train"] = f"{meta.get('nTrain', 0):,}"
    cv = meta.get("tuning") or {}
    if cv.get("mean_pr_auc") is not None:
        r["pr_auc"] = f"{cv['mean_pr_auc']:.4f}"
    if os.path.exists(mp):
        r["size"] = f"{os.path.getsize(mp) / 1e6:.1f}MB"
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="モデル一覧を表にする")
    ap.add_argument("--model-dir", default=M.MODEL_DIR)
    ap.add_argument("--markdown", action="store_true",
                    help="Markdown の表で出す（実行要約に貼る用）")
    args = ap.parse_args(argv)

    rows = [row(a, args.model_dir) for a in M.ALGOS]
    head = ("モデル", "状態", "学習日", "訓練件数", "OOF件数", "探索PR-AUC", "サイズ")
    keys = ("name", "state", "trained", "n_train", "n_oof", "pr_auc", "size")

    if args.markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join(["---"] * len(head)) + "|")
        for r in rows:
            print("| " + " | ".join(str(r[k]) for k in keys) + " |")
    else:
        w = [max(_w(head[i]), *(_w(str(r[k])) for r in rows))
             for i, k in enumerate(keys)]
        print("  ".join(_pad(h, w[i]) for i, h in enumerate(head)))
        for r in rows:
            print("  ".join(_pad(str(r[k]), w[i]) for i, k in enumerate(keys)))

    ok = [r for r in rows if r["state"] == "学習済み"]
    print()
    print(f"学習済み {len(ok)}/{len(rows)} モデル")
    miss = [r["name"] for r in rows if r["state"] != "学習済み"]
    if miss:
        print(f"未学習: {', '.join(miss)}")
    # 1つも無いときだけ失敗扱い。一部欠けは要約に出すだけで止めない
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
