#!/usr/bin/env python3
"""
予測タブの過去の日を、その日に出した値に戻す（一度だけの移行。運用者の指示 2026-10-12「過去の日は最初に出した値のまま
固定してください」）。

それまでの public/data/predictions.json は、毎日の予測で直近5営業日ぶんを、その時点の最新のモデルとその日に作り直した
特徴量で採点し直して上書きしていた。固定（predict_daily.freeze_past_days）を始める前に、今のファイルの日を git の履歴
から戻す。

各日の値 = 日次予測の実行が出した版（コミットの件名が「chore(data): ブレイク予測を更新」）のうち、その日が最新の日
（asOf）だった最後の版（その日のうちに意図して予測し直した版を含む）。コードの変更と一緒にファイルを手で直した版
（例: 2026-10-09 にモデルの表示を絞ったコミット）は使わない。履歴にその日が最新だった版が無い日は、今の値のまま。
dayInfo[日] = その版のモデルの学習日（model.trainedAt）と採点した時刻（generatedAt）。

  python3 research/restore_first_predictions.py --dry-run   # 日ごとの件数と、変わる件数だけ
  python3 research/restore_first_predictions.py             # 書き換える

公開されている predictions.json の過去の版を戻すだけで、新しい値は作らない。出力は件数だけ（銘柄は出さない）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REL_PATH = "public/data/predictions.json"
#: 日次予測の実行（.github/workflows/predict.yml の Commit predictions）のコミットの件名の頭
PREDICT_SUBJECT = "chore(data): ブレイク予測を更新"


def versions_from_git(rel_path: str = REL_PATH, root: str = ROOT,
                      subject: str = PREDICT_SUBJECT) -> List[Tuple[str, Dict]]:
    """日次予測の実行が出した predictions.json の版を古い順に (コミット, 中身)。"""
    lines = subprocess.check_output(["git", "log", "--reverse", "--format=%h%x09%s", "--", rel_path],
                                    cwd=root).decode().splitlines()
    revs = [ln.split("\t", 1)[0] for ln in lines if "\t" in ln and ln.split("\t", 1)[1].startswith(subject)]
    out = []
    for rev in revs:
        try:
            out.append((rev, json.loads(subprocess.check_output(["git", "show", f"{rev}:{rel_path}"], cwd=root))))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
    return out


def restore(current: Dict, versions: List[Tuple[str, Dict]]) -> Tuple[Dict, List[Dict]]:
    """
    current の dates の各日を、その日が asOf だった最後の版の行に置き換える。戻り値: (新しい中身, 日ごとの報告)。
    versions は古い順。報告: {date, source, before, after, changed}（changed = スコアか順位が違う行と、片方だけの行の数）。
    """
    last_for: Dict[str, Tuple[str, Dict]] = {}
    for rev, v in versions:
        if v.get("asOf"):
            last_for[v["asOf"]] = (rev, v)
    cur_rows: Dict[str, List[Dict]] = {}
    for x in current.get("candidates") or []:
        cur_rows.setdefault(x["date"], []).append(x)
    rows: List[Dict] = []
    info: Dict[str, Dict] = {}
    report = []
    for d in current.get("dates") or []:
        before = cur_rows.get(d, [])
        hit = last_for.get(d)
        if hit is None:
            rows.extend(before)
            info[d] = dict((current.get("dayInfo") or {}).get(d)
                           or {"trainedAt": (current.get("model") or {}).get("trainedAt"),
                               "generatedAt": current.get("generatedAt")})
            report.append({"date": d, "source": "今の値のまま（最新だった版が無い）", "before": len(before),
                           "after": len(before), "changed": 0})
            continue
        rev, v = hit
        after = [x for x in v.get("candidates") or [] if x.get("date") == d]
        rows.extend(after)
        info[d] = dict((v.get("dayInfo") or {}).get(d)
                       or {"trainedAt": (v.get("model") or {}).get("trainedAt"), "generatedAt": v.get("generatedAt")})
        b = {x["jqCode"]: x for x in before}
        a = {x["jqCode"]: x for x in after}
        changed = len(set(a) ^ set(b)) + sum(
            1 for k in set(a) & set(b)
            if a[k].get("score") != b[k].get("score") or a[k].get("rankInDay") != b[k].get("rankInDay"))
        report.append({"date": d, "source": rev, "before": len(before), "after": len(after), "changed": changed})
    rows.sort(key=lambda x: (x["date"], x.get("score") or 0), reverse=True)
    out = dict(current)
    out["candidates"] = rows
    out["dayInfo"] = info
    return out, report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="予測タブの過去の日を、その日に出した値に戻す（一度だけの移行）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    path = os.path.join(ROOT, REL_PATH)
    with open(path, encoding="utf-8") as fh:
        current = json.load(fh)
    new, report = restore(current, versions_from_git())
    for r in report:
        print(f"[restore] {r['date']}: {r['source']} / 候補 {r['before']} → {r['after']}件・変わる {r['changed']}件")
    if args.dry_run:
        return 0
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(new, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"[done] {REL_PATH} を書き換えた（{len(new['candidates'])}件・{len(new['dates'])}日）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
