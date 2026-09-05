#!/usr/bin/env python3
"""
/fins/summary が実際に返す全項目を実測する（データ探索）。

なぜ必要か:
  決算特徴量が効かなかった原因を追ったところ、ROE は通期開示にしか
  入っておらず、四半期は 0.0% だった（docs/MODEL_FUNDAMENTAL_COVERAGE.md）。
  そもそも「何が取れるのか」を確かめずに特徴量を設計していたのが誤り。

  さらに research/jq_bulk.py の FIN_COLS はホワイトリストであり、
  そこに書いていない項目は取得時点で捨てている。
  API が BPS や各種比率を返していても手元に残らない。

ここで確かめること:
  1. /fins/summary が返す項目名の全リスト（FIN_COLS に無いものを含む）
  2. 各項目の充足率。開示種別(CurPerType)ごとにも出す
  3. 現在 FIN_COLS で捨てている項目
  4. PER / PBR / EPS / ROE / ROA を直接返す項目があるか。
     無い場合、手元の項目から計算できるか
  5. 候補エンドポイントがこのプランで叩けるか（実測。推測しない）

結果は research/probe_fins_fields.json と docs/DATA_FIELDS.md に出す。
"""
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jquants_data_fetcher import (  # noqa: E402
    JQuantsClient, JQuantsError, resolve_api_key,
)

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "probe_fins_fields.json")
OUT_MD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "docs", "DATA_FIELDS.md")

# 日本の決算発表が集中する時期を選ぶ。閑散日だと項目が出そろわない。
SAMPLE_DATES = [
    "2019-05-15", "2019-08-08", "2019-11-13", "2020-02-13",
    "2021-05-14", "2021-08-06", "2021-11-12", "2022-02-14",
    "2023-05-15", "2023-08-09", "2023-11-13", "2024-02-14",
    "2024-05-15", "2024-08-08", "2024-11-13", "2025-02-14",
    "2025-05-15", "2025-08-08",
]

# 叩けるかどうかを実測する候補。存在を主張するものではない。
CANDIDATE_ENDPOINTS = [
    ("/fins/summary", {"date": "2024-05-15"}),
    ("/fins/details", {"date": "2024-05-15"}),
    ("/fins/statements", {"date": "2024-05-15"}),
    ("/fins/dividend", {"date": "2024-05-15"}),
    ("/fins/fs_details", {"date": "2024-05-15"}),
    ("/equities/master", {}),
    ("/equities/bars/daily", {"date": "2024-05-15"}),
    ("/markets/margin-interest", {"date": "2024-05-15"}),
    ("/markets/short-selling", {"date": "2024-05-15"}),
    ("/markets/breakdown", {"date": "2024-05-15"}),
    ("/markets/trades-spec", {}),
    ("/indices/topix", {"from": "2024-05-01", "to": "2024-05-15"}),
    ("/indices/prices", {"date": "2024-05-15"}),
]

# 探している概念 -> 項目名に現れそうな断片（実測した名前と突き合わせるだけ）
WANTED = {
    "EPS(1株利益)": {"eps"},
    "BPS(1株純資産)": {"bps", "bookvalue", "bvps"},
    "PER": {"per", "pe"},
    "PBR": {"pbr", "pb"},
    "ROE": {"roe"},
    "ROA": {"roa"},
    "総資産": {"ta", "totalassets", "assets"},
    "自己資本": {"eq", "sheq", "equity", "netassets"},
    "株数": {"shout", "shares", "issuedshares"},
}


def probe_endpoints(client):
    out = []
    for path, params in CANDIDATE_ENDPOINTS:
        rec = {"path": path, "params": params}
        try:
            data = client.get(path, params)
            batch = data.get("data")
            rec["ok"] = True
            rec["rows"] = len(batch) if isinstance(batch, list) else 0
            rec["keys"] = sorted(batch[0].keys()) if batch else []
        except JQuantsError as exc:
            rec["ok"] = False
            rec["error"] = str(exc)[:200]
        out.append(rec)
        print(f"  {path:<32} {'OK' if rec.get('ok') else 'NG'} "
              f"{rec.get('rows', '')} {rec.get('error', '')}")
    return out


def probe_fins(client):
    """全項目の充足率を、開示種別ごとに集計する。"""
    seen = defaultdict(int)         # field -> 非欠測件数
    total = 0
    by_period = defaultdict(lambda: defaultdict(int))   # period -> field -> 件数
    period_total = defaultdict(int)
    sample_row = None
    failed = []

    for d in SAMPLE_DATES:
        try:
            rows = client.get_paginated("/fins/summary", {"date": d})
        except JQuantsError as exc:
            failed.append(f"{d}: {str(exc)[:120]}")
            continue
        print(f"  {d}: {len(rows):,}件")
        for r in rows:
            total += 1
            per = str(r.get("CurPerType") or r.get("TypeOfCurrentPeriod") or "?")
            period_total[per] += 1
            if sample_row is None and r:
                sample_row = r
            for k, v in r.items():
                if v not in (None, ""):
                    seen[k] += 1
                    by_period[per][k] += 1

    fields = []
    for k in sorted(set(list(seen.keys()) + list((sample_row or {}).keys()))):
        rec = {
            "field": k,
            "present_pct": round(seen.get(k, 0) / total * 100, 1) if total else 0.0,
            "by_period": {p: round(by_period[p].get(k, 0) / n * 100, 1)
                          for p, n in sorted(period_total.items()) if n},
        }
        fields.append(rec)
    fields.sort(key=lambda r: -r["present_pct"])
    return {"n_rows": total, "fields": fields, "failed": failed,
            "period_counts": dict(period_total)}


# 項目名に付く接頭辞。剥がしてから本体を比較する。
# 部分一致だと CurPerType が PER に、CashEq が Eq に当たってしまう。
PREFIXES = ["NxFNC", "NxF", "FNC", "NC", "Nx", "F", "D"]
# 接尾辞（四半期の別、平均残高など）
SUFFIXES = ["2Q", "FY", "AR", "Ann"]


def _core(name: str) -> str:
    """接頭辞・接尾辞を剥がして本体を取り出す。"""
    s = name
    for p in PREFIXES:
        if s.startswith(p) and len(s) > len(p):
            s = s[len(p):]
            break
    for q in SUFFIXES:
        if s.endswith(q) and len(s) > len(q):
            s = s[: -len(q)]
            break
    return s.lower()


def match_wanted(field_names):
    """
    探している概念が、実測した項目名のどれに当たるかを突き合わせる。

    完全一致のみ（接頭辞・接尾辞を剥がした後）。
    部分一致にすると CurPerType が PER に当たるなどの誤検出が出る。
    """
    out = {}
    for concept, frags in WANTED.items():
        hits = sorted(f for f in field_names if _core(f) in frags)
        out[concept] = hits
    return out


def build_md(fins, endpoints, wanted, current_cols):
    got = [f["field"] for f in fins["fields"]]
    dropped = ([] if current_cols is None
               else [f for f in got if f not in set(current_cols)])
    lines = [
        "# J-Quants /fins/summary の項目一覧（実測）",
        "",
        "`research/probe_fins_fields.py` の出力。**API を実際に叩いた結果のみ**を記載する。",
        "",
        "モデリングの前にデータ探索をする、という方針に沿って、",
        "「何が取れるか」を推測せずに実測した。",
        "",
        f"- 対象: {len(SAMPLE_DATES)}日分の開示（決算集中期を選定）",
        f"- 取得行数: {fins['n_rows']:,}",
        f"- 開示種別の内訳: "
        + " / ".join(f"{k}: {v:,}" for k, v in sorted(fins["period_counts"].items())),
        "",
    ]
    if fins["failed"]:
        lines += ["**取得に失敗した日**:", ""] + [f"- {f}" for f in fins["failed"]] + [""]

    lines += [
        "## 探している指標が取れるか",
        "",
        "| 指標 | 該当する項目（実測） |",
        "| --- | --- |",
    ]
    for concept, hits in wanted.items():
        lines.append(f"| {concept} | "
                     + (", ".join(f"`{h}`" for h in hits) if hits else "**該当なし**")
                     + " |")

    lines += [
        "",
        "## 現在 FIN_COLS で捨てている項目",
        "",
        "`research/jq_bulk.py` の `FIN_COLS` はホワイトリストで、",
        "ここに無い項目は取得時点で捨てている。",
        "",
    ]
    if current_cols is None:
        lines.append("現在 `FIN_COLS = None`（絞らず全項目を保持）なので、"
                     "捨てている項目は無い。")
    elif dropped:
        lines += ["| 項目 | 充足率 |", "| --- | ---: |"]
        for f in fins["fields"]:
            if f["field"] in dropped:
                lines.append(f"| `{f['field']}` | {f['present_pct']}% |")
    else:
        lines.append("捨てている項目は無い。")

    lines += [
        "",
        "## 全項目の充足率",
        "",
        "| 項目 | 全体 | 開示種別ごと |",
        "| --- | ---: | --- |",
    ]
    for f in fins["fields"]:
        by = " / ".join(f"{k}: {v}%" for k, v in f["by_period"].items())
        lines.append(f"| `{f['field']}` | {f['present_pct']}% | {by} |")

    lines += [
        "",
        "## エンドポイントの疎通（実測）",
        "",
        "叩いて確かめた結果。存在しない・権限が無いものは NG になる。",
        "",
        "| エンドポイント | 結果 | 件数 | 備考 |",
        "| --- | :---: | ---: | --- |",
    ]
    for e in endpoints:
        if e.get("ok"):
            lines.append(f"| `{e['path']}` | OK | {e['rows']} | "
                         f"{len(e.get('keys', []))}項目 |")
        else:
            lines.append(f"| `{e['path']}` | NG | — | {e.get('error', '')[:120]} |")
    lines.append("")
    return "\n".join(lines)


def read_fin_cols():
    """
    research/jq_bulk.py の FIN_COLS を、import せずに読む。

    jq_bulk は pandas に依存するが、probe は標準ライブラリだけで動かしたい
    （取得系は stdlib のみという方針。probe.yml も pandas を入れていない）。
    ソースを構文解析して定数だけ取り出す。
    """
    import ast
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jq_bulk.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "FIN_COLS":
                    val = ast.literal_eval(node.value)
                    # None = 絞らずに全項目を保持している
                    return None if val is None else list(val)
    return []


def main() -> int:
    try:
        api_key = resolve_api_key()
    except SystemExit:
        raise
    client = JQuantsClient(api_key)

    print("[1] エンドポイントの疎通")
    endpoints = probe_endpoints(client)

    print("\n[2] /fins/summary の全項目")
    fins = probe_fins(client)
    if not fins["n_rows"]:
        print("決算データを1件も取得できなかった", file=sys.stderr)
        return 1

    # 取得結果はここで先に保存する。
    # 後段の解析で落ちると、API を叩き直すことになるため。
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"fins": fins, "endpoints": endpoints}, fh,
                  ensure_ascii=False, indent=2)
    print(f"  (取得結果を保存: {OUT_JSON})")

    names = [f["field"] for f in fins["fields"]]
    wanted = match_wanted(names)
    print("\n[3] 探している指標との突き合わせ")
    for concept, hits in wanted.items():
        print(f"  {concept:<16} {hits if hits else '該当なし'}")

    current = read_fin_cols()
    dropped = [] if current is None else [n for n in names if n not in set(current)]
    if current is None:
        print("\n[4] FIN_COLS は None（全項目を保持）。捨てている項目は無い")
    else:
        print(f"\n[4] FIN_COLS で捨てている項目: {len(dropped)}件")
        for f in fins["fields"]:
            if f["field"] in dropped and f["present_pct"] >= 10:
                print(f"  {f['field']:<28} {f['present_pct']:5.1f}%")

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"fins": fins, "endpoints": endpoints, "wanted": wanted,
                   "keeps_all_fields": current is None,
                   "dropped_by_fin_cols": dropped}, fh, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(build_md(fins, endpoints, wanted, current) + "\n")
    print(f"\n[done] {OUT_JSON}\n[done] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
