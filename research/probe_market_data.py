#!/usr/bin/env python3
"""
市場環境（マクロ・地合い）に使えるデータが、この契約で実際に取れるかを実測する。

なぜ必要か:
  ブレイクアウトの成否は地合いに強く依存する、という仮説がある。
  ところが現在の市場環境の特徴量は TOPIX の 20日／120日リターンだけで
  （research/features.py の "market" グループ）、
  日経平均・金・金利などは「取れるかどうか」すら確かめていない。

  docs/DATA_FIELDS.md の実測では /indices/topix と /indices/prices は 403 で、
  /indices/bars/daily/topix だけが通っている。指数が他に何を返すかは不明。

ここで確かめること:
  1. /indices 配下に一覧（列挙）できるものがあるか
  2. /indices/bars/daily/{name} で topix 以外に通る名前があるか
  3. 通らない場合、東証上場の ETF で代用できるか。
     ETF は /equities/bars/daily に含まれるので取得先を増やさずに済む。
     どの銘柄が実在するかは /equities/master の名称から探す（推測しない）
  4. 見つかった候補が 2016-10（契約の最古日）まで遡れるか、
     毎営業日出来があるか、売買代金は十分か

このファイルの候補リストは「存在するはずのもの」ではなく
「叩いて確かめる対象」である。通らなければ NG として記録するだけで、
通ったものだけを docs/MARKET_DATA.md に残す。

結果は research/probe_market_data.json と docs/MARKET_DATA.md に出す。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jquants_data_fetcher import (  # noqa: E402
    JQuantsClient, JQuantsError, resolve_api_key,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_JSON = os.path.join(HERE, "probe_market_data.json")
OUT_MD = os.path.join(ROOT, "docs", "MARKET_DATA.md")

# 契約がカバーする最古の日付（research/probe_boundary.py で実測）。
# 特徴量に使うには、この日まで遡って値が無いと学習期間が削られる。
EARLIEST_DATE = "2016-10-01"

# --------------------------------------------------------------------------- #
# 1. 指数エンドポイント
# --------------------------------------------------------------------------- #

# 列挙できるか。一覧が返るなら候補を推測する必要が無くなる
ENUM_ENDPOINTS = [
    ("/indices", {}),
    ("/indices/master", {}),
    ("/indices/bars/daily", {"from": "2024-05-01", "to": "2024-05-15"}),
    ("/indices/bars/daily", {"code": "0000", "from": "2024-05-01", "to": "2024-05-15"}),
    ("/indices/bars", {"from": "2024-05-01", "to": "2024-05-15"}),
]

# /indices/bars/daily/{name} の {name} 候補。
# topix は既知の OK（プローブ自体が動いていることの対照になる）。
# それ以外は綴りを含めて未確認で、通るかどうかをここで測る。
INDEX_NAMES = [
    "topix",
    "topix-core30", "topix-large70", "topix100", "topix500", "topix1000",
    "topix-small", "topix-mid400",
    "nikkei225", "nikkei-225", "nikkei", "nk225", "n225",
    "prime", "standard", "growth",
    "tse-prime", "tse-standard", "tse-growth",
    "growth250", "mothers",
    "reit", "tsereit",
    "jpx400", "jpx-nikkei400",
]

# --------------------------------------------------------------------------- #
# 2. 上場商品からの代用候補
# --------------------------------------------------------------------------- #

# 名称に含まれうる断片。ここに書いた語が実在を意味するのではなく、
# /equities/master が返した名称を突き合わせる検索語にすぎない。
THEMES: Dict[str, List[str]] = {
    "金": ["ゴールド", "純金", "金価格", "金先物", "GOLD", "Gold"],
    "日経225": ["日経225", "日経平均", "225"],
    "TOPIX": ["TOPIX", "トピックス", "東証株価指数"],
    "グロース/新興": ["グロース", "マザーズ", "Growth", "250"],
    "米国株": ["S&P", "ナスダック", "NASDAQ", "米国株", "ダウ"],
    "原油/商品": ["原油", "WTI", "商品", "コモディティ"],
    "債券/金利": ["国債", "債券", "金利"],
    "為替": ["為替", "米ドル", "ドル建", "通貨"],
    "REIT": ["REIT", "リート", "不動産投信"],
    "ボラティリティ": ["VIX", "ボラティリティ", "変動率"],
}

# 内国株の市場区分。ここに入らないものが ETF/ETN/REIT などにあたる
DOMESTIC_MARKETS = ("プライム", "スタンダード", "グロース", "内国株")


def probe_endpoints(client: JQuantsClient) -> List[dict]:
    """列挙系エンドポイントを叩く。"""
    out = []
    for path, params in ENUM_ENDPOINTS:
        rec = {"path": path, "params": params}
        try:
            data = client.get(path, params)
            batch = data.get("data")
            rec["ok"] = True
            rec["rows"] = len(batch) if isinstance(batch, list) else 0
            rec["keys"] = sorted(batch[0].keys()) if batch else []
            rec["sample"] = batch[0] if batch else None
        except JQuantsError as exc:
            rec["ok"] = False
            rec["error"] = str(exc)[:200]
        out.append(rec)
        label = f"{path} {params}"
        print(f"  {label:<62} {'OK' if rec.get('ok') else 'NG'} "
              f"{rec.get('rows', '')}")
    return out


def probe_index_names(client: JQuantsClient) -> List[dict]:
    """/indices/bars/daily/{name} を総当たりする。"""
    out = []
    for name in INDEX_NAMES:
        rec = {"name": name}
        try:
            rows = client.get_paginated(
                f"/indices/bars/daily/{name}",
                {"from": "2024-05-01", "to": "2024-05-15"},
            )
            rec["ok"] = True
            rec["rows"] = len(rows)
            rec["keys"] = sorted(rows[0].keys()) if rows else []
            rec["sample"] = rows[0] if rows else None
        except JQuantsError as exc:
            rec["ok"] = False
            rec["error"] = str(exc)[:160]
        out.append(rec)
        print(f"  /indices/bars/daily/{name:<18} "
              f"{'OK' if rec.get('ok') else 'NG'} {rec.get('rows', '')}")
    return out


def _matched_themes(name: str) -> List[str]:
    hit = []
    for theme, words in THEMES.items():
        if any(w in name for w in words):
            hit.append(theme)
    return hit


def probe_instruments(client: JQuantsClient) -> dict:
    """
    /equities/master の全銘柄から、指数・商品に連動しそうな上場商品を探す。

    名称は API が返したものをそのまま使う。こちらで銘柄コードを
    書き下ろすと、実在しないコードを混ぜてしまう。
    """
    rows = client.get_paginated("/equities/master", {})
    print(f"  /equities/master: {len(rows):,}件")
    if not rows:
        return {"total": 0, "by_market": {}, "candidates": []}

    by_market: Dict[str, int] = defaultdict(int)
    candidates = []
    for r in rows:
        mkt = str(r.get("MktNm") or "?")
        by_market[mkt] += 1
        name = str(r.get("CoName") or "")
        is_domestic = any(k in mkt for k in DOMESTIC_MARKETS)
        themes = _matched_themes(name)
        # 内国株（普通株）は母集団そのものなので除く。
        # 探しているのは指数・商品に連動する上場商品のほう
        if is_domestic or not themes:
            continue
        candidates.append({
            "code": str(r.get("Code") or ""),
            "name": name,
            "market": mkt,
            "sector33": str(r.get("S33Nm") or ""),
            "themes": themes,
        })

    print(f"  市場区分: " + " / ".join(
        f"{k}:{v}" for k, v in sorted(by_market.items(), key=lambda x: -x[1])))
    print(f"  テーマに引っかかった上場商品: {len(candidates)}件")
    return {
        "total": len(rows),
        "by_market": dict(by_market),
        "candidates": candidates,
    }


def probe_history(client: JQuantsClient, code: str, end: str) -> dict:
    """
    その銘柄が契約の最古日まで遡れるか、毎営業日値が付くか、
    売買代金がどれくらいかを測る。

    薄い ETF は値が飛ぶので、価格系列として使えるかは出来高で決まる。
    """
    try:
        rows = client.get_paginated(
            "/equities/bars/daily", {"code": code, "from": EARLIEST_DATE, "to": end})
    except JQuantsError as exc:
        return {"ok": False, "error": str(exc)[:160]}
    if not rows:
        return {"ok": True, "rows": 0}

    dates = sorted(str(r.get("Date")) for r in rows)
    vols, vals = [], []
    for r in rows:
        for key, sink in (("Vo", vols), ("Va", vals)):
            v = r.get(key)
            try:
                sink.append(float(v))
            except (TypeError, ValueError):
                pass
    zero_vol = sum(1 for v in vols if v == 0)
    return {
        "ok": True,
        "rows": len(rows),
        "first": dates[0],
        "last": dates[-1],
        "zeroVolumeDays": zero_vol,
        "zeroVolumeShare": round(zero_vol / len(vols) * 100, 2) if vols else None,
        "medianTurnoverYen": int(statistics.median(vals)) if vals else None,
    }


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #

def write_md(result: dict) -> None:
    L: List[str] = []
    L.append("# 市場環境データの取得可否（実測）")
    L.append("")
    L.append("`research/probe_market_data.py` の出力。**API を実際に叩いた結果のみ**を記載する。")
    L.append("")
    L.append(f"- 実測日時: {result['probedAt']}")
    L.append(f"- 遡及の起点: {EARLIEST_DATE}（契約の最古日）")
    L.append("")

    L.append("## 指数エンドポイント（列挙）")
    L.append("")
    L.append("| エンドポイント | パラメータ | 結果 | 件数 | 備考 |")
    L.append("| --- | --- | :---: | ---: | --- |")
    for r in result["enumEndpoints"]:
        note = ", ".join(r.get("keys", [])) if r.get("ok") else r.get("error", "")
        L.append(f"| `{r['path']}` | `{r['params']}` | "
                 f"{'OK' if r.get('ok') else 'NG'} | {r.get('rows', '—')} | {note[:110]} |")
    L.append("")

    L.append("## `/indices/bars/daily/{name}`")
    L.append("")
    ok = [r for r in result["indexNames"] if r.get("ok")]
    L.append(f"候補 {len(result['indexNames'])} 件のうち **{len(ok)} 件が通った**。")
    L.append("")
    L.append("| name | 結果 | 件数 | 列 |")
    L.append("| --- | :---: | ---: | --- |")
    for r in result["indexNames"]:
        cols = ", ".join(r.get("keys", [])) if r.get("ok") else "—"
        L.append(f"| `{r['name']}` | {'OK' if r.get('ok') else 'NG'} | "
                 f"{r.get('rows', '—')} | {cols[:80]} |")
    L.append("")

    inst = result["instruments"]
    L.append("## 上場商品（ETF/ETN など）からの代用候補")
    L.append("")
    L.append(f"`/equities/master` の {inst['total']:,} 件を名称で検索した結果。")
    L.append("**銘柄コードはこちらで書き下ろしていない。API が返した名称に一致したものだけを載せる。**")
    L.append("")
    L.append("市場区分の内訳: " + " / ".join(
        f"{k} {v:,}" for k, v in sorted(inst["by_market"].items(), key=lambda x: -x[1])))
    L.append("")
    if inst["candidates"]:
        L.append("| コード | 名称 | 市場区分 | テーマ |")
        L.append("| --- | --- | --- | --- |")
        for c in inst["candidates"]:
            L.append(f"| `{c['code']}` | {c['name']} | {c['market']} | "
                     f"{'・'.join(c['themes'])} |")
    else:
        L.append("該当なし。")
    L.append("")

    L.append("## 候補の遡及範囲と流動性")
    L.append("")
    L.append("価格系列として使えるかは「最古日まで遡れるか」と「毎営業日値が付くか」で決まる。")
    L.append("出来高0の日が多い銘柄は前日終値が据え置かれるため、リターンが人為的に0になる。")
    L.append("")
    if result["history"]:
        L.append("| コード | 名称 | 行数 | 最古 | 最新 | 出来高0の日 | 売買代金の中央値 |")
        L.append("| --- | --- | ---: | --- | --- | ---: | ---: |")
        for h in result["history"]:
            s = h["stats"]
            if not s.get("ok"):
                L.append(f"| `{h['code']}` | {h['name']} | — | — | — | — | {s.get('error','')[:60]} |")
                continue
            turnover = s.get("medianTurnoverYen")
            turnover_txt = f"{turnover:,}円" if turnover is not None else "—"
            L.append(
                f"| `{h['code']}` | {h['name']} | {s.get('rows', 0):,} | "
                f"{s.get('first', '—')} | {s.get('last', '—')} | "
                f"{s.get('zeroVolumeShare', '—')}% | {turnover_txt} |")
    else:
        L.append("該当なし。")
    L.append("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[write] {OUT_MD}")


def main() -> int:
    client = JQuantsClient(resolve_api_key(), pause=0.2)
    end = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    result = {"probedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
              "earliest": EARLIEST_DATE, "historyEnd": end}

    print("[1] 指数エンドポイントの列挙")
    result["enumEndpoints"] = probe_endpoints(client)

    print("\n[2] /indices/bars/daily/{name} の総当たり")
    result["indexNames"] = probe_index_names(client)

    print("\n[3] 上場商品から代用候補を探す")
    result["instruments"] = probe_instruments(client)

    print("\n[4] 候補の遡及範囲と流動性")
    # 全部測ると件数が読めないので、テーマごとに数銘柄に絞る。
    # ここでの並びは名称順（恣意的な選別を避ける）
    per_theme: Dict[str, int] = defaultdict(int)
    picked = []
    for c in sorted(result["instruments"]["candidates"], key=lambda x: x["code"]):
        if all(per_theme[t] >= 4 for t in c["themes"]):
            continue
        for t in c["themes"]:
            per_theme[t] += 1
        picked.append(c)
    print(f"  測る候補: {len(picked)}件")
    hist = []
    for c in picked:
        s = probe_history(client, c["code"], end)
        hist.append({"code": c["code"], "name": c["name"],
                     "themes": c["themes"], "stats": s})
        print(f"  {c['code']} {c['name'][:28]:<28} "
              f"{s.get('rows', 0)}行 {s.get('first', '')}〜{s.get('last', '')} "
              f"出来高0 {s.get('zeroVolumeShare', '—')}%")
    result["history"] = hist

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n[write] {OUT_JSON}")
    write_md(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
