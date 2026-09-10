#!/usr/bin/env python3
"""
市場環境（マクロ・地合い）に使えるデータが、この契約で実際に取れるかを実測する。

なぜ必要か:
  ブレイクアウトの成否は地合いに強く依存する、という仮説がある。
  ところが現在の市場環境の特徴量は TOPIX の 20日／120日リターンだけで
  （research/features.py の "market" グループ）、
  日経平均・金などは「取れるかどうか」すら確かめていない。

ここで確かめること:
  1. 指数は何が取れるか。
     1回目の実測で `/indices/bars/daily?code=0000` が通ることが分かった
     （`/indices/bars/daily/topix` と同じ内容が Code 付きで返る）。
     つまり指数はコードで引ける。どのコードが存在するかを総当たりで測る。
  2. 東証上場の ETF/ETN で代用できるか。
     これらは /equities/bars/daily に含まれるので取得先を増やさずに済む。
     銘柄コードは書き下ろさず、/equities/master が返した名称から探す。
  3. 候補が 2016-10（契約の最古日）まで遡れるか、毎営業日値が付くか、
     売買代金は十分か

候補リストは「存在するはずのもの」ではなく「叩いて確かめる対象」である。
通ったものだけを docs/MARKET_DATA.md に残す。

名称の突き合わせは NFKC 正規化してから行う。
1回目は生の文字列で比較したため、`ＮＥＸＴ　ＦＵＮＤＳ　日経２２５連動型上場投信`
のように全角数字で書かれた銘柄が "225" に一致せず、
素の日経平均連動 ETF を丸ごと取りこぼした。

結果は research/probe_market_data.json と docs/MARKET_DATA.md に出す。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import sys
import unicodedata
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

# 指数の総当たりで使う短い窓。存在確認だけなので数日で足りる
INDEX_WINDOW = {"from": "2024-05-01", "to": "2024-05-15"}

# 総当たりのリクエスト上限。無制限にすると API を延々叩き続ける
SWEEP_BUDGET = 2500

# --------------------------------------------------------------------------- #
# 1. 指数
# --------------------------------------------------------------------------- #

# パラメータの形を変えて、何が要求されるかをエラーメッセージから読む。
# date で全指数が返るなら総当たりは不要になる
PARAM_SHAPES = [
    {},
    dict(INDEX_WINDOW),
    {"date": "2024-05-15"},
    {"code": "0000", **INDEX_WINDOW},
    {"code": "0000", "date": "2024-05-15"},
]


def probe_param_shapes(client: JQuantsClient) -> List[dict]:
    """/indices/bars/daily がどのパラメータを受け付けるかを測る。"""
    out = []
    for params in PARAM_SHAPES:
        rec = {"params": params}
        try:
            data = client.get("/indices/bars/daily", params)
            batch = data.get("data")
            rec["ok"] = True
            rec["rows"] = len(batch) if isinstance(batch, list) else 0
            rec["keys"] = sorted(batch[0].keys()) if batch else []
            rec["codes"] = sorted({str(r.get("Code")) for r in batch})[:50] if batch else []
        except JQuantsError as exc:
            rec["ok"] = False
            rec["error"] = str(exc)[:400]
        out.append(rec)
        print(f"  {str(params):<58} {'OK' if rec.get('ok') else 'NG'} "
              f"{rec.get('rows', '')}")
        if not rec.get("ok"):
            print(f"      {rec['error'][:220]}")
    return out


def _sweep_codes() -> List[str]:
    """総当たりするコード。数字4桁を全部。"""
    return [f"{i:04d}" for i in range(1000)]


def _extend_codes(hits: List[str]) -> List[str]:
    """
    当たったコードの3文字プレフィックスについて、4文字目を英数字で広げる。

    JPX の指数コードには英字を含むものがありうるので、
    数字だけの総当たりでは取りこぼす。ただし全部を広げると
    リクエストが爆発するため、当たりの近傍だけに絞る。
    """
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    seen, out = set(), []
    for h in hits:
        pre = h[:3]
        for c in alpha:
            code = pre + c
            if code not in seen:
                seen.add(code)
                out.append(code)
    return out


def _probe_index_code(client: JQuantsClient, code: str) -> dict:
    try:
        rows = client.get_paginated("/indices/bars/daily", {"code": code, **INDEX_WINDOW})
    except JQuantsError as exc:
        return {"ok": False, "error": str(exc)[:120]}
    if not rows:
        return {"ok": True, "rows": 0}
    closes = []
    for r in rows:
        try:
            closes.append(float(r.get("C")))
        except (TypeError, ValueError):
            pass
    return {
        "ok": True,
        "rows": len(rows),
        "keys": sorted(rows[0].keys()),
        # 名称は返らないので、水準で見分けるしかない。
        # TOPIX(0000) の水準が分かっているので相対的に当たりを付けられる
        "close": round(statistics.median(closes), 2) if closes else None,
    }


def sweep_indices(client: JQuantsClient) -> dict:
    """存在する指数コードを総当たりで探す。"""
    spent, hits, results = 0, [], {}
    for code in _sweep_codes():
        if spent >= SWEEP_BUDGET:
            break
        r = _probe_index_code(client, code)
        spent += 1
        if r.get("ok") and r.get("rows"):
            hits.append(code)
            results[code] = r
            print(f"    {code}: {r['rows']}日 終値中央値 {r['close']}")
    print(f"  数字4桁: {len(hits)}件ヒット / {spent}リクエスト")

    extended = _extend_codes(hits)
    for code in extended:
        if spent >= SWEEP_BUDGET:
            break
        r = _probe_index_code(client, code)
        spent += 1
        if r.get("ok") and r.get("rows"):
            hits.append(code)
            results[code] = r
            print(f"    {code}: {r['rows']}日 終値中央値 {r['close']}  (英字拡張)")
    print(f"  合計 {len(hits)}件ヒット / {spent}リクエスト（上限 {SWEEP_BUDGET}）")
    return {"hits": sorted(results), "detail": results,
            "requests": spent, "budget": SWEEP_BUDGET}


def index_history(client: JQuantsClient, code: str, end: str) -> dict:
    """指数がどこまで遡れるか。"""
    try:
        rows = client.get_paginated(
            "/indices/bars/daily", {"code": code, "from": EARLIEST_DATE, "to": end})
    except JQuantsError as exc:
        return {"ok": False, "error": str(exc)[:160]}
    if not rows:
        return {"ok": True, "rows": 0}
    dates = sorted(str(r.get("Date")) for r in rows)
    return {"ok": True, "rows": len(rows), "first": dates[0], "last": dates[-1]}


# --------------------------------------------------------------------------- #
# 2. 上場商品からの代用候補
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    """全角・半角の違いで取りこぼさないよう正規化して大文字に揃える。"""
    return unicodedata.normalize("NFKC", str(s or "")).upper()


# 名称に含まれうる断片。ここに書いた語が実在を意味するのではなく、
# /equities/master が返した名称を突き合わせる検索語にすぎない。
# 比較は _norm() を通した後の文字列で行う（全角数字も一致する）。
THEMES: Dict[str, List[str]] = {
    "金": ["ゴールド", "純金", "金価格", "金先物", "GOLD"],
    "日経平均": ["日経225", "日経平均", "NIKKEI"],
    "TOPIX": ["TOPIX", "トピックス", "東証株価指数"],
    "グロース/新興": ["グロース", "マザーズ", "グロース250"],
    "米国株": ["S&P", "ナスダック", "NASDAQ", "米国株", "ダウ"],
    "原油/商品": ["原油", "WTI", "コモディティ"],
    "債券/金利": ["国債", "債券"],
    "為替": ["為替", "米ドル", "ドル建", "通貨"],
    "REIT": ["REIT", "リート", "不動産投信"],
    "ボラティリティ": ["VIX", "ボラティリティ"],
}

# 内国株（普通株）の市場区分。ここに入るものは母集団そのものなので除く。
# 実測では ETF/ETN/REIT は市場区分が「その他」で返る
DOMESTIC_MARKETS = ("プライム", "スタンダード", "グロース", "TOKYO PRO")


def _matched_themes(name: str) -> List[str]:
    n = _norm(name)
    return [theme for theme, words in THEMES.items()
            if any(_norm(w) in n for w in words)]


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
        if any(k in mkt for k in DOMESTIC_MARKETS):
            continue
        themes = _matched_themes(name)
        if not themes:
            continue
        candidates.append({
            "code": str(r.get("Code") or ""),
            "name": name,
            "market": mkt,
            "themes": themes,
        })

    print("  市場区分: " + " / ".join(
        f"{k}:{v}" for k, v in sorted(by_market.items(), key=lambda x: -x[1])))
    print(f"  テーマに引っかかった上場商品: {len(candidates)}件")
    return {"total": len(rows), "by_market": dict(by_market),
            "candidates": candidates}


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
            try:
                sink.append(float(r.get(key)))
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

    L.append("## `/indices/bars/daily` が受け付けるパラメータ")
    L.append("")
    L.append("| パラメータ | 結果 | 件数 | 列 / エラー |")
    L.append("| --- | :---: | ---: | --- |")
    for r in result["paramShapes"]:
        note = ", ".join(r.get("keys", [])) if r.get("ok") else r.get("error", "")
        L.append(f"| `{r['params']}` | {'OK' if r.get('ok') else 'NG'} | "
                 f"{r.get('rows', '—')} | {note[:150]} |")
    L.append("")

    sw = result["indexSweep"]
    L.append("## 存在する指数コード（総当たり）")
    L.append("")
    L.append("数字4桁の全域（0000〜0999）と、当たりの近傍を英字に広げた範囲を叩いた。")
    L.append(f"リクエスト {sw['requests']:,} 件（上限 {sw['budget']:,}）で **{len(sw['hits'])} 件**が値を返した。")
    L.append("")
    L.append("指数のレスポンスに**名称は入っていない**（列は Code / Date / O / H / L / C）。")
    L.append("どの指数かは水準から見当を付けるしかないので、下表には終値の中央値を併記する。")
    L.append("`0000` は TOPIX（`/indices/bars/daily/topix` と同じ内容が返る）。")
    L.append("")
    if sw["hits"]:
        L.append("| コード | 2024-05-01〜15 の終値中央値 | 全期間の行数 | 最古 | 最新 |")
        L.append("| --- | ---: | ---: | --- | --- |")
        for code in sw["hits"]:
            d = sw["detail"][code]
            h = result["indexHistory"].get(code, {})
            L.append(f"| `{code}` | {d.get('close', '—')} | "
                     f"{h.get('rows', '—')} | {h.get('first', '—')} | {h.get('last', '—')} |")
    else:
        L.append("該当なし。")
    L.append("")

    inst = result["instruments"]
    L.append("## 上場商品（ETF/ETN など）からの代用候補")
    L.append("")
    L.append(f"`/equities/master` の {inst['total']:,} 件を名称で検索した結果。")
    L.append("**銘柄コードはこちらで書き下ろしていない。API が返した名称に一致したものだけを載せる。**")
    L.append("照合は NFKC 正規化後に行う（全角数字の `２２５` も `225` に一致する）。")
    L.append("")
    L.append("市場区分の内訳: " + " / ".join(
        f"{k} {v:,}" for k, v in sorted(inst["by_market"].items(), key=lambda x: -x[1])))
    L.append("")
    L.append("価格系列として使えるかは「最古日まで遡れるか」と「毎営業日値が付くか」で決まる。")
    L.append("出来高0の日が多い銘柄は前日終値が据え置かれるため、リターンが人為的に0になる。")
    L.append("")
    if result["history"]:
        L.append("| コード | 名称 | テーマ | 行数 | 最古 | 出来高0の日 | 売買代金の中央値 |")
        L.append("| --- | --- | --- | ---: | --- | ---: | ---: |")
        for h in result["history"]:
            s = h["stats"]
            themes = "・".join(h["themes"])
            if not s.get("ok"):
                L.append(f"| `{h['code']}` | {h['name']} | {themes} | — | — | — | "
                         f"{s.get('error', '')[:60]} |")
                continue
            turnover = s.get("medianTurnoverYen")
            turnover_txt = f"{turnover:,}円" if turnover is not None else "—"
            L.append(
                f"| `{h['code']}` | {h['name']} | {themes} | {s.get('rows', 0):,} | "
                f"{s.get('first', '—')} | {s.get('zeroVolumeShare', '—')}% | {turnover_txt} |")
    else:
        L.append("該当なし。")
    L.append("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[write] {OUT_MD}")


def _save(result: dict) -> None:
    """API を叩き直さずに済むよう、段階ごとに保存する。"""
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


def main() -> int:
    client = JQuantsClient(resolve_api_key(), pause=0.03)
    end = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    result = {"probedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
              "earliest": EARLIEST_DATE, "historyEnd": end}

    print("[1] /indices/bars/daily が受け付けるパラメータ")
    result["paramShapes"] = probe_param_shapes(client)
    _save(result)

    print("\n[2] 指数コードの総当たり")
    result["indexSweep"] = sweep_indices(client)
    _save(result)

    print("\n[3] 見つかった指数の遡及範囲")
    result["indexHistory"] = {}
    for code in result["indexSweep"]["hits"]:
        h = index_history(client, code, end)
        result["indexHistory"][code] = h
        print(f"  {code}: {h.get('rows', 0)}行 {h.get('first', '')}〜{h.get('last', '')}")
    _save(result)

    print("\n[4] 上場商品から代用候補を探す")
    result["instruments"] = probe_instruments(client)
    _save(result)

    print("\n[5] 候補の遡及範囲と流動性")
    # 全部測ると件数が読めないので、テーマごとに数銘柄に絞る。
    # 並びはコード順（恣意的な選別を避ける）
    per_theme: Dict[str, int] = defaultdict(int)
    picked = []
    for c in sorted(result["instruments"]["candidates"], key=lambda x: x["code"]):
        if all(per_theme[t] >= 6 for t in c["themes"]):
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
        print(f"  {c['code']} {c['name'][:34]:<34} "
              f"{s.get('rows', 0)}行 {s.get('first', '')} "
              f"出来高0 {s.get('zeroVolumeShare', '—')}%")
    result["history"] = hist
    _save(result)

    print(f"\n[write] {OUT_JSON}")
    write_md(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
