#!/usr/bin/env python3
"""
research/eda.json から、学習前に見るための特徴量診断ページを組み立てる。

手で書いていたが、データセットを作り直すたびに書き直しになるので生成にした。
中身（欠損・異常値・冗長・ラベルの内訳）は eda.json に入っている集計そのままで、
このスクリプトは並べ方だけを決める。

  python3 research/eda_report.py
  -> research/eda_report.html

出力は Artifact としてそのまま公開できる形（<html>/<body> は付けない）。
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = os.path.join(HERE, "viewer", "eda_style.css")

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Shippori+Mincho:wght@600;700&family=Noto+Sans+JP:wght@400;500;700'
         '&family=JetBrains+Mono:wght@400;600&display=swap">')

#: 分布を並べる列。全部出すと読めないので、判断に使うものだけ。
SHOWCASE = [
    "break_margin", "close_position", "base_length", "ret_20d", "vol_20d",
    "log_market_cap", "log_trading_value", "credit_ratio", "volume_trend",
    "per", "pbr", "psr", "dividend_yield",
    "ROE_q0", "ROA_q0", "op_margin_q0", "equity_ratio_q0",
    "eps_growth_q0", "sales_growth_q0", "eps_growth_chg", "sales_growth_chg",
    "progress_vs_base", "cfo_to_op", "guidance_op_growth",
]

e = html.escape


def num(v, digits=0, unit="") -> str:
    if v is None:
        return "—"
    return f"{v:,.{digits}f}{unit}"


def bars(rows: List[tuple], unit: str = "%", tone=None, width: int = 260) -> str:
    """横棒。rows = [(ラベル, 値, 補足)]。値の最大で幅を正規化する。"""
    if not rows:
        return ""
    top = max(abs(r[1]) for r in rows) or 1
    out = []
    for label, value, note in rows:
        w = abs(value) / top * width
        cls = tone(value) if tone else "accent"
        out.append(
            f'<div class="bar-label">{e(label)}</div>'
            f'<div><svg width="{width}" height="14" role="img" '
            f'aria-label="{e(label)} {value}{unit}">'
            f'<rect x="0" y="2" width="{w:.1f}" height="10" rx="2" '
            f'fill="var(--{cls})"></rect></svg></div>'
            f'<div class="bar-value">{value:,.2f}{unit}</div>'
            f'<div class="n">{e(note)}</div>')
    return f'<div class="bars">{"".join(out)}</div>'


def spark(hist: Optional[Dict], w: int = 200, h: int = 46) -> str:
    """ヒストグラムの小さな面。軸ラベルは1%〜99%の両端だけ出す。"""
    if not hist or not hist.get("counts"):
        return '<div class="n">分布なし</div>'
    counts = hist["counts"]
    top = max(counts) or 1
    n = len(counts)
    bw = w / n
    rects = "".join(
        f'<rect x="{i*bw:.2f}" y="{h - c/top*h:.2f}" width="{max(bw-0.6,0.6):.2f}" '
        f'height="{c/top*h:.2f}" fill="var(--accent)"></rect>'
        for i, c in enumerate(counts) if c)
    return (f'<svg width="{w}" height="{h}" role="img" aria-label="分布">'
            f'{rects}</svg>')


def section(no: str, title: str, body: str, lede: str = "") -> str:
    led = f'<p class="lede">{lede}</p>' if lede else ""
    return (f'<section><h2><span class="num">{no}</span>{e(title)}</h2>'
            f'{led}{body}</section>')


def build(d: Dict, cmp_: Optional[Dict]) -> str:
    stats = d["stats"]
    lab = d["label"]
    cont = lab.get("continuation", {})

    # --- ヘッダ --- #
    facts = [
        ("サンプル", num(d["n_rows"]), "78週高値の更新日 × 銘柄"),
        ("特徴量", num(d["n_features"]), "順位版を含む"),
        ("銘柄", num(d["n_codes"]), ""),
        ("正例率", num(lab["overall_rate"], 2, "%"), f'{num(lab["n_positive"])}件'),
        ("期間", f'{d["date_min"]}<br>{d["date_max"]}', ""),
    ]
    head = ('<div class="facts">' + "".join(
        f'<div class="fact"><div class="eyebrow">{e(k)}</div>'
        f'<div class="n">{v}</div><div class="sub">{e(s)}</div></div>'
        for k, v, s in facts) + "</div>")

    # --- 01 ラベル --- #
    body = ""
    if cont:
        reached = cont.get("reached")
        pos = cont.get("positive")
        dropped = cont.get("reached_but_dropped")
        rows = [
            ("到達（+20%）", cont.get("reached_rate", 0), f'{num(reached)}件'),
            ("継続まで満たす（正例）",
             round(pos / d["n_rows"] * 100, 2) if pos else 0, f'{num(pos)}件'),
        ]
        body += bars(rows, "%", lambda v: "accent")
        body += (f'<p class="note">到達したうち <b>{num(dropped)}件</b> が'
                 f'継続の条件で外れた。定義: <code>{e(cont.get("definition",""))}</code></p>')
    fr = lab.get("future_rise_pct")
    if fr:
        body += ('<div class="scroll" style="margin-top:22px"><table><thead>'
                 '<tr><th>将来リターン（60営業日以内の最大上昇）</th>'
                 + "".join(f"<th>{e(k)}</th>" for k in fr) + "</tr></thead><tbody><tr>"
                 "<td>全サンプル</td>"
                 + "".join(f'<td class="v">{v:+.2f}%</td>' for v in fr.values())
                 + "</tr></tbody></table></div>")
    secs = [section("01", "目的変数 — 到達と継続", body,
                    "到達だけを条件にすると一瞬の吹き上げが正例に混ざる。"
                    "継続の条件で何件が外れたかを見る。")]

    # --- 02 偏り --- #
    b = ""
    mc = lab.get("by_market_cap")
    if mc:
        b += "<h3>時価総額帯</h3>" + bars(
            [(k, v["rate"], f'{num(v["n"])}件') for k, v in mc.items()],
            "%", lambda v: "bad" if v >= 12 else "accent")
    by_year = lab.get("by_year")
    if by_year:
        # 年が抜けていると、並べただけでは気づけない。
        # 実測に無い年は 0件として明示的に出す
        yrs = [int(k) for k in by_year]
        rows = []
        for y in range(min(yrs), max(yrs) + 1):
            v = by_year.get(str(y))
            rows.append((str(y), v["rate"] if v else 0.0,
                         f'{num(v["n"])}件' if v else "サンプルなし"))
        b += "<h3>年</h3>" + bars(rows, "%", lambda v: "bad" if v == 0 else "accent")
    secs.append(section("02", "正例率の偏り", b,
                        "帯によって正例率が大きく違うなら、"
                        "同じ帯の中で比べないと実力差を測れない。"))

    # --- 03 要対応 --- #
    probs = d.get("problems", [])
    if probs:
        tone = {"欠損が9割超": "t-bad", "値が1種類しかない": "t-bad",
                "無限大が含まれる": "t-bad"}
        rows = "".join(
            f'<tr><td><code>{e(p["col"])}</code></td>'
            f'<td><span class="tag {tone.get(p["kind"], "t-warn")}">{e(p["kind"])}</span></td>'
            f'<td class="v">{e(p["detail"])}</td></tr>' for p in probs)
        b = ('<div class="scroll"><table><thead><tr><th>列</th><th>種類</th>'
             f'<th>程度</th></tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        b = '<p class="note">検出なし。</p>'
    secs.append(section("03", f"要対応の列（{len(probs)}件）", b,
                        "欠損9割超・値が1種類・無限大・外れ値1割超のいずれかに当たる列。"))

    # --- 04 グループ別の欠損 --- #
    groups = d.get("groups", {})
    rows = []
    for g, cols in groups.items():
        ms = sorted(stats[c]["missing_pct"] for c in cols if c in stats)
        if not ms:
            continue
        rows.append((g, ms[len(ms) // 2], f"{len(ms)}列 / 最大 {max(ms):.1f}%"))
    rows.sort(key=lambda r: -r[1])
    secs.append(section("04", "グループ別の欠損率（中央値）",
                        bars(rows, "%", lambda v: "bad" if v >= 40
                             else "warn" if v >= 20 else "good"),
                        "決算の履歴を遡る列は構造上どうしても欠ける。"
                        "ゼロにはできないので、どこが重いかを把握しておく。"))

    # --- 05 欠損の多い列 --- #
    worst = sorted(((s["missing_pct"], c) for c, s in stats.items()),
                   reverse=True)[:15]
    rows = "".join(
        f'<tr><td><code>{e(c)}</code></td><td class="v">{m:.2f}%</td>'
        f'<td class="v">{num(stats[c].get("n_unique"))}</td></tr>'
        for m, c in worst)
    secs.append(section("05", "欠損の多い列 上位15",
                        '<div class="scroll"><table><thead><tr><th>列</th>'
                        '<th>欠損率</th><th>値の種類</th></tr></thead>'
                        f'<tbody>{rows}</tbody></table></div>'))

    # --- 06 冗長 --- #
    red = d.get("redundant", [])[:15]
    if red:
        rows = "".join(
            f'<tr><td><code>{e(r["a"])}</code></td><td><code>{e(r["b"])}</code></td>'
            f'<td class="v">{r["corr"]:.4f}</td></tr>' for r in red)
        b = ('<div class="scroll"><table><thead><tr><th>列</th><th>列</th>'
             f'<th>相関</th></tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        b = '<p class="note">相関 0.95 以上の組は無し。</p>'
    secs.append(section("06", "冗長な列（相関 0.95 以上）", b,
                        "片方が他方の単調変換なら、木は同じ分岐しか作れない。"))

    # --- 07 分布 --- #
    cards = []
    for c in SHOWCASE:
        s = stats.get(c)
        if not s:
            continue
        h = d.get("hist", {}).get(c)
        cards.append(
            f'<div class="hcard"><h3>{e(c)}</h3>'
            f'<div class="hmeta">中央値 {num(s.get("p50"), 2)}'
            f' / 欠損 {s["missing_pct"]:.1f}%</div>{spark(h)}</div>')
    secs.append(section("07", "主要特徴量の分布", f'<div class="grid">{"".join(cards)}</div>',
                        "1%〜99%の範囲で描く。両端に張り付いていれば発散を疑う。"))

    css = open(CSS, encoding="utf-8").read()
    return (f"<title>新高値ブレイク 特徴量診断</title>{FONTS}<style>{css}</style>"
            '<div class="wrap"><header>'
            "<h1>新高値ブレイク 特徴量診断</h1>"
            '<p class="lede">モデルを学習させる前に、特徴量が使える状態かを確認する。'
            "欠損・異常値・冗長・目的変数の偏りを、集計そのままで並べてある。</p>"
            f"{head}</header>{''.join(secs)}"
            '<footer class="note">データ: J-Quants API V2（日本取引所グループ）／'
            "集計は research/eda_stats.py、組版は research/eda_report.py。</footer></div>")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="EDA の診断ページを組み立てる")
    ap.add_argument("--eda", default=os.path.join(HERE, "eda.json"))
    ap.add_argument("--compare", default=os.path.join(
        HERE, "samples", "label_comparison.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "eda_report.html"))
    args = ap.parse_args(argv)

    with open(args.eda, encoding="utf-8") as fh:
        d = json.load(fh)
    cmp_ = None
    if os.path.exists(args.compare):
        with open(args.compare, encoding="utf-8") as fh:
            cmp_ = json.load(fh)

    out = build(d, cmp_)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"[done] {args.out} ({len(out)/1000:.1f}KB / {d['n_rows']:,}サンプル)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
