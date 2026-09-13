#!/usr/bin/env python3
"""
research/_data/accgraph/eda.json から EDA ページを組み立てる。

集計は research/accgraph/eda.py が済ませてある。ここは並べ方だけを決める。
出力は Artifact としてそのまま公開できる形（<html>/<body> は付けない）。

  python3 research/accgraph/eda_report.py
  -> docs/accgraph_eda.html

## 配色について

3クラス（下落／中立／上昇）は「向き」を表すので発散配色にする。
赤↔緑にすると第2色覚では赤と緑の色差が ΔE 3.8 しかなく、
色だけでは読めない。赤↔青にすると ΔE 16 以上になるのでそちらを使う。
中間の「中立」は文字どおり何も起きていないことなので、地に近い灰色にする。
どの配色でも凡例と直接ラベルを必ず添え、色だけに意味を持たせない。

充足率のように「量」を表すものは単色の濃淡（順次配色）にする。
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
ROOT = os.path.dirname(RESEARCH)
CSS = os.path.join(RESEARCH, "viewer", "eda_style.css")
OUT = os.path.join(ROOT, "docs", "accgraph_eda.html")
IN_JSON = os.path.join(RESEARCH, "_data", "accgraph", "eda.json")

sys.path.insert(0, RESEARCH)
from accgraph.eda import CORR_WARN  # noqa: E402  （しきい値を2箇所に書かない）

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Shippori+Mincho:wght@600;700&family=Noto+Sans+JP:wght@400;500;700'
         '&family=JetBrains+Mono:wght@400;600&display=swap">')

#: 追加の配色とレイアウト。eda_style.css には無い図のためだけに足す。
#: 値はすべて検証済み（明暗どちらの地でも、明度帯・彩度下限・
#: 色覚特性での色差・通常視での色差・地とのコントラストの5点を満たす）。
EXTRA_CSS = """
:root{
  --c-down:#A33636; --c-flat:#DDE2E8; --c-up:#28619B;
  --c-flat-line:#B8C0CA;
  --heat:#28619B;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --c-down:#D96262; --c-flat:#333B47; --c-up:#5E93CF;
    --c-flat-line:#4A5462; --heat:#5E93CF;
  }
}
:root[data-theme="dark"]{
  --c-down:#D96262; --c-flat:#333B47; --c-up:#5E93CF;
  --c-flat-line:#4A5462; --heat:#5E93CF;
}
/* eda_style.css の svg{width:100%} は viewBox 無しの図と相性が悪い。
   ビューポートだけ伸びて中身は元の大きさのまま取り残され、
   棒とラベルが縦にずれる。属性どおりの大きさに戻す */
.bars svg,.stack svg{width:auto;height:auto}
/* 狭い画面では図と表を横スクロールさせる。縮めると
   ラベルが縦書きに折れたり、棒が切れたりして読めなくなる */
/* eda_style.css は table-layout:fixed で列幅を固定するが、列が多い表では
   1列目が潰れて縦書きに折れる。横スクロールできる表は内容に合わせる */
.scroll table{table-layout:auto;min-width:600px}
.scroll table td:first-child{white-space:nowrap}
.scroll .bars,.scroll .stack{min-width:max-content}
.key{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--ink-2);
     margin:0 0 14px;font-family:var(--mono)}
.key span{display:inline-flex;align-items:center;gap:6px}
.key i{width:11px;height:11px;border-radius:2px;display:inline-block}
.stack{display:grid;
       grid-template-columns:minmax(88px,max-content) max-content max-content;
       gap:4px 14px;align-items:center;margin:0;justify-content:start}
.stack .lab{font-size:12.5px;color:var(--ink-2);white-space:nowrap}
.stack .cnt{font-family:var(--mono);font-size:11px;color:var(--ink-3);
            font-variant-numeric:tabular-nums;white-space:nowrap}
.heat{border-collapse:separate;border-spacing:2px;width:100%;table-layout:fixed}
.heat th{padding:0 0 6px;font-size:10.5px}
.heat td{padding:0;border:0}
.heat .cell{height:26px;border-radius:2px;display:flex;align-items:center;
            justify-content:center;font-family:var(--mono);font-size:11px;
            font-variant-numeric:tabular-nums}
.heat .name{font-size:12.5px;color:var(--ink-2);white-space:nowrap;
            padding-right:10px;overflow:hidden;text-overflow:ellipsis}
.heat .sub{font-family:var(--mono);font-size:10px;color:var(--ink-3)}
.chart-marks{position:relative;height:15px;font-family:var(--mono);
             font-size:10.5px;color:var(--ink-3)}
.chart-marks span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.chart-axis{display:flex;justify-content:space-between;font-family:var(--mono);
            font-size:10.5px;color:var(--ink-3);margin-top:5px}
.chart svg{width:100%;height:130px}
.sublab{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);
        letter-spacing:.06em;margin:16px 0 6px}
details{margin:10px 0 0}
summary{font-family:var(--mono);font-size:11px;color:var(--ink-3);cursor:pointer}
summary:hover{color:var(--ink-2)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:26px}
.mini h3{margin-top:0}
"""

e = html.escape

#: 3クラスの並び（構成比の辞書キーと一致させる）
CLASS_ORDER = ("下落", "中立", "上昇")
CLASS_VAR = {"下落": "c-down", "中立": "c-flat", "上昇": "c-up"}


# --------------------------------------------------------------------------- #
# 小道具
# --------------------------------------------------------------------------- #

def num(v, digits=0, unit="") -> str:
    if v is None:
        return "—"
    try:
        if v != v:          # NaN
            return "—"
    except TypeError:
        return "—"
    return f"{v:,.{digits}f}{unit}"


def section(no: str, title: str, body: str, lede: str = "") -> str:
    led = f'<p class="lede">{lede}</p>' if lede else ""
    return (f'<section><h2><span class="num">{no}</span>{e(title)}</h2>'
            f'{led}{body}</section>')


def key(items: List[Tuple[str, str]]) -> str:
    """凡例。2系列以上の図には必ず付ける（色だけに意味を持たせないため）。"""
    return ('<div class="key">' + "".join(
        f'<span><i style="background:var(--{var})"></i>{e(label)}</span>'
        for label, var in items) + "</div>")


def bars(rows: List[tuple], unit: str = "%", tone=None, width: int = 240,
         digits: int = 1) -> str:
    """横棒。rows = [(ラベル, 値, 補足)]。値の最大で幅を正規化する。"""
    rows = [r for r in rows if r[1] is not None and r[1] == r[1]]
    if not rows:
        return '<p class="note">該当なし。</p>'
    top = max(abs(r[1]) for r in rows) or 1
    out = []
    for label, value, note in rows:
        w = abs(value) / top * width
        cls = tone(value) if tone else "accent"
        out.append(
            f'<div class="bar-label">{e(str(label))}</div>'
            f'<div><svg width="{width}" height="14" role="img" '
            f'aria-label="{e(str(label))} {value:.{digits}f}{unit}">'
            f'<title>{e(str(label))} {value:,.{digits}f}{unit}</title>'
            f'<rect x="0" y="2" width="{w:.1f}" height="10" rx="2" '
            f'fill="var(--{cls})"></rect></svg></div>'
            f'<div class="bar-value">{value:,.{digits}f}{unit}</div>'
            f'<div class="n">{e(str(note))}</div>')
    return f'<div class="scroll"><div class="bars">{"".join(out)}</div></div>'


def diverging_bars(rows: List[tuple], unit: str = "", width: int = 240,
                   digits: int = 3) -> str:
    """
    0 を中心に左右へ伸びる横棒。符号そのものが読みたいときに使う。
    rows = [(ラベル, 値, 補足)]
    """
    rows = [r for r in rows if r[1] is not None and r[1] == r[1]]
    if not rows:
        return '<p class="note">該当なし。</p>'
    top = max(abs(r[1]) for r in rows) or 1
    half = width / 2
    out = []
    for label, value, note in rows:
        w = abs(value) / top * half
        x = half if value >= 0 else half - w
        var = "c-up" if value >= 0 else "c-down"
        out.append(
            f'<div class="bar-label">{e(str(label))}</div>'
            f'<div><svg width="{width}" height="14" role="img" '
            f'aria-label="{e(str(label))} {value:+.{digits}f}{unit}">'
            f'<title>{e(str(label))} {value:+,.{digits}f}{unit}</title>'
            f'<line x1="{half}" y1="0" x2="{half}" y2="14" '
            f'stroke="var(--line)" stroke-width="1"></line>'
            f'<rect x="{x:.1f}" y="2" width="{w:.1f}" height="10" rx="2" '
            f'fill="var(--{var})"></rect></svg></div>'
            f'<div class="bar-value">{value:+,.{digits}f}{unit}</div>'
            f'<div class="n">{e(str(note))}</div>')
    return f'<div class="scroll"><div class="bars">{"".join(out)}</div></div>'


def stacked(rows: List[tuple], width: int = 420) -> str:
    """
    3クラスの構成比を積み上げ横棒で。rows = [(ラベル, share辞書, 件数)]

    segment どうしは 2px 空ける（隣り合う色が溶けないように）。
    幅が足りる区画には数値を直接置く。凡例は呼び出し側が付ける。
    """
    out = []
    for label, share, n in rows:
        segs, x = [], 0.0
        for cname in CLASS_ORDER:
            v = share.get(cname)
            if v is None or v != v:
                continue
            w = max(v / 100.0 * width - 2, 0.0)
            stroke = (' stroke="var(--c-flat-line)" stroke-width="1"'
                      if cname == "中立" else "")
            segs.append(
                f'<g><title>{e(str(label))} {e(cname)} {v:.1f}%</title>'
                f'<rect x="{x:.1f}" y="3" width="{w:.1f}" height="15" rx="2" '
                f'fill="var(--{CLASS_VAR[cname]})"{stroke}></rect>'
                + (f'<text x="{x + w / 2:.1f}" y="14.5" text-anchor="middle" '
                   f'class="axis" fill="{"var(--ink)" if cname == "中立" else "#fff"}"'
                   f'>{v:.0f}</text>' if w >= 26 else "")
                + '</g>')
            x += w + 2
        out.append(
            f'<div class="lab">{e(str(label))}</div>'
            f'<div><svg width="{width}" height="21" role="img" '
            f'aria-label="{e(str(label))} のクラス構成比">{"".join(segs)}</svg></div>'
            f'<div class="cnt">{num(n)}件</div>')
    return (f'<div class="scroll"><div class="stack">{"".join(out)}</div></div>'
            '<details><summary>表で見る</summary>'
            + share_table(rows, "区分") + "</details>")


def share_table(rows: List[tuple], head: str) -> str:
    """積み上げ棒と同じ内容の表。色を読めない場合の逃げ道として必ず添える。"""
    body = "".join(
        f'<tr><td>{e(str(label))}</td>'
        + "".join(f'<td class="v">{num(share.get(c), 1, "%")}</td>'
                  for c in CLASS_ORDER)
        + f'<td class="v">{num(n)}</td></tr>'
        for label, share, n in rows)
    return ('<div class="scroll"><table><thead><tr>'
            f'<th>{e(head)}</th>'
            + "".join(f"<th>{e(c)}</th>" for c in CLASS_ORDER)
            + "<th>件数</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def heatmap(rows: List[tuple], cols: List[str], caption: str = "") -> str:
    """
    行 × 列の充足率。量なので単色の濃淡（順次配色）。
    薄いほど 0% に近く、地に溶けていく。数値も必ず重ねる。
    """
    head = ('<tr><th></th>' + "".join(f"<th>{e(c)}</th>" for c in cols)
            + "<th>全体</th></tr>")
    body = []
    for name, sub, values, overall in rows:
        cells = []
        for c, v in zip(cols, values):
            if v is None or v != v:
                cells.append('<td><div class="cell" style="background:var(--surface-2);'
                             'color:var(--ink-3)">—</div></td>')
                continue
            # 0.06 を下限にして、0% でも枠として見えるようにする
            op = 0.06 + (v / 100.0) * 0.94
            ink = "#fff" if v >= 55 else "var(--ink)"
            cells.append(
                f'<td><div class="cell" title="{e(name)} {e(c)} {v:.1f}%" '
                f'style="background:color-mix(in srgb, var(--heat) {op*100:.0f}%, '
                f'var(--surface));color:{ink}">{v:.0f}</div></td>')
        cells.append(
            f'<td><div class="cell" style="background:var(--surface-2)">'
            f'{overall:.0f}</div></td>')
        body.append(f'<tr><td><div class="name">{e(name)}'
                    f' <span class="sub">{e(sub)}</span></div>'
                    f'</td>{"".join(cells)}</tr>')
    cap = f'<figcaption>{e(caption)}</figcaption>' if caption else ""
    return ('<figure><div class="scroll"><table class="heat">'
            f'<thead>{head}</thead><tbody>{"".join(body)}</tbody>'
            f'</table></div>{cap}</figure>')


def area(hist: Optional[Dict], h: int = 130,
         marks: Optional[List[Tuple[float, str]]] = None,
         unit: str = "%", scale: float = 100.0) -> str:
    """
    ヒストグラムを面で描く。1%〜99% の範囲だけを描き、
    範囲外の件数は脚注に出す（外れ値で山が潰れないように）。
    marks には縦線を引きたい値（0 やしきい値）を渡す。

    文字はすべて SVG の外（HTML）に置く。中に置くと、幅に合わせて
    SVG が拡大されたときに文字まで一緒に拡大され、画面幅ごとに
    大きさが変わってしまう。
    """
    if not hist or not hist.get("counts"):
        return '<p class="note">分布を出せる件数がありません。</p>'
    counts = hist["counts"]
    lo, hi = hist["lo"], hist["hi"]
    top = max(counts) or 1
    n = len(counts)
    w = 1000.0
    bw = w / n
    rects = "".join(
        f'<rect x="{i*bw:.2f}" y="{h - c/top*h:.2f}" width="{max(bw-2,1):.2f}" '
        f'height="{c/top*h:.2f}" fill="var(--accent)">'
        f'<title>{(lo + (i+0.5)*(hi-lo)/n)*scale:,.2f}{unit}: {c:,}件</title></rect>'
        for i, c in enumerate(counts) if c)

    lines, labels = "", []
    for value, label in (marks or []):
        if not (lo <= value <= hi):
            continue
        frac = (value - lo) / (hi - lo)
        lines += (f'<line x1="{frac*w:.1f}" y1="0" x2="{frac*w:.1f}" y2="{h}" '
                  f'stroke="var(--ink-3)" stroke-width="1" stroke-dasharray="3 3" '
                  f'vector-effect="non-scaling-stroke"></line>')
        labels.append(f'<span style="left:{frac*100:.2f}%">{e(label)}</span>')

    out = (f'<div class="chart">'
           f'<div class="chart-marks">{"".join(labels)}</div>'
           f'<svg viewBox="0 0 {w:.0f} {h}" preserveAspectRatio="none" '
           f'role="img" aria-label="分布">{rects}{lines}</svg>'
           f'<div class="chart-axis"><span>{lo*scale:,.1f}{unit}</span>'
           f'<span>{hi*scale:,.1f}{unit}</span></div></div>')
    if hist.get("outside"):
        out += (f'<figcaption>1〜99パーセンタイルの範囲のみ。'
                f'範囲外 {hist["outside"]:,}件（全 {hist["n"]:,}件）</figcaption>')
    return f"<figure>{out}</figure>"


def spark(hist: Optional[Dict], w: int = 200, h: int = 44) -> str:
    if not hist or not hist.get("counts"):
        return '<div class="n">分布なし</div>'
    counts = hist["counts"]
    top = max(counts) or 1
    bw = w / len(counts)
    rects = "".join(
        f'<rect x="{i*bw:.2f}" y="{h - c/top*h:.2f}" width="{max(bw-0.6,0.6):.2f}" '
        f'height="{c/top*h:.2f}" fill="var(--accent)"></rect>'
        for i, c in enumerate(counts) if c)
    return (f'<svg width="{w}" height="{h}" role="img" aria-label="分布">'
            f'{rects}</svg>')


# --------------------------------------------------------------------------- #
# 各節
# --------------------------------------------------------------------------- #

def sec_availability(d: Dict) -> str:
    av = d["availability"]
    nodes = av["nodes"]
    qcols = ["1Q", "2Q", "3Q", "4Q/通期"]

    rows = []
    for n in nodes:
        vals = [n["by_quarter"].get(str(k)) for k in (1, 2, 3, 4)]
        sub = ("開示値" if n["source"] == "disclosed" else "計算値") + f" / {n['statement']}"
        rows.append((n["name_ja"], sub, vals, n["overall"]))
    body = heatmap(rows, qcols,
                   "数値は当該四半期にそのノードの値が入っていた割合(%)。"
                   "濃いほど埋まっている。")

    empty = [n for n in nodes if n["overall"] < 60]
    if empty:
        body += ('<p class="note">全体の充足が6割を切るノード: '
                 + "、".join(f'<b>{e(n["name_ja"])}</b>（{n["overall"]:.0f}%）'
                             for n in empty)
                 + "。いずれもキャッシュフロー計算書に由来し、"
                   "四半期開示の義務が無いため 1Q・3Q が空く。</p>")

    half = [n for n in nodes if (n.get("half_year_share") or 0) > 50]
    if half:
        body += ('<p class="note">値が入っていても<b>半期をまとめた金額</b>に'
                 "なっているノード: "
                 + "、".join(f'{e(n["name_ja"])}（{n["half_year_share"]:.0f}%）'
                             for n in half)
                 + "。1四半期あたりに割り戻してあり、"
                   "何期ぶんかは特徴量 <code>span</code> に残してある。</p>")

    lens = av["seq_len_hist"]
    body += "<h3>過去8四半期がそろっているか</h3>"
    body += bars([(f"{k}期", v / d["n_samples"] * 100, f"{v:,}件")
                  for k, v in sorted(lens.items(), key=lambda kv: int(kv[0]))
                  if v],
                 "%", lambda v: "accent")
    body += (f'<p class="note">8期すべてそろっているのは'
             f'<b>{av["full_seq_pct"]:.1f}%</b>。'
             "足りないぶんは 0 で埋めず、欠測として <code>period_mask</code> に"
             "残してある。上場が浅い銘柄ほど短くなる。</p>")

    body += "<h3>年別のサンプル数</h3>"
    by_year = av["samples_by_year"]
    body += bars([(k, v, "") for k, v in sorted(by_year.items())],
                 "件", lambda v: "accent", digits=0)

    edges = sorted(av["edges"], key=lambda x: x["both_present"])
    body += "<h3>両端がそろっているエッジ（少ない順に10本）</h3>"
    body += bars([(f'{x["src"]} → {x["dst"]}', x["both_present"], x["kind"])
                  for x in edges[:10]], "%",
                 lambda v: "bad" if v < 55 else "warn" if v < 80 else "good")
    return section(
        "01", "何が取れて、何が取れないか", body,
        "このプロジェクトの前提を決める節。取れない項目は推定で埋めず、"
        "欠測のまま持つ方針なので、どこが空くのかを先に把握しておく。")


def sec_label(d: Dict) -> str:
    lab = d["label"]
    h = str(lab.get("primary_horizon", 20))
    horizons = lab["horizons"]

    # --- 分布 --- #
    body = "<h3>超過リターンの分布（20営業日・TOPIX控除）</h3>"
    blk = horizons.get(h, {})
    tx = blk.get("topix", {})
    body += area(tx.get("hist"), marks=[(0.0, "0"), (0.02, "+2%"), (-0.02, "-2%")])
    body += ('<div class="scroll"><table><thead><tr><th>保有期間</th>'
             '<th>控除</th><th>1%</th><th>25%</th><th>中央</th><th>75%</th>'
             '<th>99%</th><th>標準偏差</th></tr></thead><tbody>')
    for hh, blk2 in horizons.items():
        for bench, name in (("raw", "控除なし"), ("topix", "TOPIX"),
                            ("sector", "業種指数")):
            b = blk2.get(bench)
            if not b:
                continue
            qq = b.get("quantiles", {})
            pc = lambda k: (qq.get(k) * 100 if qq.get(k) is not None else None)
            body += (f'<tr><td>{e(hh)}営業日</td><td>{e(name)}</td>'
                     + "".join(f'<td class="v">{num(pc(k), 2, "%")}</td>'
                               for k in ("p01", "p25", "p50", "p75", "p99"))
                     + f'<td class="v">{num(pc("std"), 2, "%")}</td></tr>')
    body += "</tbody></table></div>"

    # --- クラス比 --- #
    body += "<h3>クラス構成比</h3>"
    body += key([(c, CLASS_VAR[c]) for c in CLASS_ORDER])
    rows = []
    for hh, blk2 in horizons.items():
        for bench, name in (("topix", "TOPIX"), ("sector", "業種指数")):
            b = blk2.get(bench)
            if b:
                rows.append((f"{hh}営業日 / {name}", b["share"], b["n"]))
    body += stacked(rows)

    # --- 偏り --- #
    groups = [("by_year", "年"), ("by_quarter", "四半期"),
              ("by_turnover", "流動性（20日平均売買代金）"),
              ("by_crowding", "決算の混雑度（同日の開示社数）"),
              ("by_sector", "業種（上位のみ）")]
    for gkey, gname in groups:
        g = lab.get(gkey)
        if not g:
            continue
        items = sorted(g.items(), key=lambda kv: -kv[1]["n"])
        if gkey in ("by_year", "by_quarter"):
            items = sorted(g.items())
        if gkey == "by_sector":
            items = items[:12]
        fmt = (lambda k: f"{k}Q") if gkey == "by_quarter" else (lambda k: k)
        rows = [(fmt(k), v["share"], v["n"]) for k, v in items]
        body += f"<h3>{e(gname)}</h3>"
        body += '<p class="sublab">クラス構成比</p>'
        body += stacked(rows)
        body += '<p class="sublab">超過リターンの中央値</p>'
        body += diverging_bars([(fmt(k), (v.get("median_excess") or 0) * 100,
                                 f'平均 {num((v.get("mean_excess") or 0) * 100, 2, "%")}')
                                for k, v in items], "%", digits=2)

    # --- ベンチマーク控除が効いているか --- #
    be = lab.get("benchmark_effect")
    if be:
        body += "<h3>ベンチマーク控除は市場全体の影響を落とせているか</h3>"
        ry = be.get("raw_mean_by_year", {})
        ey = be.get("excess_mean_by_year", {})
        body += ('<div class="two">'
                 '<div class="mini"><h3>控除なしの年平均</h3>'
                 + diverging_bars([(k, v * 100, "") for k, v in sorted(ry.items())],
                                  "%", digits=2)
                 + '</div><div class="mini"><h3>TOPIX控除後の年平均</h3>'
                 + diverging_bars([(k, v * 100, "") for k, v in sorted(ey.items())],
                                  "%", digits=2)
                 + "</div></div>")
        body += (f'<p class="note">控除なしの標準偏差 '
                 f'<b>{num((be.get("raw_std") or 0) * 100, 2, "%")}</b> → '
                 f'控除後 <b>{num((be.get("excess_std") or 0) * 100, 2, "%")}</b>。'
                 "年ごとの平均が 0 に寄っていれば、その年の地合いを落とせている。"
                 "左右に振れたままなら、モデルは決算ではなく相場つきを学ぶことになる。</p>")
    return section(
        "02", "目的変数の分布と偏り", body,
        "クラス比が帯によって大きく違うなら、同じ帯の中で比べないと実力を測れない。"
        "しきい値 ±2% は調整前提の値なので、分布と併せて妥当性を見る。")


def sec_features(d: Dict) -> str:
    f = d["features"]
    stats = f["stats"]
    body = ""

    worst = sorted(((s["missing_pct"], c) for c, s in stats.items()
                    if c.endswith(".scaled")), reverse=True)[:12]
    body += "<h3>欠損率の高いノード（当該四半期）</h3>"
    body += bars([(c.replace(".scaled", ""), m, "") for m, c in worst], "%",
                 lambda v: "bad" if v >= 50 else "warn" if v >= 20 else "good")

    clipped = sorted(((s.get("clipped_pct") or 0, c) for c, s in stats.items()),
                     reverse=True)[:10]
    clipped = [(c, v, "") for v, c in clipped if v > 0]
    body += "<h3>打ち切りに張り付いた割合</h3>"
    if clipped:
        body += bars(clipped, "%", lambda v: "bad" if v >= 1 else "warn", digits=2)
        body += ('<p class="note">売上高がほぼ 0 の四半期で比が発散するため、'
                 "比率系の特徴量は上下で打ち切ってある。"
                 "張り付きが多い列は、正規化の分母を選び直す候補。</p>")
    else:
        body += '<p class="note">張り付きなし。</p>'

    const = f.get("constant", [])
    body += f"<h3>値が1種類しかない列（{len(const)}件 / 全{f['n_columns']}列）</h3>"
    if const:
        body += ('<p class="note">'
                 + "、".join(f"<code>{e(c)}</code>" for c in const[:14])
                 + ("…" if len(const) > 14 else "")
                 + "。正規化の分母に使っているノード自身は、定義上 1 に固定される"
                   "（売上高 ÷ 売上高）。学習には効かないが、"
                   "取り除くと列の並びがスキーマとずれるので残してある。</p>")
    else:
        body += '<p class="note">検出なし。</p>'

    red = f.get("redundant", [])
    body += f"<h3>冗長かもしれない組（相関 {CORR_WARN} 以上・上位15）</h3>"
    if red:
        body += ('<div class="scroll"><table><thead><tr><th>列</th><th>列</th>'
                 '<th>相関</th></tr></thead><tbody>'
                 + "".join(f'<tr><td><code>{e(r["a"])}</code></td>'
                           f'<td><code>{e(r["b"])}</code></td>'
                           f'<td class="v">{r["corr"]:.4f}</td></tr>'
                           for r in red[:15])
                 + "</tbody></table></div>")
        body += (f'<p class="note">相関は最大 {f.get("corr_rows", 0):,}行の'
                 "抽出で計算している（総当たりは列数の2乗に比例するため）。"
                 "片方が他方の単調変換なら、木は同じ分岐しか作れない。</p>")
    else:
        body += '<p class="note">検出なし。</p>'

    hist = f.get("hist", {})
    if hist:
        body += "<h3>主要ノードの分布</h3>"
        cards = []
        for col, hh in hist.items():
            s = stats.get(col, {})
            cards.append(
                f'<div class="hcard"><h3>{e(col)}</h3>'
                f'<div class="hmeta">中央値 {num(s.get("p50"), 3)}'
                f' / 欠損 {num(s.get("missing_pct"), 1, "%")}</div>'
                f'{spark(hh)}</div>')
        body += f'<div class="grid">{"".join(cards)}</div>'
        body += ('<p class="note">欠測の期は 0 で埋めてあるが、'
                 "この分布からは外してある。混ぜると 0 に山が立ち、"
                 "本当の分布が読めなくなる。</p>")
    return section(
        "03", "特徴量の診断", body,
        "学習の前に、特徴量が使える状態かを確認する。"
        "欠損・打ち切り・定数・冗長を集計そのままで並べてある。")


def sec_ic(d: Dict) -> str:
    ic = d.get("ic") or {}
    top = ic.get("top", [])
    if not top:
        return section("04", "単変量の情報量",
                       '<p class="note">サンプル数が足りず算出していない。</p>')
    body = key([("正の相関", "c-up"), ("負の相関", "c-down")])
    body += diverging_bars([(r["col"], r["ic"], f'{r["n"]:,}件') for r in top],
                           "", digits=4)
    body += (f'<p class="note">{e(str(ic.get("horizon")))}営業日の'
             "超過リターンとの順位相関。"
             f'最大でも <b>|{ic.get("max_abs_ic", 0):.4f}|</b>。'
             "順位相関にしているのは、外れ値1件で符号が変わらないようにするため。"
             "単変量でこの水準なら、モデルに期待できる上積みも大きくない。"
             "<b>0.1 を大きく超える列が出たら、まずリークを疑うこと。</b></p>")
    return section("04", "単変量の情報量（情報係数）", body,
                   "グラフにする以前に、そもそも材料があるのかを見る。"
                   "ここが全部ゼロ付近なら、構造を足しても出てこない。")


def sec_structure(d: Dict) -> str:
    st = d["structure"]
    body = ""
    pat = st.get("cf_sign_patterns", {})
    if pat:
        meaning = {
            "+--": "本業で稼ぎ、投資し、返済している（成熟）",
            "+-+": "本業で稼ぎ、投資し、資金も調達している（拡大）",
            "++-": "本業で稼ぎ、資産を売り、返済している（縮小・整理）",
            "--+": "本業が赤字で、投資し、調達している（先行投資）",
            "-++": "本業が赤字で、資産を売り、調達している（資金繰り）",
            "+++": "3区分すべて流入",
            "---": "3区分すべて流出",
            "++ +": "",
        }
        def sign_label(code: str) -> str:
            names = ("営業", "投資", "財務")
            return " / ".join(f"{n}{c}" for n, c in zip(names, code))

        rows = [(sign_label(k), v["pct"], meaning.get(k, ""))
                for k, v in sorted(pat.items(), key=lambda kv: -kv[1]["pct"])]
        body += "<h3>営業CF・投資CF・財務CF の符号の組み合わせ</h3>"
        body += bars(rows, "%", lambda v: "accent")
        body += (f'<p class="note">CF が3区分そろっているサンプル'
                 f'（全体の {num(st.get("cf_available_pct"), 1, "%")}）のみ。'
                 "符号は会計の型そのもので、グラフのエッジ特徴量 "
                 "<code>same_sign</code> が拾っている情報でもある。</p>")

    qual = st.get("cfo_to_op", {})
    if qual.get("n"):
        body += "<h3>営業CF ÷ 営業利益（利益の質）</h3>"
        body += area(qual.get("hist"), marks=[(1.0, "1.0")], unit="", scale=1.0)
        q = qual["quantiles"]
        body += (f'<p class="note">中央値 <b>{num(q.get("p50"), 2)}</b>'
                 f'（25% {num(q.get("p25"), 2)} / 75% {num(q.get("p75"), 2)}）。'
                 f"{qual['n']:,}件。1 を大きく下回るなら、利益ほどには現金が"
                 "入っていない。減価償却費が取れないので発生主義との差を"
                 "分解はできず、比としてしか見られない。</p>")

    edges = st.get("edges", [])
    if edges:
        body += "<h3>エッジごとの比率と符号の一致</h3>"
        body += ('<div class="scroll"><table><thead><tr><th>from → to</th>'
                 '<th>種別</th><th>25%</th><th>中央</th><th>75%</th>'
                 '<th>符号一致</th><th>件数</th></tr></thead><tbody>'
                 + "".join(
                     f'<tr><td><code>{e(x["src"])} → {e(x["dst"])}</code></td>'
                     f'<td>{e(x["kind"])}</td>'
                     + "".join(f'<td class="v">{num(x["ratio"].get(k), 2)}</td>'
                               for k in ("p25", "p50", "p75"))
                     + f'<td class="v">{num(x.get("same_sign_pct"), 1, "%")}</td>'
                     f'<td class="v">{num(x["n"])}</td></tr>' for x in edges)
                 + "</tbody></table></div>")
        body += ('<p class="note">内訳エッジ（部分と全体）は恒等式なので、'
                 "比率が想定どおりなのは当たり前で、検証にはならない。"
                 "意味があるのは<b>計算書をまたぐ対応</b>のほう"
                 "（例: 当期純利益 → 営業CF）で、ここの比率と符号の一致率が"
                 "企業ごとにどれだけばらつくかが、グラフにする値打ちに直結する。</p>")
    return section("05", "グラフ構造の診断", body,
                   "組んだグラフが本当に会計の型を捉えているかを見る。"
                   "恒等式で必ず成立する部分と、実データで初めて分かる部分を分ける。")


# --------------------------------------------------------------------------- #

def build(d: Dict) -> str:
    lab = d.get("label", {})
    h = str(lab.get("primary_horizon", 20))
    tx = lab.get("horizons", {}).get(h, {}).get("topix", {})
    av = d["availability"]

    facts = [
        ("サンプル", num(d["n_samples"]), "決算開示 × 銘柄"),
        ("銘柄", num(d["n_codes"]), ""),
        ("期間", f'{d["period"][0]}<br>{d["period"][1]}', ""),
        ("8期そろい", num(av.get("full_seq_pct"), 1, "%"), "過去8四半期"),
        ("上昇クラス", num((tx.get("share") or {}).get("上昇"), 1, "%"),
         f"{h}営業日 / TOPIX控除"),
    ]
    head = ('<div class="facts">' + "".join(
        f'<div class="fact"><div class="eyebrow">{e(k)}</div>'
        f'<div class="n">{v}</div><div class="sub">{e(s)}</div></div>'
        for k, v, s in facts) + "</div>")

    secs = [sec_availability(d), sec_label(d), sec_features(d),
            sec_ic(d), sec_structure(d)]

    css = open(CSS, encoding="utf-8").read()
    return (f"<title>会計フローグラフ EDA</title>{FONTS}"
            f"<style>{css}{EXTRA_CSS}</style>"
            '<div class="wrap"><header>'
            '<p class="eyebrow">accounting graph</p>'
            "<h1>会計フローグラフ 探索的データ解析</h1>"
            '<p class="lede">決算を勘定科目のグラフとして表し、'
            "決算発表後の超過リターンを3クラスに分類する研究の前段。"
            "モデルを組む前に、何が取れて何が取れないか、"
            "目的変数がどこで偏るか、材料がそもそもあるのかを見る。</p>"
            f"{head}</header>{''.join(secs)}"
            '<footer class="note">データ: J-Quants API V2（株式会社JPX総研）／'
            "集計は research/accgraph/eda.py、組版は research/accgraph/eda_report.py。"
            f'<br>集計日時 {e(d.get("generated_at", ""))}'
            f'{"（流動性フィルタあり）" if d.get("liquid_only") else "（全銘柄）"}'
            "</footer></div>")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="会計フローグラフの EDA ページを組む")
    ap.add_argument("--eda", default=IN_JSON)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    with open(args.eda, encoding="utf-8") as fh:
        d = json.load(fh)
    out = build(d)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"[report] {args.out} ({len(out)/1000:.1f}KB / "
          f"{d['n_samples']:,}サンプル)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
