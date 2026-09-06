#!/usr/bin/env python3
"""
学習したモデルの特徴量重要度を横棒グラフで出す。

research/_data/results.json（train_model.py の出力）を読み、
テスト PR-AUC が最も高かった特徴量セットの LightGBM 重要度を並べる。

  python3 research/importance_report.py
  -> research/importance_report.html

重要度は gain（その分割で減った損失の合計）を主に見る。
split（使われた回数）だけだと、値の種類が多い列がただ多く選ばれてしまう。
両方出して、片方だけ大きい列は解釈のときに疑えるようにする。

注意: 重要度は「モデルが何を使ったか」であって因果ではない。
相関の強い列どうしでは重要度が分け合われるため、
低いことは「効かない」の証拠にならない。
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
CSS = os.path.join(HERE, "viewer", "eda_style.css")

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Shippori+Mincho:wght@600;700&family=Noto+Sans+JP:wght@400;500;700'
         '&family=JetBrains+Mono:wght@400;600&display=swap">')

#: 特徴量グループを3つに束ねる。
#: 色で見分ける系列は3つまでにする（4つ目を足すと、この配色は
#: 全ペア比較の識別しきい値を割る。dataviz の検証スクリプトで確認済み）。
MACRO = [
    ("fund", "決算", {"fund_level", "fund_lag", "fund_trend", "fund_streak",
                      "turnaround", "efficiency", "progress", "cashflow",
                      "guidance"}),
    ("val", "バリュエーション・配当", {"valuation", "dividend"}),
    ("price", "株価・需給・市場", {"price", "breakout", "volume", "liquidity",
                                   "supply", "market", "sector"}),
]

TOP_N = 30

e = html.escape


def macro_of(col: str) -> str:
    """列名から束ねたグループを引く。順位版(_r)は元の列と同じ扱い。"""
    raw = col[:-2] if col.endswith("_r") else col
    for key, _, groups in MACRO:
        for g in groups:
            if raw in F.GROUPS.get(g, ()):
                return key
    return "price"


def bars(rows: List[Dict], width: int = 300) -> str:
    """
    横棒。ランキングなので横棒（縦棒だと長い列名が読めない）。

    値は棒の右に直接書く。色だけに意味を持たせない
    （凡例と列名の両方で識別できるようにする）。
    """
    if not rows:
        return ""
    top = max(r["value"] for r in rows) or 1.0
    out = []
    for r in rows:
        w = r["value"] / top * width
        out.append(
            f'<div class="bar-label"><code>{e(r["label"])}</code></div>'
            f'<div><svg width="{width}" height="13" role="img" '
            f'aria-label="{e(r["label"])} {r["value"]:.1f}">'
            f'<rect x="0" y="1.5" width="{max(w, 1.5):.1f}" height="10" rx="4" '
            f'fill="var(--s-{r["macro"]})"></rect></svg></div>'
            f'<div class="bar-value">{r["pct"]:.2f}%</div>'
            f'<div class="n">{e(r["note"])}</div>')
    return f'<div class="bars">{"".join(out)}</div>'


def legend() -> str:
    return ('<div class="legend">' + "".join(
        f'<span><i style="background:var(--s-{k});border:0"></i>{e(ja)}</span>'
        for k, ja, _ in MACRO) + "</div>")


def build(res: Dict) -> str:
    exps = res.get("experiments", [])
    scored = []
    for x in exps:
        if not x.get("importance"):
            continue
        rows = x.get("results", {}).get("test", [])
        lgb = next((r for r in rows if r["name"].startswith("LightGBM")), None)
        if lgb:
            scored.append((lgb["pr_auc"], lgb, x))
    if not scored:
        raise SystemExit("重要度を持つ実験がありません。先に train_model.py を実行してください")
    scored.sort(key=lambda t: -t[0])
    best_ap, best_metrics, best = scored[0]

    imp = best["importance"]
    total = sum(r["gain"] for r in imp) or 1.0

    # --- 上位N列 --- #
    top = [{"label": r["col"], "value": r["gain"],
            "pct": r["gain"] / total * 100, "macro": macro_of(r["col"]),
            "note": f'{r["split"]:,}回'} for r in imp[:TOP_N]]
    used = sum(1 for r in imp if r["split"] > 0)
    head_share = sum(r["gain"] for r in imp[:TOP_N]) / total * 100

    # --- グループ別 --- #
    agg: Dict[str, float] = {k: 0.0 for k, _, _ in MACRO}
    cnt: Dict[str, int] = {k: 0 for k, _, _ in MACRO}
    for r in imp:
        m = macro_of(r["col"])
        agg[m] += r["gain"]
        cnt[m] += 1
    grp = [{"label": ja, "value": agg[k], "pct": agg[k] / total * 100,
            "macro": k, "note": f"{cnt[k]}列"}
           for k, ja, _ in MACRO]
    grp.sort(key=lambda r: -r["value"])

    facts = [
        ("特徴量セット", e(best["preset"]), f'{best["n_features"]}列'),
        ("テスト PR-AUC", f'{best_ap:.4f}', f'正例率 {best_metrics["base_rate"]*100:.2f}%'),
        ("Lift@5%", f'{best_metrics["lift@5%"]:.2f}x', ""),
        ("実際に使われた列", f'{used}', f'{len(imp)}列中'),
        ("上位30列の寄与", f'{head_share:.1f}%', "gain 合計に占める割合"),
    ]
    head = ('<div class="facts">' + "".join(
        f'<div class="fact"><div class="eyebrow">{e(k)}</div>'
        f'<div class="n">{v}</div><div class="sub">{e(s)}</div></div>'
        for k, v, s in facts) + "</div>")

    others = "".join(
        f'<tr><td>{e(x["preset"])}</td><td class="v">{ap:.4f}</td>'
        f'<td class="v">{m["lift@5%"]:.2f}x</td>'
        f'<td class="v">{x["n_features"]}</td></tr>'
        for ap, m, x in scored[:10])

    css = open(CSS, encoding="utf-8").read()
    return (f"<title>ブレイク予測モデル 重要度</title>{FONTS}<style>{css}"
            # 系列色。ダークは同じ3色をダーク面用に踏み直したもの
            ":root{--s-fund:#2a78d6;--s-val:#eb6834;--s-price:#1baf7a}"
            '@media (prefers-color-scheme:dark){:root:not([data-theme="light"])'
            "{--s-fund:#3987e5;--s-val:#d95926;--s-price:#199e70}}"
            ':root[data-theme="dark"]'
            "{--s-fund:#3987e5;--s-val:#d95926;--s-price:#199e70}"
            ".legend{display:flex;flex-wrap:wrap;gap:16px;margin:14px 0 0;"
            "font-size:11.5px;color:var(--ink-2)}"
            ".legend i{display:inline-block;width:11px;height:11px;border-radius:3px;"
            "margin-right:6px;vertical-align:-1px}"
            "</style>"
            '<div class="wrap"><header>'
            "<h1>ブレイク予測モデル 重要度</h1>"
            '<p class="lede">モデルが実際に何を使って判断しているかを見る。'
            "gain（その分割で減った損失の合計）の大きい順。</p>"
            f"{head}</header>"
            + section("01", "グループ別の寄与", legend() + bars(grp),
                      "3つに束ねた寄与。どの軸で判断しているかの全体像。")
            + section("02", f"上位{TOP_N}列", legend() + bars(top),
                      "棒の右は gain 全体に占める割合、その右は分割に使われた回数。"
                      "回数だけ多くて割合が小さい列は、値の種類が多いだけの可能性がある。")
            + section("03", "特徴量セット別のテスト成績",
                      '<div class="scroll"><table><thead><tr><th>セット</th>'
                      '<th>PR-AUC</th><th>Lift@5%</th><th>列数</th></tr></thead>'
                      f"<tbody>{others}</tbody></table></div>",
                      "上の重要度は、この表の最上段（PR-AUC 最良）のモデルのもの。")
            + '<p class="note"><b>読むときの注意。</b>重要度は「モデルが何を使ったか」'
              "であって、因果でも「効く順」でもない。相関の強い列どうしでは重要度が"
              "分け合われるため、低いことは効かないことの証拠にならない。"
              "列を落とす判断は、落として測り直してから行う。</p>"
            '<footer class="note">データ: J-Quants API V2（日本取引所グループ）／'
            "重要度は research/train_model.py、組版は research/importance_report.py。"
            "</footer></div>")


def section(no: str, title: str, body: str, lede: str = "") -> str:
    led = f'<p class="lede">{lede}</p>' if lede else ""
    return (f'<section><h2><span class="num">{no}</span>{e(title)}</h2>'
            f"{led}{body}</section>")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="特徴量重要度のページを組み立てる")
    ap.add_argument("--results", default=os.path.join(DATA_DIR, "results.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "importance_report.html"))
    args = ap.parse_args(argv)

    with open(args.results, encoding="utf-8") as fh:
        res = json.load(fh)
    out = build(res)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"[done] {args.out} ({len(out)/1000:.1f}KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
