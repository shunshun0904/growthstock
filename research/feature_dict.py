#!/usr/bin/env python3
"""
特徴量の辞書ページを作る。大区分・中区分・列名・意味・欠損率を並べる。

説明は features.py の構成から組み立てる。決算系は
「軸 × 接尾辞」で機械的に決まるので、軸と接尾辞の意味だけを書けば
全列の説明が揃う。ここを1列ずつ手書きにすると、列を足したときに
説明だけ古くなる。

  python3 research/feature_dict.py
  -> research/feature_dict.html
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
import build_dataset as B  # noqa: E402

CSS = os.path.join(HERE, "viewer", "eda_style.css")
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Shippori+Mincho:wght@600;700&family=Noto+Sans+JP:wght@400;500;700'
         '&family=JetBrains+Mono:wght@400;600&display=swap">')

e = html.escape

#: 大区分。中区分（features.GROUPS のキー）をここに束ねる。
MACRO: List[tuple] = [
    ("決算（水準）", ["fund_level", "fund_lag"],
     "決算のその時点の値と、1〜3期前の値。業種や事業モデルで水準が決まる部分が大きい"),
    ("決算（変化）", ["fund_growth", "fund_trend", "fund_streak", "turnaround",
                      "progress", "guidance", "progress_seasonal", "progress_seasonal_pct"],
     "前期との差分・加速・連続性。「良くなっているか」を測る。今回の主役"),
    ("バリュエーション・配当", ["valuation", "dividend", "cashflow", "efficiency"],
     "株価と決算の比。割高・割安と、利益の質"),
    ("株価・ブレイク", ["price", "breakout"],
     "高値からの位置と、その日の抜け方"),
    ("需給・流動性・市場", ["volume", "liquidity", "supply", "market", "sector"],
     "出来高・売買代金・信用・相場環境・業種"),
    ("業種指数", ["sector_index"],
     "その銘柄の業種の指数と、そこからの相対。市場環境と違い日付内で銘柄ごとに変わる"),
    ("母集団のフラグ", ["scale"],
     "規模の帯と、決算がそろっているか。母集団の制約を外したぶん、"
     "どちらの群の行なのかを明示する"),
    ("予想（会社/市場の見通し）", ["fwd"],
     "/equities/valuation の FwdEPS・FwdPER・FwdROE と、実績との差。"
     "実績ではなく『これから』を指す唯一の塊"),
    ("株主構成", ["holders_lvs", "holders_major", "holders_cross"],
     "大量保有報告の提出と保有比率の増減、大株主10名の集中度と主体、"
     "政策保有の残高と売却。誰が持っていて、いま買い増しているか"),
    ("市場全体の需給", ["flow", "margin_alert", "flow_mkt"],
     "空売り比率、投資部門別（外国人・投信・個人・自己）の売買差引、信用規制。"
     "地合い11列は指数のリターンだけなので、主体別の需給は別の軸"),
    ("空売りの持ち高", ["short_pos"],
     "空売り残高報告（発行済みの0.5%以上の空売り）。報告者ごとの最新の持ち高の合計・"
     "人数・増減と、最後の報告からの日数。信用残（credit_ratio）より直接的な売り方の建玉"),
    ("予想の修正イベント", ["revision"],
     "業績予想・配当予想の修正がいつ何回あったか、上方か下方か。"
     "guidance_revision が修正の『幅』なのに対し、こちらは『発生』"),
    ("次の決算まで", ["earn_ahead"],
     "公表済みの予定から、次の決算発表まで何日か。開示のタイミングの裏返し"),
    ("開示のタイミング", ["timing"],
     "直近の決算開示から何日経ったか。四半期の周期のどこでブレイクしたか"),
]

#: 決算の軸。名前だけでは何の比率か分からないので明示する。
AXIS_JA = {
    "eps_growth": "EPS成長率（前年同期比・%）",
    "sales_growth": "売上成長率（前年同期比・%）",
    "eps_growth_sym": "EPS成長率（対称版）",
    "sales_growth_sym": "売上成長率（対称版）",
    "ROE": "自己資本利益率（%）",
    "ROA": "総資産利益率（%）",
    "op_margin": "営業利益率（%）",
    "equity_ratio": "自己資本比率（%）",
}
AXIS_NOTE = {
    "eps_growth_sym": "前年が0以下だと通常の成長率は定義できず欠測になる。"
                      "赤字企業が丸ごと落ちるのを避けるための定義",
    "sales_growth_sym": "同上",
}

#: 接尾辞。決算の軸すべてに同じ意味で付く。
SUFFIX_JA = {
    "q0": "直近決算の値",
    "q1": "1期前の値",
    "q2": "2期前の値",
    "q3": "3期前の値",
    "chg1": "前期 → 今期 の差分",
    "chg2": "2期前 → 前期 の差分",
    "chg3": "3期前 → 2期前 の差分",
    "chg": "2期分の変化（今期 − 2期前）",
    "chg_3q": "3期分の変化（今期 − 3期前）",
    "accel": "変化の加速（chg1 − chg2）。伸びが加速しているか",
    "up_streak": "直近から何期連続で増えているか（0〜3）",
    "pos_ratio": "有効な4期のうち、値がプラスだった割合（0〜1）",
}

#: 決算以外の列。個別に書く。
COL_JA = {
    "r_high": "78週高値に対する終値の位置（%）。母集団が高値更新日なので全件ほぼ100",
    "r_high_3m": "3ヶ月前の同じ指標。どこから上がってきたか",
    "r_high_6m": "6ヶ月前の同じ指標",
    "base_length": "前回の高値更新から何営業日空いたか。長い保ち合いからの初回ほど強い、という仮説",
    "break_margin": "それまでの78週高値をどれだけ上回ったか（%）",
    "close_position": "その日の値幅の中で終値がどこにあるか（0=安値引け, 100=高値引け）",
    "ret_20d": "直近20営業日の上昇率（%）。すでに走った後か、静かなところからの初動か",
    "vol_20d": "直近20営業日の日次リターン標準偏差（%）。値動きの荒さ",
    "volume_trend": "その日の出来高 ÷ 直前20日平均（%）",
    "log_trading_value": "20日平均売買代金の対数。流動性",
    "log_market_cap": "時価総額の対数。規模",
    "credit_ratio": "信用買残 ÷ 信用売残。需給の重さ",
    "days_since_disc": "直近の決算開示（実績値のあるもの）から何日経ったか（暦日、上限400）。開示直後と次の開示の直前が良く、その間が悪い",
    "days_since_fy": "直近の通期決算の開示から何日経ったか（暦日、上限800）",

    # --- 会社/市場の見通し（/equities/valuation）------------------------ #
    "jq_fwdeps": "API が出す予想EPS（円）。会社予想にもとづく1株利益",
    "jq_fwdper": "API が出す予想PER（倍）。実績PER より先を見ている",
    "jq_fwdroe": "予想ROE（%）",
    "jq_fwd_earnings_yield": "予想EPS ÷ 株価（%）。予想PER の逆数にあたり、PER が発散しても安定する",
    "jq_roe_gap": "予想ROE − 実績ROE（pt）。会社が改善を見込んでいるか",
    "jq_per_gap": "実績PER ÷ 予想PER（倍。**差ではなく比**）。1より大きいほど、これからの増益が織り込まれている",

    # --- 株主構成: 大量保有報告書（EDINET）------------------------------ #
    "lvs_days": "直近の大量保有報告書の提出から何日経ったか",
    "lvs_ratio": "直近の報告書が示す保有割合（%）",
    "lvs_ratio_chg": "前回の報告書からの保有割合の変化（pt）。買い増しか売り抜けか",
    "lvs_n_20": "直近20**暦日**の提出件数。短期に集まると動きがある",
    "lvs_n_60": "直近60**暦日**の提出件数",
    "lvs_n_250": "直近250**暦日**（約8ヶ月）の提出件数",

    # --- 株主構成: 大株主（EDINET、年1回）------------------------------ #
    "mjr_days": "直近の大株主情報の提出から何日経ったか",
    "mjr_top1": "筆頭株主の保有割合（%）",
    "mjr_top10": "上位10名の保有割合の合計（%）。浮動株の少なさの代理",
    "mjr_n": "保有割合が取れた大株主の人数（最大10）",
    "mjr_conc": "筆頭株主の割合 ÷ 上位10名の合計。1位が突出しているか横並びか",
    "mjr_trust": "大株主のうち信託銀行・カストディ名義の保有割合の合計（%）。機関投資家の粗い代理",
    "mjr_indiv": "大株主のうち法人名に見えない先の保有割合の合計（%）。個人・創業家の粗い代理",

    # --- 株主構成: 政策保有株（EDINET、年1回）-------------------------- #
    "xh_days": "直近の政策保有株の開示から何日経ったか",
    "xh_iss": "政策保有している上場株の銘柄数（ListedIss）",
    "xh_bookval": "政策保有する上場株の貸借対照表計上額（円。ListedBookVal）",
    "xh_dec_amt": "当期に売却して減らした額（円。ListedDecSaleAmt）。持ち合い解消",
    "xh_inc_cost": "当期に取得して増やした額（円。ListedIncAcqCost）",
    "xh_net": "増やした額 − 減らした額（円）。負なら解消が進んでいる",
    "xh_bookval_mc": "政策保有の計上額 ÷ 時価総額（%）。規模で割って比べられる形",
    "xh_net_mc": "政策保有の増減 ÷ 時価総額（%）",

    # --- 信用取引の規制・残高警報（/markets/margin-alert）--------------- #
    "alert_days": "直近の規制・警報の公表から何日経ったか",
    "alert_longoutratio": "公表時点の信用買残の比率（API の LongOutRatio）",
    "alert_shrtoutratio": "公表時点の信用売残の比率（API の ShrtOutRatio）",
    "alert_slratio": "売残 ÷ 買残（API の SLRatio）。踏み上げの余地",

    # --- 次の決算まで（/fins/earnings-date）---------------------------- #
    "days_to_earn": "次の決算発表予定日まで何日か（0〜120で頭打ち）。**その日までに公表済みの予定だけ**を使う",

    # --- 市場全体の需給（/markets/short-ratio, /equities/investor-types）- #
    "short_ratio": "S33=9999（**「その他」業種**）の空売り比率（%）。市場全体ではない（2026-09-24 に判明。取り込みが1日34行を1行に潰しており、残った行を市場全体と取り違えていた）。同じ日はどの銘柄でも同じ値",
    "short_ratio_20": "同じものの20営業日移動平均からの乖離（pt）",
    "short_ratio_mkt": "**市場全体**の空売り比率（%）。33業種とその他の全行の合計から作る。"
                       "業種が揃っていない日は使わない。同じ日はどの銘柄でも同じ値",
    "short_ratio_mkt_20": "同じものの20営業日移動平均からの乖離（pt）",
    "ss_ratio": "空売り残高報告で、報告者ごとの最新の持ち高（0.5%以上）の合計（発行済みに対する%）。"
                "報告は公表日の翌営業日から使い、365日より古い報告と0.5%を下回った報告は数えない",
    "ss_n": "その持ち高を持つ報告者の数",
    "ss_chg_20": "ss_ratio の28暦日（約20営業日）前との差（pt）",
    "ss_days": "どの報告者でも、最後の報告が使えるようになってからの暦日数（上限730。報告が無ければ欠測）",
    "inv_foreign": "**市場全体**の投資部門別で、外国人の差引（買い−売り）が5主体の売買代金合計に占める比（%）。同じ週はどの銘柄でも同じ値",
    "inv_foreign_4w": "同じものの4週合計",
    "inv_trust": "同じ形で、投資信託の差引の比（%）",
    "inv_trust_4w": "同じものの4週合計",
    "inv_indiv": "同じ形で、個人の差引の比（%）",
    "inv_indiv_4w": "同じものの4週合計",
    "inv_prop": "同じ形で、証券会社の自己売買の差引の比（%）",
    "inv_prop_4w": "同じものの4週合計",
    "inv_busco": "同じ形で、事業法人の差引の比（%）。自社株買いを含む",
    "inv_busco_4w": "同じものの4週合計",

    # --- 予想の修正イベント（fins の DocType から作る）------------------ #
    "days_since_rev": "直近の業績予想の修正から何日経ったか",
    "rev_pct": "直近の修正の幅（%）。同じ事業年度の**営業利益予想（FOP）**を直前の開示と比べる。上方修正なら正",
    "rev_up": "直近の修正が上方なら1、下方なら0",
    "rev_n_60": "直近60**暦日**の業績予想の修正回数",
    "rev_n_250": "直近250**暦日**（約8ヶ月）の修正回数。修正が多い会社ほど見通しが動く",
    "rev_up_n_250": "直近250**暦日**（約8ヶ月）の上方修正の回数",
    "rev_dn_n_250": "直近250**暦日**（約8ヶ月）の下方修正の回数",
    "days_since_divrev": "直近の配当予想の修正から何日経ったか",
    "sector_ret_20": "その銘柄の業種指数の直近20営業日リターン（%）。業種の地合い",
    "sector_ret_120": "その銘柄の業種指数の直近120営業日リターン（%）",
    "rel_sector_20": "銘柄の20日リターン − 業種の20日リターン（pt）。業種内での相対的な強さ",
    "sector_vs_topix_20": "業種の20日リターン − TOPIX の20日リターン（pt）。その業種が市場に対して強いか",
    "cap_band": "時価総額の帯（0:〜100億 / 1:100〜300億 / 2:300〜1000億 / 3:1000〜3000億 / 4:3000億〜）",
    "fund_complete": "決算の変化（full4_sym の10列）がすべて作れる行か（0/1）",
    "topix_ret_20": "TOPIX の直近20営業日リターン（%）。相場環境",
    "topix_ret_120": "TOPIX の直近120営業日リターン（%）",
    "topix_vol_20": "TOPIX 日次リターンの直近20営業日標準偏差（%）。局面の荒さ",
    "topix_ma200_gap": "TOPIX が200日移動平均からどれだけ離れているか（%）。長期の傾き",
    "nk225_ret_20": "日経225の直近20営業日リターン（%）。ETF 13210 の調整後終値から算出",
    "nk225_ret_120": "日経225の直近120営業日リターン（%）",
    "gold_ret_20": "金価格の直近20営業日リターン（%）。ETF 15400（純金上場信託）から算出",
    "gold_ret_120": "金価格の直近120営業日リターン（%）",
    "growth250_ret_20": "東証グロース250の直近20営業日リターン（%）。ETF 25160 から算出",
    "growth250_ret_120": "東証グロース250の直近120営業日リターン（%）",
    "risk_off_20": "金 − TOPIX の20日リターン差（pt）。リスクオフの度合い",
    "progress_vs_base": "通期予想に対する累計営業利益の進捗率 − 期間経過率。予想の上振れ度",
    "progress_ratio": "進捗率 ÷ 前年同期の進捗（Q×12.5% 未満なら Q×12.5%、前年が無ければ Q×25%）。"
                      "1.0 が例年どおりのペース。画面の「進捗期待」と同じ比率",
    "progress_pct": "progress_ratio の、同じ四半期・同じ物差しの過去の開示（その日より前）の中での"
                    "順位（0〜100）。四半期ごとの幅の違いをそろえる",
    "per": "株価収益率（株価 ÷ 1株利益TTM）",
    "pbr": "株価純資産倍率（株価 ÷ BPS）",
    "psr": "株価売上高倍率（時価総額 ÷ 売上TTM）",
    "peg": "PER ÷ EPS成長率。成長を織り込んだ割高度",
    "earnings_yield": "PER の逆数（%）。1÷PER なので PER が発散しても安定する",
    "book_yield": "PBR の逆数",
    "sales_yield": "PSR の逆数（%）",
    "div_yield": "配当利回り（%）",
    "has_dividend": "配当を出しているか（0/1）",
    "payout_ratio": "配当性向（%）",
    "cfo_yield": "営業CF ÷ 時価総額（%）",
    "fcf_yield": "フリーCF ÷ 時価総額（%）",
    "cfo_to_op": "営業CF ÷ 営業利益（%）。利益が現金を伴っているか",
    "accruals": "（利益 − 営業CF）÷ 総資産。会計上の利益と現金のずれ",
    "net_margin": "純利益率（%）",
    "ordinary_margin": "経常利益率（%）",
    "asset_turnover": "総資産回転率（売上 ÷ 総資産）",
    "guidance_op_growth": "会社予想営業利益の前年比（%）",
    "guidance_revision": "会社予想営業利益の、前回予想からの修正率（%）",
    "s33_code": "東証33業種コード（時点別）",
    "s17_code": "東証17業種コード（時点別）",
    "mkt_code": "市場区分コード（時点別）。2022年4月の再編を跨ぐので時点別が必須",
    "scalecat_code": "規模区分（TOPIX Core30 など）を符号化したもの",
    "eps_growth_turn": "EPS が赤字から黒字に転じたか（0/1）",
    "sales_growth_turn": "売上成長率がマイナスからプラスに転じたか（0/1）",
}


def describe(col: str) -> str:
    """列名から説明を組み立てる。決算系は 軸 × 接尾辞 で決まる。"""
    if col in COL_JA:
        return COL_JA[col]
    for axis in sorted(AXIS_JA, key=len, reverse=True):
        if col.startswith(axis + "_"):
            suf = col[len(axis) + 1:]
            if suf in SUFFIX_JA:
                note = AXIS_NOTE.get(axis)
                out = f"{AXIS_JA[axis]} の{SUFFIX_JA[suf]}"
                return out + (f"（{note}）" if note and suf == "q0" else "")
    return "（説明未登録）"


def build(stats: Optional[Dict]) -> str:
    used = {g for _, gs, _ in MACRO for g in gs}
    known = {g for g in F.GROUPS if not g.endswith("_rank")}
    missing = sorted(known - used)
    if missing:
        raise SystemExit(f"[fatal] 大区分に割り当てられていない中区分: {missing}")

    n_raw = len(F.columns("all"))
    facts = [
        ("レコードの粒度", "銘柄 × 日", "78週高値を更新した日だけ"),
        ("特徴量（絶対値）", str(n_raw), "順位版を含めて {}".format(len(F.all_columns()))),
        ("大区分", str(len(MACRO)), ""),
        ("中区分", str(len(known)), "features.GROUPS"),
        ("決算の要求", B.FUND_REQUIREMENT, "変化が作れない行は母集団から外す"),
    ]
    head = ('<div class="facts">' + "".join(
        f'<div class="fact"><div class="eyebrow">{e(k)}</div>'
        f'<div class="n">{e(v)}</div><div class="sub">{e(s)}</div></div>'
        for k, v, s in facts) + "</div>")

    secs = []
    for i, (ja, groups, note) in enumerate(MACRO, 1):
        rows = []
        for g in groups:
            cols = F.GROUPS[g]
            for j, c in enumerate(cols):
                miss = ""
                if stats and c in stats:
                    m = stats[c]["missing_pct"]
                    cls = "t-bad" if m >= 40 else "t-warn" if m >= 10 else "t-good"
                    miss = f'<span class="tag {cls}">{m:.1f}%</span>'
                gcell = (f'<td rowspan="{len(cols)}"><code>{e(g)}</code></td>'
                         if j == 0 else "")
                rows.append(f"<tr>{gcell}<td><code>{e(c)}</code></td>"
                            f"<td>{e(describe(c))}</td><td class='v'>{miss}</td></tr>")
        secs.append(
            f'<section><h2><span class="num">{i:02d}</span>{e(ja)}</h2>'
            f'<p class="lede">{e(note)}</p>'
            '<div class="scroll"><table class="dict">'
            '<colgroup><col style="width:13%"><col style="width:21%">'
            '<col style="width:58%"><col style="width:8%"></colgroup>'
            '<thead><tr><th>中区分</th><th>列</th>'
            '<th>意味</th><th>欠損</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div></section>")

    # 順位版の説明
    ranked = sorted(g for g in F.GROUPS if g.endswith("_rank"))
    secs.append(
        f'<section><h2><span class="num">{len(MACRO)+1:02d}</span>順位版（接尾辞 <code>_r</code>）</h2>'
        '<p class="lede">同じ列を「その日の全銘柄の中でのパーセンタイル順位」に'
        '置き換えたもの。PER 15倍ではなく「その日で下から30%の位置」。'
        '相場局面で水準が動く指標を、局面に依らない量にするため。</p>'
        f'<p class="note">対象の中区分: '
        + "、".join(f"<code>{e(g[:-5])}</code>" for g in ranked)
        + "。母集団を絞ったあとに付け直す（順位は母集団の中での位置なので）。</p></section>")

    presets = "".join(
        f"<tr><td><code>{e(p)}</code></td><td class='v'>{len(F.columns(p))}</td>"
        f"<td>{'、'.join(f'<code>{e(g)}</code>' for g in F.PRESETS[p])}</td></tr>"
        for p in sorted(F.PRESETS))
    secs.append(
        f'<section><h2><span class="num">{len(MACRO)+2:02d}</span>特徴量セット</h2>'
        '<p class="lede">学習時はこの単位で選ぶ。データセットには全列を作っておき、'
        'どれを使うかだけを切り替える。</p>'
        '<div class="scroll"><table><thead><tr><th>セット</th><th>列数</th>'
        f'<th>中区分</th></tr></thead><tbody>{presets}</tbody></table></div></section>')

    css = open(CSS, encoding="utf-8").read()
    return (f"<title>ブレイク予測 特徴量辞書</title>{FONTS}<style>{css}"
            "table.dict{table-layout:fixed}"
            "table.dict td{vertical-align:top}"
            "table.dict code{word-break:break-all}"
            "</style>"
            '<div class="wrap"><header><h1>ブレイク予測 特徴量辞書</h1>'
            '<p class="lede">1レコード = <b>銘柄 × 日</b>。'
            '78週高値を更新した日にだけレコードが立つ。'
            'その日の引け時点で分かる情報だけを特徴量にし、'
            '先60営業日の値動きでラベルを付ける。</p>'
            f"{head}</header>{''.join(secs)}"
            '<footer class="note">列の一覧は research/features.py、'
            "説明の組み立ては research/feature_dict.py。"
            "欠損率は research/eda.json（現在のデータセットでの実測）。</footer></div>")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="特徴量辞書のページを作る")
    ap.add_argument("--eda", default=os.path.join(HERE, "eda.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "feature_dict.html"))
    args = ap.parse_args(argv)

    stats = None
    if os.path.exists(args.eda):
        with open(args.eda, encoding="utf-8") as fh:
            stats = json.load(fh).get("stats")
    out = build(stats)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"[done] {args.out} ({len(out)/1000:.1f}KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
