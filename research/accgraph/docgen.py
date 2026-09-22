#!/usr/bin/env python3
"""
docs/ACCOUNTING_GRAPH.md を生成する。

スキーマを手で文書に書き写すと必ずずれるので、ノード表・エッジ表・図は
research/accgraph/schema.py から作る。散文だけがここにある。

  python3 research/accgraph/docgen.py          # 書き出す
  python3 research/accgraph/docgen.py --check   # 差分があれば失敗する
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
ROOT = os.path.dirname(RESEARCH)
sys.path.insert(0, RESEARCH)

from accgraph import panel, schema  # noqa: E402
from accgraph.labels import DEFAULT_LABEL  # noqa: E402

OUT = os.path.join(ROOT, "docs", "ACCOUNTING_GRAPH.md")

STATEMENT_JA = {"PL": "損益計算書", "BS": "貸借対照表", "CF": "キャッシュフロー計算書"}
KIND_JA = {"flow": "フロー", "composition": "内訳", "link": "計算書をまたぐ対応"}


def _node_table() -> List[str]:
    lines = [
        "| ノード | 名称 | 計算書 | 更新 | 種別 | 出所 | 元の項目 / 計算式 "
        "| 正規化の分母 | 共通概念 |",
        "| --- | --- | :-: | :-: | :-: | :-: | --- | :-: | --- |",
    ]
    for n in schema.NODES:
        if n.source == "disclosed":
            origin = f"`{n.field}`"
        else:
            plus, minus = n.derive
            expr = " + ".join(f"`{f}`" for f in plus)
            if minus:
                expr += " − " + " − ".join(f"`{f}`" for f in minus)
            origin = expr
        scale = "売上高" if n.scale_by == schema.SCALE_SALES else "総資産"
        kind = "期末残高" if n.kind == "stock" else "期間フロー"
        src = "開示値" if n.source == "disclosed" else "計算値"
        cadence = "年1回" if n.source_table == "edinet" else "四半期"
        lines.append(f"| `{n.id}` | {n.name_ja} | {n.statement} | {cadence} "
                     f"| {kind} | {src} | {origin} | {scale} | `{n.level1}` |")
    return lines


def _edge_table() -> List[str]:
    lines = ["| from | to | 種別 | 意味 |", "| --- | --- | :-: | --- |"]
    name = {n.id: n.name_ja for n in schema.NODES}
    for e in schema.EDGES:
        lines.append(f"| `{e.src}` ({name[e.src]}) | `{e.dst}` ({name[e.dst]}) "
                     f"| {KIND_JA[e.kind]} | {e.note or '—'} |")
    return lines


def _mermaid() -> List[str]:
    style = {"flow": "-->", "composition": "-.->", "link": "==>"}
    lines = ["```mermaid", "graph LR"]
    for st in ("PL", "BS", "CF"):
        lines.append(f"  subgraph {st}[{STATEMENT_JA[st]}]")
        for n in schema.NODES:
            if n.statement == st:
                lines.append(f"    {n.id}[\"{n.name_ja}\"]")
        lines.append("  end")
    for e in schema.EDGES:
        lines.append(f"  {e.src} {style[e.kind]} {e.dst}")
    lines.append("```")
    return lines


def render() -> str:
    cfg = DEFAULT_LABEL
    lines = [
        "# 会計フローグラフのスキーマ",
        "",
        "`research/accgraph/docgen.py` が `research/accgraph/schema.py` から生成する。",
        "手で書き換えないこと（次の生成で消える）。",
        "",
        "## 1. 前提 — 契約で取れるものだけで組む",
        "",
        "サンキー図の画像は解析しない。決算数値から直接グラフを作る。",
        "ただし、作れる粒度は J-Quants の契約が決めている。",
        "",
        "| エンドポイント | 実測 | 影響 |",
        "| --- | :-: | --- |",
        "| `/fins/summary` | OK（111項目） | PL・BS の集計値と CF 3区分が取れる |",
        "| `/fins/details` | **HTTP 403** | 減価償却費・運転資本・売上原価などの内訳が取れない |",
        "| `/fins/fs_details` | **HTTP 403** | 同上 |",
        "| EDINET DB `/companies/{code}/financials` | OK（128項目） "
        "| 上の内訳が取れる。ただし**年1回**（有価証券報告書） |",
        "",
        "実測は `docs/DATA_FIELDS.md`（`research/probe_fins_fields.py` の出力）と",
        "`docs/DATA_EDINETDB.md`（`research/probe_edinetdb.py` の出力）。",
        "",
        "四半期のグラフは J-Quants の集計値だけで組む。取れない項目を推定で埋めず、",
        "",
        "1. 開示された集計値をノードにする（出所 = 開示値）",
        "2. 集計値どうしの差で必ず決まる残余をノードにする（出所 = 計算値）",
        "3. どちらであるかをノード特徴量 `is_disclosed` として持たせる",
        "",
        "という形にした。2 は推定ではなく恒等式なので、値そのものに不確かさは入らない。",
        "",
        "そのうえで、EDINET DB の有価証券報告書から**明細ノードを下位層として足す**。",
        "要件にあった「税引前利益 → 減価償却費 → 運転資本 → 営業CF」の鎖は、",
        "ここで初めて繋がる。粗いノード（売上原価＋販管費）は残したまま、",
        "その内訳として細かいノード（売上原価 / 販管費 / 研究開発費）をぶら下げる形なので、",
        "EDINET が取れていない会社では細かい側が欠測になるだけで、",
        "粗い側のグラフはそのまま成立する。",
        "",
        "### 年1回のものを四半期のグラフに混ぜるときの約束",
        "",
        "  1. 年次のフローは4で割って「1四半期あたり」に直す（`span` = 4）",
        "  2. 「その値が何年前の書類か」を `age_years` としてノード特徴量に持たせる",
        "  3. 基準日から400日より古い書類しか無ければ、明細は全部欠測にする",
        "",
        "粒度の違いをモデルから見える形にするのが目的で、隠して均すのが目的ではない。",
        "各四半期の開示日を基準に as-of で引くので、過去の期のグラフに",
        "その時点ではまだ出ていない有報が混ざることはない。",
        "",
        "## 2. キャッシュフローは半期でしか開示されない",
        "",
        "`CFO` / `CFI` / `CFF` / `CashEq` の開示率（`docs/DATA_FIELDS.md` の実測）",
        "",
        "| 四半期 | 1Q | 2Q | 3Q | 通期 |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| `CFO` | 9.9% | 76.0% | 8.2% | 88.8% |",
        "",
        "つまり大半の企業は CF を**半期ごとにしか出さない**。",
        "累計開示を前四半期との差で単期に展開すると、1Q・3Q の累計が無いせいで",
        "2Q も通期も差が取れず、CF ノードが全期間まるごと欠測になる。",
        "",
        "そこで「同じ会計年度内で、値を持つ直近の先行開示」との差を取り、",
        "対象期数で割って1四半期あたりに揃える。",
        "",
        "| 開示 | 先行して値を持つ開示 | 当期の値 | 対象期数 |",
        "| --- | --- | --- | :-: |",
        "| 2Q（1Qに CF 無し） | なし | 累計そのもの＝上期 | 2 |",
        "| 通期（3Qに CF 無し） | 2Q | 通期 − 上期 ＝ 下期 | 2 |",
        "| 2Q（1Qに CF あり） | 1Q | 2Q − 1Q | 1 |",
        "",
        "対象期数はノード特徴量 `span` として残すので、",
        "「半期を2で割った値」であることは失われない。",
        "1Q・3Q の CF は欠測のままとし、`is_missing` を立てて 0 で置く。",
        "0 埋めだけにすると「営業CFがゼロだった」と区別できない。",
        "",
        "## 3. ノード",
        "",
        f"{schema.N_NODES}ノード（開示値 "
        f"{sum(1 for n in schema.NODES if n.source == 'disclosed')} / 計算値 "
        f"{sum(1 for n in schema.NODES if n.source == 'derived')}）。",
        "",
    ]
    lines += _node_table()
    lines += [
        "",
        "### 階層的標準化",
        "",
        "要件の「共通概念 → 業種別概念 → 企業開示科目」を3層で持つ。",
        "現契約では企業ごとの XBRL タグに到達できないため、",
        "level3 は J-Quants が正規化した項目名になり、level2（業種別概念）は空になる。",
        "層そのものは今から持っておく。無理に統合して情報を失わないための器である。",
        "金融・保険は PL/BS の構造が違うので、このスキーマの対象外とする",
        "（フェーズ3で別スキーマにする）。",
        "",
        "## 4. エッジ",
        "",
        f"{schema.N_EDGES}エッジ。種別は",
        "**フロー**（金額が移動・変換される）、",
        "**内訳**（部分と全体）、",
        "**計算書をまたぐ対応**（金額が一致するとは限らない）の3つ。",
        "",
    ]
    lines += _edge_table()
    lines += [
        "",
        "本来 PL と CF を繋ぐのは「税引前利益 → 営業CF」だが、",
        "`/fins/summary` に税引前利益が無いため当期純利益を起点にしている。",
        "税金ぶんだけ本来の橋渡しとずれるので、このエッジはフローではなく",
        "「計算書をまたぐ対応」として扱う。",
        "",
        "## 5. 図",
        "",
        "実線 = フロー / 点線 = 内訳 / 太線 = 計算書をまたぐ対応",
        "",
    ]
    lines += _mermaid()
    lines += [
        "",
        "## 6. 特徴量",
        "",
        "### ノード（四半期ごとに変わる）",
        "",
        "| 名前 | 内容 |",
        "| --- | --- |",
        "| `scaled` | 金額 ÷ 分母（PL・CF は売上高、BS は総資産） |",
        "| `log_size` | 符号付き対数の規模（百万円） |",
        "| `to_sales` | 金額 ÷ 売上高 |",
        "| `to_assets` | 金額 ÷ 総資産 |",
        "| `yoy_sym` | 前年同期比（対称変化率 −1〜+1。前年が赤字でも定義できる） |",
        "| `qoq_sym` | 前四半期比（同上） |",
        "| `slope4` | 直近4期の `scaled` の傾き（1四半期あたり） |",
        "| `vol4` | 直近4期の `scaled` の標準偏差 |",
        "| `sign` | 符号 |",
        "| `span` | その金額が何四半期ぶんか（年次のものは4） |",
        "| `age_years` | その値が何年前の書類か（四半期のものは0） |",
        "| `fcst_gap` | 通期会社予想に対する進捗の乖離（累計÷予想 − 経過四半期÷4） |",
        "| `fcst_avail` | 上を計算できたか |",
        "| `is_missing` | その期のそのノードが欠測か |",
        "",
        "要件にあった「アナリスト予想との乖離」は、J-Quants にアナリスト予想が",
        "無いため**会社予想**で代用している。別物なので名前も `fcst_` にしてある。",
        "",
        "「一過性か継続項目か」は、集計値では特別損益を分離できないため、",
        "ノードごとの定数 `recurring` として持つ（`special_tax_net` が 0、",
        "`sales` や `op` が 1）。行ごとの判定ではない。",
        "",
        "### エッジ（四半期ごとに変わる）",
        "",
        "| 名前 | 内容 |",
        "| --- | --- |",
        "| `ratio` | \\|source\\| ÷ (\\|target\\| + ε) |",
        "| `same_sign` | 両端の符号が一致するか |",
        "| `both_present` | 両端とも欠測でないか |",
        "",
        "### 定数（サンプルによらない）",
        "",
        "ノード: " + " / ".join(f"`{c}`" for c in schema.NODE_CONSTANTS)
        + "（`is_annual` = 1 が EDINET 由来）",
        "",
        "エッジ: " + " / ".join(f"`{c}`" for c in schema.EDGE_CONSTANTS),
        "",
        "定数はテンソルに入れず、`schema.json` に1回だけ書き出す。",
        "全サンプルで同じ値なのでサンプル軸に複製する意味が無い。",
        "",
        "## 7. 時系列",
        "",
        f"直近 {panel.SEQ_LEN} 四半期を `T=0`（当該決算）から過去へ並べる。",
        "各期の値は、**その決算発表の時点で見えていた版**で引く",
        "（`research/accgraph/panel.py` の as-of 結合）。",
        "過去の期が後から訂正されていても、訂正が出る前の基準日には訂正前の値が入る。",
        "ここを最終訂正値で埋めると未来情報のリークになる。",
        "",
        "上場が浅くて期がそろわない場合は欠測のままにし、`period_mask` で示す。",
        "",
        "## 8. 目的変数",
        "",
        "- 決算発表日の**翌営業日の始値**でエントリー（発表時刻は使わない）",
        f"- {' / '.join(str(h) for h in cfg.horizons)} 営業日後の終値でエグジット",
        "- TOPIX（指数コード `0000`）控除と、業種指数控除の2種類を作る",
        f"- 3クラス: +{cfg.up_threshold * 100:.0f}%超 = 上昇 / "
        f"±{cfg.up_threshold * 100:.0f}%以内 = 中立 / "
        f"{cfg.down_threshold * 100:.0f}%未満 = 下落",
        "",
        "リターンは分割調整後の価格で測る。未調整だと分割日に偽のリターンが立つ。",
        "業種指数のコード対応は `docs/INDEX_MAPPING.md`（実測で同定したもの）。",
        "",
        "## 9. 出力",
        "",
        "```",
        "research/_data/accgraph/",
        "  accgraph_<年>.npz   node_feat / edge_feat / period_mask / anchor_id",
        "  meta.parquet        1サンプル1行。銘柄・日付・リターン・ラベル",
        "  schema.json         ノード・エッジ・定数・ラベル定義",
        "```",
        "",
        f"`node_feat` は `[n, {panel.SEQ_LEN}, {schema.N_NODES}, "
        f"{len(schema.NODE_FEATURES)}]`、",
        f"`edge_feat` は `[n, {panel.SEQ_LEN}, {schema.N_EDGES}, "
        f"{len(schema.EDGE_FEATURES)}]`。",
        "スキーマが全サンプルで同一なので、グラフ構造はサンプルごとに持たず",
        "`edge_index` 1本で足りる。PyTorch Geometric にはそのまま渡せる。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="生成結果と既存ファイルが一致するか確かめるだけ")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    text = render()
    if args.check:
        if not os.path.exists(args.out):
            print(f"[docgen] {args.out} がありません")
            return 1
        cur = open(args.out, encoding="utf-8").read()
        if cur != text:
            print(f"[docgen] {args.out} がスキーマと一致しません。"
                  "python3 research/accgraph/docgen.py で作り直してください")
            return 1
        print("[docgen] 一致")
        return 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[docgen] 書き出し {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
