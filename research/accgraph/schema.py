#!/usr/bin/env python3
"""
会計フローグラフのスキーマ定義。

サンキー図の画像は一切見ない。決算数値から直接ノードとエッジを組む。

## 何を入れられるかは実測で決まっている

J-Quants の `/fins/details`（XBRL の詳細項目）は本契約では **HTTP 403**
（docs/DATA_FIELDS.md の実測）。したがって

  - 減価償却費、運転資本増減、売上原価、販管費、税引前利益

といった内訳は **取得できない**。使えるのは `/fins/summary` の集計値だけで、
キャッシュフローは営業・投資・財務の3区分と現金同等物期末残高しかない。

そこで「取れない項目を推測で作る」のではなく、

  1. 開示された集計値をノードにする（source="disclosed"）
  2. 集計値どうしの差で必ず決まる残余をノードにする（source="derived"）
  3. どちらであるかをノード特徴量 `is_disclosed` として必ず持たせる

という方針にした。2 は推定ではなく恒等式（売上高 - 営業利益 = 売上原価+販管費）
なので、値そのものに不確かさは入らない。粒度が粗いだけである。

## 階層的標準化

要件の「共通概念 → 業種別概念 → 企業開示科目」を3層で持つ。
現契約では企業ごとの開示科目（XBRL タグ）に到達できないため、
level3 は J-Quants が正規化した項目名になり、level2（業種別概念）は空になる。
`/fins/details` が使えるようになったときに level2/level3 を埋められるよう、
層そのものは今から持っておく。無理に統合して情報を失わないための器である。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# ノード
# --------------------------------------------------------------------------- #

#: 金額の正規化に使う分母。
#:   "sales"  … 当四半期の売上高（フロー項目）
#:   "assets" … 総資産（ストック項目）
#: 分母を固定するのは、企業規模の違いをグラフ間で吸収するため。
SCALE_SALES = "sales"
SCALE_ASSETS = "assets"


@dataclass(frozen=True)
class NodeSpec:
    """会計フローグラフの1ノード。"""

    id: str
    name_ja: str
    #: "PL" / "BS" / "CF"
    statement: str
    #: "flow"（期間中の金額） / "stock"（期末残高）
    kind: str
    #: "disclosed"（API が返す値） / "derived"（他の開示値の差で決まる値）
    source: str
    #: 元データの項目名。derived のときは None。
    #: どのテーブルの項目かは source_table が決める
    field: Optional[str]
    #: derived の計算式。(加算する field, 減算する field) の組
    derive: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]
    #: 金額の正規化に使う分母
    scale_by: str
    #: 通期の会社予想が存在する場合、その項目名。進捗乖離の算出に使う
    forecast_field: Optional[str]
    #: 継続項目らしさ。1.0 = 継続、0.0 = 一過性が混ざる
    #: （/fins/summary では特別損益を分離できないため、混在を 0.5 で表す）
    recurring: float
    #: 値の出所。"fins" = J-Quants /fins/summary（四半期）、
    #: "edinet" = EDINET DB の有価証券報告書（年1回）
    source_table: str = "fins"
    #: 階層的標準化。level1 = 共通概念、level2 = 業種別概念、level3 = 企業開示科目
    level1: str = ""
    level2: Optional[str] = None
    level3: Optional[str] = None
    note: str = ""


#: 累計値として開示され、単一四半期に差分展開する必要がある項目。
#: BS のストック項目（TA / Eq / ShEq / CashEq）は期末残高なので展開しない。
CUMULATIVE_FIELDS = ("Sales", "OP", "OdP", "NP", "CFO", "CFI", "CFF")

#: 期末残高としてそのまま使う項目。
STOCK_FIELDS = ("TA", "Eq", "ShEq", "CashEq")

# --------------------------------------------------------------------------- #
# EDINET DB（有価証券報告書）の項目
# --------------------------------------------------------------------------- #
#
# J-Quants の /fins/details は本契約では 403 で、減価償却費・運転資本・
# 売上原価といった明細が取れない（docs/DATA_FIELDS.md）。EDINET DB は
# それを持っている（docs/DATA_EDINETDB.md の実測）。
#
# ただし **年1回（有価証券報告書）** しか出ない。四半期のグラフに
# そのまま混ぜると、四半期ごとに動く値と年1回しか動かない値が
# 同じ顔で並ぶことになる。そこで
#
#   - 年次のフローは4で割って「1四半期あたり」に直す（span = 4）
#   - 「その値が何年前の書類か」を age_years としてノード特徴量に持たせる
#
# ことで、粒度の違いをモデルから見える形にする。

#: EDINET の期間フロー（1事業年度ぶんの金額）。
EDINET_FLOW_FIELDS = ("cost_of_sales", "sga", "rnd_expenses",
                      "profit_before_tax", "depreciation", "capex")
#: EDINET の期末残高。
EDINET_STOCK_FIELDS = ("inventories", "trade_receivables", "trade_payables",
                       "ibd_current", "ibd_noncurrent")

#: 年次フローを1四半期あたりに直すときの割り算。
ANNUAL_TO_QUARTER = 4.0


NODES: List[NodeSpec] = [
    # ----- PL ------------------------------------------------------------- #
    NodeSpec(
        id="sales", name_ja="売上高", statement="PL", kind="flow",
        source="disclosed", field="Sales", derive=None, scale_by=SCALE_SALES,
        forecast_field="FSales", recurring=1.0,
        level1="REVENUE", level3="Sales",
    ),
    NodeSpec(
        id="cogs_sga", name_ja="売上原価＋販管費", statement="PL", kind="flow",
        source="derived", field=None, derive=(("Sales",), ("OP",)),
        scale_by=SCALE_SALES, forecast_field=None, recurring=1.0,
        level1="OPERATING_COST",
        note="売上原価と販管費の内訳は /fins/details が 403 のため分離できない",
    ),
    NodeSpec(
        id="op", name_ja="営業利益", statement="PL", kind="flow",
        source="disclosed", field="OP", derive=None, scale_by=SCALE_SALES,
        forecast_field="FOP", recurring=1.0,
        level1="PROFIT_OPERATING", level3="OP",
    ),
    NodeSpec(
        id="non_op_net", name_ja="営業外損益（純額）", statement="PL", kind="flow",
        source="derived", field=None, derive=(("OdP",), ("OP",)),
        scale_by=SCALE_SALES, forecast_field=None, recurring=0.8,
        level1="NON_OPERATING_NET",
        note="受取利息・持分法損益・為替差損益などの純額",
    ),
    NodeSpec(
        id="ordinary_profit", name_ja="経常利益", statement="PL", kind="flow",
        source="disclosed", field="OdP", derive=None, scale_by=SCALE_SALES,
        forecast_field="FOdP", recurring=1.0,
        level1="PROFIT_ORDINARY", level3="OdP",
    ),
    NodeSpec(
        id="special_tax_net", name_ja="特別損益＋税金等（純額）", statement="PL",
        kind="flow", source="derived", field=None, derive=(("NP",), ("OdP",)),
        scale_by=SCALE_SALES, forecast_field=None, recurring=0.0,
        level1="SPECIAL_AND_TAX_NET",
        note="特別損益・法人税等・非支配株主持分がまとまって入る。一過性の代表",
    ),
    NodeSpec(
        id="net_income", name_ja="当期純利益", statement="PL", kind="flow",
        source="disclosed", field="NP", derive=None, scale_by=SCALE_SALES,
        forecast_field="FNP", recurring=0.7,
        level1="PROFIT_NET", level3="NP",
    ),

    # ----- BS ------------------------------------------------------------- #
    NodeSpec(
        id="total_assets", name_ja="総資産", statement="BS", kind="stock",
        source="disclosed", field="TA", derive=None, scale_by=SCALE_ASSETS,
        forecast_field=None, recurring=1.0,
        level1="ASSETS_TOTAL", level3="TA",
    ),
    NodeSpec(
        id="liabilities", name_ja="負債", statement="BS", kind="stock",
        source="derived", field=None, derive=(("TA",), ("Eq",)),
        scale_by=SCALE_ASSETS, forecast_field=None, recurring=1.0,
        level1="LIABILITIES_TOTAL",
    ),
    NodeSpec(
        id="equity", name_ja="純資産", statement="BS", kind="stock",
        source="disclosed", field="Eq", derive=None, scale_by=SCALE_ASSETS,
        forecast_field=None, recurring=1.0,
        level1="EQUITY_TOTAL", level3="Eq",
    ),
    NodeSpec(
        id="minority_etc", name_ja="非支配株主持分等", statement="BS", kind="stock",
        source="derived", field=None, derive=(("Eq",), ("ShEq",)),
        scale_by=SCALE_ASSETS, forecast_field=None, recurring=1.0,
        level1="MINORITY_INTEREST",
    ),
    NodeSpec(
        id="shareholders_equity", name_ja="自己資本", statement="BS", kind="stock",
        source="disclosed", field="ShEq", derive=None, scale_by=SCALE_ASSETS,
        forecast_field=None, recurring=1.0,
        level1="EQUITY_SHAREHOLDERS", level3="ShEq",
    ),
    NodeSpec(
        id="cash", name_ja="現金及び現金同等物", statement="BS", kind="stock",
        source="disclosed", field="CashEq", derive=None, scale_by=SCALE_ASSETS,
        forecast_field=None, recurring=1.0,
        level1="CASH", level3="CashEq",
        note="CF計算書に付随する項目なので、開示率は CF と同じ（1Q/3Q は約1割）",
    ),

    # ----- CF ------------------------------------------------------------- #
    NodeSpec(
        id="cfo", name_ja="営業CF", statement="CF", kind="flow",
        source="disclosed", field="CFO", derive=None, scale_by=SCALE_SALES,
        forecast_field=None, recurring=1.0,
        level1="CF_OPERATING", level3="CFO",
    ),
    NodeSpec(
        id="cfi", name_ja="投資CF", statement="CF", kind="flow",
        source="disclosed", field="CFI", derive=None, scale_by=SCALE_SALES,
        forecast_field=None, recurring=0.6,
        level1="CF_INVESTING", level3="CFI",
    ),
    NodeSpec(
        id="cff", name_ja="財務CF", statement="CF", kind="flow",
        source="disclosed", field="CFF", derive=None, scale_by=SCALE_SALES,
        forecast_field=None, recurring=0.6,
        level1="CF_FINANCING", level3="CFF",
    ),
    NodeSpec(
        id="fcf", name_ja="フリーCF", statement="CF", kind="flow",
        source="derived", field=None, derive=(("CFO", "CFI"), ()),
        scale_by=SCALE_SALES, forecast_field=None, recurring=1.0,
        level1="CF_FREE",
        note="営業CF + 投資CF。投資CFは通常マイナスなので実質は差し引き",
    ),
    NodeSpec(
        id="net_cash_chg", name_ja="現金増減額", statement="CF", kind="flow",
        source="derived", field=None, derive=(("CFO", "CFI", "CFF"), ()),
        scale_by=SCALE_SALES, forecast_field=None, recurring=1.0,
        level1="CF_NET_CHANGE",
        note="為替換算差額を含まない近似。差額は /fins/summary に無い",
    ),
]

# --------------------------------------------------------------------------- #
# EDINET DB から足すノード（年1回更新）
# --------------------------------------------------------------------------- #
#
# 要件にあった「税引前利益 → 減価償却費 → 運転資本 → 営業CF」という
# 粒度は、ここで初めて組める。ただし年1回なので、四半期のノードとは
# 更新頻度が違う。source_table と age_years でそれが分かるようにしてある。

NODES += [
    # ----- PL の明細 ----------------------------------------------------- #
    NodeSpec(
        id="cost_of_sales", name_ja="売上原価", statement="PL", kind="flow",
        source="disclosed", field="cost_of_sales", derive=None,
        scale_by=SCALE_SALES, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="OPERATING_COST", level2="COST_OF_SALES", level3="cost_of_sales",
        note="J-Quants では販管費と合算でしか取れない部分を分ける",
    ),
    NodeSpec(
        id="sga", name_ja="販売費及び一般管理費", statement="PL", kind="flow",
        source="disclosed", field="sga", derive=None,
        scale_by=SCALE_SALES, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="OPERATING_COST", level2="SGA", level3="sga",
    ),
    NodeSpec(
        id="rnd", name_ja="研究開発費", statement="PL", kind="flow",
        source="disclosed", field="rnd_expenses", derive=None,
        scale_by=SCALE_SALES, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="OPERATING_COST", level2="RND", level3="rnd_expenses",
        note="販管費の内数。業種によっては開示が無い",
    ),
    NodeSpec(
        id="pretax_profit", name_ja="税引前利益", statement="PL", kind="flow",
        source="disclosed", field="profit_before_tax", derive=None,
        scale_by=SCALE_SALES, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="PROFIT_PRETAX", level3="profit_before_tax",
        note="営業CFの本来の起点。J-Quants には無い",
    ),

    # ----- CF の明細 ----------------------------------------------------- #
    NodeSpec(
        id="depreciation", name_ja="減価償却費", statement="CF", kind="flow",
        source="disclosed", field="depreciation", derive=None,
        scale_by=SCALE_SALES, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="DEPRECIATION", level3="depreciation",
        note="非現金費用として営業CFに足し戻される",
    ),
    NodeSpec(
        id="capex", name_ja="設備投資", statement="CF", kind="flow",
        source="disclosed", field="capex", derive=None,
        scale_by=SCALE_SALES, forecast_field=None, recurring=0.8,
        source_table="edinet",
        level1="CAPEX", level3="capex",
    ),

    # ----- BS の明細 ----------------------------------------------------- #
    NodeSpec(
        id="inventories", name_ja="棚卸資産", statement="BS", kind="stock",
        source="disclosed", field="inventories", derive=None,
        scale_by=SCALE_ASSETS, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="INVENTORIES", level3="inventories",
    ),
    NodeSpec(
        id="trade_receivables", name_ja="売上債権", statement="BS", kind="stock",
        source="disclosed", field="trade_receivables", derive=None,
        scale_by=SCALE_ASSETS, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="TRADE_RECEIVABLES", level3="trade_receivables",
    ),
    NodeSpec(
        id="trade_payables", name_ja="仕入債務", statement="BS", kind="stock",
        source="disclosed", field="trade_payables", derive=None,
        scale_by=SCALE_ASSETS, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="TRADE_PAYABLES", level3="trade_payables",
    ),
    NodeSpec(
        id="working_capital", name_ja="運転資本", statement="BS", kind="stock",
        source="derived", field=None,
        derive=(("inventories", "trade_receivables"), ("trade_payables",)),
        scale_by=SCALE_ASSETS, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="WORKING_CAPITAL",
        note="棚卸資産 + 売上債権 − 仕入債務。増減が営業CFを削る",
    ),
    NodeSpec(
        id="interest_bearing_debt", name_ja="有利子負債", statement="BS",
        kind="stock", source="derived", field=None,
        derive=(("ibd_current", "ibd_noncurrent"), ()),
        scale_by=SCALE_ASSETS, forecast_field=None, recurring=1.0,
        source_table="edinet",
        level1="INTEREST_BEARING_DEBT",
        note="無借金の会社では項目ごと省かれる。欠測と0は区別できない",
    ),
]

NODE_IDS: List[str] = [n.id for n in NODES]
NODE_INDEX: Dict[str, int] = {n.id: i for i, n in enumerate(NODES)}
N_NODES = len(NODES)


# --------------------------------------------------------------------------- #
# エッジ
# --------------------------------------------------------------------------- #

#: エッジの種類。
#:   flow        … 金額がそのまま移動・変換される（会計上のフロー）
#:   composition … 部分と全体（内訳）
#:   link        … 計算書をまたぐ対応関係。金額が一致するとは限らない
EDGE_KINDS = ("flow", "composition", "link")


@dataclass(frozen=True)
class EdgeSpec:
    src: str
    dst: str
    kind: str
    note: str = ""


EDGES: List[EdgeSpec] = [
    # --- PL の中のフロー --- #
    EdgeSpec("sales", "cogs_sga", "flow", "売上から費用として出ていく"),
    EdgeSpec("sales", "op", "flow", "費用を引いた残りが営業利益"),
    EdgeSpec("op", "ordinary_profit", "flow", ""),
    EdgeSpec("non_op_net", "ordinary_profit", "flow", ""),
    EdgeSpec("ordinary_profit", "net_income", "flow", ""),
    EdgeSpec("special_tax_net", "net_income", "flow", ""),

    # --- PL → CF --- #
    # 本来は「税引前利益 → 営業CF」だが、/fins/summary に税引前利益が無い。
    # 純利益を起点にすると税金ぶんだけ本来の橋渡しとずれるので、
    # このエッジは flow ではなく link として扱う。
    EdgeSpec("net_income", "cfo", "link", "税引前利益が取れないため純利益を起点にした"),

    # --- CF の中のフロー --- #
    EdgeSpec("cfo", "fcf", "flow", ""),
    EdgeSpec("cfi", "fcf", "flow", ""),
    EdgeSpec("fcf", "net_cash_chg", "flow", ""),
    EdgeSpec("cff", "net_cash_chg", "flow", ""),
    EdgeSpec("net_cash_chg", "cash", "flow", "現金増減が現金残高を動かす"),

    # --- PL → BS --- #
    EdgeSpec("net_income", "shareholders_equity", "flow", "内部留保として自己資本に積まれる"),

    # --- BS の内訳 --- #
    EdgeSpec("shareholders_equity", "equity", "composition", ""),
    EdgeSpec("minority_etc", "equity", "composition", ""),
    EdgeSpec("equity", "total_assets", "composition", ""),
    EdgeSpec("liabilities", "total_assets", "composition", ""),
    EdgeSpec("cash", "total_assets", "composition", "現金は総資産の一部"),

    # --- 計算書をまたぐ対応 --- #
    EdgeSpec("total_assets", "sales", "link", "総資産回転（資産が売上を生む）"),
    EdgeSpec("cfi", "total_assets", "link", "投資支出が資産を増やす"),
    EdgeSpec("cff", "liabilities", "link", "財務CFが負債・資本を動かす"),
]

# --------------------------------------------------------------------------- #
# EDINET DB の明細が入って初めて引けるエッジ
# --------------------------------------------------------------------------- #
#
# 要件にあった「税引前利益 → 減価償却費 → 運転資本 → 営業CF」の鎖が
# ここで繋がる。粗いノード（売上原価＋販管費）は残したまま、その内訳として
# 細かいノードをぶら下げる形にしてある。EDINET が取れていない会社では
# 細かい側が欠測になるだけで、粗い側のグラフはそのまま成立する。

EDGES += [
    # --- 粗いノードの内訳 --- #
    EdgeSpec("cost_of_sales", "cogs_sga", "composition", "J-Quantsでは合算でしか取れない"),
    EdgeSpec("sga", "cogs_sga", "composition", ""),
    EdgeSpec("rnd", "sga", "composition", "販管費の内数"),

    # --- PL から CF への、本来の橋渡し --- #
    EdgeSpec("ordinary_profit", "pretax_profit", "flow", ""),
    EdgeSpec("pretax_profit", "net_income", "flow", "税金を引くと当期純利益"),
    EdgeSpec("pretax_profit", "cfo", "flow", "営業CFの本来の起点"),
    EdgeSpec("depreciation", "cfo", "flow", "非現金費用として足し戻す"),
    EdgeSpec("working_capital", "cfo", "flow", "運転資本が増えると営業CFは減る"),

    # --- 運転資本の内訳 --- #
    EdgeSpec("inventories", "working_capital", "composition", ""),
    EdgeSpec("trade_receivables", "working_capital", "composition", ""),
    EdgeSpec("trade_payables", "working_capital", "composition", "差し引く側"),

    # --- 投資と貸借 --- #
    EdgeSpec("capex", "cfi", "flow", "投資CFの主要な中身"),
    EdgeSpec("inventories", "total_assets", "composition", ""),
    EdgeSpec("trade_receivables", "total_assets", "composition", ""),
    EdgeSpec("trade_payables", "liabilities", "composition", ""),
    EdgeSpec("interest_bearing_debt", "liabilities", "composition", ""),
]

N_EDGES = len(EDGES)


def edge_index() -> List[List[int]]:
    """PyG 互換の edge_index（[2, n_edges]、行0=source、行1=target）。"""
    return [[NODE_INDEX[e.src] for e in EDGES],
            [NODE_INDEX[e.dst] for e in EDGES]]


# --------------------------------------------------------------------------- #
# 特徴量の名前（データセットの列順はここで一元管理する）
# --------------------------------------------------------------------------- #

#: サンプルごと・四半期ごとに変わるノード特徴量。
NODE_FEATURES: List[str] = [
    "scaled",        # 金額 / 分母（売上高 or 総資産）
    "log_size",      # sign(x) * log1p(|x| 百万円)。規模そのもの
    "to_sales",      # 金額 / 売上高
    "to_assets",     # 金額 / 総資産
    "yoy_sym",       # 前年同期比（対称変化率 -1〜+1）
    "qoq_sym",       # 前四半期比（対称変化率 -1〜+1）
    "slope4",        # 直近4期の scaled の傾き（1期あたり）
    "vol4",          # 直近4期の scaled の標準偏差
    "sign",          # 符号 (-1 / 0 / +1)
    "span",          # その金額が何四半期ぶんか（年次のものは4）
    "age_years",     # その値が何年前の書類か（四半期のものは0）（CFは2になることが多い）
    "fcst_gap",      # 通期会社予想に対する進捗の乖離
    "fcst_avail",    # 上を計算できたか
    "is_missing",    # 1 = この期のこのノードは欠測
]

#: ノードごとに固定で、サンプルによらない特徴量（スキーマ由来）。
NODE_CONSTANTS: List[str] = [
    "is_disclosed",  # 1 = API の開示値、0 = 恒等式による計算値
    "stmt_pl", "stmt_bs", "stmt_cf",
    "is_stock",      # 1 = 期末残高、0 = 期間フロー
    "recurring",     # 継続項目らしさ
    "is_annual",     # 1 = 年1回しか更新されない（EDINET 由来）
]

#: サンプルごと・四半期ごとに変わるエッジ特徴量。
EDGE_FEATURES: List[str] = [
    "ratio",         # |source| / (|target| + eps)。0〜1 に収める
    "same_sign",     # 両端の符号が一致するか
    "both_present",  # 両端とも欠測でないか
]

#: エッジごとに固定の特徴量。
EDGE_CONSTANTS: List[str] = ["kind_flow", "kind_composition", "kind_link"]

N_NODE_FEATURES = len(NODE_FEATURES)
N_EDGE_FEATURES = len(EDGE_FEATURES)


def node_constants() -> List[List[float]]:
    """NODE_CONSTANTS の並びで [n_nodes, n_constants] を返す。"""
    out = []
    for n in NODES:
        out.append([
            1.0 if n.source == "disclosed" else 0.0,
            1.0 if n.statement == "PL" else 0.0,
            1.0 if n.statement == "BS" else 0.0,
            1.0 if n.statement == "CF" else 0.0,
            1.0 if n.kind == "stock" else 0.0,
            float(n.recurring),
            1.0 if n.source_table == "edinet" else 0.0,
        ])
    return out


def edge_constants() -> List[List[float]]:
    """EDGE_CONSTANTS の並びで [n_edges, n_constants] を返す。"""
    out = []
    for e in EDGES:
        out.append([1.0 if e.kind == k else 0.0 for k in EDGE_KINDS])
    return out


# --------------------------------------------------------------------------- #
# 自己検査
# --------------------------------------------------------------------------- #

def validate() -> None:
    """スキーマ自身の整合を確かめる。import 時に必ず通す。"""
    ids = [n.id for n in NODES]
    assert len(ids) == len(set(ids)), f"ノードIDが重複している: {ids}"

    known = set(ids)
    for e in EDGES:
        assert e.src in known, f"エッジの source が未定義: {e.src}"
        assert e.dst in known, f"エッジの target が未定義: {e.dst}"
        assert e.src != e.dst, f"自己ループ: {e.src}"
        assert e.kind in EDGE_KINDS, f"未知のエッジ種別: {e.kind}"
    pairs = [(e.src, e.dst) for e in EDGES]
    assert len(pairs) == len(set(pairs)), "同じ (source, target) のエッジが重複している"

    for n in NODES:
        assert n.statement in ("PL", "BS", "CF"), n.id
        assert n.kind in ("flow", "stock"), n.id
        assert n.scale_by in (SCALE_SALES, SCALE_ASSETS), n.id
        if n.source == "disclosed":
            assert n.field, f"{n.id}: 開示値なのに項目名が無い"
            assert n.derive is None, f"{n.id}: 開示値なのに計算式がある"
        else:
            assert n.derive is not None, f"{n.id}: 計算値なのに計算式が無い"
            assert n.field is None, f"{n.id}: 計算値なのに項目名がある"
        # 使う項目は累計かストックのどちらかに分類されていること。
        # ここが漏れると、ストックを差分展開して残高を増減額に変えてしまう。
        used = [n.field] if n.field else []
        if n.derive:
            used = list(n.derive[0]) + list(n.derive[1])
        assert n.source_table in ("fins", "edinet"), n.id
        known_fields = (CUMULATIVE_FIELDS + STOCK_FIELDS if n.source_table == "fins"
                        else EDINET_FLOW_FIELDS + EDINET_STOCK_FIELDS)
        for f in used:
            assert f in known_fields, \
                f"{n.id}: {f} が {n.source_table} の累計/ストックのどちらにも分類されていない"

    # 孤立ノードを作らない。どこにも繋がっていないノードは、
    # GNN にとって「ただの数値」でありグラフにする意味が無い。
    connected = {e.src for e in EDGES} | {e.dst for e in EDGES}
    isolated = sorted(known - connected)
    assert not isolated, f"どのエッジにも繋がっていないノード: {isolated}"

    assert len(NODE_FEATURES) == len(set(NODE_FEATURES))
    assert len(EDGE_FEATURES) == len(set(EDGE_FEATURES))


validate()


def summary() -> str:
    lines = [f"ノード {N_NODES}件 / エッジ {N_EDGES}件",
             f"  開示値 {sum(1 for n in NODES if n.source == 'disclosed')}件 / "
             f"計算値 {sum(1 for n in NODES if n.source == 'derived')}件"]
    for st in ("PL", "BS", "CF"):
        ns = [n.id for n in NODES if n.statement == st]
        lines.append(f"  {st}: {', '.join(ns)}")
    for k in EDGE_KINDS:
        lines.append(f"  edge[{k}]: {sum(1 for e in EDGES if e.kind == k)}件")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
