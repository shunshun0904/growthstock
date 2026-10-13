#!/usr/bin/env python3
"""
大量保有報告書（J-Quants /edinet/large-volume-shareholders、保存名 lvshld）の中身の形を実測する。

運用者の依頼（2026-10-13）「特徴量として、機関投資家による買いが直近であったかどうかのフラグを入れたいです」の下調べ。
今の特徴量（lvs_*: research/extra_features.py の large_volume）は「誰かが5%超を持った・増やした」だけで、
提出者が機関投資家かどうかを見ていない。フラグの定義を決める前に、次を件数で確かめる。

  1. 期間と件数（提出日の年ごと・銘柄数）。フラグがどの年から作れるか
  2. 列と、保有者の一覧（Hldrs。入れ子の list of dict）の項目名・埋まり方。保有目的や保有者の種類のような
     区分の項目があるか（区分らしい項目は値ごとの件数を出す）
  3. 書類の種類（大量保有報告書 / 変更報告書 / 訂正 / 短期大量譲渡）と、保有割合の増減（増えた・減った・新規）
  4. 変更の理由（ChgRsn）の言葉ごとの件数（市場内取引・市場外・貸借・担保・新株予約権 など）
  5. 提出者の名前からの粗い分類（運用会社・投資顧問 / 証券会社 / 銀行・信託 / 保険 / 個人 / その他の法人）の件数と、
     その中で保有割合が増えた・新規の件数

出すのは列名・項目名・件数・割合だけ。銘柄・提出者の名前・値は出さない（Actions のログは公開）。区分らしい項目の
値ごとの件数は、値の種類が少なく（MAX_LEVELS 以下）、名前・住所らしい項目でないときだけ出す。

    exp=probe_lvs_holders.py
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extra_features as XF  # noqa: E402

#: 値ごとの件数を出す区分の項目の、値の種類の上限（これより多い項目は名前などの自由な値とみなして出さない）
MAX_LEVELS = 25
#: 名前・住所・番号らしい項目（値を出さない）
NAME_LIKE = re.compile(r"name|nm$|addr|address|rep|person|id$|code$|no$|tel|url|en$", re.I)

#: 提出者の名前からの粗い分類（上から順に当てる）。件数を見るための代理で、定義はこの結果を見て決める
CLASSES = [
    ("証券会社", re.compile(r"証券|證券|SECURITIES", re.I)),
    ("運用会社・投資顧問", re.compile(r"アセット|投信|投資顧問|投資信託委託|インベストメント|アドバイザ|キャピタル|ファンド|"
                                     r"パートナーズ|マネジメント|ASSET|INVEST|ADVIS|CAPITAL|FUND|PARTNERS|MANAGEMENT",
                                     re.I)),
    ("銀行・信託", re.compile(r"銀行|信託|BANK|TRUST", re.I)),
    ("保険", re.compile(r"生命|保険|INSURANCE|ASSURANCE|LIFE", re.I)),
]
#: 変更の理由の言葉（件数だけ数える）
REASON_WORDS = ["市場内", "市場外", "取得", "処分", "譲渡", "貸借", "貸株", "借入", "担保", "新株予約権", "転換", "株式分割",
                "増加", "減少", "公開買付", "共同保有", "割当"]
#: 書類の種類の言葉
TITLE_WORDS = ["大量保有報告書", "変更報告書", "訂正", "短期大量譲渡", "特例対象"]
#: 自由な文の列（値の種類が少なくても値そのものは出さず、言葉ごとの件数だけ数える。中に名前が入りうる）
FREE_TEXT = {"ChgRsn"}
#: 保有目的・提出者の種類・事業内容らしい文に出てくる言葉（件数だけ数える）
HOLDER_WORDS = ["純投資", "政策投資", "経営", "重要提案", "投資一任", "顧客", "信託財産", "運用", "ポートフォリオ",
                "安定株主", "支配", "貸借", "担保", "投資運用業", "投資助言", "金融商品取引業", "銀行", "保険", "証券",
                "信託", "個人", "法人"]


def holders(cell) -> List[dict]:
    """Hldrs（list of dict。parquet から読むと ndarray）を list of dict に。"""
    return XF._holders(cell)


def holder_name(h: dict) -> str:
    """保有者の名前らしい項目（どの名前の項目かは決め打ちしない）。"""
    for k in h:
        if re.search(r"name|nm$", str(k), re.I) and not re.search(r"en$", str(k), re.I):
            v = h.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def classify(name: str) -> str:
    """名前からの粗い分類。法人らしさの言葉（XF.CORP_PAT）が無ければ個人とみなす。"""
    if not name:
        return "名前なし"
    for label, pat in CLASSES:
        if pat.search(name):
            return label
    return "その他の法人" if XF.CORP_PAT.search(name) else "個人"


def ratio_move(now, last) -> str:
    """保有割合の動き（報告書の合計の割合と前回の割合）。"""
    a = pd.to_numeric(now, errors="coerce")
    b = pd.to_numeric(last, errors="coerce")
    if not np.isfinite(a):
        return "割合なし"
    if not np.isfinite(b) or b == 0:
        return "新規（前回なし）"
    if a > b:
        return "増えた"
    if a < b:
        return "減った"
    return "同じ"


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "—"


def print_counts(title: str, counts: Counter, total: int, top: Optional[int] = None) -> None:
    print(f"  {title}（{total:,}件）")
    for k, n in counts.most_common(top):
        print(f"    {k}: {n:,}（{pct(n, total)}）")


def level_counts(values: Iterable) -> Optional[Counter]:
    """
    区分らしい値の件数。値の種類が MAX_LEVELS を超えるか、入れ子（list / dict。中に名前が入りうる）が混ざれば
    None（出さない）。
    """
    vals = list(values)
    if any(isinstance(v, (list, tuple, dict, np.ndarray)) for v in vals):
        return None
    # 法人の名前らしい値（株式会社・LTD など）が混ざる項目は、値の種類が少なくても出さない
    if any(isinstance(v, str) and XF.CORP_PAT.search(v) for v in vals):
        return None
    c = Counter("（空）" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in vals)
    return c if len(c) <= MAX_LEVELS else None


def word_counts(title: str, values: Iterable, words: List[str] = HOLDER_WORDS) -> None:
    """文の値に出てくる言葉ごとの件数（値そのものは出さない）。入れ子・数値は数えない。"""
    s = [v for v in values if isinstance(v, str) and v.strip()]
    if not s:
        return
    hits = [(w, sum(1 for v in s if w in v)) for w in words]
    hits = [(w, k) for w, k in hits if k]
    lens = np.array([len(v) for v in s])
    print(f"    {title}: 文が入っている {len(s):,}件・長さの中央 {np.median(lens):.0f}字。言葉ごとの件数: "
          + (" / ".join(f"{w} {k:,}（{pct(k, len(s))}）" for w, k in hits) if hits else "どれも出てこない"))


def report(d: pd.DataFrame) -> Dict:
    out: Dict = {}
    n = len(d)
    print(f"[rows] {n:,}件 / 列 {len(d.columns)}")
    if not n:
        return out
    sub = pd.to_datetime(d.get("SubDate"), errors="coerce")
    print(f"[period] 提出日 {sub.min().date() if sub.notna().any() else '—'} 〜 "
          f"{sub.max().date() if sub.notna().any() else '—'} / 銘柄 {d['Code'].astype(str).nunique():,}"
          if "Code" in d.columns else "[period] Code が無い")
    years = sub.dt.year.value_counts().sort_index()
    print("[years] " + " / ".join(f"{int(y)}年 {c:,}" for y, c in years.items()))
    out["years"] = {int(y): int(c) for y, c in years.items()}

    print("\n== 列（埋まっている行の割合） ==")
    for c in d.columns:
        nn = int(d[c].notna().sum())
        print(f"  {c}: {pct(nn, n)}  型 {d[c].dtype}")

    print("\n== 区分らしい列の値ごとの件数（値の種類が少なく、名前・番号らしくない列だけ） ==")
    for c in d.columns:
        if c in ("Hldrs",) or c in FREE_TEXT or NAME_LIKE.search(c) or c in ("SubDate", "RptOblgDate", "Date"):
            continue
        if d[c].dtype != object and not str(d[c].dtype).startswith("string"):
            continue
        lc = level_counts(d[c].tolist())
        if lc is not None:
            print_counts(c, lc, n)
        else:
            print(f"  {c}: 値の種類が {MAX_LEVELS} を超えるか入れ子なので値は出さない")
            word_counts(c, d[c].tolist())

    if "DocTitle" in d.columns:
        t = d["DocTitle"].astype(str)
        print("\n== 書類の題名の言葉 ==")
        for w in TITLE_WORDS:
            k = int(t.str.contains(w, regex=False).sum())
            print(f"  「{w}」を含む: {k:,}（{pct(k, n)}）")

    moves = Counter()
    if "TotalShsRatio" in d.columns:
        last = d["TotalShsRatioLast"] if "TotalShsRatioLast" in d.columns else pd.Series(np.nan, index=d.index)
        mv = [ratio_move(a, b) for a, b in zip(d["TotalShsRatio"], last)]
        d = d.assign(_move=mv)
        moves = Counter(mv)
        print()
        print_counts("保有割合の動き（報告書の合計の割合 TotalShsRatio と前回 TotalShsRatioLast）", moves, n)
        r = pd.to_numeric(d["TotalShsRatio"], errors="coerce")
        q = r.quantile([0.1, 0.5, 0.9]).to_numpy()
        print(f"  割合の分布（小数）: 10% {q[0]:.4f} / 中央 {q[1]:.4f} / 90% {q[2]:.4f}"
              "（1以下なら小数、5以上なら%）")
    out["moves"] = dict(moves)

    if "ChgRsn" in d.columns:
        cr = d["ChgRsn"]
        has = cr.notna() & (cr.astype(str).str.strip() != "")
        print(f"\n== 変更の理由（ChgRsn）: 入っている {int(has.sum()):,}（{pct(int(has.sum()), n)}） ==")
        s = cr[has].astype(str)
        for w in REASON_WORDS:
            k = int(s.str.contains(w, regex=False).sum())
            if k:
                print(f"  「{w}」を含む: {k:,}（{pct(k, int(has.sum()))}）")

    if "Hldrs" not in d.columns:
        print("\nHldrs が無い")
        return out
    hl = [holders(c) for c in d["Hldrs"]]
    nh = np.array([len(h) for h in hl])
    print(f"\n== 保有者の一覧（Hldrs） ==")
    print(f"  一覧が入っている報告書 {int((nh > 0).sum()):,}（{pct(int((nh > 0).sum()), n)}）/ "
          f"1報告書の保有者の数: 中央 {np.median(nh[nh > 0]) if (nh > 0).any() else 0:.0f}・"
          f"90% {np.quantile(nh[nh > 0], 0.9) if (nh > 0).any() else 0:.0f}・最大 {nh.max() if len(nh) else 0}")
    keys = Counter()
    filled = Counter()
    types: Dict[str, Counter] = {}
    for hs in hl:
        for h in hs:
            for k, v in h.items():
                keys[k] += 1
                if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() != "":
                    filled[k] += 1
                types.setdefault(k, Counter())[type(v).__name__] += 1
    total_h = int(nh.sum())
    print(f"  保有者の行 {total_h:,}。項目（埋まっている割合・型）:")
    for k, c in keys.most_common():
        print(f"    {k}: {pct(filled[k], c)}  型 {dict(types[k])}")
    out["holder_keys"] = sorted(keys)

    print("\n  区分らしい項目の値ごとの件数（値の種類が少なく、名前・住所・番号らしくない項目だけ）:")
    for k in keys:
        if NAME_LIKE.search(str(k)):
            continue
        vals = [h.get(k) for hs in hl for h in hs if k in h]
        if all(isinstance(v, (int, float, np.integer, np.floating)) or v is None for v in vals):
            continue
        lc = level_counts(vals)
        if lc is not None:
            print_counts(f"  {k}", lc, len(vals))
        else:
            print(f"    {k}: 値の種類が {MAX_LEVELS} を超えるか入れ子なので値は出さない")
            word_counts(f"  {k}", vals)

    # 提出者（一覧の先頭）の名前からの粗い分類と、その中の増減
    first = [classify(holder_name(hs[0])) if hs else "一覧なし" for hs in hl]
    anyc = [sorted({classify(holder_name(h)) for h in hs}) if hs else ["一覧なし"] for hs in hl]
    print("\n== 提出者の名前からの粗い分類（一覧の先頭の保有者。件数だけ） ==")
    fc = Counter(first)
    for label, k in fc.most_common():
        mv = Counter(m for f, m in zip(first, d["_move"] if "_move" in d.columns else [""] * n) if f == label)
        up = mv.get("増えた", 0) + mv.get("新規（前回なし）", 0)
        print(f"  {label}: {k:,}（{pct(k, n)}）・うち増えた／新規 {up:,}（{pct(up, k)}）")
    out["first_class"] = dict(fc)
    ac = Counter(c for cs in anyc for c in cs)
    print("  共同保有者を含めて、その分類の保有者が1人でもいる報告書:")
    for label, k in ac.most_common():
        print(f"    {label}: {k:,}（{pct(k, n)}）")
    if "_move" in d.columns and sub.notna().any():
        inst = np.array([f == "運用会社・投資顧問" for f in first])
        up = d["_move"].isin(["増えた", "新規（前回なし）"]).to_numpy()
        yy = sub.dt.year.to_numpy()
        print("  年ごとの「運用会社・投資顧問が先頭で、増えた／新規」の報告書の数: " + " / ".join(
            f"{int(y)}年 {int(((yy == y) & inst & up).sum()):,}" for y in sorted(set(yy[~pd.isna(yy)]))))
    out.update(report_detail(d, hl, sub))
    return out


# --------------------------------------------------------------------------- #
# 2回目（2026-10-13）: 種類のコード・保有目的の区分・直近60日の取得・処分の明細
# --------------------------------------------------------------------------- #

#: 区分のコード（値はコードなので出してよい）。1回目は名前らしい項目の判定（…Code）で飛ばしていた
CODE_COLS = ["LargeHldgTypeCode", "DocTypeCode"]
CODE_KEYS = ["LargeHldrTypeCode"]
#: 保有目的（HldgPurp）の文の区分（上から順に当てる）。件数を見るための代理で、定義はこの結果を見て決める
PURPOSE_CLASSES = [
    ("証券業務・貸借・担保", re.compile(r"証券業務|商品在庫|トレーディング|ディーリング|自己勘定|貸借|担保")),
    ("重要提案", re.compile(r"重要提案")),
    ("経営・安定株主・政策投資", re.compile(r"経営|安定株主|政策投資|支配|業務提携|資本提携|取引関係")),
    ("運用（純投資・投資一任・顧客の資産）", re.compile(r"純投資|投資一任|顧客|運用|信託財産|ポートフォリオ")),
]
#: 取得・処分の明細の文に出てくる言葉
ACQ_WORDS = ["取得", "処分", "市場内", "市場外", "買付", "売付", "借入", "返還", "貸付", "新株予約権", "公開買付", "現物",
             "株式分割", "合併"]


def purpose_class(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return "（空）"
    for label, pat in PURPOSE_CLASSES:
        if pat.search(text):
            return label
    return "その他"


def items(cell) -> List:
    """入れ子の明細（ndarray / list）を list に。"""
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    return list(cell) if isinstance(cell, (list, tuple)) else []


def _f(x) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if np.isfinite(v) else np.nan


def holder_move(h: dict) -> str:
    """保有者ごとの割合の動き（ShsRatio と ShsRatioLast）。"""
    return ratio_move(h.get("ShsRatio"), h.get("ShsRatioLast"))


def report_detail(d: pd.DataFrame, hl: List[List[dict]], sub: pd.Series) -> Dict:
    out: Dict = {}
    n = len(d)
    print("\n== 2回目: 区分のコード（値ごとの件数） ==")
    for c in CODE_COLS:
        if c in d.columns:
            lc = level_counts(d[c].tolist())
            if lc is not None:
                print_counts(c, lc, n)
            else:
                print(f"  {c}: 値の種類が {MAX_LEVELS} を超える（{d[c].nunique():,}種類）")
    for k in CODE_KEYS:
        vals = [h.get(k) for hs in hl for h in hs if k in h]
        lc = level_counts(vals)
        if lc is not None:
            print_counts(f"保有者の {k}", lc, len(vals))
        elif vals:
            print(f"  保有者の {k}: 値の種類が {MAX_LEVELS} を超える")

    # 保有目的の区分 × 保有者ごとの割合の動き × 種類のコード
    rows = []
    for i, hs in enumerate(hl):
        for h in hs:
            rows.append({"i": i, "cls": purpose_class(h.get("HldgPurp")), "move": holder_move(h),
                         "type": str(h.get("LargeHldrTypeCode")), "acq": items(h.get("AcqDisp"))})
    hd = pd.DataFrame(rows)
    out["purpose_classes"] = {}
    if len(hd):
        print("\n== 2回目: 保有目的（HldgPurp）の区分ごとの保有者の数と、保有者ごとの割合の動き ==")
        for cls, g in hd.groupby("cls"):
            mv = Counter(g["move"])
            up = mv.get("増えた", 0) + mv.get("新規（前回なし）", 0)
            tp = Counter(g["type"])
            print(f"  {cls}: {len(g):,}（{pct(len(g), len(hd))}）・増えた {mv.get('増えた', 0):,}・新規 "
                  f"{mv.get('新規（前回なし）', 0):,}・減った {mv.get('減った', 0):,}（増えた＋新規 {pct(up, len(g))}）"
                  f"・種類のコード {dict(tp.most_common(6))}")
            out["purpose_classes"][cls] = {"n": int(len(g)), "up": int(up)}

        # 直近60日の取得・処分の明細
        acq = [a for lst in hd["acq"] for a in lst]
        nacq = hd["acq"].map(len).to_numpy()
        print(f"\n== 2回目: 直近60日の取得・処分の明細（AcqDisp） ==")
        print(f"  明細のある保有者 {int((nacq > 0).sum()):,} / {len(hd):,}（{pct(int((nacq > 0).sum()), len(hd))}）・"
              f"明細の行 {len(acq):,}・1保有者の行数 中央 {np.median(nacq[nacq > 0]) if (nacq > 0).any() else 0:.0f}・"
              f"90% {np.quantile(nacq[nacq > 0], 0.9) if (nacq > 0).any() else 0:.0f}")
        dicts = [a for a in acq if isinstance(a, dict)]
        keys, filled, types = Counter(), Counter(), {}
        for a in dicts:
            for k, v in a.items():
                keys[k] += 1
                if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() != "":
                    filled[k] += 1
                types.setdefault(k, Counter())[type(v).__name__] += 1
        for k, c in keys.most_common():
            print(f"    {k}: {pct(filled[k], c)}  型 {dict(types[k])}")
        for k in keys:
            if NAME_LIKE.search(str(k)) or re.search(r"date|dt$|day", str(k), re.I):
                continue                                   # 名前・日付は値を出さない（日付は下で提出日との差だけ）
            vals = [a.get(k) for a in dicts]
            if all(isinstance(v, (int, float, np.integer, np.floating)) or v is None for v in vals):
                continue
            lc = level_counts(vals)
            if lc is not None:
                print_counts(f"  明細の {k}", lc, len(vals))
            else:
                word_counts(f"  明細の {k}", vals, ACQ_WORDS)
        # 日付らしい項目: 提出日の何日前の取引か
        subs = d["SubDate"].reset_index(drop=True) if "SubDate" in d.columns else None
        for k in keys:
            if not re.search(r"date|dt$|day", str(k), re.I) or subs is None:
                continue
            lag = []
            for i, lst in zip(hd["i"], hd["acq"]):
                s0 = pd.to_datetime(subs.iloc[i], errors="coerce")
                for a in lst:
                    if isinstance(a, dict):
                        t = pd.to_datetime(a.get(k), errors="coerce")
                        if pd.notna(t) and pd.notna(s0):
                            lag.append((s0 - t).days)
            if lag:
                q = np.quantile(lag, [0.1, 0.5, 0.9])
                print(f"  明細の {k}: 提出日の何日前か 10% {q[0]:.0f}・中央 {q[1]:.0f}・90% {q[2]:.0f}日（{len(lag):,}行）")

        # 候補の定義ごとの報告書の数（年ごと）
        acq_buy = hd["acq"].map(lambda lst: any("取得" in str(a) or "買付" in str(a) for a in lst))
        up_h = hd["move"].isin(["増えた", "新規（前回なし）"])
        run = hd["cls"] == "運用（純投資・投資一任・顧客の資産）"
        yy = sub.reset_index(drop=True).dt.year
        defs = {"D1 運用目的の保有者の割合が増えた／新規": run & up_h,
                "D2 D1 かつ直近60日の明細に取得": run & up_h & acq_buy,
                "D3 運用目的の保有者の明細に取得（割合は問わない）": run & acq_buy}
        print("\n== 2回目: 候補の定義ごとの報告書の数（年ごと。1報告書に当たる保有者が1人でもいれば1件） ==")
        out["defs"] = {}
        for name, m in defs.items():
            docs = sorted(set(hd.loc[m, "i"]))
            ys = Counter(int(yy.iloc[i]) for i in docs if pd.notna(yy.iloc[i]))
            print(f"  {name}: {len(docs):,}（全報告書の {pct(len(docs), n)}）・" + " / ".join(
                f"{y}年 {c:,}" for y, c in sorted(ys.items())))
            out["defs"][name] = len(docs)
    return out


def main(argv=None) -> int:
    data_dir = os.environ.get("DATA_DIR", XF.DATA_DIR)
    d = XF._read("lvshld", data_dir)
    print(f"[load] lvshld {len(d):,}件（{data_dir}）")
    report(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
