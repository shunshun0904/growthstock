#!/usr/bin/env python3
"""
実験71: 際どい候補を、EDINET DB の年次財務（粗利率・販管費率・営業外依存度）で分けると成績が違うか。

運用者（2026-10-10）: 決算サンキー（売上原価・販管費・営業外まで分けた損益の流れ）を売買の判断に使えるか、
という問いに対して「実験70 と同じ過去の予測に、EDINET DB の年次データから粗利率・販管費率・営業外依存度を付け、
際どい候補をそれで分けた成績を出す」ことになった。サンキー図を読む人が目で見る3つの比率を、過去の候補に
機械的に付けて、同じ目で分けたらどうだったかを測る（実験70 の日証金の節と同じ作り）。

材料
----
- 実験70 と同じ表（本番の239列の OOF。木3つ + logit。百分位は前の窓の分布。+10%指値と持ち切り。銘柄ごとの SE）
- EDINET DB の年次財務（research/_data/edinet_fin.parquet）。各行には「提出日の翌日 ≤ Date」の最新の有報を付ける
  （research/edinet_features.attach と同じ時点整合。400日より古ければ欠測）
  粗利率       gross_profit ÷ revenue（edinet_features の gpm と同じ定義）
  販管費率     sga ÷ revenue（同 sga_r）
  営業外依存度 経常利益と営業利益の対称変化率 sym(経常, 営業) = (経常 − 営業) ÷ ((|経常| + |営業|) / 2)。
               [−2, 2] に収まる（営業利益 0 で経常がプラスなら 2＝全部が営業外）。IFRS には経常利益が無いので
               税引前利益で代える（金融収益・持分法など、営業利益から税引前までの合流）。日本基準で経常利益が
               無い行は欠測（税引前で代えると特別損益が混ざる）
- 分け方は3通り（結果を見る前に決めた。見てから足さない）
  水準     その比率そのもの。業種で決まる部分が大きい（商社・小売は粗利率が低く、ソフトは高い）ので、
           母集団の分布で切ると業種を分けているのに近い。それでも「サンキーを見て高い・低いと思う」のはこれ
  前年差   同じ会社の前年度との差（会計基準・連結/個別が変わった年度とは比べない）。会社の中の変化
  業種相対 水準 − 同じ業種（EDINET DB の industry）の中央値。中央値は **その行の前月末の時点で** 各社の最新の
           有報（400日以内）から取る（先の書類を使わない。5社未満の業種は欠測）
- 線は実験70 の日証金と同じく、その月より前の母集団（比率の付いた行）の分布で引く（先の情報を使わない）。
  3分位（下 1/3・中・上 1/3）。前の行が 300 に満たない月は欠測

向き（先に決めた）
----
  粗利率       低い 1/3 が悪い側（粗利が薄い）
  販管費率     高い 1/3 が悪い側（費用が重い）
  営業外依存度 高い 1/3 が悪い側（本業以外で稼いでいる）
前年差・業種相対も同じ向き（粗利率は下がった側・業種より低い側が悪い、など）。
主の検定は「際どい候補の中で、悪い側 − それ以外（+10%指値）」の 3比率 × 3通り = 9本。9本もあると、効いていなくても
|z| ≥ 2 が1本は出る確率が 1 − 0.95^9 ≈ 37% ある。3つとも95以上と母集団でも同じ分け方を出す（際どい候補でだけ
出る差は怪しい）。おまけに、悪い印の数（0〜3。水準・前年差・業種相対それぞれ）でも並べる。

期待
----
実験22・68 で EDINET の列はモデルの精度を上げなかった。決算の情報はモデルが既に読んでいる（239列の多くが決算由来）
ので、後工程の分け方でも差は出ない見込み。悪い方向に強く効いていないかを見るのが目的。

注意: 比率の付く行は EDINET DB で取れている約1,670社ぶん（母集団の 7割）。際どい候補 × 比率あり × 3分位は
数百件なので、差が出なくても「効かない」とは言えない。公開ログには件数・割合だけを出す（銘柄名・銘柄ごとの値・
分位の線の値は出さない）。

  exp=e71_borderline_edinet.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import e70_borderline as E70  # noqa: E402
import edinet_features as EF  # noqa: E402
import lab  # noqa: E402

OUT = os.path.join(lab.DATA_DIR, "oof", "e71_borderline_edinet.json")
COMPANIES = os.path.join(lab.DATA_DIR, "edinet_companies.parquet")
MIN_REF = 300          # 3分位の線を引くのに要る、その月より前の行
MIN_INDUSTRY = 5       # 業種の中央値に要る社数
TERT = (100.0 / 3.0, 200.0 / 3.0)

#: (鍵, 表示, 悪い側が "lo" か "hi" か)
MEASURES = [
    ("gpm", "粗利率", "lo"),
    ("sga_r", "販管費率", "hi"),
    ("nonop", "営業外依存度", "hi"),
]
VARIANTS = [("level", "水準"), ("chg1", "前年差"), ("rel", "業種相対")]
TERT_JA = {"lo": "下 1/3", "mid": "中", "hi": "上 1/3"}
log = E70.log


# ---------------------------------------------------------------------- #
# EDINET DB から3つの比率
# ---------------------------------------------------------------------- #

def _num(f: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(f[c], errors="coerce").astype(float) if c in f.columns else pd.Series(np.nan, index=f.index)


def nonop_dependence(f: pd.DataFrame) -> pd.Series:
    """
    営業外依存度 = sym(経常利益, 営業利益)。IFRS（accounting_standard が "JP" でない）で経常利益が無ければ
    税引前利益で代える。日本基準で経常利益が無い行は欠測。
    """
    op = _num(f, "operating_income")
    ordi = _num(f, "ordinary_income")
    pbt = _num(f, "profit_before_tax")
    std = f["accounting_standard"].astype("string").fillna("").astype(str) if "accounting_standard" in f.columns \
        else pd.Series("", index=f.index)
    # 会計基準が分かっていて日本基準でない（IFRS・米国基準）ときだけ税引前利益で代える
    top = ordi.where(ordi.notna(), pbt.where((std != "JP") & (std != "")))
    return EF.sym(top, op)


def measures(fin: pd.DataFrame) -> pd.DataFrame:
    """
    1行 = (jq_code, fiscal_year)。ed_gpm / ed_sga_r / ed_nonop の水準（level）と前年差（chg1）、avail_date。
    同じ年度に複数の書類があれば最初の提出（edinet_features.annual_panel と同じ）。前年差は会計基準・連結/個別が
    同じ年度とだけ比べる。列名は ed_<比率>_<level|chg1>。業種相対（rel）は attach のあとで付ける。
    """
    f = fin.copy()
    f[EF.KEY] = f[EF.KEY].astype(str)
    f["fiscal_year"] = pd.to_numeric(f["fiscal_year"], errors="coerce")
    f["submit_date"] = pd.to_datetime(f.get("submit_date"), errors="coerce")
    f = f.dropna(subset=["fiscal_year"])
    f["fiscal_year"] = f["fiscal_year"].astype(int)
    for c in EF.GUARDS:
        f[c] = f[c].astype("string").fillna("").astype(str) if c in f.columns else ""
    f = (f.sort_values([EF.KEY, "fiscal_year", "submit_date"], na_position="last")
          .drop_duplicates([EF.KEY, "fiscal_year"], keep="first").reset_index(drop=True))
    rev = _num(f, "revenue")
    out = f[[EF.KEY, "fiscal_year", "submit_date"] + EF.GUARDS].copy()
    out["gpm"] = EF._ratio(_num(f, "gross_profit"), rev)
    out["sga_r"] = EF._ratio(_num(f, "sga"), rev)
    out["nonop"] = nonop_dependence(f)
    keys = [m for m, _, _ in MEASURES]
    lag = out[[EF.KEY, "fiscal_year"] + EF.GUARDS + keys].copy()
    lag["fiscal_year"] = lag["fiscal_year"] + 1
    lag = lag.rename(columns={c: f"{c}_y1" for c in EF.GUARDS + keys})
    out = out.merge(lag, on=[EF.KEY, "fiscal_year"], how="left")
    bad = pd.Series(False, index=out.index)
    for g in EF.GUARDS:
        bad |= out[f"{g}_y1"].notna() & (out[f"{g}_y1"] != out[g])
    out.loc[bad, [f"{k}_y1" for k in keys]] = np.nan
    for k in keys:
        out[f"ed_{k}_level"] = out[k]
        out[f"ed_{k}_chg1"] = out[k] - out[f"{k}_y1"]
    out["avail_date"] = out["submit_date"].dt.normalize() + pd.Timedelta(days=1)
    cols = [EF.KEY, "fiscal_year", "avail_date"] + [f"ed_{k}_{v}" for k in keys for v in ("level", "chg1")]
    return out[cols]


def industry_reference(feats: pd.DataFrame, companies: pd.DataFrame, month_ends: pd.DatetimeIndex,
                       stale_days: int = EF.STALE_DAYS, min_n: int = MIN_INDUSTRY) -> pd.DataFrame:
    """
    各月末の時点で、各社の最新の有報（avail_date ≤ 月末、stale_days 以内）から業種ごとの中央値。
    返す表の列: month_end, industry, ref_<比率>, n。min_n 社に満たない業種は出さない。
    """
    keys = [m for m, _, _ in MEASURES]
    comp = companies[["code", "industry"]].dropna().copy()
    comp["code"] = comp["code"].astype(str)
    comp["industry"] = comp["industry"].astype(str)
    comp = comp[comp["industry"] != ""].drop_duplicates("code")
    f = feats.dropna(subset=["avail_date"]).merge(comp, left_on=EF.KEY, right_on="code", how="inner")
    rows = []
    for m in month_ends:
        cur = f[(f["avail_date"] <= m) & ((m - f["avail_date"]).dt.days <= stale_days)]
        cur = cur.sort_values("avail_date").drop_duplicates(EF.KEY, keep="last")
        for ind, g in cur.groupby("industry"):
            vals = {f"ref_{k}": float(g[f"ed_{k}_level"].median()) if g[f"ed_{k}_level"].notna().sum() >= min_n
                    else np.nan for k in keys}
            rows.append({"month_end": m, "industry": ind, "n": int(len(g)), **vals})
    cols = ["month_end", "industry", "n"] + [f"ref_{k}" for k in keys]
    return pd.DataFrame(rows, columns=cols)


def attach_relative(d: pd.DataFrame, ref: pd.DataFrame, companies: pd.DataFrame) -> pd.DataFrame:
    """各行に、前月末の時点の業種の中央値との差 ed_<比率>_rel を付ける。業種が無い・中央値が無ければ欠測。"""
    keys = [m for m, _, _ in MEASURES]
    d = d.copy()
    comp = companies[["code", "industry"]].dropna().copy()
    comp["code"] = comp["code"].astype(str)
    comp = comp.drop_duplicates("code")
    d["_ind"] = d["Code"].astype(str).map(dict(zip(comp["code"], comp["industry"].astype(str))))
    d["_me"] = (pd.to_datetime(d["Date"]).dt.to_period("M") - 1).dt.end_time.dt.normalize()
    r = ref.rename(columns={"month_end": "_me", "industry": "_ind"})
    m = d[["_ind", "_me"]].merge(r[["_ind", "_me"] + [f"ref_{k}" for k in keys]], on=["_ind", "_me"], how="left")
    for k in keys:
        d[f"ed_{k}_rel"] = d[f"ed_{k}_level"].to_numpy() - m[f"ref_{k}"].to_numpy()
    return d.drop(columns=["_ind", "_me"])


def tertiles(dates: pd.Series, x: pd.Series, min_rows: int = MIN_REF) -> pd.Series:
    """各行を、その月より前の母集団の分布で lo / mid / hi に分ける。線が引けない・値が無い行は None。"""
    xv = pd.to_numeric(x, errors="coerce")
    lo = E70.past_threshold(dates, xv, q=TERT[0], min_rows=min_rows)
    hi = E70.past_threshold(dates, xv, q=TERT[1], min_rows=min_rows)
    ok = xv.notna() & lo.notna() & hi.notna()
    out = np.where(xv <= lo, "lo", np.where(xv >= hi, "hi", "mid"))
    return pd.Series(np.where(ok, out, None), index=x.index, dtype=object)


def load() -> tuple:
    """実験70 の表に、EDINET DB の3つの比率（水準・前年差・業種相対）を付けて返す。(表, logit の有無, 業種相対の有無)"""
    d, has_logit = E70.load()
    if not os.path.exists(EF.FIN):
        raise SystemExit(f"EDINET DB の年次財務が無い: {EF.FIN}")
    feats = measures(EF.load_fin(EF.FIN))
    log(f"EDINET DB: {feats[EF.KEY].nunique():,}社 / {len(feats):,}年度 / 粗利率あり {feats['ed_gpm_level'].notna().mean()*100:.0f}%"
        f" / 販管費率あり {feats['ed_sga_r_level'].notna().mean()*100:.0f}% / 営業外依存度あり {feats['ed_nonop_level'].notna().mean()*100:.0f}%")
    d = EF.attach(d, feats)
    has_rel = os.path.exists(COMPANIES)
    if has_rel:
        comp = pd.read_parquet(COMPANIES)
        has_rel = "industry" in comp.columns and "code" in comp.columns
    if has_rel:
        months = pd.date_range(pd.to_datetime(d["Date"]).min() - pd.offsets.MonthEnd(2),
                               pd.to_datetime(d["Date"]).max(), freq=pd.offsets.MonthEnd())
        ref = industry_reference(feats, comp, months)
        d = attach_relative(d, ref, comp)
        log(f"業種相対: 業種 {ref['industry'].nunique()}・月末 {ref['month_end'].nunique()}")
    else:
        log("業種の対応表（edinet_companies.parquet）が無いので業種相対は飛ばす")
        for k, _, _ in MEASURES:
            d[f"ed_{k}_rel"] = np.nan
    return d, has_logit, has_rel


# ---------------------------------------------------------------------- #

def main(argv=None) -> int:
    d, has_logit, has_rel = load()
    d["shape"] = E70.shape_of(d)
    d["laggard"] = E70.laggard_of(d)
    d = d[d["shape"].notna()].copy().reset_index(drop=True)
    d["border"] = d["shape"].isin(E70.BORDER)
    log(f"対象 {len(d):,}行（OOF の窓2以降）/ {d['Date'].min().date()}〜{d['Date'].max().date()}")
    res: Dict[str, object] = {"rows": int(len(d)), "from": str(d["Date"].min().date()),
                              "to": str(d["Date"].max().date()), "has_logit": has_logit, "has_rel": has_rel}

    # 0. 付いた行の割合
    print("\n=== 0. 比率の付いた行（時点整合・400日以内）===")
    res["coverage"] = {}
    for k, nm, _ in MEASURES:
        for v, vn in VARIANTS:
            c = f"ed_{k}_{v}"
            cov = {"all": float(d[c].notna().mean()), "border": float(d.loc[d["border"], c].notna().mean()),
                   "all95": float(d.loc[d["shape"] == "all95", c].notna().mean())}
            res["coverage"][c] = cov
            print(f"  {nm}・{vn:<6} 母集団 {cov['all']*100:5.1f}% / 際どい候補 {cov['border']*100:5.1f}% / 3つとも95以上 {cov['all95']*100:5.1f}%")

    # 3分位（その月より前の母集団の分布）
    for k, _, _ in MEASURES:
        for v, _ in VARIANTS:
            d[f"t_{k}_{v}"] = tertiles(d["Date"], d[f"ed_{k}_{v}"])

    subsets = [("border", "際どい候補", d["border"]), ("all95", "3つとも95以上", d["shape"] == "all95"),
               ("all", "母集団", pd.Series(True, index=d.index))]

    # 1〜3. 比率ごと・分け方ごと
    res["split"] = {}
    for v, vn in VARIANTS:
        if v == "rel" and not has_rel:
            continue
        print(f"\n=== {vn}: 3分位で分ける（線はその月より前の母集団。悪い側は先に決めたとおり）===")
        print(E70.HEAD)
        for k, nm, side in MEASURES:
            t = d[f"t_{k}_{v}"]
            for sk, sn, sm in subsets:
                sub = d[sm & t.notna()]
                for tk in ("lo", "mid", "hi"):
                    s = E70.summarize(sub[t[sub.index] == tk])
                    res["split"][f"{k}_{v}_{sk}_{tk}"] = s
                    mark = "（悪い側）" if tk == side else ""
                    print(E70.line(f"{nm}{vn}・{sn}・{TERT_JA[tk]}{mark}", s))
                bad = E70.summarize(sub[t[sub.index] == side])
                rest = E70.summarize(sub[t[sub.index] != side])
                res["split"][f"{k}_{v}_{sk}_rest"] = rest
                for key in ("tp10", "hold"):
                    dd = E70.diff(bad, rest, key)
                    res["split"][f"{k}_{v}_{sk}_diff_{key}"] = dd
                    print(f"    {sn}: 悪い側 − それ以外（{'+10%指値' if key == 'tp10' else '持ち切り'}）"
                          f"{dd['d']:+.2f}pt ± {dd['se']:.2f}（z {dd['z']:+.2f}）")

    # 4. 悪い印の数
    res["nbad"] = {}
    for v, vn in VARIANTS:
        if v == "rel" and not has_rel:
            continue
        cols = [f"t_{k}_{v}" for k, _, _ in MEASURES]
        ok = d[cols].notna().all(axis=1)
        nbad = sum((d[f"t_{k}_{v}"] == side).astype(int) for k, _, side in MEASURES)
        print(f"\n=== 悪い印の数（{vn}。3つとも線を引けた行だけ）===")
        print(E70.HEAD)
        for sk, sn, sm in subsets:
            sub = d[sm & ok]
            for n in range(4):
                s = E70.summarize(sub[nbad[sub.index] == n])
                res["nbad"][f"{v}_{sk}_{n}"] = s
                print(E70.line(f"{vn}・{sn}・悪い印 {n}", s))
            two = E70.summarize(sub[nbad[sub.index] >= 2])
            zero = E70.summarize(sub[nbad[sub.index] == 0])
            res["nbad"][f"{v}_{sk}_2plus"] = two
            dd = E70.diff(two, zero)
            res["nbad"][f"{v}_{sk}_diff"] = dd
            print(f"    {sn}: 悪い印 2つ以上 − 0（+10%指値）{dd['d']:+.2f}pt ± {dd['se']:.2f}（z {dd['z']:+.2f}）")
        # 参考: 実験70 で弱かった形（2つ95以上・最下位 lgbm）
        weak = d["shape"].isin(("two95_hi", "two95_mid", "two95_lo")) & (d["laggard"] == "lgbm") & ok
        for nm, m in (("悪い印なし", nbad == 0), ("悪い印あり", nbad >= 1)):
            s = E70.summarize(d[weak & m])
            res["nbad"][f"{v}_weak_{nm}"] = s
            print(E70.line(f"{vn}・2つ95以上・最下位 lgbm・{nm}", s))

    print("\n  ※ 主の検定は際どい候補の「悪い側 − それ以外（+10%指値）」9本。1本の |z| ≥ 2 は偶然でも 37% の確率で出る。"
          "3つとも95以上・母集団で同じ向きが出ているかも見る")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda x: None if x is None else float(x))
    log(f"記録: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
