#!/usr/bin/env python3
"""
実験72: 日証金と EDINET DB の特徴量 **だけ** でモデルを組むと、どれだけ当たるか（本番の239列は一切使わない）。

運用者（2026-10-10）「日証金データと EDINET DB から作成できる特徴量のみでモデル構築した場合の精度をみたい。
現行の特徴量は一切使わずに」。

腕（データは同じ1本。列だけが違う。**腕ごとに探索し直す**（列が違えば合うパラメータも違う））
  A   いまの本番（239列）。参照
  J   日証金だけ（research/jsf_features.py の17列）
  E   EDINET DB だけ（research/edinet_features.py の 523列から、時価総額と組み合わせた7列を除いた 516列）
  JE  日証金 + EDINET DB（533列）

列の決めごと
  - EDINET の「時価総額と組み合わせた」7列（ed_fcf_yield・ed_ev_ebitda など）は株価が要る（本番の log_market_cap と
    同じ情報）ので外す。EDINET DB の数字だけで組める列にする
  - 日証金の残高の列は 20日平均出来高で割ってある（割らないと会社の大きさで決まる。出来高は J-Quants の日足）。
    元の数字は日証金なので、そのまま使う。逆日歩・制限措置・貸借値段の列は日証金だけ
  - 本番の239列は A にしか入れない。Code・Date・ラベル以外は J / E / JE に渡さない

行
  学習は本番の母集団（全行）。付いていない行は欠測のまま（木は欠測を分岐で扱う。logit は前処理で埋める）。
  日証金は 2023年10月から、EDINET DB は取れている約1,670社（行の 7割弱）だけなので、
  評価は3通りの行で出す:
    all   全行（本番と同じ）
    ed    EDINET の付いた行
    both  EDINET と日証金の両方が付いた行（2023年10月以降の貸借銘柄。J / JE の列が全部そろう行）
  窓ごとの比較は、その行が 100件以上あり正例と負例の両方がある窓だけ（both は 2023年10月以降の窓だけになる）。

探索・評価は実験67・68 と同じ（50試行 × 5分割・本番と同じ OOF（36/6/6か月・エンバーゴ20営業日・
種42）・窓のずらし 0/2/4か月 × 種3つの平均。logit は種1つ）。モデルは木3つ + ロジスティック回帰。
探索の分割は本番と同じ（E68.CV_SCHEME = tuning.PRODUCTION_CV）。2026-10-10 の回は year_cap_date
（当時の本番。層別・日付単位）で、同日の運用者の指示で本番が walkforward（本番の窓と同じ前進分割）に
変わったので、以後に回せば walkforward になる。
精度は PR-AUC（正例率に対するリフトも出す。正例率が「当てずっぽう」の水準）・ROC-AUC・日内 AUC・上位10%の超過リターン。

Actions の1回の上限（330分）に収まるよう、モデルを分けて回す（同じ実験は同時に1つしか走らない）。
表（ed_* と jsf_* を付けた frame）は1回目に作って research/_data/oof/<tag>_* に置き、以降の回はそれを使う。
--budget-min（既定 290分）を過ぎたら新しい計算を始めずに終わる（同じ引数でもう1回回すと続きから）。
    exp=e72_jsf_edinet_only.py args="--algos lgbm"
    exp=e72_jsf_edinet_only.py args="--algos xgb"
    exp=e72_jsf_edinet_only.py args="--algos cat"
    exp=e72_jsf_edinet_only.py args="--algos logit"
充足だけ見る: exp=e72_jsf_edinet_only.py args="--dry"
試運転:       exp=e72_jsf_edinet_only.py args="--tag e72smoke --algos lgbm --n-trials 2 --shifts 0"

期待（結果を見る前に）
  実験67・68 で、日証金・EDINET を本番の列に **足しても** 精度は上がらなかった。単独なら本番よりかなり低い見込み
  （本番の OOF PR-AUC は正例率の約 2倍）。見たいのは「単独でどこまで当たるか」= 正例率に対するリフトが 1 を
  どれだけ超えるか、と、日証金と EDINET のどちらに情報があるか（J / E / JE の順）。

公開ログには件数・割合・日付・精度だけを出す（日証金・EDINET の値は出さない）。本番の設定（research/lgbm_params.json、
research/multi_params.json、features.py）には書かない。

結果（2026-10-10、run 38066879529、lgbm だけ（運用者の指示）。docs/MODEL_ADOPTION_RULES.md §31）
--------
- 本番と同じ OOF（全行 16,320件・正例率 17.9%）の PR-AUC: A 0.2767（1.55x）/ J 0.1846（1.03x）/ E 0.1892（1.06x）/
  JE 0.1908（1.07x）。ROC-AUC は A 0.637、J/E/JE 0.52 前後。両方の付いた行（7,282件）でも JE 0.2087（1.08x）対 A 0.3117（1.61x）
- 32窓: JE−A −0.0818 ± 0.0075（A が上 32/32）。JE−正例率 +0.0083 ± 0.0032（20/32）。両方の付いた行の 20窓で +0.0166 ± 0.0056
- 上位10% の超過リターンは J/E/JE とも 0 か負（A +0.87pt）。日証金と EDINET DB だけでは、ほぼ当てずっぽう
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edinet_features as EF  # noqa: E402
import features as F  # noqa: E402
import jsf_features as JF  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import ab_oof as AB  # noqa: E402
import e68_edinet_ab as E68  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

ALGOS = E68.ALGOS                       # lgbm / xgb / cat / logit
ARMS = ("A", "J", "E", "JE")
COLS_OF = {"A": "base", "J": "jsf", "E": "ed", "JE": "je"}
LABELS = {"A": "A いまの本番（239列）", "J": "J 日証金だけ（17列）",
          "E": "E EDINET DB だけ（516列）", "JE": "JE 日証金 + EDINET DB（533列）"}
#: 表に出す差（「後 − 前」）
PAIRS = (("A", "J"), ("A", "E"), ("A", "JE"), ("E", "JE"), ("J", "JE"))
ROWSETS = {"all": "全行", "ed": "EDINET の付いた行", "both": "EDINET と日証金の両方が付いた行"}
MIN_WINDOW_ROWS = E68.MIN_WINDOW_ROWS
TAG = "e72"
log = E68.log


# --------------------------------------------------------------------------- #
# 列
# --------------------------------------------------------------------------- #
def column_sets() -> dict:
    """base（本番）/ jsf / ed（時価総額の組み合わせを除く）/ je。互いに重ならないことを確かめる。"""
    base = F.columns(F.DEFAULT_PRESET)
    jsf = JF.columns("all")
    mcap = set(EF.columns("mcap"))
    ed = [c for c in EF.columns("all") if c not in mcap]
    sets = {"base": base, "jsf": jsf, "ed": ed, "je": jsf + ed}
    for a, b in (("base", "jsf"), ("base", "ed"), ("jsf", "ed")):
        dup = sorted(set(sets[a]) & set(sets[b]))
        if dup:
            raise SystemExit(f"列が重なる（{a} と {b}）: {dup[:5]}")
    return sets


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def prepare(rebuild: bool = False):
    """lab.frame に ed_*（EDINET DB）と jsf_*（日証金）を付けた表（全行）と記録。"""
    pf, ps = E68.fpath("frame.parquet"), E68.fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}・"
            f"EDINET の付いた行 {stamp['covered_ed']:,}・日証金 {stamp['covered_jsf']:,}・両方 {stamp['covered_both']:,}）")
        return pd.read_parquet(pf), stamp
    if not os.path.exists(EF.FIN):
        raise SystemExit(f"{EF.FIN} がありません（Release data-raw の edinet_*）")
    if not os.path.exists(os.path.join(lab.DATA_DIR, "jsf_hist.parquet")):
        raise SystemExit("jsf_hist.parquet がありません（Release data-jsf）")
    os.makedirs(E68.OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    fin = EF.load_fin(EF.FIN)
    df = EF.attach(frame, EF.feature_frame(EF.annual_panel(fin)))
    log(f"[data] ed_* を付けた（EDINET {fin[EF.KEY].nunique():,}社）/ {time.time()-t0:.0f}秒")
    t0 = time.time()
    df = JF.build(df, lab.DATA_DIR)
    log(f"[data] jsf_* を付けた / {time.time()-t0:.0f}秒")
    have_ed, have_jsf = df["ed_fiscal_year"].notna(), df["jsf_ratio"].notna()
    yr = df["Date"].dt.year
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
             "rows": int(len(df)), "date_max": str(df["Date"].max().date()),
             "n_codes": int(fin[EF.KEY].nunique()),
             "covered_ed": int(have_ed.sum()), "covered_jsf": int(have_jsf.sum()),
             "covered_both": int((have_ed & have_jsf).sum()),
             "by_year": {str(y): {"rows": int((yr == y).sum()), "ed": int((have_ed & (yr == y)).sum()),
                                  "jsf": int((have_jsf & (yr == y)).sum()),
                                  "both": int((have_ed & have_jsf & (yr == y)).sum())}
                         for y in sorted(yr.unique())}}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1)
    return df, stamp


def eval_keysets(df: pd.DataFrame) -> dict:
    """評価する行の組（ROWSETS の順）。all は None（全行）。"""
    have_ed, have_jsf = df["ed_fiscal_year"].notna(), df["jsf_ratio"].notna()
    return {"all": None, "ed": E68.keys_of(df[have_ed]), "both": E68.keys_of(df[have_ed & have_jsf])}


def coverage_report(df: pd.DataFrame, stamp: dict, colsets: dict) -> dict:
    """付き方（年別）と、列の充足（付いた行の中で）。件数と割合だけ。"""
    have_ed, have_jsf = df["ed_fiscal_year"].notna(), df["jsf_ratio"].notna()
    both = have_ed & have_jsf
    print(f"\n■ 0. 付き方（表は {stamp['built_utc']} 作成・EDINET {stamp['n_codes']:,}社）")
    print(f"  母集団 {len(df):,}行・{df['Code'].nunique():,}銘柄 / EDINET の付いた行 {have_ed.sum():,}（{have_ed.mean()*100:.1f}%）"
          f" / 日証金 {have_jsf.sum():,}（{have_jsf.mean()*100:.1f}%）/ 両方 {both.sum():,}（{both.mean()*100:.1f}%）")
    print("  年別（EDINET / 日証金 / 両方 / 全行）: " + "  ".join(
        f"{y} {v['ed']:,}/{v['jsf']:,}/{v['both']:,}/{v['rows']:,}" for y, v in stamp["by_year"].items()))
    fill = {}
    for name, cols, m in (("jsf", colsets["jsf"], have_jsf), ("ed", colsets["ed"], have_ed)):
        r = df.loc[m, cols].notna().mean()
        fill[name] = {"columns": len(cols), "median": float(r.median()), "min": float(r.min()),
                      "under_10pct": int((r < 0.10).sum())}
        print(f"  {name:<4}{len(cols):>4}列  付いた行の中の充足: 中央値 {r.median()*100:5.1f}%  最小 {r.min()*100:5.1f}%  "
              f"10% 未満 {int((r < 0.10).sum())}列")
    return {"covered_ed": int(have_ed.sum()), "covered_jsf": int(have_jsf.sum()), "covered_both": int(both.sum()),
            "share_ed": float(have_ed.mean()), "share_jsf": float(have_jsf.mean()), "share_both": float(both.mean()),
            "fill": fill}


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #
def safe_metrics(o: pd.DataFrame) -> dict:
    """
    PR-AUC・リフト・ROC-AUC・日内 AUC・上位10%の超過リターン（lab.threshold_edge。しきい値はその窓より前の分布）。
    行が少なくてしきい値を引けない（前の窓が 500件未満）ときは、超過リターンだけ NaN にする。
    """
    y = o["label"].to_numpy(dtype=int)
    s = o["score"].to_numpy(dtype=float)
    rate = float(y.mean())
    m = {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
         "day_auc": float(lab.auc_in_day(o)), "rate": rate}
    m["lift"] = m["pr_auc"] / rate if rate > 0 else float("nan")
    e = lab.threshold_edge(o, outcome="ret_o1_20")
    m["ret_o1_20_mean"] = float(e.get("thr_fold_mean", float("nan")))
    m["ret_o1_20_won"] = int(e.get("thr_folds_won", 0))
    m["ret_o1_20_n"] = int(e.get("thr_folds", 0))
    return m


# --------------------------------------------------------------------------- #
# 窓ごとの AUC（正例率も付ける。正例率が「当てずっぽう」の PR-AUC）
# --------------------------------------------------------------------------- #
def windows_on(o: pd.DataFrame, keys) -> pd.DataFrame:
    """窓ごとの PR / ROC / 正例率 / 件数。keys があればその行だけ。MIN_WINDOW_ROWS 未満・片方のクラスしか無い窓は落とす。"""
    empty = pd.DataFrame(columns=["fold", "pr", "roc", "rate", "n"])
    sub = o if keys is None else E68.on_rows(o, keys)
    if len(sub) == 0:
        return empty
    n = sub.groupby("fold").size()
    sub = sub[sub["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if len(sub) == 0:
        return empty
    w = AB.auc_by_window(sub)
    if len(w) == 0:
        return empty
    r = sub.groupby("fold").agg(rate=("label", "mean"), n=("label", "size")).reset_index()
    r["fold"] = r["fold"].astype(int)
    return w.merge(r, on="fold")


def run(algo: str, df: pd.DataFrame, keysets: dict, colsets: dict, cutoff, n_trials: int, stamp: dict,
        shifts: list, compute: bool) -> dict | None:
    recs = {}
    for arm in ARMS:
        recs[arm] = E68.tune(algo, arm, df, colsets[COLS_OF[arm]], cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
    params = {arm: recs[arm]["params"] for arm in ARMS}
    prod = {rs: {} for rs in keysets}
    for arm in ARMS:
        cs = COLS_OF[arm]
        o = E68.oof(algo, cs, df, colsets[cs], params[arm], 0, (E68.PROD_SEED,), compute)
        if o is None:
            return None
        for rs, keys in keysets.items():
            prod[rs][arm] = o if keys is None else E68.on_rows(o, keys)
    wins = {rs: {} for rs in keysets}
    for sh in shifts:
        for arm in ARMS:
            cs = COLS_OF[arm]
            o = E68.oof(algo, cs, df, colsets[cs], params[arm], sh, E68.SEEDS[algo], compute)
            if o is None:
                return None
            for rs, keys in keysets.items():
                wins[rs][(sh, arm)] = windows_on(o, keys)
    return {"recs": recs, "params": params, "prod": prod, "wins": wins}


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #
def report(results: dict, shifts: list, stamp: dict, keysets: dict, cover: dict) -> dict:
    algos = [a for a in ALGOS if results.get(a)]
    missing = [a for a in ALGOS if not results.get(a)]
    print("\n" + "=" * 78)
    print(f"結果のそろったモデル: {', '.join(algos) or 'なし'}"
          + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, "coverage": cover, "tuning": {}, "production_oof": {}, "windows": {}, "headline": {}}

    print("\n■ 1. 探索の CV（層別5分割。楽観側に出る。パラメータ選び用）")
    print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC")
    for a in algos:
        for arm in ARMS:
            rec = results[a]["recs"][arm]
            cv = rec["_cv"]
            fs = cv.get("fold_scores") or []
            print(f"  {a:<7}{arm:<4}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}"
                  f"{rec.get('_seconds', 0):>6}  " + " / ".join(f"{v:.4f}" for v in fs))
            summary["tuning"][f"{a}_{arm}"] = {k: cv.get(k) for k in ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
        print(f"  {'':<7}選ばれたパラメータ: " + " / ".join(f"{arm} {E68.short(results[a]['params'][arm])}" for arm in ARMS))

    for i, rs in enumerate(keysets):
        print(f"\n■ 2{'abc'[i]}. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。{ROWSETS[rs]}）")
        print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'件数':>8}{'正例率':>8}")
        for a in algos:
            ms = {}
            for arm in ARMS:
                o = results[a]["prod"][rs][arm]
                m = safe_metrics(o)
                ms[arm] = m
                rate = m["rate"]
                summary["production_oof"][f"{a}_{arm}_{rs}"] = {**{k: float(v) for k, v in m.items()}, "n": int(len(o))}
                print(f"  {a:<7}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
                      f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
                      f"{len(o):>8,}{rate*100:>7.1f}%")
            for x, y in PAIRS:
                print(f"  {a:<7}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}"
                      f"{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}"
                      f"{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    for i, rs in enumerate(keysets):
        rows_ = []
        for a in algos:
            for sh in shifts:
                w = None
                for arm in ARMS:
                    x = results[a]["wins"][rs][(sh, arm)]
                    x = x.rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
                    if arm != ARMS[0]:
                        x = x.drop(columns=["rate", "n"])
                    w = x if w is None else w.merge(x, on="fold")
                if w is None or len(w) == 0:
                    continue
                for _, r in w.iterrows():
                    rows_.append({"algo": a, "shift": sh, "fold": int(r["fold"]), "rate": float(r["rate"]), "n": int(r["n"]),
                                  **{f"{m}_{arm}": float(r[f"{m}_{arm}"]) for m in ("pr", "roc") for arm in ARMS}})
        s = pd.DataFrame(rows_)
        if not len(s):
            continue
        s.to_csv(E68.path(f"auc_by_window_{rs}.csv"), index=False)
        print(f"\n■ 3{'abc'[i]}. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り。logit は種1つ、"
              f"ほかは種3つの平均。{ROWSETS[rs]}・{MIN_WINDOW_ROWS}件以上の窓）")
        print(f"  {'':<7}{'窓':>4}{'正例率':>8}" + "".join(f"{'PR ' + arm:>9}" for arm in ARMS)
              + "".join(f"{'ROC ' + arm:>9}" for arm in ARMS))
        for a in algos:
            g = s[s["algo"] == a]
            print(f"  {a:<7}{len(g):>4}{g['rate'].mean()*100:>7.1f}%"
                  + "".join(f"{g[f'pr_{arm}'].mean():>9.4f}" for arm in ARMS)
                  + "".join(f"{g[f'roc_{arm}'].mean():>9.4f}" for arm in ARMS))
            summary["windows"][f"{a}_{rs}_mean"] = {"n": int(len(g)), "rate": float(g["rate"].mean()),
                                                   **{f"pr_{arm}": float(g[f"pr_{arm}"].mean()) for arm in ARMS},
                                                   **{f"roc_{arm}": float(g[f"roc_{arm}"].mean()) for arm in ARMS}}
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for a in algos:
            g = s[s["algo"] == a]
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                for x, y in PAIRS:
                    line, st = E68.pair(f"{a} {nm} {y}−{x}", g[f"{met}_{x}"].to_numpy(), g[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{a}_{met}_{y}-{x}_{rs}"] = st
            # 当てずっぽう（正例率）との差: 単独のモデルに情報があるか
            for arm in ("J", "E", "JE"):
                line, st = E68.pair(f"{a} PR {arm}−正例率", g["rate"].to_numpy(), g[f"pr_{arm}"].to_numpy())
                print(line)
                summary["windows"][f"{a}_pr_{arm}-rate_{rs}"] = st

    print("\n■ 4. 要約（本番と同じ OOF の PR-AUC と、正例率に対するリフト）")
    print(f"  {'':<7}{'行':<6}" + "".join(f"{arm:>14}" for arm in ARMS) + f"{'正例率':>8}")
    for a in algos:
        for rs in keysets:
            cells = []
            for arm in ARMS:
                m = summary["production_oof"][f"{a}_{arm}_{rs}"]
                cells.append(f"{m['pr_auc']:>7.4f}({m['lift']:>4.2f}x)")
            rate = summary["production_oof"][f"{a}_{ARMS[0]}_{rs}"]["rate"]
            summary["headline"][f"{a}_{rs}"] = {arm: summary["production_oof"][f"{a}_{arm}_{rs}"]["pr_auc"] for arm in ARMS}
            print(f"  {a:<7}{rs:<6}" + "".join(f"{c:>14}" for c in cells) + f"{rate*100:>7.1f}%")
    with open(E68.path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験72: 日証金と EDINET DB の特徴量だけのモデルの精度")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算するモデル")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0,
                    help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="行の充足だけ出して終わる")
    args = ap.parse_args(argv)
    E68.TAG = args.tag
    E68.RUN = f"{args.tag}_only"
    E68.DEADLINE = time.time() + args.budget_min * 60 if args.budget_min > 0 else None
    E68.TIMED_OUT = False
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    colsets = column_sets()

    print("=" * 78)
    print("実験72 日証金と EDINET DB の特徴量だけのモデル（5分割 CV の探索 + 本番と同じ OOF + 32窓）")
    for arm in ARMS:
        print(f"  {LABELS[arm]}: {len(colsets[COLS_OF[arm]])}列")
    print(f"  この回に計算: {', '.join(algos)} / 探索 {args.n_trials}試行 × {E68.N_SPLITS}分割（{E68.CV_SCHEME}）/ "
          f"窓のずらし {shifts}か月 / 種 {E68.SEEDS} / 上限 {args.budget_min:g}分 / 保存 {E68.RUN}_*")
    print("=" * 78)

    full, stamp = prepare(args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    for name in ("jsf", "ed"):
        miss = [c for c in colsets[name] if c not in full.columns]
        if miss:
            raise SystemExit(f"表に無い列（{name}）: {miss[:8]}")
    cover = coverage_report(full, stamp, colsets)
    if args.dry:
        return 0
    df = full
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    keysets = eval_keysets(df)
    print(f"\n  行 {len(df):,}（{df['Date'].min().date()} 〜 {df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(df['Date'] <= cutoff).sum():,}行。評価の行: " + " / ".join(
              f"{ROWSETS[rs]} {len(df) if k is None else len(k):,}" for rs, k in keysets.items()))

    results = {}
    for a in ALGOS:
        t0 = time.time()
        results[a] = run(a, df, keysets, colsets, cutoff, args.n_trials, stamp, shifts, compute=a in algos)
        if a in algos:
            log(f"[{a}] {'そろった' if results[a] else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    report(results, shifts, stamp, keysets, cover)
    if E68.TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {E68.OOF_DIR}/{E68.RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
