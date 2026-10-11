#!/usr/bin/env python3
"""
実験76: 財務指標の相対位置（市場全体の中・同じ33業種の中の百分位。research/relative_features.py）を本番の239列に
足すと、探索（5分割 CV）・本番と同じ作りの out-of-fold・32窓がどう変わるか（実験68 と同じ手順。腕は「列」の違い）。

運用者の問い（2026-10-11）「同業種内での相対的な位置がわかる特徴量は入っているか」→ 無い（株価・出来高の
業種相対だけ）。「サンプル外の市場全体の中での相対位置も入れたい。ノイズになるか」→ 参照を上場全銘柄に
する必要がある（学習データの行だけで業種内の順位を付けると母集団が 1〜2件）ので、市場全体の参照はむしろ
前提。ノイズかどうかはこの実験で測る（対照 P と比べる）。

腕（データは同じ1本。列だけが違う）
  A   いまの本番（239列）。A の列で探索
  C1  239 + 業種内の百分位 10列（rel_ind_*）を A のパラメータで（探索なし）
  C2  239 + 業種内 + 市場全体 20列（rel_ind_* + rel_mkt_*）を A のパラメータで
  P1  対照: C1 で足した 10列の値を同じ日付の行どうしで入れ替えたもの（列の情報を壊して列数だけそろえる。
      実験51・56・68 と同じ作り）を A のパラメータで
  P2  対照: C2 で足した 20列を入れ替えたものを A のパラメータで
  B   239 + 20列で探索し直し（日曜の週次実行がやること）
  C1−A・C2−A が「列を足すことの効果」、P1−A・P2−A が「列数が増えただけの幅」、C2−C1 が「市場全体の百分位を
  足すことの効果」、B−A が「再学習で起きること」

判定は docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）: 木3モデル（lgbm / xgb / cat）のうち 2つ以上で、
窓ごとの PR-AUC の差 C−A の平均が正で、対照 P−A の平均を上回り、C が上の窓が過半。C1 は P1 と、C2 は P2 と比べる。
ロジスティック回帰は参考（§7 の票に入れない）。

探索・評価は実験68 と同じ（50試行 × 5分割 year_cap_date・本番と同じ OOF（36/6/6か月・エンバーゴ20営業日・種42）・
窓のずらし 0/2/4か月 × 種3つの平均）。評価は全行と、参照（業種内・市場全体とも）の付いた行の両方で出す。

Actions の1回の上限に収まるよう、モデルを分けて回す（既定は lgbm だけ。運用者の「lgbm のみで良い」に合わせ、
§7 の票に要る xgb / cat は結果を見てから）。表（rel_* を付けた frame）は1回目に作って research/_data/oof/<tag>_*
に置き、以降の回はそれを使う。--budget-min（既定 290分）を過ぎたら新しい計算を始めずに終わる（続きは次の回）。
    exp=e76_relative_ab.py args="--algos lgbm"
    exp=e76_relative_ab.py args="--algos xgb"
    exp=e76_relative_ab.py args="--algos cat"
充足だけ見る: exp=e76_relative_ab.py args="--dry"
試運転:       exp=e76_relative_ab.py args="--tag e76smoke --algos lgbm --n-trials 2 --shifts 0"

公開ログには件数・割合・日付・精度だけを出す（財務の値は出さない）。本番の設定（research/lgbm_params.json、
research/multi_params.json、features.py）には書かない。
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
import features as F  # noqa: E402
import lab  # noqa: E402
import relative_features as RF  # noqa: E402
import train_model as T  # noqa: E402
import e68_edinet_ab as E68  # noqa: E402
from e44_shortsale import permuted  # noqa: E402
from e72_jsf_edinet_only import safe_metrics  # noqa: E402

log = E68.log
ALGOS = E68.ALGOS                       # lgbm / xgb / cat / logit
JUDGE_ALGOS = E68.JUDGE_ALGOS           # §7 の票は木3つ
DEFAULT_ALGOS = "lgbm"                  # 既定は lgbm だけ（運用者の「lgbm のみで良い」）
ARMS = ("A", "C1", "C2", "P1", "P2", "B")
COLS_OF = {"A": "base", "C1": "ind", "C2": "both", "P1": "perm_ind", "P2": "perm_both", "B": "both"}
PARAMS_OF = {"A": "A", "C1": "A", "C2": "A", "P1": "A", "P2": "A", "B": "B"}
TUNE_ARMS = ("A", "B")
LABELS = {"A": "A いまの本番（239列）",
          "C1": "C1 239 + 業種内の百分位 10列・A のパラメータ",
          "C2": "C2 239 + 業種内 + 市場全体 20列・A のパラメータ",
          "P1": "P1 対照（C1 の 10列を日付内で入れ替え）・A のパラメータ",
          "P2": "P2 対照（C2 の 20列を日付内で入れ替え）・A のパラメータ",
          "B": "B 239 + 20列・探索し直し"}
#: 表に出す差（「後 − 前」）
PAIRS = (("A", "C1"), ("A", "C2"), ("A", "P1"), ("A", "P2"), ("P1", "C1"), ("P2", "C2"), ("C1", "C2"),
         ("A", "B"), ("C2", "B"))
#: §7 の判定の組（候補, 対照）
JUDGE_PAIRS = (("C1", "P1"), ("C2", "P2"))
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261011
ROWSETS = {"all": "全行", "covered": "参照の付いた行（業種内・市場全体とも）"}
TAG = "e76"


def column_sets() -> dict:
    base = F.columns(F.DEFAULT_PRESET)
    ind, both = RF.columns("ind"), RF.columns("all")
    dup = sorted(set(base) & set(both))
    if dup:
        raise SystemExit(f"本番の列と重なる: {dup[:5]}")
    return {"base": base, "ind": base + ind, "both": base + both, "perm_ind": base + ind, "perm_both": base + both,
            "add_ind": ind, "add_both": both}


def covered_mask(df: pd.DataFrame) -> pd.Series:
    """業種内と市場全体の両方の百分位が（どれか1つの指標で）付いた行。"""
    ind = df[RF.columns("ind")].notna().any(axis=1)
    mkt = df[RF.columns("mkt")].notna().any(axis=1)
    return ind & mkt


def eval_keysets(df: pd.DataFrame) -> dict:
    return {"all": None, "covered": E68.keys_of(df[covered_mask(df)])}


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def prepare(rebuild: bool = False):
    """lab.frame に rel_*（相対位置）を付けた表（全行）と記録。"""
    pf, ps = E68.fpath("frame.parquet"), E68.fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}・"
            f"参照 {stamp['reference']['month_ends']}月末・付いた行 {stamp['covered']:,}）")
        return pd.read_parquet(pf), stamp
    os.makedirs(E68.OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    panel = RF.reference_panel(lab.DATA_DIR)
    ref = RF.coverage(panel)
    log(f"[data] 参照 {ref['month_ends']}月末（{ref['first']} 〜 {ref['last']}）・月末あたり "
        f"{ref['names']['median']:.0f}銘柄（{ref['names']['min']}〜{ref['names']['max']}）・"
        f"{RF.MIN_INDUSTRY}銘柄以上の業種 {ref['industries_ok']['median']:.0f}/{ref['industries']} / {time.time()-t0:.0f}秒")
    t0 = time.time()
    df = RF.attach(frame, panel)
    have = covered_mask(df)
    yr = df["Date"].dt.year
    log(f"[data] rel_* {len(RF.columns('all'))}列を付けた {len(df):,}行（参照の付いた行 {have.sum():,}・"
        f"{have.mean()*100:.1f}%）/ {time.time()-t0:.0f}秒")
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
             "rows": int(len(df)), "date_max": str(df["Date"].max().date()),
             "reference": ref, "metrics": list(RF.METRICS),
             "min_industry": RF.MIN_INDUSTRY, "min_market": RF.MIN_MARKET,
             "covered": int(have.sum()),
             "fill": {c: float(df[c].notna().mean()) for c in RF.columns("all")},
             "by_year": {str(y): {"rows": int((yr == y).sum()), "covered": int((have & (yr == y)).sum())}
                         for y in sorted(yr.unique())}}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1)
    return df, stamp


def coverage_report(df: pd.DataFrame, stamp: dict) -> dict:
    """参照の大きさと、付けた列の充足（年別）。件数と割合だけ。"""
    ref = stamp["reference"]
    print(f"\n■ 0. 参照と充足（表は {stamp['built_utc']} 作成）")
    print(f"  参照: {ref['month_ends']}月末（{ref['first']} 〜 {ref['last']}）・月末あたり {ref['names']['median']:.0f}銘柄"
          f"（{ref['names']['min']}〜{ref['names']['max']}）・業種 {ref['industries']}・"
          f"{stamp['min_industry']}銘柄以上の業種 {ref['industries_ok']['median']:.0f}"
          f"（{ref['industries_ok']['min']}〜{ref['industries_ok']['max']}）・業種の付いた割合 {ref['with_industry']*100:.1f}%")
    print("  参照の指標の充足（月末ごとの中央値）: " + "  ".join(
        f"{RF.METRIC_JA.get(m, m)} {v*100:.0f}%" for m, v in ref["fill"].items()))
    have = covered_mask(df)
    print(f"  母集団 {len(df):,}行・{df['Code'].nunique():,}銘柄 / 参照の付いた行 {have.sum():,}（{have.mean()*100:.1f}%）")
    yrs = sorted(df["Date"].dt.year.unique())
    print("  年別（付いた行 / 全行）: " + "  ".join(
        f"{y} {stamp['by_year'].get(str(y), {}).get('covered', 0):,}/{stamp['by_year'].get(str(y), {}).get('rows', 0):,}"
        for y in yrs))
    fill = {}
    for kind in RF.KINDS:
        r = df[RF.columns(kind)].notna().mean()
        fill[kind] = {"median": float(r.median()), "min": float(r.min()), "max": float(r.max())}
        print(f"  {RF.KIND_JA[kind]}の百分位 {len(r)}列の充足: 中央値 {r.median()*100:.1f}%・最小 {r.min()*100:.1f}%・"
              f"最大 {r.max()*100:.1f}%")
    return {"covered": int(have.sum()), "covered_share": float(have.mean()), "fill": fill}


# --------------------------------------------------------------------------- #
# 探索・out-of-fold（実験68 の関数）
# --------------------------------------------------------------------------- #
def run(algo: str, frames: dict, keysets: dict, colsets: dict, cutoff, n_trials: int, stamp: dict,
        shifts: list, compute: bool) -> dict | None:
    recs = {}
    for arm in TUNE_ARMS:
        cs = COLS_OF[arm]
        recs[arm] = E68.tune(algo, arm, frames[cs], colsets[cs], cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
    params = {arm: recs[PARAMS_OF[arm]]["params"] for arm in ARMS}
    prod = {rs: {} for rs in keysets}
    for arm in ARMS:
        cs = COLS_OF[arm]
        o = E68.oof(algo, cs, frames[cs], colsets[cs], params[arm], 0, (E68.PROD_SEED,), compute)
        if o is None:
            return None
        for rs, keys in keysets.items():
            prod[rs][arm] = o if keys is None else E68.on_rows(o, keys)
    wins = {rs: {} for rs in keysets}
    for sh in shifts:
        for arm in ARMS:
            cs = COLS_OF[arm]
            o = E68.oof(algo, cs, frames[cs], colsets[cs], params[arm], sh, E68.SEEDS[algo], compute)
            if o is None:
                return None
            for rs, keys in keysets.items():
                wins[rs][(sh, arm)] = E68.windows_on(o, keys)
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
    summary = {"data": {k: v for k, v in stamp.items() if k != "fill"}, "coverage": cover, "perm_seed": PERM_SEED,
               "tuning": {}, "production_oof": {}, "windows": {}, "judge7": {}}
    sec = {rs: "ab"[i] for i, rs in enumerate(keysets)}

    print(f"\n■ 1. 探索の CV（{E68.CV_SCHEME} 5分割。楽観側に出る。パラメータ選び用）")
    print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC")
    for a in algos:
        r = results[a]
        for arm in TUNE_ARMS:
            cv = r["recs"][arm]["_cv"]
            fs = cv.get("fold_scores") or []
            print(f"  {a:<7}{arm:<4}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}"
                  f"{r['recs'][arm].get('_seconds', 0):>6}  " + " / ".join(f"{v:.4f}" for v in fs))
            summary["tuning"][f"{a}_{arm}"] = {k: cv.get(k) for k in
                                               ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
        pa, pb = r["recs"]["A"]["params"], r["recs"]["B"]["params"]
        fa = r["recs"]["A"]["_cv"].get("fold_scores") or []
        fb = r["recs"]["B"]["_cv"].get("fold_scores") or []
        if fa and len(fa) == len(fb):
            d = np.array(fb) - np.array(fa)
            print(f"  {a:<7}B−A {d.mean():>+8.4f}{'':>8}"
                  f"{r['recs']['B']['_cv']['mean_roc_auc'] - r['recs']['A']['_cv']['mean_roc_auc']:>+8.4f}"
                  f"{'':>6}  分割ごと " + " / ".join(f"{v:+.4f}" for v in d) + f"（B が上 {(d > 0).sum()}/{len(d)}）")
        print(f"  {'':<7}選ばれたパラメータ: " + ("A と B で同じ" if E68.params_hash(pa) == E68.params_hash(pb)
                                              else f"A {E68.short(pa)} / B {E68.short(pb)}"))

    for rs in keysets:
        print(f"\n■ 2{sec[rs]}. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。{ROWSETS[rs]}）")
        print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'件数':>8}")
        for a in algos:
            ms = {}
            for arm in ARMS:
                o = results[a]["prod"][rs][arm]
                m = safe_metrics(o)
                ms[arm] = m
                summary["production_oof"][f"{a}_{arm}_{rs}"] = {k: float(v) for k, v in m.items()}
                print(f"  {a:<7}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
                      f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
                      f"{len(o):>8,}")
            for x, y in PAIRS:
                print(f"  {a:<7}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}"
                      f"{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}"
                      f"{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    for rs in keysets:
        rows_ = []
        for a in algos:
            for sh in shifts:
                w = None
                for arm in ARMS:
                    x = results[a]["wins"][rs][(sh, arm)].rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
                    w = x if w is None else w.merge(x, on="fold")
                if w is None or len(w) == 0:
                    continue
                for _, r in w.iterrows():
                    rows_.append({"algo": a, "shift": sh, "fold": int(r["fold"]),
                                  **{f"{m}_{arm}": float(r[f"{m}_{arm}"]) for m in ("pr", "roc") for arm in ARMS}})
        s = pd.DataFrame(rows_)
        if not len(s):
            continue
        s.to_csv(E68.path(f"auc_by_window_{rs}.csv"), index=False)
        note = f"・{E68.MIN_WINDOW_ROWS}件以上の窓" if keysets[rs] is not None else ""
        print(f"\n■ 3{sec[rs]}. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り。"
              f"logit は種1つ、ほかは種3つの平均。{ROWSETS[rs]}{note}）")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for a in algos:
            g = s[s["algo"] == a]
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                for x, y in PAIRS:
                    line, st = E68.pair(f"{a} {nm} {y}−{x}", g[f"{met}_{x}"].to_numpy(), g[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{a}_{met}_{y}-{x}_{rs}"] = st
        print(f"\n■ 4{sec[rs]}. §7（2026-09-26 改訂）の判定（窓の PR-AUC。{ROWSETS[rs]}）: "
              "平均 C−A が正 / 対照 P−A を上回る / C が上の窓が過半")
        for c, p in JUDGE_PAIRS:
            for a in algos:
                g = s[s["algo"] == a]
                j = E68.judge7((g[f"pr_{c}"] - g["pr_A"]).to_numpy(), (g[f"pr_{p}"] - g["pr_A"]).to_numpy())
                summary["judge7"][f"{a}_{c}_{rs}"] = j
                ref = "" if a in JUDGE_ALGOS else "（参考。§7 の票には入れない）"
                print(f"  {a:<7}{c}−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / "
                      f"{p}−A {j['mean_pa']:>+8.4f} {'○' if j['beats_placebo'] else '×'} / "
                      f"上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）{'○' if j['majority'] else '×'} "
                      f"→ {'満たす' if j['pass'] else '満たさない'}{ref}")
            judged = [a for a in algos if a in JUDGE_ALGOS]
            passed = sum(int(summary["judge7"][f"{a}_{c}_{rs}"]["pass"]) for a in judged)
            complete = all(a in algos for a in JUDGE_ALGOS)
            verdict = (("満たす（木3モデル中2つ以上）" if passed >= 2 else "満たさない（木3モデル中2つ未満）") if complete
                       else f"{passed}/{len(judged)} モデルが満たす（木3モデルがそろってから判定）")
            summary["judge7"][f"verdict_{c}_{rs}"] = {"passed": passed, "of": len(judged), "complete": complete,
                                                      "reference": [a for a in algos if a not in JUDGE_ALGOS]}
            print(f"  → §7 {c}: {verdict}")
    with open(E68.path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験76: 財務指標の相対位置（市場全体・業種内の百分位）を本番の239列に足す前後")
    ap.add_argument("--algos", default=DEFAULT_ALGOS, help="この回に計算するモデル（既定 lgbm。xgb / cat は別の回に）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0,
                    help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="参照と充足だけ出して終わる")
    args = ap.parse_args(argv)
    E68.TAG = args.tag
    E68.RUN = f"{args.tag}_rel"
    E68.DEADLINE = time.time() + args.budget_min * 60 if args.budget_min > 0 else None
    E68.TIMED_OUT = False
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    colsets = column_sets()

    print("=" * 78)
    print("実験76 財務指標の相対位置（市場全体・同じ33業種の中の百分位）を本番の239列に足す前後"
          "（5分割 CV の探索 + OOF + 32窓・対照 P）")
    print(f"  列: {F.DEFAULT_PRESET} {len(colsets['base'])}列 指紋 {F.signature(colsets['base'])} / "
          f"+ 業種内 {len(colsets['add_ind'])}列 → {len(colsets['ind'])}列 / + 市場全体 → {len(colsets['both'])}列 "
          f"指紋 {F.signature(colsets['both'])}")
    print(f"  指標: " + "・".join(RF.METRIC_JA.get(m, m) for m in RF.METRICS)
          + f" / 業種の母集団 {RF.MIN_INDUSTRY}銘柄以上・市場全体 {RF.MIN_MARKET}銘柄以上 / 参照は行の日付より前の月末")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  この回に計算: {', '.join(algos)} / 探索 {args.n_trials}試行 × {E68.N_SPLITS}分割（{E68.CV_SCHEME}）/ "
          f"窓のずらし {shifts}か月 / 種 {E68.SEEDS} / 対照の種 {PERM_SEED} / 上限 {args.budget_min:g}分 / 保存 {E68.RUN}_*")
    print("=" * 78)

    full, stamp = prepare(args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    miss = [c for c in colsets["both"] if c not in full.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    cover = coverage_report(full, stamp)
    if args.dry:
        return 0
    df = full
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    keysets = eval_keysets(df)
    print(f"\n  行 {len(df):,}（{df['Date'].min().date()} 〜 {df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(df['Date'] <= cutoff).sum():,}行。評価の行: " + " / ".join(
              f"{ROWSETS[rs]} {len(df) if k is None else len(k):,}" for rs, k in keysets.items()))
    t0 = time.time()
    frames = {"base": df, "ind": df, "both": df}
    for cs, add in (("perm_ind", "add_ind"), ("perm_both", "add_both")):
        frames[cs] = permuted(df, colsets[add], seed=PERM_SEED)
        frames[cs].attrs["built_utc"] = stamp["built_utc"]
    log(f"[data] 対照 P1 / P2: 足した {len(colsets['add_ind'])} / {len(colsets['add_both'])}列を日付内で入れ替えた"
        f"（種 {PERM_SEED}）/ {time.time()-t0:.0f}秒")

    results = {}
    for a in ALGOS:
        t0 = time.time()
        results[a] = run(a, frames, keysets, colsets, cutoff, args.n_trials, stamp, shifts, compute=a in algos)
        if a in algos:
            log(f"[{a}] {'そろった' if results[a] else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    report(results, shifts, stamp, keysets, cover)
    if E68.TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {E68.OOF_DIR}/{E68.RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
