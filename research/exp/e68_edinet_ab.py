#!/usr/bin/env python3
"""
実験68: EDINET DB の年次特徴量（research/edinet_features.py）を本番の239列に足すと、木3モデル + ロジスティック回帰の
探索（5分割 CV）・本番と同じ作りの out-of-fold・32窓がどう変わるか（実験67 と同じ手順。腕は「列」の違い）。

運用者の計画（2026-10-09）「EDINET はいま取れている行で A/B を先に1回回し、12月にそろった時点で確かめ直す」。
実験28（2026-09、本番153列・5モデル・B1/B2 の腕）の台本を、239列・木3モデル + ロジスティック回帰（運用者の指示
2026-10-09 夜「実験にもロジスティックを含めてください」。MLP は外したまま）・実験67 と同じ A/B/C の手順に作り直したもの。候補の列の中身と充足は docs/FEATURE_IDEAS_EDINET.md、
取得の仕組みと制約は docs/DATA_EDINETDB.md。

腕（データは同じ1本。列だけが違う）
  A  いまの本番（239列）。A の列で探索
  B  239 + ed（既定 all 523列 → 762列）で探索し直し → 日曜の週次実行がやること
  C  239 + ed を A のパラメータで（探索なし）→ 列を足したことそのものの効果
  P  対照: 239 + ed の値を同じ日付の行どうしで入れ替えたもの（列の情報を壊して列数だけそろえる。
     実験51・56 と同じ作り）を A のパラメータで。§7（2026-09-26 改訂）の「対照 P の改善を上回る」に使う
  C−A が「列を足すことの効果」、P−A が「列数が増えただけの幅」、B−A が「再学習で起きること」、
  B−C が「探索し直しの揺れ」

行（--rows）
  covered  EDINET の y0 が付いた行だけ（既定。全腕とも同じ行。運用者の「いま取れている行で」）。
           取得の順は母集団に多く出る銘柄が先（research/edinet_targets.txt）なので、付いた行は
           ブレイクの多い銘柄に寄る。12月に母集団がそろったら all で確かめ直す
  all      全行（付いていない行は ed_* が欠測。12月以降の確かめ直し用。評価は全行と付いた行の両方で出す）

列（--set。検定（実験22）の結果では選ばない。同じ窓で選ぶと楽観になる）
  all        research/edinet_features.py の全列 523（既定。本番に入れるなら「全部足す」が最も単純で、
             週次の探索がそのまま扱える）
  core       売上・営業利益・純利益・EPS の軌道 40列
  core+mcap  core + 時価総額との組み合わせ 7列（実験28 の既定）
  unique     core を除く 483列（J-Quants に無い明細・資本政策・人的資本・比率・時価総額の組み合わせ）

判定は docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）: 木3モデル（lgbm / xgb / cat。運用の合議と同じ）のうち
2つ以上で、窓ごとの PR-AUC の差 C−A の平均が正で、対照 P−A の平均を上回り、C が上の窓が過半（32窓なら 17以上）。
ロジスティック回帰は同じ表に並べる（§7 の票には入れず参考。決定的なので種は1つ）。B は「日曜がやること」の参考として
並べる（探索し直しの条件は §7 で外した）。

探索・評価は実験67 と同じ（50試行 × 5分割 year_cap_date・本番と同じ OOF（36/6/6か月・エンバーゴ20営業日・
種42）・窓のずらし 0/2/4か月 × 種3つの平均。logit は種1つ）。

Actions の1回の上限（330分）に収まるよう、モデルを分けて回す。表（ed_* を付けた frame）は1回目に作って
research/_data/oof/<tag>_* に置き、以降の回はそれを使う。列が多いので --budget-min（既定 290分）を過ぎたら
新しい計算を始めずに終わる（それまでの探索・out-of-fold は保存済み。同じ引数でもう1回回すと続きから。
actions/cache は job が成功したときだけ保存されるので、上限で打ち切られるより自分で止めるほうが安全）。
    exp=e68_edinet_ab.py args="--algos lgbm"
    exp=e68_edinet_ab.py args="--algos xgb"
    exp=e68_edinet_ab.py args="--algos cat"
    exp=e68_edinet_ab.py args="--algos logit"
充足だけ見る: exp=e68_edinet_ab.py args="--dry"
試運転:       exp=e68_edinet_ab.py args="--tag e68smoke --algos lgbm --n-trials 2 --shifts 0"

結果（2026-10-09、run 37913419095 lgbm 63分 / 37913461491 xgb 202分 / 37920515264 cat 103分 / 37943266657 logit 57分、
docs/MODEL_ADOPTION_RULES.md §29）: 付いた行 14,816（67.6%）で、列を足した C−A は 32窓の PR-AUC で lgbm −0.0049 ± 0.0025・
xgb −0.0056 ± 0.0028・cat +0.0003 ± 0.0034・logit −0.0263 ± 0.0037 と、どれも対照 P（値を日付内で入れ替え）と区別がつかない
（C−P は全部 ±1.5 SE 以内）。探索し直し（B）は列を足して下がったぶんを戻すだけ（B−A +0.0003〜+0.0034、15〜19/32）。
logit の大きな落ち方は P でも同じで、239列で選んだ正則化が 762列では効きすぎる列数の効果（探索し直すと戻る）。
§7 の票は木3モデル中 0 → 採用しない。12月に母集団がそろったら --rows all と、小さな組（--set core+mcap / unique）で確かめ直す。

公開ログには件数・割合・日付・精度だけを出す（財務の値は出さない）。本番の設定（research/lgbm_params.json、
research/multi_params.json、features.py）には書かない。
"""
from __future__ import annotations

import argparse
import hashlib
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
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e44_shortsale import permuted  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV     # 本番の retrain-weekly.yml と同じ（year_cap_date。前進分割は実験73 で検証中）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE   # 探索の目的関数（pr_auc。リフトは実験73 で検証中）
#: 2026-10-09 夜の運用者の指示「実験にもロジスティックを含める」（MLP は外したまま）
ALGOS = ("lgbm", "xgb", "cat", "logit")
#: §7 の票は木3モデル（運用の合議と同じ）。logit は同じ表に並べて参考にする
JUDGE_ALGOS = ("lgbm", "xgb", "cat")
#: logit は決定的なので種は1つ
SEEDS = {"lgbm": (42, 7, 123), "xgb": (42, 7, 123), "cat": (42, 7, 123), "logit": (42,)}
PROD_SEED = 42
ARMS = ("A", "B", "C", "P")
COLS_OF = {"A": "base", "B": "ed", "C": "ed", "P": "perm"}     # どの列の組（perm は ed と同じ列・値を入れ替え）
PARAMS_OF = {"A": "A", "B": "B", "C": "A", "P": "A"}           # どの探索のパラメータ
LABELS = {"A": "A いまの本番（239列）", "B": "B 239 + ed・探索し直し",
          "C": "C 239 + ed・A のパラメータ", "P": "P 対照（ed を日付内で入れ替え）・A のパラメータ"}
#: 表に出す差（左 − 右 ではなく「後 − 前」）
PAIRS = (("A", "C"), ("A", "P"), ("P", "C"), ("A", "B"), ("C", "B"))
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261009
MIN_WINDOW_ROWS = 100
SET_NAMES = ("all", "core", "core+mcap", "unique")
TAG = "e68"              # 表（frame / stamp）の頭。--tag で変える
RUN = "e68_covered_all"  # 探索・out-of-fold・要約の頭（tag_rows_set）。main で決める
#: 計算の時間の上限（時刻）。None なら無制限。tune / oof は新しい計算の前にこれを見る
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fpath(name: str) -> str:
    """表（rows・set によらない）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def path(name: str) -> str:
    """探索・out-of-fold・要約（rows・set ごと）。"""
    return os.path.join(OOF_DIR, f"{RUN}_{name}")


def params_hash(p: dict) -> str:
    return hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:8]


def keys_of(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([df["Code"].astype(str), pd.to_datetime(df["Date"])])


def on_rows(o: pd.DataFrame, keys: pd.MultiIndex) -> pd.DataFrame:
    return o[keys_of(o).isin(keys)].reset_index(drop=True)


def time_left() -> bool:
    """上限の手前か。過ぎていれば一度だけ知らせる。"""
    global TIMED_OUT
    if DEADLINE is None or time.time() < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min を過ぎたので、新しい計算は始めない（保存済みの分だけで報告。続きは次の回）")
    return False


def column_set(name: str) -> list:
    """--set の名前から ed_* の列（research/edinet_features.py の定義順）。"""
    core, mcap, every = EF.columns("core"), EF.columns("mcap"), EF.columns("all")
    sets = {"all": every, "core": core, "core+mcap": core + mcap,
            "unique": [c for c in every if c not in core]}
    if name not in sets:
        raise KeyError(f"未知の列の組: {name}. 利用可能: {SET_NAMES}")
    return sets[name]


def select_rows(df: pd.DataFrame, rows: str) -> pd.DataFrame:
    """--rows の行。covered は EDINET の y0 が付いた行だけ。"""
    if rows == "covered":
        return df[df["ed_fiscal_year"].notna()].reset_index(drop=True)
    if rows == "all":
        return df
    raise KeyError(f"未知の行の指定: {rows}")


def eval_keysets(df: pd.DataFrame, rows: str) -> dict:
    """評価する行の組。all のときは全行と付いた行の両方、covered のときは付いた行（= 全行）だけ。"""
    if rows == "all":
        return {"all": None, "covered": keys_of(df[df["ed_fiscal_year"].notna()])}
    return {"covered": None}


def judge7(d_ca: np.ndarray, d_pa: np.ndarray) -> dict:
    """
    §7（2026-09-26 改訂）の1モデル分。窓ごとの PR-AUC の差 C−A と P−A から
      平均 C−A > 0、平均 C−A > 平均 P−A、C が上の窓が過半（n//2 + 1 以上）
    """
    n = int(len(d_ca))
    wins = int((d_ca > 0).sum())
    need = n // 2 + 1
    out = {"n": n, "mean_ca": float(d_ca.mean()) if n else float("nan"),
           "mean_pa": float(d_pa.mean()) if n else float("nan"), "wins": wins, "need": need}
    out["positive"] = bool(n and out["mean_ca"] > 0)
    out["beats_placebo"] = bool(n and out["mean_ca"] > out["mean_pa"])
    out["majority"] = bool(n and wins >= need)
    out["pass"] = out["positive"] and out["beats_placebo"] and out["majority"]
    return out


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def prepare(rebuild: bool = False):
    """lab.frame に ed_* を付けた表（全行）と記録。"""
    pf, ps = fpath("frame.parquet"), fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}・"
            f"EDINET {stamp['n_codes']:,}社・y0 の付いた行 {stamp['covered']:,}）")
        return pd.read_parquet(pf), stamp
    if not os.path.exists(EF.FIN):
        raise SystemExit(f"{EF.FIN} がありません（fetch-edinetdb.yml の成果物。Release data-raw の edinet_*）")
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    fin = EF.load_fin(EF.FIN)
    df = EF.attach(frame, EF.feature_frame(EF.annual_panel(fin)))
    have = df["ed_fiscal_year"].notna()
    yr = df["Date"].dt.year
    log(f"[data] ed_* を付けた {len(df):,}行（EDINET {fin[EF.KEY].nunique():,}社・y0 の付いた行 {have.sum():,}・"
        f"{have.mean()*100:.1f}%）/ {time.time()-t0:.0f}秒")
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
             "rows": int(len(df)), "date_max": str(df["Date"].max().date()),
             "n_codes": int(fin[EF.KEY].nunique()), "covered": int(have.sum()),
             "covered_codes": int(df.loc[have, "Code"].nunique()),
             "covered_by_year": {str(y): int((have & (yr == y)).sum()) for y in sorted(yr.unique())},
             "rows_by_year": {str(y): int((yr == y).sum()) for y in sorted(yr.unique())},
             "ed_columns": len(EF.columns("all"))}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1)
    return df, stamp


def coverage_report(df: pd.DataFrame, stamp: dict, ed: list, set_name: str) -> dict:
    """y0 の付き方（年別）と、足す列の充足（付いた行の中で。グループごとの中央値）。件数と割合だけ。"""
    have = df["ed_fiscal_year"].notna()
    print(f"\n■ 0. EDINET の付き方（表は {stamp['built_utc']} 作成・EDINET {stamp['n_codes']:,}社）")
    print(f"  母集団 {len(df):,}行・{df['Code'].nunique():,}銘柄 / y0 の付いた行 {have.sum():,}（{have.mean()*100:.1f}%）・"
          f"{df.loc[have, 'Code'].nunique():,}銘柄")
    yrs = sorted(df["Date"].dt.year.unique())
    print("  年別（付いた行 / 全行）: " + "  ".join(
        f"{y} {stamp['covered_by_year'].get(str(y), 0):,}/{stamp['rows_by_year'].get(str(y), 0):,}" for y in yrs))
    groups = [("core", EF.columns("core")), ("detail", EF.columns("detail")), ("capital", EF.columns("capital")),
              ("people", EF.columns("people")), ("ratio", EF.columns("ratio")),
              ("ratio_chg", EF.columns("ratio_chg")), ("mcap", EF.columns("mcap"))]
    fill = {}
    print(f"  足す列 {set_name}（{len(ed)}列）の充足（付いた行の中で。グループごとに列の中央値・最小）:")
    sub = df.loc[have]
    for g, cols in groups:
        cols = [c for c in cols if c in ed]
        if not cols:
            continue
        r = sub[cols].notna().mean()
        fill[g] = {"columns": len(cols), "median": float(r.median()), "min": float(r.min()),
                   "under_10pct": int((r < 0.10).sum())}
        print(f"    {g:<10}{len(cols):>4}列  中央値 {r.median()*100:5.1f}%  最小 {r.min()*100:5.1f}%  "
              f"充足 10% 未満 {int((r < 0.10).sum())}列")
    return {"covered": int(have.sum()), "covered_share": float(have.mean()), "fill": fill}


# --------------------------------------------------------------------------- #
# 探索（本番の週次実行と同じ関数・条件）
# --------------------------------------------------------------------------- #
def tune(algo: str, arm: str, frame: pd.DataFrame, cols: list, cutoff, n_trials: int,
         stamp: dict, compute: bool = True):
    p = path(f"params_{algo}_{arm}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if (rec.get("_n_trials") == n_trials and rec.get("_cutoff") == str(cutoff.date())
                and rec.get("_data") == stamp["built_utc"]):
            return rec
    if not compute or not time_left():
        return None
    sub = frame[(frame["Date"] <= cutoff) & frame["label"].notna()]
    t0 = time.time()
    if algo == "lgbm":
        log(f"  [{algo} {arm}] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列）")
        params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS,
                             embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME, objective=CV_OBJECTIVE, model="classifier",
                             verbose=False)
        rec = {"params": tuning.params_for("x", store={"x": params}), "_cv": dict(tuning.LAST_CV)}
    else:
        log(f"  [{algo} {arm}] 探索（tuning_multi.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列・"
            f"前処理 {TM.preprocess_version(algo) or 'なし'}）")
        TM.STUDY_DB = path(f"optuna_{arm}.db")
        rec = TM.tune(algo, sub, cols, n_trials=n_trials, n_splits=N_SPLITS, verbose=False, scheme=CV_SCHEME,
                      objective=CV_OBJECTIVE)
    rec["_seconds"] = round(time.time() - t0)
    rec["_n_trials"] = n_trials
    rec["_cutoff"] = str(cutoff.date())
    rec["_data"] = stamp["built_utc"]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    cv = rec["_cv"]
    log(f"  [{algo} {arm}] {rec['_seconds']}秒 / CV PR-AUC {cv['mean_pr_auc']:.4f}（±{cv['std']:.4f}）"
        f" / ROC {cv['mean_roc_auc']:.4f}")
    return rec


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #
def oof(algo: str, colset: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds,
        compute: bool = True):
    """列の組（base / ed / perm）× パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(frame.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{algo}_c{colset}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    missing = [f for f in files if not os.path.exists(f)]
    if missing and (not compute or not time_left()):
        return None
    folds = E41.folds_for(frame["Date"], shift) if missing else None
    parts = []
    for sd, f in zip(seeds, files):
        if os.path.exists(f):
            parts.append(pd.read_parquet(f))
            continue
        if not time_left():
            return None
        t0 = time.time()
        o = E41.oof_folds(algo, frame, cols, params, sd, folds)
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {algo} 列{colset} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def windows_on(o: pd.DataFrame, keys: pd.MultiIndex | None) -> pd.DataFrame:
    """窓ごとの AUC。keys があればその行だけで、MIN_WINDOW_ROWS 未満の窓は落とす。"""
    sub = o if keys is None else on_rows(o, keys)
    if len(sub) == 0:
        return pd.DataFrame(columns=["fold", "pr", "roc"])
    n = sub.groupby("fold").size()
    sub = sub[sub["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if len(sub) == 0:
        return pd.DataFrame(columns=["fold", "pr", "roc"])
    return AB.auc_by_window(sub)


def pair(name: str, a: np.ndarray, b: np.ndarray) -> tuple:
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    line = (f"  {name:<24}{a.mean():>9.4f}{b.mean():>9.4f}{d.mean():>+10.4f}{se:>9.4f}"
            f"{int((d > 0).sum()):>5}/{len(d):<4}{int((d == 0).sum()):>6}")
    return line, {"n": int(len(d)), "mean_diff": float(d.mean()), "se": float(se),
                  "wins": int((d > 0).sum()), "ties": int((d == 0).sum())}


def run(algo: str, frames: dict, keysets: dict, colsets: dict, cutoff, n_trials: int, stamp: dict,
        shifts: list, compute: bool) -> dict | None:
    recs = {}
    for arm in ("A", "B"):
        recs[arm] = tune(algo, arm, frames[COLS_OF[arm]], colsets[COLS_OF[arm]], cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
    params = {arm: recs[PARAMS_OF[arm]]["params"] for arm in ARMS}
    prod = {rs: {} for rs in keysets}
    for arm in ARMS:
        cs = COLS_OF[arm]
        o = oof(algo, cs, frames[cs], colsets[cs], params[arm], 0, (PROD_SEED,), compute)
        if o is None:
            return None
        for rs, keys in keysets.items():
            prod[rs][arm] = o if keys is None else on_rows(o, keys)
    wins = {rs: {} for rs in keysets}
    for sh in shifts:
        for arm in ARMS:
            cs = COLS_OF[arm]
            o = oof(algo, cs, frames[cs], colsets[cs], params[arm], sh, SEEDS[algo], compute)
            if o is None:
                return None
            for rs, keys in keysets.items():
                wins[rs][(sh, arm)] = windows_on(o, keys)
    return {"recs": recs, "params": params, "prod": prod, "wins": wins}


def report(results: dict, shifts: list, stamp: dict, keysets: dict, cover: dict, set_name: str, rows: str) -> dict:
    algos = [a for a in ALGOS if results.get(a)]
    missing = [a for a in ALGOS if not results.get(a)]
    print("\n" + "=" * 78)
    print(f"結果のそろったモデル: {', '.join(algos) or 'なし'}"
          + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, "coverage": cover, "rows": rows, "set": set_name, "perm_seed": PERM_SEED,
               "tuning": {}, "production_oof": {}, "windows": {}, "judge7": {}}
    titles = {"all": "全行", "covered": "y0 の付いた行だけ"}
    sec = {rs: ("" if len(keysets) == 1 else "ab"[i]) for i, rs in enumerate(keysets)}

    print("\n■ 1. 探索の CV（層別5分割。楽観側に出る。パラメータ選び用）")
    print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC / パラメータ")
    for a in algos:
        r = results[a]
        for arm in ("A", "B"):
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
                  f"{'':>6}  分割ごと " + " / ".join(f"{v:+.4f}" for v in d)
                  + f"（B が上 {(d > 0).sum()}/{len(d)}）")
        print(f"  {'':<7}選ばれたパラメータ: " + ("A と B で同じ" if params_hash(pa) == params_hash(pb)
                                              else f"A {short(pa)} / B {short(pb)}"))

    for rs in keysets:
        print(f"\n■ 2{sec[rs]}. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。{titles[rs]}）")
        print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
        for a in algos:
            ms = {}
            for arm in ARMS:
                o = results[a]["prod"][rs][arm]
                m = metrics(o)
                ms[arm] = m
                summary["production_oof"][f"{a}_{arm}_{rs}"] = {k: float(v) for k, v in m.items()}
                print(f"  {a:<7}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}"
                      f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
                      f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt"
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
        s.to_csv(path(f"auc_by_window_{rs}.csv"), index=False)
        note = f"・{MIN_WINDOW_ROWS}件以上の窓" if keysets[rs] is not None else ""
        print(f"\n■ 3{sec[rs]}. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り。"
              f"logit は種1つ、ほかは種3つの平均。{titles[rs]}{note}）")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for a in algos:
            g = s[s["algo"] == a]
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                for x, y in PAIRS:
                    line, st = pair(f"{a} {nm} {y}−{x}", g[f"{met}_{x}"].to_numpy(), g[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{a}_{met}_{y}-{x}_{rs}"] = st
        print(f"\n■ 4{sec[rs]}. §7（2026-09-26 改訂）の判定（窓の PR-AUC。{titles[rs]}）: "
              "平均 C−A が正 / 対照 P−A を上回る / C が上の窓が過半")
        for a in algos:
            g = s[s["algo"] == a]
            j = judge7((g["pr_C"] - g["pr_A"]).to_numpy(), (g["pr_P"] - g["pr_A"]).to_numpy())
            summary["judge7"][f"{a}_{rs}"] = j
            ref = "" if a in JUDGE_ALGOS else "（参考。§7 の票には入れない）"
            print(f"  {a:<7}C−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / "
                  f"P−A {j['mean_pa']:>+8.4f} {'○' if j['beats_placebo'] else '×'} / "
                  f"上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）{'○' if j['majority'] else '×'} "
                  f"→ {'満たす' if j['pass'] else '満たさない'}{ref}")
        judged = [a for a in algos if a in JUDGE_ALGOS]
        passed = sum(int(summary["judge7"][f"{a}_{rs}"]["pass"]) for a in judged)
        complete = all(a in algos for a in JUDGE_ALGOS)
        verdict = (("満たす（木3モデル中2つ以上）" if passed >= 2 else "満たさない（木3モデル中2つ未満）") if complete
                   else f"{passed}/{len(judged)} モデルが満たす（木3モデルがそろってから判定）")
        summary["judge7"][f"verdict_{rs}"] = {"passed": passed, "of": len(judged), "complete": complete,
                                              "reference": [a for a in algos if a not in JUDGE_ALGOS]}
        print(f"  → §7: {verdict}")
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    return summary


_FIXED_KEYS = {"objective", "boosting_type", "n_jobs", "verbose", "random_state", "subsample_freq"}


def short(p: dict) -> str:
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return "{" + ", ".join(f"{k} {fmt(v)}" for k, v in sorted(p.items()) if k not in _FIXED_KEYS) + "}"


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE
    ap = argparse.ArgumentParser(description="実験68: EDINET の年次特徴量を本番の239列に足す前後")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算するモデル")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--rows", choices=("covered", "all"), default="covered",
                    help="covered: y0 の付いた行だけ（既定）/ all: 全行（12月以降の確かめ直し用）")
    ap.add_argument("--set", choices=SET_NAMES, default="all", help="足す ed_* の列の組")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0,
                    help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="行の充足だけ出して終わる（取得の進み具合の確認）")
    args = ap.parse_args(argv)
    TAG = args.tag
    RUN = f"{TAG}_{args.rows}_{args.set.replace('+', '_')}"
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    base = F.columns(F.DEFAULT_PRESET)
    ed = column_set(args.set)
    dup = sorted(set(base) & set(ed))
    if dup:
        raise SystemExit(f"本番の列と重なる: {dup[:5]}")
    colsets = {"base": base, "ed": base + ed, "perm": base + ed}

    print("=" * 78)
    print("実験68 EDINET の年次特徴量を本番の239列に足す前後（5分割 CV の探索 + OOF + 32窓・対照 P）")
    print(f"  列: {F.DEFAULT_PRESET} {len(base)}列 指紋 {F.signature(base)} / + ed {args.set} {len(ed)}列 "
          f"→ {len(base) + len(ed)}列 指紋 {F.signature(base + ed)}")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  行: {args.rows} / この回に計算: {', '.join(algos)} / 探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}）/ "
          f"窓のずらし {shifts}か月 / 種 {SEEDS} / 対照の種 {PERM_SEED} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    full, stamp = prepare(args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    miss = [c for c in colsets["ed"] if c not in full.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    cover = coverage_report(full, stamp, ed, args.set)
    if args.dry:
        return 0
    df = select_rows(full, args.rows)
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    have = df["ed_fiscal_year"].notna()
    print(f"\n  この回の行 {len(df):,}（{args.rows}。y0 の付いた行 {have.sum():,}・{have.mean()*100:.1f}%・"
          f"{df['Date'].min().date()} 〜 {df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(df['Date'] <= cutoff).sum():,}行（付いた行 {(have & (df['Date'] <= cutoff)).sum():,}）")
    if len(df) < 2000:
        print("  ※ 2,000行未満。窓ごとの判定には足りない")
    t0 = time.time()
    perm = permuted(df, ed, seed=PERM_SEED)
    perm.attrs["built_utc"] = stamp["built_utc"]
    log(f"[data] 対照 P: ed {len(ed)}列を日付内で入れ替えた（種 {PERM_SEED}）/ {time.time()-t0:.0f}秒")
    frames = {"base": df, "ed": df, "perm": perm}
    keysets = eval_keysets(df, args.rows)

    results = {}
    for a in ALGOS:
        t0 = time.time()
        results[a] = run(a, frames, keysets, colsets, cutoff, args.n_trials, stamp, shifts, compute=a in algos)
        if a in algos:
            log(f"[{a}] {'そろった' if results[a] else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    report(results, shifts, stamp, keysets, cover, args.set, args.rows)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
