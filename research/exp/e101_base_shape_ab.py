#!/usr/bin/env python3
"""
実験101: 「抜いた78週高値の古さ」と「前回の更新日の高値に対する今回の高値」の2列（下の shape_columns）を本番の239列に
足すと、木3モデル + ロジスティック回帰の探索（5分割 CV）・本番と同じ作りの out-of-fold・32窓がどう変わるか（実験89 と
同じ手順。腕は「列」の違い）。

運用者の質問（2026-10-17）「前回の高値更新からの日数（base_length）が1000日を超えるものがある。78週ぶりの高値更新が
母集団なので最高でも約200日と思っていた」→ 定義を確かめた（probe_base_length、run 39042976197。上限は78週の窓では決まらず、
長く下げた後の戻りでは何年にもなる）→ 運用者の決定「戻りの度合いも足して検証」。運用者の方針「いきなり変えるよりも、
まずは検証してほしいです。本番導入するかはその結果次第です」→ 本番の設定（research/features.py・research/lgbm_params.json・
research/multi_params.json）と build_dataset.py には書かない。2列はこの台本の中で日足から作る。

足す2列（どちらも予測日 t の引けまでに分かる値だけで作る。t より先は見ない）
  broken_high_age  今回抜いた「それまでの78週高値」（t の前の 368営業日の日中の高値の最大 = high52w_prior）を付けた日から
                   t まで何営業日か。同じ高値が複数あれば一番新しい日。窓の中なので 1〜368（base_length は上限が無い）
  prev_nh_ratio    1つ前に78週高値を更新した日（base_length の起点）の日中の高値に対する、t の日中の高値（%）。100 未満は
                   前回の更新の高値にまだ届いていない戻り（長い下げからの戻り）、100 以上は前回の更新を超えた。
                   1つ前の更新がデータの中に無い行（base_length が欠測）は欠測
  高値は分割を調整した値（price_panel の high = AdjH、無ければ H）。数え方は probe_base_length と同じ（再計算の一致は
  そちらで 185,353行すべて確かめた）

腕（データは同じ1本。列だけが違う）
  A  いまの本番（239列）。A の列で探索
  B  239 + 2列で探索し直し → 日曜の週次実行がやること
  C  239 + 2列を A のパラメータで（探索なし）→ 列を足したことそのものの効果
  P  対照: 2列の値を同じ日付の行どうしで入れ替えたもの（種 PERM_SEED）を A のパラメータで。§7 の「対照 P の改善を
     上回る」に使う
  C−A が「列を足すことの効果」、P−A が「列数が増えただけの幅」、B−A が「再学習で起きること」、B−C が「探索し直しの揺れ」

ラベル（--k。1回の実行で1つ。両方回す）
  1.5  到達しきい値 1.5σ（既定）。選定「1.5σ の3本がすべて95以上」のモデルのラベル
  1.2  1.2σ。基準モデル（1.2σ の LightGBM。画面の帯・順位・シート）と、§7 がもともと対象にしているモデルのラベル
  ラベル以外（列・探索・窓・種）は同じ。1.5σ はデータセットの材料から引き直す（B.rise_label_for_k。週次の学習と同じ）

行: 学習も評価も全行（2列は全行で作れる。prev_nh_ratio は base_length が欠測の約15% で欠測、モデルは欠測として扱う）

判定（回す前に決めた。2026-10-17）
  docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）をラベルごとに当てる: 木3モデル（lgbm / xgb / cat）のうち2つ以上で、
  窓ごとの PR-AUC の差 C−A の平均が正・対照 P−A の平均を上回る・C が上の窓が過半（n//2 + 1 以上）。
  本番に入れる案として運用者に出すのは、1.5σ と 1.2σ の両方で §7 を満たすとき。片方だけなら、結果を見せて運用者が決める。
  logit は同じ表に並べる（§7 の票には入れない。決定的なので種は1つ）。B は日曜がやることの参考。
  **変更（2026-10-17、結果を見る前）**: 運用者の指示「検証は、lgbmの1.5σのみでよいです」により、回すのは
  `--k 1.5 --algos lgbm` の1本だけ（2本目の xgb は待ちのうちに取り消した）。判定は lgbm 1.5σ が §7 の1モデル分の3条件
  （窓ごとの C−A の平均が正・対照 P−A の平均を上回る・C が上の窓が過半）を満たすか。木3モデルの票は数えられないので、
  本番に入れるかは結果を見て運用者が決める（ログの「木3モデルがそろってから判定」の行はこの回では使わない）。

結果（2026-10-17。lgbm 1.5σ、run 39045199490。docs/MODEL_ADOPTION_RULES.md §62）: 32窓の PR-AUC で C−A +0.0019（SE 0.0018、
上の窓 21/32）・対照 P−A +0.0015 → §7 の1モデル分の3条件は字面では満たす。ただし本物と入れ替えの差 C−P は +0.0004（SE 0.0014、
18/32）で、2列そのものの情報は見えない。本番と同じ OOF でも C−A +0.0023・P−A +0.0022。
運用者の決定（2026-10-17）: 本番に入れる。2列は build_dataset._breakout_shape が作るようになった（この台本の shape_columns と
同じ値。tests/test_breakout_shape.py・research/exp/probe_breakout_shape.py）。この台本は当時の比べ方の記録として残す。

出すもの: 2列の分布と、値の区切りごとの正例率（件数・割合だけ）、探索の CV（分割ごと）、本番と同じ OOF（種42・ずらし0）、
窓ごとの差の平均・SE・上の窓の数、§7 の判定、窓ごとの値の一覧（運用者の指示「各cv（異なる窓）でもみたいです」）。
公開ログには件数・割合・日付・精度だけ（銘柄・値は出さない）。

    exp=e101_base_shape_ab.py args="--k 1.5 --algos lgbm"      ← 回すのはこの1本（上の変更）
    （もとの計画: 同じく xgb・cat・logit と、--k 1.2 で4回）
列の付き方だけ見る: exp=e101_base_shape_ab.py args="--dry"
試運転:             exp=e101_base_shape_ab.py args="--tag e101smoke --algos lgbm --n-trials 2 --shifts 0"

Run Experiment は同じ実験を同時に1本しか走らせず、待ちに置けるのも1本だけ（3本目を入れると待ちの1本が取り消される）
なので、1本ずつ起動する。表（2列を付けた frame）は1回目に作って research/_data/oof/<tag>_* に置き、以降の回は
actions/cache で引き継いだそれを使う（どの回も同じ行で比べる）。--budget-min（既定 290分）を過ぎたら新しい計算を
始めずに終わる（保存済みの探索・out-of-fold は次の回が続きから使う）。
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
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e44_shortsale import permuted  # noqa: E402
from e68_edinet_ab import judge7, keys_of, on_rows, pair, params_hash, short  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV       # 本番の retrain-weekly.yml と同じ（year_cap_date）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
ALGOS = ("lgbm", "xgb", "cat", "logit")
#: §7 の票は木3モデル。logit は同じ表に並べて参考にする
JUDGE_ALGOS = ("lgbm", "xgb", "cat")
SEEDS = {"lgbm": (42, 7, 123), "xgb": (42, 7, 123), "cat": (42, 7, 123), "logit": (42,)}
PROD_SEED = 42
ARMS = ("A", "B", "C", "P")
COLS_OF = {"A": "base", "B": "new", "C": "new", "P": "perm"}
PARAMS_OF = {"A": "A", "B": "B", "C": "A", "P": "A"}
LABELS = {"A": "A いまの本番（239列）", "B": "B 239 + 2列・探索し直し",
          "C": "C 239 + 2列・A のパラメータ", "P": "P 対照（2列を日付内で入れ替え）・A のパラメータ"}
PAIRS = (("A", "C"), ("A", "P"), ("P", "C"), ("A", "B"), ("C", "B"))
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261017
MIN_WINDOW_ROWS = 100
KS = (1.5, 1.2)
#: 判定に使う評価の行（回す前に決めた。2列は全行で作れるので全行）
PRIMARY_ROWS = "all"
#: 足す2列（下の shape_columns が作る）と、画面・ログ用の名前
NEW_COLS = ("broken_high_age", "prev_nh_ratio")
NEW_JA = {"broken_high_age": "抜いた78週高値の古さ（営業日）", "prev_nh_ratio": "前回の更新日の高値に対する今回の高値（%）"}
#: 値の区切り（正例率を見る。回す前に決めた）
BINS = {"broken_high_age": (0, 5, 20, 60, 120, 250, 368),
        "prev_nh_ratio": (0, 60, 80, 95, 100, 105, 120, float("inf"))}
TAG = "e101"
RUN = "e101_k15"
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fpath(name: str) -> str:
    """表（ラベルによらない）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def path(name: str) -> str:
    """探索・out-of-fold・要約（ラベルごと）。"""
    return os.path.join(OOF_DIR, f"{RUN}_{name}")


def k_tag(k: float) -> str:
    return f"k{int(round(k * 10)):02d}"


def time_left() -> bool:
    global TIMED_OUT
    if DEADLINE is None or time.time() < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min を過ぎたので、新しい計算は始めない（保存済みの分だけで報告。続きは次の回）")
    return False


def eval_keysets(df: pd.DataFrame) -> dict:
    """評価する行の組（全行だけ）。"""
    return {"all": None}


def with_label(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """k のラベルに引き直した表（1.2σ はデータセットのラベルのまま）。ラベルの決まらない行は落とす。"""
    if abs(k - B.VOL_NORM_K) < 1e-9:
        return df[df["label"].notna()].reset_index(drop=True)
    out = df.copy()
    out["label"] = B.rise_label_for_k(out, k)
    return out[out["label"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 足す2列
# --------------------------------------------------------------------------- #
def peak_age(high: np.ndarray, p: int, w: int) -> float:
    """位置 p の行が抜いた「それまでの w 行の高値の最大」を付けた行から何行か（同じ値が複数なら一番新しいもの）。"""
    win = high[max(0, p - w):p]
    if not len(win) or np.all(np.isnan(win)):
        return np.nan
    r = len(win) - 1 - int(np.argmax(win[::-1] == np.nanmax(win)))   # 最大を付けた一番新しい行（窓の中の位置）
    return float(len(win) - r)


def shape_columns(keys: pd.DataFrame, panel: pd.DataFrame, w: int = B.HIGH_WINDOW) -> pd.DataFrame:
    """
    keys（Code, Date）の行ごとに2列を作る（keys の並びのまま）。panel は銘柄・日付の順に並んだ日足の表で、Code・Date・
    high・base_length（build_dataset.add_breakout_context の値）を持つこと。panel に無い行は欠測。
    """
    p = panel[["Code", "Date", "high", "base_length"]].copy()
    p["Code"] = p["Code"].astype(str)
    p["Date"] = pd.to_datetime(p["Date"])
    p["_pos"] = p.groupby("Code", sort=False).cumcount()
    k = keys[["Code", "Date"]].copy()
    k["Code"] = k["Code"].astype(str)
    k["Date"] = pd.to_datetime(k["Date"])
    m = k.merge(p[["Code", "Date", "_pos", "base_length"]], on=["Code", "Date"], how="left")
    highs = {c: g["high"].to_numpy(dtype=float) for c, g in p.groupby("Code", sort=False)}
    age, ratio = [], []
    for code, pos, bl in zip(m["Code"], m["_pos"], m["base_length"]):
        h = highs.get(code)
        if h is None or pd.isna(pos):
            age.append(np.nan)
            ratio.append(np.nan)
            continue
        pos = int(pos)
        age.append(peak_age(h, pos, w))
        q = pos - int(bl) if pd.notna(bl) else -1
        ratio.append(h[pos] / h[q] * 100.0 if 0 <= q < pos and h[q] > 0 and np.isfinite(h[pos]) else np.nan)
    return pd.DataFrame({"broken_high_age": age, "prev_nh_ratio": ratio})


def build_panel(data_dir: str) -> pd.DataFrame:
    """build_dataset.build と同じ作りの日足の表（78週の窓・更新日・base_length まで）。"""
    bars = B.load_parts("bars", data_dir)
    segments = B.load_market_segments(data_dir)
    gm = B.general_market_start(bars, segments)
    start = gm.loc[gm["tpm"]].set_index("Code")["start"] if B.GENERAL_MARKET_START else None
    return B.add_breakout_context(B.mark_new_highs(B.price_panel(bars, start=start)))


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def prepare(rebuild: bool = False):
    """lab.frame に2列を付けた表（全行。ラベルは 1.2σ のまま）と記録。"""
    pf, ps = fpath("frame.parquet"), fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}）")
        return pd.read_parquet(pf), stamp
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    panel = build_panel(lab.DATA_DIR)
    panel["Code"] = panel["Code"].astype(str)
    # 表の base_length と、組み直した日足の base_length が全行一致するか（同じ作りであることの確かめ）
    chk = frame[["Code", "Date", "base_length"]].merge(panel[["Code", "Date", "base_length"]], on=["Code", "Date"],
                                                         how="left", suffixes=("", "_panel"))
    same = int(((chk["base_length"] == chk["base_length_panel"])
                | (chk["base_length"].isna() & chk["base_length_panel"].isna())).sum())
    if same != len(frame):
        raise SystemExit(f"組み直した日足の base_length が表と合わない: {len(frame) - same:,}行（作りが違う）")
    shape = shape_columns(frame[["Code", "Date"]], panel)
    # 2026-10-17 からはデータセットにも同じ2列がある（build_dataset._breakout_shape）。この台本の数え方で付け直す
    frame = frame.drop(columns=[c for c in NEW_COLS if c in frame.columns])
    df = pd.concat([frame.reset_index(drop=True), shape], axis=1)
    yr = df["Date"].dt.year
    log(f"[data] 2列を付けた {len(df):,}行（base_length が表と全行一致）/ {time.time()-t0:.0f}秒")
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
             "rows": int(len(df)), "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
             "rows_by_year": {str(y): int((yr == y).sum()) for y in sorted(yr.unique())},
             "missing": {c: int(df[c].isna().sum()) for c in NEW_COLS}}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return df, stamp


def coverage_report(df: pd.DataFrame, stamp: dict, k: float) -> dict:
    """2列の分布と、値の区切りごとの正例率（件数と割合だけ）。"""
    print(f"\n■ 0. 2列の付き方（表は {stamp['built_utc']} 作成）")
    out = {"label_k": k}
    lab_ = df["label"]
    for c in NEW_COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        ok = v.dropna()
        q = ok.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]) if len(ok) else pd.Series(dtype=float)
        print(f"  {c}（{NEW_JA[c]}）: 欠測 {int(v.isna().sum()):,} / {len(v):,}・分位 "
              + " / ".join(f"{int(kk * 100)}%: {q[kk]:,.1f}" for kk in q.index))
        edges = BINS[c]
        cut = pd.cut(v, bins=list(edges), right=True, include_lowest=False)
        rows = []
        for iv, g in lab_.groupby(cut, observed=False):
            n = int(g.notna().sum())
            r = float(g.mean()) if n else float("nan")
            rows.append({"bin": str(iv), "n": n, "pos_rate": r})
        print(f"    ラベル {k:.1f}σ の正例率（区切りごと）: " + "  ".join(
            f"{x['bin']} {x['n']:,}行 {x['pos_rate'] * 100:.1f}%" for x in rows))
        miss = lab_[v.isna()]
        if len(miss):
            print(f"    欠測の行 {len(miss):,} の正例率 {miss.mean() * 100:.1f}%")
        out[c] = {"missing": int(v.isna().sum()), "quantiles": {str(kk): float(x) for kk, x in q.items()}, "bins": rows}
    return out


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
    """列の組（base / new / perm）× パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。"""
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
    """窓ごとの AUC と、窓の期間・件数・正例率。keys があればその行だけで、MIN_WINDOW_ROWS 未満の窓は落とす。"""
    cols = ["fold", "pr", "roc", "start", "end", "n", "pos"]
    sub = o if keys is None else on_rows(o, keys)
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)
    n = sub.groupby("fold").size()
    sub = sub[sub["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)
    w = AB.auc_by_window(sub)
    if not len(w):
        return pd.DataFrame(columns=cols)
    g = sub.groupby("fold")
    info = pd.DataFrame({"start": g["Date"].min(), "end": g["Date"].max(), "n": g.size(),
                         "pos": g["label"].mean()}).reset_index()
    return w.merge(info, on="fold", how="left")[cols]


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


def window_table(results: dict, algos: list, shifts: list, rs: str) -> pd.DataFrame:
    """窓ごとの値（モデル × ずらし × 窓）。4腕がそろった窓だけ。"""
    rows = []
    for a in algos:
        for sh in shifts:
            w = None
            for arm in ARMS:
                x = results[a]["wins"][rs][(sh, arm)]
                x = x.rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
                if arm != "A":
                    x = x[["fold", f"pr_{arm}", f"roc_{arm}"]]
                w = x if w is None else w.merge(x, on="fold")
            if w is None or len(w) == 0:
                continue
            for _, r in w.iterrows():
                rows.append({"algo": a, "shift": sh, "fold": int(r["fold"]), "start": r["start"], "end": r["end"],
                             "n": int(r["n"]), "pos": float(r["pos"]),
                             **{f"{m}_{arm}": float(r[f"{m}_{arm}"]) for m in ("pr", "roc") for arm in ARMS}})
    return pd.DataFrame(rows)


def report(results: dict, shifts: list, stamp: dict, keysets: dict, cover: dict, k: float) -> dict:
    algos = [a for a in ALGOS if results.get(a)]
    missing = [a for a in ALGOS if not results.get(a)]
    print("\n" + "=" * 78)
    print(f"ラベル {k:.1f}σ / 結果のそろったモデル: {', '.join(algos) or 'なし'}"
          + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, "coverage": cover, "label_k": k, "perm_seed": PERM_SEED, "primary_rows": PRIMARY_ROWS,
               "tuning": {}, "production_oof": {}, "windows": {}, "judge7": {}, "per_window": {}}
    titles = {"all": "全行"}
    sec = {rs: "ab"[i] for i, rs in enumerate(keysets)}

    print(f"\n■ 1. 探索の CV（{CV_SCHEME}・{N_SPLITS}分割。楽観側に出る。パラメータ選び用）")
    print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC / パラメータ")
    for a in algos:
        r = results[a]
        for arm in ("A", "B"):
            cv = r["recs"][arm]["_cv"]
            fs = cv.get("fold_scores") or []
            print(f"  {a:<7}{arm:<4}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}"
                  f"{r['recs'][arm].get('_seconds', 0):>6}  " + " / ".join(f"{v:.4f}" for v in fs))
            summary["tuning"][f"{a}_{arm}"] = {kk: cv.get(kk) for kk in
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
                summary["production_oof"][f"{a}_{arm}_{rs}"] = {kk: float(v) for kk, v in m.items()}
                print(f"  {a:<7}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}"
                      f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
                      f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt"
                      f"{len(o):>8,}")
            for x, y in PAIRS:
                print(f"  {a:<7}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}"
                      f"{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}"
                      f"{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    tables = {}
    for rs in keysets:
        s = window_table(results, algos, shifts, rs)
        tables[rs] = s
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
        print(f"\n■ 4{sec[rs]}. §7（2026-09-26 改訂）の判定（窓の PR-AUC。{titles[rs]}"
              f"{'。判定に使う行（回す前に決めた）' if rs == PRIMARY_ROWS else '。参考'}）: "
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
        print(f"  → §7（ラベル {k:.1f}σ・{titles[rs]}）: {verdict}")

    for rs in keysets:
        s = tables[rs]
        if not len(s):
            continue
        print(f"\n■ 5{sec[rs]}. 窓ごとの値（{titles[rs]}。PR-AUC。差は「後 − 前」。運用者の指示「各cv（異なる窓）でもみたい」）")
        for a in algos:
            for sh in shifts:
                g = s[(s["algo"] == a) & (s["shift"] == sh)].sort_values("fold")
                if not len(g):
                    continue
                print(f"  {a}・ずらし{sh}か月（{len(g)}窓）")
                print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}{'A':>8}{'B':>8}{'C':>8}{'P':>8}"
                      f"{'C−A':>9}{'P−A':>9}{'B−A':>9}")
                for _, r in g.iterrows():
                    print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                          f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%{r['pr_A']:>8.4f}{r['pr_B']:>8.4f}{r['pr_C']:>8.4f}"
                          f"{r['pr_P']:>8.4f}{r['pr_C'] - r['pr_A']:>+9.4f}{r['pr_P'] - r['pr_A']:>+9.4f}"
                          f"{r['pr_B'] - r['pr_A']:>+9.4f}")
        summary["per_window"][rs] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE
    ap = argparse.ArgumentParser(description="実験101: 抜いた78週高値の古さ・前回の更新日の高値に対する今回の高値の2列を239列に足す前後")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算するモデル")
    ap.add_argument("--k", type=float, default=KS[0], help="ラベルの到達しきい値（σ の倍数。1.5 か 1.2）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0,
                    help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="列の付き方だけ出して終わる")
    args = ap.parse_args(argv)
    if not any(abs(args.k - k) < 1e-9 for k in KS):
        raise SystemExit(f"--k は {KS} のどれか（実際: {args.k}）")
    TAG = args.tag
    RUN = f"{TAG}_{k_tag(args.k)}"
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    base = F.columns(F.DEFAULT_PRESET)
    new = list(NEW_COLS)
    dup = sorted(set(base) & set(new))
    if dup:
        raise SystemExit(f"本番の列と重なる: {dup}")
    colsets = {"base": base, "new": base + new, "perm": base + new}

    print("=" * 78)
    print("実験101 抜いた78週高値の古さ・前回の更新日の高値に対する今回の高値の2列を本番の239列に足す前後"
          "（5分割 CV の探索 + OOF + 32窓・対照 P）")
    print(f"  列: {F.DEFAULT_PRESET} {len(base)}列 指紋 {F.signature(base)} / + {', '.join(new)} → {len(base) + len(new)}列 "
          f"指紋 {F.signature(base + new)}")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  ラベル {args.k:.1f}σ / この回に計算: {', '.join(algos)} / 探索 {args.n_trials}試行 × {N_SPLITS}分割"
          f"（{CV_SCHEME}・{CV_OBJECTIVE}）/ 窓のずらし {shifts}か月 / 種 {SEEDS} / 対照の種 {PERM_SEED} / "
          f"判定の行 {PRIMARY_ROWS} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    full, stamp = prepare(args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    miss = [c for c in colsets["new"] if c not in full.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    df = with_label(full, args.k)
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    cover = coverage_report(df, stamp, args.k)
    if args.dry:
        return 0
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"\n  この回の行 {len(df):,}（ラベル {args.k:.1f}σ の正例率 {df['label'].mean()*100:.1f}%・"
          f"{df['Date'].min().date()} 〜 {df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(df['Date'] <= cutoff).sum():,}行")
    t0 = time.time()
    perm = permuted(df, new, seed=PERM_SEED)
    perm.attrs["built_utc"] = stamp["built_utc"]
    log(f"[data] 対照 P: {len(new)}列を日付内で入れ替えた（種 {PERM_SEED}）/ {time.time()-t0:.0f}秒")
    frames = {"base": df, "new": df, "perm": perm}
    keysets = eval_keysets(df)

    results = {}
    for a in ALGOS:
        t0 = time.time()
        results[a] = run(a, frames, keysets, colsets, cutoff, args.n_trials, stamp, shifts, compute=a in algos)
        if a in algos:
            log(f"[{a}] {'そろった' if results[a] else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    report(results, shifts, stamp, keysets, cover, args.k)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
