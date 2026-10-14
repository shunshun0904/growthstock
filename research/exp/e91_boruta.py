#!/usr/bin/env python3
"""
実験91: Boruta（影の列に勝てない列を外す）で本番の239列を絞ると、木3モデル（1.5σ）の分離力が上がるか。
あわせて、本番の列に「先の期間で逆に働いている列」があるかを、窓のテスト期間での入れ替え重要度で見る。

運用者の問い（2026-10-14）「特徴量239列のうち、いらない特徴量が逆に精度を下げていないか。Boruta による特徴量選択を
実施してみたい」。同日の回答（AskUserQuestion）:

  進め方      推奨の形（窓の中で選ぶ Boruta・日付に合わせた影・対照 R・先の期間での入れ替え重要度。木3モデル・32窓）
  採用の条件  改善したら採用（§7 と同じ形。下の「判定」）
  ラベル      1.5σ だけ（2026-10-12 からの選定「1.5σ の3本がすべて95以上」のモデルのラベル）
  tentative   残す（外すのは rejected だけ）

腕（データ・パラメータは同じ。列だけが違う。パラメータは本番の 1.5σ の3本（multi_params.json の *_k15）を読むだけ）
  A  いまの本番の239列
  S  窓ごとに、その窓の訓練期間の行だけで回した Boruta が rejected にした列を外したもの
  R  対照: S と同じ本数を、でたらめに選んで外したもの（窓ごとに固定。種 R_SEED）
  S−A が「Boruta で外すことの効果」、R−A が「列が減っただけの幅」、S−R が「どの列を外したかの効果」

Boruta（Kursa & Rudnicki 2010 の手順。判定は BorutaPy 0.4.3 の既定と同じ: 影の最大（perc 100）・二段の多重比較
（two_step: FDR → Bonferroni）・α 0.05・最大100回）。違うのは次の3点で、どれもこのデータの形に合わせたもの:
  1. 影の作り。銘柄ごとの列は同じ日付の行どうしで入れ替える（実験68 などの対照 P と同じ。1行しか無い日は同じ月の
     1行の日どうし）。その日は全銘柄同じ値の列（地合い・市場全体の需給。データから見つける）は日付ごと入れ替える。
     全行で入れ替えると影は「どの日か」を表せなくなり、時期をなぞるだけの列でも影に勝ってしまう
  2. 影との比べ方。銘柄ごとの列は銘柄ごとの列の影の最大と、日付単位の列は日付単位の列の影の最大と比べる
     （日付単位の影は日の当たり外れを訓練の中で拾えるぶん重要度が大きく出やすいと見込んだ。混ぜると銘柄ごとの列の
     物差しが日付単位の影で決まってしまう。見込みが合っているかは ■1 の「影の物差し」で確かめる）
  3. 重要度。本番の 1.5σ の LightGBM（lgbm_k15 のパラメータ・木200本）の gain（BorutaPy は feature_importances_ で、
     LightGBM の既定では split の回数）
  回すのはずらし0の窓（訓練期間の行だけ）。ずらし2・4か月の窓は、訓練の終わりがそれ以前のずらし0の窓のうち最も後の
  ものの選定を使う（窓 i の訓練の終わりはずらし0の窓 i より 2・4か月後ろなので、先読みにならない）。選ぶのは LightGBM
  1本で、xgb / cat も同じ列で学習する（本番は1つの列で全モデルを学習する）。実験61 の教訓（列を選ぶモデルは評価する
  期間を学習していないものにする）を、窓ごとに選び直すことで守る

判定（回す前に決めた。2026-10-14）
  docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）を「外す」向きに当てる: 木3モデル（lgbm / xgb / cat の 1.5σ）の
  うち2つ以上で、32窓の PR-AUC の差 S−A の平均が正・対照 R−A の平均を上回る・S が上の窓が過半（17 以上）。満たせば、
  本番に入れる案（その時点までの行で選んだ固定の一覧。毎週は選び直さない）を運用者に出す。満たさなければ本番は239列のまま。

診断（判定には使わない）: A の窓モデルで、テスト期間の列・グループを入れ替えたときの PR-AUC・ROC-AUC の下がり方
（入れ替え重要度）。負なら「入れ替えたほうが当たる」= 先の期間で逆に働いている列。入れ替えは影と同じ作り
（銘柄ごとの列は日付内、日付単位の列は日付ごと。グループは中の列を同じ並べ替えでまとめて入れ替える）。

公開ログには件数・割合・日付・精度・列の名前だけ（銘柄・値は出さない）。

    exp=e91_boruta.py args="--algos lgbm"   （1回目。Boruta の選定（ずらし0の窓）+ lgbm の3腕）
    exp=e91_boruta.py args="--algos xgb"
    exp=e91_boruta.py args="--algos cat"
選定のようすだけ見る: exp=e91_boruta.py args="--select-only"
行と影の単位だけ見る: exp=e91_boruta.py args="--dry"
試運転:               exp=e91_boruta.py args="--tag e91smoke --algos lgbm --max-iter 10 --shifts 0 --seeds 42"

Run Experiment は同じ実験を同時に1本しか走らせず、待ちに置けるのも1本だけなので、1本ずつ起動する。表は1回目に作って
research/_data/oof/<tag>_* に置き、以降の回は actions/cache で引き継いだそれを使う（どの回も同じ行・同じ選定で比べる）。
パラメータも1回目の値を表の記録（stamp）に固定する（途中で土曜の Retrain Weekly が *_k15 を探索し直しても変えない）。
--budget-min（既定 290分）を過ぎたら新しい計算を始めずに終わる（保存済みの選定・out-of-fold は次の回が続きから使う）。
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
import features as F  # noqa: E402
import lab  # noqa: E402
import models as M  # noqa: E402
import tuning_multi as TM  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e20_annual_trajectory import chance_hits  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e68_edinet_ab import judge7, pair, params_hash  # noqa: E402
from e89_inst_buy_ab import windows_on, with_label  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
#: ラベルの到達しきい値（運用者の回答「1.5σ だけ」）
K = 1.5
#: 比べるモデル（本番の 1.5σ の木3本）。§7 の票もこの3つ。--algos は lgbm / xgb / cat でも名指しできる
ALGOS = ("lgbm_k15", "xgb_k15", "cat_k15")
#: Boruta の重要度を出すモデル
SELECT_ALGO = "lgbm_k15"
SEEDS = (42, 7, 123)
PROD_SEED = 42
ARMS = ("A", "S", "R")
LABELS = {"A": "A いまの本番（239列）",
          "S": "S 窓ごとの Boruta で rejected の列を外す（tentative は残す）・A のパラメータ",
          "R": "R 対照（S と同じ本数をでたらめに外す）・A のパラメータ"}
PAIRS = (("A", "S"), ("A", "R"), ("R", "S"))
#: Boruta の設定（回す前に決めた。BorutaPy 0.4.3 の既定と同じ）
MAX_ITER = 100
ALPHA = 0.05
PERC = 100
#: 種（結果を見る前に決めた）
BORUTA_SEED = 20261014
R_SEED = 20261015
PI_SEED = 20261016
#: これより行の少ない日は、同じ月の少ない日どうしでまとめて入れ替える（1行の日は入れ替えても自分の値のまま）
MIN_DATE_ROWS = 2
#: 窓の学習に要る行（E41.oof_folds と同じ）
MIN_TRAIN, MIN_TEST = 1000, 200
KEEP = ["Code", "Date", "label", "ret_o1_20", "ret_o1_40"]
TAG = "e91"
RUN = "e91_k15"
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fpath(name: str) -> str:
    """表（ラベルによらない）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def path(name: str) -> str:
    """選定・out-of-fold・要約。"""
    return os.path.join(OOF_DIR, f"{RUN}_{name}")


def time_left() -> bool:
    global TIMED_OUT
    if DEADLINE is None or time.time() < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min を過ぎたので、新しい計算は始めない（保存済みの分だけで報告。続きは次の回）")
    return False


def algo_name(a: str) -> str:
    """lgbm / xgb / cat を 1.5σ の名前に（lgbm_k15 などそのままの名前も受ける）。"""
    n = a if a in ALGOS else f"{a}_k15"
    if n not in ALGOS:
        raise SystemExit(f"知らないモデル: {a}（使えるもの: lgbm / xgb / cat）")
    return n


def short_name(a: str) -> str:
    return TM.base_algo(a)


def prod_params(algo: str) -> dict:
    """本番の 1.5σ のモデルのパラメータ（土曜の Retrain Weekly が239列・1.5σ で探索したもの）。読むだけ。"""
    with open(TM.PARAMS_PATH, encoding="utf-8") as fh:
        rec = json.load(fh)[algo]
    cv = rec.get("_cv", {})
    base = F.columns(F.DEFAULT_PRESET)
    if cv.get("features_sig") != F.signature(base) or float(cv.get("label_k", 0)) != K:
        raise SystemExit(f"{algo} の探索結果が本番の列・{K}σ のものではない（features_sig {cv.get('features_sig')}・"
                         f"label_k {cv.get('label_k')}）")
    return dict(rec["params"])


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def prepare(rebuild: bool = False):
    """lab.frame（全行。ラベルは 1.2σ のまま）と記録。"""
    pf, ps = fpath("frame.parquet"), fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}）")
        return pd.read_parquet(pf), stamp
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    df = lab.frame(rebuild=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rows": int(len(df)),
             "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
             "params": {a: prod_params(a) for a in ALGOS}}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1)
    log(f"[data] lab.frame {len(df):,}行 × {df.shape[1]}列 / {time.time()-t0:.0f}秒")
    return df, stamp


def data_hash(stamp: dict) -> str:
    return hashlib.sha1(str(stamp.get("built_utc", "")).encode()).hexdigest()[:6]


def params_of(algo: str, stamp: dict) -> dict:
    """1回目に表と一緒に控えたパラメータ（控えが無ければ今の multi_params.json）。"""
    return dict(stamp.get("params", {}).get(algo) or prod_params(algo))


# --------------------------------------------------------------------------- #
# 影（入れ替え）の作り
# --------------------------------------------------------------------------- #
def date_level_columns(df: pd.DataFrame, cols: list) -> list:
    """その日は全銘柄同じ値の列（欠測も1つの値として数える。全期間で1つの値しか無い列は除く）。"""
    if not len(df):
        return []
    per_day = df.groupby("Date")[list(cols)].nunique(dropna=False).max()
    overall = df[list(cols)].nunique(dropna=False)
    return [c for c in cols if per_day[c] <= 1 and overall[c] > 1]


def shadow_groups(dates) -> np.ndarray:
    """
    銘柄ごとの列を入れ替える単位（行ごとの番号）。行が MIN_DATE_ROWS 以上の日はその日。少ない日は同じ月の少ない日
    どうし、それでも1行なら同じ年の少ない日どうし（1行の単位は入れ替えても自分の値のままで、影が元の列と同じになる）。
    """
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    if not len(d):
        return np.zeros(0)
    n_day = d.map(d.value_counts()).to_numpy()
    small = n_day < MIN_DATE_ROWS
    key = np.where(small, "m" + d.dt.strftime("%Y-%m"), d.dt.strftime("%Y-%m-%d"))
    n_key = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy()
    key = np.where(small & (n_key < 2), "y" + d.dt.strftime("%Y"), key)
    return pd.factorize(pd.Series(key))[0].astype(float)


class Shuffler:
    """
    同じ行の集まりに何度も使う並べ替え。戻り値 perm は「行 i に持ってくる行の番号」（X[perm] が入れ替えた写し）。

      within   shadow_groups の単位の中だけで一様に並べ替える（単位の中の値の集まりは変わらない）
      by_date  日付ごと入れ替える（行 i に、でたらめに対応させた別の日の代表の行を持ってくる。日付単位の列用）
    """

    def __init__(self, dates):
        d = pd.to_datetime(pd.Series(np.asarray(dates)))
        self.groups = shadow_groups(d)
        self.n = len(self.groups)
        self.base = np.argsort(self.groups, kind="stable")
        codes, uniq = pd.factorize(d)
        self.date_idx = np.asarray(codes)
        self.n_dates = len(uniq)
        _, self.first_row = np.unique(self.date_idx, return_index=True)

    def within(self, rng) -> np.ndarray:
        shuf = np.argsort(self.groups + rng.random(self.n), kind="stable")
        perm = np.empty(self.n, dtype=np.int64)
        perm[self.base] = shuf
        return perm

    def by_date(self, rng) -> np.ndarray:
        return self.first_row[rng.permutation(self.n_dates)[self.date_idx]]


def shadows(X: np.ndarray, sh: Shuffler, is_date: np.ndarray, rng) -> np.ndarray:
    """X の列ごとの影（列ごとに別の並べ替え）。"""
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = X[sh.by_date(rng) if is_date[j] else sh.within(rng), j]
    return out


# --------------------------------------------------------------------------- #
# Boruta
# --------------------------------------------------------------------------- #
def fdr(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini–Hochberg で棄却するか（BorutaPy の _fdrcorrection と同じ）。"""
    n = len(p)
    if not n:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ok = p[order] <= np.arange(1, n + 1) / n * alpha
    if ok.any():
        ok[: np.nonzero(ok)[0].max() + 1] = True
    out = np.empty(n, dtype=bool)
    out[order] = ok
    return out


def decide(dec: np.ndarray, hits: np.ndarray, it: int, alpha: float = ALPHA) -> np.ndarray:
    """
    BorutaPy の _do_tests（two_step=True）と同じ判定。dec は 1 confirmed / 0 tentative / −1 rejected。
    帰無仮説は「影の最大に勝つ確率が 0.5」。勝ちが多ければ confirmed、少なければ rejected。
    """
    from scipy import stats

    act = np.where(dec >= 0)[0]
    h = hits[act]
    p_acc = stats.binom.sf(h - 1, it, 0.5)
    p_rej = stats.binom.cdf(h, it, 0.5)
    acc = fdr(p_acc, alpha) & (p_acc <= alpha / it)
    rej = fdr(p_rej, alpha) & (p_rej <= alpha / it)
    und = dec[act] == 0
    out = dec.copy()
    out[act[und & acc]] = 1
    out[act[und & rej]] = -1
    return out


def boruta(X: np.ndarray, y: np.ndarray, sh: Shuffler, is_date: np.ndarray, fit_imp, *,
           max_iter: int = MAX_ITER, alpha: float = ALPHA, perc: float = PERC, seed: int = BORUTA_SEED,
           progress=None) -> dict:
    """
    Boruta の本体。fit_imp(X_with_shadows, y, it) は列ごとの重要度を返す。rejected の列はその回から外し、影も作らない
    （BorutaPy と同じ）。影との比べ方は列の種類ごと（is_date が同じ列の影の perc 百分位）。
    """
    p = X.shape[1]
    dec = np.zeros(p, dtype=int)
    hits = np.zeros(p, dtype=int)
    imp_sum = np.zeros(p)
    imp_n = np.zeros(p, dtype=int)
    thr_hist = []
    rng = np.random.default_rng(seed)
    it = 0
    while it < max_iter and (dec == 0).any():
        act = np.where(dec >= 0)[0]
        xa, isd = X[:, act], is_date[act]
        imp = np.asarray(fit_imp(np.hstack([xa, shadows(xa, sh, isd, rng)]), y, it), dtype=float)
        real, sha = imp[:len(act)], imp[len(act):]
        thr = np.full(len(act), np.inf)
        rec = {}
        for kind, nm in ((False, "stock"), (True, "date")):
            m = isd == kind
            if m.any():
                thr[m] = np.percentile(sha[m], perc)
                rec[nm] = float(thr[m][0])
                rec[f"{nm}_from"] = int(act[np.where(m)[0][np.argmax(sha[m])]])    # 影の最大になった列（診断）
        it += 1
        hits[act] += real > thr
        imp_sum[act] += real
        imp_n[act] += 1
        thr_hist.append(rec)
        dec = decide(dec, hits, it, alpha)
        if progress is not None:
            progress(it, dec)
    return {"decision": dec, "hits": hits, "n_iter": it,
            "mean_imp": np.where(imp_n > 0, imp_sum / np.maximum(imp_n, 1), np.nan), "thr": thr_hist}


def gain_fitter(params: dict):
    """本番の 1.5σ の LightGBM（tuning_multi.build と同じ組み方）の gain。回ごとに種を変える。"""
    def fit_imp(X, y, it):
        TM.SEED = BORUTA_SEED + it
        try:
            m = TM.build(SELECT_ALGO, params, y)
            m.fit(X, y)
        finally:
            TM.SEED = 0
        return m.booster_.feature_importance(importance_type="gain")
    return fit_imp


def select_key(fold, cols: list, params: dict, stamp: dict, date_cols: list, max_iter: int) -> dict:
    return {"train_end": str(fold.train_end), "cols": F.signature(cols), "params": params_hash(params),
            "data": stamp["built_utc"], "max_iter": max_iter, "alpha": ALPHA, "perc": PERC, "seed": BORUTA_SEED,
            "min_date_rows": MIN_DATE_ROWS, "date_cols": F.signature(date_cols)}


def select_window(df: pd.DataFrame, cols: list, fold, params: dict, stamp: dict, date_cols: list,
                  max_iter: int, compute: bool = True) -> dict | None:
    """ずらし0の窓 fold の訓練期間の行だけで Boruta。保存済みなら読む（設定と表の版が同じときだけ）。"""
    p = path(f"boruta_f{int(fold.index):02d}.json")
    key = select_key(fold, cols, params, stamp, date_cols, max_iter)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("key") == key:
            return rec
    if not compute or not time_left():
        return None
    d = pd.to_datetime(df["Date"])
    tr = df[(d <= pd.Timestamp(fold.train_end)) & df["label"].notna()]
    if len(tr) < MIN_TRAIN:
        return None
    X = tr[cols].to_numpy(dtype=float)
    y = tr["label"].to_numpy(dtype=int)
    dset = set(date_cols)
    is_date = np.array([c in dset for c in cols])
    sh = Shuffler(tr["Date"])
    t0 = time.time()

    def progress(it, dec):
        if it % 10 == 0:
            log(f"    窓{fold.index} {it}回: confirmed {(dec == 1).sum()} / tentative {(dec == 0).sum()} / "
                f"rejected {(dec == -1).sum()}（{time.time()-t0:.0f}秒）")

    res = boruta(X, y, sh, is_date, gain_fitter(params), max_iter=max_iter, progress=progress)
    dec = res["decision"]
    rec = {"key": key, "fold": int(fold.index), "train_end": str(fold.train_end), "n_train": int(len(tr)),
           "pos_rate": float(y.mean()), "n_iter": int(res["n_iter"]), "seconds": round(time.time() - t0),
           "confirmed": [c for c, v in zip(cols, dec) if v == 1],
           "tentative": [c for c, v in zip(cols, dec) if v == 0],
           "rejected": [c for c, v in zip(cols, dec) if v == -1],
           "hits": {c: int(h) for c, h in zip(cols, res["hits"])},
           "mean_gain": {c: (None if not np.isfinite(g) else float(g)) for c, g in zip(cols, res["mean_imp"])},
           "thr_last": res["thr"][-1] if res["thr"] else {},
           "thr_median": {k: float(np.median([t[k] for t in res["thr"] if k in t]))
                          for k in ("stock", "date") if any(k in t for t in res["thr"])},
           "max_shadow_from": {k: {cols[j]: int(n) for j, n in pd.Series(
               [t[f"{k}_from"] for t in res["thr"] if f"{k}_from" in t], dtype=int).value_counts().items()}
               for k in ("stock", "date")}}
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    log(f"  [Boruta] 窓{fold.index}（訓練 〜{fold.train_end}・{len(tr):,}行・正例率 {y.mean()*100:.1f}%）: "
        f"{rec['n_iter']}回 / confirmed {len(rec['confirmed'])}・tentative {len(rec['tentative'])}・"
        f"rejected {len(rec['rejected'])} / {rec['seconds']}秒")
    return rec


def selection_for(fold, sel_folds: list):
    """fold（ずらした窓）に使う選定の窓: 訓練の終わりが fold の訓練の終わり以前のずらし0の窓のうち、最も後のもの。"""
    te = pd.Timestamp(fold.train_end)
    ok = [f for f in sel_folds if pd.Timestamp(f.train_end) <= te]
    return max(ok, key=lambda f: pd.Timestamp(f.train_end)) if ok else None


def colmaps(base: list, folds_by_shift: dict, sel_folds: list, sels: dict) -> dict:
    """(腕, ずらし) → {窓の番号: 列}。選定が無い窓は、どの腕にも入れない（同じ窓で比べる）。"""
    out = {(arm, sh): {} for arm in ARMS for sh in folds_by_shift}
    for sh, folds in folds_by_shift.items():
        for f in folds:
            sf = selection_for(f, sel_folds)
            rec = sels.get(int(sf.index)) if sf is not None else None
            if rec is None:
                continue
            rej = set(rec["rejected"])
            rng = np.random.default_rng([R_SEED, int(sh), int(f.index)])
            drop = set(rng.choice(np.array(base, dtype=object), size=len(rej), replace=False)) if rej else set()
            out[("A", sh)][int(f.index)] = list(base)
            out[("S", sh)][int(f.index)] = [c for c in base if c not in rej]
            out[("R", sh)][int(f.index)] = [c for c in base if c not in drop]
    return out


def colmap_sig(cm: dict) -> str:
    return hashlib.sha1(json.dumps({str(k): F.signature(v) for k, v in sorted(cm.items())},
                                   sort_keys=True).encode()).hexdigest()[:8]


def selection_sig(sels: dict) -> str:
    """選定の指紋（窓ごとの rejected）。lgbm / xgb / cat の回が同じ選定で比べたかをログで確かめる。"""
    return hashlib.sha1(json.dumps({str(k): sorted(v["rejected"]) for k, v in sorted(sels.items())},
                                   sort_keys=True).encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# out-of-fold（窓ごとに列が違う）と、入れ替え重要度
# --------------------------------------------------------------------------- #
def perm_importance(model, Xte: np.ndarray, yte: np.ndarray, dates, cols: list, base_score: np.ndarray,
                    groups: dict, date_cols: set, seed: int, fold: int) -> pd.DataFrame:
    """列・グループごとに、テスト期間の値を入れ替えたときの PR-AUC・ROC-AUC の下がり方（正 = その列が先で効いている）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    sh = Shuffler(dates)
    rng = np.random.default_rng([PI_SEED, int(seed), int(fold)])
    b_pr, b_roc = average_precision_score(yte, base_score), roc_auc_score(yte, base_score)
    idx = {c: j for j, c in enumerate(cols)}
    units = [("col", c, [c]) for c in cols] + [("group", g, [c for c in cs if c in idx]) for g, cs in groups.items()]
    rows = []
    for kind, name, cs in units:
        if not cs:
            continue
        st = [idx[c] for c in cs if c not in date_cols]
        dt = [idx[c] for c in cs if c in date_cols]
        xp = Xte.copy()
        if st:
            xp[:, st] = Xte[sh.within(rng)][:, st]
        if dt:
            xp[:, dt] = Xte[sh.by_date(rng)][:, dt]
        s = M.predict(model, xp)
        rows.append({"fold": int(fold), "kind": kind, "name": name,
                     "d_pr": float(b_pr - average_precision_score(yte, s)),
                     "d_roc": float(b_roc - roc_auc_score(yte, s))})
    return pd.DataFrame(rows)


def oof_cols(algo: str, df: pd.DataFrame, colmap: dict, params: dict, seed: int, folds,
             pi: dict | None = None):
    """
    窓ごとの列（colmap: 窓の番号 → 列）で out-of-fold。学習器の組み方・行の選び方は E41.oof_folds の lgbm 以外の枝と
    同じ（models.fit。種は tuning_multi.SEED）。pi があれば、その窓モデルで入れ替え重要度も出す。
    """
    d = pd.to_datetime(df["Date"])
    parts, imps = [], []
    try:
        for f in folds:
            if int(f.index) not in colmap:
                continue
            tr = df[(d <= pd.Timestamp(f.train_end)) & df["label"].notna()]
            te = df[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end)) & df["label"].notna()]
            if len(te) < MIN_TEST or len(tr) < MIN_TRAIN:
                continue
            cols = colmap[int(f.index)]
            Xte = te[cols].to_numpy(dtype=float)
            TM.SEED = seed
            m = M.fit(algo, tr[cols].to_numpy(dtype=float), tr["label"].to_numpy(dtype=int), cols, params=params)
            sc = M.predict(m, Xte)
            part = te[[c for c in KEEP if c in te.columns]].copy()
            part["score"] = sc
            part["fold"] = int(f.index)
            parts.append(part)
            if pi is not None:
                imps.append(perm_importance(m, Xte, te["label"].to_numpy(dtype=int), te["Date"], cols, sc,
                                            pi["groups"], pi["date_cols"], seed, int(f.index)))
    finally:
        TM.SEED = 0
    o = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEEP + ["score", "fold"])
    return o, (pd.concat(imps, ignore_index=True) if imps else None)


def oof(algo: str, arm: str, df: pd.DataFrame, cms: dict, params: dict, shift: int, seeds, stamp: dict,
        pi: dict | None, compute: bool = True):
    """腕 × ずらし × 種の out-of-fold（種の平均）と、A なら入れ替え重要度（種の平均）。保存済みなら読む。"""
    cm = cms[(arm, shift)]
    ph = f"{params_hash(params)}_{colmap_sig(cm)}_d{data_hash(stamp)}"
    files = [path(f"{algo}_{arm}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    pfiles = [path(f"{algo}_{arm}_pi_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds] if arm == "A" else None
    need = [i for i in range(len(seeds))
            if not os.path.exists(files[i]) or (pfiles is not None and not os.path.exists(pfiles[i]))]
    if need and (not compute or not time_left()):
        return None, None
    folds = E41.folds_for(df["Date"], shift) if need else None
    parts, pis = [], []
    for i, sd in enumerate(seeds):
        if i not in need:
            parts.append(pd.read_parquet(files[i]))
            if pfiles is not None:
                pis.append(pd.read_parquet(pfiles[i]))
            continue
        if not time_left():
            return None, None
        t0 = time.time()
        o, imp = oof_cols(algo, df, cm, params, sd, folds, pi if arm == "A" else None)
        o.to_parquet(files[i], index=False)
        parts.append(o)
        if pfiles is not None:
            imp.to_parquet(pfiles[i], index=False)
            pis.append(imp)
        log(f"    {short_name(algo)} 腕{arm} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    imp = None
    if pis:
        imp = (pd.concat(pis, ignore_index=True).groupby(["fold", "kind", "name"], as_index=False)[["d_pr", "d_roc"]]
               .mean())
    return o, imp


def run(algo: str, df: pd.DataFrame, cms: dict, stamp: dict, shifts: list, seeds, pi: dict,
        compute: bool) -> dict | None:
    params = params_of(algo, stamp)
    prod = {}
    if 0 in shifts:
        for arm in ARMS:
            o, _ = oof(algo, arm, df, cms, params, 0, (PROD_SEED,), stamp, pi, compute)
            if o is None:
                return None
            prod[arm] = o
    wins, imps = {}, {}
    for sh in shifts:
        for arm in ARMS:
            o, imp = oof(algo, arm, df, cms, params, sh, seeds, stamp, pi, compute)
            if o is None:
                return None
            wins[(sh, arm)] = windows_on(o, None)
            if imp is not None:
                imps[sh] = imp
    return {"params": params, "prod": prod, "wins": wins, "imps": imps}


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #
def window_table(results: dict, algos: list, shifts: list, cms: dict) -> pd.DataFrame:
    """窓ごとの値（モデル × ずらし × 窓）。3腕がそろった窓だけ。"""
    rows = []
    for a in algos:
        for sh in shifts:
            w = None
            for arm in ARMS:
                x = results[a]["wins"][(sh, arm)].rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
                if arm != "A":
                    x = x[["fold", f"pr_{arm}", f"roc_{arm}"]]
                w = x if w is None else w.merge(x, on="fold")
            if w is None or not len(w):
                continue
            for _, r in w.iterrows():
                f = int(r["fold"])
                rows.append({"algo": a, "shift": sh, "fold": f, "start": r["start"], "end": r["end"],
                             "n": int(r["n"]), "pos": float(r["pos"]),
                             "n_cols_S": len(cms[("S", sh)].get(f, [])), "n_cols_R": len(cms[("R", sh)].get(f, [])),
                             **{f"{m}_{arm}": float(r[f"{m}_{arm}"]) for m in ("pr", "roc") for arm in ARMS}})
    return pd.DataFrame(rows)


def verdict(judged: dict, complete: bool) -> dict:
    """§7 の票（木3モデル中2つ以上で満たせば採用の案）。"""
    passed = sum(int(j["pass"]) for j in judged.values())
    if complete:
        text = "満たす（木3モデル中2つ以上）→ 本番に入れる案を運用者に出す" if passed >= 2 else \
            "満たさない（木3モデル中2つ未満）→ 本番は239列のまま"
    else:
        text = f"{passed}/{len(judged)} モデルが満たす（木3モデル・3通りのずらし・種3つがそろってから判定）"
    return {"passed": passed, "of": len(judged), "complete": complete, "text": text}


def selection_report(sels: dict, base: list, groups: dict, date_cols: list) -> dict:
    """Boruta の選定のようす（ずらし0の窓ごと・グループごと・列ごと・窓どうしの重なり）。"""
    out = {"windows": [], "groups": {}, "rejected_count": {}, "jaccard_next": []}
    if not sels:
        print("\n■ 1. Boruta の選定: まだ1窓も無い")
        return out
    ks = sorted(sels)
    print(f"\n■ 1. Boruta の選定（ずらし0の窓ごと。訓練期間の行だけ。最大 {MAX_ITER_IN_USE}回・α {ALPHA}・影の {PERC} 百分位）")
    print(f"  {'窓':>3}  {'訓練の終わり':<12}{'行':>8}{'正例率':>8}{'回数':>6}{'confirmed':>11}{'tentative':>11}"
          f"{'rejected':>10}{'秒':>7}  影の物差し（gain の中央値: 銘柄ごと / 日付単位）")
    for k in ks:
        r = sels[k]
        tm = r.get("thr_median", {})
        print(f"  {k:>3}  {r['train_end']:<12}{r['n_train']:>8,}{r['pos_rate']*100:>7.1f}%{r['n_iter']:>6}"
              f"{len(r['confirmed']):>11}{len(r['tentative']):>11}{len(r['rejected']):>10}{r['seconds']:>7}  "
              + " / ".join(f"{tm[x]:,.1f}" if x in tm else "—" for x in ("stock", "date")))
        out["windows"].append({kk: r[kk] for kk in ("fold", "train_end", "n_train", "pos_rate", "n_iter", "seconds")}
                              | {"n_confirmed": len(r["confirmed"]), "n_tentative": len(r["tentative"]),
                                 "n_rejected": len(r["rejected"]), "thr_median": tm})
    n = len(ks)
    cnt = {c: sum(c in set(sels[k]["rejected"]) for k in ks) for c in base}
    out["rejected_count"] = cnt
    print(f"\n  グループごと（{n}窓。rejected の本数の平均 / {n}窓すべてで rejected / 1窓も rejected にならない）")
    for g, cs in groups.items():
        cs = [c for c in cs if c in cnt]
        if not cs:
            continue
        mean_rej = float(np.mean([sum(c in set(sels[k]["rejected"]) for c in cs) for k in ks]))
        always = sum(cnt[c] == n for c in cs)
        never = sum(cnt[c] == 0 for c in cs)
        kind = "（日付単位）" if all(c in date_cols for c in cs) else ""
        print(f"    {g:<22}{len(cs):>4}列{kind}: 平均 {mean_rej:>5.1f}本 / すべて {always:>3} / なし {never:>3}")
        out["groups"][g] = {"n": len(cs), "mean_rejected": mean_rej, "always": always, "never": never}
    for kind, nm in (("stock", "銘柄ごと"), ("date", "日付単位")):
        tot = {}
        for k in ks:
            for c, v in sels[k].get("max_shadow_from", {}).get(kind, {}).items():
                tot[c] = tot.get(c, 0) + v
        if tot:
            n_it = sum(tot.values())
            top = sorted(tot.items(), key=lambda x: -x[1])[:6]
            print(f"  影の最大になった列（{nm}・{n}窓の {n_it}回のうち）: "
                  + ", ".join(f"{c} {v}回（{v / n_it * 100:.0f}%）" for c, v in top))
            out[f"max_shadow_from_{kind}"] = tot
    hist = pd.Series(list(cnt.values())).value_counts().sort_index()
    print(f"\n  列ごとの rejected になった窓の数（{n}窓中）: " + " / ".join(f"{int(k)}窓 {int(v)}列"
                                                               for k, v in hist.items()))
    for a, b in zip(ks, ks[1:]):
        sa, sb = set(sels[a]["rejected"]), set(sels[b]["rejected"])
        jac = len(sa & sb) / len(sa | sb) if sa | sb else 1.0
        out["jaccard_next"].append({"from": a, "to": b, "jaccard": jac})
    if out["jaccard_next"]:
        print("  隣の窓との rejected の重なり（Jaccard）: "
              + " / ".join(f"{x['from']}→{x['to']} {x['jaccard']:.2f}" for x in out["jaccard_next"]))
    last = sels[ks[-1]]
    print(f"\n  最後の窓（訓練 〜{last['train_end']}。本番に入れるならこれに近い一覧）の rejected {len(last['rejected'])}列: "
          + ", ".join(last["rejected"]))
    dl = [c for c in date_cols if c in cnt]
    if dl:
        print(f"  日付単位の列 {len(dl)}本の rejected の窓の数: " + ", ".join(f"{c} {cnt[c]}" for c in dl))
    return out


def pi_report(results: dict, algos: list, shifts: list, cnt: dict, n_sel: int) -> dict:
    """診断: A の窓モデルの入れ替え重要度（窓は3通りのずらし。種の平均）。"""
    out = {}
    print("\n■ 6. 診断（判定には使わない）: A の窓モデルで、テスト期間の値を入れ替えたときの下がり方"
          "（正 = 先の期間で効いている、負 = 入れ替えたほうが当たる）")
    for a in algos:
        parts = [results[a]["imps"][sh].assign(shift=sh) for sh in shifts if sh in results[a]["imps"]]
        if not parts:
            continue
        imp = pd.concat(parts, ignore_index=True)
        g = imp.groupby(["kind", "name"])
        st = pd.DataFrame({"mean_pr": g["d_pr"].mean(), "se_pr": g["d_pr"].std(ddof=1) / np.sqrt(g.size()),
                           "pos_pr": g["d_pr"].apply(lambda v: int((v > 0).sum())),
                           "neg_pr": g["d_pr"].apply(lambda v: int((v < 0).sum())),
                           "mean_roc": g["d_roc"].mean(), "n_win": g.size()}).reset_index()
        st["t_pr"] = st["mean_pr"] / st["se_pr"].replace(0, np.nan)
        out[a] = st.to_dict("records")
        print(f"\n  {short_name(a)}（{int(st['n_win'].median())}窓）グループ（PR-AUC の下がり方の平均 ± SE・"
              "正の窓 / 負の窓・ROC の下がり方）")
        gs = st[st["kind"] == "group"].sort_values("mean_pr", ascending=False)
        for _, r in gs.iterrows():
            print(f"    {r['name']:<22}{r['mean_pr']:>+9.4f} ± {r['se_pr']:.4f}（t {r['t_pr']:>+5.1f}）"
                  f"{int(r['pos_pr']):>4} / {int(r['neg_pr']):<4}ROC {r['mean_roc']:>+8.4f}")
        cs = st[st["kind"] == "col"].copy()
        used = cs[(cs["mean_pr"] != 0) | (cs["se_pr"] > 0)]
        exp_two, _, nw = chance_hits(used, z=2.0)
        n_neg = int((used["t_pr"] < -2).sum())
        n_pos = int((used["t_pr"] > 2).sum())
        print(f"  列（{len(cs)}列。モデルが使っていて値の動く列 {len(used)}）: t < −2 が {n_neg}列・t > +2 が {n_pos}列"
              f"（偶然の見込みは片側それぞれ約 {exp_two / 2:.1f}列。窓 {nw}本の t 分布）")
        for title, sub in (("先の期間でいちばん効いている15列", used.sort_values("t_pr", ascending=False).head(15)),
                           ("入れ替えたほうが当たる側の15列（t の小さい順）", used.sort_values("t_pr").head(15))):
            print(f"    {title}: " + ", ".join(f"{r['name']} {r['mean_pr']:+.4f}（t {r['t_pr']:+.1f}）"
                                             for _, r in sub.iterrows()))
        if cnt and n_sel:
            cs["rej"] = cs["name"].map(cnt).fillna(0)
            print(f"    Boruta の扱いごとの平均（ずらし0の {n_sel}窓で rejected になった回数）: " + " / ".join(
                f"{lab_} {len(s)}列 {s['mean_pr'].mean():+.5f}" for lab_, s in (
                    ("一度も外れない", cs[cs["rej"] == 0]),
                    ("ときどき外れる", cs[(cs["rej"] > 0) & (cs["rej"] < n_sel / 2)]),
                    ("半分以上の窓で外れる", cs[cs["rej"] >= n_sel / 2])) if len(s)))
    return out


def report(results: dict, shifts: list, seeds, stamp: dict, cms: dict, sels: dict, base: list, groups: dict,
           date_cols: list) -> dict:
    algos = [a for a in ALGOS if results.get(a)]
    missing = [a for a in ALGOS if not results.get(a)]
    print("\n" + "=" * 78)
    print(f"ラベル {K:.1f}σ / 結果のそろったモデル: {', '.join(map(short_name, algos)) or 'なし'}"
          + (f" / まだ: {', '.join(map(short_name, missing))}" if missing else ""))
    summary = {"data": stamp, "label_k": K, "boruta": {"max_iter": MAX_ITER_IN_USE, "alpha": ALPHA, "perc": PERC,
                                                        "seed": BORUTA_SEED, "min_date_rows": MIN_DATE_ROWS},
               "r_seed": R_SEED, "pi_seed": PI_SEED, "date_level_columns": date_cols,
               "selection_sig": selection_sig(sels),
               "production_oof": {}, "windows": {}, "judge7": {}, "per_window": []}
    summary["selection"] = selection_report(sels, base, groups, date_cols)

    if algos and 0 in shifts:
        print(f"\n■ 2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種{PROD_SEED}）")
        print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}"
              f"{'最悪':>9}{'件数':>8}")
        for a in algos:
            ms = {}
            for arm in ARMS:
                o = results[a]["prod"][arm]
                m = metrics(o)
                ms[arm] = m
                summary["production_oof"][f"{short_name(a)}_{arm}"] = {kk: float(v) for kk, v in m.items()}
                print(f"  {short_name(a):<7}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}"
                      f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
                      f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt"
                      f"{len(o):>8,}")
            for x, y in PAIRS:
                print(f"  {short_name(a):<7}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}"
                      f"{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}"
                      f"{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    s = window_table(results, algos, shifts, cms) if algos else pd.DataFrame()
    complete = (all(a in algos for a in ALGOS) and sorted(shifts) == [0, 2, 4] and tuple(seeds) == SEEDS
                and MAX_ITER_IN_USE == MAX_ITER)
    if len(s):
        s.to_csv(path("auc_by_window.csv"), index=False)
        print(f"\n■ 3. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り。種 {len(seeds)}つの平均）")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for a in algos:
            g = s[s["algo"] == a]
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                for x, y in PAIRS:
                    line, st = pair(f"{short_name(a)} {nm} {y}−{x}", g[f"{met}_{x}"].to_numpy(),
                                    g[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{short_name(a)}_{met}_{y}-{x}"] = st
        print("\n■ 4. 判定（回す前に決めた。§7 を「外す」向きに: 窓の PR-AUC の平均 S−A が正 / 対照 R−A を上回る / "
              "S が上の窓が過半）")
        judged = {}
        for a in algos:
            g = s[s["algo"] == a]
            j = judge7((g["pr_S"] - g["pr_A"]).to_numpy(), (g["pr_R"] - g["pr_A"]).to_numpy())
            judged[a] = j
            summary["judge7"][short_name(a)] = j
            print(f"  {short_name(a):<7}S−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / "
                  f"R−A {j['mean_pa']:>+8.4f} {'○' if j['beats_placebo'] else '×'} / "
                  f"上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）{'○' if j['majority'] else '×'} "
                  f"→ {'満たす' if j['pass'] else '満たさない'}")
        v = verdict(judged, complete)
        summary["judge7"]["verdict"] = v
        print(f"  → {v['text']}")

        print("\n■ 5. 窓ごとの値（PR-AUC。差は「後 − 前」。S・R の列の数は窓ごと）")
        for a in algos:
            for sh in shifts:
                g = s[(s["algo"] == a) & (s["shift"] == sh)].sort_values("fold")
                if not len(g):
                    continue
                print(f"  {short_name(a)}・ずらし{sh}か月（{len(g)}窓）")
                print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}{'S列':>5}{'A':>8}{'S':>8}{'R':>8}"
                      f"{'S−A':>9}{'R−A':>9}{'S−R':>9}")
                for _, r in g.iterrows():
                    print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                          f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%{int(r['n_cols_S']):>5}{r['pr_A']:>8.4f}"
                          f"{r['pr_S']:>8.4f}{r['pr_R']:>8.4f}{r['pr_S'] - r['pr_A']:>+9.4f}"
                          f"{r['pr_R'] - r['pr_A']:>+9.4f}{r['pr_S'] - r['pr_R']:>+9.4f}")
        summary["per_window"] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
    if algos:
        summary["perm_importance"] = pi_report(results, algos, shifts, summary["selection"].get("rejected_count", {}),
                                               len(sels))
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


#: この回の Boruta の最大回数（試運転で減らしたときは判定を「参考」にする）
MAX_ITER_IN_USE = MAX_ITER


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE, MAX_ITER_IN_USE
    ap = argparse.ArgumentParser(description="実験91: Boruta で本番の239列を絞ると、木3モデル（1.5σ）の分離力が上がるか")
    ap.add_argument("--algos", default="lgbm,xgb,cat", help="この回に計算するモデル（lgbm / xgb / cat）")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--max-iter", type=int, default=MAX_ITER, help="Boruta の最大回数（試運転で減らす）")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0,
                    help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--select-only", action="store_true", help="Boruta の選定だけ回して報告する")
    ap.add_argument("--dry", action="store_true", help="行・日付単位の列・影の単位だけ出して終わる")
    args = ap.parse_args(argv)
    TAG = args.tag
    RUN = f"{TAG}_k15"
    MAX_ITER_IN_USE = args.max_iter
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    algos = [algo_name(a) for a in args.algos.split(",") if a.strip()]
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    base = F.columns(F.DEFAULT_PRESET)
    groups = {g: [c for c in F.GROUPS[g] if c in set(base)] for g in F.PRESETS[F.DEFAULT_PRESET]}

    print("=" * 78)
    print("実験91 Boruta（窓の中で選ぶ・日付に合わせた影）で本番の239列を絞ると、木3モデル（1.5σ）の分離力が上がるか")
    print(f"  列: {F.DEFAULT_PRESET} {len(base)}列 指紋 {F.signature(base)} / グループ {len(groups)}")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  ラベル {K:.1f}σ / この回に計算: {', '.join(map(short_name, algos))} / 選定のモデル {SELECT_ALGO}（gain）/ "
          f"Boruta 最大 {args.max_iter}回・α {ALPHA}・影の {PERC} 百分位・種 {BORUTA_SEED} / 窓のずらし {shifts}か月 / "
          f"種 {seeds} / 対照の種 {R_SEED} / 入れ替え重要度の種 {PI_SEED} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    full, stamp = prepare(args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    miss = [c for c in base if c not in full.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    df = with_label(full, K)
    date_cols = date_level_columns(df, base)
    sh_all = Shuffler(df["Date"])
    n_day = df.groupby("Date").size()
    print(f"\n■ 0. 行（ラベル {K:.1f}σ）: {len(df):,}行・正例率 {df['label'].mean()*100:.1f}%・"
          f"{df['Date'].min().date()} 〜 {df['Date'].max().date()}・{len(n_day):,}日")
    print(f"  日付単位の列（その日は全銘柄同じ値）{len(date_cols)}本: " + ", ".join(date_cols))
    print(f"  影の単位: {int(sh_all.groups.max()) + 1 if len(sh_all.groups) else 0:,}（1行の日 {int((n_day < MIN_DATE_ROWS).sum()):,}日・"
          f"{int(n_day[n_day < MIN_DATE_ROWS].sum()):,}行は同じ月の1行の日どうしでまとめた）")
    if args.dry:
        return 0

    folds0 = E41.folds_for(df["Date"], 0)
    sel_params = params_of(SELECT_ALGO, stamp)
    log("[params] 1回目に控えたパラメータ: " + " / ".join(f"{short_name(a)} {params_hash(params_of(a, stamp))}"
                                                     for a in ALGOS))
    sels = {}
    log(f"[Boruta] ずらし0の {len(folds0)}窓で選ぶ（保存済みの窓は読む）")
    for f in folds0:
        rec = select_window(df, base, f, sel_params, stamp, date_cols, args.max_iter)
        if rec is not None:
            sels[int(f.index)] = rec
    log(f"[Boruta] 選定のそろった窓 {len(sels)}/{len(folds0)}・選定の指紋 {selection_sig(sels)}")
    folds_by_shift = {sh: E41.folds_for(df["Date"], sh) for sh in shifts}
    cms = colmaps(base, folds_by_shift, folds0, sels)
    pi = {"groups": groups, "date_cols": set(date_cols)}

    results = {}
    if not args.select_only:
        if len(sels) < len(folds0):
            log(f"[Boruta] 選定のそろっていない窓がある（{len(sels)}/{len(folds0)}）。そろってから out-of-fold を作る")
        else:
            for a in ALGOS:
                t0 = time.time()
                results[a] = run(a, df, cms, stamp, shifts, seeds, pi, compute=a in algos)
                if a in algos:
                    log(f"[{short_name(a)}] {'そろった' if results[a] else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    report(results, shifts, seeds, stamp, cms, sels, base, groups, date_cols)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの選定・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
