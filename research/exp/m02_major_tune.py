#!/usr/bin/env python3
"""
本ブレイク予測モデル: 新しいラベル用にパラメータを探索し、日次ボラ1本を上回るかを見る（ブランチ
claude/major-breakout-model）。

運用者の依頼（2026-09-23）「１をお願いします」（= このラベル用に205列でパラメータを探索し、
合格ラインを「日次ボラ1本を上回ること」に置く）。以前の指示「パラメータチューニングする際も
205 全特徴量を使ったモデルでお願いします」「oofで最良にすることがチューニングの目的ではないです」。

なぜ今のモデルの探索のやり方を使わないか
  今の探索（tuning.year_folds の year_cap_date）は、日付をまとめて年×時価総額帯で層別した
  k 分割で、訓練と検証が同じ時期を含む。20営業日のラベルならまだよいが、新しいラベルは
  120営業日先まで見るので、同じ銘柄の重なった期間が両側に入り、検証が簡単になる
  （その銘柄が走った期間を覚えるだけで当たる）。探索はそういうパラメータを選んでしまう。
  そこで、時期で5つの塊に切り、検証の塊の前後121営業日を学習から外す（purged k-fold）。
  前は「その行のラベルが検証期間の値動きを含む」、後は「その行の特徴量が検証のラベルと
  同じ値動きを含む」ため。目的関数は塊ごとの PR-AUC ÷ 正例率（リフト）の平均
  （塊によって正例率が 1〜4% と違うので、PR-AUC のままだと正例率の高い塊で決まる）

なぜ評価を窓6〜10に絞るか
  探索は窓6（2023-11〜）の学習の最後（テスト開始の121営業日前 = 2023-05-10）までの行で行う。
  窓1〜5 は探索に使った期間と重なるので、そこで測ると「評価の答えを見て探索した」ことになる。
  窓6〜10（2023-11〜2026-03）は探索に一度も使っていない

比べるもの（すべて窓6〜10・同じ行）
  探索したパラメータの3モデル（種3つの平均）と、その平均順位
  本番のパラメータの3モデル（m01 と同じ）と、その平均順位
  列1本: 日次ボラが高い順（合格ライン）ほか
  差の幅: 月ごとに入れ替えるブートストラップ（1,000回）で、平均順位 − 日次ボラ の PR-AUC リフトと
  ROC-AUC の差の 5〜95% 点
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "major"))
import features as F  # noqa: E402
import lab  # noqa: E402
import tuning_multi as TM  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import label_eda as LE  # noqa: E402
import m01_major_oof as M1  # noqa: E402

OOF_DIR = M1.OOF_DIR
ALGOS = ("lgbm", "xgb", "cat")
SEEDS = M1.SEEDS
EVAL_FROM = 6                   # 評価する最初の窓（探索に使わなかった窓）
N_SPLITS = 5                    # 探索の塊の数
N_TRIALS = 50                   # 本番の探索と同じ試行数
N_BOOT = 1000
PURGE = M1.PURGE                # 121営業日
LABEL_VERSION = "v3"            # ラベルの定義を変えたら上げる（探索の引き継ぎを切るため）
PREFIX = "m02"                  # 保存するファイルの頭（--quick では別にする）
#: ボラ系として外す列（運用者の案 2026-09-23「シンプルに日次ボラを使うのと、それ（ボラ系）を除いて
#: モデル構築して、2つの予測結果を使うのはどうでしょうか？後者はファンダメンタルズ的（ボラ以外の
#: 要素で）に、本ブレイク起こりそうかを判断してもらうのが意図です」）
#:   株価・出来高から作る列のグループと、全銘柄で同じ値になる地合いの列。
#:   残りのうち、探索に使う期間の行で日次ボラとの順位相関の絶対値が RHO_MAX 以上の列も外す
#:   （vol_20d を1本抜くだけだと、直近20日の上昇率 +0.71 などから荒さを組み立て直すため）
VOL_GROUPS = ("price", "breakout", "volume", "liquidity", "sector_index", "margin_alert",
              "supply", "market", "flow")
VOL_EXTRA = ("cap_band",)
RHO_MAX = 0.3
VOL_TOP = (0.2, 0.4)            # 「ボラで絞ってからファンダで並べる」の絞り方。結果を見る前に決めた


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def purged_blocks(d: pd.DataFrame, cal: pd.DatetimeIndex, k: int = N_SPLITS, purge: int = PURGE) -> list:
    """
    時期で k 個の塊に切り（行数で等分）、塊ごとに「その塊を検証、前後 purge 営業日を除いた
    残りを学習」にする。d は Date 列を持ち、日付順でなくてよい。
    """
    dates = pd.to_datetime(d["Date"])
    order = np.argsort(dates.to_numpy(), kind="stable")
    blocks = np.array_split(order, k)
    pos = cal.searchsorted(dates.to_numpy())               # 営業日の番号
    out = []
    for b in blocks:
        lo, hi = pos[b].min(), pos[b].max()
        va = np.zeros(len(d), dtype=bool)
        va[b] = True
        tr = ~va & ((pos < lo - purge) | (pos > hi + purge))
        out.append((np.flatnonzero(tr), np.flatnonzero(va)))
    return out


def cv_lift(algo: str, params: dict, d: pd.DataFrame, cols: list, folds: list, label: str = "y") -> dict:
    """探索の1試行ぶん: 塊ごとに学習・検証して、リフト（PR-AUC ÷ 正例率）と ROC-AUC を返す。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    X = d[cols].to_numpy(dtype=float)
    y = d[label].to_numpy(dtype=int)
    lifts, rocs = [], []
    for tr, va in folds:
        if y[tr].sum() < 5 or y[va].sum() < 3:
            continue
        m = TM.build(algo, params, y[tr], cols)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        lifts.append(average_precision_score(y[va], p) / y[va].mean())
        rocs.append(roc_auc_score(y[va], p))
    return {"lift": float(np.mean(lifts)), "lift_sd": float(np.std(lifts)),
            "roc": float(np.mean(rocs)), "folds": [round(x, 3) for x in lifts]}


def tune(algo: str, d: pd.DataFrame, cols: list, folds: list, tag: str, kind: str = "") -> dict:
    """Optuna（TPE、種0）で N_TRIALS 回。途中で止まっても続きから回せるように保存する。"""
    import optuna
    path = os.path.join(OOF_DIR, f"{PREFIX}_params_{kind + '_' if kind else ''}{algo}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("_cv", {}).get("tag") == tag:
            log(f"  [{algo}] 探索済みを読む（CV リフト {rec['_cv']['lift']:.3f}）")
            return rec
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    TM.SEED = 0
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
        storage=f"sqlite:///{os.path.join(OOF_DIR, PREFIX + '_optuna.db')}",
        study_name=f"{kind + '_' if kind else ''}{algo}_{tag}", load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    t0 = time.time()

    def objective(trial):
        r = cv_lift(algo, TM.SPACES[algo](trial), d, cols, folds)
        trial.set_user_attr("roc", r["roc"])
        trial.set_user_attr("lift_sd", r["lift_sd"])
        trial.set_user_attr("folds", r["folds"])
        return r["lift"]

    if N_TRIALS - done > 0:
        study.optimize(objective, n_trials=N_TRIALS - done, show_progress_bar=False)
    at = study.best_trial.user_attrs
    rec = {"params": dict(study.best_params),
           "_cv": {"tag": tag, "lift": round(study.best_value, 4), "lift_sd": round(at["lift_sd"], 4),
                   "roc": round(at["roc"], 4), "folds": at["folds"], "n_trials": N_TRIALS,
                   "n_splits": len(folds), "purge": PURGE, "seconds": round(time.time() - t0),
                   "trials_resumed": done}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    log(f"  [{algo}] 探索 {time.time() - t0:.0f}秒（引き継ぎ {done}試行）/ CV リフト {rec['_cv']['lift']:.3f}"
        f"（±{rec['_cv']['lift_sd']:.3f}）/ ROC {rec['_cv']['roc']:.3f}")
    return rec


def exvol_columns(cols: list, rows: pd.DataFrame) -> tuple:
    """ボラ系を外した列。戻り値: 残す列、グループで外した列、相関で外した列（相関つき）。"""
    by_group = [c for g in VOL_GROUPS for c in F.GROUPS[g] if c in cols] + [c for c in VOL_EXTRA if c in cols]
    rest = [c for c in cols if c not in set(by_group)]
    rho = rows[rest].rank().corrwith(rows["vol_20d"].rank())
    by_rho = rho[rho.abs() >= RHO_MAX].sort_values(key=lambda v: -v.abs())
    keep = [c for c in rest if c not in set(by_rho.index)]
    return keep, by_group, by_rho


def cv_predict(algo: str, params: dict, d: pd.DataFrame, cols: list, folds: list) -> tuple:
    """探索期間の各塊を、その塊を外して学習したモデルで予測する（組み合わせ方の重みを学ぶため）。"""
    X, y = d[cols].to_numpy(dtype=float), d["y"].to_numpy(dtype=int)
    out, blk = np.full(len(d), np.nan), np.full(len(d), -1)
    TM.SEED = 0
    for i, (tr, va) in enumerate(folds):
        m = TM.build(algo, params, y[tr], cols)
        m.fit(X[tr], y[tr])
        out[va] = m.predict_proba(X[va])[:, 1]
        blk[va] = i
    return out, blk


def logit_pct(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """塊（窓）の中の順位を 0〜1 にして、ロジットに直す（重みを足し算で学ぶため）。"""
    q = pd.Series(x).groupby(group).rank(pct=True).to_numpy()
    q = np.clip(q, 0.001, 0.999)
    return np.log(q / (1 - q))


def oof_built(algo: str, seed: int, d: pd.DataFrame, cols: list, params: dict, plan: list) -> pd.DataFrame:
    """探索したパラメータで out-of-fold。組み方は探索と同じ TM.build（木200本）。"""
    parts = []
    try:
        for f, cut in plan:
            if cut is None:
                continue
            tr = d[(d["Date"] <= cut) & d["y"].notna()]
            te = d[(d["Date"] >= pd.Timestamp(f.test_start)) & (d["Date"] <= pd.Timestamp(f.test_end))
                   & d["y"].notna()]
            if len(te) < M1.MIN_TEST or len(tr) < M1.MIN_TRAIN or tr["y"].sum() < M1.MIN_TRAIN_POS:
                continue
            TM.SEED = seed
            ytr = tr["y"].to_numpy(dtype=int)
            m = TM.build(algo, params, ytr, cols)
            m.fit(tr[cols].to_numpy(dtype=float), ytr)
            part = te[["Code", "Date"]].copy()
            part["score"] = m.predict_proba(te[cols].to_numpy(dtype=float))[:, 1]
            part["fold"] = f.index
            parts.append(part)
    finally:
        TM.SEED = 0
    return pd.concat(parts, ignore_index=True)


def seed_avg(name: str, make) -> pd.DataFrame:
    """種3つの確率平均。種ごとに保存し、2回目以降は読む。"""
    outs = []
    for sd in SEEDS:
        path = os.path.join(OOF_DIR, f"{PREFIX}_{name}_s{sd}.parquet")
        if os.path.exists(path):
            o = pd.read_parquet(path)
            o["Date"] = pd.to_datetime(o["Date"])
        else:
            t0 = time.time()
            o = make(sd)
            o.to_parquet(path, index=False)
            log(f"  [{name}] 種{sd} {time.time() - t0:.0f}秒（{o['fold'].nunique()}窓・{len(o):,}行）")
        outs.append(o.set_index(["Code", "Date"]))
    base = outs[0].copy()
    base["score"] = np.mean([o["score"].reindex(base.index).to_numpy() for o in outs], axis=0)
    return base.reset_index()


def boot_diff(y, a, b, month, fold, n=N_BOOT, seed=0) -> dict:
    """月ごとに入れ替えるブートストラップで、a − b の リフトと ROC-AUC の差の分布。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    ms = np.unique(month)
    idx = {m: np.flatnonzero(month == m) for m in ms}
    qa = pd.Series(a).groupby(fold).rank(pct=True).to_numpy()
    qb = pd.Series(b).groupby(fold).rank(pct=True).to_numpy()
    dl, dr = [], []
    for _ in range(n):
        pick = np.concatenate([idx[m] for m in rng.choice(ms, size=len(ms), replace=True)])
        yy = y[pick]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        base = yy.mean()
        dl.append((average_precision_score(yy, qa[pick]) - average_precision_score(yy, qb[pick])) / base)
        dr.append(roc_auc_score(yy, qa[pick]) - roc_auc_score(yy, qb[pick]))
    dl, dr = np.array(dl), np.array(dr)
    return {"lift_p05": float(np.percentile(dl, 5)), "lift_p50": float(np.percentile(dl, 50)),
            "lift_p95": float(np.percentile(dl, 95)), "lift_pos": float((dl > 0).mean()),
            "roc_p05": float(np.percentile(dr, 5)), "roc_p50": float(np.percentile(dr, 50)),
            "roc_p95": float(np.percentile(dr, 95)), "roc_pos": float((dr > 0).mean()), "n": len(dl)}


def main(argv=None) -> int:
    global N_TRIALS, SEEDS, N_BOOT, PREFIX
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true", help="手元の動作確認用（探索3試行・種1つ・入れ替え50回）")
    args = ap.parse_args(argv)
    if args.quick:
        N_TRIALS, SEEDS, N_BOOT, PREFIX = 3, (42,), 50, "m02quick"
    os.makedirs(OOF_DIR, exist_ok=True)

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    bars = LE.load_bars()
    cal = pd.DatetimeIndex(np.sort(bars["Date"].unique()))
    df["y"] = LE.label_frame(df[["Code", "Date"]], bars)["y_major"].to_numpy()
    fw = LE.forward_closes(bars, df[["Code", "Date"]])
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        df["mx"] = np.nanmax(fw["C"] / fw["entry"][:, None], axis=1) - 1.0
    cols = [c for c in F.columns(F.DEFAULT_PRESET) if c in df.columns]
    if len(cols) != len(F.columns(F.DEFAULT_PRESET)):
        raise SystemExit("本番の列がデータセットにそろっていない")

    plan = M1.fold_plan(df["Date"], cal)
    cut6 = dict((f.index, cut) for f, cut in plan)[EVAL_FROM]
    tune_rows = df[(df["Date"] <= cut6) & df["y"].notna()].reset_index(drop=True)
    folds = purged_blocks(tune_rows, cal)
    tag = f"{LABEL_VERSION}_{F.signature(cols)}_{cut6.date()}_k{N_SPLITS}_p{PURGE}"
    print("=== 0. 前提 ===")
    print(f"  列: {F.DEFAULT_PRESET}（{len(cols)}列）/ 探索に使う行: 〜{cut6.date()} の {len(tune_rows):,}件"
          f"（正例 {int(tune_rows['y'].sum())}件）")
    print(f"  探索の分割: 時期で{N_SPLITS}つの塊。検証の塊の前後{PURGE}営業日を学習から外す")
    for i, (tr, va) in enumerate(folds, 1):
        dv = tune_rows["Date"].iloc[va]
        print(f"    塊{i}: 検証 {dv.min().date()}〜{dv.max().date()} {len(va):,}件（正例 {int(tune_rows['y'].iloc[va].sum())}）"
              f" / 学習 {len(tr):,}件（正例 {int(tune_rows['y'].iloc[tr].sum())}）")
    ev = [(f, cut) for f, cut in plan if f.index >= EVAL_FROM]
    print(f"  評価: 窓{EVAL_FROM}〜（探索に使っていない期間）。学習はテスト開始の{PURGE}営業日前まで（m01 と同じ）")

    print(f"\n=== 1. 探索（{N_TRIALS}試行 × {N_SPLITS}塊。目的 = 塊ごとのリフトの平均）===")
    tuned, prod_cv = {}, {}
    for a in ALGOS:
        tuned[a] = tune(a, tune_rows, cols, folds, tag)
    print(f"  {'モデル':<10}{'探索後 CV リフト':>16}{'塊ごと':>34}{'ROC':>7}   パラメータ")
    for a in ALGOS:
        c = tuned[a]["_cv"]
        print(f"  {a:<10}{c['lift']:>12.3f}±{c['lift_sd']:.2f}{str(c['folds']):>36}{c['roc']:>7.3f}   {tuned[a]['params']}")

    # --- 評価（窓6〜10）--- #
    print(f"\n=== 2. 窓{EVAL_FROM}〜で out-of-fold（種{len(SEEDS)}つの平均）===")
    sc_t, sc_p = {}, {}
    for a in ALGOS:
        sc_t[a] = seed_avg(f"tuned_{a}_{tag}", lambda sd, a=a: oof_built(a, sd, df, cols, tuned[a]["params"], ev))
        prod = E27.prod_params(a)
        sc_p[a] = seed_avg(f"prod_{a}", lambda sd, a=a, prod=prod: M1.oof(
            a, sd, df, cols, prod, ev, "y", lambda f, cut: cut))
    key = ["Code", "Date"]
    common = sc_t["lgbm"][key + ["fold"]]
    for o in [sc_t[a] for a in ALGOS[1:]] + [sc_p[a] for a in ALGOS]:
        common = common.merge(o[key], on=key)
    common = common.merge(df[key + ["y", "ret_o1_60", "ret_o1_120", "mx", "vol_20d"]
                             + [c for c, _, _ in M1.SINGLE if c != "vol_20d"]], on=key, how="left")
    y = common["y"].to_numpy(dtype=int)
    mi = pd.MultiIndex.from_frame(common[key])

    def on_common(o):
        return common[key + ["fold"]].assign(score=o.set_index(key)["score"].reindex(mi).to_numpy())

    rows = {}
    for a in ALGOS:
        rows[f"t_{a}"] = on_common(sc_t[a])
        rows[f"p_{a}"] = on_common(sc_p[a])
    for g in ("t", "p"):
        e = common[key + ["fold"]].copy()
        e["score"] = np.mean([M1.pct_in_fold(rows[f"{g}_{a}"]) for a in ALGOS], axis=0)
        rows[f"{g}_ens"] = e
    for c, sign, _ in M1.SINGLE:
        rows[c] = common[key + ["fold"]].assign(score=sign * common[c].to_numpy(dtype=float))
    names = {**{f"t_{a}": f"探索後 {a}" for a in ALGOS}, "t_ens": "探索後 3モデルの平均順位",
             **{f"p_{a}": f"本番の値 {a}" for a in ALGOS}, "p_ens": "本番の値 3モデルの平均順位",
             **{c: f"列1本: {lab_}" for c, _, lab_ in M1.SINGLE}}
    order = ["t_ens", "t_lgbm", "t_xgb", "t_cat", "p_ens", "p_lgbm", "p_xgb", "p_cat"] + [c for c, _, _ in M1.SINGLE]
    print(f"  窓{EVAL_FROM}〜{common['fold'].max()}: {len(common):,}行・正例 {int(y.sum())}件・正例率 {y.mean()*100:.2f}%")
    print(M1.HEAD)
    res = {}
    for k in order:
        o = rows[k]
        ok = np.isfinite(o["score"].to_numpy(dtype=float))
        res[k] = M1.measure(o[ok], y[ok], common[ok])
        print(M1.line(names[k], res[k]))

    print("\n=== 3. 合格ライン（日次ボラ1本）との差。月ごとに入れ替えるブートストラップ ===")
    month = common["Date"].dt.to_period("M").astype(str).to_numpy()
    fold = common["fold"].to_numpy()
    vol = common["vol_20d"].to_numpy(dtype=float)
    ok = np.isfinite(vol)
    print(f"  {'比べる組':<40}{'リフトの差 5%':>12}{'中央':>8}{'95%':>8}{'差>0':>7}"
          f"{'ROCの差 5%':>11}{'中央':>8}{'95%':>8}{'差>0':>7}")
    for k in ("t_ens", "t_lgbm", "t_xgb", "t_cat", "p_ens"):
        a_ = rows[k]["score"].to_numpy(dtype=float)
        b = boot_diff(y[ok], a_[ok], vol[ok], month[ok], fold[ok])
        print(f"  {names[k] + ' − 日次ボラ':<40}{b['lift_p05']:>+12.2f}{b['lift_p50']:>+8.2f}{b['lift_p95']:>+8.2f}"
              f"{b['lift_pos']*100:>6.0f}%{b['roc_p05']:>+11.3f}{b['roc_p50']:>+8.3f}{b['roc_p95']:>+8.3f}{b['roc_pos']*100:>6.0f}%")
    b = boot_diff(y, rows["t_ens"]["score"].to_numpy(dtype=float), rows["p_ens"]["score"].to_numpy(dtype=float),
                  month, fold)
    print(f"  {'探索後 − 本番の値（3モデルの平均順位）':<40}{b['lift_p05']:>+12.2f}{b['lift_p50']:>+8.2f}{b['lift_p95']:>+8.2f}"
          f"{b['lift_pos']*100:>6.0f}%{b['roc_p05']:>+11.3f}{b['roc_p50']:>+8.3f}{b['roc_p95']:>+8.3f}{b['roc_pos']*100:>6.0f}%")
    print("  リフトの差 = PR-AUC の差 ÷ 正例率。スコアは窓の中の順位にそろえてから比べる。"
          f"月（{len(np.unique(month))}か月）を入れ替えて {N_BOOT}回。")

    print("\n=== 4. 日次ボラの近い銘柄どうしで（ボラの五分位ごとの ROC-AUC、件数で重み付け）===")
    from sklearn.metrics import roc_auc_score
    qv = pd.qcut(pd.Series(vol[ok]).rank(method="first"), 5, labels=False).to_numpy()
    band = np.full(len(vol), -1)
    band[ok] = qv
    for k in ("t_ens", "t_lgbm", "t_xgb", "t_cat", "p_ens", "vol_20d"):
        s = rows[k]["score"].to_numpy(dtype=float)
        acc = []
        for b_ in range(5):
            m = band == b_
            if 0 < y[m].sum() < m.sum():
                acc.append((roc_auc_score(y[m], s[m]), int(m.sum())))
        w = sum(a_ * n for a_, n in acc) / sum(n for _, n in acc)
        print(f"  {names[k]:<34}{w:.3f}   （五分位ごと {' / '.join(f'{a_:.3f}' for a_, _ in acc)}）")

    print("\n=== 5. 上位10%に入った行のその後（買値から）===")
    print(f"  {'並べ方':<34}{'60日後(5日平均)':>14}{'120日後(5日平均)':>16}{'120日の最高値':>14}")
    r0 = res["t_ens"]
    print(f"  {'（全体）':<34}{r0['ret_o1_60_all']*100:>+13.1f}%{r0['ret_o1_120_all']*100:>+15.1f}%{r0['mx_all']*100:>+13.1f}%")
    for k in ("t_ens", "p_ens", "vol_20d"):
        r = res[k]
        print(f"  {names[k]:<34}{r['ret_o1_60_top10']*100:>+13.1f}%{r['ret_o1_120_top10']*100:>+15.1f}%{r['mx_top10']*100:>+13.1f}%")

    # ------------------------------------------------------------------ #
    print("\n=== 6. ボラ系を外したモデル（ファンダメンタルズ中心）===")
    keep, by_group, by_rho = exvol_columns(cols, tune_rows)
    print(f"  外した列: 株価・出来高・地合いのグループ {len(by_group)}本 / 日次ボラとの順位相関 |ρ|≥{RHO_MAX} の"
          f"{len(by_rho)}本 → 残り {len(keep)}本")
    print("  相関で外した列: " + " / ".join(f"{c} {v:+.2f}" for c, v in by_rho.items()))
    tag_x = f"{LABEL_VERSION}_{F.signature(keep)}_{cut6.date()}_k{N_SPLITS}_p{PURGE}"
    tuned_x = {a: tune(a, tune_rows, keep, folds, tag_x, kind="exvol") for a in ALGOS}
    for a in ALGOS:
        c = tuned_x[a]["_cv"]
        print(f"  {a:<6} 探索後 CV リフト {c['lift']:.3f}±{c['lift_sd']:.2f}  ROC {c['roc']:.3f}  塊ごと {c['folds']}")
    sc_x = {a: seed_avg(f"exvol_{a}_{tag_x}", lambda sd, a=a: oof_built(a, sd, df, keep, tuned_x[a]["params"], ev))
            for a in ALGOS}
    for a in ALGOS:
        rows[f"x_{a}"] = on_common(sc_x[a])
        names[f"x_{a}"] = f"ボラ系なし {a}"
    e = common[key + ["fold"]].copy()
    e["score"] = np.mean([M1.pct_in_fold(rows[f"x_{a}"]) for a in ALGOS], axis=0)
    rows["x_ens"], names["x_ens"] = e, "ボラ系なし 3モデルの平均順位"

    # 組み合わせ方（窓の中の順位で作る）
    qv_ = pd.Series(np.where(ok, vol, -np.inf)).groupby(fold).rank(pct=True).to_numpy()
    qx_ = rows["x_ens"]["score"].to_numpy(dtype=float)
    comb = {"c_avg": ("ボラ + ボラ系なし 順位の平均", (qv_ + qx_) / 2)}
    for t in VOL_TOP:
        comb[f"c_top{int(t*100)}"] = (f"ボラ上位{int(t*100)}%をボラ系なしで並べる", (qv_ >= 1 - t) * 1.0 + qx_)
    # 探索期間の CV の予測で重みを学ぶ（評価の窓は使わない）
    from sklearn.linear_model import LogisticRegression
    pr_tune = []
    for a in ALGOS:
        o, blk = cv_predict(a, tuned_x[a]["params"], tune_rows, keep, folds)
        pr_tune.append(pd.Series(o).groupby(blk).rank(pct=True).to_numpy())
    qx_t = np.mean(pr_tune, axis=0)
    blk_t = blk
    Xt = np.c_[logit_pct(tune_rows["vol_20d"].fillna(tune_rows["vol_20d"].median()).to_numpy(), blk_t),
               logit_pct(qx_t, blk_t)]
    lr = LogisticRegression(C=100.0, max_iter=1000).fit(Xt, tune_rows["y"].to_numpy(dtype=int))
    w = lr.coef_[0]
    Xe = np.c_[logit_pct(np.where(ok, vol, np.nanmedian(vol)), fold), logit_pct(qx_, fold)]
    comb["c_lr"] = (f"学習した重み（ボラ {w[0]:.2f} : ボラ系なし {w[1]:.2f}）", Xe @ w)
    for k, (nm, sc) in comb.items():
        rows[k], names[k] = common[key + ["fold"]].assign(score=sc), nm

    from scipy.stats import spearmanr
    rs = spearmanr(rows["x_ens"]["score"][ok], vol[ok]).correlation
    rf = spearmanr(rows["t_ens"]["score"][ok], vol[ok]).correlation
    print(f"  評価の窓で、点数と日次ボラの順位相関: ボラ系なし {rs:+.2f} / 全列（探索後）{rf:+.2f}")

    print(f"\n=== 7. 2つの予測を組み合わせる（窓{EVAL_FROM}〜、{len(common):,}行・正例 {int(y.sum())}件）===")
    print(M1.HEAD)
    order7 = ["vol_20d", "x_ens", "x_lgbm", "x_xgb", "x_cat", "c_avg"] + [f"c_top{int(t*100)}" for t in VOL_TOP] \
        + ["c_lr", "t_ens"]
    for k in order7:
        o = rows[k]
        okk = np.isfinite(o["score"].to_numpy(dtype=float))
        res[k] = M1.measure(o[okk], y[okk], common[okk])
        print(M1.line(names[k], res[k]))
    print(f"\n  日次ボラ1本との差（月ごとに入れ替えるブートストラップ {N_BOOT}回）")
    print(f"  {'比べる組':<40}{'リフトの差 5%':>12}{'中央':>8}{'95%':>8}{'差>0':>7}"
          f"{'ROCの差 5%':>11}{'中央':>8}{'95%':>8}{'差>0':>7}")
    for k in ["x_ens", "c_avg"] + [f"c_top{int(t*100)}" for t in VOL_TOP] + ["c_lr", "t_ens"]:
        a_ = rows[k]["score"].to_numpy(dtype=float)
        b = boot_diff(y[ok], a_[ok], vol[ok], month[ok], fold[ok])
        print(f"  {names[k][:24] + ' − 日次ボラ':<40}{b['lift_p05']:>+12.2f}{b['lift_p50']:>+8.2f}{b['lift_p95']:>+8.2f}"
              f"{b['lift_pos']*100:>6.0f}%{b['roc_p05']:>+11.3f}{b['roc_p50']:>+8.3f}{b['roc_p95']:>+8.3f}{b['roc_pos']*100:>6.0f}%")
    print("\n  日次ボラの近い銘柄どうしで（五分位ごとの ROC-AUC、件数で重み付け）")
    for k in ("x_ens", "t_ens", "vol_20d"):
        s_ = rows[k]["score"].to_numpy(dtype=float)
        acc = [(roc_auc_score(y[band == b_], s_[band == b_]), int((band == b_).sum())) for b_ in range(5)
               if 0 < y[band == b_].sum() < (band == b_).sum()]
        print(f"  {names[k]:<34}{sum(a_ * n for a_, n in acc) / sum(n for _, n in acc):.3f}"
              f"   （{' / '.join(f'{a_:.3f}' for a_, _ in acc)}）")
    print("\n  上位10%に入った行のその後（買値から）")
    print(f"  {'並べ方':<34}{'60日後(5日平均)':>14}{'120日後(5日平均)':>16}{'120日の最高値':>14}")
    for k in ["vol_20d", "x_ens", "c_avg"] + [f"c_top{int(t*100)}" for t in VOL_TOP] + ["c_lr"]:
        r = res[k]
        print(f"  {names[k][:30]:<34}{r['ret_o1_60_top10']*100:>+13.1f}%{r['ret_o1_120_top10']*100:>+15.1f}%"
              f"{r['mx_top10']*100:>+13.1f}%")

    pd.DataFrame(res).T.to_csv(os.path.join(OOF_DIR, f"{PREFIX}_summary.csv"))
    log(f"記録: {OOF_DIR}/{PREFIX}_summary.csv / {PREFIX}_params_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
