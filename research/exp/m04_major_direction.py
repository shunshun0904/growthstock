#!/usr/bin/env python3
"""
本ブレイク予測モデル: 目的変数を「+30%に −20%より先に届くか」に変えて学習し直す（ブランチ claude/major-breakout-model）。

運用者の依頼（2026-09-23）「a.でお願いします」
  a = 学習させる答えを「+30%に −20%より先に届くか」に変える。今のラベル（60日で+50%かつ120日で2倍）は
  荒さを当てる形になっていて、3モデル98%超の候補の 56〜62% が +30% より先に −20% に触れていた（m03）。
  続けて「どちらにも届かなかった行は学習に使わず — これは、推論時は事前にはわからないので、
  このフィルターは無しでお願いします。」（届かなかった行も 0 として学習に入れる）
  以前の指示「パラメータチューニングする際も205 全特徴量を使ったモデルでお願いします」
  「oof だけでなく、各cv（異なる窓）でもみたいです」

ラベル A（買値 = ブレイク翌営業日の寄り。買った日を1日目として120営業日）
  1 = 場中の高値が +30% に、場中の安値が −20% より先に届いた（上が先）
  0 = それ以外。−20% が先（同じ日に両方なら下が先とみなす。控えめ）と、どちらにも届かなかった行（約半分）
  売買を数えられない行（120営業日たっていない・データの外）は使わない

探索: m02 と同じ組み方（205列・〜2023-05-10 の行・時期で5つの塊・検証の塊の前後121営業日を学習から外す・
  各50試行・Optuna TPE 種0）。目的は塊ごとの ROC-AUC の平均（正例率が 27〜52% あり、上位の当たりだけを
  見る PR-AUC のリフトより素直）
評価: m03 と同じ売買（3モデルがそろって前の窓の98%点・95%点を超えたら翌営業日の寄りで買い、+30%に届いたら
  売る。届かなければ120営業日目の終値。参考に −20%損切り）。今のラベルの3モデル（m02 のパラメータ）、
  ブレイク全件、日次ボラ98%超と、同じ窓・同じ行で比べる。窓の切り方3通り（0/2/4か月ずらし）
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
import label_eda as LE  # noqa: E402
import trade as TR  # noqa: E402
import m01_major_oof as M1  # noqa: E402
import m02_major_tune as M2  # noqa: E402
import m03_major_exit as M3  # noqa: E402

OOF_DIR = M1.OOF_DIR
ALGOS = M2.ALGOS
S3 = M3.S3
KEY = M3.KEY
PREFIX = "m04"
N_TRIALS = 50
LABEL_VERSION = "d1"            # ラベルの定義を変えたら上げる（探索の引き継ぎを切るため）
UP, DOWN = M3.UP, M3.DOWN       # +30% / −20%
LABELS = (("A", "届かなければ0"),)
SETS = ("今のラベル", "A")
SHIFTS = (0, 2, 4)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def labels_from(cat: np.ndarray, tradable: np.ndarray) -> tuple:
    """
    touch_order の区分からラベル。y_a（学習に使う）: 上が先 = 1、それ以外 = 0。
    y_b（学習には使わない。当てる力の内訳を見る参考だけ）: 上下どちらかに届いた行で、上が先 = 1。
    同じ日に両方（4）は下が先（0）。売買を数えられない行は NaN。
    """
    y_a = np.where(cat == 1, 1.0, 0.0)
    y_b = np.where(cat == 1, 1.0, np.where(cat == 5, np.nan, 0.0))
    y_a[~tradable] = np.nan
    y_b[~tradable] = np.nan
    return y_a, y_b


def cv_auc(algo: str, params: dict, d: pd.DataFrame, cols: list, folds: list, label: str) -> dict:
    """探索の1試行ぶん: 塊ごとに学習・検証して ROC-AUC を返す。"""
    from sklearn.metrics import roc_auc_score
    X = d[cols].to_numpy(dtype=float)
    y = d[label].to_numpy(dtype=int)
    aucs = []
    for tr, va in folds:
        if y[tr].sum() < 20 or y[va].sum() < 10 or (1 - y[va]).sum() < 10:
            continue
        m = TM.build(algo, params, y[tr], cols)
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    return {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)), "folds": [round(a, 4) for a in aucs]}


def tune_auc(algo: str, d: pd.DataFrame, cols: list, folds: list, tag: str, kind: str, label: str) -> dict:
    """Optuna（TPE、種0）で N_TRIALS 回。途中で止まっても続きから回せるように保存する。"""
    import optuna
    path = os.path.join(OOF_DIR, f"{M2.PREFIX}_params_{kind}_{algo}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("_cv", {}).get("tag") == tag and rec["_cv"].get("n_trials") == N_TRIALS:
            log(f"  [{kind} {algo}] 探索済みを読む（CV AUC {rec['_cv']['auc']:.4f}）")
            return rec
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    TM.SEED = 0
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
        storage=f"sqlite:///{os.path.join(OOF_DIR, M2.PREFIX + '_optuna.db')}",
        study_name=f"{kind}_{algo}_{tag}", load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    t0 = time.time()

    def objective(trial):
        r = cv_auc(algo, TM.SPACES[algo](trial), d, cols, folds, label)
        trial.set_user_attr("auc_sd", r["auc_sd"])
        trial.set_user_attr("folds", r["folds"])
        return r["auc"]

    if N_TRIALS - done > 0:
        study.optimize(objective, n_trials=N_TRIALS - done, show_progress_bar=False)
    at = study.best_trial.user_attrs
    rec = {"params": dict(study.best_params),
           "_cv": {"tag": tag, "auc": round(study.best_value, 4), "auc_sd": round(at["auc_sd"], 4),
                   "folds": at["folds"], "n_trials": N_TRIALS, "n_splits": len(folds), "purge": M2.PURGE,
                   "seconds": round(time.time() - t0), "trials_resumed": done}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    log(f"  [{kind} {algo}] 探索 {time.time() - t0:.0f}秒（引き継ぎ {done}試行）/ CV AUC {rec['_cv']['auc']:.4f}"
        f"（±{rec['_cv']['auc_sd']:.4f}）")
    return rec


def oof_label(algo: str, seed: int, d: pd.DataFrame, cols: list, params: dict, plan: list, label: str) -> pd.DataFrame:
    """ラベル label で学習した out-of-fold。テストは売買を数えられる行すべて（m03 と同じ行）。"""
    parts = []
    try:
        for f, cut in plan:
            if cut is None:
                continue
            tr = d[(d["Date"] <= cut) & d[label].notna()]
            te = d[(d["Date"] >= pd.Timestamp(f.test_start)) & (d["Date"] <= pd.Timestamp(f.test_end)) & d["tradable"]]
            if len(te) < M1.MIN_TEST or len(tr) < M1.MIN_TRAIN or tr[label].sum() < M1.MIN_TRAIN_POS:
                continue
            TM.SEED = seed
            ytr = tr[label].to_numpy(dtype=int)
            m = TM.build(algo, params, ytr, cols)
            m.fit(tr[cols].to_numpy(dtype=float), ytr)
            part = te[KEY].copy()
            part["score"] = m.predict_proba(te[cols].to_numpy(dtype=float))[:, 1]
            part["fold"] = f.index
            parts.append(part)
    finally:
        TM.SEED = 0
    return pd.concat(parts, ignore_index=True)


def power(bk, df, cat, mask: np.ndarray, y_a, y_b) -> str:
    """3モデルの平均順位（窓の中）で、A・B を当てる AUC と、上位2%・5%の中の上下の内訳。"""
    from sklearn.metrics import roc_auc_score
    fr = bk.fr
    q = np.mean([fr.groupby("fold")[c].rank(pct=True).to_numpy() for c in S3], axis=0)
    rows = fr["row"].to_numpy()
    out = []
    for y in (y_a, y_b):
        yy = y[rows]
        m = mask & np.isfinite(yy)
        out.append(roc_auc_score(yy[m], q[m]) if 0 < yy[m].sum() < m.sum() else float("nan"))
    s = f"{out[0]:>8.3f}{out[1]:>8.3f}"
    for top in (0.02, 0.05):
        qq = fr.assign(_q=q).groupby("fold")["_q"].rank(pct=True).to_numpy()
        m = mask & (qq > 1 - top)
        c = cat[rows[m]]
        s += (f"{int(m.sum()):>7}{(c == 1).mean()*100:>7.1f}%{np.isin(c, (2, 3, 4)).mean()*100:>7.1f}%"
              f"{(c == 5).mean()*100:>7.1f}%")
    return s


def main(argv=None) -> int:
    global N_TRIALS, SHIFTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true", help="手元の動作確認用（探索3試行・種1つ・ずらし0か月だけ）")
    args = ap.parse_args(argv)
    M2.PREFIX = PREFIX
    if args.quick:
        N_TRIALS, M2.SEEDS, SHIFTS = 3, (42,), (0,)
        M2.PREFIX = "m04quick"
    global_prefix = M2.PREFIX
    os.makedirs(OOF_DIR, exist_ok=True)

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)
    bars = LE.load_bars(extra=("AdjL",))
    cal = pd.DatetimeIndex(np.sort(bars["Date"].unique()))
    df["y"] = LE.label_frame(df[KEY], bars)["y_major"].to_numpy()
    cols = [c for c in F.columns(F.DEFAULT_PRESET) if c in df.columns]
    if len(cols) != len(F.columns(F.DEFAULT_PRESET)):
        raise SystemExit("本番の列がデータセットにそろっていない")
    plan0 = M1.fold_plan(df["Date"], cal)
    cut6 = dict((f.index, cut) for f, cut in plan0)[M2.EVAL_FROM]
    tuned_major, tag_major = M3.load_params(cols, cut6)

    P = TR.forward_hc(bars, df[KEY])
    df["tradable"] = P["tradable"]
    tradable = df["tradable"].to_numpy()
    ex = {name: TR.exit_target(P, target=tp) if sl is None else TR.exit_bracket(P, tp, sl) for name, tp, sl in M3.EXITS}
    cat, du, dd = TR.touch_order(P, UP, DOWN)
    df["y_A"], df["y_B"] = labels_from(cat, tradable)
    buy_all = cal.searchsorted(df["Date"].to_numpy()) + 1
    del bars, P

    print("=== 0. 前提 ===")
    n_t = int(tradable.sum())
    c_t = cat[tradable]
    print(f"  売買を数えられるブレイク {n_t:,}件: 上が先 {(c_t == 1).mean()*100:.1f}% / 下が先 "
          f"{np.isin(c_t, (2, 3, 4)).mean()*100:.1f}% / どちらにも届かない {(c_t == 5).mean()*100:.1f}%")
    print(f"  正例率（上が先 = 1、それ以外 = 0）: {np.nanmean(df['y_A'])*100:.1f}%。どちらにも届かなかった行も 0 として学習に入れる")
    print(f"  列: {F.DEFAULT_PRESET}（{len(cols)}列）/ 今のラベルのパラメータ: {tag_major}")

    # --- 1. 探索 --- #
    print(f"\n=== 1. 探索（{N_TRIALS}試行 × {M2.N_SPLITS}塊・前後{M2.PURGE}営業日を外す。目的 = 塊ごとの ROC-AUC の平均）===")
    tuned = {}
    for kind, name in LABELS:
        label = f"y_{kind}"
        rows = df[(df["Date"] <= cut6) & df[label].notna()].reset_index(drop=True)
        folds = M2.purged_blocks(rows, cal)
        tag = f"{LABEL_VERSION}{kind}_{F.signature(cols)}_{cut6.date()}_k{M2.N_SPLITS}_p{M2.PURGE}"
        print(f"  [{kind}: {name}] 探索に使う行 〜{cut6.date()} の {len(rows):,}件（正例 {rows[label].mean()*100:.1f}%）")
        tuned[kind] = {a: tune_auc(a, rows, cols, folds, tag, kind, label) for a in ALGOS}
        for a in ALGOS:
            c = tuned[kind][a]["_cv"]
            print(f"    {a:<5} CV AUC {c['auc']:.4f} ± {c['auc_sd']:.4f}  塊ごと {c['folds']}  {tuned[kind][a]['params']}")

    # --- 点数（窓ごとの out-of-fold）--- #
    books = {}
    for shift in SHIFTS:
        plan = M1.fold_plan(df["Date"], cal, shift)
        sc = {a: M2.seed_avg(f"trade_{a}_{tag_major}_sh{shift}",
                             lambda sd, a=a, plan=plan: M2.oof_trade(a, sd, df, cols, tuned_major[a], plan))
              for a in ALGOS}
        books[("今のラベル", shift)] = M3.Book(df, cal, plan, sc, ex, buy_all)
        for kind, _ in LABELS:
            sc = {a: M2.seed_avg(f"dir{kind}_{a}_sh{shift}_{LABEL_VERSION}",
                                 lambda sd, a=a, plan=plan, kind=kind: oof_label(
                                     a, sd, df, cols, tuned[kind][a]["params"], plan, f"y_{kind}"))
                  for a in ALGOS}
            books[(kind, shift)] = M3.Book(df, cal, plan, sc, ex, buy_all)

    # --- 2. 当てる力 --- #
    print("\n=== 2. 上が先を当てる力（ずらし0か月。3モデルの窓の中の平均順位）===")
    print(f"  {'点数':<14}{'期間':<14}{'AUC:A':>8}{'AUC:B':>8}  {'上位2%: 件数':>11}{'上が先':>8}{'下が先':>8}{'届かず':>8}"
          f"  {'上位5%: 件数':>11}{'上が先':>8}{'下が先':>8}{'届かず':>8}")
    for s in SETS:
        bk = books[(s, 0)]
        for pname in ("全窓", "探索に使っていない窓"):
            print(f"  {s:<14}{pname:<14}" + power(bk, df, cat, bk.part_mask(pname), df["y_A"].to_numpy(),
                                                  df["y_B"].to_numpy()))
    bk = books[("今のラベル", 0)]
    allm = bk.part_mask("全窓")
    c_all = cat[bk.fr["row"].to_numpy()[allm]]
    print(f"  （ブレイク全件 {int(allm.sum()):,}件: 上が先 {(c_all == 1).mean()*100:.1f}% / 下が先 "
          f"{np.isin(c_all, (2, 3, 4)).mean()*100:.1f}% / 届かず {(c_all == 5).mean()*100:.1f}%）")
    print("  AUC:A = 上が先（1）とそれ以外（0）を分ける力（学習したラベル）。AUC:B = 参考。上下どちらかに届いた行だけで、"
          "上が先を分ける力（向きを当てているか、動く銘柄を当てているかの内訳を見るため。学習には使っていない）")

    # --- 3. 売買 --- #
    print("\n=== 3. 売買の損益（3モデル98%超・95%超で買い、+30%に届いたら売る。届かなければ120営業日目の終値）===")
    for shift in SHIFTS:
        print(f"\n  --- ずらし{shift}か月 ---")
        print(M3.HEAD1)
        for part in M3.PARTS:
            ref = books[("今のラベル", shift)]
            mask = ref.part_mask(part)
            if not mask.any():
                continue
            mo = ref.months(mask)
            print(f"  [{part}] {mo:.0f}か月")
            for s in SETS:
                bk = books[(s, shift)]
                pm = bk.part_mask(part)
                for rule in ("3モデル 98%超", "3モデル 95%超"):
                    for xname in ("+30%", "+30%/−20%"):
                        print(M3.sline(bk, f"{s}・{rule.split()[1]}", xname, pm & bk.sel[rule], mo, bk.prio(rule)))
            print(M3.sline(ref, "日次ボラ98%超", "+30%", mask & ref.sel["日次ボラ 98%超"], mo, ref.prio("日次ボラ")))
            print(M3.sline(ref, "ブレイク全件", "+30%", mask, mo, ref.prio("3モデル")))
    print("  選び方の「98%超」「95%超」は、3モデルがそろってその窓より前の窓の点数のその%点を超えたもの")

    # --- 4. 上下どちらが先か --- #
    print(f"\n=== 4. 買った銘柄の上下どちらが先か（全窓。+30% と −20%）===")
    head = (f"  {'選び方':<20}{'件数':>6}" + "".join(f"{TR.ORDER[k]:>10}" for k in (1, 2, 3, 5))
            + f"{'上の日':>8}{'下の日':>8}{'日次ボラ':>9}")
    vol = df["vol_20d"].to_numpy(dtype=float)
    for shift in SHIFTS:
        print(f"\n  --- ずらし{shift}か月 ---")
        print(head)
        items = [(f"{s}・{r.split()[1]}", books[(s, shift)], books[(s, shift)].sel[r]) for s in SETS
                 for r in ("3モデル 98%超", "3モデル 95%超")]
        items.append(("ブレイク全件", books[("今のラベル", shift)], np.ones(len(books[("今のラベル", shift)].fr), bool)))
        for name, bk, m in items:
            ii = bk.fr["row"].to_numpy()[m & bk.part_mask("全窓")]
            c = cat[ii]
            if not len(ii):
                print(f"  {name:<20}{0:>6}")
                continue
            up_d = np.median(du[ii][c == 1]) if (c == 1).any() else float("nan")
            dn_d = np.median(dd[ii][np.isin(c, (2, 3))]) if np.isin(c, (2, 3)).any() else float("nan")
            print(f"  {name:<20}{len(ii):>6}" + "".join(f"{(c == k).mean()*100:>9.1f}%" for k in (1, 2, 3, 5))
                  + f"{up_d:>8.0f}{dn_d:>8.0f}{np.nanmedian(vol[ii]):>8.2f}%")
    print("  同じ日に両方は下が先に数えた（ずっと 0件）。日次ボラ = 買った銘柄の日次ボラ（vol_20d）の中央値。"
          "上の日・下の日 = +30%・−20% に触れるまでの営業日（中央値）")

    # --- 5. 窓ごと --- #
    print("\n=== 5. 窓ごと（ずらし0か月・98%超・+30%で売る）===")
    print(f"  {'窓':>3} {'テスト':<24}" + "".join(f"{s + ': 件数':>12}{'平均':>8}{'到達':>7}" for s in SETS)
          + f"{'全件: 平均':>11}")
    ref = books[("今のラベル", 0)]
    for k in sorted(np.unique(ref.fr["fold"][ref.fr["evald"]])):
        f = ref.wins[k]
        line = f"  {k:>3} {f.test_start}〜{f.test_end}"
        for s in SETS:
            bk = books[(s, 0)]
            m = (bk.fr["fold"] == k).to_numpy() & bk.sel["3モデル 98%超"]
            st = TR.summarize(*(x[bk.fr["row"].to_numpy()[m]] for x in ex["+30%"]))
            line += (f"{st['n']:>12}{st['mean']*100:>+7.1f}%{st['hit']*100:>6.0f}%" if st["n"]
                     else f"{0:>12}{'-':>8}{'-':>7}")
        mm = (ref.fr["fold"] == k).to_numpy()
        r_all = ex["+30%"][0][ref.fr["row"].to_numpy()[mm]]
        print(line + f"{r_all.mean()*100:>+10.1f}%  {'重ならない' if pd.Timestamp(f.test_start) >= M2.CLEAN_FROM else '重なる'}")
    print("  到達 = +30% に届いた割合。全件 = その窓のブレイクを全部買って +30% で売ったときの平均")

    for kind, _ in LABELS:
        for a in ALGOS:
            tuned[kind][a]["_cv"]["label"] = kind
    with open(os.path.join(OOF_DIR, f"{global_prefix}_params_all.json"), "w", encoding="utf-8") as fh:
        json.dump(tuned, fh, ensure_ascii=False, indent=1, default=float)
    log(f"記録: {OOF_DIR}/{global_prefix}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
