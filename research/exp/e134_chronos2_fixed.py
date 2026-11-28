#!/usr/bin/env python3
"""
実験134: 実験109 の腕 Z0（Chronos-2 に株価だけ）を、今の本番の目的変数（固定の +10%・+15%。label_specs の fixed_return_v1）で
測り直す。比べる相手は同じ目的変数の LightGBM（本番の記録のパラメータと列）。検証だけ（本番の設定・データセットの作りには書かない）。

運用者の問い（2026-11-28）「Z0: Chronos-2 に株価だけを入れる　この実験だけで良いですが、目的変数を今の本番と同じにした検証は
やりましたか？」→ やっていない（実験109 の正解は当時の本番の 1.2σ。本番は 2026-11-03 に固定の +10%・+15% へ切り替わった。
docs/MODEL_ADOPTION_RULES.md の「固定+10%・+15%の分類モデルへの切替」）→ 運用者の選択（AskUserQuestion）「+10% と +15% の両方」。

回す前に決めたこと（2026-11-28）

行と窓
  データセットの行（78週高値の新規の更新日 t）。窓は実験109 と同じ本番の前進分割（36/6/6か月・エンバーゴ20営業日・ずらし0。
  E41.folds_for）。Chronos-2 は学習しない（ゼロショット）ので、窓は比べる単位としてだけ使う。

目的変数（label_specs の fixed_return_v1）
  +10%: 翌営業日の寄りで買い、20営業日以内の終値の最高値が +10% 以上・20日目の直近5日平均終値が +5% 以上・20日目に MA5≥MA20
        （lgbm・xgb・cat のラベル。データセットの label。label_specs.labels で引き直して全行一致を確かめる）
  +15%: 同じ形で +15% / +7.5%（*_k15 のラベル。label_specs.labels(frame, "lgbm_k15") で引き直す）

腕（目的変数ごと）
  A   LightGBM。本番の記録のパラメータと列（+10% は research/model/meta.json、+15% は research/model/models/lgbm_k15/meta.json。
      記録の labelSpec が label_specs.for_model と違えば止める）。窓ごとに学習し、種3つ（42・7・123）の平均
  Z0  Chronos-2 に株価だけ（実験109 の Z0 と同じ入力: 日足の t までの直近256行の終値の対数・先20行・ゼロショット・
      版 29ec3766d36d6f73f0696f85560a422f50e8498c。E112.load_pipeline）。分位点は目的変数によらないので1回だけ計算する

Z0 の点数
  判定に使う点数: p_end = P(log C[t+20] − log C[t] ≥ log(1 + θ))（θ = 0.10 / 0.15）。実験109 の p_end の必要上昇率（1.2σ×√20）を
  固定の θ に替えたもの（E109.scores）。基準の価格は t の終値（ラベルの基準は翌営業日の寄りで、t の時点では分からない）。
  参考の点数（判定には使わない）: p_reach（t+1〜t+20 の各日の到達確率の最大）、p_half（20日目に +θ/2 を超える確率。終盤の
  条件の水準）、med（20日目の中央値の上昇）。

判定（実験109 と同じ。目的変数ごと）
  1. Z0 − A: 窓ごとの PR-AUC の差の平均・SE・上の窓の数と、テストの全行を日付で束ねたブートストラップ（2,000回）での PR-AUC の
     差の 95% 区間。上回る = 平均が正・区間の下端が正・上の窓が過半。下回る = 平均が負・区間の上端が負・上の窓が過半に届かない。
     ほかは「決められない」（E109.verdict）。
  2. Z0 に当てる力があるか: lift と ROC-AUC の 95% 区間の下端が 1・0.5 を超えるか。
  参考: ROC-AUC・日付内 AUC、モデルの公開より後（2025-11-01 以降）だけの集計、窓の中の上位2%（Precision・+10% の指値に
  届いた割合・1取引の平均）、予測の当たり具合（p_end の平均と、実際に20日目の終値が t の終値の +θ を超えた割合）、
  A の種42 と本番の記録の OOF の PR-AUC。
  実験109 とのつなぎ（参考）: 同じ行で 1.2σ のラベル（B.rise_label_for_k）と 1.2σ の p_end を作り、実験109 のテストの期間
  （2021-05-02〜2026-09-17）の PR-AUC を出す（実験109 の Z0 は 0.2380）。

確かめ（合わなければ止める）
  - +10% のラベルが label_specs.labels で引き直したものと全行一致し、+15% のラベルが引き直せる（材料の列がそろう）
  - 本番の記録の labelSpec が label_specs.for_model と一致し、記録の列がすべて表にある
  - 日足と学習データの行の並び・調整が同じ（E105.check_alignment）
  - A の学習が実験の共通部品（E41.oof_folds）と同じ点数（最初の窓・種42）
  - Chronos-2 の版（E112.load_pipeline）

回し方（Run Experiment）
    exp=e134_chronos2_fixed.py                 本番の回（Chronos-2 の分位点は research/_data/oof/e134_* に置き、次の回に引き継ぐ）
    exp=e134_chronos2_fixed.py args="--dry"    Chronos-2 の手前まで（ラベル・記録・日足の確かめと A）

公開ログには件数・割合・日付・精度だけを出す（銘柄・値は出さない）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import label_specs as LS  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e105_technical_ab as E105  # noqa: E402
import e109_chronos2 as E109  # noqa: E402
import e112_chronos2_cols as E112  # noqa: E402
from e104_market_drift import top_by_window  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
MODEL_DIR = E112.MODEL_DIR
TAG = "e134"
SEEDS = (42, 7, 123)
SHIFT = 0
CONTEXT = E109.CONTEXT
HORIZON = E109.HORIZON
TP = LT.TAKE_PROFIT
POST_RELEASE = E109.POST_RELEASE
#: 目的変数の鍵 → (本番のモデルの名前, 到達のしきい値)
TARGETS = {"p10": ("lgbm", 0.10), "p15": ("lgbm_k15", 0.15)}
TARGET_NAMES = {"p10": "+10%", "p15": "+15%"}
ARMS = ("A", "Z0")
LABELS = {"A": "A   LightGBM（本番の記録）", "Z0": "Z0  Chronos-2 株価だけ"}
#: 実験109 とのつなぎ（参考）: 実験109 のテストの期間と、その回の Z0 の PR-AUC
E109_PERIOD = (pd.Timestamp("2021-05-02"), pd.Timestamp("2026-09-17"))
E109_Z0_PR = 0.2380
N_BOOT = E109.N_BOOT


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def path(name: str) -> str:
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


# --------------------------------------------------------------------------- #
# 目的変数と本番の記録
# --------------------------------------------------------------------------- #

def target_labels(frame: pd.DataFrame) -> dict:
    """
    目的変数ごとのラベル（label_specs.labels）。+10% はデータセットの label と全行一致しなければ止める（本番の学習の確かめと同じ）。
    材料の列（future_rise・end_level・uptrend_end・entry_price）が無ければ label_specs が止める。
    """
    LS.check_training_frame(frame, "lgbm")
    out = {}
    for key, (name, thr) in TARGETS.items():
        spec = LS.for_model(name)
        if spec is None or abs(spec["threshold"] - thr) > 1e-12:
            raise SystemExit(f"{name} のラベルの定義が想定（+{thr * 100:g}%）と違う: {spec}")
        out[key] = LS.labels(frame, name)
    return out


def prod_model(name: str, frame_cols, root: str = MODEL_DIR) -> tuple[dict, list, dict]:
    """
    本番の記録（meta.json）のパラメータ・列と出どころ。labelSpec が今の定義と違う・記録の列が表に無いときは止める。
    LightGBM のパラメータは本番の組み方の固定の値で補う（E112.full_lgbm）。
    """
    with open(E112.meta_path(name, root), encoding="utf-8") as fh:
        meta = json.load(fh)
    want = LS.for_model(name)
    if meta.get("labelSpec") != want:
        raise SystemExit(f"{name}: 本番の記録の labelSpec が今の定義と違う（{meta.get('labelSpec')}）")
    cols = list(meta.get("features") or [])
    miss = [c for c in cols if c not in set(frame_cols)]
    if not cols or miss:
        raise SystemExit(f"{name}: 本番の記録の列が表に無い（{len(miss)}列。先頭 {miss[:5]}）")
    params = E112.full_lgbm(meta["params"])
    info = {"model": name, "label": meta.get("label"), "preset": meta.get("preset"), "n_features": len(cols),
            "trained": str(meta.get("trainedAt", ""))[:10], "hash": E112.params_hash(params),
            "oof": meta.get("oofMetrics") or {}}
    return params, cols, info


# --------------------------------------------------------------------------- #
# A（本番の記録の LightGBM）
# --------------------------------------------------------------------------- #

def arm_a(key: str, df: pd.DataFrame, cols: list, params: dict, folds) -> pd.DataFrame:
    """
    df（label が目的変数）の out-of-fold（種3つの平均と種42）。保存済みなら読む。
    最初の窓・種42 で、実験の共通部品（E41.oof_folds）と同じ点数になることを確かめる（合わなければ止める）。
    """
    fp = AB.fingerprint(df, cols) + "_" + E112.params_hash(params)
    po = path(f"A_{key}_{fp}_oof.parquet")
    if os.path.exists(po):
        log(f"[A {TARGET_NAMES[key]}] 保存済みの out-of-fold を読む（{fp}）")
        return pd.read_parquet(po)
    first = next(f for f in folds if E109.fold_rows(df, f) is not None)
    want = E41.oof_folds("lgbm", df, cols, params, SEEDS[0], [first])
    got = E109.fold_fit(df, cols, params, SEEDS[0], first)
    diff = float(np.max(np.abs(want["score"].to_numpy() - got[1])))
    print(f"  確かめ: A の学習が E41.oof_folds と同じ点数か（{TARGET_NAMES[key]}・窓{first.index}・種{SEEDS[0]}）"
          f" {len(got[1]):,}行の差の最大 {diff:.2e}")
    if diff > 1e-12:
        raise SystemExit("A の学習が実験の共通部品と合わない")
    rows = []
    for f in folds:
        t0 = time.time()
        sc, te = [], None
        for sd in SEEDS:
            r = E109.fold_fit(df, cols, params, sd, f)
            if r is None:
                break
            te = r[0]
            sc.append(r[1])
        if te is None:
            continue
        part = te[["Code", "Date", "label"]].copy()
        part["score"] = np.mean(sc, axis=0)
        part["score42"] = sc[0]
        part["fold"] = f.index
        rows.append(part)
        log(f"  A {TARGET_NAMES[key]} 窓{f.index}（{f.test_start}〜{f.test_end}）: テスト {len(part):,}行 {time.time()-t0:.0f}秒")
    o = pd.concat(rows, ignore_index=True)
    os.makedirs(OOF_DIR, exist_ok=True)
    o.to_parquet(po, index=False)
    return o


# --------------------------------------------------------------------------- #
# Z0 の点数
# --------------------------------------------------------------------------- #

def fixed_scores(q: np.ndarray, levels, last_log: np.ndarray, theta: float) -> pd.DataFrame:
    """
    固定の到達 theta の点数。p_end・p_reach・med・clip_hi・crossed は E109.scores（必要上昇率を全行 theta にしたもの）、
    p_half は20日目に +theta/2 を超える確率（終盤の条件の水準）。
    """
    n = q.shape[0]
    sc = E109.scores(q, levels, last_log, np.full(n, float(theta)))
    sc["p_half"] = E109.prob_above(q[:, :, -1], list(levels), last_log + np.log1p(theta / 2.0))
    return sc


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #

def judge(o: pd.DataFrame) -> dict:
    """判定 1・2（回す前に決めた。E109 と同じ言い方）と全行の区間。o は fold・Date・label・s_A・s_Z0 を持つ。"""
    w = E109.by_window(o, ARMS)
    bt = E109.boot(o, ARMS, n=N_BOOT)
    d = (w["pr_Z0"] - w["pr_A"]).to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    lo, hi = E109.ci(bt["pr_Z0"] - bt["pr_A"])
    wins = int((d > 0).sum())
    out = {"windows": w, "pair": {"mean": float(d.mean()), "se": se, "wins": wins, "n": int(len(d)), "ci": [lo, hi],
                                  "verdict": E109.verdict(float(d.mean()), lo, hi, wins, len(d))}, "arms": {}}
    y = o["label"].to_numpy(dtype=int)
    for a in ARMS:
        pr, roc = E109.pr_roc(y, o[f"s_{a}"].to_numpy(dtype=float))
        plo, phi = E109.ci(bt[f"pr_{a}"])
        llo, lhi = E109.ci(bt[f"pr_{a}"] / bt["pos"])
        rlo, rhi = E109.ci(bt[f"roc_{a}"])
        out["arms"][a] = {"pr": pr, "pr_ci": [plo, phi], "lift": pr / float(y.mean()), "lift_ci": [llo, lhi], "roc": roc,
                          "roc_ci": [rlo, rhi], "day_auc": float(lab.auc_in_day(o.assign(score=o[f"s_{a}"]))),
                          "skill": bool(llo > 1.0 and rlo > 0.5) if a != "A" else None}
    return out


def report_target(key: str, o: pd.DataFrame, sc: pd.DataFrame, real_up: np.ndarray, info: dict) -> dict:
    """目的変数1つの判定と参考の集計を出す（件数・割合・精度だけ）。sc は o と同じ並びの Z0 の点数。"""
    name = TARGET_NAMES[key]
    theta = TARGETS[key][1]
    res = judge(o)
    w = res["windows"]
    print(f"\n{'=' * 78}\n目的変数 {name}（{info['label']}）: {len(o):,}行・{len(w)}窓・正例率 {o['label'].mean()*100:.1f}%"
          f"（{o['Date'].min().date()}〜{o['Date'].max().date()}）")
    print(f"  A のパラメータ: 本番の {info['model']}（{info['preset']}・{info['n_features']}列・学習 {info['trained']}・"
          f"指紋 {info['hash']}）")
    print(f"\n■ 1. 窓ごとの PR-AUC（{name}）")
    print(f"  {'窓':<4}{'行':>7}{'正例率':>8}{'A':>9}{'Z0':>9}{'Z0−A':>9}")
    for _, r in w.iterrows():
        print(f"  {int(r['fold']):<4}{int(r['n']):>7,}{r['pos']*100:>7.1f}%{r['pr_A']:>9.4f}{r['pr_Z0']:>9.4f}"
              f"{r['pr_Z0'] - r['pr_A']:>+9.4f}")
    print(f"\n■ 2. 全行（日付で束ねたブートストラップ {N_BOOT:,}回の 95% 区間）")
    print(f"  {'腕':<30}{'PR-AUC':>9}{'区間':>18}{'lift':>7}{'lift の区間':>16}{'ROC-AUC':>9}{'区間':>18}{'日付内AUC':>10}")
    for a in ARMS:
        s = res["arms"][a]
        print(f"  {LABELS[a]:<30}{s['pr']:>9.4f}  [{s['pr_ci'][0]:.4f}, {s['pr_ci'][1]:.4f}]{s['lift']:>7.2f}"
              f"  [{s['lift_ci'][0]:.2f}, {s['lift_ci'][1]:.2f}]{s['roc']:>9.4f}  [{s['roc_ci'][0]:.4f}, {s['roc_ci'][1]:.4f}]"
              f"{s['day_auc']:>10.4f}")
    p = res["pair"]
    print("\n■ 3. 判定（回す前に決めた）")
    print(f"  Z0 − A: 窓ごとの PR-AUC の差 平均 {p['mean']:+.4f}（SE {p['se']:.4f}・上の窓 {p['wins']}/{p['n']}）・"
          f"全行の差の95%区間 [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] → Z0 は A を「{p['verdict']}」")
    z = res["arms"]["Z0"]
    print(f"  Z0 に当てる力があるか: lift の区間の下端 {z['lift_ci'][0]:.2f}（1 を超えるか）・ROC-AUC の区間の下端 "
          f"{z['roc_ci'][0]:.4f}（0.5 を超えるか）→ {'ある' if z['skill'] else '言えない'}")

    # 参考: 上位2%（+10% の指値。今の売り方）
    print("\n■ 4. 参考: 窓の中の上位2%（Precision・+10% の指値に届いた割合・1取引の平均）")
    res["top2"] = {}
    for a in ARMS:
        t = top_by_window(o.assign(score=o[f"s_{a}"]))
        if not len(t):
            continue
        r = {"k": int(t["k"].sum()), "prec": float(t["prec"].mean()), "hit10": float(t["hit10"].mean()),
             "tp10c": float(t["tp10c"].mean())}
        res["top2"][a] = r
        print(f"  {LABELS[a]:<30} {r['k']:>4}件  Precision {r['prec']*100:5.1f}%  +10% {r['hit10']*100:5.1f}%  "
              f"1取引 {r['tp10c']:+.2f}%")

    # 参考: 公開より後
    post = o[o["Date"] >= POST_RELEASE]
    res["post"] = {"rows": int(len(post))}
    if len(post) >= 200 and post["label"].nunique() == 2:
        bp = E109.boot(post, ARMS, n=N_BOOT)
        print(f"\n■ 5. 参考: モデルの公開より後（{POST_RELEASE.date()} 以降。{len(post):,}行・正例率 {post['label'].mean()*100:.1f}%）")
        for a in ARMS:
            pr, roc = E109.pr_roc(post["label"].to_numpy(dtype=int), post[f"s_{a}"].to_numpy(dtype=float))
            lo, hi = E109.ci(bp[f"pr_{a}"])
            res["post"][a] = {"pr": pr, "pr_ci": [lo, hi], "roc": roc}
            print(f"  {LABELS[a]:<30}PR-AUC {pr:.4f} [{lo:.4f}, {hi:.4f}]  ROC-AUC {roc:.4f}")
        lo, hi = E109.ci(bp["pr_Z0"] - bp["pr_A"])
        res["post"]["diff_ci"] = [lo, hi]
        print(f"  Z0 − A: PR-AUC の差の95%区間 [{lo:+.4f}, {hi:+.4f}]")

    # 参考: 当たり具合・参考の点数・本番の記録
    ok = np.isfinite(real_up)
    res["calib"] = {"p_end_mean": float(np.nanmean(sc["p_end"])),
                    "hit_end_rate": float((real_up[ok] >= np.log1p(theta)).mean()) if ok.any() else float("nan"),
                    "clip_hi": float(sc["clip_hi"].mean()), "crossed": float(sc["crossed"].mean())}
    c = res["calib"]
    print(f"\n■ 6. 参考: 予測の当たり具合: p_end の平均 {c['p_end_mean']*100:.1f}% / 実際に20日目の終値が t の終値の {name} を"
          f"超えた割合 {c['hit_end_rate']*100:.1f}%・+{theta*100:g}% が最上位の分位点より上の行 {c['clip_hi']*100:.1f}%・"
          f"分位点が交差した行 {c['crossed']*100:.1f}%")
    y = o["label"].to_numpy(dtype=int)
    res["ref_scores"] = {}
    line = []
    for k in ("p_reach", "p_half", "med"):
        pr, roc = E109.pr_roc(y, sc[k].to_numpy(dtype=float))
        res["ref_scores"][k] = {"pr": pr, "roc": roc}
        line.append(f"{k} {pr:.4f} / {roc:.4f}")
    print("■ 7. 参考の点数（判定には使わない。全行の PR-AUC / ROC-AUC）: " + "  ".join(line))
    pr42, roc42 = E109.pr_roc(y, o["score42"].to_numpy(dtype=float))
    rec = info.get("oof") or {}
    res["prod_oof"] = {"a_seed42": {"pr": pr42, "roc": roc42, "n": int(len(o))}, "record": rec}
    print(f"■ 8. 参考: A の種42 の PR-AUC {pr42:.4f}・ROC-AUC {roc42:.4f}（{len(o):,}行）/ 本番の記録の OOF "
          f"PR-AUC {rec.get('prAuc', float('nan'))}・ROC-AUC {rec.get('rocAuc', float('nan'))}（{rec.get('n', '?')}行。"
          f"学習 {info['trained']}。種と行が違うので目安）")
    res["windows"] = w.to_dict(orient="records")
    return res


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験134: Chronos-2 の Z0 を今の本番の目的変数（+10%・+15%）で測り直す")
    ap.add_argument("--targets", default="p10,p15", help="p10（+10%）・p15（+15%）")
    ap.add_argument("--dry", action="store_true", help="Chronos-2 の手前まで（ラベル・記録・日足の確かめと A）")
    args = ap.parse_args(argv)
    keys = [k for k in args.targets.split(",") if k]
    bad = [k for k in keys if k not in TARGETS]
    if bad or not keys:
        raise SystemExit(f"目的変数は {tuple(TARGETS)} から: {bad}")
    E112.TAG = TAG                      # Chronos-2 の分位点の置き場（E112.z0_quantiles）をこの実験の名前にする
    os.makedirs(OOF_DIR, exist_ok=True)
    print("=" * 78)
    print("実験134 Chronos-2 株価だけ（実験109 の Z0）を、今の本番の目的変数で測り直す")
    print(f"  目的変数: {', '.join(f'{TARGET_NAMES[k]}（{LS.name(TARGETS[k][0])}）' for k in keys)}")
    print(f"  Chronos-2: {E112.MODEL_ID}（commit {E112.REVISION}）・文脈 {CONTEXT}行・先 {HORIZON}行・ゼロショット")
    print(f"  窓: 本番の前進分割（ずらし{SHIFT}か月）/ A の種 {list(SEEDS)}")
    print("=" * 78)

    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"]).astype("datetime64[ns]")
    frame["Code"] = frame["Code"].astype(str)
    frame = frame[frame["label"].notna()].reset_index(drop=True)
    labels = target_labels(frame)
    log(f"[data] lab.frame {len(frame):,}行（{frame['Date'].min().date()}〜{frame['Date'].max().date()}）/ {time.time()-t0:.0f}秒")
    print(f"  確かめ: +10% のラベルが label_specs で引き直したものと全行一致（{len(frame):,}行）")
    for k in keys:
        lb = labels[k]
        print(f"  {TARGET_NAMES[k]} のラベル: 決まる行 {int(lb.notna().sum()):,}・正例率 {lb.mean()*100:.1f}%")
    models = {k: prod_model(TARGETS[k][0], frame.columns, root=MODEL_DIR) for k in keys}
    for k in keys:
        _p, cols, info = models[k]
        print(f"  A {TARGET_NAMES[k]}: 本番の {info['model']}（{info['label']}・{info['preset']}・{len(cols)}列・学習 {info['trained']}・"
              f"指紋 {info['hash']}）")
    folds = E41.folds_for(frame["Date"], SHIFT)

    # --- A ---
    oA = {}
    for k in keys:
        params, cols, _info = models[k]
        df = frame.assign(label=labels[k].to_numpy())
        df = df[df["label"].notna()].reset_index(drop=True)
        oA[k] = arm_a(k, df, cols, params, folds)
        oA[k]["Date"] = pd.to_datetime(oA[k]["Date"]).astype("datetime64[ns]")
        oA[k]["Code"] = oA[k]["Code"].astype(str)
    keyset = pd.concat([o[["Code", "Date"]] for o in oA.values()]).drop_duplicates()
    rows = keyset.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)
    rows = rows.merge(frame[["Code", "Date", "close_position", "vol_20d", "ret_o1_20", "future_rise", "end_level",
                             "uptrend_end", "label"]], on=["Code", "Date"], how="left", validate="one_to_one")
    log(f"[rows] Chronos-2 で採点する行 {len(rows):,}（目的変数の窓のテストの行の和）")

    # --- 日足 ---
    t0 = time.time()
    panel = E105.build_panel(lab.DATA_DIR, rows["Code"].unique())
    panel = panel.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)
    if panel.duplicated(["Code", "Date"]).any():
        raise SystemExit("日足の Code・Date が重なっている")
    log(f"[panel] 日足 {len(panel):,}行（{panel['Code'].nunique():,}銘柄）/ {time.time()-t0:.0f}秒")
    checks = {"alignment": E105.check_alignment(rows, panel)}
    start, end = E109.context_bounds(panel, rows, CONTEXT)
    span = end - start + 1
    print(f"  文脈の長さ（行）: 中央値 {int(np.median(span))}・{CONTEXT} 行に満たない行 {(span < CONTEXT).mean()*100:.1f}%")
    if args.dry:
        print("\n[dry] Chronos-2 の手前で止める")
        return 0

    # --- 出口（+10% の指値）---
    bars_px = LT.load_bars(lab.DATA_DIR, start=rows["Date"].min() - pd.Timedelta(days=10))
    bars_px["Date"] = bars_px["Date"].astype("datetime64[ns]")
    ex, chk = E83.attach_exits(rows[["Code", "Date", "ret_o1_20"]], bars_px, targets=(TP,))
    del bars_px
    if len(ex) != len(rows) or not (ex["Code"].to_numpy() == rows["Code"].to_numpy()).all():
        raise SystemExit("出口の突き合わせで行の並びが変わった")
    tk = E83.tp_key(TP)
    exit_ok = chk["hold5_match"] == chk["hold5_both"] and chk["order_violations"] == 0
    rows["hit10"] = ex[f"hit{tk}"].to_numpy(dtype=float) if exit_ok else np.nan
    rows["tp10c"] = pd.to_numeric(ex[f"tp{tk}c"], errors="coerce").to_numpy(dtype=float) if exit_ok else np.nan
    checks["exits"] = {**chk, "ok": bool(exit_ok)}
    print(f"  確かめ: 出口の5日平均と ret_o1_20 の一致 {chk['hold5_match']:,} / {chk['hold5_both']:,}行 → "
          f"{'一致' if exit_ok else '合わない（上位2% の +10% は出さない）'}")

    # --- Chronos-2（目的変数によらず1回）---
    t0 = time.time()
    q, levels, cinfo = E112.z0_quantiles(rows, panel)
    close = panel["close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        last_log = np.log(close[end])
        fwd = end + HORIZON
        same = (fwd < len(panel)) & (panel["Code"].to_numpy()[np.minimum(fwd, len(panel) - 1)] == rows["Code"].to_numpy())
        real_up = np.where(same, np.log(close[np.minimum(fwd, len(panel) - 1)]), np.nan) - last_log
    del panel
    checks["chronos"] = {"model": E112.MODEL_ID, "revision": E112.REVISION, "levels": levels, "rows": int(len(rows)),
                         "computed_rows": int(cinfo["computed"]), "seconds": round(cinfo["seconds"])}
    log(f"[chronos] {len(rows):,}行（今回計算 {cinfo['computed']:,}行・{cinfo['seconds']:.0f}秒）/ {time.time()-t0:.0f}秒")

    # --- 目的変数ごとの判定 ---
    summary = {"targets": {}, "checks": checks}
    keyidx = pd.MultiIndex.from_frame(rows[["Code", "Date"]])
    for k in keys:
        sc_all = fixed_scores(q, levels, last_log, TARGETS[k][1])
        o = oA[k].copy()
        pos = keyidx.get_indexer(pd.MultiIndex.from_frame(o[["Code", "Date"]]))
        if (pos < 0).any():
            raise SystemExit("A のテストの行が Chronos-2 の行に無い")
        o["s_A"] = o["score"]
        o["s_Z0"] = sc_all["p_end"].to_numpy()[pos]
        o["hit10"] = rows["hit10"].to_numpy()[pos]
        o["tp10c"] = rows["tp10c"].to_numpy()[pos]
        if o["s_Z0"].isna().any():
            raise SystemExit(f"Z0 の点数が欠測の行がある: {int(o['s_Z0'].isna().sum()):,}行")
        summary["targets"][k] = report_target(k, o, sc_all.iloc[pos].reset_index(drop=True), real_up[pos], models[k][2])

    # --- 実験109 とのつなぎ（参考）---
    cfg12 = replace(B.DEFAULT_RISE, vol_norm_k=1.2)
    lab12 = B.rise_label_for_k(rows, 1.2)
    need12, _ = B.rise_thresholds(rows["vol_20d"], cfg12)
    s12 = E109.scores(q, levels, last_log, need12.to_numpy(dtype=float))["p_end"].to_numpy()
    m = (rows["Date"] >= E109_PERIOD[0]) & (rows["Date"] <= E109_PERIOD[1]) & lab12.notna() & np.isfinite(s12)
    pr12, roc12 = E109.pr_roc(lab12[m].to_numpy(dtype=int), s12[m.to_numpy()])
    summary["link_e109"] = {"rows": int(m.sum()), "pos": float(lab12[m].mean()), "pr": pr12, "roc": roc12}
    print(f"\n■ 9. 参考: 実験109 とのつなぎ（1.2σ のラベル・1.2σ の p_end。{E109_PERIOD[0].date()}〜{E109_PERIOD[1].date()}・"
          f"{int(m.sum()):,}行・正例率 {lab12[m].mean()*100:.1f}%）: PR-AUC {pr12:.4f}・ROC-AUC {roc12:.4f}"
          f"（実験109 の Z0 は {E109_Z0_PR:.4f}。データセットを作り直したぶん行と値がわずかに違う）")

    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    log(f"記録: {path('summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
