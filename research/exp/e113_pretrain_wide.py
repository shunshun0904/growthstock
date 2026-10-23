#!/usr/bin/env python3
"""
実験113: 78週高値の更新日すべて（クールダウンなし）で事前学習し、今の母集団（新規の更新日）で仕上げると、木3モデル（lgbm・xgb・cat）
の本番と同じ作りの out-of-fold・32窓の分離力（PR-AUC）が上がるか（§7。腕は「学習に使う行」と「仕上げ方」の違い）。検証だけ。

運用者の問い（2026-10-23）「今の特徴量で転移学習（転移予測）的なことはできないのでしょうか？ LLM的に言えば、蒸留ですかね。」→
答え（docs/MODEL_ADOPTION_RULES.md §76）: 今の360列のまま転移学習に当たるのは「広い母集団で先に学び、今の母集団で仕上げる」
（LLM の事前学習 → 微調整）。蒸留（先生の確率を生徒が真似る）は主に軽くする・揺れを減らす技術で、木のモデルの今の運用では
上積みは期待しにくい → 運用者の選択（AskUserQuestion）「事前学習→仕上げ」・広い母集団「78週の更新日すべて」。運用者の方針
「いきなり変えるよりも、まずは検証してほしいです。本番導入するかはその結果次第です」→ 本番の設定と build_dataset.py には書かない。

行（回す前に決めた。2026-10-23）
  目標 T  今のデータセットの行（78週高値の新規の更新日 = 直前20営業日に更新なし。lab.frame。ラベルの決まる行）
  元 S    同じ作りのデータセットを、直前20営業日の条件を外して作り（78週高値の更新日すべて）、T の行を除いたもの。台本の中で
          build_dataset.build を回す（mark_new_highs の is_fresh_break を is_new_high に差し替える。BREAKOUT_COOLDOWN = 0 は今の
          コードでは rolling(0, min_periods=1) が通らないため）。手元で数えた見込み（2026-10-23。決算の条件の前）: 更新日すべて
          164,882行・新規の更新日 23,860行（正例率 18.1%・17.3%）
  T と S の区別の列は足さない（base_length（前回の更新日からの営業日数）が 20 を超えるかで、ほぼ分かれる）

腕（評価はどれも T のテストの行。パラメータは本番のモデルの記録。実験112 と同じ e112.prod_params）
  A  T だけで学習（本番の360列）
  W  T と S を混ぜて学習（360列）。S の重みは窓ごとに「S の重みの合計 = T の行数」になる値（T は 1）
  P  対照: W と同じ行・列・重みで、S のラベルを同じ日付の S の行どうしで入れ替えたもの（種 PERM_SEED）。日付ごとの正例率
     （相場の良し悪し）は残り、同じ日の銘柄を見分ける情報だけが壊れる
  F  参考（仕上げ）: S だけで学んだモデルの点数（生の値）から始めて、T で続けて学ぶ（LightGBM の init_score・XGBoost の
     base_margin・CatBoost の baseline）。テストの点数は S の生の値 + 仕上げのモデルの生の値。学習の行の S の点数は、S が同じ銘柄の
     近い日の行で学んだぶん楽観側に出る（交差の当てはめはしない。参考の腕のため）
  S  参考: S だけで学んだモデルを、そのまま T のテストの行に当てたもの（F の途中の値。計算は増えない）
  どの腕も、学習に使う行は T・S とも窓の学習の終わり（エンバーゴ20営業日の手前）までの日付だけ

判定（回す前に決めた。2026-10-23）
  docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）: 木3モデル（lgbm・xgb・cat）のうち2つ以上で、窓ごとの PR-AUC の差 W−A の
  平均が正・対照 P−A の平均を上回る・W が上の窓が過半（n//2 + 1 以上）。窓は本番の out-of-fold と同じ作り（36/6/6か月・
  エンバーゴ20営業日）の境界を 0/2/4か月ずらした3通り（約32窓）、点数は種3つ（42・7・123）の平均。1.2σ のラベルで判定し、
  本番に入れるかは結果を見て運用者が決める。F・S は参考（F−A・S−A の平均・SE・上の窓を並べる）

参考（判定には使わない）: 本番と同じ作りの out-of-fold（ずらし0・種42）、窓の中の上位2% の Precision・+10% の指値に届いた割合・
1取引の平均、S の行の付き方（年ごと・正例率・base_length の分布）

確かめ（合わなければ止める）
  - T の行がすべて広いデータセットにあり、360列の値とラベルが全行一致すること（相対 1e-6・絶対 1e-9。E105.compare_columns）
  - S と T の行が重ならないこと。学習に使う S の行が窓の学習の終わりより後に無いこと
  - 部品: tests/test_e113_pretrain_wide.py（本番の組み方と同じ点数・init の足し方・重み・入れ替え）

回し方（Run Experiment。同じ実験は同時に1本だけ）
    exp=e113_pretrain_wide.py args="--prepare-only"   ← 1回目: 広いデータセットを作り、確かめて表を保存（out-of-fold は回さない）
    exp=e113_pretrain_wide.py args="--algos lgbm"
    exp=e113_pretrain_wide.py args="--algos xgb"
    exp=e113_pretrain_wide.py args="--algos cat"
  表（T と S）は1回目に作って research/_data/oof/e113_* に置き、以降の回は actions/cache で引き継いだそれを使う。--budget-min
  （既定 290分）を過ぎたら新しい計算を始めずに報告して終わる（保存済みの out-of-fold は次の回が続きから使う）。判定の腕
  （A・W・P）を先に、F を後に計算する。

公開ログには件数・割合・日付・精度・列名だけを出す（銘柄・値は出さない）。
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
import live_track as LT  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e105_technical_ab as E105  # noqa: E402
import e112_chronos2_cols as E112  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e44_shortsale import permuted  # noqa: E402
from e68_edinet_ab import judge7, pair, params_hash, short  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
TAG = "e113"
RUN = "e113_k12"
ALGOS = ("lgbm", "xgb", "cat")
SEEDS = (42, 7, 123)
PROD_SEED = 42
SHIFTS = (0, 2, 4)
K_DEFAULT = 1.2
K = K_DEFAULT
#: 計算する腕（S は F の途中の値なので計算しない）。判定の腕を先に
COMPUTE = ("A", "W", "P", "F")
ARMS = ("A", "W", "P", "F", "S")
LABELS = {"A": "A  今の母集団だけ", "W": "W  広い母集団と混ぜて学習", "P": "P  対照（広い母集団のラベルを日付内で入れ替え）",
          "F": "F  広い母集団で学んだ点数から仕上げ", "S": "S  広い母集団だけで学んだモデル"}
PAIRS = (("A", "W"), ("A", "P"), ("P", "W"), ("A", "F"), ("A", "S"))
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261024
TP = LT.TAKE_PROFIT
#: 広い母集団の表に残す列（360列のほかに、ラベルの引き直しに要る列）
KEEP_EXTRA = ("future_rise", "end_level", "uptrend_end", "vol_20d")
MIN_TRAIN, MIN_TEST = 1000, 200
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cpath(name: str) -> str:
    """表（ラベルによらない）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def path(name: str) -> str:
    """out-of-fold・要約（ラベルごと）。"""
    return os.path.join(OOF_DIR, f"{RUN}_{name}")


def time_left() -> bool:
    global TIMED_OUT
    if DEADLINE is None or time.time() < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min を過ぎたので、新しい計算は始めない（保存済みの分だけで報告。続きは次の回）")
    return False


def keys_of(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([df["Code"].astype(str), pd.to_datetime(df["Date"]).astype("datetime64[ns]")])


# --------------------------------------------------------------------------- #
# 学習器（本番の組み方。重み・始めの点数を渡せる）
# --------------------------------------------------------------------------- #

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def fit_predict(algo: str, params: dict, seed: int, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                weight: np.ndarray | None = None, init_tr: np.ndarray | None = None,
                init_te: np.ndarray | None = None) -> np.ndarray:
    """
    学習して Xte の生の値（対数オッズ）を返す。weight は行の重み、init_tr / init_te は始めの点数（生の値）。始めの点数で学んだ
    モデルの値には始めの点数を足して返す（どの学習器でも、予測はモデル自身の値 + init_te）。
    LightGBM は本番と同じ組み方（e112.full_lgbm。E41.oof_folds と同じ）、XGBoost・CatBoost は tuning_multi.build。
    """
    spw = tuning.scale_pos_weight(ytr)
    if algo == "lgbm":
        import lightgbm as lgb

        m = lgb.LGBMClassifier(**{**E112.full_lgbm(params), "random_state": seed}, scale_pos_weight=spw)
        m.fit(Xtr, ytr, sample_weight=weight, init_score=init_tr)
        raw = m.predict(Xte, raw_score=True)
    else:
        old = TM.SEED
        try:
            TM.SEED = seed
            m = TM.build(algo, params, ytr)
        finally:
            TM.SEED = old
        if algo == "xgb":
            m.fit(Xtr, ytr, sample_weight=weight, base_margin=init_tr)
            # 始めの点数で学んだモデルは、予測でも始めの点数を渡すと全体の値になる（base_score を使わない）
            raw = m.predict(Xte, output_margin=True, base_margin=init_te if init_tr is not None else None)
            return np.asarray(raw, dtype=float)
        if algo == "cat":
            m.fit(Xtr, ytr, sample_weight=weight, baseline=init_tr)
            raw = m.predict(Xte, prediction_type="RawFormulaVal")
        else:
            raise ValueError(f"知らない学習器: {algo}")
    raw = np.asarray(raw, dtype=float)
    if init_tr is not None:
        raw = raw + np.asarray(init_te, dtype=float)
    return raw


# --------------------------------------------------------------------------- #
# 表（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #

def build_wide(data_dir: str, out_path: str) -> None:
    """build_dataset.build を、直前20営業日の条件を外して（78週高値の更新日すべてを候補にして）回す。終わったら元に戻す。"""
    orig_mark, orig_cd = B.mark_new_highs, B.BREAKOUT_COOLDOWN

    def all_highs(df, cooldown=None, on_high=None):
        out = orig_mark(df, cooldown=1, on_high=on_high)       # cooldown 0 は rolling(0, min_periods=1) が通らない
        out["is_fresh_break"] = out["is_new_high"]
        return out

    B.mark_new_highs, B.BREAKOUT_COOLDOWN = all_highs, 0
    try:
        B.build(data_dir, out_path)
    finally:
        B.mark_new_highs, B.BREAKOUT_COOLDOWN = orig_mark, orig_cd


def split_wide(wide: pd.DataFrame, target: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, dict]:
    """
    広いデータセットから T の行を除いた S と、確かめの記録。T の行がすべて広いデータセットにあり、cols とラベルが全行一致する
    ことを確かめる（合わなければ止める）。
    """
    kw, kt = keys_of(wide), keys_of(target)
    if kw.duplicated().any() or kt.duplicated().any():
        raise SystemExit("Code・Date が重なっている")
    pos = kw.get_indexer(kt)
    found = int((pos >= 0).sum())
    chk = {"target_rows": int(len(target)), "found_in_wide": found, "wide_rows": int(len(wide))}
    print(f"  確かめ: 今の母集団の行のうち広いデータセットにある {found:,} / {len(target):,}行（広いデータセット {len(wide):,}行）")
    if found != len(target):
        raise SystemExit("今の母集団の行が広いデータセットにそろっていない")
    sub = wide.iloc[pos].reset_index(drop=True)
    bad = E105.compare_columns(target[cols].reset_index(drop=True), sub[cols], cols=cols)
    lab_bad = int((pd.to_numeric(target["label"], errors="coerce").to_numpy()
                   != pd.to_numeric(sub["label"], errors="coerce").to_numpy()).sum())
    chk["bad_cells"], chk["bad_columns"], chk["label_mismatch"] = int(sum(bad.values())), int(sum(v > 0 for v in bad.values())), lab_bad
    print(f"  確かめ: その行の {len(cols)}列の一致 → 食い違い {chk['bad_cells']:,}セル（{chk['bad_columns']}列）・ラベルの食い違い {lab_bad:,}行")
    if chk["bad_cells"] or lab_bad:
        raise SystemExit("今の母集団の行の値が広いデータセットと合わない: "
                         + ", ".join(f"{c} {v}" for c, v in bad.items() if v)[:400])
    src = wide[~kw.isin(kt)].reset_index(drop=True)
    chk["source_rows"] = int(len(src))
    return src, chk


def prepare(cols: list, rebuild: bool = False):
    """T（lab.frame に出口を付けた表）と S（広いデータセットから T を除いた表）と記録。1回目に作って保存し、以降は読む。"""
    pt, ps, pj = cpath("target.parquet"), cpath("source.parquet"), cpath("stamp.json")
    if not rebuild and all(os.path.exists(p) for p in (pt, ps, pj)):
        with open(pj, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・T {stamp['target_rows']:,}行・S {stamp['source_rows']:,}行）")
        return pd.read_parquet(pt), pd.read_parquet(ps), stamp
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    target = lab.frame(rebuild=True)
    target["Date"] = pd.to_datetime(target["Date"]).astype("datetime64[ns]")
    target["Code"] = target["Code"].astype(str)
    target = target[target["label"].notna()].reset_index(drop=True)
    need = cols + ["ret_o1_20", "ret_o1_40"] + list(KEEP_EXTRA)
    miss = [c for c in need if c not in target.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    log(f"[data] T（lab.frame）{len(target):,}行（{target['Date'].min().date()}〜{target['Date'].max().date()}）/ {time.time()-t0:.0f}秒")

    t0 = time.time()
    full_path = os.path.join(lab.DATA_DIR, "e113_wide_full.parquet")       # 大きいので actions/cache には入れない
    print("\n■ 広いデータセットを作る（build_dataset.build・78週高値の更新日すべて）", flush=True)
    build_wide(lab.DATA_DIR, full_path)
    wide = pd.read_parquet(full_path, columns=list(dict.fromkeys(["Code", "Date", "label"] + cols + list(KEEP_EXTRA))))
    for ext in ("", "_meta.json", "_general_market_start.json"):
        p = full_path if not ext else os.path.splitext(full_path)[0] + ext
        if os.path.exists(p):
            os.remove(p)
    wide["Date"] = pd.to_datetime(wide["Date"]).astype("datetime64[ns]")
    wide["Code"] = wide["Code"].astype(str)
    wide = wide[wide["label"].notna()].reset_index(drop=True)
    log(f"[data] 広いデータセット {len(wide):,}行 / {time.time()-t0:.0f}秒")
    print("\n■ 確かめ（今の母集団の行が、広いデータセットで同じ値か）")
    src, checks = split_wide(wide, target, cols)
    del wide

    t0 = time.time()
    bars_px = LT.load_bars(lab.DATA_DIR, start=target["Date"].min() - pd.Timedelta(days=10))
    bars_px["Date"] = bars_px["Date"].astype("datetime64[ns]")
    ex, chk = E83.attach_exits(target[["Code", "Date", "ret_o1_20"]], bars_px, targets=(TP,))
    del bars_px
    if len(ex) != len(target) or not (ex["Code"].to_numpy() == target["Code"].to_numpy()).all():
        raise SystemExit("出口の突き合わせで行の並びが変わった")
    k = E83.tp_key(TP)
    exit_ok = chk["hold5_match"] == chk["hold5_both"] and chk["order_violations"] == 0
    target["hit10"] = ex[f"hit{k}"].to_numpy(dtype=float) if exit_ok else np.nan
    target["tp10c"] = pd.to_numeric(ex[f"tp{k}c"], errors="coerce").to_numpy(dtype=float) if exit_ok else np.nan
    checks["exits"] = {**chk, "ok": bool(exit_ok)}
    print(f"  確かめ: 出口の5日平均と ret_o1_20 の一致 {chk['hold5_match']:,} / {chk['hold5_both']:,}行 → "
          f"{'一致' if exit_ok else '合わない（上位2% の +10% は出さない）'}（{time.time()-t0:.0f}秒）")

    keep_t = ["Code", "Date", "label", "ret_o1_20", "ret_o1_40", "hit10", "tp10c"] + cols + list(KEEP_EXTRA)
    target = target[list(dict.fromkeys(keep_t))]
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "target_rows": int(len(target)),
             "source_rows": int(len(src)), "date_min": str(target["Date"].min().date()),
             "date_max": str(target["Date"].max().date()), "checks": checks}
    target.to_parquet(pt, index=False)
    src.to_parquet(ps, index=False)
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return target, src, stamp


def source_report(target: pd.DataFrame, src: pd.DataFrame) -> dict:
    """S の付き方（年ごとの行・正例率、base_length の分布、T との比）。件数と割合だけ。"""
    out = {}
    yt, ys = target["Date"].dt.year, src["Date"].dt.year
    print(f"\n■ 0. 行の付き方（{K:.1f}σ のラベル）: T {len(target):,}行（正例率 {target['label'].mean()*100:.1f}%）・"
          f"S {len(src):,}行（正例率 {src['label'].mean()*100:.1f}%・T の {len(src) / max(len(target), 1):.1f}倍）")
    rows = []
    for y in sorted(set(yt) | set(ys)):
        a, b = target[yt == y], src[ys == y]
        rows.append({"year": int(y), "T": int(len(a)), "S": int(len(b)), "pos_T": float(a["label"].mean()) if len(a) else None,
                     "pos_S": float(b["label"].mean()) if len(b) else None})
    out["by_year"] = rows
    print("  年ごと（T / S / 正例率 T・S）: " + " ".join(
        f"{r['year']}:{r['T']:,}/{r['S']:,}/{(r['pos_T'] or 0)*100:.0f}・{(r['pos_S'] or 0)*100:.0f}%" for r in rows))
    if "base_length" in src.columns:
        bl_s = pd.to_numeric(src["base_length"], errors="coerce")
        bl_t = pd.to_numeric(target["base_length"], errors="coerce")
        q = [0.1, 0.5, 0.9]
        out["base_length"] = {"S": [float(x) for x in bl_s.quantile(q)], "T": [float(x) for x in bl_t.quantile(q)],
                              "S_le20": float((bl_s <= 20).mean()), "T_le20": float((bl_t <= 20).mean())}
        print(f"  base_length（前回の更新日からの営業日数）の 10/50/90%: S {' / '.join(f'{x:.0f}' for x in bl_s.quantile(q))}・"
              f"T {' / '.join(f'{x:.0f}' for x in bl_t.quantile(q))}（20 以下の行 S {(bl_s <= 20).mean()*100:.1f}%・"
              f"T {(bl_t <= 20).mean()*100:.1f}%）")
    return out


# --------------------------------------------------------------------------- #
# out-of-fold（T のテストの行だけを採点する）
# --------------------------------------------------------------------------- #

def fold_oof(algo: str, arm: str, target: pd.DataFrame, src: pd.DataFrame, cols: list, params: dict, seed: int,
             folds, y_src: np.ndarray) -> pd.DataFrame:
    """1つの腕・種の out-of-fold（T のテストの行。窓の切り方は E41.oof_folds と同じ）。F は S の点数（score_S）も返す。"""
    dt = pd.to_datetime(target["Date"]).to_numpy()
    ds = pd.to_datetime(src["Date"]).to_numpy()
    Xt = target[cols].to_numpy(dtype=float)
    yt = target["label"].to_numpy(dtype=int)
    Xs = src[cols].to_numpy(dtype=float) if arm != "A" else None
    keep = ["Code", "Date", "label", "ret_o1_20", "ret_o1_40", "hit10", "tp10c"]
    parts = []
    for f in folds:
        tr = dt <= np.datetime64(pd.Timestamp(f.train_end))
        te = (dt >= np.datetime64(pd.Timestamp(f.test_start))) & (dt <= np.datetime64(pd.Timestamp(f.test_end)))
        if te.sum() < MIN_TEST or tr.sum() < MIN_TRAIN:
            continue
        part = target.loc[te, [c for c in keep if c in target.columns]].copy()
        if arm == "A":
            part["score"] = sigmoid(fit_predict(algo, params, seed, Xt[tr], yt[tr], Xt[te]))
        else:
            sm = ds <= np.datetime64(pd.Timestamp(f.train_end))
            if arm in ("W", "P"):
                X = np.vstack([Xt[tr], Xs[sm]])
                y = np.concatenate([yt[tr], y_src[sm]])
                w = np.concatenate([np.ones(int(tr.sum())), np.full(int(sm.sum()), tr.sum() / max(int(sm.sum()), 1))])
                part["score"] = sigmoid(fit_predict(algo, params, seed, X, y, Xt[te], weight=w))
            elif arm == "F":
                both = np.vstack([Xt[tr], Xt[te]])
                m_all = fit_predict(algo, params, seed, Xs[sm], y_src[sm], both)
                m_tr, m_te = m_all[: int(tr.sum())], m_all[int(tr.sum()):]
                part["score_S"] = sigmoid(m_te)
                part["score"] = sigmoid(fit_predict(algo, params, seed, Xt[tr], yt[tr], Xt[te], init_tr=m_tr, init_te=m_te))
            else:
                raise ValueError(f"知らない腕: {arm}")
        part["fold"] = f.index
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def oof(algo: str, arm: str, target: pd.DataFrame, src: pd.DataFrame, cols: list, params: dict, shift: int, seeds,
        y_src: dict, stamp: dict):
    """学習器 × 腕 × ずらし × 種の out-of-fold（種の平均。F は S の点数も平均）。保存済みなら読む。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(stamp.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{algo}_{arm}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    missing = [f for f in files if not os.path.exists(f)]
    if missing and not time_left():
        return None
    folds = E41.folds_for(target["Date"], shift) if missing else None
    parts = []
    for sd, f in zip(seeds, files):
        if os.path.exists(f):
            parts.append(pd.read_parquet(f))
            continue
        if not time_left():
            return None
        t0 = time.time()
        o = fold_oof(algo, arm, target, src, cols, params, sd, folds, y_src["P" if arm == "P" else "W"])
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {algo} {arm} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    if "score_S" in o.columns:
        o["score_S"] = np.mean([x["score_S"].to_numpy(dtype=float) for x in parts], axis=0)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def with_s(o: pd.DataFrame) -> pd.DataFrame:
    """F の out-of-fold から、S の点数を score にした写し（腕 S）。"""
    return o.drop(columns=["score"]).rename(columns={"score_S": "score"})


def run(algos, target, src, cols, params, shifts, seeds, y_src, stamp, arms=COMPUTE) -> dict:
    """学習器ごとに、本番と同じ作り（ずらし0・種42）と、ずらし × 種の平均を、腕ごとに。判定の腕を先に。時間切れの分は入らない。"""
    res = {}
    for algo in algos:
        r = {"prod": {}, "oofs": {}}
        for group in ([a for a in ("A", "W", "P") if a in arms], [a for a in ("F",) if a in arms]):
            for arm in group:
                o = oof(algo, arm, target, src, cols, params[algo], 0, (PROD_SEED,), y_src, stamp)
                if o is not None:
                    r["prod"][arm] = o
            for sh in shifts:
                for arm in group:
                    o = oof(algo, arm, target, src, cols, params[algo], sh, seeds, y_src, stamp)
                    if o is not None:
                        r["oofs"][(sh, arm)] = o
        if "F" in r["prod"]:
            r["prod"]["S"] = with_s(r["prod"]["F"])
        for sh in shifts:
            if (sh, "F") in r["oofs"]:
                r["oofs"][(sh, "S")] = with_s(r["oofs"][(sh, "F")])
        res[algo] = r
    return res


def have(r: dict, arms, shifts) -> list:
    return [a for a in arms if a in r["prod"] and all((sh, a) in r["oofs"] for sh in shifts)]


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #

def report(res: dict, shifts, seeds, stamp: dict, extra: dict, pinfo: dict) -> dict:
    primary = ("A", "W", "P")
    algos = [a for a in ALGOS if a in res and set(primary) <= set(have(res[a], ARMS, shifts))]
    full_design = tuple(shifts) == SHIFTS and tuple(seeds) == SEEDS
    print("\n" + "=" * 78)
    print(f"ラベル {K:.1f}σ / 判定の腕（A・W・P）のそろった学習器: {', '.join(algos) or 'なし'}"
          + ("" if full_design else f"（この回は ずらし {list(shifts)}・種 {list(seeds)} だけ。判定の形ではない）"))
    summary = {"data": stamp, **extra, "label_k": K, "perm_seed": PERM_SEED, "params": pinfo, "shifts": list(shifts),
               "seeds": list(seeds), "full_design": full_design, "production_oof": {}, "windows": {}, "judge7": {},
               "top": {}, "per_window": {}}

    print("\n■ 1. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。T のテストの行）")
    print(f"  {'':<6}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
    for a in algos:
        arms = have(res[a], ARMS, shifts)
        ms = {}
        for arm in arms:
            o = res[a]["prod"][arm]
            m = metrics(o)
            ms[arm] = m
            summary["production_oof"][f"{a}_{arm}"] = {k: float(v) for k, v in m.items()}
            print(f"  {a:<6}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
                  f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
                  f"{m['ret_o1_20_worst']:>+8.2f}pt{len(o):>8,}")
        for x, y in PAIRS:
            if x in ms and y in ms:
                print(f"  {a:<6}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}"
                      f"{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    tables = {}
    print(f"\n■ 2. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り × 種{len(seeds)}つの平均）")
    print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
    for a in algos:
        arms = have(res[a], ARMS, shifts)
        s = E105.window_table(res[a], arms, shifts)
        tables[a] = (s, arms)
        s.to_csv(path(f"{a}_auc_by_window.csv"), index=False)
        for met, nm in (("pr", "PR"), ("roc", "ROC")):
            for x, y in PAIRS:
                if x in arms and y in arms:
                    line, st = pair(f"{a} {nm} {y}−{x}", s[f"{met}_{x}"].to_numpy(), s[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{a}_{met}_{y}-{x}"] = st

    print(f"\n■ 3. §7（2026-09-26 改訂）の判定（窓の PR-AUC。回す前に決めた。{K:.1f}σ）: "
          "平均 W−A が正 / 対照 P−A を上回る / W が上の窓が過半")
    for a in algos:
        s, _ = tables[a]
        j = judge7((s["pr_W"] - s["pr_A"]).to_numpy(), (s["pr_P"] - s["pr_A"]).to_numpy())
        summary["judge7"][a] = j
        print(f"  {a:<6}W−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / P−A {j['mean_pa']:>+8.4f} "
              f"{'○' if j['beats_placebo'] else '×'} / 上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）"
              f"{'○' if j['majority'] else '×'} → {'満たす' if j['pass'] else '満たさない'}")
    passed = sum(int(summary["judge7"][a]["pass"]) for a in algos)
    done = len(algos) == len(ALGOS) and full_design
    verdict = (("満たす（木3モデル中2つ以上）" if passed >= 2 else "満たさない（木3モデル中2つ未満）") if done
               else f"{passed}/{len(algos)} が満たす（木3モデル・ずらし3通り・種3つがそろってから判定）")
    summary["judge7"]["verdict"] = {"passed": passed, "of": len(algos), "complete": done, "text": verdict}
    print(f"  → §7: {verdict}" + ("" if abs(K - 1.2) < 1e-9 else "（1.5σ は参考。判定は 1.2σ の回）"))

    print("\n■ 4. 窓の中の上位2%（参考）: Precision / +10% に届いた割合 / 1取引の平均（%。届けば +10、届かなければ20日目の終値）")
    for a in algos:
        _, arms = tables[a]
        tt = E105.top_table(res[a], arms, shifts)
        if not len(tt):
            continue
        tt.to_csv(path(f"{a}_top_by_window.csv"), index=False)
        summary["top"][a] = {}
        for arm in arms:
            r = {m: float(tt[f"{m}_{arm}"].mean()) for m in ("prec", "hit10", "tp10c")}
            line = f"  {a:<6}{arm:<4}{r['prec']*100:>7.1f}%{r['hit10']*100:>8.1f}%{r['tp10c']:>+8.2f}%"
            if arm != "A":
                for m, nm, sc in (("prec", "Precision", 100), ("hit10", "+10%", 100), ("tp10c", "平均", 1)):
                    d = (tt[f"{m}_{arm}"] - tt[f"{m}_A"]).to_numpy(dtype=float) * sc
                    d = d[np.isfinite(d)]
                    if len(d) > 1:
                        se = d.std(ddof=1) / np.sqrt(len(d))
                        line += f" ｜ {nm} {d.mean():+.2f} ± {se:.2f}（上 {int((d > 0).sum())}/{len(d)}）"
                        r[f"d_{m}"] = {"mean": float(d.mean()), "se": float(se), "wins": int((d > 0).sum()), "n": int(len(d))}
            summary["top"][a][arm] = r
            print(line + f"（{len(tt)}窓・{int(tt['k'].sum()):,}件）")

    print("\n■ 5. 窓ごとの PR-AUC の差（W−A / P−A / F−A。ずらしごと）")
    for a in algos:
        s, arms = tables[a]
        summary["per_window"][a] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
        for sh in shifts:
            g = s[s["shift"] == sh].sort_values("fold")
            if not len(g):
                continue
            line = f"  {a:<6}ずらし{sh}か月（{len(g)}窓）W−A " + " ".join(f"{v:+.3f}" for v in g["pr_W"] - g["pr_A"])
            line += " ｜ P−A " + " ".join(f"{v:+.3f}" for v in g["pr_P"] - g["pr_A"])
            if "F" in arms:
                line += " ｜ F−A " + " ".join(f"{v:+.3f}" for v in g["pr_F"] - g["pr_A"])
            print(line)
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE, K
    ap = argparse.ArgumentParser(description="実験113: 78週高値の更新日すべてで事前学習 → 今の母集団で仕上げる（木3モデル・§7。検証だけ）")
    ap.add_argument("--k", type=float, default=K_DEFAULT, choices=(1.2, 1.5), help="ラベルの到達しきい値（σ の倍数）。既定 1.2")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算する学習器")
    ap.add_argument("--arms", default=",".join(COMPUTE), help="この回に計算する腕（A・W・P・F。S は F の途中の値）")
    ap.add_argument("--shifts", default=",".join(map(str, SHIFTS)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の out-of-fold は使えなくなる）")
    ap.add_argument("--prepare-only", action="store_true", help="表を作って確かめるだけ（out-of-fold は回さない）")
    ap.add_argument("--budget-min", type=float, default=290.0, help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    args = ap.parse_args(argv)
    TAG = args.tag
    K = float(args.k)
    RUN = f"{TAG}_k{int(round(K * 10)):02d}"
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    algos = [a for a in args.algos.split(",") if a]
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in algos if a not in ALGOS] + [a for a in arms if a not in COMPUTE]
    if bad:
        raise SystemExit(f"知らない学習器・腕: {bad}（学習器 {ALGOS}・腕 {COMPUTE}）")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cols = F.columns(F.DEFAULT_PRESET)
    params, pinfo = {}, {}
    for a in ALGOS:
        params[a], pinfo[a] = E112.prod_params(a, K)

    print("=" * 78)
    print(f"実験113 78週高値の更新日すべてで事前学習 → 今の母集団で仕上げる（{K:.1f}σ・木3モデル・OOF + 窓・対照 P）")
    print(f"  列: {F.DEFAULT_PRESET} {len(cols)}列 指紋 {F.signature(cols)}")
    for a in ARMS:
        print(f"  {LABELS[a]}")
    for a in ALGOS:
        i = pinfo[a]
        print(f"  パラメータ {a}: 本番の {i['model']}（{i['preset']}・{i['n_features']}列・学習 {i['trained']}・指紋 {i['hash']}）"
              f"{'' if a in algos else '（この回は計算しない）'}  {short(params[a])}")
    print(f"  腕 {arms} / 窓のずらし {shifts}か月 / 種 {seeds} / 対照の種 {PERM_SEED} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    target, src, stamp = prepare(cols, args.rebuild)
    for d in (target, src):
        d["Date"] = pd.to_datetime(d["Date"]).astype("datetime64[ns]")
        d["Code"] = d["Code"].astype(str)
    label_check = E105.check_label(target, K)
    target = E105.with_label(target, K)
    src = E105.with_label(src, K)
    if keys_of(src).isin(keys_of(target)).any():
        raise SystemExit("S と T の行が重なっている")
    extra = {"source": source_report(target, src), "label_check": label_check}
    if args.prepare_only:
        log("[prepare-only] 表を作って確かめた。out-of-fold は次の回から")
        return 0
    t0 = time.time()
    y_src = {"W": src["label"].to_numpy(dtype=int),
             "P": permuted(src[["Date", "label"]], ["label"], seed=PERM_SEED)["label"].to_numpy(dtype=int)}
    log(f"[data] 対照 P: S のラベルを日付内で入れ替えた（種 {PERM_SEED}）/ {time.time()-t0:.0f}秒")
    t0 = time.time()
    res = run(algos, target, src, cols, params, shifts, seeds, y_src, stamp, arms=arms)
    log(f"[oof] 計算 {time.time()-t0:.0f}秒")
    # この回に計算しない学習器も、判定の腕（A・W・P）の保存済みがそろっていれば集計に入れる（保存済みの腕だけを読む）
    for a in ALGOS:
        saved = _saved_arms(a, params, shifts, seeds, stamp)
        if a not in res and {"A", "W", "P"} <= set(saved):
            res.update(run([a], target, src, cols, params, shifts, seeds, y_src, stamp, arms=saved))
    report(res, shifts, seeds, stamp, extra, pinfo)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


def _saved_arms(algo: str, params: dict, shifts, seeds, stamp: dict) -> list:
    """algo の腕のうち、out-of-fold（本番と同じ作りと、ずらし × 種）がすべて保存済みのもの（読むだけで集計に入れられる腕）。"""
    ph = params_hash(params[algo]) + "_d" + hashlib.sha1(str(stamp.get("built_utc", "")).encode()).hexdigest()[:6]
    want = [(0, PROD_SEED)] + [(sh, sd) for sh in shifts for sd in seeds]
    return [arm for arm in COMPUTE
            if all(os.path.exists(path(f"{algo}_{arm}_{ph}_sh{sh}_s{sd}.parquet")) for sh, sd in want)]


if __name__ == "__main__":
    raise SystemExit(main())
