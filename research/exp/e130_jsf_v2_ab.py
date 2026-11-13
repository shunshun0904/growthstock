#!/usr/bin/env python3
"""
実験130: 日証金の需給の特徴量・作り直し版（research/jsf_features_v2.py の20列）を今の373列に足すと、+10% の LightGBM の
探索（5分割 CV）・本番と同じ作りの out-of-fold・32窓がどう変わるか（実験67 の作り直し。腕は「列」の違い）。検証だけ。

運用者の依頼（2026-11-12）「（過去3年が）そろい…であれば、特徴量として、作成してほしいです」「列方向、横方向、銘柄間での
バリデーションチェックも欠かさずに」。運用者の選択（同日、AskUserQuestion）: 「373列で検証し直す」（本番は変えず、結果を見て
入れるかを決める）・「16列＋過去との比較の列」（正しくは17列＋60営業日の位置3列 = 20列。docs/DATA_JSF.md）・「写しと
J-Quants で作る」・「全銘柄の過去を取り直す」・「控えを残して上書き」・「定時の続きで取る」・「3列: 60日の位置だけ」・
「+10% の lgbm だけ」（腕 A/B/C/P、本番と同じ探索、本番と同じ OOF と32窓、全行と日証金の付いた行だけの両方）・「営業日で数える」。
運用者の方針「いきなり変えるよりも、まずは検証してほしいです。本番導入するかはその結果次第です」→ 本番の設定（features.py・
lgbm_params.json）と build_dataset.py には書かない。20列はこの台本の中で付ける（jsf_features_v2.build）。
運用者の指示「パラメータチューニングする際も全特徴量を使ったモデルでお願いします」→ 探索は 373 + 20 = 393列で行う（腕 B）。

腕（データは同じ1本。列だけが違う）
  A   今の373列（features.DEFAULT_PRESET。回した時点の既定を名前で固定）。A の列で探索
  B   373 + 20列で探索し直し → 入れたときに週次の再学習がやること
  C   373 + 20列を A のパラメータで（探索なし）→ 列を足したことそのものの効果
  P   対照: 20列の値を同じ日付の行どうしで入れ替えたもの（種 PERM_SEED）を A のパラメータで。§7 の「対照 P の改善を上回る」
  C−A が「20列を足すことの効果」、P−A が「列数が増えただけの幅」、C−P が「20列そのものの情報」、B−A が「再学習で起きること」

ラベル: 本番の +10%（label_specs.for_model("lgbm")。fixed_return_v1）。データセットの label が本番の定義と全行一致することを
確かめてから使う（label_specs.check_training_frame）。行: 学習は全行（日証金の値が無い行は欠測のまま。木は欠測として扱う）。
評価は2通り: 全行（32窓）と、日証金の付いた行だけ（jsf2_lag がある行。その行が MIN_WINDOW_ROWS 件以上ある窓）。

判定（回す前に運用者に確かめてから、この台本と docs/MODEL_ADOPTION_RULES.md に書く。下は案）
  docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）の1モデル分の3条件を、32窓（ずらし 0/2/4か月 × 種3つの平均）の PR-AUC に
  当てる: 窓ごとの C−A の平均が正・対照 P−A の平均を上回る・C が上の窓が過半（n//2 + 1 以上）。木3モデルの票は数えられないので、
  本番に入れるかは結果を見て運用者が決める（実験101〜105 と同じ扱い）。

出すもの（公開ログには件数・割合・日付・精度だけ。銘柄・値は出さない。日証金は私的利用の範囲）
  0. 20列の付き方（欠測の割合・分位）、既存の373列との順位相関の最大、学習の寄与（gain）の割合と上位の列
  1. 探索の CV（A・B）  2. 本番と同じ作りの OOF（ずらし0・種42。全行と付いた行だけ）  3. 32窓の差（全行）
  3b. 窓の差（付いた行だけ）  4. §7 の判定  5. 窓の中の上位2% の Precision・+10% の指値に届いた割合・1取引の平均（参考）
  6. 窓ごとの値

    exp=e130_jsf_v2_ab.py args="--restrict-shift 1"            ← 写しの日 + 1営業日 = 申込日（運用者に確かめてから決める）
列の付き方だけ見る: exp=e130_jsf_v2_ab.py args="--restrict-shift 1 --dry"
試運転:             exp=e130_jsf_v2_ab.py args="--tag e130smoke --n-trials 2 --shifts 0 --arms A,C,P --restrict-shift 1"

Run Experiment は同じ実験を同時に1本しか走らせない。表（20列を付けた frame）は1回目に作って research/_data/oof/<tag>_* に置き、
以降の回は actions/cache で引き継いだそれを使う（日証金の過去ファイル・作りの台本・写しの対応が変わったら作り直す）。
--budget-min（既定 290分）を過ぎたら新しい計算を始めずに終わる（保存済みの探索・out-of-fold は次の回が続きから使う）。
判定の腕（A・C・P）を先に、B を後に計算する。
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
import jsf_features_v2 as V2  # noqa: E402
import lab  # noqa: E402
import label_specs as LS  # noqa: E402
import live_track as LT  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e44_shortsale import permuted  # noqa: E402
from e68_edinet_ab import judge7, pair, params_hash, short  # noqa: E402
from e104_market_drift import top_by_window  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV       # 週次の再学習（retrain-weekly.yml）と同じ（year_cap_date）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
ALGO = "lgbm"
SEEDS = (42, 7, 123)
PROD_SEED = 42
#: 比べる土台。回した時点の既定（373列）を名前で固定する
BASE_PRESET = "all_plus_prog_listing_vol_book_shape_dd_recent_tx_financial"
NEW_COLS = tuple(V2.columns("all"))
ARMS = ("A", "C", "P", "B")
#: 判定に使う腕（先に計算する）
PRIMARY = ("A", "C", "P")
COLS_OF = {"A": "base", "B": "new", "C": "new", "P": "perm"}
PARAMS_OF = {"A": "A", "B": "B", "C": "A", "P": "A"}
LABELS = {"A": "A  今の373列（探索した A のパラメータ）", "B": "B  373 + 20列・探索し直し",
          "C": "C  373 + 20列・A のパラメータ", "P": "P  対照（20列を日付内で入れ替え）・A のパラメータ"}
PAIRS = (("A", "C"), ("A", "P"), ("P", "C"), ("A", "B"), ("C", "B"))
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261113
MIN_WINDOW_ROWS = 100
#: 重なりの多い列の目安（既存の列との順位相関の絶対値）
OVERLAP = (0.9, 0.95)
TP = LT.TAKE_PROFIT
TAG = "e130"
RUN = "e130"
DEADLINE: float | None = None
TIMED_OUT = False
#: 評価の行の組（all = 全行、jsf = 日証金の付いた行だけ）
ROWSETS = ("all", "jsf")
ROWSET_JA = {"all": "全行", "jsf": "日証金の付いた行だけ"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fpath(name: str) -> str:
    """表（腕によらない）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def path(name: str) -> str:
    """探索・out-of-fold・要約。"""
    return os.path.join(OOF_DIR, f"{RUN}_{name}")


def time_left() -> bool:
    global TIMED_OUT
    if DEADLINE is None or time.time() < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min を過ぎたので、新しい計算は始めない（保存済みの分だけで報告。続きは次の回）")
    return False


def shift_tag(shift) -> str:
    return "rsN" if shift is None else f"rs{int(shift):+d}".replace("+", "p").replace("-", "m")


def file_sha(p: str) -> str:
    if not os.path.exists(p):
        return "none"
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def source_stamp(data_dir: str) -> dict:
    """表を作り直すかを決める材料: 日証金の元のファイル・作りの台本の指紋。"""
    return {"jsf_hist": file_sha(os.path.join(data_dir, "jsf_hist.parquet")),
            "jsf_balance": file_sha(os.path.join(data_dir, "jsf_balance.parquet")),
            "jsf_lending": file_sha(os.path.join(data_dir, "jsf_lending.parquet")),
            "jsf_restrict": file_sha(os.path.join(data_dir, "jsf_restrict.parquet")),
            "v2_code": file_sha(V2.__file__)}


def check_label(df: pd.DataFrame) -> dict:
    """データセットの label が本番の +10% の定義（label_specs）と全行一致するか。違えば止める。"""
    want = LS.labels(df, "lgbm")
    have = pd.to_numeric(df["label"], errors="coerce")
    same = (want.isna() & have.isna()) | (want.notna() & have.eq(want))
    out = {"rows": int(len(df)), "same": int(same.sum()), "defined": int(want.notna().sum()),
           "spec": LS.for_model("lgbm")}
    print(f"  確かめ: データセットの label と本番の +10% の定義の一致 {out['same']:,} / {out['rows']:,}行"
          f"（決まる行 {out['defined']:,}）")
    if out["same"] != out["rows"]:
        raise SystemExit("データセットの label が本番の +10% の定義と合わない")
    return out


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def jsf_checks(df: pd.DataFrame) -> dict:
    """付けた20列の最小限の確かめ（値の確認そのものは research/exp/probe_jsf_v2.py）。範囲の外・無限があれば止める。"""
    rng = {"jsf2_fee": (0, None), "jsf2_fee_days20": (0, 20), "jsf2_fee_max20": (0, None),
           "jsf2_restrict": (0, 3), "jsf2_lendable": (0, 1), "jsf2_loan_pos60": (0, 1),
           "jsf2_stock_pos60": (0, 1), "jsf2_ratio_pos60": (0, 1), "jsf2_lag": (1, V2.MAX_STALE - 1)}
    bad = {}
    for c in V2.columns("both"):
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        n = int(np.isinf(s).sum())
        lo, hi = rng.get(c, (None, None))
        if lo is not None:
            n += int(np.nansum(s < lo))
        if hi is not None:
            n += int(np.nansum(s > hi))
        if n:
            bad[c] = n
    have = df["jsf2_lag"].notna()
    out = {"rows": int(len(df)), "attached": int(have.sum()),
           "attached_since_2023_10": float(have[df["Date"] >= "2023-10-01"].mean()) if (df["Date"] >= "2023-10-01").any()
           else float("nan"), "bad": bad}
    print(f"  確かめ: 20列が付いた行 {out['attached']:,} / {out['rows']:,}（2023年10月以降で "
          f"{out['attached_since_2023_10'] * 100:.1f}%）・範囲の外か無限の値 {sum(bad.values())}")
    if bad:
        raise SystemExit(f"20列に範囲の外か無限の値がある: {bad}")
    return out


def prepare(cols: list, shift, rebuild: bool = False):
    """lab.frame に20列と出口（+10% の指値）を付けた表（全行）と記録。"""
    pf, ps = fpath("frame.parquet"), fpath("stamp.json")
    src = source_stamp(lab.DATA_DIR)
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        df = pd.read_parquet(pf)
        miss = [c for c in cols + list(V2.columns("both")) if c not in df.columns]
        same_src = stamp.get("sources") == src and stamp.get("restrict_shift") == shift
        if not miss and same_src:
            log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}）")
            return df, stamp
        log(f"[data] 保存済みの表を作り直す（無い列 {miss[:5]}・元のファイルか写しの対応が変わった {not same_src}）")
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"]).astype("datetime64[ns]")
    frame["Code"] = frame["Code"].astype(str)
    miss = [c for c in cols + ["ret_o1_20", "ret_o1_40", "label"] if c not in frame.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    have = [c for c in V2.columns("both") if c in frame.columns]
    if have:
        raise SystemExit(f"学習データに jsf2_* の列がある（本番に入れた後に回すなら作りを見直す）: {have[:5]}")
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    keys = frame[["Code", "Date"]].copy()
    att = V2.build(keys, lab.DATA_DIR, shift=shift)
    if len(att) != len(frame) or not (att["Code"].to_numpy() == frame["Code"].to_numpy()).all() \
            or not (pd.to_datetime(att["Date"]).to_numpy() == frame["Date"].to_numpy()).all():
        raise SystemExit("20列を付けたら行の並びが変わった")
    df = pd.concat([frame.reset_index(drop=True), att[V2.columns("both")].reset_index(drop=True)], axis=1)
    log(f"[data] 20列を付けた / {time.time()-t0:.0f}秒")
    print("\n■ 確かめ")
    checks = {"jsf": jsf_checks(df)}

    t0 = time.time()
    bars_px = LT.load_bars(lab.DATA_DIR, start=df["Date"].min() - pd.Timedelta(days=10))
    bars_px["Date"] = bars_px["Date"].astype("datetime64[ns]")
    ex, chk = E83.attach_exits(df[["Code", "Date", "ret_o1_20"]], bars_px, targets=(TP,))
    del bars_px
    if len(ex) != len(df) or not (ex["Code"].to_numpy() == df["Code"].to_numpy()).all():
        raise SystemExit("出口の突き合わせで行の並びが変わった")
    k = E83.tp_key(TP)
    exit_ok = chk["hold5_match"] == chk["hold5_both"] and chk["order_violations"] == 0
    df["hit10"] = ex[f"hit{k}"].to_numpy(dtype=float) if exit_ok else np.nan
    df["tp10c"] = pd.to_numeric(ex[f"tp{k}c"], errors="coerce").to_numpy(dtype=float) if exit_ok else np.nan
    checks["exits"] = {**chk, "ok": bool(exit_ok)}
    print(f"  確かめ: 出口の5日平均と ret_o1_20 の一致 {chk['hold5_match']:,} / {chk['hold5_both']:,}行 → "
          f"{'一致' if exit_ok else '合わない（上位2% の +10% は出さない）'}（{time.time()-t0:.0f}秒）")

    yr = df["Date"].dt.year
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rows": int(len(df)),
             "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
             "rows_by_year": {str(y): int((yr == y).sum()) for y in sorted(yr.unique())}, "checks": checks,
             "missing": {c: int(df[c].isna().sum()) for c in NEW_COLS}, "sources": src, "restrict_shift": shift}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return df, stamp


def coverage_report(df: pd.DataFrame, base: list) -> dict:
    """20列の欠測の割合と分位、既存の373列との順位相関の最大（件数と割合・相関だけ）。"""
    out = {"columns": {}, "overlap": {}}
    att = df[df["jsf2_lag"].notna()]
    print(f"\n■ 0. 20列の付き方（{len(df):,}行のうち付いた行 {len(att):,}。分位は付いた行の 5% / 50% / 95%）")
    for c in NEW_COLS:
        v = pd.to_numeric(att[c], errors="coerce")
        qs = v.quantile([0.05, 0.5, 0.95]) if v.notna().any() else pd.Series([np.nan] * 3, index=[0.05, 0.5, 0.95])
        out["columns"][c] = {"missing_all": float(df[c].isna().mean()), "missing_attached": float(v.isna().mean()),
                             "q05": float(qs[0.05]), "q50": float(qs[0.5]), "q95": float(qs[0.95])}
        print(f"    {c:<22} 欠測 全行 {df[c].isna().mean()*100:5.1f}%・付いた行 {v.isna().mean()*100:5.1f}%  "
              f"{qs[0.05]:>10.3g} / {qs[0.5]:>10.3g} / {qs[0.95]:>10.3g}")
    ranks = att[base + list(NEW_COLS)].rank()
    corr = ranks.corr()
    best = {}
    for c in NEW_COLS:
        s = corr.loc[c, base].abs()
        s = s[s.notna()]
        best[c] = (s.idxmax(), float(s.max())) if len(s) else (None, float("nan"))
    inner = corr.loc[list(NEW_COLS), list(NEW_COLS)].abs().to_numpy()
    iu = np.triu_indices(len(NEW_COLS), 1)
    pairs = inner[iu]
    for th in OVERLAP:
        n_base = sum(1 for c in NEW_COLS if best[c][1] >= th)
        n_pair = int(np.nansum(pairs >= th))
        out["overlap"][str(th)] = {"vs_base": n_base, "pairs_within": n_pair}
        print(f"  既存の373列のどれかと順位相関の絶対値が {th} 以上の列 {n_base} / {len(NEW_COLS)}・20列どうしで {th} 以上の組 {n_pair}")
    print("  既存の列といちばん重なる列: " + " / ".join(f"{c}↔{best[c][0]} {best[c][1]:.2f}" for c in NEW_COLS))
    out["max_abs_rho_vs_base"] = {c: {"col": best[c][0], "rho": best[c][1]} for c in NEW_COLS}
    return out


def gain_report(df: pd.DataFrame, base: list, params: dict, cutoff) -> dict:
    """探索の期間の全行で C（373 + 20列・A のパラメータ）を1本学習し、寄与（gain）の割合と上位の列（参考）。"""
    import lightgbm as lgb

    cols = base + list(NEW_COLS)
    sub = df[(df["Date"] <= cutoff) & df["label"].notna()]
    y = sub["label"].to_numpy(dtype=int)
    m = lgb.LGBMClassifier(**{**params, "random_state": PROD_SEED}, scale_pos_weight=tuning.scale_pos_weight(y))
    m.fit(sub[cols].to_numpy(dtype=float), y)
    gain = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=cols)
    tot = float(gain.sum()) or 1.0
    share = {"既存の373列": float(gain[base].sum() / tot), "日証金の20列": float(gain[list(NEW_COLS)].sum() / tot)}
    print(f"\n■ 0b. 学習の寄与（gain）の割合（探索の期間 〜{cutoff.date()} の {len(sub):,}行で C を1本。参考）: "
          + " / ".join(f"{k} {v*100:.1f}%" for k, v in share.items()))
    all_rank = gain.rank(ascending=False)
    new_rank = gain[list(NEW_COLS)].sort_values(ascending=False)
    print(f"  20列の寄与（全{len(cols)}列の中の順位）: " + " / ".join(f"{c} {int(all_rank[c])}位" for c in new_rank.index))
    return {"share": share, "new": {c: {"gain_share": float(gain[c] / tot), "rank": int(all_rank[c])} for c in new_rank.index}}


# --------------------------------------------------------------------------- #
# 探索（週次の再学習と同じ関数・条件）
# --------------------------------------------------------------------------- #
def tune(arm: str, frame: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict):
    p = path(f"params_{arm}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if (rec.get("_n_trials") == n_trials and rec.get("_cutoff") == str(cutoff.date())
                and rec.get("_data") == stamp["built_utc"] and rec.get("_sig") == F.signature(cols)):
            return rec
    if not time_left():
        return None
    sub = frame[(frame["Date"] <= cutoff) & frame["label"].notna()]
    t0 = time.time()
    log(f"  [{arm}] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列）")
    params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS, embargo_days=T.EMBARGO_DAYS,
                         scheme=CV_SCHEME, objective=CV_OBJECTIVE, model="classifier", verbose=False,
                         label_spec=LS.for_model("lgbm"))
    rec = {"params": tuning.params_for("x", store={"x": params}), "_cv": dict(tuning.LAST_CV)}
    rec["_seconds"] = round(time.time() - t0)
    rec["_n_trials"] = n_trials
    rec["_cutoff"] = str(cutoff.date())
    rec["_data"] = stamp["built_utc"]
    rec["_sig"] = F.signature(cols)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    cv = rec["_cv"]
    log(f"  [{arm}] {rec['_seconds']}秒 / CV PR-AUC {cv['mean_pr_auc']:.4f}（±{cv['std']:.4f}）/ ROC {cv['mean_roc_auc']:.4f}")
    return rec


# --------------------------------------------------------------------------- #
# out-of-fold（本番と同じ作り。E41.oof_folds）
# --------------------------------------------------------------------------- #
def oof(arm: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds):
    """腕 × パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(frame.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{ALGO}_c{COLS_OF[arm]}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    missing = [f for f in files if not os.path.exists(f)]
    if missing and not time_left():
        return None
    folds = E41.folds_for(frame["Date"], shift) if missing else None
    parts = []
    extra = frame[["Code", "Date", "hit10", "tp10c"]].assign(jsf=frame["jsf2_lag"].notna().to_numpy())
    for sd, f in zip(seeds, files):
        if os.path.exists(f):
            parts.append(pd.read_parquet(f))
            continue
        if not time_left():
            return None
        t0 = time.time()
        o = E41.oof_folds(ALGO, frame, cols, params, sd, folds)
        o = o.merge(extra, on=["Code", "Date"], how="left")
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {arm} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def rows_of(o: pd.DataFrame, rows: str) -> pd.DataFrame:
    if rows == "all":
        return o
    if rows == "jsf":
        return o[o["jsf"].eq(True)]                   # 突き合わせに無い行（欠測）は付いていない側に数える
    raise KeyError(rows)


def windows_on(o: pd.DataFrame, rows: str = "all") -> pd.DataFrame:
    """窓ごとの AUC と、窓の期間・件数・正例率（MIN_WINDOW_ROWS 未満の窓は落とす）。rows で評価の行を選ぶ。"""
    cols = ["fold", "pr", "roc", "start", "end", "n", "pos"]
    o = rows_of(o, rows)
    n = o.groupby("fold").size()
    sub = o[o["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if not len(sub):
        return pd.DataFrame(columns=cols)
    # 正例と負例の両方がある窓だけ（付いた行だけにすると、古い窓で片方しかないことがある）
    both = sub.groupby("fold")["label"].agg(lambda s: s.nunique() == 2)
    sub = sub[sub["fold"].isin(both[both].index)]
    if not len(sub):
        return pd.DataFrame(columns=cols)
    w = AB.auc_by_window(sub)
    if not len(w):
        return pd.DataFrame(columns=cols)
    g = sub.groupby("fold")
    info = pd.DataFrame({"start": g["Date"].min(), "end": g["Date"].max(), "n": g.size(),
                         "pos": g["label"].mean()}).reset_index()
    return w.merge(info, on="fold", how="left")[cols]


def run(arms: list, frames: dict, colsets: dict, cutoff, n_trials: int, stamp: dict, shifts: list) -> dict:
    """判定の腕（A・C・P）を先に、B を後に計算する。時間切れの腕は結果に入らない。"""
    res = {"recs": {}, "params": {}, "prod": {}, "oofs": {}}
    res["recs"]["A"] = tune("A", frames["base"], colsets["base"], cutoff, n_trials, stamp)
    if res["recs"]["A"] is None:
        return res
    first = [a for a in PRIMARY if a in arms]
    later = [a for a in ARMS if a in arms and a not in PRIMARY]
    for group in (first, later):
        if "B" in group:
            res["recs"]["B"] = tune("B", frames["new"], colsets["new"], cutoff, n_trials, stamp)
        for arm in group:
            rec = res["recs"].get(PARAMS_OF[arm])
            if rec is None:
                continue
            res["params"][arm] = rec["params"]
            cs = COLS_OF[arm]
            o = oof(arm, frames[cs], colsets[cs], rec["params"], 0, (PROD_SEED,))
            if o is not None:
                res["prod"][arm] = o
        for sh in shifts:
            for arm in group:
                if arm not in res["params"]:
                    continue
                cs = COLS_OF[arm]
                o = oof(arm, frames[cs], colsets[cs], res["params"][arm], sh, SEEDS)
                if o is not None:
                    res["oofs"][(sh, arm)] = o
    return res


def complete_arms(res: dict, shifts: list) -> list:
    return [a for a in ARMS if a in res["prod"] and all((sh, a) in res["oofs"] for sh in shifts)]


def window_table(res: dict, arms: list, shifts: list, rows: str = "all") -> pd.DataFrame:
    out = []
    for sh in shifts:
        w = None
        for arm in arms:
            x = windows_on(res["oofs"][(sh, arm)], rows).rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
            if w is not None:
                x = x[["fold", f"pr_{arm}", f"roc_{arm}"]]
            w = x if w is None else w.merge(x, on="fold")
        if w is None or not len(w):
            continue
        for _, r in w.iterrows():
            out.append({"shift": sh, "fold": int(r["fold"]), "start": r["start"], "end": r["end"], "n": int(r["n"]),
                        "pos": float(r["pos"]), **{f"{m}_{a}": float(r[f"{m}_{a}"]) for m in ("pr", "roc") for a in arms}})
    return pd.DataFrame(out)


def top_table(res: dict, arms: list, shifts: list) -> pd.DataFrame:
    out = []
    for sh in shifts:
        w = None
        for arm in arms:
            x = top_by_window(res["oofs"][(sh, arm)]).rename(columns={c: f"{c}_{arm}" for c in ("prec", "hit10", "tp10c")})
            x = x.drop(columns=["n", "k"]) if w is not None else x
            w = x if w is None else w.merge(x, on="fold")
        if w is not None and len(w):
            out += [{"shift": sh, **r} for r in w.to_dict("records")]
    return pd.DataFrame(out)


def judge_line(s: pd.DataFrame, label: str) -> dict | None:
    if not len(s) or not all(f"pr_{a}" in s.columns for a in ("A", "C", "P")):
        print(f"  {label}: 腕がそろっていない（A・C・P）")
        return None
    j = judge7((s["pr_C"] - s["pr_A"]).to_numpy(), (s["pr_P"] - s["pr_A"]).to_numpy())
    print(f"  {label}: C−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / P−A {j['mean_pa']:>+8.4f} "
          f"{'○' if j['beats_placebo'] else '×'} / 上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）"
          f"{'○' if j['majority'] else '×'} → {'満たす' if j['pass'] else '満たさない'}")
    return j


def report(res: dict, shifts: list, stamp: dict, extra: dict) -> dict:
    arms = complete_arms(res, shifts)
    missing = [a for a in ARMS if a not in arms]
    print("\n" + "=" * 78)
    print("ラベル +10%（本番） / 結果のそろった腕: " + (", ".join(arms) or "なし") + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, **extra, "perm_seed": PERM_SEED, "tuning": {}, "production_oof": {},
               "windows": {}, "judge7": {}, "top": {}, "per_window": {}}

    print(f"\n■ 1. 探索の CV（{CV_SCHEME}・{N_SPLITS}分割。楽観側に出る。パラメータ選び用）")
    for arm in ("A", "B"):
        rec = res["recs"].get(arm)
        if not rec:
            continue
        cv = rec["_cv"]
        fs = cv.get("fold_scores") or []
        print(f"  {arm:<4}{cv['mean_pr_auc']:>8.4f} ±{cv['std']:.4f}  ROC {cv['mean_roc_auc']:.4f}  {rec.get('_seconds', 0)}秒  "
              f"分割ごと " + " / ".join(f"{v:.4f}" for v in fs))
        print(f"      パラメータ {short(rec['params'])}")
        summary["tuning"][arm] = {k: cv.get(k) for k in ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
    if "A" not in arms:
        print("  基準 A の out-of-fold がまだそろっていない（続きは次の回）")
        with open(path("summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
        return summary

    for rows in ROWSETS:
        print(f"\n■ 2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。{ROWSET_JA[rows]}）")
        print(f"  {'腕':<14}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
        ms = {}
        for arm in arms:
            o = rows_of(res["prod"][arm], rows)
            if o["label"].nunique() < 2:
                continue
            m = metrics(o)
            ms[arm] = m
            summary["production_oof"][f"{rows}_{arm}"] = {k: float(v) for k, v in m.items()}
            print(f"  {arm:<14}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
                  f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
                  f"{m['ret_o1_20_worst']:>+8.2f}pt{len(o):>8,}")
        for x, y in PAIRS:
            if x in ms and y in ms:
                print(f"  {y + '−' + x:<14}{ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}"
                      f"{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    tables = {}
    for rows in ROWSETS:
        s = window_table(res, arms, shifts, rows)
        tables[rows] = s
        s.to_csv(path(f"auc_by_window_{rows}.csv"), index=False)
        print(f"\n■ 3{'b' if rows == 'jsf' else ''}. 窓ごと（{ROWSET_JA[rows]}。境界を {'/'.join(map(str, shifts))}か月ずらした"
              f"{len(shifts)}通り × 種3つの平均。窓は{ROWSET_JA[rows]}が {MIN_WINDOW_ROWS}件以上で正例・負例がそろうもの）: {len(s)}窓")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        if not len(s):
            continue
        for met, nm in (("pr", "PR"), ("roc", "ROC")):
            for x, y in PAIRS:
                if x in arms and y in arms:
                    line, st = pair(f"{nm} {y}−{x}", s[f"{met}_{x}"].to_numpy(), s[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{rows}_{met}_{y}-{x}"] = st

    print("\n■ 4. §7（2026-09-26 改訂）の1モデル分の3条件（窓の PR-AUC）: 平均 C−A が正 / 対照 P−A を上回る / C が上の窓が過半")
    for rows in ROWSETS:
        j = judge_line(tables[rows], f"lgbm +10%・{ROWSET_JA[rows]}")
        if j is not None:
            summary["judge7"][rows] = j

    tt = top_table(res, arms, shifts)
    if len(tt):
        tt.to_csv(path("top_by_window.csv"), index=False)
        has_exit = bool(tt[[f"hit10_{a}" for a in arms]].notna().any().any())
        mets = (("prec", "Precision", 100), ("hit10", "+10%", 100), ("tp10c", "平均", 1)) if has_exit \
            else (("prec", "Precision", 100),)
        print(f"\n■ 5. 窓の中の上位2%（参考。全行。{len(tt)}窓・合計 {int(tt['k'].sum()):,}件）: Precision / +10% に届いた割合 / "
              "1取引の平均（%。届けば +10、届かなければ20日目の終値）")
        for arm in arms:
            r = {m: float(tt[f"{m}_{arm}"].mean()) for m in ("prec", "hit10", "tp10c")}
            line = f"  {arm:<14}{r['prec']*100:>7.1f}%" + (f"{r['hit10']*100:>8.1f}%{r['tp10c']:>+8.2f}%" if has_exit else "")
            if arm != "A":
                for m, nm, sc in mets:
                    d = (tt[f"{m}_{arm}"] - tt[f"{m}_A"]).to_numpy(dtype=float) * sc
                    d = d[np.isfinite(d)]
                    if len(d) > 1:
                        line += f" ｜ {nm} {d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(len(d)):.2f}（上 {int((d > 0).sum())}/{len(d)}）"
                        r[f"d_{m}"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                       "wins": int((d > 0).sum()), "n": int(len(d))}
            summary["top"][arm] = r
            print(line)

    show = [a for a in ("A", "B", "C", "P") if a in arms]
    for rows in ROWSETS:
        s = tables[rows]
        print(f"\n■ 6{'b' if rows == 'jsf' else ''}. 窓ごとの値（PR-AUC。{ROWSET_JA[rows]}。差は「後 − 前」）")
        for sh in shifts:
            g = s[s["shift"] == sh].sort_values("fold") if len(s) else s
            if not len(g):
                continue
            print(f"  ずらし{sh}か月（{len(g)}窓）")
            print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}" + "".join(f"{a[:8]:>9}" for a in show)
                  + f"{'C−A':>9}{'P−A':>9}")
            for _, r in g.iterrows():
                print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                      f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%" + "".join(f"{r[f'pr_{a}']:>9.4f}" for a in show)
                      + (f"{r['pr_C'] - r['pr_A']:>+9.4f}" if "C" in arms else "")
                      + (f"{r['pr_P'] - r['pr_A']:>+9.4f}" if "P" in arms else ""))
        summary["per_window"][rows] = (s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
                                       if len(s) else [])
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def parse_shift(x: str):
    x = str(x).strip().lower()
    if x in ("none", "n", ""):
        return None
    return int(x)


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE
    ap = argparse.ArgumentParser(description="実験130: 日証金の20列（v2）を373列に足す前後（lgbm +10%・検証だけ）")
    ap.add_argument("--restrict-shift", required=True,
                    help="制限措置の写しの日と申込日の対応（営業日。none で写しを使わない）。運用者に確かめてから決める")
    ap.add_argument("--arms", default=",".join(ARMS), help="この回に計算する腕（A は必ず含める）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0, help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="20列の付き方だけ出して終わる")
    args = ap.parse_args(argv)
    shift = parse_shift(args.restrict_shift)
    TAG = f"{args.tag}_{shift_tag(shift)}"
    RUN = TAG
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"知らない腕: {bad}（使えるもの {ARMS}）")
    if "A" not in arms:
        raise SystemExit("基準 A は外せません")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    base = F.columns(BASE_PRESET)
    new = list(NEW_COLS)
    dup = sorted(set(base) & set(new))
    if dup:
        raise SystemExit(f"今の列と重なる: {dup}")
    colsets = {"base": base, "new": base + new, "perm": base + new}

    print("=" * 78)
    print("実験130 日証金の需給の20列（v2）を今の373列に足す前後（lgbm +10%・5分割 CV の探索 + OOF + 32窓・対照 P）")
    print(f"  列: {BASE_PRESET} {len(base)}列 指紋 {F.signature(base)}"
          f"{'' if BASE_PRESET == F.DEFAULT_PRESET else '（いまの既定とは違う）'} / + 20列 → {len(base) + len(new)}列 "
          f"指紋 {F.signature(base + new)}")
    print(f"  制限措置の写しの対応: {'使わない' if shift is None else f'写しの日 + {shift}営業日 = 申込日'}")
    for a in ARMS:
        print(f"  {LABELS[a]}{'' if a in arms else '（この回は計算しない）'}")
    print(f"  探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}・{CV_OBJECTIVE}・ラベル +10%）/ 窓のずらし {shifts}か月 / "
          f"種 {SEEDS} / 対照の種 {PERM_SEED} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    full, stamp = prepare(base, shift, args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    label_check = check_label(full)
    df = full[full["label"].notna()].sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    extra = {"coverage": coverage_report(df, base), "label_check": label_check, "restrict_shift": shift}
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"\n  この回の行 {len(df):,}（+10% の正例率 {df['label'].mean()*100:.1f}%・{df['Date'].min().date()} 〜 "
          f"{df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では {(df['Date'] <= cutoff).sum():,}行・"
          f"日証金の付いた行 {int(df['jsf2_lag'].notna().sum()):,}")
    if args.dry:
        return 0
    t0 = time.time()
    perm = permuted(df, new, seed=PERM_SEED)
    perm.attrs["built_utc"] = stamp["built_utc"]
    log(f"[data] 対照 P: 20列を日付内で入れ替えた（種 {PERM_SEED}）/ {time.time()-t0:.0f}秒")
    frames = {"base": df, "new": df, "perm": perm}
    t0 = time.time()
    res = run(arms, frames, colsets, cutoff, args.n_trials, stamp, shifts)
    log(f"[{ALGO}] 計算 {time.time()-t0:.0f}秒 / そろった腕 {', '.join(complete_arms(res, shifts)) or 'なし'}")
    if res["recs"].get("A"):
        extra["gain"] = gain_report(df, base, res["recs"]["A"]["params"], cutoff)
    report(res, shifts, stamp, extra)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
