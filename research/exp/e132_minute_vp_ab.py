#!/usr/bin/env python3
"""
実験132: 分足から作る価格帯別出来高の特徴量（research/minute_features.py の33列から VWAP の3列を除いた30列）を今の373列に
足すと、+10% の LightGBM の探索（5分割 CV）・本番と同じ作りの out-of-fold・32窓がどう変わるか。検証だけ（本番の features.py・
lgbm_params.json は変えない）。

運用者の依頼（2026-11-19）「他のセッションで途中まで実施した、特徴量追加の検証です。研究用のPR8があるはずです。今回の分足の
データから作成できる新規特徴量の検証です。まずは、lgbmの+10%のモデルのみで、パラメータチューニングあり+oof検証してほしい」。
運用者の選択（同日、AskUserQuestion）:
  - 特徴量「PR8の21列＋時間帯の12列」: PR #8 の7指標（POC・VWAP・70%範囲の下限/上限と終値の差、POC・終値より上・終値±1% の
    株数の割合）＋ 時間帯の4指標（寄り・前場・後場・引けの株数の割合）× 1・5・20営業日 = 33列（分足だけ）
  - 評価「本番の作りだけ」: 実験130と同じ（全行で学習。新しい列は分足の前の行で欠測のまま。本番と同じ OOF と32窓を、全行と
    分足の付いた行だけの両方で数える）
  - 分足の日「当日 D まで」: D の行に D の引けまでの分足を使う（日足の特徴量と同じ。本番に入れるなら予測を分足の取得の後に回す）
  - 腕「AとBだけ」: §7 の判定（対照 P・A のパラメータの C が要る）は出さない。B−A を見る
運用者の選択（同日、33列の EDA（e132_minute_eda.py）の後）「VWAPの3列を外した30列」:
  vp_minute_vwap_gap_{1,5,20}d は日足の売買代金 ÷ 出来高でも作れる値（EDA で D の日足と相対差 2e-16 で一致）で、20日は既存の
  tx_vwap_gap_20 と順位相関 0.99。分足に固有の30列（価格帯の形・時間帯）だけで効果を見る。表には33列を付けて確かめは33列で行い、
  学習に足すのは30列だけ

腕（データは同じ1本。列だけが違う）
  A   今の373列（features.DEFAULT_PRESET。回した時点の既定を名前で固定）。A の列で探索
  B   373 + 30列で探索し直し → 入れたときに週次の再学習がやること
  C   373 + 30列・A のパラメータ（探索しない）→ パラメータの違いを除いた、列だけの効果。運用者の選択（同日、A・B の結果の後）
      「C の腕で列の効果を切り分ける」。A・B の探索と out-of-fold は保存済みのものを使い、C だけを計算する（--keep-frame で
      A・B と同じ表を使う）。対照 P は回さないので §7 の判定は出さない

ラベル: 本番の +10%（label_specs.for_model("lgbm")）。データセットの label が本番の定義と全行一致することを確かめてから使う。
探索: 週次の再学習と同じ（tuning.tune・50試行・5分割・year_cap_date・ホールドアウト12か月の手前まで）。
out-of-fold: 本番と同じ作り（36/6/6か月・エンバーゴ20営業日。E41.oof_folds）。ずらし0・種42 と、ずらし 0/2/4か月 × 種3つ（32窓）。
評価の行: 全行と、分足の付いた行だけ（30列のどれかがある行。分足は 2024-11-01 から）。

出すもの（公開ログには件数・割合・日付・精度だけ。銘柄・値は出さない。分足は J-Quants のアドオンのデータで、私的利用の範囲）
  0. 30列の付き方（欠測の割合・分位）・既存の373列との順位相関の最大  0b. 学習の寄与（gain）の割合（参考）
  1. 探索の CV（A・B）  2. 本番と同じ作りの OOF（全行・付いた行だけ）  3. 窓ごとの B−A（全行）  3b. 同（付いた行だけ）
  5. 窓の中の上位2% の Precision・+10% の指値に届いた割合・1取引の平均（参考）  6. 窓ごとの値
  3・3b には「訓練に分足の付いた行がある窓だけ」の行も出す（試運転 run 44519658877 の後、本番の回の前に足した）。訓練に分足の
  行が無い窓では30列が訓練の全行で欠測になり、LightGBM はその列を使わない。そこでの B−A はパラメータの違いだけを表す
  （A と B のパラメータが同じなら差はちょうど 0。試運転はそうだった）。「訓練に分足の無い窓」の行も出す（本番の回の後に足した）

置き場の分足（下書きの Release data-minute）は research/_data/minute/ に置く（Run Minute Experiment が落とす）。
    exp=e132_minute_vp_ab.py
列の付き方だけ見る: args="--dry"
試運転:             args="--tag e132smoke --n-trials 2 --shifts 0"
C を足す回:         args="--keep-frame"（A・B は保存済みの探索・out-of-fold を読む。C の out-of-fold だけ計算する）

表（33列を付けた frame）は1回目に作って research/_data/oof/<tag>_* に置き、以降の回は actions/cache で引き継いだそれを使う
（分足の取得記録・作りの台本が変わったら作り直す）。--budget-min（既定 290分）を過ぎたら新しい計算を始めずに終わる
（保存済みの探索・out-of-fold は次の回が続きから使う）。
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
import label_specs as LS  # noqa: E402
import live_track as LT  # noqa: E402
import minute_features as MF  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import volume_profile as VP  # noqa: E402
import volume_profile_data as VD  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e68_edinet_ab import pair, params_hash, short  # noqa: E402
from e104_market_drift import top_by_window  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
MINUTE_DIR = os.path.join(lab.DATA_DIR, "minute")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV       # 週次の再学習（retrain-weekly.yml）と同じ（year_cap_date）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
ALGO = "lgbm"
SEEDS = (42, 7, 123)
PROD_SEED = 42
#: 比べる土台。回した時点の既定（373列）を名前で固定する
BASE_PRESET = "all_plus_prog_listing_vol_book_shape_dd_recent_tx_financial"
#: 表に付ける列（minute_features の33列。付けた後の確かめは33列すべてで行う）
BUILT_COLS = tuple(MF.COLUMNS)
#: 学習に足さない列（運用者の選択 2026-11-19「VWAPの3列を外した30列」。日足だけで作れ、20日は tx_vwap_gap_20 と ρ=0.99）
DROPPED_COLS = tuple(MF.column("vwap_gap", n) for n in MF.WINDOWS)
#: B で足す列（分足に固有の30列。並びは minute_features のまま）
NEW_COLS = tuple(c for c in BUILT_COLS if c not in DROPPED_COLS)
ARMS = ("A", "B", "C")
COLS_OF = {"A": "base", "B": "new", "C": "new"}
#: パラメータをどの腕の探索から取るか（C は A のパラメータのまま列だけ足す）
PARAMS_OF = {"A": "A", "B": "B", "C": "A"}
LABELS = {"A": "A  今の373列（A の列で探索）", "B": f"B  373 + {len(NEW_COLS)}列（VWAP の3列を除く）・探索し直し",
          "C": f"C  373 + {len(NEW_COLS)}列・A のパラメータ（探索しない。列だけの効果）"}
#: 比べる組（後 − 前）
PAIRS = (("A", "B"), ("A", "C"))
MIN_WINDOW_ROWS = 100
#: 重なりの多い列の目安（既存の列との順位相関の絶対値）
OVERLAP = (0.9, 0.95)
TP = LT.TAKE_PROFIT
TAG = "e132"
RUN = "e132"
DEADLINE: float | None = None
TIMED_OUT = False
#: 評価の行の組（all = 全行、minute = 分足の付いた行だけ）
ROWSETS = ("all", "minute")
ROWSET_JA = {"all": "全行", "minute": "分足の付いた行だけ"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def path(name: str) -> str:
    return os.path.join(OOF_DIR, f"{RUN}_{name}")


def time_left() -> bool:
    global TIMED_OUT
    if DEADLINE is None or time.time() < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min を過ぎたので、新しい計算は始めない（保存済みの分だけで報告。続きは次の回）")
    return False


def file_sha(p: str) -> str:
    if not os.path.exists(p):
        return "none"
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def source_stamp() -> dict:
    """表を作り直すかを決める材料: 分足の取得記録・作りの台本の指紋。"""
    return {"minute_manifest": file_sha(os.path.join(MINUTE_DIR, "minute_manifest.json")),
            "minute_features": file_sha(MF.__file__), "volume_profile": file_sha(VP.__file__),
            "volume_profile_data": file_sha(VD.__file__)}


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
def minute_checks(df: pd.DataFrame, report: dict) -> dict:
    """付けた33列の確かめ。割合が 0〜1 の外・無限・「終値との差」が −100% 以下・期間の中で11指標がそろわない行があれば止める。"""
    bad = {}
    for c in BUILT_COLS:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        n = int(np.isinf(s).sum())
        if c.endswith(tuple(f"_share_{w}d" for w in MF.WINDOWS)):
            n += int(np.nansum((s < -1e-9) | (s > 1 + 1e-9)))
        else:
            n += int(np.nansum(s <= -1))
        if n:
            bad[c] = n
    torn = {}
    for w in MF.WINDOWS:
        cols = [MF.column(m, w) for m in MF.METRICS]
        have = df[cols].notna().sum(axis=1)
        torn[f"{w}d"] = int(((have > 0) & (have < len(cols))).sum())
    sums = {}
    for w in MF.WINDOWS:
        s = df[[MF.column(f"{b}_share", w) for b in VP.BUCKETS]].sum(axis=1, min_count=1)
        sums[f"{w}d"] = int((s.notna() & (s - 1).abs().gt(1e-6)).sum())
    out = {**report, "bad": bad, "torn": torn, "bucket_sum_off": sums}
    attached = df[list(BUILT_COLS)].notna().any(axis=1)
    since = df["Date"] >= pd.Timestamp(report["minute_from"]) if report.get("minute_from") else attached
    out["attached_share_since_minute"] = float(attached[since].mean()) if since.any() else float("nan")
    print(f"  確かめ: 33列が付いた行 {int(attached.sum()):,} / {len(df):,}（分足の始まり {report.get('minute_from')} 以降で "
          f"{out['attached_share_since_minute'] * 100:.1f}%）・範囲の外か無限 {sum(bad.values())}・"
          f"期間の中で欠けのある行 {sum(torn.values())}・時間帯の割合の合計が1でない行 {sum(sums.values())}")
    print(f"  分足の集計の状態（期間ごと。件数）: {report.get('status')}")
    if bad or sum(torn.values()) or sum(sums.values()):
        raise SystemExit(f"33列の確かめに失敗: 範囲 {bad} / 欠け {torn} / 合計 {sums}")
    return out


def prepare(cols: list, rebuild: bool = False, keep: bool = False):
    """
    lab.frame に33列と出口（+10% の指値）を付けた表（全行）と記録。

    keep: 保存済みの表があれば、分足の取得記録・台本が変わっていてもそのまま使う（腕を足す回で、前の回の腕と同じ表で比べる
    ため。毎晩の分足の取得で取得記録は変わるが、学習データの行はラベルの決まる日までなので、付けた列は変わらない）。
    """
    pf, ps = path("frame.parquet"), path("stamp.json")
    src = source_stamp()
    if src["minute_manifest"] == "none":
        raise SystemExit(f"分足の取得記録が無い: {MINUTE_DIR}/minute_manifest.json（Run Minute Experiment が落とす）")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        df = pd.read_parquet(pf)
        miss = [c for c in cols + list(BUILT_COLS) if c not in df.columns]
        if not miss and (stamp.get("sources") == src or keep):
            changed = sorted(k for k in src if (stamp.get("sources") or {}).get(k) != src[k])
            log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}）"
                + (f"。--keep-frame: 作ったときから変わったもの {changed}（表は作り直さない）" if changed else ""))
            return df, stamp
        if keep:
            log("[data] --keep-frame だが、保存済みの表に無い列があるので作り直す")
        log(f"[data] 保存済みの表を作り直す（無い列 {miss[:5]}・分足の記録か台本が変わった {stamp.get('sources') != src}）")
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"]).astype("datetime64[ns]")
    frame["Code"] = frame["Code"].astype(str)
    miss = [c for c in cols + ["ret_o1_20", "ret_o1_40", "label"] if c not in frame.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    have = [c for c in BUILT_COLS if c in frame.columns]
    if have:
        raise SystemExit(f"学習データに vp_minute_* の列がある（本番に入れた後に回すなら作りを見直す）: {have[:5]}")
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    att, report = MF.build(frame[["Code", "Date"]], MINUTE_DIR, lab.DATA_DIR)
    if len(att) != len(frame) or not (att["Code"].to_numpy() == frame["Code"].to_numpy()).all() \
            or not (pd.to_datetime(att["Date"]).to_numpy() == frame["Date"].to_numpy()).all():
        raise SystemExit("33列を付けたら行の並びが変わった")
    df = pd.concat([frame.reset_index(drop=True), att[list(BUILT_COLS)].reset_index(drop=True)], axis=1)
    log(f"[data] 33列を付けた / {time.time()-t0:.0f}秒（分足 {report['minute_from']} 〜 {report['minute_to']}・"
        f"{report['minute_days']}日）")
    print("\n■ 確かめ")
    checks = {"minute": minute_checks(df, report)}

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
             "missing": {c: int(df[c].isna().sum()) for c in BUILT_COLS}, "sources": src}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return df, stamp


def attached_mask(df: pd.DataFrame) -> pd.Series:
    return df[list(NEW_COLS)].notna().any(axis=1)


def coverage_report(df: pd.DataFrame, base: list) -> dict:
    """足す30列の欠測の割合と分位、既存の373列との順位相関の最大（件数と割合・相関だけ）。"""
    out = {"columns": {}, "overlap": {}}
    att = df[attached_mask(df)]
    k = len(NEW_COLS)
    print(f"\n■ 0. {k}列の付き方（{len(df):,}行のうち付いた行 {len(att):,}。分位は付いた行の 5% / 50% / 95%。"
          f"学習に足さない列: {' / '.join(DROPPED_COLS)}）")
    for c in NEW_COLS:
        v = pd.to_numeric(att[c], errors="coerce")
        qs = v.quantile([0.05, 0.5, 0.95]) if v.notna().any() else pd.Series([np.nan] * 3, index=[0.05, 0.5, 0.95])
        out["columns"][c] = {"missing_all": float(df[c].isna().mean()), "missing_attached": float(v.isna().mean()),
                             "q05": float(qs[0.05]), "q50": float(qs[0.5]), "q95": float(qs[0.95])}
        print(f"    {c:<36} 欠測 全行 {df[c].isna().mean()*100:5.1f}%・付いた行 {v.isna().mean()*100:5.1f}%  "
              f"{qs[0.05]:>9.3g} / {qs[0.5]:>9.3g} / {qs[0.95]:>9.3g}")
    if len(att) < 3:
        print("  付いた行が少なく、既存の列との重なりは出さない")
        return out
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
        print(f"  既存の373列のどれかと順位相関の絶対値が {th} 以上の列 {n_base} / {k}・{k}列どうしで {th} 以上の組 {n_pair}")
    print("  既存の列といちばん重なる列: " + " / ".join(f"{c}↔{best[c][0]} {best[c][1]:.2f}" for c in NEW_COLS))
    out["max_abs_rho_vs_base"] = {c: {"col": best[c][0], "rho": best[c][1]} for c in NEW_COLS}
    return out


def gain_report(df: pd.DataFrame, base: list, params: dict, cutoff, arm: str = "B") -> dict:
    """探索の期間の全行で B か C（373 + 30列・その腕のパラメータ）を1本学習し、寄与（gain）の割合と上位の列（参考）。"""
    import lightgbm as lgb

    cols = base + list(NEW_COLS)
    sub = df[(df["Date"] <= cutoff) & df["label"].notna()]
    y = sub["label"].to_numpy(dtype=int)
    m = lgb.LGBMClassifier(**{**params, "random_state": PROD_SEED}, scale_pos_weight=tuning.scale_pos_weight(y))
    m.fit(sub[cols].to_numpy(dtype=float), y)
    gain = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=cols)
    tot = float(gain.sum()) or 1.0
    share = {"既存の373列": float(gain[base].sum() / tot), f"分足の{len(NEW_COLS)}列": float(gain[list(NEW_COLS)].sum() / tot)}
    print(f"\n■ 0b. 学習の寄与（gain）の割合（探索の期間 〜{cutoff.date()} の {len(sub):,}行で {arm} を1本。参考）: "
          + " / ".join(f"{k} {v*100:.1f}%" for k, v in share.items()))
    all_rank = gain.rank(ascending=False)
    new_rank = gain[list(NEW_COLS)].sort_values(ascending=False)
    print(f"  {len(NEW_COLS)}列の寄与（全{len(cols)}列の中の順位。上位10列）: "
          + " / ".join(f"{c} {int(all_rank[c])}位" for c in new_rank.index[:10]))
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
    # 列の指紋も名前に入れる（探索の最良がはじめの無作為な試行だと、列が違っても同じパラメータになり得るため）
    files = [path(f"{ALGO}_c{COLS_OF[arm]}{F.signature(cols)}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    missing = [f for f in files if not os.path.exists(f)]
    if missing and not time_left():
        return None
    folds = E41.folds_for(frame["Date"], shift) if missing else None
    parts = []
    extra = frame[["Code", "Date", "hit10", "tp10c"]].assign(minute=attached_mask(frame).to_numpy())
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
    if rows == "minute":
        return o[o["minute"].eq(True)]                # 突き合わせに無い行（欠測）は付いていない側に数える
    raise KeyError(rows)


def windows_on(o: pd.DataFrame, rows: str = "all") -> pd.DataFrame:
    """窓ごとの AUC と、窓の期間・件数・正例率（MIN_WINDOW_ROWS 未満の窓は落とす）。rows で評価の行を選ぶ。"""
    cols = ["fold", "pr", "roc", "start", "end", "n", "pos"]
    o = rows_of(o, rows)
    n = o.groupby("fold").size()
    sub = o[o["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if not len(sub):
        return pd.DataFrame(columns=cols)
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


def train_minute_rows(frame: pd.DataFrame, shifts) -> dict:
    """
    窓（ずらし・窓の番号）ごとの、訓練に入る分足の付いた行の数。訓練は E41.oof_folds と同じく窓の train_end までの
    ラベルのある行（窓は oof と同じ E41.folds_for）。0 の窓では B の30列は訓練の全行で欠測になる。
    """
    d = pd.to_datetime(frame["Date"])
    att = attached_mask(frame) & frame["label"].notna()
    out = {}
    for sh in sorted(set(shifts) | {0}):
        for f in E41.folds_for(frame["Date"], sh):
            out[(sh, int(f.index))] = int((att & (d <= pd.Timestamp(f.train_end))).sum())
    return out


def run(arms: list, frames: dict, colsets: dict, cutoff, n_trials: int, stamp: dict, shifts: list) -> dict:
    """ARMS の順（A・B・C）に計算する。C は A の探索のパラメータを使う。時間切れの腕は結果に入らない。"""
    res = {"recs": {}, "params": {}, "prod": {}, "oofs": {}, "train_minute": train_minute_rows(frames["new"], shifts)}
    for arm in [a for a in ARMS if a in arms]:
        cs = COLS_OF[arm]
        src = PARAMS_OF[arm]
        if src not in res["recs"]:
            rec = tune(src, frames[COLS_OF[src]], colsets[COLS_OF[src]], cutoff, n_trials, stamp)
            if rec is None:
                continue
            res["recs"][src] = rec
        rec = res["recs"][src]
        res["params"][arm] = rec["params"]
        o = oof(arm, frames[cs], colsets[cs], rec["params"], 0, (PROD_SEED,))
        if o is not None:
            res["prod"][arm] = o
        for sh in shifts:
            o = oof(arm, frames[cs], colsets[cs], rec["params"], sh, SEEDS)
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
                        "pos": float(r["pos"]),
                        "train_minute": int(res.get("train_minute", {}).get((sh, int(r["fold"])), -1)),
                        **{f"{m}_{a}": float(r[f"{m}_{a}"]) for m in ("pr", "roc") for a in arms}})
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


def report(res: dict, shifts: list, stamp: dict, extra: dict) -> dict:
    arms = complete_arms(res, shifts)
    missing = [a for a in ARMS if a not in arms]
    print("\n" + "=" * 78)
    print("ラベル +10%（本番） / 結果のそろった腕: " + (", ".join(arms) or "なし") + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, **extra, "tuning": {}, "production_oof": {}, "windows": {}, "top": {}, "per_window": {}}

    print(f"\n■ 1. 探索の CV（{CV_SCHEME}・{N_SPLITS}分割。楽観側に出る。パラメータ選び用）")
    for arm in ARMS:
        rec = res["recs"].get(arm)
        if not rec:
            continue
        cv = rec["_cv"]
        fs = cv.get("fold_scores") or []
        print(f"  {arm:<4}{cv['mean_pr_auc']:>8.4f} ±{cv['std']:.4f}  ROC {cv['mean_roc_auc']:.4f}  {rec.get('_seconds', 0)}秒  "
              f"分割ごと " + " / ".join(f"{v:.4f}" for v in fs))
        print(f"      パラメータ {short(rec['params'])}")
        summary["tuning"][arm] = {k: cv.get(k) for k in ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
    if "C" in res["params"]:
        print("  C   探索しない（A のパラメータのまま30列を足す）")
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
                print(f"  {arm:<14}（正例・負例の片方しかない）")
                continue
            m = metrics(o)
            ms[arm] = m
            summary["production_oof"][f"{rows}_{arm}"] = {k: float(v) for k, v in m.items()}
            print(f"  {arm:<14}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
                  f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
                  f"{m['ret_o1_20_worst']:>+8.2f}pt{len(o):>8,}")
        for lo, hi in PAIRS:
            if lo in ms and hi in ms:
                a, b = ms[lo], ms[hi]
                print(f"  {hi + '−' + lo:<14}{b['pr_auc'] - a['pr_auc']:>+8.4f}{'':>7}{b['roc_auc'] - a['roc_auc']:>+8.4f}"
                      f"{b['day_auc'] - a['day_auc']:>+8.4f}{b['ret_o1_20_mean'] - a['ret_o1_20_mean']:>+10.2f}pt")

    tables = {}
    for rows in ROWSETS:
        s = window_table(res, arms, shifts, rows)
        tables[rows] = s
        s.to_csv(path(f"auc_by_window_{rows}.csv"), index=False)
        print(f"\n■ 3{'b' if rows == 'minute' else ''}. 窓ごと（{ROWSET_JA[rows]}。境界を {'/'.join(map(str, shifts))}か月ずらした"
              f"{len(shifts)}通り × 種3つの平均。窓は{ROWSET_JA[rows]}が {MIN_WINDOW_ROWS}件以上で正例・負例がそろうもの）: {len(s)}窓")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        pairs = [(lo, hi) for lo, hi in PAIRS if lo in arms and hi in arms]
        if not len(s) or not pairs:
            continue
        for lo, hi in pairs:
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                line, st = pair(f"{nm} {hi}−{lo}", s[f"{met}_{lo}"].to_numpy(), s[f"{met}_{hi}"].to_numpy())
                print(line)
                summary["windows"][f"{rows}_{met}_{hi}-{lo}"] = st
        t = s[s["train_minute"] > 0]
        u = s[s["train_minute"] == 0]
        print(f"  うち訓練に分足の付いた行がある窓（{len(t)}窓）と無い窓（{len(u)}窓。30列が訓練で全行欠測で B・C は30列を使えない。"
              "そこでの差はパラメータの違いだけで、C は A とちょうど同じになる）")
        for lo, hi in pairs:
            for part, key, nm2 in ((t, "trained", "訓練に分足"), (u, "untrained", "訓練に分足なし")):
                for met, nm in (("pr", "PR"), ("roc", "ROC")):
                    if not len(part):
                        break
                    line, st = pair(f"{nm} {hi}−{lo}（{nm2}）", part[f"{met}_{lo}"].to_numpy(), part[f"{met}_{hi}"].to_numpy())
                    print(line)
                    summary["windows"][f"{rows}_{met}_{hi}-{lo}_{key}"] = st
    print("\n■ 4. §7 の判定は出さない（運用者の選択。対照 P は回していない）")

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

    for rows in ROWSETS:
        s = tables[rows]
        print(f"\n■ 6{'b' if rows == 'minute' else ''}. 窓ごとの値（PR-AUC。{ROWSET_JA[rows]}。差は「B − A」「C − A」）")
        for sh in shifts:
            g = s[s["shift"] == sh].sort_values("fold") if len(s) else s
            if not len(g):
                continue
            print(f"  ずらし{sh}か月（{len(g)}窓）")
            diffs = [(lo, hi) for lo, hi in PAIRS if lo in arms and hi in arms]
            print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}{'訓練の分足':>9}" + "".join(f"{a:>9}" for a in arms)
                  + "".join(f"{hi + '−' + lo:>9}" for lo, hi in diffs))
            for _, r in g.iterrows():
                print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                      f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%{int(r['train_minute']):>11,}"
                      + "".join(f"{r[f'pr_{a}']:>9.4f}" for a in arms)
                      + "".join(f"{r[f'pr_{hi}'] - r[f'pr_{lo}']:>+9.4f}" for lo, hi in diffs))
        summary["per_window"][rows] = (s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
                                       if len(s) else [])
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE
    ap = argparse.ArgumentParser(description="実験132: 分足の価格帯別出来高の30列を373列に足す前後（lgbm +10%・検証だけ）")
    ap.add_argument("--arms", default=",".join(ARMS), help="この回に計算する腕（A は必ず含める）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--keep-frame", action="store_true",
                    help="保存済みの表を、分足の取得記録が変わっていてもそのまま使う（腕を足す回で前の回と同じ表で比べる）")
    ap.add_argument("--budget-min", type=float, default=290.0, help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="列の付き方だけ出して終わる")
    args = ap.parse_args(argv)
    TAG = args.tag
    RUN = TAG
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"知らない腕: {bad}（使えるもの {ARMS}）")
    if "A" not in arms:
        raise SystemExit("基準 A は外せません")
    if args.rebuild and args.keep_frame:
        raise SystemExit("--rebuild と --keep-frame は一緒に使えません")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    base = F.columns(BASE_PRESET)
    new = list(NEW_COLS)
    dup = sorted(set(base) & set(new))
    if dup:
        raise SystemExit(f"今の列と重なる: {dup}")
    colsets = {"base": base, "new": base + new}

    print("=" * 78)
    print(f"実験132 分足の価格帯別出来高の{len(new)}列を今の373列に足す前後（lgbm +10%・5分割 CV の探索 + OOF + 32窓）")
    print(f"  列: {BASE_PRESET} {len(base)}列 指紋 {F.signature(base)}"
          f"{'' if BASE_PRESET == F.DEFAULT_PRESET else '（いまの既定とは違う）'} / + {len(new)}列 → {len(base) + len(new)}列 "
          f"指紋 {F.signature(base + new)}")
    print(f"  足さない列: {' / '.join(DROPPED_COLS)}（日足だけで作れる。EDA の後の運用者の選択。表には付けて確かめる）")
    print("  分足: 予測日 D の当日の引けまで（1・5・20営業日）。2024-11-01 より前の行は欠測")
    for a in ARMS:
        print(f"  {LABELS[a]}{'' if a in arms else '（この回は計算しない）'}")
    print(f"  探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}・{CV_OBJECTIVE}・ラベル +10%）/ 窓のずらし {shifts}か月 / "
          f"種 {SEEDS} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    full, stamp = prepare(base, args.rebuild, args.keep_frame)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    label_check = check_label(full)
    df = full[full["label"].notna()].sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    extra = {"coverage": coverage_report(df, base), "label_check": label_check}
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    att = attached_mask(df)
    print(f"\n  この回の行 {len(df):,}（+10% の正例率 {df['label'].mean()*100:.1f}%・{df['Date'].min().date()} 〜 "
          f"{df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では {(df['Date'] <= cutoff).sum():,}行・"
          f"うち分足の付いた行 {int((att & (df['Date'] <= cutoff)).sum()):,}（全体では {int(att.sum()):,}）")
    if args.dry:
        return 0
    frames = {"base": df, "new": df}
    t0 = time.time()
    res = run(arms, frames, colsets, cutoff, args.n_trials, stamp, shifts)
    log(f"[{ALGO}] 計算 {time.time()-t0:.0f}秒 / そろった腕 {', '.join(complete_arms(res, shifts)) or 'なし'}")
    if res["recs"].get("B") and "B" in res["params"]:
        extra["gain"] = gain_report(df, base, res["params"]["B"], cutoff, "B")
    if "C" in res["params"]:
        extra["gain_C"] = gain_report(df, base, res["params"]["C"], cutoff, "C")
    report(res, shifts, stamp, extra)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
