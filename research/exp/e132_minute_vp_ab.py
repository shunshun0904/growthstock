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
短い期間の値動き:   args="--keep-frame --short"（分足の30列が、高値更新の翌日から数営業日の値動きを当てるか。今の373列の予測を
                    超えて当たりが残るか。運用者の選択 2026-11-19「短い期間の値動きで見る」。docs の 97-11）
残差で確かめる:     args="--keep-frame --residual"（保存済みの A の out-of-fold で説明しきれない部分を、分足の30列が説明するか。
                    運用者の選択 2026-11-19「残差で確かめる」。docs の 97-9）
重要度の顔ぶれ:     args="--keep-frame --importance"（A・C・B を同じ行・種で学習し直し、gain の割合の上位を比べる。
                    運用者の依頼 2026-11-19「腕Cにおいて、重要度の高い特徴量の中に今回追加した特徴量は含まれてますか？
                    重要度の高い上位の特徴量の顔ぶれがどれくらい変わったのか見たいです」）

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
# 重要度の顔ぶれ（--importance）
# --------------------------------------------------------------------------- #
#: 顔ぶれを比べる上位の列数
TOP_NS = (10, 20, 30, 50)
#: 表に並べる上位の列数
SHOW_TOP = 20


def importance_shares(df: pd.DataFrame, cols: list, params: dict, seed: int):
    """df（ラベルのある行）で1本学習し、列ごとの gain と split の割合（合計1）を返す。学習器は gain_report と同じ。"""
    import lightgbm as lgb

    y = df["label"].to_numpy(dtype=int)
    m = lgb.LGBMClassifier(**{**params, "random_state": seed}, scale_pos_weight=tuning.scale_pos_weight(y))
    m.fit(df[cols].to_numpy(dtype=float), y)
    out = []
    for kind in ("gain", "split"):
        v = pd.Series(m.booster_.feature_importance(importance_type=kind), index=cols, dtype=float)
        out.append(v / (float(v.sum()) or 1.0))
    return out[0], out[1]


def top_cols(share: pd.Series, n: int) -> list:
    """割合の大きい順の上位 n 列（同じ値は列名の順）。"""
    order = sorted(share.index, key=lambda c: (-float(share[c]), c))
    return order[:n]


def ranks_of(share: pd.Series) -> pd.Series:
    """割合の大きい順の順位（1 が最上位。同じ値は同じ順位）。"""
    return share.rank(ascending=False, method="min").astype(int)


def overlap(a: pd.Series, b: pd.Series, n: int) -> int:
    return len(set(top_cols(a, n)) & set(top_cols(b, n)))


def importance_report(df: pd.DataFrame, base: list, params_of: dict, cutoff, seeds=None) -> dict:
    """
    A（373列・A のパラメータ）・C（403列・A のパラメータ）・B（403列・B のパラメータ）を、同じ行と種で学習し直して
    gain の割合を比べる。期間は探索の期間（〜cutoff）と全期間（ラベルのある全行。本番の全期間の学習に近い）。

    - 分足の30列が上位に入るか（割合・いちばん上の順位・上位 20/50/100 に入る列の数）
    - 上位 N 列の顔ぶれの重なり（A と C）。偶然の入れ替わりの目安として、同じ腕で種だけ変えたときの重なりも出す
      （列の抜き取り colsample があるので、種や列の数が変わるだけでも木は変わる）
    - 既存の373列の割合の順位相関（A と C）と、同じ腕の種どうしの順位相関
    - 種3つの平均の割合で並べた上位 SHOW_TOP 列（A・C の順位を並べる）
    """
    seeds = tuple(seeds or SEEDS)
    new = list(NEW_COLS)
    groups = F.column_groups()
    arms = {"A": (list(base), params_of["A"]), "C": (list(base) + new, params_of["A"])}
    if params_of.get("B") is not None:
        arms["B"] = (list(base) + new, params_of["B"])
    periods = (("tuning", f"探索の期間（〜{pd.Timestamp(cutoff).date()}）", df[df["Date"] <= cutoff]),
               ("all", f"全期間（〜{df['Date'].max().date()}。本番の全期間の学習に近い）", df))
    out = {"seeds": list(seeds), "periods": {}}
    for key, title, sub in periods:
        sub = sub[sub["label"].notna()]
        att = attached_mask(sub)
        print(f"\n■ 重要度（gain の割合）: {title} {len(sub):,}行・うち分足の付いた行 {int(att.sum()):,}"
              f"（{att.mean() * 100:.0f}%）・種 {list(seeds)}。A・C は A のパラメータ、B は B のパラメータ")
        gain, split = {}, {}
        for arm, (cols, prm) in arms.items():
            for sd in seeds:
                gain[(arm, sd)], split[(arm, sd)] = importance_shares(sub, cols, prm, sd)
        mean = {arm: pd.concat([gain[(arm, sd)] for sd in seeds], axis=1).mean(axis=1) for arm in arms}
        rank = {arm: ranks_of(mean[arm]) for arm in arms}
        rec = {"rows": int(len(sub)), "minute_rows": int(att.sum()), "arms": {}}

        # 1. 分足の30列は上位に入るか
        print("  1. 分足の30列（種3つの平均の割合で順位を付ける。括弧は種ごとの値の幅）")
        for arm in [a for a in ("C", "B") if a in arms]:
            per_seed = [float(gain[(arm, sd)][new].sum()) for sd in seeds]
            per_seed_best = [int(ranks_of(gain[(arm, sd)])[new].min()) for sd in seeds]
            sp_share = float(np.mean([split[(arm, sd)][new].sum() for sd in seeds]))
            r = rank[arm][new]
            best = r.sort_values(kind="mergesort").index[0]
            cnt = {n: int((r <= n).sum()) for n in (20, 50, 100)}
            print(f"    {arm}: gain の割合 {np.mean(per_seed) * 100:.2f}%（{min(per_seed) * 100:.2f}〜{max(per_seed) * 100:.2f}%）・"
                  f"split の割合 {sp_share * 100:.2f}%・いちばん上 {best} {int(r[best])}位 / {len(arms[arm][0])}列"
                  f"（種ごと {'・'.join(str(b) for b in per_seed_best)}位）・上位20に {cnt[20]}列・50に {cnt[50]}列・"
                  f"100に {cnt[100]}列")
            rec["arms"][arm] = {"minute_gain_share": float(np.mean(per_seed)), "minute_gain_share_by_seed": per_seed,
                                "minute_split_share": sp_share, "best_col": best, "best_rank": int(r[best]),
                                "best_rank_by_seed": per_seed_best, "in_top": {str(n): v for n, v in cnt.items()},
                                "minute_ranks": {c: int(r[c]) for c in new}}

        # 2. 上位 N 列の顔ぶれの重なり
        print("  2. 上位 N 列の顔ぶれの重なり（同じ列の数 / N）")
        print(f"    {'':<34}" + "".join(f"{'上位' + str(n):>8}" for n in TOP_NS))
        lines = [("A と C（種3つの平均で並べた順位）", [overlap(mean["A"], mean["C"], n) for n in TOP_NS])]
        same = [[overlap(gain[("A", sd)], gain[("C", sd)], n) for sd in seeds] for n in TOP_NS]
        lines.append(("A と C（同じ種どうし・3組の平均）", [float(np.mean(v)) for v in same]))
        pairs = [(seeds[i], seeds[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
        for arm in ("A", "C"):
            v = [float(np.mean([overlap(gain[(arm, s1)], gain[(arm, s2)], n) for s1, s2 in pairs])) for n in TOP_NS]
            lines.append((f"{arm} の種どうし（{len(pairs)}組の平均。偶然の目安）", v))
        if "B" in arms:
            lines.append(("A と B（種3つの平均で並べた順位）", [overlap(mean["A"], mean["B"], n) for n in TOP_NS]))
        for name, vals in lines:
            print(f"    {name:<34}" + "".join(f"{v:>6.1f}/{n:<2}" if isinstance(v, float) else f"{v:>6}/{n:<2}"
                                             for v, n in zip(vals, TOP_NS)))
        rec["overlap"] = {name: {str(n): v for n, v in zip(TOP_NS, vals)} for name, vals in lines}

        # 3. 既存の373列の順位相関
        b = list(base)
        rho_ac = float(mean["A"][b].corr(mean["C"][b], method="spearman"))
        rho_same = float(np.mean([gain[("A", sd)][b].corr(gain[("C", sd)][b], method="spearman") for sd in seeds]))
        rho_seed = {arm: float(np.mean([gain[(arm, s1)][b].corr(gain[(arm, s2)][b], method="spearman")
                                        for s1, s2 in pairs])) for arm in ("A", "C")}
        print(f"  3. 既存の{len(b)}列の gain の割合の順位相関（スピアマン）: A と C（平均どうし）{rho_ac:.3f}・"
              f"A と C（同じ種どうし）{rho_same:.3f}・A の種どうし {rho_seed['A']:.3f}・C の種どうし {rho_seed['C']:.3f}")
        rec["spearman"] = {"A_C_mean": rho_ac, "A_C_same_seed": rho_same, "A_seeds": rho_seed["A"],
                           "C_seeds": rho_seed["C"]}

        # 4. 上位の列の表
        print(f"  4. 上位{SHOW_TOP}列（種3つの平均の gain の割合。順位の「→」は A → C）")
        print(f"    {'順位':>4}  {'A の上位':<34}{'割合':>7}{'C での順位':>9}   {'C の上位':<34}{'割合':>7}{'A での順位':>9}")
        ta, tc = top_cols(mean["A"], SHOW_TOP), top_cols(mean["C"], SHOW_TOP)
        for i in range(SHOW_TOP):
            ca, cc = ta[i], tc[i]
            ra_c = f"{int(rank['C'][ca])}位"
            rc_a = f"{int(rank['A'][cc])}位" if cc in rank["A"].index else "（新）"
            print(f"    {i + 1:>4}  {ca:<34}{mean['A'][ca] * 100:>6.2f}%{ra_c:>9}   {cc:<34}{mean['C'][cc] * 100:>6.2f}%"
                  f"{rc_a:>9}")
        left = [c for c in ta if c not in tc]
        came = [c for c in tc if c not in ta]
        print(f"    上位{SHOW_TOP}から外れた列: " + (" / ".join(f"{c}（{int(rank['A'][c])} → {int(rank['C'][c])}位）"
                                                    for c in left) or "なし"))
        print(f"    上位{SHOW_TOP}に入った列: " + (" / ".join(f"{c}（{int(rank['A'][c]) if c in rank['A'].index else '新'} → "
                                                    f"{int(rank['C'][c])}位）" for c in came) or "なし"))
        grp = {arm: mean[arm].groupby(lambda c: groups.get(c) or ("vp_minute" if c in new else "?")).sum()
               for arm in ("A", "C")}
        topg = grp["A"].sort_values(ascending=False).index[:8]
        print("    グループごとの割合（A → C。上位8グループ）: "
              + " / ".join(f"{g} {grp['A'][g] * 100:.1f} → {grp['C'].get(g, 0.0) * 100:.1f}%" for g in topg)
              + f" / vp_minute 0.0 → {grp['C'].get('vp_minute', 0.0) * 100:.1f}%")
        rec["top"] = {arm: [{"col": c, "group": groups.get(c) or ("vp_minute" if c in new else "?"),
                             "share": float(mean[arm][c]), "rank": int(rank[arm][c])}
                            for c in top_cols(mean[arm], max(TOP_NS))] for arm in arms}
        rec["ranks_in_other"] = {"A_top_in_C": {c: int(rank["C"][c]) for c in ta},
                                 "C_top_in_A": {c: (int(rank["A"][c]) if c in rank["A"].index else None) for c in tc}}
        rec["groups"] = {arm: {g: float(v) for g, v in grp[arm].items()} for arm in grp}
        out["periods"][key] = rec
    with open(path("importance.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    return out


# --------------------------------------------------------------------------- #
# 残差で確かめる（--residual）
# --------------------------------------------------------------------------- #
#: 並べ替え検定の回数（列ごと・30列まとめて）
N_PERM = 1000
N_PERM_JOINT = 500
#: 時間で分けた検証で、30列の係数に掛ける L2 の重さ
RIDGE = 1.0
#: 列ごとの判定の FDR（Benjamini–Hochberg）
BH_Q = 0.10
#: 30列まとめての判定の有意水準
ALPHA = 0.05


def logit_fit(X: np.ndarray, y: np.ndarray, beta=None, ridge=None, iters: int = 50, tol: float = 1e-9):
    """ロジスティック回帰（ニュートン法）。ridge は係数ごとの L2 の重さ（None なら無し）。(係数, 対数尤度) を返す。"""
    k = X.shape[1]
    b = np.zeros(k) if beta is None else np.asarray(beta, dtype=float).copy()
    lam = np.zeros(k) if ridge is None else np.asarray(ridge, dtype=float)
    for _ in range(iters):
        eta = np.clip(X @ b, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)
        g = X.T @ (y - p) - lam * b
        H = (X * w[:, None]).T @ X + np.diag(lam) + 1e-10 * np.eye(k)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ b
    return b, float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def normal_scores(v) -> np.ndarray:
    """順位を標準正規の分位に直した値（外れ値に引っ張られない）。"""
    from scipy.special import ndtri

    r = pd.Series(np.asarray(v, dtype=float)).rank(method="average").to_numpy()
    return ndtri((r - 0.5) / len(r))


def within_date(v: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """日付ごとの平均を引いた値（同じ日の候補どうしの違いだけを残す）。"""
    mean = pd.Series(v).groupby(codes).transform("mean").to_numpy()
    return v - mean


def date_permuter(codes: np.ndarray, rng):
    """同じ日付の行の中だけで並べ替える添字を返す関数（日付ごとの分布を保ったまま、ラベルとの結びつきを切る）。"""
    n = len(codes)
    base = np.lexsort((np.arange(n), codes))

    def perm() -> np.ndarray:
        shuffled = np.lexsort((rng.random(n), codes))
        out = np.empty(n, dtype=int)
        out[base] = shuffled
        return out
    return perm


def bh_qvalues(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        run = min(run, p[i] * n / rank)
        q[i] = run
    return q


def base_design(score: np.ndarray) -> np.ndarray:
    """A の予測の logit（標準化）とその2乗（A の予測の目盛りのずれを吸収する）。"""
    s = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(s / (1 - s))
    z = (z - z.mean()) / (z.std() or 1.0)
    return np.column_stack([np.ones(len(z)), z, z * z])


def added_ll(X0: np.ndarray, y: np.ndarray, extra: np.ndarray, b0: np.ndarray, ll0: float) -> float:
    if extra.ndim == 1:
        extra = extra[:, None]
    _, ll1 = logit_fit(np.column_stack([X0, extra]), y, beta=np.concatenate([b0, np.zeros(extra.shape[1])]))
    return ll1 - ll0


def residual_rows(df: pd.DataFrame, base: list, params_a: dict, shifts) -> pd.DataFrame:
    """
    A の out-of-fold（保存済み。ずらし × 種3つ）を行ごとに平均した予測と、分足の30列・ラベル・日付の表（分足の付いた行だけ）。
    どの予測も、その行を学習に使っていないモデルが出したもの。
    """
    parts = []
    for sh in shifts:
        o = oof("A", df, base, params_a, sh, SEEDS)
        if o is None:
            raise SystemExit(f"A の out-of-fold（ずらし{sh}）が無い")
        parts.append(o[["Code", "Date", "score"]].rename(columns={"score": f"s{sh}"}))
    m = parts[0]
    for x in parts[1:]:
        m = m.merge(x, on=["Code", "Date"], how="outer")
    sc = [c for c in m.columns if c.startswith("s") and c[1:].isdigit()]
    m["score"] = m[sc].mean(axis=1)
    m["n_scores"] = m[sc].notna().sum(axis=1)
    keep = df.loc[attached_mask(df), ["Code", "Date", "label"] + list(NEW_COLS)]
    out = keep.merge(m[["Code", "Date", "score", "n_scores"]], on=["Code", "Date"], how="inner")
    return out[out["label"].notna() & out["score"].notna()].reset_index(drop=True)


def residual_report(rows: pd.DataFrame, cols: list, n_perm: int = N_PERM, n_perm_joint: int = N_PERM_JOINT,
                    seed: int = 0) -> dict:
    """
    A の予測で説明しきれない部分を、列が説明するか（分足の付いた行で）。

    基準: ラベル ~ A の予測の logit（標準化）＋ その2乗 のロジスティック回帰。
    列ごと: 基準に列（順位を標準正規に直した値）を足したときの対数尤度の伸び（ΔLL）。並べ替え検定は、同じ日付の行の中で
            列の値を入れ替えて n_perm 回（日付ごとの偏りでは有意にならない）。「日付の中」は列から日付ごとの平均を引いた値で
            同じことをする（同じ日の候補の並べ方に効くか）。q は Benjamini–Hochberg
    まとめて: 30列（すべてそろう行）を一度に足した ΔLL と、行ごとに30列まとめて日付の中で入れ替えた並べ替え検定。
              日付で前半・後半に分け、片方で学習してもう片方で A だけ と A＋30列 を比べる（両方向）
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    rows = rows.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    y_all = rows["label"].to_numpy(dtype=float)
    out = {"rows": int(len(rows)), "pos": int(y_all.sum()), "dates": int(rows["Date"].nunique()),
           "a_auc": float(roc_auc_score(y_all, rows["score"])), "a_pr_auc": float(average_precision_score(y_all, rows["score"])),
           "columns": {}}
    print(f"\n■ 残差で確かめる: 分足の付いた行 {len(rows):,}（正例 {out['pos']:,}・{out['dates']}日）。A の予測は保存済みの "
          f"out-of-fold（行ごとに {int(rows['n_scores'].min())}〜{int(rows['n_scores'].max())}本の平均）で、この行での "
          f"ROC {out['a_auc']:.4f}・PR-AUC {out['a_pr_auc']:.4f}")
    print(f"  列ごと（並べ替え {n_perm}回・同じ日付の中で入れ替え。q は BH。ΔLL は対数尤度の伸び。"
          "残差は「実際の正例率 − A から見込む正例率」を列の5分位ごとに（ポイント。低い → 高い））")
    print(f"    {'列':<34}{'行':>6}{'単独AUC':>8}{'向き':>4}{'ΔLL':>7}{'p':>7}{'q':>6}  {'日付の中 ΔLL':>10}{'p':>7}{'q':>6}"
          f"  5分位の残差")
    res = []
    for c in cols:
        m = rows[c].notna().to_numpy()
        sub = rows[m]
        y = sub["label"].to_numpy(dtype=float)
        codes = pd.factorize(sub["Date"])[0]
        X0 = base_design(sub["score"].to_numpy())
        b0, ll0 = logit_fit(X0, y)
        p0 = 1.0 / (1.0 + np.exp(-(X0 @ b0)))
        r = normal_scores(sub[c].to_numpy())
        rw = within_date(r, codes)
        perm = date_permuter(codes, rng)
        stats = {}
        for key, v in (("pooled", r), ("within", rw)):
            d = added_ll(X0, y, v, b0, ll0)
            null = np.array([added_ll(X0, y, v[perm()], b0, ll0) for _ in range(n_perm)])
            b1, _ = logit_fit(np.column_stack([X0, v]), y, beta=np.append(b0, 0.0))
            stats[key] = {"dll": float(d), "p": float((1 + np.sum(null >= d)) / (n_perm + 1)), "coef": float(b1[-1])}
        bins = pd.qcut(pd.Series(r).rank(method="first"), 5, labels=False).to_numpy()
        quint = [float(np.mean((y - p0)[bins == k]) * 100) for k in range(5)]
        auc = float(roc_auc_score(y, sub[c].to_numpy(dtype=float))) if 0 < y.sum() < len(y) else float("nan")
        res.append({"col": c, "n": int(m.sum()), "auc": auc, "pooled": stats["pooled"], "within": stats["within"],
                    "quintile_resid_pt": quint})
    for key in ("pooled", "within"):
        q = bh_qvalues([r_[key]["p"] for r_ in res])
        for r_, qq in zip(res, q):
            r_[key]["q"] = float(qq)
    for r_ in sorted(res, key=lambda x: -x["pooled"]["dll"]):
        po, wi = r_["pooled"], r_["within"]
        print(f"    {r_['col']:<34}{r_['n']:>6,}{r_['auc']:>8.3f}{'+' if po['coef'] > 0 else '−':>4}{po['dll']:>7.2f}"
              f"{po['p']:>7.3f}{po['q']:>6.2f}  {wi['dll']:>10.2f}{wi['p']:>7.3f}{wi['q']:>6.2f}  "
              + " / ".join(f"{v:+.1f}" for v in r_["quintile_resid_pt"]))
        out["columns"][r_["col"]] = r_
    hit = {key: [r_["col"] for r_ in res if r_[key]["q"] < BH_Q] for key in ("pooled", "within")}
    print(f"  q < {BH_Q} の列: 全体 {len(hit['pooled'])}列{('（' + ' / '.join(hit['pooled']) + '）') if hit['pooled'] else ''}・"
          f"日付の中 {len(hit['within'])}列{('（' + ' / '.join(hit['within']) + '）') if hit['within'] else ''}")
    out["hits"] = hit

    # 30列まとめて
    full = rows.dropna(subset=list(cols)).reset_index(drop=True)
    y = full["label"].to_numpy(dtype=float)
    codes = pd.factorize(full["Date"])[0]
    X0 = base_design(full["score"].to_numpy())
    b0, ll0 = logit_fit(X0, y)
    R = np.column_stack([normal_scores(full[c].to_numpy()) for c in cols])
    d = added_ll(X0, y, R, b0, ll0)
    perm = date_permuter(codes, rng)
    null = np.array([added_ll(X0, y, R[perm()], b0, ll0) for _ in range(n_perm_joint)])
    p_joint = float((1 + np.sum(null >= d)) / (n_perm_joint + 1))
    print(f"  {len(cols)}列まとめて（{len(full):,}行。並べ替え {n_perm_joint}回）: ΔLL {d:.2f}・並べ替えの 95% 点 "
          f"{np.quantile(null, 0.95):.2f}・p {p_joint:.3f}")
    dates = np.sort(full["Date"].unique())
    cut = dates[len(dates) // 2]
    halves = {"前半で学習 → 後半で評価": (full["Date"] < cut, full["Date"] >= cut),
              "後半で学習 → 前半で評価": (full["Date"] >= cut, full["Date"] < cut)}
    split = {}
    for name, (tr, te) in halves.items():
        tr, te = tr.to_numpy(), te.to_numpy()
        Xb = base_design(full["score"].to_numpy())
        # 標準化は学習側の値で決める（評価側の分布を使わない）
        Rtr = R[tr]
        mu, sd = Rtr.mean(axis=0), Rtr.std(axis=0)
        sd[sd == 0] = 1.0
        Rz = (R - mu) / sd
        bb, _ = logit_fit(Xb[tr], y[tr])
        lam = np.concatenate([np.zeros(Xb.shape[1]), np.full(len(cols), RIDGE)])
        bf, _ = logit_fit(np.column_stack([Xb, Rz])[tr], y[tr], ridge=lam)
        pb = 1 / (1 + np.exp(-(Xb[te] @ bb)))
        pf = 1 / (1 + np.exp(-(np.column_stack([Xb, Rz])[te] @ bf)))
        yt = y[te]

        def ll(pp):
            pp = np.clip(pp, 1e-9, 1 - 1e-9)
            return float(-np.mean(yt * np.log(pp) + (1 - yt) * np.log(1 - pp)))
        rec = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
               "auc_a": float(roc_auc_score(yt, pb)), "auc_full": float(roc_auc_score(yt, pf)),
               "pr_a": float(average_precision_score(yt, pb)), "pr_full": float(average_precision_score(yt, pf)),
               "logloss_a": ll(pb), "logloss_full": ll(pf)}
        split[name] = rec
        print(f"  {name}（学習 {rec['n_train']:,}行・評価 {rec['n_test']:,}行。分かれ目 {pd.Timestamp(cut).date()}）: "
              f"ROC A だけ {rec['auc_a']:.4f} → A＋{len(cols)}列 {rec['auc_full']:.4f}（{rec['auc_full'] - rec['auc_a']:+.4f}）・"
              f"PR-AUC {rec['pr_a']:.4f} → {rec['pr_full']:.4f}（{rec['pr_full'] - rec['pr_a']:+.4f}）・"
              f"対数損失 {rec['logloss_a']:.4f} → {rec['logloss_full']:.4f}")
    both_up = all(v["auc_full"] > v["auc_a"] for v in split.values())
    if p_joint < ALPHA and both_up:
        reading = "情報が残っている（A が使い切れていない情報が30列にある）"
    elif p_joint >= ALPHA:
        reading = "残っていない（A の予測で説明できない部分を、30列はまとめても説明しない）"
    else:
        reading = "はっきりしない（まとめての検定は有意だが、時間で分けた検証の両方では上がらない）"
    print(f"  読み（回す前に決めたもの。まとめての p < {ALPHA} かつ 時間で分けた両方向で ROC が上がれば「残っている」、"
          f"p ≥ {ALPHA} なら「残っていない」）: {reading}")
    out["joint"] = {"rows": int(len(full)), "dll": float(d), "p": p_joint, "null_q95": float(np.quantile(null, 0.95)),
                    "split": split, "reading": reading}
    return out


# --------------------------------------------------------------------------- #
# 短い期間の値動き（--short）
# --------------------------------------------------------------------------- #
#: 当てる値動き（株価は分割調整後。買いは本番と同じ D+1 の寄り。gap だけは買う前の窓）
SHORT_TARGETS = (("gap", "翌日の寄りの窓（D の終値 → D+1 の寄り）"),
                 ("r1", "1日目（D+1 の寄り → D+1 の引け）"),
                 ("r3", "3日目まで（D+1 の寄り → D+3 の引け）"),
                 ("r5", "5日目まで（D+1 の寄り → D+5 の引け）"))
#: 日付ごとの相関を取る最少の行数
MIN_DATE_ROWS = 5
#: 30列まとめた信号の判定に使う t（4つの値動きを見るので 2.5）
SHORT_T = 2.5
#: 30列まとめた信号のリッジの重さ
SHORT_RIDGE = 10.0


def short_returns(keys: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """keys（Code, Date）ごとの短い期間の値動き（gap・r1・r3・r5）。日足の並び（取引のあった日）で数える。"""
    bars = LT.load_bars(data_dir, start=pd.to_datetime(keys["Date"]).min() - pd.Timedelta(days=10))
    b = bars.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)
    g = b.groupby("Code", sort=False)
    o1 = g["AdjO"].shift(-1)
    out = b[["Code", "Date"]].copy()
    out["gap"] = o1 / b["AdjC"] - 1.0
    for h in (1, 3, 5):
        out[f"r{h}"] = g["AdjC"].shift(-h) / o1 - 1.0
    k = keys[["Code", "Date"]].copy()
    k["Date"] = pd.to_datetime(k["Date"])
    k["Code"] = k["Code"].astype(str)
    return k.merge(out, on=["Code", "Date"], how="left")


def date_scores(v, codes: np.ndarray) -> np.ndarray:
    """日付ごとの順位を標準正規の分位に直した値（欠測は欠測のまま。その日に1行だけなら 0）。"""
    from scipy.special import ndtri

    x = pd.Series(np.asarray(v, dtype=float))
    g = x.groupby(codes)
    r = g.rank(method="average")
    n = g.transform("count")
    return ndtri(((r - 0.5) / n).to_numpy())


def date_ic(x, y, codes: np.ndarray, min_rows: int = MIN_DATE_ROWS, control=None) -> dict:
    """
    日付ごとの順位相関（日付の中で順位を標準正規に直した値どうしの相関）の平均・SE・t・p（日付を1件として数える）。
    control を渡すと、x と y からそれぞれ control（同じく日付の中の順位）で説明できる分を引いた残りどうしの相関（偏相関）。
    """
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if control is not None:
        m &= np.isfinite(np.asarray(control, dtype=float))
    c = codes[m]
    xs, ys = date_scores(x[m], c), date_scores(y[m], c)
    if control is not None:
        zs = date_scores(np.asarray(control, dtype=float)[m], c)
        zz = float(zs @ zs) or 1.0
        xs, ys = xs - (xs @ zs) / zz * zs, ys - (ys @ zs) / zz * zs
    d = pd.DataFrame({"c": c, "x": xs, "y": ys})
    d["x"] -= d.groupby("c")["x"].transform("mean")
    d["y"] -= d.groupby("c")["y"].transform("mean")
    d["xy"], d["xx"], d["yy"] = d["x"] * d["y"], d["x"] ** 2, d["y"] ** 2
    a = d.groupby("c").agg(n=("x", "size"), xy=("xy", "sum"), xx=("xx", "sum"), yy=("yy", "sum"))
    a = a[(a["n"] >= min_rows) & (a["xx"] > 0) & (a["yy"] > 0)]
    ic = (a["xy"] / np.sqrt(a["xx"] * a["yy"])).to_numpy()
    if len(ic) < 3:
        return {"ic": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan"), "dates": int(len(ic)),
                "by_date": ic}
    se = float(ic.std(ddof=1) / np.sqrt(len(ic)))
    t = float(ic.mean() / se) if se > 0 else float("nan")
    return {"ic": float(ic.mean()), "se": se, "t": t, "p": float(2 * stats.t.sf(abs(t), len(ic) - 1)),
            "dates": int(len(ic)), "by_date": ic, "dates_index": a.index.to_numpy()}


def short_oof(df: pd.DataFrame, base: list, params: dict, target: np.ndarray, rows_mask: np.ndarray) -> np.ndarray:
    """
    今の列（base）で target（日付の中の順位を標準正規に直した値）を当てる LightGBM の回帰の out-of-fold。
    窓は本番と同じ作り（E41.folds_for のずらし0）。rows_mask の行がテストに入る窓だけ学習する。パラメータは A の探索のもの
    （+10% の分類で探したものを流用。この目的では探索しない）。
    """
    import lightgbm as lgb

    d = pd.to_datetime(df["Date"])
    pred = np.full(len(df), np.nan)
    ok = np.isfinite(target)
    X = df[base].to_numpy(dtype=float)
    for f in E41.folds_for(df["Date"], 0):
        te = ((d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))).to_numpy()
        if not (te & rows_mask).any():
            continue
        tr = (d <= pd.Timestamp(f.train_end)).to_numpy() & ok
        m = lgb.LGBMRegressor(**{**params, "random_state": PROD_SEED})
        m.fit(X[tr], target[tr])
        pred[te] = m.predict(X[te])
    return pred


def short_report(rows: pd.DataFrame, cols: list, targets=SHORT_TARGETS, pred_cols: dict | None = None) -> dict:
    """
    rows: Date・値動き（targets の列）・列（cols）・今の列の予測（pred_cols[値動き] の列。無ければ偏相関は出さない）。
    1. 列ごと・値動きごとの日付の中の順位相関（IC。日付を1件として t）。q は全部（列 × 値動き）で BH
    2. 今の列の予測で説明できる分を引いた偏相関（今の列を超えて当たりが残るか）
    3. 30列まとめた信号（リッジ）を日付の前半で作って後半で測る（両方向）。偏相関の側は、今の列の予測に30列を足した信号と
       予測だけの IC の差（日付ごとの差の平均と t）
    """
    rows = rows.sort_values(["Date"], kind="mergesort").reset_index(drop=True)
    codes = pd.factorize(rows["Date"])[0]
    pred_cols = pred_cols or {}
    out = {"rows": int(len(rows)), "dates": int(rows["Date"].nunique()), "targets": {}, "columns": {}, "joint": {}}
    print(f"\n■ 短い期間の値動き: 分足の付いた行 {len(rows):,}・{out['dates']}日（相関は日付ごとに {MIN_DATE_ROWS}行以上の日で取り、"
          "日付を1件として t を出す。順位相関は日付の中の順位どうし）")
    for key, title in targets:
        v = rows[key].to_numpy(dtype=float)
        q = np.nanquantile(v, [0.05, 0.5, 0.95])
        rec = {"title": title, "n": int(np.isfinite(v).sum()), "q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2])}
        if key in pred_cols:
            r = date_ic(rows[pred_cols[key]].to_numpy(dtype=float), v, codes)
            rec["pred_ic"] = {k: r[k] for k in ("ic", "se", "t", "dates")}
        out["targets"][key] = rec
        print(f"  {key}: {title}。{rec['n']:,}行・中央 {q[1] * 100:+.2f}%（5% {q[0] * 100:+.2f}% / 95% {q[2] * 100:+.2f}%）"
              + (f"・今の373列の予測の IC {rec['pred_ic']['ic']:+.4f}（t {rec['pred_ic']['t']:+.2f}・{rec['pred_ic']['dates']}日）"
                 if "pred_ic" in rec else ""))

    # 1・2. 列ごと
    raw, part = {}, {}
    for c in cols:
        for key, _ in targets:
            raw[(c, key)] = date_ic(rows[c].to_numpy(dtype=float), rows[key].to_numpy(dtype=float), codes)
            if key in pred_cols:
                part[(c, key)] = date_ic(rows[c].to_numpy(dtype=float), rows[key].to_numpy(dtype=float), codes,
                                         control=rows[pred_cols[key]].to_numpy(dtype=float))
    for name, dct in (("raw", raw), ("partial", part)):
        if not dct:
            continue
        ks = list(dct)
        qs = bh_qvalues([dct[k]["p"] if np.isfinite(dct[k]["p"]) else 1.0 for k in ks])
        for k, qq in zip(ks, qs):
            dct[k]["q"] = float(qq)
    for name, dct, title in (("raw", raw, "1. 列ごとの IC（分足の列だけ。セルは IC と t。* は q < 0.10）"),
                             ("partial", part, "2. 今の373列の予測を引いた偏相関（今の列を超えて残るか。セルは IC と t。* は q < 0.10）")):
        if not dct:
            continue
        print(f"  {title}")
        print(f"    {'列':<34}" + "".join(f"{key:>16}" for key, _ in targets))
        order = sorted(cols, key=lambda c: -max(abs(dct[(c, k)]["t"]) if np.isfinite(dct[(c, k)]["t"]) else 0
                                                  for k, _ in targets))
        for c in order:
            cells = []
            for key, _ in targets:
                r = dct[(c, key)]
                star = "*" if r.get("q", 1.0) < BH_Q else " "
                cells.append(f"{r['ic']:+.3f}({r['t']:+.1f}){star}" if np.isfinite(r["ic"]) else "—")
            print(f"    {c:<34}" + "".join(f"{x:>16}" for x in cells))
        hits = [(c, k) for (c, k) in dct if dct[(c, k)].get("q", 1.0) < BH_Q]
        print(f"    q < {BH_Q}: {len(hits)}組" + (("（" + " / ".join(f"{c}×{k}" for c, k in hits) + "）") if hits else ""))
        for (c, k), r in dct.items():
            out["columns"].setdefault(c, {}).setdefault(name, {})[k] = {kk: r[kk] for kk in ("ic", "se", "t", "p", "q", "dates")}

    # 3. 30列まとめた信号
    X = np.column_stack([np.nan_to_num(date_scores(rows[c].to_numpy(dtype=float), codes)) for c in cols])
    dates = np.sort(rows["Date"].unique())
    cut = dates[len(dates) // 2]
    first = (rows["Date"] < cut).to_numpy()
    print(f"  3. {len(cols)}列まとめた信号（リッジ・重さ {SHORT_RIDGE:g}）。日付で前半・後半に分け、片方で作ってもう片方で測る"
          f"（分かれ目 {pd.Timestamp(cut).date()}）")
    for key, _ in targets:
        y = date_scores(rows[key].to_numpy(dtype=float), codes)
        has_pred = key in pred_cols
        P = date_scores(rows[pred_cols[key]].to_numpy(dtype=float), codes) if has_pred else None
        res = {}
        for name, tr in (("前半 → 後半", first), ("後半 → 前半", ~first)):
            te = ~tr
            ok_tr = tr & np.isfinite(y)
            beta = np.linalg.solve(X[ok_tr].T @ X[ok_tr] + SHORT_RIDGE * np.eye(X.shape[1]), X[ok_tr].T @ y[ok_tr])
            sig = X @ beta
            r_raw = date_ic(np.where(te, sig, np.nan), np.where(te, rows[key].to_numpy(dtype=float), np.nan), codes)
            rec = {"raw": {k: r_raw[k] for k in ("ic", "se", "t", "dates")}}
            if has_pred:
                ok2 = ok_tr & np.isfinite(P)
                Z = np.column_stack([P, X])
                b2 = np.linalg.solve(Z[ok2].T @ Z[ok2] + np.diag([0.0] + [SHORT_RIDGE] * X.shape[1]), Z[ok2].T @ y[ok2])
                sig2 = np.column_stack([np.nan_to_num(P), X]) @ b2
                yy = np.where(te & np.isfinite(P), rows[key].to_numpy(dtype=float), np.nan)
                a = date_ic(np.where(te, rows[pred_cols[key]].to_numpy(dtype=float), np.nan), yy, codes)
                b = date_ic(np.where(te & np.isfinite(P), sig2, np.nan), yy, codes)
                both = np.intersect1d(a.get("dates_index", []), b.get("dates_index", []))
                da = pd.Series(a["by_date"], index=a.get("dates_index", [])).reindex(both)
                db = pd.Series(b["by_date"], index=b.get("dates_index", [])).reindex(both)
                diff = (db - da).to_numpy()
                se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 2 else float("nan")
                rec["added"] = {"ic_pred": a["ic"], "ic_pred_plus": b["ic"], "diff": float(diff.mean()) if len(diff) else float("nan"),
                                "se": se, "t": float(diff.mean() / se) if se and np.isfinite(se) and se > 0 else float("nan"),
                                "dates": int(len(diff))}
            res[name] = rec
        out["joint"][key] = res
        line = f"    {key}: 分足だけ " + " / ".join(f"{n} IC {r['raw']['ic']:+.4f}（t {r['raw']['t']:+.2f}）" for n, r in res.items())
        if has_pred:
            line += "｜今の列に足した差 " + " / ".join(
                f"{n} {r['added']['ic_pred']:+.4f} → {r['added']['ic_pred_plus']:+.4f}（差 {r['added']['diff']:+.4f}・t {r['added']['t']:+.2f}）"
                for n, r in res.items())
        print(line)
    works = [k for k, _ in targets if all(r["raw"]["t"] >= SHORT_T for r in out["joint"][k].values())]
    beyond = [k for k, _ in targets if k in pred_cols and all(r["added"]["t"] >= SHORT_T for r in out["joint"][k].values())]
    reading = ("分足の列は短い期間の値動きに効く（" + "・".join(works) + "）" if works else "分足の列だけで短い期間の値動きを当てるとは言えない")
    reading += "。" + ("今の列を超えて効く（" + "・".join(beyond) + "）" if beyond else "今の列を超えて効くとは言えない")
    print(f"  読み（回す前に決めたもの。30列まとめた信号の IC が両方向で t ≥ {SHORT_T} なら「効く」、今の列の予測に足した差が"
          f"両方向で t ≥ {SHORT_T} なら「今の列を超えて効く」）: {reading}")
    out["reading"] = {"works": works, "beyond": beyond, "text": reading}
    return out


def short_rows(df: pd.DataFrame, base: list, params_a: dict) -> tuple:
    """分足の付いた行に、短い期間の値動きと、今の列で当てた予測（値動きごと）を付けた表。"""
    ret = short_returns(df[["Code", "Date"]], lab.DATA_DIR)
    if not (ret["Code"].to_numpy() == df["Code"].astype(str).to_numpy()).all():
        raise SystemExit("短い期間の値動きを付けたら行の並びが変わった")
    full = df.reset_index(drop=True).copy()
    for key, _ in SHORT_TARGETS:
        full[key] = ret[key].to_numpy()
    check = None
    if "entry_gap" in full.columns:
        both = full[["gap", "entry_gap"]].dropna()
        check = {"rows": int(len(both)), "max_abs_diff": float((both["gap"] - both["entry_gap"]).abs().max()) if len(both) else None}
        diff = "—" if check["max_abs_diff"] is None else f"{check['max_abs_diff']:.2e}"
        print(f"  確かめ: gap と学習データの entry_gap の差の最大 {diff}（{check['rows']:,}行）")
    codes_all = pd.factorize(full["Date"])[0]
    att = attached_mask(full).to_numpy()
    pred_cols = {}
    t0 = time.time()
    for key, _ in SHORT_TARGETS:
        target = date_scores(full[key].to_numpy(dtype=float), codes_all)
        full[f"pred_{key}"] = short_oof(full, base, params_a, target, att)
        pred_cols[key] = f"pred_{key}"
    log(f"[short] 今の373列の回帰（値動き {len(SHORT_TARGETS)}つ・本番と同じ窓）/ {time.time() - t0:.0f}秒")
    rows = full[att].reset_index(drop=True)
    return rows, pred_cols, check


# --------------------------------------------------------------------------- #
# 探索（週次の再学習と同じ関数・条件）
# --------------------------------------------------------------------------- #
def load_tuned(arm: str, cols: list, cutoff, n_trials: int, stamp: dict):
    """保存済みの探索の結果（試行の数・探索の期間・表の版・列の指紋がそろうものだけ）。無ければ None。"""
    p = path(f"params_{arm}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if (rec.get("_n_trials") == n_trials and rec.get("_cutoff") == str(cutoff.date())
                and rec.get("_data") == stamp["built_utc"] and rec.get("_sig") == F.signature(cols)):
            return rec
    return None


def tune(arm: str, frame: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict):
    p = path(f"params_{arm}.json")
    rec = load_tuned(arm, cols, cutoff, n_trials, stamp)
    if rec is not None:
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
    ap.add_argument("--short", action="store_true",
                    help="分足の30列が、高値更新の翌日から数営業日の値動きを当てるか（今の列の予測を超えて残るか）を調べて終わる")
    ap.add_argument("--residual", action="store_true",
                    help="保存済みの A の out-of-fold で説明しきれない部分を、分足の30列が説明するかを調べて終わる")
    ap.add_argument("--importance", action="store_true",
                    help="保存済みの探索のパラメータで A・C・B を学習し直し、重要度（gain の割合）の上位の顔ぶれを比べて終わる")
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
    if args.short:
        rec = load_tuned("A", colsets["base"], cutoff, args.n_trials, stamp)
        if rec is None:
            raise SystemExit("A の保存済みの探索の結果が無い（--short は探索し直さない。--keep-frame で前の回の表を使う）")
        rows, pred_cols, check = short_rows(df, base, rec["params"])
        out = short_report(rows, list(NEW_COLS), pred_cols=pred_cols)
        out["gap_check"] = check
        with open(path("short.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        log(f"記録: {path('short.json')}（公開しない）")
        return 0
    if args.residual:
        rec = load_tuned("A", colsets["base"], cutoff, args.n_trials, stamp)
        if rec is None:
            raise SystemExit("A の保存済みの探索の結果が無い（--residual は探索し直さない。--keep-frame で前の回の表を使う）")
        rows = residual_rows(df, base, rec["params"], shifts)
        out = residual_report(rows, list(NEW_COLS))
        with open(path("residual.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
        log(f"記録: {path('residual.json')}（公開しない）")
        return 0
    if args.importance:
        recs = {}
        for arm, cs in (("A", "base"), ("B", "new")):
            rec = load_tuned(arm, colsets[cs], cutoff, args.n_trials, stamp)
            if rec is None:
                raise SystemExit(f"{arm} の保存済みの探索の結果が無い（--importance は探索し直さない。--keep-frame で前の回の表を使う）")
            recs[arm] = rec["params"]
        print(f"\n重要度の顔ぶれ（探索した A・B のパラメータ。A {short(recs['A'])} / B {short(recs['B'])}）")
        importance_report(df, base, recs, cutoff)
        log(f"記録: {path('importance.json')}（公開しない）")
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
