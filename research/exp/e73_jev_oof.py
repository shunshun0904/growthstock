#!/usr/bin/env python3
"""
実験73: Jev（TypeSafe AI の判断モデル）に、直近1年の OOF 候補へ本番と同じ問い（「翌営業日の寄りで買い、
20営業日以内に +10% に届く」の確率）を当て、較正・分離力・3モデル合議との重なりを測る。

運用者の決定（2026-10-10）: Jev の値は画面と台帳に出すだけで、選定の規則には入れない。前向きの記録を
貯めつつ、過去分で一度測る（本実験）。

先に決めること（結果を見てから分け方を足さない）
----------------------------------------------
対象: 本番の OOF（lgbm / xgb / cat / あれば logit）の行のうち、最新日から --days（既定 365 暦日）以内で、
      全モデルの百分位（その行より前の窓の分布。実験70 と同じ基準）と +10% の結果（実験32 の値動きの表）が
      付く行。
state: 本番（research/jev_predict.build_state）と同じ作り。百分位は前の窓の分布、寄与は**その行の窓の**
      LightGBM（本番のパラメータで窓ごとに学習し直したもの）の TreeSHAP。本番で学習済みのモデルの SHAP を
      使うと、その行を学習に使った in-sample の寄与（ラベルを少し知っている）を Jev に渡すことになるため。
腕:   prod（本番と同じ。銘柄名・コード・業種・市場・日付を含む）と anon（それらを伏せる）。Jev は
      2026-09-15 公開のモデルで、過去の日付の銘柄については「その後どうなったか」を学習データから知っている
      疑いがある。anon との差がその大きさの目安（差が無ければ先読みの心配は小さい。prod だけ良ければ
      prod の数字は当てにならない）。
物差し: hit10（20営業日以内に高値が買値の1.1倍に届いたか）、+10%指値の収益（tp10）、持ち切り（r）。
      SE は銘柄ごとにまとめて計算する（実験70 と同じ）。
見るもの:
  1. 較正: Jev の確率を 10pt 刻みの帯に分け、帯ごとの実際の到達率。Brier スコアを「母集団の到達率を
     常に答える」基準と比べる
  2. 分離力: hit10 に対する ROC-AUC。比較は各モデルの百分位と 3モデルの最小。日付内（その日の候補の
     中での順位。地合いの差を除く）の AUC も
  3. 運用との重なり: 線の上（3つとも95以上）・際どい候補（実験70 の形）・母集団を、Jev 50 以上 / 未満で
     分ける。差と SE・z
  4. 相関: Jev と各モデルの百分位の Spearman。prod と anon の Spearman
  5. 費用: 入力トークンの合計と 1件あたり
記録: 集計だけをログに出す（銘柄ごとの値は出さない）。答えは research/_data/oof/e73_jev_answers.parquet に
      控える（Actions のキャッシュに乗る。公開の成果物には上げない）。途中で止まっても続きから。

  exp=e73_jev_oof.py  args="--dry-run"     # 件数と state の例（匿名の腕）、文字数だけ。通信しない
  exp=e73_jev_oof.py  args="--arms prod --limit 200"   # 本番と同じ腕を新しい方から200件だけ（費用の確かめ）
  exp=e73_jev_oof.py                        # prod と anon の両方・直近1年（既定）

鍵は Actions の Secrets TYPESAFE_API_KEY（run-experiment.yml が env で渡す）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(RESEARCH), "scripts"))
import build_dataset as B  # noqa: E402
import jev_predict as JV  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import predict_daily as PD  # noqa: E402
import tuning  # noqa: E402
import walkforward as WF  # noqa: E402
import e70_borderline as E70  # noqa: E402
from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS  # noqa: E402

ALGOS = ("lgbm", "xgb", "cat", "logit")
ARMS = ("prod", "anon")
#: 画面の注意の線（src/lib/strategy.js の JEV.line）と同じ
JEV_LINE = 50.0
#: 較正の帯（%）。最後は 100 を含める
BINS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.001)
CACHE = os.path.join(lab.DATA_DIR, "oof", "e73_jev_answers.parquet")
OUT = os.path.join(lab.DATA_DIR, "oof", "e73_jev_oof.json")
CACHE_COLS = ["arm", "Code", "Date", "question", "state_sha", "model", "prob",
              "input_tokens", "output_tokens", "asked_at"]
#: state に要る、データセットの素の列（predictions.json の候補の値と同じ元）
FIELDS = {
    "close": "close_raw", "high52w": "high52w", "rHigh": "r_high", "breakMargin": "break_margin",
    "baseLength": "base_length", "ret20d": "ret_20d", "vol20d": "vol_20d",
    "volumeTrend": "volume_trend", "marketCap": "market_cap", "tradingValue": "tv_ma20",
    "creditRatio": "credit_ratio", "per": "per", "pbr": "pbr", "divYield": "div_yield",
    "epsGrowth": "eps_growth_q0", "salesGrowth": "sales_growth_q0", "roe": "ROE_q0",
    "opMargin": "op_margin_q0",
}

log = E70.log


# ---------------------------------------------------------------------- #
# 集計の部品（データ無しで単体テストできる）
# ---------------------------------------------------------------------- #

def calib_table(prob: Sequence[float], hit: Sequence[bool], tp10: Sequence[float],
                hold: Sequence[float], codes: Sequence, bins: Sequence[float] = BINS) -> List[Dict]:
    """Jev の確率（%）の帯ごとに、件数・平均の確率・実際の到達率・+10%指値・持ち切り。"""
    p = np.asarray(prob, dtype=float)
    h = np.asarray(hit, dtype=bool)
    t = np.asarray(tp10, dtype=float)
    r = np.asarray(hold, dtype=float)
    c = np.asarray(codes)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        row = {"lo": float(lo), "hi": float(min(hi, 100.0)), "n": n}
        if n:
            row.update({"mean_prob": float(p[m].mean()), "hit": float(h[m].mean() * 100),
                        "hit_se": E70.cluster_se(h[m].astype(float) * 100, c[m]),
                        "tp10": float(t[m].mean()), "tp10_se": E70.cluster_se(t[m], c[m]),
                        "hold": float(r[m].mean())})
        out.append(row)
    return out


def brier(prob01: Sequence[float], hit: Sequence[bool]) -> float:
    p = np.asarray(prob01, dtype=float)
    y = np.asarray(hit, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(p) else np.nan


def roc_auc(y: Sequence[bool], s: Sequence[float]) -> float:
    """順位で数える ROC-AUC（同点は 0.5）。片方の群しか無ければ NaN。"""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_in_day(dates: Sequence, y: Sequence[bool], s: Sequence[float]) -> Tuple[float, int]:
    """日付ごとの AUC（両方の群がある日だけ）を行数で重み付けした平均と、数えた日数。"""
    d = pd.DataFrame({"d": list(dates), "y": np.asarray(y, dtype=bool), "s": np.asarray(s, dtype=float)})
    num = 0.0
    den = 0
    days = 0
    for _, g in d.groupby("d"):
        a = roc_auc(g["y"], g["s"])
        if np.isfinite(a):
            num += a * len(g)
            den += len(g)
            days += 1
    return (num / den if den else np.nan), days


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    x = pd.Series(np.asarray(a, dtype=float))
    y = pd.Series(np.asarray(b, dtype=float))
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    return float(x[ok].corr(y[ok], method="spearman"))


def split_by_line(sub: pd.DataFrame, col: str, line: float = JEV_LINE) -> Dict[str, Dict]:
    """col が line 以上 / 未満で分けた成績と差（+10%指値・持ち切り）。"""
    hi = E70.summarize(sub[sub[col] >= line])
    lo = E70.summarize(sub[sub[col] < line])
    return {"hi": hi, "lo": lo,
            "diff_tp10": E70.diff(hi, lo, "tp10"), "diff_hold": E70.diff(hi, lo, "hold")}


def cache_key(arm: str, code: str, date: str, question: str = JV.QUESTION_VERSION) -> tuple:
    return (str(arm), str(code), str(date), str(question))


def arm_state(c: Dict, tone: Optional[str], arm: str) -> Dict:
    """腕ごとの state。anon は銘柄コード・名前・業種・市場・日付を伏せる。"""
    if arm not in ARMS:
        raise ValueError(f"知らない腕: {arm}")
    return JV.build_state(c, tone, anonymous=(arm == "anon"))


# ---------------------------------------------------------------------- #
# データ
# ---------------------------------------------------------------------- #

def load_rows(days: int) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """本番の OOF（全モデル）に、前の窓の百分位・+10% の結果・データセットの素の列を付けた表。"""
    from e32_takeprofit import attach, forward_paths

    model_dir = os.path.join(RESEARCH, "model")
    algos = [a for a in ALGOS if LT.find_oof(a, lab.DATA_DIR, model_dir) is not None]
    if "lgbm" not in algos:
        raise SystemExit("lgbm の OOF が無い（Release data-raw の oof.parquet）")
    base = None
    for a in algos:
        o = pd.read_parquet(LT.find_oof(a, lab.DATA_DIR, model_dir))
        o["Date"] = pd.to_datetime(o["Date"])
        o["Code"] = o["Code"].astype(str)
        keep = ["Code", "Date", "score"] + (["label", "ret_o1_20"] if a == "lgbm" else [])
        o = o[keep].rename(columns={"score": f"s_{a}"})
        base = o if base is None else base.merge(o, on=["Code", "Date"], how="inner")
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    base["fold"] = E70.assign_folds(base["Date"], frame["Date"])
    base = E70.hist_pct(base, algos)
    base["n_break"] = base.groupby("Date")["Code"].transform("size")
    base["rank_in_day"] = base.groupby("Date")["s_lgbm"].rank(ascending=False, method="min")
    base["r"] = pd.to_numeric(base["ret_o1_20"], errors="coerce") * 100
    d = attach(base, forward_paths())
    d = d[d["entry"].notna() & d["r"].notna()]
    d = d[d[[f"hp_{a}" for a in algos]].notna().all(axis=1)]
    cutoff = d["Date"].max() - pd.Timedelta(days=int(days))
    d = d[d["Date"] >= cutoff].copy()
    cols = [c for c in dict.fromkeys(FIELDS.values()) if c in frame.columns]
    d = d.merge(frame[["Code", "Date"] + cols], on=["Code", "Date"], how="left")
    return d.sort_values(["Date", "rank_in_day"]).reset_index(drop=True), algos, frame


def fold_contrib(d: pd.DataFrame, frame: pd.DataFrame, cols: List[str], params: Dict,
                 log=log) -> Dict[tuple, Dict]:
    """
    行ごとの寄与（predict_daily.contributions と同じ形）を、**その行の窓の** LightGBM で出す。
    窓は本番の OOF と同じ切り方。学習はその窓の訓練終了日までのラベル付きの行。
    """
    import lightgbm as lgb

    folds = {f.index: f for f in WF.make_folds(
        pd.to_datetime(frame["Date"]), min_train_months=OOF_MIN_TRAIN_MONTHS,
        test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS, embargo_days=B.RISE_HORIZON)}
    fd = pd.to_datetime(frame["Date"])
    feat = frame.set_index(["Code", "Date"])[cols]
    out: Dict[tuple, Dict] = {}
    for fi in sorted(int(x) for x in d["fold"].unique()):
        f = folds.get(fi)
        rows = d[d["fold"] == fi]
        if f is None or rows.empty:
            continue
        tr = frame[(fd <= pd.Timestamp(f.train_end)) & frame["label"].notna()]
        if len(tr) < 1000:
            log(f"  窓{fi}: 訓練が {len(tr)} 行しか無いので寄与を付けない")
            continue
        ytr = tr["label"].to_numpy(dtype=int)
        gbm = lgb.LGBMClassifier(**params, scale_pos_weight=tuning.scale_pos_weight(ytr))
        gbm.fit(tr[cols].to_numpy(dtype=float), ytr)
        keys = list(zip(rows["Code"], rows["Date"]))
        X = feat.loc[keys].to_numpy(dtype=float)
        for k, c in zip(keys, PD.contributions(gbm.booster_, X, cols)):
            out[k] = c
        log(f"  窓{fi}（〜{f.train_end} で学習）: {len(rows):,}行の寄与")
    return out


def candidate_rows(d: pd.DataFrame, algos: Sequence[str], contrib: Dict[tuple, Dict],
                   names: Dict[str, Dict], prog: Dict[tuple, Dict]) -> List[Dict]:
    """OOF の行を predictions.json の候補と同じ形にする（jev_predict.build_state が読む鍵）。"""
    import jquants_data_fetcher as JF

    need, _ = B.rise_thresholds(d["vol_20d"] if "vol_20d" in d.columns else pd.Series(np.nan, index=d.index),
                                B.DEFAULT_RISE)
    out = []
    for i, s in enumerate(d.itertuples(index=False)):
        row = s._asdict()
        jq = str(row["Code"])
        date = pd.Timestamp(row["Date"]).date().isoformat()
        nm = names.get(jq, {})
        pf = prog.get((jq, date), {})
        c = {
            "code": JF.display_code(jq), "jqCode": jq,
            "name": nm.get("CoName"), "sector": nm.get("S33Nm"), "market": nm.get("MktNm"),
            "scale": nm.get("ScaleCat") if isinstance(nm.get("ScaleCat"), str) else None,
            "date": date,
            "rankInDay": int(row["rank_in_day"]), "nInDay": int(row["n_break"]),
            "needPct": float(need.iloc[i] * 100) if np.isfinite(need.iloc[i]) else None,
            "progressRate": pf.get("progressRate"), "quarter": pf.get("quarter"),
            "progressBenchmark": pf.get("progressBenchmark"), "progressBasis": pf.get("progressBasis"),
            "contrib": contrib.get((jq, pd.Timestamp(row["Date"]))),
            "byModel": {a: {"score": float(row[f"s_{a}"]), "pctHistorical": float(row[f"hp_{a}"])}
                        for a in algos},
        }
        for key, col in FIELDS.items():
            c[key] = row.get(col)
        out.append(c)
    return out


# ---------------------------------------------------------------------- #
# Jev に問う
# ---------------------------------------------------------------------- #

def load_cache(path: str = CACHE) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=CACHE_COLS)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:                  # noqa: BLE001
        log(f"控えを読めないので作り直す: {type(exc).__name__}")
        return pd.DataFrame(columns=CACHE_COLS)
    for c in CACHE_COLS:
        if c not in df.columns:
            df[c] = None
    return df[CACHE_COLS]


def save_cache(df: pd.DataFrame, path: str = CACHE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[CACHE_COLS].drop_duplicates(["arm", "Code", "Date", "question"], keep="first") \
        .to_parquet(path, index=False, compression="zstd")


def ask_arm(arm: str, cands: List[Dict], tones: Dict[str, Optional[str]], client,
            cache: pd.DataFrame, workers: int, log=log) -> Tuple[Dict[tuple, float], Dict, pd.DataFrame]:
    """
    腕の全候補に問う。控えにある (腕, コード, 日付, 問いの版) は問い直さない。
    戻り値: {(Code, Date): prob}, 費用の要約, 更新した控え。
    """
    known = {cache_key(x["arm"], x["Code"], x["Date"], x["question"]): x
             for x in cache.to_dict("records")}
    probs: Dict[tuple, float] = {}
    usage = {"asked": 0, "cached": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
             "sha_mismatch": 0}
    todo = []
    for c in cands:
        state = arm_state(c, tones.get(c["date"]), arm)
        sha = JV.state_sha(state)
        k = cache_key(arm, c["jqCode"], c["date"])
        if k in known:
            rec = known[k]
            probs[(c["jqCode"], c["date"])] = float(rec["prob"])
            usage["cached"] += 1
            if str(rec.get("state_sha") or "") != sha:
                usage["sha_mismatch"] += 1
            continue
        todo.append((c, state, sha))
    log(f"[{arm}] 控えから {usage['cached']:,}件（state の作りが変わったもの {usage['sha_mismatch']}件）"
        f" / 問う {len(todo):,}件")
    if not todo:
        return probs, usage, cache
    new_records: List[Dict] = []
    asked_at = pd.Timestamp.now(tz="UTC").isoformat()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(client.system_one, state): (c, sha) for c, state, sha in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            c, sha = futs[fut]
            try:
                ans = JV.parse_noul(fut.result())
            except JV.JevError as exc:
                usage["failed"] += 1
                if usage["failed"] <= 5:
                    log(f"  [{arm}] 失敗: {exc}")
                continue
            usage["asked"] += 1
            usage["input_tokens"] += ans["input_tokens"]
            usage["output_tokens"] += ans["output_tokens"]
            probs[(c["jqCode"], c["date"])] = ans["prob"]
            new_records.append({"arm": arm, "Code": c["jqCode"], "Date": c["date"],
                                "question": JV.QUESTION_VERSION, "state_sha": sha,
                                "model": ans["model"] or client.model, "prob": ans["prob"],
                                "input_tokens": ans["input_tokens"],
                                "output_tokens": ans["output_tokens"], "asked_at": asked_at})
            if n % 100 == 0 or n == len(todo):
                new_df = pd.DataFrame(new_records, columns=CACHE_COLS)
                cache = new_df if cache.empty else pd.concat([cache, new_df], ignore_index=True)
                save_cache(cache)
                new_records = []
                per = usage["input_tokens"] / max(1, usage["asked"])
                log(f"  [{arm}] {n:,}/{len(todo):,} 済み（{time.time() - t0:.0f}秒・失敗 {usage['failed']}）"
                    f" / 入力 {per:,.0f} トークン/件 → 全体の見込み {per * len(todo):,.0f}")
    if usage["failed"] and usage["asked"] == 0:
        raise SystemExit(f"[{arm}] 全件失敗した（鍵・疎通・問いの形を確かめる）")
    return probs, usage, cache


# ---------------------------------------------------------------------- #

def report(d: pd.DataFrame, algos: Sequence[str], arms: Sequence[str], usage: Dict[str, Dict]) -> Dict:
    res: Dict[str, object] = {"rows": int(len(d)), "from": str(d["Date"].min().date()),
                              "to": str(d["Date"].max().date()), "codes": int(d["Code"].nunique()),
                              "arms": list(arms), "usage": usage}
    base_hit = float(d["hit10"].mean())
    d = d.copy()
    trees = [a for a in E70.TREES if a in algos]
    if len(trees) == len(E70.TREES):
        d["shape"] = E70.shape_of(d)
    else:                                   # 木3つがそろわなければ形は付けない（運用の線が引けない）
        d["shape"] = None
    d["hp_min"] = d[[f"hp_{a}" for a in trees]].min(axis=1)

    print(f"\n=== 0. 対象 ===\n  {len(d):,}行 / {d['Code'].nunique():,}銘柄 / {res['from']}〜{res['to']} / "
          f"+10% 到達率 {base_hit * 100:.1f}% / +10%指値 {d['tp10'].mean():+.2f}% / 持ち切り {d['r'].mean():+.2f}%")
    for arm in arms:
        col = f"jev_{arm}"
        ok = d[col].notna()
        print(f"  {arm}: Jev の値あり {int(ok.sum()):,}行 / 平均 {d.loc[ok, col].mean():.1f}% / "
              f"中央値 {d.loc[ok, col].median():.1f}%")

    # 1. 較正
    print("\n=== 1. 較正（Jev の確率の帯ごとに、実際に +10% に届いた割合）===")
    res["calibration"] = {}
    for arm in arms:
        col = f"jev_{arm}"
        g = d[d[col].notna()]
        tab = calib_table(g[col], g["hit10"], g["tp10"], g["r"], g["Code"])
        b_jev = brier(g[col] / 100.0, g["hit10"])
        b_base = brier(np.full(len(g), base_hit), g["hit10"])
        res["calibration"][arm] = {"bins": tab, "brier": b_jev, "brier_base": b_base}
        print(f"  [{arm}] Brier: Jev {b_jev:.4f} / 母集団の到達率（{base_hit * 100:.1f}%）を常に答える基準 "
              f"{b_base:.4f}（小さいほど良い）")
        print(f"  {'帯(%)':<10}{'件数':>6}{'Jevの平均':>9}{'到達率':>8}{'SE':>6}{'+10%指値':>9}{'SE':>6}{'持ち切り':>9}")
        for row in tab:
            if not row["n"]:
                continue
            print(f"  {row['lo']:>3.0f}〜{row['hi']:<4.0f}{row['n']:>8}{row['mean_prob']:>8.1f}%"
                  f"{row['hit']:>7.1f}%{row['hit_se']:>6.1f}{row['tp10']:>+8.2f}%{row['tp10_se']:>6.2f}"
                  f"{row['hold']:>+8.2f}%")

    # 2. 分離力
    print("\n=== 2. 分離力（hit10 に対する ROC-AUC。日付内は、その日の候補の中での順位）===")
    res["auc"] = {}
    rows = [(f"Jev（{arm}）", f"jev_{arm}") for arm in arms]
    rows += [(f"{a} の百分位", f"hp_{a}") for a in algos] + [("3モデルの最小", "hp_min")]
    for nm, col in rows:
        g = d[d[col].notna()]
        a = roc_auc(g["hit10"], g[col])
        a_lbl = roc_auc(g["label"].astype(bool), g[col]) if g["label"].notna().all() else np.nan
        a_day, n_days = auc_in_day(g["Date"], g["hit10"], g[col])
        res["auc"][col] = {"hit10": a, "label": a_lbl, "in_day": a_day, "days": n_days, "n": int(len(g))}
        print(f"  {nm:<18} hit10 {a:.3f} / モデルの正例 {a_lbl:.3f} / 日付内 {a_day:.3f}（{n_days}日）")

    # 3. 運用との重なり
    print(f"\n=== 3. 運用との重なり（Jev {JEV_LINE:.0f}% 以上 / 未満。差は +10%指値）===")
    print(E70.HEAD)
    res["split"] = {}
    subsets = [("線の上（3つとも95以上）", d["shape"] == "all95"),
               ("際どい候補（実験70 の形）", d["shape"].isin(E70.BORDER)),
               ("母集団", pd.Series(True, index=d.index))]
    for arm in arms:
        col = f"jev_{arm}"
        res["split"][arm] = {}
        for nm, m in subsets:
            sub = d[m & d[col].notna()]
            sp = split_by_line(sub, col)
            res["split"][arm][nm] = sp
            print(E70.line(f"[{arm}] {nm}・Jev≥{JEV_LINE:.0f}", sp["hi"]))
            print(E70.line(f"[{arm}] {nm}・Jev<{JEV_LINE:.0f}", sp["lo"]))
            dd = sp["diff_tp10"]
            print(f"      差 {dd['d']:+.2f}pt ± {dd['se']:.2f}（z {dd['z']:+.2f}）")

    # 4. 相関
    print("\n=== 4. 相関（Spearman）===")
    res["corr"] = {}
    for arm in arms:
        col = f"jev_{arm}"
        res["corr"][arm] = {a: spearman(d[col], d[f"hp_{a}"]) for a in algos}
        print(f"  [{arm}] " + " / ".join(f"{a} {res['corr'][arm][a]:+.3f}" for a in algos))
    if len(arms) == 2:
        rho = spearman(d[f"jev_{arms[0]}"], d[f"jev_{arms[1]}"])
        res["corr"]["prod_anon"] = rho
        both = d[d[f"jev_{arms[0]}"].notna() & d[f"jev_{arms[1]}"].notna()]
        mean_diff = float((both[f"jev_{arms[0]}"] - both[f"jev_{arms[1]}"]).mean()) if len(both) else np.nan
        res["corr"]["prod_minus_anon_mean"] = mean_diff
        print(f"  {arms[0]} と {arms[1]} の相関 {rho:+.3f} / 平均の差（{arms[0]} − {arms[1]}）{mean_diff:+.1f}pt")

    # 5. 費用
    print("\n=== 5. 費用（この実行で問うた分。控えから読んだ分は含まない）===")
    for arm in arms:
        u = usage.get(arm, {})
        per = u.get("input_tokens", 0) / max(1, u.get("asked", 0))
        print(f"  [{arm}] 問うた {u.get('asked', 0):,}件 / 控え {u.get('cached', 0):,}件 / 失敗 {u.get('failed', 0):,}件 / "
              f"入力 {u.get('input_tokens', 0):,} トークン（{per:,.0f}/件）/ 出力 {u.get('output_tokens', 0):,}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験73: 直近1年の OOF 候補に Jev を当てる")
    ap.add_argument("--days", type=int, default=365, help="最新日から何暦日ぶんを対象にするか")
    ap.add_argument("--arms", default="prod,anon", help="腕（prod / anon。カンマ区切り）")
    ap.add_argument("--limit", type=int, default=0, help="腕ごとに問う行の上限（新しい日から）。0 で全部")
    ap.add_argument("--workers", type=int, default=4, help="同時に投げる数")
    ap.add_argument("--dry-run", action="store_true", help="件数と state の例・文字数だけ（通信しない）")
    args = ap.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"知らない腕: {a}（{', '.join(ARMS)}）")

    d, algos, frame = load_rows(args.days)
    log(f"対象 {len(d):,}行 / {d['Code'].nunique():,}銘柄 / {d['Date'].min().date()}〜{d['Date'].max().date()} "
        f"/ モデル {', '.join(algos)}")
    if args.limit:
        keep = d.sort_values(["Date", "rank_in_day"], ascending=[False, True]).head(args.limit)
        d = d.loc[keep.index].sort_values(["Date", "rank_in_day"]).reset_index(drop=True)
        log(f"--limit {args.limit}: 新しい日から {len(d):,}行に絞る")

    meta = json.load(open(os.path.join(RESEARCH, "model", "meta.json"), encoding="utf-8"))
    cols = [c for c in meta["features"] if c in frame.columns]
    if len(cols) != len(meta["features"]):
        log(f"データセットに無い特徴量 {len(meta['features']) - len(cols)}列は寄与から外す")
    params = {k: v for k, v in (meta.get("params") or {}).items() if not k.startswith("_")}
    if not params:
        params = tuning.params_for(meta.get("preset", "all"))
    log("窓ごとの LightGBM で寄与を付ける（本番のパラメータ）")
    contrib = fold_contrib(d, frame, cols, params)
    names = {str(r["Code"]): r for r in PD.name_map(lab.DATA_DIR).to_dict("records")}
    prog = PD.progress_fields(lab.DATA_DIR, zip(d["Code"].astype(str), d["Date"].dt.strftime("%Y-%m-%d")))
    cands = candidate_rows(d, algos, contrib, names, prog)
    tones: Dict[str, Optional[str]] = {}
    for c in cands:
        tones.setdefault(c["date"], []).append(c)
    tones = {k: JV.market_tone(v) for k, v in tones.items()}
    n_contrib = sum(1 for c in cands if c.get("contrib"))
    log(f"候補の形にした {len(cands):,}件（寄与あり {n_contrib:,} / 銘柄名あり {sum(1 for c in cands if c.get('name')):,}）")

    chars = {arm: sum(len(json.dumps(arm_state(c, tones.get(c["date"]), arm), ensure_ascii=False,
                                     separators=(",", ":"))) for c in cands) for arm in arms}
    for arm in arms:
        log(f"[{arm}] state の合計 {chars[arm]:,}文字（1件 {chars[arm] / max(1, len(cands)):,.0f}文字）")
    if args.dry_run or not JV.enabled():
        if not args.dry_run:
            log(f"{JV.API_KEY_ENV} が未設定なので問わない（Actions の Secrets に登録する）")
        if cands:
            ex = arm_state(cands[-1], tones.get(cands[-1]["date"]), "anon")
            print("\n[例] anon の state（銘柄・日付を伏せたもの）:")
            print(json.dumps(ex, ensure_ascii=False, indent=1))
            print("\n[問い]")
            print(json.dumps(JV.QUESTIONS, ensure_ascii=False, indent=1))
        print("\n通信していません。費用はトークン単価 × 上の文字数（≒ トークン数の目安）で見積もる")
        return 0 if args.dry_run else 1

    client = JV.Client()
    log(f"Jev: {client.base_url} / モデル {client.model}")
    cache = load_cache()
    usage: Dict[str, Dict] = {}
    for arm in arms:
        probs, u, cache = ask_arm(arm, cands, tones, client, cache, args.workers)
        usage[arm] = u
        d[f"jev_{arm}"] = [probs.get((c["jqCode"], c["date"]), np.nan) for c in cands]
    res = report(d, algos, arms, usage)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x))
    log(f"記録: {OUT}（集計だけ。銘柄ごとの値は控え {CACHE} にあり、公開の成果物には上げない）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
