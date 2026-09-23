#!/usr/bin/env python3
"""
実運用の追跡: 日次予測が画面に出した判断を、out-of-fold（OOF）で見込んだ
成績と比べる。

何と何を比べるか
  実運用  毎日の予測（public/data/predictions.json）は Predict Breakouts が
          コミットするので、git の履歴に「その日、画面に何が出ていたか」が
          残っている。日付ごとに、**翌営業日の寄り（9:00 JST = 0:00 UTC）より
          前の最後の版**を「判断に使えた版」とする。寄りまでに版が無ければ
          最初に出た版を使い、「遅れ」の印を付ける。
          そこに運用の規則を当てて、紙の上の売買を作る。値動きは保存済みの
          日足（分割調整後）。
  見込み  本番モデルの OOF（Release の oof.parquet / xgb_oof.parquet /
          cat_oof.parquet）に同じ規則を当てる。百分位は画面と同じく
          「OOF 全体のうち、そのスコアより低い割合」。
          **1本の数字にしない。** 実運用と同じ長さの期間を1か月（21営業日）
          ずつずらして切り出し、期間ごとの成績の散らばりを見込みの幅にする。
          1つの OOF の平均で決めると過適合する（運用者の方針）。

規則（docs/PLAYBOOK.md。画面の src/lib/strategy.js の STRATEGY と同じ値で、
tests/test_live_track.py が両者を突き合わせる）
  その日の発火（候補数）が 8件未満なら買わない
  ブースティング3モデル（LightGBM / XGBoost / CatBoost）すべてが 90 以上
  3モデルの最小の百分位が高い順に上位2件（同点は LightGBM のスコア順）
  翌営業日の寄りで買い、+20% の指値、届かなければ 20営業日で手仕舞い
  同時に持つのは 3銘柄まで。埋まっていたら見送る（先着順・乗り換えなし）

収益の数え方は OOF の実験（実験32〜35）と同じ
  買値       翌営業日の始値（分割調整後）
  +20%到達   1〜20営業日目（1 = 買った日）の高値が買値の1.2倍以上なら +20% ちょうど
  届かない   20営業日目までの5日平均終値（lab.realized_returns の ret_o1_20）
  参考に「20営業日目の終値で売った場合」も並べる
  手数料・税・スリッページは含まない

途中の取引を平均に入れない
  +20% に早く届いた取引だけが先に確定するので、確定した取引だけで平均を
  取ると勝ち側に偏る。**選定日から20営業日が過ぎた日の取引だけ**を成績に
  数え、それより後の取引は「途中」として含み損益だけ出す。

使い方
  python3 research/live_track.py                  # 集計して表示
  python3 research/live_track.py --json out.json  # 要約を JSON にも書く
  GitHub Actions では run-experiment.yml に exp=live_vs_oof.py を渡す
  （Release の日足・OOF を落とし、ラベル付きのデータセットを作ってから回る）
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import subprocess
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import trading_calendar as TC  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
MODEL_DIR = os.path.join(HERE, "model")
OUT_DIR = os.path.join(DATA_DIR, "oof")
PRED_PATH = "public/data/predictions.json"

#: 合議に使う3モデル（線形・MLP は入れない）。画面の BOOST と同じ
BOOST = ("lgbm", "xgb", "cat")

# --- 運用の規則。src/lib/strategy.js の STRATEGY と同じ値 --- #
AGREE_PCT = 90.0      # agreePct: 3モデルすべてがこの百分位以上
STRONG_BREAKS = 20    # strongBreaks: この件数以上の発火なら本命の日
SKIP_BREAKS = 8       # skipBreaks: これ未満の発火なら見送る
TOP_K = 2             # topK: 買うのは上位何件まで
TAKE_PROFIT = 20.0    # takeProfit: 利確の指値（%）
HOLD = 20             # holdDays: 届かなければ何営業日で手仕舞いするか
SLOTS = 3             # maxSlots: 同時に持てる銘柄数

#: 見込みの幅を作るときに期間をずらす幅（営業日）。約1か月
STEP = 21
#: 1か月の営業日数（月あたりに直すときの換算）
MONTH = 21.0

TAKEN = "買う"
FULL = "枠が満杯"


# ---------------------------------------------------------------- #
# 実運用: git の履歴から「その日、画面に出ていた判断」を復元する
# ---------------------------------------------------------------- #

def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, check=True).stdout


def ensure_history(repo: str, fetch: bool = True) -> bool:
    """
    浅い clone（GitHub Actions の checkout は既定で最新1コミット）なら
    履歴を取り寄せる。取り寄せられなければ False（復元できる日が減る）。
    """
    try:
        shallow = _git(repo, "rev-parse", "--is-shallow-repository").strip() == "true"
    except (OSError, subprocess.CalledProcessError):
        return False
    if shallow and fetch:
        print("[git] 浅い clone なので履歴を取り寄せる（git fetch --unshallow）")
        subprocess.run(["git", "-C", repo, "fetch", "--unshallow", "--quiet"],
                       check=False)
        shallow = _git(repo, "rev-parse", "--is-shallow-repository").strip() == "true"
    return not shallow


def load_versions(repo: str, path: str = PRED_PATH) -> List[dict]:
    """predictions.json の版を古い順に。読めない版は飛ばす。"""
    out = []
    log = _git(repo, "log", "--reverse", "--format=%H %cI", "--", path)
    for line in log.splitlines():
        sha, _, ts = line.partition(" ")
        try:
            data = json.loads(_git(repo, "show", f"{sha}:{path}"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        out.append({"sha": sha,
                    "committed": pd.Timestamp(ts).tz_convert("UTC"),
                    "data": data})
    return out


def trading_days(data_dir: str) -> List[dt.date]:
    """保存済みの取引所カレンダーの営業日。無ければ空。"""
    return sorted(TC.load(data_dir).days)


def next_trading_day(day: dt.date, days: Sequence[dt.date]) -> dt.date:
    """day の次の営業日。カレンダーの外なら次の平日で代える。"""
    i = int(np.searchsorted(np.array(days, dtype="datetime64[D]"),
                            np.datetime64(day, "D"), side="right"))
    if days and i < len(days):
        return days[i]
    nxt = day + dt.timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += dt.timedelta(days=1)
    return nxt


def next_open_utc(day: dt.date, days: Sequence[dt.date]) -> pd.Timestamp:
    """day の翌営業日の寄り（9:00 JST = 0:00 UTC）。"""
    return pd.Timestamp(next_trading_day(day, days)).tz_localize("UTC")


def _pct(v) -> float:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else np.nan


def live_rows(versions: Sequence[dict], days: Sequence[dt.date]) -> pd.DataFrame:
    """
    日付ごとに「判断に使えた版」を選び、その版の候補を1行ずつにする。

    列: Date, Code, code, name, score, p_lgbm, p_xgb, p_cat,
        sha, committed, trained, late
    """
    by_date: Dict[str, List[int]] = {}
    for i, v in enumerate(versions):
        for d in sorted({c.get("date") for c in v["data"].get("candidates", [])
                         if c.get("date")}):
            by_date.setdefault(d, []).append(i)
    rows = []
    for d, idx in sorted(by_date.items()):
        opens = next_open_utc(dt.date.fromisoformat(d), days)
        before = [i for i in idx if versions[i]["committed"] < opens]
        late = not before
        v = versions[before[-1] if before else idx[0]]
        trained = (v["data"].get("model") or {}).get("trainedAt")
        for c in v["data"].get("candidates", []):
            if c.get("date") != d:
                continue
            per = c.get("byModel") or {}
            rec = {"Date": pd.Timestamp(d),
                   "Code": str(c.get("jqCode") or c.get("code") or ""),
                   "code": str(c.get("code") or ""),
                   "name": c.get("name") or "",
                   "score": _pct(c.get("score")),
                   "sha": v["sha"][:7], "committed": v["committed"],
                   "trained": (trained or "")[:10], "late": late}
            for a in BOOST:
                p = (per.get(a) or {}).get("pctHistorical")
                if a == "lgbm" and p is None:
                    p = c.get("pctHistorical")
                rec[f"p_{a}"] = _pct(p)
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
# 規則: 日ごとの判断と、買う銘柄
# ---------------------------------------------------------------- #

def decide(rows: pd.DataFrame, *, agree: float = AGREE_PCT,
           min_break: int = SKIP_BREAKS, top_k: int = TOP_K):
    """
    画面の strategySignal と同じ判断を、日ごとにまとめて出す。

    rows: 1行 = その日の候補1件（Date, Code, score, p_lgbm, p_xgb, p_cat）。
          発火数はその日の行数（画面と同じ数え方）。
    戻り値: (rows に p_min / n_break / passed / rank を足したもの,
             1日1行の判断, 買う銘柄)
    """
    r = rows.copy()
    pc = [f"p_{a}" for a in BOOST]
    # 1つでも欠けていれば NaN（欠けたまま「満たした」としない。画面と同じ）
    r["p_min"] = r[pc].min(axis=1, skipna=False)
    r["n_break"] = r.groupby("Date")["Code"].transform("size")
    r["passed"] = (r["p_min"] >= agree).fillna(False).astype(bool)
    r = r.sort_values(["Date", "p_min", "score"], ascending=[True, False, False],
                      na_position="last").reset_index(drop=True)
    r["rank"] = np.nan
    ok = r["passed"]
    r.loc[ok, "rank"] = r[ok].groupby("Date").cumcount() + 1
    r["buyable_day"] = r["n_break"] >= min_break
    picks = r[ok & r["buyable_day"] & (r["rank"] <= top_k)].copy()
    picks["rank"] = picks["rank"].astype(int)

    days = []
    for d, g in r.groupby("Date", sort=True):
        n = len(g)
        n_pass = int(g["passed"].sum())
        n_missing = int(g["p_min"].isna().sum())
        if n < min_break:
            verdict = f"見送り（発火{n}件）"
        elif n_missing:
            verdict = "判定できない（3モデルの百分位が無い）"
        elif not n_pass:
            verdict = "買わない（基準を満たす銘柄なし）"
        else:
            verdict = f"買う（{min(n_pass, top_k)}件）"
        days.append({"Date": d, "n_break": n, "n_pass": n_pass,
                     "n_missing": n_missing, "verdict": verdict,
                     "buy": n >= min_break and n_pass > 0 and not n_missing,
                     "strong": n >= STRONG_BREAKS})
    return r, pd.DataFrame(days), picks


# ---------------------------------------------------------------- #
# 値動き: 買値・+20% 到達・満了
# ---------------------------------------------------------------- #

def load_bars(data_dir: str, start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """日足（分割調整後の始値・高値・終値）。start の年より前のファイルは読まない。"""
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if start is not None:
        paths = [p for p in paths
                 if os.path.basename(p)[5:9].isdigit()
                 and int(os.path.basename(p)[5:9]) >= start.year]
    if not paths:
        raise SystemExit(f"日足（bars_*.parquet）がありません: {data_dir}")
    b = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "AdjO", "AdjH", "AdjC"])
                   for p in paths], ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str)
    return b


def forward(bars: pd.DataFrame, keys: pd.DataFrame, *, target: float = TAKE_PROFIT,
            hold: int = HOLD) -> pd.DataFrame:
    """
    (Code, Date) ごとの値動き。Date は選定日（ブレイクの日）。

    列: status（満了 / +20%到達 / 保有中 / 未約定 / 寄り付かず / 日足なし）,
        buy_date, entry, hit_day, exit_date, ret（規則どおりの収益 %）,
        ret_hold（ret_o1_20 %）, ret_close（20営業日目の終値で売った場合 %）,
        elapsed（選定日から数えた営業日。1 = 買った日）, ret_now（保有中の含み %）
    """
    need = keys[["Code", "Date"]].drop_duplicates()
    want = need.groupby("Code")["Date"].apply(list).to_dict()
    b = bars[bars["Code"].isin(want)].sort_values(["Code", "Date"])
    out = []
    seen = set()
    for code, g in b.groupby("Code", sort=False):
        dates = g["Date"].to_numpy(dtype="datetime64[ns]")
        o = g["AdjO"].to_numpy(dtype=float)
        h = g["AdjH"].to_numpy(dtype=float)
        c = g["AdjC"].to_numpy(dtype=float)
        ma5 = pd.Series(c).rolling(5, min_periods=1).mean().to_numpy()
        n = len(dates)
        for d in want[code]:
            seen.add((code, d))
            rec = {"Code": code, "Date": d, "status": "", "buy_date": pd.NaT,
                   "entry": np.nan, "hit_day": np.nan, "exit_date": pd.NaT,
                   "ret": np.nan, "ret_hold": np.nan, "ret_close": np.nan,
                   "elapsed": np.nan, "ret_now": np.nan}
            i = int(np.searchsorted(dates, np.datetime64(d, "ns")))
            if i >= n or dates[i] != np.datetime64(d, "ns"):
                rec["status"] = "日足なし"
                out.append(rec)
                continue
            if i + 1 >= n:
                rec["status"] = "未約定"
                out.append(rec)
                continue
            entry = o[i + 1]
            rec["buy_date"] = pd.Timestamp(dates[i + 1])
            if not (np.isfinite(entry) and entry > 0):
                rec["status"] = "寄り付かず"
                out.append(rec)
                continue
            rec["entry"] = entry
            level = entry * (1.0 + target / 100.0)
            last = min(n - 1, i + hold)
            hit = None
            for k in range(1, last - i + 1):
                if np.isfinite(h[i + k]) and h[i + k] >= level:
                    hit = k
                    break
            if i + hold < n:
                rec["ret_hold"] = (ma5[i + hold] / entry - 1.0) * 100
                rec["ret_close"] = (c[i + hold] / entry - 1.0) * 100
            if hit is not None:
                rec.update(status="+20%到達", hit_day=hit, ret=target,
                           exit_date=pd.Timestamp(dates[i + hit]))
            elif i + hold < n:
                rec.update(status="満了", ret=rec["ret_hold"],
                           exit_date=pd.Timestamp(dates[i + hold]))
            else:
                rec.update(status="保有中", elapsed=n - 1 - i,
                           ret_now=(c[n - 1] / entry - 1.0) * 100)
            out.append(rec)
    for code, d in need.itertuples(index=False):
        if (code, d) not in seen:
            out.append({"Code": code, "Date": d, "status": "日足なし"})
    f = pd.DataFrame(out)
    return keys.merge(f, on=["Code", "Date"], how="left")


# ---------------------------------------------------------------- #
# 枠: 同時に持つのは3銘柄まで、先着順
# ---------------------------------------------------------------- #

def simulate(picks: pd.DataFrame, bar_days: Sequence, *, slots: int = SLOTS,
             hold: int = HOLD) -> pd.DataFrame:
    """
    買う銘柄（decide の picks に forward の列を足したもの）を先着順に枠へ入れる。

    実験35（research/exp/e35_slack.py の simulate）と同じ約束:
      翌営業日（選定日の次の日足の日）に買う
      +20% に k営業日目で届けば、その翌営業日から枠が空く
      届かなければ 20営業日目の翌営業日から空く
      空きが無ければ見送る。保有中の玉は切らない
    まだ終わっていない玉（保有中・未約定）は、終わるまで枠をふさぐ。

    戻り値: picks に taken（買う / 枠が満杯 / 買えない）と slot を足したもの
    """
    idx = {pd.Timestamp(d): i for i, d in enumerate(bar_days)}
    free = [-1.0] * slots
    out = []
    for r in picks.sort_values(["Date", "rank"]).to_dict("records"):
        di = idx.get(pd.Timestamp(r["Date"]))
        rec = dict(r, taken="買えない", slot=np.nan)
        if di is None or r.get("status") in ("日足なし", "寄り付かず"):
            out.append(rec)
            continue
        buy = di + 1
        empty = [s for s in range(slots) if free[s] <= buy]
        if not empty:
            rec["taken"] = FULL
            out.append(rec)
            continue
        s = empty[0]
        if r.get("status") == "+20%到達":
            free[s] = buy + float(r["hit_day"])
        elif r.get("status") == "満了":
            free[s] = buy + hold
        else:
            free[s] = math.inf
        rec.update(taken=TAKEN, slot=s)
        out.append(rec)
    return pd.DataFrame(out, columns=list(picks.columns) + ["taken", "slot"])


def trade_stats(t: pd.DataFrame, n_days: int, slots: int = SLOTS) -> dict:
    """
    取引の成績。月あたりの収益は「取引の収益の合計 ÷ 枠数 ÷ 月数」の概算
    （1取引に資金の 1/枠 を入れる。複利にしない）。
    """
    r = pd.to_numeric(t.get("ret", pd.Series(dtype=float)), errors="coerce").dropna()
    months = max(n_days, 1) / MONTH
    n = int(len(r))
    hit = (t.loc[r.index, "status"] == "+20%到達") if n else pd.Series(dtype=bool)
    return {"n": n,
            "mean": float(r.mean()) if n else np.nan,
            "median": float(r.median()) if n else np.nan,
            "win": float((r > 0).mean() * 100) if n else np.nan,
            "se": float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else np.nan,
            "hit": float(hit.mean() * 100) if n else np.nan,
            "worst": float(r.min()) if n else np.nan,
            "per_month": n / months,
            "monthly": float(r.sum() / slots / months) if n else 0.0}


# ---------------------------------------------------------------- #
# 見込み: 本番モデルの OOF に同じ規則を当てる
# ---------------------------------------------------------------- #

def pct_of(scores: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """
    画面の pctHistorical と同じ: 過去分布のうち、そのスコアより低い割合 × 100
    を小数1桁に丸める。丸めも本番（models.pct_historical）と同じ Python の round
    にしてある。90.0 ちょうどの境目で基準の判定が割れないように。
    """
    s = np.sort(np.asarray(hist, dtype=float))
    k = np.searchsorted(s, np.asarray(scores, dtype=float), side="left")
    return np.array([round(float(x / len(s)) * 100, 1) for x in k])


def find_oof(algo: str, oof_dir: str, model_dir: str) -> Optional[str]:
    """
    本番モデルの OOF の置き場所。
      Release から落とした形   <oof_dir>/oof.parquet, <oof_dir>/<algo>_oof.parquet
      日次予測が並べ直した形   <model_dir>/oof.parquet, <model_dir>/models/<algo>/oof.parquet
    """
    if algo == "lgbm":
        cands = [os.path.join(oof_dir, "oof.parquet"),
                 os.path.join(model_dir, "oof.parquet")]
    else:
        cands = [os.path.join(oof_dir, f"{algo}_oof.parquet"),
                 os.path.join(model_dir, "models", algo, "oof.parquet")]
    return next((p for p in cands if os.path.exists(p)), None)


def oof_rows(oof_dir: str, model_dir: str) -> pd.DataFrame:
    """3モデルの OOF を (Code, Date) で揃え、画面と同じ百分位を付ける。"""
    base = None
    for a in BOOST:
        path = find_oof(a, oof_dir, model_dir)
        if path is None:
            raise SystemExit(f"{a} の OOF が見つかりません（{oof_dir} / {model_dir}）")
        o = pd.read_parquet(path)
        o["Date"] = pd.to_datetime(o["Date"])
        o["Code"] = o["Code"].astype(str)
        keep = ["Code", "Date", "score"] + (["label"] if a == "lgbm" and "label" in o else [])
        o = o[keep].rename(columns={"score": f"s_{a}"})
        o[f"p_{a}"] = pct_of(o[f"s_{a}"].to_numpy(), o[f"s_{a}"].to_numpy())
        base = o if base is None else base.merge(o, on=["Code", "Date"], how="inner")
    base["score"] = base["s_lgbm"]
    return base


# ---------------------------------------------------------------- #
# 比べ方: 同じ長さの期間を切り出した散らばり
# ---------------------------------------------------------------- #

def auc(y: pd.Series, s: pd.Series) -> float:
    """ROC-AUC。片方のクラスしか無い、または 20件未満なら NaN。"""
    from sklearn.metrics import roc_auc_score

    m = y.notna() & s.notna()
    y, s = y[m].astype(int), s[m]
    if len(y) < 20 or y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def period_stats(rows: pd.DataFrame, days_tbl: pd.DataFrame, picks_fwd: pd.DataFrame,
                 all_fwd: pd.DataFrame, bar_days: Sequence, lo, hi, n_days: int,
                 *, complete_to=None) -> dict:
    """
    [lo, hi] の期間の成績。complete_to を渡すと、取引の成績（平均・勝率・月あたり）は
    選定日が complete_to 以前の取引だけで数える（途中の取引で勝ち側に偏らないため）。
    枠の埋まり方は期間全体で回す（途中の玉も後の信号をふさぐ）。
    """
    inside = lambda df: df[(df["Date"] >= lo) & (df["Date"] <= hi)]  # noqa: E731
    dt_ = inside(days_tbl)
    rw = inside(rows)
    sim = simulate(inside(picks_fwd), bar_days)
    taken = sim[sim["taken"] == TAKEN]
    cut = hi if complete_to is None else min(pd.Timestamp(hi), pd.Timestamp(complete_to))
    n_cut = n_days if complete_to is None else sum(
        1 for d in bar_days if pd.Timestamp(lo) <= pd.Timestamp(d) <= cut)
    done = taken[taken["Date"] <= cut]
    allp = inside(all_fwd)
    allp = allp[allp["passed"] & allp["buyable_day"] & (allp["Date"] <= cut)]
    out = {"days": len(dt_),
           "break_mean": float(dt_["n_break"].mean()) if len(dt_) else np.nan,
           "pass_rate": float(dt_["n_pass"].sum() / max(dt_["n_break"].sum(), 1) * 100)
           if len(dt_) else np.nan,
           # モデルごとに 90 以上の割合。OOF 全体では定義上 10%。本番モデルの
           # スコアの水準が OOF とずれると、ここが 10% から離れる
           **{f"share_{a}": float((rw[f"p_{a}"] >= AGREE_PCT).mean() * 100)
              if len(rw) else np.nan for a in BOOST},
           "buy_days_pm": float(dt_["buy"].sum() / max(n_days, 1) * MONTH),
           "signals": int(len(sim)), "taken": int(len(taken)),
           "full": int((sim["taken"] == FULL).sum()),
           "trade": trade_stats(done, n_cut),
           "trade_days": n_cut,
           "all_pass": trade_stats(allp, n_cut, slots=max(len(allp), 1))}
    # 分離力（ラベルがある行だけ）
    if "label" in rw:
        lab = rw[rw["Date"] <= cut]
        for a in BOOST:
            out[f"auc_{a}"] = auc(lab["label"], lab[f"p_{a}"])
        out["n_label"] = int(lab["label"].notna().sum())
    return out


def blocks(rows, days_tbl, picks_fwd, all_fwd, bar_days, length: int,
           step: int = STEP) -> pd.DataFrame:
    """OOF の期間を length 営業日ずつ切り出し、step 営業日ずつずらす。"""
    lo_all, hi_all = rows["Date"].min(), rows["Date"].max()
    days = [pd.Timestamp(d) for d in bar_days if lo_all <= pd.Timestamp(d) <= hi_all]
    out = []
    for s in range(0, len(days) - length + 1, step):
        lo, hi = days[s], days[s + length - 1]
        st = period_stats(rows, days_tbl, picks_fwd, all_fwd, bar_days, lo, hi, length)
        out.append({"start": lo, "end": hi, **_flat(st)})
    return pd.DataFrame(out)


def _flat(st: dict) -> dict:
    out = {}
    for k, v in st.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}_{k2}"] = v2
        else:
            out[k] = v
    return out


def where(value: float, dist: Iterable[float]) -> dict:
    """value が分布のどこにいるか（下から何%目）と、10/50/90% 点。"""
    d = np.array([x for x in dist if x is not None and np.isfinite(x)], dtype=float)
    if not len(d):
        return {"n": 0}
    out = {"n": int(len(d)), "p10": float(np.percentile(d, 10)),
           "p50": float(np.percentile(d, 50)), "p90": float(np.percentile(d, 90))}
    if value is not None and np.isfinite(value):
        out["rank"] = float((d < value).mean() * 100 + (d == value).mean() * 50)
    return out


def verdict_of(w: dict) -> str:
    if not w.get("n") or "rank" not in w:
        return "比べられない"
    if w["rank"] < 10:
        return "見込みの幅より下"
    if w["rank"] > 90:
        return "見込みの幅より上"
    return "見込みの幅の中"


# ---------------------------------------------------------------- #
# 表示
# ---------------------------------------------------------------- #

def _width(s: str) -> int:
    """端末での表示幅（全角 = 2）。日本語の列を揃えるため。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _l(v, w: int) -> str:
    s = str(v)
    return s + " " * max(0, w - _width(s))


def _r(v, w: int) -> str:
    s = str(v)
    return " " * max(0, w - _width(s)) + s


def _f(v, fmt="{:+.2f}", na="-"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    return fmt.format(v)


def _cut(s: str, w: int) -> str:
    """表示幅 w に収まるように切る。"""
    out = ""
    for ch in str(s):
        if _width(out + ch) > w:
            break
        out += ch
    return out


def compare_line(name: str, live: float, w: dict, fmt: str = "{:+.2f}") -> str:
    head = f"  {_l(name, 24)}{_r(_f(live, fmt), 9)}"
    if not w.get("n"):
        return head + "   （OOF の期間が足りない）"
    return (head + f"{_r(_f(w['p10'], fmt), 9)}{_r(_f(w['p50'], fmt), 9)}"
            f"{_r(_f(w['p90'], fmt), 9)}{_r(_f(w.get('rank'), '{:.0f}%'), 8)}   {verdict_of(w)}")


COMPARE_HEAD = (f"  {_l('指標', 24)}{_r('実運用', 9)}{_r('OOF 10%', 9)}{_r('中央値', 9)}"
                f"{_r('90%', 9)}{_r('下から', 8)}")


def n_bar_days(bar_days: Sequence, lo, hi) -> int:
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    return sum(1 for d in bar_days if lo <= pd.Timestamp(d) <= hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実運用の成績を OOF の見込みと比べる")
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--oof-dir", default=DATA_DIR,
                    help="Release から落とした OOF（oof.parquet / <algo>_oof.parquet）の場所")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--since", default="",
                    help="この日以降を実運用として数える（既定: 寄りまでに版が出た最初の日）")
    ap.add_argument("--no-fetch", action="store_true", help="浅い clone でも履歴を取り寄せない")
    ap.add_argument("--json", default="", help="要約を JSON で書く先")
    args = ap.parse_args(argv)

    # ---- 実運用の判断を復元 ---- #
    full = ensure_history(args.repo, fetch=not args.no_fetch)
    vers = load_versions(args.repo)
    cal = trading_days(args.data_dir)
    live = live_rows(vers, cal)
    if live.empty:
        print("予測の履歴が見つかりません（public/data/predictions.json のコミットが無い）")
        return 1
    if args.since:
        start = pd.Timestamp(args.since)
    else:
        rt = live.loc[~live["late"], "Date"]
        start = rt.min() if len(rt) else live["Date"].min()
    early = live[live["Date"] < start]
    live = live[live["Date"] >= start].reset_index(drop=True)

    bars = load_bars(args.data_dir, start=pd.Timestamp("2021-01-01"))
    bar_days = sorted(pd.Timestamp(d) for d in bars["Date"].unique())
    last_bar = bar_days[-1]
    # 選定日がこの日以前なら、20営業日の出口まで日足が揃っている
    complete_to = bar_days[-1 - HOLD] if len(bar_days) > HOLD else None

    # ラベル（学習データと同じ定義）。ラベル付きのデータセットがあれば付ける
    ds_path = os.path.join(args.data_dir, "dataset.parquet")
    if os.path.exists(ds_path):
        labels = pd.read_parquet(ds_path, columns=["Code", "Date", "label"])
        labels["Date"] = pd.to_datetime(labels["Date"])
        labels["Code"] = labels["Code"].astype(str)
        live = live.merge(labels, on=["Code", "Date"], how="left")

    lrows, ldays, lpicks = decide(live)
    lfwd_all = forward(bars, lrows)
    lpicks_fwd = forward(bars, lpicks)
    live_lo, live_hi = lrows["Date"].min(), lrows["Date"].max()
    n_live = max(n_bar_days(bar_days, live_lo, live_hi), int(ldays["Date"].nunique()))
    lsim = simulate(lpicks_fwd, bar_days)
    lst = period_stats(lrows, ldays, lpicks_fwd, lfwd_all, bar_days, live_lo, live_hi,
                       n_live, complete_to=complete_to)

    # ---- 見込み（本番モデルの OOF に同じ規則）---- #
    orows = oof_rows(args.oof_dir, args.model_dir)
    orows, odays, opicks = decide(orows)
    ofwd_all = forward(bars, orows)
    opicks_fwd = forward(bars, opicks)
    o_lo, o_hi = orows["Date"].min(), orows["Date"].max()
    whole = period_stats(orows, odays, opicks_fwd, ofwd_all, bar_days, o_lo, o_hi,
                         n_bar_days(bar_days, o_lo, o_hi))
    blk_all = blocks(orows, odays, opicks_fwd, ofwd_all, bar_days, max(n_live, 1))
    n_done = (lst["trade_days"]
              if complete_to is not None and live_lo <= complete_to else 0)
    blk_done = (blocks(orows, odays, opicks_fwd, ofwd_all, bar_days, n_done)
                if n_done >= 5 else pd.DataFrame())

    # ---- 表示 ---- #
    today = dt.datetime.now(dt.timezone.utc).date()
    t_live = lsim[lsim["taken"] == TAKEN]
    t_done = (t_live[t_live["Date"] <= complete_to] if complete_to is not None
              else t_live.iloc[:0])
    t_open = t_live[~t_live.index.isin(t_done.index)]
    first_mature = None
    if len(t_live):
        fut = [d for d in cal if d > pd.Timestamp(t_live["Date"].min()).date()]
        first_mature = fut[HOLD - 1] if len(fut) >= HOLD else None
    wt = whole["trade"]

    print("=" * 78)
    print(f"実運用の追跡（ミニブレイク予測・運用の規則）  集計 {today} / 日足 〜{last_bar.date()}")
    print("=" * 78)
    print("\n■ 結論")
    print(f"  ・記録: {live_lo.date()}〜{live_hi.date()} の {int(ldays['Date'].nunique())}営業日"
          f"（画面の判断を git の履歴から復元）。買う日 {int(ldays['buy'].sum())}日、"
          f"規則どおりの買い {len(t_live)}件（成績に数える {len(t_done)}件、途中 {len(t_open)}件）")
    print(f"  ・見込み（本番モデルの OOF に同じ規則）: 1取引 {_f(wt['mean'])}%"
          f"（{wt['n']}件、勝率 {_f(wt['win'], '{:.0f}')}%）、月あたり {_f(wt['monthly'])}%"
          f"（枠{SLOTS}・複利なしの概算）")
    if len(t_done):
        ts = lst["trade"]
        w = where(ts["mean"], blk_done.get("trade_mean", []))
        print(f"  ・実運用の1取引の平均 {_f(ts['mean'])}%（{ts['n']}件、勝率 "
              f"{_f(ts['win'], '{:.0f}')}%）。OOF を同じ長さで切り出すと中央値 "
              f"{_f(w.get('p50'))}%、10〜90% の幅 {_f(w.get('p10'))}〜{_f(w.get('p90'))}%"
              f" → {verdict_of(w)}")
        wm = where(ts["monthly"], blk_done.get("trade_monthly", []))
        print(f"  ・実運用の月あたり（概算）{_f(ts['monthly'])}%。OOF の同じ長さの期間では"
              f"中央値 {_f(wm.get('p50'))}%（10〜90%: {_f(wm.get('p10'))}〜"
              f"{_f(wm.get('p90'))}%）→ {verdict_of(wm)}")
    else:
        when = (f"最初の満期は {first_mature} の見込み" if first_mature
                else "まだ買いが無い")
        print(f"  ・実運用の取引の成績はまだ比べられない（選定日から{HOLD}営業日たった"
              f"取引が無い。{when}）")
    wp = where(lst["pass_rate"], blk_all.get("pass_rate", []))
    print(f"  ・選定の頻度: 基準を満たした候補 {int(ldays['n_pass'].sum())}/"
          f"{int(ldays['n_break'].sum())}件（{_f(lst['pass_rate'], '{:.1f}')}%）。"
          f"OOF の同じ長さの期間では中央値 {_f(wp.get('p50'), '{:.1f}')}%"
          f"（10〜90%: {_f(wp.get('p10'), '{:.1f}')}〜{_f(wp.get('p90'), '{:.1f}')}%）"
          f"→ {verdict_of(wp)}")
    if lst.get("n_label"):
        aucs = " / ".join(f"{a} {_f(lst.get(f'auc_{a}'), '{:.3f}')}" for a in BOOST)
        print(f"  ・分離力（ラベル確定 {lst['n_label']}件の ROC-AUC）: {aucs}")
    else:
        print(f"  ・分離力: ラベルが確定した候補がまだ無い（選定日から{HOLD}営業日かかる）")

    print("\n■ 1. 記録の状況")
    print(f"  予測ファイルの版 {len(vers)}個（履歴が"
          f"{'揃っている' if full else '浅い。古い日が欠ける'}）")
    if len(early):
        print(f"  {early['Date'].min().date()}〜{early['Date'].max().date()} は運用開始前に"
              f"まとめて出た版しか無いので数えない（{early['Date'].nunique()}日）")
    meta = lrows.drop_duplicates("Date").set_index("Date")
    print(f"  寄りまでに版が出なかった日（遅れ）: {int(meta['late'].sum())}日")
    print(f"  日足の最終日 {last_bar.date()} / 成績に数えるのは選定日が "
          f"{complete_to.date() if complete_to is not None else '-'} 以前の取引")

    print("\n■ 2. 日ごとの判断（画面が出していた版）")
    print(f"  {_l('選定日', 11)}{_r('発火', 5)}{_r('通過', 5)}  {_l('判断', 34)}"
          f"{_l('買う銘柄', 12)}{_l('版', 9)}{_l('出た時刻(UTC)', 14)}{_l('学習日', 11)}")
    for r in ldays.itertuples(index=False):
        m = meta.loc[r.Date]
        codes = " ".join(p.code for p in lpicks[lpicks["Date"] == r.Date].itertuples())
        mark = "（遅れ）" if m["late"] else ""
        print(f"  {_l(r.Date.date(), 11)}{_r(r.n_break, 5)}{_r(r.n_pass, 5)}  "
              f"{_l(r.verdict, 34)}{_l(codes, 12)}{_l(m['sha'], 9)}"
              f"{_l(m['committed'].strftime('%m-%d %H:%M'), 14)}{_l(m['trained'], 11)}{mark}")

    print(f"\n■ 3. 取引（規則どおり・枠{SLOTS}・先着順）")
    if not len(lsim):
        print("  （買う銘柄が出た日が無い）")
    else:
        print(f"  {_l('選定日', 11)}{_l('銘柄', 6)}{_l('名前', 16)}{_r('順', 3)}{_r('最小%', 7)}"
              f"{_r('枠', 4)}{_r('買った日', 12)}{_r('買値', 9)}{_r('状態', 12)}"
              f"{_r('収益%', 8)}{_r('20日目終値%', 13)}{_r('含み%', 8)}")
        names = lrows.set_index(["Date", "Code"])["name"].to_dict()
        for r in lsim.itertuples(index=False):
            state = r.status if r.taken == TAKEN else r.taken
            bought = r.buy_date.date() if pd.notna(r.buy_date) else "-"
            print(f"  {_l(r.Date.date(), 11)}{_l(r.Code[:4], 6)}"
                  f"{_l(_cut(names.get((r.Date, r.Code)) or '', 15), 16)}{_r(r.rank, 3)}"
                  f"{_r(_f(r.p_min, '{:.1f}'), 7)}{_r(_f(r.slot, '{:.0f}'), 4)}"
                  f"{_r(bought, 12)}{_r(_f(r.entry, '{:,.0f}'), 9)}{_r(state, 12)}"
                  f"{_r(_f(r.ret), 8)}{_r(_f(r.ret_close), 13)}{_r(_f(r.ret_now), 8)}")

    print("\n■ 4. 見込みとの比較（OOF を同じ長さで切り出した散らばり）")
    print(f"  OOF 全体: {o_lo.date()}〜{o_hi.date()}、候補 {len(orows):,}件。"
          f"規則どおりの取引 {wt['n']}件、平均 {_f(wt['mean'])}%（SE {_f(wt['se'], '{:.2f}')}）、"
          f"勝率 {_f(wt['win'], '{:.0f}')}%、+20%到達 {_f(wt['hit'], '{:.0f}')}%、"
          f"月あたり {_f(wt['monthly'])}%（概算）")
    osim = simulate(opicks_fwd, bar_days)
    ot = osim[osim["taken"] == TAKEN].assign(year=lambda x: x["Date"].dt.year)
    print(f"  {_l('年', 6)}{_r('取引', 6)}{_r('平均%', 9)}{_r('勝率%', 8)}{_r('月あたり%', 11)}")
    for y, g in ot.groupby("year"):
        ys = trade_stats(g, n_bar_days(bar_days, max(o_lo, pd.Timestamp(f"{y}-01-01")),
                                       min(o_hi, pd.Timestamp(f"{y}-12-31"))))
        print(f"  {_l(y, 6)}{_r(ys['n'], 6)}{_r(_f(ys['mean']), 9)}"
              f"{_r(_f(ys['win'], '{:.0f}'), 8)}{_r(_f(ys['monthly']), 11)}")
    print(f"\n  (a) 選定の流れ: 実運用と同じ {n_live}営業日の期間 × {len(blk_all)}本"
          f"（{STEP}営業日ずつずらす）")
    print(COMPARE_HEAD)
    print(compare_line("1日の発火数", lst["break_mean"],
                       where(lst["break_mean"], blk_all.get("break_mean", [])), "{:.1f}"))
    print(compare_line("基準を満たす割合 %", lst["pass_rate"], wp, "{:.1f}"))
    for a in BOOST:
        v = lst[f"share_{a}"]
        print(compare_line(f"{a} が90以上の割合 %", v,
                           where(v, blk_all.get(f"share_{a}", [])), "{:.1f}"))
    print(compare_line("買う日（月あたり）", lst["buy_days_pm"],
                       where(lst["buy_days_pm"], blk_all.get("buy_days_pm", [])), "{:.1f}"))
    if len(blk_done):
        ts = lst["trade"]
        print(f"\n  (b) 取引の成績: 選定日が {complete_to.date()} 以前の {n_done}営業日 × "
              f"{len(blk_done)}本（OOF でも取引0件の期間 "
              f"{int((blk_done['trade_n'] == 0).sum())}本）")
        print(COMPARE_HEAD)
        for key, name, fmt in (("n", "取引数", "{:.0f}"), ("mean", "1取引の平均 %", "{:+.2f}"),
                               ("win", "勝率 %", "{:.0f}"), ("hit", "+20%到達 %", "{:.0f}"),
                               ("monthly", "月あたり（概算）%", "{:+.2f}")):
            print(compare_line(name, ts[key],
                               where(ts[key], blk_done.get(f"trade_{key}", [])), fmt))
        ap_ = lst["all_pass"]
        print("\n  (c) 枠を気にしない場合（発火8件以上の日の基準通過すべて）")
        print(COMPARE_HEAD)
        print(compare_line("件数", ap_["n"], where(ap_["n"], blk_done.get("all_pass_n", [])),
                           "{:.0f}"))
        print(compare_line("平均 %", ap_["mean"],
                           where(ap_["mean"], blk_done.get("all_pass_mean", []))))
        print(compare_line("勝率 %", ap_["win"],
                           where(ap_["win"], blk_done.get("all_pass_win", [])), "{:.0f}"))
        if lst.get("n_label"):
            print(f"\n  (d) 分離力（ラベル確定 {lst['n_label']}件、ROC-AUC）")
            print(COMPARE_HEAD)
            for a in BOOST:
                v = lst.get(f"auc_{a}")
                print(compare_line(a, v, where(v, blk_done.get(f"auc_{a}", [])), "{:.3f}"))
    else:
        print(f"\n  (b) 取引の成績: 選定日から{HOLD}営業日たった期間がまだ無いので比べない")

    print("\n■ 注意")
    print("  ・取引が少ないうちは、どの数字も幅が広い。1取引の標準偏差は約9pt あるので、"
          "10件の平均でも ±3pt はぶれる")
    print("  ・百分位は OOF 全体の分布で付けている（画面と同じ）。本番モデルは全期間で"
          "学習しているので、スコアの水準が OOF とずれると基準を満たす割合が変わる。"
          "(a) の「90以上の割合」がそのずれの目安（OOF 全体では定義上 10%）")
    print("  ・紙の上の売買（規則どおり）で、実際の約定・手数料・税は含まない。"
          "実際の売買はスプレッドシートの台帳（建値・手仕舞い値の列）に記録する")

    # ---- 保存 ---- #
    os.makedirs(OUT_DIR, exist_ok=True)
    lsim.to_csv(os.path.join(OUT_DIR, "live_track_trades.csv"), index=False)
    ldays.to_csv(os.path.join(OUT_DIR, "live_track_days.csv"), index=False)
    blk_all.to_csv(os.path.join(OUT_DIR, "live_track_blocks_flow.csv"), index=False)
    if len(blk_done):
        blk_done.to_csv(os.path.join(OUT_DIR, "live_track_blocks_trade.csv"), index=False)
    if args.json:
        num = (int, float, np.integer, np.floating)
        summary = {"asOf": str(today), "lastBar": str(last_bar.date()),
                   "live": {"from": str(live_lo.date()), "to": str(live_hi.date()),
                            "days": int(ldays["Date"].nunique()),
                            "buyDays": int(ldays["buy"].sum()),
                            "trades": int(len(t_live)), "tradesComplete": int(len(t_done)),
                            **{k: v for k, v in _flat(lst).items() if isinstance(v, num)}},
                   "oof": {k: v for k, v in _flat(whole).items() if isinstance(v, num)}}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
