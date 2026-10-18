#!/usr/bin/env python3
"""
上位2%（百分位98以上）の候補を、時価総額・決算発表・配当・信用倍率・実績PER で絞ると Precision はどうなるか（probe。検証だけ）。

運用者の問い（2026-10-18）「上位2%の中で、時価総額500億以内、直近20日以内に決算発表あり、配当金あり、信用倍率10倍以内、
PER30倍以内のものに限定するとどうなりますか？」。運用者の決定（同日）: PER は実績PER、20日は営業日で数える、5つは AND。
続けて「目的変数の定義ではなく、売り指値到達の+10%に到達したかどうかで、Precisionを算出すると、80%は行くのではないか」
→ ラベルの Precision に並べて、+10% の指値に届いた割合（実験83 の tp10c と同じ判定）も数える。
さらに「決算発表から20営業日以内ではなく、目的変数の期間内に、決算発表があるかです」→ 決算の条件を、基準日 t の翌営業日から
目的変数の期間（RISE_HORIZON = 20営業日）の終わりまでに決算発表があるか、に直した（最初の版は t までの20営業日に開示があるか
だった。結果は docs/MODEL_ADOPTION_RULES.md §65）。本番の設定・規則・画面には何も書かない。

- 行: 12本の out-of-fold（Release の data-raw。oof.parquet / <名前>_oof.parquet）。Precision は2通り:
  ラベル = 各モデル自身のラベル（到達しきい値に届いたか）、+10% = 翌営業日の寄りで買い、1〜20営業日目（1 = 買った日）の
  高値が買値の 1.10倍以上になったか（実験83 の hit10。live_track.forward）。1取引の平均は +10% に届けば +10%、届かなければ
  20営業日目の終値で売った収益（tp10c。手数料・税・すべりは含まない）
- 上位2%: 画面と同じ百分位（同点は数えない。小数1桁に丸める）で LINE 以上。物差しは2通り:
  全体 = OOF 全体の分布（今の画面と同じ）、前向き = その窓より前の窓の OOF の分布（実験88・90 と同じ。窓3〜11）
- 条件（学習データ research/_data/dataset.parquet の列と、決算の開示日・発表予定日）:
  - 時価総額 CAP_MAX 億円以下: market_cap（未調整終値 × 開示時点の株数、億円。列が無ければ log_market_cap から戻す）
  - 目的変数の期間内に決算発表あり: 発表の日（休日ならその後の最初の営業日）が t の 1〜RISE_HORIZON 営業日後。2通りで数える:
      実際の開示日 = 売上か純利益の実績がある開示（四半期を含む。days_since_disc と同じ開示）の日。t より先の事実なので、
                    t の時点では分からない（予定日がずれた分だけ運用とは違う）
      予定日       = J-Quants の決算発表予定日（earndate）のうち、t までに公表済み（PubDate ≤ t）で t より後の予定の一番早いもの
                    （特徴量 days_to_earn と同じ規則。t の時点で分かる。予定が出ていない銘柄は満たさない）
  - 配当あり: has_dividend == 1（会社予想の年間配当、無ければ実績が 0 より大きい）
  - 信用倍率 CREDIT_MAX 倍以下: credit_ratio（信用買残 ÷ 信用売残。公表日で結合）。売残 0・データ無しは欠測で、満たさない
  - 実績PER PER_MAX 倍以下: per（J-Quants の PER、無ければ株価 ÷ 直近4四半期の1株利益。黒字のときだけ値がある）。
    赤字・欠測は満たさない
  - 5つすべて（AND）は、決算の条件を実際の開示日で数えた版と予定日で数えた版の2つ
- 出すのは件数・割合・平均だけ（銘柄名・コード・銘柄ごとの値は出さない。Actions のログは公開）

  python3 research/exp/probe_top2_filters.py [--line 98] [--data-dir research/_data]
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import build_dataset as B  # noqa: E402
import extra_features as XF  # noqa: E402
import lab  # noqa: E402
import models as M  # noqa: E402
import walkforward as WF  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import live_track as LT  # noqa: E402

#: 上位の線（百分位）。98 = 上位2%
LINE = 98.0
#: 条件（運用者の問いのまま。PER は実績、決算は目的変数の期間内）
CAP_MAX = 500.0          # 億円
EARN_WINDOW = B.RISE_HORIZON   # 目的変数の期間（営業日）。+10% の判定の期間（LT.HOLD）と同じ 20
CREDIT_MAX = 10.0        # 倍
PER_MAX = 30.0           # 倍
#: days_to_earn の頭打ち（extra_features.earnings_ahead の clip）
DAYS_TO_EARN_CLIP = 120
FILTERS = ("cap", "earn_act", "earn_sch", "div", "credit", "per")
#: 5つすべて（AND）。決算の条件だけ2通り
ALL_SETS = {"all_act": ("cap", "earn_act", "div", "credit", "per"),
            "all_sch": ("cap", "earn_sch", "div", "credit", "per")}
FILTER_JA = {"cap": f"時価総額 {CAP_MAX:g}億円以下",
             "earn_act": "期間内に決算（実際の開示日）", "earn_sch": "期間内に決算（予定日）",
             "div": "配当あり", "credit": f"信用倍率 {CREDIT_MAX:g}倍以下", "per": f"実績PER {PER_MAX:g}倍以下",
             "all_act": "5つすべて（実際の開示日）", "all_sch": "5つすべて（予定日）"}
#: 規則の3本（画面と同じ。src/lib/strategy.js の RULE_MODELS）
RULE_MODELS = ("lgbm_k15", "xgb_k15", "cat_k15")
OUTCOME = lab.OUTCOME
#: 売りの指値（%）。運用と同じ（live_track.TAKE_PROFIT）
TP = LT.TAKE_PROFIT
MODEL_META = os.path.join(os.path.dirname(HERE), "model", "meta.json")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """正例の割合の 95% の幅（Wilson）。n = 0 なら NaN。"""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def _keys(keys: pd.DataFrame) -> pd.DataFrame:
    left = keys[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    left["Date"] = pd.to_datetime(left["Date"]).astype("datetime64[ns]")
    left["_i"] = np.arange(len(left))
    return left


def trading_days_since(keys: pd.DataFrame, disc: pd.DataFrame, days: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    (Code, Date) ごとに、Date 以前で一番新しい開示（disc の DiscDate。同じ日を含む）からの営業日数と暦日数。
    営業日数 = days のうち (開示日, Date] に入る日の数（Date に開示なら 0。休日の開示はその後の最初の営業日が 1）。
    開示が無ければ NaN。keys の並びのまま返す。暦日数は学習データの days_since_disc との突き合わせに使う。
    """
    left = _keys(keys)
    d = disc[["Code", "DiscDate"]].copy()
    d["Code"] = d["Code"].astype(str)
    d["DiscDate"] = pd.to_datetime(d["DiscDate"]).astype("datetime64[ns]")
    d = d.dropna(subset=["DiscDate"]).sort_values("DiscDate")
    m = pd.merge_asof(left.sort_values("Date"), d, left_on="Date", right_on="DiscDate", by="Code",
                      direction="backward", allow_exact_matches=True).sort_values("_i")
    t = m["Date"].to_numpy(dtype="datetime64[ns]")
    dd = m["DiscDate"].to_numpy(dtype="datetime64[ns]")
    has = ~np.isnat(dd)
    days = np.sort(np.asarray(days, dtype="datetime64[ns]"))
    td = np.full(len(m), np.nan)
    cal = np.full(len(m), np.nan)
    td[has] = (np.searchsorted(days, t[has], side="right") - np.searchsorted(days, dd[has], side="right")).astype(float)
    cal[has] = ((t[has] - dd[has]) / np.timedelta64(1, "D")).astype(float)
    return td, cal


def trading_days_ahead(dates: np.ndarray, events: np.ndarray, days: np.ndarray) -> np.ndarray:
    """
    基準日 dates から発表の日 events まで何営業日先か。発表の日が休日ならその後の最初の営業日で数える
    （t の翌営業日なら 1、t の次の金曜の後の土曜なら翌月曜の分）。events が NaT、または営業日の表の外なら NaN。
    """
    days = np.sort(np.asarray(days, dtype="datetime64[ns]"))
    t = np.asarray(dates, dtype="datetime64[ns]")
    e = np.asarray(events, dtype="datetime64[ns]")
    out = np.full(len(t), np.nan)
    ok = ~np.isnat(e)
    pe = np.searchsorted(days, e[ok], side="left")
    pt = np.searchsorted(days, t[ok], side="left")
    k = (pe - pt).astype(float)
    k[pe >= len(days)] = np.nan                       # 営業日の表より先（数えられない）
    out[ok] = k
    return out


def next_disclosure(keys: pd.DataFrame, disc: pd.DataFrame) -> np.ndarray:
    """(Code, Date) ごとに、Date より後（同じ日は含まない）で一番早い開示の日。無ければ NaT。keys の並び。"""
    left = _keys(keys)
    d = disc[["Code", "DiscDate"]].copy()
    d["Code"] = d["Code"].astype(str)
    d["DiscDate"] = pd.to_datetime(d["DiscDate"]).astype("datetime64[ns]")
    d = d.dropna(subset=["DiscDate"]).sort_values("DiscDate")
    m = pd.merge_asof(left.sort_values("Date"), d, left_on="Date", right_on="DiscDate", by="Code",
                      direction="forward", allow_exact_matches=False).sort_values("_i")
    return m["DiscDate"].to_numpy(dtype="datetime64[ns]")


def next_scheduled(keys: pd.DataFrame, sched: pd.DataFrame, strict: bool = True) -> np.ndarray:
    """
    (Code, Date) ごとに、Date までに公表済み（PubDate ≤ Date）の決算発表予定（SchDate）のうち、Date より後
    （strict=False なら Date を含む。特徴量 days_to_earn と同じ）で一番早いもの。無ければ NaT。keys の並び。
    """
    left = _keys(keys)
    out = np.full(len(left), np.datetime64("NaT"), dtype="datetime64[ns]")
    if sched is None or not len(sched):
        return out
    s = sched[["Code", "PubDate", "SchDate"]].copy()
    s["Code"] = s["Code"].astype(str)
    for c in ("PubDate", "SchDate"):
        s[c] = pd.to_datetime(s[c], errors="coerce").astype("datetime64[ns]")
    s = s.dropna(subset=["PubDate", "SchDate"])
    j = left.merge(s, on="Code", how="inner")
    after = j["SchDate"] > j["Date"] if strict else j["SchDate"] >= j["Date"]
    j = j[(j["PubDate"] <= j["Date"]) & after]
    nxt = j.groupby("_i")["SchDate"].min()
    out[nxt.index.to_numpy()] = nxt.to_numpy(dtype="datetime64[ns]")
    return out


def filter_masks(f: pd.DataFrame) -> Dict[str, np.ndarray]:
    """条件の真偽（欠測は満たさない）と、5つすべて（AND）の2通り。"""
    def num(c):
        return pd.to_numeric(f[c], errors="coerce").to_numpy(dtype=float)
    per = num("per")
    with np.errstate(invalid="ignore"):
        ea, es = num("earn_ahead_act"), num("earn_ahead_sch")
        m = {
            "cap": num("market_cap") <= CAP_MAX,
            "earn_act": (ea >= 1) & (ea <= EARN_WINDOW),
            "earn_sch": (es >= 1) & (es <= EARN_WINDOW),
            "div": num("has_dividend") == 1.0,
            "credit": num("credit_ratio") <= CREDIT_MAX,
            "per": (per > 0) & (per <= PER_MAX),
        }
    for name, parts in ALL_SETS.items():
        m[name] = np.logical_and.reduce([m[c] for c in parts])
    return m


def pct_all(score: np.ndarray) -> np.ndarray:
    """OOF 全体の分布での百分位（画面と同じ: 同点は数えない・小数1桁）。"""
    s = np.asarray(score, dtype=float)
    return E82.rounded(E79.pct_within(np.sort(s), s))


def pct_forward(dates: pd.Series, score: np.ndarray, meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    前向きの百分位（実験79・88・90 と同じ）: テスト窓ごとに、それより前の窓の OOF（窓の始まりの手前 RISE_HORIZON 営業日
    ぶんは外す）の分布で付ける。前の窓が MIN_TRAIN_WINDOWS 未満の窓は NaN。窓の区切りは学習データの最初と最後の日
    （meta の trainFrom / trainTo。walkforward.make_folds は日付の最小・最大だけを見る）で決める。戻りは (百分位, 窓の番号)。
    """
    folds = E79.make_windows(pd.Series(pd.to_datetime([meta["trainFrom"], meta["trainTo"]])))
    w = E79.assign_windows(dates, folds)
    if (w == 0).any():
        raise SystemExit(f"どの窓にも入らない OOF の行がある: {(w == 0).sum()}行（窓の区切りが本番と違う）")
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    start = {f.index: pd.Timestamp(f.test_start) for f in folds}
    gap = pd.Timedelta(days=int(round(B.RISE_HORIZON * WF.TRADING_TO_CALENDAR)))
    present = sorted(int(x) for x in np.unique(w))
    s = np.asarray(score, dtype=float)
    out = np.full(len(s), np.nan)
    for ww in present:
        prior = [x for x in present if x < ww]
        if len(prior) < E79.MIN_TRAIN_WINDOWS:
            continue
        tr = np.isin(w, prior) & (d <= np.datetime64(start[ww] - gap))
        te = w == ww
        if tr.sum() < 200:
            continue
        out[te] = E82.rounded(E79.pct_within(np.sort(s[tr]), s[te]))
    return out, w


def cell(y: np.ndarray, hit: np.ndarray, tp: np.ndarray, m: np.ndarray) -> str:
    """行数・ラベルの Precision（95% の幅）・+10% に届いた割合（95% の幅）・1取引の平均（tp10c）。"""
    n = int(m.sum())
    if n == 0:
        return f"{0:>5}行  {'—':>20}  {'—':>20}  {'—':>7}"
    k = int(np.nansum(y[m]))
    h = int(np.sum(hit[m]))
    lo, hi = wilson(k, n)
    hlo, hhi = wilson(h, n)
    t = tp[m]
    mt = float(np.nanmean(t)) if np.isfinite(t).any() else np.nan
    return (f"{n:>5}行  {k / n * 100:>5.1f}%（{lo * 100:>4.1f}〜{hi * 100:>4.1f}）  "
            f"{h / n * 100:>5.1f}%（{hlo * 100:>4.1f}〜{hhi * 100:>4.1f}）  {mt:>+6.2f}%")


def load_frame(data_dir: str) -> Tuple[pd.DataFrame, dict]:
    """12本の OOF（Code, Date, score_<名前>, label_<名前>, 実収益）に、条件の列を (Code, Date) で付けた表。"""
    wide = None
    for a in E79.MODELS:
        p = E79.oof_path(a, data_dir)
        if not os.path.exists(p):
            raise SystemExit(f"{a} の out-of-fold がありません（{os.path.basename(p)}）")
        o = pd.read_parquet(p)
        o["Code"] = o["Code"].astype(str)
        o["Date"] = pd.to_datetime(o["Date"]).astype("datetime64[ns]")
        if o.duplicated(["Code", "Date"]).any():
            raise SystemExit(f"{a} の out-of-fold に (Code, Date) の重複がある")
        keep = o[["Code", "Date", "score", "label"]].rename(columns={"score": f"score_{a}", "label": f"label_{a}"})
        if a == M.BASELINE:
            keep[OUTCOME] = o[OUTCOME].to_numpy()
        n0 = 0 if wide is None else len(wide)
        wide = keep if wide is None else wide.merge(keep, on=["Code", "Date"], how="inner")
        if n0 and len(wide) != n0:
            raise SystemExit(f"{a} の out-of-fold の行がほかのモデルとそろわない（{n0} -> {len(wide)}）")
    info = {"oof_rows": int(len(wide))}

    ds_path = os.path.join(data_dir, "dataset.parquet")
    import pyarrow.parquet as pq
    have = set(pq.read_schema(ds_path).names)
    want = ["Code", "Date", "log_market_cap", "has_dividend", "credit_ratio", "per", "days_since_disc", "days_to_earn"]
    if "market_cap" in have:
        want.append("market_cap")
    miss = [c for c in want if c not in have]
    if miss:
        raise SystemExit(f"学習データに列が無い: {miss}")
    ds = pd.read_parquet(ds_path, columns=want)
    ds["Code"] = ds["Code"].astype(str)
    ds["Date"] = pd.to_datetime(ds["Date"]).astype("datetime64[ns]")
    if ds.duplicated(["Code", "Date"]).any():
        raise SystemExit("学習データに (Code, Date) の重複がある")
    if "market_cap" not in ds.columns:
        ds["market_cap"] = np.expm1(ds["log_market_cap"].to_numpy(dtype=float))
        info["market_cap_from"] = "log_market_cap"
    else:
        info["market_cap_from"] = "market_cap"
    f = wide.merge(ds, on=["Code", "Date"], how="left", indicator=True)
    info["matched"] = int((f["_merge"] == "both").sum())
    f = f.drop(columns="_merge")

    fins = B.load_parts("fins", data_dir)
    disc = fins.loc[fins[["Sales", "NP"]].notna().any(axis=1), ["Code", "DiscDate"]]
    bars = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not bars:
        raise SystemExit("日足（bars_*.parquet）が無い")
    days = np.unique(np.concatenate([pd.to_datetime(pd.read_parquet(p, columns=["Date"])["Date"]).to_numpy(
        dtype="datetime64[ns]") for p in bars]))
    info["trading_days"] = int(len(days))
    t = f["Date"].to_numpy(dtype="datetime64[ns]")
    # 確かめ用: t 以前の開示からの暦日数（days_since_disc と同じになるはず）
    _, f["disc_cal"] = trading_days_since(f, disc, days)
    # 目的変数の期間内の決算: 実際の開示日 / t までに公表済みの予定日
    f["earn_ahead_act"] = trading_days_ahead(t, next_disclosure(f, disc), days)
    sched = XF._read("earndate", data_dir, columns=["Code", "PubDate", "SchDate"])
    info["sched_rows"] = int(len(sched))
    f["earn_ahead_sch"] = trading_days_ahead(t, next_scheduled(f, sched, strict=True), days)
    # 確かめ用: 特徴量 days_to_earn と同じ規則（t を含む）での暦日数
    ge = next_scheduled(f, sched, strict=False)
    with np.errstate(invalid="ignore"):
        f["sched_cal"] = np.clip((ge - t) / np.timedelta64(1, "D"), 0, DAYS_TO_EARN_CLIP)
    # +10% の指値の出口（実験83 と同じ。1 = 買った日〜20営業日目の高値）。確かめに使う 5日平均は ret_o1_20 と一致するはず
    bars_px = LT.load_bars(data_dir, start=f["Date"].min() - pd.Timedelta(days=10))
    f, chk = E83.attach_exits(f, bars_px, targets=(TP,))
    info["exit_check"] = chk
    return f, info


def check_or_stop(label: str, want: np.ndarray, got: np.ndarray, clip_at: float) -> None:
    """want（学習データの列）と got（ここで数え直した値）が、want が頭打ちより下の行で一致し、欠測もそろうか。"""
    ok = np.isfinite(want) & (want < clip_at)
    same = ok & np.isfinite(got) & (np.abs(got - want) < 1e-9)
    nan_mis = int((np.isnan(want) != np.isnan(got)).sum())
    print(f"  確かめ: {label} 一致 {int(same.sum()):,} / {int(ok.sum()):,}行（欠測の食い違い {nan_mis:,}行）")
    if int(same.sum()) != int(ok.sum()) or nan_mis:
        raise SystemExit(f"{label} が学習データと合わない")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="上位2% を5つの条件で絞った Precision（probe）")
    ap.add_argument("--data-dir", default=lab.DATA_DIR)
    ap.add_argument("--line", type=float, default=LINE, help="上位の線（百分位）。98 = 上位2%%")
    args = ap.parse_args(argv)
    with open(MODEL_META, encoding="utf-8") as fh:
        meta = json.load(fh)

    f, info = load_frame(args.data_dir)
    n = len(f)
    print("=" * 100)
    print(f"上位 {100 - args.line:g}%（百分位 {args.line:g} 以上）を5つの条件（AND）で絞る（probe。件数・割合だけ）")
    print(f"  OOF {info['oof_rows']:,}行（{f['Date'].min().date()}〜{f['Date'].max().date()}）/ 学習データと突き合わせ "
          f"{info['matched']:,}行 / 時価総額は {info['market_cap_from']} から / 営業日 {info['trading_days']:,}日 / "
          f"決算発表予定 {info['sched_rows']:,}件")
    if info["matched"] != n:
        raise SystemExit("学習データに無い OOF の行がある（学習データとモデルの版が違う）")

    # 確かめ: 開示日・予定日の突き合わせが、学習データの days_since_disc（暦日、上限400）・days_to_earn（暦日、上限120）と合うか
    check_or_stop("開示からの暦日数が days_since_disc と", f["days_since_disc"].to_numpy(dtype=float),
                  f["disc_cal"].to_numpy(dtype=float), 400)
    check_or_stop("次の予定までの暦日数が days_to_earn と", f["days_to_earn"].to_numpy(dtype=float),
                  f["sched_cal"].to_numpy(dtype=float), DAYS_TO_EARN_CLIP)
    chk = info["exit_check"]
    k = E83.tp_key(TP)
    print(f"  確かめ: 20営業日目の5日平均が ret_o1_20 と一致 {chk['hold5_match']:,} / {chk['hold5_both']:,}行、"
          f"20営業日目の終値がある行 {chk['with_close']:,} / {chk['rows']:,}、+{TP:g}% に届いた行 {chk['hit_share'][k]:.1f}%")
    if chk["hold5_match"] != chk["hold5_both"]:
        raise SystemExit("出口の値動きが学習データの実収益と合わない")

    masks = filter_masks(f)
    ea, es = masks["earn_act"], masks["earn_sch"]
    sch_known = np.isfinite(f["earn_ahead_sch"].to_numpy(dtype=float))
    print(f"  期間内（1〜{EARN_WINDOW}営業日先）に決算: 実際の開示日で {ea.mean() * 100:.1f}% / 予定日で {es.mean() * 100:.1f}%"
          f"（t までに公表済みの次の予定がある行 {sch_known.mean() * 100:.1f}%）。両方 {np.mean(ea & es) * 100:.1f}%・"
          f"実際だけ {np.mean(ea & ~es) * 100:.1f}%・予定だけ {np.mean(~ea & es) * 100:.1f}%")

    hit = f[f"hit{E83.tp_key(TP)}"].to_numpy(dtype=bool)
    tp = pd.to_numeric(f[f"tp{E83.tp_key(TP)}c"], errors="coerce").to_numpy(dtype=float)
    rows = ("none",) + FILTERS + tuple(ALL_SETS)

    def name(c):
        return "（条件なし）" if c == "none" else FILTER_JA[c]

    print("\n■ 0. 条件だけ（モデルの線なし。OOF の全行）: 行の割合と、ラベルごとの正例率・+10% に届いた割合")
    src = {"cap": "market_cap", "earn_act": "earn_ahead_act", "earn_sch": "earn_ahead_sch", "div": "has_dividend",
           "credit": "credit_ratio", "per": "per"}
    for c in FILTERS:
        print(f"  {name(c):<26} 満たす {masks[c].mean() * 100:>5.1f}%（列の欠測 {f[src[c]].isna().mean() * 100:>4.1f}%）")
    for c in ALL_SETS:
        print(f"  {name(c):<26} 満たす {masks[c].mean() * 100:>5.1f}%")
    everyone = np.ones(n, dtype=bool)
    for lab_algo, ja in ((M.BASELINE, "1.2σ"), ("lgbm_k15", "1.5σ"), ("lgbm_k18", "1.8σ")):
        y = f[f"label_{lab_algo}"].to_numpy(dtype=float)
        print(f"  {ja} のラベル: 全行 {cell(y, hit, tp, everyone)}")
        for c in ("earn_act", "earn_sch") + tuple(ALL_SETS):
            print(f"    {name(c):<26} {cell(y, hit, tp, masks[c])}")

    yardsticks = {}
    for a in E79.MODELS:
        s = f[f"score_{a}"].to_numpy(dtype=float)
        pf, _ = pct_forward(f["Date"], s, meta)
        yardsticks[a] = (pct_all(s), pf)
    fwd_ok = np.isfinite(yardsticks[M.BASELINE][1])
    fd = f.loc[fwd_ok, "Date"]
    months = (fd.max() - fd.min()).days / 30.4375
    print(f"\n  前向きの物差しが付く行 {int(fwd_ok.sum()):,}（{fd.min().date()}〜{fd.max().date()}、{months:.1f}か月）")

    def per_month(top):
        return " / ".join(f"{name(c)} {int((top & masks[c]).sum()) / months:.1f}" for c in ALL_SETS)

    print(f"\n■ 1. モデルごと: 百分位 {args.line:g} 以上の行を条件で絞る")
    print(f"  各セル: 行 / ラベルの Precision（95%の幅）/ +{TP:g}% に届いた割合（95%の幅）/ 1取引の平均"
          f"（+{TP:g}% の指値・届かなければ20営業日目の終値）。左が全体の物差し、右が前向きの物差し")
    for a in E79.MODELS:
        y = f[f"label_{a}"].to_numpy(dtype=float)
        pa, pf = yardsticks[a]
        top_a, top_f = pa >= args.line, pf >= args.line
        print(f"\n  [{a}] 正例率 {np.mean(y) * 100:.1f}%")
        for c in rows:
            m = masks.get(c, everyone)
            print(f"  {name(c):<26}  {cell(y, hit, tp, top_a & m)}  |  {cell(y, hit, tp, top_f & m)}")
        print(f"  前向き・月あたりの行: 条件なし {int(top_f.sum()) / months:.1f} / {per_month(top_f)}")

    print(f"\n■ 2. 規則の3本（{' / '.join(RULE_MODELS)}）がすべて線以上（1.5σ のラベル）")
    y = f["label_lgbm_k15"].to_numpy(dtype=float)
    for line in sorted({95.0, args.line}):
        all3_a = np.logical_and.reduce([yardsticks[a][0] >= line for a in RULE_MODELS])
        all3_f = np.logical_and.reduce([yardsticks[a][1] >= line for a in RULE_MODELS])
        print(f"\n  3本とも {line:g} 以上")
        for c in rows:
            m = masks.get(c, everyone)
            print(f"  {name(c):<26}  {cell(y, hit, tp, all3_a & m)}  |  {cell(y, hit, tp, all3_f & m)}")
        print(f"  前向き・月あたりの行: 条件なし {int(all3_f.sum()) / months:.1f} / {per_month(all3_f)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
