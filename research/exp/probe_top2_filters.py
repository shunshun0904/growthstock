#!/usr/bin/env python3
"""
上位2%（百分位98以上）の候補を、時価総額・直近の決算発表・配当・信用倍率・実績PER で絞ると Precision はどうなるか（probe。検証だけ）。

運用者の問い（2026-10-18）「上位2%の中で、時価総額500億以内、直近20日以内に決算発表あり、配当金あり、信用倍率10倍以内、
PER30倍以内のものに限定するとどうなりますか？」。運用者の決定（同日）: PER は実績PER、20日は営業日で数える。
本番の設定・規則・画面には何も書かない。

- 行: 12本の out-of-fold（Release の data-raw。oof.parquet / <名前>_oof.parquet）。Precision は各モデル自身のラベルで数える
- 上位2%: 画面と同じ百分位（同点は数えない。小数1桁に丸める）で LINE 以上。物差しは2通り:
  全体 = OOF 全体の分布（今の画面と同じ）、前向き = その窓より前の窓の OOF の分布（実験88・90 と同じ。窓3〜11）
- 条件（学習データ research/_data/dataset.parquet の列と決算の開示日。どれも基準日 t の引けで分かる値）:
  - 時価総額 CAP_MAX 億円以下: market_cap（未調整終値 × 開示時点の株数、億円。列が無ければ log_market_cap から戻す）
  - 直近 EARN_TD 営業日以内に決算発表: 売上か純利益の実績がある開示（四半期を含む。days_since_disc と同じ開示）の日から
    t までの営業日数が 0〜EARN_TD（営業日は日足のある日。t に開示なら 0、開示日が休日ならその後の最初の営業日が 1）
  - 配当あり: has_dividend == 1（会社予想の年間配当、無ければ実績が 0 より大きい）
  - 信用倍率 CREDIT_MAX 倍以下: credit_ratio（信用買残 ÷ 信用売残。公表日で結合）。売残 0・データ無しは欠測で、満たさない
  - 実績PER PER_MAX 倍以下: per（J-Quants の PER、無ければ株価 ÷ 直近4四半期の1株利益。黒字のときだけ値がある）。
    赤字・欠測は満たさない
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
import lab  # noqa: E402
import models as M  # noqa: E402
import walkforward as WF  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402

#: 上位の線（百分位）。98 = 上位2%
LINE = 98.0
#: 条件（運用者の問いのまま。PER は実績、日数は営業日）
CAP_MAX = 500.0          # 億円
EARN_TD = 20             # 営業日
CREDIT_MAX = 10.0        # 倍
PER_MAX = 30.0           # 倍
FILTERS = ("cap", "earn", "div", "credit", "per")
FILTER_JA = {"cap": f"時価総額 {CAP_MAX:g}億円以下", "earn": f"決算発表から {EARN_TD}営業日以内",
             "div": "配当あり", "credit": f"信用倍率 {CREDIT_MAX:g}倍以下", "per": f"実績PER {PER_MAX:g}倍以下"}
#: 規則の3本（画面と同じ。src/lib/strategy.js の RULE_MODELS）
RULE_MODELS = ("lgbm_k15", "xgb_k15", "cat_k15")
OUTCOME = lab.OUTCOME
MODEL_META = os.path.join(os.path.dirname(HERE), "model", "meta.json")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """正例の割合の 95% の幅（Wilson）。n = 0 なら NaN。"""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def trading_days_since(keys: pd.DataFrame, disc: pd.DataFrame, days: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    (Code, Date) ごとに、Date 以前で一番新しい開示（disc の DiscDate。同じ日を含む）からの営業日数と暦日数。
    営業日数 = days のうち (開示日, Date] に入る日の数（Date に開示なら 0。休日の開示はその後の最初の営業日が 1）。
    開示が無ければ NaN。keys の並びのまま返す。
    """
    left = keys[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    left["Date"] = pd.to_datetime(left["Date"]).astype("datetime64[ns]")
    left["_i"] = np.arange(len(left))
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


def filter_masks(f: pd.DataFrame) -> Dict[str, np.ndarray]:
    """5つの条件の真偽（欠測は満たさない）。"""
    def num(c):
        return pd.to_numeric(f[c], errors="coerce").to_numpy(dtype=float)
    per = num("per")
    with np.errstate(invalid="ignore"):
        return {
            "cap": num("market_cap") <= CAP_MAX,
            "earn": num("earn_td") <= EARN_TD,
            "div": num("has_dividend") == 1.0,
            "credit": num("credit_ratio") <= CREDIT_MAX,
            "per": (per > 0) & (per <= PER_MAX),
        }


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


def cell(y: np.ndarray, r: np.ndarray, m: np.ndarray) -> str:
    """行数・Precision（95% の幅）・平均の実収益。"""
    n = int(m.sum())
    if n == 0:
        return f"{0:>5}行  {'—':>22}  {'—':>7}"
    k = int(np.nansum(y[m]))
    lo, hi = wilson(k, n)
    rr = r[m]
    mr = float(np.nanmean(rr)) * 100.0 if np.isfinite(rr).any() else np.nan      # ret_o1_20 は比率 → %
    return f"{n:>5}行  {k / n * 100:>5.1f}%（{lo * 100:>4.1f}〜{hi * 100:>4.1f}）  {mr:>+6.2f}%"


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
    want = ["Code", "Date", "log_market_cap", "has_dividend", "credit_ratio", "per", "days_since_disc"]
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
    f["earn_td"], f["earn_cal"] = trading_days_since(f, disc, days)
    return f, info


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
    print(f"上位 {100 - args.line:g}%（百分位 {args.line:g} 以上）を5つの条件で絞る（probe。件数・割合だけ）")
    print(f"  OOF {info['oof_rows']:,}行（{f['Date'].min().date()}〜{f['Date'].max().date()}）/ 学習データと突き合わせ "
          f"{info['matched']:,}行 / 時価総額は {info['market_cap_from']} から / 営業日 {info['trading_days']:,}日")
    if info["matched"] != n:
        raise SystemExit("学習データに無い OOF の行がある（学習データとモデルの版が違う）")

    # 確かめ: 開示日の突き合わせが学習データの days_since_disc（暦日、上限400）と一致するか
    dsd = f["days_since_disc"].to_numpy(dtype=float)
    cal = f["earn_cal"].to_numpy(dtype=float)
    ok = np.isfinite(dsd) & (dsd < 400)
    same = ok & np.isfinite(cal) & (np.abs(cal - dsd) < 1e-9)
    print(f"  確かめ: 開示からの暦日数が days_since_disc と一致 {int(same.sum()):,} / {int(ok.sum()):,}行"
          f"（開示が無い行 {int(np.isnan(cal).sum()):,}）")
    if int(same.sum()) != int(ok.sum()):
        raise SystemExit("開示日の突き合わせが学習データと合わない")
    td = f["earn_td"].to_numpy(dtype=float)
    q = np.nanpercentile(td, [0, 25, 50, 75, 100])
    print(f"  開示からの営業日数: 分位 0/25/50/75/100% = {' / '.join(f'{v:.0f}' for v in q)}、"
          f"{EARN_TD}営業日以内 {np.mean(td <= EARN_TD) * 100:.1f}%")

    masks = filter_masks(f)
    masks["all"] = np.logical_and.reduce([masks[c] for c in FILTERS])
    names = {**FILTER_JA, "all": "5つすべて"}
    r = f[OUTCOME].to_numpy(dtype=float)

    print("\n■ 0. 条件だけ（モデルの線なし。OOF の全行）: 行の割合と、ラベルごとの正例率")
    for c in FILTERS:
        src = {"cap": "market_cap", "earn": "earn_td", "div": "has_dividend", "credit": "credit_ratio", "per": "per"}[c]
        miss = f[src].isna().mean() * 100
        print(f"  {names[c]:<22} 満たす {masks[c].mean() * 100:>5.1f}%（列の欠測 {miss:>4.1f}%）")
    print(f"  {'5つすべて':<22} 満たす {masks['all'].mean() * 100:>5.1f}%")
    for lab_algo, ja in ((M.BASELINE, "1.2σ"), ("lgbm_k15", "1.5σ"), ("lgbm_k18", "1.8σ")):
        y = f[f"label_{lab_algo}"].to_numpy(dtype=float)
        everyone = np.ones(n, dtype=bool)
        print(f"  {ja} のラベル: 全行 {cell(y, r, everyone)} / 5つすべて {cell(y, r, masks['all'])}")

    yardsticks = {}
    for a in E79.MODELS:
        s = f[f"score_{a}"].to_numpy(dtype=float)
        pf, _ = pct_forward(f["Date"], s, meta)
        yardsticks[a] = (pct_all(s), pf)
    fwd_ok = np.isfinite(yardsticks[M.BASELINE][1])
    fd = f.loc[fwd_ok, "Date"]
    months = (fd.max() - fd.min()).days / 30.4375
    print(f"\n  前向きの物差しが付く行 {int(fwd_ok.sum()):,}（{fd.min().date()}〜{fd.max().date()}、{months:.1f}か月）")

    print(f"\n■ 1. モデルごと: 百分位 {args.line:g} 以上の行を条件で絞る（Precision は各モデル自身のラベル。"
          f"実収益は {OUTCOME} の平均）")
    print(f"  {'条件':<22}  {'全体の物差し: 行・Precision（95%の幅）・実収益':<44}  前向きの物差し")
    for a in E79.MODELS:
        y = f[f"label_{a}"].to_numpy(dtype=float)
        pa, pf = yardsticks[a]
        top_a, top_f = pa >= args.line, pf >= args.line
        print(f"\n  [{a}] 正例率 {np.mean(y) * 100:.1f}%")
        for c in ("none",) + FILTERS + ("all",):
            m = masks.get(c, np.ones(n, dtype=bool))
            print(f"  {('（条件なし）' if c == 'none' else names[c]):<22}  {cell(y, r, top_a & m)}  |  {cell(y, r, top_f & m)}")
        print(f"  {'前向き・月あたりの行':<22}  条件なし {int(top_f.sum()) / months:.1f} / 5つすべて "
              f"{int((top_f & masks['all']).sum()) / months:.1f}")

    print(f"\n■ 2. 規則の3本（{' / '.join(RULE_MODELS)}）がすべて線以上（1.5σ のラベル）")
    y = f["label_lgbm_k15"].to_numpy(dtype=float)
    for line in sorted({95.0, args.line}):
        all3_a = np.logical_and.reduce([yardsticks[a][0] >= line for a in RULE_MODELS])
        all3_f = np.logical_and.reduce([yardsticks[a][1] >= line for a in RULE_MODELS])
        print(f"\n  3本とも {line:g} 以上")
        for c in ("none",) + FILTERS + ("all",):
            m = masks.get(c, np.ones(n, dtype=bool))
            print(f"  {('（条件なし）' if c == 'none' else names[c]):<22}  {cell(y, r, all3_a & m)}  |  {cell(y, r, all3_f & m)}")
        print(f"  {'前向き・月あたりの行':<22}  条件なし {int(all3_f.sum()) / months:.1f} / 5つすべて "
              f"{int((all3_f & masks['all']).sum()) / months:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
