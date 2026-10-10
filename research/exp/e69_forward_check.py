#!/usr/bin/env python3
"""
実験69: 先に決めた仮説を、決めたあとに増えたデータだけで確かめ直す（前向きの確かめ直し）。

運用者の了承（2026-10-10）:
  「EDINET: 平均年収の前年比と販管費の2年変化を見ます。12月に新しく EDINET が付く銘柄だけで同じ検定を
    回し、同じ向きが出るかを確かめます。… 日証金: 融資残高の20日変化と逆日歩を見ます。2027年以降の
    新しい窓でも同じ向きが出るかを確かめます。」

なぜ要るか
--------
実験22（EDINET 523列）と実験66（日証金 測れた16列）は、たくさんの列を同じデータで並べて上位を選んだ。
選んだ列の z は「同じデータで選んだ」ぶん高く出る（選択の楽観）。選んだあとに増えたデータだけで
同じ検定を回せば、この楽観は入らない。
  EDINET  2026-10-10 にまだ財務が無かった銘柄だけ（銘柄が重ならない。取得の記録の first_ok_at で分ける）
  日証金   2026-11-01 以降の日付の行だけ（暦月の窓。10/10 に見た行は入らない）

何を先に決めたか（HYPOTHESES。結果を見てから変えない。docs/FEATURE_IDEAS_EDINET.md・docs/DATA_JSF.md）
--------
- 列と向き。物差しは実験20 の screen と同じ「窓ごとに、列が上位10% の行の ret_o1_20 の平均 − 窓全体の平均」
- 判定（列ごと。決めた向きでの片側 t 検定。自由度は窓の数 − 1）
    再現          平均が決めた向き かつ 片側 p < 0.05
    向きは同じ     平均が決めた向き だが p ≥ 0.05（決められない。採用しない）
    逆向き         平均が逆（候補から外す）
    測れない       測れた窓が足りない
  SE は窓の並びの隣どうしの相関（正のときだけ）で広げる。隣の窓とは結果の期間（20営業日）が重なり、
  地合いも続くので、広げないと t が甘くなる（日証金の月の窓で効く）
- 「再現」でも本番には入れない。その列だけを足した §7 の A/B（実験67・68 と同じ作り）に進むだけ
- 列の作りも 10/10 のまま使う（物差しは順位だけなので単調な変換は効かないが、順位の変わる作り直しをしたらやり直し）
- 日証金は 2026-11 〜 2027-10 の12か月がそろうまでは「途中経過」で、判定に使わない

  python3 research/exp/e69_forward_check.py --source edinet   # 2026-12-11 以降（12月の取得が終わってから）
  python3 research/exp/e69_forward_check.py --source jsf      # 2027-12-06 以降（途中経過は 2027-06）

公開ログに出すのは件数・窓ごとの集計・判定だけ（銘柄名・銘柄ごとの値は出さない）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import walkforward as WF  # noqa: E402
from e20_annual_trajectory import OUTCOME, chance_hits, screen, window_stats  # noqa: E402
from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS  # noqa: E402

OUT_DIR = os.path.join(lab.DATA_DIR, "oof")
DECIDED = "2026-10-10"
ALPHA = 0.05

#: 先に決めた仮説。結果を見てから変えない（変えるなら新しい日付で決め直し、別の確かめ直しにする）
HYPOTHESES: Dict[str, dict] = {
    "edinet": {
        # 列 -> 決めた向き（+1: 高いほど良い / −1: 低いほど良い）
        "features": {"ed_avg_annual_salary_yoy1": +1, "ed_sga_chg2y": -1},
        "reference": {},
        # 2026-10-10 の実験22（1,670社・15,309行・窓11本）での値: (超過 pt, z, 超過が正の窓)
        "origin": {"ed_avg_annual_salary_yoy1": (+0.84, +2.94, "正10/11"),
                   "ed_sga_chg2y": (-0.98, -2.70, "正2/11")},
        # 取得の記録（edinet_manifest.json）で first_ok_at がこれ以降の銘柄だけを使う
        "new_since": "2026-10-11T00:00:00+09:00",
        # 2026-10-10 の時点で取れていた社数（run 38012202750 の [state]、実験22 の「EDINET 取得済み」）。
        # 記録から数え直してこれと合わなければ、記録が欠けているので止める
        "old_companies": 1670,
        "min_windows": 8,
        "run_after": "2026-12-11",
    },
    "jsf": {
        "features": {"jsf_loan_chg20_v": +1, "jsf_fee_days20": -1},
        # 逆日歩のもう1本。ほぼ同じ列なので判定には数えず、並べて見るだけ
        "reference": {"jsf_fee_max20": -1},
        # 2026-10-09 の実験66（run 37888171436。9,540〜9,590行・半年の窓6本）での値
        "origin": {"jsf_loan_chg20_v": (+0.87, +3.05, "正6/6"),
                   "jsf_fee_days20": (-0.80, -2.75, "正1/6"),
                   "jsf_fee_max20": (-0.92, -2.77, "正1/6")},
        "start": "2026-11-01",
        "end": "2027-10-31",
        "min_windows": 9,
        "run_after": "2027-12-06",
    },
}

LABELS = {"pass": "再現", "weak": "向きは同じ（弱い）", "reverse": "逆向き", "short": "測れない"}
#: 列全体の数え上げに要る行（screen が列ごとに要る数と同じ）
FAMILY_MIN_ROWS = 500


# ---------------------------------------------------------------------- #
# 判定
# ---------------------------------------------------------------------- #

def lag1_autocorr(e: np.ndarray) -> float:
    """窓の並びの隣どうしの相関（標本の lag-1 自己相関）。3本未満なら 0。"""
    e = np.asarray(e, dtype=float)
    if len(e) < 3:
        return 0.0
    d = e - e.mean()
    den = float((d * d).sum())
    return float((d[:-1] * d[1:]).sum() / den) if den > 0 else 0.0


def se_inflation(rho: float, n: int) -> float:
    """
    隣どうしの相関 rho（AR(1) とみなす）のときの、平均の分散の倍率
    1 + 2 Σ_{k=1}^{n−1} (1 − k/n) rho^k。rho は 0〜0.9 に収める（負は広げない＝保守側）。
    """
    r = min(max(float(rho), 0.0), 0.9)
    return 1.0 + 2.0 * sum((1.0 - k / n) * r ** k for k in range(1, n))


def verdict(edges, sign: int, min_windows: int, alpha: float = ALPHA) -> dict:
    """
    窓ごとの超過（pt）を、決めた向き sign（+1 / −1）で片側 t 検定する。
    戻り値: n, mean, se, rho, se_adj, t, p, agree（決めた向きの窓）, label（LABELS のキー）
    """
    from scipy import stats

    e = np.asarray([v for v in edges if np.isfinite(v)], dtype=float)
    n = len(e)
    out = {"n": n, "mean": float(e.mean()) if n else np.nan, "agree": int((np.sign(e) == sign).sum())}
    if n < 2:
        return {**out, "se": np.nan, "rho": np.nan, "se_adj": np.nan, "t": np.nan, "p": np.nan,
                "label": "short"}
    se = float(e.std(ddof=1) / np.sqrt(n))
    rho = lag1_autocorr(e)
    se_adj = se * np.sqrt(se_inflation(rho, n))
    t = sign * out["mean"] / se_adj if se_adj > 0 else np.nan
    p = float(stats.t.sf(t, n - 1)) if np.isfinite(t) else np.nan
    if n < min_windows or not np.isfinite(t):
        label = "short"
    elif sign * out["mean"] <= 0:
        label = "reverse"
    elif p < alpha:
        label = "pass"
    else:
        label = "weak"
    return {**out, "se": se, "rho": rho, "se_adj": float(se_adj), "t": float(t), "p": p, "label": label}


# ---------------------------------------------------------------------- #
# 窓
# ---------------------------------------------------------------------- #

def oof_windows(dates: pd.Series) -> List[tuple]:
    """本番の OOF と同じ作りの窓（実験22・66 と同じ）。"""
    folds = WF.make_folds(dates, min_train_months=OOF_MIN_TRAIN_MONTHS, test_months=OOF_TEST_MONTHS,
                          step_months=OOF_STEP_MONTHS, embargo_days=B.RISE_HORIZON)
    return [(np.datetime64(f.test_start), np.datetime64(f.test_end)) for f in folds]


def month_windows(start: str, end: str) -> List[tuple]:
    """start の月から end の月までの暦月（両端を含む）。"""
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    return [(np.datetime64(m.start_time.normalize()), np.datetime64(m.end_time.normalize())) for m in months]


def last_labeled(frame: pd.DataFrame) -> Optional[pd.Timestamp]:
    """結果（OUTCOME）とラベルが付いている最後の日付。"""
    ok = np.isfinite(pd.to_numeric(frame[OUTCOME], errors="coerce")) & frame["label"].notna()
    return pd.Timestamp(frame.loc[ok, "Date"].max()) if ok.any() else None


def complete_windows(windows: List[tuple], last: Optional[pd.Timestamp]) -> List[bool]:
    """
    窓の行の結果がすべて出ているか。結果は日付の順に出るので、窓の終わりより後の日付に結果の付いた行が
    あれば、その窓の行はすべて出ている（最後の日ちょうどまでしか無いときは「まだ」と数える。保守側）。
    """
    if last is None:
        return [False] * len(windows)
    return [pd.Timestamp(e) < last for _, e in windows]


# ---------------------------------------------------------------------- #
# 行の選び方
# ---------------------------------------------------------------------- #

def edinet_split(manifest: dict, since: str) -> Tuple[set, set]:
    """取得の記録を (since 以降に初めて取れた銘柄, それより前から取れていた銘柄) に分ける。"""
    import edinet_fetch as EFT

    return EFT.split_by_first_ok(manifest.get("companies", {}), since)


def edinet_rows(df: pd.DataFrame, new: set, old: set, expected_old: int) -> pd.Series:
    """
    新しい銘柄の行のうち y0（ed_fiscal_year）が付いた行。前からの銘柄の数が記録と合わなければ止める
    （記録が欠けていると、10/10 に見た銘柄が「新しい」側に混ざりうる）。
    """
    if len(old) != expected_old:
        raise SystemExit(f"[stop] 2026-10-10 に取れていた社数が {len(old):,}（決めた値は {expected_old:,}）。"
                         "取得の記録が欠けているか、分け方が違う。判定しない")
    if new & old:
        raise SystemExit("[stop] 新しい銘柄と前からの銘柄が重なっている")
    return df["Code"].astype(str).isin(new) & df["ed_fiscal_year"].notna()


# ---------------------------------------------------------------------- #

def measure(df: pd.DataFrame, feats: Dict[str, int], windows: List[tuple], min_windows: int) -> dict:
    y = df["label"].to_numpy(dtype=float)
    r = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy(dtype=float)
    d = df["Date"].to_numpy()
    out = {}
    for f, sign in feats.items():
        x = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
        if ok.sum() < 500 or len(np.unique(y[ok])) < 2 or np.nanstd(x[ok]) == 0:
            out[f] = {"rows": int(ok.sum()), "windows": [], **verdict([], sign, min_windows)}
            continue
        ws = window_stats(x, y, r, d, ok, windows)
        out[f] = {"rows": int(ok.sum()), "windows": ws,
                  **verdict([w["edge"] for w in ws], sign, min_windows)}
    return out


def report(h: dict, res: dict, final: bool) -> None:
    head = "判定" if final else "途中経過（判定ではない）"
    print(f"\n=== {head}: 決めた向きでの片側 t 検定（α={ALPHA}）===")
    print(f"  {'列':<28}{'向き':>4}{'行':>7}{'窓':>4}{'超過pt':>8}{'SE':>6}{'隣相関':>7}"
          f"{'t':>7}{'p':>7}{'同じ向き':>8}  結果      （決めた時点: 超過 / z / 正の窓）")
    for f, sign in list(h["features"].items()) + list(h["reference"].items()):
        v = res[f]
        ref = f in h["reference"]
        o = h["origin"].get(f)
        lab_ = (LABELS[v["label"]] if final else "途中（判定しない）") + ("（参考）" if ref else "")
        print(f"  {f:<28}{'+' if sign > 0 else '−':>4}{v['rows']:>7,}{v['n']:>4}"
              f"{v['mean']:>+8.2f}{v.get('se_adj', np.nan):>6.2f}{v.get('rho', np.nan):>+7.2f}"
              f"{v.get('t', np.nan):>+7.2f}{v.get('p', np.nan):>7.3f}{v['agree']:>5}/{v['n']:<2}  "
              f"{lab_:<12}" + (f"（{o[0]:+.2f} / {o[1]:+.2f} / {o[2]}）" if o else ""))
    print("\n  窓ごとの超過（pt）")
    for f in list(h["features"]) + list(h["reference"]):
        ws = res[f]["windows"]
        cells = " ".join(f"{str(pd.Timestamp(w['start']).date())[:7]}:{w['edge']:+.1f}({w['n']})" for w in ws)
        print(f"  {f}: {cells or '(なし)'}")


def run_edinet(args) -> dict:
    import edinet_features as EF
    import edinet_fetch as EFT

    h = HYPOTHESES["edinet"]
    man_path = os.path.join(os.path.dirname(args.fin), EFT.MANIFEST)
    if not (os.path.exists(args.fin) and os.path.exists(man_path)):
        raise SystemExit(f"{args.fin} と {man_path} が要る（Release data-raw の edinet_*）")
    with open(man_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    EFT.backfill_first_ok(manifest.get("companies", {}))
    new, old = edinet_split(manifest, h["new_since"])
    print(f"取得の記録: {h['new_since'][:10]} より前から {len(old):,}社 / それ以降に初めて取れた {len(new):,}社")

    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    df = EF.attach(frame, EF.feature_frame(EF.annual_panel(EF.load_fin(args.fin))))
    rows = edinet_rows(df, new, old, h["old_companies"])
    have = df["ed_fiscal_year"].notna()
    old_rows = df["Code"].astype(str).isin(old) & have
    print(f"母集団 {len(df):,}行 / 新しい銘柄の行（y0 付き）{int(rows.sum()):,}行・{df.loc[rows, 'Code'].nunique():,}銘柄"
          f"（{rows.mean()*100:.1f}%）/ 前からの銘柄の行 {int(old_rows.sum()):,}行")
    windows = oof_windows(df["Date"])
    print(f"窓 {len(windows)}本（本番の OOF と同じ作り）/ 物差し {OUTCOME}")
    sub = df[rows].reset_index(drop=True)
    res = measure(sub, {**h["features"], **h["reference"]}, windows, h["min_windows"])
    report(h, res, final=True)

    # 列全体（523列）で |z|>2 が偶然より多いか。列の名前は出さない（ここから選び直さないため）
    if not args.no_family:
        fam = screen(sub, EF.columns("all"), windows) if len(sub) >= FAMILY_MIN_ROWS else None
        meas = fam["edge_z"].notna() if fam is not None else None
        if fam is None or not meas.any():
            print(f"\n列全体: 測れる列が無い（新しい銘柄の行 {len(sub):,}。列ごとに 500行・窓ごとに 100行が要る）")
        else:
            exp_n, p2, n_win = chance_hits(fam[meas])
            print(f"\n列全体: 測れた {int(meas.sum())}/{len(fam)}列 / |z|>2 {int((fam['abs_z'] > 2).sum())}本"
                  f"（偶然の見込み {exp_n:.1f}本、窓 {n_win}本の t 分布）")
    return {"source": "edinet", "new_companies": len(new), "old_companies": len(old),
            "rows": int(rows.sum()), "windows": len(windows), "result": res}


def run_jsf(args) -> dict:
    import jsf_features as JF

    h = HYPOTHESES["jsf"]
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    df = JF.build(frame, lab.DATA_DIR)
    have = df["jsf_ratio"].notna()
    t0, t1 = pd.Timestamp(h["start"]), pd.Timestamp(h["end"])
    rows = have & (df["Date"] >= t0) & (df["Date"] <= t1)
    windows = month_windows(h["start"], h["end"])
    last = last_labeled(df)
    done = complete_windows(windows, last)
    final = all(done)
    print(f"母集団 {len(df):,}行 / {h['start']} 〜 {h['end']} の日証金の付いた行 {int(rows.sum()):,}行 / "
          f"結果の出ている最後の日 {last.date() if last is not None else '-'}")
    print(f"窓 暦月 {len(windows)}本のうち結果がそろった {sum(done)}本"
          + ("" if final else f"（{h['run_after']} 以降に判定。いまは途中経過）"))
    use = [w for w, ok in zip(windows, done) if ok]
    sub = df[rows].reset_index(drop=True)
    res = measure(sub, {**h["features"], **h["reference"]}, use, h["min_windows"])
    report(h, res, final=final)
    return {"source": "jsf", "rows": int(rows.sum()), "windows_complete": int(sum(done)),
            "final": final, "result": res}


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.datetime64, pd.Timestamp)):
        return str(pd.Timestamp(x).date())
    return str(x)


def main(argv=None) -> int:
    import edinet_features as EF

    ap = argparse.ArgumentParser(description="先に決めた仮説を、そのあと増えたデータだけで確かめ直す")
    ap.add_argument("--source", choices=sorted(HYPOTHESES), required=True)
    ap.add_argument("--fin", default=EF.FIN, help="EDINET の年次財務（同じ場所に edinet_manifest.json）")
    ap.add_argument("--no-family", action="store_true", help="EDINET の列全体の数え上げを省く")
    args = ap.parse_args(argv)
    h = HYPOTHESES[args.source]
    print(f"実験69（{args.source}）: {DECIDED} に決めた仮説 / 判定してよいのは {h['run_after']} 以降")
    out = run_edinet(args) if args.source == "edinet" else run_jsf(args)
    out.update({"decided": DECIDED, "hypotheses": h})
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"e69_{args.source}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=_js)
    print(f"\n記録: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
