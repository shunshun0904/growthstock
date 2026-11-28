#!/usr/bin/env python3
"""
実験133: 学習に使う年数を変えると、+10% の LightGBM の精度はどう変わるか（学習曲線）。検証だけ。本番は変えない。

運用者の問い（2026-11-27）「JQUANTS APIをプレミアムプランに引き上げて、昔のデータも使おうと思ってます。…料金が高いので、
データ数増加による効果が確実なら1ヶ月だけ課金したい」。運用者の選択（同日、AskUserQuestion）: 「無料で先に測る」・
「2〜6年・15窓」・パラメータは「本番のまま」・判定は「厳しめ」。課金するかの判断材料にする（docs/MODEL_ADOPTION_RULES.md §105）。

腕（データ・列・パラメータ・テストの行は同じ。学習に使う行の期間だけが違う）
  L2〜L6  窓の学習の最終日（train_end）から L 年さかのぼった日の翌日 〜 train_end の行で学習
  ALL     データの最初 〜 train_end（今の本番の out-of-fold と同じ作り。参考）
窓: 本番と同じ作り（36/6/6か月・エンバーゴ20営業日）を 0/2/4か月ずらした3通りのうち、train_end − 6年 がデータの最初の日
  以降の窓（6年分の学習が取れる窓 = 2024年5月以降のテスト）。どの腕も同じ窓・同じテストの行で比べる。種 42・7・123 の平均。
列: 今の373列（回した時点の既定を名前で固定）。ラベル: 本番の +10%（label_specs.for_model("lgbm")。fixed_return_v1）。
パラメータ: 本番の lgbm_params.json の値（探索し直さない）。本番の値は全期間（約7年）で選んだので、短い年数にはやや不利
  （伸びが大きめに出る側）。それでも頭打ちなら「見込み薄」は堅い。

判定（運用者の選択 2026-11-27「厳しめ」。回す前に決めた）
  窓ごとの PR-AUC の差 L6 − L4 で ① 平均が正 ② L6 が上の窓が過半（n//2 + 1 以上。同じ値の窓は上に数えない）
  ③ 平均が 2SE を超える。3つとも満たせば「伸び続けている（古いデータに見込みあり）」、1つでも欠ければ「頭打ち（見込み薄）」。
  2016年より前（デフレ期）のデータの効き目そのものは、データが無いので測れない（この判定は「データの量」の効き目の目安）。

出すもの（公開ログには件数・割合・日付・精度だけ。銘柄・値は出さない）
  0. 確かめ（ラベル・出口・窓・学習の行の入れ子・テストの行が腕どうしで同じ）
  1. 腕ごとの窓の平均（PR-AUC・ROC・学習の行数）  2. 差（隣どうし・L6−L4・L6−L2・ALL−L6）  3. 判定
  4. 参考: 年数を2倍にしたときの PR-AUC の伸び（窓ごとに L2〜L6 の PR-AUC を log2(年数) に当てた傾き）
  5. 窓の中の上位2%（参考）  6. 窓ごとの値

    exp=e133_train_years.py
窓と学習の行数だけ見る: exp=e133_train_years.py args="--dry"
試運転:                 exp=e133_train_years.py args="--tag e133smoke --shifts 0 --seeds 42 --years 2,6"

out-of-fold は research/_data/oof/<tag>_* に置き（Run Experiment が actions/cache で引き継ぐ）、同じデータ・同じパラメータなら
次の回はそれを読む。
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
import tuning  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
from e25_auc_noise import average  # noqa: E402
from e68_edinet_ab import pair, params_hash, short  # noqa: E402
from e104_market_drift import top_by_window  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
ALGO = "lgbm"
SEEDS = (42, 7, 123)
SHIFTS = (0, 2, 4)
#: 比べる年数（運用者の選択 2026-11-27「2〜6年・15窓」）
YEARS = (2, 3, 4, 5, 6)
#: 学習の期間を切らない腕（今の本番の out-of-fold と同じ作り。参考）
ALL = "ALL"
#: 判定に使う組（一番長い年数 − その2年短い年数）
JUDGE = ("L4", "L6")
#: 比べる土台。回した時点の既定（373列）を名前で固定する
BASE_PRESET = "all_plus_prog_listing_vol_book_shape_dd_recent_tx_financial"
MIN_WINDOW_ROWS = 100
#: E41.oof_folds が窓を飛ばす下限（学習 1000行未満・テスト 200行未満）と同じ
MIN_TRAIN_ROWS = 1000
MIN_TEST_ROWS = 200
TP = LT.TAKE_PROFIT
TAG = "e133"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def path(name: str) -> str:
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def arm_name(years) -> str:
    return ALL if years is None else f"L{int(years)}"


def arm_years(arm: str):
    return None if arm == ALL else int(arm[1:])


# --------------------------------------------------------------------------- #
# 窓と学習の行
# --------------------------------------------------------------------------- #
def eligible_folds(dates: pd.Series, shift: int, max_years: int) -> list:
    """本番と同じ作りの窓（ずらし shift か月）のうち、train_end − max_years 年がデータの最初の日以降のもの。"""
    d = pd.to_datetime(dates)
    first = pd.Timestamp(d.min())
    return [f for f in E41.folds_for(d, shift)
            if pd.Timestamp(f.train_end) - pd.DateOffset(years=int(max_years)) >= first]


def train_mask(dates: pd.Series, train_end, years) -> pd.Series:
    """学習に使う行: (train_end − years 年, train_end]。years が None なら最初から train_end まで。"""
    d = pd.to_datetime(dates)
    end = pd.Timestamp(train_end)
    m = d <= end
    if years is not None:
        m &= d > end - pd.DateOffset(years=int(years))
    return m


def check_label(df: pd.DataFrame) -> dict:
    """データセットの label が本番の +10% の定義（label_specs）と全行一致するか。違えば止める。"""
    want = LS.labels(df, "lgbm")
    have = pd.to_numeric(df["label"], errors="coerce")
    same = (want.isna() & have.isna()) | (want.notna() & have.eq(want))
    out = {"rows": int(len(df)), "same": int(same.sum()), "defined": int(want.notna().sum())}
    print(f"  確かめ: データセットの label と本番の +10% の定義の一致 {out['same']:,} / {out['rows']:,}行"
          f"（決まる行 {out['defined']:,}）")
    if out["same"] != out["rows"]:
        raise SystemExit("データセットの label が本番の +10% の定義と合わない")
    return out


def production_params(cols: list) -> dict:
    """本番の lgbm_params.json の値（今の373列・+10%）。列の指紋かラベルが違えば止める。"""
    rec = tuning.load_params().get(BASE_PRESET)
    if not rec:
        raise SystemExit(f"lgbm_params.json に {BASE_PRESET} が無い")
    sig = F.signature(cols)
    if rec.get("_features_sig") != sig:
        raise SystemExit(f"本番のパラメータの列の指紋 {rec.get('_features_sig')} が今の列 {sig} と違う")
    spec = (rec.get("_cv") or {}).get("label_spec") or {}
    if spec.get("version") != LS.for_model("lgbm").get("version"):
        raise SystemExit(f"本番のパラメータのラベル {spec.get('version')} が今の +10% と違う")
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def data_key(df: pd.DataFrame) -> str:
    """out-of-fold を読み直してよいかの指紋（行・日付・ラベル）。"""
    h = hashlib.sha1()
    h.update(pd.util.hash_pandas_object(df[["Code", "Date"]], index=False).to_numpy().tobytes())
    h.update(np.nan_to_num(pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=float), nan=-1).tobytes())
    return h.hexdigest()[:8]


def prepare(cols: list):
    """lab.frame に出口（+10% の指値）を付けた表と確かめの記録。"""
    t0 = time.time()
    df = lab.frame(rebuild=True)
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    df["Code"] = df["Code"].astype(str)
    miss = [c for c in cols + ["ret_o1_20", "ret_o1_40", "label"] if c not in df.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    df = df.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    log(f"[data] lab.frame {len(df):,}行 × {df.shape[1]}列 / {time.time()-t0:.0f}秒")
    print("\n■ 確かめ")
    checks = {"label": check_label(df)}
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
    stamp = {"rows": int(len(df)), "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
             "rows_by_year": {str(y): int((yr == y).sum()) for y in sorted(yr.unique())}, "checks": checks,
             "key": data_key(df)}
    return df, stamp


def plan(df: pd.DataFrame, arms: list, shifts: list, max_years: int) -> pd.DataFrame:
    """窓ごと・腕ごとの学習の行数とテストの行数（計算の前に出す）。テストが MIN_TEST_ROWS 未満の窓は使わない
    （本番の out-of-fold・E41.oof_folds と同じ下限）。"""
    ok = df["label"].notna()
    d = df["Date"]
    rows = []
    for sh in shifts:
        for f in eligible_folds(d, sh, max_years):
            te = ok & (d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))
            r = {"shift": sh, "fold": f.index, "train_end": f.train_end, "test_start": f.test_start,
                 "test_end": f.test_end, "n_test": int(te.sum()), "use": bool(te.sum() >= MIN_TEST_ROWS)}
            for arm in arms:
                m = train_mask(d, f.train_end, arm_years(arm)) & ok
                r[f"n_{arm}"] = int(m.sum())
                r[f"from_{arm}"] = str(d[m].min().date()) if m.any() else ""
            rows.append(r)
    return pd.DataFrame(rows)


def check_plan(p: pd.DataFrame, arms: list) -> dict:
    """学習の行が年数の順に入れ子（短い年数 ⊂ 長い年数 ⊂ ALL）で、使う窓の学習がどれも下限を超えるか。崩れていれば止める。"""
    order = sorted([a for a in arms if a != ALL], key=arm_years) + ([ALL] if ALL in arms else [])
    bad_nest = 0
    for _, r in p.iterrows():
        n = [r[f"n_{a}"] for a in order]
        bad_nest += int(any(a > b for a, b in zip(n, n[1:])))
    used = p[p["use"]]
    small = int((used[[f"n_{a}" for a in arms]].min(axis=1) < MIN_TRAIN_ROWS).sum())
    out = {"windows": int(len(used)), "dropped_small_test": int((~p["use"]).sum()), "not_nested": bad_nest,
           "too_small_train": small}
    print(f"  確かめ: 使う窓 {out['windows']}（6年分の学習が取れ、テストが {MIN_TEST_ROWS}行以上）・テストが少なく使わない窓 "
          f"{out['dropped_small_test']}・学習の行が入れ子でない窓 {bad_nest}・学習が {MIN_TRAIN_ROWS}行に満たない窓 {small}")
    if bad_nest or small or not len(used):
        raise SystemExit("窓の作りが崩れている（入れ子でないか、学習の行が足りないか、使える窓が無い）")
    return out


def used_folds(df: pd.DataFrame, p: pd.DataFrame, shift: int, max_years: int) -> list:
    keep = set(p.loc[p["use"] & (p["shift"] == shift), "fold"].astype(int))
    return [f for f in eligible_folds(df["Date"], shift, max_years) if f.index in keep]


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #
def oof_arm(df: pd.DataFrame, cols: list, params: dict, folds: list, arm: str, seed: int) -> pd.DataFrame:
    """1つの腕・1つの種の out-of-fold（窓ごとに学習の期間を切って E41.oof_folds に渡す）。"""
    parts = []
    for f in folds:
        m = train_mask(df["Date"], f.train_end, arm_years(arm))
        o = E41.oof_folds(ALGO, df, cols, params, seed, [f], train_mask=m.to_numpy())
        parts.append(o)
    return pd.concat(parts, ignore_index=True)


def run(df: pd.DataFrame, cols: list, params: dict, arms: list, shifts: list, seeds: list, max_years: int,
        key: str, p: pd.DataFrame) -> dict:
    extra = df[["Code", "Date", "hit10", "tp10c"]]
    ph = params_hash(params)
    oofs, tests = {}, {}
    for sh in shifts:
        folds = used_folds(df, p, sh, max_years)
        if not folds:
            continue
        for arm in arms:
            parts = []
            for sd in seeds:
                f = path(f"{arm}_sh{sh}_s{sd}_{ph}_{key}.parquet")
                if os.path.exists(f):
                    parts.append(pd.read_parquet(f))
                    continue
                t0 = time.time()
                o = oof_arm(df, cols, params, folds, arm, sd).merge(extra, on=["Code", "Date"], how="left")
                os.makedirs(OOF_DIR, exist_ok=True)
                o.to_parquet(f, index=False)
                parts.append(o)
                log(f"    {arm} ずらし{sh}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
            o = average(parts)
            o["Date"] = pd.to_datetime(o["Date"])
            o["Code"] = o["Code"].astype(str)
            oofs[(sh, arm)] = o
            tests[(sh, arm)] = hashlib.sha1(
                pd.util.hash_pandas_object(o[["fold", "Code", "Date"]], index=False).to_numpy().tobytes()).hexdigest()
    done = [sh for sh in shifts if all((sh, a) in oofs for a in arms)]
    # テストの行が腕どうしで同じか（窓・行・順番）
    same = bool(done) and all(len({tests[(sh, a)] for a in arms}) == 1 for sh in done)
    print(f"  確かめ: テストの行が腕どうしで同じ → {'同じ' if same else '違う'}")
    if not same:
        raise SystemExit("腕ごとにテストの行が違う（か、計算できた窓が無い）")
    return {"oofs": oofs, "shifts": done}


def windows_on(o: pd.DataFrame) -> pd.DataFrame:
    """窓ごとの AUC と、窓の期間・件数・正例率（MIN_WINDOW_ROWS 未満か正例・負例の片方しかない窓は落とす）。"""
    cols = ["fold", "pr", "roc", "start", "end", "n", "pos"]
    n = o.groupby("fold").size()
    sub = o[o["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    both = sub.groupby("fold")["label"].agg(lambda s: s.nunique() == 2)
    sub = sub[sub["fold"].isin(both[both].index)]
    if not len(sub):
        return pd.DataFrame(columns=cols)
    w = AB.auc_by_window(sub)
    g = sub.groupby("fold")
    info = pd.DataFrame({"start": g["Date"].min(), "end": g["Date"].max(), "n": g.size(),
                         "pos": g["label"].mean()}).reset_index()
    return w.merge(info, on="fold", how="left")[cols]


def window_table(res: dict, arms: list, shifts: list) -> pd.DataFrame:
    out = []
    for sh in shifts:
        w = None
        for arm in arms:
            x = windows_on(res["oofs"][(sh, arm)]).rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
            if w is not None:
                x = x[["fold", f"pr_{arm}", f"roc_{arm}"]]
            w = x if w is None else w.merge(x, on="fold")
        if w is None or not len(w):
            continue
        for _, r in w.iterrows():
            out.append({"shift": sh, "fold": int(r["fold"]), "start": r["start"], "end": r["end"], "n": int(r["n"]),
                        "pos": float(r["pos"]), **{f"{m}_{a}": float(r[f"{m}_{a}"]) for m in ("pr", "roc") for a in arms}})
    return pd.DataFrame(out)


def judge_strict(d: np.ndarray) -> dict:
    """厳しめの判定（運用者の選択 2026-11-27）: ① 平均が正 ② 上の窓が過半 ③ 平均が 2SE を超える。"""
    d = np.asarray(d, dtype=float)
    n = int(len(d))
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    mean = float(d.mean()) if n else float("nan")
    wins = int((d > 0).sum())
    need = n // 2 + 1
    out = {"n": n, "mean": mean, "se": se, "wins": wins, "need": need,
           "positive": bool(n and mean > 0), "majority": bool(n and wins >= need),
           "beyond_2se": bool(n > 1 and np.isfinite(se) and mean > 2 * se)}
    out["pass"] = bool(out["positive"] and out["majority"] and out["beyond_2se"])
    return out


def slope_per_doubling(s: pd.DataFrame, arms: list) -> dict:
    """窓ごとに L2〜L6 の PR-AUC を log2(年数) に当てた傾き（年数を2倍にしたときの伸び）の平均 ± SE。"""
    ys = [a for a in arms if a != ALL]
    if len(ys) < 2 or not len(s):
        return {"n": 0}
    x = np.log2(np.array([arm_years(a) for a in ys], dtype=float))
    sl = []
    for _, r in s.iterrows():
        y = np.array([r[f"pr_{a}"] for a in ys], dtype=float)
        sl.append(float(np.polyfit(x, y, 1)[0]))
    sl = np.array(sl)
    return {"n": int(len(sl)), "mean": float(sl.mean()),
            "se": float(sl.std(ddof=1) / np.sqrt(len(sl))) if len(sl) > 1 else float("nan"),
            "positive_windows": int((sl > 0).sum())}


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


def report(res: dict, arms: list, shifts: list, seeds: list, stamp: dict, p: pd.DataFrame, params: dict) -> dict:
    shifts = [sh for sh in shifts if sh in res.get("shifts", shifts)]
    pu = p[p["use"]] if "use" in p.columns else p
    print("\n" + "=" * 78)
    summary = {"data": stamp, "params": params, "arms": arms, "shifts": shifts, "seeds": seeds,
               "plan": p.to_dict("records"), "windows": {}, "judge": {}, "slope": {}, "top": {}, "per_window": []}
    s = window_table(res, arms, shifts)
    s.to_csv(path("auc_by_window.csv"), index=False)

    print(f"\n■ 1. 腕ごとの窓の平均（{len(s)}窓。ずらし {'/'.join(map(str, shifts))}か月 × 種{len(seeds)}つの平均）")
    print(f"  {'腕':<6}{'学習の行（平均）':>14}{'学習の最初（例）':>16}{'PR-AUC':>9}{'ROC':>9}")
    first = pu.sort_values(["shift", "fold"]).iloc[-1] if len(pu) else None
    for arm in arms:
        n_mean = float(pu[f"n_{arm}"].mean()) if len(pu) else float("nan")
        print(f"  {arm:<6}{n_mean:>14,.0f}{(first[f'from_{arm}'] if first is not None else ''):>16}"
              f"{s[f'pr_{arm}'].mean():>9.4f}{s[f'roc_{arm}'].mean():>9.4f}")
        summary["windows"][arm] = {"train_rows_mean": n_mean, "pr": float(s[f"pr_{arm}"].mean()),
                                   "roc": float(s[f"roc_{arm}"].mean())}
    if first is not None:
        print(f"  （学習の最初（例）は最後の窓: ずらし{int(first['shift'])}か月の窓{int(first['fold'])}・学習の最終日 {first['train_end']}）")

    ys = sorted([a for a in arms if a != ALL], key=arm_years)
    pairs = list(zip(ys, ys[1:]))
    for x, y in (("L4", "L6"), ("L2", "L6")):
        if x in arms and y in arms and (x, y) not in pairs:
            pairs.append((x, y))
    if ALL in arms and ys:
        pairs.append((ys[-1], ALL))
    print("\n■ 2. 窓ごとの差（PR-AUC・ROC。差は「後 − 前」）")
    print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
    for met, nm in (("pr", "PR"), ("roc", "ROC")):
        for x, y in pairs:
            line, st = pair(f"{nm} {y}−{x}", s[f"{met}_{x}"].to_numpy(), s[f"{met}_{y}"].to_numpy())
            print(line)
            summary["windows"][f"{met}_{y}-{x}"] = st

    x, y = JUDGE
    print(f"\n■ 3. 判定（厳しめ。運用者の選択 2026-11-27）: 窓の PR-AUC の {y}−{x} で ① 平均が正 ② {y} が上の窓が過半"
          " ③ 平均が 2SE を超える")
    if x in arms and y in arms and len(s):
        j = judge_strict((s[f"pr_{y}"] - s[f"pr_{x}"]).to_numpy())
        summary["judge"] = j
        print(f"  {y}−{x} {j['mean']:+.4f}（SE {j['se']:.4f}）{'○' if j['positive'] else '×'} / 上の窓 {j['wins']}/{j['n']}"
              f"（{j['need']} 以上）{'○' if j['majority'] else '×'} / 2SE {2 * j['se']:.4f} を"
              f"{'超える ○' if j['beyond_2se'] else '超えない ×'} → "
              f"{'伸び続けている（古いデータに見込みあり）' if j['pass'] else '頭打ち（見込み薄）'}")
    else:
        print(f"  {x}・{y} がそろっていない（判定しない）")

    sl = slope_per_doubling(s, arms)
    summary["slope"] = sl
    if sl.get("n"):
        print(f"\n■ 4. 参考: 年数を2倍にしたときの PR-AUC の伸び（窓ごとに {ys[0]}〜{ys[-1]} を log2(年数) に当てた傾き）: "
              f"{sl['mean']:+.4f} ± {sl['se']:.4f}（正の窓 {sl['positive_windows']}/{sl['n']}）")

    tt = top_table(res, arms, shifts)
    if len(tt):
        tt.to_csv(path("top_by_window.csv"), index=False)
        has_exit = bool(tt[[f"hit10_{a}" for a in arms]].notna().any().any())
        print(f"\n■ 5. 窓の中の上位2%（参考。{len(tt)}窓・合計 {int(tt['k'].sum()):,}件）: Precision / +10% に届いた割合 / "
              "1取引の平均（%。届けば +10、届かなければ20日目の終値）")
        for arm in arms:
            r = {m: float(tt[f"{m}_{arm}"].mean()) for m in ("prec", "hit10", "tp10c")}
            line = f"  {arm:<6}{r['prec']*100:>7.1f}%" + (f"{r['hit10']*100:>8.1f}%{r['tp10c']:>+8.2f}%" if has_exit else "")
            if arm == y and x in arms:
                d = (tt[f"prec_{y}"] - tt[f"prec_{x}"]).to_numpy(dtype=float) * 100
                line += f" ｜ Precision {y}−{x} {d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(len(d)):.2f}（上 {int((d > 0).sum())}/{len(d)}）"
                if has_exit:
                    e = (tt[f"tp10c_{y}"] - tt[f"tp10c_{x}"]).to_numpy(dtype=float)
                    e = e[np.isfinite(e)]
                    if len(e) > 1:
                        line += f" ｜ 平均 {e.mean():+.2f} ± {e.std(ddof=1) / np.sqrt(len(e)):.2f}（上 {int((e > 0).sum())}/{len(e)}）"
            summary["top"][arm] = r
            print(line)

    print("\n■ 6. 窓ごとの値（PR-AUC）")
    for sh in shifts:
        g = s[s["shift"] == sh].sort_values("fold") if len(s) else s
        if not len(g):
            continue
        print(f"  ずらし{sh}か月（{len(g)}窓）")
        print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}" + "".join(f"{a:>8}" for a in arms)
              + (f"{y + '−' + x:>9}" if x in arms and y in arms else ""))
        for _, r in g.iterrows():
            print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                  f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%" + "".join(f"{r[f'pr_{a}']:>8.4f}" for a in arms)
                  + (f"{r[f'pr_{y}'] - r[f'pr_{x}']:>+9.4f}" if x in arms and y in arms else ""))
    summary["per_window"] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records") \
        if len(s) else []
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def show_plan(p: pd.DataFrame, arms: list) -> None:
    print(f"\n■ 0. 窓と学習の行数（{len(p)}窓）")
    print(f"  {'ずらし':>4}{'窓':>4}  {'学習の最終日':<12}{'テスト':<24}{'件数':>7}" + "".join(f"{a:>8}" for a in arms)
          + "  （学習の行数）")
    for _, r in p.iterrows():
        print(f"  {int(r['shift']):>4}{int(r['fold']):>4}  {r['train_end']:<12}{r['test_start']}〜{r['test_end']}"
              f"{int(r['n_test']):>7,}" + "".join(f"{int(r[f'n_{a}']):>8,}" for a in arms)
              + ("" if r.get("use", True) else "  ← テストが少なく使わない"))


def main(argv=None) -> int:
    global TAG
    ap = argparse.ArgumentParser(description="実験133: 学習に使う年数を変えたときの +10% の LightGBM の精度")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--shifts", default=",".join(map(str, SHIFTS)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--years", default=",".join(map(str, YEARS)), help="比べる年数（一番長い年数が取れる窓だけを使う）")
    ap.add_argument("--no-all", action="store_true", help="参考の ALL（最初から全部）を計算しない")
    ap.add_argument("--dry", action="store_true", help="窓と学習の行数だけ出して終わる")
    args = ap.parse_args(argv)
    TAG = args.tag
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    years = sorted({int(x) for x in args.years.split(",") if x.strip()})
    if len(years) < 2 or min(years) < 1:
        raise SystemExit("--years は1以上の年数を2つ以上")
    arms = [arm_name(y) for y in years] + ([] if args.no_all else [ALL])
    cols = F.columns(BASE_PRESET)

    print("=" * 78)
    print("実験133 学習に使う年数を変えたときの +10% の LightGBM の精度（学習曲線。本番は変えない）")
    print(f"  列: {BASE_PRESET} {len(cols)}列 指紋 {F.signature(cols)}"
          f"{'' if BASE_PRESET == F.DEFAULT_PRESET else '（いまの既定とは違う）'}")
    print(f"  腕: {', '.join(arms)}（L は学習の最終日から何年さかのぼるか。ALL は最初から全部 = 今の本番の out-of-fold の作り）")
    print(f"  窓: 本番と同じ作り（36/6/6か月・エンバーゴ20営業日）をずらし {shifts}か月。{max(years)}年分の学習が取れる窓だけ"
          f" / 種 {tuple(seeds)} / 保存 {TAG}_*")
    print("=" * 78)
    params = production_params(cols)
    print(f"  パラメータ（本番の lgbm_params.json・探索し直さない）: {short(params)}")
    df, stamp = prepare(cols)
    p = plan(df, arms, shifts, max(years))
    show_plan(p, arms)
    stamp["plan_checks"] = check_plan(p, arms)
    if args.dry:
        return 0
    t0 = time.time()
    res = run(df, cols, params, arms, shifts, seeds, max(years), stamp["key"], p)
    log(f"[{ALGO}] 計算 {time.time()-t0:.0f}秒")
    report(res, arms, shifts, seeds, stamp, p, params)
    log(f"記録: {path('*')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
