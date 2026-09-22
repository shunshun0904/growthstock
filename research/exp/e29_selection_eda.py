#!/usr/bin/env python3
"""
実験29: 運用の2つの選定基準で選ばれる銘柄の EDA（out-of-fold と CV）。

選定基準（docs/MODEL_ADOPTION_RULES.md §7、2026-09-22 に固定）
  R1  LightGBM 単体でスコアが上位 5%（95 パーセンタイル以上）
  R2  ブースティング3モデル（LightGBM / XGBoost / CatBoost）すべてで上位 10%

2つの評価の場
  OOF  本番と同じウォークフォワード（36ヶ月/6ヶ月/6ヶ月・エンバーゴ20営業日、
       11窓）。しきい値は**それより前の窓のスコア分布**から出す（運用そのもの）。
       スコアは実験27 の腕 B1（153列・本番のパラメータ・種3つの平均）
  CV   探索に使う5分割（year_cap_date: 年×時価総額帯で層別・日付単位で分割）。
       各分割の検証側を、その分割の訓練側で学習したモデルで採点する。
       しきい値は**その検証側の中**の分位（時間順が無いので過去が定義できない）。
       CV は分割の訓練側に未来が混ざるので、成績は楽観に出る
       （実測 CV PR-AUC 0.31 対 OOF 0.26）。**選ばれる銘柄の性格を見るためのもの**で、
       収益の見積もりには使わない

何を見るか
  1. 頻度（日・週・月）と集中（同じ銘柄が何度選ばれるか）
  2. 成績（正例率・実収益の分布・窓ごと）
  3. 銘柄の性格: 選定と母集団で、特徴量の中央値がどれだけ違うか
     （母集団分布の何パーセンタイルに当たるか、で出す）
  4. 業種・市場区分・規模帯の構成比
  5. 年別の選定数と成績
  6. 2つの基準の重なり

結果は research/_data/oof/e29_*.csv。本番の設定には書かない。
"""

from __future__ import annotations

import glob
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
import models as M  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import ops_rule as OR  # noqa: E402
from e24_timing_ab import with_timing  # noqa: E402
from e25_auc_noise import average  # noqa: E402
from e27_timing_multi import OOF_DIR, log, prod_params  # noqa: E402

SEEDS3 = (42, 7, 123)
N_SPLITS = 5
OUT = OOF_DIR
#: 選定と母集団を比べる特徴量（意味の分かるものに絞る）
EDA_NUM = [
    ("market_cap", "時価総額（億円）"), ("log_trading_value", "売買代金の対数"),
    ("r_high", "78週高値からの位置（%）"), ("r_high_3m", "3ヶ月前の高値位置（%）"),
    ("base_length", "ベースの長さ（営業日）"), ("break_margin", "高値をどれだけ抜けたか（%）"),
    ("close_position", "その日の終値の位置（0〜100）"), ("ret_20d", "直近20日の上昇率（%）"),
    ("vol_20d", "直近20日のボラ（%）"), ("volume_trend", "出来高 ÷ 20日平均（%）"),
    ("credit_ratio", "信用買残 ÷ 売残"), ("days_since_disc", "直近決算からの日数"),
    ("per", "PER"), ("pbr", "PBR"), ("earnings_yield", "益回り（%）"),
    ("div_yield", "配当利回り（%）"), ("psr", "PSR"),
    ("eps_growth_q0", "EPS成長率（%）"), ("sales_growth_q0", "売上成長率（%）"),
    ("ROE_q0", "ROE（%）"), ("op_margin_q0", "営業利益率（%）"),
    ("equity_ratio_q0", "自己資本比率（%）"), ("progress_vs_base", "決算進捗の超過（pt）"),
    ("guidance_op_growth", "会社予想の営業利益成長（%）"),
    ("topix_ma200_gap", "TOPIX の200日線からの乖離（%）"),
    ("rel_sector_20", "業種に対する20日の相対（pt）"),
]
CAP_BAND_JA = {0: "〜100億", 1: "100〜300億", 2: "300〜1000億", 3: "1000〜3000億", 4: "3000億〜"}
SCALE_JA = {0: "（対象外）", 1: "Core30", 2: "Large70", 3: "Mid400", 4: "Small 1", 5: "Small 2"}
MKT_JA = {105: "PRO", 109: "その他(ETF)", 111: "プライム", 112: "スタンダード", 113: "グロース"}


def sector_names() -> dict:
    m = pd.read_parquet(os.path.join(lab.DATA_DIR, "master.parquet"),
                        columns=["S33", "S33Nm"]).dropna().drop_duplicates()
    return {int(a): b for a, b in zip(m["S33"], m["S33Nm"])}


# ---------------------------------------------------------------------- #
# スコア
# ---------------------------------------------------------------------- #

def oof_scores(algos=OR.BOOST) -> dict:
    """実験27 の腕 B1（153列・本番のパラメータ）を種3つで平均して読む。"""
    out = {}
    for a in algos:
        files = [os.path.join(OOF_DIR, f"e27_{a}_B1_s{s}.parquet") for s in SEEDS3]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            raise SystemExit(f"{missing[0]} がありません（先に実験27 を回す）")
        out[a] = average([pd.read_parquet(f) for f in files])
    return out


def fit_predict(algo: str, params: dict, tr: pd.DataFrame, va: pd.DataFrame,
                cols: list, seed: int) -> np.ndarray:
    """
    lgbm は本番のパラメータに objective / n_estimators まで入っているので
    LGBMClassifier に直接渡す（tuning_multi.build と二重指定になる）。
    他のモデルは探索空間のぶんだけなので build に任せる。
    実験19・27 の out-of-fold と同じ組み立て。
    """
    ytr = tr["label"].to_numpy(dtype=int)
    Xtr, Xva = tr[cols].to_numpy(dtype=float), va[cols].to_numpy(dtype=float)
    if algo == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**{**params, "random_state": seed},
                               scale_pos_weight=tuning.scale_pos_weight(ytr))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]
    TM.SEED = seed
    m = M.fit(algo, Xtr, ytr, cols, params=params)
    TM.SEED = 0
    return M.predict(m, Xva)


def cv_scores(df: pd.DataFrame, cols: list, algos=OR.BOOST) -> dict:
    """
    探索と同じ5分割（year_cap_date）で、検証側を訓練側のモデルで採点する。
    分割は tuning.year_folds に任せる（探索とずれないように）。
    """
    d = pd.to_datetime(df["Date"])
    train_end, _, _ = T.holdout_bounds(d, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()].reset_index(drop=True)
    folds = tuning.year_folds(sub, n_splits=N_SPLITS, seed=0, by_year=True,
                              by_cap=True, group_by_date=True)
    log(f"CV: 探索期間 〜{train_end.date()} / {len(sub):,}件 / {len(folds)}分割 "
        + " / ".join(f"検証{len(v):,}" for _, v in folds))
    keep = ["Code", "Date", "label", "ret_o1_20", "ret_o1_40"]
    out = {}
    for a in algos:
        path = os.path.join(OUT, f"e29_cv_{a}.parquet")
        if os.path.exists(path):
            out[a] = pd.read_parquet(path)
            log(f"  [{a}] CV スコアの保存済みを読む（{len(out[a]):,}件）")
            continue
        params = prod_params(a)["params"]
        parts = []
        t0 = time.time()
        for i, (tr, va) in enumerate(folds, 1):
            sc = np.zeros(len(va), dtype=float)
            for s in SEEDS3:
                sc += fit_predict(a, params, tr, va, cols, s)
            part = va[keep].copy()
            part["score"] = sc / len(SEEDS3)
            part["fold"] = i
            parts.append(part)
        out[a] = pd.concat(parts, ignore_index=True)
        out[a].to_parquet(path, index=False)
        log(f"  [{a}] CV スコア {len(out[a]):,}件 / {time.time()-t0:.0f}秒")
    return out


# ---------------------------------------------------------------------- #
# 選定
# ---------------------------------------------------------------------- #

def pick(scores: dict, models, pct: float, mode: str) -> pd.DataFrame:
    """
    mode="oof"     : しきい値はそれより前の窓の分布（運用の規則）
    mode="oof_win" : しきい値はその窓の中の分布（窓ごとの選定率を一定にする変種）
    mode="cv"      : しきい値はその分割の検証側の分布
    戻り値は選ばれた行（Code, Date, fold, label, ret_o1_*, s_<algo>）。
    """
    first = models[0]
    base = scores[first][["Code", "Date", "fold", "label", "ret_o1_20", "ret_o1_40"]].copy()
    for a in models:
        base = base.merge(scores[a][["Code", "Date", "score"]].rename(columns={"score": f"s_{a}"}),
                          on=["Code", "Date"], how="inner")
    out = []
    for f in sorted(base["fold"].unique()):
        cur = base[base["fold"] == f]
        if mode == "oof":
            prev = base[base["fold"] < f]
            if len(prev) < 500:
                continue
            ref = {a: prev[f"s_{a}"].to_numpy() for a in models}
        else:
            if mode == "oof_win" and len(base[base["fold"] < f]) < 500:
                continue          # 窓1 は OOF と同じ理由で外す（比較の分母を揃える）
            ref = {a: cur[f"s_{a}"].to_numpy() for a in models}
        ok = np.ones(len(cur), dtype=bool)
        for a in models:
            ok &= (cur[f"s_{a}"] > np.percentile(ref[a], pct)).to_numpy()
        out.append(cur[ok])
    return pd.concat(out, ignore_index=True) if out else base.iloc[:0]


def universe(scores: dict, models, mode: str) -> pd.DataFrame:
    """選定と同じ窓（OOF は窓1を除く）の母集団。比較の分母。"""
    base = scores[models[0]][["Code", "Date", "fold", "label", "ret_o1_20", "ret_o1_40"]]
    if mode.startswith("oof"):
        folds = sorted(base["fold"].unique())
        keep = [f for f in folds if len(base[base["fold"] < f]) >= 500]
        return base[base["fold"].isin(keep)]
    return base


# ---------------------------------------------------------------------- #
# EDA
# ---------------------------------------------------------------------- #

def trading_days() -> pd.Series:
    d = pd.concat([pd.read_parquet(p, columns=["Date"])
                   for p in sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))])["Date"]
    return pd.Series(np.sort(pd.to_datetime(d.unique())))


def frequency(sel: pd.DataFrame, uni: pd.DataFrame, cal: pd.Series) -> dict:
    d0, d1 = pd.to_datetime(uni["Date"]).min(), pd.to_datetime(uni["Date"]).max()
    days = cal[(cal >= d0) & (cal <= d1)]
    per_day = (sel.groupby(pd.to_datetime(sel["Date"])).size()
               .reindex(days, fill_value=0))
    wk = per_day.groupby(per_day.index.to_period("W")).sum()
    mo = per_day.groupby(per_day.index.to_period("M")).sum()
    vc = sel["Code"].value_counts()
    return {"n": int(len(sel)), "rate": len(sel) / len(uni),
            "days": int(len(days)), "per_day": float(per_day.mean()),
            "median_day": float(per_day.median()), "zero_day": float((per_day == 0).mean()),
            "max_day": int(per_day.max()), "per_week": float(wk.mean()),
            "zero_week": float((wk == 0).mean()), "per_month": float(mo.mean()),
            "codes": int(sel["Code"].nunique()),
            "once": float((vc == 1).mean()), "max_repeat": int(vc.max()) if len(vc) else 0,
            "span": f"{d0.date()}〜{d1.date()}"}


def performance(sel: pd.DataFrame, uni: pd.DataFrame) -> dict:
    r = pd.to_numeric(sel["ret_o1_20"], errors="coerce")
    ru = pd.to_numeric(uni["ret_o1_20"], errors="coerce")
    per_fold = []
    for f in sorted(sel["fold"].unique()):
        a = pd.to_numeric(sel.loc[sel["fold"] == f, "ret_o1_20"], errors="coerce")
        b = pd.to_numeric(uni.loc[uni["fold"] == f, "ret_o1_20"], errors="coerce")
        if len(a) >= 5 and b.notna().sum() >= 100:
            per_fold.append((int(f), float((a.mean() - b.mean()) * 100)))
    v = np.array([x for _, x in per_fold])
    q = r.quantile([0.1, 0.25, 0.5, 0.75, 0.9]) * 100
    return {"label_rate": float(sel["label"].mean()), "base_label": float(uni["label"].mean()),
            "ret20": float(r.mean() * 100), "ret20_uni": float(ru.mean() * 100),
            "ret40": float(pd.to_numeric(sel["ret_o1_40"], errors="coerce").mean() * 100),
            "ret40_uni": float(pd.to_numeric(uni["ret_o1_40"], errors="coerce").mean() * 100),
            "win": float((r > 0).mean()), "win_uni": float((ru > 0).mean()),
            "q10": q[0.1], "q25": q[0.25], "median": q[0.5], "q75": q[0.75], "q90": q[0.9],
            "fold_mean": float(v.mean()) if len(v) else np.nan,
            "fold_se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
            "fold_won": int((v > 0).sum()), "n_folds": len(v),
            "worst": float(v.min()) if len(v) else np.nan,
            "per_fold": dict(per_fold)}


def profile(sel: pd.DataFrame, uni: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """特徴量ごとに、選定の中央値が母集団分布の何パーセンタイルに当たるかを出す。"""
    key = ["Code", "Date"]
    cols = [c for c, _ in EDA_NUM if c in frame.columns]
    f = frame[key + cols]
    su = uni.merge(f, on=key, how="left")
    ss = sel.merge(f, on=key, how="left")
    rows = []
    for c, ja in EDA_NUM:
        if c not in su.columns:
            continue
        u = pd.to_numeric(su[c], errors="coerce").dropna()
        v = pd.to_numeric(ss[c], errors="coerce").dropna()
        if len(u) < 100 or len(v) < 20:
            continue
        med = v.median()
        rows.append({"col": c, "ja": ja, "uni_median": u.median(), "sel_median": med,
                     "pctile_of_uni": float((u < med).mean() * 100),
                     "uni_mean": u.mean(), "sel_mean": v.mean(),
                     "cover_sel": float(len(v) / len(ss))})
    return pd.DataFrame(rows)


def mix(sel: pd.DataFrame, uni: pd.DataFrame, frame: pd.DataFrame, col: str,
        names: dict, top: int = 12) -> pd.DataFrame:
    key = ["Code", "Date"]
    f = frame[key + [col]]
    su = uni.merge(f, on=key, how="left")[col]
    ss = sel.merge(f, on=key, how="left")[col]
    a = ss.value_counts(normalize=True) * 100
    b = su.value_counts(normalize=True) * 100
    out = pd.DataFrame({"sel_pct": a, "uni_pct": b}).fillna(0.0)
    out["diff_pt"] = out["sel_pct"] - out["uni_pct"]
    out["name"] = [names.get(int(i), str(i)) if pd.notna(i) else "欠測" for i in out.index]
    return out.sort_values("sel_pct", ascending=False).head(top)


def by_year(sel: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    s = sel.copy(); u = uni.copy()
    s["y"] = pd.to_datetime(s["Date"]).dt.year
    u["y"] = pd.to_datetime(u["Date"]).dt.year
    rows = []
    for y, g in s.groupby("y"):
        gu = u[u["y"] == y]
        rows.append({"year": int(y), "n": len(g), "uni": len(gu),
                     "rate": len(g) / max(1, len(gu)) * 100,
                     "label_rate": g["label"].mean() * 100,
                     "base_label": gu["label"].mean() * 100,
                     "ret20": pd.to_numeric(g["ret_o1_20"], errors="coerce").mean() * 100,
                     "ret20_uni": pd.to_numeric(gu["ret_o1_20"], errors="coerce").mean() * 100})
    return pd.DataFrame(rows)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    frame = with_timing(lab.frame())
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.rename(columns={"jq_days_since_disc": "days_since_disc",
                                  "jq_days_since_fy": "days_since_fy"})
    cols = [c for c in F.columns("all") if c in frame.columns]
    log(f"母集団 {len(frame):,}件 / 特徴量 {len(cols)}列 / 種 {SEEDS3}")
    scores = {"OOF": oof_scores(), "CV": cv_scores(frame, cols)}
    cal = trading_days()
    s33 = sector_names()

    store = {}
    NOTE = {
        "OOF": "本番と同じ11窓・しきい値は**過去の窓**の分布（運用そのもの）。"
               "窓ごとの選定数は大きく偏る",
        "OOF窓内": "同じ11窓・しきい値は**その窓の中**の分布。窓ごとの選定率が一定に"
                   "なるので、窓平均の比較にはこちらが素直",
        "CV": "探索と同じ5分割・しきい値は各分割の検証側の分布。分割の訓練側に未来が"
              "混ざるので成績は楽観に出る（選ばれる銘柄の性格を見るためのもの）",
    }
    MODE_KEY = {"OOF": ("OOF", "oof"), "OOF窓内": ("OOF", "oof_win"), "CV": ("CV", "cv")}
    for mode in ("OOF", "OOF窓内", "CV"):
        src, how = MODE_KEY[mode]
        sc = scores[src]
        print(f"\n{'='*100}\n{mode}（{NOTE[mode]}）\n{'='*100}")
        for rname, models, pct in OR.RULES:
            sel = pick(sc, models, pct, how)
            uni = universe(sc, models, how)
            fr_ = frequency(sel, uni, cal)
            pf = performance(sel, uni)
            store[(mode, rname)] = {"freq": fr_, "perf": pf}
            print(f"\n--- {rname} ---  対象 {fr_['span']} / 母集団 {len(uni):,}件")
            print(f"  選定 {fr_['n']:,}件（{fr_['rate']*100:.1f}%）/ 銘柄 {fr_['codes']:,} / "
                  f"1回だけ選ばれた銘柄 {fr_['once']*100:.0f}% / 同じ銘柄の最多 {fr_['max_repeat']}回")
            if src == "OOF":
                print(f"  頻度: 1営業日 {fr_['per_day']:.2f}件（中央値 {fr_['median_day']:.0f} / "
                      f"ゼロの日 {fr_['zero_day']*100:.0f}% / 最多 {fr_['max_day']}件）/ "
                      f"1週 {fr_['per_week']:.1f}件（ゼロの週 {fr_['zero_week']*100:.0f}%）/ "
                      f"1か月 {fr_['per_month']:.1f}件")
            print(f"  正例率 {pf['label_rate']*100:.1f}%（母集団 {pf['base_label']*100:.1f}%）/ "
                  f"勝率 {pf['win']*100:.1f}%（母集団 {pf['win_uni']*100:.1f}%）")
            print(f"  ret_o1_20 平均 {pf['ret20']:+.2f}%（母集団 {pf['ret20_uni']:+.2f}%）/ "
                  f"中央値 {pf['median']:+.2f}% / 四分位 {pf['q25']:+.2f}〜{pf['q75']:+.2f}% / "
                  f"10-90% {pf['q10']:+.2f}〜{pf['q90']:+.2f}%")
            print(f"  ret_o1_40 平均 {pf['ret40']:+.2f}%（母集団 {pf['ret40_uni']:+.2f}%）")
            unit = "窓" if src == "OOF" else "分割"
            print(f"  {unit}平均の超過 {pf['fold_mean']:+.2f}pt（SE {pf['fold_se']:.2f}）/ "
                  f"勝ち {pf['fold_won']}/{pf['n_folds']} / 最悪 {pf['worst']:+.2f}pt")
            print(f"  {unit}ごとの超過: " + " ".join(f"{k}:{v:+.2f}" for k, v in pf["per_fold"].items()))
            cnt = sel.groupby("fold").size()
            print(f"  {unit}ごとの選定数: " + " ".join(f"{k}:{v}" for k, v in cnt.items()))

            prof = profile(sel, uni, frame)
            prof.to_csv(os.path.join(OUT, f"e29_profile_{how}_{models[0]}{int(pct)}.csv"), index=False)
            print(f"  --- 銘柄の性格（選定の中央値が母集団分布の何%点か。50 なら母集団と同じ）---")
            print(f"    {'特徴量':<30}{'母集団の中央値':>14}{'選定の中央値':>13}{'%点':>7}")
            for _, r in prof.reindex(prof["pctile_of_uni"].sub(50).abs().sort_values(ascending=False).index).iterrows():
                print(f"    {r['ja']:<30}{r['uni_median']:>14.2f}{r['sel_median']:>13.2f}{r['pctile_of_uni']:>7.0f}")

            for col, names, ja in (("s33_code", s33, "業種（33分類）"),
                                   ("cap_band", CAP_BAND_JA, "時価総額の帯"),
                                   ("mkt_code", MKT_JA, "市場区分"),
                                   ("scalecat_code", SCALE_JA, "TOPIX 規模区分")):
                mx = mix(sel, uni, frame, col, names, top=10)
                mx.to_csv(os.path.join(OUT, f"e29_mix_{col}_{how}_{models[0]}{int(pct)}.csv"))
                print(f"  --- {ja}の構成比（選定 / 母集団 / 差）---")
                for _, r in mx.iterrows():
                    print(f"    {r['name']:<22}{r['sel_pct']:>6.1f}% {r['uni_pct']:>6.1f}% "
                          f"{r['diff_pt']:>+6.1f}pt")

            yr = by_year(sel, uni)
            yr.to_csv(os.path.join(OUT, f"e29_year_{how}_{models[0]}{int(pct)}.csv"), index=False)
            print(f"  --- 年別 ---")
            print(f"    {'年':<6}{'選定':>6}{'母集団':>7}{'選定率':>7}{'正例率':>8}{'(母)':>7}"
                  f"{'ret20':>8}{'(母)':>8}")
            for _, r in yr.iterrows():
                print(f"    {int(r['year']):<6}{int(r['n']):>6}{int(r['uni']):>7}{r['rate']:>6.1f}%"
                      f"{r['label_rate']:>7.1f}%{r['base_label']:>6.1f}%"
                      f"{r['ret20']:>+7.2f}%{r['ret20_uni']:>+7.2f}%")

        # 2つの基準の重なり
        a = pick(sc, OR.RULES[0][1], OR.RULES[0][2], how)
        b = pick(sc, OR.RULES[1][1], OR.RULES[1][2], how)
        ka = set(map(tuple, a[["Code", "Date"]].to_numpy()))
        kb = set(map(tuple, b[["Code", "Date"]].to_numpy()))
        both = ka & kb
        print(f"\n--- 2つの基準の重なり（{mode}）---")
        print(f"  lgbm95 {len(ka):,} / 3モデル90 {len(kb):,} / 両方 {len(both):,}"
              f"（lgbm95 の {len(both)/max(1,len(ka))*100:.0f}% / 3モデル90 の {len(both)/max(1,len(kb))*100:.0f}%）")
        for name, keys in (("両方", both), ("lgbm95 のみ", ka - kb), ("3モデル90 のみ", kb - ka)):
            if not keys:
                continue
            idx = pd.MultiIndex.from_tuples(list(keys), names=["Code", "Date"])
            g = a.set_index(["Code", "Date"]).reindex(idx).dropna(subset=["label"])
            if not len(g):
                g = b.set_index(["Code", "Date"]).reindex(idx).dropna(subset=["label"])
            r = pd.to_numeric(g["ret_o1_20"], errors="coerce")
            print(f"  {name:<14}{len(g):>7,}件  正例率 {g['label'].mean()*100:>5.1f}%  "
                  f"ret20 平均 {r.mean()*100:>+6.2f}%  中央値 {r.median()*100:>+6.2f}%")

    with open(os.path.join(OUT, "e29_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in store.items()}, fh,
                  ensure_ascii=False, indent=1, default=float)
    log(f"記録: {OUT}/e29_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
