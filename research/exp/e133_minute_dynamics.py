#!/usr/bin/env python3
"""
実験133: 分足の価格帯別出来高の形と推移、直近のテクニカルだけで、本番と同じ +10% を当てる独立したモデル（検証だけ）。

運用者の依頼（2026-11-19）「独立して、この情報や直近のテクニカルだけを使ったモデル（機械学習に限らない）を作りたいです。
おそらくこれだけが捉えるシグナルもあるかと思うので。モデルでなくても、ルールベースロジック構築のための統計モデルでも良いですし。
分布の形自体というよりは、その形の推移が大事なのかなとも思います。分布自体の微分ともいいますか。フィッシャー情報量とかが一番近い
概念かもしれません」。選択（AskUserQuestion）: データ「分足だけ」・行「高値更新の候補だけ」・当てるもの「本番と同じ +10%」・
モデル「統計モデルと機械学習の両方」。

実験132の台本（e132_minute_vp_ab.py の --dynamics）から呼ぶ（同じ表・A の保存済みの out-of-fold を使うため）。

列（research/minute_dynamics.py と、学習データの日足の列）
  推移だけ（dyn）              分布の推移の28列（フィッシャー・ラオ距離・ワッサースタイン距離・重心と POC の移動・エントロピーの変化・
                               70%範囲の幅の比・上値と終値近くの割合の変化・形の変わる速さとフィッシャー情報量・時間帯の割合の変化・
                               帯ごとの割合の変化）
  分布すべて（dist）           推移の28列 ＋ 形の16列（直近5日・20日の8つの帯）＋ 実験132の分足の30列
  直近のテクニカルだけ（tech） TECH_COLUMNS（日足から作った、期間の短い値動き・出来高・値幅の列）
  分布すべて＋直近のテクニカル（dist_tech）
モデル
  統計モデル（logit）: 標準化したロジスティック回帰（L2・C 0.1）。前処理は学習側の値だけで決める（1%・99% で丸め・中央値で埋める）
  機械学習（lgbm）  : LightGBM（葉15・葉の最小100・学習率0.03・300本・行と列の抜き取り 0.8）。データが少ないので探索はしない
                      （設定を変えた結果も参考に出す）
  ルール            : 深さ2の決定木（葉の最小300行）を前半の日付で作り、後半で葉ごとの正例率を確かめる
評価（分足の付いた候補の行。20営業日の推移が作れる行）
  時間順の窓: 2025-05-01 から3か月ずつのテスト窓。学習はテストの始まりの 28暦日（20営業日の目安）前までの行だけ
  A（今の本番の作りのモデル）の予測は、実験132の保存済みの out-of-fold（ずらし3通りの平均）
  1. 単独の当たり（PR-AUC・ROC・日内 AUC・リフト）。ROC は同じ日付の中でラベルを入れ替える並べ替え検定
  2. A との重なり（順位相関）と、A に足したときの上積み（窓ごとに順位を平均した組み合わせの PR-AUC − A の PR-AUC。
     並べ替え検定は、新しいモデルの点を同じ日付の中で入れ替える）
  3. A の上位20%（窓ごと）の中で、新しいモデルが当たり外れを分けるか（その中の AUC と、上半分・下半分の正例率）
  4. 統計モデルの係数（標準化。全期間で1本と、窓ごとの向きのそろい方）、ルールの葉
読み（回す前に決めたもの。主に見る組み合わせは「分布すべて＋直近のテクニカル × LightGBM」と「推移だけ × ロジスティック」の2つで、
  有意水準は 0.05 ÷ 2）
  単独で当たる          : ROC の並べ替え検定 p < 0.025
  今のモデルにない情報  : 組み合わせの PR-AUC の上積みが正で並べ替え検定 p < 0.025、かつ窓の過半で上積みが正
  A の上位を絞り込める  : A の上位20% の中の AUC の並べ替え検定 p < 0.025
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import minute_dynamics as MD

#: 直近のテクニカル（学習データの日足の列。期間の短い値動き・出来高・値幅。財務・需給・相場全体の列は入れない）
TECH_COLUMNS = (
    "ret_1d", "ret_2d", "gap_open", "ret_5d", "vol_chg_1d",                                   # recent_change
    "base_length", "break_margin", "close_position", "ret_20d", "vol_20d",                    # breakout
    "r_high_3m", "volume_trend", "vol_rel_short", "vol_rel_long", "vol_updown", "vol_gap_ratio",
    "pullback_depth", "base_depth",
    "tx_ma_gap_5", "tx_ma_gap_10", "tx_ma_gap_25", "tx_ma_ratio_5_25", "tx_rsi_9", "tx_rsi_14", "tx_stoch_k",
    "tx_williams_r_14", "tx_cci_20", "tx_roc_10", "tx_natr_14", "tx_vol_parkinson_20", "tx_bb_pctb", "tx_bb_width",
    "tx_donchian_pos_20", "tx_vwap_gap_20", "tx_cmf_20", "tx_mfi_14", "tx_obv_chg_20", "tx_up_vol_share_20",
    "tx_body_ratio", "tx_upper_shadow", "tx_lower_shadow", "tx_range_atr", "tx_bull_share_20", "tx_vol_osc_5_20",
    "tx_gap_up_count_20", "tx_close_streak",
)
TEST_START = pd.Timestamp("2025-05-01")
TEST_MONTHS = 3
EMBARGO_DAYS = 28
LOGIT_C = 0.1
LGBM = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 100, "subsample": 0.8,
        "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1, "n_jobs": 2}
#: 参考に出す別の設定
LGBM_ALT = {**LGBM, "num_leaves": 7, "min_child_samples": 200}
LOGIT_C_ALT = 1.0
TREE_DEPTH = 2
TREE_MIN_LEAF = 300
TOP_A = 0.2
N_PERM = 500
ALPHA = 0.05 / 2
PRIMARY = (("dist_tech", "lgbm"), ("dyn", "logit"))
#: 形だけの推移で測り直す回（運用者の選択 2026-11-20「形だけの推移で測り直す」。docs の 98-5）で主に見る2つ
PRIMARY_SHAPE = (("shp", "logit"), ("shp_tech", "lgbm"))
SET_JA = {"dyn": "推移だけ", "dist": "分布すべて", "tech": "直近のテクニカルだけ", "dist_tech": "分布すべて＋直近のテクニカル",
          "shp": "形の推移だけ", "shp_tech": "形の推移＋直近のテクニカル", "all": "形の推移＋分布すべて＋テクニカル"}
MODEL_JA = {"logit": "統計（ロジスティック）", "lgbm": "機械学習（LightGBM）"}


def feature_sets(minute_cols) -> dict:
    dyn = list(MD.DYNAMIC_COLUMNS)
    dist = dyn + list(MD.SHAPE_COLUMNS) + list(minute_cols)
    return {"dyn": dyn, "dist": dist, "tech": list(TECH_COLUMNS), "dist_tech": dist + list(TECH_COLUMNS)}


def shape_sets(minute_cols) -> dict:
    """形だけの推移（minute_dynamics.SHAPE_DYN_COLUMNS）の組。直近のテクニカルだけは前回と同じ列なので比べるために残す。"""
    shp = list(MD.SHAPE_DYN_COLUMNS)
    dist = list(MD.DYNAMIC_COLUMNS) + list(MD.SHAPE_COLUMNS) + list(minute_cols)
    return {"shp": shp, "tech": list(TECH_COLUMNS), "shp_tech": shp + list(TECH_COLUMNS),
            "all": shp + dist + list(TECH_COLUMNS)}


def windows(dates: pd.Series, start=TEST_START, months: int = TEST_MONTHS, embargo: int = EMBARGO_DAYS):
    """(学習の終わり, テストの始まり, テストの終わり) の並び。学習はテストの始まりの embargo 暦日前まで。"""
    d = pd.to_datetime(dates)
    out, s = [], pd.Timestamp(start)
    while s <= d.max():
        e = s + pd.DateOffset(months=months) - pd.Timedelta(days=1)
        out.append((s - pd.Timedelta(days=embargo), s, min(e, d.max())))
        s = s + pd.DateOffset(months=months)
    return out


def _prep(train: pd.DataFrame, test: pd.DataFrame, cols: list):
    """ロジスティック用: 学習側の 1%・99% で丸め、中央値で埋め、標準化（どれも学習側の値だけで決める）。"""
    Xtr = train[cols].to_numpy(dtype=float)
    Xte = test[cols].to_numpy(dtype=float)
    lo, hi = np.nanquantile(Xtr, 0.01, axis=0), np.nanquantile(Xtr, 0.99, axis=0)
    med = np.nanmedian(Xtr, axis=0)
    out = []
    for X in (Xtr, Xte):
        X = np.clip(X, lo, hi)
        X = np.where(np.isfinite(X), X, med)
        out.append(X)
    mu, sd = out[0].mean(axis=0), out[0].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    keep = np.isfinite(med)
    return (out[0] - mu)[:, keep] / sd[keep], (out[1] - mu)[:, keep] / sd[keep], [c for c, k in zip(cols, keep) if k]


def fit_predict(kind: str, train: pd.DataFrame, test: pd.DataFrame, cols: list, alt: bool = False, seed: int = 0):
    """学習して test の点を返す（と、ロジスティックなら標準化した係数）。"""
    y = train["label"].to_numpy(dtype=int)
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression

        Xtr, Xte, used = _prep(train, test, cols)
        m = LogisticRegression(C=LOGIT_C_ALT if alt else LOGIT_C, max_iter=2000)
        m.fit(Xtr, y)
        return m.predict_proba(Xte)[:, 1], dict(zip(used, m.coef_[0]))
    import lightgbm as lgb

    params = {**(LGBM_ALT if alt else LGBM), "random_state": seed}
    pos = max(int(y.sum()), 1)
    m = lgb.LGBMClassifier(**params, scale_pos_weight=(len(y) - pos) / pos)
    m.fit(train[cols].to_numpy(dtype=float), y)
    return m.predict_proba(test[cols].to_numpy(dtype=float))[:, 1], None


def oof(rows: pd.DataFrame, cols: list, kind: str, wins, alt: bool = False):
    """時間順の窓ごとに学習して当てた点（テスト窓の行だけ。ほかは NaN）と、窓の番号・係数。"""
    d = rows["Date"]
    score = np.full(len(rows), np.nan)
    fold = np.full(len(rows), -1)
    coefs = []
    for k, (train_end, s, e) in enumerate(wins):
        tr = rows[d < train_end]
        te_mask = ((d >= s) & (d <= e)).to_numpy()
        if len(tr) < 300 or tr["label"].nunique() < 2 or not te_mask.any():
            continue
        sc, coef = fit_predict(kind, tr, rows[te_mask], cols, alt=alt)
        score[te_mask] = sc
        fold[te_mask] = k
        if coef is not None:
            coefs.append(coef)
    return score, fold, coefs


def day_auc(y: np.ndarray, s: np.ndarray, codes: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    vals = []
    for c in np.unique(codes):
        m = codes == c
        if 0 < y[m].sum() < m.sum():
            vals.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(vals)) if vals else float("nan")


def metrics(y: np.ndarray, s: np.ndarray, codes: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    pr = float(average_precision_score(y, s))
    return {"n": int(len(y)), "pos": float(y.mean()), "pr_auc": pr, "lift": pr / float(y.mean()),
            "roc": float(roc_auc_score(y, s)), "day_auc": day_auc(y, s, codes)}


def _perm_within(codes: np.ndarray, rng):
    n = len(codes)
    base = np.lexsort((np.arange(n), codes))

    def perm():
        out = np.empty(n, dtype=int)
        out[base] = np.lexsort((rng.random(n), codes))
        return out
    return perm


def rank_within(v: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """グループ（窓）ごとの百分位の順位（0〜1）。"""
    return pd.Series(v).groupby(groups).rank(pct=True).to_numpy()


def report(rows: pd.DataFrame, minute_cols, seed: int = 0, n_perm: int = N_PERM, sets: dict | None = None,
           primary=PRIMARY, coef_sets=("dyn", "dist_tech"), rule_set: str = "dist_tech", need=None,
           title: str = "実験133 独立したモデル") -> dict:
    """
    rows: Code・Date・label・score（A の予測）と、列（minute_dynamics.COLUMNS・minute_cols・TECH_COLUMNS）。
    sets・primary・coef_sets・rule_set・need（値が要る列。既定は推移の列）を渡すと、形だけの推移の回（shape_sets・PRIMARY_SHAPE）にも使う。
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(seed)
    sets = sets or feature_sets(minute_cols)
    need = list(need or MD.DYNAMIC_COLUMNS)
    rows = rows[rows[need].notna().any(axis=1)].sort_values(["Date", "Code"], kind="mergesort")
    rows = rows.reset_index(drop=True)
    missing = [c for c in TECH_COLUMNS if c not in rows.columns]
    if missing:
        raise SystemExit(f"直近のテクニカルの列が表に無い: {missing[:5]}")
    wins = windows(rows["Date"])
    out = {"rows": int(len(rows)), "pos": int(rows["label"].sum()), "windows": [[str(a.date()), str(b.date()), str(c.date())]
                                                                           for a, b, c in wins], "models": {}}
    print(f"\n■ {title}: 分足の付いた候補で、20営業日の推移が作れる行 {len(rows):,}（正例 {out['pos']:,}・"
          f"{rows['label'].mean() * 100:.1f}%）。テスト窓 {len(wins)}つ（{wins[0][1].date()} 〜 {wins[-1][2].date()}・3か月ずつ。"
          f"学習はテストの始まりの {EMBARGO_DAYS}暦日前まで）")
    print("  列の数: " + " / ".join(f"{SET_JA[k]} {len(v)}" for k, v in sets.items()))
    print("  推移の列の分位（5% / 50% / 95%）: " + " / ".join(
        f"{c.replace('vp_dyn_', '').replace('vp_shp_', '')} {np.nanquantile(rows[c], .05):.3g}/"
        f"{np.nanquantile(rows[c], .5):.3g}/{np.nanquantile(rows[c], .95):.3g}" for c in need[:20]))

    test_mask = np.zeros(len(rows), dtype=bool)
    scores, folds, coefs_by = {}, None, {}
    for set_key, cols in sets.items():
        for kind in ("logit", "lgbm"):
            sc, fd, cf = oof(rows, cols, kind, wins)
            scores[(set_key, kind)] = sc
            coefs_by[(set_key, kind)] = cf
            folds = fd if folds is None else folds
            test_mask |= np.isfinite(sc)
    t = test_mask & np.isfinite(rows["score"].to_numpy(dtype=float))
    for v in scores.values():
        t &= np.isfinite(v)
    T = rows[t].reset_index(drop=True)
    y = T["label"].to_numpy(dtype=int)
    codes = pd.factorize(T["Date"])[0]
    fd = folds[t]
    a = T["score"].to_numpy(dtype=float)
    ma = metrics(y, a, codes)
    out["test_rows"] = int(len(T))
    out["A"] = ma
    print(f"\n  1. 単独の当たり（テスト窓の行 {len(T):,}・正例率 {y.mean() * 100:.1f}%）。ROC の p は同じ日付の中でラベルを"
          f"入れ替える並べ替え検定（{n_perm}回）")
    print(f"    {'列':<26}{'モデル':<22}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'p(ROC)':>8}  窓ごとの PR-AUC")
    perm = _perm_within(codes, rng)

    def per_window(s):
        return [float(average_precision_score(y[fd == k], s[fd == k])) if 0 < y[fd == k].sum() < (fd == k).sum()
                else float("nan") for k in sorted(set(fd))]

    print(f"    {'A（今のモデル）':<26}{'—':<22}{ma['pr_auc']:>8.4f}{ma['lift']:>6.2f}x{ma['roc']:>8.4f}{ma['day_auc']:>8.4f}"
          f"{'':>8}  " + " / ".join(f"{v:.3f}" for v in per_window(a)))
    out["A"]["per_window"] = per_window(a)
    for (set_key, kind), sc in scores.items():
        s = sc[t]
        m = metrics(y, s, codes)
        null = np.array([roc_auc_score(y[perm()], s) for _ in range(n_perm)])
        m["p_roc"] = float((1 + np.sum(null >= m["roc"])) / (n_perm + 1))
        m["per_window"] = per_window(s)
        out["models"][f"{set_key}/{kind}"] = m
        print(f"    {SET_JA[set_key]:<26}{MODEL_JA[kind]:<22}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc']:>8.4f}"
              f"{m['day_auc']:>8.4f}{m['p_roc']:>8.3f}  " + " / ".join(f"{v:.3f}" for v in m["per_window"]))

    # 2. A との重なりと上積み
    print("\n  2. A との重なりと、A に足したときの上積み（窓ごとに A と新しいモデルの順位を平均した点。並べ替えは新しいモデルの点を"
          "同じ日付の中で入れ替える）")
    print(f"    {'列':<26}{'モデル':<22}{'順位相関':>8}{'組み合わせ':>10}{'上積み':>9}{'p':>7}{'上の窓':>8}")
    ra = rank_within(a, fd)
    pr_a = float(average_precision_score(y, ra))
    pw_a = [float(average_precision_score(y[fd == k], ra[fd == k])) for k in sorted(set(fd))]
    for (set_key, kind), sc in scores.items():
        s = sc[t]
        rho = float(pd.Series(s).corr(pd.Series(a), method="spearman"))
        rs = rank_within(s, fd)
        blend = (ra + rs) / 2
        gain = float(average_precision_score(y, blend)) - pr_a
        null = []
        for _ in range(n_perm):
            rsp = rank_within(s[perm()], fd)
            null.append(float(average_precision_score(y, (ra + rsp) / 2)) - pr_a)
        p = float((1 + np.sum(np.array(null) >= gain)) / (n_perm + 1))
        pw = [float(average_precision_score(y[fd == k], blend[fd == k])) - pw_a[i] for i, k in enumerate(sorted(set(fd)))]
        wins_up = int(np.sum(np.array(pw) > 0))
        rec = out["models"][f"{set_key}/{kind}"]
        rec.update({"spearman_with_a": rho, "blend_pr_auc": gain + pr_a, "blend_gain": gain, "p_blend": p,
                    "blend_gain_per_window": pw, "blend_windows_up": wins_up})
        print(f"    {SET_JA[set_key]:<26}{MODEL_JA[kind]:<22}{rho:>8.3f}{gain + pr_a:>10.4f}{gain:>+9.4f}{p:>7.3f}"
              f"{wins_up:>5}/{len(pw)}")
    print(f"    （A だけの PR-AUC は窓ごとの順位で {pr_a:.4f}）")

    # 3. A の上位20% の中
    top = np.zeros(len(T), dtype=bool)
    for k in sorted(set(fd)):
        m = fd == k
        top[m] = a[m] >= np.quantile(a[m], 1 - TOP_A)
    yt, ct = y[top], pd.factorize(T.loc[top, "Date"])[0]
    permt = _perm_within(ct, rng)
    out["top_a"] = {"rows": int(top.sum()), "pos": float(yt.mean())}
    print(f"\n  3. A の上位{int(TOP_A * 100)}%（窓ごと）の中: {int(top.sum()):,}行・正例率 {yt.mean() * 100:.1f}%"
          f"（全体 {y.mean() * 100:.1f}%）。新しいモデルの点で上半分・下半分に分けた正例率と、その中の AUC")
    print(f"    {'列':<26}{'モデル':<22}{'AUC':>7}{'p':>7}{'上半分':>8}{'下半分':>8}")
    for (set_key, kind), sc in scores.items():
        s = sc[t][top]
        auc = float(roc_auc_score(yt, s)) if 0 < yt.sum() < len(yt) else float("nan")
        null = np.array([roc_auc_score(yt, s[permt()]) for _ in range(n_perm)])
        p = float((1 + np.sum(null >= auc)) / (n_perm + 1))
        hi = s >= np.median(s)
        rec = out["models"][f"{set_key}/{kind}"]
        rec.update({"top_a_auc": auc, "p_top_a": p, "top_a_upper_pos": float(yt[hi].mean()),
                    "top_a_lower_pos": float(yt[~hi].mean())})
        print(f"    {SET_JA[set_key]:<26}{MODEL_JA[kind]:<22}{auc:>7.3f}{p:>7.3f}{yt[hi].mean() * 100:>7.1f}%"
              f"{yt[~hi].mean() * 100:>7.1f}%")

    # 参考: 設定を変えた結果
    print("\n  参考: 設定を変えた結果（LightGBM 葉7・葉の最小200 / ロジスティック C 1.0。単独の PR-AUC と ROC）")
    alt_line = []
    for set_key, cols in sets.items():
        for kind in ("logit", "lgbm"):
            sc, _, _ = oof(rows, cols, kind, wins, alt=True)
            s = sc[t]
            ok = np.isfinite(s)
            if ok.sum() and 0 < y[ok].sum() < ok.sum():
                alt_line.append(f"{SET_JA[set_key]}×{kind} {average_precision_score(y[ok], s[ok]):.4f}/"
                                f"{roc_auc_score(y[ok], s[ok]):.4f}")
    print("    " + " / ".join(alt_line))

    # 4. 統計モデルの係数とルール
    for set_key in coef_sets:
        cols = sets[set_key]
        Xall, _, used = _prep(rows, rows.head(1), cols)
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=LOGIT_C, max_iter=2000).fit(Xall, rows["label"].to_numpy(dtype=int))
        coef = pd.Series(m.coef_[0], index=used)
        per = coefs_by[(set_key, "logit")]
        same = {c: float(np.mean([np.sign(cf.get(c, 0.0)) == np.sign(coef[c]) for cf in per])) if per else float("nan")
                for c in used}
        topc = coef.abs().sort_values(ascending=False).index[:15]
        print(f"\n  4. 統計モデルの係数（{SET_JA[set_key]}。標準化した列1つぶんの変化での対数オッズ。全期間で1本。"
              f"括弧は窓ごとの係数で向きがそろった割合）")
        print("    " + " / ".join(f"{c} {coef[c]:+.3f}（{same[c] * 100:.0f}%）" for c in topc))
        out.setdefault("coefficients", {})[set_key] = {c: {"coef": float(coef[c]), "same_sign": same[c]} for c in used}

    cols = sets[rule_set]
    dates = np.sort(rows["Date"].unique())
    cut = dates[len(dates) // 2]
    first = (rows["Date"] < cut).to_numpy()
    X = rows[cols].to_numpy(dtype=float)
    med = np.nanmedian(X[first], axis=0)
    Xf = np.where(np.isfinite(X), X, med)
    tree = DecisionTreeClassifier(max_depth=TREE_DEPTH, min_samples_leaf=TREE_MIN_LEAF, random_state=0)
    tree.fit(Xf[first], rows.loc[first, "label"].to_numpy(dtype=int))
    leaf = tree.apply(Xf)
    tr_ = tree.tree_

    def path_of(node_id):
        conds, node = [], 0
        while node != node_id:
            f, thr = tr_.feature[node], tr_.threshold[node]
            left, right = tr_.children_left[node], tr_.children_right[node]
            go_left = node_id in _subtree(left)
            conds.append(f"{cols[f]} {'≦' if go_left else '＞'} {thr:.4g}")
            node = left if go_left else right
        return " かつ ".join(conds) or "（すべて）"

    def _subtree(n):
        if n < 0:
            return set()
        return {n} | _subtree(tr_.children_left[n]) | _subtree(tr_.children_right[n])

    yl = rows["label"].to_numpy(dtype=int)
    a_all = rows["score"].to_numpy(dtype=float)
    sec = ~first
    top_sec = np.zeros(len(rows), dtype=bool)
    if sec.any():
        top_sec[sec] = a_all[sec] >= np.nanquantile(a_all[sec], 1 - TOP_A)
    print(f"\n  5. ルール（深さ{TREE_DEPTH}の決定木・葉の最小{TREE_MIN_LEAF}行。前半 〜{pd.Timestamp(cut).date()} で作り、後半で確かめる。"
          f"{SET_JA[rule_set]}）")
    leaves = []
    for lf in sorted(set(leaf)):
        m1, m2 = (leaf == lf) & first, (leaf == lf) & sec
        m3 = (leaf == lf) & top_sec
        rec = {"rule": path_of(lf), "n_first": int(m1.sum()), "pos_first": float(yl[m1].mean()) if m1.any() else float("nan"),
               "n_second": int(m2.sum()), "pos_second": float(yl[m2].mean()) if m2.any() else float("nan"),
               "n_top_a_second": int(m3.sum()), "pos_top_a_second": float(yl[m3].mean()) if m3.any() else float("nan")}
        leaves.append(rec)
        print(f"    {rec['rule']}: 前半 {rec['n_first']:,}行 {rec['pos_first'] * 100:.1f}% / 後半 {rec['n_second']:,}行 "
              f"{rec['pos_second'] * 100:.1f}% / 後半の A の上位{int(TOP_A * 100)}% の中 {rec['n_top_a_second']:,}行 "
              f"{rec['pos_top_a_second'] * 100:.1f}%")
    print(f"    （全体の正例率 前半 {yl[first].mean() * 100:.1f}% / 後半 {yl[sec].mean() * 100:.1f}%・後半の A の上位"
          f"{int(TOP_A * 100)}% {yl[top_sec].mean() * 100:.1f}%）")
    out["rule"] = {"cut": str(pd.Timestamp(cut).date()), "leaves": leaves}

    # 読み
    read = {}
    for set_key, kind in primary:
        r = out["models"][f"{set_key}/{kind}"]
        read[f"{set_key}/{kind}"] = {
            "単独で当たる": r["p_roc"] < ALPHA,
            "今のモデルにない情報": (r["blend_gain"] > 0 and r["p_blend"] < ALPHA
                                    and r["blend_windows_up"] > len(r["blend_gain_per_window"]) / 2),
            "A の上位を絞り込める": r["p_top_a"] < ALPHA,
        }
    print(f"\n  読み（回す前に決めたもの。主に見る2つ。有意水準 {ALPHA:.3f}）")
    for k, v in read.items():
        s_, m_ = k.split("/")
        print(f"    {SET_JA[s_]} × {MODEL_JA[m_]}: " + " / ".join(f"{kk} {'はい' if vv else 'いいえ'}" for kk, vv in v.items()))
    out["reading"] = read
    return out


# --------------------------------------------------------------------------- #
# A の上位の中の逆向き（運用者の選択 2026-11-20「上位の中の逆向きを確かめる」。docs の 98-7）
# --------------------------------------------------------------------------- #
#: 候補（分布・形の列を使うモデル）
INVERSE_CANDIDATES = (("dyn", "logit"), ("dyn", "lgbm"), ("dist", "logit"), ("dist", "lgbm"), ("shp", "logit"), ("shp", "lgbm"))
#: 前半（どのモデルを使うかを決める窓）と後半（確かめる窓）。窓の番号は windows() の順
DISCOVER_WINDOWS = (0, 1, 2)
CONFIRM_WINDOWS = (3, 4, 5)
INVERSE_ALPHA = 0.05


def inverse_check(rows: pd.DataFrame, minute_cols, seed: int = 0, n_perm: int = 1000) -> dict:
    """
    A の上位20%（窓ごと）の中で、分布・形のモデルの点が高いほど正例が少ないか。前半の窓で候補のうちいちばん逆向き（その中の AUC が
    いちばん低い）のモデルを選び、後半の窓で「そのモデルの点が上半分の候補を外す」を確かめる（外し方は回す前に決めた）。
    後半の検定は、A の上位の行の中で、新しいモデルの点を同じ日付の中で入れ替える並べ替え（片側。残した側 − 外した側が観測以上）。
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    sets = {**feature_sets(minute_cols), **shape_sets(minute_cols)}
    need = list(MD.DYNAMIC_COLUMNS) + list(MD.SHAPE_DYN_COLUMNS)
    rows = rows[rows[need].notna().any(axis=1)].sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    wins = windows(rows["Date"])
    scores, fold = {}, None
    for key in INVERSE_CANDIDATES:
        sc, fd, _ = oof(rows, sets[key[0]], key[1], wins)
        scores[key] = sc
        fold = fd if fold is None else fold
    a = rows["score"].to_numpy(dtype=float)
    ok = (fold >= 0) & np.isfinite(a)
    for sc in scores.values():
        ok &= np.isfinite(sc)
    top = np.zeros(len(rows), dtype=bool)
    for k in sorted(set(fold[ok])):
        m = ok & (fold == k)
        top[m] = a[m] >= np.quantile(a[m], 1 - TOP_A)
    y = rows["label"].to_numpy(dtype=int)
    disc = top & np.isin(fold, DISCOVER_WINDOWS)
    conf = top & np.isin(fold, CONFIRM_WINDOWS)
    out = {"windows": [[str(x.date()) for x in w] for w in wins], "discover_rows": int(disc.sum()),
           "confirm_rows": int(conf.sum()), "candidates": {}}
    print(f"\n■ A の上位{int(TOP_A * 100)}% の中の逆向き: 前半の窓 {[str(wins[k][1].date()) for k in DISCOVER_WINDOWS]} で選び、"
          f"後半の窓 {[str(wins[k][1].date()) for k in CONFIRM_WINDOWS]} で確かめる。A の上位の行 前半 {int(disc.sum()):,}"
          f"（正例率 {y[disc].mean() * 100:.1f}%）・後半 {int(conf.sum()):,}（正例率 {y[conf].mean() * 100:.1f}%）")

    def halves(sc, mask):
        """窓ごとに、A の上位の行を新しいモデルの点の中央で上半分・下半分に分ける。"""
        upper = np.zeros(len(rows), dtype=bool)
        for k in sorted(set(fold[mask])):
            m = mask & (fold == k)
            upper[m] = sc[m] >= np.median(sc[m])
        return upper

    print(f"  1. 前半の窓での、A の上位の中の AUC（低いほど逆向き）と、上半分・下半分の正例率")
    for key in INVERSE_CANDIDATES:
        sc = scores[key]
        up = halves(sc, disc)
        auc = float(roc_auc_score(y[disc], sc[disc]))
        rec = {"discover_auc": auc, "discover_upper_pos": float(y[disc & up].mean()),
               "discover_lower_pos": float(y[disc & ~up].mean())}
        out["candidates"][f"{key[0]}/{key[1]}"] = rec
        print(f"    {SET_JA[key[0]]} × {MODEL_JA[key[1]]}: AUC {auc:.3f}・上半分 {rec['discover_upper_pos'] * 100:.1f}%・"
              f"下半分 {rec['discover_lower_pos'] * 100:.1f}%")
    chosen = min(INVERSE_CANDIDATES, key=lambda k: out["candidates"][f"{k[0]}/{k[1]}"]["discover_auc"])
    out["chosen"] = f"{chosen[0]}/{chosen[1]}"
    print(f"  選んだモデル（前半でいちばん逆向き）: {SET_JA[chosen[0]]} × {MODEL_JA[chosen[1]]}")

    # 2. 後半で確かめる
    sc = scores[chosen]
    up = halves(sc, conf)
    kept, cut = conf & ~up, conf & up
    obs = float(y[kept].mean() - y[cut].mean())
    codes = pd.factorize(rows["Date"])[0]
    idx = np.flatnonzero(conf)
    perm_codes = codes[idx]
    base = np.lexsort((np.arange(len(idx)), perm_codes))
    null = []
    for _ in range(n_perm):
        shuffled = np.empty(len(idx), dtype=int)
        shuffled[base] = np.lexsort((rng.random(len(idx)), perm_codes))
        s2 = sc.copy()
        s2[idx] = sc[idx][shuffled]
        u2 = halves(s2, conf)
        null.append(float(y[conf & ~u2].mean() - y[conf & u2].mean()))
    p = float((1 + np.sum(np.array(null) >= obs)) / (n_perm + 1))
    per = []
    for k in CONFIRM_WINDOWS:
        m = conf & (fold == k)
        if m.sum() == 0:
            continue
        per.append({"window": str(wins[k][1].date()), "rows": int(m.sum()), "all": float(y[m].mean()),
                    "kept": float(y[m & ~up].mean()), "cut": float(y[m & up].mean())})
    same = sum(1 for r in per if r["kept"] > r["cut"])
    out["confirm"] = {"all_pos": float(y[conf].mean()), "kept_pos": float(y[kept].mean()), "cut_pos": float(y[cut].mean()),
                      "kept_rows": int(kept.sum()), "cut_rows": int(cut.sum()), "diff": obs, "p": p,
                      "per_window": per, "windows_same_direction": same,
                      "auc": float(roc_auc_score(y[conf], sc[conf]))}
    print(f"\n  2. 後半の窓で確かめる（選んだモデルの点の上半分を外す）: A の上位 {int(conf.sum()):,}行の正例率 {y[conf].mean() * 100:.1f}% → "
          f"残した下半分 {int(kept.sum()):,}行 {y[kept].mean() * 100:.1f}%・外した上半分 {int(cut.sum()):,}行 {y[cut].mean() * 100:.1f}%"
          f"（差 {obs * 100:+.1f}ポイント・並べ替え {n_perm}回の片側 p {p:.3f}・AUC {out['confirm']['auc']:.3f}）")
    for r in per:
        print(f"    {r['window']} からの窓: {r['rows']:,}行 {r['all'] * 100:.1f}% → 残した側 {r['kept'] * 100:.1f}%・"
              f"外した側 {r['cut'] * 100:.1f}%")
    print("  参考（後半の窓。ほかの候補で同じ外し方をした場合）: " + " / ".join(
        f"{SET_JA[k[0]]}×{k[1]} 残した側 {y[conf & ~halves(scores[k], conf)].mean() * 100:.1f}%・外した側 "
        f"{y[conf & halves(scores[k], conf)].mean() * 100:.1f}%" for k in INVERSE_CANDIDATES))

    # 3. 選んだモデルの点は何を表すか（前半の A の上位の中で、直近のテクニカルの列との順位相関）
    rho = {}
    for c in TECH_COLUMNS:
        v = rows.loc[disc, c]
        if v.notna().sum() > 30:
            rho[c] = float(pd.Series(sc[disc]).corr(v.reset_index(drop=True), method="spearman"))
    topc = sorted(rho, key=lambda c: -abs(rho[c]))[:8]
    out["what_it_is"] = {c: rho[c] for c in topc}
    print("  3. 選んだモデルの点と、A の上位の中で重なる直近のテクニカルの列（前半の窓の順位相関。上位8つ）: "
          + " / ".join(f"{c} {rho[c]:+.2f}" for c in topc))

    real = obs > 0 and p < INVERSE_ALPHA and same >= 2
    out["reading"] = {"inverse_is_real": bool(real)}
    print(f"\n  読み（回す前に決めたもの。後半で、残した側 − 外した側が正・並べ替えの片側 p < {INVERSE_ALPHA}・後半の3つの窓の2つ以上で"
          f"同じ向き）: {'逆向きは偶然ではない（外すルールの候補になる）' if real else '逆向きは確かめられない'}")
    return out


# --------------------------------------------------------------------------- #
# A にない情報を公平に測る（運用者の選択 2026-11-20「公平な測り方で測り直す」。docs の 98-10）
#   98-4・98-6 の「A と等しい重みで混ぜて上積みが出るか」は、A よりずっと弱いモデルを重くしすぎて、小さい独立の情報を
#   見つけられない作りだった（98-9）。ここでは、残差（A の予測を土台にしたロジスティック回帰に足す）と、重みを過去の窓で
#   決めて A に足す方法で測る。新しいモデルの点は、同じ日の候補の中の順位だけを使う
# --------------------------------------------------------------------------- #
#: 候補の列の組（98-4 の4つと、98-6 で増えた3つ）× ロジスティック・LightGBM
INCREMENTAL_SETS = ("dyn", "dist", "tech", "dist_tech", "shp", "shp_tech", "all")
INCREMENTAL_ALPHA = 0.05
#: 重みを過去の窓で決めるときに、学習に要る最小の行数と、上積みが正であってほしい窓の数
STACK_MIN_ROWS = 200
STACK_MIN_UP = 3


def date_normal(v, codes: np.ndarray) -> np.ndarray:
    """同じ日付の中の順位を標準正規の分位に直した値（1行だけの日は 0）。ほかの日付の値は使わない。"""
    from scipy.special import ndtri

    g = pd.Series(np.asarray(v, dtype=float)).groupby(codes)
    r = g.rank(method="average").to_numpy()
    n = g.transform("size").to_numpy()
    return ndtri((r - 0.5) / n)


def fast_day_auc(y, s, codes: np.ndarray) -> float:
    """day_auc と同じ値（両方のラベルがある日の、日付の中の AUC の平均）を、日付ごとの順位の和から求める。codes は 0 から。"""
    r = pd.Series(np.asarray(s, dtype=float)).groupby(codes).rank(method="average").to_numpy()
    y = np.asarray(y, dtype=float)
    n = np.bincount(codes).astype(float)
    n1 = np.bincount(codes, weights=y, minlength=len(n))
    r1 = np.bincount(codes, weights=r * y, minlength=len(n))
    n0 = n - n1
    ok = (n1 > 0) & (n0 > 0)
    if not ok.any():
        return float("nan")
    return float(np.mean((r1[ok] - n1[ok] * (n1[ok] + 1) / 2) / (n1[ok] * n0[ok])))


def _logit(X: np.ndarray, y: np.ndarray, beta=None, cov: bool = False, iters: int = 50, tol: float = 1e-9):
    """罰則の無いロジスティック回帰（ニュートン法）。(係数, 対数尤度, 係数の共分散 か None) を返す。"""
    k = X.shape[1]
    b = np.zeros(k) if beta is None else np.asarray(beta, dtype=float).copy()
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -35, 35)))
        H = (X * (p * (1.0 - p))[:, None]).T @ X + 1e-10 * np.eye(k)
        step = np.linalg.solve(H, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ b
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    if not cov:
        return b, ll, None
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    H = (X * (p * (1.0 - p))[:, None]).T @ X + 1e-10 * np.eye(k)
    return b, ll, np.linalg.inv(H)


def _base(score) -> np.ndarray:
    """A の予測の logit（標準化）とその2乗（97-9 の残差の土台と同じ。A の予測の目盛りのずれを吸収する）。"""
    s = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(s / (1 - s))
    z = (z - z.mean()) / (z.std() or 1.0)
    return np.column_stack([np.ones(len(z)), z, z * z])


def walk_forward(y, la, v, fold, dates, wins, eval_windows, start=None):
    """
    窓 k（eval_windows）ごとに「ラベル ~ 1 + A の logit + v」を、その窓の始まりの EMBARGO_DAYS 暦日前までの、前のテスト窓の行で
    学習し、窓 k の行の点（A の logit と v の重み付きの和）にする。学習できない窓の行は NaN。start は窓ごとの係数の初期値。
    """
    out = np.full(len(y), np.nan)
    coefs = {}
    for k in eval_windows:
        tr = (fold >= 0) & (fold < k) & (dates < np.datetime64(wins[k][0]))
        te = fold == k
        if tr.sum() < STACK_MIN_ROWS or not te.any() or y[tr].min() == y[tr].max():
            continue
        X = np.column_stack([np.ones(int(tr.sum())), la[tr], v[tr]])
        b, _, _ = _logit(X, y[tr], beta=None if start is None else start.get(k))
        out[te] = b[1] * la[te] + b[2] * v[te]
        coefs[k] = b
    return out, coefs


def _stack_gain(y, st, la, fold, codes, eval_windows):
    """重みを過去の窓で決めた点の日内 AUC − A の日内 AUC（点のある窓をまとめて）と、窓ごとの差。"""
    m = np.isfinite(st)
    if not m.any():
        return float("nan"), []
    c = pd.factorize(codes[m])[0]
    total = fast_day_auc(y[m], st[m], c) - fast_day_auc(y[m], la[m], c)
    per = []
    for k in eval_windows:
        w = m & (fold == k)
        if w.any():
            cw = pd.factorize(codes[w])[0]
            per.append(fast_day_auc(y[w], st[w], cw) - fast_day_auc(y[w], la[w], cw))
    return float(total), per


def incremental_check(rows: pd.DataFrame, minute_cols, seed: int = 0, n_perm: int = 1000) -> dict:
    """
    独立したモデル（14の組）が A にない情報を持つか。残差（A の予測を土台に足した ΔLL）と、重みを過去の窓で決めて A に足した
    日内 AUC の上積み。並べ替えは新しいモデルの点を同じ日付の中で入れ替える（14の組に同じ入れ替え。調整後 p は 14の組の最大と比べる）。
    """
    from scipy.stats import norm
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    sets = {**feature_sets(minute_cols), **shape_sets(minute_cols)}
    need = list(MD.DYNAMIC_COLUMNS) + list(MD.SHAPE_DYN_COLUMNS)
    rows = rows[rows[need].notna().any(axis=1)].sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    wins = windows(rows["Date"])
    keys = [(s, k) for s in INCREMENTAL_SETS for k in ("logit", "lgbm")]
    scores, fold = {}, None
    for key in keys:
        sc, fd, _ = oof(rows, sets[key[0]], key[1], wins)
        scores[key] = sc
        fold = fd if fold is None else fold
    a = rows["score"].to_numpy(dtype=float)
    ok = (fold >= 0) & np.isfinite(a)
    for sc in scores.values():
        ok &= np.isfinite(sc)
    T = rows[ok].reset_index(drop=True)
    y = T["label"].to_numpy(dtype=float)
    codes = pd.factorize(T["Date"])[0]
    fd = fold[ok]
    dates = T["Date"].to_numpy()
    a = a[ok]
    ac = np.clip(a, 1e-6, 1 - 1e-6)
    la = np.log(ac / (1 - ac))
    V = {}
    for key in keys:
        v = date_normal(scores[key][ok], codes)
        V[key] = v / (v.std() or 1.0)
    eval_windows = list(range(1, len(wins)))
    auc_a = fast_day_auc(y, a, codes)
    d_a = float(np.sqrt(2) * norm.ppf(auc_a))
    a_n = date_normal(a, codes)
    out = {"rows": int(len(rows)), "test_rows": int(len(T)), "pos": float(y.mean()), "dates": int(codes.max() + 1),
           "a_day_auc": auc_a, "d_a": d_a, "windows": [[str(x.date()) for x in w] for w in wins], "candidates": {}}
    print(f"\n■ A にない情報を公平に測る: テスト窓の行 {len(T):,}（正例率 {y.mean() * 100:.1f}%・{out['dates']}日）。"
          f"A の日内 AUC {auc_a:.4f}（正例と負例の点の差 {d_a:.3f}）")
    print("  新しいモデルの点は、同じ日の候補の中の順位を正規分布の分位に直した値（日付をまたぐずれを入れない）")

    # 1. 残差
    X0 = _base(a)
    b0, ll0, _ = _logit(X0, y)
    Q, _ = np.linalg.qr(X0)

    def dll(v):
        return _logit(np.column_stack([X0, v]), y, beta=np.append(b0, 0.0))[1] - ll0

    obs = {}
    for key in keys:
        v = V[key]
        b1, ll1, cv = _logit(np.column_stack([X0, v]), y, beta=np.append(b0, 0.0), cov=True)
        se = float(np.sqrt(cv[-1, -1]))
        sd_rest = float((v - Q @ (Q.T @ v)).std())
        d_pt, d_hi = abs(b1[-1]) * sd_rest, (abs(b1[-1]) + 1.96 * se) * sd_rest
        obs[key] = {"day_auc": fast_day_auc(y, scores[key][ok], codes),
                    "within_corr_a": float(np.corrcoef(date_normal(scores[key][ok], codes), a_n)[0, 1]),
                    "coef": float(b1[-1]), "se": se, "dll": float(ll1 - ll0), "sd_rest": sd_rest,
                    "gain_point": float(norm.cdf(np.hypot(d_a, d_pt) / np.sqrt(2)) - norm.cdf(d_a / np.sqrt(2))),
                    "gain_upper": float(norm.cdf(np.hypot(d_a, d_hi) / np.sqrt(2)) - norm.cdf(d_a / np.sqrt(2)))}

    # 2. 重みを過去の窓で決めて足す
    for key in keys:
        st, coefs = walk_forward(y, la, V[key], fd, dates, wins, eval_windows)
        g, per = _stack_gain(y, st, la, fd, codes, eval_windows)
        pr = []
        for k in eval_windows:
            w = np.isfinite(st) & (fd == k)
            if w.any() and 0 < y[w].sum() < w.sum():
                pr.append(float(average_precision_score(y[w], st[w]) - average_precision_score(y[w], la[w])))
        obs[key].update({"stack_gain": g, "stack_per_window": per, "stack_up": int(np.sum(np.array(per) > 0)),
                         "stack_weight": float(np.mean([c[2] for c in coefs.values()])) if coefs else float("nan"),
                         "stack_pr_gain": float(np.mean(pr)) if pr else float("nan"), "_start": coefs})

    # 並べ替え（14の組に同じ入れ替え）。学習できる窓は v によらないので、A の日内 AUC は先に1度だけ求める
    m_eval = np.isfinite(walk_forward(y, la, V[keys[0]], fd, dates, wins, eval_windows)[0])
    c_eval = pd.factorize(codes[m_eval])[0]
    auc_a_eval = fast_day_auc(y[m_eval], la[m_eval], c_eval) if m_eval.any() else float("nan")
    perm = _perm_within(codes, rng)
    null_dll = np.zeros((n_perm, len(keys)))
    null_gain = np.zeros((n_perm, len(keys)))
    for i in range(n_perm):
        idx = perm()
        for j, key in enumerate(keys):
            vp = V[key][idx]
            null_dll[i, j] = dll(vp)
            st, _ = walk_forward(y, la, vp, fd, dates, wins, eval_windows, start=obs[key]["_start"])
            if not np.array_equal(np.isfinite(st), m_eval):
                raise RuntimeError("重みを過去の窓で決める行が、並べ替えで変わった")
            null_gain[i, j] = (fast_day_auc(y[m_eval], st[m_eval], c_eval) - auc_a_eval) if m_eval.any() else np.nan
    max_dll, max_gain = null_dll.max(axis=1), null_gain.max(axis=1)
    for j, key in enumerate(keys):
        r = obs[key]
        r.pop("_start")
        r["p_dll"] = float((1 + np.sum(null_dll[:, j] >= r["dll"])) / (n_perm + 1))
        r["p_dll_adj"] = float((1 + np.sum(max_dll >= r["dll"])) / (n_perm + 1))
        r["p_stack"] = float((1 + np.sum(null_gain[:, j] >= r["stack_gain"])) / (n_perm + 1))
        r["p_stack_adj"] = float((1 + np.sum(max_gain >= r["stack_gain"])) / (n_perm + 1))
        out["candidates"][f"{key[0]}/{key[1]}"] = r

    print(f"\n  1. 残差（土台は A の予測の logit とその2乗のロジスティック回帰。新しいモデルの点を足した対数尤度の伸び ΔLL。"
          f"並べ替えは同じ日付の中で {n_perm}回。調整後 p は 14の組の最大と比べる。上積みは係数を日内 AUC に直したもの）")
    print(f"    {'列の組':<26}{'モデル':<22}{'日内AUC':>8}{'日内の相関':>9}{'係数':>8}{'標準誤差':>8}{'ΔLL':>7}{'p':>7}"
          f"{'調整後p':>8}{'上積み':>9}{'上限':>9}")
    for key in keys:
        r = out["candidates"][f"{key[0]}/{key[1]}"]
        print(f"    {SET_JA[key[0]]:<26}{MODEL_JA[key[1]]:<22}{r['day_auc']:>8.4f}{r['within_corr_a']:>9.3f}{r['coef']:>+8.3f}"
              f"{r['se']:>8.3f}{r['dll']:>7.2f}{r['p_dll']:>7.3f}{r['p_dll_adj']:>8.3f}{r['gain_point']:>+9.4f}"
              f"{r['gain_upper']:>+9.4f}")
    out["a_day_auc_stack_windows"] = auc_a_eval
    print(f"\n  2. 重みを過去の窓で決めて A に足す（{len(eval_windows)}つの窓。各窓の重みは、その窓の始まりの {EMBARGO_DAYS}暦日前"
          f"までの前の窓の行で決める。この窓での A の日内 AUC {auc_a_eval:.4f}。並べ替えは 1. と同じ入れ替えで学習からやり直す）")
    print(f"    {'列の組':<26}{'モデル':<22}{'重み':>8}{'上積み':>9}{'上の窓':>7}{'p':>7}{'調整後p':>8}{'PR-AUC差':>10}"
          f"  窓ごとの日内 AUC の差")
    for key in keys:
        r = out["candidates"][f"{key[0]}/{key[1]}"]
        print(f"    {SET_JA[key[0]]:<26}{MODEL_JA[key[1]]:<22}{r['stack_weight']:>+8.3f}{r['stack_gain']:>+9.4f}"
              f"{r['stack_up']:>4}/{len(r['stack_per_window'])}{r['p_stack']:>7.3f}{r['p_stack_adj']:>8.3f}"
              f"{r['stack_pr_gain']:>+10.4f}  " + " / ".join(f"{v:+.3f}" for v in r["stack_per_window"]))

    info = [k for k, r in out["candidates"].items() if r["p_dll_adj"] < INCREMENTAL_ALPHA]
    usable = [k for k, r in out["candidates"].items()
              if r["stack_gain"] > 0 and r["p_stack_adj"] < INCREMENTAL_ALPHA and r["stack_up"] >= STACK_MIN_UP]
    best = max(out["candidates"].items(), key=lambda kv: kv[1]["gain_upper"])
    out["reading"] = {"information": bool(info), "information_sets": info, "usable": bool(usable), "usable_sets": usable,
                      "largest_upper": {"set": best[0], "gain_upper": best[1]["gain_upper"]}}

    def ja(k):
        s_, m_ = k.split("/")
        return f"{SET_JA[s_]} × {MODEL_JA[m_]}"

    print(f"\n  読み（回す前に決めたもの。残差の調整後 p < {INCREMENTAL_ALPHA}・足した上積みが正で調整後 p < {INCREMENTAL_ALPHA}"
          f"かつ {STACK_MIN_UP}つ以上の窓で正）")
    print(f"    A にない情報がある: {'はい（' + ' / '.join(ja(k) for k in info) + '）' if info else 'いいえ'}")
    print(f"    A に足して使える: {'はい（' + ' / '.join(ja(k) for k in usable) + '）' if usable else 'いいえ'}")
    if not info and not usable:
        print(f"    見つからない。あるとしても、日内 AUC の上積みは大きくて {best[1]['gain_upper']:+.4f}"
              f"（95% 区間の上の端。14の組でいちばん大きい {ja(best[0])}）")
    return out
