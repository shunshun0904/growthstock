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
SET_JA = {"dyn": "推移だけ", "dist": "分布すべて", "tech": "直近のテクニカルだけ", "dist_tech": "分布すべて＋直近のテクニカル"}
MODEL_JA = {"logit": "統計（ロジスティック）", "lgbm": "機械学習（LightGBM）"}


def feature_sets(minute_cols) -> dict:
    dyn = list(MD.DYNAMIC_COLUMNS)
    dist = dyn + list(MD.SHAPE_COLUMNS) + list(minute_cols)
    return {"dyn": dyn, "dist": dist, "tech": list(TECH_COLUMNS), "dist_tech": dist + list(TECH_COLUMNS)}


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


def report(rows: pd.DataFrame, minute_cols, seed: int = 0, n_perm: int = N_PERM) -> dict:
    """rows: Code・Date・label・score（A の予測）と、列（minute_dynamics.COLUMNS・minute_cols・TECH_COLUMNS）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(seed)
    sets = feature_sets(minute_cols)
    rows = rows[rows[list(MD.DYNAMIC_COLUMNS)].notna().any(axis=1)].sort_values(["Date", "Code"], kind="mergesort")
    rows = rows.reset_index(drop=True)
    missing = [c for c in TECH_COLUMNS if c not in rows.columns]
    if missing:
        raise SystemExit(f"直近のテクニカルの列が表に無い: {missing[:5]}")
    wins = windows(rows["Date"])
    out = {"rows": int(len(rows)), "pos": int(rows["label"].sum()), "windows": [[str(a.date()), str(b.date()), str(c.date())]
                                                                           for a, b, c in wins], "models": {}}
    print(f"\n■ 実験133 独立したモデル: 分足の付いた候補で、20営業日の推移が作れる行 {len(rows):,}（正例 {out['pos']:,}・"
          f"{rows['label'].mean() * 100:.1f}%）。テスト窓 {len(wins)}つ（{wins[0][1].date()} 〜 {wins[-1][2].date()}・3か月ずつ。"
          f"学習はテストの始まりの {EMBARGO_DAYS}暦日前まで）")
    print("  列の数: " + " / ".join(f"{SET_JA[k]} {len(v)}" for k, v in sets.items()))
    print("  推移の列の分位（5% / 50% / 95%）: " + " / ".join(
        f"{c.replace('vp_dyn_', '')} {np.nanquantile(rows[c], .05):.3g}/{np.nanquantile(rows[c], .5):.3g}/"
        f"{np.nanquantile(rows[c], .95):.3g}" for c in MD.DYNAMIC_COLUMNS[:17]))

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
    for set_key in ("dyn", "dist_tech"):
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

    cols = sets["dist_tech"]
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
          f"{SET_JA['dist_tech']}）")
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
    for set_key, kind in PRIMARY:
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
