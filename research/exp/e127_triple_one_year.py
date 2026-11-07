#!/usr/bin/env python3
"""
実験127: 目的変数を「1年以内に買値の3倍」「半年以内に買値の2倍」にしたとき、列の区分ごとの重要度（gain の割合）が今の +10%
ラベルとどう違うか（検証だけ。本番のモデル・ラベル・買いの規則・画面は変えない）。

運用者の依頼（2026-11-07）
  「目的変数のラベルを1年後に購入価格の3倍に到達するかどうかでラベルを作ってほしいです。
    仮説としては、正例のものはかなりファンダメンタルズが効いてくるはずで、今の本番のモデルよりは、テクニカルの重要度は
    下がってくると思います。」
  選んだもの（AskUserQuestion）: 判定は「1年以内に届けば正例」、母集団は「今と同じ新高値の候補」、確かめ方は「重要度の割合だけ」。
  追加の依頼（同日）「もう一つ、半年後に買値の2倍 も検証してほしいです。」→ 3倍と同じ考え方（半年＝125営業日以内に届けば正例）で
  同じ回に並べる。

ラベル（TARGETS。どれも同じ作りで、窓の長さ h と倍率 m だけが違う: 3倍は h=250・m=3、2倍は h=125・m=2）
  買値   翌営業日の寄り（分割調整後の始値。データセットの entry_price と同じ取り方: AdjO を素の O で埋めたもの）
  正例   t+1〜t+h 営業日の終値（AdjC を素の C で埋めたもの）の最大が 買値×m 以上（本番と同じく終値で測る。高値で測ると一瞬
         触れただけも正例になる。比べ方は label_specs.at_least と同じ許容誤差）
  負例   h 営業日先まで行があって届かなかった行。または、h 営業日たつ前に売買が止まり（その銘柄の最後の日が全体の最後の日の
         STOP_GAP 営業日より前。上場廃止など）、それまでに届かなかった行（届かずに消えた銘柄を外すと、負例が健全な銘柄に偏る）
  不明   先がまだ h 営業日ない行（3倍は直近1年ほど）・買値が無い行。使わない（途中で届いた行だけを正例にすると、直近の行が正例に偏る）
  窓の数え方は本番の build_dataset.attach_rise_label と同じ（銘柄ごとの行で数える。行が無い日は数えない）。
  使う行は、すべてのラベル（+10%・3倍・2倍）が分かる行（3つのラベルを同じ行で比べるため。2倍だけなら使える直近半年ぶんは使わない）。

比べ方（読み方に入れるもの）
  同じ行（すべてのラベルが分かる新高値の候補）・同じ373列（features.DEFAULT_PRESET）・同じ木（本番の research/lgbm_params.json の
  373列のパラメータ。木200本。scale_pos_weight = 負例/正例 は本番と同じ扱い）で、ラベルだけを替えて LightGBM を学習し、
  列ごとの gain を合計に対する割合にして、区分ごとに足す。ラベルの効き目だけを見るため、新しいラベル用に探索し直しはしない。
  ぶれ: 学習に使う期間（判断日が 2021・2022・2023・2024 年末まで / 全部）× 種（0・1・2）の15組。

区分（グループ → 区分。financial_inputs の13列は中身がまちまちなので列ごと）
  tech    テクニカル（値動き・出来高の形）: tech_indicators・price・breakout・breakout_shape・drawdown・recent_change・volume・
          vol_factors
  fund    決算・財務・会社予想: fund_level・fund_lag・fund_trend・fund_streak・fund_book・cashflow・efficiency・guidance・
          turnaround・revision・fwd・progress_seasonal、financial_inputs の残余利益・配当割引の有効性4列
  val     バリュエーション・配当: valuation・dividend、financial_inputs の益利回り・純資産利回り・配当利回り・配当性向
  supply  需給（信用・空売り・大株主・投資部門）: supply・margin_alert・flow・holders_lvs・holders_major・holders_cross
  market  地合い（市場・業種の値動き・金利・ベータ）: market・sector_index、financial_inputs のベータ2列・安全利子率2列
  other   規模・属性・時期: liquidity・scale・sector・listing・timing・earn_ahead、financial_inputs の財務の古さ

読み方（回す前に決めたもの。採用の判定ではない。3倍・2倍それぞれで別々に）
  テクニカルの割合（gain の合計に占める tech の割合。%の小数1桁に丸めて）を、15組それぞれで 新しいラベル と +10% で比べる。
  - 15組すべてで 新しいラベル < +10% →「仮説どおり（新しいラベルのほうがテクニカルの割合が低い）」
  - 15組すべてで 新しいラベル > +10% →「仮説と逆（新しいラベルのほうがテクニカルの割合が高い）」
  - それ以外 →「はっきりしない」
  見るだけ: 決算・財務＋バリュエーション（fund+val）の割合を同じ形で。区分・グループ・列ごとの割合（全期間・種の平均）。

参考（読み方には入れない）
  前進分割の PR-AUC。テストは判断日が 2021〜2025 年の暦年ごと、学習はテストの始まりの「いちばん長い窓の営業日数×1.5＋5」暦日
  （250営業日なら380日）より前の行（3倍のラベルは1年先まで見るので、それより近い行は答えがテスト期間と重なる）。+10%・2倍も同じ
  分割（比べやすさのため）。重要度は学習したモデルが
  何を使ったかであって、当たるかどうかとは別なので、当たり具合の目安として出す。

確かめ（合わなければ止まる）
  - +10% のラベルが本番の定義（label_specs.check_training_frame）
  - 買値: この台本の翌営業日の寄りとデータセットの entry_price が一致（相対誤差 MATCH_TOL。合わない行が MAX_MISMATCH を超えたら止める）
  - 窓の取り方: 同じ関数で h=20 の値を作り、データセットの future_rise と一致（同上）
  - 区分: 373列すべてに区分が付く。パラメータが373列で探索したもの（列の指紋）

公開ログには集計だけ（件数・割合・区分名・列名・指標）。銘柄ごとの値は出さない。結果は research/_data/oof/e127_results.json。

    exp=e127_triple_one_year.py  args=""   （既定は --targets x3:250:3,x2:125:2 --seeds 0,1,2）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import features as F  # noqa: E402
import label_specs as LS  # noqa: E402
import lab  # noqa: E402
import tuning  # noqa: E402

#: 新しいラベル（列名・窓の営業日数・倍率）。運用者の依頼の順
TARGETS = (("x3", 250, 3.0), ("x2", 125, 2.0))
STOP_GAP = 10
SEEDS = (0, 1, 2)
WINDOW_ENDS = ("2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "all")
OOF_YEARS = (2021, 2022, 2023, 2024, 2025)
#: 前進分割（参考）で学習とテストのあいだに空ける暦日。いちばん長い窓の営業日数×1.5＋5（250営業日なら380日）
PURGE_PER_DAY = 1.5
PURGE_PAD = 5
MATCH_TOL = 1e-9
MAX_MISMATCH = 0.001
MIN_POS = 20
OUT = os.path.join(lab.DATA_DIR, "oof", "e127_results.json")

CATEGORIES = (("tech", "テクニカル（値動き・出来高の形）"),
              ("fund", "決算・財務・会社予想"),
              ("val", "バリュエーション・配当"),
              ("supply", "需給（信用・空売り・大株主・投資部門）"),
              ("market", "地合い（市場・業種の値動き・金利・ベータ）"),
              ("other", "規模・属性・時期"))
GROUP_CATEGORY = {
    "tech_indicators": "tech", "price": "tech", "breakout": "tech", "breakout_shape": "tech",
    "drawdown": "tech", "recent_change": "tech", "volume": "tech", "vol_factors": "tech",
    "fund_level": "fund", "fund_lag": "fund", "fund_trend": "fund", "fund_streak": "fund",
    "fund_book": "fund", "cashflow": "fund", "efficiency": "fund", "guidance": "fund",
    "turnaround": "fund", "revision": "fund", "fwd": "fund", "progress_seasonal": "fund",
    "valuation": "val", "dividend": "val",
    "supply": "supply", "margin_alert": "supply", "flow": "supply", "holders_lvs": "supply",
    "holders_major": "supply", "holders_cross": "supply",
    "market": "market", "sector_index": "market",
    "liquidity": "other", "scale": "other", "sector": "other", "listing": "other",
    "timing": "other", "earn_ahead": "other",
}
#: financial_inputs（実験121 C の13列）は中身がまちまちなので列ごとに区分を決める
COLUMN_CATEGORY = {
    "tv_beta": "market", "tv_beta_n": "market", "tv_rf": "market", "tv_rf_age": "market",
    "tv_fin_age": "other",
    "tv_earnings_yield": "val", "tv_book_yield": "val", "tv_div_yield": "val", "tv_payout": "val",
    "tv_ri_valid_fraction": "fund", "tv_ddm_valid_fraction": "fund",
    "missing_tv_ri": "fund", "missing_tv_ddm": "fund",
}
BASE = ("p10", "+10%（本番）")


def target_name(horizon: int, multiple: float) -> str:
    span = {250: "1年", 125: "半年"}.get(int(horizon), f"{int(horizon)}営業日")
    return f"{multiple:g}倍（{span}以内）"


def parse_targets(text: str) -> List[Tuple[str, int, float]]:
    """"x3:250:3,x2:125:2" -> [("x3", 250, 3.0), ("x2", 125, 2.0)]。列名は英小文字・数字・_ だけ。"""
    out = []
    for part in [p.strip() for p in str(text).split(",") if p.strip()]:
        bits = part.split(":")
        if len(bits) != 3 or not bits[0].replace("_", "").isalnum() or not bits[0].islower() \
                or bits[0] in (BASE[0], "label"):
            raise SystemExit(f"--targets の形が不正: {part}（例 x3:250:3）")
        horizon, multiple = int(bits[1]), float(bits[2])
        if horizon < 1 or multiple <= 1:
            raise SystemExit(f"--targets の値が不正: {part}（窓は1営業日以上・倍率は1より大きい）")
        out.append((bits[0], horizon, multiple))
    if not out or len({k for k, _, _ in out}) != len(out):
        raise SystemExit(f"--targets が空か列名が重なっている: {text}")
    return out


def labels_for(targets: Sequence[Tuple[str, int, float]]) -> List[Tuple[str, str]]:
    return [BASE] + [(k, target_name(h, m)) for k, h, m in targets]


# --------------------------------------------------------------------------- #
# ラベル
# --------------------------------------------------------------------------- #
def price_columns(bars: pd.DataFrame) -> pd.DataFrame:
    """日足から 銘柄・日・始値・終値（分割調整後を素の値で埋めたもの。build_dataset.price_panel と同じ取り方）を作り、並べる。"""
    b = pd.DataFrame({"Code": bars["Code"].astype(str), "Date": pd.to_datetime(bars["Date"])})
    o = bars["AdjO"] if "AdjO" in bars.columns else pd.Series(np.nan, index=bars.index)
    b["open"] = (o.fillna(bars["O"]) if "O" in bars.columns else o).to_numpy(dtype=float)
    b["close"] = bars["AdjC"].fillna(bars["C"]).to_numpy(dtype=float)
    return b.sort_values(["Code", "Date"]).reset_index(drop=True)


def forward_frame(prices: pd.DataFrame, horizon: int, stop_gap: int = STOP_GAP) -> pd.DataFrame:
    """
    各行 t について、買値（翌営業日の寄り）・t+1〜t+horizon の終値の最大・先が horizon 行あるか・その銘柄の売買が止まったか。

    prices は price_columns の形（銘柄・日の順に並んでいること）。窓は銘柄ごとの行で数える（build_dataset.attach_rise_label と同じ）。
    止まった = その銘柄の最後の日が、全体の日付で最後から stop_gap 営業日目より前。
    """
    g = prices.groupby("Code", sort=False)
    entry = g["open"].shift(-1)

    def future_max(s: pd.Series) -> pd.Series:
        return s[::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)

    fmax = g["close"].transform(future_max)
    have = g.cumcount(ascending=False) >= horizon
    days = np.sort(prices["Date"].unique())
    cutoff = days[-1 - stop_gap] if len(days) > stop_gap else days[0]
    stopped = g["Date"].transform("max") < cutoff
    return pd.DataFrame({"Code": prices["Code"], "Date": prices["Date"], "entry": entry,
                         "fmax": fmax, "have_forward": have, "stopped": stopped})


def multiple_label(fr: pd.DataFrame, multiple: float) -> pd.Series:
    """forward_frame の行から「買値の multiple 倍に届いたか」（1 / 0 / 不明は NaN）。"""
    ratio = fr["fmax"] / fr["entry"]
    known = ((fr["have_forward"] | fr["stopped"]) & fr["entry"].notna() & fr["fmax"].notna()
             & (fr["entry"] > 0))
    hit = LS.at_least(ratio, multiple)
    return hit.astype(float).where(known)


def mismatch_share(a: pd.Series, b: pd.Series, tol: float = MATCH_TOL) -> float:
    """両方に値がある行のうち、相対誤差が tol を超える行の割合。片方だけ欠けている行も合わないと数える。"""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    both = a.notna() & b.notna()
    one = a.notna() ^ b.notna()
    diff = (a - b).abs() > tol * np.maximum(1.0, b.abs())
    n = int((both | one).sum())
    return float(((both & diff) | one).sum() / n) if n else 0.0


# --------------------------------------------------------------------------- #
# 区分と割合
# --------------------------------------------------------------------------- #
def category_map(cols: Sequence[str]) -> Dict[str, str]:
    """列 -> 区分。区分が付かない列があれば止める（新しいグループを足したのに区分を決め忘れるのを防ぐ）。"""
    cg = F.column_groups()
    out, missing = {}, []
    for c in cols:
        cat = COLUMN_CATEGORY.get(c) or GROUP_CATEGORY.get(cg.get(c, ""))
        if cat is None:
            missing.append(f"{c}（{cg.get(c, 'グループなし')}）")
        else:
            out[c] = cat
    if missing:
        raise SystemExit(f"区分の決まっていない列が {len(missing)}本: {missing[:5]}")
    return out


def shares(values: np.ndarray, cols: Sequence[str]) -> Dict[str, float]:
    """列ごとの重要度を合計1の割合にする。合計が0（木が1本も分岐しない）なら全部0。"""
    v = np.asarray(values, dtype=float)
    total = float(v.sum())
    return {c: (float(x) / total if total > 0 else 0.0) for c, x in zip(cols, v)}


def by_key(col_share: Dict[str, float], key: Dict[str, str]) -> Dict[str, float]:
    """列ごとの割合を区分（またはグループ）ごとに足す。"""
    out: Dict[str, float] = {}
    for c, s in col_share.items():
        out[key[c]] = out.get(key[c], 0.0) + s
    return out


def verdict(pairs: Sequence[Tuple[float, float]], name: str = "新しいラベル") -> str:
    """pairs = [(+10% の割合 %, 新しいラベルの割合 %)]。小数1桁に丸めて、すべての組で同じ向きのときだけ決める。"""
    if not pairs:
        return "比べられない（組が無い）"
    lower = all(round(new, 1) < round(base, 1) for base, new in pairs)
    higher = all(round(new, 1) > round(base, 1) for base, new in pairs)
    if lower:
        return f"仮説どおり（{name}のほうがテクニカルの割合が低い）"
    if higher:
        return f"仮説と逆（{name}のほうがテクニカルの割合が高い）"
    return "はっきりしない"


def direction(pairs: Sequence[Tuple[float, float]], name: str) -> str:
    """見るだけの向き（小数1桁に丸めて、すべての組で同じ向きか）。"""
    if pairs and all(round(b, 1) > round(a, 1) for a, b in pairs):
        return f"{name}のほうが高い（すべての組）"
    if pairs and all(round(b, 1) < round(a, 1) for a, b in pairs):
        return f"{name}のほうが低い（すべての組）"
    return "組によって違う"


# --------------------------------------------------------------------------- #
# 学習
# --------------------------------------------------------------------------- #
def production_params(cols: Sequence[str]) -> Dict:
    """本番の373列の木のパラメータ。373列で探索したもの（列の指紋）でなければ止める。"""
    store = tuning.load_params()
    rec = store.get(F.DEFAULT_PRESET) or {}
    if rec.get("_features_sig") != F.signature(list(cols)):
        raise SystemExit(f"パラメータが {len(cols)}列で探索したものではない（指紋 {rec.get('_features_sig')} / "
                         f"{F.signature(list(cols))}）")
    p = {k: v for k, v in tuning.params_for(F.DEFAULT_PRESET).items() if not str(k).startswith("_")}
    p["verbose"] = -1
    return p


def fit(X: np.ndarray, y: np.ndarray, params: Dict, seed: int):
    import lightgbm as lgb

    pos = float((y == 1).sum())
    spw = float((len(y) - pos) / pos) if pos else 1.0
    m = lgb.LGBMClassifier(**{**params, "random_state": int(seed)}, scale_pos_weight=spw)
    m.fit(X, y)
    return m


def gain_split(model) -> Tuple[np.ndarray, np.ndarray]:
    b = model.booster_
    return (np.asarray(b.feature_importance(importance_type="gain"), dtype=float),
            np.asarray(b.feature_importance(importance_type="split"), dtype=float))


def window_rows(df: pd.DataFrame, end: str) -> pd.DataFrame:
    return df if end == "all" else df[df["Date"] <= pd.Timestamp(end)]


def importance_runs(df: pd.DataFrame, cols: Sequence[str], params: Dict, seeds: Sequence[int],
                    ends: Sequence[str], labels: Sequence[Tuple[str, str]], log=print) -> List[Dict]:
    """期間×種×ラベルごとに学習し、列ごとの gain・split の割合を返す。"""
    runs = []
    for end in ends:
        w = window_rows(df, end)
        few = [name for key, name in labels if not MIN_POS <= int(w[key].sum()) <= len(w) - MIN_POS]
        if few:
            log(f"  期間 〜{end}: {len(w):,}行 / 正例か負例が {MIN_POS} 件に満たないラベル（{'・'.join(few)}）があるので飛ばす")
            continue
        X = w[list(cols)].to_numpy(dtype=float)
        for seed in seeds:
            row = {"end": end, "seed": int(seed), "n": int(len(w))}
            for key, _ in labels:
                y = w[key].to_numpy(dtype=int)
                gain, split = gain_split(fit(X, y, params, seed))
                row[key] = {"pos": int(y.sum()), "gain": shares(gain, cols), "split": shares(split, cols)}
            runs.append(row)
        log(f"  期間 〜{end}: {len(w):,}行 / 正例 " + "・".join(f"{name} {int(w[key].sum()):,}" for key, name in labels)
            + f" / 種 {len(seeds)}つ")
    return runs


def purge_days_for(targets: Sequence[Tuple[str, int, float]]) -> int:
    return int(np.ceil(max(h for _, h, _ in targets) * PURGE_PER_DAY)) + PURGE_PAD


def oof_reference(df: pd.DataFrame, cols: Sequence[str], params: Dict, years: Sequence[int],
                  labels: Sequence[Tuple[str, str]], purge_days: int, seed: int = 0) -> Dict:
    """前進分割の out-of-fold（参考）。学習はテストの始まりの purge_days 日前より前の行。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    parts = []
    for y0 in years:
        start = pd.Timestamp(f"{y0}-01-01")
        te = df[(df["Date"] >= start) & (df["Date"] < pd.Timestamp(f"{y0 + 1}-01-01"))]
        tr = df[df["Date"] < start - pd.Timedelta(days=purge_days)]
        if te.empty or tr.empty:
            continue
        part = {"year": y0, "n_train": int(len(tr)), "n_test": int(len(te))}
        Xtr = tr[list(cols)].to_numpy(dtype=float)
        Xte = te[list(cols)].to_numpy(dtype=float)
        for key, _ in labels:
            ytr = tr[key].to_numpy(dtype=int)
            if ytr.sum() == 0 or ytr.sum() == len(ytr):
                part[key] = None
                continue
            m = fit(Xtr, ytr, params, seed)
            part[key] = {"y": te[key].to_numpy(dtype=int).tolist(),
                         "s": m.predict_proba(Xte)[:, 1].tolist()}
        parts.append(part)
    out = {"years": [], "pooled": {}, "purge_days": int(purge_days)}
    for key, _ in labels:
        got = [p[key] for p in parts if p.get(key)]
        if not got:
            continue
        ys = np.concatenate([np.asarray(g["y"]) for g in got])
        ss = np.concatenate([np.asarray(g["s"]) for g in got])
        if 0 < ys.sum() < len(ys):
            ap = float(average_precision_score(ys, ss))
            out["pooled"][key] = {"n": int(len(ys)), "pos": int(ys.sum()), "base": float(ys.mean()),
                                  "pr_auc": ap, "lift": ap / float(ys.mean()),
                                  "roc_auc": float(roc_auc_score(ys, ss))}
    for p in parts:
        row = {"year": p["year"], "n_train": p["n_train"], "n_test": p["n_test"]}
        for key, _ in labels:
            if not p.get(key):
                continue
            y = np.asarray(p[key]["y"])
            s = np.asarray(p[key]["s"])
            if 0 < y.sum() < len(y):
                ap = float(average_precision_score(y, s))
                row[key] = {"pos": int(y.sum()), "base": float(y.mean()), "lift": ap / float(y.mean())}
        out["years"].append(row)
    return out


# --------------------------------------------------------------------------- #
# 集計と表示
# --------------------------------------------------------------------------- #
def summarize(runs: List[Dict], cmap: Dict[str, str], cols: Sequence[str],
              labels: Sequence[Tuple[str, str]]) -> Dict:
    groups = F.column_groups()
    gmap = {c: (groups.get(c) or "?") for c in cols}
    base = labels[0][0]
    rows = []
    for r in runs:
        row = {"end": r["end"], "seed": r["seed"], "n": r["n"]}
        for key, _ in labels:
            row[key] = {kind: by_key(r[key][kind], cmap) for kind in ("gain", "split")}
            row[key]["pos"] = r[key]["pos"]
        rows.append(row)

    def tech(row, key):
        return 100 * row[key]["gain"].get("tech", 0.0)

    def fund_val(row, key):
        return 100 * (row[key]["gain"].get("fund", 0.0) + row[key]["gain"].get("val", 0.0))

    pairs_tech = {key: [(tech(r, base), tech(r, key)) for r in rows] for key, _ in labels[1:]}
    pairs_fv = {key: [(fund_val(r, base), fund_val(r, key)) for r in rows] for key, _ in labels[1:]}

    def mean_over(level: str, key: str, kind: str = "gain") -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for r in runs:
            if level == "category":
                d = by_key(r[key][kind], cmap)
            elif level == "group":
                d = by_key(r[key][kind], gmap)
            else:
                d = r[key][kind]
            for k, v in d.items():
                acc.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in acc.items()}

    return {"rows": rows, "pairs_tech": pairs_tech, "pairs_fund_val": pairs_fv,
            "verdict": {key: verdict(pairs_tech[key], name) for key, name in labels[1:]},
            "fund_val_direction": {key: direction(pairs_fv[key], name) for key, name in labels[1:]},
            "mean": {key: {"category": mean_over("category", key), "category_split": mean_over("category", key, "split"),
                           "group": mean_over("group", key), "column": mean_over("column", key)}
                     for key, _ in labels}}


def report(summary: Dict, oof: Optional[Dict], labels: Sequence[Tuple[str, str]], log=print) -> None:
    m = summary["mean"]
    base = labels[0][0]
    log("\n" + "=" * 110)
    log(f"区分ごとの重要度の割合（gain。期間×種の{len(summary['rows'])}組の平均。括弧は split＝分岐に使われた回数の割合。"
        "差は本番の +10% との差）")
    log("=" * 110)
    log("  " + "区分".ljust(30) + "".join(f"{name:>20}" for _, name in labels)
        + "".join(f"{'差 ' + name:>20}" for _, name in labels[1:]))
    for cat, cname in CATEGORIES:
        cells = []
        for key, _ in labels:
            g = 100 * m[key]["category"].get(cat, 0.0)
            sp = 100 * m[key]["category_split"].get(cat, 0.0)
            cells.append(f"{g:>11.1f}%（{sp:4.1f}%）")
        diffs = [f"{100 * (m[key]['category'].get(cat, 0.0) - m[base]['category'].get(cat, 0.0)):>+18.1f}pt"
                 for key, _ in labels[1:]]
        log("  " + cname.ljust(28) + "".join(cells) + "".join(diffs))

    log("\nテクニカルの割合（gain）: 期間×種の組ごと（正例の件数・テクニカルの割合・本番との差）")
    for r_i, r in enumerate(summary["rows"]):
        pos = " ".join(f"{r[key]['pos']:,}" for key, _ in labels)
        techs = " / ".join(f"{name} {100 * r[key]['gain'].get('tech', 0.0):.1f}%" for key, name in labels)
        diffs = " / ".join(f"{name} {summary['pairs_tech'][key][r_i][1] - summary['pairs_tech'][key][r_i][0]:+.1f}pt"
                           for key, name in labels[1:])
        log(f"  〜{r['end']:<11} 種{r['seed']} {r['n']:>7,}行 正例 {pos} | {techs} | 差 {diffs}")
    for key, name in labels[1:]:
        diffs = [b - a for a, b in summary["pairs_tech"][key]]
        fv = [b - a for a, b in summary["pairs_fund_val"][key]]
        if diffs:
            log(f"  {name}: テクニカルの割合の差 平均 {np.mean(diffs):+.1f}pt（最小 {min(diffs):+.1f}・最大 {max(diffs):+.1f}）"
                f" / 見るだけ: 決算・財務＋バリュエーションの割合の差 平均 {np.mean(fv):+.1f}pt"
                f"（最小 {min(fv):+.1f}・最大 {max(fv):+.1f}）→ {summary['fund_val_direction'][key]}")
    log("\n読み方（回す前に決めたもの）:")
    for key, name in labels[1:]:
        log(f"  {name}: {summary['verdict'][key]}")

    for key, name in labels:
        log(f"\nグループごとの割合（gain・平均）の上位12: {name}")
        top = sorted(m[key]["group"].items(), key=lambda kv: -kv[1])[:12]
        log("  " + " / ".join(f"{g} {100 * v:.1f}%" for g, v in top))
    for key, name in labels:
        log(f"\n列ごとの割合（gain・平均）の上位15: {name}")
        top = sorted(m[key]["column"].items(), key=lambda kv: -kv[1])[:15]
        log("  " + " / ".join(f"{c} {100 * v:.1f}%" for c, v in top))

    if oof:
        log(f"\n参考: 前進分割（学習はテストの始まりの{oof['purge_days']}日前まで）の当たり具合")
        for key, name in labels:
            p = oof["pooled"].get(key)
            if p:
                log(f"  {name}: {p['n']:,}行・正例 {p['pos']:,}（{100 * p['base']:.2f}%）/ PR-AUC {p['pr_auc']:.4f}"
                    f"（正例率の {p['lift']:.2f}倍）/ ROC-AUC {p['roc_auc']:.4f}")
        for row in oof["years"]:
            cells = []
            for key, name in labels:
                c = row.get(key)
                if c:
                    cells.append(f"{name} 正例 {c['pos']:,}・{c['lift']:.2f}倍")
            log(f"  {row['year']}年: 学習 {row['n_train']:,}行・テスト {row['n_test']:,}行 / " + " / ".join(cells))


# --------------------------------------------------------------------------- #
def load_inputs(data_dir: str):
    """データセットと日足。テストでは差し替える。"""
    import build_dataset as B

    ds = pd.read_parquet(lab.DATASET)
    ds["Date"] = pd.to_datetime(ds["Date"])
    ds["Code"] = ds["Code"].astype(str)
    bars = B.load_parts("bars", data_dir)
    return ds, bars


def build_frame(ds: pd.DataFrame, bars: pd.DataFrame, targets: Sequence[Tuple[str, int, float]], cols: Sequence[str],
                log=print) -> Tuple[pd.DataFrame, Dict]:
    """候補の行に新しいラベルを付け、確かめをして、すべてのラベルが分かる行だけを返す。"""
    LS.check_training_frame(ds, "lgbm")
    missing = [c for c in cols if c not in ds.columns]
    if missing:
        raise SystemExit(f"データセットに無い列が {len(missing)}本: {missing[:5]}")
    prices = price_columns(bars)
    dup = int(prices.duplicated(["Code", "Date"]).sum())
    if dup:
        # 本番の日足の読み方（build_dataset.load_parts）は重ねを落とさない。重ねがあれば窓の数え方がずれうるので、落とした
        # うえで、下の確かめ（買値・20営業日の窓が本番と合うか）で本番とずれていないことを見る
        log(f"[warn] 日足に銘柄・日の重ねが {dup:,}行。後ろの行を残す")
        prices = prices.drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)
    keys = ["Code", "Date"]
    df = ds.copy()
    frames: Dict[int, pd.DataFrame] = {}
    for key, horizon, multiple in targets:
        if horizon not in frames:
            frames[horizon] = forward_frame(prices, horizon)
        fr = frames[horizon]
        part = fr[keys].copy()
        part[key] = multiple_label(fr, multiple).to_numpy()
        part[f"{key}_have"] = fr["have_forward"].to_numpy()
        part[f"{key}_stopped"] = fr["stopped"].to_numpy()
        df = df.merge(part, on=keys, how="left", validate="one_to_one")
    fr_20 = forward_frame(prices, LS.for_model("lgbm")["horizon"])
    fr_20["rise20"] = (fr_20["fmax"] / fr_20["entry"] - 1.0).where(fr_20["have_forward"])
    df = df.merge(fr_20[keys + ["entry", "rise20"]], on=keys, how="left", validate="one_to_one")
    del frames, fr_20, prices

    bad_entry = mismatch_share(df["entry"], df["entry_price"])
    bad_rise = mismatch_share(df["rise20"], df["future_rise"])
    log(f"[check] 買値（翌営業日の寄り）がデータセットの entry_price と合わない行 {100 * bad_entry:.3f}%"
        f" / 20営業日の窓の終値の最大がデータセットの future_rise と合わない行 {100 * bad_rise:.3f}%（許容 {100 * MAX_MISMATCH:.1f}%）")
    if bad_entry > MAX_MISMATCH or bad_rise > MAX_MISMATCH:
        raise SystemExit("買値か窓の取り方が本番のラベルと合わない")

    df["p10"] = pd.to_numeric(df["label"], errors="coerce")
    known = df["p10"].notna()
    for key, _, _ in targets:
        known &= df[key].notna()
    info: Dict = {"candidates": int(len(df)), "known": int(known.sum()), "targets": {}}
    for key, horizon, multiple in targets:
        stopped = df[f"{key}_stopped"].fillna(False).astype(bool)
        have = df[f"{key}_have"].fillna(False).astype(bool)
        short = df[key].isna() & ~have & ~stopped
        early = known & stopped & ~have
        info["targets"][key] = {"horizon": int(horizon), "multiple": float(multiple),
                                "known_alone": int(df[key].notna().sum()),
                                "unknown_recent": int(short.sum()),
                                "unknown_other": int((df[key].isna() & ~short).sum()),
                                "stopped_known": int(early.sum()),
                                "stopped_pos": int(df.loc[early, key].sum())}
    out = df.loc[known].copy()
    label_keys = ["p10"] + [k for k, _, _ in targets]
    for key in label_keys:
        out[key] = out[key].astype(int)
    info["first"] = str(out["Date"].min().date()) if len(out) else ""
    info["last"] = str(out["Date"].max().date()) if len(out) else ""
    years = out.groupby(out["Date"].dt.year)[label_keys].agg(["size", "sum"])
    info["years"] = {int(y): {"n": int(r[(label_keys[0], "size")]),
                              **{k: int(r[(k, "sum")]) for k in label_keys}} for y, r in years.iterrows()}
    return out.sort_values(keys, kind="mergesort").reset_index(drop=True), info


def main(argv: Optional[Sequence[str]] = None, log=print) -> int:
    ap = argparse.ArgumentParser(description="実験127: 1年以内に3倍・半年以内に2倍のラベルで重要度の割合を比べる（検証だけ）")
    ap.add_argument("--targets", default=",".join(f"{k}:{h}:{m:g}" for k, h, m in TARGETS),
                    help="新しいラベル（列名:窓の営業日数:倍率 をカンマで）")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--ends", default=",".join(WINDOW_ENDS))
    ap.add_argument("--no-oof", action="store_true", help="参考の前進分割を回さない")
    ap.add_argument("--data-dir", default=lab.DATA_DIR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    t0 = time.time()
    targets = parse_targets(args.targets)
    labels = labels_for(targets)

    cols = F.columns(F.DEFAULT_PRESET)
    cmap = category_map(cols)
    params = production_params(cols)
    log(f"[setup] 列 {F.DEFAULT_PRESET}（{len(cols)}列・指紋 {F.signature(list(cols))}）/ 木 {params.get('n_estimators')}本"
        f" / 学習率 {params.get('learning_rate'):.4g} / 葉 {params.get('num_leaves')}（本番の lgbm_params.json）")
    counts: Dict[str, int] = {}
    for c in cols:
        counts[cmap[c]] = counts.get(cmap[c], 0) + 1
    log("[setup] 区分ごとの列の数: " + " / ".join(f"{name} {counts.get(cat, 0)}" for cat, name in CATEGORIES))
    log("[setup] ラベル: " + " / ".join(name for _, name in labels))

    ds, bars = load_inputs(args.data_dir)
    df, info = build_frame(ds, bars, targets, cols, log=log)
    del bars
    log(f"[label] 候補 {info['candidates']:,}行のうち、すべてのラベルが分かる行 {info['known']:,}（{info['first']}〜{info['last']}）")
    for key, name in labels[1:]:
        t = info["targets"][key]
        log(f"[label] {name}（{t['horizon']}営業日以内に終値が買値の{t['multiple']:g}倍）: 単独で分かる行 {t['known_alone']:,}"
            f"・先が足りない行 {t['unknown_recent']:,}・ほかの理由で分からない行 {t['unknown_other']:,}"
            f" / 使う行のうち、売買が止まって窓の途中で分かった行 {t['stopped_known']:,}（うち正例 {t['stopped_pos']:,}）")
    n = max(len(df), 1)
    log("[label] 正例（使う行）: " + " / ".join(f"{name} {int(df[key].sum()):,}（{100 * df[key].sum() / n:.2f}%）"
                                             for key, name in labels))
    for key, name in labels[1:]:
        both = int((df[key].astype(bool) & df["p10"].astype(bool)).sum())
        log(f"[label] {name} の正例のうち +10% も正例 {both:,}")
    log("[label] 年ごと（行・正例 " + "・".join(name for _, name in labels) + "）: "
        + " / ".join(f"{y}: {v['n']:,}・" + "・".join(f"{v[key]:,}" for key, _ in labels) for y, v in info["years"].items()))

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ends = [e.strip() for e in args.ends.split(",") if e.strip()]
    log(f"\n[fit] 期間 {len(ends)}つ × 種 {len(seeds)}つ × ラベル{len(labels)}つ")
    runs = importance_runs(df, cols, params, seeds, ends, labels, log=log)
    summary = summarize(runs, cmap, cols, labels)
    oof = None if args.no_oof else oof_reference(df, cols, params, OOF_YEARS, labels, purge_days_for(targets))
    report(summary, oof, labels, log=log)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"info": info, "labels": labels, "targets": targets, "verdict": summary["verdict"],
                   "pairs_tech": summary["pairs_tech"], "pairs_fund_val": summary["pairs_fund_val"],
                   "fund_val_direction": summary["fund_val_direction"], "mean": summary["mean"],
                   "oof": (None if oof is None else {"pooled": oof["pooled"], "years": oof["years"],
                                                     "purge_days": oof["purge_days"]}),
                   "seeds": seeds, "ends": ends}, fh, ensure_ascii=False, indent=1)
    log(f"\n[done] {args.out} / {(time.time() - t0) / 60:.1f}分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
