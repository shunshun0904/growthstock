#!/usr/bin/env python3
"""
母集団とラベルの設計を1因子ずつ動かして、設計ごとの実力を実測する。

なぜこのスクリプトが要るか
--------------------------
特徴量を足す実験は打ち止めになった。地合い11列も業種指数4列も、
向きの揃った改善が出なかった。一方で母集団を 12,484 -> 22,909 に広げた
変更は対無情報 +0.0274 -> +0.0416 と、唯一はっきり効いた。
残っている大きな設計変数は「誰を母集団に入れるか」と「何を正例と呼ぶか」で、
それを推測ではなく同一データ上の実測で並べる。

ROC-AUC を目標にするときの落とし穴
----------------------------------
ROC-AUC も PR-AUC も *ラベルの関数* なので、ラベル定義の違う設計どうしを
これらの数字だけで比べても「モデルが良くなった」ことにはならない。
正例の条件を厳しくすれば正例は減って残った正例は極端になり、
たいてい ROC-AUC は上がる。だが取れる銘柄が減っただけかもしれない。

そこで設計ごとに、ラベル定義に依存しない物差しを必ず並べる:

    上位5%に選ばれた銘柄が、固定の参照ホライズン（60営業日）で
    実際にどれだけ上がったか（終盤水準の中央値・勝率）。

参照ホライズンは全設計で同じ RiseConfig から計算するので、ラベルを
どういじっても意味が変わらない。ROC-AUC が上がっても
この値が動かないなら、上がったのは指標であって実力ではない。

測り方
------
・特徴量は既存のデータセット（research/_data/dataset.parquet）から取る。
  ラベルだけを差し替える設計なら、作り直さずに (Code, Date) で結合できる。
・母集団を *狭める* 設計（クールダウンを伸ばす、流動性の下限を上げる）は
  同じデータセットの行の部分集合なので、これも作り直し不要。
・母集団が *増える* 設計（52週高値）はデータセットごと別に要る。
  SWEEP_HIGH_WINDOW=245 python3 research/build_dataset.py \
      --out research/_data/dataset_w245.parquet
  で作っておく。無ければその設計は飛ばす（黙って落とさず、理由を出す）。
・ハイパーパラメータは全設計で共通（既定は探索済みの all）。
  設計ごとに探索し直すと「設計の差」と「探索の差」が混ざる。
  ここで測るのは設計の差だけ。採用案は後で正規に探索し直す。
・分割は全設計で同じ日付境界を使う（下の split_bounds を参照）。
  設計ごとに末尾から取り直すと、評価期間そのものがずれて局面差が混ざる。

出力
----
research/_data/sweep_design.json と docs/MODEL_DESIGN_SWEEP.md。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

#: 高値窓ごとのデータセット。既定の 368 以外は掃引用に別途ビルドする。
DATASETS = {
    368: os.path.join(DATA_DIR, "dataset.parquet"),
    245: os.path.join(DATA_DIR, "dataset_w245.parquet"),
}

#: 実収益の物差しに使う参照ホライズン。全設計で固定する。
#: ラベルの horizon を動かしてもこちらは動かさない。動かしたら比較にならない。
REF_HORIZON = 60
REF_RISE = B.RiseConfig(horizon=REF_HORIZON, threshold=0.20, keep_days=0,
                        end_ratio=None, require_uptrend=False)

#: ラベル計算に要る列だけ。パネルは1,000万行規模あるので丸ごと copy しない。
PANEL_COLS = ["Code", "Date", "close", "is_new_high", "high52w", "tv_ma20"]


# --------------------------------------------------------------------------- #
# 設計の記述
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Design:
    """1つの設計。基準からどの1因子を動かしたかを axis に書く。"""
    key: str
    axis: str               # 動かした軸（日本語）。基準は "基準"
    label: str              # 表に出す短い説明
    high_window: int = 368
    cooldown: int = 20
    min_trading_value: Optional[float] = 0.1
    horizon: int = 60
    threshold: float = 0.20
    keep_days: int = 10
    end_ratio: Optional[float] = 0.10
    require_uptrend: bool = True
    #: 到達しきい値を銘柄自身のボラティリティで測る。None なら固定%（従来）。
    #:
    #: 「先60営業日で +20%」は銘柄のボラティリティで正規化されていない。
    #: 日次ボラ3%の銘柄の60日σは 3%×√60 ≒ 23% で、+20% は 0.85σ。
    #: 日次ボラ1%なら 7.7% で、同じ +20% が 2.6σ になる。
    #: つまり正例になりやすさが定義の時点で銘柄ごとに何倍も違う。
    #: 実測でも日付内で最も分離するのは vol_20d（AUC 0.6325）で、
    #: モデルの日付内AUC 0.6288 を単独で上回っている。
    #: 「上がる銘柄を当てている」のか「荒い銘柄を選んでいるだけ」なのかを
    #: 切り分けるために、しきい値を k×σ にした定義を並べる。
    vol_norm_k: Optional[float] = None

    @property
    def rise(self) -> B.RiseConfig:
        return B.RiseConfig(horizon=self.horizon, threshold=self.threshold,
                            keep_days=self.keep_days, end_ratio=self.end_ratio,
                            require_uptrend=self.require_uptrend)

    @property
    def forward_needed(self) -> int:
        """ラベル確定に必要な将来営業日数。エンバーゴの幅もこれで決まる。"""
        return self.horizon


BASE = Design(key="base", axis="基準", label="現行（78週 / +20% / 60日 / 維持10日 / 終盤+10% / トレンド）")


def _var(key: str, axis: str, label: str, **kw) -> Design:
    """基準から1因子だけ動かした設計を作る。"""
    d = {f.name: getattr(BASE, f.name) for f in BASE.__dataclass_fields__.values()
         if f.name not in ("key", "axis", "label")}
    d.update(kw)
    return Design(key=key, axis=axis, label=label, **d)


#: 掃引する設計。基準から1因子ずつ動かす。
#: 組み合わせは1因子の結果を見てから決める（先に全組み合わせを回すと
#: どの因子が効いたのか分からなくなるうえ、当たりを引くまで探す形になる）。
DESIGNS: List[Design] = [
    BASE,
    # --- 到達しきい値 --- #
    _var("th15", "到達しきい値", "+15%", threshold=0.15),
    _var("th25", "到達しきい値", "+25%", threshold=0.25),
    _var("th30", "到達しきい値", "+30%", threshold=0.30),
    # --- ホライズン --- #
    _var("hz40", "ホライズン", "40営業日", horizon=40),
    _var("hz90", "ホライズン", "90営業日", horizon=90),
    _var("hz120", "ホライズン", "120営業日", horizon=120),
    # --- 維持日数 --- #
    _var("keep0", "維持日数", "条件なし", keep_days=0),
    _var("keep20", "維持日数", "20日", keep_days=20),
    # --- 終盤の水準 --- #
    _var("end_off", "終盤の水準", "条件なし", end_ratio=None),
    _var("end15", "終盤の水準", "+15%", end_ratio=0.15),
    _var("end20", "終盤の水準", "+20%", end_ratio=0.20),
    # --- トレンド条件 --- #
    _var("trend_off", "トレンド条件", "条件なし", require_uptrend=False),
    # --- クールダウン（母集団を狭める） --- #
    _var("cd40", "クールダウン", "40営業日", cooldown=40),
    _var("cd60", "クールダウン", "60営業日", cooldown=60),
    # --- 流動性の下限（母集団を狭める） --- #
    _var("tv03", "流動性の下限", "0.3億円", min_trading_value=0.3),
    _var("tv10", "流動性の下限", "1.0億円", min_trading_value=1.0),
    _var("tv30", "流動性の下限", "3.0億円", min_trading_value=3.0),
    # --- 高値窓（母集団が増える。別データセットが要る） --- #
    _var("w245", "高値窓", "52週(245日)", high_window=245),
    # --- 到達しきい値をボラティリティで正規化する --- #
    # 比較相手は keep0（維持条件なし・固定+20%）。ボラ正規化のほうは
    # しきい値が行ごとに動くので維持日数を付けられず、形を揃えるため。
    #
    # 実測の60日σ: 中央15.0% / p5 6.0% / p95 47.8%（8倍の開き）。
    # 固定+20%は p5 の銘柄には 3.3σ、p95 の銘柄には 0.42σ にあたる。
    # 正例になりやすさが定義の時点で銘柄ごとに何倍も違う。
    # k は正例率が keep0 に近くなるあたりを挟んで振る。
    _var("volk10", "ボラ正規化", "k=1.0σ", vol_norm_k=1.0, keep_days=0),
    _var("volk12", "ボラ正規化", "k=1.2σ", vol_norm_k=1.2, keep_days=0),
    _var("volk14", "ボラ正規化", "k=1.4σ", vol_norm_k=1.4, keep_days=0),
    _var("volk16", "ボラ正規化", "k=1.6σ", vol_norm_k=1.6, keep_days=0),
]


# --------------------------------------------------------------------------- #
# 分割の境界
# --------------------------------------------------------------------------- #

def split_bounds(dates) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    全設計で共通の評価期間を返す (test_start, dmax)。

    基準の設計の末尾から機械的に決める。設計ごとに末尾から取り直すと、
    ホライズンを伸ばした設計はラベルが確定する末尾が手前にずれるので、
    評価期間そのものが変わってしまう。局面が変われば数字は動くので、
    それでは設計の差を見たことにならない。期間を固定して、
    ホライズンを伸ばした代償は「その期間に残る件数が減る」形で払わせる。
    """
    _, test_start, dmax = T.holdout_bounds(dates)
    return test_start, dmax


def train_end_for(design: Design, test_start: pd.Timestamp) -> pd.Timestamp:
    """
    設計ごとの訓練終端。エンバーゴはその設計のホライズンぶん空ける。

    ホライズン120日の設計を60日ぶんのエンバーゴで切ると、訓練末尾の
    サンプルのラベルが評価期間の値動きで決まる。リークなので、
    エンバーゴは必ず設計のホライズンに合わせる。訓練が短くなるのは
    「先を長く見る」設計の正当な代償。
    """
    days = int(round(design.forward_needed * T.TRADING_TO_CALENDAR))
    return test_start - pd.Timedelta(days=days)


# --------------------------------------------------------------------------- #
# パネルとラベル
# --------------------------------------------------------------------------- #

class Panels:
    """高値窓ごとのパネルを1度だけ作って使い回す。"""

    def __init__(self, bars: pd.DataFrame):
        self._bars = bars
        self._cache: Dict[int, pd.DataFrame] = {}

    def get(self, high_window: int) -> pd.DataFrame:
        if high_window not in self._cache:
            print(f"[panel] 高値窓 {high_window}営業日"
                  f"（{round(high_window/245*52)}週）でパネルを作成")
            df = B.price_panel(self._bars, B.LabelConfig(high_window=high_window))
            df = B.mark_new_highs(df, cooldown=B.BREAKOUT_COOLDOWN, on_high=True)
            self._cache[high_window] = df[PANEL_COLS].copy()
        return self._cache[high_window]


def fresh_break(panel: pd.DataFrame, cooldown: int) -> pd.Series:
    """
    クールダウンを変えたときの「新規のブレイク」フラグ。

    build_dataset.mark_new_highs と同じ式。クールダウンを伸ばすと
    fresh(20) の部分集合になる（直前20日に更新が無いことは
    直前40日に更新が無いことより緩い）。
    """
    g = panel.groupby("Code", sort=False)["is_new_high"]
    recent = g.transform(lambda s: s.shift(1).rolling(cooldown, min_periods=1).max())
    return panel["is_new_high"] & (recent.fillna(0) == 0)


def labels_for(panel: pd.DataFrame, cfg: B.RiseConfig) -> pd.DataFrame:
    """パネル全体にラベルを付け直し、結合に要る列だけ返す。"""
    df = B.attach_rise_label(panel.copy(), cfg)
    return df[["Code", "Date", "label", "future_rise", "end_level"]]


def reference_outcome(panel: pd.DataFrame) -> pd.DataFrame:
    """
    ラベル非依存の物差し。全設計で同じ参照ホライズンから計算する。

    ref_rise … 先60営業日の終値最大 / 基準日終値 - 1（一番おいしいところ）
    ref_end  … 先60営業日後の5日平均 / 基準日終値 - 1（持ち切ったときの水準）

    運用で手に入るのは ref_end に近い。ref_rise は「うまく降りられたら」の上限。
    """
    df = B.attach_rise_label(panel.copy(), REF_RISE)
    out = df[["Code", "Date", "future_rise", "end_level", "uptrend_end"]].copy()
    return out.rename(columns={"future_rise": "ref_rise", "end_level": "ref_end",
                               "uptrend_end": "ref_uptrend"})


# --------------------------------------------------------------------------- #
# 1設計ぶんの評価
# --------------------------------------------------------------------------- #

def outcome_stats(part: pd.DataFrame) -> Dict:
    """実収益の物差し。ラベルには一切触れない。"""
    e = part["ref_end"].dropna()
    r = part["ref_rise"].dropna()
    if len(e) == 0:
        return {}
    return {
        "n": int(len(e)),
        "end_median": round(float(e.median()) * 100, 2),
        "end_mean": round(float(e.mean()) * 100, 2),
        "win_rate": round(float((e > 0).mean()), 4),
        "rise_median": round(float(r.median()) * 100, 2) if len(r) else None,
    }


def run_design(design: Design, frame: pd.DataFrame, cols: List[str],
               params: Dict, test_start: pd.Timestamp, dmax: pd.Timestamp,
               ) -> Dict:
    """1設計を学習して、分離力と実収益の物差しを返す。"""
    import lightgbm as lgb

    d = pd.to_datetime(frame["Date"])
    tr_end = train_end_for(design, test_start)
    train = frame[(d <= tr_end) & frame["label"].notna()]
    test = frame[(d >= test_start) & (d <= dmax) & frame["label"].notna()]

    row: Dict = {
        "key": design.key, "axis": design.axis, "label": design.label,
        "n_all": int(frame["label"].notna().sum()),
        "positive_rate_all": round(float(frame["label"].mean()), 4),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "train_end": str(tr_end.date()),
    }
    if len(test) < 300 or test["label"].sum() < 30 or len(train) < 1000:
        row["skipped"] = (f"評価に足りない（訓練{len(train):,} / "
                          f"評価{len(test):,} / 正例{int(test['label'].sum())}）")
        return row

    ytr = train["label"].to_numpy(dtype=int)
    yte = test["label"].to_numpy(dtype=int)
    gbm = lgb.LGBMClassifier(**params, scale_pos_weight=tuning.scale_pos_weight(ytr))
    gbm.fit(train[cols].to_numpy(dtype=float), ytr)
    score = gbm.predict_proba(test[cols].to_numpy(dtype=float))[:, 1]

    ev = T.evaluate(design.key, yte, score, dates=test["Date"])
    row.update({
        "base_rate": round(ev["base_rate"], 4),
        "pr_auc": round(ev["pr_auc"], 4),
        "roc_auc": round(ev["roc_auc"], 4),
        "within_date_auc": round(ev["within_date_auc"], 4),
        "lift@5%": round(ev["lift@5%"], 3),
        "pr_gain": round(ev["pr_auc"] - ev["base_rate"], 4),
    })

    # --- ラベル非依存の物差し --- #
    # スコア上位5%と、評価期間の全件。差が「選んだことの価値」。
    t = test.copy()
    t["score"] = score
    k = max(1, int(len(t) * 0.05))
    top = t.nlargest(k, "score")
    row["outcome_top5"] = outcome_stats(top)
    row["outcome_all"] = outcome_stats(t)
    a, b = row["outcome_all"], row["outcome_top5"]
    if a and b:
        row["edge_end_median"] = round(b["end_median"] - a["end_median"], 2)
        row["edge_win_rate"] = round(b["win_rate"] - a["win_rate"], 4)
        lo, hi = edge_ci(t)
        row["edge_end_ci"] = [lo, hi]
        row["edge_significant"] = bool(lo > 0)
    # 上位5%が「どんな銘柄」なのか。数字が良くなったときに
    # 何を選ぶようになったのかが分からないと、結果を信じる根拠がない
    row["top5_profile"] = profile(top, t)
    return row


#: 上位5%の素性を見る列。無い列は飛ばす。
PROFILE_COLS = ("vol_20d", "log_market_cap", "ret_20d", "tv_ma20_log", "r_high")


def profile(top: pd.DataFrame, allrows: pd.DataFrame) -> Dict:
    """上位5%と全件の中央値を並べる。何を選ぶようになったかを見るため。"""
    out: Dict = {}
    for c in PROFILE_COLS:
        if c not in top.columns:
            continue
        a, b = allrows[c].dropna(), top[c].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        out[c] = {"top5": round(float(b.median()), 3),
                  "all": round(float(a.median()), 3)}
    return out


def edge_ci(t: pd.DataFrame, k_pct: float = 5.0, n_boot: int = 1000,
            seed: int = 0) -> Tuple[float, float]:
    """
    「上位k% - 全件」の終盤リターン中央値の差に、95%区間を付ける。

    上位5%は200件ほどしかない。中央値の差が +0.5pt 出ても、それが
    誤差なのかは目視では分からない。区間が0をまたぐなら
    「選んだ意味があった」とは言えない。

    リサンプルは日付単位（ブロックブートストラップ）にする。
    同じ日の銘柄は地合いを共有していて独立ではないので、行単位で
    resample すると実際より狭い区間が出る（有意でないものが有意に見える）。
    """
    rng = np.random.default_rng(seed)
    end = t["ref_end"].to_numpy(dtype=float)
    score = t["score"].to_numpy(dtype=float)
    groups = [np.flatnonzero(t["Date"].to_numpy() == d)
              for d in pd.unique(t["Date"])]
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        e, sc = end[idx], score[idx]
        ok = np.isfinite(e)
        e, sc = e[ok], sc[ok]
        if len(e) < 40:
            continue
        n = max(1, int(len(e) * k_pct / 100))
        top = e[np.argsort(sc, kind="stable")[-n:]]
        diffs.append(float(np.median(top) - np.median(e)) * 100)
    if not diffs:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (round(float(lo), 2), round(float(hi), 2))


# --------------------------------------------------------------------------- #
# 組み立て
# --------------------------------------------------------------------------- #

#: モデルを使わない対照。列名と「大きいほうを上位にするか」の組。
#:
#: 掃引でどれだけ数字が良くなっても、この対照に勝てないなら
#: モデルは要らない（その1列で並べれば済む）。ラベルを見ないので
#: 全設計に共通の1組として出す。
NAIVE_RULES = (
    ("低ボラ順（-vol_20d）", "vol_20d", False),
    ("モメンタム順（ret_20d）", "ret_20d", True),
    ("大型順（log_market_cap）", "log_market_cap", True),
    ("高ボラ順（vol_20d）", "vol_20d", True),
)


def naive_baselines(frame: pd.DataFrame, test_start: pd.Timestamp,
                    dmax: pd.Timestamp) -> List[Dict]:
    """
    1列で並べただけの対照。同じ評価期間・同じ物差しで測る。

    「上位5%の実収益が全件よりどれだけ良いか」はラベルを見ないので、
    ラベル定義に関係なく計算できる。モデルの数字はこれと比べて読む。
    """
    d = pd.to_datetime(frame["Date"])
    t = frame[(d >= test_start) & (d <= dmax)].copy()
    out = []
    for name, col, desc in NAIVE_RULES:
        if col not in t.columns:
            continue
        u = t[t[col].notna()].copy()
        u["score"] = u[col] if desc else -u[col]
        k = max(1, int(len(u) * 0.05))
        top = u.nlargest(k, "score")
        a, b = outcome_stats(u), outcome_stats(top)
        if not a or not b:
            continue
        lo, hi = edge_ci(u)
        out.append({"name": name, "n": int(len(u)),
                    "top5_end_median": b["end_median"],
                    "all_end_median": a["end_median"],
                    "top5_win_rate": b["win_rate"],
                    "all_win_rate": a["win_rate"],
                    "edge_end_median": round(b["end_median"] - a["end_median"], 2),
                    "edge_end_ci": [lo, hi],
                    "edge_significant": bool(lo > 0),
                    "edge_significant_negative": bool(hi < 0)})
    return out


def build_frame(design: Design, dataset: pd.DataFrame, panel: pd.DataFrame,
                ref: pd.DataFrame, cols: List[str]) -> Optional[pd.DataFrame]:
    """
    設計に対応する「特徴量 + ラベル + 物差し」を作る。

    特徴量はデータセットから取る。行は
      (a) その高値窓のデータセットに入っている
      (b) その設計のクールダウンで新規ブレイク
      (c) その設計の流動性下限を満たす
    を全部満たすもの。ラベルはパネルから付け直す。
    """
    # merge は左の行順を保つが索引は振り直す。データセットの行に戻れるよう
    # 位置を明示的に持ち歩く（索引に頼ると、右側に重複キーが1つあるだけで
    # 黙って行が増え、特徴量とラベルがずれる）
    keys = dataset[["Code", "Date"]].copy()
    keys["_row"] = np.arange(len(dataset), dtype=np.int64)

    sel = panel[["Code", "Date", "tv_ma20"]].copy()
    if design.cooldown != B.BREAKOUT_COOLDOWN:
        sel["fresh"] = fresh_break(panel, design.cooldown).to_numpy()
    else:
        # データセット自体が既定のクールダウンで作られている
        sel["fresh"] = True

    if design.vol_norm_k is None:
        lab = labels_for(panel, design.rise)[["Code", "Date", "label"]]
    else:
        # 参照ホライズンの値から作るので、パネルの付け直しは要らない
        lab = ref[["Code", "Date"]].assign(label=np.nan)

    m = keys
    for right in (sel, lab, ref):
        n_before = len(m)
        m = m.merge(right, on=["Code", "Date"], how="left")
        if len(m) != n_before:
            raise SystemExit(
                f"[{design.key}] 結合で行数が {n_before:,} -> {len(m):,} に変わった。"
                "パネル側に (Code, Date) の重複がある")

    keep = m["fresh"].fillna(False).astype(bool).to_numpy()
    if design.min_trading_value is not None:
        keep &= (m["tv_ma20"] >= design.min_trading_value).to_numpy()
    m = m[keep]

    rows = m["_row"].to_numpy()
    out = dataset.iloc[rows][cols].copy()
    out["Date"] = dataset["Date"].to_numpy()[rows]
    out["Code"] = dataset["Code"].to_numpy()[rows]
    for c in ("label", "ref_rise", "ref_end", "ref_uptrend"):
        out[c] = m[c].to_numpy()
    # True/False/NaN の object 列を数値にそろえる（NaN = 判定不能を残す）
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    if design.vol_norm_k is not None:
        out["label"] = vol_normalised_label(out, design)
    return out.reset_index(drop=True)


def vol_normalised_label(frame: pd.DataFrame, design: Design) -> pd.Series:
    """
    到達しきい値を銘柄自身の σ で測ったラベル。

        σ      = vol_20d/100 × √horizon      （日次ボラから期間ボラへ）
        到達   = ref_rise >= k×σ
        終盤   = ref_end  >= (end_ratio/threshold) × k×σ
        トレンド = 参照ホライズン終了時点で MA20 >= MA60

    維持日数は付けない。しきい値が行ごとに動くとこの条件だけ別実装になり、
    「正規化したから変わった」のか「維持条件の実装が変わったから変わった」のか
    分からなくなる。比較相手は同じ形の keep0（維持条件なし）にする。
    """
    if design.keep_days:
        raise SystemExit(f"[{design.key}] ボラ正規化と維持日数は同時に使えない")
    if design.horizon != REF_HORIZON:
        raise SystemExit(f"[{design.key}] ボラ正規化は参照ホライズン"
                         f"（{REF_HORIZON}営業日）でのみ定義している")
    if "vol_20d" not in frame.columns:
        raise SystemExit(f"[{design.key}] vol_20d が特徴量に無い")

    sigma = frame["vol_20d"] / 100.0 * np.sqrt(design.horizon)
    hit = frame["ref_rise"] >= design.vol_norm_k * sigma
    ok = hit.copy()
    determined = frame["ref_rise"].notna() & sigma.notna()
    if design.end_ratio is not None:
        frac = design.end_ratio / design.threshold
        ok &= frame["ref_end"] >= frac * design.vol_norm_k * sigma
        determined &= frame["ref_end"].notna()
    if design.require_uptrend:
        ok &= frame["ref_uptrend"] == 1.0
        determined &= frame["ref_uptrend"].notna()
    return ok.where(determined).astype("float64")


def load_dataset(high_window: int) -> Optional[pd.DataFrame]:
    path = DATASETS.get(high_window)
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="母集団・ラベル設計を掃引する")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--preset", default="all", help="特徴量セット")
    ap.add_argument("--params", default="all", help="使うハイパーパラメータの鍵")
    ap.add_argument("--only", default="", help="key をカンマ区切りで指定（絞り込み用）")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "sweep_design.json"))
    ap.add_argument("--doc", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "MODEL_DESIGN_SWEEP.md"))
    args = ap.parse_args(argv)

    designs = DESIGNS
    if args.only:
        want = {k.strip() for k in args.only.split(",") if k.strip()}
        designs = [d for d in DESIGNS if d.key in want]
        if not designs:
            raise SystemExit(f"--only に一致する設計がありません: {sorted(want)}")

    cols = F.columns(args.preset)
    params = tuning.params_for(args.params)
    print(f"[setup] 特徴量 {args.preset}（{len(cols)}列） / "
          f"パラメータ {args.params}（木{params['n_estimators']}本 "
          f"/ lr {params['learning_rate']:.4f} / 葉 {params['num_leaves']}）")
    print("[setup] ハイパーパラメータは全設計で共通。"
          "設計ごとに探索し直すと設計の差と探索の差が混ざる")

    paths = sorted(glob.glob(os.path.join(args.data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    print(f"[load] 日次バー {len(bars):,}行 / {bars['Code'].nunique():,}銘柄")
    panels = Panels(bars)

    datasets: Dict[int, pd.DataFrame] = {}
    refs: Dict[int, pd.DataFrame] = {}
    results: List[Dict] = []
    naive: Optional[List[Dict]] = None
    test_start = dmax = None

    for design in designs:
        hw = design.high_window
        if hw not in datasets:
            ds = load_dataset(hw)
            if ds is None:
                print(f"\n[skip] 高値窓 {hw} のデータセットが無い"
                      f"（{DATASETS.get(hw)}）。"
                      f"SWEEP_HIGH_WINDOW={hw} で build_dataset.py を回すと作れる")
                datasets[hw] = None
            else:
                print(f"\n[load] 高値窓 {hw} のデータセット {len(ds):,}行 "
                      f"/ 正例率 {ds['label'].mean()*100:.2f}%")
                datasets[hw] = ds
        ds = datasets[hw]
        if ds is None:
            results.append({"key": design.key, "axis": design.axis,
                            "label": design.label,
                            "skipped": f"高値窓 {hw} のデータセットが無い"})
            continue

        if hw not in refs:
            print(f"[ref] 高値窓 {hw} の参照ホライズン"
                  f"（{REF_HORIZON}営業日）を計算")
            refs[hw] = reference_outcome(panels.get(hw))

        if test_start is None:
            # 基準の設計のデータセットで期間を決め、全設計で使い回す
            test_start, dmax = split_bounds(ds["Date"])
            print(f"[split] 評価期間は全設計で共通: "
                  f"{test_start.date()} 〜 {dmax.date()}")

        frame = build_frame(design, ds, panels.get(hw), refs[hw], cols)
        if naive is None and hw == 368:
            naive = naive_baselines(frame, test_start, dmax)
            print("\n[対照] モデルを使わず1列で並べたときの"
                  "「上位5% - 全件」（同じ評価期間・同じ物差し）")
            for r in naive:
                # 負に有意なものを「有意でない」と書くと逆に読める。
                # 実際「モメンタム順 -17.90pt [-21.61,-13.67]」が
                # 「有意でない」と出ていた
                sig = ("有意に正" if r["edge_significant"]
                       else "有意に負" if r["edge_significant_negative"]
                       else "有意でない")
                print(f"    {r['name']:<26} {r['edge_end_median']:+6.2f}pt "
                      f"[{r['edge_end_ci'][0]:+.2f},{r['edge_end_ci'][1]:+.2f}] "
                      f"{sig} / 勝率 {r['top5_win_rate']*100:.1f}% "
                      f"(全件 {r['all_win_rate']*100:.1f}%)")
        print(f"\n=== [{design.axis}] {design.label} ===")
        row = run_design(design, frame, cols, params, test_start, dmax)
        results.append(row)
        if "skipped" in row:
            print(f"    飛ばした: {row['skipped']}")
            continue
        print(f"    母集団 {row['n_all']:,}（正例率 {row['positive_rate_all']*100:.2f}%） "
              f"/ 訓練 {row['n_train']:,} / 評価 {row['n_test']:,}")
        print(f"    PR-AUC {row['pr_auc']:.4f}（対無情報 +{row['pr_gain']:.4f}） "
              f"/ ROC-AUC {row['roc_auc']:.4f} "
              f"/ 日付内 {row['within_date_auc']:.4f} "
              f"/ Lift@5% {row['lift@5%']:.2f}x")
        b, a = row.get("outcome_top5", {}), row.get("outcome_all", {})
        if b and a:
            print(f"    実収益（60日後・5日平均）上位5%: 中央 {b['end_median']:+.2f}% "
                  f"勝率 {b['win_rate']*100:.1f}%  / "
                  f"全件: 中央 {a['end_median']:+.2f}% 勝率 {a['win_rate']*100:.1f}%")

    _print_table(results)
    payload = {
        "naive_baselines": naive or [],
        "reference_horizon": REF_HORIZON,
        "preset": args.preset, "params_key": args.params,
        "test_start": str(test_start.date()) if test_start is not None else None,
        "test_end": str(dmax.date()) if dmax is not None else None,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out}")
    _write_doc(args.doc, payload)
    print(f"[done] {args.doc}")
    return 0


def _rows_ok(results: List[Dict]) -> List[Dict]:
    return [r for r in results if "pr_auc" in r]


def _ci_text(r: Dict) -> str:
    ci = r.get("edge_end_ci")
    if not ci or any(v is None or not np.isfinite(v) for v in ci):
        return ""
    return f"[{ci[0]:+.2f},{ci[1]:+.2f}]"


def _print_table(results: List[Dict]) -> None:
    ok = _rows_ok(results)
    if not ok:
        print("\n結果なし")
        return
    base = next((r for r in ok if r["key"] == "base"), None)
    print("\n" + "=" * 132)
    print("設計ごとの比較（ハイパーパラメータは全設計で共通。評価期間も共通）")
    print("-" * 132)
    print(f"{'軸':<14}{'設計':<18}{'母集団':>8}{'正例率':>8}{'評価n':>7}"
          f"{'PR-AUC':>9}{'対無情報':>9}{'ROC-AUC':>9}{'日付内':>8}"
          f"{'Lift@5%':>9}{'上位5%終盤':>11}{'全件終盤':>10}{'差':>7}{'95%区間':>17}")
    print("-" * 132)
    for r in ok:
        b, a = r.get("outcome_top5", {}), r.get("outcome_all", {})
        mark = " *" if base and r["key"] != "base" and (
            r["roc_auc"] > base["roc_auc"] and r["pr_gain"] > base["pr_gain"]) else ""
        print(f"{r['axis']:<14}{r['label'][:17]:<18}{r['n_all']:>8,}"
              f"{r['positive_rate_all']*100:>7.2f}%{r['n_test']:>7,}"
              f"{r['pr_auc']:>9.4f}{r['pr_gain']:>+9.4f}{r['roc_auc']:>9.4f}"
              f"{r['within_date_auc']:>8.4f}{r['lift@5%']:>8.2f}x"
              f"{b.get('end_median', float('nan')):>+10.2f}%"
              f"{a.get('end_median', float('nan')):>+9.2f}%"
              f"{r.get('edge_end_median', float('nan')):>+6.2f}"
              f"{_ci_text(r):>17}{mark}")
    print("-" * 132)
    print("終盤 = 参照ホライズン60営業日後の5日平均終値が基準日終値から何%か。")
    print("      ラベル定義に依存しないので、ラベルを変えた設計どうしでも比べられる。")
    print("      「差」= 上位5% - 全件。選んだことの価値。ここが動かないなら")
    print("      ROC-AUC が上がっても上がったのは指標であって実力ではない。")
    print("      95%区間は日付単位のブロックブートストラップ（B=1000）。")
    print("      0 をまたぐなら「選んだ意味があった」とは言えない。")
    print("      * = 基準より ROC-AUC と PR-AUC(対無情報) の両方が上")


def _write_doc(path: str, payload: Dict) -> None:
    ok = _rows_ok(payload["results"])
    base = next((r for r in ok if r["key"] == "base"), None)
    L = [
        "# 母集団・ラベル設計の掃引",
        "",
        "`research/sweep_design.py` が生成する。手で書き換えない。",
        "",
        "基準の設計から1因子ずつ動かして、分離力（PR-AUC / ROC-AUC / 日付内AUC）と、",
        "ラベル定義に依存しない実収益の物差しを並べたもの。",
        "",
        "## 読み方",
        "",
        "PR-AUC も ROC-AUC も **ラベルの関数** なので、ラベル定義の違う設計どうしを",
        "この数字だけで比べても「モデルが良くなった」ことにはならない。",
        "正例の条件を厳しくすれば正例は減って残った正例は極端になり、",
        "たいてい ROC-AUC は上がる。だが取れる銘柄が減っただけかもしれない。",
        "",
        f"そこで参照ホライズン（{payload['reference_horizon']}営業日、全設計で固定）で",
        "実際に何%上がったかを並べる。**上位5% - 全件** の差が「選んだことの価値」で、",
        "ここが動かないなら ROC-AUC が上がっても実力は上がっていない。",
        "",
        "## 条件",
        "",
        f"- 特徴量セット: `{payload['preset']}`",
        f"- ハイパーパラメータ: `{payload['params_key']}`（全設計で共通）",
        f"- 評価期間: {payload['test_start']} 〜 {payload['test_end']}（全設計で共通）",
        "- エンバーゴは設計ごとのホライズンぶん空ける（訓練が短くなるのは長いホライズンの代償）",
        "",
        "## 結果",
        "",
        "| 軸 | 設計 | 母集団 | 正例率 | 評価n | PR-AUC | 対無情報 | ROC-AUC | 日付内AUC | Lift@5% | 上位5%終盤 | 全件終盤 | 差 | 差の95%区間 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in ok:
        b, a = r.get("outcome_top5", {}), r.get("outcome_all", {})
        name = f"**{r['label']}**" if r["key"] == "base" else r["label"]
        L.append(
            f"| {r['axis']} | {name} | {r['n_all']:,} | {r['positive_rate_all']*100:.2f}% "
            f"| {r['n_test']:,} | {r['pr_auc']:.4f} | {r['pr_gain']:+.4f} "
            f"| {r['roc_auc']:.4f} | {r['within_date_auc']:.4f} | {r['lift@5%']:.2f}x "
            f"| {b.get('end_median', float('nan')):+.2f}% "
            f"| {a.get('end_median', float('nan')):+.2f}% "
            f"| {r.get('edge_end_median', float('nan')):+.2f}pt "
            f"| {_ci_text(r) or '—'} |")
    naive = payload.get("naive_baselines") or []
    if naive:
        L += ["", "## モデルを使わない対照", "",
              "同じ評価期間・同じ物差しで、1列だけで並べたときの"
              "「上位5% - 全件」。",
              "掃引でどれだけ数字が良くなっても、ここに勝てないなら"
              "モデルは要らない。", "",
              "| 並べ方 | 上位5%終盤 | 全件終盤 | 差 | 95%区間 | 上位5%勝率 |",
              "|---|---:|---:|---:|:---:|---:|"]
        for r in naive:
            ci = r["edge_end_ci"]
            L.append(f"| {r['name']} | {r['top5_end_median']:+.2f}% "
                     f"| {r['all_end_median']:+.2f}% "
                     f"| {r['edge_end_median']:+.2f}pt "
                     f"| [{ci[0]:+.2f},{ci[1]:+.2f}] "
                     f"| {r['top5_win_rate']*100:.1f}% |")

    prof = [r for r in ok if r.get("top5_profile")]
    if prof:
        cols = [c for c in PROFILE_COLS
                if any(c in r["top5_profile"] for r in prof)]
        ja = {"vol_20d": "日次ボラ(%)", "log_market_cap": "log時価総額",
              "ret_20d": "20日リターン(%)", "r_high": "高値への近さ",
              "tv_ma20_log": "log売買代金"}
        L += ["", "## 上位5%はどんな銘柄か", "",
              "設計を変えると、モデルが選ぶ銘柄そのものが変わる。",
              "数字が良くなったときに何を選ぶようになったのかが分からないと、",
              "結果を信じる根拠がない。各セルは「上位5%の中央値（全件の中央値）」。", "",
              "| 軸 | 設計 | " + " | ".join(ja.get(c, c) for c in cols) + " |",
              "|---|---|" + "---:|" * len(cols)]
        for r in prof:
            cells = []
            for c in cols:
                v = r["top5_profile"].get(c)
                cells.append(f"{v['top5']:.2f} ({v['all']:.2f})" if v else "—")
            name = f"**{r['label']}**" if r["key"] == "base" else r["label"]
            L.append(f"| {r['axis']} | {name} | " + " | ".join(cells) + " |")

    skipped = [r for r in payload["results"] if "skipped" in r]
    if skipped:
        L += ["", "## 測れなかった設計", ""]
        for r in skipped:
            L.append(f"- {r['axis']} / {r['label']}: {r['skipped']}")
    if base:
        L += ["", "## 基準との比較", "",
              f"基準: ROC-AUC {base['roc_auc']:.4f} / PR-AUC {base['pr_auc']:.4f}"
              f"（対無情報 {base['pr_gain']:+.4f}） / "
              f"上位5%の終盤中央値 {base.get('outcome_top5', {}).get('end_median')}%"]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
