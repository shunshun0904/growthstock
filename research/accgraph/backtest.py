#!/usr/bin/env python3
"""
取引コスト控除後のバックテスト。

分類の精度が上がっても、それが取引として成立するとは限らない。
中立クラスを当てて精度が上がっただけ、ということが普通に起きるので、
「上昇と判定したものを実際に買ったらどうだったか」を別に測る。

## 建て方

超過リターンを目的にしているので、建玉は
「個別株ロング ＋ ベンチマーク（TOPIX または業種指数）ショート」に相当する。
コストは往復ぶんを1トレードにつき1回だけ引く。既定 30bp は
売買手数料と、ロング側のスプレッド・ベンチマーク側のヘッジコストを
まとめた目安であり、実際の執行コストの見積りではない。

## 損益の積み上げ方

保有期間が20営業日あるので、日々エントリーすると建玉が重なる。
重なりを含めたまま平均すると、同じ市場変動を何度も数えてしまう。
ここでは重ならないコホートだけを繋いで資産曲線を作る。

  エントリー日を古い順に見て、直前のコホートの手仕舞い日以降の
  最初の日だけを採用する。同じ日にエントリーする銘柄は等金額で分ける。

これは「常時1枚だけ持つ」戦略で、実際に運用できる形になっている。
建玉を重ねればトレード数は増えるが、そのぶん独立でないサンプルが増えるだけで、
検定としては弱くなる。重なりを含めた統計量は別途、日付でクラスタした
t値として出す。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd

#: 往復の取引コスト（ベーシスポイント）。1トレードにつき1回引く
DEFAULT_COST_BPS = 30.0
#: 年間営業日
TRADING_DAYS_PER_YEAR = 252
#: シャープレシオを出すのに必要な最小コホート数（およそ1年ぶん）
MIN_COHORTS_FOR_SHARPE = 8


@dataclass
class BacktestResult:
    rule: str
    n_trades: int
    n_cohorts: int
    cum_return: float
    ann_return: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    mean_net_per_trade: float
    t_stat_clustered: float
    cost_bps: float

    def to_dict(self) -> Dict:
        return asdict(self)


def select_trades(meta: pd.DataFrame, proba: np.ndarray, classes: np.ndarray,
                  rule: str = "predicted_up", top_frac: float = 0.2,
                  up_class: int = 2) -> np.ndarray:
    """
    どのサンプルを買うか決める。

      predicted_up  argmax が上昇クラス
      topk          同じエントリー日の中で P(上昇) 上位 top_frac
      all           全件買う（比較用の基準線）
    """
    if rule == "all":
        return np.ones(len(meta), dtype=bool)
    up_col = int(np.where(classes == up_class)[0][0])
    if rule == "predicted_up":
        return proba.argmax(axis=1) == up_col
    if rule == "topk":
        p = pd.Series(proba[:, up_col], index=meta.index)
        d = pd.to_datetime(meta["entry_date"])
        rank = p.groupby(d).rank(pct=True, ascending=False)
        return (rank <= top_frac).to_numpy()
    raise ValueError(f"未知の売買ルール: {rule}")


def _exit_dates(meta: pd.DataFrame, horizon: int, max_horizon: int) -> pd.Series:
    """
    手仕舞い日。主保有期間なら label_ready_date が実際の営業日なのでそれを使う。
    それ以外は営業日オフセットで近似する（祝日ぶんずれる）。
    """
    if horizon == max_horizon and "label_ready_date" in meta.columns:
        return pd.to_datetime(meta["label_ready_date"])
    return pd.to_datetime(meta["entry_date"]) + pd.tseries.offsets.BDay(horizon)


def cohort_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """重ならないコホートだけを繋いだ資産曲線。"""
    if trades.empty:
        return pd.DataFrame(columns=["entry_date", "exit_date", "ret", "equity",
                                     "n"])
    by_date = (trades.groupby("entry_date")
               .agg(ret=("net", "mean"), n=("net", "size"),
                    exit_date=("exit_date", "max"))
               .reset_index().sort_values("entry_date"))
    rows = []
    free_from = pd.Timestamp.min
    for r in by_date.itertuples(index=False):
        if r.entry_date < free_from:
            continue
        rows.append({"entry_date": r.entry_date, "exit_date": r.exit_date,
                     "ret": float(r.ret), "n": int(r.n)})
        free_from = r.exit_date
    curve = pd.DataFrame(rows)
    if curve.empty:
        return curve.assign(equity=[])
    curve["equity"] = (1.0 + curve["ret"]).cumprod()
    return curve


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _clustered_t(trades: pd.DataFrame) -> float:
    """
    同じ日のトレードは同じ地合いを共有するので独立ではない。
    日ごとに平均してから t 値を取る（日付でクラスタした検定）。
    """
    if trades.empty:
        return float("nan")
    per_day = trades.groupby("entry_date")["net"].mean()
    n = len(per_day)
    if n < 2:
        return float("nan")
    sd = float(per_day.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(per_day.mean() / (sd / np.sqrt(n)))


def run(meta: pd.DataFrame, proba: np.ndarray, classes: np.ndarray, *,
        excess_col: str, horizon: int, max_horizon: int,
        rule: str = "predicted_up", top_frac: float = 0.2,
        cost_bps: float = DEFAULT_COST_BPS,
        up_class: int = 2) -> BacktestResult:
    pick = select_trades(meta, proba, classes, rule=rule, top_frac=top_frac,
                         up_class=up_class)
    sub = meta.loc[pick, ["entry_date", "Code", excess_col]].copy()
    sub = sub.dropna(subset=[excess_col])
    sub["entry_date"] = pd.to_datetime(sub["entry_date"])
    sub["exit_date"] = _exit_dates(meta.loc[sub.index], horizon, max_horizon)
    sub["net"] = sub[excess_col] - cost_bps / 10_000.0

    curve = cohort_curve(sub)
    if curve.empty:
        return BacktestResult(rule, len(sub), 0, 0.0, 0.0, float("nan"), 0.0,
                              float("nan"), float("nan"), float("nan"), cost_bps)

    rets = curve["ret"].to_numpy()
    periods_per_year = TRADING_DAYS_PER_YEAR / max(horizon, 1)
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    # コホートが数本しかないとシャープレシオは運で決まる。
    # 20営業日保有だと1年で約12本しか取れないので、1年ぶんを下限にする
    if len(rets) < MIN_COHORTS_FOR_SHARPE or sd <= 0:
        sharpe = float("nan")
    else:
        sharpe = float(rets.mean()) / sd * np.sqrt(periods_per_year)
    cum = float(curve["equity"].iloc[-1] - 1.0)
    years = max(len(rets) / periods_per_year, 1e-9)
    ann = float((1.0 + cum) ** (1.0 / years) - 1.0) if cum > -1 else -1.0

    return BacktestResult(
        rule=rule, n_trades=int(len(sub)), n_cohorts=int(len(curve)),
        cum_return=cum, ann_return=ann, sharpe=sharpe,
        max_drawdown=_max_drawdown(curve["equity"].to_numpy()),
        hit_rate=float((sub["net"] > 0).mean()),
        mean_net_per_trade=float(sub["net"].mean()),
        t_stat_clustered=_clustered_t(sub),
        cost_bps=cost_bps,
    )
