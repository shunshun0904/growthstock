#!/usr/bin/env python3
"""
時系列クロスバリデーション（Purged / Embargo つき）。

決算サンプルは「発表翌営業日に入り、20営業日後に出る」ので、
1サンプルが未来の20営業日ぶんの価格情報を内包している。
発表日だけで訓練/テストを切ると、訓練の最後のサンプルのラベルが
テスト期間の価格で決まっていて、そこからリークする。

  Purge   … ラベル確定日がテスト開始日以降になる訓練サンプルを外す
  Embargo … さらに緩衝期間を置き、テスト直前の訓練サンプルも外す

銘柄はまたがない。同じ日に複数銘柄が決算を出すので、銘柄で分けると
同じ市場環境が訓練とテストの両方に入る。分けるのは時間軸だけにする。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd

#: 営業日 -> 暦日のおおよその換算（年間 252営業日 / 365日）
TRADING_TO_CALENDAR = 365.0 / 252.0


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    n_purged: int
    n_embargoed: int

    def to_dict(self) -> Dict:
        return asdict(self)


def walk_forward(meta: pd.DataFrame, *, min_train_months: int = 48,
                 test_months: int = 12, step_months: int = 12,
                 embargo_trading_days: int = 20,
                 date_col: str = "entry_date",
                 ready_col: str = "label_ready_date",
                 min_test_rows: int = 200):
    """
    訓練窓を伸ばしながらテスト窓を前に進める（expanding window）。

    金融データは履歴が限られるので、古い期間を捨てる rolling ではなく
    全履歴を使う。テスト窓は重ねない（重ねるとフォールド間が相関して、
    独立に検証したように見えてしまう）。

    返り値は (folds, train_masks, test_masks)。
    """
    d = pd.to_datetime(meta[date_col])
    ready = pd.to_datetime(meta[ready_col]) if ready_col in meta.columns else d
    # ラベル確定日が無いサンプルは、いつ確定したか分からないので
    # 訓練に使うと purge できない。エントリー日 + 保有期間で代用せず、除外する
    ready = ready.fillna(pd.Timestamp.max)

    d0, dmax = d.min(), d.max()
    embargo = pd.Timedelta(days=int(round(embargo_trading_days * TRADING_TO_CALENDAR)))

    folds: List[Fold] = []
    train_masks: List[np.ndarray] = []
    test_masks: List[np.ndarray] = []

    i = 0
    while True:
        test_start = d0 + pd.DateOffset(months=min_train_months + i * step_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_start >= dmax:
            break

        in_test = (d >= test_start) & (d < test_end)
        before = d < test_start
        # Purge: ラベルがテスト期間の価格で決まる訓練サンプルを外す
        purged = before & (ready >= test_start)
        # Embargo: テスト直前の緩衝期間
        embargoed = before & ~purged & (d >= test_start - embargo)
        in_train = before & ~purged & ~embargoed

        if int(in_test.sum()) >= min_test_rows and int(in_train.sum()) > 0:
            folds.append(Fold(
                index=len(folds),
                train_start=str(d[in_train].min().date()),
                train_end=str(d[in_train].max().date()),
                test_start=str(test_start.date()),
                test_end=str(min(test_end, dmax).date()),
                n_train=int(in_train.sum()), n_test=int(in_test.sum()),
                n_purged=int(purged.sum()), n_embargoed=int(embargoed.sum()),
            ))
            train_masks.append(in_train.to_numpy())
            test_masks.append(in_test.to_numpy())
        i += 1
        if test_end > dmax + pd.DateOffset(months=test_months):
            break

    if not folds:
        raise SystemExit(
            "フォールドが1つも作れません。min_train_months / test_months を"
            "データの長さに合わせて小さくしてください")
    return folds, train_masks, test_masks


def holdout(meta: pd.DataFrame, test_start: str, *,
            embargo_trading_days: int = 20,
            date_col: str = "entry_date", ready_col: str = "label_ready_date"):
    """単一分割。最終的な確認用で、これだけで判断はしない。"""
    d = pd.to_datetime(meta[date_col])
    ready = pd.to_datetime(meta[ready_col]).fillna(pd.Timestamp.max)
    ts = pd.Timestamp(test_start)
    embargo = pd.Timedelta(days=int(round(embargo_trading_days * TRADING_TO_CALENDAR)))
    in_test = d >= ts
    before = d < ts
    purged = before & (ready >= ts)
    embargoed = before & ~purged & (d >= ts - embargo)
    in_train = before & ~purged & ~embargoed
    f = Fold(index=0, train_start=str(d[in_train].min().date()),
             train_end=str(d[in_train].max().date()),
             test_start=str(ts.date()), test_end=str(d.max().date()),
             n_train=int(in_train.sum()), n_test=int(in_test.sum()),
             n_purged=int(purged.sum()), n_embargoed=int(embargoed.sum()))
    return [f], [in_train.to_numpy()], [in_test.to_numpy()]


def describe(folds: List[Fold]) -> str:
    lines = [f"フォールド {len(folds)}本"]
    for f in folds:
        lines.append(
            f"  #{f.index}: 訓練 {f.train_start}〜{f.train_end} ({f.n_train:,}件) "
            f"/ テスト {f.test_start}〜{f.test_end} ({f.n_test:,}件) "
            f"[purge {f.n_purged:,} / embargo {f.n_embargoed:,}]")
    return "\n".join(lines)
