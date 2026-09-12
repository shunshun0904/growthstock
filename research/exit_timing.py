#!/usr/bin/env python3
"""
出口（いつ売るか）の比較。API も Release も読まないので合成データで単体テストできる。

なぜ要るか
----------
既存モデルのラベルは「3ヶ月のどこかで +1.2σ√60 に到達し、3ヶ月後もその半分は
残っている」である。到達が**いつ**かは問うていない。ところが今までの評価は
t+56〜t+60 の水準（＝終盤）しか見ておらず、到達のピークを捨てていた。

モデルが選んでいるもの（ピークを付ける銘柄）と、測っているもの（3ヶ月後の水準）が
ずれている。ずれの大きさと、どこで売れば一番取れるかを実測する。

測り方
------
入口は Phase 1 の結論どおり「翌営業日の寄り」に固定する
（docs/ENTRY_POLICY_PHASE1.md）。動かすのは出口だけ。

  固定日数    t+k の終値で売る（k を振る）
  ラベル準拠  基準日終値から need 以上に達した最初の日の終値で売る。
              届かなければ horizon 日目の終値
  ピーク      最大値が実際に何日目に付いたか（後から見た上限。実現できない）

評価は2つ並べる。
  1件あたり   1回の売買でいくら儲かるか
  時間あたり  1件あたり ÷ 平均保有日数 × 20営業日（＝1ヶ月換算）

後者が要るのは、60日で+5%と20日で+3%が同じ資金効率ではないため。
ただし「売った翌日に次の1件へ乗り換えられる」という仮定が入る。
候補は1日数件しか出ないので、実際には常に乗り換えられるとは限らない。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: 先読みする営業日数。既存ラベルの地平（60）の4倍。
#:
#: 最初は120で測ったが、最大値の31%が111〜120日目に付いていた。
#: まだ上がり続けている途中で窓を閉じており、「どこで頭打ちになるか」を
#: measurable にしていなかった。山を跨ぐところまで延ばす。
#: 延ばした分だけ直近のイベントが母集団から外れるのは承知のうえ。
FORWARD_DAYS = 250

#: 固定日数で売る場合に並べる保有日数
HORIZONS = (1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250)

#: 1ヶ月換算に使う営業日数
MONTH_DAYS = 20

#: しきい値比較の許容差（相対）。level = base × (1+need) は割り算・掛け算で
#: 作るので、100 × 1.10 が 110.00000000000001 になる。ちょうど到達した日を
#: 浮動小数の端数で「未到達」にしない。
_REL_EPS = 1e-12


def _forward_index(panel: pd.DataFrame, ev: pd.DataFrame, days: int):
    """各イベントの t, t+1, ..., t+days が panel の何行目かと、その有効性。"""
    if "_pos" not in ev.columns:
        raise SystemExit("ev に _pos がありません。panel に通し番号を振ってください")
    remaining = panel.groupby("Code", sort=False).cumcount(
        ascending=False).to_numpy()
    pos = ev["_pos"].to_numpy(dtype=int)
    offs = np.arange(days + 1)
    idx = np.minimum(pos[:, None] + offs[None, :], len(panel) - 1)
    ok = offs[None, :] <= remaining[pos][:, None]
    return idx, ok


def forward_matrix(panel: pd.DataFrame, ev: pd.DataFrame,
                   days: int = FORWARD_DAYS) -> np.ndarray:
    """
    各イベントについて t, t+1, ..., t+days の終値を横に並べた行列を返す。

    `panel` は Code, Date で整列済みで `_pos`（通し番号）を持つこと。
    銘柄の末尾を越えた先は NaN にする。0 で埋めない
    （「まだ先が無い」と「値が0」は別物）。
    """
    idx, ok = _forward_index(panel, ev, days)
    close = panel["close"].to_numpy(dtype=float)
    return np.where(ok, close[idx], np.nan)


def benchmark_matrix(panel: pd.DataFrame, ev: pd.DataFrame,
                     topix: pd.DataFrame, days: int = FORWARD_DAYS) -> np.ndarray:
    """
    forward_matrix と同じ形で、**その銘柄の各営業日と同じ日付の** TOPIX 終値を返す。

    なぜ要るか: 250営業日持てば、その間の市場全体の上昇もそのまま乗る。
    「長く持つほど良い」が地合いなのか銘柄選定なのかは、指数を引かないと分からない。

    日付で突き合わせる。位置で合わせると、売買が成立しなかった日がある銘柄で
    「その銘柄の250日目」と「市場の250日目」がずれる。
    """
    idx, ok = _forward_index(panel, ev, days)
    dates = panel["Date"].to_numpy("datetime64[ns]")[idx]
    tp = topix.dropna().sort_values("Date")
    td = pd.to_datetime(tp["Date"]).to_numpy("datetime64[ns]")
    tv = pd.to_numeric(tp["topix"], errors="coerce").to_numpy(dtype=float)
    j = np.searchsorted(td, dates)
    jc = np.clip(j, 0, len(td) - 1)
    hit = (j < len(td)) & (td[jc] == dates)
    return np.where(ok & hit, tv[jc], np.nan)


def _excess(ret: np.ndarray, bm: Optional[np.ndarray],
            exit_col: np.ndarray) -> Optional[np.ndarray]:
    """
    銘柄のリターン（手数料込み）から、同じ期間の指数リターン（手数料なし）を引く。

    指数側に手数料を掛けないので、超過はわずかに銘柄に不利な側に出る。
    向きが分かっていれば、その方が安全。
    """
    if bm is None:
        return None
    rows = np.arange(len(ret))
    base = bm[:, 0]
    px = bm[rows, exit_col]
    bret = (px / base - 1.0) * 100.0
    return ret - bret


def _excess_stats(exc: Optional[np.ndarray]) -> Dict[str, float]:
    if exc is None:
        return {}
    ok = np.isfinite(exc)
    if not ok.any():
        return {}
    e = exc[ok]
    return {"excess_n": int(ok.sum()),
            "excess_mean": round(float(e.mean()), 3),
            "excess_median": round(float(np.median(e)), 3),
            "excess_win": round(100.0 * float((e > 0).mean()), 2)}


def _stats(ret: np.ndarray, hold: np.ndarray) -> Dict[str, float]:
    """1件あたりと時間あたりの両方を出す。平均だけだと下側が見えない。"""
    ok = np.isfinite(ret)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0}
    r = ret[ok]
    h = hold[ok].astype(float)
    mean = float(r.mean())
    mean_hold = float(h.mean())
    return {
        "n": n,
        "mean": round(mean, 3),
        "median": round(float(np.median(r)), 3),
        "win_rate": round(100.0 * float((r > 0).mean()), 2),
        "p05": round(float(np.percentile(r, 5)), 3),
        "p95": round(float(np.percentile(r, 95)), 3),
        "hold_days": round(mean_hold, 1),
        "per_month": (round(mean / mean_hold * MONTH_DAYS, 3)
                      if mean_hold > 0 else float("nan")),
    }


def fixed_horizon(F: np.ndarray, entry: np.ndarray, fee_pct: float = 0.05,
                  horizons: Sequence[int] = HORIZONS,
                  bm: Optional[np.ndarray] = None) -> List[Dict]:
    """t+k の終値で売る。k ごとの成績を並べる。"""
    out = []
    for k in horizons:
        if k >= F.shape[1]:
            continue
        ret = (F[:, k] / entry - 1.0) * 100.0 - fee_pct
        row = {"rule": f"{k}営業日で売る", "kind": "fixed", "k": k}
        row.update(_stats(ret, np.full(len(ret), k)))
        row.update(_excess_stats(_excess(ret, bm, np.full(len(ret), k))))
        out.append(row)
    return out


def target_exit(F: np.ndarray, entry: np.ndarray, need: np.ndarray,
                horizon: int = 60, fee_pct: float = 0.05,
                ratio: float = 1.0, label: str = "",
                bm: Optional[np.ndarray] = None) -> Dict:
    """
    ラベル準拠の出口。基準日終値から `ratio × need` 以上に達した
    最初の日の終値で売る。届かなければ horizon 日目の終値で売る。

    しきい値の分母を基準日終値（C_t）にするのは、ラベルがそう定義されているため
    （build_dataset.rise_thresholds）。エントリー価格に合わせて測り直すと、
    モデルが選んだ条件とは別物を測ることになる。
    """
    base = F[:, 0]
    level = base * (1.0 + ratio * need)
    win = F[:, 1:horizon + 1]
    hit = np.isfinite(win) & (win >= level[:, None] * (1.0 - _REL_EPS))
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1) + 1, horizon)
    px = F[np.arange(len(F)), first]
    ret = (px / entry - 1.0) * 100.0 - fee_pct
    name = label or f"{ratio:g}×到達しきい値で売る（届かなければ{horizon}日）"
    row = {"rule": name, "kind": "target", "k": None,
           "ratio": ratio, "horizon": horizon,
           "hit_rate": round(100.0 * float(any_hit.mean()), 2)}
    row.update(_stats(ret, first))
    row.update(_excess_stats(_excess(ret, bm, first)))
    return row


def peak_profile(F: np.ndarray, entry: np.ndarray, upto: int,
                 fee_pct: float = 0.05) -> Dict:
    """
    最大値が何日目に付いたか、そのとき何%だったか。

    **後から見た上限であって、実現できる数字ではない。**
    固定日数の成績がこれにどれだけ届いていないかを測る物差しとして出す。
    """
    win = F[:, 1:upto + 1]
    ok = np.isfinite(win).any(axis=1)
    day = np.full(len(F), np.nan)
    px = np.full(len(F), np.nan)
    if ok.any():
        sub = win[ok]
        j = np.nanargmax(np.where(np.isfinite(sub), sub, -np.inf), axis=1)
        day[ok] = j + 1
        px[ok] = sub[np.arange(len(sub)), j]
    ret = (px / entry - 1.0) * 100.0 - fee_pct
    d = day[np.isfinite(day)]
    out = {"upto": upto, "n": int(np.isfinite(ret).sum())}
    out.update({f"peak_{k}": v for k, v in
                _stats(ret, np.where(np.isfinite(day), day, np.nan)).items()})
    if d.size:
        out["day_median"] = float(np.median(d))
        out["day_q1"] = float(np.percentile(d, 25))
        out["day_q3"] = float(np.percentile(d, 75))
        # 10営業日ごとの山。どのあたりに寄っているかを形で見る
        step = 10 if upto <= 120 else 25
        edges = np.arange(0, upto + step, step)
        cnt, _ = np.histogram(d, bins=edges)
        out["day_hist"] = [{"from": int(edges[i]) + 1, "to": int(edges[i + 1]),
                            "n": int(cnt[i]),
                            "share": round(100.0 * cnt[i] / d.size, 1)}
                           for i in range(len(cnt))]
    return out


def compare_exits(F: np.ndarray, entry: np.ndarray, need: np.ndarray,
                  fee_pct: float = 0.05,
                  horizons: Sequence[int] = HORIZONS,
                  bm: Optional[np.ndarray] = None) -> List[Dict]:
    """固定日数とラベル準拠を1つの表に並べる。"""
    rows = fixed_horizon(F, entry, fee_pct, horizons, bm=bm)
    rows.append(target_exit(F, entry, need, horizon=60, fee_pct=fee_pct,
                            ratio=1.0, bm=bm,
                            label="到達しきい値(+1.2σ√60)に届いたら売る / 60日で打ち切り"))
    rows.append(target_exit(F, entry, need, horizon=60, fee_pct=fee_pct,
                            ratio=0.5, bm=bm,
                            label="到達しきい値の半分で売る / 60日で打ち切り"))
    rows.append(target_exit(F, entry, need, horizon=120, fee_pct=fee_pct,
                            ratio=1.0, bm=bm,
                            label="到達しきい値に届いたら売る / 120日で打ち切り"))
    return rows


# --------------------------------------------------------------------------- #
# 候補の選び方
# --------------------------------------------------------------------------- #

def subset_masks(ev: pd.DataFrame, quantiles: Sequence[float] = (0.80, 0.90, 0.95),
                 top_n: Sequence[int] = (1, 2, 3),
                 min_scored: int = 200) -> List[Dict]:
    """
    「スコアの絶対しきい値」と「その日の上位N件」を並べる。

    この2つは同じものではない。候補が弱い日でも上位N件は必ず選ばれるので、
    日をまたぐと低いスコアが混ざる。どちらで運用するかは取引機会と質の
    釣り合いで決まるので、件数も一緒に出す。
    """
    out: List[Dict] = [{"name": "全件", "mask": np.ones(len(ev), dtype=bool),
                        "kind": "all"}]
    # スコアが十分な件数そろっていなければ分けない。
    # 数十件で「上位10%」を作っても読めない数字が出るだけ。
    if "score" not in ev.columns or ev["score"].notna().sum() < min_scored:
        return out
    s = ev["score"]
    for q in quantiles:
        thr = float(s.quantile(q))
        out.append({"name": f"スコア>={thr:.4f}（上位{(1-q)*100:.0f}%）",
                    "mask": (s >= thr).to_numpy(dtype=bool),
                    "kind": "threshold", "threshold": thr})
    rank = ev.groupby("Date")["score"].rank(ascending=False, method="first")
    for n in top_n:
        out.append({"name": f"その日の上位{n}件",
                    "mask": (rank <= n).to_numpy(dtype=bool),
                    "kind": "top_n", "top_n": n})
    return out


def subset_profile(ev: pd.DataFrame, mask: np.ndarray) -> Dict:
    """部分集合の素性。スコアの水準が違えば成績が違うのは当たり前なので先に出す。"""
    sub = ev[mask]
    n = len(sub)
    days = pd.to_datetime(sub["Date"]) if n else pd.Series([], dtype="datetime64[ns]")
    span_years = ((days.max() - days.min()).days / 365.25) if n > 1 else float("nan")
    out = {"n": n,
           "per_year": round(n / span_years, 1) if span_years and span_years > 0
           else float("nan")}
    for col, key in (("score", "score_mean"), ("label", "label_rate"),
                     ("vol_20d", "vol_mean")):
        if col in sub.columns and n:
            v = pd.to_numeric(sub[col], errors="coerce")
            if v.notna().any():
                out[key] = round(float(v.mean()) * (100.0 if col == "label" else 1.0), 4)
    return out
