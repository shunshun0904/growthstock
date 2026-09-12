#!/usr/bin/env python3
"""
エントリー方策の期待値比較（Phase 1・日足のみ）の中身。

docs/ENTRY_TIMING_DESIGN.md の §8〜§11 を実装する。
API も Release も読まないので、合成データで単体テストできる。
実行用の入口は research/entry_policy_eval.py。

測っているもの
--------------
78週高値を更新した日 t を1件として、翌営業日以降の買い方を変えたときに
1件あたりの平均リターンがどう変わるかを測る。手仕舞いは全方策で共通
（t+56〜t+60 の平均終値。既存の ref_end と同じ）なので、
差が出たらそれはエントリーの差である。

未約定・見送りは 0 とする（建玉が無い＝損も得もしない）。
これは「見送ったぶんの資金は現金で寝かせる」という仮定であり、
他の銘柄に回せるなら過小評価になる。約定率と約定時の平均も併記するので、
そちらから読み直せるようにしてある。

なぜ楽観と悲観を両方出すのか
----------------------------
日足では「安値と高値のどちらが先に付いたか」が分からない。
指値が約定したかどうかは安値だけで決まるので約定自体は判定できるが、
同値で並んだときに約定するかは板の順番次第で決まらない。
そこで「安値≦指値なら約定（楽観）」と「安値が指値より slack_bp 以上
下なら約定（悲観）」を両方出す。**この2つの差が分足に払う金額の上限**であり、
符号が反転するかどうかが契約の判断になる（設計 §12）。
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: 手仕舞い。基準日から数えた営業日。既存ラベルの ref_end と同じ取り方
#: （t+60 時点の直近5営業日平均終値）。ここを方策ごとに変えると、
#: 入りと出を同時に動かすことになって比較が成立しない。
EXIT_HORIZON = 60
EXIT_WINDOW = 5

#: 指値を出しておける最長の営業日数。ここまでの先行値を用意する
MAX_HOLD_DAYS = 5

#: しきい値比較の許容差。ギャップは割り算で作るので、+10%ちょうどが
#: 10.000000000000009 になり「上限10%」に引っかかって見送られていた。
#: しきい値は人が決めた数字なので、浮動小数の端数で判定を変えない。
_EPS = 1e-9


@dataclasses.dataclass(frozen=True)
class Costs:
    """執行に伴う目減り。入れないと押し目指値が必ず勝つ。"""
    #: 往復の手数料・諸費用（%）。証券会社ごとに違うので設定値にしてある
    fee_pct: float = 0.05
    #: 悲観側の約定判定で要求する余裕（bp）。
    #: 呼値1枚ぶんの丸めは 3,000円の株で約3bp なので、この幅に含まれる。
    slack_bp: float = 10.0


def _forward(g: pd.core.groupby.generic.SeriesGroupBy, k: int) -> pd.Series:
    """k営業日先の値。銘柄をまたがない（groupby.shift が境界を NaN にする）。"""
    return g.shift(-k)


def build_events(panel: pd.DataFrame,
                 min_trading_value: Optional[float] = 0.1,
                 max_hold_days: int = MAX_HOLD_DAYS,
                 ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    高値更新日を1行にして、翌営業日以降の値動きと手仕舞い価格を横に並べる。

    `panel` は build_dataset.price_panel + mark_new_highs を通したもの。
    `open` 列（分割調整後の始値）は呼び出し側で付けておく。

    件数の減り方（ファネル）も一緒に返す。どの条件で何件落ちたかを
    出しておかないと、母集団が既存モデルとずれたことに気づけない。
    """
    need = {"Code", "Date", "open", "high", "low", "close", "is_fresh_break"}
    missing = need - set(panel.columns)
    if missing:
        raise SystemExit(f"panel に列がありません: {sorted(missing)}")

    df = panel.sort_values(["Code", "Date"]).reset_index(drop=True)
    g = df.groupby("Code", sort=False)

    for k in range(1, max_hold_days + 1):
        for col in ("open", "high", "low", "close"):
            df[f"{col}{k}"] = _forward(g[col], k)

    # 手仕舞い: t+60 時点の直近 EXIT_WINDOW 営業日平均終値。
    # rolling で作ってから -EXIT_HORIZON ずらす（build_dataset の end_level と同じ形）
    ma = g["close"].transform(
        lambda s: s.rolling(EXIT_WINDOW, min_periods=EXIT_WINDOW).mean())
    df["exit_price"] = ma.groupby(df["Code"], sort=False).shift(-EXIT_HORIZON)
    # 先の営業日が足りない末尾は手仕舞い価格が作れない。0 で埋めない
    have_forward = g.cumcount(ascending=False) >= EXIT_HORIZON
    df["exit_price"] = df["exit_price"].where(have_forward)

    ev = df[df["is_fresh_break"] == True].copy()   # noqa: E712
    funnel = {"新規ブレイク": len(ev)}

    if "high52w" in ev.columns:
        ev = ev[ev["high52w"].notna()]
        funnel["52週高値が定義できる"] = len(ev)
    if min_trading_value is not None and "tv_ma20" in ev.columns:
        ev = ev[ev["tv_ma20"] >= min_trading_value]
        funnel[f"20日平均売買代金>={min_trading_value}億円"] = len(ev)
    ev = ev[ev["exit_price"].notna() & ev["close"].gt(0)]
    funnel["手仕舞い価格が確定"] = len(ev)
    ev = ev[ev["open1"].notna() & ev["open1"].gt(0)]
    funnel["翌営業日に値が付いた"] = len(ev)

    ev["gap_pct"] = (ev["open1"] / ev["close"] - 1.0) * 100.0
    # 持ち切ったときの水準（基準日終値ベース）。既存の ref_end と同じ定義。
    # 突き合わせ用に残す。方策の比較そのものには使わない
    ev["ref_end_pct"] = (ev["exit_price"] / ev["close"] - 1.0) * 100.0
    return ev.reset_index(drop=True), funnel


# --------------------------------------------------------------------------- #
# 方策
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class Policy:
    """1つのエントリー方策。`kind` で執行の形が決まる。"""
    name: str
    kind: str                       # "open" | "dip"
    #: "open": この値を超えるギャップなら見送る（None なら常に買う）
    gap_cap: Optional[float] = None
    #: "dip": 指値の深さ（%）
    depth: Optional[float] = None
    #: "dip": 指値を出しておく営業日数
    hold: int = 1
    #: "dip": 指値の基準。"close"=前夜に C_t から / "open"=寄り後に O_{t+1} から
    anchor: str = "open"
    #: "dip": 期限まで約定しなければ、その日の引けで成行にするか
    chase: bool = False

    @property
    def decision_time(self) -> str:
        """いつ決める方策か。使える情報が違うので、混ぜて比べない。"""
        if self.kind == "dip" and self.anchor == "close":
            return "T0 寄り前"
        return "T1 寄り後"


def default_policies(depths: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
                     holds: Sequence[int] = (1, 3, 5),
                     gap_caps: Sequence[float] = (2.0, 3.0, 5.0, 7.0, 10.0),
                     ) -> List[Policy]:
    """
    比較する方策の格子。**先に決めて動かさない**。
    結果を見てから格子を広げると、そのぶん偶然の勝者が増える（設計 §10）。
    """
    out: List[Policy] = [Policy("寄り成行", "open")]
    for c in gap_caps:
        out.append(Policy(f"寄り成行 / ギャップ>{c:g}%は見送り", "open", gap_cap=c))
    for anchor, label in (("open", "始値基準"), ("close", "前夜・前日終値基準")):
        for d in depths:
            for h in holds:
                out.append(Policy(
                    f"押し目 -{d:g}% / {h}日 / {label} / 未約定は見送り",
                    "dip", depth=d, hold=h, anchor=anchor, chase=False))
                out.append(Policy(
                    f"押し目 -{d:g}% / {h}日 / {label} / 期限で成行",
                    "dip", depth=d, hold=h, anchor=anchor, chase=True))
    return out


def execute(ev: pd.DataFrame, pol: Policy, costs: Costs,
            optimistic: bool) -> Dict[str, np.ndarray]:
    """
    1つの方策を全イベントに執行して、約定の有無とエントリー価格を返す。

    返すのは配列だけ。集計はしない（部分集合ごとに集計し直したいため）。
    """
    n = len(ev)
    entry = np.full(n, np.nan)
    filled = np.zeros(n, dtype=bool)

    if pol.kind == "open":
        ok = np.ones(n, dtype=bool)
        if pol.gap_cap is not None:
            ok = ev["gap_pct"].to_numpy(dtype=float) <= pol.gap_cap + _EPS
        o1 = ev["open1"].to_numpy(dtype=float)
        ok &= np.isfinite(o1)
        entry[ok] = o1[ok]
        filled = ok
        return {"entry": entry, "filled": filled}

    if pol.kind != "dip":
        raise ValueError(f"未知の方策: {pol.kind}")

    base = (ev["close"] if pol.anchor == "close" else ev["open1"]).to_numpy(dtype=float)
    limit = base * (1.0 - pol.depth / 100.0)
    # 悲観: 安値が指値より slack_bp 以上下でないと約定しない。
    # 楽観: 安値が指値以下なら約定する。
    trigger = limit if optimistic else limit * (1.0 - costs.slack_bp / 10000.0)

    for k in range(1, pol.hold + 1):
        lo = ev[f"low{k}"].to_numpy(dtype=float)
        op = ev[f"open{k}"].to_numpy(dtype=float)
        hit = (~filled) & np.isfinite(lo) & (lo <= trigger)
        # 指値より下で寄れば、約定価格は指値ではなく寄り値になる。
        # anchor="open" の初日は指値 < 始値 が定義上成り立つので、この分岐は効かない
        #（寄り値を見てから置く指値が、その寄り値より下にあるため）。
        px = np.where(np.isfinite(op) & (op < limit), op, limit)
        entry[hit] = px[hit]
        filled |= hit

    if pol.chase:
        # 期限日の引けで成行。引けが取れない行は約定なしのまま
        cl = ev[f"close{pol.hold}"].to_numpy(dtype=float)
        late = (~filled) & np.isfinite(cl) & (cl > 0)
        entry[late] = cl[late]
        filled |= late

    return {"entry": entry, "filled": filled}


def returns_of(ev: pd.DataFrame, pol: Policy, costs: Costs,
               optimistic: bool) -> np.ndarray:
    """
    1件あたりのリターン（%）。未約定・見送りは 0。

    手仕舞い価格は全方策で共通なので、差が出たらエントリーの差である。
    """
    r = execute(ev, pol, costs, optimistic)
    entry, filled = r["entry"], r["filled"]
    exit_px = ev["exit_price"].to_numpy(dtype=float)
    out = np.zeros(len(ev))
    ok = filled & np.isfinite(entry) & (entry > 0) & np.isfinite(exit_px)
    out[ok] = (exit_px[ok] / entry[ok] - 1.0) * 100.0 - costs.fee_pct
    return out


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #

def summarize(ev: pd.DataFrame, pol: Policy, costs: Costs,
              optimistic: bool) -> Dict[str, float]:
    """1方策の成績。平均だけでなく約定率と下側も出す。"""
    r = execute(ev, pol, costs, optimistic)
    entry, filled = r["entry"], r["filled"]
    exit_px = ev["exit_price"].to_numpy(dtype=float)
    ret = returns_of(ev, pol, costs, optimistic)
    n = len(ev)
    took = int(filled.sum())

    # 取り逃し: 約定しなかったが、寄りで買っていれば勝っていた件
    o1 = ev["open1"].to_numpy(dtype=float)
    would = np.isfinite(o1) & (o1 > 0) & np.isfinite(exit_px) & (exit_px > o1)
    missed = int((~filled & would).sum())

    # エントリー価格の改善（寄り値比）。約定した行だけで測る
    imp = np.full(n, np.nan)
    okm = filled & np.isfinite(entry) & (entry > 0) & np.isfinite(o1) & (o1 > 0)
    imp[okm] = (o1[okm] / entry[okm] - 1.0) * 100.0

    filled_ret = ret[filled] if took else np.array([])
    return {
        "n": n,
        "fill_rate": round(100.0 * took / n, 2) if n else float("nan"),
        "mean": round(float(ret.mean()), 4) if n else float("nan"),
        "median": round(float(np.median(ret)), 4) if n else float("nan"),
        "p05": round(float(np.percentile(ret, 5)), 4) if n else float("nan"),
        "mean_if_filled": round(float(filled_ret.mean()), 4) if took else float("nan"),
        "win_rate_if_filled": (round(100.0 * float((filled_ret > 0).mean()), 2)
                               if took else float("nan")),
        "missed_rate": round(100.0 * missed / n, 2) if n else float("nan"),
        "entry_improve": (round(float(np.nanmean(imp)), 4)
                          if np.isfinite(imp).any() else float("nan")),
    }


def bootstrap_means(ev: pd.DataFrame, mat: np.ndarray, baseline: int,
                    n_boot: int = 2000, seed: int = 0) -> List[Dict[str, float]]:
    """
    全方策の「基準との平均差」の95%区間を、**日単位**の復元抽出で一度に出す。

    行単位で回してはいけない。同じ日のブレイクは地合いを共有していて
    独立でないため、区間が実際より狭く出る。

    方策ごとに引き直すと同じ抽出を何十回もやり直すことになるので、
    日ごとの合計まで畳んでから行列積で一気に回す。
    `mat` は (イベント数, 方策数) のリターン行列。
    """
    n, n_pol = mat.shape
    obs = mat.mean(axis=0) - mat[:, baseline].mean() if n else np.full(n_pol, np.nan)
    empty = [{"diff": round(float(obs[i]), 4) if n else float("nan"),
              "lo": float("nan"), "hi": float("nan"),
              "p_gt0": float("nan"), "n_boot": 0} for i in range(n_pol)]
    if n == 0 or n_boot <= 0:
        return empty

    codes, _ = pd.factorize(ev["Date"].to_numpy())
    n_dates = int(codes.max()) + 1
    if n_dates < 2:
        return empty
    sums = np.zeros((n_dates, n_pol))
    np.add.at(sums, codes, mat)
    cnt = np.bincount(codes, minlength=n_dates).astype(float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_dates, size=(n_boot, n_dates))
    # 各抽出で、その日が何回選ばれたか。重みにして行列積で畳む。
    # np.add.at は 400万要素だと目に見えて遅いので、
    # 抽出をまたいで通し番号にしてから bincount 1回で数える。
    flat = (np.arange(n_boot)[:, None] * n_dates + idx).ravel()
    w = np.bincount(flat, minlength=n_boot * n_dates).astype(float)
    w = w.reshape(n_boot, n_dates)

    tot = w @ cnt
    good = tot > 0
    means = (w @ sums)[good] / tot[good, None]
    diffs = means - means[:, [baseline]]

    out = []
    for i in range(n_pol):
        col = diffs[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            out.append(empty[i])
            continue
        out.append({"diff": round(float(obs[i]), 4),
                    "lo": round(float(np.percentile(col, 2.5)), 4),
                    "hi": round(float(np.percentile(col, 97.5)), 4),
                    "p_gt0": round(float((col > 0).mean()), 4),
                    "n_boot": int(col.size)})
    return out


def compare(ev: pd.DataFrame, policies: Sequence[Policy], costs: Costs,
            optimistic: bool, baseline: int = 0,
            n_boot: int = 2000, seed: int = 0) -> List[Dict]:
    """全方策を基準（既定は寄り成行）と比べる。基準自身も含めて返す。"""
    mat = np.column_stack([returns_of(ev, p, costs, optimistic)
                           for p in policies]) if len(ev) else \
        np.zeros((0, len(policies)))
    cis = bootstrap_means(ev, mat, baseline, n_boot, seed)
    out = []
    for i, p in enumerate(policies):
        row = {"policy": p.name, "decision_time": p.decision_time,
               "kind": p.kind, "depth": p.depth, "hold": p.hold,
               "anchor": p.anchor, "chase": p.chase, "gap_cap": p.gap_cap}
        row.update(summarize(ev, p, costs, optimistic))
        row["vs_base"] = cis[i]
        out.append(row)
    return out


def flip_check(train_or_test: pd.DataFrame, policies: Sequence[Policy],
               costs: Costs, baseline: int = 0) -> List[Dict]:
    """
    楽観と悲観で、基準に対する優劣の**符号が反転する**方策を拾う。

    設計 §12 の判断そのもの。反転するなら、決め手は日中の到達順序であり
    それは日足では分からない。分足に金を払う理由になる。
    """
    opt = {r["policy"]: r for r in compare(train_or_test, policies, costs,
                                           optimistic=True, baseline=baseline,
                                           n_boot=0)}
    pes = {r["policy"]: r for r in compare(train_or_test, policies, costs,
                                           optimistic=False, baseline=baseline,
                                           n_boot=0)}
    out = []
    for name, o in opt.items():
        p = pes[name]
        do, dp = o["vs_base"]["diff"], p["vs_base"]["diff"]
        if not (np.isfinite(do) and np.isfinite(dp)):
            continue
        out.append({"policy": name, "optimistic": do, "pessimistic": dp,
                    "flips": bool(do > 0) != bool(dp > 0),
                    "width": round(abs(do - dp), 4)})
    out.sort(key=lambda r: -r["width"])
    return out
