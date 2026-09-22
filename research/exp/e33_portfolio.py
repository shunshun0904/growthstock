#!/usr/bin/env python3
"""
実験33: 資金枠（同時に持てる銘柄数）を固定したときに、戦略がどうなるか。

実験30〜32 は1取引あたりの成績だった。実際には枠が埋まっていれば見送るので、
「年に何回入れて、資産がどれだけ増えるか」は別の数字になる。

規則（実験30〜32 の実測から）
  選定  3モデル（LGBM / XGBoost / CatBoost）すべてが上位10%
        かつ その日の母集団の発火数が MIN_BREAK 件以上
        並べ方は「3モデルの最小順位」で上位 TOP_K 件
  入り  翌営業日の寄り
  出口  +20%（または他の目標）に届いたらその日に売る。届かなければ 20営業日で売る
  枠    同時に SLOTS 銘柄まで。埋まっていたら見送る（先着順）

資産は枠ごとに複利で回す（1枠 = 資金の 1/SLOTS）。
結果は research/_data/oof/e33_*.csv。
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
from e27_timing_multi import OOF_DIR, log  # noqa: E402
from e31_strategy import prepare, RULE  # noqa: E402
from e32_takeprofit import TARGETS, HOLD, attach, forward_paths  # noqa: E402


def calendar() -> pd.Series:
    d = pd.concat([pd.read_parquet(p, columns=["Date"])
                   for p in sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))])["Date"]
    return pd.Series(np.sort(pd.to_datetime(d.unique())))


def signals(d: pd.DataFrame, min_break: int, top_k: int) -> pd.DataFrame:
    s = d[d[RULE] & (d["n_break"] >= min_break)].sort_values(
        ["Date", "p_min"], ascending=[True, False])
    s = s.assign(rank=s.groupby("Date").cumcount() + 1)
    return s[s["rank"] <= top_k].sort_values(["Date", "rank"]).reset_index(drop=True)


def simulate(sig: pd.DataFrame, cal: pd.Series, slots: int, target: float) -> dict:
    """
    先着順に枠へ入れる。枠が空くのは売った日（その日のうちに次を買えるとする）。
    戻り値は枠ごとの複利と、取れた／見送った件数。
    """
    k = int(target)
    idx = {d: i for i, d in enumerate(cal)}
    free = [-1] * slots               # 枠ごとの「次に買える営業日の番号」
    wealth = [1.0] * slots
    taken, skipped = [], 0
    for _, row in sig.iterrows():
        di = idx.get(pd.Timestamp(row["Date"]))
        if di is None:
            continue
        # 買うのは翌営業日なので、枠は di+1 に空いていればよい
        buy = di + 1
        avail = [i for i, f in enumerate(free) if f <= buy]
        if not avail:
            skipped += 1
            continue
        slot = avail[0]
        hit = bool(row[f"hit{k}"])
        days = int(row[f"day{k}"]) if hit and pd.notna(row[f"day{k}"]) else HOLD
        ret = float(row[f"tp{k}"]) / 100.0
        if not np.isfinite(ret):
            continue
        free[slot] = buy + days
        wealth[slot] *= (1.0 + ret)
        taken.append({"Date": row["Date"], "Code": row["Code"], "slot": slot,
                      "ret": ret * 100, "days": days, "hit": hit})
    t = pd.DataFrame(taken)
    if not len(t):
        return {"slots": slots, "target": k, "taken": 0}
    years = (pd.Timestamp(sig["Date"].max()) - pd.Timestamp(sig["Date"].min())).days / 365.25
    total = float(np.mean(wealth))
    return {"slots": slots, "target": k, "taken": len(t), "skipped": skipped,
            "rate": len(t) / (len(t) + skipped) * 100,
            "per_year": len(t) / years, "mean": t["ret"].mean(),
            "win": (t["ret"] > 0).mean() * 100, "hit": t["hit"].mean() * 100,
            "days": t["days"].mean(), "wealth": total,
            "cagr": (total ** (1 / years) - 1) * 100,
            "worst": t["ret"].min(), "years": years, "trades": t}


def main() -> int:
    d = prepare()
    paths = forward_paths()
    d = attach(d, paths)
    d = d[d["entry"].notna() & d["r"].notna()]
    for c in ("lgbm", "xgb", "cat"):
        d[f"p_{c}"] = d[f"s_{c}"].rank(pct=True)
    d["p_min"] = d[["p_lgbm", "p_xgb", "p_cat"]].min(axis=1)
    cal = calendar()
    log(f"対象 {len(d):,}件 / {d['Date'].min().date()}〜{d['Date'].max().date()}")

    rows = []
    print(f"\n{'条件':<34}{'枠':>4}{'出口':>7}{'取れた':>7}{'見送り':>7}{'消化率':>7}"
          f"{'年間':>6}{'平均':>8}{'勝率':>6}{'到達':>6}{'保有日':>7}{'資産倍率':>9}{'年率':>8}")
    for mb, tk in ((20, 2), (20, 1), (26, 2), (26, 1), (10, 2)):
        sig = signals(d, mb, tk)
        for slots in (1, 2, 3):
            for tgt in (20.0, 10.0):
                s = simulate(sig, cal, slots, tgt)
                if not s.get("taken"):
                    continue
                rows.append({k: v for k, v in s.items() if k != "trades"})
                print(f"  {f'発火{mb}件以上・上位{tk}件':<32}{slots:>4}{'+'+str(int(tgt))+'%':>7}"
                      f"{s['taken']:>7}{s['skipped']:>7}{s['rate']:>6.0f}%{s['per_year']:>6.1f}"
                      f"{s['mean']:>+7.2f}%{s['win']:>5.0f}%{s['hit']:>5.0f}%{s['days']:>7.1f}"
                      f"{s['wealth']:>9.2f}{s['cagr']:>+7.1f}%")
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e33_portfolio.csv"), index=False)

    # 本命の設定で、取引の明細と年ごと
    sig = signals(d, 20, 2)
    s = simulate(sig, cal, 2, 20.0)
    t = s["trades"].copy()
    t["year"] = pd.to_datetime(t["Date"]).dt.year
    print(f"\n=== 本命（発火20件以上・上位2件・枠2・+20%利確）の年ごと ===")
    print(f"  {'年':<6}{'取引':>6}{'平均':>9}{'勝率':>7}{'+20%到達':>9}{'年の倍率':>9}")
    for y, g in t.groupby("year"):
        w = float(np.prod(1 + g["ret"] / 100) ** (1 / max(1, g["slot"].nunique())))
        print(f"  {y:<6}{len(g):>6}{g['ret'].mean():>+8.2f}%{(g['ret']>0).mean()*100:>6.0f}%"
              f"{g['hit'].mean()*100:>8.0f}%{w:>9.3f}")
    print(f"\n  期間 {s['years']:.1f}年 / 取引 {s['taken']}件（見送り {s['skipped']}件）"
          f" / 資産 {s['wealth']:.2f}倍 / 年率 {s['cagr']:+.1f}%")
    print(f"  ※ 手数料・税・スリッページは含まない。1枠あたり資金の1/2を投じる前提")
    s["trades"].to_csv(os.path.join(OOF_DIR, "e33_trades.csv"), index=False)
    log(f"記録: {OOF_DIR}/e33_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
