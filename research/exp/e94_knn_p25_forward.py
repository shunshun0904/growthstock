#!/usr/bin/env python3
"""
実験94: 実験93 の後工程（同じ日の候補から、近傍の収益の下側25%点で1件選ぶ）を、決めたあとの判断日だけで確かめ直す
（前向きの確かめ直し。実験69 と同じ考え方。本番は変えない）。

運用者の選択（2026-10-15、AskUserQuestion）: 前向きの記録に進む / 記録の形 = 判定の日と条件を先に固定。

なぜ要るか
--------
実験93（§53）は先に決めた判定を通ったが、(a) 実験91 と同じ期間の out-of-fold で、分布の物差しは運用者の指摘を受けて決め直した
もの、(b) 日の復元抽出の25%点の差が 0 をまたぐ、(c) 通ったのは25%点・近い50件の形だけ、の3点があった。決めた日より後の判断日
だけで同じ物差しの判定を回せば、見たデータで決めたぶんの楽観は入らない。

何を先に決めたか（結果を見てから変えない。変えるなら新しい日付で決め直し、別の確かめ直しにする）
--------
- 行・候補・判断日・分布・距離・近傍の母集団・見積もり・選び方・判定の3条件: 実験93 と同じ台本（e93 の関数をそのまま呼ぶ）。
  近傍の母集団は、判断日より前に結果が出そろった行なので、決めた日より前の行も母集団には入る（選ぶ側の情報であって、
  判定の対象ではない）
- 判定に使う判断日: 選定日が FIRST_DAY（2026-10-16）以降で、その日の規則の行がすべて収益と2つの分布を出せる日だけ
  （1件でも欠ける日は数えない。欠けた候補を外すと選び方が変わるため）
- 判定してよい日: JUDGE_FROM（2027-10-15）以降。途中経過は INTERIM（2027-04-15）に一度出す（判定に使わない）
- 判断日が MIN_DAYS（20日）に満たなければ「判定できない」。EXTEND_TO（2028-04-15）まで一度だけ延ばし、それでも足りなければ
  打ち切り（この線は閉じる）
- 判定: 実験93 と同じ3条件（Q の25%点 ≥ M の25%点 / Q の25%点が無作為の25%点の分布で片側 p < 0.05 / 10・50・75%点が
  無作為の中央以上）がそろえば「再現」。そろわなければ「再現しない」
- 「再現」でも本番の並べ方には入れない。入れる案を運用者に出すだけ。「再現しない」ならこの線は閉じる
- 乱数の種・近傍の数・母集団の線・分布の作りは実験93 の既定のまま（変えたらやり直し）

  python3 research/exp/e94_knn_p25_forward.py                      # 今日の日付で（JUDGE_FROM より前なら途中経過）
  python3 research/exp/e94_knn_p25_forward.py --today 2027-10-15   # 判定の日を指定（記録用）

公開ログに出すのは件数・分位点・割合・判定だけ（銘柄・日付ごとの値は出さない）。

空回し（2026-10-15、run 38735940574、61b4457）: 2026-10-16 以降の判断日 0日 → 途中経過。全期間の件数は実験93 と同じ
（559件・96日）。判定に使うデータは何も見ていない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import models as M  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e91_wasserstein_knn_pick as E91  # noqa: E402
import e93_knn_quantile_pick as E93  # noqa: E402

DECIDED = "2026-10-15"               # 決めた日
FIRST_DAY = "2026-10-16"             # 判定に使う判断日（選定日）はこれ以降
INTERIM = "2027-04-15"               # 途中経過（判定に使わない）
JUDGE_FROM = "2027-10-15"            # 判定してよい日
EXTEND_TO = "2028-04-15"             # 判断日が足りなければ一度だけ延ばす
MIN_DAYS = 20                        # 判定に要る判断日の数
RULE = {"key": f"p{E93.MAIN_P}_{E93.K_MAIN}", "ascending": False}   # 実験93 の主: 近傍50件の25%点の最上位
CONDS = ("not_below_M", "p_random", "no_worse_elsewhere")           # 実験93 の3条件（e93.evaluate の conds のキー）
#: 決めた時点（実験93、run 38733526689）の値。並べて見るだけ
ORIGIN = {"days": 96, "Q_p25": +0.21, "M_p25": -0.90, "R_p25": -1.04, "p_Q": 0.021, "down_M": 0.06, "up_M": 0.69}
LABELS = {"pass": "再現", "fail": "再現しない", "interim": "途中経過（判定ではない）", "short": "判定できない（延ばす）",
          "closed": "判定できない（打ち切り。この線は閉じる）"}
OUT_DIR = E79.OUT_DIR
log = E79.log


# --------------------------------------------------------------------------- #
# 判定に使う判断日
# --------------------------------------------------------------------------- #

def rule_counts(df: pd.DataFrame, split: dict, line: float = E93.LINE) -> pd.Series:
    """日付ごとの規則の行の数（その窓の物差しで 1.5σ の3本の最小が line 以上。実験93 の collect と同じ数え方）。"""
    mins = E91.window_min(df, split)
    w_arr = df["window"].to_numpy()
    dates = pd.to_datetime(df["Date"]).to_numpy(dtype="datetime64[ns]")
    parts = []
    for w, (U, mn) in mins.items():
        rows = U[(w_arr[U] == w) & (mn >= line)]
        parts.append(pd.Series(dates[rows]))
    if not parts:
        return pd.Series(dtype=int)
    return pd.concat(parts, ignore_index=True).value_counts().sort_index()


def forward_days(c: pd.DataFrame, counts: pd.Series, first_day: str = FIRST_DAY) -> tuple[pd.DataFrame, dict]:
    """
    判定に使う判断日だけに絞る: 選定日が first_day 以降で、その日の規則の行がすべて c に入っている（収益と2つの分布が出せる）日。
    戻り値: 絞った表と件数の記録（first_day 以降の候補2件以上の日・そのうち欠けのある日・使う日）。
    """
    first = pd.Timestamp(first_day)
    info = {"first_day": first_day, "days_after": 0, "days_incomplete": 0, "days": 0}
    if c.empty:
        return c, info
    per = c.groupby("Date").size()
    per = per[per.index >= first]
    keep = []
    for d, n in per.items():
        if n < 2:
            continue
        info["days_after"] += 1
        total = int(counts.get(pd.Timestamp(d), 0))
        if total != int(n):
            info["days_incomplete"] += 1
            continue
        keep.append(d)
    info["days"] = len(keep)
    out = c[c["Date"].isin(keep)].reset_index(drop=True)
    return out, info


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #

def status(today, n_days: int) -> str:
    """LABELS のキー。判定してよい日か・判断日が足りるかで決める（pass / fail は judge が決める）。"""
    t = pd.Timestamp(today)
    if t < pd.Timestamp(JUDGE_FROM):
        return "interim"
    if n_days >= MIN_DAYS:
        return "judge"
    return "closed" if t >= pd.Timestamp(EXTEND_TO) else "short"


def judge(c_fwd: pd.DataFrame, today, draws: int = E93.DRAWS, seed: int = E93.SEED) -> dict:
    """絞った判断日で実験93 と同じ評価を回し、status に応じて判定の札を付ける。"""
    n_days = int(c_fwd["Date"].nunique()) if len(c_fwd) else 0
    st = status(today, n_days)
    ev = E93.evaluate(c_fwd, RULE["key"], RULE["ascending"], draws, seed) if n_days else {"days": 0, "effective": False, "conds": {}}
    if st == "judge":
        st = "pass" if all(ev["conds"].get(k, False) for k in CONDS) else "fail"
    return {"today": str(pd.Timestamp(today).date()), "n_days": n_days, "status": st, "label": LABELS[st], "eval": ev}


def report(res: dict, finfo: dict, draws: int = E93.DRAWS) -> None:
    ev = res["eval"]
    print(f"\n実験94: {DECIDED} に決めた後工程（{RULE['key']} の最上位）を、{FIRST_DAY} 以降の判断日だけで確かめ直す")
    print(f"  判定してよいのは {JUDGE_FROM} 以降（途中経過 {INTERIM}・足りなければ {EXTEND_TO} まで延ばす）。判断日は {MIN_DAYS}日以上")
    print(f"  今日 {res['today']}: {FIRST_DAY} 以降の候補2件以上の日 {finfo['days_after']}日、うち規則の行に欠けのある日 "
          f"{finfo['days_incomplete']}日 → 判定に使う判断日 {res['n_days']}日")
    print(f"  決めた時点（実験93、判断日 {ORIGIN['days']}日）: Q の25%点 {ORIGIN['Q_p25']:+.2f} / M {ORIGIN['M_p25']:+.2f} / "
          f"無作為 {ORIGIN['R_p25']:+.2f}・p = {ORIGIN['p_Q']:.3f}・Q 対 M 下側 {ORIGIN['down_M']:.2f} / 上側 {ORIGIN['up_M']:.2f}pt")
    if ev.get("days"):
        print(E93.line(f"Q: 近傍の{E93.MAIN_P}%点の最上位（{ev['days']}日）", ev["Q"]))
        print(E93.line(f"M: 今の並べ方（{ev['days']}日）", ev["M"]))
        print(E93.line(f"無作為（{draws:,}回を合わせた分布）", ev["R"]))
        print(f"  条件: {E93.fmt_cond(ev)}")
        print(f"  参考: {E93.fmt_pair(ev)}")
        print(f"  参考: {E93.fmt_area(ev)}")
    else:
        print("  判定に使える判断日はまだ無い")
    print(f"  結果: {res['label']}")
    if res["status"] in ("pass", "fail"):
        print("  扱い: 「再現」でも本番の並べ方には入れない（入れる案を運用者に出すだけ）。「再現しない」ならこの線は閉じる")
    print("  注意: 手数料・税・すべりは含まない。近傍の母集団には決めた日より前の行も入る（選ぶ側の情報で、判定の対象ではない）")


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #

def _js(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験94: 実験93 の後工程を、決めたあとの判断日だけで確かめ直す")
    ap.add_argument("--today", default=None, help="判定の日（既定は今日）。JUDGE_FROM より前なら途中経過")
    ap.add_argument("--draws", type=int, default=E93.DRAWS)
    args = ap.parse_args(argv)
    today = pd.Timestamp(args.today) if args.today else pd.Timestamp.today().normalize()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    df = E91.attach_vol(df, fr)
    df["score"] = pd.to_numeric(df[f"score_{M.BASELINE}"], errors="coerce")
    del fr
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=250),
                        extra=("C", "AdjVo", "Vo"))
    df, chk = E83.attach_exits(df, bars, targets=(E93.TARGET,))
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行")
    dist = E91.event_distributions(bars, df[["Code", "Date"]])
    del bars
    split = E79.window_split(df, folds)
    starts = {f.index: pd.Timestamp(f.test_start) for f in folds}
    c, info = E93.collect(df, dist, split, starts)
    log(f"今の規則の行 {info['rule_rows']}件・候補2件以上の判断日 {info['days']}日（全期間。実験93 は 559件・96日）")
    c_fwd, finfo = forward_days(c, rule_counts(df, split))
    res = judge(c_fwd, today, draws=args.draws)
    report(res, finfo, draws=args.draws)
    out = {"decided": DECIDED, "first_day": FIRST_DAY, "judge_from": JUDGE_FROM, "extend_to": EXTEND_TO, "min_days": MIN_DAYS,
           "rule": RULE, "conds": CONDS, "origin": ORIGIN, "forward": finfo, "result": res, "all_days": info["days"]}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e94_forward.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=_js)
    log(f"記録: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
