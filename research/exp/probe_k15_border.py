#!/usr/bin/env python3
"""
「1.5σで際どい」の数字: 1.5σ の3本（lgbm_k15・xgb_k15・cat_k15）で見た際どい候補の、形 × 一番低いモデルごとの成績。

運用者の依頼（2026-10-12）「1.5σで際どいにしてほしいです」。画面の「際どい」の印（src/lib/strategy.js の borderNote）は
1.2σ の3本と実験70 の数字のまま。今の規則（1.5σ の3本がすべて95以上）と同じ3本で形を決め、数字を測り直す
（見るだけ。選定の規則は変えない）。

測り方（先に決めた。結果を見てから分け方・線を足さない）
------------------------------------------------
  行: 前向き（主。テスト窓 3〜11 の候補の全部。実験70 と同じく発火の数を問わない）。参考に、発火8件以上の日に限った形と、
      画面の物差し（全期間の百分位）
  形（実験70・画面の borderShape と同じ。線 95・下 85。百分位の丸めは画面と同じ小数1桁）:
      all95（3本とも95以上）/ two95_hi・two95_mid・two95_lo（2本が95以上で、残り1本が 90〜95・85〜90・85未満）/
      min90・min85（95以上は1本以下で、最小が 90〜95・85〜90）/ other（それ以外）
  一番低いモデル: 3本のうち百分位が一番低いもの（同点は lgbm_k15 → xgb_k15 → cat_k15 の順で先。画面の laggard と同じ）
  物差し: tp10c（+10% の指値。届かなければ20営業日目の終値）。SE は銘柄ごとにまとめる（実験70 の cluster_se と同じ）
  画面に出す数字: 線の上（all95）、2本95以上（hi・mid・lo を合わせる）× 一番低いモデル3通り、
      95以上は1本以下・最小 85〜95（min90・min85 を合わせる）の合計と × 一番低いモデル3通り
  赤の注意を出す組（先に決めた線）: 件数が WARN_MIN_N（20）件以上で、平均が WARN_RET（0%）以下（空き枠と同じか悪い）。
      20件未満の組は「件数が少ない」として赤にしない。画面の文言はこの表の前向きの数字を使う
  参考: 1.2σ の3本で同じ形（今の印の物差し。実験70 は持ち切り ret_o1_20・窓2以降の百分位なので数字はずれる）

公開ログには集計だけ（件数・平均・割合）。銘柄・日付ごとの値は出さない。

    exp=probe_k15_border.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import e70_borderline as E70  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e87_monthly_return_search as E87  # noqa: E402

MODELS = E79.MODELS
LINE = 95.0                      # 画面の BORDER.line
NEAR_LO = 85.0                   # 画面の BORDER.lo
#: 形を決める3本（記号 → モデル）。K15 が主、T3 は参考
GROUPS = {"K15": E82.T15, "T3": E82.TREES}
GROUP_JA = {"K15": "1.5σ の3本", "T3": "1.2σ の3本（参考）"}
SHAPES = (("all95", "3本とも95以上（線の上）"), ("two95_hi", "2本95以上・残り 90〜95"), ("two95_mid", "2本95以上・残り 85〜90"),
          ("two95_lo", "2本95以上・残り 85未満"), ("min90", "95以上は1本以下・最小 90〜95"),
          ("min85", "95以上は1本以下・最小 85〜90"), ("other", "それ以外"))
BORDER = ("two95_hi", "two95_mid", "two95_lo", "min90", "min85")
FAMILIES = (("two95", "2本95以上", ("two95_hi", "two95_mid", "two95_lo")), ("min", "95以上は1本以下・最小 85〜95", ("min90", "min85")))
WARN_MIN_N = 20
WARN_RET = 0.0
OUT_DIR = E79.OUT_DIR

log = E79.log


def shape_of(V, line: float = LINE, lo: float = NEAR_LO) -> np.ndarray:
    """(行 × 3本) の丸めた百分位から形（SHAPES の鍵）。1つでも NaN なら None。画面の borderShape と同じ分け方。"""
    V = np.asarray(V, dtype=float)
    ok = np.isfinite(V).all(axis=1)
    n95 = (V >= line).sum(axis=1)
    mn = np.where(ok, np.min(np.where(np.isfinite(V), V, np.inf), axis=1), np.nan)
    out = np.where(n95 == 3, "all95",
                   np.where(n95 == 2, np.where(mn >= 90, "two95_hi", np.where(mn >= lo, "two95_mid", "two95_lo")),
                            np.where(mn >= 90, "min90", np.where(mn >= lo, "min85", "other"))))
    return np.where(ok, out, None).astype(object)


def laggard_of(V, models) -> np.ndarray:
    """百分位が一番低いモデル（同点は models の先）。1つでも NaN なら None。"""
    V = np.asarray(V, dtype=float)
    ok = np.isfinite(V).all(axis=1)
    idx = np.argmin(np.where(np.isfinite(V), V, np.inf), axis=1)
    return np.array([models[i] if k else None for i, k in zip(idx, ok)], dtype=object)


def add_shapes(ft: pd.DataFrame, parts: list, models=MODELS, groups=GROUPS) -> pd.DataFrame:
    """組ごとに形（shape_<組>）と一番低いモデル（lag_<組>）を足す。parts の順は E87.build と同じ（行の順がそろう）。"""
    out = ft.copy()
    for code, grp in groups.items():
        idx = [list(models).index(a) for a in grp]
        V = np.concatenate([E82.rounded(np.asarray(P, dtype=float)[:, idx]) for _, P in parts])
        out[f"shape_{code}"] = shape_of(V)
        out[f"lag_{code}"] = laggard_of(V, grp)
    return out


def stats(g: pd.DataFrame) -> dict:
    r = pd.to_numeric(g["tp10c"], errors="coerce")
    ok = r.notna().to_numpy()
    x = r.to_numpy(dtype=float)[ok]
    n = int(len(x))
    nan = float("nan")
    if not n:
        return {"n": 0, "codes": 0, "mean": nan, "se": nan, "win": nan, "hit": nan, "bad10": nan}
    hit = g["hit10"].to_numpy(dtype=bool)[ok]
    codes = g["Code"].to_numpy()[ok]
    return {"n": n, "codes": int(pd.Series(codes).nunique()),
            "mean": float(x.mean()), "se": float(E70.cluster_se(x, codes)),
            "win": float((x > 0).mean() * 100), "hit": float(hit.mean() * 100), "bad10": float((x < -10).mean() * 100)}


def warn_of(s: dict, min_n: int = WARN_MIN_N, ret: float = WARN_RET) -> bool:
    """先に決めた赤の注意の線: 件数 min_n 以上で、平均が ret 以下。"""
    return bool(s.get("n", 0) >= min_n and np.isfinite(s.get("mean", np.nan)) and s["mean"] <= ret)


def line(name: str, s: dict, width: int = 36) -> str:
    if not s["n"]:
        return f"  {LT._l(name, width)}    0件"
    few = "（件数が少ない）" if s["n"] < WARN_MIN_N else ""
    return (f"  {LT._l(name, width)}{s['n']:>5}件（{s['codes']:>3}銘柄）  +10% {s['mean']:+6.2f}% ± {s['se']:.2f}  勝率 "
            f"{s['win']:4.1f}%  到達 {s['hit']:4.1f}%  −10%割れ {s['bad10']:4.1f}%{few}")


def family_table(ft: pd.DataFrame, code: str) -> dict:
    """画面に出す形の数字: 線の上・2本95以上 × 一番低いモデル・95以上は1本以下の合計と × 一番低いモデル。"""
    grp = GROUPS[code]
    sh, lag = ft[f"shape_{code}"], ft[f"lag_{code}"]
    out = {"line": stats(ft[sh == "all95"])}
    for fam, _, keys in FAMILIES:
        m = sh.isin(keys)
        out[fam] = {"all": stats(ft[m])}
        for a in grp:
            out[fam][a] = stats(ft[m & (lag == a)])
            out[fam][a]["warn"] = warn_of(out[fam][a])
    return out


def print_family(t: dict, code: str) -> None:
    grp = GROUPS[code]
    print(line("3本とも95以上（線の上）", t["line"]))
    for fam, name, _ in FAMILIES:
        print(line(f"{name}（合計）", t[fam]["all"]))
        for a in grp:
            mark = "  ← 赤の注意" if t[fam][a]["warn"] else ""
            print(line(f"  一番低い {a}", t[fam][a]) + mark)
        for a in grp:                                         # 参考: その一番低いモデル − 同じ形のほか（SE は独立とみなす）
            rest = [b for b in grp if b != a]
            x, y = t[fam][a], stats_of_rest(t, fam, rest)
            if x["n"] and y["n"]:
                d = x["mean"] - y["mean"]
                se = float(np.sqrt(x["se"] ** 2 + y["se"] ** 2))
                z = d / se if np.isfinite(se) and se > 0 else float("nan")
                print(f"    参考 一番低い {a} − ほか2本: {d:+.2f}pt ± {se:.2f}（z {z:+.2f}）")


def stats_of_rest(t: dict, fam: str, rest: list) -> dict:
    """同じ形のほかの一番低いモデルの組を、件数で重みを付けて合わせる（平均。SE は独立とみなして合成）。"""
    xs = [t[fam][b] for b in rest if t[fam][b]["n"]]
    n = sum(s["n"] for s in xs)
    if not n:
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    mean = sum(s["n"] * s["mean"] for s in xs) / n
    se = float(np.sqrt(sum((s["n"] / n) ** 2 * (s["se"] ** 2 if np.isfinite(s["se"]) else 0.0) for s in xs)))
    return {"n": n, "mean": mean, "se": se}


def shape_block(ft: pd.DataFrame, code: str, title: str) -> dict:
    print(f"\n■ {title}")
    res = {}
    sh = ft[f"shape_{code}"]
    for k, name in SHAPES:
        res[k] = stats(ft[sh == k])
        print(line(name, res[k]))
    res["border"] = stats(ft[sh.isin(BORDER)])
    print(line("（際どい候補の合計）", res["border"]))
    res["all"] = stats(ft[sh.notna()])
    print(line("（母集団）", res["all"]))
    return res


def js_lines(t: dict, code: str) -> list:
    """strategy.js の BORDER_STATS に写す形（件数・平均を小数2桁）。"""
    def one(s):
        return f"{{ n: {s['n']}, ret: {s['mean']:.2f} }}" if s["n"] else "{ n: 0, ret: null }"
    grp = GROUPS[code]
    out = [f"  line: {one(t['line'])},"]
    for fam, _, _ in FAMILIES:
        parts = [f"all: {one(t[fam]['all'])}"] + [f"{a}: {one(t[fam][a])}" for a in grp]
        out.append(f"  {fam}: {{ " + ", ".join(parts) + " },")
    return out


def report(ft: pd.DataFrame, ff: pd.DataFrame | None = None) -> dict:
    summary: dict = {}
    k = "K15"
    summary["shapes"] = shape_block(ft, k, f"1. 形ごとの成績（{GROUP_JA[k]}・前向き・発火を問わない）")
    print(f"\n■ 2. 形 × 一番低いモデル（{GROUP_JA[k]}・前向き・発火を問わない）")
    sh, lag = ft[f"shape_{k}"], ft[f"lag_{k}"]
    summary["shape_lag"] = {}
    for key, name in SHAPES:
        if key not in BORDER:
            continue
        for a in GROUPS[k]:
            s = stats(ft[(sh == key) & (lag == a)])
            summary["shape_lag"][f"{key}_{a}"] = s
            print(line(f"{name}・一番低い {a}", s, 44))
    print(f"\n■ 3. 画面に出す数字（{GROUP_JA[k]}・前向き・発火を問わない。赤の注意 = {WARN_MIN_N}件以上で平均 {WARN_RET:.0f}% 以下）")
    t = family_table(ft, k)
    print_family(t, k)
    summary["screen"] = t
    print("  strategy.js に写す形:")
    for s in js_lines(t, k):
        print("  " + s)
    hot = ft[ft["n_break"].to_numpy() >= LT.SKIP_BREAKS]
    print(f"\n■ 4. 参考: 発火{LT.SKIP_BREAKS}件以上の日に限る（{GROUP_JA[k]}・前向き）")
    summary["hot"] = family_table(hot, k)
    print_family(summary["hot"], k)
    if ff is not None and len(ff):
        print(f"\n■ 5. 参考: 画面の物差し（全期間の百分位。{GROUP_JA[k]}・発火を問わない）")
        summary["full"] = family_table(ff, k)
        print_family(summary["full"], k)
    print(f"\n■ 6. 参考: {GROUP_JA['T3']}で同じ形（前向き・発火を問わない。今の画面の印の物差し）")
    summary["t3"] = family_table(ft, "T3")
    print_family(summary["t3"], "T3")
    print("\n■ まとめ")
    reds = [f"{fam}・一番低い {a}" for fam, _, _ in FAMILIES for a in GROUPS[k] if t[fam][a]["warn"]]
    print(f"  {GROUP_JA[k]}（前向き）: 線の上 {t['line']['n']}件 {t['line']['mean']:+.2f}%・際どい候補の合計 "
          f"{summary['shapes']['border']['n']}件 {summary['shapes']['border']['mean']:+.2f}%。"
          f"赤の注意の組: {'、'.join(reds) if reds else 'なし'}")
    print("  注意: 同じ期間の out-of-fold だけ。手数料・税・すべりは含まない")
    summary["reds"] = reds
    return summary


def main(argv=None) -> int:
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = E83.attach_exits(df, bars, targets=E87.TARGETS)
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・到達の順序の食い違い {chk['order_violations']}件")
    split = E79.window_split(df, folds)
    parts_f = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
    parts_s = [(np.arange(len(df)), E79.pct_matrix(df, MODELS))]
    ft = add_shapes(E87.build(df, parts_f), parts_f)
    ff = add_shapes(E87.build(df, parts_s), parts_s)
    k = (ft["shape_K15"] == "all95").to_numpy()
    print(f"■ 0. 確かめ: 前向き {len(ft):,}行（テスト窓 {len(split)}つ）。1.5σ の3本 95以上 {int(k.sum())}件"
          f"（probe_k15_baseline は 559件）・min_K15 で数えても {int((ft['min_K15'].to_numpy(dtype=float) >= LINE).sum())}件")
    summary = report(ft, ff)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "probe_k15_border.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"記録: {path} / 全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
