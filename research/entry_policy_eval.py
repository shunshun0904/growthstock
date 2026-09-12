#!/usr/bin/env python3
"""
エントリー方策の期待値比較（Phase 1・日足のみ）を実行して、報告を書く。

  python3 research/entry_policy_eval.py                 # research/_data の生データを使う
  python3 research/entry_policy_eval.py --n-boot 500    # 早く回す

分足は使わない。使えるのは日足10年ぶんで、これは今の契約のまま回せる。
設計は docs/ENTRY_TIMING_DESIGN.md。出力は docs/ENTRY_POLICY_PHASE1.md。

何が決まるか
------------
  1. 翌日の寄りで成行するのと、押し目を待つのと、どちらが良いか
  2. ギャップが何%を超えたら見送るべきか
  3. 分足アドオン（月5,500円）を契約する価値があるか（設計 §12）

3 は「楽観的な約定仮定と悲観的な約定仮定で結論の符号が反転するか」で決まる。
反転するなら、決め手は日中の到達順序であり日足では分からない。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_dataset as B  # noqa: E402
import entry_policy as EP  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
MODEL_DIR = os.path.join(HERE, "model")
OUT_JSON = os.path.join(HERE, "entry_policy_phase1.json")
OUT_MD = os.path.join(ROOT, "docs", "ENTRY_POLICY_PHASE1.md")

#: 合格ライン。**先に決める**（設計 §11）。結果を見てから動かさない。
PASS_DIFF = 0.3     # 基準（寄り成行）に対する1件あたりの差（pt）
#: 感度を見る手数料（往復・%）と、悲観側の余裕（bp）
FEE_GRID = (0.0, 0.05, 0.1)
SLACK_GRID = (0.0, 10.0, 25.0)

#: 上位だけ並べると「待って逃したときの損」が表から消える。
#: 成績に関係なく必ず出す代表的な方策を決めておく。
HEADLINE_POLICIES = (
    "寄り成行",
    "寄り成行 / ギャップ>3%は見送り",
    "寄り成行 / ギャップ>5%は見送り",
    "寄り成行 / ギャップ>10%は見送り",
    "押し目 -1% / 1日 / 始値基準 / 未約定は見送り",
    "押し目 -2% / 1日 / 始値基準 / 未約定は見送り",
    "押し目 -3% / 1日 / 始値基準 / 未約定は見送り",
    "押し目 -2% / 3日 / 始値基準 / 未約定は見送り",
    "押し目 -2% / 5日 / 始値基準 / 未約定は見送り",
    "押し目 -2% / 1日 / 始値基準 / 期限で成行",
    "押し目 -3% / 3日 / 始値基準 / 期限で成行",
)


# --------------------------------------------------------------------------- #
# データ
# --------------------------------------------------------------------------- #

def load_events(data_dir: str, min_trading_value: Optional[float],
                ) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    bars = B.load_parts("bars", data_dir)
    n_bars = len(bars)
    print("[panel] 株価系の指標を算出")
    panel = B.price_panel(bars)
    # price_panel は始値を作らない（既存モデルが使っていないため）。
    # 分割調整後を優先するのは他の四本値と同じ理由（調整前だと分割日に
    # 偽のギャップが出る）。
    if "AdjO" in panel.columns:
        panel["open"] = panel["AdjO"].fillna(panel["O"])
    else:
        panel["open"] = panel["O"]
    print("[panel] 78週高値の更新日を判定")
    panel = B.mark_new_highs(panel)
    print("[events] 翌営業日以降の値動きと手仕舞い価格を並べる")
    ev, funnel = EP.build_events(panel, min_trading_value=min_trading_value)
    for k, v in funnel.items():
        print(f"  {k}: {v:,}件")
    return ev, funnel, n_bars


def attach_scores(ev: pd.DataFrame, model_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """
    out-of-fold スコアがあれば結合する。実運用で買うのは上位だけなので、
    全件の平均だけ見ても運用の答えにならない。

    同時に、手仕舞い価格の計算が既存の ref_end と一致するかを突き合わせる。
    ここがずれていたら、以降の数字は全部信用できない。
    """
    path = os.path.join(model_dir, "oof.parquet")
    info: Dict = {"available": False, "path": path}
    if not os.path.exists(path):
        print(f"[score] {path} が無いので、全件のみで評価する")
        return ev, info

    oof = pd.read_parquet(path)
    oof["Date"] = pd.to_datetime(oof["Date"])
    ev = ev.copy()
    ev["Date"] = pd.to_datetime(ev["Date"])
    ev["Code"] = ev["Code"].astype(str)
    oof["Code"] = oof["Code"].astype(str)
    merged = ev.merge(oof[["Code", "Date", "score", "ref_end"]],
                      on=["Code", "Date"], how="left")
    n_hit = int(merged["score"].notna().sum())
    info.update({"available": n_hit > 0, "oof_rows": len(oof), "matched": n_hit})
    print(f"[score] out-of-fold {len(oof):,}件のうち {n_hit:,}件が突き合った")

    both = merged[merged["ref_end"].notna()]
    if len(both):
        diff = (both["ref_end_pct"] - both["ref_end"] * 100.0).abs()
        info["ref_end_check"] = {
            "n": int(len(both)),
            "max_abs_diff_pt": round(float(diff.max()), 6),
            "median_abs_diff_pt": round(float(diff.median()), 6),
        }
        print(f"[check] 手仕舞い価格と既存 ref_end の差: "
              f"中央値 {diff.median():.6f}pt / 最大 {diff.max():.6f}pt")
    return merged, info


def subsets(ev: pd.DataFrame, has_score: bool) -> List[Tuple[str, pd.DataFrame]]:
    """評価する部分集合。運用で実際に買う形に近いものを併記する。"""
    out = [("全件", ev)]
    if not has_score:
        return out
    scored = ev[ev["score"].notna()]
    if len(scored) < 200:
        return out
    thr = scored["score"].quantile(0.90)
    out.append((f"スコア上位10%（score>={thr:.4f}）", scored[scored["score"] >= thr]))
    top1 = scored.loc[scored.groupby("Date")["score"].idxmax()]
    out.append(("その日の最上位1件", top1))
    return out


# --------------------------------------------------------------------------- #
# 評価
# --------------------------------------------------------------------------- #

def split(ev: pd.DataFrame, test_start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.to_datetime(ev["Date"])
    ts = pd.Timestamp(test_start)
    return ev[d < ts], ev[d >= ts]


def pick_best(rows: List[Dict], kind: str) -> Optional[Dict]:
    """訓練期間の平均で最良の方策を選ぶ。テストの結果は見ない（設計 §10）。"""
    cand = [r for r in rows if r["kind"] == kind and r["policy"] != "寄り成行"]
    cand = [r for r in cand if np.isfinite(r["mean"])]
    if not cand:
        return None
    return max(cand, key=lambda r: r["mean"])


def evaluate(ev: pd.DataFrame, policies: Sequence[EP.Policy], costs: EP.Costs,
             test_start: str, n_boot: int, seed: int) -> Dict:
    tr, te = split(ev, test_start)
    out: Dict = {"n_all": len(ev), "n_train": len(tr), "n_test": len(te),
                 "test_start": test_start}
    for tag, sub in (("train", tr), ("test", te)):
        for opt, label in ((True, "optimistic"), (False, "pessimistic")):
            out[f"{tag}_{label}"] = EP.compare(sub, policies, costs, opt,
                                               n_boot=n_boot, seed=seed)
    # 訓練で選び、テストで測る
    sel: Dict = {}
    for label in ("optimistic", "pessimistic"):
        tr_rows = out[f"train_{label}"]
        te_rows = {r["policy"]: r for r in out[f"test_{label}"]}
        for kind, jp in (("dip", "押し目"), ("open", "見送り付き寄り成行")):
            best = pick_best(tr_rows, kind)
            if best is None:
                continue
            sel[f"{label}_{kind}"] = {
                "kind": kind, "kind_ja": jp, "policy": best["policy"],
                "train": {k: best[k] for k in ("mean", "fill_rate", "missed_rate")},
                "train_vs_base": best["vs_base"],
                "test": {k: te_rows[best["policy"]][k]
                         for k in ("mean", "fill_rate", "missed_rate",
                                   "median", "p05", "entry_improve")}
                if best["policy"] in te_rows else None,
                "test_vs_base": (te_rows[best["policy"]]["vs_base"]
                                 if best["policy"] in te_rows else None),
            }
    out["selected"] = sel
    out["flip"] = EP.flip_check(te, policies, costs)
    # 「寄り成行で買ったら実際どうなるか」の散らばり。
    # 平均だけ見て良し悪しを決めると、負ける年があることが見えない。
    base = policies[0]   # compare() の基準と同じ（寄り成行）
    out["baseline_spread"] = {
        tag: (EP.summarize(sub, base, costs, False) if len(sub) else None)
        for tag, sub in (("all", ev), ("train", tr), ("test", te))
    }
    return out


def by_year(ev: pd.DataFrame, policies: Sequence[EP.Policy], names: Sequence[str],
            costs: EP.Costs, optimistic: bool) -> List[Dict]:
    """局面依存を見る。1局面だけで勝っている方策を採らないため（設計 §11）。"""
    want = [p for p in policies if p.name in set(names)]
    years = sorted(pd.to_datetime(ev["Date"]).dt.year.unique())
    out = []
    for y in years:
        sub = ev[pd.to_datetime(ev["Date"]).dt.year == y]
        if len(sub) < 30:
            continue
        row = {"year": int(y), "n": len(sub)}
        for p in want:
            row[p.name] = round(float(
                EP.returns_of(sub, p, costs, optimistic).mean()), 3)
        out.append(row)
    return out


def sensitivity(ev: pd.DataFrame, policies: Sequence[EP.Policy],
                names: Sequence[str], test_start: str) -> List[Dict]:
    """手数料と余裕を振る。入れ方ひとつで結論が変わるなら、それは結論ではない。"""
    _, te = split(ev, test_start)
    want = [p for p in policies if p.name in set(names)]
    base = EP.Policy("寄り成行", "open")
    out = []
    for fee in FEE_GRID:
        for slack in SLACK_GRID:
            costs = EP.Costs(fee_pct=fee, slack_bp=slack)
            for opt in (True, False):
                b = EP.returns_of(te, base, costs, opt).mean() if len(te) else np.nan
                row = {"fee_pct": fee, "slack_bp": slack,
                       "fill": "楽観" if opt else "悲観",
                       "寄り成行": round(float(b), 3)}
                for p in want:
                    v = EP.returns_of(te, p, costs, opt).mean() if len(te) else np.nan
                    row[p.name] = round(float(v - b), 3)
                out.append(row)
    return out


def verdict(res: Dict) -> Dict:
    """
    設計 §11 の合格ラインと §12 の契約判断を、書いてあるとおりに当てる。
    文章で誤魔化さず、数字から機械的に決める。
    """
    v: Dict = {"pass_diff": PASS_DIFF}
    ok = {}
    for label in ("optimistic", "pessimistic"):
        s = res["selected"].get(f"{label}_dip")
        if not s or not s.get("test_vs_base"):
            ok[label] = None
            continue
        d = s["test_vs_base"]
        ok[label] = bool(d["diff"] >= PASS_DIFF and np.isfinite(d["lo"])
                         and d["lo"] > 0)
    v["dip_passes"] = ok

    flips = [f for f in res.get("flip", []) if f["flips"]]
    v["n_flipping"] = len(flips)
    v["widest_flip"] = flips[0] if flips else None

    if ok.get("optimistic") and ok.get("pessimistic"):
        v["dip"] = "採用できる。楽観・悲観のどちらでも合格ラインを超えた"
        v["minute_addon"] = "当面は契約しない。日足だけで方策が立つ"
    elif not ok.get("optimistic") and not ok.get("pessimistic"):
        v["dip"] = "採用しない。どちらの約定仮定でも合格ラインに届かない"
        v["minute_addon"] = ("契約しない。押し目待ちが効かないので、"
                             "日中の解像度を上げても効かない")
    else:
        v["dip"] = "保留。約定の仮定しだいで結論が変わる"
        v["minute_addon"] = ("契約する価値がある。決め手は日中の到達順序で、"
                             "それは日足では分からない")
    return v


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def _ci(d: Optional[Dict]) -> str:
    if not d or not np.isfinite(d.get("lo", np.nan)):
        return "—"
    return f"{d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def _table(header: List[str], rows: List[List]) -> List[str]:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def top_rows(rows: List[Dict], n: int = 8) -> List[Dict]:
    return sorted([r for r in rows if np.isfinite(r["mean"])],
                  key=lambda r: -r["mean"])[:n]


def write_md(report: Dict, path: str) -> None:
    L: List[str] = []
    A = L.append
    A("# エントリー方策の期待値比較（Phase 1・日足のみ）")
    A("")
    A("`research/entry_policy_eval.py` の出力。設計は "
      "[docs/ENTRY_TIMING_DESIGN.md](ENTRY_TIMING_DESIGN.md)。")
    A("")
    A(f"- 実行日時: {report['ranAt']}")
    A(f"- 日次バー: {report['bars_rows']:,}行")
    A(f"- 手数料: 往復 {report['costs']['fee_pct']}% / "
      f"悲観側の余裕: {report['costs']['slack_bp']}bp")
    A(f"- 訓練/テストの境目: {report['test_start']}")
    A(f"- ブートストラップ: 日単位・{report['n_boot']:,}回")
    A("")
    A("手仕舞いは全方策で共通（基準日から60営業日後の5日平均終値）。"
      "未約定・見送りは 0（建玉が無い）として1件あたりで平均する。")
    A("")

    ck = report.get("score_info", {}).get("ref_end_check")
    if ck:
        A(f"> 手仕舞い価格は既存の `ref_end` と突き合わせてある"
          f"（{ck['n']:,}件・最大差 {ck['max_abs_diff_pt']:.6f}pt）。"
          "別々に計算した2つが一致しているので、以降の数字はここで壊れていない。")
        A("")

    # --- 判定 --- #
    A("## 判定")
    A("")
    A(f"合格ライン（**先に決めてある**・設計 §11）: 基準に対して "
      f"**+{PASS_DIFF}pt 以上**、かつ95%区間が0を跨がない、"
      "かつ楽観・悲観で符号が変わらない。")
    A("")
    body = []
    for name, sec in report["subsets"].items():
        v = sec["verdict"]
        ok = v["dip_passes"]
        body.append([
            name, f"{sec['n_test']:,}",
            "○" if ok.get("optimistic") else "×",
            "○" if ok.get("pessimistic") else "×",
            f"{v['n_flipping']}件", v["dip"]])
    A("\n".join(_table(
        ["部分集合", "テスト件数", "楽観で合格", "悲観で合格",
         "符号が反転した方策", "押し目待ちの扱い"], body)))
    A("")
    A(f"**分足アドオン（月5,500円）**: "
      f"{report['subsets']['全件']['verdict']['minute_addon']}")
    A("")

    # --- 代表的な方策 --- #
    A("## 代表的な方策（テスト期間・全件・悲観側の約定）")
    A("")
    A("成績の上位だけを並べると、**待って逃したときの損**が表から消える。"
      "成績に関係なく出す方策を先に決めてある。")
    A("")
    rows = {r["policy"]: r for r in report["subsets"]["全件"]["test_pessimistic"]}
    body = []
    for name in HEADLINE_POLICIES:
        r = rows.get(name)
        if not r:
            continue
        body.append([
            "**" + name + "（基準）**" if name == "寄り成行" else name,
            f"{r['mean']:+.2f}", f"{r['fill_rate']:.0f}%",
            f"{r['missed_rate']:.0f}%",
            f"{r['mean_if_filled']:+.2f}"
            if np.isfinite(r["mean_if_filled"]) else "—",
            "—" if name == "寄り成行" else _ci(r["vs_base"]),
        ])
    A("\n".join(_table(
        ["方策", "1件あたり平均", "約定率", "取り逃し", "約定した分の平均",
         "基準との差 [95%区間]"], body)))
    A("")
    A("設計 §3 は、「安く買えた分より、買えずに逃した上昇のほうが大きい」"
      "と先に予想している。"
      "**「未約定は見送り」の行と基準を見比べれば、そうなっているかが分かる。**"
      "約定率が下がるほど取り逃しが増える形になっているか、"
      "「約定した分の平均」が基準より良くても1件あたりでは負けていないかを見る。")
    A("")

    A("## 寄り成行で買ったときのばらつき")
    A("")
    A("平均だけ見ても、実際にどれくらい振れるかが分からない。"
      "**基準（翌日寄り成行・60営業日保有）の1件あたりリターンの散らばり**を出す。")
    A("")
    body = []
    for name, sec in report["subsets"].items():
        for tag, ja in (("all", "全期間"), ("test", "テスト期間")):
            b = (sec.get("baseline_spread") or {}).get(tag)
            if not b:
                continue
            body.append([name, ja, f"{b['n']:,}",
                         f"{b['mean']:+.2f}", f"{b['median']:+.2f}",
                         f"{b['win_rate_if_filled']:.1f}%", f"{b['p05']:+.2f}"])
    A("\n".join(_table(
        ["部分集合", "期間", "件数", "平均", "中央値", "勝率", "下側5%"], body)))
    A("")
    A("平均が中央値より大きいのは、大きく伸びる少数が平均を押し上げているため。"
      "**1件ずつの結果は大きく振れる。**下側5%は「20回に1回はこれ以下になる」水準。")
    A("")
    A("## 母集団の作られ方")
    A("")
    A("\n".join(_table(["条件", "残った件数"],
                       [[k, f"{n:,}"] for k, n in report["funnel"].items()])))
    A("")

    for name, sec in report["subsets"].items():
        A(f"## {name}（{sec['n_all']:,}件 / 訓練 {sec['n_train']:,} / "
          f"テスト {sec['n_test']:,}）")
        A("")
        for label, ja in (("optimistic", "楽観（安値≦指値なら約定）"),
                          ("pessimistic", "悲観（安値が指値より余裕ぶん下で約定）")):
            rows = sec.get(f"test_{label}") or []
            if not rows:
                continue
            A(f"### テスト期間・{ja}")
            A("")
            base = rows[0]
            body = [["**寄り成行（基準）**", f"{base['mean']:+.2f}",
                     f"{base['fill_rate']:.0f}%", "—", "—", "—"]]
            for r in top_rows(rows):
                if r["policy"] == "寄り成行":
                    continue
                body.append([r["policy"], f"{r['mean']:+.2f}",
                             f"{r['fill_rate']:.0f}%",
                             f"{r['missed_rate']:.0f}%",
                             f"{r['entry_improve']:+.2f}%"
                             if np.isfinite(r["entry_improve"]) else "—",
                             _ci(r["vs_base"])])
            A("\n".join(_table(
                ["方策", "1件あたり平均", "約定率", "取り逃し", "エントリー改善",
                 "基準との差 [95%区間]"], body)))
            A("")

        sel = sec.get("selected", {})
        if sel:
            A("### 訓練で選んで、テストで測る")
            A("")
            body = []
            for key, s in sel.items():
                t = s.get("test") or {}
                body.append([
                    "楽観" if key.startswith("optimistic") else "悲観",
                    s["kind_ja"], s["policy"],
                    f"{s['train']['mean']:+.2f}",
                    f"{t.get('mean', float('nan')):+.2f}" if t else "—",
                    _ci(s.get("test_vs_base")),
                ])
            A("\n".join(_table(
                ["約定の仮定", "種類", "訓練で選ばれた方策", "訓練の平均",
                 "テストの平均", "テストでの基準との差 [95%区間]"], body)))
            A("")

    flips = report["subsets"]["全件"].get("flip") or []
    A("## 楽観と悲観の幅（分足に払う金額の上限）")
    A("")
    A("テスト期間で、基準との差が約定の仮定でどれだけ動くか。"
      "**符号が反転する方策があれば、決め手は日中の到達順序であり日足では決まらない。**")
    A("")
    flipped = [f for f in flips if f["flips"]]
    if flipped:
        A("### 符号が反転した方策")
        A("")
        A("\n".join(_table(["方策", "楽観での差", "悲観での差", "幅"],
                           [[f["policy"], f"{f['optimistic']:+.2f}",
                             f"{f['pessimistic']:+.2f}", f"{f['width']:.2f}"]
                            for f in flipped])))
        A("")
        A("差そのものが小さい方策は、揺らせば符号が変わる。"
          "反転したという事実だけでなく、**差の大きさが合格ラインに届いているか**"
          "を併せて見ること。")
    else:
        A("符号が反転した方策は**無い**。")
    A("")
    A("### 幅の大きい方策")
    A("")
    A("\n".join(_table(["方策", "楽観での差", "悲観での差", "幅", "符号が反転"],
                       [[f["policy"], f"{f['optimistic']:+.2f}",
                         f"{f['pessimistic']:+.2f}", f"{f['width']:.2f}",
                         "**する**" if f["flips"] else "しない"]
                        for f in flips[:12]])))
    A("")

    sens = report.get("sensitivity") or []
    if sens:
        A("## 手数料と余裕への感度（テスト期間・基準との差）")
        A("")
        keys = [k for k in sens[0] if k not in
                ("fee_pct", "slack_bp", "fill", "寄り成行")]
        A("\n".join(_table(
            ["手数料", "余裕", "約定", "寄り成行の平均"] + keys,
            [[f"{r['fee_pct']}%", f"{r['slack_bp']:.0f}bp", r["fill"],
              f"{r['寄り成行']:+.2f}"] + [f"{r[k]:+.2f}" for k in keys]
             for r in sens])))
        A("")

    yr = report.get("by_year") or []
    if yr:
        A("## 年ごと（全件・楽観・基準との差ではなく実額）")
        A("")
        keys = [k for k in yr[0] if k not in ("year", "n")]
        A("\n".join(_table(["年", "件数"] + keys,
                           [[r["year"], f"{r['n']:,}"]
                            + [f"{r[k]:+.2f}" for k in keys] for r in yr])))
        A("")
        A("1つの年だけで勝っている方策は採らない。"
          "局面が変われば消える見込みが高いため。")
        A("")

    A("## この報告の限界")
    A("")
    A("- 日足なので、安値と高値のどちらが先に付いたかが分からない。"
      "楽観と悲観の2つで挟んでいるのはそのため")
    A("- 呼値の刻みは実装していない。3,000円の株で1呼値は約3bpなので、"
      "余裕 0/10/25bp の振れ幅に収まる")
    A("- 建玉の大きさを考えていない。薄い銘柄では、"
      "安値が指値を割っても実際には買えない量がある")
    A("- 見送ったぶんの資金は現金で寝かせる前提。"
      "他の銘柄に回せるなら見送りの評価は過小になる")
    A("- 手仕舞い日は**基準日から固定**している（設計 §8.2）。"
      "2日目以降に約定した方策は、そのぶん保有期間が短くなる。"
      "上昇局面では待つ方策に不利な側に働くので、"
      "押し目待ちが勝つという結論が出た場合は過大評価ではない")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--test-start", default="2024-10-01")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fee", type=float, default=EP.Costs.fee_pct)
    ap.add_argument("--slack", type=float, default=EP.Costs.slack_bp)
    ap.add_argument("--min-trading-value", type=float, default=0.1)
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--out-json", default=OUT_JSON)
    args = ap.parse_args(argv)

    costs = EP.Costs(fee_pct=args.fee, slack_bp=args.slack)
    policies = EP.default_policies()
    print(f"[policy] 比較する方策: {len(policies)}件（格子は先に固定してある）")

    ev, funnel, n_bars = load_events(args.data_dir, args.min_trading_value)
    if ev.empty:
        raise SystemExit("イベントが0件。生データを確認してください")
    ev, score_info = attach_scores(ev, args.model_dir)

    report: Dict = {
        "ranAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bars_rows": n_bars,
        "funnel": funnel,
        "test_start": args.test_start,
        "n_boot": args.n_boot,
        "costs": {"fee_pct": costs.fee_pct, "slack_bp": costs.slack_bp},
        "score_info": score_info,
        "n_policies": len(policies),
        "subsets": {},
    }

    for name, sub in subsets(ev, score_info.get("available", False)):
        print(f"\n[eval] {name}: {len(sub):,}件")
        res = evaluate(sub, policies, costs, args.test_start,
                       args.n_boot, args.seed)
        res["verdict"] = verdict(res)
        report["subsets"][name] = res
        print(f"  押し目: {res['verdict']['dip']}")
        print(f"  分足  : {res['verdict']['minute_addon']}")

    # 感度と年別は、全件の選ばれた方策について出す
    chosen = []
    for key, s in report["subsets"]["全件"]["selected"].items():
        if s["policy"] not in chosen:
            chosen.append(s["policy"])
    print(f"\n[sens] 感度を見る方策: {chosen}")
    report["sensitivity"] = sensitivity(ev, policies, chosen, args.test_start)
    report["by_year"] = by_year(ev, policies, ["寄り成行"] + chosen, costs, True)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    write_md(report, args.out_md)
    print(f"\n[write] {args.out_json}\n[write] {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
