#!/usr/bin/env python3
"""
実験74: Jev への問いを「儲かるか」「モデルの正例になるか」に変えて、モデルが陽性と見た候補の中で
真陽性と偽陽性を分けられるかを測る（実験73 の続き）。

運用者の決定（2026-10-11）: 問いは A・B・C の3つ。腕は prod（本番と同じ state）と anon（銘柄名・コード・業種・
市場・日付を伏せる）。採否の主の物差しは **モデルの正例（+1.2σ）**。他の物差しも表には出す。
参考として、実験73 の問い R（+10% 到達）も同じ呼び出しに同梱する（費用はほぼ変わらない。同じ state への答えの
再現性と、新しい問いが R の言い換えになっていないかを行ごとに比べるため）。

問い（全部 noul。1回の呼び出しに4つ入れる。版 prec_v1）
  A a_tp10   翌営業日の寄りで買い、+10% の指値（届かなければ20営業日目までの5日平均終値で売る）の収益が正
  B b_tp20   同じく +20% の指値（届かなければ20営業日目までの5日平均終値で売る）の収益が正。運用の出口
  C c_label  20営業日以内に終値が買値から必要上昇率（ボラの1.2σ、state に記載）以上に上がり、終盤もその半分以上を
             保ち、5日移動平均が20日移動平均を上回る（モデルの正例の定義）
  R r_rise10 実験73 と同じ「20営業日以内に高値が買値の +10% に達する」
答え合わせ（手元の値）: A ↔ tp10 > 0 / B ↔ tp20 > 0 / C ↔ label / R ↔ hit10

先に決めること（結果を見てから分け方を足さない）
----------------------------------------------
対象と state は実験73 と同じ（直近1年の OOF、前の窓の百分位、窓ごとの LightGBM の SHAP）。
1. 較正: 問いごとに、自分の答え合わせに対する Brier（母集団の率を常に答える基準と比べる）と 10pt 刻みの帯
2. 分離力: 問い A/B/C/R と素朴な基準（日次ボラ・20日リターン）・LightGBM の百分位・3モデルの最小の ROC-AUC を、
   4つの答え合わせ（tp10>0 / tp20>0 / 正例 / hit10）それぞれに対して。母集団と「3モデルとも90以上」の中で
3. Precision との相関（主の物差し = 正例）: 選定の集合（3モデル90以上 / 3モデル95以上 / LightGBM 95以上 /
   LightGBM 90以上 / 母集団）ごと・問いごとに、(a) 真陽性（正例）と偽陽性の答えの平均の差（SE は銘柄ごと）、
   (b) 集合の中の AUC（正例）、(c) 答えの3分位ごとの Precision（正例・tp20>0）と +10%/+20% 指値の平均、
   (d) 50 と集合の中央値をゲートにしたときの件数・Precision・+20%指値
4. 問い同士の相関（Spearman。素朴な基準とも）
5. prod と anon の相関・平均の差（問いごと）
6. 費用
採否を検討する条件（全部そろったとき。1つの実験で決めない）: 主の物差し（正例）で「3モデルとも90以上」の中の
真陽性 − 偽陽性 の差が正で |z| ≥ 2、anon でも同じ向き、答えの3分位で +20%指値が単調に上がる。

記録: 集計だけをログに出す（銘柄ごとの値は出さない）。答えは research/_data/oof/e74_jev_answers.parquet に控える
（Actions のキャッシュ。公開の成果物には上げない）。

  exp=e74_jev_questions.py  args="--dry-run"              # 件数・問い・state の例（通信しない）
  exp=e74_jev_questions.py  args="--arms prod --limit 200"
  exp=e74_jev_questions.py                                 # prod と anon・直近1年（既定）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)
sys.path.insert(0, HERE)
import jev_predict as JV  # noqa: E402
import lab  # noqa: E402
import predict_daily as PD  # noqa: E402
import tuning  # noqa: E402
import e70_borderline as E70  # noqa: E402
import e73_jev_oof as E73  # noqa: E402

log = E73.log

QSET = "prec_v1"
KEYS = ("a_tp10", "b_tp20", "c_label", "r_rise10")
JA = {"a_tp10": "A +10%指値の収益が正", "b_tp20": "B +20%指値の収益が正",
      "c_label": "C モデルの正例", "r_rise10": "R +10%到達（実験73）"}


def _trade(pct: int) -> str:
    return (f"この日本株を翌営業日の寄り付き（始値）で買い、買った日を1日目として{JV.HOLD_DAYS}営業日以内に"
            f"高値が買値の +{pct}% に達したら、その日に +{pct}% で売る（+{pct}% の指値）。達しなければ、"
            f"{JV.HOLD_DAYS}営業日目までの直近5日間の平均終値で売る。")


QUESTIONS: Dict[str, Dict] = {
    "a_tp10": {
        "type": "noul",
        "instructions": _trade(10) + "この取引の収益は正になる（売値が買値を上回る）。",
        "criteria": {"true": "+10% の指値に届く。または届かなくても、20営業日目の売値が買値を上回る",
                     "false": "+10% に届かず、20営業日目の売値が買値以下"},
    },
    "b_tp20": {
        "type": "noul",
        "instructions": _trade(20) + "この取引の収益は正になる（売値が買値を上回る）。",
        "criteria": {"true": "+20% の指値に届く。または届かなくても、20営業日目の売値が買値を上回る",
                     "false": "+20% に届かず、20営業日目の売値が買値以下"},
    },
    "c_label": {
        "type": "noul",
        "instructions": (
            f"この日本株を翌営業日の寄り付き（始値）で買ったとして、{JV.HOLD_DAYS}営業日以内に終値が買値から"
            "「必要上昇率」（この銘柄の日次ボラティリティの1.2σ。state の『必要上昇率』）以上に上がり、"
            f"かつ{JV.HOLD_DAYS}営業日目の終盤（直近5日間の平均終値）でもその半分以上の上昇を保ち、"
            "5日移動平均が20日移動平均を上回っている。つまり、上がって、上がったまま終える"
            "（機械学習モデルが「正例」とする形）。"),
        "criteria": {"true": "必要上昇率以上に上がり、終盤もその半分以上を保ち、短期の移動平均が長期を上回って終える",
                     "false": "必要上昇率に届かない。または届いても終盤に半分未満まで戻す、"
                              "または短期の移動平均が長期を下回って終える"},
    },
    "r_rise10": JV.QUESTIONS[JV.QUESTION_KEY],
}

CACHE = os.path.join(lab.DATA_DIR, "oof", "e74_jev_answers.parquet")
OUT = os.path.join(lab.DATA_DIR, "oof", "e74_jev_questions.json")
CACHE_COLS = ["arm", "Code", "Date", "qset", "state_sha", "model", *KEYS,
              "input_tokens", "output_tokens", "asked_at"]
#: 答え合わせの列（d に作る）。主の物差しは y_c（モデルの正例）
TRUTHS = (("y_a", "tp10 > 0"), ("y_b", "tp20 > 0"), ("y_c", "モデルの正例"), ("y_r", "+10% 到達"))
PRIMARY = "y_c"
TRUTH_OF = {"a_tp10": "y_a", "b_tp20": "y_b", "c_label": "y_c", "r_rise10": "y_r"}
GATE = 50.0


def add_truths(d: pd.DataFrame) -> pd.DataFrame:
    """答え合わせの列を足す。tp20 が無い（古い値動きの表）なら作れないので止める。"""
    for c in ("tp10", "tp20", "hit10"):
        if c not in d.columns:
            raise SystemExit(f"{c} が無い（実験32 の値動きの表を作り直す）")
    d = d.copy()
    d["y_a"] = pd.to_numeric(d["tp10"], errors="coerce") > 0
    d["y_b"] = pd.to_numeric(d["tp20"], errors="coerce") > 0
    d["y_c"] = pd.to_numeric(d["label"], errors="coerce").fillna(0).astype(int) == 1
    d["y_r"] = d["hit10"].astype(bool)
    return d


def parse_answers(resp: Dict) -> Dict[str, float]:
    """4つの noul をまとめて % に。1つでも形が違えば JevError（部分的な答えを控えない）。"""
    out = {}
    for k in KEYS:
        out[k] = JV.parse_noul(resp, k)["prob"]
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    out["model"] = (resp.get("model") or "")
    out["input_tokens"] = int(usage.get("input_tokens") or 0)
    out["output_tokens"] = int(usage.get("output_tokens") or 0)
    return out


def cache_key(arm: str, code: str, date: str, qset: str = QSET) -> tuple:
    return (str(arm), str(code), str(date), str(qset))


def load_cache(path: str = CACHE) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=CACHE_COLS)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:                  # noqa: BLE001
        log(f"控えを読めないので作り直す: {type(exc).__name__}")
        return pd.DataFrame(columns=CACHE_COLS)
    for c in CACHE_COLS:
        if c not in df.columns:
            df[c] = None
    return df[CACHE_COLS]


def save_cache(df: pd.DataFrame, path: str = CACHE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[CACHE_COLS].drop_duplicates(["arm", "Code", "Date", "qset"], keep="first") \
        .to_parquet(path, index=False, compression="zstd")


def ask_arm(arm: str, cands: List[Dict], tones: Dict[str, Optional[str]], client,
            cache: pd.DataFrame, workers: int, path: str = CACHE,
            log=log) -> Tuple[Dict[tuple, Dict[str, float]], Dict, pd.DataFrame]:
    """腕の全候補に4つの問いをまとめて問う。控えにある (腕, コード, 日付, 版) は問い直さない。"""
    known = {cache_key(x["arm"], x["Code"], x["Date"], x["qset"]): x for x in cache.to_dict("records")}
    probs: Dict[tuple, Dict[str, float]] = {}
    usage = {"asked": 0, "cached": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "models": {}}
    todo = []
    for c in cands:
        state = E73.arm_state(c, tones.get(c["date"]), arm)
        k = cache_key(arm, c["jqCode"], c["date"])
        if k in known:
            rec = known[k]
            probs[(c["jqCode"], c["date"])] = {q: float(rec[q]) for q in KEYS}
            usage["cached"] += 1
            m = str(rec.get("model") or "")
            usage["models"][m] = usage["models"].get(m, 0) + 1
            continue
        todo.append((c, state, JV.state_sha(state)))
    log(f"[{arm}] 控えから {usage['cached']:,}件 / 問う {len(todo):,}件（1件に {len(KEYS)} 問）")
    if not todo:
        return probs, usage, cache
    new_records: List[Dict] = []
    asked_at = pd.Timestamp.now(tz="UTC").isoformat()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(client.system_one, state, QUESTIONS): (c, sha) for c, state, sha in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            c, sha = futs[fut]
            try:
                ans = parse_answers(fut.result())
            except JV.JevError as exc:
                usage["failed"] += 1
                if usage["failed"] <= 5:
                    log(f"  [{arm}] 失敗: {exc}")
                continue
            usage["asked"] += 1
            usage["input_tokens"] += ans["input_tokens"]
            usage["output_tokens"] += ans["output_tokens"]
            m = ans["model"] or client.model
            usage["models"][m] = usage["models"].get(m, 0) + 1
            probs[(c["jqCode"], c["date"])] = {q: ans[q] for q in KEYS}
            new_records.append({"arm": arm, "Code": c["jqCode"], "Date": c["date"], "qset": QSET,
                                "state_sha": sha, "model": m, **{q: ans[q] for q in KEYS},
                                "input_tokens": ans["input_tokens"], "output_tokens": ans["output_tokens"],
                                "asked_at": asked_at})
            if n % 100 == 0 or n == len(todo):
                new_df = pd.DataFrame(new_records, columns=CACHE_COLS)
                cache = new_df if cache.empty else pd.concat([cache, new_df], ignore_index=True)
                save_cache(cache, path)
                new_records = []
                per = usage["input_tokens"] / max(1, usage["asked"])
                log(f"  [{arm}] {n:,}/{len(todo):,} 済み（{time.time() - t0:.0f}秒・失敗 {usage['failed']}）"
                    f" / 入力 {per:,.0f} トークン/件 → 全体の見込み {per * len(todo):,.0f}")
    if usage["failed"] and usage["asked"] == 0:
        raise SystemExit(f"[{arm}] 全件失敗した（鍵・疎通・問いの形を確かめる）")
    return probs, usage, cache


# ---------------------------------------------------------------------- #
# 集計
# ---------------------------------------------------------------------- #

def stats(sub: pd.DataFrame) -> Dict:
    """集合の Precision（正例 / tp20>0 / tp10>0）と +10%・+20% 指値の平均。"""
    if len(sub) == 0:
        return {"n": 0}
    return {"n": int(len(sub)), "codes": int(sub["Code"].nunique()),
            "prec_c": float(sub["y_c"].mean() * 100), "prec_c_se": E70.cluster_se(sub["y_c"].astype(float) * 100, sub["Code"]),
            "prec_b": float(sub["y_b"].mean() * 100), "prec_a": float(sub["y_a"].mean() * 100),
            "tp10": float(sub["tp10"].mean()), "tp10_se": E70.cluster_se(sub["tp10"], sub["Code"]),
            "tp20": float(sub["tp20"].mean()), "tp20_se": E70.cluster_se(sub["tp20"], sub["Code"])}


HEAD = (f"  {'':<36}{'件数':>6}{'Prec(正例)':>9}{'SE':>5}{'Prec(+20%指値>0)':>15}{'+10%指値':>9}{'+20%指値':>9}{'SE':>6}")


def line(name: str, s: Dict) -> str:
    if not s.get("n"):
        return f"  {name:<36}{0:>6}"
    if s["n"] < E73.MIN_ROWS:
        return f"  {name:<36}{s['n']:>6}   （件数が少なすぎる）"
    return (f"  {name:<36}{s['n']:>6}{s['prec_c']:>8.1f}%{s['prec_c_se']:>5.1f}{s['prec_b']:>14.1f}%"
            f"{s['tp10']:>+8.2f}%{s['tp20']:>+8.2f}%{s['tp20_se']:>6.2f}")


def report(d: pd.DataFrame, arms: Sequence[str], usage: Dict[str, Dict]) -> Dict:
    res: Dict[str, object] = {"rows": int(len(d)), "from": str(d["Date"].min().date()),
                              "to": str(d["Date"].max().date()), "codes": int(d["Code"].nunique()),
                              "arms": list(arms), "usage": usage, "qset": QSET}
    naive = [c for c in ("vol_20d", "ret_20d") if c in d.columns]
    base = {y: float(d[y].mean()) for y, _ in TRUTHS}
    print(f"\n=== 0. 対象 ===\n  {len(d):,}行 / {d['Code'].nunique():,}銘柄 / {res['from']}〜{res['to']}")
    print("  答え合わせの率: " + " / ".join(f"{lab_} {base[y] * 100:.1f}%" for y, lab_ in TRUTHS))
    for arm in arms:
        for q in KEYS:
            col = f"{arm}_{q}"
            ok = d[col].notna()
            print(f"  [{arm}] {JA[q]:<22} 値あり {int(ok.sum()):,} / 平均 {d.loc[ok, col].mean():.1f}% / "
                  f"中央値 {d.loc[ok, col].median():.1f}%")

    # 1. 較正
    print("\n=== 1. 較正（問いごとに、自分の答え合わせに対して）===")
    res["calibration"] = {}
    for arm in arms:
        res["calibration"][arm] = {}
        for q in KEYS:
            col, y = f"{arm}_{q}", TRUTH_OF[q]
            g = d[d[col].notna()]
            b_j = E73.brier(g[col] / 100.0, g[y])
            b_0 = E73.brier(np.full(len(g), base[y]), g[y])
            tab = E73.calib_table(g[col], g[y], g["tp10"], g["r"], g["Code"])
            res["calibration"][arm][q] = {"brier": b_j, "brier_base": b_0, "bins": tab}
            print(f"  [{arm}] {JA[q]:<22} Brier {b_j:.4f} / 基準（率 {base[y] * 100:.1f}% を常に答える）{b_0:.4f}")
            print(f"  {'帯(%)':<10}{'件数':>6}{'答えの平均':>9}{'実際の率':>8}{'SE':>6}{'+10%指値':>9}")
            for row in tab:
                if row["n"] >= E73.MIN_ROWS:
                    print(f"  {row['lo']:>3.0f}〜{row['hi']:<4.0f}{row['n']:>8}{row['mean_prob']:>8.1f}%"
                          f"{row['hit']:>7.1f}%{row['hit_se']:>6.1f}{row['tp10']:>+8.2f}%")

    # 2. 分離力
    print("\n=== 2. 分離力（ROC-AUC。行 = 並べる物差し、列 = 答え合わせ）===")
    res["auc"] = {}
    scopes = [("母集団", pd.Series(True, index=d.index))]
    for name, mask in E73.selections(d):
        if name.startswith("3モデルとも90") or name.startswith("LightGBM 90"):
            scopes.append((name, mask))
    for sname, mask in scopes:
        sub = d[mask]
        print(f"  --- {sname}: {len(sub):,}行 ---")
        print(f"  {'物差し':<30}" + "".join(f"{lab_:>14}" for _, lab_ in TRUTHS))
        res["auc"][sname] = {}
        rows = [(f"Jev {JA[q]}（{arm}）", f"{arm}_{q}") for arm in arms for q in KEYS]
        rows += [(f"{c}（素朴な基準）", c) for c in naive]
        rows += [("LightGBM の百分位", "hp_lgbm")]
        if "hp_min" in sub.columns:
            rows += [("3モデルの最小", "hp_min")]
        for nm, col in rows:
            if col not in sub.columns:
                continue
            g = sub[sub[col].notna()]
            a = {y: E73.roc_auc(g[y], g[col]) for y, _ in TRUTHS}
            res["auc"][sname][col] = a
            print(f"  {nm:<30}" + "".join(f"{a[y]:>14.3f}" for y, _ in TRUTHS))

    # 3. Precision との相関（主の物差し = 正例）
    print(f"\n=== 3. モデルが陽性と見た候補の中で、答えは真陽性（正例）と偽陽性を分けるか ===")
    res["precision"] = {}
    for arm in arms:
        res["precision"][arm] = {}
        for sname, mask in E73.selections(d):
            sub = d[mask]
            out: Dict[str, object] = {"total": stats(sub)}
            print(f"\n  --- [{arm}] {sname}: {len(sub):,}行 / 正例率 {sub['y_c'].mean() * 100:.1f}% ---")
            if len(sub) < E73.MIN_ROWS:
                print("    件数が少なすぎる（10件未満）")
                res["precision"][arm][sname] = out
                continue
            out["gap"], out["auc"], out["terciles"], out["gates"] = {}, {}, {}, {}
            for q in KEYS:
                col = f"{arm}_{q}"
                g = E73.tp_fp_gap(sub, col, PRIMARY)
                g_own = E73.tp_fp_gap(sub, col, TRUTH_OF[q])
                out["gap"][q] = {"primary": g, "own": g_own}
                a = {y: E73.roc_auc(sub[y], sub[col]) for y, _ in TRUTHS}
                out["auc"][q] = a
                print(f"    (a)(b) {JA[q]:<22} 正例の真陽性 {g['tp_n']:>4}件 {g['tp']:5.1f}% / 偽陽性 {g['fp_n']:>4}件 {g['fp']:5.1f}%"
                      f" / 差 {g['d']:+5.1f}pt ± {g['se']:.1f}（z {g['z']:+.2f}）/ AUC 正例 {a['y_c']:.3f}・自分の答え合わせ"
                      f" {a[TRUTH_OF[q]]:.3f}")
            # (c) 3分位
            if len(sub) >= E73.MIN_TERCILE:
                print("    (c) 答えの3分位（集合の中で低い / 中 / 高い）")
                print(HEAD)
                for q in KEYS:
                    col = f"{arm}_{q}"
                    try:
                        qs = pd.qcut(sub[col], 3, labels=False, duplicates="drop")
                    except ValueError:
                        continue
                    if qs.nunique() < 2:
                        continue
                    out["terciles"][q] = []
                    for k in sorted(qs.dropna().unique()):
                        gg = sub[qs == k]
                        s = stats(gg)
                        s["lo"], s["hi"] = float(gg[col].min()), float(gg[col].max())
                        out["terciles"][q].append(s)
                        print(line(f"{JA[q][:1]} {s['lo']:.0f}〜{s['hi']:.0f}%", s))
            # (d) ゲート（50 と集合の中央値）
            print(f"    (d) ゲート（答え ≥ t を買う / < t を見送る。t = {GATE:.0f} と集合の中央値）")
            print(HEAD)
            print(line("ゲート無し（集合の全部）", out["total"]))
            for q in KEYS:
                col = f"{arm}_{q}"
                med = float(sub[col].median())
                out["gates"][q] = {}
                for t, tn in ((GATE, f"{GATE:.0f}"), (med, f"中央値 {med:.0f}")):
                    hi, lo = stats(sub[sub[col] >= t]), stats(sub[sub[col] < t])
                    out["gates"][q][tn] = {"ge": hi, "lt": lo, "diff_tp20": E70.diff(hi, lo, "tp20")}
                    print(line(f"{JA[q][:1]} ≥ {tn}", hi))
                    print(line(f"{JA[q][:1]} < {tn}", lo))
            res["precision"][arm][sname] = out

    # 4. 問い同士の相関
    print("\n=== 4. 問い同士の相関（Spearman。prod）===")
    arm0 = arms[0]
    cols = [f"{arm0}_{q}" for q in KEYS] + naive
    names = [JA[q][:1] for q in KEYS] + naive
    res["corr"] = {}
    print(f"  {'':<10}" + "".join(f"{n:>10}" for n in names))
    for ci, ni in zip(cols, names):
        row = {nj: E73.spearman(d[ci], d[cj]) for cj, nj in zip(cols, names)}
        res["corr"][ni] = row
        print(f"  {ni:<10}" + "".join(f"{row[nj]:>+10.3f}" for nj in names))

    # 5. prod と anon
    if len(arms) == 2:
        print(f"\n=== 5. {arms[0]} と {arms[1]}（問いごと）===")
        res["arms"] = {}
        for q in KEYS:
            a, b = d[f"{arms[0]}_{q}"], d[f"{arms[1]}_{q}"]
            both = a.notna() & b.notna()
            rho = E73.spearman(a, b)
            md = float((a[both] - b[both]).mean()) if both.any() else np.nan
            res["arms"][q] = {"rho": rho, "mean_diff": md}
            print(f"  {JA[q]:<22} 相関 {rho:+.3f} / 平均の差 {md:+.1f}pt")

    # 6. 費用
    print("\n=== 6. 費用（この実行で問うた分）===")
    for arm in arms:
        u = usage.get(arm, {})
        per = u.get("input_tokens", 0) / max(1, u.get("asked", 0))
        print(f"  [{arm}] 問うた {u.get('asked', 0):,}件 / 控え {u.get('cached', 0):,}件 / 失敗 {u.get('failed', 0):,}件 / "
              f"入力 {u.get('input_tokens', 0):,} トークン（{per:,.0f}/件）/ 出力 {u.get('output_tokens', 0):,}")
        models = u.get("models") or {}
        if models:
            print(f"  [{arm}] 答えたモデルの版: " + " / ".join(f"{m} {n:,}件" for m, n in sorted(models.items())))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験74: 問いを変えて Jev を当てる")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--arms", default="prod,anon")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in E73.ARMS:
            raise SystemExit(f"知らない腕: {a}（{', '.join(E73.ARMS)}）")

    d, algos, frame = E73.load_rows(args.days)
    d = add_truths(d)
    trees = [a for a in E70.TREES if a in algos]
    d["hp_min"] = d[[f"hp_{a}" for a in trees]].min(axis=1)
    log(f"対象 {len(d):,}行 / {d['Code'].nunique():,}銘柄 / {d['Date'].min().date()}〜{d['Date'].max().date()} "
        f"/ モデル {', '.join(algos)}")
    if args.limit:
        keep = d.sort_values(["Date", "rank_in_day"], ascending=[False, True]).head(args.limit)
        d = d.loc[keep.index].sort_values(["Date", "rank_in_day"]).reset_index(drop=True)
        log(f"--limit {args.limit}: 新しい日から {len(d):,}行に絞る")

    meta = json.load(open(os.path.join(RESEARCH, "model", "meta.json"), encoding="utf-8"))
    cols = [c for c in meta["features"] if c in frame.columns]
    params = {k: v for k, v in (meta.get("params") or {}).items() if not k.startswith("_")} \
        or tuning.params_for(meta.get("preset", "all"))
    log("窓ごとの LightGBM で寄与を付ける（本番のパラメータ）")
    contrib = E73.fold_contrib(d, frame, cols, params)
    names = {str(r["Code"]): r for r in PD.name_map(lab.DATA_DIR).to_dict("records")}
    prog = PD.progress_fields(lab.DATA_DIR, zip(d["Code"].astype(str), d["Date"].dt.strftime("%Y-%m-%d")))
    cands = E73.candidate_rows(d, algos, contrib, names, prog)
    tones: Dict[str, Optional[str]] = {}
    for c in cands:
        tones.setdefault(c["date"], []).append(c)
    tones = {k: JV.market_tone(v) for k, v in tones.items()}
    log(f"候補の形にした {len(cands):,}件（寄与あり {sum(1 for c in cands if c.get('contrib')):,}）")
    qchars = len(json.dumps(QUESTIONS, ensure_ascii=False, separators=(",", ":")))
    log(f"問い {len(KEYS)}つ（{qchars:,}文字。実験73 は 1つ "
        f"{len(json.dumps(JV.QUESTIONS, ensure_ascii=False, separators=(',', ':'))):,}文字）")

    if args.dry_run or not JV.enabled():
        if not args.dry_run:
            log(f"{JV.API_KEY_ENV} が未設定なので問わない")
        if cands:
            print("\n[例] anon の state（銘柄・日付を伏せたもの）:")
            print(json.dumps(E73.arm_state(cands[-1], tones.get(cands[-1]["date"]), "anon"),
                             ensure_ascii=False, indent=1))
        print("\n[問い]")
        print(json.dumps(QUESTIONS, ensure_ascii=False, indent=1))
        print("\n通信していません")
        return 0 if args.dry_run else 1

    client = JV.Client()
    log(f"Jev: {client.base_url} / モデル {client.model}")
    cache = load_cache()
    usage: Dict[str, Dict] = {}
    for arm in arms:
        probs, u, cache = ask_arm(arm, cands, tones, client, cache, args.workers)
        usage[arm] = u
        for q in KEYS:
            d[f"{arm}_{q}"] = [probs.get((c["jqCode"], c["date"]), {}).get(q, np.nan) for c in cands]
    res = report(d, arms, usage)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x))
    log(f"記録: {OUT}（集計だけ。銘柄ごとの値は控え {CACHE} にあり、公開の成果物には上げない）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
