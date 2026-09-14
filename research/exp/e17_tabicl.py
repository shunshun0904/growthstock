#!/usr/bin/env python3
"""
実験17: TabICLv2 を既存5モデルと同じ土俵で測り、並べる価値があるかを見る。

    python3 research/exp/e17_tabicl.py --probe          # 所要時間だけ実測
    python3 research/exp/e17_tabicl.py                  # 本走
    python3 research/exp/e17_tabicl.py --max-context 8000 --n-estimators 4

何を測るか
---------
実験16（既存5モデルの並べ比べ）と同じものを測る。土俵を変えると
「TabICL が良かった」のか「測り方を変えたから良く見えた」のかが
分からなくなる。

  1. 運用指標。しきい値運用（スコアが過去分布の上位10%なら買う・
     翌営業日の寄り買い・40営業日後の5日平均終値売り）の優位と、
     その再現性（窓平均・窓SD・勝った窓・最悪の窓）。
     しきい値は窓ごとに**過去の窓だけ**から決める。
  2. 既存5モデルとのスコア相関と、上位10%の重複。
     **並べる価値はここで決まる。** 相関が高いモデルは画面で
     2本ぶんの場所を取るだけで、情報が増えない。

判定の足切りは実験11 のノイズ床（同一設定で種5個）と同じ。

  しきい値優位  レンジ 0.484pt
  窓平均        レンジ 0.143pt  <- 最も解像度が高い

条件の非対称性（結果を読むときに必ず添えること）
--------------------------------------------
  探索      既存5モデルは Optuna で50試行ずつ探索済み。
            TabICL は探索しない（既定値のまま）。
            「調整なしで効く」ことが TabICL の売りなので、
            探索して勝たせるのは主旨から外れる。
            この非対称は TabICL に不利な向き。
  種平均    既存5モデルは3種平均（実験16 と同じ）。TabICL は既定1種。
            種平均はそれ自体がアンサンブルなので、この非対称は
            既存5モデル側に有利な向き。--tabicl-seeds で揃えられる。
  文脈      --max-context を掛けたときは、TabICL だけ訓練行の一部
            （直近側）しか見ていない。既存5モデルは全行で学習する。
            この非対称も TabICL に不利な向き。

打ち切りと再開
------------
TabICL は窓ごとに out-of-fold を書き出す（research/_data/oof/tabicl/）。
--budget-seconds を超えたらそこで止め、次の実行が残りの窓から続ける。
CI のジョブ上限は6時間で、CPU での所要時間が事前に読めないため。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import e16_lineup as E16  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import models as M  # noqa: E402
import tabicl_model as T  # noqa: E402
import tuning_multi as TM  # noqa: E402

REPORT = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      "docs", "MODEL_TABICL.md")
RESULT_JSON = os.path.join(lab.DATA_DIR, "e17_tabicl.json")

#: 既存5モデルの種。実験16 と同じにして、数字をそのまま突き合わせられるようにする
BASELINE_SEEDS = E16.SCREEN_SEEDS


def run_baselines(df: pd.DataFrame, cols, algos, seeds, log=print) -> Dict[str, lab.Result]:
    """既存モデルの out-of-fold。実験16 と同じコードで作る。"""
    os.makedirs(E16.OOF_DIR, exist_ok=True)
    out: Dict[str, lab.Result] = {}
    for algo in algos:
        p = os.path.join(E16.OOF_DIR, f"lineup_{algo}.parquet")
        if os.path.exists(p):
            o = lab.attach_outcomes(pd.read_parquet(p), df)
            out[algo] = lab.Result(algo, o, lab.metrics(o))
            log(f"  {algo:<7} 保存済みを読む")
            continue
        t0 = time.time()
        r = lab.run_multi(df, lambda s, a=algo: E16.seeded(a, cols, s),
                          seeds=seeds, cols=cols, name=algo)
        r.oof.to_parquet(p, index=False)
        out[algo] = r
        log(f"  {algo:<7} {time.time() - t0:>5.0f}秒  "
            f"窓平均 {r.metrics['thr_fold_mean']:+.2f}pt")
    return out


def run_tabicl(df: pd.DataFrame, cols, args, log=print) -> lab.Result:
    """TabICL の out-of-fold。種ごとに回して平均する。"""
    seeds = [int(s) for s in str(args.tabicl_seeds).split(",") if s.strip()]
    per_seed: List[lab.Result] = []
    spent = 0.0
    for s in seeds:
        log(f"  種 {s}")
        left = (args.budget_seconds - spent) if args.budget_seconds else 0.0
        if args.budget_seconds and left <= 0:
            log("  予算切れ。残りの種は次の実行へ")
            break
        t0 = time.time()
        per_seed.append(T.walk_forward(
            df, cols, seed=s, max_context=args.max_context,
            n_estimators=args.n_estimators, batch_size=args.batch_size,
            budget_seconds=left, log=log))
        spent += time.time() - t0
    if not per_seed:
        raise SystemExit("TabICL の out-of-fold を作れませんでした")
    if len(per_seed) == 1:
        r = per_seed[0]
        r.name = "tabicl"
        return r
    return T.average_seeds(per_seed)


def align(results: Dict[str, lab.Result], log=print) -> Dict[str, lab.Result]:
    """
    全モデルを共通の (Code, Date) に揃える。

    TabICL が予算切れで一部の窓しか回せていないと、行数が違うまま
    表に並ぶことになる。同じ行の上で比べないと、相関も上位10%重複も
    「どの期間を見たか」の差になってしまう。
    """
    common = None
    for r in results.values():
        k = set(map(tuple, r.oof[["Code", "Date"]].to_numpy()))
        common = k if common is None else (common & k)
    sizes = {n: len(r.oof) for n, r in results.items()}
    if len(set(sizes.values())) == 1:
        return results
    log(f"  行数が揃っていません {sizes} -> 共通の {len(common):,}行に揃えます")
    out = {}
    for n, r in results.items():
        key = pd.MultiIndex.from_frame(r.oof[["Code", "Date"]])
        mask = np.array([t in common for t in key], dtype=bool)
        o = r.oof[mask].reset_index(drop=True)
        out[n] = lab.Result(r.name, o, lab.metrics(o), r.per_seed)
        out[n].metrics.update({k: v for k, v in r.metrics.items()
                               if k.startswith("_")})
    return out


def report(results: Dict[str, lab.Result], args, df, log=print) -> str:
    """結果を Markdown にする。docs/MODEL_TABICL.md に残す。"""
    tic = results["tabicl"]
    names = list(results)
    lines: List[str] = []
    A = lines.append

    A("# TabICLv2 の検証")
    A("")
    A("実験17（`research/exp/e17_tabicl.py`）の結果。"
      "既存5モデルと同じウォークフォワード・同じ運用指標で測った。")
    A("")
    A(T.DESIGN_MD)
    A("## 条件")
    A("")
    A("| 項目 | 値 |")
    A("| --- | --- |")
    A(f"| チェックポイント | `{T.ckpt_version()}` |")
    A(f"| 特徴量 | {len(F.columns(args.features))}列（`{args.features}`）|")
    A(f"| 窓 | 訓練36ヶ月以上 / 検証6ヶ月 / 6ヶ月刻み / "
      f"エンバーゴ60営業日 |")
    A(f"| out-of-fold | {tic.metrics['n']:,}行 |")
    A(f"| 文脈の上限 | "
      f"{'なし（全訓練行）' if args.max_context <= 0 else f'{args.max_context:,}行（直近側）'} |")
    A(f"| アンサンブル数 | {args.n_estimators}（同時 {args.batch_size}）|")
    A(f"| TabICL の種 | {args.tabicl_seeds} |")
    A(f"| 既存5モデルの種 | {', '.join(str(s) for s in BASELINE_SEEDS)} |")
    sec = tic.metrics.get("_seconds") or 0.0
    A(f"| TabICL の所要時間 | {sec / 60:.0f}分"
      f"（この実行で回した窓のみ。保存済みを読んだ窓は含まない）|")
    A("")
    A("### 条件の非対称性")
    A("")
    A("- **探索**: 既存5モデルは Optuna 50試行ずつ探索済み。TabICL は既定値のまま"
      "（調整なしで効くことが売りなので、探索して勝たせるのは主旨から外れる）。"
      "TabICL に不利な向き。")
    A("- **種平均**: 既存5モデルは3種平均。"
      + (f"TabICL は {len(str(args.tabicl_seeds).split(','))}種。"
         if len(str(args.tabicl_seeds).split(",")) < 3 else "TabICL も同数。")
      + " 種平均はそれ自体がアンサンブルなので、既存5モデル側に有利な向き。")
    if args.max_context > 0:
        A(f"- **文脈**: TabICL は直近 {args.max_context:,}行しか見ていない。"
          "既存5モデルは全訓練行（最大19,726行）で学習する。TabICL に不利な向き。"
          "絞ったのは好みではなくメモリの制約による。実測で文脈1行あたり"
          "736.9KB 使い、全訓練行なら 17.6GB になる。CI ランナーは16GB なので"
          "そもそも載らない。")
    A("")

    A("## 運用指標")
    A("")
    A("しきい値運用（スコアが過去分布の上位10%なら買う・翌営業日の寄り買い・"
      "40営業日後の5日平均終値売り）。しきい値は窓ごとに過去の窓だけから決める。")
    A("")
    A("```")
    A(lab.table(results))
    A("```")
    A("")
    A("### 再現性（窓平均で見る。足切りは窓SD/√窓数）")
    A("")
    A("| モデル | 窓平均 | 標準誤差 | 勝ち窓 | 最悪の窓 | PR-AUC |")
    A("| --- | ---: | ---: | :---: | ---: | ---: |")
    for n, r in results.items():
        m = r.metrics
        se = m["thr_fold_sd"] / np.sqrt(max(1, m["thr_folds"]))
        ja = T.JA if n == "tabicl" else M.JA.get(n, n)
        A(f"| {ja} | {m['thr_fold_mean']:+.2f}pt | {se:.2f} | "
          f"{m['thr_folds_won']}/{m['thr_folds']} | {m['thr_worst']:+.2f}pt | "
          f"{m['pr_auc']:.4f} |")
    A("")

    A("## 並べる価値（スコア相関と上位10%の重複）")
    A("")
    A("相関が高い組は画面で2本ぶんの場所を取るだけで、情報が増えない。"
      "RF を外したときの基準は lgbm との相関 0.858。")
    A("")
    A("| 組 | Spearman 相関 | 上位10%の重複 |")
    A("| --- | ---: | ---: |")
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho, jac, n_ = E16.overlap(results[a].oof, results[b].oof)
            ja_a = T.JA if a == "tabicl" else M.JA.get(a, a)
            ja_b = T.JA if b == "tabicl" else M.JA.get(b, b)
            pairs.append((a, b, rho, jac))
            A(f"| {ja_a} × {ja_b} | {rho:.3f} | {jac * 100:.1f}% |")
    A("")

    # 判定。数字から機械的に導ける部分だけを書く
    A("## 読み取り")
    A("")
    tm = tic.metrics
    se = tm["thr_fold_sd"] / np.sqrt(max(1, tm["thr_folds"]))
    A(f"- TabICLv2 の窓平均は **{tm['thr_fold_mean']:+.2f}pt**"
      f"（標準誤差 {se:.2f} / 勝ち窓 {tm['thr_folds_won']}/{tm['thr_folds']}）。")
    others = [n for n in names if n != "tabicl"]
    if others:
        best = max(others, key=lambda n: results[n].metrics["thr_fold_mean"])
        bm = results[best].metrics
        A(f"- 既存モデルの最良は {M.JA.get(best, best)} の "
          f"{bm['thr_fold_mean']:+.2f}pt。差は "
          f"{tm['thr_fold_mean'] - bm['thr_fold_mean']:+.2f}pt。"
          "実験11 のノイズ床（窓平均のレンジ 0.143pt）と比べて読むこと。")
    tic_pairs = [(b, rho, jac) for a, b, rho, jac in pairs if a == "tabicl"] \
        + [(a, rho, jac) for a, b, rho, jac in pairs if b == "tabicl"]
    if tic_pairs:
        # 絶対値で取る。強い負の相関は「順位が逆」なだけで、
        # 情報としては同じものを見ていることになる
        mx = max(tic_pairs, key=lambda t: abs(t[1]))
        A(f"- 既存モデルと最も似ているのは {M.JA.get(mx[0], mx[0])}（相関 "
          f"{mx[1]:.3f} / 上位10%重複 {mx[2] * 100:.1f}%）。"
          "絶対値が 0.858（RF を外した基準）を超えていれば、"
          "並べても情報は増えない。")
    if tm.get("_stopped_at") is not None:
        A(f"- **未完**: 予算切れで窓{tm['_stopped_at']} 以降を回していない。"
          "同じワークフローをもう一度走らせると残りの窓から続く。")
    A("")
    A("---")
    A("")
    A(f"生成: `python3 research/exp/e17_tabicl.py`"
      f"（{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}）")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TabICLv2 を既存5モデルと比べる")
    ap.add_argument("--features", default="all")
    ap.add_argument("--max-context", type=int, default=T.DEFAULT_MAX_CONTEXT,
                    help="文脈に入れる訓練行の上限（直近側）。0 で全行")
    ap.add_argument("--n-estimators", type=int, default=T.DEFAULT_N_ESTIMATORS,
                    help="TabICL 内部のアンサンブル数")
    ap.add_argument("--batch-size", type=int, default=T.DEFAULT_BATCH_SIZE,
                    help="同時に流すアンサンブル数。メモリの上限を決める")
    ap.add_argument("--tabicl-seeds", default="42",
                    help="TabICL の種（カンマ区切り）")
    ap.add_argument("--budget-seconds", type=float, default=0.0,
                    help="TabICL に割く秒数。超えたら窓の途中で打ち切る")
    ap.add_argument("--baselines", default=",".join(M.ALGOS),
                    help="比較する既存モデル。空にすると TabICL だけ")
    ap.add_argument("--probe", action="store_true",
                    help="所要時間の実測だけして終わる")
    ap.add_argument("--probe-sizes", default="2000,6000,12000")
    ap.add_argument("--mem-limit-gb", type=float, default=0.0,
                    help="使えるメモリの上限。超えたら MemoryError を上げて "
                         "窓の途中で止める。0 で無効")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--placeholder", action="store_true",
                    help="データ無しで設計だけ docs/MODEL_TABICL.md に書く")
    args = ap.parse_args(argv)

    if args.placeholder:
        # 実行前のブランチでも、何を試そうとしているのかが読めるようにする。
        # 本走が終われば report() が同じ前書きごと上書きする
        os.makedirs(os.path.dirname(REPORT), exist_ok=True)
        with open(REPORT, "w", encoding="utf-8") as fh:
            fh.write("# TabICLv2 の検証\n\n"
                     "実験17（`research/exp/e17_tabicl.py`）の設計。\n"
                     "**結果はまだ入っていない。**"
                     "`research/tabicl_request.txt` の1行目を `run` にして "
                     "push すると、Experiment TabICLv2 ワークフローが回り、"
                     "このファイルが実測値で上書きされる。\n\n"
                     + T.DESIGN_MD + "\n"
                     "## 結果\n\n未実行。\n")
        print(f"書き出し: {REPORT}")
        return 0

    if args.mem_limit_gb > 0:
        ok = T.limit_memory(args.mem_limit_gb)
        print(f"メモリ上限 {args.mem_limit_gb}GB "
              f"{'を掛けた' if ok else 'は掛けられなかった'}")

    cols = F.columns(args.features)
    df = lab.frame()
    print(f"データセット {len(df):,}行 / 特徴量 {len(cols)}列 "
          f"/ ラベル確定 {int(df['label'].notna().sum()):,}行")
    print(f"チェックポイント {T.ckpt_version()} "
          f"(TABICL_CKPT={os.environ.get('TABICL_CKPT') or '未設定（HF から取得）'})")
    print(f"いまのメモリ {T.rss_gb():.2f}GB")
    print()

    if args.probe:
        print("=== 所要時間とメモリの実測（本走と同じ条件）===")
        print(f"  アンサンブル{args.n_estimators} / 同時{args.batch_size} / "
              "検証は本番の窓をそのまま使う")
        sizes = [int(s) for s in args.probe_sizes.split(",") if s.strip()]
        p = T.probe(df, cols, sizes=sizes, n_estimators=args.n_estimators,
                    batch_size=args.batch_size)
        os.makedirs(lab.DATA_DIR, exist_ok=True)
        with open(os.path.join(lab.DATA_DIR, "e17_probe.json"), "w") as fh:
            json.dump(p, fh, ensure_ascii=False, indent=2)
        print()
        if p.get("failed_at"):
            print(f"  文脈{p['failed_at']:,}行でメモリ不足。"
                  "本走はこれより小さい上限で回すこと")
        if "seconds_per_context_row" not in p:
            print("  2点そろわなかったので外挿できない")
            return 1
        print(f"  文脈1行あたり {p['seconds_per_context_row'] * 1000:.2f}ミリ秒 "
              f"/ 固定費 {p['fixed_seconds']:.1f}秒")
        print(f"  文脈1行あたり {p['gb_per_context_row'] * 1e6:.1f}KB "
              f"/ データ読み込みぶん {p['baseline_rss_gb']:.2f}GB")
        print(f"  全訓練行（{p['n_train_full']:,}行）まで上げたときの"
              f"最大メモリ見積もり {p['peak_rss_gb_at_full']:.1f}GB"
              "（CI ランナーは16GB）")
        bend = [(n, p[k]) for n, k in (("時間", "curvature_seconds"),
                                       ("メモリ", "curvature_gb")) if k in p]
        if bend:
            print("  傾きの伸び "
                  + " / ".join(f"{n} {v:.2f}倍" for n, v in bend)
                  + "（1.0 なら線形。大きいほど外挿は過小評価）")
        print()
        for mc in (0, 4000, 8000, 12000, 16000):
            est = T.estimate_total(df, p, max_context=mc,
                                   n_estimators=args.n_estimators, n_seeds=1)
            if est:
                n_ctx = max(f["n_context"] for f in est["per_fold"])
                gb = (p["baseline_rss_gb"]
                      + p["gb_per_context_row"] * n_ctx
                      + (p["points"][0]["tabicl_gb"]
                         - p["gb_per_context_row"] * p["points"][0]["n_context"]))
                print(f"    文脈上限{'全行' if mc == 0 else f'{mc:,}':>7}"
                      f"（最大の窓 {n_ctx:,}行）-> 1種 {est['hours']:.1f}時間 "
                      f"/ 最大メモリ {gb:.1f}GB")
        print()
        print("  注: 文脈の行数に線形と仮定した外挿。上の「傾きの伸び」が")
        print("      1.0 より大きければ、実際はこれより掛かる")
        return 0

    results: Dict[str, lab.Result] = {}

    print("=== TabICLv2 ===")
    print(f"  文脈上限 {'全行' if args.max_context <= 0 else f'{args.max_context:,}行'}"
          f" / アンサンブル {args.n_estimators} / 種 {args.tabicl_seeds}")
    results["tabicl"] = run_tabicl(df, cols, args)
    print()

    algos = [a for a in args.baselines.split(",") if a.strip()]
    if algos:
        print("=== 既存モデル（実験16 と同じ条件）===")
        results.update(run_baselines(df, cols, algos, BASELINE_SEEDS))
        print()

    results = align(results)

    print(lab.table(results))
    print()
    md = report(results, args, df)
    print(md)

    os.makedirs(lab.DATA_DIR, exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as fh:
        json.dump({n: {k: v for k, v in r.metrics.items()
                       if not isinstance(v, (list, dict))}
                   for n, r in results.items()}, fh,
                  ensure_ascii=False, indent=2, default=str)
    if not args.no_report:
        os.makedirs(os.path.dirname(REPORT), exist_ok=True)
        with open(REPORT, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"\n書き出し: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
