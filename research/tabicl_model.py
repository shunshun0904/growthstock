#!/usr/bin/env python3
"""
TabICLv2 を lab.py のウォークフォワードに載せるための薄い層。

TabICL とは何か
--------------
表形式データ向けの事前学習済み Transformer（soda-inria/tabicl、
PyPI `tabicl`）。ICML 2026 の TabICLv2 が既定のチェックポイント
`tabicl-classifier-v2-20260212.ckpt`。

木や線形モデルと決定的に違うのは、**このデータでは重みを学習しない**こと。
fit() は訓練データを手元に置くだけで、predict() のときに訓練行と検証行を
まとめて1回 Transformer に通し、文脈内学習（in-context learning）で答えを
出す。合成データで事前学習された「表を読む能力」をそのまま当てる。

なぜ並べる価値がありそうか
------------------------
既存5モデルは全部このデータから学習する。lgbm と logit のスコア相関が
0.412 しかないのは関数クラスの違いによるもので、それでも「同じ
22,896行から何を学ぶか」という点では同じ土俵にいる。TabICL は
学習そのものをしないので、誤りの出方がさらに違うことが期待できる。
それが本当かは exp/e17_tabicl.py が測る（相関と上位10%重複）。

なぜ models.py に入れないか
-------------------------
models.ALGOS は画面・日次予測・週次学習が読む正本で、ここに足すと
検証前に本番へ入ることになる。TabICL は predict のたびに訓練行ぜんぶを
Transformer に通すので、日次予測の所要時間が既存モデルと桁で違う
可能性がある。まず research で測り、入れるかどうかはその結果で決める。

実行環境の要件
------------
チェックポイント（重み）は Hugging Face の jingang/TabICL にしか無く、
pip では入らない。したがって実行環境が huggingface.co に到達できるか、
あるいは落としたファイルを TABICL_CKPT で渡せることが要る。

  TABICL_CKPT          チェックポイントのパス。既定は None（HF から取得）
  TABICL_CKPT_VERSION  v1 / v1.1 と比べたいときに切り替える
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import tuning_multi as TM  # noqa: E402

#: 画面に出すときの名前。まだ本番には入れないが、実験の表で使う
JA = "TabICLv2"
SHORT = "TIC"
NOTE = ("表形式の事前学習済み Transformer。このデータでは重みを学習せず、"
        "訓練行を文脈として1回通して答えを出す")

#: 検証レポート（docs/MODEL_TABICL.md）の前書き。
#:
#: ここを正本にして exp/e17_tabicl.py が埋め込む。実行前に置く
#: プレースホルダと、実行後に上書きされる本番のレポートで、
#: 「なぜ試すか」の説明が食い違わないようにするため。
DESIGN_MD = """\
## TabICLv2 とは

表形式データ向けの事前学習済み Transformer（[soda-inria/tabicl](https://github.com/soda-inria/tabicl)、
PyPI `tabicl`）。ICML 2026 の TabICLv2 が既定のチェックポイント。

木や線形モデルと決定的に違うのは、**このデータでは重みを学習しない**こと。
`fit()` は訓練データを手元に置くだけで、`predict()` のときに訓練行と検証行を
まとめて1回 Transformer に通し、文脈内学習（in-context learning）で答えを出す。
合成データで事前学習された「表を読む能力」をそのまま当てる。

## なぜ試すか

既存5モデルは全部このデータから学習する。LightGBM とロジスティック回帰の
スコア相関が 0.412 しかないのは関数クラスの違いによるもので、それでも
「同じ22,896行から何を学ぶか」という点では同じ土俵にいる。
TabICL は学習そのものをしないので、誤りの出方がさらに違うことが期待できる。

アンサンブルはしない運用なので、単体の強さより**既存5モデルと違う銘柄を
選ぶか**が採否の分かれ目になる。相関が高ければ、画面で1本ぶんの場所を
取るだけで情報は増えない（RF を外したときの基準は LightGBM との相関 0.858）。

## 運用に入れる前に確かめること

TabICL は `predict` のたびに訓練行ぜんぶを Transformer に通す。
日次予測の所要時間が既存モデルと桁で違う可能性があるため、
**まず research で測り、本番（画面・週次学習・日次予測）に入れるかどうかは
この結果で決める**。`research/models.py` の `ALGOS` にはまだ入れていない。
"""

#: 既定のチェックポイント。tabicl 2.2.0 の既定と同じものを明示的に持つ。
#: パッケージが更新されて既定が変わったときに、黙って別の重みで
#: 測っていた、という事故を避ける
DEFAULT_CKPT_VERSION = "tabicl-classifier-v2-20260212.ckpt"

#: 文脈に入れる訓練行の上限。0 なら全行。
#:
#: TabICL の計算量は文脈の行数に効く。GPU 前提の設計なので、CPU では
#: 全22,896行を毎窓通すと現実的でない可能性がある。上限を掛けるときは
#: **直近側を残す**（古い相場から捨てる）。ランダムに間引くと窓ごとに
#: 別の期間を見ることになり、窓間の比較ができなくなる。
DEFAULT_MAX_CONTEXT = 0

#: アンサンブル数。TabICL 内部で特徴量の並べ替え・正規化を変えた
#: 複数ビューを平均する。既定8はパッケージの既定と同じ。
#: CPU ではここが素直に線形に効くので、予算が足りなければ下げる
DEFAULT_N_ESTIMATORS = 8

#: 同時に処理するアンサンブル数。メモリの上限を決めるのはここ。
#:
#: 列方向の埋め込みが (batch, 行数, 列数, 埋め込み次元) の実体を持つ。
#: 22,896行 × 151列 で埋め込み次元を128とすると、1バッチあたり
#: 約1.8GB。パッケージの既定 8 をそのまま使うと14GB になり、
#: 16GB の CI ランナーでは落ちる。CPU では速度より先にメモリが効くので
#: 小さく取る。offload_mode="auto" が CPU/ディスクへ退避する経路も
#: あるが、それに頼ると退避のぶんさらに遅くなる。
DEFAULT_BATCH_SIZE = 2

#: 1窓ぶんの out-of-fold を置く場所。窓ごとに書き出して、
#: 途中で打ち切られても次の実行が続きから走れるようにする。
#: CI のジョブ上限は6時間で、TabICL の所要時間は事前に読めない
FOLD_DIR = os.path.join(lab.DATA_DIR, "oof", "tabicl")


def ckpt_path() -> Optional[str]:
    """TABICL_CKPT が指すファイル。無ければ None（Hugging Face から取る）。"""
    p = os.environ.get("TABICL_CKPT") or None
    if p and not os.path.exists(p):
        raise SystemExit(f"TABICL_CKPT が指すファイルがありません: {p}")
    return p


def ckpt_version() -> str:
    return os.environ.get("TABICL_CKPT_VERSION") or DEFAULT_CKPT_VERSION


def build(seed: int = lab.SEED, *, n_estimators: int = DEFAULT_N_ESTIMATORS,
          batch_size: int = DEFAULT_BATCH_SIZE,
          n_jobs: int = -1, verbose: bool = False):
    """
    TabICLClassifier を組む。

    既定値から動かすのは4つだけにする。

      random_state    種平均のため。ここを変えると特徴量の並べ替えが変わる
      n_estimators    CPU の予算に合わせて下げられるように
      batch_size      メモリの上限（DEFAULT_BATCH_SIZE の説明を参照）
      n_jobs          CI ランナーの論理コアを使い切る（既定は物理コア数）

    正規化・外れ値処理・softmax 温度は既定のまま。TabICL の売りは
    「調整なしで効く」ことなので、ここを触ると何を測っているのか
    分からなくなる。既存5モデルは Optuna で50試行ずつ探索してあるので、
    条件としてはむしろ TabICL に不利な比較になる。それは結果に添えて書く。

    batch_size と n_jobs は精度に影響しない（何個まとめて流すか、
    何スレッドで回すか）ので、ここを予算に合わせて動かしても
    「測っているもの」は変わらない。n_estimators だけは結果が変わる。
    """
    from tabicl import TabICLClassifier

    return TabICLClassifier(
        n_estimators=n_estimators,
        batch_size=max(1, min(batch_size, n_estimators)),
        random_state=seed,
        n_jobs=n_jobs,
        device="cpu" if os.environ.get("TABICL_FORCE_CPU") else None,
        model_path=ckpt_path(),
        checkpoint_version=ckpt_version(),
        verbose=verbose,
    )


# --------------------------------------------------------------------------- #
# 入力の形
# --------------------------------------------------------------------------- #

def to_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    TabICL に渡す形にする。

    numpy 配列ではなく DataFrame で渡す。TabICL は DataFrame のときだけ
    列の型を見て、カテゴリ列を OrdinalEncoder、数値列を平均補完に回す
    （tabicl/_sklearn/preprocessing.py の TransformToNumerical）。
    float 配列で渡すと s33_code=5250 が 1050 の5倍の量として
    正規化に掛かる。logit と MLP で one-hot にしているのと同じ理由。

    無限大は欠損に倒す。補完は平均（SimpleImputer の既定）なので、
    inf が1つでも混じると列ぜんぶが inf になる。比率から作った列
    （per / pbr / *_yield など）は分母が0に近いと inf を出しうる。
    """
    out = df.loc[:, list(cols)].copy()
    for c in cols:
        if c in TM.CATEGORICAL:
            # 整数コードだが順序に意味がない列。category にしておくと
            # TabICL 側が 0..k-1 の密なコードに振り直す
            out[c] = out[c].astype("category")
        else:
            s = pd.to_numeric(out[c], errors="coerce")
            out[c] = s.replace([np.inf, -np.inf], np.nan).astype(float)
    return out


def cap_context(tr: pd.DataFrame, max_context: int) -> pd.DataFrame:
    """
    文脈に入れる訓練行を直近 max_context 行に絞る。

    古い側から捨てるのは、相場のレジームが近いほど文脈として効く
    という前提より前に、**窓をまたいで比較できる形にする**ため。
    ランダムに間引くと窓ごとに違う期間を見ることになり、
    「窓9 だけ良い」が期間のせいなのか偶然なのか分からなくなる。

    0 以下なら何もしない。
    """
    if max_context <= 0 or len(tr) <= max_context:
        return tr
    d = pd.to_datetime(tr["Date"])
    return tr.loc[d.sort_values(kind="mergesort").index[-max_context:]]


# --------------------------------------------------------------------------- #
# ウォークフォワード
# --------------------------------------------------------------------------- #

@dataclass
class FoldTiming:
    fold: int
    n_train: int
    n_test: int
    seconds: float


def _fold_path(seed: int, fold: int, max_context: int, n_estimators: int) -> str:
    return os.path.join(
        FOLD_DIR,
        f"f{fold:02d}_s{seed}_c{max_context}_e{n_estimators}.parquet")


def walk_forward(df: pd.DataFrame, cols: Sequence[str], *,
                 seed: int = lab.SEED,
                 max_context: int = DEFAULT_MAX_CONTEXT,
                 n_estimators: int = DEFAULT_N_ESTIMATORS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 budget_seconds: float = 0.0,
                 resume: bool = True,
                 log=print) -> lab.Result:
    """
    lab.run と同じ窓で out-of-fold を作る。

    lab.run をそのまま使わないのは2つ理由がある。

      1. 文脈の上限を掛けるのに Date が要る。lab.run の fit ファクトリには
         (X, y) しか渡らないので、日付を見て直近だけ残すことができない。
      2. 窓ごとに書き出したい。TabICL の所要時間は事前に読めず、CI の
         ジョブ上限（6時間）に収まる保証がない。途中で打ち切られても
         次の実行が残りの窓から続けられるようにする。

    窓の切り方・足切り（検証200行・訓練1000行）・持ち回す列は lab.run と
    同じものを使う。ここがずれると既存5モデルと比較できない。

    budget_seconds を超えたら、そこまでの窓で打ち切って返す。
    足りない窓は次の実行で埋まる。
    """
    cols = list(cols)
    os.makedirs(FOLD_DIR, exist_ok=True)
    d = pd.to_datetime(df["Date"])
    labeled = df["label"].notna()

    parts: List[pd.DataFrame] = []
    timings: List[FoldTiming] = []
    t_start = time.time()
    stopped = None

    for f in lab.folds(df):
        tr = df[(d <= f.train_end) & labeled]
        te = df[(d >= f.test_start) & (d <= f.test_end) & labeled]
        if len(te) < lab.MIN_TEST or len(tr) < lab.MIN_TRAIN:
            continue

        p = _fold_path(seed, f.index, max_context, n_estimators)
        if resume and os.path.exists(p):
            parts.append(pd.read_parquet(p))
            log(f"  窓{f.index} 保存済みを読む ({len(parts[-1]):,}行)")
            continue

        if budget_seconds and (time.time() - t_start) > budget_seconds:
            stopped = f.index
            log(f"  窓{f.index} 予算切れで打ち切り "
                f"({time.time() - t_start:.0f}秒 > {budget_seconds:.0f}秒)")
            break

        ctx = cap_context(tr, max_context)
        t0 = time.time()
        clf = build(seed, n_estimators=n_estimators, batch_size=batch_size)
        clf.fit(to_frame(ctx, cols), ctx["label"].to_numpy(dtype=int))
        score = np.asarray(clf.predict_proba(to_frame(te, cols)),
                           dtype=float)[:, 1]
        dt = time.time() - t0

        keep = ["Code", "Date"] + [c for c in lab.OUT_COLS if c in te.columns]
        part = te[keep].copy()
        part["score"] = score
        part["fold"] = f.index
        part.to_parquet(p, index=False, compression="zstd")
        parts.append(part)
        timings.append(FoldTiming(f.index, len(ctx), len(te), dt))
        log(f"  窓{f.index} {f.test_start.date()}〜{f.test_end.date()}: "
            f"文脈{len(ctx):,} / 検証{len(te):,} / {dt:.0f}秒")

    if not parts:
        raise SystemExit("out-of-fold を1窓も作れませんでした")

    oof = pd.concat(parts, ignore_index=True)
    res = lab.Result(name=f"tabicl#{seed}", oof=oof, metrics=lab.metrics(oof))
    # 後で報告に使うので、所要時間と打ち切りの有無を結果に付けておく。
    # 「速いモデルと同じ表に並べる」のはこの数字を伏せると誤解を生む
    res.metrics["_seconds"] = sum(t.seconds for t in timings)
    res.metrics["_folds_run"] = len(timings)
    res.metrics["_folds_total"] = len(parts)
    res.metrics["_stopped_at"] = stopped
    res.metrics["_max_context"] = max_context
    res.metrics["_n_estimators"] = n_estimators
    return res


def average_seeds(results: Sequence[lab.Result], name: str = "tabicl") -> lab.Result:
    """
    種ごとの out-of-fold を平均する。lab.run_multi と同じ扱い。

    確率の単純平均。同じ学習器・同じ出力なのでスケールが揃っている。
    """
    base = results[0].oof
    key = [c for c in (["Code", "Date", "fold"] + list(lab.OUT_COLS))
           if c in base.columns]
    acc = base[key].copy()
    # 窓が揃っていない種があると行数が食い違う。揃っている窓だけで平均する
    common = set(base["fold"].unique())
    for r in results[1:]:
        common &= set(r.oof["fold"].unique())
    acc = acc[acc["fold"].isin(common)].reset_index(drop=True)
    stack = []
    for r in results:
        o = r.oof[r.oof["fold"].isin(common)]
        o = o.set_index(["Code", "Date"]).loc[
            pd.MultiIndex.from_frame(acc[["Code", "Date"]])]
        stack.append(o["score"].to_numpy())
    acc["score"] = np.mean(stack, axis=0)
    out = lab.Result(name=name, oof=acc, metrics=lab.metrics(acc),
                     per_seed=list(results))
    out.metrics["_seconds"] = sum(r.metrics.get("_seconds", 0.0) for r in results)
    out.metrics["_max_context"] = results[0].metrics.get("_max_context")
    out.metrics["_n_estimators"] = results[0].metrics.get("_n_estimators")
    return out


# --------------------------------------------------------------------------- #
# 所要時間の実測
# --------------------------------------------------------------------------- #

def peak_rss_gb() -> float:
    """このプロセスがこれまでに使ったメモリの最大値（GB）。"""
    import resource
    # Linux の ru_maxrss は KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def probe(df: pd.DataFrame, cols: Sequence[str], *,
          sizes: Sequence[int] = (1000, 3000),
          n_estimators: int = 2, batch_size: int = DEFAULT_BATCH_SIZE,
          seed: int = lab.SEED, log=print) -> Dict:
    """
    最後の窓で、文脈の行数を変えて fit+predict の時間を実測する。

    なぜ先に測るか
    -------------
    TabICL は GPU 前提の設計（H100 で 50,000行×100列を10秒未満）で、
    CPU で何倍掛かるかは公表されていない。全10窓を回してから
    「6時間で終わらなかった」と分かるのは、CI の実行1回を捨てることになる。

    2点で測って外挿する。文脈の行数に対して線形と仮定した見積もりなので、
    注意（attention）が行数に二乗で効く部分があれば**過小評価になる**。
    そのことは呼び出し側で明示して使う。
    """
    folds = lab.folds(df)
    d = pd.to_datetime(df["Date"])
    labeled = df["label"].notna()
    f = folds[-1]
    tr = df[(d <= f.train_end) & labeled]
    te = df[(d >= f.test_start) & (d <= f.test_end) & labeled]
    # 検証側も切る。予測時間は検証行数にも効くので、全部通すと
    # 「文脈を変えた効果」に検証側の時間が定数で乗って傾きが読めない
    te_probe = te.iloc[:500]

    out = {"n_train_full": int(len(tr)), "n_test_full": int(len(te)),
           "n_test_probe": int(len(te_probe)), "n_estimators": n_estimators,
           "batch_size": batch_size, "points": []}
    for n in sizes:
        ctx = cap_context(tr, n)
        t0 = time.time()
        clf = build(seed, n_estimators=n_estimators, batch_size=batch_size)
        clf.fit(to_frame(ctx, cols), ctx["label"].to_numpy(dtype=int))
        clf.predict_proba(to_frame(te_probe, cols))
        dt = time.time() - t0
        # メモリは文脈の行数に比例して増える。全行に上げたときに
        # CI ランナー（16GB）に収まるかは、ここの実測から見積もる
        rss = peak_rss_gb()
        out["points"].append({"n_context": int(len(ctx)), "seconds": dt,
                              "peak_rss_gb": rss})
        log(f"  文脈{len(ctx):,}行 / 検証{len(te_probe):,}行 / "
            f"アンサンブル{n_estimators} / 同時{batch_size} : "
            f"{dt:.1f}秒 / 最大メモリ {rss:.2f}GB")

    pts = out["points"]
    if len(pts) >= 2:
        # 単純な2点直線。切片は「文脈に依らない固定費」（重みの読み込み・
        # 検証側の処理）に相当する
        (x0, y0), (x1, y1) = ((p["n_context"], p["seconds"]) for p in pts[:2])
        slope = (y1 - y0) / max(1, (x1 - x0))
        out["seconds_per_context_row"] = slope
        out["fixed_seconds"] = y0 - slope * x0
        # メモリも2点の直線で外挿する。全行に上げたときの見積もりを出して
        # おかないと、6時間走らせた末に OOM で落ちるのがいちばん高くつく
        m0, m1 = (p["peak_rss_gb"] for p in pts[:2])
        gb_slope = (m1 - m0) / max(1, (x1 - x0))
        out["gb_per_context_row"] = gb_slope
        out["peak_rss_gb_at_full"] = m1 + gb_slope * (out["n_train_full"] - x1)
    return out


def estimate_total(df: pd.DataFrame, probe_out: Dict, *,
                   max_context: int, n_estimators: int,
                   n_seeds: int = 1) -> Dict:
    """
    probe の結果から、全窓を回したときの所要時間を見積もる。

    文脈の行数に線形、アンサンブル数に線形、検証行数に線形と仮定する。
    どれも実際には最適化やバッチ処理でずれるので、桁を見るための数字。
    """
    slope = probe_out.get("seconds_per_context_row")
    fixed = probe_out.get("fixed_seconds")
    if slope is None:
        return {}
    scale_e = n_estimators / max(1, probe_out["n_estimators"])
    scale_t = 1.0  # 検証行数ぶんは下で窓ごとに掛ける
    d = pd.to_datetime(df["Date"])
    labeled = df["label"].notna()
    total = 0.0
    per_fold = []
    for f in lab.folds(df):
        tr = df[(d <= f.train_end) & labeled]
        te = df[(d >= f.test_start) & (d <= f.test_end) & labeled]
        if len(te) < lab.MIN_TEST or len(tr) < lab.MIN_TRAIN:
            continue
        n_ctx = len(cap_context(tr, max_context))
        # 固定費は検証行数に比例する部分が主なので、probe の検証行数で割って掛け直す
        fix = fixed * (len(te) / max(1, probe_out["n_test_probe"]))
        sec = (slope * n_ctx + max(0.0, fix)) * scale_e * scale_t
        per_fold.append({"fold": f.index, "n_context": n_ctx,
                         "n_test": int(len(te)), "seconds": sec})
        total += sec
    return {"per_fold": per_fold, "seconds": total * n_seeds,
            "hours": total * n_seeds / 3600.0, "n_seeds": n_seeds}
