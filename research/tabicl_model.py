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

    extra = {}
    # 列方向の埋め込みをどこに置くか。既定（auto）は TabICL が空きメモリを
    # 見て決めるが、こちらが RLIMIT_AS で上限を掛けていることは見えないので、
    # 機械の 16GB を基準に「まだ載る」と判断してしまう。
    # disk を明示すればメモリマップに逃がせる。遅くなる代わりに
    # 文脈を絞らずに済む。既定は触らない（TABICL_OFFLOAD が無ければ auto）
    off = os.environ.get("TABICL_OFFLOAD")
    if off:
        extra["offload_mode"] = off
        if off == "disk":
            extra["disk_offload_dir"] = (os.environ.get("TABICL_OFFLOAD_DIR")
                                         or os.path.join(lab.DATA_DIR, "offload"))
            os.makedirs(extra["disk_offload_dir"], exist_ok=True)

    return TabICLClassifier(
        n_estimators=n_estimators,
        batch_size=max(1, min(batch_size, n_estimators)),
        random_state=seed,
        n_jobs=n_jobs,
        device="cpu" if os.environ.get("TABICL_FORCE_CPU") else None,
        model_path=ckpt_path(),
        checkpoint_version=ckpt_version(),
        verbose=verbose,
        **extra,
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
        import gc
        gc.collect()
        reset_peak_rss()
        t0 = time.time()
        try:
            clf = build(seed, n_estimators=n_estimators, batch_size=batch_size)
            clf.fit(to_frame(ctx, cols), ctx["label"].to_numpy(dtype=int))
            score = np.asarray(clf.predict_proba(to_frame(te, cols)),
                               dtype=float)[:, 1]
        except (MemoryError, RuntimeError) as e:
            # ここで止めて、そこまでの窓を返す。メモリ上限を掛けてあれば
            # ランナーごと落ちずに済み、保存済みの窓は次の実行で活きる
            stopped = f.index
            log(f"  窓{f.index} メモリ不足で打ち切り "
                f"({type(e).__name__}: {str(e)[:120]})")
            log(f"          文脈{len(ctx):,} / 検証{len(te):,} / "
                f"最大 {peak_rss_gb():.2f}GB")
            break
        dt = time.time() - t0
        peak = peak_rss_gb()

        keep = ["Code", "Date"] + [c for c in lab.OUT_COLS if c in te.columns]
        part = te[keep].copy()
        part["score"] = score
        part["fold"] = f.index
        part.to_parquet(p, index=False, compression="zstd")
        parts.append(part)
        timings.append(FoldTiming(f.index, len(ctx), len(te), dt))
        log(f"  窓{f.index} {f.test_start.date()}〜{f.test_end.date()}: "
            f"文脈{len(ctx):,} / 検証{len(te):,} / {dt:.0f}秒 / "
            f"最大 {peak:.2f}GB")
        del clf
        gc.collect()

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

def _vm_gb(key: str) -> float:
    """/proc/self/status から VmRSS / VmHWM を GB で読む。"""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith(key):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("nan")


def reset_peak_rss() -> bool:
    """
    メモリの最高水位（VmHWM）を現在値まで下げる。

    なぜ要るか
    ---------
    最高水位はプロセスを通した高水位線なので、先に lab.frame() が
    株価バー全期間を読んで 5.4GB まで上げてしまうと、そのあと TabICL が
    何GB 使ったのかが埋もれて見えない。実際、それで「文脈を増やしても
    メモリが増えない」という誤った読みをして本走をメモリ不足で落とした。

    Linux は /proc/self/clear_refs に 5 を書くと最高水位をリセットする。
    使えない環境（他OS・権限なし）では False を返すので、
    呼び出し側は「リセットできなかった」として読む。
    """
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as fh:
            fh.write("5\n")
        return True
    except OSError:
        return False


def peak_rss_gb() -> float:
    """最後のリセット以降でこのプロセスが使ったメモリの最大値（GB）。"""
    return _vm_gb("VmHWM:")


def rss_gb() -> float:
    """いま使っているメモリ（GB）。"""
    return _vm_gb("VmRSS:")


def limit_memory(gb: float) -> bool:
    """
    使えるメモリに上限を掛ける。

    なぜ要るか
    ---------
    上限が無いと、確保しすぎたときに**ランナーごと**落ちる。実測では
    exit 143（SIGTERM）でジョブが死に、`if: always()` を付けたはずの
    「途中結果を Release に戻す」まで飛ばされ、18分ぶんの計算が消えた。

    上限を掛けておけば、超えた時点で Python 側に MemoryError が上がる。
    そこまでに終わった窓は保存済みなので、次の実行が続きから走れる。

    RLIMIT_AS は仮想アドレス空間の上限で、torch は実体より広く確保する
    ことがある。低く取りすぎると動くはずのものが落ちるので、
    既定は無効（0）にして、実行側が明示したときだけ掛ける。
    """
    if gb <= 0:
        return False
    import resource

    # torch を先に読み込んでから掛ける。torch は起動時に実体より広い
    # アドレス空間を確保するので、読み込む前に上限を掛けると
    # 「本来なら動くのに import で落ちる」ことがある
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    n = int(gb * 1024 ** 3)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,
                           (n, hard if hard != resource.RLIM_INFINITY
                            else resource.RLIM_INFINITY))
        return True
    except (ValueError, OSError):
        return False


def probe(df: pd.DataFrame, cols: Sequence[str], *,
          sizes: Sequence[int] = (2000, 6000, 12000),
          n_estimators: int = DEFAULT_N_ESTIMATORS,
          batch_size: int = DEFAULT_BATCH_SIZE,
          seed: int = lab.SEED, log=print) -> Dict:
    """
    最後の窓で、文脈の行数を変えて fit+predict の時間とメモリを実測する。

    なぜ先に測るか
    -------------
    TabICL は GPU 前提の設計（H100 で 50,000行×100列を10秒未満）で、
    CPU で何倍掛かるかは公表されていない。全10窓を回してから
    「6時間で終わらなかった」と分かるのは、CI の実行1回を捨てることになる。

    本走と同じ条件で測る（2026-09-14 の失敗を受けて）
    --------------------------------------------
    最初はアンサンブル2・同時2・検証500行で測り、本走はアンサンブル8・
    検証1,400行で回した。測った条件と回す条件が違えば見積もりは当たらない。
    実際 18分でランナーがメモリ不足に落ち、しかも最高水位が
    lab.frame() の 5.4GB に埋もれてメモリの増加が見えていなかった。

    いまは (1) 本走と同じ n_estimators / batch_size を使い、
    (2) 検証側も本番の窓をそのまま使い、(3) 1点ごとに最高水位を
    リセットしてから測る。

    1点ずつ出力する。途中でメモリ不足に落ちても、
    「どの行数までは通ったか」が残る。
    """
    folds = lab.folds(df)
    d = pd.to_datetime(df["Date"])
    labeled = df["label"].notna()
    f = folds[-1]
    tr = df[(d <= f.train_end) & labeled]
    # 検証側は本番の窓をそのまま使う。ここを小さくすると、予測側の
    # メモリと時間を過小に見積もることになる（前回それで落ちた）
    te = df[(d >= f.test_start) & (d <= f.test_end) & labeled]

    resettable = reset_peak_rss()
    out = {"n_train_full": int(len(tr)), "n_test_full": int(len(te)),
           "n_test_probe": int(len(te)), "n_estimators": n_estimators,
           "batch_size": batch_size, "peak_resettable": resettable,
           "baseline_rss_gb": rss_gb(), "points": []}
    if not resettable:
        log("  [warn] 最高水位をリセットできない。メモリの実測は "
            f"データ読み込みぶん（{out['baseline_rss_gb']:.2f}GB）を含む")

    for n in sizes:
        ctx = cap_context(tr, n)
        import gc
        gc.collect()
        reset_peak_rss()
        before = rss_gb()
        t0 = time.time()
        try:
            clf = build(seed, n_estimators=n_estimators, batch_size=batch_size)
            clf.fit(to_frame(ctx, cols), ctx["label"].to_numpy(dtype=int))
            clf.predict_proba(to_frame(te, cols))
        except (MemoryError, RuntimeError) as e:
            log(f"  文脈{len(ctx):,}行 : メモリ不足で失敗 ({type(e).__name__})")
            out["failed_at"] = int(len(ctx))
            break
        dt = time.time() - t0
        peak = peak_rss_gb()
        out["points"].append({"n_context": int(len(ctx)), "seconds": dt,
                              "peak_rss_gb": peak,
                              "tabicl_gb": max(0.0, peak - before)})
        log(f"  文脈{len(ctx):,}行 / 検証{len(te):,}行 / "
            f"アンサンブル{n_estimators} / 同時{batch_size} : "
            f"{dt:.1f}秒 / 最大 {peak:.2f}GB "
            f"(TabICL ぶん {max(0.0, peak - before):.2f}GB)")
        del clf
        gc.collect()

    pts = out["points"]
    if len(pts) >= 2:
        # 最初と最後の2点で直線を引く。切片は「文脈に依らない固定費」
        # （重みの読み込み・検証側の処理）に相当する
        (x0, y0), (x1, y1) = ((p["n_context"], p["seconds"])
                              for p in (pts[0], pts[-1]))
        slope = (y1 - y0) / max(1, (x1 - x0))
        out["seconds_per_context_row"] = slope
        out["fixed_seconds"] = y0 - slope * x0
        m0, m1 = (p["tabicl_gb"] for p in (pts[0], pts[-1]))
        gb_slope = (m1 - m0) / max(1, (x1 - x0))
        out["gb_per_context_row"] = gb_slope
        out["tabicl_gb_at_full"] = m1 + gb_slope * (out["n_train_full"] - x1)
        out["peak_rss_gb_at_full"] = (out["baseline_rss_gb"]
                                      + out["tabicl_gb_at_full"])
        # 3点以上あれば、線形の仮定が妥当かを見る。最後の区間の傾きが
        # 最初の区間より大きければ、外挿は過小評価になっている
        if len(pts) >= 3:
            def seg(a, b, k):
                return (pts[b][k] - pts[a][k]) / max(
                    1, pts[b]["n_context"] - pts[a]["n_context"])

            def ratio(k, floor):
                # 最初の区間の傾きがほぼ0だと比は意味を持たない
                # （0で割った大きな数が出るだけ）。測れないときは出さない
                base = seg(0, 1, k)
                if base < floor:
                    return None
                return seg(len(pts) - 2, len(pts) - 1, k) / base

            for key, k, floor in (("curvature_seconds", "seconds", 1e-4),
                                  ("curvature_gb", "tabicl_gb", 1e-8)):
                v = ratio(k, floor)
                if v is not None:
                    out[key] = v
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
