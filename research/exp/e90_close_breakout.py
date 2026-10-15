#!/usr/bin/env python3
"""
実験90: 候補の入り口（サンプルに入る条件）を「終値 > これまでの終値の最大」にしたとき、母集団・モデル・今の規則の
成績がどう変わるか（実験89 と同じ手順。腕は「母集団」の違い）。

運用者の指摘（2026-10-15）: 買取王国（10/14 の候補）は日中の高値だけが78週高値を超え、終値は前の高値より下で引けた
（行ってこい）。「サンプルにはいる条件も終値判定の方がよい気がしました。そこは変更によりどの程度影響があるか、
まずは検証したいです」。定義は運用者の選択「終値 ＞ これまでの終値の最大」（両側を終値にする）。
運用者の方針「いきなり変えるよりも、まずは検証してほしいです。本番導入するかはその結果次第です」→ 本番の設定
（build_dataset の既定・研究の設定ファイル）は変えない。

母集団（build_dataset の BREAKOUT_REF / BREAKOUT_ON_HIGH。実験の中で書き換えて build() を呼ぶ）
  high   いまの本番: 日中の高値 > それまでの78週（368営業日）の日中の高値の最大（high / True）
  close  終値 > それまでの78週の終値の最大（close / False）。78週高値の基準（high52w・high52w_prior）も終値の最大に
         なるので、r_high（更新日は常に 100）・break_margin（終値の最大からの上抜け幅）・base_length の意味も変わる。
         列の名前と数（239列）は同じ
  どちらも連続した更新日は直前20営業日に更新が無い日だけ（BREAKOUT_COOLDOWN）。ラベルの定義は同じ

腕
  A  high の母集団で探索・学習（いまの本番）
  B  close の母集団で探索し直し → 入り口を変えたときに日曜の週次実行がやること
  C  close の母集団を A のパラメータで（探索なし）→ 母集団を変えたことそのものの効果
  P  対照: high の母集団から close の母集団と同じ件数を無作為に抜いた行で学習し（種 SUBSET_SEED）、予測は high の
     全行に出す（A のパラメータ）→ 「学習の行が減っただけ」の幅。§7 の「対照 P の改善を上回る」に使う
  C−A が「母集団を変えることの効果」、P−A が「行数が減っただけの幅」、B−A が「再学習で起きること」、B−C が「探索し直しの揺れ」

ラベル（--k。1回の実行で1つ）
  1.5  到達しきい値 1.5σ（既定）。今の規則「1.5σ の3本がすべて95以上」のモデルのラベル → 判定はこちら
  1.2  1.2σ。本番の基準モデル（画面の帯・順位・シート）のラベル → 参考

行
  学習は各腕の母集団の全行。モデルの比較（探索の CV・OOF・32窓・§7）は **共通の行**（両方の母集団にある銘柄×日付）で
  行う（回す前に決めた）。母集団が違う腕を同じ行で比べるため。規則の成績（§6）は各腕の自分の母集団で測る（入り口が
  変われば候補そのものが変わるので）。

判定（回す前に決めた。2026-10-15）
  (i)  モデル: docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）を共通の行・ラベル 1.5σ で当てる。木3モデルのうち
       2つ以上で、窓ごとの PR-AUC の差 C−A の平均が正・対照 P−A の平均を上回る・C が上の窓が過半（n//2 + 1 以上）
       → 「終値の母集団で学習するほうが、共通の行の順位付けが良い」
  (ii) 規則: 各腕の自分の母集団で、前向きの百分位（テスト窓より前の窓の OOF の分布。実験88 と同じ）が 1.5σ の3本
       すべて95以上の行の、+10% の指値・届かなければ20営業日目の終値（tp10c）の1取引の平均。B − A の窓ごとの差の
       平均が正で B が上の窓が過半、かつ B の月あたりの規則の行の数が A の半分以上 → 「入り口を変えても規則の成績は
       落ちない」
  (iii) 軽い代わり: A の規則の行のうち共通の行（= 入り口は変えずに、規則に「終値も前の終値の最大を超えた日」を足す）の
       成績が B と同等以上（1取引の平均の差が B の SE の範囲）なら、再学習より規則の条件を足すほうが軽い
  (i) と (ii) の両方を満たせば入り口の変更を運用者に出す。片方だけ・どちらも満たさないときは、結果を見せて運用者が
  決める（(iii) が立てば規則の条件だけを出す）。

  運用者の指示（2026-10-15、lgbm の回の途中）「時間がかかるので、比較にはlgbmだけで良いです」→ xgb・cat は回さない。
  (i) は lgbm の1票（「2つ以上」は満たせないので、lgbm が §7 を満たすかどうかを出す）。(ii)(iii) の規則「1.5σ の3本すべて
  95以上」は、回したモデルだけの前向きの百分位（lgbm だけなら lgbm が95以上）で代える。

結果（2026-10-15。run 38729598853 表 / 38731224835 lgbm / 38732935819 規則。docs/MODEL_ADOPTION_RULES.md §50）
  母集団: high 21,949行・close 21,674行（正例率 13.0% / 13.1%）で、共通の行は 11,525（約半分。同じ銘柄の更新日が前後に
  ずれる）。high だけの行（行ってこい など）の正例率 13.8% は共通の行の 12.3% より低くない。
  (i)  lgbm: 32窓の PR-AUC の C−A −0.0017 ± 0.0028（上の窓 16/32）・P−A −0.0027 → §7 を満たさない（対照と区別がつかない）
  (ii) 規則（lgbm の1本が 95 以上）: 1取引の平均 A +2.51%（708行・月 13.4）/ B +2.09%（802行・月 15.1）/ C +1.87% / P +2.40%。
       B−A の窓ごとの差 −0.22 ± 0.96pt（上の窓 2/9）→ 満たさない
  (iii) A の規則の行のうち共通の行 309件 +2.18% 対 B +2.09%（SE 0.29）→ 同等以上で立つ。ただし A のそれ以外の行（行ってこい
       など 399件）は +2.77% で共通の行より良く、規則に終値の条件を足すと A から −0.18 ± 0.88pt（上の窓 2/9）
  → 入り口の変更も規則の条件の追加も、いまの A を上回らない。本番は変えない（決めるのは運用者）。

出すもの: 母集団の違い（件数・年別・正例率・共通の行）、探索の CV、本番と同じ OOF（共通の行）、32窓の差・§7、窓ごとの値、
規則の成績（各腕の母集団・A の規則の行を共通/それ以外に分けた成績 = 行ってこいの行の成績・取引の形・月の候補数）。
公開ログには件数・割合・日付・精度だけ（銘柄・値は出さない）。

    exp=e90_close_breakout.py args="--dry"                （母集団の違いだけ。4〜5分）
    exp=e90_close_breakout.py args="--k 1.5 --algos lgbm"  （運用者の指示で lgbm だけ。§6・§7 は回したモデルで出る）
    （xgb・cat・--k 1.2 も同じ形で回せる）
試運転: exp=e90_close_breakout.py args="--tag e90smoke --algos lgbm --n-trials 2 --shifts 0"

Run Experiment は同じ実験を同時に1本しか走らせず、待ちに置けるのも1本だけなので、1本ずつ起動する。表（2つの母集団）は
1回目に作って research/_data/oof/<tag>_* に置き、以降の回は actions/cache で引き継いだそれを使う。--budget-min
（既定 290分）を過ぎたら新しい計算を始めずに終わる（保存済みの探索・out-of-fold は次の回が続きから使う）。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import walkforward as WF  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e87_monthly_return_search as E87  # noqa: E402
import e88_breaks_returns as E88  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e68_edinet_ab import judge7, keys_of, on_rows, pair, params_hash, short  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV       # 本番の retrain-weekly.yml と同じ（year_cap_date）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
ALGOS = ("lgbm", "xgb", "cat", "logit")
#: §7 の票は木3モデル。logit は同じ表に並べて参考にする
JUDGE_ALGOS = ("lgbm", "xgb", "cat")
SEEDS = {"lgbm": (42, 7, 123), "xgb": (42, 7, 123), "cat": (42, 7, 123), "logit": (42,)}
PROD_SEED = 42
POPS = ("high", "close")
#: 母集団の定義（build_dataset の BREAKOUT_REF, BREAKOUT_ON_HIGH）
POP_DEF = {"high": ("high", True), "close": ("close", False)}
POP_JA = {"high": "high（日中の高値 > 日中の高値の最大。いまの本番）", "close": "close（終値 > 終値の最大）"}
ARMS = ("A", "B", "C", "P")
POP_OF = {"A": "high", "B": "close", "C": "close", "P": "high"}
PARAMS_OF = {"A": "A", "B": "B", "C": "A", "P": "A"}
LABELS = {"A": "A いまの母集団（high）・A の探索", "B": "B 終値の母集団（close）・探索し直し",
          "C": "C 終値の母集団（close）・A のパラメータ",
          "P": "P 対照（high から close と同じ件数を無作為に抜いて学習・予測は high の全行）・A のパラメータ"}
PAIRS = (("A", "C"), ("A", "P"), ("P", "C"), ("A", "B"), ("C", "B"))
#: 対照の無作為の抜き方の種（結果を見る前に決めた）
SUBSET_SEED = 20261015
MIN_WINDOW_ROWS = 100
KS = (1.5, 1.2)
#: 判定に使う評価の行（回す前に決めた）
PRIMARY_ROWS = "common"
#: 規則（画面と同じ）: 1.5σ の3本の前向きの百分位がすべて RULE_LINE 以上。出口は +TP% の指値・届かなければ20営業日目の終値
RULE_LINE = LT.AGREE_PCT
TP = LT.TAKE_PROFIT
MIN_BREAK = LT.SKIP_BREAKS
#: 前向きの百分位に要る前の窓の数（実験79・88 と同じ）
MIN_TRAIN_WINDOWS = E79.MIN_TRAIN_WINDOWS
#: 窓ごとの差に要る規則の行の数（実験80・82 と同じ）
MIN_GROUP = E82.MIN_GROUP
#: (iii) の「同等」: 1取引の平均の差が B の SE 何倍まで
EQUIV_SE = 1.0
TAG = "e90"
RUN = "e90_k15"
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fpath(name: str) -> str:
    """表（ラベルによらない）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def path(name: str) -> str:
    """探索・out-of-fold・要約（ラベルごと）。"""
    return os.path.join(OOF_DIR, f"{RUN}_{name}")


def k_tag(k: float) -> str:
    return f"k{int(round(k * 10)):02d}"


def time_left() -> bool:
    global TIMED_OUT
    if DEADLINE is None or time.time() < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min を過ぎたので、新しい計算は始めない（保存済みの分だけで報告。続きは次の回）")
    return False


def with_label(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """k のラベルに引き直した表（1.2σ はデータセットのラベルのまま）。ラベルの決まらない行は落とす。"""
    if abs(k - B.VOL_NORM_K) < 1e-9:
        return df[df["label"].notna()].reset_index(drop=True)
    out = df.copy()
    out["label"] = B.rise_label_for_k(out, k)
    return out[out["label"].notna()].reset_index(drop=True)


def common_keys(fa: pd.DataFrame, fb: pd.DataFrame) -> pd.MultiIndex:
    """両方の母集団にある (Code, Date)。"""
    return keys_of(fa).intersection(keys_of(fb))


def in_keys(df: pd.DataFrame, keys: pd.MultiIndex) -> np.ndarray:
    return pd.MultiIndex.from_frame(df[["Code", "Date"]]).isin(keys)


# --------------------------------------------------------------------------- #
# データ（2つの母集団。1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def load_bars() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def with_outcomes(ds: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """lab.frame() と同じ実収益の列を付ける（同じ関数・同じ結合。実験47 と同じ）。"""
    import sweep_design as S

    ds = ds.copy()
    ds["Date"] = pd.to_datetime(ds["Date"])
    ds["Code"] = ds["Code"].astype(str)
    ref = S.reference_outcome(S.Panels(bars).get(B.HIGH_WINDOW))
    out = ds.merge(ref, on=["Code", "Date"], how="left")
    out = out.merge(lab.realized_returns(bars), on=["Code", "Date"], how="left")
    return out.reset_index(drop=True)


def build_population(pop: str, out_path: str) -> pd.DataFrame:
    """
    母集団 pop のデータセットを同じ生データから作る（保存済みなら読む）。build_dataset の既定は触らず、呼ぶ間だけ
    書き換える（実験47 の GENERAL_MARKET_START と同じやり方）。
    """
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)
    ref, on_high = POP_DEF[pop]
    saved = (B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH, dict(B.SWEEP_OVERRIDES))
    B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH = ref, on_high
    B.SWEEP_OVERRIDES["BREAKOUT_REF"] = ref
    B.SWEEP_OVERRIDES["BREAKOUT_ON_HIGH"] = on_high
    try:
        B.build(lab.DATA_DIR, out_path)
    finally:
        B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH = saved[0], saved[1]
        B.SWEEP_OVERRIDES.clear()
        B.SWEEP_OVERRIDES.update(saved[2])
    return pd.read_parquet(out_path)


def population_stamp(frames: dict) -> dict:
    """件数・年別・正例率（1.2σ）・共通の行（公開ログに出してよい数だけ）。"""
    fa, fb = frames["high"], frames["close"]
    ck = common_keys(fa, fb)
    st = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "common": int(len(ck)),
          "common_by_year": {str(y): int(n) for y, n in
                             sorted(pd.Series(ck.get_level_values("Date")).dt.year.value_counts().items())}}
    for pop, f in frames.items():
        yr = f["Date"].dt.year
        lab_ = f["label"]
        st[pop] = {"rows": int(len(f)), "labeled": int(lab_.notna().sum()),
                   "pos_rate_k12": float(lab_.mean()) if lab_.notna().any() else None,
                   "date_min": str(f["Date"].min().date()), "date_max": str(f["Date"].max().date()),
                   "rows_by_year": {str(y): int((yr == y).sum()) for y in sorted(yr.unique())},
                   "pos_by_year": {str(y): float(lab_[yr == y].mean()) if lab_[yr == y].notna().any() else None
                                   for y in sorted(yr.unique())},
                   "codes": int(f["Code"].nunique())}
    return st


def prepare(rebuild: bool = False):
    """2つの母集団の表（全行。ラベルは 1.2σ のまま）と記録。"""
    ps = fpath("stamp.json")
    paths = {pop: fpath(f"frame_{pop}.parquet") for pop in POPS}
    if not rebuild and os.path.exists(ps) and all(os.path.exists(p) for p in paths.values()):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・high {stamp['high']['rows']:,}行 / "
            f"close {stamp['close']['rows']:,}行 / 共通 {stamp['common']:,}行）")
        frames = {pop: pd.read_parquet(p) for pop, p in paths.items()}
        for f in frames.values():
            f["Date"] = pd.to_datetime(f["Date"])
            f["Code"] = f["Code"].astype(str)
        return frames, stamp
    if (B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH) != POP_DEF["high"]:
        raise SystemExit(f"build_dataset の既定が high / True ではない: {B.BREAKOUT_REF} / {B.BREAKOUT_ON_HIGH}"
                         "（SWEEP_* の環境変数が入っている？ いまの本番の母集団が作れない）")
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    fa = lab.frame(rebuild=True)              # ワークフローが既定で作った dataset.parquet（high）
    fa["Date"] = pd.to_datetime(fa["Date"])
    fa["Code"] = fa["Code"].astype(str)
    log(f"[data] high（いまの本番）: lab.frame {len(fa):,}行 × {fa.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    ds = build_population("close", fpath("dataset_close.parquet"))
    log(f"[data] close: build_dataset {len(ds):,}行 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    fb = with_outcomes(ds, load_bars())
    log(f"[data] close: 実収益を付けた {len(fb):,}行 × {fb.shape[1]}列 / {time.time()-t0:.0f}秒")
    miss = sorted(set(fa.columns) - set(fb.columns))
    if miss:
        raise SystemExit(f"close の表に無い列: {miss[:8]}")
    frames = {"high": fa, "close": fb[fa.columns]}
    stamp = population_stamp(frames)
    for pop, f in frames.items():
        f.to_parquet(paths[pop], index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return frames, stamp


def nearby(only: pd.DataFrame, other: pd.DataFrame, days: int = B.BREAKOUT_COOLDOWN) -> pd.DataFrame:
    """
    片方の母集団にしかない行（only）について、もう片方の母集団（other）に同じ銘柄の更新日が 直前 days 営業日 / 直後 days
    営業日（暦日に直して ±round(days × 営業日→暦日)）にあるか。before / after の列（真偽）を付けて返す。
    終値の基準（終値の最大）は日中の高値の最大より低いので、終値の母集団のほうが先に抜けることがある。その行は
    「直前に close の更新日がある」側に入り、連続更新をまとめる cooldown で片方だけになったもの。
    """
    tol = pd.Timedelta(days=int(round(days * WF.TRADING_TO_CALENDAR)))
    o = only[["Code", "Date"]].copy().sort_values("Date").reset_index()
    t = other[["Code", "Date"]].copy().sort_values("Date")
    t["_other"] = t["Date"]
    back = pd.merge_asof(o, t, on="Date", by="Code", direction="backward", tolerance=tol, allow_exact_matches=False)
    fwd = pd.merge_asof(o, t, on="Date", by="Code", direction="forward", tolerance=tol, allow_exact_matches=False)
    out = only.copy()
    out["before"] = back.set_index("index")["_other"].notna().reindex(only.index).to_numpy()
    out["after"] = fwd.set_index("index")["_other"].notna().reindex(only.index).to_numpy()
    return out


def population_report(frames_k: dict, stamp: dict, k: float) -> dict:
    """母集団の違い（件数・割合だけ）。"""
    print(f"\n■ 0. 母集団の違い（表は {stamp['built_utc']} 作成。ラベル {k:.1f}σ）")
    yrs = sorted(set(stamp["high"]["rows_by_year"]) | set(stamp["close"]["rows_by_year"]))
    out = {"label_k": k}
    for pop in POPS:
        f = frames_k[pop]
        st = stamp[pop]
        print(f"  {pop:<6} {POP_JA[pop]}")
        print(f"         全行 {st['rows']:,}（{st['date_min']} 〜 {st['date_max']}・{st['codes']:,}銘柄）/ "
              f"ラベル {k:.1f}σ の決まる行 {len(f):,}・正例率 {f['label'].mean()*100:.1f}%")
        print("         年別（行）: " + "  ".join(f"{y} {st['rows_by_year'].get(y, 0):,}" for y in yrs))
        out[pop] = {"rows": st["rows"], "labeled_k": int(len(f)), "pos_rate_k": float(f["label"].mean())}
    ck = common_keys(frames_k["high"], frames_k["close"])
    na, nb = len(frames_k["high"]), len(frames_k["close"])
    print(f"  共通の行（両方にある銘柄×日付。判定に使う）: {len(ck):,}（high の {len(ck)/na*100:.1f}% / "
          f"close の {len(ck)/nb*100:.1f}%）/ high だけ {na - len(ck):,} / close だけ {nb - len(ck):,}")
    cy = pd.Series(ck.get_level_values("Date")).dt.year.value_counts()
    print("         年別（共通）: " + "  ".join(f"{y} {int(cy.get(int(y), 0)):,}" for y in yrs))
    fa = frames_k["high"]
    inc = in_keys(fa, ck)
    print(f"  high の行の正例率: 共通の行 {fa.loc[inc, 'label'].mean()*100:.1f}%（{int(inc.sum()):,}件）/ "
          f"high だけの行（終値は前の終値の最大を超えていない、など） {fa.loc[~inc, 'label'].mean()*100:.1f}%"
          f"（{int((~inc).sum()):,}件）")
    out.update(common=int(len(ck)), only_high=int(na - len(ck)), only_close=int(nb - len(ck)),
               pos_rate_high_common=float(fa.loc[inc, "label"].mean()),
               pos_rate_high_only=float(fa.loc[~inc, "label"].mean()) if (~inc).any() else None)
    # 片方だけの行の内訳: もう片方の母集団に同じ銘柄の更新日が直前 / 直後の20営業日にあるか
    print(f"  片方だけの行の内訳（もう片方の母集団に同じ銘柄の更新日が 直前 / 直後 {B.BREAKOUT_COOLDOWN}営業日にあるか。"
          "件数と正例率）:")
    out["nearby"] = {}
    for pop, oth in (("high", "close"), ("close", "high")):
        f = frames_k[pop]
        only = f[~in_keys(f, ck)]
        if not len(only):
            continue
        nb_ = nearby(only, frames_k[oth])
        groups = (("直前にある（終値では先に抜けていた など）" if pop == "high" else "直前にある", nb_["before"] & ~nb_["after"]),
                  ("直後にある", ~nb_["before"] & nb_["after"]),
                  ("直前にも直後にもある", nb_["before"] & nb_["after"]),
                  ("どちらにもない（終値では抜けていない など）" if pop == "high" else "どちらにもない",
                   ~nb_["before"] & ~nb_["after"]))
        out["nearby"][pop] = {}
        for name, m in groups:
            n = int(m.sum())
            r = float(nb_.loc[m, "label"].mean()) if n else float("nan")
            out["nearby"][pop][name] = {"n": n, "pos_rate": r}
            print(f"    {pop} だけ {len(only):,}件のうち {name}: {n:,}（{n / len(only) * 100:.1f}%・正例率 "
                  f"{r * 100:.1f}%）" if n else f"    {pop} だけ {len(only):,}件のうち {name}: 0")
    return out


# --------------------------------------------------------------------------- #
# 探索（本番の週次実行と同じ関数・条件）
# --------------------------------------------------------------------------- #
def tune(algo: str, arm: str, frame: pd.DataFrame, cols: list, cutoff, n_trials: int,
         stamp: dict, compute: bool = True):
    p = path(f"params_{algo}_{arm}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if (rec.get("_n_trials") == n_trials and rec.get("_cutoff") == str(cutoff.date())
                and rec.get("_data") == stamp["built_utc"]):
            return rec
    if not compute or not time_left():
        return None
    sub = frame[(frame["Date"] <= cutoff) & frame["label"].notna()]
    t0 = time.time()
    if algo == "lgbm":
        log(f"  [{algo} {arm}] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列）")
        params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS,
                             embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME, objective=CV_OBJECTIVE, model="classifier",
                             verbose=False)
        rec = {"params": tuning.params_for("x", store={"x": params}), "_cv": dict(tuning.LAST_CV)}
    else:
        log(f"  [{algo} {arm}] 探索（tuning_multi.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列・"
            f"前処理 {TM.preprocess_version(algo) or 'なし'}）")
        TM.STUDY_DB = path(f"optuna_{arm}.db")
        rec = TM.tune(algo, sub, cols, n_trials=n_trials, n_splits=N_SPLITS, verbose=False, scheme=CV_SCHEME,
                      objective=CV_OBJECTIVE)
    rec["_seconds"] = round(time.time() - t0)
    rec["_n_trials"] = n_trials
    rec["_cutoff"] = str(cutoff.date())
    rec["_data"] = stamp["built_utc"]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    cv = rec["_cv"]
    log(f"  [{algo} {arm}] {rec['_seconds']}秒 / CV PR-AUC {cv['mean_pr_auc']:.4f}（±{cv['std']:.4f}）"
        f" / ROC {cv['mean_roc_auc']:.4f}")
    return rec


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #
def oof(algo: str, pop: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds,
        compute: bool = True, train_mask=None, mask_tag: str = ""):
    """母集団 × パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。train_mask は対照 P の学習の行。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(frame.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{algo}_p{pop}{mask_tag}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    missing = [f for f in files if not os.path.exists(f)]
    if missing and (not compute or not time_left()):
        return None
    folds = E41.folds_for(frame["Date"], shift) if missing else None
    parts = []
    for sd, f in zip(seeds, files):
        if os.path.exists(f):
            parts.append(pd.read_parquet(f))
            continue
        if not time_left():
            return None
        t0 = time.time()
        o = E41.oof_folds(algo, frame, cols, params, sd, folds, train_mask=train_mask)
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {algo} 母集団{pop}{mask_tag} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def windows_on(o: pd.DataFrame, keys: pd.MultiIndex | None) -> pd.DataFrame:
    """窓ごとの AUC と、窓の期間・件数・正例率。keys があればその行だけで、MIN_WINDOW_ROWS 未満の窓は落とす。"""
    cols = ["fold", "pr", "roc", "start", "end", "n", "pos"]
    sub = o if keys is None else on_rows(o, keys)
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)
    n = sub.groupby("fold").size()
    sub = sub[sub["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)
    w = AB.auc_by_window(sub)
    if not len(w):
        return pd.DataFrame(columns=cols)
    g = sub.groupby("fold")
    info = pd.DataFrame({"start": g["Date"].min(), "end": g["Date"].max(), "n": g.size(),
                         "pos": g["label"].mean()}).reset_index()
    return w.merge(info, on="fold", how="left")[cols]


def subset_mask(fa: pd.DataFrame, n_take: int, seed: int = SUBSET_SEED) -> np.ndarray:
    """対照 P の学習の行: high の母集団から n_take 行を無作為に（種を固定）。"""
    rng = np.random.default_rng(seed)
    m = np.zeros(len(fa), dtype=bool)
    m[rng.choice(len(fa), size=min(n_take, len(fa)), replace=False)] = True
    return m


def run(algo: str, frames: dict, keysets: dict, cols: list, cutoff, n_trials: int, stamp: dict,
        shifts: list, compute: bool, mask: np.ndarray) -> dict | None:
    recs = {}
    for arm in ("A", "B"):
        recs[arm] = tune(algo, arm, frames[POP_OF[arm]], cols, cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
    params = {arm: recs[PARAMS_OF[arm]]["params"] for arm in ARMS}

    def mask_of(arm):
        return (mask, "_sub") if arm == "P" else (None, "")

    prod = {rs: {} for rs in keysets}
    prod_full = {}
    for arm in ARMS:
        pop = POP_OF[arm]
        tm, tag = mask_of(arm)
        o = oof(algo, pop, frames[pop], cols, params[arm], 0, (PROD_SEED,), compute, tm, tag)
        if o is None:
            return None
        prod_full[arm] = o
        for rs, keys in keysets.items():
            prod[rs][arm] = o if keys is None else on_rows(o, keys)
    wins = {rs: {} for rs in keysets}
    for sh in shifts:
        for arm in ARMS:
            pop = POP_OF[arm]
            tm, tag = mask_of(arm)
            o = oof(algo, pop, frames[pop], cols, params[arm], sh, SEEDS[algo], compute, tm, tag)
            if o is None:
                return None
            for rs, keys in keysets.items():
                wins[rs][(sh, arm)] = windows_on(o, keys)
    return {"recs": recs, "params": params, "prod": prod, "prod_full": prod_full, "wins": wins}


def window_table(results: dict, algos: list, shifts: list, rs: str) -> pd.DataFrame:
    """窓ごとの値（モデル × ずらし × 窓）。4腕がそろった窓だけ。"""
    rows = []
    for a in algos:
        for sh in shifts:
            w = None
            for arm in ARMS:
                x = results[a]["wins"][rs][(sh, arm)]
                x = x.rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
                if arm != "A":
                    x = x[["fold", f"pr_{arm}", f"roc_{arm}"]]
                w = x if w is None else w.merge(x, on="fold")
            if w is None or len(w) == 0:
                continue
            for _, r in w.iterrows():
                rows.append({"algo": a, "shift": sh, "fold": int(r["fold"]), "start": r["start"], "end": r["end"],
                             "n": int(r["n"]), "pos": float(r["pos"]),
                             **{f"{m}_{arm}": float(r[f"{m}_{arm}"]) for m in ("pr", "roc") for arm in ARMS}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 規則の成績（各腕の自分の母集団。前向きの百分位 → 1.5σ の3本 95以上 → +10% の指値）
# --------------------------------------------------------------------------- #
def forward_pct(o: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    """
    fold（テスト窓）ごとに、それより前の窓の out-of-fold の分布で付けた百分位（画面と同じ丸め）。テスト窓の始まりの
    手前 RISE_HORIZON 営業日ぶんは結果が出ていないので外す。前の窓が MIN_TRAIN_WINDOWS 未満の窓は NaN。
    """
    out = pd.DataFrame(index=o.index)
    w = o["fold"].to_numpy()
    d = pd.to_datetime(o["Date"]).to_numpy()
    present = sorted(int(x) for x in np.unique(w))
    gap = pd.Timedelta(days=int(round(B.RISE_HORIZON * WF.TRADING_TO_CALENDAR)))
    for c in score_cols:
        out[f"pct_{c}"] = np.nan
    for ww in present:
        prior = [x for x in present if x < ww]
        te = w == ww
        if len(prior) < MIN_TRAIN_WINDOWS or not te.any():
            continue
        cut = np.datetime64(pd.Timestamp(d[te].min()) - gap)
        tr = np.isin(w, prior) & (d <= cut)
        if tr.sum() < 200:
            continue
        for c in score_cols:
            ref = np.sort(o.loc[tr, c].to_numpy(dtype=float))
            out.loc[te, f"pct_{c}"] = E82.rounded(E79.pct_within(ref, o.loc[te, c].to_numpy(dtype=float)))
    return out


def rule_frame(prods: dict, pop_frame: pd.DataFrame, bars: pd.DataFrame, ck: pd.MultiIndex) -> pd.DataFrame:
    """
    木モデル（prods にあるもの。本番と同じ OOF、ずらし0・種42）を (Code, Date) で突き合わせ、前向きの百分位・規則の最小・
    発火（その日の母集団の行数）・出口（tp10c）・共通の行の印を付けた表。前向きの百分位が付く窓の行だけ。
    規則の最小（min_K15）は prods にあるモデルの百分位の最小（3本そろえば画面の規則と同じ。運用者の指示で lgbm だけの
    ときは lgbm の百分位）。並べ替えのスコアは lgbm（無ければ最初のモデル）。
    """
    algos = [a for a in JUDGE_ALGOS if a in prods]
    if not algos:
        raise ValueError("木モデルの OOF が無い")
    o = None
    for a in algos:
        x = prods[a][["Code", "Date", "fold", "label", lab.OUTCOME, "score"]].rename(columns={"score": f"score_{a}"})
        if o is None:
            o = x
        else:
            o = o.merge(x[["Code", "Date", "fold", f"score_{a}"]], on=["Code", "Date", "fold"], how="inner")
    o = o.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    pct = forward_pct(o, [f"score_{a}" for a in algos])
    o["min_K15"] = pct[[f"pct_score_{a}" for a in algos]].min(axis=1)   # どれか欠ければ NaN = 満たさない
    o["score"] = o[f"score_{'lgbm' if 'lgbm' in algos else algos[0]}"]
    nb = pop_frame.groupby("Date").size()
    o["n_break"] = o["Date"].map(nb).fillna(0).astype(int)
    o, _ = E83.attach_exits(o, bars, targets=(TP,))
    o["common"] = in_keys(o, ck)
    o["window"] = o["fold"].astype(int)
    has = pct.notna().all(axis=1).to_numpy()
    return o[has].reset_index(drop=True)


def rule_stats(ft: pd.DataFrame, months: float) -> dict:
    s = E88.stats(ft)
    s["per_month"] = s["n"] / months if months else float("nan")
    cnt = ft.groupby(pd.to_datetime(ft["Date"]).dt.to_period("M")).size() if len(ft) else pd.Series(dtype=int)
    s["month_median"] = float(cnt.median()) if len(cnt) else 0.0
    s["by_window"] = {int(w): (int(len(x)), float(pd.to_numeric(x["tp10c"], errors="coerce").mean()))
                      for w, x in ft.groupby("window")}
    return s


def trade_form(ft: pd.DataFrame, bar_days, months: float) -> dict:
    """画面の形（今の規則・上位 TOP_K 件・同時に SLOTS 件・+TP%・発火 MIN_BREAK 件以上）で売買したときの成績。"""
    t = E87.run_one(ft, bar_days, "K15", RULE_LINE, MIN_BREAK, LT.TOP_K, LT.SLOTS, 0, TP)
    s = E88.stats(t)
    r = pd.to_numeric(t["tp10c"], errors="coerce").dropna()
    s["capital_pm"] = float(r.sum() / LT.SLOTS / months) if months else float("nan")
    return s


def paired_windows(a: dict, b: dict) -> dict:
    """窓ごとの規則の行の平均の差 b − a（両方に MIN_GROUP 行以上ある窓）。"""
    d = [b["by_window"][w][1] - a["by_window"][w][1] for w in a["by_window"]
         if w in b["by_window"] and a["by_window"][w][0] >= MIN_GROUP and b["by_window"][w][0] >= MIN_GROUP]
    return E79.summarize(np.array(d, dtype=float))


def strategy(results: dict, frames_k: dict, ck: pd.MultiIndex) -> dict | None:
    """各腕の規則の成績（結果のある木モデルで。1本も無ければ None）。"""
    avail = [a for a in JUDGE_ALGOS if results.get(a)]
    if not avail:
        return None
    t0 = time.time()
    start = min(f["Date"].min() for f in frames_k.values()) - pd.Timedelta(days=10)
    bars = LT.load_bars(lab.DATA_DIR, start=start)
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    out = {"models": avail}
    for arm in ARMS:
        pop = POP_OF[arm]
        ft = rule_frame({a: results[a]["prod_full"][arm] for a in avail}, frames_k[pop], bars, ck)
        months = E82.n_months(ft)
        rule = ft[ft["min_K15"] >= RULE_LINE]
        rec = {"pop": pop, "rows": int(len(ft)), "windows": sorted(int(w) for w in ft["window"].unique()),
               "months": months, "rule": rule_stats(rule, months),
               "rule_common": rule_stats(rule[rule["common"]], months),
               "rule_other": rule_stats(rule[~rule["common"]], months),
               "trade": trade_form(ft, bar_days, months)}
        out[arm] = rec
    out["pairs"] = {f"{y}-{x}": paired_windows(out[x]["rule"], out[y]["rule"]) for x, y in PAIRS}
    out["pairs"]["A_common-A"] = paired_windows(out["A"]["rule"], out["A"]["rule_common"])
    log(f"[rule] 規則の成績を測った（{', '.join(avail)}） / {time.time()-t0:.0f}秒")
    return out


def strategy_report(st: dict, k: float) -> dict:
    """§6（規則の成績）と §7（判定のまとめ）。"""
    models = st.get("models", list(JUDGE_ALGOS))
    rule_ja = ("1.5σ の3本すべて" if len(models) == 3 else
               f"{', '.join(models)} の{len(models)}本（運用者の指示で絞った。画面の規則「3本すべて」の代わり）が")
    print(f"\n■ 6. 規則の成績（各腕の自分の母集団。前向きの百分位（前の窓 {MIN_TRAIN_WINDOWS}つ以上）が {rule_ja} "
          f"{RULE_LINE:g} 以上の行。出口 +{TP:g}% の指値・届かなければ20営業日目の終値。手数料・税・すべりは含まない）")
    head = f"  {'':<34}{'件数':>6}{'月あたり':>8}{'1取引':>9}{'中央':>8}{'SE':>6}{'勝率':>7}{'到達':>7}{'−10%割れ':>9}"
    print(head)

    def line(name, s):
        return (f"  {LT._l(name, 34)}{s['n']:>6,}{s['per_month']:>8.1f}{s['mean']:>+8.2f}%{s['med']:>+7.2f}%"
                f"{s['se']:>6.2f}{s['win']:>6.1f}%{s['hit']:>6.1f}%{s['bad10']:>8.1f}%")

    for arm in ARMS:
        r = st[arm]
        print(f"  {LABELS[arm]}（前向きの行 {r['rows']:,}・窓 {r['windows'][0]}〜{r['windows'][-1]}・{r['months']}か月）")
        print(line(f"  規則の行", r["rule"]))
        print(line(f"  うち共通の行（終値も前の終値の最大を超えた日）", r["rule_common"]))
        print(line(f"  うちそれ以外（行ってこい など）", r["rule_other"]))
        t = r["trade"]
        print(f"  {LT._l('  取引の形（上位' + str(LT.TOP_K) + '件・' + str(LT.SLOTS) + '枠・先着）', 34)}{t['n']:>6,}"
              f"{'':>8}{t['mean']:>+8.2f}%{t['med']:>+7.2f}%{t['se']:>6.2f}{t['win']:>6.1f}%{t['hit']:>6.1f}%"
              f"{t['bad10']:>8.1f}%  資金全体の月 {t['capital_pm']:+.2f}%")
    print("  窓ごとの差（規則の行の平均。両方に " + str(MIN_GROUP) + " 行以上ある窓）:")
    for key, s in st["pairs"].items():
        print(f"    {key:<12} {E79.fmt_s(s)}")

    a, b, c = st["A"], st["B"], st["C"]
    d_ba = st["pairs"]["B-A"]
    ok_ret = bool(d_ba["n"] and d_ba["mean"] > 0 and d_ba["wins"] >= d_ba["n"] // 2 + 1)
    ok_count = bool(np.isfinite(a["rule"]["per_month"]) and b["rule"]["per_month"] >= a["rule"]["per_month"] * 0.5)
    ac = a["rule_common"]
    equiv = bool(np.isfinite(ac["mean"]) and np.isfinite(b["rule"]["mean"]) and np.isfinite(b["rule"]["se"])
                 and ac["mean"] >= b["rule"]["mean"] - EQUIV_SE * b["rule"]["se"])
    print(f"\n■ 7. 判定（回す前に決めた読み方。ラベル {k:.1f}σ）")
    print(f"  (ii) 規則: B − A の窓ごとの差 {E79.fmt_s(d_ba)} → {'○' if ok_ret else '×'} / "
          f"月あたりの規則の行 A {a['rule']['per_month']:.1f} → B {b['rule']['per_month']:.1f}"
          f"（A の半分以上 {'○' if ok_count else '×'}） → {'満たす' if ok_ret and ok_count else '満たさない'}")
    print(f"  (iii) 軽い代わり: A の規則の行のうち共通の行 {ac['n']:,}件 {ac['mean']:+.2f}% vs B {b['rule']['n']:,}件 "
          f"{b['rule']['mean']:+.2f}%（SE {b['rule']['se']:.2f}） → "
          f"{'同等以上（規則の条件を足すほうが軽い）' if equiv else '下回る'}")
    print(f"  参考: C − A {E79.fmt_s(st['pairs']['C-A'])} / A の規則の行のうち それ以外（行ってこい など）"
          f" {a['rule_other']['n']:,}件 {a['rule_other']['mean']:+.2f}%")
    return {"models": models, "rule_ok": ok_ret, "count_ok": ok_count, "filter_equiv": equiv,
            "arms": {arm: st[arm] for arm in ARMS}, "pairs": st["pairs"]}


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #
def report(results: dict, shifts: list, stamp: dict, keysets: dict, pop: dict, k: float,
           strat: dict | None = None) -> dict:
    algos = [a for a in ALGOS if results.get(a)]
    missing = [a for a in ALGOS if not results.get(a)]
    print("\n" + "=" * 78)
    print(f"ラベル {k:.1f}σ / 結果のそろったモデル: {', '.join(algos) or 'なし'}"
          + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, "population": pop, "label_k": k, "subset_seed": SUBSET_SEED,
               "primary_rows": PRIMARY_ROWS, "tuning": {}, "production_oof": {}, "windows": {}, "judge7": {},
               "per_window": {}, "strategy": None}
    titles = {"common": "共通の行（両方の母集団にある銘柄×日付）"}
    sec = {rs: "ab"[i] for i, rs in enumerate(keysets)}

    print(f"\n■ 1. 探索の CV（{CV_SCHEME}・{N_SPLITS}分割。楽観側に出る。パラメータ選び用。A は high・B は close の母集団）")
    print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC / パラメータ")
    for a in algos:
        r = results[a]
        for arm in ("A", "B"):
            cv = r["recs"][arm]["_cv"]
            fs = cv.get("fold_scores") or []
            print(f"  {a:<7}{arm:<4}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}"
                  f"{r['recs'][arm].get('_seconds', 0):>6}  " + " / ".join(f"{v:.4f}" for v in fs))
            summary["tuning"][f"{a}_{arm}"] = {kk: cv.get(kk) for kk in
                                               ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
        pa, pb = r["recs"]["A"]["params"], r["recs"]["B"]["params"]
        print(f"  {'':<7}選ばれたパラメータ: " + ("A と B で同じ" if params_hash(pa) == params_hash(pb)
                                              else f"A {short(pa)} / B {short(pb)}")
              + "（母集団が違うので CV の値そのものは比べない）")

    for rs in keysets:
        print(f"\n■ 2{sec[rs]}. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。{titles[rs]}）")
        print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
        for a in algos:
            ms = {}
            for arm in ARMS:
                o = results[a]["prod"][rs][arm]
                m = metrics(o)
                ms[arm] = m
                summary["production_oof"][f"{a}_{arm}_{rs}"] = {kk: float(v) for kk, v in m.items()}
                print(f"  {a:<7}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}"
                      f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
                      f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt"
                      f"{len(o):>8,}")
            for x, y in PAIRS:
                print(f"  {a:<7}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}"
                      f"{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}"
                      f"{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    tables = {}
    for rs in keysets:
        s = window_table(results, algos, shifts, rs)
        tables[rs] = s
        if not len(s):
            continue
        s.to_csv(path(f"auc_by_window_{rs}.csv"), index=False)
        print(f"\n■ 3{sec[rs]}. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り。"
              f"logit は種1つ、ほかは種3つの平均。{titles[rs]}・{MIN_WINDOW_ROWS}件以上の窓）")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for a in algos:
            g = s[s["algo"] == a]
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                for x, y in PAIRS:
                    line, stt = pair(f"{a} {nm} {y}−{x}", g[f"{met}_{x}"].to_numpy(), g[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{a}_{met}_{y}-{x}_{rs}"] = stt
        print(f"\n■ 4{sec[rs]}. §7（2026-09-26 改訂）の判定（窓の PR-AUC。{titles[rs]}"
              f"{'。判定に使う行（回す前に決めた）' if rs == PRIMARY_ROWS else '。参考'}）: "
              "平均 C−A が正 / 対照 P−A を上回る / C が上の窓が過半")
        for a in algos:
            g = s[s["algo"] == a]
            j = judge7((g["pr_C"] - g["pr_A"]).to_numpy(), (g["pr_P"] - g["pr_A"]).to_numpy())
            summary["judge7"][f"{a}_{rs}"] = j
            ref = "" if a in JUDGE_ALGOS else "（参考。§7 の票には入れない）"
            print(f"  {a:<7}C−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / "
                  f"P−A {j['mean_pa']:>+8.4f} {'○' if j['beats_placebo'] else '×'} / "
                  f"上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）{'○' if j['majority'] else '×'} "
                  f"→ {'満たす' if j['pass'] else '満たさない'}{ref}")
        judged = [a for a in algos if a in JUDGE_ALGOS]
        passed = sum(int(summary["judge7"][f"{a}_{rs}"]["pass"]) for a in judged)
        complete = all(a in algos for a in JUDGE_ALGOS)
        verdict = (("満たす（木3モデル中2つ以上）" if passed >= 2 else "満たさない（木3モデル中2つ未満）") if complete
                   else f"{passed}/{len(judged)} モデルが満たす（木3モデルがそろってから判定）")
        summary["judge7"][f"verdict_{rs}"] = {"passed": passed, "of": len(judged), "complete": complete,
                                              "reference": [a for a in algos if a not in JUDGE_ALGOS]}
        print(f"  → (i) §7（ラベル {k:.1f}σ・{titles[rs]}）: {verdict}")

    for rs in keysets:
        s = tables[rs]
        if not len(s):
            continue
        print(f"\n■ 5{sec[rs]}. 窓ごとの値（{titles[rs]}。PR-AUC。差は「後 − 前」。運用者の指示「各cv（異なる窓）でもみたい」）")
        for a in algos:
            for sh in shifts:
                g = s[(s["algo"] == a) & (s["shift"] == sh)].sort_values("fold")
                if not len(g):
                    continue
                print(f"  {a}・ずらし{sh}か月（{len(g)}窓）")
                print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}{'A':>8}{'B':>8}{'C':>8}{'P':>8}"
                      f"{'C−A':>9}{'P−A':>9}{'B−A':>9}")
                for _, r in g.iterrows():
                    print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                          f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%{r['pr_A']:>8.4f}{r['pr_B']:>8.4f}{r['pr_C']:>8.4f}"
                          f"{r['pr_P']:>8.4f}{r['pr_C'] - r['pr_A']:>+9.4f}{r['pr_P'] - r['pr_A']:>+9.4f}"
                          f"{r['pr_B'] - r['pr_A']:>+9.4f}")
        summary["per_window"][rs] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")

    if strat is not None:
        summary["strategy"] = strategy_report(strat, k)
    else:
        print(f"\n■ 6. 規則の成績: 木モデル（{', '.join(JUDGE_ALGOS)} のどれか）の結果がそろった回で出す")
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE
    ap = argparse.ArgumentParser(description="実験90: 候補の入り口を終値ベース（終値 > これまでの終値の最大）にしたときの影響")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算するモデル")
    ap.add_argument("--k", type=float, default=KS[0], help="ラベルの到達しきい値（σ の倍数。1.5 か 1.2）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0,
                    help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="母集団の違いだけ出して終わる")
    args = ap.parse_args(argv)
    if not any(abs(args.k - k) < 1e-9 for k in KS):
        raise SystemExit(f"--k は {KS} のどれか（実際: {args.k}）")
    TAG = args.tag
    RUN = f"{TAG}_{k_tag(args.k)}"
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    cols = F.columns(F.DEFAULT_PRESET)

    print("=" * 78)
    print("実験90 候補の入り口を「終値 > これまでの終値の最大」にしたときの影響（5分割 CV の探索 + OOF + 32窓・対照 P・規則の成績）")
    print(f"  列: {F.DEFAULT_PRESET} {len(cols)}列 指紋 {F.signature(cols)}（両方の母集団で同じ列）")
    for pop in POPS:
        print(f"  母集団 {POP_JA[pop]}")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  ラベル {args.k:.1f}σ / この回に計算: {', '.join(algos)} / 探索 {args.n_trials}試行 × {N_SPLITS}分割"
          f"（{CV_SCHEME}・{CV_OBJECTIVE}）/ 窓のずらし {shifts}か月 / 種 {SEEDS} / 対照の種 {SUBSET_SEED} / "
          f"判定の行 {PRIMARY_ROWS} / 規則 1.5σ の3本 {RULE_LINE:g} 以上・+{TP:g}% / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    frames, stamp = prepare(args.rebuild)
    frames_k = {}
    for pop, f in frames.items():
        miss = [c for c in cols if c not in f.columns]
        if miss:
            raise SystemExit(f"{pop} の表に無い列: {miss[:8]}")
        fk = with_label(f, args.k)
        fk.attrs["built_utc"] = stamp["built_utc"] + f" {pop}"   # out-of-fold の名前に表の版を入れる
        frames_k[pop] = fk
    pop_rep = population_report(frames_k, stamp, args.k)
    if args.dry:
        return 0
    cutoff, _, _ = T.holdout_bounds(frames_k["high"]["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    ck = common_keys(frames_k["high"], frames_k["close"])
    keysets = {"common": ck}
    mask = subset_mask(frames_k["high"], len(frames_k["close"]))
    print(f"\n  探索の期間（〜{cutoff.date()}）: high {(frames_k['high']['Date'] <= cutoff).sum():,}行 / "
          f"close {(frames_k['close']['Date'] <= cutoff).sum():,}行。対照 P の学習の行 {int(mask.sum()):,} / "
          f"{len(mask):,}（{mask.mean()*100:.1f}%）")

    results = {}
    for a in ALGOS:
        t0 = time.time()
        results[a] = run(a, frames_k, keysets, cols, cutoff, args.n_trials, stamp, shifts, compute=a in algos, mask=mask)
        if a in algos:
            log(f"[{a}] {'そろった' if results[a] else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    strat = strategy(results, frames_k, ck)
    report(results, shifts, stamp, keysets, pop_rep, args.k, strat)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
