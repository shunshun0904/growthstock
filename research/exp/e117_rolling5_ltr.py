#!/usr/bin/env python3
"""
実験117: ランキング学習を「直近5営業日の候補」の組で毎日並べ、今の規則（1.5σ の3本 95以上）を満たす上位だけを買う（検証だけ。
基準モデル・買いの規則・画面・本番のランキング学習は変えない）。

運用者の指摘（2026-10-29）
  「週次ランキングと言いましたが、厳密には、日次で、直近5営業日でのサンプル内でのランキング になるかと思います。こちらも、
    モデル自体は週次更新を想定してます。」→ AskUserQuestion: 毎日の買い方「上位2件のうち未購入を買う」
  「買うのは毎日ではなく、1.5σ95以上の条件を満たす必要もあるので、毎日ではないです。」→ AskUserQuestion: 条件の使い方
    「上位2件かつ条件を満たすもの」
実験115（金曜の夜に暦の週の候補を並べる形。docs/MODEL_ADOPTION_RULES.md §79）の作り直し。§79-6・§82。

毎日の組（判断日 t の夜に、t までの直近5営業日の候補を並べる）
  - 判断日 t: 取引日の暦の各日。組は、候補の日が t の4営業日前〜t の行（データセットの行 = 78週高値の更新日）。同じ銘柄が組の中に
    2回以上入っていれば、最後の候補の日の行を残す
  - 特徴量: 残した行の 360列（features.DEFAULT_PRESET。候補の日の値）＋候補の日から判断日までの値動き4列（実験115 の運用者の選択
    「使う」を同じ形で）
      r5_days  候補の日から判断日までの営業日数（取引日の暦で数える。0 = 判断日に入った候補。0〜4）
      r5_ret   候補の日の終値 → 判断日の終値の騰落（比。調整後終値）
      r5_high  候補の日の翌営業日〜判断日の高値の最大 ÷ 候補の日の終値 − 1（判断日に入った候補は欠測）
      r5_low   同じ期間の安値の最小 ÷ 候補の日の終値 − 1（同じ）
  - 収益: 判断日の翌営業日の寄りで買い、+10% の指値、届かなければ20営業日目の終値（今の売り方。live_track.forward に選定日 =
    判断日として渡すのと同じ）。学ぶ並びは組の中の「翌営業日の寄りで買い20営業日目の終値」の5分位（実験114 の L と同じ作りで、
    組が「その日」→「その日までの直近5営業日」）
  - 条件 R15（今日の戦略の規則）: 候補の日の 1.5σ の木3本（lgbm_k15・xgb_k15・cat_k15）の前向きの百分位（本番の out-of-fold を
    本番と同じ窓に分け、前の窓のスコアの分布での位置を画面と同じ小数1桁に丸めたもの。実験82 と同じ作り）がすべて95以上。候補ごとに
    1回決まり（候補の日の値）、どのずらし方でも同じものを使う
  - 毎日の買い方（運用者の選択「上位2件のうち未購入を買う」「上位2件かつ条件を満たすもの」）: 判断日ごとに組をスコアの上から並べ
    （同じスコアは銘柄コードの順）、上位2件のうち、R15 を満たし、まだ買っていないものを翌営業日の寄りで買う。上位2件より下には
    降りない（上位2件が R15 を満たさなければ、その日は買わない）。「まだ買っていない」は、
      - 同じ候補（銘柄・候補の日）を前の判断日に買っていない
      - その銘柄を持っていない（前に買った玉が終わっていない。+10% に k営業日目で届けば買った日から k営業日後、届かなければ
        20営業日後に終わる。live_track.simulate の枠の空き方と同じ数え方）
  - 数える判断日: R15 の印が組のすべての行に付いた日（印の無い候補が上位に来た日を「条件を満たさない」扱いにしないため。外した
    日数は表に出す）

腕（どれも同じ組を採点し、同じ買い方で買う）
  W5   毎日の組のランキング学習（LGBMRanker。組 = 判断日の直近5営業日の候補。360列＋値動き4列。種42）。木の形は実験114 の L と
       同じ探索（探索の期間は run_tuning.py と同じ打ち切り・種0・TPE・50試行・木200本・目的は検証分割の組の中の NDCG@5）。探索の
       5分割は、判断日を暦月の塊にまとめて年で層別し、塊ごとに分ける（同じ月の判断日は同じ側）。訓練側からは、検証側の判断日の
       前後4営業日の判断日を外す（同じ候補が両側の組に入らないように）
  W5X  参考: W5 から値動き4列を外したもの（360列だけ。木の形は W5 と同じ）。組の作りそのものと、値動きの列の効き目を分ける
  D    今のランキング学習（日次。research/ltr_params.json の木の形・種42。日ごとの組で学んだモデルで組の行を採点）
  B    基準モデル（1.2σ の LightGBM。lgbm_params.json の 360列の木の形。本番の out-of-fold と同じ当てはめ）
  R15  今の買い（腕ではなく比べる相手）: 候補の日に R15 を満たした候補を、候補の日の翌営業日の寄りで買う（持っている銘柄は
       買い増さない）
  窓: 基準モデルの out-of-fold と同じ作り（36か月 / 6か月 / 6か月、エンバーゴ20営業日。e41.folds_for）を判断日の日付で作り、
  0 / 2 / 4か月ずらした3通り。W5・W5X は判断日が窓の訓練の最終日以前の組の行で、D・B は候補の日が訓練の最終日以前の日次の行
  （本番と同じ）で学ぶ。採点と買いは判断日がテスト窓に入る組。本番では週1回学び直す想定だが、ここではほかのモデルの out-of-fold
  と同じく窓ごと（6か月ごと）に学び直す（週ごとに学び直すより古いモデルで採点する）

比べるもの（ずらし方ごと）
  1. 毎日の買い（上位2件のうち R15 を満たす未購入）と今の買い R15: 件数（月あたり）/ 平均収益（全件・窓の平均）/ −10% 未満
     （全件・窓の平均）/ +10% 到達 / 20営業日目の終値。W5 − R15・W5 − B・W5 − D・W5 − W5X の窓ごとの差。R15 を満たす候補の
     うち、上位2件に入って買ったものの件数。候補全体の平均（組の行すべてを判断日の翌営業日に買ったとき）
  2. 同じ買いを画面の3枠（同時に3銘柄まで・先着順・空きが無ければ見送る。live_track.simulate）で回した取引の件数・平均・資金全体の
     月あたり（1件に1/3）。今の買いはその日の上位2件（3本の最小の順。live_track.decide と同じ）
  3. 参考: 条件なしの上位2件のうち未購入（R15 を問わない）の件数・平均収益・−10% 未満
  4. 組の中の並び: NDCG@5（組の中の20営業日目の終値の段階）・組の中の順位相関（今の売り方 / 20営業日目の終値）
  5. 新しさ（実験115 の運用者の見立て「後から入った候補のほうが上に来やすいのでは」の毎日の形）: 候補の日からの営業日数（0〜4）
     ごとの組の行の割合・腕ごとの組の中の百分位の平均、買った候補の営業日数の割合と平均

読み方（回す前に決めたもの。採用の判定ではない。入れるかは運用者が決める）
  - 3通りのずらしすべてで、W5 の買い（上位2件かつ R15）の平均収益（窓の平均）が、今の買い R15・D の買い・B の買いのどれよりも
    高く、かつ −10% 未満の割合（窓の平均）がどれよりも低い →「直近5営業日の上位2件で絞る形は、今の買いや今のスコアで絞るより
    良い」。それ以外 →「はっきりとは良くない」。件数（月あたり）は並べて出すだけで、読み方には入れない
  - 新しさ: W5 の組の中の百分位の平均の「入った日の候補（0営業日）− 入ってから1〜4営業日の候補」（小数2桁に丸める）が、3通り
    すべてで +0.05 以上 →「入ったばかりの候補を上に置きやすい」、すべてで −0.05 以下 →「入ってから日がたった候補を上に置き
    やすい」、それ以外 →「はっきりとは寄らない」

確かめ（合わなければ止まる）
  - 組の行は (銘柄, 判断日) で1行・候補の日は判断日の0〜4営業日前・データセットの日付はすべて取引日
  - 無作為に選んだ判断日の組の銘柄と候補の日が、データセットから直接選び直したものと同じ
  - 判断日に入った候補（r5_days = 0）は、収益が日次の収益（実験114 の attach_exit）とすべて同じ（選定日が同じなので）
  - 値動き4列と r5_days を、無作為に選んだ300行で銘柄の日足から直接計算し直した値と同じ
  - 本番の out-of-fold（1.5σ の木3本）が置き場にある（無ければ R15 を作れないので止まる）
  - 参考に数えるもの: 収益がある行のうち、判断日の翌取引日に買えた割合（売買停止などで遅れる行がある）、R15 の印が付いた行の数

参考（--triple。運用者の確認 2026-10-30「当日のみでのランキングで1位、直近5営業日ランキングでも1位、1.5σモデルでも全部95以上の
銘柄のみ買ったら、収益がどうなるか確認したいです。そんなに候補はないはずです。」。見るだけ。§82-4）
  - 候補の日 t の夜に、その日の候補の中の今のランキング学習（日次。D）で1位・t の組の W5 で1位・R15 の3つとも満たす候補を、t の
    翌営業日の寄りで買う（持っている銘柄は買い増さない）。分けて、R15＋日次の1位・R15＋組の1位・R15 だけ・今の買いのうち3つとも
    満たすもの以外も出す。数える判断日は本体と同じ。件数・平均収益（全件 ± 標準誤差・窓の平均）・勝率・−10% 未満など
  - 続く確認（同じ日）「1.2σモデルでも95以上の条件だとどうなるか知りたいです。3条件、4条件全て満たすサンプルの集計をしてほしい
    です。」: 1.2σ の木3本（R0。実験115 と同じ作り）も足した4つとも満たす候補と、R15＋R0（順位は問わない）も出す。手元で同じ
    out-of-fold を作り直せるように、W5 の木の形を全桁で出す（データの値ではない）

公開ログには集計だけ（件数・平均・割合・木の形）。銘柄ごとの値は出さない。結果は research/_data/oof/e117_*。

    exp=e117_rolling5_ltr.py  args=""      （既定は --trials 50 --shifts 0,2,4 --top 2。探索の結果と out-of-fold は置き場に
                                            残し、同じデータ・同じ設定なら次の回は読む）
    exp=e117_rolling5_ltr.py  args="--triple"   （参考の表も出す）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import live_track as LT  # noqa: E402
import ltr as LTR  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e115_weekly_ltr as W  # noqa: E402

PREFIX = "e117"
OOF_DIR = E.OOF_DIR
PARAMS_PATH = os.path.join(OOF_DIR, f"{PREFIX}_params.json")
SUMMARY_PATH = os.path.join(OOF_DIR, f"{PREFIX}_summary.json")
RET, HIT, HITDAY, R20 = E.RET, E.HIT, E.HITDAY, E.R20
D_RET, D_HIT, D_R20 = W.D_RET, W.D_HIT, W.D_R20
K = 5                             # 組の長さ（運用者の指摘「直近5営業日」）
MOVE_COLS = ("r5_days", "r5_ret", "r5_high", "r5_low")
#: 組の行に足す列（データセットに同じ名前の列があれば止める）
ADDED = MOVE_COLS + (D_RET, D_HIT, D_R20, "cand_date", "buy_date")
TOP_N = 2                         # 毎日見る上位の件数（運用者の選択「上位2件のうち未購入を買う」「上位2件かつ条件を満たすもの」）
SEED = W.SEED                     # ランキング学習の種（本番と同じ 42）
LINE = W.LINE                     # 今の規則の線（95）
FRESH_LINE = 0.05                 # 新しさの読み方の線（組の中の百分位の平均の差）
BAD = E.BAD
MIN_TEST, MIN_TRAIN = W.MIN_TEST, W.MIN_TRAIN
N_CHECK = 300                     # 値動きの列を直接計算し直す行の数
N_POOL_CHECK = 60                 # 組の中身を直接選び直す判断日の数
ARMS = ("W5", "W5X", "D", "B")
LABELS = {"W5": "W5 直近5営業日の組", "W5X": "W5X 同じ組（値動きの列なし）",
          "D": "D 今のランキング学習（日次）", "B": "B 基準モデル（1.2σ）"}
#: 条件 R15（今日の戦略の規則。2026-10-29 から 1.5σ の木3本）。本番の out-of-fold（基準モデルと同じ窓 = ずらし0）を読む
RULE_TREES = tuple(LT.K15)
OBJECTIVE_JA = (f"検証分割の組の中の NDCG@{E.NDCG_K}（候補{E.MIN_DAY}件以上の組。利得は組の中の20営業日目の終値の段階 "
                f"0〜{E.N_GRADES - 1}）")

log = E.log


# --------------------------------------------------------------------------- #
# 毎日の組
# --------------------------------------------------------------------------- #

def rolling_frame(full: pd.DataFrame, bars: pd.DataFrame, days, cols: Sequence[str],
                  k: int = K) -> Tuple[pd.DataFrame, Dict]:
    """
    日次の候補（データセットの行。今の売り方の収益 RET・HIT・HITDAY・R20 が付いたもの）から毎日の組の行を作る。判断日 t ごとに、
    候補の日が t の (k−1) 営業日前〜t の行を集め、同じ (銘柄, 判断日) は最後の候補の日の行だけ残す。Date を判断日にし、元の候補の
    日を cand_date に残す。値動き4列を足す。日次の収益は d_ret・d_hit・d_r20 に名前を変えて残す（確かめと参考）。列は cols と
    label・識別の列だけ（行が k 倍になるので）。(Date, Code) の順。
    """
    clash = [c for c in ADDED if c in full.columns]
    if clash:
        raise SystemExit(f"データセットに組の行で足す列と同じ名前の列がある: {clash}")
    cal = W.calendar(days)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    keep = ["Code", "Date", "label", RET, HIT, R20] + [c for c in cols if c not in ("Code", "Date", "label")]
    d = full[[c for c in dict.fromkeys(keep) if c in full.columns]].copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Code"] = d["Code"].astype(str)
    off = int((~d["Date"].isin(cal)).sum())
    if off:
        raise SystemExit(f"データセットに取引日の暦に無い日付の行がある（{off}行）")
    d = d.rename(columns={RET: D_RET, HIT: D_HIT, R20: D_R20})
    d["_p"] = d["Date"].map(pos).to_numpy(dtype=np.int64)
    parts = []
    for a in range(k):
        x = d[d["_p"] + a < len(cal)].copy()
        x["_t"] = x["_p"] + a
        parts.append(x)
    r = pd.concat(parts, ignore_index=True)
    before = len(r)
    r = r.sort_values(["Code", "_t", "_p"], kind="mergesort").drop_duplicates(["Code", "_t"], keep="last")
    r["cand_date"] = r["Date"]
    r["Date"] = cal[r["_t"].to_numpy()]
    r["r5_days"] = (r["_t"] - r["_p"]).astype(np.int64)
    r = r.drop(columns=["_p", "_t"]).sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    p0 = r["cand_date"].map(pos).to_numpy(dtype=np.int64)

    # 値動き: 候補の日・判断日の終値と、その間（候補の日の翌取引日〜判断日）の高値の最大・安値の最小
    px = bars[bars["Code"].isin(set(r["Code"]))][["Code", "Date", "AdjH", "AdjL", "AdjC"]].copy()
    px["Date"] = pd.to_datetime(px["Date"])
    px["Code"] = px["Code"].astype(str)
    rid = np.arange(len(r))
    codes = r["Code"].to_numpy()

    def close_at(dates) -> np.ndarray:
        q = pd.DataFrame({"Code": codes, "Date": pd.to_datetime(dates), "_rid": rid})
        m = q.merge(px[["Code", "Date", "AdjC"]], on=["Code", "Date"], how="left", validate="many_to_one")
        return m.sort_values("_rid")["AdjC"].to_numpy(dtype=float)

    c0 = close_at(r["cand_date"].to_numpy())
    c1 = close_at(r["Date"].to_numpy())
    span = r["r5_days"].to_numpy()
    seg = [pd.DataFrame({"_rid": rid[span >= j], "Code": codes[span >= j], "Date": cal[p0[span >= j] + j]})
           for j in range(1, int(span.max(initial=0)) + 1)]
    hmax = np.full(len(r), np.nan)
    lmin = np.full(len(r), np.nan)
    if seg:
        s = pd.concat(seg, ignore_index=True).merge(px[["Code", "Date", "AdjH", "AdjL"]], on=["Code", "Date"],
                                                    how="left", validate="many_to_one")
        agg = s.groupby("_rid").agg(h=("AdjH", "max"), l=("AdjL", "min"))
        hmax = agg["h"].reindex(rid).to_numpy(dtype=float)
        lmin = agg["l"].reindex(rid).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r["r5_ret"] = np.where(c0 > 0, c1 / c0 - 1.0, np.nan)
        r["r5_high"] = np.where(c0 > 0, hmax / c0 - 1.0, np.nan)
        r["r5_low"] = np.where(c0 > 0, lmin / c0 - 1.0, np.nan)

    if r.duplicated(["Code", "Date"]).any():
        raise SystemExit("組の行に同じ (銘柄, 判断日) の行が2つ以上ある")
    if ((r["r5_days"] < 0) | (r["r5_days"] > k - 1)).any() or (r["cand_date"] > r["Date"]).any():
        raise SystemExit(f"候補の日が判断日の0〜{k - 1}営業日前に入らない行がある")
    size = r.groupby("Date").size()
    info = {"daily_rows": int(len(full)), "rows": int(len(r)), "days": int(r["Date"].nunique()),
            "dropped_repeats": int(before - len(r)),
            "per_day_median": float(size.median()) if len(size) else np.nan,
            "days_ge5": float((size >= E.MIN_DAY).mean()) if len(size) else np.nan,
            "age_share": [float((r["r5_days"] == j).mean()) if len(r) else np.nan for j in range(k)],
            "first": str(r["Date"].min().date()) if len(r) else None,
            "last": str(r["Date"].max().date()) if len(r) else None}
    return r, info


def check_pools(rl: pd.DataFrame, full: pd.DataFrame, days, n: int = N_POOL_CHECK, seed: int = 0,
                k: int = K) -> Dict:
    """無作為に選んだ n 個の判断日について、組の (銘柄, 候補の日) が、データセットから直接選び直したものと同じかを数える。"""
    cal = W.calendar(days)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    f = full[["Code", "Date"]].copy()
    f["Date"] = pd.to_datetime(f["Date"])
    f["Code"] = f["Code"].astype(str)
    f["_p"] = f["Date"].map(pos)
    ds = np.sort(rl["Date"].unique())
    rng = np.random.default_rng(seed)
    pick = ds[np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))] if len(ds) else []
    same = 0
    for t in pick:
        pt = int(pos[pd.Timestamp(t)])
        g = f[(f["_p"] >= pt - (k - 1)) & (f["_p"] <= pt)].sort_values(["Code", "Date"]).drop_duplicates("Code", keep="last")
        want = set(zip(g["Code"], g["Date"]))
        got_rows = rl[rl["Date"] == t]
        got = set(zip(got_rows["Code"], pd.to_datetime(got_rows["cand_date"])))
        same += int(want == got)
    return {"days": int(len(pick)), "same": int(same)}


def check_move_cols(rl: pd.DataFrame, bars: pd.DataFrame, days, n: int = N_CHECK, seed: int = 0) -> Dict:
    """
    値動き4列と r5_days を、無作為に選んだ n 行について銘柄の日足を直接切り出して計算し直し、同じかを数える（欠測どうしは同じ、
    値は差が 1e-12 以下なら同じ）。
    """
    cal = W.calendar(days)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rl), size=min(n, len(rl)), replace=False)) if len(rl) else np.array([], dtype=int)
    sub = rl.iloc[idx][["Code", "Date", "cand_date"] + list(MOVE_COLS)]
    b = bars[bars["Code"].astype(str).isin(set(sub["Code"]))].copy()
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str)
    groups = {c: g.set_index("Date").sort_index() for c, g in b.groupby("Code", sort=False)}
    same = 0
    worst = 0.0
    for r in sub.itertuples(index=False):
        g = groups.get(r.Code)
        c0 = float(g["AdjC"].get(r.cand_date, np.nan)) if g is not None else np.nan
        c1 = float(g["AdjC"].get(r.Date, np.nan)) if g is not None else np.nan
        win = g[(g.index > r.cand_date) & (g.index <= r.Date)] if g is not None else None
        want = {"r5_days": float(((cal > r.cand_date) & (cal <= r.Date)).sum()),
                "r5_ret": c1 / c0 - 1.0 if c0 > 0 else np.nan,
                "r5_high": float(win["AdjH"].max()) / c0 - 1.0 if (win is not None and len(win) and c0 > 0) else np.nan,
                "r5_low": float(win["AdjL"].min()) / c0 - 1.0 if (win is not None and len(win) and c0 > 0) else np.nan}
        ok = True
        for key, v in want.items():
            got = float(getattr(r, key))
            if np.isnan(v) and np.isnan(got):
                continue
            if np.isnan(v) or np.isnan(got) or abs(v - got) > 1e-12:
                ok = False
                worst = max(worst, abs(v - got)) if (np.isfinite(v) and np.isfinite(got)) else float("inf")
        same += int(ok)
    return {"rows": int(len(sub)), "same": int(same), "max_diff": float(worst)}


def check_same_day(rl: pd.DataFrame) -> Dict:
    """判断日に入った候補（r5_days = 0）は、組の収益と日次の収益が同じはず（欠測の有無も含めて）。合わない行を数える。"""
    return W.check_same_day(rl.rename(columns={"r5_days": "wk_days"}))


# --------------------------------------------------------------------------- #
# 探索
# --------------------------------------------------------------------------- #

def search_frame(rl: pd.DataFrame, cutoff) -> Tuple[pd.DataFrame, int]:
    """探索に使う組の行（判断日が打ち切り以前・収益がある・ラベルがある行）と、ラベルが無くて外した行の数。"""
    return W.search_frame(rl, cutoff)


def search_sig(cols_w: Sequence[str], sub: pd.DataFrame, n_trials: int) -> Dict:
    """探索の結果を使い回してよいかの目印（列・行・期間・試行数が同じなら同じ）。"""
    return {"features_sig": F.signature(list(cols_w)), "n_features": len(cols_w), "rows": int(len(sub)),
            "days": int(sub["Date"].nunique()), "train_to": str(pd.to_datetime(sub["Date"]).max().date()),
            "n_trials": int(n_trials), "target": R20, "group": f"rolling{K}", "folds": "month_purged", "v": 1}


def _year_groups(years: np.ndarray, n_splits: int) -> np.ndarray:
    """月の塊の年。月が n_splits 未満の年は隣の年とまとめる（層別の層が分割数より少ないと分けられない）。"""
    uniq = sorted(set(int(y) for y in years))
    cnt = {y: int((years == y).sum()) for y in uniq}
    groups: List[List[int]] = []
    cur: List[int] = []
    n = 0
    for y in uniq:
        cur.append(y)
        n += cnt[y]
        if n >= n_splits:
            groups.append(cur)
            cur, n = [], 0
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    gid = {y: i for i, g in enumerate(groups) for y in g}
    return np.array([gid[int(y)] for y in years])


def search_folds(sub: pd.DataFrame, days, n_splits: int = E.N_SPLITS, seed: int = E.SEED,
                 purge: int = K - 1) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    探索の分割。判断日を暦月の塊にまとめ、年で層別して塊ごとに n_splits 分割する（同じ月の判断日は同じ側）。訓練側からは、検証側の
    判断日の前後 purge 営業日以内の判断日を外す（組は直近 K 営業日の候補なので、判断日が purge 営業日より離れていれば同じ候補を
    含まない）。
    """
    from sklearn.model_selection import StratifiedKFold

    cal = W.calendar(days)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    d = pd.to_datetime(sub["Date"])
    month = d.dt.to_period("M")
    months = np.array(sorted(month.unique()))
    if len(months) < n_splits:
        return []
    years = np.array([m.year for m in months])
    strata = _year_groups(years, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = d.map(pos).to_numpy(dtype=np.int64)
    out = []
    for _, va_m in skf.split(months, strata):
        va = month.isin(months[va_m]).to_numpy()
        near = np.zeros(len(cal) + 2 * purge + 1, dtype=bool)
        vp = np.unique(p[va])
        for j in range(-purge, purge + 1):
            near[np.clip(vp + j, 0, len(cal) - 1)] = True
        tr = (~va) & (~near[p])
        out.append((sub[tr], sub[va]))
    return out


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #

KEEP_R = ("Code", "Date", "cand_date", "r5_days", RET, HIT, HITDAY, R20, D_RET, D_R20)


def run_tag(params: Dict[str, Dict], cols: Sequence[str], cols_w: Sequence[str], rl: pd.DataFrame,
            daily: pd.DataFrame) -> str:
    """保存名に入れる目印（木の形・列・データが変わったら別の名前にして、古い out-of-fold を読まない）。"""
    key = {"params": {a: {k: p[k] for k in sorted(p)} for a, p in params.items()},
           "cols": F.signature(list(cols)), "cols_w": F.signature(list(cols_w)),
           "rl_rows": int(len(rl)), "rl_last": str(pd.to_datetime(rl["Date"]).max().date()),
           "rl_r20": int(rl[R20].notna().sum()), "d_rows": int(len(daily)),
           "d_last": str(pd.to_datetime(daily["Date"]).max().date()), "d_r20": int(daily[R20].notna().sum()),
           "seed": SEED, "k": K, "v": 1}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


def oof_shift(rl: pd.DataFrame, daily: pd.DataFrame, cols: Sequence[str], cols_w: Sequence[str],
              params: Dict[str, Dict], shift: int, out_dir: str = OOF_DIR) -> pd.DataFrame:
    """
    1つのずらし方の out-of-fold。組の行（収益があるもの）を4本で採点した表（score_W5・score_W5X・score_D・score_B・fold）。
    rl は組の行（今の売り方の収益と20営業日目の終値がある行）、daily は日次の候補（ラベルがある行）。保存済みなら読む。
    """
    tag = run_tag(params, cols, cols_w, rl, daily)
    path = os.path.join(out_dir, f"{PREFIX}_pool_{tag}_sh{shift}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    t0 = time.time()
    folds = E41.folds_for(rl["Date"], shift)
    dr = pd.to_datetime(rl["Date"])
    dd = pd.to_datetime(daily["Date"])
    has_r20 = daily[R20].notna()
    rows = []
    for f in folds:
        lo, hi, end = pd.Timestamp(f.test_start), pd.Timestamp(f.test_end), pd.Timestamp(f.train_end)
        te = rl[(dr >= lo) & (dr <= hi)]
        tr_r = rl[dr <= end]
        if len(te) < MIN_TEST or len(tr_r) < MIN_TRAIN:
            continue
        tr_l = daily[(dd <= end) & has_r20]
        tr_c = daily[dd <= end]
        part = te[list(KEEP_R)].copy()
        part["score_W5"] = W.fit_rank(params["W5"], tr_r, te, cols_w)
        part["score_W5X"] = W.fit_rank(params["W5"], tr_r, te, cols)
        part["score_D"] = W.fit_rank(params["D"], tr_l, te, cols)
        part["score_B"] = W.fit_base(params["B"], tr_c, [te], cols)[0]
        part["fold"] = int(f.index)
        rows.append(part)
    if not rows:
        raise SystemExit(f"ずらし{shift}か月: 窓を作れない（期間が短すぎる）")
    o = pd.concat(rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    o.to_parquet(path, index=False)
    log(f"  ずらし{shift}か月: {o['fold'].nunique()}窓・組の行 {len(o):,}件・{o['Date'].nunique():,}日 {time.time() - t0:.0f}秒")
    return o


# --------------------------------------------------------------------------- #
# 毎日の買い
# --------------------------------------------------------------------------- #

def _take(top: pd.DataFrame, ok: np.ndarray, pos: Dict, hold: int) -> pd.DataFrame:
    """
    並べた行（判断日・順位の順）を上から見て、ok で、同じ候補（銘柄・候補の日）をまだ買っておらず、その銘柄を持っていないものを
    買う。持っている間 = 買った日から、+10% に k営業日目で届けば k営業日、届かなければ hold 営業日（live_track.simulate の枠の
    空き方と同じ）。
    """
    bought = set()
    until: Dict[str, float] = {}
    keep = []
    for r, good in zip(top.itertuples(), ok):
        if not good:
            continue
        key = (r.Code, pd.Timestamp(r.cand_date))
        if key in bought:
            continue
        b = pos[pd.Timestamp(r.Date)] + 1
        if until.get(r.Code, -np.inf) > b:
            continue
        bought.add(key)
        hit = bool(getattr(r, HIT))
        hd = getattr(r, HITDAY)
        until[r.Code] = b + (float(hd) if hit and np.isfinite(hd) else float(hold))
        keep.append(r.Index)
    return top.loc[keep]


def daily_buys(o: pd.DataFrame, score: str, days, top_n: int = TOP_N, need: str | None = None,
               hold: int = E.HOLD) -> pd.DataFrame:
    """
    毎日の買い方（運用者の選択「上位2件のうち未購入を買う」「上位2件かつ条件を満たすもの」）。判断日ごとに組をスコアの上から並べ
    （同じスコアは銘柄コードの順）、上位 top_n 件のうち、同じ候補をまだ買っておらず、その銘柄を持っておらず、need の列が 1 のもの
    （need が None なら条件なし。参考）を、判断日の翌営業日の寄りで買う。上位 top_n 件より下には降りない。返すのは買った行
    （rank = その日の上位の中の順位 1〜top_n）。判断日の順。
    """
    cal = W.calendar(days)
    pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    s = o.sort_values(["Date", score, "Code"], ascending=[True, False, True], kind="mergesort")
    top = s.groupby("Date", sort=True).head(top_n).copy()
    top["rank"] = top.groupby("Date", sort=True).cumcount() + 1
    ok = (np.ones(len(top), dtype=bool) if need is None
          else pd.to_numeric(top[need], errors="coerce").to_numpy(dtype=float) == 1.0)
    return _take(top, ok, pos, hold)


def rule_buys(o: pd.DataFrame, days, hold: int = E.HOLD) -> pd.DataFrame:
    """
    今の買い（R15）: 候補の日を判断日とする組の行（r5_days = 0）のうち R15 の印が付いたものを、候補の日の翌営業日の寄りで買う
    （持っている銘柄は買い増さない）。同じ日は3本の最小の順、同点は基準モデルのスコアの順（live_track.decide と同じ）。rank は
    その日の印の付いた行の中の順位（3枠は上位2件を先着順。live_track.TOP_K）。
    """
    cal = W.calendar(days)
    pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    z = o[(o["r5_days"] == 0) & (pd.to_numeric(o["r15"], errors="coerce") == 1.0)]
    s = z.sort_values(["Date", "pmin", "score_B", "Code"], ascending=[True, False, False, True], kind="mergesort").copy()
    s["rank"] = s.groupby("Date", sort=True).cumcount() + 1
    return _take(s, np.ones(len(s), dtype=bool), pos, hold)


def slot_trades(buys: pd.DataFrame, days, n_months: int) -> Dict:
    """買いを画面の3枠（live_track.simulate。先着順・空きが無ければ見送る）で回した取引の件数・平均収益・資金全体の月あたり。"""
    if not len(buys):
        return {"n": 0, "ret": np.nan, "bad": np.nan, "capital_pm": np.nan, "offered": 0}
    p = buys[["Date", "Code", "rank", RET, HIT, HITDAY]].copy()
    p["status"] = np.where(p[HIT].astype(bool), LT.HIT, "満了")
    p["hit_day"] = p[HITDAY]
    sim = LT.simulate(p, list(W.calendar(days)))
    t = sim[sim["taken"] == LT.TAKEN]
    r = t[RET].to_numpy(dtype=float)
    return {"n": int(len(t)), "offered": int(len(p)), "ret": float(np.mean(r)) if len(r) else np.nan,
            "bad": float(np.mean(r < BAD)) if len(r) else np.nan,
            "capital_pm": float(np.sum(r) / LT.SLOTS / n_months) if (len(r) and n_months) else np.nan}


def attach_rule(o: pd.DataFrame, rule: pd.DataFrame, names: Tuple[str, str] = ("r15", "pmin")) -> pd.DataFrame:
    """
    組の行に、候補の日の印と3本の最小を足す（銘柄・候補の日で突き合わせる。印の無い候補は欠測）。rule は daily_rule の戻り値
    （列 r15・pmin）で、names の名前で足す（既定は R15 の r15・pmin。1.2σ の3本なら r0・pmin0）。
    """
    r = rule[["Code", "Date", "r15", "pmin"]].rename(columns={"Date": "cand_date", "r15": names[0], "pmin": names[1]}).copy()
    r["Code"] = r["Code"].astype(str)
    r["cand_date"] = pd.to_datetime(r["cand_date"])
    base = o.drop(columns=[c for c in names if c in o.columns])
    return base.merge(r, on=["Code", "cand_date"], how="left", validate="many_to_one")


def flagged(o: pd.DataFrame) -> pd.DataFrame:
    """R15 の印が組のすべての行に付いた判断日の組だけ（印の無い候補が上位に来た日を、条件を満たさない扱いにしないため）。"""
    ok = o["r15"].notna().groupby(o["Date"]).transform("all").to_numpy(dtype=bool)
    return o[ok].reset_index(drop=True)


def age_stats(o: pd.DataFrame, buys: Dict[str, pd.DataFrame], k: int = K) -> Dict:
    """
    新しさ: 候補の日からの営業日数（0〜k−1）ごとの組の行の割合、腕ごとの組の中の百分位（0 = 最下位 / 1 = 最上位）の平均、
    「入った日（0）− 1〜k−1」の差、買った候補の営業日数の割合と平均。
    """
    age = o["r5_days"].to_numpy()
    out: Dict = {"share": [float((age == j).mean()) if len(o) else np.nan for j in range(k)],
                 "mean": float(age.mean()) if len(o) else np.nan, "pct": {}, "fresh_gap": {}, "buy_share": {},
                 "buy_mean": {}}
    for a, b in buys.items():
        pct = LTR.within_date_pct(o["Date"], o[f"score_{a}"].to_numpy(dtype=float))
        out["pct"][a] = [float(pct[age == j].mean()) if (age == j).any() else np.nan for j in range(k)]
        out["fresh_gap"][a] = (float(pct[age == 0].mean() - pct[age > 0].mean())
                               if (age == 0).any() and (age > 0).any() else np.nan)
        ba = b["r5_days"].to_numpy() if len(b) else np.array([], dtype=int)
        out["buy_share"][a] = [float((ba == j).mean()) if len(ba) else np.nan for j in range(k)]
        out["buy_mean"][a] = float(ba.mean()) if len(ba) else np.nan
    return out


def n_months(o: pd.DataFrame) -> int:
    """判断日がある月の数（資金全体の月あたり・月あたりの件数の分母）。"""
    return int(pd.to_datetime(o["Date"]).dt.to_period("M").nunique()) if len(o) else 0


def evaluate(o: pd.DataFrame, days, top_n: int = TOP_N) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
    """
    1つのずらし方の集計。o は組の out-of-fold に R15 の印（attach_rule）を足したもの。数えるのは R15 の印が組のすべてに付いた
    判断日（flagged）。腕ごとの買い（上位 top_n 件のうち未購入で R15 を満たすもの）・今の買い（R15）・3枠・窓ごとの差（W5 − R15・
    D・B・W5X）・参考に条件なしの上位 top_n 件・組の中の並び・新しさ。返すのは集計と、腕ごとの買った行。
    """
    of = flagged(o)
    nm = n_months(of)
    buys = {a: daily_buys(of, f"score_{a}", days, top_n, need="r15") for a in ARMS}
    r15 = rule_buys(of, days)
    free = {a: daily_buys(of, f"score_{a}", days, top_n) for a in ARMS}
    rule_cands = of.loc[pd.to_numeric(of["r15"], errors="coerce") == 1.0, ["Code", "cand_date"]].drop_duplicates()
    res: Dict = {"windows": int(of["fold"].nunique()) if len(of) else 0, "rows": int(len(of)),
                 "days": int(of["Date"].nunique()) if len(of) else 0, "months": nm,
                 "rows_all": int(len(o)), "days_all": int(o["Date"].nunique()),
                 "all": W.pick_stats(of), "buys": {a: W.pick_stats(b) for a, b in buys.items()},
                 "rule": W.pick_stats(r15), "rule_cands": int(len(rule_cands)),
                 "slots": {a: slot_trades(b, days, nm) for a, b in buys.items()},
                 "free": {a: W.pick_stats(f) for a, f in free.items()},
                 "rank": {a: W.ranking_metrics(of, f"score_{a}") for a in ARMS}, "diff": {}, "diff_bad": {}}
    res["slots"]["R15"] = slot_trades(r15[r15["rank"] <= LT.TOP_K], days, nm)
    base = {a: W.per_window(b) for a, b in buys.items()}
    base["R15"] = W.per_window(r15)
    for b in ("R15", "B", "D", "W5X"):
        res["diff"][b] = E.fold_diff(base[b], base["W5"])
        res["diff_bad"][b] = E.fold_diff(base[b], base["W5"], "bad")
    res["age"] = age_stats(of, buys)
    return res, buys


def reading(rows: List[Dict]) -> Dict:
    """
    読み方（回す前に決めたもの。採用の判定ではない）。rows はずらし方ごとの evaluate の結果（shift を足したもの）。
      better: すべてのずらし方で、W5 の買い（上位2件かつ R15）の平均収益（窓の平均）が、今の買い（R15）・D・B の買いのどれよりも
              高く、−10% 未満（窓の平均）がどれよりも低い
      lean: W5 の組の中の百分位の平均の「入った日 − 1〜4営業日」（小数2桁に丸める）が、すべてで +FRESH_LINE 以上なら "fresh"、
            すべてで −FRESH_LINE 以下なら "aged"、それ以外は "none"
    """
    each = []
    for r in rows:
        w = r["buys"]["W5"]
        others = (r["rule"], r["buys"]["D"], r["buys"]["B"])
        ok = bool(all(w["ret_w"] > x["ret_w"] for x in others) and all(w["bad_w"] < x["bad_w"] for x in others))
        # 表に出す桁（小数2桁）に丸めてから線と比べる（表の値と読み方を食い違わせない）
        g = r["age"]["fresh_gap"]["W5"]
        gap = round(float(g), 2) if np.isfinite(g) else float("nan")
        each.append({"shift": int(r["shift"]), "better": ok, "fresh_gap": gap})
    gaps = [x["fresh_gap"] for x in each]
    if each and all(np.isfinite(g) and g >= FRESH_LINE for g in gaps):
        lean = "fresh"
    elif each and all(np.isfinite(g) and g <= -FRESH_LINE for g in gaps):
        lean = "aged"
    else:
        lean = "none"
    return {"rows": each, "better": bool(each) and all(x["better"] for x in each), "lean": lean}


LEAN_JA = {"fresh": "入ったばかりの候補を上に置きやすい", "aged": "入ってから日がたった候補を上に置きやすい",
           "none": "はっきりとは寄らない"}


# --------------------------------------------------------------------------- #
# 参考: 日次の1位・組の1位・R15 の3つとも満たす候補（運用者の確認 2026-10-30。見るだけ）
# --------------------------------------------------------------------------- #

TRIPLE_SETS = ("Q4", "D1W1", "R15R0", "D1", "W1", "R15", "REST")
TRIPLE_JA = {"Q4": "4つとも満たす（3つ＋1.2σ の3本 95以上）", "D1W1": "3つとも満たす（日次1位・組1位・R15）",
             "R15R0": "R15＋R0（1.5σ・1.2σ の6本 95以上）", "D1": "R15＋日次の1位", "W1": "R15＋組の1位（候補の日）",
             "R15": "R15 だけ（今の買い）", "REST": "今の買いのうち3つとも満たすもの以外"}
#: 1.2σ の木3本（2026-10-23〜10-29 の規則。実験115 の R0 と同じ）
R0_TREES = tuple(W.RULE_TREES)


def triple_buys(o: pd.DataFrame, days, hold: int = E.HOLD) -> Dict[str, pd.DataFrame]:
    """
    運用者の確認（2026-10-30）「当日のみでのランキングで1位、直近5営業日ランキングでも1位、1.5σモデルでも全部95以上の銘柄のみ
    買ったら」。判断日 t の組の行に、その日の候補（r5_days = 0）の中の今のランキング学習（日次。D）の順位 d_rank と、組全体（直近
    5営業日）の中の W5 の順位 w5_rank を付ける（どちらも同じスコアは銘柄コードの順。候補の日でない行の d_rank は 0）。候補の日の行で
    R15 を満たし、d_rank と w5_rank がどちらも1位のものを、候補の日の翌営業日の寄りで買う（持っている銘柄は買い増さない。_take）。
    分けて見るために、R15＋日次の1位・R15＋組の1位・R15 だけ（今の買い）と、今の買いのうち3つとも満たすもの以外も返す。
    組の行に r0（1.2σ の3本の印。attach_rule の names = ("r0", "pmin0")）があれば、3つに R0 も満たす4つとも満たす候補（Q4）と
    R15＋R0（順位は問わない）も返す（運用者の確認 2026-10-30「1.2σモデルでも95以上の条件だとどうなるか」）。
    """
    cal = W.calendar(days)
    pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    x = o.reset_index(drop=True).copy()
    s = x.sort_values(["Date", "score_W5", "Code"], ascending=[True, False, True], kind="mergesort")
    x["w5_rank"] = 0
    x.loc[s.index, "w5_rank"] = (s.groupby("Date", sort=False).cumcount() + 1).to_numpy()
    s = x[x["r5_days"] == 0].sort_values(["Date", "score_D", "Code"], ascending=[True, False, True], kind="mergesort")
    x["d_rank"] = 0
    x.loc[s.index, "d_rank"] = (s.groupby("Date", sort=False).cumcount() + 1).to_numpy()
    base = (x["r5_days"] == 0).to_numpy() & (pd.to_numeric(x["r15"], errors="coerce") == 1.0).to_numpy()
    d1 = (x["d_rank"] == 1).to_numpy()
    w1 = (x["w5_rank"] == 1).to_numpy()
    masks = {"D1W1": base & d1 & w1, "D1": base & d1, "W1": base & w1, "R15": base}
    if "r0" in x.columns:
        r0 = (pd.to_numeric(x["r0"], errors="coerce") == 1.0).to_numpy()
        masks.update({"Q4": base & d1 & w1 & r0, "R15R0": base & r0})
    out: Dict[str, pd.DataFrame] = {}
    for k, m in masks.items():
        sel = x[m].sort_values(["Date", "Code"], kind="mergesort")
        out[k] = _take(sel, np.ones(len(sel), dtype=bool), pos, hold)
    r = out["R15"]
    out["REST"] = r[~((r["d_rank"] == 1) & (r["w5_rank"] == 1))]
    return out


def triple_stats(o: pd.DataFrame, days, hold: int = E.HOLD) -> Dict:
    """1つのずらし方の参考の集計。数えるのは evaluate と同じ判断日（R15 の印が組のすべてに付いた日）。"""
    of = flagged(o)
    nm = n_months(of)
    sets = triple_buys(of, days, hold)
    out: Dict = {"days": int(of["Date"].nunique()) if len(of) else 0, "months": nm, "sets": {}}
    for k in [k for k in TRIPLE_SETS if k in sets]:
        b = sets[k]
        st = W.pick_stats(b)
        r = b[RET].to_numpy(dtype=float)
        st["pm"] = float(len(b) / nm) if nm else np.nan
        st["se"] = float(np.std(r, ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
        st["win"] = float(np.mean(r > 0)) if len(r) else np.nan
        out["sets"][k] = st
    return out


# --------------------------------------------------------------------------- #
# 今の買いの印（R15）
# --------------------------------------------------------------------------- #

def daily_rule(daily: pd.DataFrame, data_dir: str | None = None, models: Sequence[str] = RULE_TREES,
               line: float = LINE) -> Tuple[pd.DataFrame, Dict]:
    """
    今の買い（R15）の印と3本の最小。実験82 と同じ作り（実験115 の daily_rule と同じで、3本を今の規則の 1.5σ の木にしたもの）:
    本番の out-of-fold を本番と同じ窓に分け、前向きの百分位（前の窓のスコアの分布での位置）を画面と同じ小数1桁に丸めて、3本
    すべて line 以上。返すのは daily の (Code, Date) と r15（1.0 / 0.0。百分位が付かない行は欠測）・pmin（3本の最小）。3本の
    ファイルがそろわなければ found = False。印は候補ごとに決まるので、どのずらし方でも同じものを使う。
    """
    import lab
    import e79_model_clusters as E79
    import e82_and_rule_variants as E82

    data_dir = data_dir or lab.DATA_DIR
    models = tuple(models)
    keys = daily[["Code", "Date"]].copy()
    keys["Code"] = keys["Code"].astype(str)
    keys["Date"] = pd.to_datetime(keys["Date"])
    if not all(os.path.exists(E79.oof_path(a, data_dir)) for a in models):
        return keys.iloc[:0].assign(r15=pd.Series(dtype=float), pmin=pd.Series(dtype=float)), {"found": False}
    w, info0 = E79.load_oofs(data_dir, models=models)
    folds = E79.make_windows(keys["Date"])
    w["window"] = E79.assign_windows(w["Date"], folds)
    w = w[w["window"] > 0].reset_index(drop=True)
    flag = np.full(len(w), np.nan)
    pmin = np.full(len(w), np.nan)
    for _, (_, te, _, Pte) in E79.window_split(w, folds, models=models).items():
        rp = E82.rounded(Pte)
        flag[te] = (np.nan_to_num(rp, nan=-1.0) >= line).all(axis=1).astype(float)
        pmin[te] = np.nanmin(np.where(np.isfinite(rp), rp, np.nan), axis=1)
    w["r15"] = flag
    w["pmin"] = pmin
    out = keys.merge(w[["Code", "Date", "r15", "pmin"]], on=["Code", "Date"], how="left", validate="many_to_one")
    info = {"found": True, "joined": int(info0["kept"]), "rows": int(len(w)), "with_pct": int(np.isfinite(flag).sum()),
            "r15": int(np.nansum(flag)), "matched": int(out["r15"].notna().sum())}
    return out, info


# --------------------------------------------------------------------------- #

def _p(x: float, nd: int = 2) -> str:
    return W._p(x, nd)


def _s(x: float) -> str:
    return W._s(x)


def _pm(n: int, months: int) -> str:
    return f"{n / months:.2f}" if months else "—"


def report(sh: int, res: Dict, top_n: int) -> None:
    """1つのずらし方の表（公開ログ。集計だけ）。"""
    a0, nm = res["all"], res["months"]
    print(f"\n■ ずらし{sh}か月（{res['windows']}窓・R15 の印が組のすべてに付いた判断日 {res['days']:,}日 / テスト窓の "
          f"{res['days_all']:,}日・{nm}か月・組の行 {res['rows']:,}件・候補全体の平均 {_p(a0['ret'])}（窓の平均 {_p(a0['ret_w'])}）・"
          f"−10% 未満 {_s(a0['bad'])}）")
    print(f"  ◆ 毎日の買い（上位{top_n}件のうち、未購入で 1.5σ の3本が{LINE:.0f}以上のものを翌営業日の寄りで買う）: 件数（月あたり）/ "
          f"平均収益（全件・窓の平均）/ −10% 未満（全件・窓の平均）/ +10%到達 / 20営業日目の終値")
    rows = [(LABELS[a] + "＋R15", res["buys"][a]) for a in ARMS] + [("今の買い R15（候補の翌営業日）", res["rule"])]
    for name, t in rows:
        print(f"    {name:<30}{t['n']:>5}件（{_pm(t['n'], nm)}）{_p(t['ret']):>8} {_p(t['ret_w']):>8} {_s(t['bad']):>6} "
              f"{_s(t['bad_w']):>6} {_s(t['hit']):>6} {_p(t['r20']):>8}")
    print(f"    {'候補全体（組の行すべて）':<28}{a0['n']:>5}件 {_p(a0['ret']):>8} {_p(a0['ret_w']):>8} "
          f"{_s(a0['bad']):>6} {_s(a0['bad_w']):>6} {_s(a0['hit']):>6} {_p(a0['r20']):>8}")
    print(f"    R15 の印が付いた候補 {res['rule_cands']}件のうち、上位{top_n}件に入って買ったもの: "
          + " / ".join(f"{a} {res['buys'][a]['n']}件" for a in ARMS))
    print("  ◆ 窓ごとの差（W5＋R15 − 比べる買い。平均収益 / −10% 未満）: " + " / ".join(
        f"W5 − {b} {W._d(res['diff'][b])}・−10%未満 {res['diff_bad'][b]['mean'] * 100:+.1f}pt" if res['diff'][b].get('n')
        else f"W5 − {b} —" for b in ("R15", "B", "D", "W5X")))
    print(f"  ◆ 同じ買いを画面の3枠で回すと（今の買いはその日の上位{LT.TOP_K}件）: " + " / ".join(
        f"{a} {res['slots'][a]['n']}取引 {_p(res['slots'][a]['ret'])}・月 {_p(res['slots'][a]['capital_pm'])}"
        for a in list(ARMS) + ["R15"]))
    print(f"  ◆ 参考: 条件なしの上位{top_n}件のうち未購入（R15 を問わない）: " + " / ".join(
        f"{a} {res['free'][a]['n']}件 {_p(res['free'][a]['ret_w'])}・−10%未満 {_s(res['free'][a]['bad_w'])}" for a in ARMS))
    print(f"  ◆ 組の中の並び（NDCG@{E.NDCG_K} / 順位相関 今の売り方・20営業日目の終値）: " + " / ".join(
        f"{a} {res['rank'][a]['ndcg']:.4f} / {res['rank'][a]['rho']:+.3f} / {res['rank'][a]['rho20']:+.3f}" for a in ARMS))
    ag = res["age"]
    print("  ◆ 新しさ: 候補の日からの営業日数ごとの組の行の割合 / 腕ごとの組の中の百分位の平均（0.5 が真ん中）/ 入った日 − 1〜4営業日"
          " ｜ 買った候補の営業日数の割合と平均")
    print("    組の行 " + " ".join(f"{j}日 {_s(ag['share'][j])}" for j in range(K)) + f" ｜ 平均 {ag['mean']:.2f}日")
    for a in ARMS:
        g = ag["fresh_gap"][a]
        print(f"    {a:<4}" + " ".join(f"{j}日{ag['pct'][a][j]:.2f}" for j in range(K))
              + f" ｜ 差 {g:+.2f}" + " ｜ 買い " + " ".join(f"{j}日{_s(ag['buy_share'][a][j])}" for j in range(K))
              + f" ・平均 {ag['buy_mean'][a]:.2f}日")


def report_triple(rows: List[Tuple[int, Dict]]) -> None:
    """参考（運用者の確認 2026-10-30・見るだけ）の表（公開ログ。集計だけ）。"""
    print(f"\n■ 参考（運用者の確認 2026-10-30・見るだけ）: 候補の日に「その日の候補の中の今のランキング学習（日次）で1位」"
          f"「直近{K}営業日の組の W5 で1位」「1.5σ の3本が{LINE:.0f}以上」をすべて満たす候補だけを、翌営業日の寄りで買う")
    print("  件数（月あたり）/ 平均収益（全件 ± 標準誤差）/ 窓の平均 / 勝率 / −10% 未満 / +10%到達 / 20営業日目の終値")
    for sh, t in rows:
        print(f"  ずらし{sh}か月（R15 の印が組のすべてに付いた判断日 {t['days']:,}日・{t['months']}か月）")
        for k in [k for k in TRIPLE_SETS if k in t["sets"]]:
            s = t["sets"][k]
            se = f"{s['se'] * 100:.2f}pt" if np.isfinite(s["se"]) else "—"
            print(f"    {TRIPLE_JA[k]:<26}{s['n']:>5}件（{_pm(s['n'], t['months'])}）{_p(s['ret']):>8} ± {se:<7}"
                  f"{_p(s['ret_w']):>8} {_s(s['win']):>6} {_s(s['bad']):>6} {_s(s['hit']):>6} {_p(s['r20']):>8}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験117: 直近5営業日の組で毎日並べるランキング学習（検証だけ）")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--triple", action="store_true",
                    help="参考（運用者の確認 2026-10-30）: 日次の1位・組の1位・R15 の3つとも満たす候補だけを買った成績も出す")
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    import warnings
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)

    t_all = time.time()
    cols = list(F.columns(F.DEFAULT_PRESET))
    cols_w = cols + list(MOVE_COLS)
    p_base = W.base_params(cols)
    rec_d = W.daily_ltr_params(cols)
    os.makedirs(OOF_DIR, exist_ok=True)

    full, bars, cutoff, chk = W.load_all(cols)
    days = W.calendar(bars["Date"].unique())
    rl, info = rolling_frame(full, bars, days, cols)
    rl, xinfo = W.attach_week_exit(rl, bars, days)
    cm = check_move_cols(rl, bars, days)
    cp = check_pools(rl, full, days)
    sd = check_same_day(rl)
    del bars
    daily = full[full["label"].notna()].reset_index(drop=True)
    del full

    print("=" * 100)
    print(f"実験117 直近{K}営業日の組で毎日並べるランキング学習（検証だけ）: 日次の候補 {info['daily_rows']:,}件 → 組の行 "
          f"{info['rows']:,}件・判断日 {info['days']:,}日（{info['first']} 〜 {info['last']}）/ {F.DEFAULT_PRESET} {len(cols)}列＋"
          f"値動き{len(MOVE_COLS)}列 / ずらし {shifts}か月 / 毎日 上位{args.top}件のうち未購入で 1.5σ の3本が{LINE:.0f}以上のもの")
    print(f"  収益: 判断日の翌営業日の寄りで買い、+{E.TARGET:g}% の指値、届かなければ{E.HOLD}営業日目の終値（今の売り方）/ "
          f"W5 が学ぶ並び: 組の中の{E.HOLD}営業日目の終値の{E.N_GRADES}分位")
    print("=" * 100)
    log(f"[確かめ] 日次の出口の計算: 同じ計算の5日平均が ret_o1_20 と一致 {chk['hold5_match']:,} / {chk['hold5_both']:,} 行")
    if chk["hold5_both"] and chk["hold5_match"] / chk["hold5_both"] < 0.99:
        raise SystemExit("出口の計算が ret_o1_20 と合わない（日足と学習データの株価がずれている）")
    log(f"[確かめ] 組: 同じ銘柄が組の中に2回以上入って外した行 {info['dropped_repeats']:,}件・1日の組 中央 "
        f"{info['per_day_median']:.0f}件・{E.MIN_DAY}件以上の日 {_s(info['days_ge5'])} ｜ 候補の日からの営業日数: "
        + " / ".join(f"{j}日 {_s(info['age_share'][j])}" for j in range(K)))
    log(f"[確かめ] 組の中身をデータセットから直接選び直す: {cp['same']} / {cp['days']} 日が同じ")
    log(f"[確かめ] 組の収益: 20営業日目の終値がある {xinfo['with_r20']:,} / {xinfo['rows']:,}件・+{E.TARGET:g}% 到達 "
        f"{_s(xinfo['hit_share'])}・判断日の翌取引日に買えた {_s(xinfo['next_day_share'])}（遅れた {xinfo['late']}件）")
    log(f"[確かめ] 判断日に入った候補（r5_days = 0）の組の収益と日次の収益: {sd['rows']:,}件中 合わない行 今の売り方 {sd['ret']}・"
        f"20営業日目の終値 {sd['r20']}・到達の印 {sd['hit']}")
    log(f"[確かめ] 値動き4列と r5_days を日足から直接計算し直す: {cm['same']} / {cm['rows']} 行が同じ（差の最大 {cm['max_diff']:.1e}）")
    if cp["same"] != cp["days"]:
        raise SystemExit("組の中身が、データセットから直接選び直したものと合わない")
    if sd["mismatch"]:
        raise SystemExit("判断日に入った候補で、組の収益が日次の収益と合わない（収益の付け方がずれている）")
    if cm["same"] != cm["rows"]:
        raise SystemExit("値動きの列が日足から直接計算した値と合わない")
    rule, rinfo = daily_rule(daily)
    if not rinfo.get("found"):
        raise SystemExit("本番の out-of-fold（1.5σ の木3本）が置き場に無いので、今の買いの印（R15）を作れない")
    log(f"[確かめ] 今の買いの印 R15（本番の out-of-fold・実験82 と同じ作り）: 3本がそろう {rinfo['joined']:,}行・窓に入る "
        f"{rinfo['rows']:,}行・前向きの百分位が付く {rinfo['with_pct']:,}行・R15 {rinfo['r15']:,}件（学習データの {len(daily):,}行の"
        f"うち印が付く {rinfo['matched']:,}行）")
    rule0 = None
    if args.triple:
        rule0, r0info = daily_rule(daily, models=R0_TREES)
        if not r0info.get("found"):
            raise SystemExit("本番の out-of-fold（1.2σ の木3本）が置き場に無いので、R0 の印を作れない")
        both = rule[["Code", "Date", "r15"]].merge(rule0[["Code", "Date", "r15"]].rename(columns={"r15": "r0"}),
                                                   on=["Code", "Date"], how="left", validate="one_to_one")
        log(f"[確かめ] 参考の R0（1.2σ の木3本・同じ作り）: 前向きの百分位が付く {r0info['with_pct']:,}行・R0 {r0info['r15']:,}件・"
            f"R15 と R0 の両方 {int(((both['r15'] == 1) & (both['r0'] == 1)).sum()):,}件・R15 の印があって R0 の印が無い行 "
            f"{int((both['r15'].notna() & both['r0'].isna()).sum()):,}")

    # ---- 探索 ----
    sub, no_label = search_frame(rl, cutoff)
    sig = search_sig(cols_w, sub, args.trials)
    log(f"[探索] 期間 〜{cutoff.date()}（run_tuning.py と同じ打ち切り）: 組の行 {len(sub):,}件・{sub['Date'].nunique()}日"
        f"（層別に使うラベルが無くて外した {no_label}件）")
    store = None
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            prev = json.load(fh)
        if prev.get("_sig") == sig:
            store = prev
            log("[探索] 保存済みの探索を読む（同じデータ・同じ列・同じ試行数）")
    folds_s = search_folds(sub, days)
    if not folds_s:
        raise SystemExit("探索の分割を作れない")
    log("[探索] 分割（暦月の塊・前後4営業日を訓練から外す）: " + " / ".join(
        f"訓練 {len(tr):,}・検証 {len(va):,}" for tr, va in folds_s))
    if store is None:
        log(f"[探索] W5（LambdaRank・組 = 直近{K}営業日・目的 {OBJECTIVE_JA}・{args.trials}試行）")
        rec_w = E.tune_l(sub, cols_w, args.trials, folds_s)
        rec_w["_cv"]["objective"] = OBJECTIVE_JA
        store = {"_sig": sig, "_cutoff": str(cutoff.date()), "W5": rec_w}
        with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
            json.dump(store, fh, ensure_ascii=False, indent=2, default=str)
        log(f"[探索] W5 {rec_w['_minutes']}分 / 分割平均 NDCG@{E.NDCG_K} {rec_w['_cv']['ndcg']}（最良 {rec_w['_cv']['best_trial']}"
            f"試行目）")
    params = {"W5": store["W5"]["params"], "D": rec_d["params"], "B": p_base}
    print("\n■ 木の形")
    print(f"  W5 直近{K}営業日の組（探索 〜{store['_cutoff']}）: {E._tree_line(params['W5'])}")
    print(f"  D  今のランキング学習（research/ltr_params.json・探索 〜{rec_d.get('_cutoff', '?')}）: {E._tree_line(params['D'])}")
    print(f"  B  基準モデル（lgbm_params.json）: {E._tree_line(params['B'])}")
    rnd = E.random_ndcg(folds_s)
    cvm = {"W5": E.cv_metrics("L", params["W5"], folds_s, cols_w), "W5X": E.cv_metrics("L", params["W5"], folds_s, cols)}
    print(f"  探索の5分割で（分割の平均）: 組の中の NDCG@{E.NDCG_K} W5 {cvm['W5']['ndcg']:.4f} / W5X（値動きの列なし）"
          f"{cvm['W5X']['ndcg']:.4f} / でたらめな順 {rnd:.4f} ｜ 組の中の順位相関（20営業日目の終値）W5 {cvm['W5']['rho']:+.3f} / "
          f"W5X {cvm['W5X']['rho']:+.3f}")

    # ---- out-of-fold と集計 ----
    rl_eval = rl[rl[RET].notna() & rl[R20].notna()].reset_index(drop=True)
    summary: Dict = {"info": info, "exit": xinfo, "check_cols": cm, "check_pools": cp, "check_same_day": sd, "sig": sig,
                     "rule_info": rinfo, "cutoff": str(cutoff.date()),
                     "params": {a: {k: v for k, v in p.items()} for a, p in params.items()},
                     "search_cv": {k: v for k, v in store["W5"]["_cv"].items() if k != "trials"},
                     "cv": cvm, "random_ndcg": rnd, "top": args.top, "shifts": {}}
    rows = []
    tri: List[Tuple[int, Dict]] = []
    for sh in shifts:
        o = attach_rule(oof_shift(rl_eval, daily, cols, cols_w, params, sh), rule)
        res, _ = evaluate(o, days, args.top)
        report(sh, res, args.top)
        summary["shifts"][str(sh)] = res
        rows.append({"shift": sh, **res})
        if args.triple:
            tri.append((sh, triple_stats(attach_rule(o, rule0, ("r0", "pmin0")), days)))

    rd = reading(rows)
    summary["reading"] = rd
    print("\n■ 読み方（回す前に決めたもの。採用の判定ではない）")
    for x, r in zip(rd["rows"], rows):
        t = r["buys"]
        print(f"  ずらし{x['shift']}か月: 買いの平均収益（窓の平均）W5 {_p(t['W5']['ret_w'])} / 今の買い R15 {_p(r['rule']['ret_w'])} / "
              f"D {_p(t['D']['ret_w'])} / B {_p(t['B']['ret_w'])}・−10% 未満 W5 {_s(t['W5']['bad_w'])} / R15 "
              f"{_s(r['rule']['bad_w'])} / D {_s(t['D']['bad_w'])} / B {_s(t['B']['bad_w'])}（{'○' if x['better'] else '×'}）・"
              f"新しさの差 {x['fresh_gap']:+.2f}")
    print(f"  → 買い方: {'直近5営業日の上位2件で絞る形は、今の買いや今のスコアで絞るより良い' if rd['better'] else 'はっきりとは良くない'}"
          f" ／ 新しさ: {LEAN_JA[rd['lean']]}")
    if tri:
        report_triple(tri)
        summary["triple"] = {str(sh): t for sh, t in tri}
        # 手元で同じ out-of-fold を作り直すため（木の形。データの値ではない）
        print("  W5 の木の形（全桁）: " + json.dumps(params["W5"], sort_keys=True, default=str))
    summary["minutes"] = round((time.time() - t_all) / 60, 1)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"記録: {OOF_DIR}/{PREFIX}_* （{summary['minutes']}分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
