#!/usr/bin/env python3
"""
購入条件を満たした銘柄ごとのリカレンス・ネットワーク（ε-recurrence network）。画面の「リカレンス・ネットワーク」タブ
（src/views/NetworkView.jsx）に渡す（Release data-raw の recurrence.tar.gz。デプロイのときに dist/data/recurrence/ へ展開する）。

運用者の依頼（2026-10-19）「Obsidian の球状のプロットのようなことを、ノードを1営業日として、本予測モデルに使えそうな可視化を
してほしい。市場全体よりも、その日の各候補銘柄単体で。指数リカレントネットワークが良いと思う。特徴量はいったん、重要度の高い
テクニカル系のみ」。運用者の選択（AskUserQuestion）: ε-リカレンス・ネットワーク / 実験105 の gain 上位9列 / 取れる全期間 /
画面の新しいタブ / データは Release に置く（全期間だと1日あたり約0.7〜1.6MB になり、毎日コミットすると履歴が年に数百MB 膨らむ。
モデル本体を Release に置くのと同じ考え方。この見込みは高値更新の全銘柄で作る前提で、下の対象に絞り色付けの値を足した後は
2026-10-09〜16 の5日で5銘柄・0.7MB）。
対象（同日の運用者の指定「候補といったのは、予測サンプルではなく、その中でも購入条件を満たした銘柄です」）: その日の高値更新の
銘柄（予測ファイルの候補）のうち、購入条件（今日の戦略の規則。1.5σ の3本 RULE_MODELS の百分位がすべて AGREE_PCT 以上）を
満たした銘柄だけ。画面の strategy.js の strategySignal の passed・候補銘柄のニュース（news_links.rule_codes）と同じ。

**参考の表示であって、選定には使わない。** 買う銘柄は今日の戦略（1.5σ の3本の最小の順）のまま。

手法（Marwan ほか 2009 Phys. Lett. A 373:4246・Donner ほか 2010 New J. Phys. 12:033025 の recurrence network）
  ノード: 対象の銘柄の、予測日までの各営業日（売買が成立し、下の9列がそろった日。いちばん長い窓の指標（出来高の250日平均）が
        そろうまでの約250取引日は入らない）。予測日（候補の日）が最後のノード
  状態: テクニカル指標9列（FEATURES。research/technicals.py の値そのもの）を、列の型（linear_preprocess の型と同じ。正で裾が
        重い列は上の0.5%で切って log1p、符号つきで裾が重い列は上下0.5%で切って asinh、ほかは上下0.1%で切る）で変換し、その銘柄の
        ノード全体で平均0・標準偏差1にそろえた9次元のベクトル。|z| は Z_CAP で切り、1/Q σ 刻みの整数で持つ。距離も画面もこの
        丸めた整数から数える（画面で辺の密度を変えて引き直しても、既定の密度ならこのファイルの要約と同じ辺になる）
  辺: 2日の状態のユークリッド距離が ε 以下で、かつ THEILER（目的変数の期間 20営業日）より離れている組。隣り合う日は指標の窓が
      重なってほぼ同じ状態になるので結ばない（リカレンスの解析でいうタイラー窓）。ε は結べる組のうち辺になる割合（辺の密度
      = リカレンス率）が RR_BP（1万分率）になる値。結べる組の距離の2乗（整数）を小さい順に並べた m 番目（m = RR × 組の数を
      四捨五入）を ε² とし、それ以下を全部結ぶ（同じ値は全部入る）。Donner ほか 2010 の「リカレンス率を固定して ε を決める」
  結果（ノードの色）: その日に候補になっていたら、の結果。予測日までに分かったものだけ（予測日より後の日足は読まない）
     ret  … 今日の戦略の売り方（翌営業日の寄りで買い、+10% の指値に届けば +10、届かなければ20営業日目の終値。
             live_track.forward。予測日までに指値に届いた日は、20営業日たっていなくても +10 で分かっている）
     l12 / l15 … 目的変数（1.2σ・1.5σ。build_dataset.attach_rise_label・rise_label_for_k。20営業日たつまでは未確定）
     fresh … 78週高値の更新日（直前20営業日に更新なし。build_dataset.mark_new_highs の is_fresh_break。モデルの母集団の日。
             流動性・市場区分の除外は掛けていない）
     rise / fall … 先20営業日の最大上昇率・最大下落率（%）。t+1〜t+20 の終値の最大・最小 ÷ 翌営業日の始値 − 1。rise は目的変数の
             到達を測る build_dataset の future_rise そのもの（運用者の選択 2026-10-19「終値・翌日寄り基準」）。20営業日たつまでは
             未確定
     r20  … 20営業日目の騰落率（%）。t+20 の終値 ÷ 翌営業日の始値 − 1（live_track.forward の ret_close。指値なしで持ち切った
             ときの値）。20営業日たつまでは未確定。20営業日目に売買が成立しなかった日（2020-10-01 の終日の売買停止・売買の
             少ない銘柄）も分からない（ret と同じ。rise / fall は窓の中の売買の無い日を飛ばして数える）。画面はこの3つでも
             ノードを色分けする（5段・4段と未確定）
  要約（summary）: 今日のノードの次数（ε の中にある過去の日の数。タイラー窓のぶん、近傍はすべて結果が出そろった日）と、その銘柄の
        ノードの中での位置（中位の順位の百分位）、今日の近傍の結果（+10% 到達・目的変数の正例の割合・収益の中央値）と銘柄の全期間の
        割合、今日に近い過去の更新日（母集団の日）。画面は辺の密度を変えるたびに同じ式で数え直す（src/lib/recurrence.js。
        tests/recurrence.test.js と tests/test_recurrence_net.py が同じフィクスチャで一致を確かめる）
  過去の日: その日に作った値のまま（前の archive にある日は引き継ぐ。knn_compare と同じ考え方）。前の archive に無い過去の日は
        その日までの日足で作る（builtAt でいつ作ったか分かる）
  確かめ: 予測用のデータセット（dataset_predict.parquet）があれば、候補の日の9列がデータセットの値と一致するかを数える
        （check_train_serve と同じ許し。相対 1e-6・絶対 1e-9）

出すもの（--out-dir。archive は --archive）
  index.json            日ごとの購入条件を満たした銘柄と要約・その日の高値更新の件数（nBreak）。画面は最初にこれだけ読む
  <日付>_<jqCode>.json  銘柄ごとのノード（状態・結果・色付けの値・終値）。約2,200ノードの銘柄で約130KB（日足の短い銘柄は小さい）

作れなくても日次予測は止めない（predict.yml の段は continue-on-error。銘柄ごとの失敗は理由を残して先へ進む）。

  python3 research/recurrence_net.py --data-dir research/_data --predictions public/data/predictions.json \\
      --out-dir research/_data/recurrence --archive research/_data/recurrence.tar.gz
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys
import tarfile
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import linear_preprocess as LP  # noqa: E402
import live_track as LT  # noqa: E402
import technicals as TX  # noqa: E402

VERSION = 2                         # 2: 先20営業日の値動き（rise・fall・r20）を足した
#: 状態の9列。実験105 の学習の寄与（gain。360列の lgbm）の上位で、1.5σ の上位8列（360列中 28・31・32・38・40・42・44・50位）と、
#: 1.2σ でだけ上位に入った出来高の60日 ÷ 250日（37位）。docs/MODEL_ADOPTION_RULES.md §68
FEATURES: Tuple[str, ...] = ("tx_vol_60", "tx_mdi_14", "tx_vortex_pos_14", "tx_atr_ratio_14_50", "tx_ichi_kijun_gap",
                             "tx_bb_width", "tx_tsi", "tx_stoch_k", "tx_vol_ratio_60_250")
#: 実験105 の順位（画面の説明用。§68 の「学習の寄与」）
GAIN_RANK = {"tx_vol_60": {"k15": 28, "k12": 30}, "tx_mdi_14": {"k15": 31}, "tx_vortex_pos_14": {"k15": 32, "k12": 16},
             "tx_atr_ratio_14_50": {"k15": 38, "k12": 25}, "tx_ichi_kijun_gap": {"k15": 40, "k12": 39},
             "tx_bb_width": {"k15": 42}, "tx_tsi": {"k15": 44}, "tx_stoch_k": {"k15": 50}, "tx_vol_ratio_60_250": {"k12": 37}}
#: 画面の短い名前（正式な名前は technicals.JA。画面ではホバーで出す）
SHORT = {"tx_vol_60": "60日ボラ", "tx_mdi_14": "−DI(14)", "tx_vortex_pos_14": "VI+(14)", "tx_atr_ratio_14_50": "ATR14÷ATR50",
         "tx_ichi_kijun_gap": "基準線乖離", "tx_bb_width": "BB幅", "tx_tsi": "TSI", "tx_stoch_k": "ストキャス%K",
         "tx_vol_ratio_60_250": "出来高60日÷250日"}
THEILER = int(B.RISE_HORIZON)       # 20営業日。これ以下しか離れていない日どうしは結ばない
RR_BP = 100                         # 既定の辺の密度（1万分率。100 = 1%）
RR_CHOICES_BP = (50, 100, 200, 300)  # 画面で選べる密度
Q = 20                              # 状態を 1/Q σ 刻みの整数で持つ
Z_CAP = 6.0                         # |z| をここで切る（Q 倍して ±120。int8 に収まる）
MIN_NODES = 100                     # ノードがこれより少ない銘柄は作らない
TARGET = float(LT.TAKE_PROFIT)      # +10% の指値
HOLD = int(LT.HOLD)                 # 20営業日目の終値
LABEL_KS = (("l12", 1.2), ("l15", 1.5))  # 目的変数（到達しきい値の σ の倍数）
N_NEAREST = 5                       # 今日に近い過去の更新日を何件出すか
#: 対象にする銘柄の規則（今日の戦略の購入条件）。live_track と同じ値（画面の strategy.js の RULE_MODELS・STRATEGY.agreePct と
#: 同じことは tests/test_live_track.py が確かめる）
RULE_MODELS: Tuple[str, ...] = tuple(LT.RULE_MODELS)
AGREE_PCT = float(LT.AGREE_PCT)
BARS_COLS = ["Date", "Code", "O", "H", "L", "C", "Vo", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo"]
ARCHIVE_DIR = "recurrence"          # archive の中のフォルダ名（デプロイで dist/data/ の下に展開する）


def log(msg: str) -> None:
    print(f"[recurrence] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 状態（9列の変換・標準化・丸め）
# --------------------------------------------------------------------------- #

def transform_states(X: np.ndarray, cols: Sequence[str] = FEATURES) -> Tuple[np.ndarray, List[dict]]:
    """
    n × 列 の生の値を、列の型（linear_preprocess.kind_of / STEPS）で変換し、平均0・標準偏差1（母標準偏差）にそろえる。
    統計は渡された行（その銘柄のノード全体）だけで決める。戻り値: (z, 列ごとの変換の値)。z は Z_CAP で切ってある。
    変換の値（lo・hi・s・mean・std）は画面が z から生の値の目安へ戻すのに使う。
    """
    X = np.asarray(X, dtype=float)
    Z = np.empty_like(X)
    params = []
    for j, col in enumerate(cols):
        kind = LP.kind_of(col)
        x = X[:, j:j + 1]
        p = {"col": col, "kind": kind, "lo": None, "hi": None, "s": None}
        for step in LP.STEPS[kind]():
            x = step.fit_transform(x)
            if isinstance(step, LP.ClipQuantile):
                lo, hi = float(step.lo_[0]), float(step.hi_[0])
                p["lo"] = lo if np.isfinite(lo) else None
                p["hi"] = hi if np.isfinite(hi) else None
            elif isinstance(step, LP.Asinh):
                p["s"] = float(step.scale_[0])
        x = x[:, 0]
        mean, std = float(np.mean(x)), float(np.std(x))
        if not (np.isfinite(std) and std > 0):
            std = 1.0                                        # 値が動かない列は 0 にそろうだけ（距離に効かない）
        Z[:, j] = np.clip((x - mean) / std, -Z_CAP, Z_CAP)
        p.update(mean=mean, std=std)
        params.append(p)
    return Z, params


def quantize(Z: np.ndarray, q: int = Q) -> np.ndarray:
    """1/q σ 刻みの整数（四捨五入。0.5 ちょうどは偶数へ寄せる numpy の rint）。"""
    return np.rint(np.asarray(Z, dtype=float) * q).astype(np.int16)


# --------------------------------------------------------------------------- #
# 辺（整数の距離の2乗・リカレンス率で決める ε）
# --------------------------------------------------------------------------- #

def sq_dist(zq: np.ndarray) -> np.ndarray:
    """
    丸めた状態どうしの距離の2乗（整数。n × n）。内積は浮動小数点の行列積で数える（値は |z|·Q ≤ 120 の整数なので、9列の積の和は
    高々 9 × 120² で 2^53 よりずっと小さく、足す順番によらず正確に整数になる）。
    """
    a = np.asarray(zq, dtype=np.float64)
    g = np.rint(a @ a.T).astype(np.int64)
    nrm = np.diag(g)
    return nrm[:, None] + nrm[None, :] - 2 * g


def threshold(s: np.ndarray, theiler: int = THEILER, rr_bp: int = RR_BP) -> Dict[str, int]:
    """
    結べる組（j − i > theiler）の距離の2乗を小さい順に並べた m 番目を ε² にする。m = rr_bp × 組の数 ÷ 10000 を四捨五入
    （整数で数える。画面の recurrence.js と同じ）、最小 1。結べる組が無ければ t = -1（辺なし）。
    """
    n = len(s)
    if n <= theiler + 1:
        return {"t": -1, "pairs": 0, "m": 0}
    iu, ju = np.triu_indices(n, k=theiler + 1)
    v = s[iu, ju]
    pairs = int(len(v))
    m = max(1, (int(rr_bp) * pairs + 5000) // 10000)
    m = min(m, pairs)
    t = int(np.partition(v, m - 1)[m - 1])
    return {"t": t, "pairs": pairs, "m": int(m)}


def adjacency(s: np.ndarray, t: int, theiler: int = THEILER) -> np.ndarray:
    """距離の2乗が t 以下で、theiler より離れた組を結ぶ（対称・対角は False）。"""
    n = len(s)
    idx = np.arange(n)
    far = np.abs(idx[:, None] - idx[None, :]) > theiler
    return (s <= t) & far


# --------------------------------------------------------------------------- #
# 要約（今日のノードの次数・近傍の結果）
# --------------------------------------------------------------------------- #

def _median(v: np.ndarray) -> Optional[float]:
    return float(np.median(v)) if len(v) else None


def _share(v: np.ndarray) -> Optional[float]:
    return float(np.mean(v)) if len(v) else None


def outcome_stats(idx: np.ndarray, ret: np.ndarray, labels: Dict[str, np.ndarray], target: float = TARGET) -> dict:
    """
    ノードの集まり idx の結果。ret は % （分からない日は NaN）。hit は ret が target に届いた割合（+10 は指値に届いた日だけ。
    20営業日目の終値が +10% 以上なら途中の高値も届いているので、ret >= target と指値に届いたことは同じ）。
    labels は目的変数（1 / 0 / NaN）。割合・中央値は分かっている日だけで数える（無ければ None）。
    """
    idx = np.asarray(idx, dtype=int)
    r = ret[idx]
    r = r[np.isfinite(r)]
    out = {"n": int(len(idx)), "known": int(len(r)),
           "hit": _share(r >= target - 1e-9), "retMed": _median(r)}
    for key, lab in labels.items():
        v = lab[idx]
        v = v[np.isfinite(v)]
        out[key] = _share(v)
        out[f"{key}N"] = int(len(v))
    return out


def summarize(zq: np.ndarray, ret: np.ndarray, labels: Dict[str, np.ndarray], fresh: Sequence[int],
              dates: Sequence[str], theiler: int = THEILER, rr_bp: int = RR_BP, q: int = Q) -> dict:
    """
    ネットワークの要約。今日 = 最後のノード。ret・labels は丸めた後の値（ファイルに書く値。画面と同じ数になるように）。
    戻り値の数値は丸めない（書き出しで丸める）。
    """
    n = len(zq)
    s = sq_dist(zq)
    th = threshold(s, theiler, rr_bp)
    A = adjacency(s, th["t"], theiler)
    deg = A.sum(axis=1)
    today = n - 1
    k = int(deg[today])
    nb = np.flatnonzero(A[today])
    fresh = np.asarray(sorted(set(int(i) for i in fresh)), dtype=int)
    fr_set = set(fresh.tolist())
    nb_fresh = np.array([i for i in nb if i in fr_set], dtype=int)
    # 今日に近い過去の更新日（タイラー窓より前。ε の外でも近い順に出す）
    past_fresh = fresh[fresh < today - theiler]
    order = past_fresh[np.argsort(s[today, past_fresh], kind="stable")][:N_NEAREST]
    nearest = []
    for i in order:
        rec = {"i": int(i), "date": dates[i], "d": float(np.sqrt(s[today, i]) / q), "inEps": bool(A[today, i]),
               "ret": None if not np.isfinite(ret[i]) else float(ret[i])}
        for key, lab in labels.items():
            rec[key] = None if not np.isfinite(lab[i]) else int(lab[i])
        nearest.append(rec)
    return {
        "rrBp": int(rr_bp), "theiler": int(theiler), "pairs": th["pairs"], "m": th["m"],
        "eps2": th["t"], "eps": (float(np.sqrt(th["t"]) / q) if th["t"] >= 0 else None),
        "edges": int(A.sum() // 2), "isolated": int((deg == 0).sum()),
        "deg": k, "degPct": float(100.0 * (np.sum(deg < k) + 0.5 * np.sum(deg == k)) / n),
        "degMed": float(np.median(deg)),
        "nbr": outcome_stats(nb, ret, labels), "base": outcome_stats(np.arange(n), ret, labels),
        "nbrFresh": outcome_stats(nb_fresh, ret, labels), "baseFresh": outcome_stats(fresh, ret, labels),
        "nearestFresh": nearest,
    }


def round_summary(x, nd: int = 4):
    """書き出し用に小数を丸める（入れ子もまとめて）。"""
    if isinstance(x, float):
        return round(x, nd) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: round_summary(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [round_summary(v, nd) for v in x]
    return x


# --------------------------------------------------------------------------- #
# 日足 → パネル（price_panel・更新日・目的変数・テクニカル指標）
# --------------------------------------------------------------------------- #

def load_bars(data_dir: str, codes: Iterable[str]) -> pd.DataFrame:
    """codes の銘柄の日足（price_panel に要る列）。年ごとの bars_<年>.parquet から、その銘柄の行だけを読む。"""
    codes = sorted({str(c) for c in codes})
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"日足（bars_*.parquet）がありません: {data_dir}")
    import pyarrow.parquet as pq

    parts = []
    for p in paths:
        have = set(pq.read_schema(p).names)
        cols = [c for c in BARS_COLS if c in have]
        part = pd.read_parquet(p, columns=cols, filters=[("Code", "in", codes)])
        for c in BARS_COLS:
            if c not in part.columns:
                part[c] = np.nan
        parts.append(part[BARS_COLS])
    b = pd.concat(parts, ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str)
    return b.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)


def build_panel(bars: pd.DataFrame, segments: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    日足（予測日で切ったもの）から、学習データと同じ作りの日足パネル（build_dataset.price_panel。一般市場で数え始める日も同じ）に
    78週高値の更新日・目的変数（LABEL_KS）・先20営業日の最大上昇率（future_rise）と最大下落率（future_fall）・テクニカル指標9列を
    付ける。予測日より後の行は渡さないこと（目的変数が先を見る）。
    """
    gm = B.general_market_start(bars, segments)
    start = gm.loc[gm["tpm"]].set_index("Code")["start"] if B.GENERAL_MARKET_START else None
    df = B.price_panel(bars, start=start)
    df = B.mark_new_highs(df)
    df = B.attach_rise_label(df)
    for key, k in LABEL_KS:
        df[key] = B.rise_label_for_k(df, k)
    # 最大下落率: future_rise（attach_rise_label）と同じ窓（t+1〜t+horizon の終値。売買不成立の日は飛ばす）・同じ基準
    # （entry_price = 翌営業日の始値）・同じ「先の行が horizon あるか」の判定で、最大を最小にしたもの
    h = int(B.RISE_HORIZON)
    g = df.groupby("Code", sort=False)
    have_forward = g.cumcount(ascending=False) >= h
    fmin = g["close"].transform(lambda s: s[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)).where(have_forward)
    df["future_fall"] = fmin / df["entry_price"] - 1.0
    tx = TX.compute(df[[c for c in B.TECH_PANEL_COLS if c in df.columns]])
    for c in FEATURES:
        df[c] = tx[c].to_numpy()
    return df


def _pct_list(a: np.ndarray) -> List[Optional[float]]:
    """% の値（0.1 刻みに丸めたもの）を書き出す形に（分からない日は None、整数は int）。"""
    return [None if not np.isfinite(x) else (int(round(x)) if float(x).is_integer() else float(x)) for x in a]


def _sig(x: float, digits: int = 4) -> Optional[float]:
    """有効数字 digits 桁（終値の表示用）。"""
    if not np.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def candidate_payload(g: pd.DataFrame, bars_code: pd.DataFrame, cand: dict, as_of: str) -> Tuple[Optional[dict], str]:
    """
    1銘柄のファイルの中身。g はその銘柄のパネルの行（予測日まで）、bars_code はその銘柄の日足（予測日まで）。
    作れなければ (None, 理由)。
    """
    day = pd.Timestamp(as_of)
    g = g[g["Date"] <= day]
    ok = np.isfinite(g[list(FEATURES)].to_numpy(dtype=float)).all(axis=1)
    g = g[ok].reset_index(drop=True)
    if not len(g) or pd.Timestamp(g["Date"].iloc[-1]) != day:
        return None, "予測日の9列がそろわない"
    if len(g) < MIN_NODES:
        return None, f"ノードが {len(g)} 日しかない（{MIN_NODES} 日未満）"
    X = g[list(FEATURES)].to_numpy(dtype=float)
    Z, params = transform_states(X)
    zq = quantize(Z)
    n = len(g)
    dates = [d.date().isoformat() for d in pd.to_datetime(g["Date"])]
    # 結果: live_track.forward（予測日までの日足だけ）。0.1% 刻みに丸めた値を書き、要約もその値で数える
    fw = LT.forward(bars_code, pd.DataFrame({"Code": str(cand["jqCode"]), "Date": pd.to_datetime(g["Date"]).to_numpy()}),
                    target=TARGET, hold=HOLD, fallback="close")
    ret = np.round(pd.to_numeric(fw["ret"], errors="coerce").to_numpy(dtype=float), 1)
    # 色付けの値（要約には使わない）: 先20営業日の最大上昇率・最大下落率（パネル）と20営業日目の騰落率（live_track.forward）
    fwd = {"rise": np.round(100.0 * pd.to_numeric(g["future_rise"], errors="coerce").to_numpy(dtype=float), 1),
           "fall": np.round(100.0 * pd.to_numeric(g["future_fall"], errors="coerce").to_numpy(dtype=float), 1),
           "r20": np.round(pd.to_numeric(fw["ret_close"], errors="coerce").to_numpy(dtype=float), 1)}
    labels = {key: pd.to_numeric(g[key], errors="coerce").to_numpy(dtype=float) for key, _ in LABEL_KS}
    fresh = np.flatnonzero(g["is_fresh_break"].fillna(False).to_numpy(dtype=bool)).tolist()
    summ = summarize(zq, ret, labels, fresh, dates)
    d = pd.to_datetime(g["Date"]).to_numpy(dtype="datetime64[D]")
    gaps = np.diff(d).astype(int).tolist()
    payload = {
        "v": VERSION, "date": as_of, "code": str(cand.get("code") or str(cand["jqCode"])[:4]),
        "jqCode": str(cand["jqCode"]), "name": cand.get("name") or "",
        "n": n, "start": dates[0], "gaps": gaps,
        "close": [_sig(x) for x in g["close"].to_numpy(dtype=float)],
        "q": Q, "features": list(FEATURES),
        "scale": [round_summary(p, 6) for p in params],
        "z": zq.reshape(-1).astype(int).tolist(),
        "ret": _pct_list(ret),
        **{key: [None if not np.isfinite(x) else int(x) for x in labels[key]] for key, _ in LABEL_KS},
        **{key: _pct_list(v) for key, v in fwd.items()},
        "fresh": [int(i) for i in fresh],
        "raw": [_sig(float(x), 6) for x in X[-1]],             # 今日の9列の生の値（有効数字6桁。画面の説明）
        "summary": round_summary(summ),
        "_raw": [float(x) for x in X[-1]],                     # 丸めない値（データセットとの突き合わせだけ。書き出さない）
    }
    return payload, ""


# --------------------------------------------------------------------------- #
# 対象（購入条件を満たした銘柄）
# --------------------------------------------------------------------------- #

def rule_min(cand: dict, models: Sequence[str] = RULE_MODELS) -> Optional[float]:
    """
    規則の3本の百分位（byModel の pctHistorical）の最小。1本でも無い（数でない・有限でない）なら None。画面の strategy.js の
    minPct と同じ（どれか1本でも欠けたまま「満たした」としない）。
    """
    per = cand.get("byModel")
    if not isinstance(per, dict):
        return None
    vals = []
    for a in models:
        m = per.get(a)
        v = m.get("pctHistorical") if isinstance(m, dict) else None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return None
        vals.append(float(v))
    return min(vals) if vals else None


def rule_candidates(cands: Sequence[dict], models: Sequence[str] = RULE_MODELS, line: float = AGREE_PCT) -> List[dict]:
    """
    購入条件を満たした候補（規則の3本の最小が line 以上）だけを、strategy.js の strategySignal の passed と同じ並び
    （最小の降順、同点は基準モデルのスコアの降順）で返す。
    """
    got = [(rule_min(c, models), i, c) for i, c in enumerate(cands)]
    got = [(m, i, c) for m, i, c in got if m is not None and m >= line]
    got.sort(key=lambda x: (-x[0], -float(x[2].get("score") or 0.0), x[1]))
    return [c for _, _, c in got]


# --------------------------------------------------------------------------- #
# 日ごと・全体
# --------------------------------------------------------------------------- #

def file_name(day: str, jq_code: str) -> str:
    return f"{day}_{jq_code}.json"


def index_entry(payload: dict, file: str) -> dict:
    s = payload["summary"]
    return {"jqCode": payload["jqCode"], "code": payload["code"], "name": payload["name"], "file": file,
            "n": payload["n"], "start": payload["start"], "summary": s}


def build_day(day: str, cands: List[dict], bars: pd.DataFrame, segments: Optional[pd.DataFrame]) -> Tuple[dict, Dict[str, dict]]:
    """
    1日ぶん。cands は作る銘柄（build では購入条件を満たした銘柄だけ）、bars はその銘柄の日足（全期間。ここで予測日で切る）。
    戻り値: (index の日の項目, {ファイル名: 中身})。銘柄ごとの失敗は理由を残して先へ進む。
    """
    t0 = time.time()
    built_at = pd.Timestamp.utcnow().isoformat()
    blk = {"date": day, "builtAt": built_at, "status": "ok", "candidates": [], "skipped": []}
    files: Dict[str, dict] = {}
    if not cands:
        blk["status"] = "no_candidates"
        return blk, files
    cut = bars[bars["Date"] <= pd.Timestamp(day)]
    codes = {str(c["jqCode"]) for c in cands}
    cut = cut[cut["Code"].isin(codes)]
    panel = build_panel(cut, segments) if len(cut) else pd.DataFrame(columns=["Code", "Date"])
    by_code = {c: g for c, g in panel.groupby("Code", sort=False)} if len(panel) else {}
    bars_by_code = {c: g for c, g in cut.groupby("Code", sort=False)}
    for cand in cands:
        jq = str(cand["jqCode"])
        try:
            if jq in by_code:
                payload, why = candidate_payload(by_code[jq], bars_by_code[jq], cand, day)
            else:
                payload, why = None, "日足が無い"
        except Exception as exc:                 # noqa: BLE001
            payload, why = None, f"{type(exc).__name__}: {exc}"[:200]
        if payload is None:
            blk["skipped"].append({"jqCode": jq, "code": str(cand.get("code") or jq[:4]), "reason": why})
            continue
        f = file_name(day, jq)
        files[f] = payload
        blk["candidates"].append(index_entry(payload, f))
    if blk["skipped"]:
        blk["status"] = "partial" if blk["candidates"] else "unavailable"
    # ログ（公開）には銘柄コードを出さない（理由だけ。どの銘柄かは目次の skipped にある）
    log(f"{day}: 購入条件を満たした {len(cands)}件のうち {len(blk['candidates'])}件を作った"
        + (f"（作れない {len(blk['skipped'])}件: " + "・".join(sorted({s['reason'] for s in blk["skipped"]}))[:300] + "）"
           if blk["skipped"] else "") + f" / {time.time() - t0:.1f}秒")
    return blk, files


def method_block() -> dict:
    ja = {c: TX.JA.get(c, c) for c in FEATURES}
    return {
        "features": [{"col": c, "ja": ja[c], "short": SHORT.get(c, ja[c]),
                      "group": TX.GROUP_JA.get(TX.GROUP_OF.get(c, ""), ""), "kind": LP.kind_of(c),
                      "gainRank": GAIN_RANK.get(c, {})} for c in FEATURES],
        "theiler": THEILER, "rrBp": RR_BP, "rrChoicesBp": list(RR_CHOICES_BP), "q": Q, "zCap": Z_CAP,
        "target": TARGET, "hold": HOLD, "labels": [{"key": k, "k": v} for k, v in LABEL_KS], "minNodes": MIN_NODES,
        "rule": {"models": list(RULE_MODELS), "agreePct": AGREE_PCT},
        "note": ("参考の表示。購入条件（1.5σ の3本の百分位がすべて95以上）を満たした銘柄ごとに、各営業日のテクニカル指標9列の"
                 "状態が似た日どうし（タイラー窓 20営業日より離れた組のうち、距離の近い順に辺の密度ぶん）を結んだリカレンス・"
                 "ネットワーク。選定には使わない（今日の戦略の並べ方のまま）。過去の日は、その日に作った値のまま"),
    }


def read_archive(path: Optional[str]) -> Tuple[Optional[dict], Dict[str, dict]]:
    """前の archive（無い・読めない・版が違うなら (None, {})）。"""
    if not path or not os.path.exists(path):
        return None, {}
    try:
        files: Dict[str, dict] = {}
        index = None
        with tarfile.open(path, "r:gz") as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(".json"):
                    continue
                name = os.path.basename(m.name)
                data = json.load(tf.extractfile(m))
                if name == "index.json":
                    index = data
                else:
                    files[name] = data
        if not index or index.get("v") != VERSION:
            log(f"前の archive は版が違うので使わない（{(index or {}).get('v')}）")
            return None, {}
        return index, files
    except Exception as exc:                     # noqa: BLE001
        log(f"前の archive を読めないので使わない: {type(exc).__name__}: {exc}")
        return None, {}


def carry(prev_index: Optional[dict], prev_files: Dict[str, dict], day: str, cands: List[dict]) -> Optional[Tuple[dict, Dict[str, dict]]]:
    """前の archive にこの日があり、作る銘柄（購入条件を満たした銘柄）が同じで、ファイルがそろっていれば、そのまま引き継ぐ。"""
    if not prev_index:
        return None
    blk = next((d for d in prev_index.get("days", []) if d.get("date") == day), None)
    if not blk or blk.get("status") not in ("ok", "partial", "no_candidates"):
        return None
    had = {c["jqCode"] for c in blk.get("candidates", [])} | {s["jqCode"] for s in blk.get("skipped", [])}
    if had != {str(c["jqCode"]) for c in cands}:
        return None
    files = {}
    for c in blk.get("candidates", []):
        if c["file"] not in prev_files:
            return None
        files[c["file"]] = prev_files[c["file"]]
    return blk, files


def check_dataset(files: Dict[str, dict], dataset_path: Optional[str]) -> Optional[dict]:
    """
    候補の日の9列（ファイルの raw）が、予測用のデータセットの同じ行の値と一致するか（check_train_serve と同じ許し）。
    データセットが無ければ None。
    """
    if not dataset_path or not os.path.exists(dataset_path) or not files:
        return None
    ds = pd.read_parquet(dataset_path, columns=["Code", "Date", *FEATURES])
    ds["Code"] = ds["Code"].astype(str)
    ds["Date"] = pd.to_datetime(ds["Date"]).dt.date.astype(str)
    ds = ds.set_index(["Code", "Date"])
    n = bad = 0
    worst: List[str] = []
    bad_cols: set = set()
    for f, p in files.items():
        key = (p["jqCode"], p["date"])
        if key not in ds.index:
            continue
        want = ds.loc[key, list(FEATURES)]
        want = want.iloc[0] if isinstance(want, pd.DataFrame) else want
        have = np.array([np.nan if v is None else v for v in p.get("_raw", p["raw"])], dtype=float)
        same = np.isclose(have, want.to_numpy(dtype=float), rtol=1e-6, atol=1e-9, equal_nan=True)
        n += 1
        if not same.all():
            bad += 1
            cols = [FEATURES[j] for j in np.flatnonzero(~same)]
            worst.append(f"{p['code']} {p['date']}: {cols}")
            bad_cols.update(cols)
    return {"checked": n, "mismatch": bad, "examples": worst[:5], "columns": sorted(bad_cols)}


def build(pred: dict, data_dir: str, prev_archive: Optional[str] = None, dataset_path: Optional[str] = None,
          days: Optional[Sequence[str]] = None) -> Tuple[dict, Dict[str, dict]]:
    """
    predictions.json（pred）の日ごとの候補のうち、購入条件を満たした銘柄（rule_candidates）にネットワークを作る。前の archive に
    ある日は引き継ぐ。日の項目の nBreak はその日の高値更新の件数（予測ファイルの候補の数。満たさなかった銘柄も数える）。
    戻り値: (index, {ファイル名: 中身})。
    """
    t0 = time.time()
    days = list(days or pred.get("dates") or [])
    by_day: Dict[str, List[dict]] = {d: [] for d in days}
    for c in pred.get("candidates") or []:
        if c.get("date") in by_day:
            by_day[c["date"]].append(c)
    n_break = {d: len(v) for d, v in by_day.items()}
    by_day = {d: rule_candidates(v) for d, v in by_day.items()}
    log("購入条件（" + " / ".join(RULE_MODELS) + f" がすべて{AGREE_PCT:g}以上）を満たした銘柄: "
        + ("・".join(f"{d} {len(by_day[d])}件（高値更新 {n_break[d]}件のうち）" for d in days) or "日なし"))
    prev_index, prev_files = read_archive(prev_archive)
    index = {"v": VERSION, "generatedAt": pd.Timestamp.utcnow().isoformat(), "asOf": pred.get("asOf") or (days[-1] if days else None),
             "method": method_block(), "days": []}
    files: Dict[str, dict] = {}
    todo = []
    for d in days:
        got = carry(prev_index, prev_files, d, by_day[d])
        if got is not None:
            blk, fs = got
            blk = dict(blk, carried=True, nBreak=n_break[d])
            index["days"].append(blk)
            files.update(fs)
        else:
            todo.append(d)
            index["days"].append({"date": d, "status": "pending"})
    log(f"日 {len(days)}日（引き継ぎ {len(days) - len(todo)}日・作る {len(todo)}日: {', '.join(todo) or 'なし'}）")
    if todo:
        codes = {str(c["jqCode"]) for d in todo for c in by_day[d]}
        bars = load_bars(data_dir, codes) if codes else pd.DataFrame(columns=BARS_COLS)
        segments = B.load_market_segments(data_dir)
        for d in todo:
            try:
                blk, fs = build_day(d, by_day[d], bars, segments)
            except Exception as exc:             # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"[:300]
                log(f"{d}: 作れない: {reason}")
                blk, fs = {"date": d, "builtAt": pd.Timestamp.utcnow().isoformat(), "status": "error",
                           "reason": reason, "candidates": [], "skipped": []}, {}
            blk["nBreak"] = n_break[d]
            i = next(k for k, x in enumerate(index["days"]) if x["date"] == d)
            index["days"][i] = blk
            files.update(fs)
            if d == days[-1]:
                chk = check_dataset(fs, dataset_path)
                if chk is not None:
                    blk["datasetCheck"] = chk
                    log(f"データセットとの突き合わせ（{d}）: {chk['checked']}件中 食い違い {chk['mismatch']}件"
                        + (f"（列: {', '.join(chk['columns'])}。候補は目次の datasetCheck）" if chk["mismatch"] else ""))
    log(f"{len(files)}ファイル / {time.time() - t0:.0f}秒")
    return index, files


FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_[0-9A-Za-z]+\.json$")


def write(index: dict, files: Dict[str, dict], out_dir: Optional[str], archive: Optional[str]) -> None:
    """
    index.json と銘柄ごとのファイルを out_dir に、同じものを archive（tar.gz。中は recurrence/）に書く。
    out_dir に前からある自分のファイル（index.json と <日付>_<コード>.json）は消してから書く（ほかのファイルには触らない）。
    「_」で始まる項目（突き合わせ用の丸めない値）は書き出さない。
    """
    def dump(p: dict) -> bytes:
        return json.dumps({k: v for k, v in p.items() if not k.startswith("_")},
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    blobs = {"index.json": dump(index)}
    for f, p in files.items():
        blobs[f] = dump(p)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            if f == "index.json" or FILE_RE.match(f):
                os.remove(os.path.join(out_dir, f))
        for f, b in blobs.items():
            with open(os.path.join(out_dir, f), "wb") as fh:
                fh.write(b)
    if archive:
        os.makedirs(os.path.dirname(os.path.abspath(archive)), exist_ok=True)
        tmp = archive + ".tmp"
        with tarfile.open(tmp, "w:gz") as tf:
            for f, b in sorted(blobs.items()):
                info = tarfile.TarInfo(f"{ARCHIVE_DIR}/{f}")
                info.size = len(b)
                info.mtime = int(time.time())
                tf.addfile(info, io.BytesIO(b))
        os.replace(tmp, archive)
    total = sum(len(b) for b in blobs.values())
    log(f"書いた: {len(blobs)}ファイル・{total / 1e6:.1f}MB"
        + (f" → {out_dir}" if out_dir else "") + (f" / {archive}（{os.path.getsize(archive) / 1e6:.1f}MB）" if archive else ""))


def summary_line(index: dict) -> str:
    parts = []
    for d in index.get("days", []):
        tag = "引き継ぎ" if d.get("carried") else d.get("status")
        nb = f"高値更新 {d['nBreak']}件のうち・" if isinstance(d.get("nBreak"), int) else ""
        parts.append(f"{d['date']} {len(d.get('candidates', []))}件（{nb}{tag}）")
    return "・".join(parts) or "日なし"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="購入条件を満たした銘柄ごとのリカレンス・ネットワーク（画面の「リカレンス・ネットワーク」タブ用）")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data"))
    ap.add_argument("--predictions", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                          "public", "data", "predictions.json"))
    ap.add_argument("--out-dir", default=None, help="index.json と銘柄ごとのファイルを書くフォルダ（中身は作り直す）")
    ap.add_argument("--archive", default=None, help="同じものを tar.gz に書く（Release に上げる。中は recurrence/）")
    ap.add_argument("--prev", default=None, help="前の archive（過去の日を引き継ぐ。既定は --archive と同じ場所）")
    ap.add_argument("--dataset", default=None, help="予測用のデータセット（候補の日の9列を突き合わせる）。既定は data-dir の dataset_predict.parquet")
    ap.add_argument("--days", nargs="*", default=None, help="作る日（既定は predictions.json の dates）")
    args = ap.parse_args(argv)
    if not args.out_dir and not args.archive:
        ap.error("--out-dir か --archive のどちらかを指定してください")
    with open(args.predictions, encoding="utf-8") as fh:
        pred = json.load(fh)
    prev = args.prev or args.archive
    dataset = args.dataset or os.path.join(args.data_dir, "dataset_predict.parquet")
    index, files = build(pred, args.data_dir, prev_archive=prev, dataset_path=dataset, days=args.days)
    write(index, files, args.out_dir, args.archive)
    log(summary_line(index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
