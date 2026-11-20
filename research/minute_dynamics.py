"""
分足から作る価格帯別出来高の「推移」の特徴量（実験133。研究用で、本番の特徴量には入れていない）。

運用者の依頼（2026-11-19）「独立して、この情報や直近のテクニカルだけを使ったモデル（機械学習に限らない）を作りたい。…
分布の形自体というよりは、その形の推移が大事なのかなとも思います。分布自体の微分ともいいますか。フィッシャー情報量とかが
一番近い概念かもしれません」。

行（銘柄・日 D）ごとに、D を含む直近20営業日の分足を、日ごとに「D の終値からの距離」の 1% 刻みの帯（−30%〜+15%。外は両端に
まとめる）に置いた分布にする。価格は分割・併合の係数で D の基準にそろえる（volume_profile.share_factors。株数は割る）。
20営業日の窓が集計できる行（volume_profile.window_profiles の 20日が ok）だけに値を付け、ほかは欠測のまま。

直近 = D を含む5営業日、その前 = それより前の15営業日。

  fr_recent_prior        直近とその前の分布のフィッシャー・ラオ距離 2·arccos(Σ√(p·q))（形がどれだけ変わったか）
  hellinger_recent_prior 同じくヘリンジャー距離 √(1 − Σ√(p·q))
  w1_recent_prior        ワッサースタイン距離（価格の軸の上で出来高をどれだけ動かせば重なるか。終値に対する割合）
  mean_shift             出来高の重心の移動（直近 − その前。終値に対する割合。＋ は上へ）
  poc_shift              最多出来高の帯の移動（同上）
  entropy_recent         直近の分布のエントロピー（小さいほど1か所に集まる）
  entropy_change         エントロピーの変化（直近 − その前。− は集まってきた）
  va_width_recent        直近の 70% 範囲の幅（終値に対する割合）
  va_width_log_ratio     70% 範囲の幅の比の対数（直近 ÷ その前。− は狭まった）
  above_close_change     終値より上の株数の割合の変化（直近 − その前。− は上値の出来高が減った＝しこりの消化）
  near_close_change      終値 ±1% の株数の割合の変化（＋ は今の値の近くに集まってきた）
  speed_recent           日ごとの分布の、隣り合う日どうしのフィッシャー・ラオ距離の平均（直近の5回。形の変わる速さ）
  speed_prior            同じく、その前の14回
  speed_log_ratio        速さの比の対数（直近 ÷ その前。＋ は形の変わり方が速くなった＝加速）
  fisher_info_recent     隣り合う日どうしの距離の2乗の平均（直近。時間に対するフィッシャー情報量の見積もり）
  fisher_info_prior      同じく、その前
  last_day_fr            D と前日の分布のフィッシャー・ラオ距離（いちばん新しい変化）
  volume_log_ratio       1日あたりの株数の比の対数（直近 ÷ その前）
  close_share_change     引けの株数の割合の変化（直近 − その前）
  open_share_change      寄りの株数の割合の変化（直近 − その前）
  delta_b0〜b7           帯（minute_features.SHAPE_EDGES の8つ）ごとの株数の割合の変化（直近 − その前。出来高がどこへ移ったか）
  shape5_b0〜b7          直近5営業日の分布の形（8つの帯の割合）
  shape20_b0〜b7         20営業日の分布の形（8つの帯の割合）

形だけの推移（vp_shp_*。運用者の選択 2026-11-20「形だけの推移で測り直す」。上の距離は D の終値からの距離の帯で測るので、値が
数%動くだけで分布が重ならなくなり、形の変化より値の移動を測っていた）: 日ごと・期間ごとの分布を、位置（VWAP）・広がり（株数で
重みを付けた価格の標準偏差）・形（VWAP を引いて標準偏差で割った分布。Z_EDGES の帯）に分ける。値が動いても広がりが変わっても、
形が同じなら形の距離は 0。

  shape_fr_recent_prior    直近とその前の形のフィッシャー・ラオ距離
  shape_speed_recent       隣り合う日どうしの形のフィッシャー・ラオ距離の平均（直近の5回）
  shape_speed_prior        同じく、その前の14回
  shape_speed_log_ratio    形の変わる速さの比の対数（(直近 + 0.01) ÷ (その前 + 0.01)）
  shape_info_recent        隣り合う日どうしの形の距離の2乗の平均（直近。形のフィッシャー情報量の見積もり）
  shape_info_prior         同じく、その前
  last_day_shape_fr        D と前日の形のフィッシャー・ラオ距離
  shape_entropy_recent     直近の形のエントロピー
  shape_entropy_change     形のエントロピーの変化（直近 − その前）
  skew_recent              直近の分布の歪度（株数で重み。＋ は上側に裾が長い）
  skew_change              歪度の変化（直近 − その前）
  kurt_recent              直近の分布の尖度（正規分布で 0）
  kurt_change              尖度の変化（直近 − その前）
  sigma_recent             直近の広がり（VWAP に対する割合）
  sigma_log_ratio          広がりの比の対数（直近 ÷ その前）
  sigma_slope              日ごとの広がり（VWAP に対する割合）の対数の、日数に対する傾き
  vwap_drift_sigma         VWAP の移動を、その前の広がりで割った値（位置が広がりの何倍動いたか）
  vwap_up_share            日ごとの VWAP が前の日より上がった日の割合
  z_close_recent           D の終値の、直近の分布の中での位置（VWAP からの距離 ÷ 広がり）
  z_poc_change             形の最多の帯の位置の変化（直近 − その前。広がりの単位）
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import minute_features as MF
import volume_profile as VP

WINDOW = 20
RECENT = 5
#: 1% 刻みの帯の境目（D の終値からの距離）。両端は外をまとめる
FINE_EDGES = np.concatenate([[-np.inf], np.round(np.arange(-0.30, 0.15 + 1e-9, 0.01), 4), [np.inf]])
#: 帯の代表値（両端は 0.5% 外側）
FINE_CENTERS = np.concatenate([[-0.305], (FINE_EDGES[1:-2] + FINE_EDGES[2:-1]) / 2, [0.155]])
#: 帯の幅（両端も 1% とみなす）
FINE_WIDTH = 0.01

#: 形（標準化した分布）の帯。VWAP から広がりの何倍離れているか（±3 の外は両端にまとめる）
Z_EDGES = np.concatenate([[-np.inf], np.round(np.arange(-3.0, 3.0 + 1e-9, 0.25), 4), [np.inf]])
Z_CENTERS = np.concatenate([[-3.125], (Z_EDGES[1:-2] + Z_EDGES[2:-1]) / 2, [3.125]])
#: 形の変わる速さの比を取るときに足す値（速さがちょうど 0 の日が続いても比が出るように）
SPEED_EPS = 0.01
SHAPE_DYN_SCALAR = ["shape_fr_recent_prior", "shape_speed_recent", "shape_speed_prior", "shape_speed_log_ratio",
                    "shape_info_recent", "shape_info_prior", "last_day_shape_fr", "shape_entropy_recent",
                    "shape_entropy_change", "skew_recent", "skew_change", "kurt_recent", "kurt_change", "sigma_recent",
                    "sigma_log_ratio", "sigma_slope", "vwap_drift_sigma", "vwap_up_share", "z_close_recent", "z_poc_change"]
#: 形だけの推移の列
SHAPE_DYN_COLUMNS = [f"vp_shp_{c}" for c in SHAPE_DYN_SCALAR]

SCALAR_COLUMNS = ["fr_recent_prior", "hellinger_recent_prior", "w1_recent_prior", "mean_shift", "poc_shift",
                  "entropy_recent", "entropy_change", "va_width_recent", "va_width_log_ratio", "above_close_change",
                  "near_close_change", "speed_recent", "speed_prior", "speed_log_ratio", "fisher_info_recent",
                  "fisher_info_prior", "last_day_fr", "volume_log_ratio", "close_share_change", "open_share_change"]
N_SHAPE = len(MF.SHAPE_EDGES) - 1
DELTA_COLUMNS = [f"delta_b{i}" for i in range(N_SHAPE)]
SHAPE5_COLUMNS = [f"shape5_b{i}" for i in range(N_SHAPE)]
SHAPE20_COLUMNS = [f"shape20_b{i}" for i in range(N_SHAPE)]
#: 推移の列（分布の微分）
DYNAMIC_COLUMNS = [f"vp_dyn_{c}" for c in SCALAR_COLUMNS + DELTA_COLUMNS]
#: 形の列（直近5日・20日）
SHAPE_COLUMNS = [f"vp_dyn_{c}" for c in SHAPE5_COLUMNS + SHAPE20_COLUMNS]
COLUMNS = DYNAMIC_COLUMNS + SHAPE_COLUMNS


def day_hist(points: np.ndarray, factor: float, close: float, edges=FINE_EDGES) -> np.ndarray:
    """1日の分足（価格ごとの株数）を、D の値の基準（係数 factor）にそろえ、D の終値からの距離の帯に置いた株数。"""
    pts = np.asarray(points, dtype=float)
    if not len(pts):
        return np.zeros(len(edges) - 1)
    rel = pts[:, 0] * factor / close - 1.0
    return np.histogram(rel, bins=edges, weights=pts[:, 1] / factor)[0].astype(float)


def normalize(h: np.ndarray) -> Optional[np.ndarray]:
    total = float(h.sum())
    return h / total if total > 0 else None


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.clip(np.sum(np.sqrt(p * q)), 0.0, 1.0))


def fisher_rao(p: np.ndarray, q: np.ndarray) -> float:
    """確率の単体の上のフィッシャー・ラオ距離（0〜π）。"""
    return 2.0 * math.acos(bhattacharyya(p, q))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return math.sqrt(max(0.0, 1.0 - bhattacharyya(p, q)))


def wasserstein1(p: np.ndarray, q: np.ndarray, width: float = FINE_WIDTH) -> float:
    """帯の並びの上の1次のワッサースタイン距離（累積の差の面積）。"""
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))[:-1]) * width)


def entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


def value_area(p: np.ndarray, share: float = 0.7) -> Tuple[float, float]:
    """最多の帯から左右の多い側へ広げて share に達する範囲（volume_profile.make_profile と同じ広げ方）。両端の境目。"""
    peak = int(np.argmax(p))
    lo = hi = peak
    included = p[peak]
    while included < share - 1e-12:
        left = p[lo - 1] if lo > 0 else -1.0
        right = p[hi + 1] if hi + 1 < len(p) else -1.0
        if left < 0 and right < 0:
            break
        if left >= right:
            lo -= 1
            included += left
        else:
            hi += 1
            included += right
    bounds = np.concatenate([[FINE_CENTERS[0] - FINE_WIDTH / 2], FINE_EDGES[1:-1], [FINE_CENTERS[-1] + FINE_WIDTH / 2]])
    return float(bounds[lo]), float(bounds[hi + 1])


def coarse(p: np.ndarray) -> np.ndarray:
    """1% の帯を minute_features.SHAPE_EDGES の8つの帯にまとめる（帯の代表値で振り分ける）。"""
    idx = np.digitize(FINE_CENTERS, MF.SHAPE_EDGES[1:-1], right=False)
    return np.bincount(idx, weights=p, minlength=N_SHAPE)[:N_SHAPE]


def moments(price: np.ndarray, shares: np.ndarray):
    """株数で重みを付けた (VWAP, 標準偏差, 歪度, 尖度（正規分布で 0）)。株数が無ければ None。広がりが 0 なら歪度・尖度は 0。"""
    w = np.asarray(shares, dtype=float)
    p = np.asarray(price, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return None
    v = float(w @ p / total)
    sd = math.sqrt(max(float(w @ (p - v) ** 2 / total), 0.0))
    if sd <= 1e-12 * max(abs(v), 1.0):
        return v, 0.0, 0.0, 0.0
    z = (p - v) / sd
    return v, sd, float(w @ z ** 3 / total), float(w @ z ** 4 / total) - 3.0


def standard_shape(price: np.ndarray, shares: np.ndarray, mom) -> np.ndarray:
    """位置（VWAP）と広がり（標準偏差）を除いた形（Z_EDGES の帯の割合）。広がりが 0 なら 0 の帯にすべて置く。"""
    v, sd = mom[0], mom[1]
    w = np.asarray(shares, dtype=float)
    z = (np.asarray(price, dtype=float) - v) / sd if sd > 0 else np.zeros(len(w))
    h = np.histogram(z, bins=Z_EDGES, weights=w)[0].astype(float)
    return h / h.sum()


def shape_features(days: List[Optional[Tuple[np.ndarray, np.ndarray]]], close: float) -> Optional[Dict[str, float]]:
    """
    20営業日の日ごとの (価格, 株数)（古い順。D の値の基準。約定の無い日は None）から、形だけの推移の列。
    直近（最後の RECENT 日）とその前のどちらかに約定が無ければ None。
    """
    if len(days) != WINDOW:
        return None

    def window(part):
        got = [d for d in part if d is not None and len(d[0])]
        if not got:
            return None
        return np.concatenate([d[0] for d in got]), np.concatenate([d[1] for d in got])

    rec, pri = window(days[-RECENT:]), window(days[:-RECENT])
    if rec is None or pri is None:
        return None
    m_r, m_p = moments(*rec), moments(*pri)
    if m_r is None or m_p is None or m_r[1] <= 0 or m_p[1] <= 0:
        return None
    f_r, f_p = standard_shape(*rec, m_r), standard_shape(*pri, m_p)
    daily = []
    for d in days:
        mom = moments(*d) if d is not None and len(d[0]) else None
        daily.append(None if mom is None else (mom, standard_shape(d[0], d[1], mom)))
    steps = [(t, fisher_rao(daily[t - 1][1], daily[t][1])) for t in range(1, WINDOW)
             if daily[t - 1] is not None and daily[t] is not None]
    rs = [x for t, x in steps if t >= WINDOW - RECENT]
    ps = [x for t, x in steps if t < WINDOW - RECENT]
    rel_sd = [(t, daily[t][0][1] / daily[t][0][0]) for t in range(WINDOW) if daily[t] is not None and daily[t][0][1] > 0]
    slope = float(np.polyfit([t for t, _ in rel_sd], np.log([x for _, x in rel_sd]), 1)[0]) if len(rel_sd) >= 3 else np.nan
    vw = [daily[t][0][0] if daily[t] is not None else None for t in range(WINDOW)]
    ups = [vw[t] > vw[t - 1] for t in range(1, WINDOW) if vw[t] is not None and vw[t - 1] is not None]
    out = {
        "shape_fr_recent_prior": fisher_rao(f_r, f_p),
        "shape_speed_recent": float(np.mean(rs)) if rs else np.nan,
        "shape_speed_prior": float(np.mean(ps)) if ps else np.nan,
        "shape_info_recent": float(np.mean(np.square(rs))) if rs else np.nan,
        "shape_info_prior": float(np.mean(np.square(ps))) if ps else np.nan,
        "last_day_shape_fr": float(dict(steps).get(WINDOW - 1, np.nan)),
        "shape_entropy_recent": entropy(f_r),
        "shape_entropy_change": entropy(f_r) - entropy(f_p),
        "skew_recent": m_r[2], "skew_change": m_r[2] - m_p[2],
        "kurt_recent": m_r[3], "kurt_change": m_r[3] - m_p[3],
        "sigma_recent": m_r[1] / m_r[0],
        "sigma_log_ratio": math.log((m_r[1] / m_r[0]) / (m_p[1] / m_p[0])),
        "sigma_slope": slope,
        "vwap_drift_sigma": (m_r[0] - m_p[0]) / m_p[1],
        "vwap_up_share": float(np.mean(ups)) if ups else np.nan,
        "z_close_recent": (close - m_r[0]) / m_r[1],
        "z_poc_change": float(Z_CENTERS[int(np.argmax(f_r))] - Z_CENTERS[int(np.argmax(f_p))]),
    }
    sr, sp = out["shape_speed_recent"], out["shape_speed_prior"]
    # 張り付き（1つの値だけ）の日が続くと形の距離がちょうど 0 になるので、0.01 を足してから比べる
    out["shape_speed_log_ratio"] = (math.log((sr + SPEED_EPS) / (sp + SPEED_EPS)) if np.isfinite(sr) and np.isfinite(sp)
                                    else np.nan)
    return out


def features(hists: List[np.ndarray], buckets: List[np.ndarray]) -> Optional[Dict[str, float]]:
    """
    20営業日の日ごとの株数（古い順。hists は 1% の帯、buckets は寄り・前場・後場・引け）から、推移と形の列。
    直近（最後の RECENT 日）とその前のどちらかに約定が無ければ None。
    """
    if len(hists) != WINDOW:
        return None
    H = np.vstack(hists)
    recent, prior = normalize(H[-RECENT:].sum(axis=0)), normalize(H[:-RECENT].sum(axis=0))
    whole = normalize(H.sum(axis=0))
    if recent is None or prior is None or whole is None:
        return None
    daily = [normalize(h) for h in hists]
    steps = [(t, fisher_rao(daily[t - 1], daily[t])) for t in range(1, WINDOW)
             if daily[t - 1] is not None and daily[t] is not None]
    rec = [d for t, d in steps if t >= WINDOW - RECENT]
    pri = [d for t, d in steps if t < WINDOW - RECENT]
    above = FINE_CENTERS > 0
    near = np.abs(FINE_CENTERS) < 0.01
    lo_r, hi_r = value_area(recent)
    lo_p, hi_p = value_area(prior)
    B = np.vstack(buckets).astype(float)
    b_rec, b_pri = B[-RECENT:].sum(axis=0), B[:-RECENT].sum(axis=0)
    share_rec = b_rec / b_rec.sum() if b_rec.sum() > 0 else np.full(len(b_rec), np.nan)
    share_pri = b_pri / b_pri.sum() if b_pri.sum() > 0 else np.full(len(b_pri), np.nan)
    vol_rec = H[-RECENT:].sum() / RECENT
    vol_pri = H[:-RECENT].sum() / (WINDOW - RECENT)
    out = {
        "fr_recent_prior": fisher_rao(recent, prior),
        "hellinger_recent_prior": hellinger(recent, prior),
        "w1_recent_prior": wasserstein1(recent, prior),
        "mean_shift": float(recent @ FINE_CENTERS - prior @ FINE_CENTERS),
        "poc_shift": float(FINE_CENTERS[int(np.argmax(recent))] - FINE_CENTERS[int(np.argmax(prior))]),
        "entropy_recent": entropy(recent),
        "entropy_change": entropy(recent) - entropy(prior),
        "va_width_recent": hi_r - lo_r,
        "va_width_log_ratio": math.log((hi_r - lo_r) / (hi_p - lo_p)),
        "above_close_change": float(recent[above].sum() - prior[above].sum()),
        "near_close_change": float(recent[near].sum() - prior[near].sum()),
        "speed_recent": float(np.mean(rec)) if rec else np.nan,
        "speed_prior": float(np.mean(pri)) if pri else np.nan,
        "fisher_info_recent": float(np.mean(np.square(rec))) if rec else np.nan,
        "fisher_info_prior": float(np.mean(np.square(pri))) if pri else np.nan,
        "last_day_fr": float(dict(steps).get(WINDOW - 1, np.nan)),
        "volume_log_ratio": math.log(vol_rec / vol_pri) if vol_rec > 0 and vol_pri > 0 else np.nan,
        "close_share_change": float(share_rec[3] - share_pri[3]),
        "open_share_change": float(share_rec[0] - share_pri[0]),
    }
    sr, sp = out["speed_recent"], out["speed_prior"]
    out["speed_log_ratio"] = math.log(sr / sp) if np.isfinite(sr) and np.isfinite(sp) and sr > 0 and sp > 0 else np.nan
    for i, v in enumerate(coarse(recent) - coarse(prior)):
        out[f"delta_b{i}"] = float(v)
    for i, v in enumerate(coarse(recent)):
        out[f"shape5_b{i}"] = float(v)
    for i, v in enumerate(coarse(whole)):
        out[f"shape20_b{i}"] = float(v)
    return out


def dynamics(keys: pd.DataFrame, minute_dir: str, data_dir: str) -> Tuple[pd.DataFrame, dict]:
    """
    keys（Code, Date）に推移と形の列（COLUMNS）を付けた表（keys と同じ行・同じ並び）と、件数だけの記録。
    20営業日の窓が集計できる行（window_profiles の 20日が ok・D の終値がある）だけに値を付ける。
    """
    out = pd.DataFrame({"Code": keys["Code"].to_numpy(), "Date": pd.to_datetime(keys["Date"]).to_numpy()})
    for c in COLUMNS + SHAPE_DYN_COLUMNS:
        out[c] = np.nan
    report = {"rows": int(len(out))}
    inp = MF.load_inputs(keys, minute_dir, data_dir, lookback_days=WINDOW)
    if inp is None:
        report["attached"] = 0
        return out, report
    status = Counter()
    col_index = {c: out.columns.get_loc(f"vp_dyn_{c}") for c in SCALAR_COLUMNS + DELTA_COLUMNS
                 + SHAPE5_COLUMNS + SHAPE20_COLUMNS}
    shp_index = {c: out.columns.get_loc(f"vp_shp_{c}") for c in SHAPE_DYN_SCALAR}
    shp_status = Counter()
    for i in inp.target:
        code, day = inp.codes[i], inp.iso[i]
        daily = inp.quotes.get(code, {})
        series = {"tick": {}, "minute": inp.minute.get(code, {})}
        dates = VP.lookback(inp.sessions, day, WINDOW)
        result = VP.window_profiles(dates, WINDOW, daily, series)["minute"]
        close = daily.get(day).close if daily.get(day) and daily[day].valid else None
        if result["status"] != "ok" or close is None:
            status[result["status"] if result["status"] != "ok" else "close_missing"] += 1
            continue
        factors = VP.share_factors(dates, daily)
        if factors is None:
            status["share_basis_unknown"] += 1
            continue
        hists, buckets, days = [], [], []
        for d in dates:
            sess = series["minute"].get(d)
            if sess is None or not daily[d].valid:              # 売買の無い日（window_profiles は通している）
                hists.append(np.zeros(len(FINE_EDGES) - 1))
                buckets.append(np.zeros(len(VP.BUCKETS)))
                days.append(None)
                continue
            hists.append(day_hist(sess.points, factors[d], close))
            buckets.append(np.asarray(sess.buckets, dtype=float) / factors[d])
            pts = np.asarray(sess.points, dtype=float)
            days.append((pts[:, 0] * factors[d], pts[:, 1] / factors[d]) if len(pts) else None)
        shp = shape_features(days, close)
        if shp is None:
            shp_status["no_shape"] += 1
        else:
            shp_status["ok"] += 1
            for k, v in shp.items():
                out.iat[i, shp_index[k]] = v
        feats = features(hists, buckets)
        if feats is None:
            status["no_trades_in_part"] += 1
            continue
        status["ok"] += 1
        for k, v in feats.items():
            out.iat[i, col_index[k]] = v
    report["target_rows"] = int(len(inp.target))
    report["attached"] = int(out[COLUMNS].notna().any(axis=1).sum())
    report["status"] = dict(sorted(status.items()))
    report["shape_status"] = dict(sorted(shp_status.items()))
    report["shape_attached"] = int(out[SHAPE_DYN_COLUMNS].notna().any(axis=1).sum())
    return out, report
