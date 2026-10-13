#!/usr/bin/env python3
"""
週次の再学習で新しく学習に加わったサンプルから、正例・負例を最大 N 件ずつ選び、画面のチャート用に書き出す。

運用者の依頼（2026-10-13）「新たにタブとして追加してほしいのが、直近追加されたサンプルのチャート（週次のリトレインで
新たに追加されたサンプルのチャート）、正例、負例で5例ずつ程度で良いので、表示してほしいです。正例は5例もないかも
しれませんが、なければないで大丈夫です」。

加わったサンプル
  本番の学習（research/train_production.py）は、ラベルが確定した全期間で学び直す。学習期間の終わり（meta.json の
  trainTo）は毎週延びるので、**前回の trainTo より後・今回の trainTo まで**の更新日のサンプルが、今回はじめて学習に
  入った分。前回の trainTo は research/model/meta.json の git の履歴から取る（週次の再学習が毎回コミットする）。
  例: 2026-10-11 の再学習は trainTo 2026-09-08、前回（10/4）は 2026-09-01 → 9/2〜9/8 の更新日のサンプル。

選び方
  正例・負例それぞれから、学習期間の終わりの日付を種にした無作為抽出で最大 N 件（同じ週なら何度回しても同じ）。
  足りなければあるだけ。

チャート
  research/export_label_samples.build_case（ラベルの目視検証と同じ部品）で、更新日の前後の終値・判定の20営業日・
  到達に必要な水準（翌営業日の寄りから +k×σ）・終盤に必要な水準・78週高値・移動平均を、学習のラベルと同じ式で引く。
  成否の内訳（到達したか・終盤の水準・20日目の MA5≥MA20）はデータセットの材料（future_rise / end_level /
  uptrend_end と vol_20d から引いたしきい値）から出す。

確かめ（件数だけ出す）
  - 内訳から引き直したラベルが、データセットのラベルと加わった全件で一致するか
  - チャートの基準の価格・最大上昇率が、データセットの値と食い違わないか（表示する分）

出力: public/data/new_samples.json（画面のタブ「再学習で加わったサンプル」が読む）。ログには件数と日付だけ（銘柄は出さない）。

  python3 research/new_samples.py                      # research/_data の dataset.parquet と日足から
  python3 research/new_samples.py --prev-to 2026-09-01  # 前回の学習期間の終わりを手で与える
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import build_dataset as B  # noqa: E402
import export_label_samples as EL  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
META = os.path.join(HERE, "model", "meta.json")
META_REL = "research/model/meta.json"
OUT = os.path.join(ROOT, "public", "data", "new_samples.json")
N_EACH = 5
#: 更新日の前に何営業日ぶん見せるか。78週（78週高値を測る期間と同じ368営業日。運用者の指示 2026-10-13 夜「チャートの
#: 表示期間は78週にしてください」。それまでは40営業日）。後ろは判定の20営業日＋5日
BEFORE = B.HIGH_WINDOW
AFTER = B.RISE_HORIZON + 5
DS_COLS = ["Code", "Date", "label", "vol_20d", "entry_price", "future_rise", "end_level", "uptrend_end"]


def prev_window(cur_to: str, root: str = ROOT, rel: str = META_REL) -> Optional[Dict]:
    """meta.json の git の履歴を新しい順にたどり、trainTo が今回より前の最初の版（前回の再学習）。無ければ None。"""
    try:
        revs = subprocess.check_output(["git", "log", "--format=%H", "--", rel], cwd=root,
                                       stderr=subprocess.DEVNULL).decode().split()
    except (OSError, subprocess.CalledProcessError):
        return None
    for rev in revs:
        try:
            m = json.loads(subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=root,
                                                   stderr=subprocess.DEVNULL))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        to = m.get("trainTo")
        if to and to < cur_to:
            return {"trainTo": to, "trainedAt": m.get("trainedAt"), "nTrain": m.get("nTrain")}
    return None


def added_rows(ds: pd.DataFrame, prev_to: Optional[str], cur_to: str) -> pd.DataFrame:
    """前回の学習期間の終わりより後・今回の終わりまでの、ラベルが確定した行。"""
    d = pd.to_datetime(ds["Date"])
    m = (d <= pd.Timestamp(cur_to)) & ds["label"].notna()
    if prev_to:
        m &= d > pd.Timestamp(prev_to)
    return ds[m].copy()


def verdict_parts(rows: pd.DataFrame) -> pd.DataFrame:
    """
    ラベルの内訳（データセットの材料から。export_label_samples.verdict と同じ式）。
    列: need / end_need（しきい値、小数）、reached / end_ok / uptrend（bool）、ok（3つとも）。
    """
    cfg = B.DEFAULT_RISE
    need, end_need = B.rise_thresholds(rows.get("vol_20d"), cfg)
    need.index = rows.index
    end_need.index = rows.index
    rise = pd.to_numeric(rows["future_rise"], errors="coerce")
    end = pd.to_numeric(rows["end_level"], errors="coerce")
    up = pd.to_numeric(rows["uptrend_end"], errors="coerce")
    out = pd.DataFrame(index=rows.index)
    out["need"] = need
    out["end_need"] = end_need
    out["reached"] = (rise >= need).fillna(False)
    out["end_ok"] = (end >= end_need).fillna(False) if cfg.end_ratio is not None else True
    out["uptrend"] = (up == 1.0).fillna(False) if cfg.require_uptrend else True
    out["ok"] = out["reached"] & out["end_ok"] & out["uptrend"]
    return out


def pick(rows: pd.DataFrame, n: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """正例・負例からそれぞれ最大 n 件（種を固定した無作為抽出）。日付・銘柄の順に並べる。"""
    out = []
    for lab in (1, 0):
        g = rows[rows["label"].astype(float) == lab]
        s = g.sample(n=min(n, len(g)), random_state=seed) if len(g) else g
        out.append(s.sort_values(["Date", "Code"]))
    return out[0], out[1]


def load_bars_for(data_dir: str, codes: List[str]) -> pd.DataFrame:
    """選んだ銘柄だけの日足（年ごとのファイルから Code で絞って読む）。"""
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    parts = []
    for p in paths:
        try:
            parts.append(pd.read_parquet(p, filters=[("Code", "in", list(codes))]))
        except Exception:                                     # noqa: BLE001
            b = pd.read_parquet(p)
            parts.append(b[b["Code"].astype(str).isin(codes)])
    bars = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(bars):
        bars["Code"] = bars["Code"].astype(str)
    return bars


def load_names(data_dir: str) -> Dict[str, str]:
    """銘柄名（master.parquet の CoName）。無ければ空。"""
    p = os.path.join(data_dir, "master.parquet")
    if not os.path.exists(p):
        return {}
    try:
        m = pd.read_parquet(p)
    except Exception:                                         # noqa: BLE001
        return {}
    if "Code" not in m.columns or "CoName" not in m.columns:
        return {}
    return dict(zip(m["Code"].astype(str), m["CoName"].astype(str)))


def display_code(jq: str) -> str:
    """5桁の J-Quants コードを画面の4桁に（末尾 0 の普通株だけ）。"""
    return jq[:4] if len(jq) == 5 and jq.endswith("0") else jq


def _r(v, d=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if np.isfinite(v) else None


#: チャートに要る項目だけ持ち出す（厳しい案の比較・出来高は使わない。画面が描かない移動平均・78週高値の系列も
#: 持ち出さない。78週にして日足が約6倍になったので、使わない系列でファイルを膨らませない）
CASE_KEYS = ["t", "tPos", "horizon", "closeAtT", "entryPrice", "target", "thresholdPct", "hitPos",
             "endLineAdopted", "endRatioAdopted", "endLevel", "uptrendEnd", "maxGain", "maxDraw", "endGain",
             "dates", "close"]


def build_cases(sel: pd.DataFrame, parts: pd.DataFrame, panel: pd.DataFrame, names: Dict[str, str],
                before: int = BEFORE, after: int = AFTER) -> Tuple[List[Dict], Dict[str, int]]:
    """選んだ行のチャート。戻り値: (ケース, 確かめの件数)。"""
    cases, checks = [], {"noChart": 0, "entryMismatch": 0, "riseMismatch": 0}
    for idx, row in sel.iterrows():
        code = str(row["Code"])
        c = EL.build_case(panel, code, pd.Timestamp(row["Date"]), names.get(code, ""), int(row["label"]),
                          before=before, after=after)
        if c is None:
            checks["noChart"] += 1
            continue
        v = parts.loc[idx]
        ds_entry = _r(row.get("entry_price"), 4)
        if ds_entry and c.get("entryPrice") and abs(c["entryPrice"] / ds_entry - 1) > 0.005:
            checks["entryMismatch"] += 1
        ds_rise = pd.to_numeric(row.get("future_rise"), errors="coerce")
        if np.isfinite(ds_rise) and c.get("maxGain") is not None and abs(c["maxGain"] - ds_rise * 100) > 0.5:
            checks["riseMismatch"] += 1
        out = {k: c.get(k) for k in CASE_KEYS}
        out.update({
            "code": display_code(code), "jqCode": code, "name": names.get(code, ""), "label": int(row["label"]),
            # 成否の内訳（データセットの材料から。チャートの値は表示用に日足から引き直したもの）
            "verdict": {
                "reached": bool(v["reached"]), "endOk": bool(v["end_ok"]), "uptrend": bool(v["uptrend"]),
                "maxRisePct": _r(ds_rise * 100), "needPct": _r(v["need"] * 100),
                "endLevelPct": _r(pd.to_numeric(row.get("end_level"), errors="coerce") * 100),
                "endNeedPct": _r(v["end_need"] * 100),
            },
        })
        cases.append(out)
    return cases, checks


def build(ds: pd.DataFrame, meta: Dict, prev: Optional[Dict], data_dir: str, n: int = N_EACH,
          before: int = BEFORE, after: int = AFTER, bars: Optional[pd.DataFrame] = None,
          names: Optional[Dict[str, str]] = None) -> Tuple[Dict, Dict]:
    """書き出す中身と、ログに出す件数。"""
    cur_to = meta["trainTo"]
    prev_to = prev.get("trainTo") if prev else None
    rows = added_rows(ds, prev_to, cur_to)
    parts = verdict_parts(rows)
    mismatch = int((parts["ok"].astype(int) != rows["label"].astype(float).astype(int)).sum())
    seed = int(cur_to.replace("-", ""))
    pos, neg = pick(rows, n, seed)
    sel = pd.concat([pos, neg])
    cases: List[Dict] = []
    checks = {"noChart": 0, "entryMismatch": 0, "riseMismatch": 0}
    if len(sel):
        codes = sorted(sel["Code"].astype(str).unique())
        bars = load_bars_for(data_dir, codes) if bars is None else bars[bars["Code"].astype(str).isin(codes)]
        # 学習のラベルと同じ部品で、選んだ銘柄だけ指標を作り直す（どれも銘柄ごとの計算）
        panel = B.price_panel(bars)
        panel = B.add_breakout_context(B.attach_rise_label(B.mark_new_highs(panel)))
        panel["Code"] = panel["Code"].astype(str)
        cases, checks = build_cases(sel, parts, panel, names if names is not None else load_names(data_dir),
                                    before, after)
    payload = {
        "generatedAt": pd.Timestamp.utcnow().isoformat(),
        "model": {"trainedAt": meta.get("trainedAt"), "trainFrom": meta.get("trainFrom"), "trainTo": cur_to,
                  "nTrain": meta.get("nTrain"), "label": meta.get("label"), "riseHorizon": meta.get("riseHorizon")},
        "prev": prev,
        "window": {"after": prev_to, "to": cur_to},
        "counts": {"added": int(len(rows)), "positive": int((rows["label"].astype(float) == 1).sum()),
                   "negative": int((rows["label"].astype(float) == 0).sum())},
        "pick": {"each": n, "seed": seed, "positive": int(len(pos)), "negative": int(len(neg))},
        "cases": cases,
    }
    log = {"mismatch": mismatch, **checks}
    return payload, log


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="週次の再学習で加わったサンプルのチャートを書き出す")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--meta", default=META)
    ap.add_argument("--prev-to", default="", help="前回の学習期間の終わり（省くと meta.json の git の履歴から）")
    ap.add_argument("--n", type=int, default=N_EACH)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    with open(args.meta, encoding="utf-8") as fh:
        meta = json.load(fh)
    prev = ({"trainTo": args.prev_to, "trainedAt": None, "nTrain": None} if args.prev_to
            else prev_window(meta["trainTo"]))
    if prev is None:
        print("[warn] 前回の学習期間の終わりが分からない（meta.json の履歴が無い）。今回の終わりまでの全期間から選ぶ")
    ds = pd.read_parquet(args.dataset, columns=DS_COLS)
    ds["Code"] = ds["Code"].astype(str)
    payload, log = build(ds, meta, prev, args.data_dir, n=args.n)
    c, p = payload["counts"], payload["pick"]
    print(f"[window] 学習期間の終わり {prev['trainTo'] if prev else '—'} → {meta['trainTo']}"
          f"（{(meta.get('trainedAt') or '')[:10]} の再学習）")
    print(f"[added] 加わったサンプル {c['added']}件（正例 {c['positive']}・負例 {c['negative']}）/ "
          f"表示 正例 {p['positive']}・負例 {p['negative']}（種 {p['seed']}）")
    print(f"[check] 内訳から引き直したラベルとデータセットのラベルの食い違い {log['mismatch']}件 / "
          f"チャートが作れない {log['noChart']}件 / 基準の価格の食い違い {log['entryMismatch']}件 / "
          f"最大上昇率の食い違い {log['riseMismatch']}件")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"[done] {os.path.relpath(args.out, ROOT)}（{len(payload['cases'])}件）")
    return 1 if log["mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
