#!/usr/bin/env python3
"""
実験116: 決算の出し直し（同じ期の後からの版）を「その日に分かっていた版」で付け直した学習データ（直す版）と、今の学習データ
（今の版）で、1.2σ の3本（LightGBM・XGBoost・CatBoost）の out-of-fold の PR-AUC を比べる（検証だけ。本番の学習データ・モデル・
規則は変えない）。

経緯: 2026-10-28 の学習と予測の一致チェックで、決算の出し直しがあった期は、元の版の開示日から出し直しの日までの行に後の版が
先回りして効き、予測の時点の値（元の版）と食い違うことが分かった（docs/DATA_TIMING.md）。運用者の選択（2026-10-29）
「まず影響を測る」→ probe（research/exp/probe_fins_versions.py。当たる行 376件・1.71%）→「直す版を作って比べる」（勧めた）。
docs/MODEL_ADOPTION_RULES.md §80。

2つの版（行・ラベルは同じ。決算の列だけが違う）
  今の版  Run Experiment の Build dataset が作った research/_data/dataset.parquet（本番と同じ作り）
  直す版  同じデータで build_dataset.py --pit-fins（PIT_FINS: 出し直しの窓の行だけ、その日までに開示された決算だけで作った
          決算パネルの値に付け直す。進捗の順位は付いた版の開示日の順位。そのあとの計算（1年以上古い決算の欠測・時価総額・
          PER・PBR・全銘柄の中の順位など）は今の作りのまま）。research/_data/pit/dataset.parquet

腕（どれも本番の out-of-fold と同じ当てはめ・同じ種。窓ごとに同じ行を採点する）
  A0  今の版で学習し、今の版で採点（今の out-of-fold。採点する行にも後の版が先回りしている）
  A   今の版で学習し、直す版で採点（今の運用で起きていること: 学習は今の版、予測の時点の値はその日の版）
  B   直す版で学習し、直す版で採点（直したあと）
  B − A が「学習データを直す効き目」、A0 − A が「今の out-of-fold が先回りで良く見えている幅」

窓: 本番と同じ（36か月 / 6か月 / 6か月、エンバーゴ20営業日）を 0 / 2 / 4か月ずらした3通り（e41.folds_for）。木の形は本番の
探索の結果（lgbm_params.json・multi_params.json の 360列の鍵。360列で探索したものでなければ止める）

確かめ（合わなければ止める）
  - 2つの版で行（Code・Date）の並び・ラベル・列が同じ。直す版の学習データの記録（_meta.json）が pitFins = true
  - 値が変わった行と列を数える: 出し直しの窓の行（決算の列）と、窓の外の行（全銘柄の中の順位のように、同じ日のほかの行の
    値で決まる列だけが動くはず。動いた列の名前を出す）

読み方（回す前に決めたもの。判定ではない。直すかどうかは運用者が決める）
  3本 × 3通りのずらしそれぞれで、B − A の窓ごとの PR-AUC の差の平均を見る。平均の大きさが 0.002 以上かつ標準誤差の2倍以上の
  組を「変わる」とし、
  - 「変わる」組が無い →「精度は変わらない（直すのは学習と予測の食い違いをなくすため）」
  - 下がる組があって上がる組が無い →「直すと下がる（今の版の先回りが学習を助けていた）」
  - 上がる組があって下がる組が無い →「直すと上がる」、両方ある →「モデルやずらしで逆向き」
  参考: A0 − A（今の out-of-fold が良く見えている幅）、ずらし0 の全期間の PR-AUC・ROC-AUC、当たる行のスコアの差

公開ログには集計だけ（件数・割合・精度・列の名前）。銘柄ごとの値は出さない。結果は research/_data/oof/e116_*。

    exp=e116_fins_pit_ab.py                      （既定 --shifts 0,2,4 --algos lgbm,xgb,cat）

結果（2026-10-29、run 40851737540、819fb9a、実験の部分 32.4分。docs/MODEL_ADOPTION_RULES.md §80-4）
  - 確かめ: 2つの版で行・ラベル・列が同じ。窓に入る行 376。学習の 360列が変わったのは窓の中の行だけ
  - 窓の外で動いたのは、名前が _r で終わる日付内の順位の列だけ
  - B − A（窓の平均 PR-AUC）: 9組の平均 −0.0012。下がる組は1つ（xgb ずらし4か月 −0.0057 ± 0.0023）、ほか8組は変わらない
    → 読み方「直すと下がる（今の版の先回りが学習を助けていた）」
  - A0 − A: 9組とも大きさ 0.0013 以下。今の out-of-fold が先回りで良く見えている形跡は無い
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
import build_dataset as BD  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import models as M  # noqa: E402
import tuning  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e114b_ltr_rule as RB  # noqa: E402

PREFIX = "e116"
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PIT_DIR = os.path.join(lab.DATA_DIR, "pit")
PIT_PATH = os.path.join(PIT_DIR, "dataset.parquet")
BUILD_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build_dataset.py")
SUMMARY_PATH = os.path.join(OOF_DIR, f"{PREFIX}_summary.json")
TREES = ("lgbm", "xgb", "cat")
ARMS = ("A0", "A", "B")
ARM_JA = {"A0": "今の版で学習・今の版で採点", "A": "今の版で学習・直す版で採点", "B": "直す版で学習・直す版で採点"}
MIN_GAP = 0.002                  # 読み方の「変わる」の大きさ（PR-AUC）
MIN_TEST, MIN_TRAIN = 200, 1000  # 窓を使う最小の件数（本番の out-of-fold と同じ）

log = E.log


# --------------------------------------------------------------------------- #
# 2つの版
# --------------------------------------------------------------------------- #

def build_pit(data_dir: str, out_path: str, script: str = BUILD_SCRIPT) -> Dict:
    """
    同じデータで直す版を作る（build_dataset.py --pit-fins を別のプロセスで。今の版と同じ作りで、学習データを作る間の
    メモリを残さない）。学習データの記録（_meta.json）を返す。
    """
    import subprocess

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = subprocess.run([sys.executable, script, "--data-dir", data_dir, "--out", out_path, "--pit-fins"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    keep = [x for x in r.stdout.splitlines() if x.startswith(("[merge]", "[mode]", "[done]", "[save]"))]
    for x in keep[-12:]:
        print(f"  （直す版）{x}")
    if r.returncode != 0:
        print("\n".join(r.stdout.splitlines()[-30:]))
        raise SystemExit(f"直す版を作れなかった（終了コード {r.returncode}）")
    with open(os.path.splitext(out_path)[0] + "_meta.json", encoding="utf-8") as fh:
        return json.load(fh)


def _read(path: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["Date"] = pd.to_datetime(d["Date"])
    d["Code"] = d["Code"].astype(str)
    return d.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)


def load_pair(cur_path: str, pit_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """2つの版を (Date, Code) の順に読む。行の並び・ラベル・列が同じでなければ止める。"""
    cur, pit = _read(cur_path), _read(pit_path)
    if len(cur) != len(pit) or not np.array_equal(cur[["Code", "Date"]].astype(str).to_numpy(),
                                                   pit[["Code", "Date"]].astype(str).to_numpy()):
        raise SystemExit(f"2つの版で行が違う（今の版 {len(cur):,}行・直す版 {len(pit):,}行）")
    if set(cur.columns) != set(pit.columns):
        raise SystemExit(f"2つの版で列が違う: {sorted(set(cur.columns) ^ set(pit.columns))[:8]}")
    a, b = cur["label"].to_numpy(dtype=float), pit["label"].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]):
        raise SystemExit("2つの版でラベルが違う")
    return cur, pit[list(cur.columns)]


def changed_cells(cur: pd.DataFrame, pit: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """列ごとに値が変わった行の印（欠測どうしは同じ。数値は相対 1e-9 まで同じ）。行 × 列の真偽の表。"""
    out = {}
    for c in cols:
        a, b = cur[c], pit[c]
        an, bn = a.isna().to_numpy(), b.isna().to_numpy()
        if pd.api.types.is_datetime64_any_dtype(a) or pd.api.types.is_datetime64_any_dtype(b):
            eq = pd.to_datetime(a).to_numpy() == pd.to_datetime(b).to_numpy()
            out[c] = ~((an & bn) | (~an & ~bn & eq))
            continue
        x, y = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        if ((x.notna() | an).all()) and ((y.notna() | bn).all()):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                eq = np.isclose(xv, yv, rtol=1e-9, atol=0.0)
        else:
            eq = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        out[c] = ~((an & bn) | (~an & ~bn & eq))
    return pd.DataFrame(out)


def change_report(cur: pd.DataFrame, pit: pd.DataFrame, win_mask: np.ndarray, model_cols: Sequence[str]) -> Dict:
    """値が変わった行と列（出し直しの窓の行 / 窓の外の行）。model_cols は学習に使う列（360列）。"""
    cols = [c for c in cur.columns if c not in ("Code", "Date")]
    ch = changed_cells(cur, pit, cols)
    row_any = ch.any(axis=1).to_numpy()
    in_w, out_w = row_any & win_mask, row_any & ~win_mask
    by_col = ch.sum(axis=0)
    outside = ch[~win_mask].sum(axis=0)
    mc = [c for c in model_cols if c in ch.columns]
    return {"rows": int(len(cur)), "window_rows": int(win_mask.sum()), "changed_rows": int(row_any.sum()),
            "changed_in_windows": int(in_w.sum()), "changed_outside": int(out_w.sum()),
            "model_changed_rows": int(ch[mc].any(axis=1).sum()) if mc else 0,
            "cols_changed": {c: int(n) for c, n in by_col[by_col > 0].sort_values(ascending=False).items()},
            "cols_outside": {c: int(n) for c, n in outside[outside > 0].sort_values(ascending=False).items()},
            "labelled_window_rows": int((win_mask & cur["label"].notna().to_numpy()).sum())}


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #

def fit_predict(algo: str, params: Dict, tr: pd.DataFrame, tes: Sequence[pd.DataFrame],
                cols: Sequence[str]) -> List[np.ndarray]:
    """本番の out-of-fold と同じ当てはめで tr を学習し、tes をそれぞれ採点（lgbm は train_production、ほかは train_multi と同じ）。"""
    cols = list(cols)
    X = tr[cols].to_numpy(dtype=float)
    y = tr["label"].to_numpy(dtype=int)
    if algo == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**params, scale_pos_weight=tuning.scale_pos_weight(y))
        m.fit(X, y)
        return [m.predict_proba(t[cols].to_numpy(dtype=float))[:, 1] for t in tes]
    m = M.fit(algo, X, y, cols, params=params)
    return [M.predict(m, t[cols].to_numpy(dtype=float)) for t in tes]


def run_tag(algo: str, params: Dict, cols: Sequence[str], cur: pd.DataFrame, pit: pd.DataFrame,
            ch_rows: int) -> str:
    key = {"algo": algo, "params": {k: params[k] for k in sorted(params)}, "cols": F.signature(list(cols)),
           "rows": int(len(cur)), "last": str(cur["Date"].max().date()), "labels": int(cur["label"].notna().sum()),
           "changed": int(ch_rows), "pit_sum": float(np.nansum(pit[list(cols)].to_numpy(dtype=float))), "v": 1}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


def oof_pair(cur: pd.DataFrame, pit: pd.DataFrame, cols: Sequence[str], algo: str, params: Dict, shift: int,
             tag: str, out_dir: str = OOF_DIR) -> pd.DataFrame:
    """
    1モデル・1つのずらし方の out-of-fold: 窓ごとに、今の版で学習したモデルで今の版（A0）と直す版（A）を採点し、直す版で
    学習したモデルで直す版（B）を採点する。学習はラベルのある行（本番と同じ）、採点は窓の中のラベルのある行。保存済みなら読む。
    """
    path = os.path.join(out_dir, f"{PREFIX}_{algo}_{tag}_sh{shift}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    t0 = time.time()
    lab_ok = cur["label"].notna()
    folds = E41.folds_for(cur.loc[lab_ok, "Date"], shift)
    d = cur["Date"]
    rows = []
    for f in folds:
        trm = (d <= pd.Timestamp(f.train_end)) & lab_ok
        tem = (d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end)) & lab_ok
        if int(tem.sum()) < MIN_TEST or int(trm.sum()) < MIN_TRAIN:
            continue
        a0, a = fit_predict(algo, params, cur[trm], [cur[tem], pit[tem]], cols)
        (b,) = fit_predict(algo, params, pit[trm], [pit[tem]], cols)
        part = cur.loc[tem, ["Code", "Date", "label"]].copy()
        part["score_A0"], part["score_A"], part["score_B"] = a0, a, b
        part["fold"] = int(f.index)
        rows.append(part)
    if not rows:
        raise SystemExit(f"{algo} ずらし{shift}か月: 窓を作れない")
    o = pd.concat(rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    o.to_parquet(path, index=False)
    log(f"  {algo} ずらし{shift}か月: {o['fold'].nunique()}窓・{len(o):,}件 {time.time() - t0:.0f}秒")
    return o


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #

def window_scores(o: pd.DataFrame) -> pd.DataFrame:
    """窓ごとの PR-AUC（腕ごと）。正例と負例の両方がある窓だけ。"""
    from sklearn.metrics import average_precision_score

    rows = []
    for f, g in o.groupby("fold", sort=True):
        y = g["label"].to_numpy(dtype=int)
        if y.min() == y.max():
            continue
        rows.append({"fold": int(f), **{a: float(average_precision_score(y, g[f"score_{a}"].to_numpy(dtype=float)))
                                        for a in ARMS}})
    return pd.DataFrame(rows, columns=["fold"] + list(ARMS))


def paired(ws: pd.DataFrame, a: str, b: str) -> Dict:
    """窓ごとの差（a − b）の平均・標準誤差・上回った窓の数。"""
    dv = (ws[a] - ws[b]).to_numpy(dtype=float)
    if not len(dv):
        return {"mean": float("nan"), "se": float("nan"), "won": 0, "n": 0}
    se = float(dv.std(ddof=1) / np.sqrt(len(dv))) if len(dv) > 1 else float("nan")
    return {"mean": float(dv.mean()), "se": se, "won": int((dv > 0).sum()), "n": int(len(dv))}


def pooled(o: pd.DataFrame) -> Dict[str, Dict]:
    """全期間（窓をまとめた）PR-AUC・ROC-AUC（腕ごと）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = o["label"].to_numpy(dtype=int)
    return {a: {"pr": float(average_precision_score(y, o[f"score_{a}"])),
                "roc": float(roc_auc_score(y, o[f"score_{a}"]))} for a in ARMS}


def verdict_of(d: Dict, min_gap: float = MIN_GAP) -> str:
    """1つの組（モデル × ずらし）の B − A: 'up' / 'down' / 'same'（大きさ min_gap 以上かつ標準誤差の2倍以上で変わる）。"""
    m, se = d["mean"], d["se"]
    if not (np.isfinite(m) and np.isfinite(se)) or abs(m) < min_gap or abs(m) < 2 * se:
        return "same"
    return "up" if m > 0 else "down"


def reading(kinds: Sequence[str]) -> str:
    """読み方（回す前に決めたもの）: 組ごとの判定の並びから全体の読み。"""
    up, down = "up" in kinds, "down" in kinds
    if up and down:
        return "mixed"
    if down:
        return "down"
    if up:
        return "up"
    return "same"


READING_JA = {"same": "精度は変わらない（直すのは学習と予測の食い違いをなくすため）",
              "down": "直すと下がる（今の版の先回りが学習を助けていた）", "up": "直すと上がる",
              "mixed": "モデルやずらしで逆向き"}


def _f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else "—"


def _d(d: Dict) -> str:
    if not d.get("n"):
        return "—"
    return f"{d['mean']:+.4f} ± {d['se']:.4f}（上の窓 {d['won']}/{d['n']}）"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験116: 決算の出し直しをその日の版で付け直した学習データと比べる（検証だけ）")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--algos", default=",".join(TREES))
    ap.add_argument("--skip-build", action="store_true", help="直す版を作り直さず、置き場のものを読む（同じデータのとき）")
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    algos = [a for a in args.algos.split(",") if a.strip()]
    if any(a not in TREES for a in algos):
        ap.error(f"知らないモデル: {algos}")
    t_all = time.time()
    cols = list(F.columns(F.DEFAULT_PRESET))
    params = RB.prod_params(cols)                     # 360列で探索した木の形でなければ止まる
    os.makedirs(OOF_DIR, exist_ok=True)

    if not (args.skip_build and os.path.exists(PIT_PATH)):
        log("[直す版] build_dataset.py --pit-fins で作る（今の版と同じデータ）")
        meta = build_pit(lab.DATA_DIR, PIT_PATH)
    else:
        with open(os.path.splitext(PIT_PATH)[0] + "_meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
    if not meta.get("pitFins"):
        raise SystemExit("直す版の学習データの記録が pitFins = true でない")
    cur, pit = load_pair(lab.DATASET, PIT_PATH)
    with open(os.path.splitext(lab.DATASET)[0] + "_meta.json", encoding="utf-8") as fh:
        if json.load(fh).get("pitFins"):
            raise SystemExit("今の版の学習データが PIT_FINS で作られている")
    miss = [c for c in cols if c not in cur.columns]
    if miss:
        raise SystemExit(f"学習データに無い列: {miss[:6]}")
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        fins = BD.load_parts("fins", lab.DATA_DIR)
    win = BD.reissue_windows(BD.fin_versions(fins))
    wmask = BD.in_reissue_windows(cur, win)
    rep = change_report(cur, pit, wmask, cols)

    print("=" * 100)
    print(f"実験116 決算の出し直しをその日の版で付け直した学習データ（直す版）と今の版で、1.2σ の3本の PR-AUC を比べる（検証だけ）: "
          f"{rep['rows']:,}行・{cur['Date'].min().date()} 〜 {cur['Date'].max().date()} / {len(cols)}列 / ずらし {shifts}か月 / "
          f"{' '.join(algos)}")
    print("=" * 100)
    log(f"[確かめ] 2つの版で行・ラベル・列が同じ。出し直しの窓 {len(win):,}・窓に入る行 {rep['window_rows']:,}"
        f"（ラベルあり {rep['labelled_window_rows']:,}）/ 値が変わった行 {rep['changed_rows']:,}（窓の中 "
        f"{rep['changed_in_windows']:,}・窓の外 {rep['changed_outside']:,}）/ 学習の {len(cols)}列のどれかが変わった行 "
        f"{rep['model_changed_rows']:,}")
    top = list(rep["cols_changed"].items())[:12]
    log("[確かめ] 変わった列（行数の多い順）: " + " / ".join(f"{c} {n}" for c, n in top)
        + (f" ほか {len(rep['cols_changed']) - len(top)}列" if len(rep["cols_changed"]) > len(top) else ""))
    if rep["cols_outside"]:
        log("[確かめ] 窓の外で変わった列（同じ日のほかの行の値で決まる列のはず）: "
            + " / ".join(f"{c} {n}" for c, n in rep["cols_outside"].items()))
    if rep["changed_rows"] == 0:
        raise SystemExit("直す版と今の版で値が変わった行が無い（PIT_FINS が効いていない）")

    summary: Dict = {"report": rep, "windows": int(len(win)), "shifts": {}, "params": params}
    kinds = []
    for algo in algos:
        tag = run_tag(algo, params[algo], cols, cur, pit, rep["changed_rows"])
        summary["shifts"][algo] = {}
        for sh in shifts:
            o = oof_pair(cur, pit, cols, algo, params[algo], sh, tag)
            ws = window_scores(o)
            res = {"windows": int(len(ws)), "mean": {a: float(ws[a].mean()) for a in ARMS},
                   "B_A": paired(ws, "B", "A"), "A0_A": paired(ws, "A0", "A"), "B_A0": paired(ws, "B", "A0"),
                   "pooled": pooled(o), "folds": json.loads(ws.to_json(orient="records"))}
            # 当たる行（窓に入る・ラベルあり）のスコアの差（参考）
            hit = BD.in_reissue_windows(o, win)
            res["hit_rows"] = int(hit.sum())
            res["hit_absdiff"] = {"A_A0": float(np.mean(np.abs(o.loc[hit, "score_A"] - o.loc[hit, "score_A0"])))
                                  if hit.any() else float("nan"),
                                  "B_A": float(np.mean(np.abs(o.loc[hit, "score_B"] - o.loc[hit, "score_A"])))
                                  if hit.any() else float("nan")}
            res["kind"] = verdict_of(res["B_A"])
            kinds.append(res["kind"])
            summary["shifts"][algo][str(sh)] = res
            print(f"■ {algo} ずらし{sh}か月（{res['windows']}窓）: 窓の平均 PR-AUC " + " / ".join(
                f"{a} {_f(res['mean'][a])}" for a in ARMS)
                + f" ｜ B − A {_d(res['B_A'])} → {res['kind']} ｜ A0 − A {_d(res['A0_A'])}"
                + f" ｜ 全期間 PR-AUC A0 {_f(res['pooled']['A0']['pr'])} A {_f(res['pooled']['A']['pr'])} "
                  f"B {_f(res['pooled']['B']['pr'])}・ROC-AUC B − A {res['pooled']['B']['roc'] - res['pooled']['A']['roc']:+.4f}"
                + f" ｜ 当たる行 {res['hit_rows']}件のスコアの差の平均 |A − A0| {_f(res['hit_absdiff']['A_A0'], 5)}・"
                  f"|B − A| {_f(res['hit_absdiff']['B_A'], 5)}")

    overall = reading(kinds)
    summary["reading"] = {"kinds": kinds, "overall": overall}
    print(f"\n■ 読み方（回す前に決めたもの。判定ではない）: B − A の窓ごとの差が、大きさ {MIN_GAP} 以上かつ標準誤差の2倍以上の組を"
          f"「変わる」とする → 上がる {kinds.count('up')}組・下がる {kinds.count('down')}組・変わらない {kinds.count('same')}組"
          f" → {READING_JA[overall]}")
    print("  腕: " + " / ".join(f"{a} {ARM_JA[a]}" for a in ARMS))
    summary["minutes"] = round((time.time() - t_all) / 60, 1)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"記録: {OOF_DIR}/{PREFIX}_* （{summary['minutes']}分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
