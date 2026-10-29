#!/usr/bin/env python3
"""
実験114c: ランキング学習を毎週探索し直してよいかを確かめる（検証だけ。本番の設定は変えない）。

運用者の質問（2026-10-24）「ランキング学習において５cvのチューニングは実施していないのですか？」→ 実験114 で 50試行 ×
5分割の探索を1回だけ行い、週次では探索し直していなかった（設定を固定）。AskUserQuestion「ランキング学習の設定
（ハイパーパラメータ）の探索を、これからどうしますか？」→「毎週探索し直す」（基準モデルと同じく日曜の再学習で毎回
50試行 × 5分割。いきなり切り替えず、探索し直した版を並べて確かめてから入れ替える）。docs/MODEL_ADOPTION_RULES.md §78-10。

データ: 実験114・114b と同じ最終日（ラベルがある最後の日 2026-09-24。--data-end）で切る。10/24 のあとに増えた日は使わない
（増えた日を入れると、探索の打ち切りも比べの行も実験114・114b と変わり、「同じものが出るか」を確かめられない）。

確かめること
  1. 再現: 本番の探索（research/tune_ltr.py と同じ: ltr.load_frame → ltr.search_cutoff → ltr.search_rows → ltr.tune）が、
     実験114 の L と同じ木の形を出すか（12個の鍵・最良の試行・分割ごとの NDCG@5・置き場にある実験114 の試行ごとの値
     50試行すべて）。違ったときは、同じデータで実験114 の探索そのもの（実験114 の読み方 E.load_frame・打ち切り・分割で
     E.tune_l）も回し、本番の探索と全部同じなら「作り（コード）は同じで、10/24 のあとにデータの値が変わった」として続ける
  2. 探索し直しの揺れ: 週次の再学習が k = 1〜4週前に探索していたら選んでいた木の形 P1〜P4（データの最終日から 7k 日
     さかのぼった日までの日付で、run_tuning.py と同じ式で打ち切りを決める）を作り、実験114b と同じ比べ方（今の規則の形・
     0 / 2 / 4か月のずらし・ランキング学習は種42）で、両方使う形（R0D）と今の規則（R0）を並べる。今の木の形（実験114 の
     L = P0）も同じ表に並べ、実験114b の数（§78-8）と同じになることも確かめる（参考。データの値が変わっていれば違ってよい）

判定（回す前に決めたもの）
  - 1 が一致し（置き場の L と同じ、または同じデータの実験114 の探索と同じ）、かつ 2 の P1〜P4 の4通りすべてで実験114b と
    同じ判定（R0D が3通りのずらしすべてで、R0 より平均収益（窓の平均）が高く、−10% 未満（窓の平均）が低い）なら
    「探索し直しても、両方使う形の結論は変わらない」→ 週次の再学習に探索を入れる
  - 1 が一致しなければ止める（本番の探索が実験114 と違う。入れない）
  - 2 で1通りでも判定が変われば入れずに、数字を運用者に見せて決めてもらう
  参考に並べるもの（判定には使わない）: 木の形・探索の NDCG@5・本番と同じ窓（ずらし0）の out-of-fold の NDCG@5・R0D に
  選ばれた行のうち P0 と同じ行の割合

公開ログには集計だけ（件数・平均・割合・木の形）。銘柄ごとの値は出さない。結果は research/_data/oof/e114c_*。
探索は1回約27分（実験114 の実測）× 5回（1 が置き場と違えばもう1回）。時間の上限（--budget-min）を過ぎたら新しい探索も
比べも始めずに終わる（終わった探索は置き場に残るので、同じ引数でもう1回回すと続きから）。

    exp=e114_ltr_retest.py  args="--retune"
    （実験114 の探索の結果と、L・3本の out-of-fold・実験114b の集計を Actions のキャッシュから読むため、実験114 の名前で回す。
     手元なら python3 research/exp/e114c_ltr_retune.py）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import ltr  # noqa: E402
import tuning  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e114b_ltr_rule as B  # noqa: E402

PREFIX = "e114c"
OOF_DIR = E.OOF_DIR
SUMMARY_PATH = os.path.join(OOF_DIR, f"{PREFIX}_summary.json")
WEEKS = (1, 2, 3, 4)
SHIFTS = (0, 2, 4)
BUDGET_MIN = 280.0
#: 実験114・114b のデータの最終日（ラベルがある最後の日。§78-3「2018-04-03 〜 2026-09-24」・§78-8 と同じ 22,012件）
DATA_END = "2026-09-24"
#: 実験114b の数と突き合わせる鍵（P0 は実験114b のランキング学習と同じものなので、同じ数になるはず）
CHECK_KEYS = ("R0", "R0D", "LN")
TREE_KEYS = E.TREE_KEYS

log = E.log


def until(frame: pd.DataFrame, data_end: Optional[pd.Timestamp]) -> pd.DataFrame:
    """data_end までの行（None なら全部）。行の順は変えない。"""
    if data_end is None:
        return frame
    return frame[pd.to_datetime(frame["Date"]) <= pd.Timestamp(data_end)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 週ごとの打ち切り
# --------------------------------------------------------------------------- #

def week_ends(dates, weeks: Sequence[int] = WEEKS) -> Dict[int, pd.Timestamp]:
    """
    k 週前の週次の再学習が見ていたデータの最終日（今の最終日から 7k 日さかのぼった日までの、データにある最後の日）。
    k = 0 は今の最終日。
    """
    d = pd.Series(pd.to_datetime(pd.Series(dates)).unique()).sort_values().reset_index(drop=True)
    dmax = pd.Timestamp(d.iloc[-1])
    out = {0: dmax}
    for k in weeks:
        prev = d[d <= dmax - pd.Timedelta(days=7 * int(k))]
        if not len(prev):
            raise SystemExit(f"{k}週前のデータが無い")
        out[int(k)] = pd.Timestamp(prev.iloc[-1])
    return out


def week_cutoffs(dates, weeks: Sequence[int] = WEEKS) -> Dict[int, pd.Timestamp]:
    """k 週前の探索の打ち切り（その週のデータの最終日から、run_tuning.py と同じ式。ltr.search_cutoff）。"""
    d = pd.to_datetime(pd.Series(dates))
    return {k: ltr.search_cutoff(d[d <= end]) for k, end in week_ends(d, weeks).items()}


# --------------------------------------------------------------------------- #
# 探索（本番と同じ ltr.tune）
# --------------------------------------------------------------------------- #

def tune_path(name: str, cols: Sequence[str], sub: pd.DataFrame, n_trials: int, out_dir: str = OOF_DIR) -> str:
    """探索の結果の置き場（name は w0〜w4 か e114。列・行・期間・試行数が変われば別の名前）。"""
    d = pd.to_datetime(sub["Date"])
    key = {"name": name, "cols": F.signature(list(cols)), "rows": int(len(sub)), "from": str(d.min().date()),
           "to": str(d.max().date()), "r20": round(float(sub[ltr.R20].sum()), 6), "n_trials": int(n_trials), "v": 2}
    tag = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(out_dir, f"{PREFIX}_tune_{name}_{tag}.json")


def _cached(path: str, make) -> Dict:
    """置き場にあれば読み、無ければ make() で作って置く（_cached に読んだかどうか）。"""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        rec["_cached"] = True
        return rec
    rec = make()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    rec["_cached"] = False
    return rec


def tune_week(ds: pd.DataFrame, cols: Sequence[str], k: int, cutoff: pd.Timestamp, n_trials: int,
              out_dir: str = OOF_DIR, tune_fn=None) -> Dict:
    """k 週前の打ち切りで探索する（置き場にあれば読む）。tune_fn はテスト用（既定は ltr.tune）。"""
    sub = ltr.search_rows(ds, cutoff)

    def make():
        rec = (tune_fn or ltr.tune)(sub, cols, n_trials=n_trials)
        rec["_week"], rec["_cutoff"] = int(k), str(pd.Timestamp(cutoff).date())
        return rec

    return _cached(tune_path(f"w{int(k)}", cols, sub, n_trials, out_dir), make)


def e114_search(full: pd.DataFrame, cols: Sequence[str], n_trials: int, out_dir: str = OOF_DIR,
                tune_fn=None) -> Dict:
    """
    実験114 の探索そのもの（実験114 の main と同じ: E.load_frame の行・全日付からの打ち切り・ラベルの無い行が無いことの確かめ・
    今の売り方の収益と20営業日目の終値がある行・年×時価総額帯で層別した日付単位の分割・E.tune_l）を、full のデータで回す
    （置き場にあれば読む）。本番の探索の P0 が置き場の L と違ったときに、違いが作りかデータの変化かを分けるため。
    返す形は ltr.tune にそろえる（params・_cv・_trials）。tune_fn はテスト用（既定は E.tune_l）。
    """
    cutoff = pd.Timestamp(E.holdout_bounds(full["Date"], E.HOLDOUT_MONTHS, E.EMBARGO_DAYS)[0])
    sub = full[pd.to_datetime(full["Date"]) <= cutoff]
    if sub["label"].isna().any():
        raise SystemExit(f"探索の期間にラベルの無い行がある（{int(sub['label'].isna().sum())}行）")
    sub_l = sub[sub[E.RET].notna() & sub[E.R20].notna()].reset_index(drop=True)

    def make():
        folds = tuning.cv_folds(sub_l, tuning.PRODUCTION_CV, n_splits=E.N_SPLITS, seed=E.SEED,
                                embargo_days=E.EMBARGO_DAYS)
        if not folds:
            raise SystemExit("探索の分割を作れない")
        rec = (tune_fn or E.tune_l)(sub_l, cols, n_trials, folds)
        cv = {k: v for k, v in rec["_cv"].items() if k != "trials"}
        return {"params": rec["params"], "_cv": cv, "_trials": rec["_cv"].get("trials"),
                "_cutoff": str(cutoff.date()), "_rows": int(len(sub_l))}

    return _cached(tune_path("e114", cols, sub_l, n_trials, out_dir), make)


def same_as_114(rec: Dict, ref: Dict, ref_trials: Optional[list] = None) -> Dict:
    """
    再現の確かめ: 本番の探索の結果 rec が、実験114 の L（ref: ltr_params.json と同じ形。params と _cv）と同じか。
    ref_trials（実験114 の試行ごとの値）があれば、それも全試行で比べる。
    """
    p, q = rec["params"], ref.get("params") or {}
    diff = [k for k in sorted(set(p) | set(q)) if p.get(k) != q.get(k)]
    cv, rc = rec["_cv"], ref.get("_cv") or {}
    same_cv = all(cv.get(k) == rc.get(k) for k in ("best_trial", "ndcg", "ndcg_folds"))
    trials = None
    if ref_trials is not None:
        trials = rec.get("_trials") == ref_trials
    ok = not diff and same_cv and trials is not False
    return {"diff_keys": diff, "same_params": not diff, "same_cv": same_cv, "same_trials": trials, "ok": ok}


# --------------------------------------------------------------------------- #
# 今の規則の形での比べ（実験114b と同じ）
# --------------------------------------------------------------------------- #

def rule_eval(df: pd.DataFrame, cols: Sequence[str], trees: Dict[str, pd.DataFrame], params_l: Dict, shift: int,
              seed: int = B.L_SEED, oof_fn=None) -> Dict:
    """
    1つの木の形・1つのずらし方で、実験114b と同じ表を作る（R0 / R0D / LN の集計・R0D − R0 の窓ごとの差・R0D の行・
    ずらし0なら out-of-fold の NDCG@5）。oof_fn はテスト用（既定は E.oof_arm）。
    """
    lo = (oof_fn or E.oof_arm)(df, cols, "L", params_l, shift, (seed,))
    ft = B.forward_frame(B.wide({**trees, B.LTR: lo}, df), E41.folds_for(df["Date"], shift))
    masks = B.rule_masks(ft)
    st = {key: B.rule_stats(ft, masks[key]) for key in CHECK_KEYS}
    st["R0D"]["diff"] = B.window_diff(ft, masks["R0"], masks["R0D"])
    st["R0D"]["diff_bad"] = B.window_diff(ft, masks["R0"], masks["R0D"], "bad")
    picks = set(map(tuple, ft.loc[masks["R0D"], ["Code", "Date"]].astype(str).to_numpy()))
    nd = float("nan")
    if E.R20 in lo.columns and bool(lo[E.R20].notna().all()):
        nd = ltr.ndcg_at_k(lo["Date"], lo["score"], ltr.grades(lo["Date"], lo[E.R20].to_numpy(dtype=float)))
    return {"stats": st, "picks": picks, "ndcg_oof": nd, "windows": int(ft["window"].nunique()), "rows": int(len(ft))}


def verdict_row(stats_by_shift: Dict[int, Dict]) -> Dict:
    """実験114b と同じ判定（R0D がすべてのずらし方で、R0 より平均収益（窓の平均）が高く −10% 未満（窓の平均）が低い）。"""
    rows = [{"shift": int(sh), **{k: st[k] for k in ("R0", "R0D", "LN")}} for sh, st in sorted(stats_by_shift.items())]
    return B.verdict(rows, B.PRIMARY)


def decide(repro_ok: bool, verdicts: Dict[int, Dict]) -> str:
    """判定（回す前に決めたもの）: stop（再現しない）/ adopt（4通りすべて同じ判定）/ ask（判定が変わる木の形がある）。"""
    if not repro_ok:
        return "stop"
    retuned = [v for k, v in verdicts.items() if int(k) != 0]
    if retuned and all(v["pass"] for v in retuned):
        return "adopt"
    return "ask"


DECISION_JA = {
    "stop": "本番の探索が実験114 と同じ木の形を出さない → 週次の再学習に入れない（作りを調べる）",
    "adopt": "探索し直しても、両方使う形の判定は変わらない → 週次の再学習に探索を入れる",
    "ask": "探索し直すと判定が変わる木の形がある → 入れずに、数字を運用者に見せて決めてもらう",
}


def check_114b(st: Dict, prev: Optional[Dict], shift: int) -> Optional[bool]:
    """P0 の R0 / R0D / LN の件数・平均収益・−10% 未満が実験114b の集計（e114b_summary.json）と同じか（無ければ None）。"""
    rec = (((prev or {}).get("shifts") or {}).get(str(shift)) or {}).get("stats")
    if not rec:
        return None
    for key in CHECK_KEYS:
        a, b = st[key], rec.get(key) or {}
        if int(a["n"]) != int(b.get("n", -1)):
            return False
        for f in ("ret_w", "bad_w"):
            if not np.isclose(float(a[f]), float(b.get(f, np.nan)), rtol=0, atol=1e-12):
                return False
    return True


def _p(x: float, nd: int = 2) -> str:
    return f"{x * 100:+.{nd}f}%" if np.isfinite(x) else "—"


def _s(x: float) -> str:
    return f"{x * 100:.1f}%" if np.isfinite(x) else "—"


def _tree(params: Dict) -> str:
    return " / ".join(f"{k} {params[k]:.4g}" if isinstance(params[k], float) else f"{k} {params[k]}"
                      for k in TREE_KEYS if k in params)


def _same_ja(same: Dict, n_trials: int) -> str:
    trials_ja = {True: f"{n_trials}試行すべて同じ", False: "違う",
                 None: "比べない（試行ごとの値が置き場に無いか、試行数が違う）"}[same["same_trials"]]
    return (f"木の形 {'同じ' if same['same_params'] else '違う'}（違う鍵 {same['diff_keys'] or 'なし'}）・最良の試行と分割ごとの "
            f"NDCG@5 {'同じ' if same['same_cv'] else '違う'}・試行ごとの値 {trials_ja}")


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験114c: ランキング学習を毎週探索し直してよいかを確かめる（検証だけ）")
    ap.add_argument("--trials", type=int, default=ltr.TUNE_TRIALS)
    ap.add_argument("--weeks", default=",".join(str(k) for k in WEEKS))
    ap.add_argument("--budget-min", type=float, default=BUDGET_MIN)
    ap.add_argument("--data-end", default=DATA_END,
                    help=f"この日までのデータで回す（既定 {DATA_END} は実験114・114b と同じ最終日。all で全部）")
    args = ap.parse_args(argv)
    weeks = tuple(int(x) for x in args.weeks.split(",") if x.strip())
    data_end = None if args.data_end == "all" else pd.Timestamp(args.data_end)
    import warnings
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)

    t_all = time.time()
    cols = F.columns(F.DEFAULT_PRESET)
    ref = ltr.load_params()
    why = ltr.params_mismatch(ref, cols)
    if why:
        raise SystemExit(f"research/ltr_params.json が今の列で探索したものではない（{why}）")
    store = None
    if os.path.exists(E.PARAMS_PATH):
        with open(E.PARAMS_PATH, encoding="utf-8") as fh:
            store = json.load(fh)
    ref_trials = ((store or {}).get("L") or {}).get("_cv", {}).get("trials")
    if store is not None and store["L"]["params"] != ref["params"]:
        raise SystemExit("置き場の実験114 の L と research/ltr_params.json の木の形が違う")

    # 本番と同じ読み方（tune_ltr.py と同じ）で、実験114 と同じ最終日まで
    ds_all = ltr.load_frame(lab.DATASET, lab.DATA_DIR, cols)
    ds = until(ds_all, data_end)
    n_all, last_all = len(ds_all), ds_all["Date"].max().date()
    del ds_all
    cutoffs = week_cutoffs(ds["Date"], weeks)
    ends = week_ends(ds["Date"], weeks)
    print("=" * 96)
    print(f"実験114c ランキング学習を毎週探索し直してよいかを確かめる（検証だけ）: {len(ds):,}件（ラベルあり）・"
          f"{ds['Date'].nunique():,}日・{ds['Date'].min().date()} 〜 {ds['Date'].max().date()}"
          f"（データの最終日を {'切らない' if data_end is None else str(data_end.date()) + ' で切った。今のデータは'} "
          f"{n_all:,}件・〜{last_all}）/ {len(cols)}列（指紋 {F.signature(cols)}）/ 探索 {args.trials}試行 × "
          f"{ltr.TUNE_SPLITS}分割・種 {ltr.TUNE_SEED}")
    print(f"  今の木の形（P0 で再現を確かめる）: research/ltr_params.json = 実験114 の L（{ref.get('_cutoff')} まで）: "
          f"{_tree(ref['params'])}")
    print("  探索の打ち切り（k 週前の週次の再学習のデータの最終日 → 打ち切り）: " + " / ".join(
        f"P{k} {ends[k].date()} → {cutoffs[k].date()}" for k in sorted(cutoffs)))
    print("=" * 96)

    full_e: Optional[pd.DataFrame] = None

    def e_frame() -> pd.DataFrame:
        """実験114・114b の読み方（E.load_frame）で、同じ最終日まで。"""
        full, _, _ = E.load_frame(cols)
        return until(full, data_end)

    recs: Dict[int, Dict] = {}
    summary: Dict = {"data_end": None if data_end is None else str(data_end.date()), "rows": int(len(ds)),
                     "cutoffs": {str(k): str(v.date()) for k, v in cutoffs.items()},
                     "ends": {str(k): str(v.date()) for k, v in ends.items()}, "trials": args.trials}
    for k in sorted(cutoffs):
        elapsed = (time.time() - t_all) / 60
        if elapsed > args.budget_min:
            log(f"[時間] {elapsed:.0f}分で上限 {args.budget_min:.0f}分を過ぎた。P{k} から先は次の回に（同じ引数でもう1回）")
            summary["unfinished"] = True
            break
        sub_n = len(ltr.search_rows(ds, cutoffs[k]))
        log(f"[探索] P{k}（〜{cutoffs[k].date()}・{sub_n:,}件）")
        rec = tune_week(ds, cols, k, cutoffs[k], args.trials)
        cv = rec["_cv"]
        took = "置き場から読んだ" if rec.get("_cached") else f"{cv['minutes']}分"
        log(f"[探索] P{k} {took} / 最良は試行{cv['best_trial']}・分割平均 NDCG@{ltr.NDCG_K} {cv['ndcg']:.4f} / "
            f"{_tree(rec['params'])}")
        recs[k] = rec
        if k == 0:
            same = same_as_114(rec, ref, ref_trials if args.trials == ltr.TUNE_TRIALS else None)
            summary["repro"] = same
            log(f"[確かめ] 再現: 本番の探索の P0 と置き場の実験114 の L: {_same_ja(same, args.trials)}")
            ok = same["ok"]
            if not ok:
                # 作り（コード）の違いか、10/24 のあとのデータの値の変化かを分ける: 同じデータで実験114 の探索そのものを回す
                log("[確かめ] 置き場の L と違うので、同じデータで実験114 の探索（実験114 の読み方・打ち切り・分割で tune_l）を回す")
                full_e = e_frame()
                er = e114_search(full_e, cols, args.trials)
                same_e = same_as_114(rec, er, er.get("_trials"))
                summary["repro_same_data"] = same_e
                took = "置き場から読んだ" if er.get("_cached") else f"{er['_cv'].get('minutes', '?')}分"
                log(f"[確かめ] 同じデータの実験114 の探索（{took}・〜{er['_cutoff']}・{er['_rows']:,}件）と本番の探索の P0: "
                    f"{_same_ja(same_e, args.trials)}")
                ok = same_e["ok"]
                if ok:
                    log("[確かめ] 作りは同じ。置き場の L との違いは 10/24 のあとのデータの値の変化による")
            summary["repro_ok"] = bool(ok)
            if not ok:
                summary["decision"] = "stop"
                print(f"\n■ 判定: {DECISION_JA['stop']}")
                _save(summary, t_all)
                return 1
    summary["tunes"] = {str(k): {"params": r["params"], "_cv": r["_cv"], "_cutoff": r.get("_cutoff")}
                        for k, r in recs.items()}
    elapsed = (time.time() - t_all) / 60
    if not summary.get("unfinished") and elapsed > args.budget_min:
        log(f"[時間] 探索が終わった時点で {elapsed:.0f}分（上限 {args.budget_min:.0f}分）。比べは次の回に（同じ引数でもう1回）")
        summary["unfinished"] = True
    if summary.get("unfinished"):
        _save(summary, t_all)
        return 0
    del ds

    # 今の規則の形での比べ（実験114b と同じ読み方・同じ3本・同じ最終日）
    if full_e is None:
        full_e = e_frame()
    df = full_e[full_e["label"].notna()].reset_index(drop=True)
    del full_e
    print(f"\n比べの行: {len(df):,}件・{df['Date'].nunique():,}日・{df['Date'].min().date()} 〜 {df['Date'].max().date()}"
          "（実験114b は 22,012件・2,008日）")
    tparams = B.prod_params(cols)
    prev_b = None
    if os.path.exists(B.SUMMARY_PATH):
        with open(B.SUMMARY_PATH, encoding="utf-8") as fh:
            prev_b = json.load(fh)
    res: Dict[int, Dict[int, Dict]] = {k: {} for k in recs}
    for sh in SHIFTS:
        trees = {a: B.oof_tree(df, cols, a, tparams[a], sh) for a in B.TREES}
        for k in sorted(recs):
            res[k][sh] = rule_eval(df, cols, trees, recs[k]["params"], sh)
        same_b = check_114b(res[0][sh]["stats"], prev_b, sh)
        p0 = res[0][sh]
        print(f"\n■ ずらし{sh}か月（{p0['windows']}窓・{p0['rows']:,}件）: R0 今の規則 {p0['stats']['R0']['n']}件 "
              f"{_p(p0['stats']['R0']['ret_w'])} −10%未満 {_s(p0['stats']['R0']['bad_w'])}（窓の平均）/ P0 の数が実験114b と"
              + {True: "同じ", False: "違う", None: "比べられない（実験114b の集計が置き場に無い）"}[same_b])
        print("  木の形  R0D 件数 / 平均収益 / −10%未満（窓の平均）｜ R0 との窓ごとの差 ｜ P0 と同じ行 ｜ LN 件数 / 平均収益 ｜ 判定")
        for k in sorted(recs):
            r = res[k][sh]
            s0, sd, sl = r["stats"]["R0"], r["stats"]["R0D"], r["stats"]["LN"]
            dd = sd["diff"]
            ok = bool(sd["ret_w"] > s0["ret_w"] and sd["bad_w"] < s0["bad_w"])
            share = len(r["picks"] & res[0][sh]["picks"]) / max(1, len(res[0][sh]["picks"]))
            print(f"    P{k}  {sd['n']:>4}件 {_p(sd['ret_w']):>8} {_s(sd['bad_w']):>6} ｜ {dd['mean'] * 100:+.2f} ± "
                  f"{dd['se'] * 100:.2f}pt（上の窓 {dd['won']}/{dd['n']}）｜ {share * 100:5.1f}% ｜ {sl['n']:>4}件 "
                  f"{_p(sl['ret_w']):>8} ｜ {'○' if ok else '×'}")
        summary.setdefault("check114b", {})[str(sh)] = same_b

    verdicts = {k: verdict_row({sh: res[k][sh]["stats"] for sh in SHIFTS}) for k in sorted(recs)}
    print("\n■ 木の形と、探索・本番と同じ窓の NDCG@5（参考）")
    for k in sorted(recs):
        cv = recs[k]["_cv"]
        print(f"  P{k}（〜{recs[k].get('_cutoff')}）探索 {cv['ndcg']:.4f} / out-of-fold（ずらし0）"
              f"{res[k][0]['ndcg_oof']:.4f} / {_tree(recs[k]['params'])}")
    decision = decide(summary["repro_ok"], verdicts)
    print("\n■ 判定（回す前に決めたもの: 再現が一致し、P1〜P4 の4通りすべてで R0D が3通りのずらしすべてで R0 より平均収益"
          "（窓の平均）が高く −10% 未満（窓の平均）が低い）")
    for k in sorted(recs):
        v = verdicts[k]
        print(f"  P{k}: " + " / ".join(f"ずらし{r['shift']} {'○' if r['ok'] else '×'}" for r in v["rows"])
              + f" → {'満たす' if v['pass'] else '満たさない'}" + ("（今の木の形。参考）" if k == 0 else ""))
    print(f"  → {DECISION_JA[decision]}")
    summary["verdicts"] = verdicts
    summary["decision"] = decision
    summary["results"] = {str(k): {str(sh): {"stats": res[k][sh]["stats"], "ndcg_oof": res[k][sh]["ndcg_oof"],
                                             "same_as_p0": len(res[k][sh]["picks"] & res[0][sh]["picks"])
                                             / max(1, len(res[0][sh]["picks"]))}
                                   for sh in SHIFTS} for k in res}
    _save(summary, t_all)
    return 0


def _save(summary: Dict, t_all: float) -> None:
    summary["minutes"] = round((time.time() - t_all) / 60, 1)
    os.makedirs(OOF_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"記録: {SUMMARY_PATH} （{summary['minutes']}分）")


if __name__ == "__main__":
    raise SystemExit(main())
