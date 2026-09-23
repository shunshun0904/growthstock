#!/usr/bin/env python3
"""
本ブレイク予測モデル: 利確を +30% にし、3モデル98%超で絞ったときの損益。あわせて「下落が先に来る
銘柄」の割合と、ファンダメンタルズ・信用残でそれを見分けられるか（ブランチ claude/major-breakout-model）。

運用者の依頼（2026-09-23）
  「利確タイミングを30%にしてほしいです。それと思った以上に対象銘柄があるので、3モデルとも、
   98%超えで購入銘柄を絞るとどうなりますか？」
  「「上下におおきく動く銘柄」を拾う。下落が先に来てしまうものも拾ってしまうということですよね？
   それをファンダメンタルズや、信用倍率等で拾いたいのですが。」

§1 売買の損益: 売り方（場中に +30% / 2倍 に届いたらその値で売る。届かなければ120営業日目の終値）×
   選び方（3モデル95%超 / 98%超、比べ物に日次ボラ98%超・ブレイク全件）。窓の切り方3通り
   （0/2/4か月ずらし）と、探索と重なる窓 / 使っていない窓に分けて出す
§2 上下どちらが先か: 買値から +30%（場中の高値）と −20%（場中の安値）のどちらに先に触れたか
§3 見分けられるか（1列ずつ）: 3モデルの点数が上位10%の行で、「上が先」と「下が先」を分ける力（AUC）を
   ファンダメンタルズ・信用の列ごとに測る。偶然の幅は、窓の中でラベルを入れ替えて測る
§4 見分けられるか（モデル）: ファンダメンタルズ・信用の列だけで「上が先か下が先か」を学習した
   LightGBM で、買い候補を「前の窓の中央値より上 / 下」に分け、損益を比べる

点数は m02 で探索したパラメータ（research/major/params_v3.json）の3モデルを、窓ごとにテスト開始の
121営業日前まで学習し直した out-of-fold（m02 §8 と同じ作り方。キャッシュが別なので計算し直す）。
しきい値（95%点・98%点・中央値）は、その窓より前の窓の点数から決める（先の情報を使わない）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "major"))
import features as F  # noqa: E402
import lab  # noqa: E402
import tuning_multi as TM  # noqa: E402
import label_eda as LE  # noqa: E402
import margin_feats as MF  # noqa: E402
import trade as TR  # noqa: E402
import m01_major_oof as M1  # noqa: E402
import m02_major_tune as M2  # noqa: E402

OOF_DIR = M1.OOF_DIR
ALGOS = M2.ALGOS
S3 = tuple(f"s_{a}" for a in ALGOS)
PARAMS = os.path.join(os.path.dirname(HERE), "major", "params_v3.json")
PREFIX = "m03"
UP, DOWN = 1.30, 0.80           # 利確 +30%。「下が先」の線は −20%（結果を見る前に決めた）
#: 売り方（名前, 利確の倍率, 損切りの倍率）。損切りは運用者の依頼ではなく、「下が先」の痛さを見るための参考
EXITS = (("+30%", 1.30, None), ("2倍", 2.00, None), ("+30%/−20%", 1.30, 0.80))
RULES = (("3モデル 95%超", S3, 95.0), ("3モデル 98%超", S3, 98.0), ("日次ボラ 98%超", ("vol_20d",), 98.0))
SHIFTS = (0, 2, 4)
TOP_SET = 0.90                  # §3 で見る行: 3モデルの点数（前の窓の中での位置の平均）が90%以上
N_PERM = 200
#: 「ファンダメンタルズ・信用」に入れない列（株価・出来高・地合いから作るもの）
TECH_GROUPS = ("price", "breakout", "volume", "sector_index", "market", "flow")
TECH_EXTRA = ("log_trading_value",)
MARGIN_COLS = ("credit_ratio", "alert_days", "alert_longoutratio", "alert_shrtoutratio", "alert_slratio") \
    + MF.COLUMNS
KEY = ["Code", "Date"]
PARTS = ("全窓", "探索と重なる窓", "探索に使っていない窓")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fund_margin_columns(cols: list) -> list:
    """ファンダメンタルズ・信用の列（株価・出来高・地合いから作る列を除く）と、信用残の新しい3本。"""
    tech = {c for g in TECH_GROUPS for c in F.GROUPS[g]} | set(TECH_EXTRA)
    return [c for c in cols if c not in tech] + list(MF.COLUMNS)


def load_params(cols: list, cut6: pd.Timestamp) -> tuple:
    with open(PARAMS, encoding="utf-8") as fh:
        rec = json.load(fh)
    tag = f"{M2.LABEL_VERSION}_{F.signature(cols)}_{cut6.date()}_k{M2.N_SPLITS}_p{M2.PURGE}"
    if rec["tag"] != tag:
        raise SystemExit(f"params_v3.json の探索条件（{rec['tag']}）が今の列・期間（{tag}）と合わない")
    return {a: rec[a] for a in ALGOS}, tag


def dir_oof(seed: int, df: pd.DataFrame, fcols: list, ydir: np.ndarray, plan: list, params: dict) -> pd.DataFrame:
    """ファンダメンタルズ・信用の列で「上が先（1）/ 下が先（0）」を学習した LightGBM の out-of-fold。"""
    parts = []
    X = df[fcols].to_numpy(dtype=float)
    ok = df["tradable"].to_numpy() & np.isfinite(ydir)
    dates = df["Date"].to_numpy()
    try:
        for f, cut in plan:
            if cut is None:
                continue
            tr = ok & (dates <= np.datetime64(cut))
            te = df["tradable"].to_numpy() & (dates >= np.datetime64(pd.Timestamp(f.test_start))) \
                & (dates <= np.datetime64(pd.Timestamp(f.test_end)))
            if te.sum() < M1.MIN_TEST or tr.sum() < M1.MIN_TRAIN:
                continue
            TM.SEED = seed
            ytr = ydir[tr].astype(int)
            m = TM.build("lgbm", params, ytr, fcols)
            m.fit(X[tr], ytr)
            part = df.loc[te, KEY].copy()
            part["score"] = m.predict_proba(X[te])[:, 1]
            part["fold"] = f.index
            parts.append(part)
    finally:
        TM.SEED = 0
    return pd.concat(parts, ignore_index=True)


class Book:
    """1つの切り方（ずらし）ぶんの点数と、行ごとの売買の結果をまとめて持つ。"""

    def __init__(self, df, cal, plan, sc, ex, buy_all):
        self.plan = plan
        self.wins = {f.index: f for f, _ in plan}
        fr = sc["lgbm"][KEY + ["fold"]].copy()
        mi = pd.MultiIndex.from_frame(fr[KEY])
        for a in ALGOS:
            fr[f"s_{a}"] = sc[a].set_index(KEY)["score"].reindex(mi).to_numpy()
        row_of = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_frame(df[KEY]))
        fr["row"] = row_of.reindex(mi).to_numpy().astype(np.int64)
        fr["vol_20d"] = df["vol_20d"].to_numpy(dtype=float)[fr["row"]]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)    # 最初の窓は前の窓が無く全部 NaN
            fr["prio_m"] = np.nanmean([M2.prior_pct(fr, f"s_{a}") for a in ALGOS], axis=0)
        fr["prio_v"] = M2.prior_pct(fr, "vol_20d")
        fr["evald"] = np.isfinite(fr["prio_m"])
        fr["clean"] = fr["fold"].map(lambda k: pd.Timestamp(self.wins[k].test_start) >= M2.CLEAN_FROM).to_numpy()
        self.fr, self.ex, self.buy = fr, ex, buy_all
        self.sel = {name: TR.select_prior(fr, list(c), pct) for name, c, pct in RULES}
        self.sel["ブレイク全件"] = np.ones(len(fr), dtype=bool)

    def part_mask(self, part: str) -> np.ndarray:
        e, c = self.fr["evald"].to_numpy(), self.fr["clean"].to_numpy()
        return {"全窓": e, "探索と重なる窓": e & ~c, "探索に使っていない窓": e & c}[part]

    def months(self, mask: np.ndarray) -> float:
        ks = np.unique(self.fr["fold"].to_numpy()[mask])
        return sum((pd.Timestamp(self.wins[k].test_end) - pd.Timestamp(self.wins[k].test_start)).days + 1
                   for k in ks) / 30.44 if len(ks) else float("nan")

    def prio(self, rule: str) -> np.ndarray:
        return (self.fr["prio_v"] if rule.startswith("日次ボラ") else self.fr["prio_m"]).to_numpy()


HEAD1 = (f"  {'選び方':<22}{'売り方':<10}{'件数':>6}{'月あたり':>8}{'利確到達':>8}{'到達日':>6}{'損切り':>7}{'平均':>8}"
         f"{'中央値':>8}{'勝率':>6}{'下位10%':>9}{'最悪':>8}{'保有日':>7}  {'1銘柄ずつ: 回数':>12}{'資産':>8}{'年率':>8}"
         f"{'最大DD':>8}")


def sline(bk: Book, name: str, xname: str, mask: np.ndarray, months: float, prio: np.ndarray) -> str:
    ii = bk.fr["row"].to_numpy()[mask]
    r, h, w = (x[ii] for x in bk.ex[xname])
    st = TR.summarize(r, h, w)
    if not st["n"]:
        return f"  {name:<22}{xname:<10}{0:>6}"
    target = {x[0]: x[1] for x in EXITS}[xname]
    o = TR.one_at_a_time(bk.buy[ii], h, r, prio[mask], target=target)
    return (f"  {name:<22}{xname:<10}{st['n']:>6}{st['n'] / months:>8.2f}{st['hit']*100:>7.1f}%{st['hit_days']:>6.0f}"
            f"{st['stop']*100:>6.1f}%{st['mean']*100:>+7.1f}%{st['median']*100:>+7.1f}%{st['win']*100:>5.0f}%"
            f"{st['p10']*100:>+8.1f}%"
            f"{st['worst']*100:>+7.1f}%{st['days']:>7.1f}  {o['trades']:>12}{o['multiple']:>7.2f}倍"
            f"{o['cagr']*100:>+7.1f}%{o['mdd']*100:>+7.1f}%")


def section1(bk: Book, shift: int, df: pd.DataFrame) -> None:
    fr = bk.fr
    if shift == 0:
        for name in ("3モデル 98%超", "3モデル 95%超"):
            print(f"\n  --- ずらし0か月: 窓ごと（{name}・場中に +30% で売る）---")
            print(f"  {'窓':>3} {'テスト':<24}{'候補':>7}{'買った':>7}{'利確到達':>8}{'平均':>8}{'中央値':>8}"
                  f"{'最悪':>8}{'保有日':>7}{'本ブレイク':>9}  探索と")
            for k in sorted(np.unique(fr["fold"][fr["evald"]])):
                w = (fr["fold"] == k).to_numpy()
                pick = w & bk.sel[name]
                ii = fr["row"].to_numpy()[pick]
                st = TR.summarize(*(x[ii] for x in bk.ex["+30%"]))
                f = bk.wins[k]
                nmaj = int(np.nansum(df["y"].to_numpy(dtype=float)[ii]))
                tail = (f"{st['hit']*100:>7.1f}%{st['mean']*100:>+7.1f}%{st['median']*100:>+7.1f}%"
                        f"{st['worst']*100:>+7.1f}%{st['days']:>7.1f}{nmaj:>9}" if st["n"]
                        else f"{'-':>8}" * 4 + f"{'-':>7}{'-':>9}")
                print(f"  {k:>3} {f.test_start}〜{f.test_end}{int(w.sum()):>7,}{int(pick.sum()):>7}{tail}"
                      f"  {'重ならない' if pd.Timestamp(f.test_start) >= M2.CLEAN_FROM else '重なる'}")
        print("  候補 = 売買を数えられるブレイク。本ブレイク = 買った銘柄のうち本ブレイクのラベルが1の数")

    print(f"\n  --- ずらし{shift}か月: まとめ ---")
    print(HEAD1)
    for part in PARTS:
        mask = bk.part_mask(part)
        if not mask.any():
            continue
        mo = bk.months(mask)
        print(f"  [{part}] {mo:.0f}か月")
        for name in [r[0] for r in RULES] + ["ブレイク全件"]:
            for xname, _, _ in EXITS:
                print(sline(bk, name, xname, mask & bk.sel[name], mo, bk.prio(name)))
    print("  利確到達 = 期限までに場中でその値に届いた割合。到達日 = 届いたものの、買ってから届くまでの営業日（中央値）。"
          "損切り = +30%/−20% で −20% に先に触れて売った割合（寄りで −20% を割っていたら寄りで売る。同じ日に両方なら損切り）")
    print("  1銘柄ずつ = 持っている間は次を買わず、全額で複利。同じ日に複数出たら点数の位置が高いものを1つ"
          "（ブレイク全件は、空いたらその日で点数の一番高いもの）。最大DD は売った時点の資産で測る")


def order_lines(name: str, ii: np.ndarray, cat, du, dd, ret30) -> list:
    c = cat[ii]
    n = len(ii)
    if not n:
        return [f"  {name:<24}{0:>6}"]
    share = [(c == k).mean() * 100 for k in range(1, 6)]
    mean = [ret30[ii][c == k].mean() * 100 if (c == k).any() else float("nan") for k in range(1, 6)]
    up_d = np.median(du[ii][c == 1]) if (c == 1).any() else float("nan")
    dn_d = np.median(dd[ii][np.isin(c, (2, 3))]) if np.isin(c, (2, 3)).any() else float("nan")
    return [f"  {name:<24}{n:>6}" + "".join(f"{s:>9.1f}%" for s in share) + f"{up_d:>8.0f}{dn_d:>8.0f}",
            f"  {'  └ +30%で売ったときの平均':<24}{'':>6}"
            + "".join(f"{m:>+9.1f}%" if np.isfinite(m) else f"{'-':>10}" for m in mean)]


def section2(books: dict, cat, du, dd, ret30) -> None:
    print(f"\n=== 2. 上下どちらが先か（買値から +{(UP-1)*100:.0f}% と −{(1-DOWN)*100:.0f}% のどちらに先に触れたか・120営業日）===")
    print("  +30% は場中の高値、−20% は場中の安値で判定。同じ日に両方に触れたものは日の中の順番が分からないので別に数える")
    head = (f"  {'行':<24}{'件数':>6}" + "".join(f"{TR.ORDER[k]:>10}" for k in range(1, 6))
            + f"{'上の日':>8}{'下の日':>8}")
    for shift, bk in books.items():
        fr = bk.fr
        rows = fr["row"].to_numpy()
        print(f"\n  --- ずらし{shift}か月・全窓 ---")
        print(head)
        sets = [("3モデル 98%超", bk.sel["3モデル 98%超"]), ("3モデル 95%超", bk.sel["3モデル 95%超"])]
        if shift == 0:
            sets += [("3モデルの点数 上位10%", fr["prio_m"].to_numpy() >= TOP_SET),
                     ("日次ボラ 98%超", bk.sel["日次ボラ 98%超"]), ("ブレイク全件", np.ones(len(fr), dtype=bool))]
        for name, m in sets:
            for ln in order_lines(name, rows[m & fr["evald"].to_numpy()], cat, du, dd, ret30):
                print(ln)
    print("  上の日 = 上が先のものが +30% に届くまでの営業日（中央値）。下の日 = 下が先のものが −20% に触れるまで")
    be = (1 - DOWN) / ((UP - 1) + (1 - DOWN))
    print(f"  +{(UP-1)*100:.0f}%で利確・−{(1-DOWN)*100:.0f}%で損切りなら、損益がとんとんになる「上が先」の割合は"
          f" {be*100:.0f}%（p × {(UP-1)*100:.0f} = (1 − p) × {(1-DOWN)*100:.0f}。どちらも無しの行と窓を開けた下げは除いた概算）")


def auc_screen(bk: Book, df: pd.DataFrame, fcols: list, ydir: np.ndarray, n_perm: int = N_PERM, seed: int = 0):
    """上位10%の行で、列ごとに「上が先」を分ける AUC。窓の中の順位で比べる。偶然の幅は窓の中の入れ替えで。"""
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    fr = bk.fr
    m0 = fr["evald"].to_numpy() & (fr["prio_m"].to_numpy() >= TOP_SET)
    rows = fr["row"].to_numpy()[m0]
    y = ydir[rows]
    ok = np.isfinite(y)
    rows, y = rows[ok], y[ok].astype(int)
    fold = fr["fold"].to_numpy()[m0][ok]
    clean = fr["clean"].to_numpy()[m0][ok]
    in95 = bk.sel["3モデル 95%超"][m0][ok]
    X = df[fcols].to_numpy(dtype=float)[rows]
    R = pd.DataFrame(X).groupby(fold).rank(pct=True).to_numpy()
    rng = np.random.default_rng(seed)
    Yp = np.empty((n_perm, len(y)))
    for p in range(n_perm):
        yy = y.copy()
        for f in np.unique(fold):
            mf = fold == f
            yy[mf] = rng.permutation(yy[mf])
        Yp[p] = yy
    maxdev = np.zeros(n_perm)
    out = []
    for j, c in enumerate(fcols):
        m = np.isfinite(R[:, j])
        rec = {"col": c, "cov": m.mean(), "n": int(m.sum())}
        if m.sum() < 50 or y[m].sum() < 10 or (1 - y[m]).sum() < 10:
            out.append(rec)
            continue
        rk = rankdata(R[m, j])
        yp = Yp[:, m]
        n1 = yp.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ap = (yp @ rk - n1 * (n1 + 1) / 2) / (n1 * (m.sum() - n1))
        dev = np.abs(ap - 0.5)
        maxdev = np.fmax(maxdev, dev)
        rec["auc"] = roc_auc_score(y[m], R[m, j])
        for nm, sub in (("auc_ov", ~clean), ("auc_cl", clean), ("auc_95", in95)):
            ms = m & sub
            rec[nm] = roc_auc_score(y[ms], R[ms, j]) if 0 < y[ms].sum() < ms.sum() else float("nan")
        rec["dev95"] = float(np.nanpercentile(dev, 95))
        out.append(rec)
    res = pd.DataFrame(out)
    return res, float(np.percentile(maxdev, 95)), len(y), int(y.sum())


def section3(bk: Book, df: pd.DataFrame, fcols: list, ydir: np.ndarray) -> None:
    print(f"\n=== 3. ファンダメンタルズ・信用の列1本ずつで「上が先」を見分けられるか（ずらし0か月）===")
    scols = [c for c in fcols if c not in F.GROUPS["sector"]]      # 業種・市場の番号は大小に意味が無いので測らない
    res, floor, n, npos = auc_screen(bk, df, scols, ydir)
    print(f"  行: 3モデルの点数（前の窓の中での位置の平均）が{TOP_SET*100:.0f}%以上のうち、上が先か下が先かが"
          f"決まった {n:,}件（上が先 {npos}件・{npos / n * 100:.1f}%）。列 {len(scols)}本（業種・市場の番号は除く）")
    print("  AUC = その列が大きいほど「上が先」になりやすい度合い（0.5 = 見分けられない、0.5未満は小さいほど上が先）。"
          "窓の中の順位で比べる")
    print(f"  偶然の幅（窓の中でラベルを{N_PERM}回入れ替え）: 1列だけ見たときの |AUC−0.5| の95%点 = 列の中央値 "
          f"{res['dev95'].median():.3f} / {len(scols)}列の中の最大の95%点 = {floor:.3f}")
    res["dev"] = (res["auc"] - 0.5).abs()
    top = res.dropna(subset=["auc"]).sort_values("dev", ascending=False)
    print(f"\n  {'列':<28}{'埋まり':>7}{'件数':>7}{'AUC':>8}{'重なる窓':>9}{'使っていない窓':>13}{'95%超の中':>10}  向き")
    shown = set()
    for _, r in top.head(20).iterrows():
        print(fmt_auc(r, floor))
        shown.add(r["col"])
    print(f"\n  信用の列（順位に関係なく全部）")
    for c in MARGIN_COLS:
        r = res[res["col"] == c]
        if len(r):
            print(fmt_auc(r.iloc[0], floor))
    res.to_csv(os.path.join(OOF_DIR, f"{PREFIX}_auc_screen.csv"), index=False)
    print(f"  向き: ↑ = 大きいほど上が先 / ↓ = 小さいほど上が先。★ = 最大の偶然の幅（{floor:.3f}）を超え、"
          "重なる窓と使っていない窓で向きが同じ")


def fmt_auc(r, floor: float) -> str:
    if not np.isfinite(r.get("auc", np.nan)):
        return f"  {r['col']:<28}{r['cov']*100:>6.0f}%{int(r['n']):>7}{'（少なすぎて測らない）':>20}"
    same = np.sign(r["auc_ov"] - 0.5) == np.sign(r["auc_cl"] - 0.5)
    star = "★" if abs(r["auc"] - 0.5) > floor and same else ""
    return (f"  {r['col']:<28}{r['cov']*100:>6.0f}%{int(r['n']):>7}{r['auc']:>8.3f}{r['auc_ov']:>9.3f}{r['auc_cl']:>13.3f}"
            f"{r['auc_95']:>10.3f}  {'↑' if r['auc'] > 0.5 else '↓'}{star}")


def section4(books: dict, df: pd.DataFrame, fcols: list, ydir: np.ndarray, params: dict, tag_d: str) -> None:
    from sklearn.metrics import roc_auc_score
    print(f"\n=== 4. ファンダメンタルズ・信用の列だけの「上が先か」モデルで、買い候補を分ける ===")
    print(f"  学習: 売買を数えられるブレイクのうち、上が先（1）か下が先（0）かが決まった行。列 {len(fcols)}本。"
          f"LightGBM（m02 で探索したパラメータ）、種{len(M2.SEEDS)}つの平均。窓ごとにテスト開始の121営業日前まで")
    print("  分け方: その窓より前の窓の点数の中央値より上 = 上半分（残す）、以下 = 下半分（外す）")
    for shift, bk in books.items():
        sd = M2.seed_avg(f"dir_{tag_d}_sh{shift}", lambda s, plan=bk.plan: dir_oof(s, df, fcols, ydir, plan, params))
        fr = bk.fr
        fr["s_dir"] = sd.set_index(KEY)["score"].reindex(pd.MultiIndex.from_frame(fr[KEY])).to_numpy()
        keep = TR.select_prior(fr, ["s_dir"], 50.0)
        rows = fr["row"].to_numpy()
        y = ydir[rows]
        q = fr.groupby("fold")["s_dir"].rank(pct=True).to_numpy()
        ev = fr["evald"].to_numpy()
        print(f"\n  --- ずらし{shift}か月 ---")
        line = "  上が先を分ける AUC（窓の中の順位）:"
        for name, m in (("決まった行すべて", ev), ("3モデルの点数 上位10%", ev & (fr["prio_m"].to_numpy() >= TOP_SET)),
                        ("3モデル 95%超", ev & bk.sel["3モデル 95%超"]), ("3モデル 98%超", ev & bk.sel["3モデル 98%超"])):
            mm = m & np.isfinite(y) & np.isfinite(q)
            a = roc_auc_score(y[mm], q[mm]) if 0 < y[mm].sum() < mm.sum() else float("nan")
            line += f" {name} {a:.3f}（{int(mm.sum())}件）/"
        print(line.rstrip("/"))
        print(HEAD1.replace("選び方", "選び方・分け方"))
        for part in PARTS:
            mask = bk.part_mask(part)
            if not mask.any():
                continue
            mo = bk.months(mask)
            print(f"  [{part}] {mo:.0f}か月")
            for name in ("3モデル 98%超", "3モデル 95%超"):
                base = mask & bk.sel[name]
                for tag, m in (("", base), ("・上半分", base & keep), ("・下半分", base & ~keep)):
                    print(sline(bk, name + tag, "+30%", m, mo, bk.prio(name)))
                for tag, m in (("・上半分", base & keep), ("・下半分", base & ~keep)):
                    yy = y[m]
                    yy = yy[np.isfinite(yy)]
                    print(f"    {name + tag} の上が先の割合: {yy.mean()*100:.1f}%（決まった {len(yy)}件）"
                          if len(yy) else f"    {name + tag}: 0件")


def main(argv=None) -> int:
    global SHIFTS, N_PERM
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true", help="手元の動作確認用（種1つ・ずらし0か月だけ・入れ替え20回）")
    args = ap.parse_args(argv)
    M2.PREFIX = PREFIX
    if args.quick:
        M2.PREFIX, M2.SEEDS, SHIFTS, N_PERM = "m03quick", (42,), (0,), 20
    os.makedirs(OOF_DIR, exist_ok=True)

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)
    bars = LE.load_bars(extra=("AdjL", "C", "AdjVo"))
    cal = pd.DatetimeIndex(np.sort(bars["Date"].unique()))
    df["y"] = LE.label_frame(df[KEY], bars)["y_major"].to_numpy()
    cols = [c for c in F.columns(F.DEFAULT_PRESET) if c in df.columns]
    if len(cols) != len(F.columns(F.DEFAULT_PRESET)):
        raise SystemExit("本番の列がデータセットにそろっていない")
    plan0 = M1.fold_plan(df["Date"], cal)
    cut6 = dict((f.index, cut) for f, cut in plan0)[M2.EVAL_FROM]
    tuned, tag = load_params(cols, cut6)

    t0 = time.time()
    mf = MF.build(df[KEY], bars, MF.load_margin(lab.DATA_DIR))
    for c in MF.COLUMNS:
        df[c] = mf[c].to_numpy()
    log(f"信用残の3本を作った（{time.time() - t0:.0f}秒）。埋まり: "
        + " / ".join(f"{c} {df[c].notna().mean()*100:.0f}%" for c in MF.COLUMNS))
    fcols = fund_margin_columns(cols)

    P = TR.forward_hc(bars, df[KEY])
    df["tradable"] = P["tradable"]
    ex = {name: TR.exit_target(P, target=tp) if sl is None else TR.exit_bracket(P, tp, sl) for name, tp, sl in EXITS}
    cat, du, dd = TR.touch_order(P, UP, DOWN)
    ydir = TR.direction_label(cat)
    ydir[~df["tradable"].to_numpy()] = np.nan
    buy_all = cal.searchsorted(df["Date"].to_numpy()) + 1
    del bars

    print("=== 0. 前提 ===")
    print(f"  点数: m02 で探索したパラメータ（{tag}）の3モデル、種{len(M2.SEEDS)}つの平均。列 {len(cols)}本")
    print(f"  売買を数えられるブレイク {int(df['tradable'].sum()):,}件。上が先 {int(np.nansum(ydir == 1)):,}件 / "
          f"下が先 {int(np.nansum(ydir == 0)):,}件 / どちらも無し・同じ日 {int(np.isnan(ydir[df['tradable'].to_numpy()]).sum()):,}件")
    print(f"  ファンダメンタルズ・信用の列: {len(fcols)}本（外した株価・出来高・地合いのグループ: {', '.join(TECH_GROUPS)}"
          f" と {', '.join(TECH_EXTRA)}）")
    print("  売り方: 場中の高値がその値に届いたらちょうどその値で売る（指値）。届かなければ120営業日目の終値、"
          "途中で上場廃止したら最後の終値。手数料・税なし")

    books = {}
    for shift in SHIFTS:
        plan = M1.fold_plan(df["Date"], cal, shift)
        sc = {a: M2.seed_avg(f"trade_{a}_{tag}_sh{shift}",
                             lambda sd, a=a, plan=plan: M2.oof_trade(a, sd, df, cols, tuned[a], plan)) for a in ALGOS}
        books[shift] = Book(df, cal, plan, sc, ex, buy_all)

    print("\n=== 1. 売買の損益（利確 +30% と 2倍、3モデル 95%超と 98%超）===")
    for shift, bk in books.items():
        section1(bk, shift, df)
    section2(books, cat, du, dd, ex["+30%"][0])
    section3(books[0], df, fcols, ydir)
    tag_d = f"v1_{F.signature(fcols)}_u{UP}_d{DOWN}"
    section4(books, df, fcols, ydir, tuned["lgbm"], tag_d)
    log(f"記録: {OOF_DIR}/{PREFIX}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
