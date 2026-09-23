#!/usr/bin/env python3
"""
会計フローグラフの Temporal GNN（GraphSAGE + GRU）。

## 何をするか

四半期ごとの会計フローグラフ（J-Quants の18ノード・21エッジ）を
GraphSAGE でまとめ、過去8四半期ぶんを GRU でつなぎ、決算後20営業日の
超過リターンの3クラス（下落 / 中立 / 上昇）を予測する。

EDINET の明細ノードは使わない。2段構えの測定（`increment.py`、
2026-09-22）で、106社の範囲では上積みが見つからなかったため。

## 比べるもの

ベースライン（`latest_jq` / `seq_jq` × logit / lgbm / mlp）と、
**同じ母集団・同じ walk-forward の分割・同じ行**で比べる。

GNN が勝ったとき・負けたときに理由を分けるため、切り離し版も回す。

  sage_gru     本体。GraphSAGE（エッジ特徴つき）2層 -> GRU（8四半期）
  mlp_gru      エッジを外した版。ノードごとに同じ深さの層を通すだけで、
               ノードの間で情報をやりとりしない。本体との差がエッジ
               （会計フロー）の効果
  sage_latest  時系列を外した版。当該四半期のグラフだけ。本体との差が
               時系列の効果

ニューラルネットは初期値で結果が揺れるので、各版を乱数3通りで学習し、
予測確率を平均したものを評価する。乱数ごとの AUC も並べる。

## 判定基準（結果を見る前に決めておく）

差 = AUC(sage_gru) − AUC(最良のベースライン) を同じ行で取り、
95%区間（発表日単位のブートストラップ）が

  GNN が有効               0 を上回る
  ベースラインのほうが良い  0 を下回る
  差が見えない             0 をまたぐ

最良のベースラインは、6本の中で AUC の点推定が最も高いもの。
ベースラインに最も有利な選び方なので、GNN に甘くならない。
バックテストは参考として並べ、判定には使わない。

切り離し版の読み方も同じ区間で決める。

  AUC(sage_gru) − AUC(mlp_gru)      0 を上回ればエッジが効いている
  AUC(sage_gru) − AUC(sage_latest)  0 を上回れば時系列が効いている

## 使い方

  python3 research/accgraph/gnn.py baselines   # ベースラインの予測を保存
  python3 research/accgraph/gnn.py train       # GNN の予測を保存
  python3 research/accgraph/gnn.py report      # 比べて docs/ACCGRAPH_GNN.md
  python3 research/accgraph/gnn.py all         # 3つを続けて

予測は research/_data/accgraph/gnn/ に保存する。段を分けてあるので、
GNN だけ直して回し直すときにベースラインを作り直さなくてよい。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
ROOT = os.path.dirname(RESEARCH)
sys.path.insert(0, RESEARCH)

from accgraph import (backtest, baselines, build, diagnose, increment,  # noqa: E402
                      schema, splits)
from accgraph import labels as L  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "ACCGRAPH_GNN.md")
PRED_DIR = os.path.join(build.OUT_DIR, "gnn")
OUT_JSON = os.path.join(PRED_DIR, "gnn_results.json")
CLASSES = np.array([0, 1, 2])

BASELINE_SETS = ("latest_jq", "seq_jq")
BASELINE_MODELS = ("logit", "lgbm", "mlp")
VARIANTS = ("sage_gru", "mlp_gru", "sage_latest")
MAIN = "sage_gru"
SEEDS = (0, 1, 2)

VARIANT_JA = {
    "sage_gru": "GraphSAGE + GRU（本体）",
    "mlp_gru": "エッジを外した版（ノードごと + GRU）",
    "sage_latest": "時系列を外した版（当該四半期の GraphSAGE）",
}


@dataclass
class TrainConfig:
    hidden: int = 32
    layers: int = 2
    id_dim: int = 8
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 25
    patience: int = 4
    #: これより前は早期打ち切りしない。学習の出だしに検証の損失が
    #: 横ばいの区間があり（合成データで更新40〜60回ぶん）、そこで
    #: 打ち切ると何も学ばないまま終わる
    min_epochs: int = 5
    #: 訓練窓の末尾をこの割合だけ検証に回し、早期打ち切りに使う
    val_frac: float = 0.15
    #: 標準化したあとの上下限。比率は裾が重い
    clip: float = 5.0


# --------------------------------------------------------------------------- #
# グラフ
# --------------------------------------------------------------------------- #

def jq_graph() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    J-Quants のノードと、両端が J-Quants のノードであるエッジ。

    ベースラインの `_jq` と同じ範囲（`baselines.flatten`）。
    返り値は (ノード番号, エッジ番号, エッジの始点, エッジの終点)。
    始点・終点は J-Quants のノードだけに振り直した番号。
    """
    node_idx = [j for j, n in enumerate(schema.NODES) if n.source_table == "fins"]
    pos = {j: k for k, j in enumerate(node_idx)}
    edge_idx, src, dst = [], [], []
    for i, e in enumerate(schema.EDGES):
        s, d = schema.NODE_INDEX[e.src], schema.NODE_INDEX[e.dst]
        if s in pos and d in pos:
            edge_idx.append(i)
            src.append(pos[s])
            dst.append(pos[d])
    return (np.array(node_idx), np.array(edge_idx),
            np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64))


# --------------------------------------------------------------------------- #
# 前処理
# --------------------------------------------------------------------------- #

class Scaler:
    """
    訓練行だけで平均と標準偏差を取り、最後の軸以外（行・時点）で集計する。

    ノード特徴量は (ノード, 特徴量) ごと、エッジ特徴量は (エッジ, 特徴量) ごと。
    保存済みの特徴量は欠測が0で埋まっていて、欠測かどうかは is_missing が持つ。
    """

    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, a: np.ndarray, rows: np.ndarray) -> "Scaler":
        sub = a[rows].astype(np.float64)
        axes = (0, 1) if sub.ndim == 4 else (0,)
        self.mu = sub.mean(axis=axes)
        sd = sub.std(axis=axes)
        self.sd = np.where(sd > 1e-8, sd, 1.0)
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        out = ((a - self.mu) / self.sd).astype(np.float32)
        np.clip(out, -self.clip, self.clip, out=out)
        return out


def inner_split(dates: pd.Series, ready: pd.Series, train_rows: np.ndarray,
                val_frac: float, embargo_days: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    訓練窓の末尾 val_frac を検証に回す（早期打ち切り用）。

    検証の前にも purge / embargo を置く。訓練に残す行は、ラベルが検証の
    開始前に確定し、かつ検証の開始より embargo 以上前に入ったものだけ。
    """
    d = pd.to_datetime(dates).to_numpy()[train_rows]
    r = pd.to_datetime(ready).fillna(pd.Timestamp.max).to_numpy()[train_rows]
    cut = np.quantile(d.astype("datetime64[ns]").astype(np.int64), 1.0 - val_frac)
    cut = np.datetime64(int(cut), "ns")
    va = train_rows[d >= cut]
    keep = (d < cut - np.timedelta64(embargo_days, "D")) & (r < cut)
    return train_rows[keep], va


# --------------------------------------------------------------------------- #
# モデル
# --------------------------------------------------------------------------- #

def _torch():
    import torch
    return torch


def build_model(variant: str, n_nodes: int, n_feat: int, n_efeat: int,
                n_ctx: int, src: np.ndarray, dst: np.ndarray,
                cfg: TrainConfig):
    """variant ごとのモデルを組み立てる（torch はここで初めて読む）。"""
    torch = _torch()
    nn = torch.nn
    H = cfg.hidden

    class SageLayer(nn.Module):
        """
        エッジ特徴つきの GraphSAGE（平均集約）。流れの向きと逆向きで重みを分ける。

        use_edges=False のときはノード自身の変換だけになり、ノードの間で
        情報をやりとりしない（mlp_gru の切り離し）。
        """

        def __init__(self, use_edges: bool):
            super().__init__()
            self.use_edges = use_edges
            self.self_lin = nn.Linear(H, H)
            self.norm = nn.LayerNorm(H)
            self.drop = nn.Dropout(cfg.dropout)
            if use_edges:
                self.fwd = nn.Linear(H + n_efeat, H)
                self.bwd = nn.Linear(H + n_efeat, H)
                s = torch.as_tensor(src, dtype=torch.long)
                d = torch.as_tensor(dst, dtype=torch.long)
                self.register_buffer("src", s)
                self.register_buffer("dst", d)
                deg_in = torch.zeros(n_nodes).index_add_(0, d, torch.ones(len(d)))
                deg_out = torch.zeros(n_nodes).index_add_(0, s, torch.ones(len(s)))
                self.register_buffer("deg_in", deg_in.clamp(min=1.0)[None, :, None])
                self.register_buffer("deg_out", deg_out.clamp(min=1.0)[None, :, None])

        def forward(self, h, e):
            out = self.self_lin(h)
            if self.use_edges:
                m_f = self.fwd(torch.cat([h[:, self.src], e], dim=-1))
                m_b = self.bwd(torch.cat([h[:, self.dst], e], dim=-1))
                out = (out
                       + torch.zeros_like(h).index_add(1, self.dst, m_f) / self.deg_in
                       + torch.zeros_like(h).index_add(1, self.src, m_b) / self.deg_out)
            return h + self.drop(torch.relu(self.norm(out)))

    class GraphEncoder(nn.Module):
        """1四半期のグラフを H 次元にまとめる。"""

        def __init__(self, use_edges: bool):
            super().__init__()
            # グラフは全サンプル共通なので、どのノードかを埋め込みで持たせる
            self.node_id = nn.Parameter(torch.randn(n_nodes, cfg.id_dim) * 0.1)
            self.inp = nn.Linear(n_feat + cfg.id_dim, H)
            self.layers = nn.ModuleList([SageLayer(use_edges)
                                         for _ in range(cfg.layers)])
            self.readout = nn.Linear(2 * H, H)

        def forward(self, x, e):
            g = x.shape[0]
            h = torch.relu(self.inp(torch.cat(
                [x, self.node_id.expand(g, -1, -1)], dim=-1)))
            for layer in self.layers:
                h = layer(h, e)
            return torch.relu(self.readout(
                torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)))

    class TemporalGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = variant != "sage_latest"
            self.enc = GraphEncoder(use_edges=variant != "mlp_gru")
            if self.temporal:
                # 各時点のグラフ要約 + その四半期が存在したか
                self.gru = nn.GRU(H + 1, H, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(H + n_ctx, H), nn.ReLU(), nn.Dropout(cfg.dropout),
                nn.Linear(H, len(CLASSES)))

        def forward(self, x, e, m, c):
            """
            x (B, T, n, F) / e (B, T, E, Fe) / m (B, T) / c (B, C)。
            時点 0 が当該四半期、1 以降が過去。
            """
            if not self.temporal:
                z = self.enc(x[:, 0], e[:, 0])
            else:
                b, t = x.shape[:2]
                g = self.enc(x.reshape((b * t,) + x.shape[2:]),
                             e.reshape((b * t,) + e.shape[2:])).reshape(b, t, -1)
                seq = torch.cat([g, m[..., None]], dim=-1)
                # GRU には古い順に入れる。最後の隠れ状態が当該四半期を踏まえたものになる
                _, hn = self.gru(torch.flip(seq, dims=[1]))
                z = hn[-1]
            return self.head(torch.cat([z, c], dim=-1))

    if variant not in VARIANTS:
        raise ValueError(f"未知の版: {variant}")
    return TemporalGNN()


# --------------------------------------------------------------------------- #
# 学習
# --------------------------------------------------------------------------- #

@dataclass
class Arrays:
    """1フォールドぶんの、標準化済みの入力。"""
    x: np.ndarray       # (N, T, n, F) float32
    e: np.ndarray       # (N, T, E, Fe) float32
    m: np.ndarray       # (N, T) float32
    c: np.ndarray       # (N, C) float32


def prepare(nf: np.ndarray, ef: np.ndarray, pm: np.ndarray, ctx: np.ndarray,
            train_rows: np.ndarray, cfg: TrainConfig, t_len: int) -> Arrays:
    """訓練行だけで標準化の統計を取り、全行を変換する。"""
    x = nf[:, :t_len]
    e = ef[:, :t_len]
    xs = Scaler(cfg.clip).fit(x, train_rows)
    es = Scaler(cfg.clip).fit(e, train_rows)
    cs = Scaler(cfg.clip).fit(ctx, train_rows)
    return Arrays(xs.transform(x), es.transform(e),
                  pm[:, :t_len].astype(np.float32), cs.transform(ctx))


def _batch(arr: Arrays, idx: np.ndarray):
    torch = _torch()
    return (torch.from_numpy(arr.x[idx]), torch.from_numpy(arr.e[idx]),
            torch.from_numpy(arr.m[idx]), torch.from_numpy(arr.c[idx]))


def predict(model, arr: Arrays, idx: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(idx), batch_size):
            b = idx[a:a + batch_size]
            out.append(torch.softmax(model(*_batch(arr, b)), dim=-1).numpy())
    return (np.concatenate(out, axis=0) if out
            else np.zeros((0, len(CLASSES)), dtype=np.float32))


def train_one(variant: str, arr: Arrays, y: np.ndarray, tr: np.ndarray,
              va: np.ndarray, src: np.ndarray, dst: np.ndarray,
              cfg: TrainConfig, seed: int):
    """
    1つの版を1つの乱数で学習する。検証の対数損失で早期打ち切りし、
    最も良かったエポックの重みを返す。検証が空なら最後まで回す。
    """
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(variant, arr.x.shape[2], arr.x.shape[3], arr.e.shape[3],
                        arr.c.shape[1], src, dst, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    yt = torch.from_numpy(y.astype(np.int64))

    best = (float("inf"), None, -1)
    bad = 0
    history = []
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = rng.permutation(tr)
        tot, cnt = 0.0, 0
        for a in range(0, len(perm), cfg.batch_size):
            b = perm[a:a + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(*_batch(arr, b)), yt[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        train_loss = tot / max(cnt, 1)
        if len(va):
            p = predict(model, arr, va)
            val_loss = float(-np.log(np.clip(p[np.arange(len(va)), y[va]],
                                             1e-12, None)).mean())
        else:
            val_loss = train_loss
        history.append((train_loss, val_loss))
        if val_loss < best[0] - 1e-5:
            best = (val_loss, {k: v.detach().clone()
                               for k, v in model.state_dict().items()}, epoch)
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience and epoch + 1 >= cfg.min_epochs:
                break
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, {"best_epoch": best[2], "best_val_loss": best[0],
                   "epochs_run": len(history), "history": history}


# --------------------------------------------------------------------------- #
# データと分割
# --------------------------------------------------------------------------- #

def load_data(data_dir: str, benchmark: str, horizon: int):
    meta, nf, ef, pm = build.load(data_dir, liquid_only=True)
    y_col = f"y_{benchmark}_{horizon}d"
    if y_col not in meta.columns:
        raise SystemExit(f"{y_col} がデータセットにありません")
    usable = meta[y_col].to_numpy() >= 0
    meta = meta[usable].reset_index(drop=True)
    nf, ef, pm = nf[usable], ef[usable], pm[usable]
    y = meta[y_col].to_numpy().astype(int)
    return meta, nf, ef, pm, y


def make_folds(meta: pd.DataFrame, *, min_train_months: int, test_months: int,
               step_months: int, max_horizon: int, min_test_rows: int):
    return splits.walk_forward(
        meta, min_train_months=min_train_months, test_months=test_months,
        step_months=step_months, embargo_trading_days=max_horizon,
        min_test_rows=min_test_rows)


def _fingerprint(meta: pd.DataFrame, y: np.ndarray) -> Dict:
    """段を分けて保存した予測が、同じ行に付いたものかを確かめるための印。"""
    return {"n": int(len(y)), "y_sum": int(y.sum()),
            "first": str(meta["anchor_id"].iloc[0]) if len(meta) else "",
            "last": str(meta["anchor_id"].iloc[-1]) if len(meta) else ""}


def _save(path: str, preds: Dict[str, np.ndarray], fp: Dict, extra: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{k.replace("/", "__"): v for k, v in preds.items()})
    with open(path + ".json", "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": fp, "names": list(preds), **extra}, fh,
                  ensure_ascii=False, indent=1)


def _load(path: str, fp: Dict) -> Tuple[Dict[str, np.ndarray], Dict]:
    with open(path + ".json", encoding="utf-8") as fh:
        info = json.load(fh)
    if info["fingerprint"] != fp:
        raise SystemExit(f"{path} は別のデータで作られています。作り直してください "
                         f"({info['fingerprint']} != {fp})")
    z = np.load(path)
    return {n: z[n.replace("/", "__")] for n in info["names"]}, info


# --------------------------------------------------------------------------- #
# 段 1: ベースライン
# --------------------------------------------------------------------------- #

def run_baselines(data_dir: str, pred_dir: str, *, benchmark: str, horizon: int,
                  fold_kw: Dict, sets: Sequence[str] = BASELINE_SETS,
                  models: Sequence[str] = BASELINE_MODELS, seed: int = 0) -> str:
    meta, nf, ef, pm, y = load_data(data_dir, benchmark, horizon)
    folds, tr, te = make_folds(meta, **fold_kw)
    print(splits.describe(folds))
    preds = {}
    for fs in sets:
        X, _ = baselines.flatten(nf, ef, pm, meta, kind=fs)
        for m in models:
            t0 = time.time()
            p = diagnose.oof_full(meta, X, y, m, tr, te, seed=seed)
            ok = np.isfinite(p).all(axis=1)
            auc = diagnose.macro_auc(y[ok], p[ok])
            print(f"[gnn] ベースライン {fs}/{m}: AUC {auc:.4f} "
                  f"({X.shape[1]:,}次元, {time.time() - t0:.0f}秒)", flush=True)
            preds[f"{fs}/{m}"] = p.astype(np.float32)
    path = os.path.join(pred_dir, "oof_baselines.npz")
    _save(path, preds, _fingerprint(meta, y), {"folds": [f.to_dict() for f in folds]})
    return path


# --------------------------------------------------------------------------- #
# 段 2: GNN
# --------------------------------------------------------------------------- #

def run_gnn(data_dir: str, pred_dir: str, *, benchmark: str, horizon: int,
            fold_kw: Dict, variants: Sequence[str] = VARIANTS,
            seeds: Sequence[int] = SEEDS, cfg: Optional[TrainConfig] = None) -> str:
    torch = _torch()
    torch.set_num_threads(os.cpu_count() or 1)
    cfg = cfg or TrainConfig()
    meta, nf, ef, pm, y = load_data(data_dir, benchmark, horizon)
    folds, tr_masks, te_masks = make_folds(meta, **fold_kw)
    print(splits.describe(folds))

    node_idx, edge_idx, src, dst = jq_graph()
    nf = nf[:, :, node_idx]
    ef = ef[:, :, edge_idx]
    ctx, _ = baselines.context_features(meta, pm)
    ctx = ctx.astype(np.float32)
    t_full = nf.shape[1]
    embargo_days = int(round(fold_kw["max_horizon"] * splits.TRADING_TO_CALENDAR))
    print(f"[gnn] ノード {len(node_idx)} / エッジ {len(edge_idx)} / "
          f"時点 {t_full} / 設定 {asdict(cfg)}", flush=True)

    preds: Dict[str, np.ndarray] = {}
    logs: Dict[str, List] = {}
    for v in variants:
        t_len = 1 if v == "sage_latest" else t_full
        per_seed = {s: np.full((len(y), len(CLASSES)), np.nan, dtype=np.float32)
                    for s in seeds}
        logs[v] = []
        for f, trm, tem in zip(folds, tr_masks, te_masks):
            train_rows = np.flatnonzero(trm)
            test_rows = np.flatnonzero(tem)
            arr = prepare(nf, ef, pm, ctx, train_rows, cfg, t_len)
            tr, va = inner_split(meta["entry_date"], meta["label_ready_date"],
                                 train_rows, cfg.val_frac, embargo_days)
            for s in seeds:
                t0 = time.time()
                model, info = train_one(v, arr, y, tr, va, src, dst, cfg, seed=s)
                per_seed[s][test_rows] = predict(model, arr, test_rows)
                auc = diagnose.macro_auc(y[test_rows], per_seed[s][test_rows])
                print(f"[gnn] {v:<11} fold{f.index} seed{s}: AUC {auc:.4f} / "
                      f"エポック {info['best_epoch'] + 1}/{info['epochs_run']} / "
                      f"訓練 {len(tr):,} 検証 {len(va):,} / "
                      f"{time.time() - t0:.0f}秒", flush=True)
                logs[v].append({"fold": f.index, "seed": s, "auc": auc,
                                "n_train": int(len(tr)), "n_val": int(len(va)),
                                **{k: info[k] for k in ("best_epoch", "epochs_run",
                                                        "best_val_loss")}})
        for s in seeds:
            preds[f"{v}/seed{s}"] = per_seed[s]
        preds[v] = np.mean([per_seed[s] for s in seeds], axis=0).astype(np.float32)
    path = os.path.join(pred_dir, "oof_gnn.npz")
    _save(path, preds, _fingerprint(meta, y),
          {"config": asdict(cfg), "seeds": list(seeds), "logs": logs})
    return path


# --------------------------------------------------------------------------- #
# 段 3: 比較
# --------------------------------------------------------------------------- #

def _verdict(ci: Sequence[float], pos: str, neg: str, none: str) -> str:
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "判定できない"
    if lo > 0.0:
        return pos
    if hi < 0.0:
        return neg
    return none


def verdict_main(ci: Sequence[float]) -> str:
    return _verdict(ci, "GNN が有効", "ベースラインのほうが良い", "差が見えない")


def verdict_edges(ci: Sequence[float]) -> str:
    return _verdict(ci, "エッジが効いている", "エッジが害になっている",
                    "エッジの効果は見えない")


def verdict_time(ci: Sequence[float]) -> str:
    return _verdict(ci, "時系列が効いている", "時系列が害になっている",
                    "時系列の効果は見えない")


def report(data_dir: str, pred_dir: str, *, benchmark: str, horizon: int,
           fold_kw: Dict, cost_bps: float = backtest.DEFAULT_COST_BPS,
           n_boot: int = 1000, seed: int = 0) -> Dict:
    from accgraph.evaluate import classification_metrics

    meta, _, _, _, y = load_data(data_dir, benchmark, horizon)
    fp = _fingerprint(meta, y)
    base, base_info = _load(os.path.join(pred_dir, "oof_baselines.npz"), fp)
    gnn, gnn_info = _load(os.path.join(pred_dir, "oof_gnn.npz"), fp)
    folds, _, _ = make_folds(meta, **fold_kw)

    names_base = [k for k in base]
    names_gnn = [v for v in VARIANTS if v in gnn]
    if MAIN not in gnn:
        raise SystemExit(f"{MAIN} の予測がありません")
    allp = {**base, **{v: gnn[v] for v in names_gnn}}
    ok = np.logical_and.reduce([np.isfinite(p).all(axis=1) for p in allp.values()])
    print(f"[gnn] 比較する行 {int(ok.sum()):,}（全モデルに予測が付いた行）")
    m_ok = meta[ok].reset_index(drop=True)
    y_ok = y[ok]
    excess_col = f"excess_{benchmark}_{horizon}d"

    rows = {}
    for name, p in allp.items():
        q = p[ok]
        met = classification_metrics(y_ok, q)
        bt = {rule: backtest.run(m_ok, q, CLASSES, excess_col=excess_col,
                                 horizon=horizon, max_horizon=fold_kw["max_horizon"],
                                 rule=rule, cost_bps=cost_bps).to_dict()
              for rule in ("predicted_up", "topk", "all")}
        rows[name] = {"metrics": met, "backtest": bt}

    best = max(names_base, key=lambda k: rows[k]["metrics"]["roc_auc_macro"])
    comp = {MAIN: gnn[MAIN][ok], best: base[best][ok]}
    pairs = [(MAIN, best)]
    for v in ("mlp_gru", "sage_latest"):
        if v in gnn:
            comp[v] = gnn[v][ok]
            pairs.append((MAIN, v))
    boot = increment.bootstrap_models(y_ok, comp, m_ok["entry_date"], pairs,
                                      n_boot=n_boot, seed=seed)

    diff_main = boot["diff"][f"{MAIN}-{best}"]
    out = {
        "benchmark": benchmark, "horizon": horizon, "cost_bps": cost_bps,
        "n_boot": n_boot, "n_rows": int(ok.sum()),
        "n_codes": int(m_ok["Code"].nunique()),
        "period": [str(pd.to_datetime(m_ok["entry_date"]).min().date()),
                   str(pd.to_datetime(m_ok["entry_date"]).max().date())],
        "class_share": [float((y_ok == c).mean()) for c in CLASSES],
        "folds": [f.to_dict() for f in folds],
        "config": gnn_info.get("config"), "seeds": gnn_info.get("seeds"),
        "rows": rows, "best_baseline": best, "boot": boot,
        "verdict": verdict_main(diff_main["ci"]),
        "logs": gnn_info.get("logs", {}),
        "seed_auc": {k: diagnose.macro_auc(y_ok, gnn[k][ok])
                     for k in gnn if "/seed" in k},
    }
    if f"{MAIN}-mlp_gru" in boot["diff"]:
        out["verdict_edges"] = verdict_edges(boot["diff"][f"{MAIN}-mlp_gru"]["ci"])
    if f"{MAIN}-sage_latest" in boot["diff"]:
        out["verdict_time"] = verdict_time(boot["diff"][f"{MAIN}-sage_latest"]["ci"])
    print(f"[gnn] 判定: {out['verdict']}（{MAIN} − {best} = "
          f"{diff_main['point']:+.4f} [{diff_main['ci'][0]:+.4f}, "
          f"{diff_main['ci'][1]:+.4f}]）")
    return out


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #

def _f(x, fmt):
    return "—" if x is None or not np.isfinite(x) else format(x, fmt)


def _ci(c, fmt="+.4f") -> str:
    if c is None or any(x is None or not np.isfinite(x) for x in c):
        return "—"
    return f"[{format(c[0], fmt)}, {format(c[1], fmt)}]"


def _label(name: str) -> str:
    if name in VARIANT_JA:
        return f"**{name}**"
    return f"`{name}`"


def to_markdown(d: Dict) -> str:
    b = d["benchmark"]
    best = d["best_baseline"]
    boot = d["boot"]
    dm = boot["diff"][f"{MAIN}-{best}"]
    lines = [
        "# 会計フローグラフ — Temporal GNN（GraphSAGE + GRU）",
        "",
        "`research/accgraph/gnn.py` の出力。**実行した結果のみ**を載せる。",
        "",
        "## 結論",
        "",
        f"**{d['verdict']}**。{MAIN} − 最良のベースライン（`{best}`）の AUC の差は "
        f"{dm['point']:+.4f} {_ci(dm['ci'])}。",
    ]
    if "verdict_edges" in d:
        de = boot["diff"][f"{MAIN}-mlp_gru"]
        lines.append(f"エッジを外した版との差は {de['point']:+.4f} {_ci(de['ci'])}"
                     f"（{d['verdict_edges']}）。")
    if "verdict_time" in d:
        dt = boot["diff"][f"{MAIN}-sage_latest"]
        lines.append(f"時系列を外した版との差は {dt['point']:+.4f} {_ci(dt['ci'])}"
                     f"（{d['verdict_time']}）。")
    lines += [
        "",
        "## 条件",
        "",
        f"- ベンチマーク: {'TOPIX' if b == 'topix' else '業種指数'}控除 / "
        f"{d['horizon']}営業日（発表翌営業日の始値でエントリー）",
        f"- 母集団: 流動性1億円以上の全銘柄。比べた行 {d['n_rows']:,}件・"
        f"{d['n_codes']:,}社（{d['period'][0]} 〜 {d['period'][1]}）",
        "- クラス比: " + " / ".join(f"{L.CLASS_NAMES[i]} {s * 100:.1f}%"
                                   for i, s in enumerate(d["class_share"])),
        f"- 分割: 訓練窓を伸ばす walk-forward {len(d['folds'])}本"
        "（Purge / Embargo つき）。全モデルで同じ分割・同じ行",
        f"- GNN の入力: J-Quants の18ノード・21エッジ、過去8四半期。"
        f"乱数 {d['seeds']} の予測確率を平均",
        f"- GNN の設定: {d['config']}",
        f"- 区間: 発表日単位のブートストラップ {d['n_boot']:,}回の95%区間",
        f"- 取引コスト: 往復 {d['cost_bps']:.0f}bp（バックテストは参考）",
        "",
        "## 判定基準（結果を見る前に決めたもの）",
        "",
        "| 比較 | 0 を上回る | 0 を下回る | 0 をまたぐ |",
        "| --- | --- | --- | --- |",
        "| sage_gru − 最良のベースライン | GNN が有効 | ベースラインのほうが良い | 差が見えない |",
        "| sage_gru − mlp_gru | エッジが効いている | エッジが害になっている | エッジの効果は見えない |",
        "| sage_gru − sage_latest | 時系列が効いている | 時系列が害になっている | 時系列の効果は見えない |",
        "",
        "最良のベースラインは6本の中で AUC の点推定が最も高いもの"
        "（ベースラインに最も有利な選び方）。",
        "",
        "## AUC と損益",
        "",
        "| モデル | ROC-AUC(macro) | Accuracy | F1(macro) | Sharpe(上昇判定) | 累積 | 最大DD "
        "| Sharpe(上位20%) | トレード数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = sorted(d["rows"], key=lambda k: -d["rows"][k]["metrics"]["roc_auc_macro"])
    for name in order:
        r = d["rows"][name]
        m, bu, bk = r["metrics"], r["backtest"]["predicted_up"], r["backtest"]["topk"]
        lines.append(
            f"| {_label(name)} | {_f(m['roc_auc_macro'], '.4f')} "
            f"| {_f(m['accuracy'], '.4f')} | {_f(m['f1_macro'], '.4f')} "
            f"| {_f(bu['sharpe'], '.2f')} | {_f(bu['cum_return'] * 100, '.1f')}% "
            f"| {_f(bu['max_drawdown'] * 100, '.1f')}% | {_f(bk['sharpe'], '.2f')} "
            f"| {bu['n_trades']:,} |")
    all_sh = d["rows"][order[0]]["backtest"]["all"]["sharpe"]
    lines += [
        "",
        f"参考: 全件買い（判定を使わない）の Sharpe は {_f(all_sh, '.2f')}。",
        "",
        "## 差の区間",
        "",
        "| 比較 | AUC の差 | 95%区間 | 読み |",
        "| --- | ---: | --- | --- |",
        f"| {MAIN} − `{best}` | {dm['point']:+.4f} | {_ci(dm['ci'])} | {d['verdict']} |",
    ]
    if "verdict_edges" in d:
        de = boot["diff"][f"{MAIN}-mlp_gru"]
        lines.append(f"| {MAIN} − mlp_gru | {de['point']:+.4f} | {_ci(de['ci'])} "
                     f"| {d['verdict_edges']} |")
    if "verdict_time" in d:
        dt = boot["diff"][f"{MAIN}-sage_latest"]
        lines.append(f"| {MAIN} − sage_latest | {dt['point']:+.4f} | {_ci(dt['ci'])} "
                     f"| {d['verdict_time']} |")

    lines += [
        "",
        "## 乱数ごとの AUC",
        "",
        "平均した予測の AUC と、乱数1本ずつの AUC。乱数でどれだけ揺れるかの目安。",
        "",
        "| 版 | 平均した予測 | 乱数ごと |",
        "| --- | ---: | --- |",
    ]
    for v in VARIANTS:
        if v not in d["rows"]:
            continue
        per = [f"{d['seed_auc'][k]:.4f}" for k in sorted(d["seed_auc"])
               if k.startswith(v + "/")]
        lines.append(f"| {v}（{VARIANT_JA[v]}） "
                     f"| {d['rows'][v]['metrics']['roc_auc_macro']:.4f} "
                     f"| {' / '.join(per)} |")

    lines += [
        "",
        "## フォールドごとの学習",
        "",
        "| 版 | フォールド | 乱数 | テスト AUC | 採ったエポック / 回したエポック | 訓練 | 検証 |",
        "| --- | :-: | :-: | ---: | ---: | ---: | ---: |",
    ]
    for v, logs in d["logs"].items():
        for g in logs:
            lines.append(f"| {v} | {g['fold']} | {g['seed']} | {_f(g['auc'], '.4f')} "
                         f"| {g['best_epoch'] + 1} / {g['epochs_run']} "
                         f"| {g['n_train']:,} | {g['n_val']:,} |")

    lines += [
        "",
        "## 読み方",
        "",
        "- 採ったエポックが毎回1〜2なら、学習がすぐ過学習に入っている。"
        "検証の損失が最初から下がらない（信号が弱い）か、モデルが大きすぎる。",
        "- ベースラインは `latest_jq`（当該四半期のみ）と `seq_jq`（過去8四半期）。"
        "どちらも J-Quants のノードだけで、GNN と同じ情報を平らに並べたもの。",
        "- 損益は判定に使っていない。20営業日保有だと独立な期間が少なく、"
        "運の影響が大きいため。",
        "",
    ]
    return "\n".join(lines)


def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="会計フローグラフの Temporal GNN")
    ap.add_argument("stage", choices=["baselines", "train", "report", "all"])
    ap.add_argument("--data-dir", default=build.OUT_DIR)
    ap.add_argument("--pred-dir", default=PRED_DIR)
    ap.add_argument("--benchmark", choices=["topix", "sector"], default="topix")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--min-train-months", type=int, default=48)
    ap.add_argument("--test-months", type=int, default=12)
    ap.add_argument("--step-months", type=int, default=12)
    ap.add_argument("--min-test-rows", type=int, default=200)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    ap.add_argument("--cost-bps", type=float, default=backtest.DEFAULT_COST_BPS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--out-json", default=OUT_JSON)
    args = ap.parse_args(argv)

    fold_kw = dict(min_train_months=args.min_train_months,
                   test_months=args.test_months, step_months=args.step_months,
                   max_horizon=20, min_test_rows=args.min_test_rows)
    common = dict(benchmark=args.benchmark, horizon=args.horizon, fold_kw=fold_kw)
    if args.stage in ("baselines", "all"):
        run_baselines(args.data_dir, args.pred_dir, **common)
    if args.stage in ("train", "all"):
        run_gnn(args.data_dir, args.pred_dir, variants=args.variants,
                seeds=args.seeds, cfg=TrainConfig(max_epochs=args.max_epochs),
                **common)
    if args.stage in ("report", "all"):
        d = report(args.data_dir, args.pred_dir, cost_bps=args.cost_bps,
                   n_boot=args.n_boot, **common)
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(_sanitize(d), fh, ensure_ascii=False, indent=1)
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(d))
        print(f"[gnn] 書き出し {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
