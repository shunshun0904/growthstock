#!/usr/bin/env python3
"""
学習データの見本を Google スプレッドシートに書き出す（運用者の依頼 2026-10-17「学習データの一部を見たいです。ブラウザ上の
エディターとかで実際にテーブル形式で見たいです。サンプルは１０００サンプル程度で良いです」。置き場所は「いつものシートに
新しいタブ」、選び方は「直近の1000行」）。

学習データ = build_dataset.py（--keep-unlabeled なし）が作る research/_data/dataset.parquet。週次の再学習と同じ作り方なので、
次の週次の再学習が学習するのと同じ行と値になる。ラベルが確定している最新の日からさかのぼり、日をまたがずに --rows 行に
届くまでの日の行をすべて出す（1日の途中で切らない）。

  タブ「学習データの見本」     1行目に日本語名、2行目に列名。列は 日付・コード・銘柄名・ラベル（1.2σ・1.5σ・1.8σ）と、
                               学習に使う特徴量（features.DEFAULT_PRESET の239列、学習器に渡す並びのまま）。
                               値は学習器に渡す値のまま（欠測は空欄、無限大は inf）
  タブ「学習データの列の説明」 行の選び方・ラベルの定義・今のモデルの学習期間と、列ごとの日本語名・大区分・グループ・説明

どちらのタブも書くたびに中身を入れ替える（ほかのタブには触らない）。銘柄ごとの値は非公開のスプレッドシートにだけ書く。
Actions のログは公開なので、行数・日付の範囲・正例の数だけを出す（銘柄コード・銘柄名・値は出さない）。

  GOOGLE_SERVICE_ACCOUNT_JSON='{...}' GSHEET_ID=... python3 research/export_train_sample.py [--rows 1000]
  python3 research/export_train_sample.py --dry-run    # 書かずに件数だけ
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_dataset as B  # noqa: E402
import feature_dict as FD  # noqa: E402
import features as F  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
META_PATH = os.path.join(HERE, "model", "meta.json")

SAMPLE_TITLE = "学習データの見本"
NOTES_TITLE = "学習データの列の説明"
ROWS = 1000
#: 1回の書き込みで送る行数（Sheets API の1リクエストを小さく保つ）
CHUNK = 200
#: ラベルの列（列名, 到達しきい値の倍数）。1.2σ は学習データの label そのもの
LABELS = (("label", 1.2), ("label_k15", 1.5), ("label_k18", 1.8))
#: 先頭の列（列名, 日本語名）
ID_COLS = (("Date", "日付"), ("Code", "コード"), ("CoName", "銘柄名"))


def code4(code) -> str:
    """J-Quants の5桁のコード（末尾0）を、画面と同じ4桁に。それ以外はそのまま。"""
    s = str(code or "").strip()
    return s[:4] if len(s) == 5 and s.endswith("0") else s


def pick_recent(ds: pd.DataFrame, rows: int) -> pd.DataFrame:
    """ラベルが確定している最新の日からさかのぼり、rows 行に届くまでの日の行をすべて（日付・コードの順）。"""
    d = ds[ds["label"].notna()].copy()
    d["Date"] = pd.to_datetime(d["Date"])
    per_day = d.groupby("Date").size().sort_index(ascending=False)
    keep, total = [], 0
    for day, n in per_day.items():
        if total >= rows:
            break
        keep.append(day)
        total += int(n)
    out = d[d["Date"].isin(keep)].copy()
    out["Code"] = out["Code"].astype(str)
    return out.sort_values(["Date", "Code"]).reset_index(drop=True)


def attach_labels(sample: pd.DataFrame) -> pd.DataFrame:
    """1.5σ・1.8σ のラベルを、データセットに残る到達・終盤・トレンドの列から引き直す（ラベル違いのモデルと同じ）。"""
    out = sample.copy()
    for col, k in LABELS:
        if col != "label":
            out[col] = B.rise_label_for_k(out, k)
    return out


def cell(v):
    """シートに書く値。欠測は空欄、無限大は文字の inf、整数に見える値はそのまま数で。"""
    if v is None:
        return ""
    if isinstance(v, (pd.Timestamp, dt.date)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return ""
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return int(f) if f.is_integer() and abs(f) < 1e15 else f


def table(sample: pd.DataFrame, names: Dict[str, str], cols: Sequence[str]) -> List[List]:
    """見本のタブの中身（1行目に日本語名、2行目に列名）。"""
    label_ja = {"label": "ラベル 1.2σ", "label_k15": "ラベル 1.5σ", "label_k18": "ラベル 1.8σ"}
    head_ja = [ja for _, ja in ID_COLS] + [label_ja[c] for c, _ in LABELS] + [FD.name_ja(c) for c in cols]
    head_id = [c for c, _ in ID_COLS] + [c for c, _ in LABELS] + list(cols)
    out: List[List] = [head_ja, head_id]
    # itertuples は識別子にならない列名を位置の名前に置き換えるので使わない（列名のまま引く）
    for row in sample.to_dict("records"):
        code = str(row["Code"])
        out.append([cell(row["Date"]), code4(code), names.get(code, "")]
                   + [cell(row[c]) for c, _ in LABELS]
                   + [cell(row[c]) for c in cols])
    return out


def label_text(k: float) -> str:
    """ラベルの定義（build_dataset の DEFAULT_RISE を k だけ変えたもの。名前と、コードどおりの言い換え）。"""
    cfg = replace(B.DEFAULT_RISE, vol_norm_k=float(k))
    h = cfg.horizon
    return (f"{cfg.name}。予測日（78週高値を更新した日）の翌営業日の寄りを基準に、(1) 翌営業日から{h}営業日の終値の最大が"
            f" 基準から {k:.1f}σ 以上上がる（σ = 20日の日次リターンの標準偏差 × √{h}）、(2) {h}営業日後の直近"
            f"{cfg.end_window}営業日平均の終値が基準から (1) の上昇率の {cfg.end_ratio / cfg.threshold:.2f}倍以上、"
            f"(3) {h}営業日後に {cfg.trend_short}日移動平均 ≥ {cfg.trend_long}日移動平均、をすべて満たせば 1、"
            "どれかを満たさなければ 0")


def notes(sample: pd.DataFrame, cols: Sequence[str], rows: int, meta: Optional[dict], built_at: str) -> List[List]:
    """列の説明のタブの中身。上に行の選び方とラベルの定義、下に列ごとの説明。"""
    days = sorted(pd.to_datetime(sample["Date"]).dt.strftime("%Y-%m-%d").unique())
    lines: List[List] = [
        ["学習データの見本（research/export_train_sample.py）"],
        [f"書き出し: {built_at}。週次の再学習と同じ手順（build_dataset.py）で、この時点の生データから作った学習データ"],
        [f"行: ラベルが確定している最新の日からさかのぼり、日をまたがずに {rows}行に届くまで。"
         f"{days[0] if days else '-'} 〜 {days[-1] if days else '-'} の {len(days)}日・{len(sample):,}行（日付・コードの順）"],
    ]
    if meta:
        lines.append([f"今のモデル（{str(meta.get('trainedAt', ''))[:10]} の週次の再学習）が学習したのは "
                      f"{meta.get('trainFrom', '-')} 〜 {meta.get('trainTo', '-')} の {meta.get('nTrain', 0):,}行。"
                      "それより後の日の行は、次の週次の再学習から学習に入る"])
    lines += [
        ["値は学習器に渡す値のまま。空欄は欠測（学習器は欠測として扱う）、inf は無限大"],
        [],
        ["ラベル", "定義"],
    ]
    for c, k in LABELS:
        use = {"label": "1.2σ の4本（基準モデル = 1.2σ の LightGBM を含む）",
               "label_k15": "1.5σ の4本（今日の戦略の規則に使う3本を含む）",
               "label_k18": "1.8σ の4本"}[c]
        lines.append([c, f"{label_text(k)}。{use}の正解"])
    group_ja = {g: ja for ja, groups, _ in FD.MACRO for g in groups}
    lines += [[], ["列名", "日本語名", "大区分", "グループ", "説明"]]
    lines += [[c, ja, "", "", ""] for c, ja in ID_COLS]
    for c in cols:
        g = F.group_of(c)
        lines.append([c, FD.name_ja(c), group_ja.get(g, "その他"), g, FD.describe(c).replace("**", "")])
    return lines


def write_tab(book, title: str, values: List[List], freeze_rows: int = 0, freeze_cols: int = 0,
              bold_rows: int = 0, chunk: int = CHUNK):
    """タブの中身を入れ替える（無ければ作る）。大きさは中身に合わせる。"""
    import gspread

    nrows = max(len(values), 1)
    ncols = max([len(r) for r in values] + [1])
    try:
        ws = book.worksheet(title)
        ws.clear()
        ws.resize(rows=nrows, cols=ncols)
    except gspread.WorksheetNotFound:
        ws = book.add_worksheet(title=title, rows=nrows, cols=ncols)
    for i in range(0, len(values), chunk):
        ws.update(values[i:i + chunk], f"A{i + 1}", value_input_option="RAW")
    if freeze_rows or freeze_cols:
        ws.freeze(rows=freeze_rows, cols=freeze_cols)
    if bold_rows:
        import export_sheets as ES
        ws.format(f"A1:{ES.a1(ncols, bold_rows)}", {"textFormat": {"bold": True}})
    return ws


def load_names(data_dir: str) -> Dict[str, str]:
    """5桁のコード → 最新の銘柄名（master.parquet。無い銘柄は空欄）。"""
    path = os.path.join(data_dir, "master.parquet")
    if not os.path.exists(path):
        return {}
    m = pd.read_parquet(path)
    if not {"Code", "CoName"} <= set(m.columns):
        return {}
    m = m.dropna(subset=["Code"]).drop_duplicates("Code", keep="last")
    return {str(c): str(n) for c, n in zip(m["Code"], m["CoName"]) if isinstance(n, str)}


def load_meta(path: str = META_PATH) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def summary(sample: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """ログの行（行数・日付の範囲・正例の数だけ。銘柄・値は出さない）。"""
    days = pd.to_datetime(sample["Date"])
    lines = [f"[sample] {len(sample):,}行 / {days.dt.date.nunique()}日 "
             f"（{days.min().date() if len(sample) else '-'} 〜 {days.max().date() if len(sample) else '-'}）"
             f" / 列 {len(ID_COLS) + len(LABELS) + len(cols)}（特徴量 {len(cols)}）"]
    for c, _ in LABELS:
        v = pd.to_numeric(sample[c], errors="coerce")
        lines.append(f"[sample]   {c}: 正例 {int((v == 1).sum()):,} / 負例 {int((v == 0).sum()):,}"
                     f" / 判定できない {int(v.isna().sum()):,}")
    filled = sample[list(cols)].notna().to_numpy().mean() * 100 if len(sample) else 0.0
    lines.append(f"[sample]   特徴量の埋まり具合 {filled:.1f}%")
    return lines


def main(argv: Optional[Sequence[str]] = None, book=None) -> int:
    """book はテスト用（既定は GSHEET_ID のシートをサービスアカウントで開く）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--sheet-id", default=os.environ.get("GSHEET_ID", ""))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not 1 <= args.rows <= 20000:
        raise SystemExit(f"--rows は 1〜20000: {args.rows}")

    cols = F.columns(F.DEFAULT_PRESET)
    ds = pd.read_parquet(args.dataset)
    missing = [c for c in cols if c not in ds.columns]
    if missing:
        raise SystemExit(f"学習に使う特徴量がデータセットにありません: {len(missing)}列（例: {missing[:3]}）")
    sample = attach_labels(pick_recent(ds, args.rows))
    for line in summary(sample, cols):
        print(line)

    built_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data = table(sample, load_names(args.data_dir), cols)
    info = notes(sample, cols, args.rows, load_meta(), built_at)
    if args.dry_run:
        print(f"[dry-run] 書かない（見本 {len(data)}行 × {len(data[0])}列、説明 {len(info)}行）")
        return 0
    if book is None:
        if not args.sheet_id:
            raise SystemExit("GSHEET_ID が設定されていません（シートの URL の /d/ と /edit の間）")
        import export_sheets as ES
        book = ES.open_sheet(args.sheet_id)
    write_tab(book, SAMPLE_TITLE, data, freeze_rows=2, freeze_cols=3, bold_rows=2)
    write_tab(book, NOTES_TITLE, info, bold_rows=1)
    print(f"[sheet] タブ「{SAMPLE_TITLE}」に {len(data) - 2:,}行 × {len(data[0])}列、"
          f"「{NOTES_TITLE}」に {len(info)}行を書いた")
    return 0


if __name__ == "__main__":
    sys.exit(main())
