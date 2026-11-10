"""Shared historical/live financial input calculation (E121 C arm)."""
import io
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
import theoretical_value as V

from financial_schema import COLUMNS
YIELD_URL = 'https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv'
# 財務省の国債金利は2つのファイルに分かれる: jgbcme_all.csv（過去分。前月末まで）と jgbcme.csv（今月分）。過去分だけを
# 読んでいたので、月の初めから数えて国債10年金利の基準日が7日を超えて古くなり（riskfree が7日超を欠測にする）、理論株価
# （RIM・配当割引法）と予測の金利の列が月の8日ごろから月末まで欠けていた（2026-11-10 に運用者が「理論株価の算出が失敗して
# いる」と指摘して判明。11/9 の基準日は 10/30）。今月分も取ってつなぐ。今月分が取れない日は過去分だけで続け、そう書く
CURRENT_YIELD_URL = 'https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv'


def _fetch(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def _parse(content):
    frame = pd.read_csv(io.BytesIO(content), skiprows=1, na_values=['-'])
    frame.columns = [str(c).strip() for c in frame.columns]
    if not {'Date', '10Y'} <= set(frame):
        raise ValueError('Unexpected MOF yield format')
    frame['Date'] = pd.to_datetime(frame.Date.astype(str).str.strip(), errors='coerce')
    return frame.dropna(subset=['Date'])


def merge_yields(history, current=None):
    """過去分と今月分をつなぐ（同じ日は今月分を採る）。日付の順・重複なし。"""
    parts = [history] + ([current] if current is not None and len(current) else [])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)
    return out


def yields(data_dir, refresh=True):
    path = Path(data_dir) / 'jgb_yields.csv'
    if refresh or not path.exists():
        history = _parse(_fetch(YIELD_URL))
        try:
            current = _parse(_fetch(CURRENT_YIELD_URL))
        except Exception as exc:  # 今月分が取れなくても過去分で続ける（金利が7日超古ければ riskfree が欠測にする）
            print(f'[yields] 今月分（jgbcme.csv）を取れなかった: {type(exc).__name__}。過去分だけで続ける', flush=True)
            current = None
        merged = merge_yields(history, current)
        tmp = path.with_suffix('.tmp')
        body = merged.assign(Date=merged.Date.dt.strftime('%Y-%m-%d')).to_csv(index=False)
        tmp.write_text('MOF JGB yields (jgbcme_all.csv + jgbcme.csv)\n' + body)
        tmp.replace(path)
        print(f'[yields] 国債金利 {len(merged):,}日・最新 {merged.Date.max().date()}（過去分 〜{history.Date.max().date()}・'
              f'今月分 {0 if current is None else len(current)}日）', flush=True)
    out = pd.read_csv(path, skiprows=1, na_values=['-'])
    out['Date'] = pd.to_datetime(out.Date)
    return out

def calculate(keys, prices, fins, market, rates):
    keys = keys.reset_index(drop=True).copy()
    keys['Date'] = pd.to_datetime(keys.Date)
    keys['Code'] = keys.Code.astype(str)
    prices = prices.copy(); prices['Date'] = pd.to_datetime(prices.Date); prices['Code'] = prices.Code.astype(str)
    fins = fins.copy(); fins['Code'] = fins.Code.astype(str)
    market = market.copy(); market['Date'] = pd.to_datetime(market.Date)
    rates = rates.copy(); rates['Date'] = pd.to_datetime(rates.Date)
    parts = [keys, V.beta(keys, prices, market, rates), V.riskfree(keys.Date, rates),
             V.financial_inputs(keys, fins, prices)]
    out = V.valuations(pd.concat(parts, axis=1))
    for method in ['ri', 'ddm']:
        out['missing_tv_' + method] = out['tv_' + method + '_mean_rel'].isna().astype(float)
    for col in ['tv_fin_source_date', 'tv_div_source_date', 'tv_rf_date']:
        if not (out[col].isna() | out[col].lt(out.Date)).all():
            raise ValueError('Unreleased financial input: ' + col)
    if np.isinf(out[COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError('Infinite financial input')
    return out
