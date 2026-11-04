"""Shared historical/live financial input calculation (E121 C arm)."""
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
import theoretical_value as V

from financial_schema import COLUMNS
YIELD_URL = 'https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv'

def yields(data_dir, refresh=True):
    path = Path(data_dir) / 'jgb_yields.csv'
    if refresh or not path.exists():
        with urlopen(YIELD_URL, timeout=60) as response:
            content = response.read()
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(content)
        check = pd.read_csv(tmp, skiprows=1, na_values=['-'])
        if not {'Date', '10Y'} <= set(check):
            raise ValueError('Unexpected MOF yield format')
        tmp.replace(path)
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
