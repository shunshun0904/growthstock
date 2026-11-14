#!/usr/bin/env python3
"""Add display profiles to an existing public snapshot using cached daily bars.

By default no API calls; --refresh-margin explicitly fetches fresh margin rows.
No imputation of missing contract amounts. Requires pandas/pyarrow
from research/requirements.txt. Normal daily refresh uses the stdlib producer
directly; this command is only a reproducible initial backfill.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

from supply_demand import build_supply_demand


def backfill(dataset, rows_by_code, margin_by_code=None):
    observed = dataset['generatedAt'][:10]
    summary = dict(stocks=len(dataset['stocks']), available=0, missing=0, calibrated=0)
    for stock in dataset['stocks']:
        metrics = stock.get('metrics') or {}
        day = metrics.get('marginDate')
        code = stock.get('jqCode') or stock['code']
        if margin_by_code is not None:
            margins = margin_by_code.get(code, [])
        else:
            margins = ([dict(Date=day, LongVol=metrics.get('marginLong'), ShrtVol=metrics.get('marginShort'))]
                       if day else [])
        profile = build_supply_demand(rows_by_code.get(stock.get('jqCode') or stock['code'], []), margins,
                                      as_of=stock.get('asOf'),
                                      observed_on=observed if margin_by_code is None else None)
        if profile['currentPrice'] is not None and metrics.get('price') is not None:
            if abs(profile['currentPrice'] - metrics['price']) > max(.01, metrics['price'] * 1e-6):
                raise ValueError(f"Price basis mismatch for {stock['code']}; do not overwrite the public snapshot")
        profile['inputSource'] = ('jquants_margin_api_and_cached_daily_bars' if margin_by_code is not None
                                 else 'public_stocks_snapshot_and_cached_daily_bars')
        profile['snapshotGeneratedAt'] = dataset['generatedAt']
        stock['supplyDemand'] = profile
        summary['available' if profile['status'] == 'ok' else 'missing'] += 1
        summary['calibrated'] += int(any(profile.get(s, {}).get('averagePrice') is not None
                                       for s in ('long', 'short') if profile.get(s)))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stocks', default='public/data/stocks.json')
    parser.add_argument('--bars-dir', required=True, help='Directory containing bars_YYYY.parquet')
    parser.add_argument('--output', required=True)
    parser.add_argument('--refresh-margin', action='store_true',
                        help='Fetch margin rows with existing J-Quants credentials; do not update other metrics')
    args = parser.parse_args()
    import pandas as pd
    import pyarrow.parquet as pq
    dataset = json.loads(Path(args.stocks).read_text())
    codes = [s.get('jqCode') or s['code'] for s in dataset['stocks']]
    tables = []
    for file in sorted(Path(args.bars_dir).glob('bars_*.parquet')):
        tables.append(pq.read_table(file, filters=[('Code', 'in', codes)]).to_pandas())
    if not tables:
        raise ValueError('No cached daily bars found')
    bars = pd.concat(tables).drop_duplicates(['Code', 'Date'], keep='last')
    bars['Date'] = pd.to_datetime(bars.Date).dt.strftime('%Y-%m-%d')
    bars = bars.astype(object).where(pd.notna(bars), None)
    grouped = {code: group.sort_values('Date').to_dict('records') for code, group in bars.groupby('Code')}
    margins = None
    if args.refresh_margin:
        from jquants_data_fetcher import JQuantsClient, resolve_api_key
        if len(codes) > 100:
            raise ValueError('Manual margin refresh is limited to 100 detailed stocks')
        client = JQuantsClient(resolve_api_key(), pause=.6)
        margins = {}
        for stock in dataset['stocks']:
            end = dt.date.fromisoformat(stock['asOf'])
            code = stock.get('jqCode') or stock['code']
            margins[code] = client.margin_interest(code, (end - dt.timedelta(days=400)).isoformat(), end.isoformat())
            if not margins[code]:
                raise ValueError(f'No margin response for {code}; public data has not been overwritten')
    summary = backfill(dataset, grouped, margins)
    Path(args.output).write_text(json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
