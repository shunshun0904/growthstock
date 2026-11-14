#!/usr/bin/env python3
"""Publish display-only per-security profiles, independent of prediction picks.

Called by the Pages build. Reads cached bars and published margin records; never
writes stock details, model inputs, predictions or releases. Output is atomic.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import re
import sys
import tempfile

from supply_demand import build_supply_demand, DAILY_MARGIN_START


def collect_margin_rows(client, trading_days, as_of, lookback=10):
    rows = []
    for day in sorted(set(trading_days), reverse=True)[:lookback]:
        batch = client.get_paginated('/markets/margin-interest', {'date': day})
        seen = set()
        for row in batch:
            fields = {'Date', 'Code', 'PubDate', 'LongVol', 'ShrtVol', 'LongVal', 'ShrtVal'}
            if fields - row.keys():
                raise ValueError('margin response missing required fields')
            if str(row['Date'])[:10] != day or str(row['Code']) in seen:
                raise ValueError('margin reference date mismatch or duplicate security')
            seen.add(str(row['Code']))
            if str(row.get('PubDate') or '')[:10] <= as_of and row.get('PubDate'):
                rows.append(row)
        print(json.dumps({'margin_date': day, 'rows': len(batch)}), flush=True)
    if not rows:
        raise ValueError('no published margin records available')
    return rows


def cached_margin_rows(directory):
    import pandas as pd
    paths = sorted(Path(directory).glob('margin_*.parquet'))
    if not paths:
        return []
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    frame['Date'] = pd.to_datetime(frame.Date).dt.strftime('%Y-%m-%d')
    known = frame.Date < DAILY_MARGIN_START
    if 'PubDate' in frame:
        known |= frame.PubDate.notna()
    return frame[known].astype(object).where(pd.notna(frame[known]), None).to_dict('records')


def collect_stale_margin_rows(client, last_closes, recent_rows):
    import pandas as pd
    earliest = min(str(r['Date'])[:10] for r in recent_rows)
    dates = pd.to_datetime(last_closes)
    stale = last_closes[(dates < earliest) & (dates >= DAILY_MARGIN_START)]
    rows = []
    for code, last in stale.items():
        end = pd.Timestamp(last).date()
        historical = client.get_paginated('/markets/margin-interest', {'code': str(code),
            'from': (end - dt.timedelta(days=400)).isoformat(), 'to': end.isoformat()})
        rows.extend(historical)
        print(json.dumps({'historical_margin_code': str(code), 'rows': len(historical)}), flush=True)
    return rows


def build_catalog(bars, margin_rows, output_dir, trading_days=None, as_of=None):
    import pandas as pd

    bars = bars.copy()
    bars['Date'] = pd.to_datetime(bars.Date, errors='raise').dt.strftime('%Y-%m-%d')
    bars['Code'] = bars.Code.astype(str)
    if bars.duplicated(['Code', 'Date']).any():
        raise ValueError('duplicate quote security/date')
    as_of = as_of or bars.Date.max()
    bars = bars[bars.Date <= as_of].sort_values(['Code', 'Date'])
    bars = bars.astype(object).where(pd.notna(bars), None)
    by_code = {c: g.to_dict('records') for c, g in bars.groupby('Code', sort=False)}
    margins = collections.defaultdict(list)
    for row in margin_rows:
        r = dict(row)
        for field in ['Date', 'PubDate']:
            if r.get(field) is not None:
                r[field] = str(r[field])[:10]
        margins[str(r['Code'])].append(r)
    codes = sorted(set(by_code) | set(margins))
    if not codes or any(not re.fullmatch(r'[0-9A-Za-z]{5}', c) for c in codes):
        raise ValueError('no valid security universe or unsafe security code')
    out = Path(output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('catalog destination must be new')
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    index = dict(schemaVersion=1, generatedAt=generated, asOf=as_of, stocks={}, counts={})
    counts = collections.Counter()
    with tempfile.TemporaryDirectory(prefix='supply-demand-', dir=out.parent) as tmp:
        staged = Path(tmp) / 'catalog'
        staged.mkdir()
        for i, code in enumerate(codes, 1):
            profile = build_supply_demand(by_code.get(code, []), margins.get(code, []),
                                          as_of=as_of, trading_days=trading_days)
            document = dict(code=code, generatedAt=generated, supplyDemand=profile)
            (staged / f'{code}.json').write_text(json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
            index['stocks'][code] = dict(status=profile['status'], asOf=profile['asOf'])
            counts[profile['status']] += 1
            if i % 500 == 0:
                print(json.dumps({'profile_codes_completed': i}), flush=True)
        index['counts'] = dict(counts)
        (staged / 'index.json').write_text(json.dumps(index, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
        staged.replace(out)
    return index


def main():
    import pandas as pd
    from jquants_data_fetcher import JQuantsClient, resolve_api_key

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--margin-snapshot', help='Offline validation only: previously collected parquet with publication dates/amounts')
    args = parser.parse_args()
    try:
        paths = sorted(Path(args.bars_dir).glob('bars_*.parquet'))
        if not paths:
            raise ValueError('no cached daily bars')
        columns = ['Code', 'Date', 'C', 'AdjC', 'Vo', 'Va', 'AdjFactor']
        bars = pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True)
        as_of = str(pd.to_datetime(bars.Date).max().date())
        calendar = None
        if args.margin_snapshot:
            margins = pd.read_parquet(args.margin_snapshot).to_dict('records')
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                key = resolve_api_key()
            client = JQuantsClient(key)
            calendar_rows = client.get_paginated('/markets/calendar', {'from': f'{int(as_of[:4])-1}-01-01', 'to': as_of})
            calendar = sorted(str(r['Date'])[:10] for r in calendar_rows if str(r.get('HolDiv')) in ['1', '2'])
            margins = collect_margin_rows(client, calendar, as_of)
            # Include historical issues too, even after they leave recent snapshots.
            # Never apply a balance published after the issue's last price date.
            last_closes = bars[pd.to_numeric(bars.C, errors='coerce') > 0].groupby('Code').Date.max()
            margins.extend(collect_stale_margin_rows(client, last_closes, margins))
            margins.extend(cached_margin_rows(args.bars_dir))
        index = build_catalog(bars, margins, args.output_dir, calendar, as_of)
        print(json.dumps({'status': 'complete', 'asOf': index['asOf'], 'codes': len(index['stocks']), 'counts': index['counts']}), flush=True)
        return 0
    except Exception as exc:
        # Authentication and upstream HTTP bodies must never reach public build logs.
        print(json.dumps({'error_type': type(exc).__name__, 'error': 'supply-demand catalog generation failed'}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
