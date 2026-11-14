#!/usr/bin/env python3
"""Read-only collection for the approved mean-entry-price experiment.

Retains original margin quantities, contract values and publication dates. It
does not update releases, the production data store, predictions or Pages.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'research'))
from jquants_data_fetcher import JQuantsClient, resolve_api_key
from trading_calendar import parse_rows

JST = dt.timezone(dt.timedelta(hours=9))
MARGIN_FIELDS = {'Date', 'Code', 'PubDate', 'LongVol', 'ShrtVol', 'LongVal', 'ShrtVal'}


class CollectionError(ValueError):
    """A safe-to-log source-contract failure; contains no credential values."""


def verify_rows(rows, day, required, label):
    seen = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise CollectionError(f'{label} {day}: missing fields {sorted(missing)}')
        if str(row['Date'])[:10] != day:
            raise CollectionError(f'{label} {day}: response reference date mismatch')
        code = str(row['Code'])
        if code in seen:
            raise CollectionError(f'{label} {day}: duplicate issue/date')
        seen.add(code)


def collect_history(client, start, end, output_dir):
    import pandas as pd

    first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if first < dt.date(2026, 9, 25) or first > last:
        raise CollectionError('contract-value history requires 2026-09-25 <= start <= end')
    destination = pathlib.Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    calendar_start = dt.date(first.year - 1, 1, 1).isoformat()
    calendar_rows = client.get_paginated('/markets/calendar', {'from': calendar_start, 'to': end})
    days = [x.isoformat() for x in parse_rows(calendar_rows) if first <= x <= last]
    if not days:
        raise CollectionError('no trading calendar dates available')
    margins, masters, empty_days, day_counts = [], [], [], {}
    for day in days:
        rows = client.get_paginated('/markets/margin-interest', {'date': day})
        verify_rows(rows, day, MARGIN_FIELDS, 'margin')
        day_counts[day] = len(rows)
        if rows:
            margins.extend(rows)
        else:
            empty_days.append(day)
        print(json.dumps({'endpoint': 'margin', 'date': day, 'rows': len(rows)}), flush=True)
    if not margins:
        raise CollectionError('no margin history returned')
    observed_end = max(str(r['Date'])[:10] for r in margins)
    internal_missing = [d for d in empty_days if d < observed_end]
    if internal_missing:
        raise CollectionError(f'missing internal trading dates: {internal_missing}')
    for day in days:
        if day > observed_end:
            continue
        rows = client.get_paginated('/equities/master', {'date': day})
        verify_rows(rows, day, {'Date', 'Code'}, 'master')
        if not rows:
            raise CollectionError(f'master {day}: no rows')
        masters.extend(rows)
        print(json.dumps({'endpoint': 'master', 'date': day, 'rows': len(rows)}), flush=True)
    files = {}
    frames = {
        'margin_amount_history.parquet': pd.DataFrame(margins),
        'master_validation_dates.parquet': pd.DataFrame(masters),
        'validation_calendar.parquet': pd.DataFrame(calendar_rows),
    }
    for name, frame in frames.items():
        path = destination / name
        temporary = destination / (name + '.tmp')
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        files[name] = {'rows': len(frame), 'columns': list(frame.columns),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        'status': 'complete', 'retrieved_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'requested_start': start, 'requested_end': end, 'observed_end': observed_end,
        'margin_rows': len(margins), 'master_rows': len(masters), 'day_counts': day_counts,
        'trailing_unpublished_dates': [d for d in empty_days if d > observed_end],
        'files': files, 'production_outputs_modified': False,
    }
    (destination / 'collection_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2026-09-25')
    parser.add_argument('--end', default=dt.datetime.now(JST).date().isoformat())
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    try:
        # Suppress even the existing resolver's credential-shape diagnostics.
        with contextlib.redirect_stdout(io.StringIO()):
            key = resolve_api_key()
        result = collect_history(JQuantsClient(key), args.start, args.end, args.output_dir)
    except CollectionError as exc:
        print(json.dumps({'error_type': type(exc).__name__, 'error': str(exc)}), file=sys.stderr)
        return 1
    except Exception as exc:
        # Never forward arbitrary authentication/HTTP bodies into public Actions logs.
        print(json.dumps({'error_type': type(exc).__name__, 'error': 'collection failed; no credential details logged'}), file=sys.stderr)
        return 1
    print(json.dumps({'status': result['status'], 'margin_rows': result['margin_rows'],
                      'observed_end': result['observed_end']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
