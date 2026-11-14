#!/usr/bin/env python3
"""Run the locked price-band mean validation once into a new output directory.

No production modules, forecasts, releases or Pages files are written. The
selection checkpoint is persisted before any held-out RMSPE is calculated.
Raw contract-value data and per-record derived data stay in local artifacts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_band_validation import (estimate_means, point_in_time_shares, make_window,
                                   select_coefficient, split_dates, rmspe, paired_block_interval)
from trading_calendar import parse_rows

FIN_COLS = ['Code', 'DiscDate', 'DiscTime', 'DiscNo', 'CurPerEn', 'ShOutFY', 'TrShFY']
BAR_COLS = ['Code', 'Date', 'Vo', 'Va', 'AdjFactor']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def normalise(frame, date_column):
    d = frame.copy()
    d['Code'] = d.Code.astype(str)
    d[date_column] = pd.to_datetime(d[date_column], errors='raise').dt.normalize()
    return d


def validate_unique(frame, columns, label):
    if frame.duplicated(columns).any():
        raise ValueError(f'duplicate {label} on {columns}')


def extend_adjustment_history(bars, financials, calendar, requests, older_raw_dir):
    """Load older split chains only for carried share pairs that require them."""
    f = financials.copy()
    f['CurPerEn'] = pd.to_datetime(f.CurPerEn, errors='coerce')
    f = f[(f.ShOutFY > f.TrShFY) & (f.TrShFY >= 0) &
          (f.CurPerEn <= f.DiscDate) & f.Code.isin(requests.Code)]
    f = f.sort_values(['DiscDate', 'DiscTime', 'DiscNo'], kind='stable')
    window_start = calendar[calendar.get_indexer([requests.Date.min()])[0] - 59]
    prior = f[f.DiscDate <= window_start].groupby('Code').tail(1)
    subsequent = f[(f.DiscDate > window_start) & (f.DiscDate <= requests.Date.max())]
    needed = pd.concat([prior, subsequent])
    needed = needed[needed.CurPerEn < bars.Date.min()]
    if needed.empty:
        return bars, []
    codes = sorted(needed.Code.unique())
    earliest = int(needed.CurPerEn.min().year)
    paths = []
    extras = []
    for year in range(earliest, int(bars.Date.min().year)):
        path = Path(older_raw_dir) / f'bars_{year}.parquet'
        if not path.exists():
            raise ValueError(f'required old corporate-action chain unavailable: {path.name}')
        paths.append(path)
        extras.append(pd.read_parquet(path, columns=['Code', 'Date', 'AdjFactor'], filters=[('Code', 'in', codes)]))
    extra = normalise(pd.concat(extras, ignore_index=True), 'Date')
    print(json.dumps({'older_adjustment_codes': len(codes), 'older_adjustment_rows': len(extra),
                      'earliest_measurement_needed': str(needed.CurPerEn.min().date())}), flush=True)
    return pd.concat([extra, bars], ignore_index=True), paths


def ordinary_universe(margin, master):
    """Historical issue classification, including foreign common shares.

    J-Quants product codes 011 domestic, 021 foreign shares, 024 foreign
    depositary shares. The fifth local-code digit 0 denotes ordinary shares;
    other fifth digits denote preferred/subordinate issues (official FAQ).
    Classification is dated, never a current-listing-only screen.
    """
    validate_unique(margin, ['Code', 'Date'], 'margin')
    validate_unique(master, ['Code', 'Date'], 'master')
    required = {'ProdCat', 'CoName', 'Mkt'}
    if not required.issubset(master):
        raise ValueError('historical master lacks product/market classification')
    joined = margin.merge(master[['Code', 'Date', 'ProdCat', 'CoName', 'Mkt']],
                          on=['Code', 'Date'], how='left', validate='one_to_one')
    known = joined.ProdCat.notna() & ~joined.ProdCat.astype(str).isin(['null', '', 'None'])
    ordinary = known & joined.ProdCat.isin(['011', '021', '024']) & joined.Code.str.endswith('0')
    joined['universe_status'] = np.where(ordinary, 'ordinary',
                                         np.where(known, 'non_ordinary', 'unknown_historical_product_type'))
    return joined.loc[ordinary].copy(), joined.groupby('universe_status').size().rename('issue_dates').reset_index()


def scope_coverage(margin, master):
    """Account for listed ordinary issues with no returned margin record too."""
    ordinary = master[master.ProdCat.isin(['011', '021', '024']) & master.Code.str.endswith('0')]
    joined = ordinary[['Code', 'Date']].merge(margin[['Code', 'Date']], on=['Code', 'Date'],
                                             how='left', indicator=True, validate='one_to_one')
    table = joined.groupby('Date').agg(ordinary_listed_issue_dates=('Code', 'size'),
        margin_available_issue_dates=('_merge', lambda x: int((x == 'both').sum())))
    table['margin_unavailable_issue_dates'] = table.ordinary_listed_issue_dates - table.margin_available_issue_dates
    return table.reset_index()


def scope_summary(margin, master):
    ordinary = master[master.ProdCat.isin(['011', '021', '024']) & master.Code.str.endswith('0')]
    coverage = scope_coverage(margin, master)
    return {'ordinary_listed_codes_period': int(ordinary.Code.nunique()),
            'ordinary_listed_issue_dates': len(ordinary),
            'margin_unavailable_issue_dates': int(coverage.margin_unavailable_issue_dates.sum())}


def build_features(requests, bars, filings, calendar, lookback):
    """Accept only code/date requests, never contract values or target means."""
    if set(requests.columns) != {'Code', 'Date'}:
        raise ValueError('feature requests must contain code/date only')
    validate_unique(bars, ['Code', 'Date'], 'bars')
    days = pd.DatetimeIndex(sorted(calendar))
    days = days[(days >= bars.Date.min()) & (days <= requests.Date.max())]
    bar_groups = dict(tuple(bars.groupby('Code', sort=False)))
    financial_groups = dict(tuple(filings.groupby('Code', sort=False)))
    metadata, prices, volumes, turnovers = [], [], [], []
    for counter, (code, requested) in enumerate(requests.groupby('Code', sort=True), 1):
        source = bar_groups.get(code, pd.DataFrame(columns=BAR_COLS))
        b = source.set_index('Date').reindex(days)
        b['present'] = b.Code.notna()
        financials = financial_groups.get(code, pd.DataFrame(columns=FIN_COLS))
        shares = point_in_time_shares(days, b.AdjFactor.to_numpy(float), financials)
        b = b.join(shares)
        for day in sorted(requested.Date):
            pos = days.get_indexer([day])[0]
            if pos < 0:
                raise ValueError('requested reference date absent from exchange calendar')
            window = b.iloc[max(0, pos - lookback + 1):pos + 1]
            base, base_reason = make_window(window, lookback, require_shares=False)
            common, reason = make_window(window, lookback)
            row = {'Code': code, 'Date': day, 'feature_status': reason or 'available',
                   'baseline_available': base_reason is None, 'feature_id': -1,
                   'baseline_mean': (float(np.average(base['price'], weights=base['volume'])) if base else np.nan)}
            if common:
                row.update(feature_id=len(prices), price_min=common['price_min'], price_max=common['price_max'],
                           split_in_window=common['split_in_window'], zero_volume_days=common['zero_volume_days'],
                           latest_share_disclosure=str(window.disclosure_date.max().date()),
                           oldest_share_measurement=str(window.measurement_date.min().date()))
                prices.append(common['price'])
                volumes.append(common['volume'])
                turnovers.append(common['turnover'])
            elif reason == 'missing_point_in_time_net_shares':
                unavailable = window[window.share_reason != 'available'].share_reason.unique()
                row['share_detail'] = '|'.join(sorted(unavailable))
            metadata.append(row)
        if counter % 500 == 0:
            print(json.dumps({'feature_codes_completed': counter, 'feature_windows': len(prices)}), flush=True)
    if not prices:
        raise ValueError('no common computable feature windows')
    return pd.DataFrame(metadata), np.stack(prices), np.stack(volumes), np.stack(turnovers)


def make_records(universe, features):
    sides = []
    for side, quantity, amount in [('long', 'LongVol', 'LongVal'), ('short', 'ShrtVol', 'ShrtVal')]:
        d = universe[['Code', 'Date', 'PubDate', 'CoName', quantity, amount]].copy()
        d = d.rename(columns={quantity: 'quantity', amount: 'contract_value'})
        q, a = (pd.to_numeric(d[x], errors='coerce') for x in ['quantity', 'contract_value'])
        valid = np.isfinite(q) & np.isfinite(a) & (q > 0) & (a > 0)
        d['side'] = side
        d['target_status'] = np.where(valid, 'available',
                                       np.where(q == 0, 'zero_position_mean_undefined', 'missing_or_invalid_target'))
        d['actual_mean'] = np.where(valid, a / q.replace(0, np.nan), np.nan)
        sides.append(d)
    records = pd.concat(sides, ignore_index=True).merge(features, on=['Code', 'Date'], validate='many_to_one')
    records['included'] = (records.target_status == 'available') & (records.feature_status == 'available')
    records['exclusion_reason'] = np.where(records.target_status != 'available', records.target_status, records.feature_status)
    return records


def score_table(records):
    summaries, daily = [], []
    for phase, phase_rows in records.groupby('phase', sort=True):
        for side in ['all', 'long', 'short']:
            d = phase_rows if side == 'all' else phase_rows[phase_rows.side == side]
            if d.empty:
                continue
            for method in ['baseline', 'candidate']:
                predicted = d[f'{method}_mean']
                relative = (predicted - d.actual_mean) / d.actual_mean
                summaries.append({'phase': phase, 'side': side, 'method': method, 'records': len(d),
                                  'codes': d.Code.nunique(), 'dates': d.Date.nunique(),
                                  'rmspe_percent': rmspe(d.actual_mean, predicted),
                                  'mape_percent': float(100 * relative.abs().mean()),
                                  'signed_mean_percentage_error': float(100 * relative.mean()),
                                  'mae_yen': float((predicted - d.actual_mean).abs().mean())})
            for day, group in d.groupby('Date'):
                e0 = ((group.baseline_mean - group.actual_mean) / group.actual_mean) ** 2
                e1 = ((group.candidate_mean - group.actual_mean) / group.actual_mean) ** 2
                daily.append({'phase': phase, 'side': side, 'Date': day.strftime('%Y-%m-%d'), 'n': len(group),
                              'baseline_sse': float(e0.sum()), 'candidate_sse': float(e1.sum()),
                              'baseline_rmspe': float(100 * np.sqrt(e0.mean())),
                              'candidate_rmspe': float(100 * np.sqrt(e1.mean()))})
    return pd.DataFrame(summaries), pd.DataFrame(daily)


def run(config_path, raw_dir, older_raw_dir, source_dir, output_dir):
    config_path, raw_dir, older_raw_dir, source_dir, output_dir = map(Path,
        [config_path, raw_dir, older_raw_dir, source_dir, output_dir])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('use a new output directory: final evaluation is not automatically repeated')
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    if not config.get('execution_authorized'):
        raise ValueError('experiment execution is not authorised')
    margin_path, master_path, calendar_path = [source_dir / n for n in
        ['margin_amount_history.parquet', 'master_validation_dates.parquet', 'validation_calendar.parquet']]
    margin = normalise(pd.read_parquet(margin_path), 'Date')
    master = normalise(pd.read_parquet(master_path), 'Date')
    source_manifest = json.loads((source_dir / 'collection_manifest.json').read_text())
    margin = margin[margin.Date >= config['start_date']]
    universe, universe_counts = ordinary_universe(margin, master)
    universe_counts.to_csv(output_dir / 'universe.csv', index=False)
    scope_coverage(margin, master).to_csv(output_dir / 'scope_coverage.csv', index=False)
    write_json(output_dir / 'scope_summary.json', scope_summary(margin, master))
    all_dates = sorted(margin.Date.unique())
    train_dates, test_dates = split_dates(all_dates, config['tuning_date_fraction'])
    old_calendar_path = older_raw_dir / 'calendar.parquet'
    calendar_rows = pd.concat([pd.read_parquet(old_calendar_path), pd.read_parquet(calendar_path)], ignore_index=True)
    calendar_rows['Date'] = pd.to_datetime(calendar_rows.Date).dt.strftime('%Y-%m-%d')
    calendar_rows = calendar_rows.drop_duplicates('Date', keep='last')
    calendar = pd.DatetimeIndex(parse_rows(calendar_rows.to_dict('records')))
    bar_paths = sorted(raw_dir.glob('bars_*.parquet'))
    fin_paths = sorted(p for p in older_raw_dir.glob('fins_*.parquet') if not (raw_dir / p.name).exists()) + sorted(raw_dir.glob('fins_*.parquet'))
    bars = normalise(pd.concat([pd.read_parquet(p, columns=BAR_COLS) for p in bar_paths], ignore_index=True), 'Date')
    financials = normalise(pd.concat([pd.read_parquet(p, columns=FIN_COLS) for p in fin_paths], ignore_index=True), 'DiscDate')
    requests = universe[['Code', 'Date']].drop_duplicates()
    bars, factor_paths = extend_adjustment_history(bars, financials, calendar, requests, older_raw_dir)
    # Hash the exact inputs before coefficient selection or final scoring.
    input_paths = [config_path, margin_path, master_path, calendar_path, old_calendar_path] + bar_paths + fin_paths + factor_paths
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    write_json(output_dir / 'source_manifest.json', {'frozen_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'input_sha256': hashes, 'collection': source_manifest,
               'note': 'retrospective current-vintage API values; historical corrections may be reflected'})
    write_json(output_dir / 'config.json', config)
    features, p, v, r = build_features(requests, bars, financials,
                                      calendar, config['lookback_trading_days'])
    features.to_parquet(output_dir / 'feature_audit.parquet', index=False)
    records = make_records(universe, features)
    records['phase'] = np.where(records.Date.isin(train_dates), 'tuning', 'final')
    records.groupby(['phase', 'side', 'exclusion_reason'], dropna=False).size().rename('records').reset_index().to_csv(output_dir / 'coverage.csv', index=False)
    records.groupby(['Date', 'side', 'target_status', 'feature_status', 'baseline_available'], dropna=False).size().rename('records').reset_index().to_csv(output_dir / 'coverage_daily.csv', index=False)
    records.to_parquet(output_dir / 'record_coverage_audit.parquet', index=False)
    common = records[records.included].copy()
    if set(common.Date.unique()) != set(all_dates):
        raise ValueError('one or more pre-split dates has no common computable records')
    training = common[common.phase == 'tuning']
    training_feature_ids = np.unique(training.feature_id.to_numpy(int))
    grid_predictions = estimate_means(p[training_feature_ids], v[training_feature_ids], r[training_feature_ids], config['coefficient_grid'])
    map_to_train = np.searchsorted(training_feature_ids, training.feature_id.to_numpy(int))
    chosen, curve, ties = select_coefficient(training.actual_mean.to_numpy(), grid_predictions[map_to_train], config['coefficient_grid'])
    curve.to_csv(output_dir / 'training_curve.csv', index=False)
    checkpoint = {'chosen_coefficient': chosen, 'exact_ties': ties,
                  'at_search_boundary': chosen in [min(config['coefficient_grid']), max(config['coefficient_grid'])],
                  'tuning_start': str(pd.Timestamp(train_dates[0]).date()), 'tuning_end': str(pd.Timestamp(train_dates[-1]).date()),
                  'final_start': str(pd.Timestamp(test_dates[0]).date()), 'final_end': str(pd.Timestamp(test_dates[-1]).date()),
                  'tuning_dates': len(train_dates), 'final_dates': len(test_dates),
                  'tuning_records': len(training), 'persisted_before_final_scoring': True,
                  'selected_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(output_dir / 'selection.json', checkpoint)
    print(json.dumps({'selection_complete': checkpoint}), flush=True)
    # Single locked coefficient only: final targets never enter the grid search.
    chosen_predictions = estimate_means(p, v, r, [chosen])[:, 0]
    common['candidate_mean'] = chosen_predictions[common.feature_id.to_numpy(int)]
    common['outside_60_day_price_support'] = (common.actual_mean < common.price_min) | (common.actual_mean > common.price_max)
    summaries, daily = score_table(common)
    summaries.to_csv(output_dir / 'summary.csv', index=False)
    daily.to_csv(output_dir / 'daily_metrics.csv', index=False)
    common.to_parquet(output_dir / 'evaluation_records.parquet', index=False)
    final_daily = daily[(daily.phase == 'final') & (daily.side == 'all')]
    boot_config = config['bootstrap']
    interval = paired_block_interval(final_daily, boot_config['block_trading_days'], boot_config['replicates'],
                                     boot_config['seed'], boot_config['coverage'])
    draws = interval.pop('resampled_differences', [])
    pd.DataFrame({'candidate_minus_baseline_rmspe_pp': draws}).to_csv(output_dir / 'bootstrap.csv', index=False)
    valid_targets = records[records.target_status == 'available']
    result = {'status': 'completed', 'selection': checkpoint, 'bootstrap': interval,
              'all_reference_dates': len(all_dates), 'ordinary_codes': int(universe.Code.nunique()),
              'ordinary_issue_dates': len(universe), 'potential_side_records': len(records),
              'valid_target_records': len(valid_targets), 'common_records': len(common),
              'coverage_among_valid_targets_percent': float(100 * len(common) / len(valid_targets)),
              'baseline_computable_valid_target_records': int(valid_targets.baseline_available.sum()),
              'common_codes': int(common.Code.nunique()),
              'outside_60_day_price_support_records': int(common.outside_60_day_price_support.sum()),
              'split_window_records': int(common.split_in_window.sum()),
              'metrics': summaries.to_dict('records'),
              'distribution_shape_ground_truth_obtained': False, 'production_modified': False,
              'completed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(output_dir / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'metrics'}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('price_band_validation_config.json')))
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--older-raw-dir', required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    run(args.config, args.raw_dir, args.older_raw_dir, args.source_dir, args.output_dir)


if __name__ == '__main__':
    main()
