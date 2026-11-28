"""Price-band estimates: conservation, publication timing, and missing-data semantics."""
import datetime as dt
import math
import os
import sys
import unittest
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from supply_demand import build_supply_demand


def quotes(prices=(80, 100, 120), volumes=None):
    return [dict(Date=f'2026-11-{i + 2:02}', C=p, H=p, L=p, Vo=v, Va=p*v,
                 AdjC=p, AdjVo=v) for i, (p, v) in enumerate(
                     zip(prices, volumes or [100] * len(prices)))]


def balance(**extra):
    return dict(Date='2026-11-04', PubDate='2026-11-04', LongVol=1000,
                ShrtVol=500, LongVal=110000, ShrtVal=45000, **extra)


class SupplyDemandTests(unittest.TestCase):
    def test_calibration_conserves_shares_and_contract_amount(self):
        p = build_supply_demand(quotes(), [balance()])
        self.assertEqual(p['balance']['publishedOn'], '2026-11-04')
        for side, amount, qty in [('long', 110000, 1000), ('short', 45000, 500)]:
            s = p['profiles']['60'][side]
            self.assertEqual(s['method'], 'mean_calibrated')
            self.assertAlmostEqual(sum(b['shares'] for b in s['bins']), qty)
            self.assertAlmostEqual(sum(b['shares'] * b['price'] for b in s['bins']), amount, places=5)

    def test_reference_pnl_has_opposite_signs_for_short_and_long(self):
        p = build_supply_demand(quotes(), [balance()])
        self.assertAlmostEqual(p['long']['referencePnlRate'], 120 / 110 - 1)
        self.assertAlmostEqual(p['short']['referencePnlRate'], 1 - 120 / 90)
        self.assertEqual(p['long']['averagePriceKind'], 'contract_amount')

    def test_missing_amount_is_explicitly_an_uncalibrated_estimate(self):
        row = balance(); row['LongVal'] = None
        p = build_supply_demand(quotes(), [row])
        self.assertIsNone(p['long']['contractAmount'])
        self.assertIsNone(p['long']['averagePrice'])
        self.assertIsNone(p['long']['referencePnlRate'])
        s = p['profiles']['60']['long']
        self.assertEqual(s['method'], 'volume_prior')
        self.assertAlmostEqual(s['estimatedAveragePrice'], 100)
        self.assertAlmostEqual(sum(b['shares'] for b in s['bins']), 1000)

    def test_missing_position_is_not_zero_but_reported_zero_is_zero(self):
        row = balance(); row.update(LongVol=None, ShrtVol=0, ShrtVal=0)
        p = build_supply_demand(quotes(), [row])
        self.assertEqual(p['profiles']['60']['long']['status'], 'missing_quantity')
        self.assertEqual(p['profiles']['60']['short']['status'], 'no_positions')
        self.assertIsNone(p['short']['averagePrice'])
        self.assertIsNone(p['ratio'])

    def test_publication_date_blocks_future_data(self):
        future = balance(); future.update(PubDate='2026-11-05', LongVol=99999)
        older = balance(); older.update(Date='2026-11-02', PubDate='2026-11-03')
        p = build_supply_demand(quotes(), [future, older])
        self.assertEqual(p['long']['shares'], 1000)
        self.assertEqual(p['balance']['date'], '2026-11-02')

    def test_unknown_daily_publication_needs_observed_evidence(self):
        row = balance(); row.pop('PubDate')
        self.assertEqual(build_supply_demand(quotes(), [row])['status'], 'missing_balance')
        p = build_supply_demand(quotes(), [row], observed_on='2026-11-04')
        self.assertEqual(p['balance']['availabilityBasis'], 'public_snapshot')
        self.assertIsNone(p['balance']['publishedOn'])
        self.assertEqual(p['balance']['availableOn'], '2026-11-04')

    def test_legacy_weekly_publication_uses_exchange_calendar(self):
        q = [dict(Date='2026-09-23', C=100, Vo=100, Va=10000)]
        row = dict(Date='2026-09-18', LongVol=100, ShrtVol=50)
        days = [dt.date(2026, 9, 22), dt.date(2026, 9, 24)]
        p = build_supply_demand(q, [row], trading_days=days)
        self.assertEqual(p['status'], 'missing_balance')

    def test_split_between_balance_and_asof_adjusts_shares_not_amount(self):
        q = quotes((100, 50, 60))
        q[0].update(AdjC=50, AdjVo=200)
        row = balance(); row.update(Date='2026-11-02', LongVal=100000, ShrtVal=50000)
        p = build_supply_demand(q, [row])
        self.assertEqual(p['long']['shares'], 2000)
        self.assertEqual(p['long']['reportedShares'], 1000)
        self.assertEqual(p['long']['contractAmount'], 100000)
        self.assertEqual(p['long']['averagePrice'], 50)
        self.assertEqual(p['currentPrice'], 60)

    def test_asof_rebases_adjusted_quotes_and_discards_future_bars(self):
        q = quotes((100, 110, 60))
        q[0].update(AdjC=50, AdjVo=200)
        q[1].update(AdjC=55, AdjVo=200)
        row = balance(); row.update(Date='2026-11-03', PubDate='2026-11-03')
        p = build_supply_demand(q, [row], as_of='2026-11-03')
        self.assertEqual(p['currentPrice'], 110)
        self.assertEqual(p['long']['shares'], 1000)
        self.assertAlmostEqual(p['profiles']['60']['long']['estimatedAveragePrice'], 110)

    def test_known_mean_outside_observed_support_is_not_forced(self):
        row = balance(); row['LongVal'] = 200000
        p = build_supply_demand(quotes(), [row])
        self.assertEqual(p['long']['averagePrice'], 200)
        self.assertEqual(p['profiles']['60']['long']['status'], 'mean_outside_window')
        self.assertEqual(p['profiles']['60']['long']['bins'], [])

    def test_no_trades_does_not_invent_prior_and_missing_price_is_explicit(self):
        p = build_supply_demand(quotes(volumes=[0, 0, 0]), [balance()])
        self.assertEqual(p['profiles']['60']['long']['status'], 'missing_volume')
        self.assertEqual(build_supply_demand([], [balance()])['status'], 'missing_price')

    def test_nontraded_sessions_count_in_window_without_inventing_prices(self):
        q = quotes(); q[1].update(C=None, AdjC=None, Vo=0, Va=0, AdjVo=0)
        p = build_supply_demand(q, [balance()])
        self.assertEqual(p['profiles']['60']['sessions'], 3)
        self.assertEqual(p['profiles']['60']['tradedSessions'], 2)

    def test_profile_prior_ends_at_balance_date_not_after_it(self):
        row = balance(); row.update(Date='2026-11-03', LongVal=None)
        p = build_supply_demand(quotes((80, 100, 900)), [row])
        self.assertEqual(p['profiles']['60']['to'], '2026-11-03')
        self.assertAlmostEqual(p['profiles']['60']['long']['estimatedAveragePrice'], 90)

    def test_daily_fetch_reuses_exactly_the_existing_four_api_calls(self):
        from jquants_data_fetcher import build_stock

        class Client:
            def __init__(self): self.calls = []
            def equities_master(self, code):
                self.calls.append('master'); return {'CoName': '合成テスト株式会社'}
            def daily_bars(self, *args):
                self.calls.append('bars'); return quotes()
            def fin_summary(self, code):
                self.calls.append('fins'); return []
            def margin_interest(self, *args):
                self.calls.append('margin'); return [balance()]

        client = Client()
        stock = build_stock(client, '1234')
        self.assertEqual(client.calls, ['master', 'bars', 'fins', 'margin'])
        self.assertEqual(stock['metrics']['price'], 120)
        self.assertEqual(stock['history'][-1]['close'], 120)
        self.assertEqual(stock['supplyDemand']['long']['averagePrice'], 110)

    def test_backfill_preserves_every_existing_field_and_does_not_invent_amounts(self):
        from backfill_supply_demand import backfill
        dataset = {'generatedAt': '2026-11-04T20:00:00+00:00', 'stocks': [
            {'jqCode': '12340', 'code': '1234', 'asOf': '2026-11-04', 'history': [{'close': 120}],
             'metrics': {'price': 120, 'marginDate': '2026-11-03', 'marginLong': 1000, 'marginShort': 500}}]}
        original = deepcopy(dataset)
        summary = backfill(dataset, {'12340': quotes()})
        self.assertEqual(summary['available'], 1)
        self.assertIsNone(dataset['stocks'][0]['supplyDemand']['long']['contractAmount'])
        dataset['stocks'][0].pop('supplyDemand')
        self.assertEqual(dataset, original)

    def test_live_margin_backfill_uses_response_publication_not_snapshot_evidence(self):
        from backfill_supply_demand import backfill
        dataset = {'generatedAt': '2026-11-04T20:00:00+00:00', 'stocks': [
            {'jqCode': '12340', 'code': '1234', 'asOf': '2026-11-04',
             'metrics': {'price': 120}}]}
        backfill(dataset, {'12340': quotes()}, {'12340': [balance()]})
        p = dataset['stocks'][0]['supplyDemand']
        self.assertEqual(p['balance']['availabilityBasis'], 'published')
        self.assertEqual(p['long']['averagePrice'], 110)
        self.assertEqual(p['balance']['publishedOn'], '2026-11-04')


if __name__ == '__main__':
    unittest.main()
