"""Public catalog covers codes outside today's detailed-stock snapshot."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_supply_demand_catalog import build_catalog, collect_margin_rows


def quotes(code):
    return [dict(Code=code, Date=d, C=p, AdjC=p, Vo=100, Va=100*p, AdjFactor=1)
            for d, p in [('2026-11-11', 100), ('2026-11-12', 120), ('2026-11-13', 130)]]


def balance(code, **updates):
    row = dict(Code=code, Date='2026-11-12', PubDate='2026-11-13',
               LongVol=1000, LongVal=110000, ShrtVol=0, ShrtVal=0)
    row.update(updates)
    return row


class CatalogTests(unittest.TestCase):
    def test_all_codes_are_generated_without_a_predictions_or_watchlist_filter(self):
        bars = pd.DataFrame(quotes('11110') + quotes('28940') + quotes('64670'))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'catalog'
            index = build_catalog(bars, [balance('11110'), balance('28940'), balance('64670')], out)
            self.assertEqual(set(index['stocks']), {'11110', '28940', '64670'})
            self.assertEqual(index['counts']['ok'], 3)
            d = json.loads((out / '28940.json').read_text())
            self.assertEqual(d['code'], '28940')
            self.assertEqual(d['supplyDemand']['long']['averagePrice'], 110)
            self.assertEqual(d['supplyDemand']['short']['shares'], 0)
            self.assertEqual(d['supplyDemand']['profiles']['60']['short']['status'], 'no_positions')

    def test_future_publication_is_not_used_and_missing_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'catalog'
            index = build_catalog(pd.DataFrame(quotes('28940')), [balance('28940', PubDate='2026-11-16')], out)
            d = json.loads((out / '28940.json').read_text())['supplyDemand']
            self.assertEqual(index['stocks']['28940']['status'], 'missing_balance')
            self.assertIsNone(d['long'])

    def test_missing_price_and_missing_margin_have_distinct_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = build_catalog(pd.DataFrame(quotes('28940')), [balance('64670')], Path(tmp) / 'catalog')
            self.assertEqual(index['stocks']['28940']['status'], 'missing_balance')
            self.assertEqual(index['stocks']['64670']['status'], 'missing_price')

    def test_duplicate_quotes_fail_before_publishing_an_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'catalog'
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                build_catalog(pd.DataFrame(quotes('28940') * 2), [balance('28940')], out)
            self.assertFalse((out / 'index.json').exists())

    def test_snapshot_collection_uses_published_dates_and_retains_contract_values(self):
        class Client:
            def get_paginated(self, endpoint, params):
                return [balance('28940')] if params['date'] == '2026-11-12' else []
        rows = collect_margin_rows(Client(), ['2026-11-12', '2026-11-13'], '2026-11-13')
        self.assertEqual(rows[0]['LongVal'], 110000)
        self.assertEqual(rows[0]['PubDate'], '2026-11-13')

    def test_missing_contract_field_or_empty_history_is_a_collection_error(self):
        class Client:
            def get_paginated(self, endpoint, params):
                return [{'Code': '28940', 'Date': '2026-11-12'}]
        with self.assertRaisesRegex(ValueError, 'fields'):
            collect_margin_rows(Client(), ['2026-11-12'], '2026-11-13')
        class Empty:
            def get_paginated(self, endpoint, params):
                return []
        with self.assertRaisesRegex(ValueError, 'no published'):
            collect_margin_rows(Empty(), ['2026-11-12'], '2026-11-13')


if __name__ == '__main__':
    unittest.main()
