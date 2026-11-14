"""The research collector must retain cost amounts and publication vintages."""
import datetime as dt
import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'research'))
from collect_price_band_validation_data import collect_history, CollectionError


def margin(day, **updates):
    row = dict(Date=day, Code='13010', PubDate='2026-09-30', LongVol=100,
               ShrtVol=20, LongVal=12345, ShrtVal=3456)
    row.update(updates)
    return row


class FakeClient:
    def __init__(self, responses):
        self.responses = responses

    def get_paginated(self, path, params):
        if path == '/markets/calendar':
            return [dict(Date=d, HolDiv='1') for d in
                    ('2026-09-25', '2026-09-28', '2026-09-29')]
        if path == '/equities/master':
            return [dict(Date=params['date'], Code='13010', ProdCat='011', Mkt='0111')]
        return self.responses.get(params['date'], [])


class PriceBandCollectorTests(unittest.TestCase):
    def test_amounts_publication_dates_and_trailing_unpublished_day(self):
        client = FakeClient({d: [margin(d)] for d in ('2026-09-25', '2026-09-28')})
        with tempfile.TemporaryDirectory() as directory:
            result = collect_history(client, '2026-09-25', '2026-09-29', directory)
            frame = pd.read_parquet(pathlib.Path(directory) / 'margin_amount_history.parquet')
            self.assertEqual(frame.LongVal.tolist(), [12345, 12345])
            self.assertEqual(frame.ShrtVal.tolist(), [3456, 3456])
            self.assertEqual(frame.PubDate.tolist(), ['2026-09-30', '2026-09-30'])
            self.assertEqual(result['observed_end'], '2026-09-28')
            self.assertEqual(result['trailing_unpublished_dates'], ['2026-09-29'])
            self.assertEqual(result['margin_rows'], 2)
            self.assertEqual(json.loads((pathlib.Path(directory) / 'collection_manifest.json').read_text())['status'], 'complete')

    def test_internal_missing_business_day_does_not_silently_change_period(self):
        client = FakeClient({d: [margin(d)] for d in ('2026-09-25', '2026-09-29')})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CollectionError, 'missing.*2026-09-28'):
                collect_history(client, '2026-09-25', '2026-09-29', directory)

    def test_missing_amount_field_is_not_dropped_or_filled_with_zero(self):
        row = margin('2026-09-25')
        del row['LongVal']
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CollectionError, 'LongVal'):
                collect_history(FakeClient({'2026-09-25': [row]}), '2026-09-25', '2026-09-25', directory)

    def test_duplicate_issue_and_date_is_rejected(self):
        row = margin('2026-09-25')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CollectionError, 'duplicate'):
                collect_history(FakeClient({'2026-09-25': [row, row.copy()]}), '2026-09-25', '2026-09-25', directory)


if __name__ == '__main__':
    unittest.main()
