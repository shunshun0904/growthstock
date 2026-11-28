import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from export_valuations import DAYS, recent_candidates
PRED = {'asOf':'2026-11-02','dates':['2026-10-26','2026-10-27','2026-10-28','2026-10-29','2026-10-30','2026-11-02'],
        'candidates':[{'jqCode':'Z','date':'2026-10-26'},{'jqCode':'A','date':'2026-10-28'},{'jqCode':'B','date':'2026-10-29'},
                      {'jqCode':'B','date':'2026-10-30'},{'jqCode':'C','date':'2026-10-29'}]}
class CandidateScope(unittest.TestCase):
    def test_last_five_inference_days_count_empty_days_and_deduplicate_latest(self):
        # 2026-11-21 から直近5営業日（運用者の選択「直近5営業日の候補すべて」「タブも5営業日に広げる」）。候補ゼロの日も数える
        self.assertEqual(DAYS, 5)
        days,c = recent_candidates(PRED)
        self.assertEqual(days,['2026-10-27','2026-10-28','2026-10-29','2026-10-30','2026-11-02'])
        self.assertEqual(set(c),{'A','B','C'})            # 6日前の Z は入らない
        self.assertEqual(c['B']['date'],'2026-10-30')     # 同じ銘柄は最新の推論日で1件
    def test_window_is_a_parameter(self):
        days,c = recent_candidates(PRED, n_days=3)       # 2026-11-20 までの3営業日と同じ選び方
        self.assertEqual(days,['2026-10-29','2026-10-30','2026-11-02'])
        self.assertEqual(set(c),{'B','C'})
    def test_dates_after_as_of_are_ignored_and_candidates_fall_back(self):
        days,_ = recent_candidates({'asOf':'2026-10-29','dates':['2026-10-28','2026-10-29','2026-10-30'],'candidates':[]})
        self.assertEqual(days,['2026-10-28','2026-10-29'])
        days,c = recent_candidates({'candidates':[{'jqCode':'A','date':'2026-10-01'},{'jqCode':'B','date':'2026-10-02'}]})
        self.assertEqual(days,['2026-10-01','2026-10-02'])
        self.assertEqual(set(c),{'A','B'})
