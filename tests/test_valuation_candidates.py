import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from export_valuations import recent_candidates
class CandidateScope(unittest.TestCase):
    def test_last_three_inference_days_count_empty_days_and_deduplicate_latest(self):
        days,c = recent_candidates({'asOf':'2026-11-02','dates':['2026-10-27','2026-10-28','2026-10-29','2026-10-30','2026-11-02'],
          'candidates':[{'jqCode':'A','date':'2026-10-28'},{'jqCode':'B','date':'2026-10-29'},{'jqCode':'B','date':'2026-10-30'},{'jqCode':'C','date':'2026-10-29'}]})
        self.assertEqual(days,['2026-10-29','2026-10-30','2026-11-02'])
        self.assertEqual(set(c),{'B','C'})
        self.assertEqual(c['B']['date'],'2026-10-30')
