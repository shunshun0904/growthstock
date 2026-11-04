#!/usr/bin/env python3
"""RIM / DDM valuations for ALL unique candidates in the last three inference days."""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import build_dataset as B
import financial_inputs as FI
import theoretical_value as V

METHODS = {
 'ri': ('RIM（残余利益）', 'ROEを5・10年で資本コストへ収束。ERP3〜7%、配当性向25〜75%。'),
 'ddm': ('配当割引法', '実績年間配当、成長率0〜2%、ERP3〜7%。無配・赤字等は算出不可。'),
}

def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k,v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    if pd.isna(value): return None
    if isinstance(value, pd.Timestamp): return value.strftime('%Y-%m-%d')
    return value

def recent_candidates(pred):
    days = sorted({str(d) for d in pred.get('dates', []) if str(d) <= pred.get('asOf', '9999')})[-3:]
    if not days:
        days = sorted({str(c['date']) for c in pred.get('candidates', [])})[-3:]
    latest = {}
    for c in pred.get('candidates', []):
        if c['date'] not in days:
            continue
        code = str(c['jqCode'])
        if code not in latest or c['date'] > latest[code]['date']:
            latest[code] = c
    return days, latest


def export(raw, predictions, output):
    pred = json.loads(Path(predictions).read_text())
    days, candidates = recent_candidates(pred)
    stocks = []
    asof = pred.get('asOf')
    if candidates:
        bars = B.load_parts('bars', str(raw))
        bars['Code'] = bars.Code.astype(str)
        bars['Date'] = pd.to_datetime(bars.Date)
        bars = bars[bars.Code.isin(candidates)].sort_values(['Code','Date']).drop_duplicates(['Code','Date'])
        prices = bars[['Code','Date','C','AdjC']].rename(columns={'C':'close_raw','AdjC':'close'})
        prices['close'] = prices.close.fillna(prices.close_raw)
        prices = prices[prices.close_raw.gt(0) & prices.close.gt(0)]
        asof = prices.Date.max().strftime('%Y-%m-%d')
        # Evaluate every company at its latest available close, using only then-known inputs.
        keys = prices.groupby('Code').tail(1)[['Code','Date','close_raw']].reset_index(drop=True)
        fins = B.load_parts('fins', str(raw))
        market = B.load_parts('topix', str(raw))
        tv = FI.calculate(keys, prices, fins, market, FI.yields(raw, refresh=False))
        by_code = {r.Code:r for _,r in tv.iterrows()}
        for code,c in sorted(candidates.items()):
            r = by_code.get(code)
            methods = []
            for name,(label,note) in METHODS.items():
                values = {s:r['tv_'+name+'_'+s] if r is not None else np.nan for s in V.STATS}
                main = values['median']
                available = pd.notna(main)
                methods.append({'id':name,'label':label,'note':note,'kind':'price','available':available,
                    'reason':None if available else '必要データ・鮮度・計算条件を満たさない',
                    **values,'deviation':main/r.close_raw-1 if available else None})
            stocks.append({'code':code,'name':c.get('name',c.get('code',code)),
                'inferenceDate':c['date'],'date':r.Date if r is not None else None,
                'price':r.close_raw if r is not None else None,
                'financialDate':r.get('tv_fin_source_date') if r is not None else None,
                'yieldDate':r.get('tv_rf_date') if r is not None else None,'methods':methods})
    payload = clean({'schemaVersion':1,'asOf':asof,'inferenceDates':days,
        'generatedAt':datetime.now(timezone.utc).isoformat(),
        'warning':'理論株価のレンジは前提条件の感応度であり、将来株価の予測区間ではありません。',
        'stocks':stocks})
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False));tmp.replace(output)
    print('Valuations exported:',len(stocks),'unique stocks; inference dates',days,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path('research/_data'));p.add_argument('--predictions',default='public/data/predictions.json');p.add_argument('--output',type=Path,default=Path('public/data/valuations.json'))
    a=p.parse_args();export(a.data_dir,a.predictions,a.output)
