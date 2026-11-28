#!/usr/bin/env python3
"""RIM / DDM valuations for ALL unique candidates in the last five inference days.

2026-11-21 に直近3営業日から5営業日に広げた（運用者の依頼「サマリー」の割安度を「直近5日の予測対象の全銘柄内での
相対評価」にするため → 選択「直近5営業日の候補すべて」「（理論株価の）タブも5営業日に広げる」。
docs/MODEL_ADOPTION_RULES.md §100）。
"""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import build_dataset as B
import financial_inputs as FI
import theoretical_value as V
from valuation_display import price_keys, latest_financials, unavailable_reason, rim_with_provider_inputs, rim_with_loss_recovery

METHODS = {
 'ri': ('RIM（残余利益）', 'ROEを5・10年で資本コストへ収束。ERP3〜7%、黒字時の配当性向25〜75%。赤字中は無配・損失を純資産から全額控除。'),
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

#: 理論株価を出す推論日の数（予測ファイルの直近の推論日。候補ゼロの日も1日と数える）。2026-11-21 に 3 → 5
DAYS = 5


def recent_candidates(pred, n_days=DAYS):
    days = sorted({str(d) for d in pred.get('dates', []) if str(d) <= pred.get('asOf', '9999')})[-n_days:]
    if not days:
        days = sorted({str(c['date']) for c in pred.get('candidates', [])})[-n_days:]
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
        # A new stock bar can precede publication of TOPIX. Align both inputs.
        market = B.load_parts('topix', str(raw))
        keys, latest_prices = price_keys(prices, market)
        if len(keys):
            asof = keys.Date.max().strftime('%Y-%m-%d')
        fins = B.load_parts('fins', str(raw))
        financials = latest_financials(fins)
        tv = FI.calculate(keys, prices, fins, market, FI.yields(raw, refresh=False)) if len(keys) else pd.DataFrame()
        if len(tv):
            calendar = pd.DatetimeIndex(pd.to_datetime(market.Date).unique()).sort_values()
            indicator_dates = set()
            for date in tv.Date:
                pos = calendar.get_indexer([date])[0]
                if pos >= 0:
                    indicator_dates.update(calendar[max(0,pos-1):pos+1])
            indicator_parts = []
            for year in sorted({d.year for d in indicator_dates}):
                path = raw/f'valuation_{year}.parquet'
                if not path.exists():
                    continue
                part = pd.read_parquet(path,columns=['Code','Date','EPS','BPS'])
                part['Code'] = part.Code.astype(str)
                part['Date'] = pd.to_datetime(part.Date)
                indicator_parts.append(part[part.Code.isin(tv.Code) & part.Date.isin(indicator_dates)])
            indicators = pd.concat(indicator_parts,ignore_index=True) if indicator_parts else pd.DataFrame()
            tv = rim_with_loss_recovery(rim_with_provider_inputs(tv, prices, indicators, market))
        by_code = {r.Code:r for _,r in tv.iterrows()}
        for code,c in sorted(candidates.items()):
            r = by_code.get(code)
            methods = []
            for name,(label,note) in METHODS.items():
                values = {s:r['tv_'+name+'_'+s] if r is not None else np.nan for s in V.STATS}
                main = values['median']
                available = pd.notna(main)
                methods.append({'id':name,'label':label,'note':note,'kind':'price','available':available,
                    'reason':None if available else unavailable_reason(name, r, financials),
                    **values,'deviation':main/r.close_raw-1 if available else None,
                    'inputSource':r.get('ri_input_source') if name=='ri' and r is not None else None,
                    'inputDate':r.get('ri_input_date') if name=='ri' and r is not None else None,
                    'recoveryAssumed':bool(r.get('ri_recovery_assumed',False)) if name=='ri' and r is not None else False,
                    'validFraction':r.get('tv_'+name+'_valid_fraction') if r is not None else None,
                    'eps':r.tv_eps if name=='ri' and r is not None else None,
                    'bps':r.tv_bps if name=='ri' and r is not None else None})
            stocks.append({'code':code,'name':c.get('name',c.get('code',code)),
                'inferenceDate':c['date'],'date':r.Date if r is not None else None,
                'price':r.close_raw if r is not None else None,
                'latestStockPriceDate':latest_prices.loc[code,'Date'] if code in latest_prices.index else None,
                'priceFallback':bool(r is not None and code in latest_prices.index and r.Date < latest_prices.loc[code,'Date']),
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
