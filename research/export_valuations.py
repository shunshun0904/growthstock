#!/usr/bin/env python3
"""Publish current candidate valuation comparisons; no valuation model features."""
from pathlib import Path
import argparse
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent / 'exp'))
import build_dataset as B
import capital_efficiency as CE
import financial_inputs as FI
import valuation_methods as M
from e122_valuation_feasibility import sectors, ev_fields

METHODS = {
 'ri': ('RIM（残余利益）', 'ROEを5・10年で資本コストへ収束。ERP3〜7%、配当性向25〜75%。'),
 'ddm': ('配当割引法', '実績年間配当、成長率0〜2%、ERP3〜7%。無配・赤字等は算出不可。'),
 'per': ('同業PER法', '同日・同業種の正の実績PER。他社10社以上、自社除外。成長率等の差は未調整。'),
 'pbr': ('同業PBR法', '同日・同業種の正のPBR。他社10社以上、自社除外。ROE等の差は未調整。'),
 'ev_ebitda': ('同業EV/EBITDA法（近似）', 'EBITDAは営業利益＋減価償却。負債の網羅性・リース等は未検証。'),
 'dcf_cash': ('DCF（現金設備投資）', '5年予測・324シナリオ。売上成長−5〜10%、税25〜35%。運転資本は売掛＋棚卸−買掛。'),
 'dcf_flex': ('DCF（設備投資補完）', '現金支出がない場合、開示設備投資額で補完。支出時点・範囲が一致しない場合がある。'),
 'mc_cash': ('確率的DCF（現金設備投資）', '1024試行。分布・独立性は仮定で未較正。実際の株価の信頼区間や上昇確率ではない。'),
 'mc_flex': ('確率的DCF（設備投資補完）', '1024試行。売上成長−5〜10%等の仮定。分布未較正、損失転落の表現が不十分。'),
 'reverse_cash': ('リバースDCF（現金設備投資）', '現在株価と整合する5年間の年率売上成長。ERP5%、税30%、成長探索−50〜50%。'),
 'reverse_flex': ('リバースDCF（設備投資補完）', '開示設備投資で補完した条件下の織り込み成長率。独立した株価予測ではない。'),
 'asset_stress': ('簿価資産ストレス試算', 'PPE・棚卸0/20/40%、無形資産0/50/100%減額。時価修正純資産や清算価値の下限ではない。'),
 'adjusted_nav': ('修正純資産法（時価）', '資産別時価・簿外債務等が不足しているため算出不可。'),
}

def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k,v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    if pd.isna(value): return None
    if isinstance(value, pd.Timestamp): return value.strftime('%Y-%m-%d')
    return value

def export(raw, predictions, output):
    pred = json.loads(Path(predictions).read_text())
    candidates = pred.get('candidates', [])
    codes = {str(x['jqCode']) for x in candidates}
    if not codes:
        raise ValueError('No candidate codes; preserve previous valuation data')
    bars = B.load_parts('bars', str(raw))
    bars['Code'] = bars.Code.astype(str)
    bars['Date'] = pd.to_datetime(bars.Date)
    bars = bars[bars.Code.isin(codes)].sort_values(['Code','Date']).drop_duplicates(['Code','Date'])
    prices = bars[['Code','Date','C','AdjC']].rename(columns={'C':'close_raw','AdjC':'close'})
    prices['close'] = prices.close.fillna(prices.close_raw)
    prices = prices[prices.close_raw.gt(0) & prices.close.gt(0)]
    asof = prices.Date.max()
    keys = prices.groupby('Code').tail(1)[['Code','Date','close_raw']].reset_index(drop=True)
    fins = B.load_parts('fins', str(raw))
    market = B.load_parts('topix', str(raw));market['Date'] = pd.to_datetime(market.Date)
    print('Current financial inputs:',len(keys),flush=True)
    tv = FI.calculate(keys, prices, fins, market, FI.yields(raw))
    fin = pd.read_parquet(raw/'edinet_fin.parquet')
    companies = pd.read_parquet(raw/'edinet_companies.parquet')
    # Same source-backed normalization/corrections as E122, when supplied in source.
    corr_path = Path(__file__).parent/'valuation_corrections.csv'
    corr = pd.read_csv(corr_path,dtype={'Code':str}) if corr_path.exists() else None
    panel = CE.annual_panel(fin, CE.periods_from_fins(fins), companies, corr)
    states = M.annual_states(panel, corr)
    master = pd.concat([pd.read_parquet(f) for f in sorted(raw.glob('master_hist_*.parquet'))+[raw/'master.parquet']],ignore_index=True)
    master['Code'] = master.Code.astype(str);master['Date'] = pd.to_datetime(master.Date)
    master = master.sort_values(['Date','Code']).drop_duplicates(['Code','Date'],keep='last')
    vals = pd.concat([pd.read_parquet(raw/f'valuation_{year}.parquet') for year in keys.Date.dt.year.unique()],ignore_index=True)
    vals['Date']=pd.to_datetime(vals.Date);vals['Code']=vals.Code.astype(str)
    vals = vals[vals.Date.isin(keys.Date.unique())][['Code','Date','PER','PBR','MktCap']].drop_duplicates(['Code','Date'])
    vals['MktCap'] = pd.to_numeric(vals.MktCap,errors='coerce')*1e6
    vals = sectors(vals,master[['Code','Date','S33']])
    d = ev_fields(M.attach(tv.merge(vals,on=['Code','Date'],how='left',validate='one_to_one'),states))
    peer = ev_fields(M.attach(vals,states))
    results = {name:pd.DataFrame({s:tv['tv_'+name+'_'+s] for s in M.STATS}) for name in ['ri','ddm']}
    for name, metric in [('per','PER'),('pbr','PBR'),('ev_ebitda','EV_EBITDA')]:
        results[name] = M.peer_prices(peer,d,metric,enterprise=name=='ev_ebitda')
    for kind in ['cash','flex']:
        results['dcf_'+kind] = M.value_grid(d,kind)
        results['mc_'+kind] = M.monte_carlo(d,kind,1024)
        results['reverse_'+kind] = M.reverse(d,kind)
    results['asset_stress'] = M.asset_stress(d)
    names = {str(x['jqCode']):x.get('name',x.get('code',x['jqCode'])) for x in candidates}
    stocks = []
    for i,r in d.iterrows():
        history = prices[prices.Code.eq(r.Code) & prices.Date.le(r.Date)].tail(20)
        # Express historical closes in the latest share unit (split-comparable yen).
        factor = r.close_raw/history.close.iloc[-1]
        h = [{'date':t.Date.strftime('%Y-%m-%d'),'close':float(t.close*factor)} for t in history.itertuples()]
        methods=[]
        for name,(label,note) in METHODS.items():
            values = results[name].loc[i].to_dict() if name in results else {}
            reverse = name.startswith('reverse')
            main = values.get('growth' if reverse else 'median',np.nan)
            available = pd.notna(main)
            methods.append({'id':name,'label':label,'note':note,'kind':'growth' if reverse else 'price',
                'available':available,'reason':None if available else ('資産別時価・簿外債務等のデータ不足' if name=='adjusted_nav' else '必要データ・鮮度・計算条件を満たさない'),
                **{k:v for k,v in values.items() if k in {*M.STATS,'p05','p95','p10','p90','growth','peers','valid_fraction'}},
                'deviation':main/r.close_raw-1 if available and not reverse else None})
        stocks.append({'code':r.Code,'name':names.get(r.Code,r.Code),'date':r.Date,'price':r.close_raw,
          'history':h,'financialDate':r.get('tv_fin_source_date'),'annualDate':r.get('filing_avail_date'),
          'yieldDate':r.get('tv_rf_date'),'beta':r.tv_beta,'riskFree':r.tv_rf,
          'debtReviewed':bool(r.v_debt_reviewed) if pd.notna(r.v_debt_reviewed) else False,'methods':methods})
    payload=clean({'schemaVersion':1,'asOf':asof,'generatedAt':datetime.now(timezone.utc).isoformat(),
      'warning':'レンジは前提条件の感応度・仮定分布であり、将来株価の予測区間ではありません。DCF系・EV法の負債の網羅性は未検証です。',
      'stocks':stocks})
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False));tmp.replace(output)
    print('Valuations exported:',len(stocks),'stocks; asOf',asof.date(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path('research/_data'));p.add_argument('--predictions',default='public/data/predictions.json');p.add_argument('--output',type=Path,default=Path('public/data/valuations.json'))
    a=p.parse_args();export(a.data_dir,a.predictions,a.output)
