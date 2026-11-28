"""Display-only price alignment and explicit valuation availability reasons.

These rules do not modify the financial inputs used to train/serve models.
"""
import numpy as np
import pandas as pd


def price_keys(prices, market):
    """Use latest stock/TOPIX common close, allowing at most one stock bar back."""
    dates = pd.to_datetime(market.Date).dropna().unique()
    ordered = prices.sort_values(['Code', 'Date'])
    latest = ordered.groupby('Code').tail(1).set_index('Code')
    recent = ordered.groupby('Code').tail(2)
    common = recent[recent.Date.isin(dates)].groupby('Code').tail(1)
    return common[['Code', 'Date', 'close_raw']].reset_index(drop=True), latest


def latest_financials(fins):
    f = fins.copy()
    for c in ['DiscDate', 'CurPerEn']:
        f[c] = pd.to_datetime(f[c], errors='coerce')
    f = f[f.DocType.str.contains('FinancialStatements', na=False) &
          f.CurPerType.isin(['FY','1Q','2Q','3Q']) & f.CurPerEn.le(f.DiscDate)]
    return {str(code): g for code, g in f.groupby('Code')}


def unavailable_reason(method, row, financials):
    if row is None:
        return '株価とTOPIXの共通データが未取得（前営業日まで確認）'
    reasons = []
    def known(name):
        value = row.get(name, np.nan)
        return pd.notna(value) and np.isfinite(float(value))
    if not known('tv_beta'):
        n = row.get('tv_beta_n', 0)
        reasons.append(f'β推定に必要な株価・TOPIXの有効観測が不足（{int(n) if pd.notna(n) else 0}件、最低252件）')
    if not known('tv_rf'):
        reasons.append('利用可能な国債10年金利がない、または基準日が7日超前')
    # Explain why the reconstruction returned no per-share data.
    f = financials.get(str(row.Code))
    snap = f[f.DiscDate.lt(row.Date)] if f is not None else pd.DataFrame()
    if not known('tv_eps') or (method == 'ri' and not known('tv_bps')):
        if snap.empty:
            reasons.append('評価日より前に利用可能な決算データがない')
        else:
            order = ['CurPerEn','DiscDate'] + (['DiscTime'] if 'DiscTime' in snap else [])
            latest = snap.sort_values(order).iloc[-1]
            if (row.Date-latest.DiscDate).days > 365:
                reasons.append('決算開示から365日超経過')
            elif pd.isna(latest.get('TrShFY')):
                reasons.append('自己株式数が欠損し、1株利益・純資産を再計算できない（ゼロとは仮定しない）')
            elif pd.isna(latest.get('ShOutFY')):
                reasons.append('発行済株式数が欠損')
            elif not known('tv_eps'):
                reasons.append('直近12か月利益の再計算に必要な前年同期・前期通期などの情報が不足')
            else:
                reasons.append('1株純資産の再計算に必要な財務・株数調整データが不足')
    if method == 'ddm' and known('tv_eps') and row.tv_eps <= 0:
        reasons.append(f'直近12か月EPSが0以下（{row.tv_eps:.2f}円）。現行の評価は黒字企業のみ対象')
    if method == 'ri' and known('tv_bps') and row.tv_bps <= 0:
        reasons.append('1株純資産が0以下で、現行RIMの評価対象外')
    if method == 'ddm':
        if not known('tv_div0'):
            reasons.append('利用可能な実績年間配当データが不足')
        elif row.tv_div0 <= 0:
            reasons.append('実績年間配当が0以下（無配など）')
        if known('tv_payout') and not 0 < row.tv_payout <= 1:
            reasons.append('実績配当性向が0%以下または100%超で、現行評価の対象外')
    if not reasons:
        fraction = row.get('tv_'+method+'_valid_fraction', np.nan)
        if method == 'ri' and row.get('ri_recovery_assumed',False):
            reasons.append('回復仮定の途中で純資産が0以下になる等、有効シナリオが80%未満')
        else:
            reasons.append(f'有効な評価シナリオが80%未満（{fraction*100:.1f}%）' if pd.notna(fraction) else '評価シナリオを作成できない')
    return '／'.join(reasons)


def rim_with_provider_inputs(frame, prices, indicators, market):
    """Fill missing RIM inputs from a coherent historical provider EPS/BPS pair.

    Display only: never change training/serving inputs. Preserve nonpositive
    observed earnings/equity, reject future/stale indicator dates, and convert
    per-share units using observed split factors rather than invent treasury zero.
    """
    import theoretical_value as V
    out = frame.copy()
    out['ri_input_source'] = 'reconstructed'
    out['ri_input_date'] = pd.to_datetime(out.tv_fin_source_date)
    if indicators.empty:
        return out
    right = indicators[['Code','Date','EPS','BPS']].copy()
    right['Code'] = right.Code.astype(str)
    right['Date'] = pd.to_datetime(right.Date)
    right = right.sort_values('Date').drop_duplicates(['Code','Date'],keep='last')
    right = right.rename(columns={'Date':'indicator_date','EPS':'provider_eps','BPS':'provider_bps'})
    left = out[['Code','Date']].copy().assign(_i=np.arange(len(out)))
    joined = pd.merge_asof(left.sort_values('Date'),right.sort_values('indicator_date'),
                           left_on='Date',right_on='indicator_date',by='Code',direction='backward')
    joined = joined.sort_values('_i').reset_index(drop=True)
    calendar = pd.DatetimeIndex(pd.to_datetime(market.Date).dropna().unique()).sort_values()
    factors = prices.assign(_factor=prices.close_raw/prices.close).set_index(['Code','Date'])._factor
    for i, r in joined.iterrows():
        original = out.iloc[i]
        eps, book = original.tv_eps, original.tv_bps
        if pd.notna(eps) and pd.notna(book):
            continue
        if (pd.notna(eps) and eps <= 0) or (pd.notna(book) and book <= 0):
            continue
        if pd.isna(r.indicator_date):
            continue
        target_pos, source_pos = calendar.get_indexer([r.Date, r.indicator_date])
        if min(target_pos, source_pos) < 0 or not 0 <= target_pos-source_pos <= 1:
            continue
        pair = pd.to_numeric(pd.Series([r.provider_eps, r.provider_bps]),errors='coerce').to_numpy(float)
        if not (np.isfinite(pair).all() and (pair > 0).all()):
            continue
        source_factor = factors.get((r.Code,r.indicator_date),np.nan)
        target_factor = factors.get((r.Code,r.Date),np.nan)
        if not (np.isfinite(source_factor) and np.isfinite(target_factor) and source_factor>0 and target_factor>0):
            continue
        converted = pair * target_factor / source_factor
        idx = out.index[i]
        out.loc[idx,['tv_eps','tv_bps']] = converted
        out.loc[idx,'ri_input_source'] = 'provider_eps_bps'
        out.loc[idx,'ri_input_date'] = r.indicator_date
    valued = V.valuations(out)
    # DDM, coverage flags and the production C features keep their original definitions.
    for col in [c for c in valued if c.startswith('tv_ri_')]:
        out[col] = valued[col]
    return out


def loss_aware_residual_value(book, earnings, ke, horizon, payout):
    """Conditional RIM: ROE converges to ke; losses reduce book in full.

    A loss never causes a negative dividend/capital injection. Any exhausted
    book invalidates the path: no recapitalization or return from insolvency is
    silently assumed. Terminal residual income is zero, so terminal value is book.
    """
    if horizon < 1 or not 0 <= payout <= 1:
        raise ValueError('Invalid RIM horizon or payout')
    b, earnings, ke = np.broadcast_arrays(np.asarray(book,float),np.asarray(earnings,float),np.asarray(ke,float))
    b = b.copy()
    valid = np.isfinite(b)&np.isfinite(earnings)&np.isfinite(ke)&(b>0)&(ke>0)
    with np.errstate(all='ignore'):
        roe = earnings/b
        value = b.copy()
        for t in range(1,horizon+1):
            rt = ke+(roe-ke)*(horizon-t)/horizon
            e = rt*b
            value = value+(e-ke*b)/(1+ke)**t
            dividend = np.maximum(e,0)*payout
            b = b+e-dividend
            valid &= np.isfinite(b)&(b>0)
            b = np.where(valid,b,np.nan)
    return np.where(valid&np.isfinite(value),value,np.nan)


def rim_with_loss_recovery(frame):
    """Display-only extension of the existing 30 scenarios to nonpositive EPS."""
    import theoretical_value as V
    out = frame.copy()
    out['ri_recovery_assumed'] = False
    earn = pd.to_numeric(out.tv_eps,errors='coerce').to_numpy(dtype=float,na_value=np.nan)
    book = pd.to_numeric(out.tv_bps,errors='coerce').to_numpy(dtype=float,na_value=np.nan)
    mask = np.isfinite(earn)&(earn<=0)&np.isfinite(book)&(book>0)
    if not mask.any():
        return out
    target = out.loc[mask]
    book, earn, beta, rf = [target[c].to_numpy(float) for c in ['tv_bps','tv_eps','tv_beta','tv_rf']]
    scenarios = []
    for erp in V.ERP:
        ke = rf+beta*erp
        for horizon in V.HORIZONS:
            for payout in V.PAYOUTS:
                scenarios.append(loss_aware_residual_value(book,earn,ke,horizon,payout))
    summary, fraction = V.summarize(np.column_stack(scenarios))
    out.loc[mask,'ri_recovery_assumed'] = True
    out.loc[mask,'tv_ri_valid_fraction'] = fraction
    price = target.close_raw.to_numpy(float)
    for j, stat in enumerate(V.STATS):
        out.loc[mask,'tv_ri_'+stat] = summary[:,j]
        out.loc[mask,'tv_ri_'+stat+'_rel'] = summary[:,j]/price-(0 if stat=='std' else 1)
    return out
