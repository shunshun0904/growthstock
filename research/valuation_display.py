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
    if known('tv_eps') and row.tv_eps <= 0:
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
        reasons.append(f'有効な評価シナリオが80%未満（{fraction*100:.1f}%）' if pd.notna(fraction) else '評価シナリオを作成できない')
    return '／'.join(reasons)
