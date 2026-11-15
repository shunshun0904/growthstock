/* Research-only static data. No API keys, market APIs, or automatic retries. */
const finite = x => Number.isFinite(x);
const positive = x => finite(x) && x > 0;

export function normalizeCode(value) {
  const code = String(value).trim().toUpperCase();
  if (!/^[0-9A-Z]{4,5}$/.test(code)) throw new Error('銘柄コードの形式が不正です');
  return code.length === 4 ? `${code}0` : code;
}

export function decodeSeries(payload) {
  if (payload?.schemaVersion !== 1 || payload.researchOnly !== true) throw new Error('研究データの形式が不正です');
  const { columns, data } = payload;
  if (!Array.isArray(columns) || new Set(columns).size !== columns.length || !columns.includes('Date')
      || !columns.includes('price') || !Array.isArray(data) || data.length > 25000
      || !data.every(row => Array.isArray(row) && row.length === columns.length)) {
    throw new Error('日次系列の形式が不正です');
  }
  const rows = data.map(row => Object.fromEntries(columns.map((name, i) => [name, row[i]])));
  if (!rows.every((r, i) => /^\d{4}-\d{2}-\d{2}$/.test(r.Date) && (!i || rows[i - 1].Date < r.Date))) {
    throw new Error('日次系列の日付が不正です');
  }
  return { ...payload, rows };
}

export function createRoicWaccClient({ baseUrl = import.meta.env?.BASE_URL ?? '/', fetchImpl = globalThis.fetch, timeoutMs = 30000 } = {}) {
  const base = `${baseUrl.replace(/\/?$/, '/')}data/roic-wacc/`;
  async function read(file, signal) {
    const control = new AbortController();
    const abort = () => control.abort();
    if (signal?.aborted) abort();
    signal?.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(abort, timeoutMs);
    try {
      const response = await fetchImpl(`${base}${file}`, { signal: control.signal });
      if (!response.ok) throw new Error(`研究データを取得できません（HTTP ${response.status}）`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      let body;
      // Pages may deliver an already decompressed body; inspect magic bytes.
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
        if (typeof DecompressionStream === 'undefined') throw new Error('このブラウザーは圧縮データに対応していません');
        body = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
      } else body = new TextDecoder().decode(bytes);
      if (body.length > 16000000) throw new Error('研究データが大きすぎます');
      return JSON.parse(body);
    } catch (error) {
      if (error.name === 'AbortError') throw new Error(signal?.aborted ? '読み込みを中止しました' : '研究データの読み込みがタイムアウトしました');
      throw error;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
    }
  }
  return {
    async catalog(signal) {
      const index = await read('index.json', signal);
      if (index.schemaVersion !== 1 || index.researchOnly !== true || !Array.isArray(index.stocks)) throw new Error('銘柄一覧の形式が不正です');
      return index;
    },
    async series(stock, signal) {
      const code = normalizeCode(stock.code);
      const file = stock.file || `${code}.json`;
      if (file !== `${code}.json` && file !== `${code}.json.gz`) throw new Error('銘柄データのパスが不正です');
      const payload = decodeSeries(await read(file, signal));
      if (normalizeCode(payload.code) !== code) throw new Error('銘柄データが一致しません');
      return payload;
    },
  };
}

export function scenarioSeries(payload, { model = 'eva_fade5', tax = 'tax30', erp = 'all', growth = 'all', epsBasis = 'forecast' } = {}) {
  const scenarios = (payload.scenarios || []).filter(s => s.model === model && s.tax === tax
    && (erp === 'all' || Math.abs(s.erp - Number(erp)) < 1e-10)
    && (growth === 'business' ? s.growth === 'business' : s.growth !== 'business' && (growth === 'all' || Math.abs(s.growth - Number(growth)) < 1e-10)));
  const rows = payload.rows.map(row => {
    const values = scenarios.map(s => row.scenarios?.[s.id]).filter(positive).sort((a, b) => a - b);
    const n = values.length;
    const median = n ? (values[Math.floor((n - 1) / 2)] + values[Math.floor(n / 2)]) / 2 : null;
    const mean = n ? values.reduce((sum, value) => sum + value, 0) / n : null;
    const eps = row[`eps_${epsBasis}`];
    const waccKey = `${tax}_${String(Math.round((erp === 'all' ? .05 : Number(erp)) * 100)).padStart(2, '0')}`;
    const independent = model.startsWith('eva_');
    const bridge = row[independent ? 'ev_bridge_per_share' : 'comparison_ev_bridge_per_share'];
    const cost = row[independent ? 'waccScenarios' : 'comparisonWaccScenarios'];
    const reinvest = scenarios.map(s => row.reinvestmentScenarios?.[s.id]).filter(finite).sort((a, b) => a - b);
    return { ...row, median, mean, min: n ? values[0] : null, max: n ? values[n - 1] : null,
      std: n ? Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / n) : null,
      scenarioCount: n, scenarioTotal: scenarios.length,
      gap: positive(row.price) && positive(median) ? row.price / median : null,
      selectedEvGap: finite(bridge) && positive(row.price + bridge) && positive(median + bridge) && positive(median)
        ? (row.price + bridge) / (median + bridge) : null,
      marketPer: positive(eps) && positive(row.price) ? row.price / eps : null,
      theoreticalPer: positive(eps) && positive(median) ? median / eps : null,
      selectedRoic: row[`roic_operating_${tax}`] ?? (tax === 'tax30' ? row.roic : null),
      conservativeRoic: row[`roic_conservative_${tax}`] ?? null,
      selectedWacc: cost?.[waccKey] ?? null,
      selectedReinvestment: reinvest.length ? (reinvest[Math.floor((reinvest.length - 1) / 2)] + reinvest[Math.floor(reinvest.length / 2)]) / 2 : null,
      valuation_model: model, tax_assumption: tax, equity_premium_scenario: erp, growth_assumption: growth, price_independent_model: independent,
      range: n ? [values[0], values[n - 1]] : null };
  });
  // Compute BEFORE slicing the display period; use the selected assumptions.
  rows.forEach((row, i) => {
    const history = rows.slice(Math.max(0, i - 251), i + 1).map(r => r.gap).filter(finite);
    row.gapZ = null; row.gapPercentile = null;
    if (finite(row.gap) && history.length >= 126) {
      const mean = history.reduce((a, b) => a + b, 0) / history.length;
      const sd = Math.sqrt(history.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (history.length - 1));
      row.gapZ = sd > 1e-12 ? (row.gap - mean) / sd : null;
      row.gapPercentile = (history.filter(x => x < row.gap).length + (history.filter(x => x === row.gap).length + 1) / 2) / history.length;
    }
    row.selectedSpread = finite(row.selectedRoic) && finite(row.selectedWacc) ? row.selectedRoic - row.selectedWacc : null;
    row.valuation_gap = row.gap;
    row.valuation_log_gap = positive(row.gap) ? Math.log(row.gap) : null;
    row.valuation_gap_zscore_252 = row.gapZ;
    row.valuation_gap_pct_252 = row.gapPercentile;
    for (const lag of [5, 20, 60]) {
      const old = rows[i - lag];
      row[`valuation_gap_change_${lag}d`] = finite(row.gap) && finite(old?.gap) ? row.gap - old.gap : null;
    }
    for (const n of [20, 60]) {
      const sample = rows.slice(Math.max(0, i - n + 1), i + 1).map(r => r.gap).filter(finite);
      const mean = sample.length === n ? sample.reduce((a, b) => a + b, 0) / n : null;
      row[`valuation_gap_ma_${n}`] = mean;
      if (n === 60) row.valuation_gap_vol_60 = finite(mean) ? Math.sqrt(sample.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (n - 1)) : null;
    }
    const prior = rows[i - 20];
    const adjusted = r => positive(r?.median) && positive(r?.share_unit_index) ? r.median / r.share_unit_index : null;
    const va = adjusted(row), vb = adjusted(prior);
    row.theoretical_price_change_20d = positive(va) && positive(vb) ? va / vb - 1 : null;
    row.wacc_change_20d = finite(row.selectedWacc) && finite(prior?.selectedWacc) ? row.selectedWacc - prior.selectedWacc : null;
    row.valuation_price_log_change_20d = positive(row.price_adjusted) && positive(prior?.price_adjusted) ? Math.log(row.price_adjusted / prior.price_adjusted) : null;
    row.valuation_value_log_change_20d = positive(va) && positive(vb) ? Math.log(va / vb) : null;
    row.valuation_gap_log_change_20d = positive(row.gap) && positive(prior?.gap) ? Math.log(row.gap / prior.gap) : null;
    if (finite(row.action_uncertainty) && row.action_uncertainty !== prior?.action_uncertainty) {
      for (const key of ['theoretical_price_change_20d', 'valuation_price_log_change_20d', 'valuation_value_log_change_20d', 'valuation_gap_log_change_20d']) row[key] = null;
    }
  });
  return rows;
}

export const REASONS = {
  financial_or_unknown: '金融業または業種を確認できない', annual_stale_or_missing: '利用可能な年次財務がない、または400日超',
  company_loss_or_profit_unverified: '赤字または親会社利益の範囲を確認できない', debt_missing: '有利子負債を確認できない',
  nci_missing: '非支配株主持分を確認できない', non_operating_cash_unidentified: '非事業性現金を識別できない',
  shares_missing_or_nonpositive: '自己株式控除後の株数を確認できない', risk_free_missing: '利用可能な国債金利がない',
  beta_unavailable: 'ベータの有効観測数などが不足', operating_roic_unavailable_or_nonpositive: '営業投下資本ROICが不明または0以下',
  nopat_nonpositive_or_missing: 'NOPATが不明または0以下', no_valid_positive_equity_scenario: '割引率・再投資率・正の株式価値の条件を満たさない',
  corporate_action_type_unverified: '企業行動が分割・併合と確認できない、またはライツ後の株数が未確認',
};
export const reasonText = text => (text || '').split(';').filter(Boolean).map(x => REASONS[x] || x).join('／');

export function disclosureMarkers(allRows, displayedRows) {
  if (!displayedRows.length) return [];
  const first = displayedRows[0].Date, last = displayedRows.at(-1).Date;
  const dates = new Set(displayedRows.map(row => row.Date));
  const markers = allRows.filter((row, i) => row.financial_source_date
    && (!i || row.financial_source_date !== allRows[i - 1].financial_source_date)
    && row.financial_source_date >= first && row.financial_source_date <= last)
    .map(row => ({ date: dates.has(row.financial_source_date) ? row.financial_source_date : row.financial_available_date,
      sourceDate: row.financial_source_date }));
  return markers.filter((m, i) => dates.has(m.date) && markers.findIndex(x => x.date === m.date) === i);
}

export function seriesCsv(rows) {
  const columns = ['Date', 'price', 'median', 'min', 'max', 'mean', 'std', 'gap', 'marketPer', 'theoreticalPer',
    'gapZ', 'gapPercentile', 'selectedRoic', 'conservativeRoic', 'selectedWacc', 'selectedSpread',
    'selectedEvGap', 'valuation_log_gap', 'valuation_gap_change_5d', 'valuation_gap_change_20d', 'valuation_gap_change_60d',
    'valuation_gap_ma_20', 'valuation_gap_ma_60', 'valuation_gap_vol_60', 'theoretical_price_change_20d', 'wacc_change_20d',
    'valuation_price_log_change_20d', 'valuation_value_log_change_20d', 'valuation_gap_log_change_20d',
    'selectedReinvestment', 'valuation_model', 'tax_assumption', 'equity_premium_scenario', 'growth_assumption', 'price_independent_model',
    'scenarioCount', 'scenarioTotal', 'financial_source_date', 'financial_available_date', 'unavailable_reason'];
  const cell = value => {
    const s = value == null ? '' : String(value);
    return `"${s.replace(/"/g, '""')}"`;
  };
  return '\ufeff' + [columns.join(','), ...rows.map(row => columns.map(c => cell(row[c])).join(','))].join('\r\n');
}
