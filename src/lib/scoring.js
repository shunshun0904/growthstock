/**
 * GrowthStockAnalyzer - Focus : 8軸スコアリングエンジン
 *
 * 仕様書 §4 「スコアリング & 判定アルゴリズム」の実装。
 * 純粋関数のみで構成し、UI とデータパイプラインの両方から利用する。
 *
 * 【重要な設計方針】
 *  指標が未取得 (null / undefined) の場合、その軸のスコアは 0 ではなく null を返す。
 *  0 点として扱うと「実測でゼロ」と「データが無い」の区別が消え、
 *  存在しない評価をでっち上げることになるため。
 *  総合スコアは「値が存在する軸」のみの平均とし、coverage (n/8) を併記する。
 */

/** 線形正規化: S(x, min, max) = clamp(0, 10, (x - min) / (max - min) * 10) */
export function normalize(x, min, max) {
  if (!Number.isFinite(x)) return null;
  if (max === min) return null;
  return Math.max(0, Math.min(10, ((x - min) / (max - min)) * 10));
}

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/* ------------------------------------------------------------------ *
 * 軸ごとのスコア関数 (仕様書 §4.1 の表に対応)
 * ------------------------------------------------------------------ */

/** 軸1: 直近四半期EPS成長率 -> S(x, 0, 50) */
export function scoreEpsGrowth(epsGrowth) {
  return isNum(epsGrowth) ? normalize(epsGrowth, 0, 50) : null;
}

/** 軸2: 直近四半期売上高成長率 -> S(x, 0, 40) */
export function scoreSalesGrowth(salesGrowth) {
  return isNum(salesGrowth) ? normalize(salesGrowth, 0, 40) : null;
}

/** 軸3: ROE -> S(x, 5, 25) */
export function scoreRoe(roe) {
  return isNum(roe) ? normalize(roe, 5, 25) : null;
}

/** 軸4: 営業利益率 -> S(x, 0, 20) */
export function scoreOpMargin(opMargin) {
  return isNum(opMargin) ? normalize(opMargin, 0, 20) : null;
}

/**
 * 軸5: テクニカル (52週高値接近率 R_high)
 *   R >= 98            -> 10.0
 *   90 <= R < 98       -> 8.0 + (R-90)/8  * 1.5
 *   80 <= R < 90       -> 6.0 + (R-80)/10 * 2.0
 *   R < 80             -> 仕様未定義のため線形外挿 R/80 * 6.0 (R=80 で 6.0 と連続)
 */
export function scoreTechnical(highRatio) {
  if (!isNum(highRatio)) return null;
  if (highRatio >= 98) return 10;
  if (highRatio >= 90) return 8.0 + ((highRatio - 90) / 8) * 1.5;
  if (highRatio >= 80) return 6.0 + ((highRatio - 80) / 10) * 2.0;
  return Math.max(0, (highRatio / 80) * 6.0);
}

/**
 * 軸6: 出来高 (出来高モメンタム × 機関参入レベルによる減衰)
 *
 * 仕様書は「出来高増減率を基本とし、機関参入レベルに応じて減衰補正」「満点条件:
 * 10億円以上の売買代金かつ増」とのみ記述しているため、以下を実装として確定させた:
 *
 *   base   = clamp(0, 10, 5.0 + (Trend_vol - 100) / 100 * 5.0)
 *            → 出来高が20日平均と同水準(100%)で 5.0、2倍(200%)で 10.0
 *   score  = base × decay(機関参入レベル)
 *            → high / mega (売買代金10億円以上) は decay = 1.0 なので満点到達可能
 */
export const VOLUME_DECAY = {
  mega: 1.0,
  high: 1.0,
  moderate: 0.85,
  low: 0.65,
  cap_low: 0.7,
  none: 0.35,
};

export function scoreVolume(volumeTrend, institutionalLevel) {
  if (!isNum(volumeTrend)) return null;
  const base = Math.max(0, Math.min(10, 5.0 + ((volumeTrend - 100) / 100) * 5.0));
  const decay = VOLUME_DECAY[institutionalLevel] ?? 1.0;
  return Math.max(0, Math.min(10, base * decay));
}

/**
 * 軸7: 需給 (信用倍率 C_ratio)
 *   C <= 1.0   -> 10.0
 *   C <= 3.0   -> 10.0 - (C-1.0)/2.0 * 3.0     (C=3 で 7.0)
 *   C <= 10.0  ->  7.0 - (C-3.0)/7.0 * 5.0     (C=10 で 2.0)
 *   C > 10.0   -> 仕様未定義のため線形外挿 max(0, 2.0 - (C-10)/10 * 2.0) (C=20 で 0)
 */
export function scoreCreditRatio(creditRatio) {
  if (!isNum(creditRatio) || creditRatio < 0) return null;
  if (creditRatio <= 1.0) return 10;
  if (creditRatio <= 3.0) return 10.0 - ((creditRatio - 1.0) / 2.0) * 3.0;
  if (creditRatio <= 10.0) return 7.0 - ((creditRatio - 3.0) / 7.0) * 5.0;
  return Math.max(0, 2.0 - ((creditRatio - 10.0) / 10.0) * 2.0);
}

/**
 * 軸8: 進捗期待 (決算進捗率 vs 経過基準)
 *   B      = Quarter × 25
 *   Expect = clamp(0, 10, 5.0 + (Progress - B) / 2.0)
 */
export function scoreProgress(progressRate, quarter) {
  if (!isNum(progressRate) || !isNum(quarter)) return null;
  const benchmark = quarter * 25;
  return Math.max(0, Math.min(10, 5.0 + (progressRate - benchmark) / 2.0));
}

/* ------------------------------------------------------------------ *
 * §4.2 機関投資家参入度判定
 * ------------------------------------------------------------------ */

/** 売買代金(億円)のみによる流動性ティア */
export function liquidityTier(tradingValueOku) {
  if (!isNum(tradingValueOku)) return null;
  if (tradingValueOku >= 30) return 'mega';
  if (tradingValueOku >= 10) return 'high';
  if (tradingValueOku >= 5) return 'moderate';
  if (tradingValueOku >= 1) return 'low';
  return 'none';
}

/**
 * 機関投資家参入度。時価総額 100億円未満は大型機関の投資制約対象のため
 * 'cap_low' が流動性ティアを上書きする (流動性ティアは liquidityTier で別途保持)。
 */
export function institutionalLevel(tradingValueOku, marketCapOku) {
  const tier = liquidityTier(tradingValueOku);
  if (isNum(marketCapOku) && marketCapOku < 100) return 'cap_low';
  return tier;
}

export const INSTITUTIONAL_META = {
  mega:     { label: '機関熱狂・Monster', short: 'MEGA',     tone: 'violet', desc: '売買代金30億円以上・超高流動性' },
  high:     { label: '機関主導・強',       short: 'HIGH',     tone: 'blue',   desc: '売買代金10〜30億円・明確な機関買い' },
  moderate: { label: '機関参入圏内',       short: 'MODERATE', tone: 'green',  desc: '売買代金5〜10億円・中小型ファンド参入可' },
  low:      { label: '個人・小口主導',     short: 'LOW',      tone: 'amber',  desc: '売買代金1〜5億円' },
  none:     { label: '流動性不足',         short: 'NONE',     tone: 'red',    desc: '売買代金1億円未満・投機対象外' },
  cap_low:  { label: '時価総額不足',       short: 'CAP LOW',  tone: 'slate',  desc: '時価総額100億円未満・大型機関の投資制約対象' },
};

/* ------------------------------------------------------------------ *
 * §4.3 株価ゾーン判定
 * ------------------------------------------------------------------ */

export function priceZone(highRatio) {
  if (!isNum(highRatio)) return null;
  if (highRatio >= 98) return 'BREAKOUT';
  if (highRatio >= 90) return 'HANDLE';
  if (highRatio >= 80) return 'BASE';
  return 'CORRECTION';
}

export const ZONE_META = {
  BREAKOUT:   { label: 'BREAKOUT',   ja: 'ブレイクアウト', tone: 'violet', desc: '52週高値更新・青天井' },
  HANDLE:     { label: 'HANDLE',     ja: '取っ手形成',     tone: 'blue',   desc: 'Cup with Handle のハンドル形成圏' },
  BASE:       { label: 'BASE',       ja: '土台築造',       tone: 'green',  desc: 'ベース形成・底固め' },
  CORRECTION: { label: 'CORRECTION', ja: '調整中',         tone: 'amber',  desc: '深い調整・トレンド修復待ち' },
};

/* ------------------------------------------------------------------ *
 * 8軸まとめ
 * ------------------------------------------------------------------ */

export const AXES = [
  { key: 'eps',       label: 'EPS成長',   full: '直近四半期EPS成長率',   unit: '%',  metric: 'epsGrowth',    rule: 'S(x, 0, 50)' },
  { key: 'sales',     label: '売上成長',  full: '直近四半期売上高成長率', unit: '%',  metric: 'salesGrowth',  rule: 'S(x, 0, 40)' },
  { key: 'roe',       label: '収益質',    full: 'ROE (自己資本利益率)',   unit: '%',  metric: 'roe',          rule: 'S(x, 5, 25)' },
  { key: 'margin',    label: '利益率',    full: '営業利益率',             unit: '%',  metric: 'opMargin',     rule: 'S(x, 0, 20)' },
  { key: 'technical', label: 'テクニカル', full: '52週高値接近率',         unit: '%',  metric: 'highRatio',    rule: '98%以上で満点' },
  { key: 'volume',    label: '出来高',    full: '出来高モメンタム',       unit: '%',  metric: 'volumeTrend',  rule: '機関参入度で減衰補正' },
  { key: 'supply',    label: '需給',      full: '信用倍率',               unit: '倍', metric: 'creditRatio',  rule: '1.0倍以下で満点' },
  { key: 'progress',  label: '進捗期待',  full: '決算進捗率 vs 経過基準', unit: '%',  metric: 'progressRate', rule: '5.0 + (進捗率 - Q×25)/2' },
];

/**
 * 指標セットから 8軸スコア・総合スコア・各種判定を算出する。
 *
 * @param {object} m 指標オブジェクト
 *   epsGrowth, salesGrowth, roe, opMargin, highRatio, volumeTrend,
 *   creditRatio, progressRate, quarter, tradingValue (億円), marketCap (億円)
 * @returns {{scores: object, axisScores: Array, totalScore: number|null,
 *            strictTotalScore: number|null, coverage: number,
 *            institutional: string|null, liquidity: string|null, zone: string|null}}
 */
export function computeScores(m = {}) {
  const institutional = institutionalLevel(m.tradingValue, m.marketCap);
  const liquidity = liquidityTier(m.tradingValue);

  const scores = {
    eps:       scoreEpsGrowth(m.epsGrowth),
    sales:     scoreSalesGrowth(m.salesGrowth),
    roe:       scoreRoe(m.roe),
    margin:    scoreOpMargin(m.opMargin),
    technical: scoreTechnical(m.highRatio),
    volume:    scoreVolume(m.volumeTrend, institutional),
    supply:    scoreCreditRatio(m.creditRatio),
    progress:  scoreProgress(m.progressRate, m.quarter),
  };

  const present = AXES.map((a) => scores[a.key]).filter(isNum);
  const coverage = present.length;

  // 仕様書 §4.1: TotalScore = (1/8) Σ Score_k
  // 未取得軸を 0 とみなす厳密版と、取得済み軸のみの平均の両方を返す。
  const strictTotalScore = present.reduce((s, v) => s + v, 0) / AXES.length;
  const totalScore = coverage > 0 ? present.reduce((s, v) => s + v, 0) / coverage : null;

  const axisScores = AXES.map((a) => ({
    ...a,
    score: scores[a.key],
    value: isNum(m[a.metric]) ? m[a.metric] : null,
  }));

  return {
    scores,
    axisScores,
    coverage,
    totalScore: totalScore === null ? null : round1(totalScore),
    strictTotalScore: coverage > 0 ? round1(strictTotalScore) : null,
    institutional,
    liquidity,
    zone: priceZone(m.highRatio),
  };
}

export function round1(v) {
  return isNum(v) ? Math.round(v * 10) / 10 : null;
}

/** レーダーチャート (Recharts) 用データ形状へ変換 */
export function toRadarData(scoreSets) {
  return AXES.map((a) => {
    const row = { axis: a.label, fullMark: 10, _key: a.key };
    for (const [name, s] of Object.entries(scoreSets)) {
      row[name] = isNum(s?.[a.key]) ? round1(s[a.key]) : 0;
    }
    return row;
  });
}
