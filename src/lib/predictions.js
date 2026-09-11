/**
 * ブレイク予測データのロードと、画面で使う小さな変換。
 *
 * predictions.json は research/predict_daily.py が毎営業日の終値後に書く。
 * 無くても他のタブは動くので、読めないことは致命的ではない（画面に理由を出す）。
 */

/**
 * 公開パスの前置き。Vite が import.meta.env.BASE_URL を入れる。
 *
 * 素の node（単体テスト）では env が無いので、モジュールの読み込み時点で
 * 評価すると落ちる。純粋な変換関数までテストできなくなるので、
 * URL は使うときに組み立てる。
 */
const base = () => import.meta.env?.BASE_URL ?? '/';
const PRED_URL = () => `${base()}data/predictions.json`;
const HIST_URL = () => `${base()}data/prediction_history.json`;

async function getJson(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`読み込めませんでした (HTTP ${res.status})`);
  return res.json();
}

export async function loadPredictions() {
  const data = await getJson(PRED_URL());
  if (!data || !Array.isArray(data.candidates)) {
    throw new Error('predictions.json の形式が不正です');
  }
  return data;
}

/** 追跡ファイルは初回実行時にはまだ薄い。読めなくても空で返す。 */
export async function loadHistory() {
  try {
    const d = await getJson(HIST_URL());
    return Array.isArray(d?.entries) ? d : { entries: [] };
  } catch {
    return { entries: [] };
  }
}

/* ------------------------------------------------------------------ */

/**
 * スコアの色。帯（out-of-fold の10分位）で決める。
 * 生スコアの絶対値で決めると、モデルを学習し直したときに意味が変わる。
 */
export function bandColor(band) {
  if (!Number.isFinite(band)) return 'var(--text-faint)';
  if (band >= 9) return 'var(--violet)';
  if (band >= 7) return 'var(--green)';
  if (band >= 4) return 'var(--amber)';
  return 'var(--red)';
}

export function bandLabel(band) {
  if (!Number.isFinite(band)) return '—';
  if (band >= 9) return '最上位';
  if (band >= 7) return '上位';
  if (band >= 4) return '中位';
  return '下位';
}

/**
 * 候補を「8軸オクタゴン」がそのまま描ける形に変換する。
 *
 * 予測タブで見つけた銘柄を、エントリー判断のためにオクタゴン側へ渡すための橋。
 * 値はすべて予測パイプラインが持っている実測値で、ここで作り出すものは無い。
 * 取れていない指標は null のままにする（0 で埋めると「実測でゼロ」と混ざる）。
 */
export function candidateToStock(c) {
  const n = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  return {
    id: `pred:${c.jqCode}`,
    code: c.code,
    jqCode: c.jqCode,
    name: c.name || c.code,
    sector: c.sector || null,
    market: c.market || null,
    scale: c.scale || null,
    note: `ブレイク予測から追加（${c.date} / ${c.rankInDay}位）`,
    asOf: c.date,
    origin: 'prediction',
    metrics: {
      date: c.date,
      price: n(c.close),
      high52w: n(c.high52w),
      highRatio: n(c.rHigh),
      tradingValue: n(c.tradingValue),
      turnoverValue: n(c.tradingValue),
      volumeTrend: n(c.volumeTrend),
      epsGrowth: n(c.epsGrowth),
      salesGrowth: n(c.salesGrowth),
      roe: n(c.roe),
      opMargin: n(c.opMargin),
      progressRate: n(c.progressRate),
    },
  };
}

/**
 * その日の地合いの向き。候補ごとの地合い寄与の中央値で判断する。
 *
 * 地合い11列は、ラベルをボラ正規化にしてから効きが大きくなった
 * （外すと CV PR-AUC 0.4016 -> 0.3337）。全体が沈んでいる日に
 * 「1位だから買い」と読まないための注意書きに使う。
 */
export function marketTone(candidates) {
  const v = candidates
    .map((c) => c?.contrib?.marketContrib)
    .filter((x) => typeof x === 'number' && Number.isFinite(x))
    .sort((a, b) => a - b);
  if (!v.length) return null;
  const med = v[Math.floor(v.length / 2)];
  return {
    median: med,
    label: med > 0.05 ? '追い風' : med < -0.05 ? '向かい風' : '中立',
    tone: med > 0.05 ? 'green' : med < -0.05 ? 'red' : 'slate',
  };
}
