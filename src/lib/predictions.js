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

/**
 * ビルド識別子。取得 URL に付けて、古い JSON を掴まないようにする。
 *
 * データが変わるのは必ずデプロイを伴う（予測ワークフローが JSON を
 * コミットし、それが Pages の再デプロイを起こす）ので、デプロイごとに
 * 変わるこの値を付ければ URL が必ず変わる。ブラウザにも CDN にも
 * 古い中身を返す余地が無くなる。
 *
 * 素の node（単体テスト）では define が効かないので、既定値を置く。
 */
const BUILD = typeof __BUILD_ID__ === 'string' ? __BUILD_ID__ : 'dev';
const PRED_URL = () => `${base()}data/predictions.json?v=${BUILD}`;
const HIST_URL = () => `${base()}data/prediction_history.json?v=${BUILD}`;

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

const FILINGS_URL = () => `${base()}data/filings.json?v=${BUILD}`;

/**
 * 有報・半期報告書の損益（public/data/filings.json。Fetch Filings が EDINET から作る）。
 * 無ければ null（取り込みがまだ）。形が違えば空の docs。
 */
export async function loadFilings() {
  try {
    const d = await getJson(FILINGS_URL());
    return d && typeof d.docs === 'object' && d.docs ? d : { docs: {} };
  } catch {
    return null;
  }
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

/**
 * 百分位の区分（上から順）。運用者の指示（2026-10-09）: 30 / 60 / 80 / 90 / 95 で分ける。
 * 95 は運用の選定「lgbm 単体 95以上」（上位5%）、90 は「3モデル 90以上」の線。
 */
export const PCT_TIERS = [95, 90, 80, 60, 30];
/** 凡例に並べる代表値（薄い順）。 */
export const PCT_LEGEND = [0, 30, 60, 80, 90, 95];
/** 上位5% の線（後方互換）。 */
export const TOP_PCT = PCT_TIERS[0];

/** 百分位が入る区分の下限（95 / 90 / 80 / 60 / 30 / 0）。値が無ければ null。 */
export function pctTier(pct) {
  if (!Number.isFinite(pct)) return null;
  for (const t of PCT_TIERS) if (pct >= t) return t;
  return 0;
}

/**
 * パーセンタイル（過去スコア分布での位置）の色。
 *
 * 1つの色相（青）の濃淡で、区分（PCT_TIERS）が上がるほど明るく鮮やか（暗い画面なので
 * 「濃い」= 明るい）。運用者の指示（2026-10-09）: 色相で分けると見分けづらい組（シアンと緑）が
 * 出るので濃淡にし、区分は 30 / 60 / 80 / 90 / 95。帯（10分位）の色（bandColor）とは別。
 * 文字に使うときは、低い区分が暗くて読めないので、文字を塗らずに四角（.pct-sw）を添える。
 */
export function pctColor(pct) {
  const t = pctTier(pct);
  return t === null ? 'var(--text-faint)' : `var(--pct-${t})`;
}

/**
 * モデル別の棒に添える短い記号。日本語名は横に並べると幅が足りない。
 *
 * 正本は research/models.py の SHORT で、predictions.json の
 * models[].short に載ってくる。ここはそれが無い古い予測ファイル用の控え。
 */
export const MODEL_SHORT = {
  lgbm: 'LGB', xgb: 'XGB', cat: 'CAT', logit: 'LR', mlp: 'NN', rf: 'RF',
};

/**
 * モデルの塊。実測のスコア相関で2つに分かれる（docs/MODEL_LINEUP.md）。
 *
 *   木3つ      相関 0.80〜0.89 / 上位10%の重複 50〜58%
 *   木以外2つ  相関 0.823      / 重複 44.7%
 *   塊をまたぐ 相関 0.63〜0.71 / 重複 29〜33%
 *
 * 木3本が揃って高いのは「同じ見方が3回出ている」だけで、3つの独立した
 * 賛成ではない。塊をまたいで揃ったときだけ、見方の違うモデルが同じ結論に
 * 達したと読める。その読み違いを防ぐために画面で区切る。
 */
export const MODEL_FAMILY = {
  lgbm: 'tree', xgb: 'tree', cat: 'tree', rf: 'tree',
  logit: 'other', mlp: 'other',
};

export const FAMILY_JA = { tree: '決定木系', other: '木以外' };

/**
 * 候補1件を「モデル別に並べられる形」にする。
 *
 * 混ぜない（アンサンブルにしない）。学習器が違えばスコアのスケールも
 * 意味も違うので、各モデル自身の過去スコア分布での位置に揃えて返す。
 * 買うかどうかは、並んだモデル（2026-10-09 からはツリー系3つ）を見て人間が決める。
 *
 * 並び順は payload の models（= research/models.py の ALGOS 順）に従う。
 * 候補ごとにスコア順で並べ替えると、行をまたいで同じ位置が同じモデルに
 * ならず、縦に読めなくなる。
 */
export function modelRows(candidate, models) {
  const per = candidate?.byModel;
  if (!per) return [];
  const order = (models?.length ? models.map((m) => m.algo) : Object.keys(per))
    .filter((a) => per[a]);
  const meta = new Map((models || []).map((m) => [m.algo, m]));
  return order.map((algo) => ({
    algo,
    name: meta.get(algo)?.name || algo,
    note: meta.get(algo)?.note || '',
    short: meta.get(algo)?.short || MODEL_SHORT[algo]
           || algo.slice(0, 3).toUpperCase(),
    family: MODEL_FAMILY[algo] || 'other',
    pct: per[algo].pctHistorical,
    score: per[algo].score,
  }));
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
      // 進捗期待。predict_daily がデータ取得と同じ関数で出した値（基準つき）。
      // 2026-09-25 より前の predictions.json の progressRate は「進捗率 − Q×25」で
      // 別物だった。そのころの候補は quarter を持たないので点数は付かない
      progressRate: n(c.progressRate),
      quarter: n(c.quarter),
      progressBenchmark: n(c.progressBenchmark),
      progressBasis: typeof c.progressBasis === 'string' ? c.progressBasis : null,
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
