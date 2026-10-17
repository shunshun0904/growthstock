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
const SAMPLES_URL = () => `${base()}data/new_samples.json?v=${BUILD}`;
const NEWS_LINKS_URL = () => `${base()}data/news_links.json?v=${BUILD}`;

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
const NAMES_URL = () => `${base()}data/feature_names.json?v=${BUILD}`;

/**
 * 特徴量の日本語名（public/data/feature_names.json。research/feature_dict.py --names-json が書く）。
 * {列名: {name: 短い日本語名, desc: 説明}}。読めなければ空（画面は元の列名のまま出す）。
 */
export async function loadFeatureNames() {
  try {
    const d = await getJson(NAMES_URL());
    return d && typeof d === 'object' ? d : {};
  } catch {
    return {};
  }
}

/**
 * 寄与の行（contrib.top の1件）に出す名前と、ホバーで出す説明。
 * 日本語名は feature_names.json → 行の name（predict_daily が付ける）→ 元の列名 の順で探す。
 * 説明は feature_names.json の desc → 行の ja。説明の末尾に元の列名を添える（モデルの入力の名前と突き合わせるため）。
 */
export function featureLabel(t, names) {
  const col = t?.col ?? '';
  const entry = names?.[col];
  const name = entry?.name || t?.name || col;
  const desc = String(entry?.desc || t?.ja || '').replace(/\*\*/g, '');
  return { name, title: desc ? `${desc}（${col}）` : col, translated: name !== col };
}

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

/**
 * 週次の再学習で加わったサンプルのチャート（public/data/new_samples.json。New Training Samples が
 * research/new_samples.py で書く）。無ければ null（まだ作っていない）。形が違えば cases を空に。
 */
export async function loadNewSamples() {
  try {
    const d = await getJson(SAMPLES_URL());
    return d && Array.isArray(d.cases) ? d : { ...(d || {}), cases: [] };
  } catch {
    return null;
  }
}

/**
 * 候補銘柄のニュースのリンクを確かめた結果（public/data/news_links.json。Check News Links が research/news_links.py で
 * 毎晩の予測の後に書く）。無ければ null（まだ確かめていない）。形が違えば stocks / market を空に。
 */
export async function loadNewsLinks() {
  try {
    const d = await getJson(NEWS_LINKS_URL());
    if (!d || typeof d !== 'object') return null;
    return {
      ...d,
      stocks: d.stocks && typeof d.stocks === 'object' ? d.stocks : {},
      market: Array.isArray(d.market) ? d.market : [],
    };
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
  // ラベル違い（research/models.py の SHORT と同じ。元の記号の頭 + k）
  lgbm_k15: 'L15', xgb_k15: 'X15', cat_k15: 'C15', logit_k15: 'LR15',
  lgbm_k18: 'L18', xgb_k18: 'X18', cat_k18: 'C18', logit_k18: 'LR18',
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
  // ラベル違い（到達しきい値 1.5σ / 1.8σ のラベルで学習した4学習器。research/models.py の VARIANTS）。
  // 1.2σ の4本の「上位10%と見たモデル」の数え上げ（agree90）には入れない（1.5σ・1.8σ は agreeK90 で別に数える）。
  // 2026-10-12 から、今日の戦略の規則は 1.5σ の木3本（strategy.js の RULE_MODELS）がすべて95以上。
  // k ごとに塊を分ける（1.5σ の4本、1.8σ の4本。同じ k の中の並びは 1.2σ と同じ LGB / XGB / CAT / LR）
  lgbm_k15: 'k15', xgb_k15: 'k15', cat_k15: 'k15', logit_k15: 'k15',
  lgbm_k18: 'k18', xgb_k18: 'k18', cat_k18: 'k18', logit_k18: 'k18',
};

export const FAMILY_JA = { tree: '決定木系', other: '木以外', k15: 'ラベル1.5σ', k18: 'ラベル1.8σ' };

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

/**
 * pctColor で塗ったマスに載せる字の色。明るい区分（60 以上）は暗い字、暗い区分（60 未満）は明るい字。
 * コントラスト比（WCAG の式で計算）: 95 / 90 / 80 / 60 の地に --bg の字で 15.8 / 10.7 / 7.6 / 5.3、
 * 30 / 0 の地に --text の字で 5.7 / 8.8。どの組も 4.5 以上（逆の組み合わせは 1.0〜3.1 で読めない）。
 */
export function pctInk(pct) {
  const t = pctTier(pct);
  if (t === null) return 'var(--text-faint)';
  return t >= 60 ? 'var(--bg)' : 'var(--text)';
}

/**
 * スコアそのものの色（運用者の依頼 2026-10-16「目的変数横断での比較ができないので、モデルのスコア自体も表示してほしい。
 * 同じ形式で、青白の濃淡でモデル×目的変数横断で色付け」）。
 *
 * 百分位は各モデル自身の分布での位置なので、ラベル（1.2σ / 1.5σ / 1.8σ）をまたいで水準を比べられない。スコアの表は
 * **その表の12本（学習器 × ラベル）の中の最小〜最大の線形**で色を付け、同じ表の中では同じ色が同じスコアを表す
 * （同日の指示「OOF の下限ではなく、この表の中での最大最小で」。最初は予測ファイルの全候補で1つにしていた）。
 * 色相は百分位の表と同じ青の濃淡（暗い画面なので明るいほど高い）。SCORE_COLORS は --pct-0 / --pct-95 と同じ値。
 * scoreRange は候補の配列を受けるので、1件だけ渡せばその表の範囲になる。
 */
export const SCORE_COLORS = { lo: [30, 58, 138], hi: [219, 234, 254] };   // #1e3a8a → #dbeafe

/** 渡した候補の全モデルのスコアの範囲（候補1件なら、その表の中の最小〜最大）。スコアが1つも無ければ null。 */
export function scoreRange(candidates) {
  let lo = Infinity;
  let hi = -Infinity;
  let n = 0;
  for (const c of candidates || []) {
    for (const v of Object.values(c?.byModel || {})) {
      const x = v?.score;
      if (!Number.isFinite(x)) continue;
      n += 1;
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
  }
  return n ? { lo, hi, n } : null;
}

/** 範囲の中の位置（0〜1）。範囲が無い・値が無ければ null。範囲の幅が 0 なら 1（最大と同じ）。 */
export function scoreT(score, range) {
  if (!range || !Number.isFinite(score)) return null;
  if (range.hi <= range.lo) return 1;
  return Math.min(1, Math.max(0, (score - range.lo) / (range.hi - range.lo)));
}

export function scoreColor(score, range) {
  const t = scoreT(score, range);
  if (t === null) return 'var(--bg-input)';
  const c = SCORE_COLORS.lo.map((a, i) => Math.round(a + (SCORE_COLORS.hi[i] - a) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/** スコアのマスの字の色。明るい地（上半分）は暗い字、暗い地は明るい字。 */
export function scoreInk(score, range) {
  const t = scoreT(score, range);
  if (t === null) return 'var(--text-faint)';
  return t >= 0.5 ? 'var(--bg)' : 'var(--text)';
}

/** ラベル違いの接尾辞（research/models.py の VARIANTS）。無ければ 1.2σ のラベル。 */
const LABEL_SUFFIX = [['_k15', 'k15'], ['_k18', 'k18']];
export const LABEL_JA = { base: '1.2σ', k15: '1.5σ', k18: '1.8σ' };

/**
 * モデル別の百分位を表の形にする（運用者の指示 2026-10-12「12個全てを載せてほしい。棒グラフ形式でなくてよい。
 * 大小が分かればよい」）。行 = 学習に使ったラベル（1.2σ / 1.5σ / 1.8σ）、列 = 学習器（LGB / XGB / CAT / LR）。
 *
 * 列の並びは payload の models の順で、学習器が最初に出た順。予測ファイルに無いモデルのマスは null。
 * 古い予測ファイル（ラベル違いが無い4本・5本）は 1.2σ の1行だけになり、MLP がある日は NN の列が付く。
 */
export function modelGrid(candidate, models) {
  const rows = modelRows(candidate, models);
  const cols = [];
  const lines = new Map();
  for (const r of rows) {
    const hit = LABEL_SUFFIX.find(([sfx]) => r.algo.endsWith(sfx));
    const label = hit ? hit[1] : 'base';
    const learner = hit ? r.algo.slice(0, -hit[0].length) : r.algo;
    if (!cols.includes(learner)) cols.push(learner);
    if (!lines.has(label)) lines.set(label, new Map());
    lines.get(label).set(learner, r);
  }
  return {
    cols: cols.map((l) => ({
      learner: l,
      short: lines.get('base')?.get(l)?.short || MODEL_SHORT[l] || l.slice(0, 3).toUpperCase(),
    })),
    rows: ['base', 'k15', 'k18'].filter((k) => lines.has(k)).map((k) => ({
      label: k,
      labelJa: LABEL_JA[k],
      cells: cols.map((l) => lines.get(k).get(l) || null),
    })),
  };
}


/**
 * 選んだ日の値の素性（research/predict_daily.py が書く dayInfo）。過去の日は、その日に出した値のまま固定している
 * （運用者の指示 2026-10-12「過去の日は最初に出した値のまま固定してください」）。
 *   past        今の予測ファイルの最新の日（asOf）より前の日か
 *   trainedAt   その日の値を出したモデルの学習日。dayInfo が無い（固定を始める前の形の）ファイルは今のモデルの学習日
 *   generatedAt その日の値を出した時刻（無ければ null）
 *   sameModel   今のモデルと同じ学習日か（違えば、今のモデルの訓練件数などはその日には当てはまらない）
 */
export function dayOrigin(data, day) {
  const info = data?.dayInfo?.[day] || null;
  const current = data?.model?.trainedAt || null;
  const trainedAt = info?.trainedAt || current;
  return {
    past: Boolean(data?.asOf) && Boolean(day) && day !== data.asOf,
    trainedAt,
    generatedAt: info?.generatedAt || null,
    sameModel: trainedAt === current,
  };
}

/** 決算の四半期の呼び名（J-Quants の決算発表予定日の FQName） */
export const QUARTER_JA = { FY: '通期', '1Q': '第1四半期', '2Q': '第2四半期', '3Q': '第3四半期' };

/**
 * 次の決算発表の予定（research/predict_daily.py の next_earnings。予測日までに公表済みの予定だけ）。
 * 運用者の依頼 2026-10-13「予測が実質的に、次の決算の予測になっていないか確認するために、次の決算発表までの日数も
 * どこかに表示してほしい」。horizon は買ってから持つ営業日（売りのルールの20営業日）。
 *   known       予定が公表済みか
 *   inWindow    発表日が予測日から horizon 営業日以内（買ってから持つ間に決算が来る）
 *   label/tone  行に出す短い印。text は詳しい文言
 * この項目が無い予測ファイル（その日に出した値のまま固定した過去の日など）は null。
 */
export function earningsNote(c, horizon = 20) {
  if (!c || !Object.prototype.hasOwnProperty.call(c, 'nextEarnDate')) return null;
  const k = c.nextEarnTradingDays;
  if (c.nextEarnDate && Number.isFinite(k)) {
    const q = QUARTER_JA[c.nextEarnQuarter] || c.nextEarnQuarter || '';
    const inWindow = k <= horizon;
    const when = k === 0 ? '予測日の当日' : `${k}営業日後`;
    return {
      known: true, inWindow, tradingDays: k,
      label: k === 0 ? '決算 当日' : `決算 ${k}営業日後`,
      tone: inWindow ? 'amber' : 'slate',
      text: `次の決算発表 ${String(c.nextEarnDate).replace(/-/g, '/')}${q ? `（${q}）` : ''}: ${when}`
        + `${Number.isFinite(c.nextEarnDays) ? `・${c.nextEarnDays}日後` : ''}。`
        + (inWindow ? `買ってから${horizon}営業日の中に入る（決算の結果で動きうる）`
          : `買ってから${horizon}営業日より先`),
    };
  }
  const since = Number.isFinite(c.daysSinceDisc) ? `前回の決算開示から ${c.daysSinceDisc}日。` : '';
  return {
    known: false, inWindow: false, tradingDays: null, label: '決算予定 未公表', tone: 'slate',
    text: '次の決算発表の予定は、予測日までに公表されていない（今年の予定では9割が発表の32日以上前に公表されている）。'
      + since,
  };
}

/** その日の候補のうち、次の決算が horizon 営業日以内の件数。項目の無いファイルなら null。 */
export function earningsSummary(rows, horizon = 20) {
  const notes = (Array.isArray(rows) ? rows : []).map((c) => earningsNote(c, horizon)).filter(Boolean);
  if (!notes.length) return null;
  return { total: notes.length, known: notes.filter((n) => n.known).length,
           soon: notes.filter((n) => n.inWindow).length };
}
