/**
 * 運用の戦略（docs/PLAYBOOK.md）を画面で再現する。
 *
 * 手順（2026-10-12 に選定の線を運用者の決定で 1.5σ の3本に変え、2026-10-23 に運用者の依頼で 1.2σ の3本に、2026-10-29 に
 * 運用者の依頼で 1.5σ の3本に戻した。下の 1 の最初の数字は 2026-10-12 より前の規則で測ったもの）:
 *
 *   1. その日の発火数（78週高値を更新した銘柄数）を見る
 *        20件以上 … 本命。8〜19件 … 枠が空いていれば取る。7件以下 … 発火が少ない日（枠が空いていれば取る）
 *      2026-10-12 までは 7件以下の日は見送っていた。運用者の決定（「Aでお願いします」）で、発火の数では見送らない
 *      以前の規則（1.2σ の3モデル 90以上、out-of-fold 2021-11〜2026-08）では、20件以上 +3.27%・勝率63%、
 *      8〜19件 +1.0〜2.2%、7件以下は142件で −0.11%・−10%割れ 14.1%（7件以下と8件以上の差は z≈2.6）。
 *      1.5σ の3本 95以上（2026-10-12〜10-23 の規則）で前向きの百分位で測り直すと（実験88）、7件以下の日 36件 +3.71%・
 *      8件以上 523件 +3.05% で、見送りの線を支える数字は無かった（だから見送りをやめた）。
 *      今の規則（1.5σ の3本 95以上、画面と同じ百分位、360列の OOF）では 20件以上 218件 +4.25%、8〜19件 116件 +2.22%、
 *      7件以下は20件と少なく +4.22%（+10% の指値・20営業日目の終値。research/exp/probe_k15_panel.py）。前向きの百分位では
 *      7件以下 32件 +2.16%（−10%割れ 12.5%）・8件以上 529件 +2.74%（5.9%）で、7件以下が低い（見送りの線は変えていない）。
 *      2026-10-23〜10-29 の規則（1.2σ の3本 95以上）は、同じ OOF の前向きで 7件以下 23件 +0.61%（30.4%）・8件以上 406件 +2.99%。
 *      **20件未満を全部見送るのは誤り。** 空き枠の収益は 0% なので、
 *      プラスの帯を捨てて枠を遊ばせると損になる（実験35）
 *   2. 1.5σ のラベルで学んだ木の3モデル（RULE_MODELS: LightGBM / XGBoost / CatBoost の 1.5σ 版）**すべて**が過去分布の95以上
 *      （運用者の依頼 2026-10-29「候補銘柄の選定モデルに、1.2σではなく、1.5σのモデルにしてほしいです。」→「すぐ（今夜から）」。
 *      2026-10-12〜10-23 も同じ。2026-10-23〜10-29 は 1.2σ の3本すべてが95以上、2026-10-12 より前は 1.2σ の3本すべてが90以上）。
 *      切り替える前に今の OOF（360列）・前向きで測り直すと、今の手順（3枠・発火の数を問わない）では 1.5σ 115取引 +1.84%・
 *      1.2σ 98取引 +0.90%、候補全体の平均は 561件 +2.70%・429件 +2.86%（窓ごとの差 1.5σ − 1.2σ は +0.85 ± 0.79pt）。
 *      先に決めた判定（候補全体と3枠の両方で 1.5σ が上）は満たさず、それを見た上での運用者の判断（probe_k15_panel、
 *      docs/MODEL_ADOPTION_RULES.md §81）
 *   3. 通った銘柄を「3モデルの最小順位」で並べて上位1〜2件（同点は 1.2σ の LightGBM のスコア順）
 *        lgbm 単体の順位で並べると2番手が +0.87% まで落ちる（最小順位なら +2.99%。以前の規則で測ったもの）
 *   4. 翌営業日の寄りで買う
 *   5. +10% の指値、届かなければ 20営業日目の終値で売る（2026-10-12 に運用者の実際の売りに合わせた。
 *      それまでは +20% の指値・届かなければ 20営業日目までの5日平均。research/live_track.py も同じ）
 *   6. 同時に持つのは 3銘柄まで。枠が埋まっていたら**見送る**
 *        保有中の銘柄は、より良い候補が出ても切らない。実測（実験34）で
 *        乗り換え37回のうち24回（65%）は切らないほうが良く、切った銘柄が
 *        そのまま持っていれば得ていた分は平均 +5.69%、実現は +3.20%。
 *        「含み損でなければ」という条件は、**勝っている玉しか切れない**
 *        ことを意味するため、構造的に勝ち玉を刈ることになる
 *
 * ここは純粋な変換だけを置く（単体テストできるように）。表示は
 * views/PredictionView.jsx。
 */

/** 1.2σ のラベルのブースティング3モデル。2026-10-23〜10-29 の規則・2026-10-12 までの規則・実験の再現に使う。 */
export const BOOST = ['lgbm', 'xgb', 'cat'];

/**
 * 今日の戦略の規則に使う3モデル（2026-10-29 から）。1.5σ のラベルで学んだ木の3本（運用者の依頼「候補銘柄の選定モデルに、
 * 1.2σではなく、1.5σのモデルにしてほしいです」。2026-10-12〜10-23 も同じ。2026-10-23〜10-29 は 1.2σ の3本 lgbm / xgb / cat）。
 * research/live_track.py の RULE_MODELS と同じ（tests/test_live_track.py が突き合わせる）。
 */
export const RULE_MODELS = ['lgbm_k15', 'xgb_k15', 'cat_k15'];

/** 規則の3モデルの呼び名（画面の文言） */
export const RULE_JA = '1.5σ の3本';

/** 規則の3モデルの短い記号（予測ファイルのモデルの short。research/models.py と同じ） */
export const RULE_SHORT = 'L15・X15・C15';

/** 規則のラベル（際どい印の頭に付ける） */
export const RULE_SIGMA = '1.5σ';

/** モデル別の表で規則の3本が並ぶ行（predictions.js の modelGrid の label。1.2σ の行は 'base'） */
export const RULE_ROW = 'k15';

/** 画面に出す短い呼び名。正本は research/models.py。 */
export const MODEL_JA = {
  lgbm: 'LightGBM', xgb: 'XGBoost', cat: 'CatBoost',
  lgbm_k15: 'LightGBM（1.5σ）', xgb_k15: 'XGBoost（1.5σ）', cat_k15: 'CatBoost（1.5σ）',
};

/**
 * 決算の寄与（水準＋変化）。正本は research/feature_dict.py の MACRO。
 *
 * 決算は「モデルの外の材料」ではない。本番153列のうち118列（77%）が
 * 決算由来なので、候補を見て別途決算を確かめるのは、モデルが既に読んだ
 * ものを読み直すことになる。どちらに効いているかを画面に出す。
 */
export const FUND_GROUPS = ['決算（水準）', '決算（変化）'];

export function fundContrib(candidate) {
  const g = candidate?.contrib?.groups;
  if (!g) return null;
  const v = FUND_GROUPS.map((k) => g[k]).filter((x) => Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) : null;
}

export const STRATEGY = {
  agreePct: 95,      // 規則の3モデル（RULE_MODELS）すべてがこの百分位以上。2026-10-12 に 1.2σ の3本 90 → 1.5σ の3本 95、
                     // 2026-10-23 に 1.2σ の3本 95、2026-10-29 に 1.5σ の3本 95
  strongBreaks: 20,  // この件数以上の発火なら本命
  skipBreaks: 1,     // これ未満の発火なら見送る。2026-10-12 に 8 → 1（発火の数では見送らない。運用者の決定「Aでお願いします」・実験88）
  quietBreaks: 8,    // これ未満は「発火が少ない日」（買うが、件数が少なく当たり外れが大きいと添える。以前の見送りの線）
  topK: 2,           // 買うのは上位何件まで
  takeProfit: 10,    // 利確の指値（%）。2026-10-12 に 20 → 10（運用者の実際の売り）
  holdDays: 20,      // 届かなければ何営業日目の終値で売るか
  maxSlots: 3,       // 同時に持てる銘柄数。埋まっていたら見送る
  nearLo: 85,        // 「惜しい候補」として画面に出す下限（規則の3モデルの最小が nearLo〜agreePct）
  frozenVol: 0.3,    // 直近20日の日次ボラ（%）がこれ未満なら「値動きが止まっている」注意を出す（TOB 中の可能性。実験63）
};

/**
 * 「値動きが止まっている」注意。直近20日の日次ボラ（candidate.vol20d、%）が s.frozenVol 未満なら文言を返す。
 *
 * TOB の公表後は株価が買付価格に張り付き、日々の値動きがほぼ 0 になる。そのまま「78週高値の更新」として
 * 候補に入り、5モデルは「静かな高値更新」を最高評価するので最上位に来る。実験63（docs/MODEL_SELECTION_EDA.md §8）で
 * 全5モデル 95以上を満たした 97件のうち 20日ボラ 0.3% 未満は 2件あり、どちらも TOB 中で外れた（上がりようがない）。
 * 値が無ければ null（注意を出さない）。選定の規則（strategySignal）は変えない。注意だけ出す。
 */
export function frozenNote(candidate, s = STRATEGY) {
  const raw = candidate?.vol20d;
  if (raw === null || raw === undefined || raw === '') return null;   // 値が無い（Number(null) は 0 になるので先に弾く）
  const v = Number(raw);
  if (!Number.isFinite(v) || v >= s.frozenVol) return null;
  return `値動きなし（20日ボラ ${v.toFixed(2)}%。TOB 中の可能性）`;
}

/**
 * 空いている枠の数。手で入れた保有数から出す。
 *
 * 保有数は玉の本数なので整数のはず。小数や負の値は入力の誤りなので
 * 黙って丸めず null を返す（画面は枠の話をしない）。
 */
export function freeSlots(held, s = STRATEGY) {
  if (!Number.isInteger(held) || held < 0) return null;
  return Math.max(0, s.maxSlots - held);
}

/**
 * models の百分位（既定は今の規則の3モデル）。1つでも欠けていれば null を混ぜて返す
 * （欠けたまま「満たした」と判定しないため）。
 */
export function pctsOf(candidate, models = RULE_MODELS) {
  const per = candidate?.byModel;
  if (!per) return null;
  const out = {};
  for (const a of models) {
    const v = per[a]?.pctHistorical;
    out[a] = Number.isFinite(v) ? v : null;
  }
  return out;
}

/** 1.2σ の3モデルの百分位。 */
export function boostPcts(candidate) {
  return pctsOf(candidate, BOOST);
}

/** 3モデル（既定は今の規則のモデル）の最小の百分位。1つでも欠けていれば null。 */
export function minPct(candidate, models = RULE_MODELS) {
  const p = pctsOf(candidate, models);
  if (!p) return null;
  const vals = models.map((a) => p[a]);
  return vals.some((v) => v === null) ? null : Math.min(...vals);
}

/**
 * 予測タブの一覧の並び順（運用者の決定 2026-10-13「1.5σ の3本のうち一番低い百分位でいきましょう」。それまでは
 * 基準モデル（1.2σ の LightGBM）のスコア順＝rankInDay。2026-10-23〜10-29 は規則が 1.2σ の3本だったので、その3本の最小。
 * 2026-10-29 に規則と一緒に 1.5σ の3本に戻した）。
 *
 * 規則の3モデル（RULE_MODELS）の百分位の最小が高い順。同点は基準モデルのスコアが高い順（strategySignal と同じ）。
 * 3本のどれかが欠けた候補（そのモデルの値が無い日。1.5σ の3本では、1.5σ を予測に入れる前の固定した過去の日）は最小が出せないので後ろに
 * 置き、基準モデルのスコア順に並べる。戻り値は各行に order（1始まりの並び順）と ruleMin（最小。出せなければ
 * null）を足した新しい配列（元の行は変えない）。rankInDay（基準モデルの順位）は帯・較正・予測ログ・追跡の
 * 物差しなので、そのまま残す。
 */
export function orderRows(rows, models = RULE_MODELS) {
  const list = Array.isArray(rows) ? rows : [];
  const byBase = (a, b) => ((b.c.score ?? 0) - (a.c.score ?? 0))
    || ((a.c.rankInDay ?? Infinity) - (b.c.rankInDay ?? Infinity));
  return list
    .map((c) => ({ c, m: minPct(c, models) }))
    .sort((a, b) => {
      if (a.m === null || b.m === null) {
        if (a.m === null && b.m === null) return byBase(a, b);
        return a.m === null ? 1 : -1;
      }
      return (b.m - a.m) || byBase(a, b);
    })
    .map(({ c, m }, i) => ({ ...c, order: i + 1, ruleMin: m }));
}

/**
 * 一覧の順位の数字の色の区分（規則の3本の最小）。'pass' = 規則の線（agreePct）以上、'near' = nearLo〜線
 * （惜しい候補・際どい候補と同じ範囲）、'below' = それ未満。最小が出せなければ null。
 */
export function ruleMinTier(m, s = STRATEGY) {
  if (!Number.isFinite(m)) return null;
  if (m >= s.agreePct) return 'pass';
  if (m >= s.nearLo) return 'near';
  return 'below';
}

/** 3モデルの百分位の幅（最大−最小）。1つでも欠けていれば null。 */
export function pctSpread(candidate, models = RULE_MODELS) {
  const p = pctsOf(candidate, models);
  if (!p) return null;
  const vals = models.map((a) => p[a]);
  if (vals.some((v) => v === null)) return null;
  return Math.max(...vals) - Math.min(...vals);
}

/** 3モデルのうち、いちばん低い評価を出しているモデル。 */
export function laggard(candidate, models = RULE_MODELS) {
  const p = pctsOf(candidate, models);
  if (!p) return null;
  const vals = models.map((a) => p[a]);
  if (vals.some((v) => v === null)) return null;
  return models[vals.indexOf(Math.min(...vals))];
}

/**
 * 規則の3本（RULE_MODELS。2026-10-29 から 1.5σ の3本）で見た「際どい」範囲（運用者の依頼 2026-10-12「1.5σで際どいに
 * してほしいです」で 1.5σ の3本にし、2026-10-23〜10-29 は規則に合わせて 1.2σ の3本、2026-10-29 に規則と一緒に 1.5σ の3本に
 * 戻した）。今の規則の線「3本とも95以上」の下で、
 * 次のどちらか:
 *   - 2本が 95以上で、残り1本が 95未満
 *   - 95以上は1本以下で、3本の最小が 85〜95
 * 例: xgb 96・cat 98・lgbm 85。
 */
export const BORDER = { line: 95, lo: 85 };

/**
 * 際どい候補の過去の成績（research/exp/probe_k15_panel.py の ■4、2026-10-29・run 40876903076・360列の OOF（最後の再学習は
 * 10/24）。前向きの百分位・テスト窓 3〜11（2022-05〜2026-09）・発火の数を問わない。probe_k15_border と同じ形。
 * docs/MODEL_ADOPTION_RULES.md §81）。物差しは +10% の指値（届かなければ20営業日目の終値）の平均（%）。形は「2本95以上（two95）」と
 * 「95以上は1本以下・最小 85〜95（min）」で、それぞれ一番低いモデルで分けた（1.5σ の3本）:
 *   3本とも95以上（線の上）       561件 +2.70%
 *   2本95以上                     502件 +1.92%（一番低いのが LightGBM 154件 +1.54% / XGBoost 208件 +2.07% / CatBoost 140件 +2.11%）
 *   95以上は1本以下・最小 85〜95  894件 +1.41%（LightGBM 323件 +1.21% / XGBoost 342件 +1.31% / CatBoost 229件 +1.84%）
 * どの組もほかの2本との差は SE の範囲（いちばん離れたのが「1本以下で CatBoost が一番低い」+0.58 ± 0.63pt）。
 * 先に決めた赤の線（BORDER_WARN: 20件以上で平均 0% 以下）に当たる組は無いので、今の数字では赤は出ない（全部黄）。
 * 2026-10-23〜10-29 の 1.2σ の3本の数字（probe_t3_panel）は §75、2026-10-12〜10-23 の 1.5σ の3本の数字（probe_k15_border、
 * 249列のころの OOF）は §48。
 */
export const BORDER_STATS = {
  line: { n: 561, ret: 2.70 },                     // 3本とも95以上（線の上）
  two95: { all: { n: 502, ret: 1.92 }, lgbm_k15: { n: 154, ret: 1.54 }, xgb_k15: { n: 208, ret: 2.07 }, cat_k15: { n: 140, ret: 2.11 } },
  min: { all: { n: 894, ret: 1.41 }, lgbm_k15: { n: 323, ret: 1.21 }, xgb_k15: { n: 342, ret: 1.31 }, cat_k15: { n: 229, ret: 1.84 } },
};

/** 赤の注意を出す線（先に決めたもの）: 件数が minN 件以上で、平均が ret（%）以下。件数が少ない組は赤にしない。 */
export const BORDER_WARN = { minN: 20, ret: 0 };

/**
 * 際どい候補の形。際どくなければ null。3本の百分位が1つでも欠けていれば null。
 *   two95_hi / two95_mid / two95_lo  2本が95以上、残り1本が 90〜95 / 85〜90 / 85未満
 *   min90 / min85                    95以上は1本以下で、最小が 90〜95 / 85〜90
 */
export function borderShape(candidate, b = BORDER, models = RULE_MODELS) {
  const p = pctsOf(candidate, models);
  if (!p) return null;
  const vals = models.map((a) => p[a]);
  if (vals.some((v) => v === null)) return null;
  const n95 = vals.filter((v) => v >= b.line).length;
  const mn = Math.min(...vals);
  if (n95 === 3) return null;
  if (n95 === 2) return mn >= 90 ? 'two95_hi' : mn >= b.lo ? 'two95_mid' : 'two95_lo';
  if (mn >= 90) return 'min90';
  if (mn >= b.lo) return 'min85';
  return null;
}

/** 収益の表示。小数1桁に丸めてから符号を付ける（−0.01 を「−0.0%」と出さない）。 */
const fmtRet = (v) => {
  const r = Math.round(v * 10) / 10;
  return r === 0 ? '0.0%' : `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}%`;
};

/** 「+1.8%（420件）」。過去に0件なら「0件」、数字が無ければ「（測っていない）」。 */
const past = (x) => {
  if (x?.n === 0) return '0件';
  return Number.isFinite(x?.ret) && x?.n ? `${fmtRet(x.ret)}（${x.n}件）` : '（測っていない）';
};

/** 短い呼び名（「LightGBM（1.5σ）」→「LightGBM」。印の頭と注意の見出しに規則のラベル（RULE_SIGMA）を書くので）。 */
const shortJa = (a) => MODEL_JA[String(a).replace(/_k1[58]$/, '')] || a;

/** 先に決めた線（BORDER_WARN）に当たるか: 件数 minN 以上で平均が ret 以下。 */
export function borderWarn(x, w = BORDER_WARN) {
  return Number.isFinite(x?.ret) && Number.isFinite(x?.n) && x.n >= w.minN && x.ret <= w.ret;
}

/**
 * 際どい候補への注意。一番低いモデルと、その形・そのモデルが一番低いときの過去の成績を添える。際どくなければ null。
 *   warn   その組が先に決めた線（BORDER_WARN）に当たる（過去の平均が 0% 以下 = 空き枠と同じか悪い）
 *   few    その組の件数が BORDER_WARN.minN 未満（目安にしにくい）
 *   badge  行に出す短い文言
 *   text   詳しい文言（行の title と、開いたときの本文）
 * 選定の規則（strategySignal）は変えない。注意だけ出す。
 */
export function borderNote(candidate, b = BORDER, st = BORDER_STATS, w = BORDER_WARN) {
  const shape = borderShape(candidate, b);
  if (!shape) return null;
  const lag = laggard(candidate);
  const p = pctsOf(candidate);
  const name = shortJa(lag);
  const fam = shape.startsWith('two95') ? 'two95' : 'min';
  const g = st?.[fam]?.[lag];
  const warn = borderWarn(g, w);
  const few = !(Number.isFinite(g?.n) && g.n >= w.minN);
  const form = fam === 'two95' ? `2本が${b.line}以上` : `${b.line}以上が1本以下・最小 ${b.lo}〜${b.line} `;
  const others = RULE_MODELS.filter((a) => a !== lag).map((a) => `${shortJa(a)} ${past(st?.[fam]?.[a])}`).join('・');
  const text = `${name} が一番低い（${p[lag].toFixed(1)}）。`
    + `${form}の形で ${name} が一番低いときは、過去 ${past(g)}`
    + (warn ? '。空き枠（0%）と同じか悪い' : '')
    + (few ? '。件数が少なく目安にしにくい' : '')
    + `。ほかが一番低いときは ${others}、3本とも${b.line}以上は ${past(st?.line)}`
    + `（${RULE_JA}・前向きの百分位。+10% の指値で売り、届かなければ20営業日目の終値で売ったときの平均）`;
  return {
    shape, laggard: lag, laggardPct: p[lag], warn, few,
    badge: warn ? `${RULE_SIGMA}で際どい・${name} が最下位` : `${RULE_SIGMA}で際どい・最下位 ${name}`,
    text,
  };
}

/**
 * 基準に届かなかったが惜しい候補（規則の3モデルの最小が nearLo〜agreePct。今は 1.5σ の3本で 85〜95）。
 *
 * 画面に出すのは「なぜ買わないか」を毎日考え直さないため。今の規則（1.5σ の3本、画面と同じ百分位、発火8件以上の日、
 * +10% の指値・20営業日目の終値。research/exp/probe_k15_panel.py、360列の OOF）では、最小 85〜95 で幅5未満 387件 +1.41%・
 * 勝率58.7%、幅5以上 633件 +1.72%・勝率61.5%（線の上の354件は +3.59%）。幅5以上（weak）でも弱くはない。
 * （249列のころの 1.5σ の3本（probe_k15_baseline）は 幅5未満 369件 +1.05%・幅5以上 584件 +2.08%。2026-10-23〜10-29 の
 * 1.2σ の3本（probe_t3_panel）は 幅5未満 335件 +1.49%・幅5以上 576件 +2.28%。）
 * 以下は以前の規則（1.2σ の3モデル、最小 85〜90）の実測（実験36、発火8件以上の日・枠の制約なし）:
 *
 *   最小 85〜90 で 3モデルの幅 < 5   118件 +2.70% 勝率58% −10%割れ 5.1%
 *   最小 85〜90 で 3モデルの幅 5〜10  270件 +0.30% 勝率49% −10%割れ 12.6%
 *   （参考）最小 90以上               943件 +2.84% 勝率62% −10%割れ 5.1%
 *
 * つまり「1つのモデルだけが5〜10pt下」の形は、空き枠（0%）と変わらない。
 * とくに最下位が LightGBM のときが弱い（104件 +0.39%）。
 * ※ 上は 153列のころの実験36。いまの239列のモデル（実験70、2026-10-10）では、85〜90 の帯で最下位が
 *   LightGBM の形は弱くない（LightGBM が最下位 202件 +1.3%、XGBoost / CatBoost が最下位 445件 +0.7%。
 *   形ごとの表の値から合成）。今の 1.5σ の3本（probe_k15_panel、前向き）では、最小 85〜95 の形で一番低いのが
 *   LightGBM の組が +1.21%（323件）といちばん低いが、ほかとの差は SE の範囲（borderNote の BORDER_STATS）。
 */
export function nearMisses(rows, s = STRATEGY) {
  const list = Array.isArray(rows) ? rows : [];
  return list
    .map((c) => ({ c, m: minPct(c), sp: pctSpread(c) }))
    .filter((x) => x.m !== null && x.m >= s.nearLo && x.m < s.agreePct)
    .sort((a, b) => b.m - a.m)
    .map(({ c, m, sp }) => ({
      ...c,
      minPct: m,
      spread: sp,
      laggard: laggard(c),
      // 幅が5以上なら「1つだけ下」の弱い形
      weak: sp !== null && sp >= 5,
    }));
}

/** 規則の3モデル（RULE_MODELS）すべてが pct（既定 95）以上か。 */
export function passesAgree(candidate, pct = STRATEGY.agreePct) {
  const m = minPct(candidate);
  return m !== null && m >= pct;
}

/**
 * ランキング学習（予測の ltr。research/ltr.py）の、その日の候補の中の百分位（0〜100。候補が1件の日は 50）。無ければ null。
 * 2026-10-24 から予測に入る（それより前の日には無い）。参考に並べるだけで、買いの規則には使わない
 * （運用者の選択「まず並べて見る」。docs/MODEL_ADOPTION_RULES.md §78-9）。
 */
export function ltrDayPct(candidate) {
  const v = candidate?.ltr?.dayPct;
  return Number.isFinite(v) ? v : null;
}

/** その日の中でランキング学習が上半分か（予測の ltr.upper。丸める前の百分位で決めてある）。無ければ null。 */
export function ltrUpper(candidate) {
  const u = candidate?.ltr?.upper;
  return typeof u === 'boolean' ? u : null;
}

/**
 * 直近5営業日の組のランキング学習（予測の ltr5。research/ltr5.py。実験117 の W5）の、予測の日の組（その日までの直近5営業日の
 * 候補）の中の順位（1 = 最上位。同じスコアは銘柄コードの順）。無ければ null。2026-11-02 の予測から入る（それより前の日には
 * 無い）。参考に並べるだけで、買いの規則には使わない（運用者の選択「画面に並べて確かめる」。docs/MODEL_ADOPTION_RULES.md §82-6）。
 */
export function ltr5Rank(candidate) {
  const v = candidate?.ltr5?.rank;
  return Number.isInteger(v) ? v : null;
}

/** 直近5営業日の組の行の数（予測の ltr5.nPool）。無ければ null。 */
export function ltr5Pool(candidate) {
  const v = candidate?.ltr5?.nPool;
  return Number.isInteger(v) ? v : null;
}

/**
 * 3条件・4条件の印（予測の conds。research/ltr5.py の conds。実験117 の参考の表と同じ組み合わせ）。
 *   'c4'  4条件: その日の候補の中で日次のランキング学習が1位・直近5営業日の組の中で1位・1.5σ の木3本がすべて95以上・
 *         1.2σ の木3本もすべて95以上
 *   'c3'  3条件（4条件ではない）: 上の3つ（1.2σ は問わない）
 *   null  どちらでもない・値が無い
 */
export function condLevel(candidate) {
  const c = candidate?.conds;
  if (!c) return null;
  if (c.c4 === true) return 'c4';
  if (c.c3 === true) return 'c3';
  return null;
}

/**
 * 両方使う形（参考）: 今の規則（RULE_MODELS がすべて agreePct 以上）を満たし、かつその日の中でランキング学習が上半分。
 * 実験114b の R0D（§78-8）。買う銘柄は strategySignal のまま変えない。research/live_track.py の ■7 と同じ判定。
 */
export function passesBoth(candidate, s = STRATEGY) {
  return passesAgree(candidate, s.agreePct) && ltrUpper(candidate) === true;
}

/**
 * その日の「両方使う形」の要約。rows はその日の候補、signal はその日の strategySignal。
 *   available  その日の候補にランキング学習の値があるか（無い日は数えない）
 *   both       今の規則を満たした候補（signal.passed）のうち、ランキング学習が上半分のもの（並びは passed のまま）
 */
export function bothSummary(rows, signal) {
  const list = Array.isArray(rows) ? rows : [];
  const passed = signal?.passed ?? [];
  const available = list.length > 0 && list.every((c) => ltrUpper(c) !== null);
  return { available, both: available ? passed.filter((c) => ltrUpper(c) === true) : [], nPassed: passed.length };
}

/**
 * ランキング学習の順位表（予測タブ。運用者の依頼 2026-10-29「ランキング学習モデルによるランキングをわかりやくどこかに
 * 表示してください」→「予測タブに順位表」）。その日の候補を、ランキング学習のその日の中の順位（ltr.dayRank。1 = 最上位）の
 * 順に並べる。同じ順位は銘柄コードの順。参考に並べるだけで、買う銘柄は今日の戦略の規則のまま。
 *   available  その日の候補すべてに順位がある（1件でも欠ければ出さない。bothSummary と同じ扱い）
 *   items      { c, rank, nDay, dayPct, upper（上半分）, rule（今の規則を満たす）, both（両方使う形）,
 *                rank5・pool5（直近5営業日の組の中の順位と組の行の数。無ければ null）, cond（condLevel） } の並び
 *   nUpper     上半分の件数（表に区切りの線を引く位置）
 */
export function ltrRanking(rows, s = STRATEGY) {
  const list = Array.isArray(rows) ? rows : [];
  const has = (c) => Number.isInteger(c?.ltr?.dayRank) && ltrUpper(c) !== null;
  if (!list.length || !list.every(has)) return { available: false, items: [], nUpper: 0 };
  const items = list.map((c) => {
    const rule = passesAgree(c, s.agreePct);
    const upper = ltrUpper(c) === true;
    return { c, rank: c.ltr.dayRank, nDay: Number.isInteger(c.ltr.nDay) ? c.ltr.nDay : list.length,
             dayPct: ltrDayPct(c), upper, rule, both: rule && upper,
             rank5: ltr5Rank(c), pool5: ltr5Pool(c), cond: condLevel(c) };
  });
  items.sort((a, b) => (a.rank - b.rank) || String(a.c.code ?? '').localeCompare(String(b.c.code ?? '')));
  return { available: true, items, nUpper: items.filter((x) => x.upper).length };
}

/**
 * その日の発火数の帯。発火数だけで決める（2026-10-12 からは発火の数では見送らない。skipBreaks = 1）。
 *   strong  20件以上（本命）/ ok  8〜19件 / quiet  1〜7件（発火が少ない日。買うが件数が少ないと添える）/ none  0件
 * 地合いの条件は足さない（発火15件以上の日は100%が TOPIX 200日線より上で、
 * 発火数が地合いを内包している。docs/MODEL_DAY_REGIME.md）。
 */
export function dayVerdict(nBreak, s = STRATEGY) {
  if (!Number.isFinite(nBreak) || nBreak <= 0) {
    return { kind: 'none', tone: 'slate', label: '発火なし',
             note: 'この日は新規の高値更新がありません' };
  }
  if (nBreak >= s.strongBreaks) {
    return { kind: 'strong', tone: 'green', label: '本命の日',
             note: `発火 ${nBreak} 件。20件以上の帯（${RULE_JA}が95以上で218件 +4.25%・勝率74%）` };
  }
  if (nBreak >= s.quietBreaks) {
    return { kind: 'ok', tone: 'amber', label: '枠が空いていれば',
             note: `発火 ${nBreak} 件。8〜19件の帯（116件 +2.22%・勝率62%）。`
                   + '枠が遊ぶより買うほうが良い' };
  }
  if (nBreak >= s.skipBreaks) {
    return { kind: 'quiet', tone: 'amber', label: '発火が少ない日',
             note: `発火 ${nBreak} 件。枠が空いていれば買う（2026-10-12 までは見送っていた）。`
                   + `7件以下の帯は件数が少ない（20件 +4.22%・勝率65%。その時点の百分位で測ると32件 +2.16%・`
                   + '−10%割れ 12.5%と当たり外れが大きい）' };
  }
  return { kind: 'skip', tone: 'red', label: '見送り',
           note: `発火 ${nBreak} 件。${s.skipBreaks - 1}件以下の日は枠が空いていても買わない` };
}

/**
 * その日の戦略シグナル。
 *
 * rows はその日の候補（predictions.json の candidates を日で絞ったもの）。
 * 戻り値の picks が「買う銘柄」、passed が「基準を満たした全銘柄」。
 */
export function strategySignal(rows, s = STRATEGY) {
  const list = Array.isArray(rows) ? rows : [];
  const nBreak = list.length;
  const verdict = dayVerdict(nBreak, s);
  // 規則の3モデルの百分位が1つでも欠けた候補の数。1件でもあればその日は判定できない（買わない）。
  // research/live_track.py の decide（「判定できない」）と同じ。過去の日は、その日に出した値のまま固定するので
  // （research/predict_daily.py の freeze_past_days）、規則のモデル（1.5σ の3本）を予測に入れる前の日は欠ける
  const missing = list.filter((c) => minPct(c) === null).length;
  const passed = list
    .filter((c) => passesAgree(c, s.agreePct))
    .map((c) => ({ ...c, minPct: minPct(c) }))
    // 最小順位の降順。同点なら基準モデルのスコアで決める（表示順を安定させる）
    .sort((a, b) => (b.minPct - a.minPct) || ((b.score ?? 0) - (a.score ?? 0)));
  const buyable = missing === 0
    && (verdict.kind === 'strong' || verdict.kind === 'ok' || verdict.kind === 'quiet');
  return {
    nBreak,
    verdict,
    passed,
    picks: buyable ? passed.slice(0, s.topK) : [],
    // 基準は満たしたが、その日を取らない判断で見送るもの
    held: buyable ? passed.slice(s.topK) : passed,
    buyable,
    missing,
  };
}

/**
 * 想定の出口。買値は翌営業日の寄りなので当日は分からない。
 * 終値を仮の買値として指値の目安を出す（あくまで目安と画面に書く）。
 */
export function exitPlan(close, s = STRATEGY) {
  if (!Number.isFinite(close) || close <= 0) return null;
  return {
    target: close * (1 + s.takeProfit / 100),
    takeProfit: s.takeProfit,
    holdDays: s.holdDays,
  };
}
