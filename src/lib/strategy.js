/**
 * 運用の戦略（docs/PLAYBOOK.md）を画面で再現する。
 *
 * 実測（out-of-fold 2021-11〜2026-08、14,966件）から決めた手順:
 *
 *   1. その日の発火数（78週高値を更新した銘柄数）で、その日を取るか決める
 *        20件以上 … 本命。持ち切り +3.27%、勝率63%
 *         8〜19件 … 枠が空いていれば取る（+1.0〜2.2%、勝率54〜55%）
 *         7件以下 … 見送る（142件で −0.11%、−10%割れが 14.1%）
 *      7件以下と8件以上の差は 2.44pt / z≈2.6 で、ここだけは線が実在する。
 *      **20件未満を全部見送るのは誤り。** 空き枠の収益は 0% なので、
 *      +2% の帯を捨てて枠を遊ばせると損になる（実験35）
 *   2. ブースティング3モデル（LightGBM / XGBoost / CatBoost）**すべて**が
 *      過去分布の上位10%（90パーセンタイル以上）
 *   3. 通った銘柄を「3モデルの最小順位」で並べて上位1〜2件
 *        lgbm 単体の順位で並べると2番手が +0.87% まで落ちる（最小順位なら +2.99%）
 *   4. 翌営業日の寄りで買う
 *   5. +20% の指値、届かなければ 20営業日で手仕舞い
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

/** ブースティング3モデル。合議の対象はこの3つだけ（線形・MLP は入れない）。 */
export const BOOST = ['lgbm', 'xgb', 'cat'];

/** 画面に出す短い呼び名。正本は research/models.py。 */
export const MODEL_JA = { lgbm: 'LightGBM', xgb: 'XGBoost', cat: 'CatBoost' };

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
  agreePct: 90,      // 3モデルすべてがこの百分位以上
  strongBreaks: 20,  // この件数以上の発火なら本命
  skipBreaks: 8,     // これ未満の発火なら見送る
  topK: 2,           // 買うのは上位何件まで
  takeProfit: 20,    // 利確の指値（%）
  holdDays: 20,      // 届かなければ何営業日で手仕舞いするか
  maxSlots: 3,       // 同時に持てる銘柄数。埋まっていたら見送る
  nearLo: 85,        // 「惜しい候補」として画面に出す下限
};

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
 * 3モデルの百分位。1つでも欠けていれば null を混ぜて返す
 * （欠けたまま「満たした」と判定しないため）。
 */
export function boostPcts(candidate) {
  const per = candidate?.byModel;
  if (!per) return null;
  const out = {};
  for (const a of BOOST) {
    const v = per[a]?.pctHistorical;
    out[a] = Number.isFinite(v) ? v : null;
  }
  return out;
}

/** 3モデルの最小の百分位。1つでも欠けていれば null。 */
export function minPct(candidate) {
  const p = boostPcts(candidate);
  if (!p) return null;
  const vals = BOOST.map((a) => p[a]);
  return vals.some((v) => v === null) ? null : Math.min(...vals);
}

/** 3モデルの百分位の幅（最大−最小）。1つでも欠けていれば null。 */
export function pctSpread(candidate) {
  const p = boostPcts(candidate);
  if (!p) return null;
  const vals = BOOST.map((a) => p[a]);
  if (vals.some((v) => v === null)) return null;
  return Math.max(...vals) - Math.min(...vals);
}

/** 3モデルのうち、いちばん低い評価を出しているモデル。 */
export function laggard(candidate) {
  const p = boostPcts(candidate);
  if (!p) return null;
  const vals = BOOST.map((a) => p[a]);
  if (vals.some((v) => v === null)) return null;
  return BOOST[vals.indexOf(Math.min(...vals))];
}

/**
 * 基準に届かなかったが惜しい候補（3モデルの最小が 85〜90）。
 *
 * 画面に出すのは「なぜ買わないか」を毎日考え直さないため。実測（実験36、
 * 発火8件以上の日・枠の制約なし）:
 *
 *   最小 85〜90 で 3モデルの幅 < 5   118件 +2.70% 勝率58% −10%割れ 5.1%
 *   最小 85〜90 で 3モデルの幅 5〜10  270件 +0.30% 勝率49% −10%割れ 12.6%
 *   （参考）最小 90以上               943件 +2.84% 勝率62% −10%割れ 5.1%
 *
 * つまり「1つのモデルだけが5〜10pt下」の形は、空き枠（0%）と変わらない。
 * とくに最下位が LightGBM のときが弱い（104件 +0.39%）。
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

/** 3モデルすべてが上位10%か。 */
export function passesAgree(candidate, pct = STRATEGY.agreePct) {
  const m = minPct(candidate);
  return m !== null && m >= pct;
}

/**
 * その日を取るかの判定。発火数だけで決める。
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
             note: `発火 ${nBreak} 件。実測でいちばん成績が良い帯（20件以上）` };
  }
  if (nBreak >= s.skipBreaks) {
    return { kind: 'ok', tone: 'amber', label: '枠が空いていれば',
             note: `発火 ${nBreak} 件。悪くはない帯（8〜19件、+1.0〜2.2%）。`
                   + '枠が遊ぶより買うほうが良い' };
  }
  return { kind: 'skip', tone: 'red', label: '見送り',
           note: `発火 ${nBreak} 件。7件以下は142件で −0.11%、`
                 + '−10%割れが 14.1%。枠が空いていても買わない' };
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
  const passed = list
    .filter((c) => passesAgree(c, s.agreePct))
    .map((c) => ({ ...c, minPct: minPct(c) }))
    // 最小順位の降順。同点なら基準モデルのスコアで決める（表示順を安定させる）
    .sort((a, b) => (b.minPct - a.minPct) || ((b.score ?? 0) - (a.score ?? 0)));
  const buyable = verdict.kind === 'strong' || verdict.kind === 'ok';
  return {
    nBreak,
    verdict,
    passed,
    picks: buyable ? passed.slice(0, s.topK) : [],
    // 基準は満たしたが、その日を取らない判断で見送るもの
    held: buyable ? passed.slice(s.topK) : passed,
    buyable,
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
