/**
 * 運用の戦略（docs/PLAYBOOK.md）を画面で再現する。
 *
 * 実測（out-of-fold 2021-11〜2026-08、14,966件）から決めた手順:
 *
 *   1. その日の発火数（78週高値を更新した銘柄数）で、その日を取るか決める
 *        20件以上 … 本命。選定銘柄の持ち切り +3.60%、正例率 39%
 *         8〜19件 … 枠が空いていれば取る（+1.8〜2.0%）
 *         7件以下 … 見送る（上位2件でも +0.02%、−10%割れが 11.9%）
 *   2. ブースティング3モデル（LightGBM / XGBoost / CatBoost）**すべて**が
 *      過去分布の上位10%（90パーセンタイル以上）
 *   3. 通った銘柄を「3モデルの最小順位」で並べて上位1〜2件
 *        lgbm 単体の順位で並べると2番手が +0.87% まで落ちる（最小順位なら +2.99%）
 *   4. 翌営業日の寄りで買う
 *   5. +20% の指値、届かなければ 20営業日で手仕舞い
 *
 * ここは純粋な変換だけを置く（単体テストできるように）。表示は
 * views/PredictionView.jsx。
 */

/** ブースティング3モデル。合議の対象はこの3つだけ（線形・MLP は入れない）。 */
export const BOOST = ['lgbm', 'xgb', 'cat'];

export const STRATEGY = {
  agreePct: 90,      // 3モデルすべてがこの百分位以上
  strongBreaks: 20,  // この件数以上の発火なら本命
  skipBreaks: 8,     // これ未満の発火なら見送る
  topK: 2,           // 買うのは上位何件まで
  takeProfit: 20,    // 利確の指値（%）
  holdDays: 20,      // 届かなければ何営業日で手仕舞いするか
};

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
             note: `発火 ${nBreak} 件。悪くはない帯（8〜19件、+1.8〜2.0%）` };
  }
  return { kind: 'skip', tone: 'red', label: '見送り',
           note: `発火 ${nBreak} 件。7件以下の日は上位2件でも +0.02%、`
                 + '−10%割れが 11.9%' };
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
