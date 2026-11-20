/**
 * サマリー（一番左のタブ）。運用者の依頼（2026-11-21）「パット見で、すべての良し悪しを測りたいので、レーダーチャートで、
 * 下記の項目を頂点とした六角形で表示してほしいです。タイムマシーンのタグのチャートとは別です。6分類モデルの平均、最低値、偏差、
 * ランキング学習モデルのランキング、理論株価から割安度です」。
 *
 * 候補ごとに6つの頂点を 0〜1 にそろえる（どれも外側ほど良い）。選択（AskUserQuestion 2026-11-21）:
 *   6つ目の頂点  「ランキングを日次と5日に分ける」
 *   分類の物差し 「過去の分布の百分位」（各モデルの pctHistorical。今の買いの規則と同じ物差し）
 *   ランキング   「その日の候補の中の順位」（日次はその日の候補の中、5日は直近5営業日の組の中）
 *   割安度       「直近5日の予測対象の全銘柄内での相対評価にしてほしいです」→「直近5営業日の候補すべて」
 *                （理論株価の計算と「理論株価」のタブも5営業日に広げた。選択「タブも5営業日に広げる」）
 * docs/MODEL_ADOPTION_RULES.md §100。買いの規則・買う銘柄は変えない（並べて見るだけ）。
 */
import { rankedValuations } from './valuation.js';

/** 6本の分類モデル（+10% の木3本と +15% の木3本）。 */
export const SUMMARY_MODELS = ['lgbm', 'xgb', 'cat', 'lgbm_k15', 'xgb_k15', 'cat_k15'];

/** 偏差の頂点の目盛り: 1 − 標準偏差÷SD_SCALE（SD_SCALE 以上は 0）。6本の百分位の標準偏差（母標準偏差。6で割る）。 */
export const SD_SCALE = 20;

/** 割安度に使う理論株価の算出方法（RIM = 残余利益法。理論株価のタブの既定と同じ）。 */
export const VALUE_METHOD = 'ri';

/** 六角形の頂点（時計回り。真上から）。short は図の中の短い名前。 */
export const SUMMARY_AXES = [
  { key: 'mean', label: '平均（6本）', short: '平均' },
  { key: 'min', label: '最低値（6本）', short: '最低値' },
  { key: 'agree', label: '偏差（6本のそろい）', short: '偏差' },
  { key: 'ltr', label: 'ランキング（日次）', short: '日次' },
  { key: 'ltr5', label: 'ランキング（5日）', short: '5日' },
  { key: 'value', label: '割安度（RIM）', short: '割安度' },
];

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * 6本の分類モデルの3つの頂点。6本すべてに百分位（pctHistorical。0〜100）があるときだけ値を返す（1本でも無ければ3つとも null）。
 *   mean   6本の平均 ÷ 100
 *   min    6本の最小 ÷ 100
 *   agree  1 − 標準偏差 ÷ SD_SCALE（0〜1 に収める。そろっているほど外側）
 * raw は表に出す元の値（mean・min・sd は百分位の単位。minModel は最小のモデル）。
 */
export function classAxes(candidate, models = SUMMARY_MODELS) {
  const pcts = models.map((a) => candidate?.byModel?.[a]?.pctHistorical);
  if (!pcts.length || !pcts.every(isNum)) return { mean: null, min: null, agree: null, raw: null };
  const mean = pcts.reduce((x, y) => x + y, 0) / pcts.length;
  const min = Math.min(...pcts);
  const sd = Math.sqrt(pcts.reduce((s, p) => s + (p - mean) ** 2, 0) / pcts.length);
  return {
    mean: clamp01(mean / 100),
    min: clamp01(min / 100),
    agree: clamp01(1 - sd / SD_SCALE),
    raw: { mean, min, sd, minModel: models[pcts.indexOf(min)] },
  };
}

/**
 * ランキング学習（日次）の頂点: その日の候補の中の百分位（予測の ltr.dayPct ÷ 100。1位 = 1・最下位 = 0・候補が1件なら 0.5）。
 * 無ければ null。raw は順位と件数。
 */
export function ltrAxis(candidate) {
  const l = candidate?.ltr;
  if (!l || !isNum(l.dayPct)) return { value: null, raw: null };
  return { value: clamp01(l.dayPct / 100), raw: { rank: l.dayRank ?? null, n: l.nDay ?? null } };
}

/**
 * ランキング学習（直近5営業日の組）の頂点: 組の中の百分位（予測の ltr5.pct ÷ 100。1位 = 1・最下位 = 0）。無ければ null。
 */
export function ltr5Axis(candidate) {
  const l = candidate?.ltr5;
  if (!l || !isNum(l.pct)) return { value: null, raw: null };
  return { value: clamp01(l.pct / 100), raw: { rank: l.rank ?? null, n: l.nPool ?? null } };
}

/**
 * 割安度の頂点（銘柄コード jqCode → { score, ratio, rank, n }）。母集団は予測ファイルの直近5営業日（pred.dates）の候補すべて
 * （同じ銘柄は1件）のうち、RIM の「使用終値 ÷ 理論株価の中央値」（理論株価のタブの priceRatio と同じ）を出せる銘柄。
 * その中で低い（割安な）順の順位 rank（1 = 最も割安。同じ値は順位の平均）を、score = (n − rank) ÷ (n − 1) で 0〜1 にする
 * （最も割安 = 1・最も割高 = 0。1銘柄だけなら 0.5）。算出できない銘柄・母集団の外の銘柄は入れない（画面は「なし」）。
 */
export function valueScores(pred, valuations, method = VALUE_METHOD) {
  const days = new Set(pred?.dates || []);
  const pool = new Set((pred?.candidates || [])
    .filter((c) => days.size === 0 || days.has(c.date))
    .map((c) => String(c.jqCode)));
  const seen = new Set();
  const ranked = rankedValuations(valuations?.stocks || [], method)
    .filter((s) => pool.has(String(s.code)) && isNum(s.priceRatio))
    .filter((s) => (seen.has(String(s.code)) ? false : seen.add(String(s.code))));
  const n = ranked.length;
  const out = new Map();
  for (let i = 0; i < n;) {
    let j = i;
    while (j + 1 < n && ranked[j + 1].priceRatio === ranked[i].priceRatio) j += 1;
    const rank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k += 1) {
      out.set(String(ranked[k].code), {
        score: n === 1 ? 0.5 : (n - rank) / (n - 1), ratio: ranked[k].priceRatio, rank, n,
      });
    }
    i = j + 1;
  }
  return out;
}

/**
 * 候補1件のサマリー: { c, axes: { mean, min, agree, ltr, ltr5, value }（0〜1 か null）, raw: { cls, ltr, ltr5, value } }。
 * values は valueScores の戻り値。
 */
export function summaryRow(candidate, values) {
  const cls = classAxes(candidate);
  const d = ltrAxis(candidate);
  const w = ltr5Axis(candidate);
  const v = values?.get(String(candidate?.jqCode)) ?? null;
  return {
    c: candidate,
    axes: { mean: cls.mean, min: cls.min, agree: cls.agree, ltr: d.value, ltr5: w.value, value: v ? v.score : null },
    raw: { cls: cls.raw, ltr: d.raw, ltr5: w.raw, value: v },
  };
}
