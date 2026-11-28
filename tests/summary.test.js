/**
 * サマリー（src/lib/summary.js）の単体テスト。六角形の6つの頂点の 0〜1 のそろえ方（運用者の選択 2026-11-21）。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SUMMARY_MODELS, SUMMARY_AXES, SD_SCALE, VALUE_METHOD, classAxes, ltrAxis, ltr5Axis, valueScores, summaryRow,
} from '../src/lib/summary.js';

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≠ ${b}`);
const cand = (jqCode, pcts, extra = {}) => ({
  code: jqCode.slice(0, 4), jqCode, date: '2026-11-19',
  byModel: Object.fromEntries(SUMMARY_MODELS.map((a, i) => [a, { pctHistorical: pcts[i] }])), ...extra,
});

test('頂点は6つ（平均・最低値・偏差・日次・5日・割安度）。6本は +10% と +15% の木3本ずつ', () => {
  assert.deepEqual(SUMMARY_AXES.map((a) => a.key), ['mean', 'min', 'agree', 'ltr', 'ltr5', 'value']);
  assert.deepEqual(SUMMARY_MODELS, ['lgbm', 'xgb', 'cat', 'lgbm_k15', 'xgb_k15', 'cat_k15']);
  assert.equal(SD_SCALE, 20);
  assert.equal(VALUE_METHOD, 'ri');
});

test('classAxes: 6本の百分位の平均÷100・最小÷100・1−標準偏差÷20（母標準偏差）', () => {
  const k = classAxes(cand('11110', [90, 80, 70, 90, 80, 70]));
  near(k.mean, 0.8);
  near(k.min, 0.7);
  const sd = Math.sqrt((100 + 0 + 100 + 100 + 0 + 100) / 6);              // 8.1650
  near(k.raw.sd, sd);
  near(k.agree, 1 - sd / 20);
  assert.equal(k.raw.minModel, 'cat');                                      // 最小の最初のモデル
  // そろっていれば 1、ばらつきが 20 以上なら 0
  assert.equal(classAxes(cand('22220', [95, 95, 95, 95, 95, 95])).agree, 1);
  assert.equal(classAxes(cand('33330', [0, 100, 0, 100, 0, 100])).agree, 0);
  near(classAxes(cand('44440', [100, 100, 100, 100, 100, 100])).mean, 1);
});

test('classAxes: 1本でも百分位が無ければ3つとも null（6本そろわないと出さない）', () => {
  const c = cand('11110', [90, 80, 70, 90, 80, 70]);
  delete c.byModel.cat_k15;
  assert.deepEqual(classAxes(c), { mean: null, min: null, agree: null, raw: null });
  assert.equal(classAxes(cand('22220', [90, 80, 70, 90, 80, NaN])).mean, null);
  assert.equal(classAxes(cand('33330', [90, 80, 70, 90, 80, '70'])).min, null);
  assert.equal(classAxes(null).agree, null);
});

test('ltrAxis / ltr5Axis: その日の候補の中・直近5営業日の組の中の百分位 ÷ 100。無ければ null', () => {
  assert.deepEqual(ltrAxis({ ltr: { dayPct: 75, dayRank: 2, nDay: 5 } }), { value: 0.75, raw: { rank: 2, n: 5 } });
  assert.equal(ltrAxis({ ltr: { dayPct: 50, dayRank: 1, nDay: 1 } }).value, 0.5);    // 候補が1件の日
  assert.equal(ltrAxis({ ltr: { dayPct: 100, dayRank: 1, nDay: 9 } }).value, 1);
  assert.equal(ltrAxis({ ltr: { dayPct: null } }).value, null);
  assert.equal(ltrAxis({}).value, null);
  assert.deepEqual(ltr5Axis({ ltr5: { pct: 87.5, rank: 2, nPool: 9 } }), { value: 0.875, raw: { rank: 2, n: 9 } });
  assert.equal(ltr5Axis({ ltr5: { rank: 2 } }).value, null);
  assert.equal(ltr5Axis(null).value, null);
});

const val = (code, ri, price = 100, available = true) => ({
  code, price, methods: [{ id: 'ri', available, median: ri }, { id: 'ddm', available: true, median: 1 }],
});

test('valueScores: 直近5営業日の候補の中で「使用終値÷理論株価の中央値（RIM）」が低いほど 1、高いほど 0', () => {
  const pred = {
    dates: ['2026-11-18', '2026-11-19'],
    candidates: [
      { jqCode: 'A0000', date: '2026-11-18' }, { jqCode: 'B0000', date: '2026-11-19' },
      { jqCode: 'C0000', date: '2026-11-19' }, { jqCode: 'D0000', date: '2026-11-19' },
      { jqCode: 'E0000', date: '2026-11-19' }, { jqCode: 'A0000', date: '2026-11-19' },   // 同じ銘柄は1件
      { jqCode: 'OLD00', date: '2026-11-01' },                                            // dates の外 → 母集団の外
    ],
  };
  const vals = { stocks: [
    val('A0000', 200),           // 50%  → いちばん割安
    val('B0000', 100),           // 100%
    val('C0000', 50),            // 200% → いちばん割高
    val('D0000', 100),           // 100%（B と同じ値 → 順位の平均）
    val('E0000', null, 100, false),   // 算出できない → 入れない
    val('OLD00', 1000),          // 母集団の外 → 入れない
    val('ZZZ00', 1000),          // 候補でない → 入れない
  ] };
  const m = valueScores(pred, vals);
  assert.deepEqual([...m.keys()].sort(), ['A0000', 'B0000', 'C0000', 'D0000']);
  assert.deepEqual(m.get('A0000'), { score: 1, ratio: 0.5, rank: 1, n: 4 });
  assert.deepEqual(m.get('C0000'), { score: 0, ratio: 2, rank: 4, n: 4 });
  near(m.get('B0000').score, (4 - 2.5) / 3);                                    // 2位と3位の平均 2.5
  assert.equal(m.get('B0000').rank, 2.5);
  assert.equal(m.get('D0000').score, m.get('B0000').score);
  assert.equal(m.has('E0000'), false);
});

test('valueScores: 1銘柄だけなら 0.5。理論株価が無い・読めないときは空', () => {
  const pred = { dates: ['2026-11-19'], candidates: [{ jqCode: 'A0000', date: '2026-11-19' }] };
  assert.deepEqual(valueScores(pred, { stocks: [val('A0000', 50)] }).get('A0000'), { score: 0.5, ratio: 2, rank: 1, n: 1 });
  assert.equal(valueScores(pred, null).size, 0);
  assert.equal(valueScores(pred, { stocks: [] }).size, 0);
  assert.equal(valueScores(null, { stocks: [val('A0000', 50)] }).size, 0);
  // 理論株価が 0 以下・終値が無いものも「算出できない」
  assert.equal(valueScores(pred, { stocks: [val('A0000', 0)] }).size, 0);
  assert.equal(valueScores(pred, { stocks: [{ ...val('A0000', 50), price: null }] }).size, 0);
});

test('summaryRow: 6つの頂点を 0〜1 か null で返し、元の値も持つ', () => {
  const c = cand('A0000', [90, 80, 70, 90, 80, 70],
    { ltr: { dayPct: 75, dayRank: 2, nDay: 5 }, ltr5: { pct: 50, rank: 5, nPool: 9 } });
  const values = new Map([['A0000', { score: 0.25, ratio: 1.6, rank: 4, n: 5 }]]);
  const r = summaryRow(c, values);
  assert.deepEqual(Object.keys(r.axes), SUMMARY_AXES.map((a) => a.key));
  near(r.axes.mean, 0.8);
  assert.equal(r.axes.ltr, 0.75);
  assert.equal(r.axes.ltr5, 0.5);
  assert.equal(r.axes.value, 0.25);
  assert.equal(r.raw.value.ratio, 1.6);
  const none = summaryRow({ jqCode: 'B0000' }, values);
  assert.deepEqual(none.axes, { mean: null, min: null, agree: null, ltr: null, ltr5: null, value: null });
});
