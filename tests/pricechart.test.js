/**
 * タイムマシーンの株価推移と出来事の破線の単体テスト。
 *   npm test
 *
 * 2026-09-26、運用者の依頼「目的変数の定義通り過去78週（1年半分）の表示にしてほしいのと、
 * 各出来事がチャート上でどこにあたるのかも、破線とかで表示してくれる見やすいです」。
 * 期間はデータ（scripts/jquants_data_fetcher.py の chart_history）が決め、画面は受け取った
 * 範囲の週数を見出しに出す。出来事は期間の中のものだけ、その日に破線を引く。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPriceSeries, quarterTicks, fmtTick, toTime } from '../src/lib/pricechart.js';

/** 3営業日ごと（ここでは3暦日ごと）の終値 */
function history(from, n, step = 3) {
  const t0 = toTime(from);
  return Array.from({ length: n }, (_, i) => ({
    date: new Date(t0 + i * step * 86400000).toISOString().slice(0, 10),
    close: 1000 + i,
  }));
}

test('78週ぶんの履歴なら見出しの週数は78', () => {
  const h = history('2025-03-10', 183);            // 182 × 3日 = 546日 = 78週
  const { span } = buildPriceSeries(h, []);
  assert.equal(span.weeks, 78);
  assert.equal(span.from, '2025-03-10');
  assert.equal(span.to, h[h.length - 1].date);
});

test('出来事は期間の中のものだけを、いちばん近い点に結び付ける', () => {
  const h = history('2026-01-05', 10);             // 1/5 〜 1/32(=2/1)
  const ms = [
    { date: '2025-12-01', type: 'earnings', title: '期間より前' },
    { date: '2026-01-09', type: 'breakout', title: '78週高値を更新' },    // 1/8 と 1/11 の間
    { date: '2026-01-12', type: 'volume_spike', title: '出来高急増' },     // 1/11 の翌日
    { date: '2026-03-01', type: 'earnings', title: '期間より後' },
  ];
  const { data, events } = buildPriceSeries(h, ms);
  assert.deepEqual(events.map((e) => e.title), ['78週高値を更新', '出来高急増']);
  const at = (d) => data.find((p) => p.date === d).events.map((e) => e.title);
  assert.deepEqual(at('2026-01-08'), ['78週高値を更新']);       // 1日違い（1/11 は2日違い）
  assert.deepEqual(at('2026-01-11'), ['出来高急増']);
  // 破線の位置は点ではなく出来事の日そのもの（間引いた点に無い日でも正しく引く）
  assert.equal(events[0].t, toTime('2026-01-09'));
});

test('履歴が無ければ何も描かない', () => {
  const r = buildPriceSeries([], [{ date: '2026-01-09', type: 'breakout' }]);
  assert.equal(r.span, null);
  assert.deepEqual(r.events, []);
});

test('横軸の目盛りは四半期の初め（1・4・7・10月の1日）', () => {
  const ticks = quarterTicks(toTime('2025-03-10'), toTime('2026-09-25'));
  assert.deepEqual(ticks.map(fmtTick), ['25/4', '25/7', '25/10', '26/1', '26/4', '26/7']);
});
