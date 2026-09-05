/**
 * 8軸スコアリングエンジンの単体テスト (仕様書 §4 の各式を直接検証する)。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  normalize, scoreEpsGrowth, scoreSalesGrowth, scoreRoe, scoreOpMargin,
  scoreTechnical, scoreVolume, scoreCreditRatio, scoreProgress,
  liquidityTier, institutionalLevel, priceZone, computeScores, AXES,
} from '../src/lib/scoring.js';

const close = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-9, msg ?? `${a} != ${b}`);

/* -------------------------------------------------- 正規化 */

test('normalize: 範囲外は 0 と 10 にクランプされる', () => {
  close(normalize(0, 0, 50), 0);
  close(normalize(50, 0, 50), 10);
  close(normalize(25, 0, 50), 5);
  close(normalize(-100, 0, 50), 0);
  close(normalize(999, 0, 50), 10);
});

test('normalize: 数値でない入力は null', () => {
  assert.equal(normalize(NaN, 0, 50), null);
  assert.equal(normalize(undefined, 0, 50), null);
  assert.equal(normalize(5, 10, 10), null);
});

/* -------------------------------------------------- 軸 1-4 (線形) */

test('軸1 EPS成長: S(x, 0, 50) — +50%で満点', () => {
  close(scoreEpsGrowth(50), 10);
  close(scoreEpsGrowth(25), 5);
  close(scoreEpsGrowth(-10), 0);
  assert.equal(scoreEpsGrowth(null), null);
});

test('軸2 売上成長: S(x, 0, 40) — +40%で満点', () => {
  close(scoreSalesGrowth(40), 10);
  close(scoreSalesGrowth(20), 5);
});

test('軸3 ROE: S(x, 5, 25) — 25%で満点、5%で0点', () => {
  close(scoreRoe(25), 10);
  close(scoreRoe(15), 5);
  close(scoreRoe(5), 0);
  close(scoreRoe(0), 0);
});

test('軸4 営業利益率: S(x, 0, 20) — 20%で満点', () => {
  close(scoreOpMargin(20), 10);
  close(scoreOpMargin(10), 5);
});

/* -------------------------------------------------- 軸5 テクニカル */

test('軸5 テクニカル: 仕様書の3段階が境界で連続 (98%のジャンプを除く)', () => {
  close(scoreTechnical(98), 10);
  close(scoreTechnical(99.9), 10);
  close(scoreTechnical(90), 8.0);
  close(scoreTechnical(94), 8.0 + (4 / 8) * 1.5);
  close(scoreTechnical(80), 6.0);
  close(scoreTechnical(85), 7.0);
  // 80%未満は仕様未定義のため線形外挿。R=80 で 6.0 と一致すること
  close(scoreTechnical(40), 3.0);
  close(scoreTechnical(0), 0);
});

test('軸5 テクニカル: 単調非減少である', () => {
  let prev = -1;
  for (let r = 0; r <= 105; r += 0.5) {
    const s = scoreTechnical(r);
    assert.ok(s >= prev - 1e-9, `R=${r} で単調性が崩れた (${prev} -> ${s})`);
    prev = s;
  }
});

/* -------------------------------------------------- 軸6 出来高 */

test('軸6 出来高: 20日平均と同水準(100%)で 5.0、2倍(200%)で満点', () => {
  close(scoreVolume(100, 'high'), 5.0);
  close(scoreVolume(200, 'high'), 10.0);
  close(scoreVolume(200, 'mega'), 10.0);
  close(scoreVolume(0, 'high'), 0);
});

test('軸6 出来高: 機関参入レベルが低いほど減衰する (満点条件は10億円以上)', () => {
  const trend = 200;
  const mega = scoreVolume(trend, 'mega');
  const high = scoreVolume(trend, 'high');
  const mod = scoreVolume(trend, 'moderate');
  const low = scoreVolume(trend, 'low');
  const none = scoreVolume(trend, 'none');
  assert.equal(mega, 10);
  assert.equal(high, 10);
  assert.ok(mod < high && low < mod && none < low, '減衰順序が正しくない');
  // 流動性不足では出来高が2倍でも満点にならない
  assert.ok(none < 5, '流動性不足の減衰が弱すぎる');
});

/* -------------------------------------------------- 軸7 需給 */

test('軸7 需給: 信用倍率の3区分が境界で連続', () => {
  close(scoreCreditRatio(0.5), 10);
  close(scoreCreditRatio(1.0), 10);
  close(scoreCreditRatio(2.0), 8.5);
  close(scoreCreditRatio(3.0), 7.0);
  close(scoreCreditRatio(6.5), 4.5);
  close(scoreCreditRatio(10.0), 2.0);
  // 10倍超は仕様未定義のため線形外挿し、20倍で 0 に到達する
  close(scoreCreditRatio(15), 1.0);
  close(scoreCreditRatio(30), 0);
});

test('軸7 需給: 単調非増加である', () => {
  let prev = 11;
  for (let c = 0.1; c <= 25; c += 0.1) {
    const s = scoreCreditRatio(c);
    assert.ok(s <= prev + 1e-9, `C=${c} で単調性が崩れた`);
    prev = s;
  }
});

/* -------------------------------------------------- 軸8 進捗期待 */

test('軸8 進捗期待: B = Quarter × 25 を基準に ±2%ptで1点動く', () => {
  close(scoreProgress(50, 2), 5.0);   // 2Q で進捗50% = 基準通り
  close(scoreProgress(60, 2), 10.0);  // 基準+10pt -> 5 + 5
  close(scoreProgress(56, 2), 8.0);
  close(scoreProgress(40, 2), 0);     // 基準-10pt -> クランプで 0
  close(scoreProgress(25, 1), 5.0);
  close(scoreProgress(75, 3), 5.0);
  assert.equal(scoreProgress(50, null), null);
});

/* -------------------------------------------------- §4.2 機関投資家参入度 */

test('§4.2 流動性ティアが仕様書の境界どおり', () => {
  assert.equal(liquidityTier(0.9), 'none');
  assert.equal(liquidityTier(1), 'low');
  assert.equal(liquidityTier(4.99), 'low');
  assert.equal(liquidityTier(5), 'moderate');
  assert.equal(liquidityTier(9.99), 'moderate');
  assert.equal(liquidityTier(10), 'high');
  assert.equal(liquidityTier(29.99), 'high');
  assert.equal(liquidityTier(30), 'mega');
  assert.equal(liquidityTier(null), null);
});

test('§4.2 時価総額100億円未満は cap_low が優先される', () => {
  assert.equal(institutionalLevel(50, 80), 'cap_low');
  assert.equal(institutionalLevel(50, 100), 'mega');
  assert.equal(institutionalLevel(50, null), 'mega');   // 時価総額不明なら流動性で判定
  assert.equal(institutionalLevel(12, 5000), 'high');
});

/* -------------------------------------------------- §4.3 株価ゾーン */

test('§4.3 株価ゾーンが仕様書の境界どおり', () => {
  assert.equal(priceZone(100), 'BREAKOUT');
  assert.equal(priceZone(98), 'BREAKOUT');
  assert.equal(priceZone(97.99), 'HANDLE');
  assert.equal(priceZone(90), 'HANDLE');
  assert.equal(priceZone(89.99), 'BASE');
  assert.equal(priceZone(80), 'BASE');
  assert.equal(priceZone(79.99), 'CORRECTION');
  assert.equal(priceZone(null), null);
});

/* -------------------------------------------------- 総合 */

test('computeScores: 全軸が満点なら総合10.0', () => {
  const r = computeScores({
    epsGrowth: 60, salesGrowth: 50, roe: 30, opMargin: 25,
    highRatio: 99, volumeTrend: 250, creditRatio: 0.8,
    progressRate: 80, quarter: 2, tradingValue: 40, marketCap: 3000,
  });
  assert.equal(r.totalScore, 10);
  assert.equal(r.strictTotalScore, 10);
  assert.equal(r.coverage, 8);
  assert.equal(r.zone, 'BREAKOUT');
  assert.equal(r.institutional, 'mega');
});

test('computeScores: 欠測軸は 0 ではなく null になり、平均から除外される', () => {
  const r = computeScores({
    epsGrowth: 50, salesGrowth: 40, highRatio: 99,
    tradingValue: 40, marketCap: 3000, volumeTrend: 250,
  });
  assert.equal(r.scores.roe, null, 'ROE 未取得なら null であるべき');
  assert.equal(r.scores.supply, null, '信用倍率 未取得なら null であるべき');
  assert.equal(r.scores.progress, null);
  // 値があるのは eps / sales / technical / volume の4軸
  assert.equal(r.coverage, 4);
  // 有効4軸すべて満点 -> totalScore は 10、8軸0埋めの厳密値は 40/8 = 5.0
  assert.equal(r.totalScore, 10);
  assert.equal(r.strictTotalScore, 5);
});

test('computeScores: 空入力でも例外を投げず null を返す', () => {
  const r = computeScores({});
  assert.equal(r.totalScore, null);
  assert.equal(r.strictTotalScore, null);
  assert.equal(r.coverage, 0);
  assert.equal(r.zone, null);
  // 売買代金が不明なときは null。「流動性不足(none)」と断定しない
  // (測っていないものを『1億円未満』と主張しないため)
  assert.equal(r.institutional, null);
  assert.equal(r.liquidity, null);
  assert.equal(computeScores(undefined).coverage, 0);
});

test('computeScores: axisScores は AXES と同じ順序・件数', () => {
  const r = computeScores({ epsGrowth: 10 });
  assert.equal(r.axisScores.length, 8);
  assert.deepEqual(r.axisScores.map((a) => a.key), AXES.map((a) => a.key));
  assert.equal(r.axisScores[0].value, 10);
});

test('computeScores: 出来高軸は売買代金の増加とともに単調非減少', () => {
  const base = { volumeTrend: 180, marketCap: 3000 };
  const scores = [0.5, 3, 7, 15, 50].map(
    (v) => computeScores({ ...base, tradingValue: v }).scores.volume
  );
  for (let i = 1; i < scores.length; i++) {
    assert.ok(scores[i] >= scores[i - 1], `売買代金増加でスコアが下がった: ${scores}`);
  }
});

test('§6.1 性能: 1000銘柄ぶんの再計算が16ms以内', () => {
  const sample = {
    epsGrowth: 32, salesGrowth: 21, roe: 17, opMargin: 13, highRatio: 94,
    volumeTrend: 150, creditRatio: 2.2, progressRate: 58, quarter: 2,
    tradingValue: 18, marketCap: 2400,
  };
  const trial = () => {
    const start = performance.now();
    for (let i = 0; i < 1000; i++) computeScores({ ...sample, epsGrowth: 32 + (i % 20) });
    return performance.now() - start;
  };

  // CI の負荷でスケジューラのノイズが乗るため、複数回試行の「最小値」で判定する。
  // (最小値はベンチマークで外乱を除く標準的な統計量。平均だと他プロセスの影響で
  //  実装が変わっていなくてもフレークする)
  trial();  // JIT ウォームアップ
  const best = Math.min(...Array.from({ length: 5 }, trial));
  assert.ok(best < 16, `1000件の再計算に最短でも ${best.toFixed(1)}ms かかった`);
});
