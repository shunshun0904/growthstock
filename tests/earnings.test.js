/**
 * 決算サンキーの元になる変換の単体テスト。
 *   npm test
 *
 * ここで守りたいのは「元データに無い数字を作らないこと」と
 * 「赤字のときに嘘の流れを描かないこと」。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { toFlow, latestQuarter, fmtYen, share, MODES } from '../src/lib/earnings.js';

const q = (o) => ({
  disclosedDate: '2026-08-04', period: '1Q', periodEnd: '2026-06-30',
  qNetSales: 1000, qOperatingProfit: 150, qOrdinaryProfit: 140, qProfit: 90,
  cumNetSales: 1000, cumOperatingProfit: 150, cumOrdinaryProfit: 140, cumProfit: 90,
  forecastNetSales: 4200, forecastOperatingProfit: 600,
  forecastOrdinaryProfit: 580, forecastProfit: 380, ...o,
});

test('段階ごとに「残った額」と「出ていった額」を出す', () => {
  const f = toFlow(q());
  assert.equal(f.sales, 1000);
  assert.deepEqual(f.steps.map((s) => [s.key, s.value]),
    [['op', 150], ['odp', 140], ['np', 90]]);
  // 営業費用は 売上高 − 営業利益 の引き算でしか出せない
  assert.equal(f.steps[0].out.value, 850);
  assert.equal(f.steps[1].out.value, 10);
  assert.equal(f.steps[2].out.value, 50);
  assert.equal(f.truncated, null);
});

test('出ていった額の合計 ＋ 最終利益 が売上高に一致する（作り出していない）', () => {
  const f = toFlow(q());
  const out = f.steps.reduce((a, s) => a + (s.out ? s.out.value : 0), 0);
  const inn = f.steps.reduce((a, s) => a + (s.in ? s.in.value : 0), 0);
  assert.equal(out - inn + f.steps.at(-1).value, f.sales);
});

test('段階利益が増えるときは流入として扱う', () => {
  // 特別利益で 当期純利益 が 経常利益 を上回る
  const f = toFlow(q({ qProfit: 200 }));
  const np = f.steps.at(-1);
  assert.equal(np.out, null);
  assert.equal(np.in.value, 60);
  assert.equal(np.in.label, '特別利益');
});

test('経常利益が無い決算では段階を1つ飛ばし、名前を変える', () => {
  const f = toFlow(q({ qOrdinaryProfit: null }));
  assert.deepEqual(f.steps.map((s) => s.key), ['op', 'np']);
  assert.deepEqual(f.missing, ['odp']);
  assert.equal(f.steps[1].out.label, '営業外・特別損益・法人税等');
  assert.equal(f.steps[1].out.value, 60);   // 150 -> 90
});

test('赤字の段でサンキーを打ち切る（負の流れは描けない）', () => {
  const f = toFlow(q({ qProfit: -30 }));
  assert.equal(f.truncated, 'np');
  const np = f.steps.at(-1);
  assert.equal(np.loss, true);
  // 幹から出せるのは直前の段階利益まで。超過分は描画側が赤で足す
  assert.equal(np.out.value, 140);
});

test('営業赤字なら、それ以降の段階は描かない', () => {
  const f = toFlow(q({ qOperatingProfit: -50 }));
  assert.equal(f.truncated, 'op');
  assert.equal(f.steps.length, 1);
});

test('売上高が無い・ゼロ・マイナスなら図を作らない', () => {
  assert.equal(toFlow(q({ qNetSales: null })), null);
  assert.equal(toFlow(q({ qNetSales: 0 })), null);
  assert.equal(toFlow(q({ qNetSales: -5 })), null);
  assert.equal(toFlow(null), null);
});

test('モードごとに別の列を読む', () => {
  assert.equal(toFlow(q(), 'cum').sales, 1000);
  assert.equal(toFlow(q(), 'forecast').sales, 4200);
  assert.equal(toFlow(q(), 'forecast').steps[0].value, 600);
  // 予想が無い決算では予想モードが作れない（ボタンを出さないため）
  assert.equal(toFlow(q({ forecastNetSales: null }), 'forecast'), null);
});

test('モードの一覧は元データにある3種だけ', () => {
  assert.deepEqual(MODES.map((m) => m.id), ['q', 'cum', 'forecast']);
});

test('直近決算は開示日がいちばん新しいもの', () => {
  const rows = [
    { disclosedDate: '2026-02-10' }, { disclosedDate: '2026-08-04' },
    { disclosedDate: '2026-05-12' },
  ];
  assert.equal(latestQuarter(rows).disclosedDate, '2026-08-04');
  assert.equal(latestQuarter([]), null);
  assert.equal(latestQuarter(undefined), null);
});

test('損失には会計慣行の △ を付ける（色だけに頼らないため）', () => {
  assert.equal(fmtYen(-45_000_000), '△4,500万円');
  assert.equal(fmtYen(4_800_000_000), '48.0億円');
  assert.equal(fmtYen(1_814_722_000_000), '1.81兆円');
  assert.equal(fmtYen(12_000_000_000), '120億円');
  assert.equal(fmtYen(null), '—');
});

test('構成比は売上高に対する割合', () => {
  assert.equal(share(150, 1000), 15);
  assert.equal(share(150, 0), null);
  assert.equal(share(null, 1000), null);
});
