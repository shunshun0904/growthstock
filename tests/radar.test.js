/**
 * 8軸オクタゴンの大きさと軸名の置き方の単体テスト。
 *   npm test
 *
 * 2026-09-26、運用者の指摘「スマホで確認したところ、オクタゴンの描画がバグっている …
 * 八角形自体が少し小さい」。390px の画面で直径が 158px しかなかった（半径を「余白を除いた
 * 正方形の 72%」にしていたため）。軸名がはみ出さない最大の半径を使う。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { AXES } from '../src/lib/scoring.js';
import {
  NARROW_WIDTH, LABEL_FONT, LINE_H, TICK_GAP, MAX_RADIUS,
  labelLines, radarRadius, textWidth,
} from '../src/lib/radar.js';

/** 軸名が図の中に収まるか（RadarPanel の AngleTick と同じ置き方で、端までの距離を測る） */
function labelsFit(width, height, narrow, r) {
  const cx = width / 2;
  const cy = height / 2;
  return AXES.every((a, i) => {
    const ang = ((90 - 45 * i) * Math.PI) / 180;
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    const x = cx + c * (r + TICK_GAP);
    const y = cy - s * (r + TICK_GAP);
    const lines = labelLines(a.label, narrow);
    const w = Math.max(...lines.map(textWidth));
    const h = (lines.length - 1) * LINE_H + LABEL_FONT;
    const left = Math.abs(c) < 0.1 ? x - w / 2 : c > 0 ? x : x - w;
    const right = left + w;
    const top = s > 0.5 ? y - h - 3 : s < -0.5 ? y : y - h / 2;
    const bottom = top + h + 3;
    return left >= 0 && right <= width && top >= 0 && bottom <= height;
  });
}

test('スマホ幅（390px の画面で図の横幅 308px）で八角形が以前より大きい', () => {
  const r = radarRadius(308, 300, true);
  // 以前は半径 79px（直径 158px）
  assert.ok(r >= 110, `半径 ${r}`);
  assert.ok(labelsFit(308, 300, true, r), '軸名がはみ出す');
});

test('360px の画面（図の横幅 278px）でも軸名がはみ出さない', () => {
  const r = radarRadius(278, 300, true);
  assert.ok(r >= 95, `半径 ${r}`);
  assert.ok(labelsFit(278, 300, true, r));
});

test('PC 幅でも軸名がはみ出さず、縦長の枠で大きくなりすぎない', () => {
  for (const [w, h] of [[962, 600], [962, 1200], [700, 420]]) {
    const r = radarRadius(w, h, false);
    assert.ok(labelsFit(w, h, false, r), `${w}x${h} 半径 ${r}`);
    assert.ok(r <= MAX_RADIUS);
  }
});

test('低い枠では高さで決まる（上下の軸名の分を空ける）', () => {
  const r = radarRadius(900, 200, false);
  assert.ok(r < 100, `半径 ${r}`);
  assert.ok(labelsFit(900, 200, false, r));
});

test('スマホ幅では横に長い軸名だけを2段にする', () => {
  assert.deepEqual(labelLines('営業利益率', true), ['営業', '利益率']);
  assert.deepEqual(labelLines('信用倍率', true), ['信用', '倍率']);
  assert.deepEqual(labelLines('EPS成長', true), ['EPS成長']);        // 真上の軸はそのまま
  assert.deepEqual(labelLines('テクニカル', true), ['テクニカル']);  // 真下の軸はそのまま
  assert.deepEqual(labelLines('営業利益率', false), ['営業利益率']);
  assert.ok(NARROW_WIDTH > 308 && NARROW_WIDTH < 700);
});
