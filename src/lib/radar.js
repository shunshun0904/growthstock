/**
 * 8軸オクタゴン（RadarPanel）の大きさと軸名の置き方。描画（JSX）から切り離して単体テストする。
 *
 * 2026-09-26、運用者の指摘（スマホで八角形が小さい・描画が崩れる）で作り直した。
 */
import { AXES } from './scoring.js';

/** これより狭いと「スマホ幅」として軸名を2段にする（px。図の横幅） */
export const NARROW_WIDTH = 520;

/** スマホ幅で2段にする軸名。横に長い名前だけ（上下の軸は横幅を食わないのでそのまま） */
const TWO_LINES = {
  '営業利益率': ['営業', '利益率'],
  '信用倍率': ['信用', '倍率'],
  '進捗期待': ['進捗', '期待'],
  '売上成長': ['売上', '成長'],
};

export const LABEL_FONT = 11.5;       // 軸名の文字の大きさ（px）
export const LINE_H = 14;             // 2段にしたときの行の高さ（px）
export const TICK_GAP = 8;            // 頂点から軸名までの間（Recharts の tickSize と同じ）
export const MAX_RADIUS = 260;        // 縦長の枠（PC で右の列が長いとき）でも大きくなりすぎないように

/** 軸名の行（スマホ幅では長い名前を2段に） */
export function labelLines(label, narrow) {
  return narrow && TWO_LINES[label] ? TWO_LINES[label] : [label];
}

/** 文字の幅の見積り（px）。全角は1文字、半角英数は0.6文字 */
export function textWidth(s) {
  let w = 0;
  for (const ch of s) w += /[ -~]/.test(ch) ? 0.6 : 1;
  return w * LABEL_FONT;
}

/**
 * 図の大きさから八角形の半径を決める（px）。
 *
 * 以前は半径を「余白を除いた正方形の 72%」にしていた。スマホ幅では左右の余白 44px ずつと
 * 72% が重なり、390px の画面で八角形の直径が 158px しかなかった（2026-09-26 実測）。
 * 軸ごとに、軸名が図の外にはみ出さない最大の半径を求め、その最小を使う。
 *   横: 頂点から外へ cos・(半径 + 間) 進んだところから軸名の幅（真上・真下は中央ぞろえ）
 *   縦: 上下の軸名は頂点の上・下に積む（AngleTick と同じ置き方）
 */
export function radarRadius(width, height, narrow, maxRadius = MAX_RADIUS) {
  const PAD = 4;
  let r = maxRadius;
  AXES.forEach((a, i) => {
    const ang = ((90 - 45 * i) * Math.PI) / 180;
    const c = Math.abs(Math.cos(ang));
    const s = Math.abs(Math.sin(ang));
    const lines = labelLines(a.label, narrow);
    const w = Math.max(...lines.map(textWidth));
    const h = (lines.length - 1) * LINE_H + LABEL_FONT + 4;
    if (c > 0.1) r = Math.min(r, (width / 2 - PAD - w) / c - TICK_GAP);
    if (s > 0.5) r = Math.min(r, (height / 2 - PAD - h) / s - TICK_GAP);
  });
  return Math.max(40, Math.floor(r));
}
