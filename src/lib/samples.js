/**
 * 再学習で加わった例（public/data/new_samples.json）のチャートの組み立て。描画（JSX）から切り離して単体テストする。
 *
 * 1件（case）は research/new_samples.py が research/export_label_samples.build_case で作ったもの。
 *   dates / close / maShort / maLong … 更新日の前40営業日〜判定の20営業日の後5日の日足（分割調整後）
 *   tPos     … 更新日（78週高値を抜けた日）の位置。horizon … 判定の窓 [tPos+1, tPos+20]
 *   entryPrice … 基準の価格（翌営業日の寄り）。target … 到達に必要な水準（寄り × (1 + k×σ)）
 *   endLineAdopted … 終盤に必要な水準（窓の最後5日の平均がこれ以上）
 *   verdict … 成否の内訳（データセットの材料から）: reached / endOk / uptrend と、その数値（%）
 */
import { fmtSigned, fmtDate } from './format.js';

/** 正例・負例に分ける（並びはファイルの順 = 日付・銘柄の順）。 */
export function splitCases(data) {
  const cases = Array.isArray(data?.cases) ? data.cases : [];
  return {
    pos: cases.filter((c) => c?.label === 1),
    neg: cases.filter((c) => c?.label === 0),
  };
}

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** 'YYYY-MM-DD' を 'M/D' に（目盛り用）。 */
export const fmtMD = (d) => {
  if (!d || typeof d !== 'string') return '';
  const [, m, day] = d.split('-');
  return m && day ? `${Number(m)}/${Number(day)}` : d;
};

/**
 * チャートの点・窓・目盛り・縦軸の範囲。
 *   points … { i, date, close, ma5, ma20 }（横軸は営業日の並び i。土日や祝日の隙間を空けない）
 *   win    … 判定の20営業日 [from, to]（i）。endWin … 窓の最後5日（終盤の水準を測るところ）
 *   entryPos … 寄り（翌営業日）の位置。yDomain … 終値・寄り・到達・終盤の線がすべて入る範囲（上下3%の余白）
 *   ticks  … 目盛り（最初・更新日・窓の終わり・最後）
 */
export function caseChart(c) {
  const dates = Array.isArray(c?.dates) ? c.dates : [];
  const close = Array.isArray(c?.close) ? c.close : [];
  const points = dates.map((date, i) => ({
    i, date, close: num(close[i]), ma5: num(c?.maShort?.[i]), ma20: num(c?.maLong?.[i]),
  }));
  const last = Math.max(0, points.length - 1);
  const tPos = Number.isInteger(c?.tPos) ? Math.min(c.tPos, last) : 0;
  const [w0, w1] = Array.isArray(c?.horizon) ? c.horizon : [tPos + 1, tPos + 20];
  const win = [Math.min(w0, last), Math.min(w1, last)];
  const endWin = [Math.max(win[0], win[1] - 4), win[1]];
  const levels = [num(c?.entryPrice), num(c?.target), num(c?.endLineAdopted),
    ...points.map((p) => p.close)].filter((v) => v !== null);
  const lo = levels.length ? Math.min(...levels) : 0;
  const hi = levels.length ? Math.max(...levels) : 1;
  const ticks = [...new Set([0, tPos, win[1], last])].filter((i) => i >= 0 && i <= last).sort((a, b) => a - b);
  const y = niceAxis(lo * 0.98, hi * 1.02);
  return {
    points, win, endWin, tPos, entryPos: Math.min(tPos + 1, last),
    yDomain: y.domain, yTicks: y.ticks, ticks,
  };
}

/**
 * 縦軸をきりのよい刻み（1・2・2.5・5 × 10^k）にそろえる。目盛りは4本前後。
 * 範囲は lo〜hi を必ず含む（線が枠の外に出ない）。
 */
export function niceAxis(lo, hi, target = 4) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { domain: [0, 1], ticks: [0, 1] };
  if (hi <= lo) hi = lo + Math.max(1, Math.abs(lo) * 0.01);
  const raw = (hi - lo) / target;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) || 10 * p;
  // 割り算の誤差（0.95 / 0.025 = 37.999…）で1刻みずれないよう、わずかに寄せてから丸める
  const r6 = (v) => Math.round(v * 1e6) / 1e6;
  const a = r6(Math.floor(lo / step + 1e-9) * step);
  const b = r6(Math.ceil(hi / step - 1e-9) * step);
  const ticks = [];
  for (let v = a; v <= b + step / 2; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return { domain: [a, b], ticks };
}

/**
 * 成否の内訳（✓ / ✗ と文）。正例は3つとも ✓。値が無ければ「—」。
 *   到達       … 判定の20営業日の終値の最大が、寄りから必要な上昇率以上か
 *   終盤の水準 … 20日目までの5日平均が、寄りから必要な水準以上か
 *   トレンド   … 20日目に MA5 ≥ MA20 か
 */
export function verdictLines(c) {
  const v = c?.verdict || {};
  return [
    { key: 'reached', ok: !!v.reached, label: '到達',
      text: `最大 ${fmtSigned(v.maxRisePct, 1)}（必要 ${fmtSigned(v.needPct, 1)}）` },
    { key: 'end', ok: !!v.endOk, label: '終盤の水準',
      text: `20日目までの5日平均 ${fmtSigned(v.endLevelPct, 1)}（必要 ${fmtSigned(v.endNeedPct, 1)}）` },
    { key: 'trend', ok: !!v.uptrend, label: '20日目のトレンド',
      text: v.uptrend ? 'MA5 ≥ MA20' : 'MA5 < MA20' },
  ];
}

/** 寄りから見た終値の上昇率（%）。ツールチップ用。 */
export function fromEntry(close, entry) {
  const a = num(close);
  const b = num(entry);
  return a !== null && b ? (a / b - 1) * 100 : null;
}

/** 見出しの「加わった更新日」。前回の終わりが分からなければ「〜今回の終わり」。 */
export function windowText(data) {
  const w = data?.window || {};
  if (!w.to) return '—';
  return w.after ? `${fmtDate(w.after)} より後 〜 ${fmtDate(w.to)}` : `〜 ${fmtDate(w.to)}`;
}

/** 正例率（%）。加わった件数が0なら null。 */
export function addedPositiveRate(data) {
  const c = data?.counts || {};
  return c.added ? (c.positive / c.added) * 100 : null;
}

