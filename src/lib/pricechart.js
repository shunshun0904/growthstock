/**
 * タイムマシーンの株価推移（目的変数と同じ78週）と、出来事の縦の破線の位置。
 * 描画（JSX）から切り離して単体テストする（2026-09-26、運用者の依頼）。
 */
export const DAY_MS = 86400000;

/** 'YYYY-MM-DD' を UTC の 0時の時刻に（履歴と出来事を同じ目盛りに載せる） */
export const toTime = (d) => Date.parse(`${d}T00:00:00Z`);

/** 四半期の初め（1・4・7・10月の1日）の目盛り。78週で6本ほど */
export function quarterTicks(t0, t1) {
  const out = [];
  const d = new Date(t0);
  let y = d.getUTCFullYear();
  let m = Math.floor(d.getUTCMonth() / 3) * 3;
  for (;;) {
    const t = Date.UTC(y, m, 1);
    if (t > t1) break;
    if (t >= t0) out.push(t);
    m += 3;
    if (m >= 12) { m -= 12; y += 1; }
  }
  return out;
}

export const fmtTick = (t) => {
  const d = new Date(t);
  return `${String(d.getUTCFullYear()).slice(2)}/${d.getUTCMonth() + 1}`;
};

/**
 * 株価推移と出来事。出来事（決算・78週高値の更新・出来高急増）は、その日に縦の破線を引く。
 *
 * 期間は目的変数と同じ78週（368営業日。scripts/jquants_data_fetcher.py の CHART_BARS）。
 * 見出しの週数は履歴から数える（取り込みを直す前の1年ぶんのデータなら「52週」と出る）。
 * 横軸は日付の目盛り（時刻）にして、間引いた点に無い日の出来事も正しい位置に引く。
 */
export function buildPriceSeries(history, milestones) {
  const data = (history || [])
    .filter((h) => h && h.date && Number.isFinite(h.close))
    .map((h) => ({ t: toTime(h.date), date: h.date, close: h.close, events: [] }));
  if (!data.length) return { data, events: [], span: null };
  const t0 = data[0].t;
  const t1 = data[data.length - 1].t;
  const events = (milestones || [])
    .filter((e) => e && e.date)
    .map((e) => ({ ...e, t: toTime(e.date) }))
    .filter((e) => e.t >= t0 && e.t <= t1)
    .sort((a, b) => a.t - b.t);
  // ツールチップで出すため、出来事をいちばん近い点に結び付ける
  for (const e of events) {
    let lo = 0;
    let hi = data.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (data[mid].t <= e.t) lo = mid; else hi = mid;
    }
    const k = Math.abs(data[lo].t - e.t) <= Math.abs(data[hi].t - e.t) ? lo : hi;
    data[k].events.push(e);
  }
  const weeks = Math.round((t1 - t0) / (7 * DAY_MS));
  return { data, events, span: { from: data[0].date, to: data[data.length - 1].date, weeks, t0, t1 } };
}
