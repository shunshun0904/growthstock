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

/**
 * 日足の本数を週に直す。予測モデルが窓を「78週」と呼ぶのと同じ換算
 * （research/build_dataset.py: 1年 = 245営業日 = 52週。368営業日 -> 78週）。
 * 株価推移の見出しはこれで出す。暦の日数から数えると、368営業日は約550暦日 = 78.6週で、
 * 四捨五入すると「79週」になる（2026-09-26 に取り直したデータで 2025-03-24 〜 2026-09-25）。
 */
export const TRADING_DAYS_PER_YEAR = 245;
export const barsToWeeks = (bars) => Math.round(((bars - 1) / TRADING_DAYS_PER_YEAR) * 52);

export const fmtTick = (t) => {
  const d = new Date(t);
  return `${String(d.getUTCFullYear()).slice(2)}/${d.getUTCMonth() + 1}`;
};

/**
 * 株価推移と出来事。出来事（決算・78週高値の更新・出来高急増）は、その日に縦の破線を引く。
 *
 * 期間は目的変数と同じ78週（368営業日。scripts/jquants_data_fetcher.py の CHART_BARS）。
 * 見出しの週数は、期間の日足の本数 bars（stocks.json の historyBars）があれば予測モデルと
 * 同じ換算で出す（78週そろえば「78週」）。無ければ（取り直す前のデータ・手入力の銘柄）
 * 履歴の最初と最後の日付から数える。
 * 横軸は日付の目盛り（時刻）にして、間引いた点に無い日の出来事も正しい位置に引く。
 */
export function buildPriceSeries(history, milestones, bars) {
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
  const weeks = Number.isFinite(bars) && bars > 1
    ? barsToWeeks(bars)
    : Math.round((t1 - t0) / (7 * DAY_MS));
  return { data, events, span: { from: data[0].date, to: data[data.length - 1].date, weeks, t0, t1 } };
}
