/**
 * リカレンス・ネットワーク（画面の「リカレンス・ネットワーク」タブ。src/views/NetworkView.jsx）の、純粋な計算。
 *
 * データは research/recurrence_net.py が日次予測で作り、Release の recurrence.tar.gz をデプロイのときに dist/data/recurrence/ に
 * 展開する。候補ごとのファイルには、その銘柄の各営業日（ノード）の状態（テクニカル指標9列を銘柄の中で標準化し、1/q σ 刻みの
 * 整数にした値）・結果（+10% の指値か20営業日目の終値・目的変数 1.2σ / 1.5σ）・78週高値の更新日が入っている。
 *
 * 辺（タイラー窓より離れた日どうしで、距離の2乗が ε² 以下）と要約は、ここで Python と同じ式で数え直す（画面で辺の密度を
 * 変えられるように）。整数の距離の2乗の m 番目（m = 密度 × 結べる組の数 を四捨五入）を ε² にするので、浮動小数点の差は出ない。
 * tests/recurrence.test.js が Python の数えた値（tests/fixtures/recurrence-expected.json）と一致することを確かめる。
 * **参考の表示で、選定には使わない。**
 */

/** 目的変数の鍵と到達しきい値の倍数（research/recurrence_net.py の LABEL_KS と同じ。index.json の method.labels が正本） */
export const DEFAULT_LABELS = [{ key: 'l12', k: 1.2 }, { key: 'l15', k: 1.5 }];

/** 既定値（index.json の method が無いときだけ使う。正本は research/recurrence_net.py） */
export const DEFAULTS = { theiler: 20, rrBp: 100, rrChoicesBp: [50, 100, 200, 300], q: 20, zCap: 6, target: 10, hold: 20 };

/** index.json の method から、計算に使う値をそろえる（欠けた値は既定値）。 */
export function methodOf(index) {
  const m = index?.method || {};
  return {
    theiler: Number.isInteger(m.theiler) ? m.theiler : DEFAULTS.theiler,
    rrBp: Number.isInteger(m.rrBp) ? m.rrBp : DEFAULTS.rrBp,
    rrChoicesBp: Array.isArray(m.rrChoicesBp) && m.rrChoicesBp.length ? m.rrChoicesBp : DEFAULTS.rrChoicesBp,
    q: Number.isInteger(m.q) ? m.q : DEFAULTS.q,
    zCap: Number.isFinite(m.zCap) ? m.zCap : DEFAULTS.zCap,
    target: Number.isFinite(m.target) ? m.target : DEFAULTS.target,
    hold: Number.isInteger(m.hold) ? m.hold : DEFAULTS.hold,
    labels: Array.isArray(m.labels) && m.labels.length ? m.labels : DEFAULT_LABELS,
    features: Array.isArray(m.features) ? m.features : [],
  };
}

const nanOf = (v) => (v === null || v === undefined ? NaN : Number(v));

/**
 * 候補のファイル → 計算に使う形。dates はノードの日付（start から gaps の暦日を足していく。UTC で数えるので時差に左右されない）。
 */
export function decode(p, labels = DEFAULT_LABELS) {
  const n = p.n;
  const dims = p.features.length;
  const z = Int16Array.from(p.z);
  if (z.length !== n * dims) throw new Error(`状態の数が合いません（${z.length} / ${n} × ${dims}）`);
  if (p.gaps.length !== n - 1) throw new Error('日付の数が合いません');
  const ret = Float64Array.from(p.ret, nanOf);
  // 先20営業日の値動き（色付け用）。版 2 の rise / fall は終値で測った別の物差しなので読まない（版 3 から高値・安値）。
  // 無いファイル（版 1）はすべて未確定
  const ver = Number(p.v) || 1;
  const fwd = (k, since) => (ver >= since && Array.isArray(p[k]) && p[k].length === n
    ? Float64Array.from(p[k], nanOf) : new Float64Array(n).fill(NaN));
  const lab = {};
  for (const { key } of labels) lab[key] = Float64Array.from(p[key] || [], nanOf);
  const dates = new Array(n);
  let t = Date.parse(`${p.start}T00:00:00Z`);
  dates[0] = p.start;
  for (let i = 1; i < n; i += 1) {
    t += p.gaps[i - 1] * 86400000;
    dates[i] = new Date(t).toISOString().slice(0, 10);
  }
  const fresh = Int32Array.from(p.fresh || []);
  const isFresh = new Uint8Array(n);
  for (const i of fresh) isFresh[i] = 1;
  return {
    n, dims, q: p.q, z, ret, rise: fwd('rise', 3), fall: fwd('fall', 3), r20: fwd('r20', 2), labels: lab, dates, fresh, isFresh,
    close: Float64Array.from(p.close || [], nanOf),
    scale: p.scale || [], features: p.features, raw: p.raw || [],
    code: p.code, jqCode: p.jqCode, name: p.name, date: p.date,
  };
}

/** 2ノードの距離の2乗（整数）。 */
export function sqDist(dec, i, j) {
  const { z, dims } = dec;
  const a = i * dims;
  const b = j * dims;
  let acc = 0;
  for (let d = 0; d < dims; d += 1) {
    const v = z[a + d] - z[b + d];
    acc += v * v;
  }
  return acc;
}

/**
 * 辺。結べる組（j − i > theiler）の距離の2乗を小さい順に並べた m 番目を ε²（t）にし、t 以下の組を全部結ぶ（同じ値も全部）。
 * m = rrBp × 組の数 ÷ 10000 を四捨五入（0.5 は切り上げ）、最小 1。結べる組が無ければ t = −1（辺なし）。
 * 戻り値: { t, pairs, m, nEdges, edges（[i0, j0, i1, j1, …]）, deg }
 */
export function network(dec, theiler, rrBp) {
  const { n, dims, z } = dec;
  const deg = new Int32Array(n);
  if (n <= theiler + 1) return { t: -1, pairs: 0, m: 0, nEdges: 0, edges: new Uint32Array(0), deg };
  const pairs = ((n - theiler - 1) * (n - theiler)) / 2;
  const s = new Int32Array(pairs);
  let maxS = 0;
  let k = 0;
  for (let i = 0; i < n; i += 1) {
    const a = i * dims;
    for (let j = i + theiler + 1; j < n; j += 1) {
      const b = j * dims;
      let acc = 0;
      for (let d = 0; d < dims; d += 1) {
        const v = z[a + d] - z[b + d];
        acc += v * v;
      }
      s[k] = acc;
      k += 1;
      if (acc > maxS) maxS = acc;
    }
  }
  const m = Math.min(pairs, Math.max(1, Math.floor((rrBp * pairs + 5000) / 10000)));
  const hist = new Uint32Array(maxS + 1);
  for (let x = 0; x < pairs; x += 1) hist[s[x]] += 1;
  let cum = 0;
  let t = 0;
  for (; t <= maxS; t += 1) {
    cum += hist[t];
    if (cum >= m) break;
  }
  const edges = new Uint32Array(2 * cum);
  let e = 0;
  k = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + theiler + 1; j < n; j += 1) {
      if (s[k] <= t) {
        edges[e] = i;
        edges[e + 1] = j;
        e += 2;
        deg[i] += 1;
        deg[j] += 1;
      }
      k += 1;
    }
  }
  return { t, pairs, m, nEdges: cum, edges, deg };
}

/** 隣接の一覧（ノードごとの近傍。ホバーで近傍を光らせるのに使う）。 */
export function neighborsOf(net, n) {
  const start = new Int32Array(n + 1);
  for (let i = 0; i < n; i += 1) start[i + 1] = start[i] + net.deg[i];
  const fill = start.slice(0, n);
  const list = new Int32Array(start[n]);
  const { edges } = net;
  for (let e = 0; e < edges.length; e += 2) {
    const a = edges[e];
    const b = edges[e + 1];
    list[fill[a]] = b;
    fill[a] += 1;
    list[fill[b]] = a;
    fill[b] += 1;
  }
  return { start, list, of: (i) => list.subarray(start[i], start[i + 1]) };
}

const median = (v) => {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
};

/**
 * ノードの集まり idx の結果（Python の outcome_stats と同じ）。hit は +10% の指値に届いた割合（ret ≥ target）。
 * 割合・中央値は分かっている日だけで数える（無ければ null）。
 */
export function outcomeStats(dec, idx, target = DEFAULTS.target, labels = DEFAULT_LABELS) {
  const r = [];
  for (const i of idx) if (Number.isFinite(dec.ret[i])) r.push(dec.ret[i]);
  const out = {
    n: idx.length,
    known: r.length,
    hit: r.length ? r.filter((x) => x >= target - 1e-9).length / r.length : null,
    retMed: median(r),
  };
  for (const { key } of labels) {
    const lab = dec.labels[key];
    let cnt = 0;
    let pos = 0;
    for (const i of idx) {
      const v = lab ? lab[i] : NaN;
      if (Number.isFinite(v)) {
        cnt += 1;
        pos += v;
      }
    }
    out[key] = cnt ? pos / cnt : null;
    out[`${key}N`] = cnt;
  }
  return out;
}

/** 今日（最後のノード）から各ノードへの距離の2乗。 */
export function todayDist(dec) {
  const today = dec.n - 1;
  const out = new Int32Array(dec.n);
  for (let i = 0; i < dec.n; i += 1) out[i] = sqDist(dec, today, i);
  return out;
}

/**
 * 要約（Python の summarize と同じ項目・同じ式）。今日 = 最後のノード。nearestFresh は今日に近い過去の更新日（タイラー窓より前。
 * ε の外も近い順。同じ距離は日付の古い順）。
 */
export function summarize(dec, net, theiler, rrBp, { target = DEFAULTS.target, labels = DEFAULT_LABELS, nNearest = 5 } = {}) {
  const { n, q } = dec;
  const today = n - 1;
  const { deg } = net;
  const k = deg[today];
  const d2 = todayDist(dec);
  const nb = [];
  for (let i = 0; i < n; i += 1) if (today - i > theiler && net.t >= 0 && d2[i] <= net.t) nb.push(i);
  const nbFresh = nb.filter((i) => dec.isFresh[i]);
  let below = 0;
  let same = 0;
  for (let i = 0; i < n; i += 1) {
    if (deg[i] < k) below += 1;
    else if (deg[i] === k) same += 1;
  }
  const pastFresh = [...dec.fresh].filter((i) => i < today - theiler)
    .sort((a, b) => (d2[a] - d2[b]) || (a - b)).slice(0, nNearest);
  const nearest = pastFresh.map((i) => {
    const rec = {
      i, date: dec.dates[i], d: Math.sqrt(d2[i]) / q, inEps: net.t >= 0 && d2[i] <= net.t,
      ret: Number.isFinite(dec.ret[i]) ? dec.ret[i] : null,
    };
    for (const { key } of labels) {
      const v = dec.labels[key]?.[i];
      rec[key] = Number.isFinite(v) ? v : null;
    }
    return rec;
  });
  const all = Array.from({ length: n }, (_, i) => i);
  let isolated = 0;
  for (let i = 0; i < n; i += 1) if (deg[i] === 0) isolated += 1;
  return {
    rrBp, theiler, pairs: net.pairs, m: net.m, eps2: net.t, eps: net.t >= 0 ? Math.sqrt(net.t) / q : null,
    edges: net.nEdges, isolated,
    deg: k, degPct: (100 * (below + 0.5 * same)) / n, degMed: median(Array.from(deg)),
    nbr: outcomeStats(dec, nb, target, labels),
    base: outcomeStats(dec, all, target, labels),
    nbrFresh: outcomeStats(dec, nbFresh, target, labels),
    baseFresh: outcomeStats(dec, [...dec.fresh], target, labels),
    nearestFresh: nearest,
  };
}

/**
 * 今日の近傍の一覧（表で見る用）。近い順（同じ距離は日付の古い順）。
 */
export function todayNeighbors(dec, net, theiler) {
  const today = dec.n - 1;
  const d2 = todayDist(dec);
  const out = [];
  for (let i = 0; i < dec.n; i += 1) {
    if (today - i > theiler && net.t >= 0 && d2[i] <= net.t) out.push(i);
  }
  return out.sort((a, b) => (d2[a] - d2[b]) || (a - b)).map((i) => ({ i, d: Math.sqrt(d2[i]) / dec.q }));
}

// --------------------------------------------------------------------------- //
// 色（結果・目的変数・時期）。暗い面（#0a0e17）で確かめた並び（dataviz の validate_palette.js。結果の4区分は青 ↔ 赤の発散で、
// 各腕は順序の検査を通る。全組の色覚の差 10.2・通常の見え方 17.5。−10〜0% の濃い赤は面との差 2.98:1 なので凡例と表を必ず添える）
// --------------------------------------------------------------------------- //

export const OUTCOME_CLASSES = [
  { id: 'hit', label: '+10% 到達', color: '#93c5fd' },
  { id: 'up', label: '0〜+10%', color: '#2563eb' },
  { id: 'down', label: '−10〜0%', color: '#b91c1c' },
  { id: 'bad', label: '−10% 未満', color: '#f87171' },
  { id: 'na', label: '未確定', color: '#64748b' },
];

/** 収益（%）の区分。+10 以上 = 指値に届いた。 */
export function outcomeOf(ret, target = DEFAULTS.target) {
  if (!Number.isFinite(ret)) return 'na';
  if (ret >= target - 1e-9) return 'hit';
  if (ret >= 0) return 'up';
  if (ret >= -10) return 'down';
  return 'bad';
}

export const LABEL_CLASSES = [
  { id: 'pos', label: '正例', color: '#93c5fd' },
  { id: 'neg', label: '負例', color: '#64748b' },
  { id: 'na', label: '未確定', color: '#334155' },
];

export function labelClassOf(v) {
  if (!Number.isFinite(v)) return 'na';
  return v >= 0.5 ? 'pos' : 'neg';
}

/**
 * 値で色分けするときの未確定（先20営業日がまだ無い日）。下の2つの段（RISE / R20）のどの色とも、通常の見え方で
 * ΔE 15.3 以上・色覚の3型で 8.6 以上離れる（dataviz の validate_palette.js の deltaE。暗い面 #0a0e17。外した最安値までの
 * 赤の段も含めて測った値）
 */
export const NA_VALUE = '#3a3a3a';

/**
 * 最高値までの上昇率（%）の5段（運用者の選択 2026-10-19「5段＋未確定」）。最高到達価格（t+1〜t+20 の高値の最大）÷ 翌営業日の寄り
 * − 1（運用者の指定「その日から最高到達価格（20日以内最高高値）までの上昇率」・選択「基準は翌営業日の始値」。10% 以上なら +10% の
 * 指値に届いた）。青の段で、低いほど暗く高いほど明るい（順序の段の検査を暗い面 #0a0e17 で通る。いちばん暗い段は面との差 2.38:1
 * なので、凡例の件数と表を添える）。並びは高い順（凡例も同じ）
 */
export const RISE_CLASSES = [
  { id: 'r20p', label: '20% 以上', color: '#cde2fb' },
  { id: 'r20', label: '10〜20%', color: '#86b6ef' },
  { id: 'r10', label: '5〜10%', color: '#5598e7' },
  { id: 'r5', label: '0〜5%', color: '#2a78d6' },
  { id: 'r0', label: '0% 未満', color: '#184f95' },
  { id: 'na', label: '未確定', color: NA_VALUE },
];

/** 最高値までの上昇率（%）の段。下の端を含む（5.0 は「5〜10%」。値は 0.1% 刻みに切り下げてあるので 10 以上 = 指値に届いた） */
export function riseOf(v) {
  if (!Number.isFinite(v)) return 'na';
  if (v >= 20) return 'r20p';
  if (v >= 10) return 'r20';
  if (v >= 5) return 'r10';
  if (v >= 0) return 'r5';
  return 'r0';
}

/**
 * 20営業日目の騰落率（%）の4段。20営業日目の終値 ÷ 翌営業日の寄り − 1（指値なしで持ち切り）。区切り（+10・0・−10）と色は
 * 結果（OUTCOME_CLASSES）と同じで、違うのは +10% の指値で売らないことだけ
 */
export const R20_CLASSES = [
  { id: 'hit', label: '+10% 以上', color: '#93c5fd' },
  { id: 'up', label: '0〜+10%', color: '#2563eb' },
  { id: 'down', label: '−10〜0%', color: '#b91c1c' },
  { id: 'bad', label: '−10% 未満', color: '#f87171' },
  { id: 'na', label: '未確定', color: NA_VALUE },
];

/** 時期の色（古い → 新しい。暗い面では新しいほど明るい。画面の百分位の濃淡と同じ青の段） */
export const TIME_STOPS = ['#1e3a8a', '#1d4ed8', '#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe'];

const hex = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));

/** 0〜1 の位置の時期の色（隣り合う段の間を線形に混ぜる）。 */
export function timeColor(f) {
  const x = Math.min(1, Math.max(0, f)) * (TIME_STOPS.length - 1);
  const k = Math.min(TIME_STOPS.length - 2, Math.floor(x));
  const a = hex(TIME_STOPS[k]);
  const b = hex(TIME_STOPS[k + 1]);
  const u = x - k;
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * u));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/**
 * 色の付け方。outcome = 結果、rise / r20 = 翌営業日の寄り（買値）から見た先20営業日の値動き（運用者の依頼 2026-10-19
 * 「ノードの色付け機能がほしい。先20日間の最大上昇率は最低限」→「最大上昇率というより、その日から最高到達価格（20日以内最高
 * 高値）までの上昇率」。選択「5段＋未確定」「基準は翌営業日の始値」「最安値までの下落率・20営業日目も同じ基準にそろえる」。
 * そのあと「安値の色付けは、なしで良いです」で最安値までの色を外した（値はツールチップと選んだ日に出す））、
 * l15 / l12 = 目的変数、time = 時期。value があるものは、ノードの値を表やツールチップに出す（short は表の見出し）
 */
export const COLOR_MODES = [
  { id: 'outcome', label: '結果（+10% / 20営業日目）',
    title: '翌営業日の寄りで買い、+10% の指値か20営業日目の終値で売ったときの収益' },
  { id: 'rise', label: '最高値までの上昇率（20日以内）', short: '最高値まで', value: 'rise',
    title: '20営業日以内の最高値（高値の最大）÷ 翌営業日の寄り − 1（10% 以上なら +10% の指値に届いた）' },
  { id: 'r20', label: '20営業日目の騰落率', short: '20日目', value: 'r20',
    title: '20営業日目の終値 ÷ 翌営業日の寄り − 1（指値なしで持ち切ったとき。20営業日目に売買が無かった日は未確定）' },
  { id: 'l15', label: '目的変数 1.5σ' },
  { id: 'l12', label: '目的変数 1.2σ' },
  { id: 'time', label: '時期' },
];

/** 区分で塗る色の付け方: 区分の並び（凡例）と、ノード i の区分。time は区分なし（連続の色） */
const BINS = {
  outcome: { classes: OUTCOME_CLASSES, of: (dec, i, target) => outcomeOf(dec.ret[i], target) },
  rise: { classes: RISE_CLASSES, of: (dec, i) => riseOf(dec.rise[i]) },
  r20: { classes: R20_CLASSES, of: (dec, i, target) => outcomeOf(dec.r20[i], target) },
};
const binOf = (mode) => BINS[mode]
  || (mode === 'time' ? null : { classes: LABEL_CLASSES, of: (dec, i) => labelClassOf(dec.labels[mode]?.[i]) });

/** 色の付け方の区分の並び（凡例に出す順）。区分の無い付け方（時期）は null */
export function classesOf(mode) {
  return binOf(mode)?.classes ?? null;
}

/** ノードごとの色（文字列の配列）。 */
export function nodeColors(dec, mode, target = DEFAULTS.target) {
  const out = new Array(dec.n);
  const bin = binOf(mode);
  const color = bin ? Object.fromEntries(bin.classes.map((c) => [c.id, c.color])) : null;
  for (let i = 0; i < dec.n; i += 1) {
    out[i] = bin ? color[bin.of(dec, i, target)] : timeColor(dec.n > 1 ? i / (dec.n - 1) : 1);
  }
  return out;
}

/** 色の区分ごとの件数（凡例に出す）。idx を渡すとそのノードだけ数える（今日に似た日の内訳）。区分の無い付け方は {} */
export function classCounts(dec, mode, target = DEFAULTS.target, idx = null) {
  const counts = {};
  const bin = binOf(mode);
  if (!bin) return counts;
  const add = (i) => { const c = bin.of(dec, i, target); counts[c] = (counts[c] || 0) + 1; };
  if (idx) for (const i of idx) add(i);
  else for (let i = 0; i < dec.n; i += 1) add(i);
  return counts;
}

// --------------------------------------------------------------------------- //
// 状態の値（丸めた z → 生の値の目安）
// --------------------------------------------------------------------------- //

/**
 * 丸めた z（整数）から、生の値の目安へ戻す（標準化を戻し、log1p は expm1・asinh は sinh × s で戻す）。切った値（上下の分位点）より
 * 外は分からないので、戻した値は切った値の中に収まる。丸め（1/q σ）のぶんの誤差がある。
 */
export function rawOf(param, zq, q = DEFAULTS.q) {
  if (!param) return NaN;
  const v = (zq / q) * param.std + param.mean;
  let x = v;
  if (param.kind === 'pos_heavy') x = Math.expm1(v);
  else if (param.kind === 'signed_heavy') x = Math.sinh(v) * (param.s || 1);
  if (param.lo !== null && param.lo !== undefined) x = Math.max(param.lo, x);
  if (param.hi !== null && param.hi !== undefined) x = Math.min(param.hi, x);
  return x;
}

/** ノード i の9列の z（σ 単位）。 */
export function zOf(dec, i) {
  const out = [];
  for (let d = 0; d < dec.dims; d += 1) out.push(dec.z[i * dec.dims + d] / dec.q);
  return out;
}
