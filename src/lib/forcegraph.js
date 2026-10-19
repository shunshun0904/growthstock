/**
 * 3次元の力学レイアウト（Obsidian のグラフ表示と同じ考え方: 辺はばね、ノードどうしは反発、中心へ弱く引く）と、回転・透視の
 * 投影。画面の「リカレンス・ネットワーク」タブ（src/views/NetworkView.jsx）が使う。依存は無し（ノード 2,200・辺 2.4万なら
 * canvas 2D で足りるので three.js などは入れない）。
 *
 * 力（d3-force の既定の式に合わせた）
 *   ばね: 辺の長さを linkDistance へ。強さは linkStrength ÷ 両端の次数の小さいほう（上限 1）、動く割合は次数で分ける
 *         （次数の多い端が動きにくい）
 *   反発: すべての組。Barnes–Hut の八分木で近似する（θ = 0.9。遠い塊は重心にまとめる）。強さ charge（負）、距離の2乗の下限 distMin²
 *   中心: 原点へ gravity × 位置（孤立した日が遠くへ飛ばない。Obsidian の「中心の力」。孤立した日は塊の外側に殻のように並ぶ）
 *   速度は毎回 (1 − velocityDecay) 倍。alpha は 1 から alphaMin まで ticks 回で下がる
 * 2次元にすると z を 0 に固定する（同じ八分木がそのまま四分木として働く）。
 */

/** 決まった種の乱数（mulberry32）。同じ候補はいつも同じ形に並ぶ。 */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 文字列から種を作る（銘柄コード・日付ごとに決まる）。 */
export function seedOf(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * 既定の力。実データ（2026-10-16 の候補。ノード 700〜2,200・辺の密度 1%）で比べて決めた。測った量は、球らしさ（線のある点の
 * いちばん短い軸 ÷ 長い軸）・辺の長さ ÷ 無作為な2点の距離（小さいほど塊の構造が見える）・近い点の間隔 ÷ 2点の距離（小さすぎると重なる）
 *   反発 −30・中心 0.06・ばね 1（d3 の既定）        球らしさ 0.47〜0.64
 *   反発 −100・中心 0.2・ばね 1                      球らしさ 0.63〜0.72・辺 0.23〜0.25（丸いが構造が薄い）
 *   反発 −80・中心 0.18・ばね 4（既定。上限 1）       球らしさ 0.39〜0.60・辺 0.18〜0.20・間隔 0.04〜0.06
 * ばねは 1 を超えると行き過ぎて潰れる（倍率 6 で上限なしだと全点が1か所に集まった）ので上限 1。画面のつまみで変えられる
 */
export const FORCE_DEFAULTS = {
  linkDistance: 30, linkStrength: 4, charge: -80, gravity: 0.18, theta: 0.9, distMin: 1, velocityDecay: 0.4,
  alphaMin: 0.002, ticks: 300,
};

const MAX_DEPTH = 24;

/** 八分木の器（ノード数の上限を見込んで一度だけ確保し、毎回使い回す）。 */
function makeTree(n) {
  const cap = Math.max(64, 8 * n + 16);
  return {
    cap, count: 0,
    ox: new Float64Array(cap), oy: new Float64Array(cap), oz: new Float64Array(cap), half: new Float64Array(cap),
    child: new Int32Array(cap * 8), leaf: new Int32Array(cap), internal: new Uint8Array(cap),
    mass: new Float64Array(cap), cx: new Float64Array(cap), cy: new Float64Array(cap), cz: new Float64Array(cap),
    next: new Int32Array(n),
  };
}

function newNode(T, ox, oy, oz, half) {
  const k = T.count;
  T.count += 1;
  T.ox[k] = ox; T.oy[k] = oy; T.oz[k] = oz; T.half[k] = half;
  T.child.fill(-1, k * 8, k * 8 + 8);
  T.leaf[k] = -1;
  T.internal[k] = 0;
  T.mass[k] = 0;
  return k;
}

function octantOf(T, k, x, y, z) {
  return (x >= T.ox[k] ? 1 : 0) | (y >= T.oy[k] ? 2 : 0) | (z >= T.oz[k] ? 4 : 0);
}

function childOf(T, k, oct) {
  let c = T.child[k * 8 + oct];
  if (c < 0) {
    const h = T.half[k] / 2;
    c = newNode(T, T.ox[k] + (oct & 1 ? h : -h), T.oy[k] + (oct & 2 ? h : -h), T.oz[k] + (oct & 4 ? h : -h), h);
    T.child[k * 8 + oct] = c;
  }
  return c;
}

function buildTree(T, pos, n) {
  let x0 = Infinity; let y0 = Infinity; let z0 = Infinity;
  let x1 = -Infinity; let y1 = -Infinity; let z1 = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const x = pos[3 * i]; const y = pos[3 * i + 1]; const z = pos[3 * i + 2];
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
    if (z < z0) z0 = z; if (z > z1) z1 = z;
  }
  T.count = 0;
  const half = Math.max(x1 - x0, y1 - y0, z1 - z0) / 2 + 1e-6;
  newNode(T, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, half);
  for (let p = 0; p < n; p += 1) {
    const x = pos[3 * p]; const y = pos[3 * p + 1]; const z = pos[3 * p + 2];
    let k = 0;
    let depth = 0;
    for (;;) {
      if (!T.internal[k]) {
        const q = T.leaf[k];
        if (q < 0) { T.leaf[k] = p; T.next[p] = -1; break; }
        const same = pos[3 * q] === x && pos[3 * q + 1] === y && pos[3 * q + 2] === z;
        if (same || depth >= MAX_DEPTH || T.count + 2 >= T.cap) { T.next[p] = q; T.leaf[k] = p; break; }
        // 葉を分ける: いまの点（同じ位置の鎖ごと）を子へ移し、続けて p を入れる
        T.internal[k] = 1;
        T.leaf[k] = -1;
        const cq = childOf(T, k, octantOf(T, k, pos[3 * q], pos[3 * q + 1], pos[3 * q + 2]));
        T.leaf[cq] = q;
      }
      k = childOf(T, k, octantOf(T, k, x, y, z));
      depth += 1;
    }
  }
  // 重心（子は親より後に作られるので、後ろから数えれば子が先にそろう）
  for (let k = T.count - 1; k >= 0; k -= 1) {
    let m = 0; let sx = 0; let sy = 0; let sz = 0;
    if (!T.internal[k]) {
      for (let q = T.leaf[k]; q >= 0; q = T.next[q]) {
        m += 1; sx += pos[3 * q]; sy += pos[3 * q + 1]; sz += pos[3 * q + 2];
      }
    } else {
      for (let o = 0; o < 8; o += 1) {
        const c = T.child[k * 8 + o];
        if (c < 0 || T.mass[c] === 0) continue;
        m += T.mass[c]; sx += T.cx[c] * T.mass[c]; sy += T.cy[c] * T.mass[c]; sz += T.cz[c] * T.mass[c];
      }
    }
    T.mass[k] = m;
    if (m > 0) { T.cx[k] = sx / m; T.cy[k] = sy / m; T.cz[k] = sz / m; }
  }
}

/**
 * レイアウトの状態。edges は [i0, j0, i1, j1, …]、deg は次数。初めの位置は種で決まる球の中の点（同じ候補はいつも同じ形）。
 */
export function createSim(n, edges, deg, { dim = 3, seed = 1, ...opts } = {}) {
  const o = { ...FORCE_DEFAULTS, ...opts };
  const rand = rng(seed);
  const pos = new Float64Array(3 * n);
  const vel = new Float64Array(3 * n);
  const r0 = o.linkDistance * Math.cbrt(n) * 1.2;
  for (let i = 0; i < n; i += 1) {
    // 球の中に一様に（向きは球面の一様、半径は立方根）
    const u = rand() * 2 - 1;
    const th = rand() * 2 * Math.PI;
    const r = r0 * Math.cbrt(rand());
    const s = Math.sqrt(1 - u * u);
    pos[3 * i] = r * s * Math.cos(th);
    pos[3 * i + 1] = r * s * Math.sin(th);
    pos[3 * i + 2] = dim === 3 ? r * u : 0;
  }
  const nE = edges.length / 2;
  const strength = new Float64Array(nE);
  const bias = new Float64Array(nE);
  const minDeg = new Float64Array(nE);
  for (let e = 0; e < nE; e += 1) {
    const a = deg[edges[2 * e]] || 1;
    const b = deg[edges[2 * e + 1]] || 1;
    minDeg[e] = Math.min(a, b);
    strength[e] = Math.min(1, o.linkStrength / minDeg[e]);   // 1 を超えると行き過ぎて揺れる・潰れる
    bias[e] = a / (a + b);
  }
  return {
    n, dim, pos, vel, edges, strength, bias, minDeg, opts: o, alpha: 1,
    alphaDecay: 1 - Math.pow(o.alphaMin, 1 / o.ticks), tree: makeTree(n), jiggle: rng(seed ^ 0x9e3779b9),
  };
}

/** もう一度動かす（密度を変えた・2D / 3D を切り替えた・並べ直す）。 */
export function reheat(sim, alpha = 0.8) {
  sim.alpha = Math.max(sim.alpha, alpha);
}

/**
 * 力を変える（画面のつまみ。Obsidian のグラフ表示の「中心の力・反発の力・リンクの力・リンクの長さ」）。変えた分だけ動かし直す。
 */
export function setForces(sim, forces = {}) {
  const o = sim.opts;
  let changed = false;
  for (const k of ['gravity', 'charge', 'linkDistance', 'linkStrength']) {
    if (Number.isFinite(forces[k]) && forces[k] !== o[k]) { o[k] = forces[k]; changed = true; }
  }
  if (!changed) return false;
  const nE = sim.edges.length / 2;
  for (let e = 0; e < nE; e += 1) sim.strength[e] = Math.min(1, o.linkStrength / sim.minDeg[e]);
  reheat(sim, 0.6);
  return true;
}

/** 2次元 / 3次元の切り替え。2次元は z を 0 に固定する。3次元へ戻すときは z を少し散らしてから動かす。 */
export function setDim(sim, dim) {
  if (sim.dim === dim) return;
  sim.dim = dim;
  for (let i = 0; i < sim.n; i += 1) {
    sim.pos[3 * i + 2] = dim === 2 ? 0 : (sim.jiggle() - 0.5) * sim.opts.linkDistance * 4;
    sim.vel[3 * i + 2] = 0;
  }
  reheat(sim, 1);
}

/** 1回ぶん動かす。動き終わっていれば false。 */
export function tick(sim) {
  if (sim.alpha < sim.opts.alphaMin) return false;
  const { n, pos, vel, edges, strength, bias, opts, tree: T } = sim;
  const alpha = sim.alpha;
  const flat = sim.dim === 2;
  // ばね
  const nE = edges.length / 2;
  for (let e = 0; e < nE; e += 1) {
    const s = edges[2 * e];
    const t = edges[2 * e + 1];
    let dx = pos[3 * t] + vel[3 * t] - pos[3 * s] - vel[3 * s];
    let dy = pos[3 * t + 1] + vel[3 * t + 1] - pos[3 * s + 1] - vel[3 * s + 1];
    let dz = flat ? 0 : pos[3 * t + 2] + vel[3 * t + 2] - pos[3 * s + 2] - vel[3 * s + 2];
    let l = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
    l = ((l - opts.linkDistance) / l) * alpha * strength[e];
    dx *= l; dy *= l; dz *= l;
    const b = bias[e];
    vel[3 * t] -= dx * b; vel[3 * t + 1] -= dy * b; vel[3 * t + 2] -= dz * b;
    vel[3 * s] += dx * (1 - b); vel[3 * s + 1] += dy * (1 - b); vel[3 * s + 2] += dz * (1 - b);
  }
  // 反発（Barnes–Hut）
  buildTree(T, pos, n);
  const theta2 = opts.theta * opts.theta;
  const dmin2 = opts.distMin * opts.distMin;
  const c = opts.charge * alpha;
  const stack = new Int32Array(Math.max(64, 8 * MAX_DEPTH + 64));
  for (let i = 0; i < n; i += 1) {
    const xi = pos[3 * i]; const yi = pos[3 * i + 1]; const zi = pos[3 * i + 2];
    let fx = 0; let fy = 0; let fz = 0;
    let sp = 0;
    stack[sp] = 0; sp += 1;
    while (sp > 0) {
      sp -= 1;
      const k = stack[sp];
      if (T.mass[k] === 0) continue;
      if (!T.internal[k]) {
        for (let q = T.leaf[k]; q >= 0; q = T.next[q]) {
          if (q === i) continue;
          let dx = pos[3 * q] - xi; let dy = pos[3 * q + 1] - yi; let dz = pos[3 * q + 2] - zi;
          let l = dx * dx + dy * dy + dz * dz;
          if (l === 0) {
            dx = (sim.jiggle() - 0.5) * 1e-6; dy = (sim.jiggle() - 0.5) * 1e-6; dz = flat ? 0 : (sim.jiggle() - 0.5) * 1e-6;
            l = dx * dx + dy * dy + dz * dz;
          }
          if (l < dmin2) l = Math.sqrt(dmin2 * l);
          fx += (dx * c) / l; fy += (dy * c) / l; fz += (dz * c) / l;
        }
        continue;
      }
      const dx = T.cx[k] - xi; const dy = T.cy[k] - yi; const dz = T.cz[k] - zi;
      let l = dx * dx + dy * dy + dz * dz;
      const w = 2 * T.half[k];
      if ((w * w) / theta2 < l) {
        if (l < dmin2) l = Math.sqrt(dmin2 * l);
        const f = (c * T.mass[k]) / l;
        fx += dx * f; fy += dy * f; fz += dz * f;
        continue;
      }
      for (let oc = 0; oc < 8; oc += 1) {
        const ch = T.child[k * 8 + oc];
        if (ch >= 0 && sp < stack.length) { stack[sp] = ch; sp += 1; }
      }
    }
    vel[3 * i] += fx; vel[3 * i + 1] += fy; vel[3 * i + 2] += flat ? 0 : fz;
  }
  // 中心へ・速度の減衰・位置
  const g = opts.gravity * alpha;
  const keep = 1 - opts.velocityDecay;
  for (let i = 0; i < n; i += 1) {
    for (let d = 0; d < 3; d += 1) {
      const k = 3 * i + d;
      if (flat && d === 2) { vel[k] = 0; pos[k] = 0; continue; }
      vel[k] = (vel[k] - pos[k] * g) * keep;
      pos[k] += vel[k];
    }
  }
  sim.alpha += (0 - sim.alpha) * sim.alphaDecay;
  return true;
}

/** 重心と、重心からの距離の q 分位（画面に収める大きさ。遠い孤立点に引っ張られないように最大ではなく分位で）。 */
export function extent(pos, n, q = 0.98) {
  let mx = 0; let my = 0; let mz = 0;
  for (let i = 0; i < n; i += 1) { mx += pos[3 * i]; my += pos[3 * i + 1]; mz += pos[3 * i + 2]; }
  mx /= n || 1; my /= n || 1; mz /= n || 1;
  const r = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const dx = pos[3 * i] - mx; const dy = pos[3 * i + 1] - my; const dz = pos[3 * i + 2] - mz;
    r[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  r.sort();
  return { cx: mx, cy: my, cz: mz, radius: Math.max(1, r[Math.min(n - 1, Math.floor(q * (n - 1)))] || 1) };
}

/**
 * 投影。view = { yaw, pitch, zoom, panX, panY }（角度はラジアン）、ext = extent の戻り値。
 * out（長さ 4n）に [画面 x, 画面 y, 大きさの倍率, 奥行き（手前ほど大きい）] を書く。透視は 3次元のときだけ。
 */
export function project(pos, n, ext, view, width, height, dim, out) {
  const cy = Math.cos(view.yaw); const sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch); const sp = Math.sin(view.pitch);
  const R = ext.radius;
  const base = (Math.min(width, height) * 0.46 * view.zoom) / R;
  const cam = 3.2 * R;
  const ox = width / 2 + view.panX;
  const oy = height / 2 + view.panY;
  for (let i = 0; i < n; i += 1) {
    const x = pos[3 * i] - ext.cx;
    const y = pos[3 * i + 1] - ext.cy;
    const z = pos[3 * i + 2] - ext.cz;
    let x1; let y2; let z2;
    if (dim === 3) {
      x1 = cy * x + sy * z;
      const z1 = -sy * x + cy * z;
      y2 = cp * y - sp * z1;
      z2 = sp * y + cp * z1;
    } else {
      x1 = cy * x - sy * y;            // 2次元は画面の中で回すだけ
      y2 = sy * x + cy * y;
      z2 = 0;
    }
    const persp = dim === 3 ? cam / (cam - z2) : 1;
    out[4 * i] = ox + x1 * base * persp;
    out[4 * i + 1] = oy + y2 * base * persp;
    out[4 * i + 2] = persp;
    out[4 * i + 3] = z2 / R;
  }
  return out;
}

/** 画面の点 (x, y) にいちばん近いノード（maxPx より遠ければ −1）。 */
export function pick(screen, n, x, y, maxPx = 14) {
  let best = -1;
  let bd = maxPx * maxPx;
  let bz = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const dx = screen[4 * i] - x;
    const dy = screen[4 * i + 1] - y;
    const d = dx * dx + dy * dy;
    // ほぼ同じ距離なら手前を選ぶ
    if (d < bd - 1 || (d <= bd + 1 && screen[4 * i + 3] > bz)) {
      if (d <= maxPx * maxPx) { best = i; bd = d; bz = screen[4 * i + 3]; }
    }
  }
  return best;
}
