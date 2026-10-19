/**
 * 力学レイアウト（src/lib/forcegraph.js。リカレンス・ネットワークのタブの図）の単体テスト。
 *
 * - 反発の八分木（Barnes–Hut）は、θ をほぼ 0 にすると総当たりと一致する（θ = 0.9 は近似）
 * - 同じ種なら同じ形（同じ候補はいつも同じ並び）。2次元にすると z は 0 のまま
 * - ばねの強さは linkStrength ÷ 両端の次数の小さいほう で、上限 1（力のつまみで変えると数え直す）
 * - 中心の力で孤立した点も遠くへ飛ばない。投影と、画面の点からいちばん近いノードを選ぶ
 *
 *   node --test tests/forcegraph.test.js
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createSim, tick, setDim, setForces, extent, project, pick, rng, seedOf, FORCE_DEFAULTS,
} from '../src/lib/forcegraph.js';

const N = 200;
const none = new Uint32Array(0);
const deg0 = new Int32Array(N);

function bruteRepulsion(sim) {
  const { pos, opts } = sim;
  const c = opts.charge * sim.alpha;
  const dmin2 = opts.distMin * opts.distMin;
  const f = new Float64Array(3 * sim.n);
  for (let i = 0; i < sim.n; i += 1) {
    for (let q = 0; q < sim.n; q += 1) {
      if (q === i) continue;
      const dx = pos[3 * q] - pos[3 * i];
      const dy = pos[3 * q + 1] - pos[3 * i + 1];
      const dz = pos[3 * q + 2] - pos[3 * i + 2];
      let l = dx * dx + dy * dy + dz * dz;
      if (l < dmin2) l = Math.sqrt(dmin2 * l);
      f[3 * i] += (dx * c) / l;
      f[3 * i + 1] += (dy * c) / l;
      f[3 * i + 2] += (dz * c) / l;
    }
  }
  return f;
}

test('反発の八分木は θ ≈ 0 で総当たりと一致する', () => {
  const sim = createSim(N, none, deg0, { seed: 3, theta: 1e-9, gravity: 0, velocityDecay: 0 });
  const want = bruteRepulsion(sim);
  const before = Float64Array.from(sim.pos);
  tick(sim);
  let err = 0;
  let mag = 0;
  for (let k = 0; k < 3 * N; k += 1) {
    err = Math.max(err, Math.abs(sim.pos[k] - before[k] - want[k]));
    mag = Math.max(mag, Math.abs(want[k]));
  }
  assert.ok(mag > 1, `力が働いている (${mag})`);
  assert.ok(err < 1e-9, `総当たりとの差 ${err}`);
  // θ = 0.9（既定）は近似だが、大きくは外れない
  const approx = createSim(N, none, deg0, { seed: 3, gravity: 0, velocityDecay: 0 });
  tick(approx);
  let e2 = 0;
  for (let k = 0; k < 3 * N; k += 1) e2 = Math.max(e2, Math.abs(approx.pos[k] - before[k] - want[k]));
  assert.ok(e2 < 0.25 * mag, `θ = 0.9 の誤差 ${e2} / ${mag}`);
});

test('同じ種なら同じ形・2次元は z が 0 のまま', () => {
  const r = rng(11);
  const edges = [];
  for (let k = 0; k < 600; k += 1) edges.push(Math.floor(r() * N), Math.floor(r() * N));
  const E = Uint32Array.from(edges);
  const deg = new Int32Array(N);
  for (const v of E) deg[v] += 1;
  const run = (seed, dim) => {
    const s = createSim(N, E, deg, { seed, dim });
    while (tick(s));
    return s;
  };
  const a = run(seedOf('46840-2026-10-16-100'), 3);
  const b = run(seedOf('46840-2026-10-16-100'), 3);
  assert.deepEqual(Array.from(a.pos), Array.from(b.pos));
  const c = run(seedOf('46840-2026-10-16-200'), 3);
  assert.notDeepEqual(Array.from(a.pos), Array.from(c.pos));
  const flat = run(5, 2);
  for (let i = 0; i < N; i += 1) assert.equal(flat.pos[3 * i + 2], 0);
  // 3次元 → 2次元に切り替えても z は 0 になり、動かし直す
  setDim(a, 2);
  assert.ok(a.alpha >= 0.99);
  while (tick(a));
  for (let i = 0; i < N; i += 1) assert.equal(a.pos[3 * i + 2], 0);
  // 孤立した点も中心の力で遠くへは飛ばない（重心からの距離の 98% 点の 3倍の中）
  const ext = extent(a.pos, N, 0.98);
  for (let i = 0; i < N; i += 1) {
    const d = Math.hypot(a.pos[3 * i] - ext.cx, a.pos[3 * i + 1] - ext.cy, a.pos[3 * i + 2] - ext.cz);
    assert.ok(d < 3 * ext.radius);
  }
});

test('ばねの強さは linkStrength ÷ 次数の小さいほう（上限 1）・つまみで数え直す', () => {
  const E = Uint32Array.from([0, 1, 1, 2, 2, 3]);
  const deg = Int32Array.from([1, 2, 8, 8]);
  const s = createSim(4, E, deg, { seed: 1, linkStrength: 4 });
  assert.deepEqual(Array.from(s.strength), [1, 1, 0.5]);       // min(1, 4/1)・min(1, 4/2)・4/8
  assert.equal(FORCE_DEFAULTS.linkStrength, 4);
  while (tick(s));
  assert.equal(setForces(s, { linkStrength: 2, gravity: 0.3 }), true);
  assert.deepEqual(Array.from(s.strength), [1, 1, 0.25]);
  assert.equal(s.opts.gravity, 0.3);
  assert.ok(s.alpha >= 0.6);                                    // 動かし直す
  assert.equal(setForces(s, { linkStrength: 2 }), false);       // 変わらなければ何もしない
});

test('投影と、いちばん近いノードを選ぶ', () => {
  const pos = Float64Array.from([0, 0, 0, 100, 0, 0, -100, 0, 0]);
  const ext = { cx: 0, cy: 0, cz: 0, radius: 100 };
  const out = new Float64Array(12);
  project(pos, 3, ext, { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 }, 400, 400, 3, out);
  assert.equal(out[0], 200);                                    // 原点は真ん中
  assert.equal(out[1], 200);
  assert.ok(out[4] > 200 && out[8] < 200);                      // 右と左
  assert.equal(pick(out, 3, out[4] + 3, out[5] - 2), 1);
  assert.equal(pick(out, 3, 5, 5), -1);                         // 遠ければ選ばない
  // 2次元は透視しない（倍率 1）
  project(pos, 3, ext, { yaw: 0, pitch: 0, zoom: 1, panX: 10, panY: 0 }, 400, 400, 2, out);
  assert.equal(out[2], 1);
  assert.equal(out[0], 210);
});
