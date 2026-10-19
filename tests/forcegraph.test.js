/**
 * 力学レイアウト（src/lib/forcegraph.js。リカレンス・ネットワークのタブの図）の単体テスト。
 *
 * - 反発の八分木（Barnes–Hut）は、θ をほぼ 0 にすると総当たりと一致する（θ = 0.9 は近似）
 * - 同じ種なら同じ形（同じ候補はいつも同じ並び）。2次元にすると z は 0 のまま
 * - ばねの強さは linkStrength ÷ 両端の次数の小さいほう で、上限 1（力のつまみで変えると数え直す）
 * - 中心の力で孤立した点も遠くへ飛ばない。投影と、画面の点からいちばん近いノードを選ぶ
 * - ほかの並べ方（2026-10-19 夜）: 対称行列の固有分解（ヤコビ法）が A v = λ v を満たす。状態の地図は主成分に置き、状態が3次元の
 *   部分空間にあれば距離をそのまま保つ・同じ状態は同じ位置。時間のらせんは1年（TURN_DAYS 営業日）で1周・今日がいちばん上
 *
 *   node --test tests/forcegraph.test.js
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createSim, tick, setDim, setForces, extent, project, pick, rng, seedOf, FORCE_DEFAULTS,
  symEigen, statePositions, helixPositions, TURN_DAYS,
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

test('対称行列の固有分解（ヤコビ法）', () => {
  const two = symEigen([2, 1, 1, 2], 2);
  assert.ok(Math.abs(two.values[0] - 3) < 1e-12 && Math.abs(two.values[1] - 1) < 1e-12);
  assert.ok(Math.abs(two.vectors[0][0] - Math.SQRT1_2) < 1e-12 && Math.abs(two.vectors[0][1] - Math.SQRT1_2) < 1e-12);
  // 乱数の 9 × 9: A v = λ v・長さ 1・互いに直交・大きい順
  const r = rng(7);
  const d = 9;
  const b = Array.from({ length: d * d }, () => r() * 2 - 1);
  const a = new Float64Array(d * d);
  for (let i = 0; i < d; i += 1) for (let j = 0; j < d; j += 1) a[i * d + j] = (b[i * d + j] + b[j * d + i]) / 2;
  const { values, vectors } = symEigen(a, d);
  for (let x = 1; x < d; x += 1) assert.ok(values[x - 1] >= values[x] - 1e-12);
  vectors.forEach((v, x) => {
    for (let i = 0; i < d; i += 1) {
      let av = 0;
      for (let j = 0; j < d; j += 1) av += a[i * d + j] * v[j];
      assert.ok(Math.abs(av - values[x] * v[i]) < 1e-9, `A v = λ v (${x}, ${i})`);
    }
    vectors.forEach((w, y) => {
      const dot = v.reduce((sum, vi, i) => sum + vi * w[i], 0);
      assert.ok(Math.abs(dot - (x === y ? 1 : 0)) < 1e-9);
    });
  });
});

test('状態の地図: 主成分に置く（3次元の部分空間なら距離を保つ・同じ状態は同じ位置）', () => {
  // 9次元の中の3つの直交する向きの組み合わせだけで作った状態（整数）
  const dims = 9;
  const u = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, -1, 1, -1, 0, 0]];
  const r = rng(11);
  const n = 60;
  const z = new Int16Array(n * dims);
  for (let i = 0; i < n; i += 1) {
    const c = [Math.round((r() - 0.5) * 80), Math.round((r() - 0.5) * 30), Math.round((r() - 0.5) * 10)];
    for (let d = 0; d < dims; d += 1) z[i * dims + d] = c[0] * u[0][d] + c[1] * u[1][d] + c[2] * u[2][d];
  }
  for (let d = 0; d < dims; d += 1) z[58 * dims + d] = z[59 * dims + d];       // 同じ状態を2つ（58 を 59 にそろえる）
  const { pos, share } = statePositions(z, n, dims, 3);
  assert.ok(Math.abs(share[0] + share[1] + share[2] - 1) < 1e-9);    // 分散はすべて上位3つに入る
  assert.ok(share[0] >= share[1] && share[1] >= share[2]);
  const dz = (i, j) => Math.hypot(...Array.from({ length: dims }, (_, d) => z[i * dims + d] - z[j * dims + d]));
  const dp = (i, j) => Math.hypot(pos[3 * i] - pos[3 * j], pos[3 * i + 1] - pos[3 * j + 1], pos[3 * i + 2] - pos[3 * j + 2]);
  for (let i = 0; i < n; i += 7) for (let j = 0; j < n; j += 5) assert.ok(Math.abs(dz(i, j) - dp(i, j)) < 1e-6, `${i}-${j}`);
  assert.deepEqual([pos[3 * 58], pos[3 * 58 + 1], pos[3 * 58 + 2]], [pos[3 * 59], pos[3 * 59 + 1], pos[3 * 59 + 2]]);
  // 2次元は上位2つだけ（z は 0）
  const flat = statePositions(z, n, dims, 2).pos;
  for (let i = 0; i < n; i += 1) assert.equal(flat[3 * i + 2], 0);
});

test('時間のらせん: 1年で1周・今日がいちばん上（2次元は外へ広がるうずまき）', () => {
  const n = 2 * TURN_DAYS + 1;
  const p = helixPositions(n, 3);
  for (let i = 0; i + TURN_DAYS < n; i += 37) {
    assert.ok(Math.abs(p[3 * i] - p[3 * (i + TURN_DAYS)]) < 1e-9 && Math.abs(p[3 * i + 2] - p[3 * (i + TURN_DAYS) + 2]) < 1e-9);
    assert.ok(Math.abs(Math.hypot(p[3 * i], p[3 * i + 2]) - 100) < 1e-9);      // 半径は一定
  }
  for (let i = 1; i < n; i += 1) assert.ok(p[3 * i + 1] < p[3 * (i - 1) + 1]);   // 新しい日ほど上（画面の上はマイナス）
  const q = helixPositions(n, 2);
  for (let i = 1; i < n; i += 1) {
    assert.equal(q[3 * i + 2], 0);
    assert.ok(Math.hypot(q[3 * i], q[3 * i + 1]) > Math.hypot(q[3 * (i - 1)], q[3 * (i - 1) + 1]));
  }
});
