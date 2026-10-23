import React, { useEffect, useRef, useState } from 'react';
import { createSim, tick, setForces, reheat, extent, project, pick, seedOf } from '../lib/forcegraph.js';

/**
 * リカレンス・ネットワークの描画（canvas 2D。力学レイアウトで3次元に並べ、ドラッグで回して見る）。
 *
 * - 点 = 1営業日（大きさは次数 = 似た日として選ばれた数を含む線の数）、線 = 状態の似た日どうし（k近傍）。今日（候補の日）は
 *   白い大きな点、78週高値の更新日は紫の輪
 * - 並べ方は力学だけ（運用者の指定 2026-10-19「3種のプロット方法がありますが、今のデフォルトのみのもので良いです」。状態の
 *   地図・時間のらせんは外した）。図はいつも3次元で、触らなければ止まっている（2026-10-20 の指定「回転機能と２D機能はいらない
 *   です」で、自動の回転と 2D の表示は外した）
 * - 見た目（運用者の依頼 2026-10-19「無理に球状にしなくても3次元ネットワーク構造でかっこよく」）: 暗い面に、薄い明るい線を
 *   重ねる（密な塊ほど明るく見える）、点は奥ほど小さく暗く（霧）、手前の点はうっすら光る。似た日は光の輪で強調。線と点の
 *   にじみは通常の重ね方で描く（足し算の重ね方 lighter は、GPU の無い描画で線 1.6万本が約8倍遅かった。足し算は強調する
 *   少しの線と光だけに使う）
 * - 強調（同日の依頼「対象日と似ている（紐づく）サンプルもハイライト」）: 触れた日 → 選んだ日 → 今日 の順に、その日の似た日
 *   （その日に近い k 日）へ細い線を引き、似た日はその日の色のまま少し大きくして、地の色の縁と細い控えめな輪で囲む（2026-10-20 の
 *   依頼「白く光らせてますが、それ自身が何色かわからなくなるので、もう少し抑えた表現に」。前は足し算の光と白い輪・光る線）。ほかの
 *   点と線は暗くしない（同日の依頼「他はくらくしないでそのままに」）。今日の強調は highlightToday で消せる
 * - 直近の軌跡（同日の依頼「単純に線で結ぶよりも、日付の順方向に流れているような（流れがわかるような）可視化」「動的にするので
 *   あれば、色自体はもう少し暗くてもよい」、2026-10-20「ビュンビュンしすぎなので、もう少し落ち着かせてほしい」「モーションさえ
 *   つければ若干他よりも線を太くする程度で、色自体は暗いままで良い」、続けて「破線もなしでよいです。モーションさえあればよい」）:
 *   暗めの橙の実線（ほかの線より少し太い。古い日ほど薄い）の上を、同じ暗い橙のやわらかい明るみが古い日から今日の向きへゆっくり
 *   流れる（画面の上で一定の速さ FLOW_PX_PER_S。前は光の粒が1日を同じ時間で進み、日の間の長いところで速く飛んでいた。その次は
 *   破線）。点の下に描く。動きを減らす設定（prefers-reduced-motion。flow = false）では、流さずに向きの矢印を置く
 * - 操作: ドラッグで回す（Shift・右ボタンはずらす）。ホイール・2本指でつまんで拡大縮小（同日の依頼「指で拡大縮小」。指の
 *   間・カーソルの位置を中心に）。2本指を動かすとずらす。点に触れる（タップ）とその日の値と似た日を出し、クリックで選ぶ
 *   （もう一度・何も無いところで外す）。ダブルクリックで元の大きさ。矢印キーで回し、＋ − で拡大、0 で元に戻す
 * - 点の大きさ（運用者の依頼 2026-10-20「プロットのサイズをもう少し大きく」→「図自体ではなく、点のサイズですかね」）: 画面の
 *   「点の大きさ」のバーの倍率 pointScale は、今日・触れた（選んだ）日・その似た日のどれでもない点にだけ掛ける。この3つの点は
 *   いつもバーの最初（1.5 倍）のときの大きさ（運用者の指定 2026-10-23「選択された点のサイズはそのままで良いです。選択された
 *   日付のノードも比例しておおきくしてしまうと、その他のノードのサイズを変える意味がなくなってしまうので」「選択された類似の点も
 *   そのままで良いです。全く関係ない他のノードを調整したいので」）。触れる範囲は点に合わせて広げる
 * - 計算は毎フレーム 10ms までに区切り、動き終わったら描き直しが要るときだけ描く
 */

const BG_IN = '#101a33';
const BG_OUT = '#04060c';
/** 線の色。奥と手前で濃さを変える */
const EDGE_RGB_BACK = '96, 165, 250';
const EDGE_RGB_FRONT = '147, 197, 253';
const EDGE_ALPHA = { back: 0.06, front: 0.12 };
/** 強調する日から似た日への線と、似た日の外の輪（控えめな薄い灰。光らせない） */
const LINK = 'rgba(203, 213, 225, 0.6)';
/** 今日・触れた（選んだ）日・その似た日の点の倍率（「点の大きさ」のバーによらず、バーの最初の 1.5 倍のときと同じ大きさ。
 *  src/views/NetworkView.jsx の POINT_SCALE.def） */
const MARK_SCALE = 1.5;
/** 直近の軌跡: 暗めの橙の実線（前は明るい黄 251, 191, 36）と、その上をゆっくり今日の向きへ流れる同じ色の明るみ */
const TRAIL_RGB = '217, 119, 6';
const FLOW_PX_PER_S = 12;    // 明るみが今日の向きへ流れる速さ（px/秒。画面の上で一定なので、日の間が長いところでも速くならない）
const FLOW_GAP_PX = 60;      // 明るみの間隔（px）
const FLOW_TAIL_PX = 30;     // 明るみの長さ（px。後ろほど薄い）
const FLOW_FRAME_MS = 50;    // 明るみだけが動くとき（並べ終わって、触っていない）の描き直しの間隔（約20コマ/秒）
const ARROW_EVERY = 4;       // 動きを減らす設定のときの矢印の間隔（日）
const RING = '#a78bfa';
const TEXT = '#e8edf6';
const ZOOM_MIN = 0.35;
const ZOOM_MAX = 12;

const clampZoom = (z) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z));

export default function RecurrenceGraph({
  dec, net, nbrs, colors, trail, seedKey, selected, onSelect, tipOf, ariaLabel, zoomCmd, forces,
  highlightToday = true, flow = true, pointScale = 1,
}) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const st = useRef({
    sim: null, view: { yaw: 0.6, pitch: -0.3, zoom: 1, panX: 0, panY: 0 }, screen: null, order: null,
    hover: -1, drag: null, pinch: null, ptrs: new Map(), dirty: true, ext: null, size: { w: 0, h: 0, dpr: 1 }, raf: 0, flowAt: 0,
    nbSet: null, nbFor: -2, nbNet: null, bg: null,
  });
  const props = useRef({});
  props.current = { dec, net, nbrs, colors, trail, selected, highlightToday, flow, pointScale };
  const [tip, setTip] = useState(null);

  // ネットワークが変わったら並べ直す（同じ候補・同じ k ならいつも同じ形。力学の種は銘柄・日付・k）
  useEffect(() => {
    const s = st.current;
    if (!dec || !net) { s.sim = null; return; }
    s.sim = createSim(dec.n, net.edges, net.deg, { seed: seedOf(seedKey || ''), ...(forces || {}) });
    s.screen = new Float64Array(4 * dec.n);
    s.order = new Int32Array(dec.n).map((_, i) => i);
    s.hover = -1;
    s.ext = null;
    s.nbFor = -2;
    s.dirty = true;
    setTip(null);
  }, [dec, net]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { st.current.dirty = true; }, [colors, trail, selected, highlightToday, nbrs, flow, pointScale]);

  // 力のつまみ（中心の力・反発の力・リンクの力・リンクの長さ）。変えた分だけ動かし直す
  useEffect(() => {
    const s = st.current;
    if (s.sim && forces && setForces(s.sim, forces)) { s.fit = true; s.dirty = true; }
  }, [forces]);

  // 外からの拡大・元に戻す（＋ − ボタン）
  useEffect(() => {
    if (!zoomCmd) return;
    const v = st.current.view;
    if (zoomCmd.kind === 'in') v.zoom = clampZoom(v.zoom * 1.25);
    else if (zoomCmd.kind === 'out') v.zoom = clampZoom(v.zoom / 1.25);
    else if (zoomCmd.kind === 'reset') Object.assign(v, { zoom: 1, panX: 0, panY: 0, yaw: 0.6, pitch: -0.3 });
    else if (zoomCmd.kind === 'reheat' && st.current.sim) reheat(st.current.sim, 0.9);
    st.current.dirty = true;
  }, [zoomCmd]);

  // 大きさ（画面の幅に合わせる。高解像度の画面は devicePixelRatio 倍で描く）
  useEffect(() => {
    const resize = () => {
      const el = wrapRef.current;
      const cv = canvasRef.current;
      if (!el || !cv) return;
      const w = el.clientWidth;
      const h = el.clientHeight;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      cv.style.width = `${w}px`;
      cv.style.height = `${h}px`;
      st.current.size = { w, h, dpr };
      st.current.bg = null;
      st.current.dirty = true;
    };
    resize();
    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(resize) : null;
    if (ro) ro.observe(wrapRef.current);
    window.addEventListener('resize', resize);
    return () => { if (ro) ro.disconnect(); window.removeEventListener('resize', resize); };
  }, []);

  // 描き続ける（計算・描画）
  useEffect(() => {
    let alive = true;
    const loop = () => {
      if (!alive) return;
      const s = st.current;
      const p = props.current;
      if (s.sim && p.dec) {
        let ran = 0;
        if (s.sim.alpha >= s.sim.opts.alphaMin) {
          const t0 = performance.now();
          while (ran < 8 && performance.now() - t0 < 10) {
            if (!tick(s.sim)) break;
            ran += 1;
          }
        }
        if (ran) { s.dirty = true; s.fit = true; }
        // 軌跡の明るみが流れている間は、ほかに動きが無くても約20コマ/秒で描き直す
        if (p.trail > 0 && p.flow && performance.now() - s.flowAt >= FLOW_FRAME_MS) s.dirty = true;
        if (s.dirty) { draw(); s.dirty = false; s.flowAt = performance.now(); }
      }
      s.raf = requestAnimationFrame(loop);
    };
    st.current.raf = requestAnimationFrame(loop);
    return () => { alive = false; cancelAnimationFrame(st.current.raf); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  function currentExtent() {
    const s = st.current;
    const n = props.current.dec.n;
    if (!s.ext || s.fit) {
      const e = extent(s.sim.pos, n, 0.985);
      // 動いている間は大きさをなめらかに追う（ふくらむたびに画面が跳ねないように）
      s.ext = s.ext ? {
        cx: s.ext.cx + (e.cx - s.ext.cx) * 0.2, cy: s.ext.cy + (e.cy - s.ext.cy) * 0.2,
        cz: s.ext.cz + (e.cz - s.ext.cz) * 0.2, radius: s.ext.radius + (e.radius - s.ext.radius) * 0.2,
      } : e;
      s.fit = false;
    }
    return s.ext;
  }

  /** 強調する日（触れた日 → 選んだ日 → 今日）。今日の強調を消していて、触れても選んでもいなければ −1 */
  function focusOf() {
    const s = st.current;
    const p = props.current;
    if (s.hover >= 0) return s.hover;
    if (Number.isInteger(p.selected)) return p.selected;
    return p.highlightToday ? p.dec.n - 1 : -1;
  }

  function neighborSet(focus) {
    const s = st.current;
    const p = props.current;
    if (s.nbFor !== focus || s.nbNet !== p.nbrs) {
      s.nbSet = focus >= 0 && p.nbrs ? new Set(p.nbrs.of(focus)) : null;
      s.nbFor = focus;
      s.nbNet = p.nbrs;
    }
    return s.nbSet;
  }

  function background(ctx, w, h) {
    const s = st.current;
    if (!s.bg) {
      const g = ctx.createRadialGradient(w * 0.5, h * 0.45, 0, w * 0.5, h * 0.5, Math.hypot(w, h) * 0.62);
      g.addColorStop(0, BG_IN);
      g.addColorStop(1, BG_OUT);
      s.bg = g;
    }
    return s.bg;
  }

  function draw() {
    const s = st.current;
    const p = props.current;
    const cv = canvasRef.current;
    const { w, h, dpr } = s.size;
    if (!cv || !w || !h || !p.dec || !p.net) return;
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = background(ctx, w, h);
    ctx.fillRect(0, 0, w, h);
    const { n, isFresh } = p.dec;
    const { deg, edges } = p.net;
    const ext = currentExtent();
    const sc = project(s.sim.pos, n, ext, s.view, w, h, s.screen);
    const today = n - 1;
    const focus = focusOf();
    const nbSet = neighborSet(focus);
    const lit = (i) => i === focus || (nbSet && nbSet.has(i));
    const depth = (i) => Math.min(1, Math.max(0, (sc[4 * i + 3] + 1) / 2));

    // 線（薄い明るい線を重ねる。密な塊ほど明るい。奥と手前で濃さを変える。強調中も同じ濃さ）
    ctx.globalCompositeOperation = 'source-over';
    ctx.lineWidth = 0.7;
    for (let pass = 0; pass < 2; pass += 1) {
      ctx.beginPath();
      for (let e = 0; e < edges.length; e += 2) {
        const a = edges[e];
        const b = edges[e + 1];
        if ((sc[4 * a + 3] + sc[4 * b + 3] >= 0) !== (pass === 1)) continue;
        ctx.moveTo(sc[4 * a], sc[4 * a + 1]);
        ctx.lineTo(sc[4 * b], sc[4 * b + 1]);
      }
      const front = pass === 1;
      const al = EDGE_ALPHA[front ? 'front' : 'back'];
      ctx.strokeStyle = `rgba(${front ? EDGE_RGB_FRONT : EDGE_RGB_BACK}, ${al.toFixed(4)})`;
      ctx.stroke();
    }
    // 強調する日から似た日への線（細い控えめな線。光らせない）
    if (focus >= 0 && nbSet && nbSet.size) {
      const fx = sc[4 * focus];
      const fy = sc[4 * focus + 1];
      ctx.beginPath();
      for (const j of nbSet) {
        ctx.moveTo(fx, fy);
        ctx.lineTo(sc[4 * j], sc[4 * j + 1]);
      }
      ctx.strokeStyle = LINK;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    // 直近の軌跡（日付の順。今日へどう入ってきたか）: 暗めの橙の実線と、今日の向きへゆっくり流れる明るみ（動きを減らす設定では
    // 矢印）。点の下に描く
    const span = p.trail > 0 && n > 1 ? { from: Math.max(0, n - 1 - p.trail), L: n - 1 - Math.max(0, n - 1 - p.trail) } : null;
    let flowTag = '';
    if (span) {
      trailBase(ctx, sc, span);
      if (p.flow) flowTag = trailFlow(ctx, sc, span, performance.now()).toFixed(2);
      else { trailArrows(ctx, sc, span); flowTag = 'arrows'; }
    }

    // 点（奥から手前へ。奥ほど小さく暗い。手前はうっすら光る）
    const order = s.order;
    order.sort((a, b) => sc[4 * a + 3] - sc[4 * b + 3]);
    // 点の大きさ（次数と遠近）。狭い画面（スマホ）では点を小さくする（約2,200点が重なって塗りつぶさないように）。ほかの点には
    // 画面の「点の大きさ」の倍率（pointScale）を掛ける
    const sizeK = Math.max(0.55, Math.min(1, Math.min(w, h) / 640));
    const radius0 = (i) => Math.min(6, (0.9 + 0.3 * Math.sqrt(deg[i])) * sc[4 * i + 2]) * sizeK;
    const radius = (i) => radius0(i) * p.pointScale;
    for (let k = 0; k < n; k += 1) {
      const i = order[k];
      if (i === today || lit(i)) continue;              // 強調する日と今日は上に重ねて描く
      const dz = depth(i);
      const r = radius(i);
      const a = 0.22 + 0.78 * dz;                         // 強調中も暗くしない（相対的な差が分かるように）
      const x = sc[4 * i];
      const y = sc[4 * i + 1];
      ctx.fillStyle = p.colors[i];
      if (dz > 0.45) {
        ctx.globalAlpha = a * 0.14;
        ctx.beginPath();
        ctx.arc(x, y, r * 2.6, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.globalAlpha = a;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fill();
      if (isFresh[i]) {
        ctx.strokeStyle = RING;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(x, y, r + 2.2, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
    // 強調する日の似た日（その日の色のまま少し大きく描き、地の色の縁と細い控えめな輪で囲む。光らせない。大きさはバーによらない）
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    if (nbSet) {
      for (const j of nbSet) {
        if (j === today) continue;
        const x = sc[4 * j];
        const y = sc[4 * j + 1];
        const r = Math.max(3.2, radius0(j) * 1.5) * MARK_SCALE;
        ctx.fillStyle = p.colors[j];
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fill();
        ring(ctx, x, y, r + 0.9, BG_OUT, 1.8);
        ring(ctx, x, y, r + 2.4, LINK, 1);
        if (isFresh[j]) ring(ctx, x, y, r + 4.2, RING, 1.2);
      }
    }
    // 今日と触れている・選んだ日の点も「点の大きさ」によらずいつも同じ大きさ（MARK_SCALE。運用者の指定 2026-10-23）
    const big = MARK_SCALE;
    // 触れている・選んだ日
    const fp = focus >= 0 && focus !== today ? [sc[4 * focus], sc[4 * focus + 1]] : null;
    if (fp) {
      const [x, y] = fp;
      glow(ctx, x, y, 18 * big, 'rgba(255, 255, 255, 0.22)');
      ctx.fillStyle = p.colors[focus];
      ctx.beginPath();
      ctx.arc(x, y, 5 * big, 0, 2 * Math.PI);
      ctx.fill();
      ring(ctx, x, y, 7 * big, '#ffffff', 1.6);
      label(ctx, x + 6 + 5 * big, y - 4 - 5 * big, p.dec.dates[focus]);
    }
    // 今日
    const tx = sc[4 * today];
    const ty = sc[4 * today + 1];
    glow(ctx, tx, ty, 26 * big, 'rgba(255, 255, 255, 0.28)');
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(tx, ty, 6.5 * big, 0, 2 * Math.PI);
    ctx.fill();
    ring(ctx, tx, ty, 6.5 * big, BG_OUT, 2);
    ring(ctx, tx, ty, 10.5 * big, '#ffffff', 1.2);
    label(ctx, tx + 3.5 + 10.5 * big, ty + 4, `今日 ${p.dec.date.slice(5).replace('-', '/')}`);
    // 今日の点の画面上の位置（ブラウザのテストが触れて確かめる。1px 以上動いたときだけ書く）。強調している日と似た日の数も
    const pos = `${Math.round(tx)},${Math.round(ty)}`;
    const wrap = wrapRef.current;
    if (wrap) {
      if (wrap.dataset.today !== pos) wrap.dataset.today = pos;
      // 触れている・選んだ日（今日以外）の点の位置（ブラウザのテストが、その点の大きさがバーで変わらないかを画素で確かめる）
      const fpos = fp ? `${Math.round(fp[0])},${Math.round(fp[1])}` : '';
      if (wrap.dataset.focuspos !== fpos) wrap.dataset.focuspos = fpos;
      const lf = focus >= 0 ? `${focus}:${nbSet ? nbSet.size : 0}` : '';
      if (wrap.dataset.focus !== lf) wrap.dataset.focus = lf;
      const zm = s.view.zoom.toFixed(3);
      if (wrap.dataset.zoom !== zm) wrap.dataset.zoom = zm;
      if (wrap.dataset.flow !== flowTag) wrap.dataset.flow = flowTag;     // 明るみの進んだ位置（px）か arrows。軌跡なしは空
      // 似た日の画面上の位置と色（ブラウザのテストが、似た日の点がその日の色のままで白く光っていないかを確かめる）
      const nbp = nbSet ? [...nbSet].filter((j) => j !== today)
        .map((j) => `${Math.round(sc[4 * j])},${Math.round(sc[4 * j + 1])},${p.colors[j]}`).join(';') : '';
      if (wrap.dataset.nbpos !== nbp) wrap.dataset.nbpos = nbp;
      if (s.sim.alpha < s.sim.opts.alphaMin) wrap.dataset.settled = '1';
      else delete wrap.dataset.settled;
    }
  }

  /** 軌跡の実線（暗めの橙。ほかの線より少し太い。古い日 → 今日で薄い → 濃い） */
  function trailBase(ctx, sc, span) {
    ctx.lineCap = 'round';
    ctx.lineWidth = 1.8;
    for (let a = 0; a < span.L; a += 1) {
      const i = span.from + a;
      const t = (a + 1) / span.L;
      ctx.beginPath();
      ctx.moveTo(sc[4 * i], sc[4 * i + 1]);
      ctx.lineTo(sc[4 * (i + 1)], sc[4 * (i + 1) + 1]);
      ctx.strokeStyle = `rgba(${TRAIL_RGB}, ${(0.3 + 0.3 * t).toFixed(3)})`;
      ctx.stroke();
    }
    ctx.lineCap = 'butt';
  }

  /**
   * 軌跡の実線の上を今日の向きへゆっくり流れる明るみ（同じ暗い橙。画面の上で FLOW_PX_PER_S の一定の速さで、FLOW_GAP_PX おきに
   * 長さ FLOW_TAIL_PX。前が濃く後ろほど薄い）。古い端で生まれ、今日の点へ薄れて消える。戻り値は明るみの進んだ位置（0〜FLOW_GAP_PX px。
   * ブラウザのテストが今日の向きへ動くことを確かめる）
   */
  function trailFlow(ctx, sc, span, now) {
    // 軌跡の上の距離（px）。日の間の長さを足していく
    const cum = new Float64Array(span.L + 1);
    for (let a = 0; a < span.L; a += 1) {
      const i = span.from + a;
      cum[a + 1] = cum[a] + Math.hypot(sc[4 * (i + 1)] - sc[4 * i], sc[4 * (i + 1) + 1] - sc[4 * i + 1]);
    }
    const total = cum[span.L];
    const at = (d) => {
      let a = 0;
      while (a < span.L - 1 && cum[a + 1] < d) a += 1;
      const len = cum[a + 1] - cum[a];
      const f = len > 0 ? Math.max(0, Math.min(1, (d - cum[a]) / len)) : 0;
      const i = span.from + a;
      return [sc[4 * i] + (sc[4 * (i + 1)] - sc[4 * i]) * f, sc[4 * i + 1] + (sc[4 * (i + 1) + 1] - sc[4 * i + 1]) * f, a];
    };
    const shift = ((now / 1000) * FLOW_PX_PER_S) % FLOW_GAP_PX;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 2.2;
    for (let head = shift; head - FLOW_TAIL_PX < total; head += FLOW_GAP_PX) {
      const d1 = Math.min(head, total);
      const d0 = Math.max(0, head - FLOW_TAIL_PX);
      if (d1 - d0 < 1) continue;
      const [x0, y0, a0] = at(d0);
      const [x1, y1, a1] = at(d1);
      const g = ctx.createLinearGradient(x0, y0, x1, y1);
      g.addColorStop(0, `rgba(${TRAIL_RGB}, 0)`);
      // 今日を越えた明るみは前が切れるので、残った後ろの濃さで終える（今日の点へ吸い込まれるように薄れる）
      g.addColorStop(1, `rgba(${TRAIL_RGB}, ${(0.85 * (1 - (head - d1) / FLOW_TAIL_PX)).toFixed(3)})`);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      for (let a = a0 + 1; a <= a1; a += 1) ctx.lineTo(sc[4 * (span.from + a)], sc[4 * (span.from + a) + 1]);
      ctx.lineTo(x1, y1);
      ctx.strokeStyle = g;
      ctx.stroke();
    }
    ctx.lineCap = 'butt';
    return shift;
  }

  /** 動きを減らす設定のときの向きの矢印（同じ暗い橙。今日に近い線から ARROW_EVERY 日おき。短い線には描かない） */
  function trailArrows(ctx, sc, span) {
    ctx.strokeStyle = `rgba(${TRAIL_RGB}, 0.85)`;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = 'round';
    for (let a = span.L - 1; a >= 0; a -= ARROW_EVERY) {
      const i = span.from + a;
      const x0 = sc[4 * i];
      const y0 = sc[4 * i + 1];
      const dx = sc[4 * (i + 1)] - x0;
      const dy = sc[4 * (i + 1) + 1] - y0;
      const len = Math.hypot(dx, dy);
      if (len < 10) continue;
      const ux = dx / len;
      const uy = dy / len;
      const mx = x0 + dx * 0.55;
      const my = y0 + dy * 0.55;
      ctx.beginPath();
      ctx.moveTo(mx - ux * 4 - uy * 3, my - uy * 4 + ux * 3);
      ctx.lineTo(mx, my);
      ctx.lineTo(mx - ux * 4 + uy * 3, my - uy * 4 - ux * 3);
      ctx.stroke();
    }
  }

  function glow(ctx, x, y, r, color) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, color);
    g.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
  }

  function ring(ctx, x, y, r, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.stroke();
  }

  function label(ctx, x, y, text) {
    ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Noto Sans JP", sans-serif';
    ctx.lineWidth = 3;
    ctx.strokeStyle = BG_OUT;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = TEXT;
    ctx.fillText(text, x, y);
  }

  // --- 操作 ---
  const local = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  /** (x, y) を中心に拡大縮小する（その点の下にある中身が、指・カーソルの下に残るように表示をずらす）。 */
  const zoomAt = (x, y, zoom, from = null) => {
    const s = st.current;
    const v = s.view;
    const base = from || { zoom: v.zoom, panX: v.panX, panY: v.panY, x, y };
    const z1 = clampZoom(zoom);
    const cx = s.size.w / 2;
    const cy = s.size.h / 2;
    const f = z1 / base.zoom;
    v.zoom = z1;
    v.panX = x - cx - (base.x - cx - base.panX) * f;
    v.panY = y - cy - (base.y - cy - base.panY) * f;
    s.dirty = true;
  };

  const hoverAt = (x, y) => {
    const s = st.current;
    const p = props.current;
    if (!s.sim || !s.screen || !p.dec) return;
    const i = pick(s.screen, p.dec.n, x, y, 12 * Math.max(1, p.pointScale));   // 点を大きくしたら触れる範囲も広げる
    if (i !== s.hover) {
      s.hover = i;
      s.dirty = true;
    }
    if (i >= 0 && tipOf) {
      const t = tipOf(i);
      setTip(t ? { ...t, x, y } : null);
    } else setTip(null);
  };

  const startDrag = (x, y, pan, moved = false) => {
    const s = st.current;
    s.drag = { x, y, yaw: s.view.yaw, pitch: s.view.pitch, panX: s.view.panX, panY: s.view.panY, moved, pan };
  };

  const pinchFrom = () => {
    const [a, b] = [...st.current.ptrs.values()];
    const v = st.current.view;
    return {
      d0: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)), zoom: v.zoom, panX: v.panX, panY: v.panY,
      x: (a.x + b.x) / 2, y: (a.y + b.y) / 2,
    };
  };

  const onPointerDown = (e) => {
    const s = st.current;
    const pt = local(e);
    // 新しく触れ始めた（最初の指・マウス）ときは、前の指の記録を捨てる（離した知らせを取りこぼしても、1本指が2本指にならない）
    if (e.isPrimary) { s.ptrs.clear(); s.pinch = null; }
    s.ptrs.set(e.pointerId, pt);
    canvasRef.current.setPointerCapture?.(e.pointerId);
    if (s.ptrs.size >= 2) {
      // 2本指: つまんで拡大縮小・動かしてずらす（回すのはやめる）
      s.pinch = pinchFrom();
      s.drag = null;
      s.hover = -1;
      setTip(null);
      s.dirty = true;
      return;
    }
    startDrag(pt.x, pt.y, e.shiftKey || e.button === 2);
    s.drag.touch = e.pointerType === 'touch';
  };

  const onPointerMove = (e) => {
    const s = st.current;
    const pt = local(e);
    if (s.ptrs.has(e.pointerId)) s.ptrs.set(e.pointerId, pt);
    if (s.pinch && s.ptrs.size >= 2) {
      const [a, b] = [...s.ptrs.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, s.pinch.zoom * (d / s.pinch.d0), s.pinch);
      return;
    }
    if (s.drag) {
      const dx = pt.x - s.drag.x;
      const dy = pt.y - s.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > (s.drag.touch ? 6 : 3)) s.drag.moved = true;
      if (s.drag.moved) {
        if (s.drag.pan) {
          s.view.panX = s.drag.panX + dx;
          s.view.panY = s.drag.panY + dy;
        } else {
          s.view.yaw = s.drag.yaw + dx * 0.008;
          s.view.pitch = Math.max(-1.45, Math.min(1.45, s.drag.pitch + dy * 0.008));
        }
        s.dirty = true;
        setTip(null);
        return;
      }
    }
    if (e.pointerType !== 'touch') hoverAt(pt.x, pt.y);
  };

  const onPointerUp = (e) => {
    const s = st.current;
    s.ptrs.delete(e.pointerId);
    if (s.pinch) {
      // 2本指から1本に戻ったら、残った指で続けて回せるように（選ぶのはしない）
      if (s.ptrs.size < 2) {
        s.pinch = null;
        if (s.ptrs.size === 1) {
          const [rest] = [...s.ptrs.values()];
          startDrag(rest.x, rest.y, false, true);
        } else s.drag = null;
      }
      return;
    }
    const d = s.drag;
    s.drag = null;
    if (d && !d.moved && onSelect && e.type === 'pointerup') {
      const { x, y } = local(e);
      const i = s.screen ? pick(s.screen, props.current.dec.n, x, y, (d.touch ? 18 : 12) * Math.max(1, props.current.pointScale)) : -1;
      onSelect(i >= 0 && i !== props.current.selected ? i : null);
      // 指で触れたときは、その日の値も出す（マウスのように触れ続けられないので）
      if (d.touch) {
        if (i >= 0 && tipOf) {
          const t = tipOf(i);
          setTip(t ? { ...t, x, y } : null);
        } else setTip(null);
      }
    }
  };

  const onLeave = (e) => {
    const s = st.current;
    if (e.pointerType === 'touch') return;
    s.hover = -1;
    s.dirty = true;
    setTip(null);
  };

  // ホイールの拡大は passive ではない受け方で（ページのスクロールを止める）。カーソルの位置を中心に
  useEffect(() => {
    const cv = canvasRef.current;
    const onWheel = (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      const v = st.current.view;
      zoomAt(e.clientX - r.left, e.clientY - r.top, v.zoom * Math.exp(-e.deltaY * 0.0012));
    };
    cv.addEventListener('wheel', onWheel, { passive: false });
    return () => cv.removeEventListener('wheel', onWheel);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const onKeyDown = (e) => {
    const v = st.current.view;
    const step = 0.12;
    if (e.key === 'ArrowLeft') v.yaw -= step;
    else if (e.key === 'ArrowRight') v.yaw += step;
    else if (e.key === 'ArrowUp') v.pitch = Math.max(-1.45, v.pitch - step);
    else if (e.key === 'ArrowDown') v.pitch = Math.min(1.45, v.pitch + step);
    else if (e.key === '+' || e.key === '=') v.zoom = clampZoom(v.zoom * 1.25);
    else if (e.key === '-') v.zoom = clampZoom(v.zoom / 1.25);
    else if (e.key === '0') Object.assign(v, { zoom: 1, panX: 0, panY: 0 });
    else return;
    e.preventDefault();
    st.current.dirty = true;
  };

  const onDouble = () => {
    Object.assign(st.current.view, { zoom: 1, panX: 0, panY: 0 });
    st.current.dirty = true;
  };

  return (
    <div className="net-canvas" ref={wrapRef}>
      <canvas
        ref={canvasRef} tabIndex={0} role="img" aria-label={ariaLabel}
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp} onPointerLeave={onLeave} onDoubleClick={onDouble} onKeyDown={onKeyDown}
        onContextMenu={(e) => e.preventDefault()}
      />
      {tip && (
        <div className="net-tip" role="status"
             style={{ left: Math.max(4, Math.min(tip.x + 14, (st.current.size.w || 0) - 240)), top: Math.min(tip.y + 14, Math.max(4, (st.current.size.h || 0) - 230)) }}>
          <div className="t">{tip.title}</div>
          {tip.lines.map(([k, v]) => (
            <div key={k}><span>{k}</span><b className="num">{v}</b></div>
          ))}
        </div>
      )}
    </div>
  );
}
