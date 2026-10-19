import React, { useEffect, useRef, useState } from 'react';
import { createSim, tick, setDim, setForces, reheat, extent, project, pick, seedOf } from '../lib/forcegraph.js';

/**
 * リカレンス・ネットワークの描画（canvas 2D。Obsidian のグラフ表示のように、力学レイアウトで球状に並べて回す）。
 *
 * - 点 = 1営業日（大きさは次数）、線 = 状態の似た日どうし。今日（候補の日）は白い大きな点、78週高値の更新日は紫の輪
 * - ドラッグで回す（2次元はずらす）。ホイール・＋−で拡大。ダブルクリックで元に戻す。点に触れるとその日の近傍を光らせ、
 *   クリックで選ぶ（もう一度クリック・何も無いところで外す）。矢印キーで回し、＋ − で拡大、0 で元に戻す
 * - 計算は毎フレーム 10ms までに区切り、動き終わったら描き直しが要るときだけ描く
 */

const BG = '#0a0e17';
const EDGE_BACK = 'rgba(148, 163, 184, 0.05)';
const EDGE_FRONT = 'rgba(148, 163, 184, 0.12)';
const EDGE_FOCUS = 'rgba(167, 139, 250, 0.85)';
const TRAIL = 'rgba(251, 191, 36, 0.8)';
const RING = '#a78bfa';
const TEXT = '#e8edf6';

export default function RecurrenceGraph({
  dec, net, nbrs, colors, dim, spin, trail, seedKey, selected, onSelect, tipOf, ariaLabel, zoomCmd, forces,
}) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const st = useRef({
    sim: null, view: { yaw: 0.6, pitch: -0.3, zoom: 1, panX: 0, panY: 0 }, screen: null, order: null,
    hover: -1, drag: null, dirty: true, ext: null, size: { w: 0, h: 0, dpr: 1 }, raf: 0, nbSet: null, nbFor: -2,
  });
  const props = useRef({});
  props.current = { dec, net, nbrs, colors, dim, spin, trail, selected };
  const [tip, setTip] = useState(null);

  // ネットワークが変わったら並べ直す（同じ候補・同じ密度ならいつも同じ形。種は銘柄・日付・密度）
  useEffect(() => {
    const s = st.current;
    if (!dec || !net) { s.sim = null; return; }
    s.sim = createSim(dec.n, net.edges, net.deg, { dim, seed: seedOf(seedKey || ''), ...(forces || {}) });
    s.screen = new Float64Array(4 * dec.n);
    s.order = new Int32Array(dec.n).map((_, i) => i);
    s.hover = -1;
    s.ext = null;
    s.nbFor = -2;
    s.dirty = true;
    setTip(null);
  }, [dec, net]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const s = st.current;
    if (s.sim) { setDim(s.sim, dim); s.ext = null; s.dirty = true; }
  }, [dim]);

  useEffect(() => { st.current.dirty = true; }, [colors, trail, selected]);

  // 力のつまみ（Obsidian の「中心の力・反発の力・リンクの力・リンクの長さ」）。変えた分だけ動かし直す
  useEffect(() => {
    const s = st.current;
    if (s.sim && forces && setForces(s.sim, forces)) { s.fit = true; s.dirty = true; }
  }, [forces]);

  // 外からの拡大・元に戻す（＋ − ボタン）
  useEffect(() => {
    if (!zoomCmd) return;
    const v = st.current.view;
    if (zoomCmd.kind === 'in') v.zoom = Math.min(8, v.zoom * 1.25);
    else if (zoomCmd.kind === 'out') v.zoom = Math.max(0.35, v.zoom / 1.25);
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
      st.current.dirty = true;
    };
    resize();
    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(resize) : null;
    if (ro) ro.observe(wrapRef.current);
    window.addEventListener('resize', resize);
    return () => { if (ro) ro.disconnect(); window.removeEventListener('resize', resize); };
  }, []);

  // 描き続ける（計算・回転・描画）
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
        const idle = !s.drag && s.hover < 0;
        if (p.spin && idle) { s.view.yaw += p.dim === 3 ? 0.0022 : 0.0010; s.dirty = true; }
        if (ran) { s.dirty = true; s.fit = true; }
        if (s.dirty) { draw(); s.dirty = false; }
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

  function neighborSet(focus) {
    const s = st.current;
    if (s.nbFor !== focus) {
      s.nbSet = focus >= 0 ? new Set(props.current.nbrs.of(focus)) : null;
      s.nbFor = focus;
    }
    return s.nbSet;
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
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    const { n, isFresh } = p.dec;
    const { deg, edges } = p.net;
    const ext = currentExtent();
    const sc = project(s.sim.pos, n, ext, s.view, w, h, s.sim.dim, s.screen);
    const today = n - 1;
    const focus = s.hover >= 0 ? s.hover : (Number.isInteger(p.selected) ? p.selected : -1);
    const nbSet = neighborSet(focus);

    // 線（奥と手前で濃さを変える）
    ctx.lineWidth = 0.6;
    for (let pass = 0; pass < 2; pass += 1) {
      ctx.beginPath();
      for (let e = 0; e < edges.length; e += 2) {
        const a = edges[e];
        const b = edges[e + 1];
        const front = sc[4 * a + 3] + sc[4 * b + 3] >= 0;
        if (front !== (pass === 1)) continue;
        ctx.moveTo(sc[4 * a], sc[4 * a + 1]);
        ctx.lineTo(sc[4 * b], sc[4 * b + 1]);
      }
      ctx.strokeStyle = pass === 1 ? EDGE_FRONT : EDGE_BACK;
      ctx.stroke();
    }
    if (focus >= 0 && nbSet) {
      ctx.beginPath();
      for (const j of nbSet) {
        ctx.moveTo(sc[4 * focus], sc[4 * focus + 1]);
        ctx.lineTo(sc[4 * j], sc[4 * j + 1]);
      }
      ctx.strokeStyle = EDGE_FOCUS;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    // 直近の軌跡（日付の順につないだ線。今日へどう入ってきたか）
    if (p.trail > 0 && n > 1) {
      ctx.beginPath();
      const from = Math.max(0, n - 1 - p.trail);
      for (let i = from; i < n; i += 1) {
        if (i === from) ctx.moveTo(sc[4 * i], sc[4 * i + 1]);
        else ctx.lineTo(sc[4 * i], sc[4 * i + 1]);
      }
      ctx.strokeStyle = TRAIL;
      ctx.lineWidth = 1.4;
      ctx.lineJoin = 'round';
      ctx.stroke();
    }
    // 点（奥から手前へ）
    const order = s.order;
    if (s.sim.dim === 3) order.sort((a, b) => sc[4 * a + 3] - sc[4 * b + 3]);
    for (let k = 0; k < n; k += 1) {
      const i = order[k];
      if (i === today) continue;
      const persp = sc[4 * i + 2];
      const r = Math.min(6, (0.9 + 0.32 * Math.sqrt(deg[i])) * persp);
      let a = s.sim.dim === 3 ? 0.35 + 0.65 * Math.min(1, Math.max(0, (sc[4 * i + 3] + 1) / 2)) : 0.95;
      if (nbSet && i !== focus && !nbSet.has(i)) a *= 0.2;
      ctx.globalAlpha = a;
      ctx.fillStyle = p.colors[i];
      ctx.beginPath();
      ctx.arc(sc[4 * i], sc[4 * i + 1], r, 0, 2 * Math.PI);
      ctx.fill();
      if (isFresh[i]) {
        ctx.strokeStyle = RING;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.arc(sc[4 * i], sc[4 * i + 1], r + 2.2, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    // 選んだ・触れている日
    if (focus >= 0 && focus !== today) {
      ring(ctx, sc[4 * focus], sc[4 * focus + 1], 6.5, '#ffffff', 1.6);
      label(ctx, sc[4 * focus] + 10, sc[4 * focus + 1] - 8, p.dec.dates[focus]);
    }
    // 今日
    const tx = sc[4 * today];
    const ty = sc[4 * today + 1];
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(tx, ty, 6.5, 0, 2 * Math.PI);
    ctx.fill();
    ring(ctx, tx, ty, 6.5, BG, 2);
    ring(ctx, tx, ty, 10, '#ffffff', 1.2);
    label(ctx, tx + 13, ty + 4, `今日 ${p.dec.date.slice(5).replace('-', '/')}`);
    // 今日の点の画面上の位置（ブラウザのテストが触れて確かめる。1px 以上動いたときだけ書く）
    const pos = `${Math.round(tx)},${Math.round(ty)}`;
    if (wrapRef.current && wrapRef.current.dataset.today !== pos) wrapRef.current.dataset.today = pos;
    if (wrapRef.current && s.sim.alpha < s.sim.opts.alphaMin) wrapRef.current.dataset.settled = '1';
    else if (wrapRef.current) delete wrapRef.current.dataset.settled;
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
    ctx.strokeStyle = BG;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = TEXT;
    ctx.fillText(text, x, y);
  }

  // --- 操作 ---
  const local = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const hoverAt = (x, y) => {
    const s = st.current;
    const p = props.current;
    if (!s.sim || !s.screen || !p.dec) return;
    const i = pick(s.screen, p.dec.n, x, y, 12);
    if (i !== s.hover) {
      s.hover = i;
      s.dirty = true;
    }
    if (i >= 0 && tipOf) {
      const t = tipOf(i);
      setTip(t ? { ...t, x, y } : null);
    } else setTip(null);
  };

  const onPointerDown = (e) => {
    const s = st.current;
    const { x, y } = local(e);
    s.drag = { x, y, yaw: s.view.yaw, pitch: s.view.pitch, panX: s.view.panX, panY: s.view.panY, moved: false,
      pan: e.shiftKey || props.current.dim === 2 || e.button === 2 };
    canvasRef.current.setPointerCapture?.(e.pointerId);
  };

  const onPointerMove = (e) => {
    const s = st.current;
    const { x, y } = local(e);
    if (s.drag) {
      const dx = x - s.drag.x;
      const dy = y - s.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) s.drag.moved = true;
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
    hoverAt(x, y);
  };

  const onPointerUp = (e) => {
    const s = st.current;
    const d = s.drag;
    s.drag = null;
    if (d && !d.moved && onSelect) {
      const { x, y } = local(e);
      const i = s.screen ? pick(s.screen, props.current.dec.n, x, y, 12) : -1;
      onSelect(i >= 0 && i !== props.current.selected ? i : null);
    }
  };

  const onLeave = () => {
    const s = st.current;
    s.hover = -1;
    s.dirty = true;
    setTip(null);
  };

  // ホイールの拡大は passive ではない受け方で（ページのスクロールを止める）
  useEffect(() => {
    const cv = canvasRef.current;
    const onWheel = (e) => {
      e.preventDefault();
      const v = st.current.view;
      v.zoom = Math.max(0.35, Math.min(8, v.zoom * Math.exp(-e.deltaY * 0.0012)));
      st.current.dirty = true;
    };
    cv.addEventListener('wheel', onWheel, { passive: false });
    return () => cv.removeEventListener('wheel', onWheel);
  }, []);

  const onKeyDown = (e) => {
    const v = st.current.view;
    const step = 0.12;
    if (e.key === 'ArrowLeft') v.yaw -= step;
    else if (e.key === 'ArrowRight') v.yaw += step;
    else if (e.key === 'ArrowUp') v.pitch = Math.max(-1.45, v.pitch - step);
    else if (e.key === 'ArrowDown') v.pitch = Math.min(1.45, v.pitch + step);
    else if (e.key === '+' || e.key === '=') v.zoom = Math.min(8, v.zoom * 1.25);
    else if (e.key === '-') v.zoom = Math.max(0.35, v.zoom / 1.25);
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
        onPointerLeave={onLeave} onDoubleClick={onDouble} onKeyDown={onKeyDown}
        onContextMenu={(e) => e.preventDefault()}
      />
      {tip && (
        <div className="net-tip" role="status"
             style={{ left: Math.min(tip.x + 14, (st.current.size.w || 0) - 240), top: tip.y + 14 }}>
          <div className="t">{tip.title}</div>
          {tip.lines.map(([k, v]) => (
            <div key={k}><span>{k}</span><b className="num">{v}</b></div>
          ))}
        </div>
      )}
    </div>
  );
}
