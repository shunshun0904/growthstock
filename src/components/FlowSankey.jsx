import React, { useState } from 'react';
import { fmtYen, share } from '../lib/earnings.js';

/**
 * 損益の「流れ」（earnings.toFlow / filings.toDetailedFlow が作る形）をサンキー図で描く。
 * データの取り方・段の意味は呼び出し側（EarningsSankey / FilingsView）が持ち、ここは描くだけ。
 *
 * 色の決め方（目視ではなく実測で決めた）
 * ------------------------------------
 * 画面の既存トークンから選び、暗い面(#141b2b)上での分離を検証した。
 *
 *   利益 #34d399 × 費用 #5c6980   2型色覚 ΔE 26.8 / 3型 28.4 / 通常 30.3
 *   （対抗案の #94a3b8 は ΔE 9.5 しかなく、採らなかった）
 *
 * 損失の赤 #f87171 だけは利益の緑と 2型色覚で ΔE 6.5 しか離れない。
 * 赤字を赤で出すのは画面の他の場所と揃える必要があるので色は変えず、
 * **色だけに頼らない**よう金額に会計慣行の △ を付け、帯に斜線を重ねる。
 *
 * 費用をわざと彩度の低い灰にしているのは、目立たせたくないから。
 * 読ませたいのは「どれだけ残ったか」であって「どれだけ出ていったか」ではない。
 *
 * 1つの段に「出ていく額」と「入ってくる額」が両方あるとき（経常利益 = 営業利益 + 営業外収益 − 営業外費用）は、
 * 前の幹の下側 out ぶんが費用へ抜け、残り（keep）が次の幹の上側になり、入ってくる額が次の幹の下側に足される。
 * 赤字の段では幹が尽きるので、前の幹も入ってきた額も費用ノードへ流れ、足りないぶん（赤字）を斜線で足す。
 */
const C = {
  profit: 'var(--green)',
  cost: 'var(--text-faint)',
  gain: 'var(--blue)',
  loss: 'var(--red)',
};

const VW = 960;          // viewBox の幅。高さは中身から決める
const BAR = 13;          // ノードの太さ
const PAD_T = 48;        // 上の余白（幹のラベル）
const PAD_B = 56;        // 下の余白（費用のラベル）
const PLOT_H = 216;      // 売上高＝この高さ
const GAP = 18;          // 幹と、その下にぶら下がる費用ノードの間隔
const LABEL_H = 40;      // ノードの下に置くラベル2行ぶんの高さ
const X0 = 10;
const X1 = VW - 10 - BAR;

/** 2つの縦線を結ぶ帯。サンキーのリボン1本。 */
function band(x0, y0a, y0b, x1, y1a, y1b) {
  const xm = (x0 + x1) / 2;
  return `M${x0},${y0a} C${xm},${y0a} ${xm},${y1a} ${x1},${y1a}`
       + ` L${x1},${y1b} C${xm},${y1b} ${xm},${y0b} ${x0},${y0b} Z`;
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/**
 * flow から図の部品（ノードとリボン）を作る。純粋関数（テストできる）。
 * 戻り値: { nodes, ribbons, H, cols, x }。
 */
export function layout(flow, shortLabel = (s) => s) {
  const { sales, steps } = flow;
  const scale = PLOT_H / sales;
  const cols = steps.length + 1;
  const dx = cols > 1 ? (X1 - X0) / (cols - 1) : 0;
  const x = (i) => X0 + dx * i;

  // 幹は上端に揃える。下に向かって費用が剥がれていく形にする
  const heights = [PLOT_H, ...steps.map((s) => Math.max(0, s.value) * scale)];
  const nodes = [];
  const ribbons = [];

  // 列ごとの「次に置ける y」。1つの列に複数ぶら下がる（費用と流入が同じ列に
  // 来ることがある）ので、積み上げて重ならないようにする。ラベルぶんの高さも空ける
  const free = heights.map((h) => h + GAP);
  const place = (col, h) => {
    const y = free[col];
    free[col] = y + h + LABEL_H;
    return y;
  };

  nodes.push({ kind: 'trunk', col: 0, y: 0, h: PLOT_H, color: C.profit,
               label: flow.salesLabel || '売上高', value: sales, full: flow.salesLabel || '売上高' });

  steps.forEach((s, i) => {
    const hPrev = heights[i];
    const hNext = s.loss ? 0 : heights[i + 1];
    const col = i + 1;
    const ih = s.in && s.in.value > 0 ? s.in.value * scale : 0;
    const oh = s.out && s.out.value > 0 ? s.out.value * scale : 0;
    // 前の幹のうち次の幹へそのまま続く高さ。通常は hPrev − out（= hNext − in）
    const keep = s.loss ? 0 : clamp(hPrev - oh, 0, hNext);

    // 幹（残った利益）。赤字の段は幅ゼロになるので帯は描かない
    if (keep > 0) {
      ribbons.push({
        d: band(x(i) + BAR, 0, keep, x(col), 0, keep),
        color: C.profit, key: `t${i}`,
        title: `${s.label} ${fmtYen(s.value)}`,
      });
    }
    nodes.push({
      kind: 'trunk', col, y: 0, h: hNext,
      color: s.loss ? C.loss : C.profit, hatch: s.loss,
      label: shortLabel(s.label), value: s.value, full: s.label,
    });

    // 出ていくぶん（費用・税）。幹の下に同じ列でぶら下げる
    let oy = null;
    if (s.out && s.out.value > 0) {
      // 赤字の段では、出ていった額が幹に入ってきた額を上回る。幹から供給できるのは hPrev（+ 流入）までなので、
      // 超過分（＝赤字）はノードの下側に赤の斜線で足す。2つを足した高さがラベルの金額と一致する
      const short = s.loss ? Math.abs(s.value) * scale : 0;
      oy = place(col, oh + short);
      const fromA = s.loss ? 0 : keep;
      ribbons.push({
        d: band(x(i) + BAR, fromA, hPrev, x(col), oy, oy + (hPrev - fromA)),
        color: C.cost, key: `o${i}`,
        title: `${s.out.label} ${fmtYen(s.out.value)}`,
      });
      nodes.push({
        kind: 'cost', col, y: oy, h: oh, short,
        color: C.cost,
        label: shortLabel(s.out.label),
        value: s.out.value,
        full: s.out.label,
      });
    }
    // 入ってくるぶん（営業外収益・特別利益）。下から幹へ合流する。赤字の段では費用へ流れる
    if (ih > 0) {
      const iy = place(i, ih);
      const toA = s.loss ? (oy ?? 0) + hPrev : keep;
      ribbons.push({
        d: band(x(i) + BAR, iy, iy + ih, x(col), toA, toA + ih),
        color: C.gain, key: `i${i}`,
        title: `${s.in.label} ${fmtYen(s.in.value)}`,
      });
      nodes.push({
        kind: 'gain', col: i, y: iy, h: ih, color: C.gain,
        label: shortLabel(s.in.label), value: s.in.value, full: s.in.label,
      });
    }
  });

  // 一番下まで積まれた列に合わせる。ラベルぶんは PAD_B で見ているので引く
  const H = PAD_T + Math.max(PLOT_H, ...free.map((f) => f - LABEL_H)) + PAD_B;
  return { nodes, ribbons, H, cols, x };
}

export default function FlowSankey({ flow, ariaLabel = '損益の流れ', shortLabel, children }) {
  const [hover, setHover] = useState(null);
  const { nodes, ribbons, H, cols, x } = layout(flow, shortLabel);
  return (
    <>
      <div className="sankey-plot">
        <svg viewBox={`0 0 ${VW} ${H}`} width="100%" height={H}
             role="img" aria-label={ariaLabel}>
          <defs>
            {/* 損失の帯には斜線を重ねる。緑と赤は2型色覚で ΔE 6.5 しか
                離れないので、色だけでは赤字と黒字を区別できない */}
            <pattern id="sk-loss" width="7" height="7" patternUnits="userSpaceOnUse"
                     patternTransform="rotate(45)">
              <rect width="7" height="7" fill="var(--red)" opacity="0.30" />
              <line x1="0" y1="0" x2="0" y2="7" stroke="var(--red)" strokeWidth="2.5" />
            </pattern>
          </defs>
          <g transform={`translate(0,${PAD_T})`}>
            {ribbons.map((r) => (
              <path key={r.key} d={r.d} fill={r.color}
                    opacity={hover && hover !== r.key ? 0.18 : 0.38}
                    onMouseEnter={() => setHover(r.key)}
                    onMouseLeave={() => setHover(null)}>
                <title>{r.title}</title>
              </path>
            ))}
            {nodes.map((n, i) => (
              <Node key={i} n={n} x={x(n.col)} sales={flow.sales} cols={cols} />
            ))}
          </g>
        </svg>
      </div>
      {children}
    </>
  );
}

function Node({ n, x, sales, cols }) {
  const pct = share(n.value, sales);
  // 端の列はラベルが figure の外へはみ出すので寄せ方を変える
  const anchor = n.col === 0 ? 'start' : n.col === cols - 1 ? 'end' : 'middle';
  const tx = anchor === 'start' ? x : anchor === 'end' ? x + BAR : x + BAR / 2;
  // 幹のラベルは上の余白へ、下にぶら下がるものは自分の真下へ
  const ty = n.kind === 'trunk' ? -30 : n.y + n.h + (n.short || 0) + 15;

  return (
    <g>
      <rect x={x} y={n.y} width={BAR} height={Math.max(1.5, n.h)}
            rx="2" fill={n.hatch ? 'url(#sk-loss)' : n.color} />
      {n.short > 0 && (
        /* 幹から供給しきれなかったぶん＝赤字。ここまで含めてラベルの金額 */
        <rect x={x} y={n.y + n.h} width={BAR} height={Math.max(2, n.short)}
              rx="2" fill="url(#sk-loss)" />
      )}
      <text x={tx} y={ty} textAnchor={anchor} className="sk-name">
        <title>{n.full}</title>{n.label}
      </text>
      <text x={tx} y={ty + 15} textAnchor={anchor} className="sk-val">
        {fmtYen(n.value)}
        {pct != null && <tspan className="sk-pct"> {pct.toFixed(1)}%</tspan>}
      </text>
    </g>
  );
}

/** 凡例。どの flow でも同じ4つ。 */
export function SankeyLegend({ flow, truncated }) {
  return (
    <div className="sankey-legend">
      <span><i style={{ background: 'var(--green)' }} />残った利益</span>
      <span><i style={{ background: 'var(--text-faint)' }} />出ていった費用・税</span>
      {flow.steps.some((s) => s.in) && (
        <span><i style={{ background: 'var(--blue)' }} />入ってきた収益</span>
      )}
      {truncated && <span><i className="hatch" />赤字（△表記）</span>}
    </div>
  );
}
