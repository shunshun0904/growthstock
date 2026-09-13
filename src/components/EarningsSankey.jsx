import React, { useMemo, useState } from 'react';
import { MODES, toFlow, latestQuarter, fmtYen, share } from '../lib/earnings.js';
import { fmtDate, DASH } from '../lib/format.js';

/**
 * 直近決算の損益をサンキー図で描く。
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

export default function EarningsSankey({ quarters }) {
  const [mode, setMode] = useState('q');
  const q = useMemo(() => latestQuarter(quarters), [quarters]);
  const [hover, setHover] = useState(null);

  // その決算にどのモードの値が入っているかは銘柄ごとに違う。
  // 選べないモードはボタンごと落とす（押せるのに何も出ない、を避ける）
  const usable = useMemo(
    () => MODES.filter((m) => toFlow(q, m.id)), [q]);
  const active = usable.some((m) => m.id === mode) ? mode : usable[0]?.id;
  const flow = useMemo(() => toFlow(q, active), [q, active]);

  if (!q) {
    return (
      <div className="empty">
        この銘柄には決算データがありません。
        <div className="sub" style={{ marginTop: 6 }}>
          ETF・REIT など、決算短信を出さない銘柄では空になります。
        </div>
      </div>
    );
  }
  if (!flow) {
    return <div className="empty">直近決算に損益の数字が入っていません。</div>;
  }

  const { sales, steps, truncated, missing } = flow;
  const scale = PLOT_H / sales;
  const cols = steps.length + 1;
  const dx = cols > 1 ? (X1 - X0) / (cols - 1) : 0;
  const x = (i) => X0 + dx * i;

  // 幹は上端に揃える。下に向かって費用が剥がれていく形にする
  const heights = [PLOT_H, ...steps.map((s) => Math.max(0, s.value) * scale)];
  const nodes = [];
  const ribbons = [];

  // 列ごとの「次に置ける y」。1つの列に複数ぶら下がる（費用と流入が同じ列に
  // 来ることがある）ので、積み上げて重ならないようにする。
  // ラベルぶんの高さも空ける
  const free = heights.map((h) => h + GAP);
  const place = (col, h) => {
    const y = free[col];
    free[col] = y + h + LABEL_H;
    return y;
  };

  nodes.push({ kind: 'trunk', col: 0, y: 0, h: PLOT_H, color: C.profit,
               label: '売上高', value: sales, full: '売上高' });

  steps.forEach((s, i) => {
    const hPrev = heights[i];
    const hNext = s.loss ? 0 : heights[i + 1];
    const col = i + 1;

    // 幹（残った利益）。赤字の段は幅ゼロになるので帯は描かない
    if (hNext > 0) {
      ribbons.push({
        d: band(x(i) + BAR, 0, hNext, x(col), 0, hNext),
        color: C.profit, key: `t${i}`,
        title: `${s.label} ${fmtYen(s.value)}`,
      });
    }
    nodes.push({
      kind: 'trunk', col, y: 0, h: hNext,
      color: s.loss ? C.loss : C.profit, hatch: s.loss,
      label: s.label, value: s.value, full: s.label,
    });

    // 出ていくぶん（費用・税）。幹の下に同じ列でぶら下げる
    if (s.out && s.out.value > 0) {
      // 赤字の段では、出ていった額が幹に入ってきた額を上回る。
      // 幹から供給できるのは hPrev までなので、超過分（＝赤字）は
      // ノードの下側に赤の斜線で足す。2つを足した高さがラベルの金額と一致する
      const short = s.loss ? Math.abs(s.value) * scale : 0;
      const oh = s.out.value * scale;
      const oy = place(col, oh + short);
      ribbons.push({
        d: band(x(i) + BAR, hNext, hPrev, x(col), oy, oy + oh),
        color: C.cost, key: `o${i}`,
        title: `${s.out.label} ${fmtYen(s.out.value + Math.abs(s.loss ? s.value : 0))}`,
      });
      nodes.push({
        kind: 'cost', col, y: oy, h: oh, short,
        color: C.cost,
        label: shortLabel(s.out.label),
        value: s.out.value + (s.loss ? Math.abs(s.value) : 0),
        full: s.out.label,
      });
    }
    // 入ってくるぶん（営業外収益・特別利益）。下から幹へ合流する
    if (s.in && s.in.value > 0) {
      const ih = s.in.value * scale;
      const iy = place(i, ih);
      ribbons.push({
        d: band(x(i) + BAR, iy, iy + ih, x(col), hPrev, hPrev + ih),
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

  return (
    <div className="sankey">
      <div className="sankey-modes">
        {usable.map((m) => (
          <button key={m.id} title={m.note}
                  className={`btn btn-ghost${m.id === active ? ' on' : ''}`}
                  onClick={() => setMode(m.id)}>
            {m.label}
          </button>
        ))}
        <span className="sub">
          {q.period || DASH}
          <span className="sep">/</span>
          {fmtDate(q.disclosedDate)} 開示
          {q.periodEnd && <> <span className="sep">/</span> {fmtDate(q.periodEnd)} 締め</>}
        </span>
      </div>

      <div className="sankey-plot">
        <svg viewBox={`0 0 ${VW} ${H}`} width="100%" height={H}
             role="img" aria-label="直近決算の損益の流れ">
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
              <Node key={i} n={n} x={x(n.col)} sales={sales} cols={cols} />
            ))}
          </g>
        </svg>
      </div>

      <Legend flow={flow} truncated={truncated} missing={missing} />
    </div>
  );
}

/** 長い名前は図の中では短くする。正式名は title 属性で読める。 */
function shortLabel(s) {
  return s.replace('（原価＋販管費）', '').replace('法人税等・特別損失', '税金・特別損失');
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

function Legend({ flow, truncated, missing }) {
  const np = flow.steps.find((s) => s.key === 'np');
  return (
    <>
      <div className="sankey-legend">
        <span><i style={{ background: 'var(--green)' }} />残った利益</span>
        <span><i style={{ background: 'var(--text-faint)' }} />出ていった費用・税</span>
        {flow.steps.some((s) => s.in) && (
          <span><i style={{ background: 'var(--blue)' }} />入ってきた収益</span>
        )}
        {truncated && <span><i className="hatch" />赤字（△表記）</span>}
      </div>
      <p className="sub sankey-note">
        帯の太さは売上高に対する割合です。<b>売上原価と販管費は決算短信の
        サマリーに無い</b>ため、その2つは「営業費用」として1本にまとめてあります
        （売上高 − 営業利益。引き算で出した値で、内訳は元データにありません）。
        {missing.includes('odp') && (
          <> この銘柄は経常利益の開示が無いため、営業利益から当期純利益まで
          1段でまとめています。</>
        )}
        {truncated && (
          <> <b>{flow.steps.find((s) => s.key === truncated)?.label}が赤字</b>の
          ため、そこで流れを止めています（サンキー図は負の流れを描けません）。</>
        )}
        {np && !truncated && (
          <> 売上高100円あたり <b className="num">{share(np.value, flow.sales)?.toFixed(1)}円</b> が
          最終的に残りました。</>
        )}
      </p>
    </>
  );
}
