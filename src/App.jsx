import React, { useEffect, useMemo, useState, useCallback } from 'react';
import CompareView from './views/CompareView.jsx';
import TimeMachineView from './views/TimeMachineView.jsx';
import SimulatorView from './views/SimulatorView.jsx';
import AddStockModal from './components/AddStockModal.jsx';
import { computeScores } from './lib/scoring.js';
import { SERIES_COLORS, fmtDateTime } from './lib/format.js';
import {
  loadDataset, loadManualStocks, saveManualStocks, loadVisibility, saveVisibility,
  manualStockFromForm, mergeStocks,
} from './lib/store.js';

const TABS = [
  { id: 'compare', label: '8軸オクタゴン比較' },
  { id: 'timemachine', label: 'タイムマシーン' },
  { id: 'simulator', label: 'What-If シミュレーター' },
];

export default function App() {
  const [dataset, setDataset] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [manual, setManual] = useState(() => loadManualStocks());
  const [tab, setTab] = useState('compare');
  const [visibleIds, setVisibleIds] = useState(() => new Set());
  const [selectedId, setSelectedId] = useState(null);
  const [showAdd, setShowAdd] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadDataset()
      .then((d) => { if (!cancelled) setDataset(d); })
      .catch((e) => { if (!cancelled) setLoadError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const stocks = useMemo(() => mergeStocks(dataset, manual), [dataset, manual]);

  const rows = useMemo(
    () => stocks.map((stock, i) => ({
      stock,
      result: computeScores(stock.metrics || {}),
      color: SERIES_COLORS[i % SERIES_COLORS.length],
    })),
    [stocks]
  );

  // 初回ロード時の可視状態: 保存済みがあれば復元、無ければ上位4銘柄を表示
  useEffect(() => {
    if (!rows.length || visibleIds.size > 0) return;
    const saved = loadVisibility();
    const ids = rows.map((r) => r.stock.id);
    const restored = Array.isArray(saved) ? saved.filter((id) => ids.includes(id)) : [];
    const initial = restored.length
      ? restored
      : [...rows].sort((a, b) => (b.result.totalScore ?? -1) - (a.result.totalScore ?? -1))
          .slice(0, 4).map((r) => r.stock.id);
    setVisibleIds(new Set(initial));
    setSelectedId((cur) => cur ?? initial[0] ?? ids[0]);
  }, [rows]);   // eslint-disable-line react-hooks/exhaustive-deps

  const toggleVisible = useCallback((id) => {
    setVisibleIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      saveVisibility([...next]);
      return next;
    });
  }, []);

  const addManual = useCallback((form) => {
    const stock = manualStockFromForm(form);
    setManual((prev) => {
      const next = [...prev, stock];
      saveManualStocks(next);
      return next;
    });
    setVisibleIds((prev) => {
      const next = new Set(prev).add(stock.id);
      saveVisibility([...next]);
      return next;
    });
    setSelectedId(stock.id);
    setShowAdd(false);
  }, []);

  const deleteManual = useCallback((id) => {
    setManual((prev) => {
      const next = prev.filter((s) => s.id !== id);
      saveManualStocks(next);
      return next;
    });
    setVisibleIds((prev) => {
      const next = new Set(prev);
      next.delete(id);
      saveVisibility([...next]);
      return next;
    });
    setSelectedId((cur) => (cur === id ? null : cur));
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <strong>GrowthStockAnalyzer — Focus</strong>
          <span>8軸モメンタム・スコアリング</span>
        </div>
        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id} role="tab" aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div style={{ fontSize: 11.5, color: 'var(--text-faint)', textAlign: 'right' }}>
          {dataset?.generatedAt ? (
            <>データ取得: <span className="num">{fmtDateTime(dataset.generatedAt)}</span></>
          ) : '—'}
        </div>
      </header>

      <main className="main">
        {loading && <div className="empty">データを読み込んでいます…</div>}

        {!loading && loadError && (
          <div className="banner warn" style={{ marginBottom: 'var(--s4)' }}>
            <span>⚠</span>
            <div>
              <strong>J-Quants データを読み込めませんでした。</strong>
              <div style={{ marginTop: 4 }}>{loadError}</div>
              <div style={{ marginTop: 8 }}>
                GitHub リポジトリの <code>Actions</code> タブから
                <code>Fetch J-Quants Data</code> を手動実行すると
                <code>public/data/stocks.json</code> が生成されます。
                それまでは「+ 銘柄を追加」で手入力した銘柄のみ分析できます。
              </div>
            </div>
          </div>
        )}

        {!loading && dataset && <DataNotices dataset={dataset} />}

        {!loading && rows.length === 0 && (
          <div className="card empty">
            分析対象の銘柄がありません。
            <div style={{ marginTop: 'var(--s4)' }}>
              <button className="btn btn-primary" onClick={() => setShowAdd(true)}>+ 銘柄を追加</button>
            </div>
          </div>
        )}

        {!loading && rows.length > 0 && (
          <>
            {tab === 'compare' && (
              <CompareView
                rows={rows} visibleIds={visibleIds} onToggle={toggleVisible}
                selectedId={selectedId} onSelect={setSelectedId}
                onAdd={() => setShowAdd(true)} onDeleteManual={deleteManual}
              />
            )}
            {tab === 'timemachine' && (
              <TimeMachineView rows={rows} selectedId={selectedId} onSelect={setSelectedId} />
            )}
            {tab === 'simulator' && (
              <SimulatorView rows={rows} selectedId={selectedId} onSelect={setSelectedId} />
            )}
          </>
        )}
      </main>

      <footer className="footer">
        データ提供: <a href="https://jpx-jquants.com/" target="_blank" rel="noreferrer">J-Quants API</a>（日本取引所グループ）
        ／ 本ツールは分析支援を目的としたものであり、投資判断・投資勧誘を行うものではありません。
      </footer>

      {showAdd && <AddStockModal onClose={() => setShowAdd(false)} onSubmit={addManual} />}
    </div>
  );
}

/** 取得できなかったエンドポイントや銘柄を必ず画面に出す (欠測を黙って隠さない) */
function DataNotices({ dataset }) {
  const unavailable = Object.entries(dataset.unavailableEndpoints || {});
  const failures = dataset.failures || [];
  if (!unavailable.length && !failures.length) return null;

  return (
    <div className="banner warn" style={{ marginBottom: 'var(--s4)' }}>
      <span>⚠</span>
      <div>
        <strong>一部のデータを取得できませんでした。</strong>
        {unavailable.length > 0 && (
          <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
            {unavailable.map(([ep, reason]) => (
              <li key={ep}>
                <code>{ep}</code> — 契約プランで利用できない可能性があります
                <div style={{ color: 'var(--text-faint)', fontSize: 11 }}>{String(reason).slice(0, 180)}</div>
              </li>
            ))}
          </ul>
        )}
        {failures.length > 0 && (
          <div style={{ marginTop: 6 }}>
            取得に失敗した銘柄: {failures.map((f) => f.code).join(', ')}
          </div>
        )}
        <div style={{ marginTop: 6, color: 'var(--text-faint)' }}>
          該当する軸は「—」と表示され、総合スコアの平均から除外されます（0点として扱いません）。
        </div>
      </div>
    </div>
  );
}
