import React, { useEffect, useState } from 'react';
import { X, Trash2, ArrowUpRight, History, Clock, Coins, Zap } from 'lucide-react';
import { API_BASE, formatActionBadge, formatDuration } from '../constants/defaultConfig';

export default function HistoryModal({ isOpen, onClose, onLoadConfig }) {
  const [sessions, setSessions] = useState([]);
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchSessions = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/sessions`);
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch (err) {
      console.error('Failed to load session history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchSessions();
      setSelectedDetail(null);
    }
  }, [isOpen]);

  const inspectSession = async (sessionId) => {
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${sessionId}`);
      if (res.ok) {
        const detail = await res.json();
        setSelectedDetail(detail);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (sessionId, e) => {
    e.stopPropagation();
    await fetch(`${API_BASE}/api/sessions/${sessionId}`, { method: 'DELETE' });
    if (selectedDetail?.session_id === sessionId) {
      setSelectedDetail(null);
    }
    fetchSessions();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-5xl max-h-[86vh] flex flex-col overflow-hidden shadow-2xl">
        {/* Modal Header */}
        <div className="px-5 py-3.5 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <History size={18} className="text-indigo-400" />
            <h2 className="text-base font-bold text-white">SQLite Session History &amp; Logs</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 grid grid-cols-1 md:grid-cols-12 overflow-hidden">
          {/* Left list */}
          <div className="md:col-span-5 border-r border-slate-800 overflow-y-auto p-3 space-y-2 max-h-[72vh]">
            {loading ? (
              <div className="text-xs text-slate-400 p-4 text-center">Loading sessions...</div>
            ) : sessions.length === 0 ? (
              <div className="text-xs text-slate-500 p-8 text-center">
                No recorded sessions yet. Launch a game session to persist telemetry automatically.
              </div>
            ) : (
              sessions.map((sess) => {
                const active = selectedDetail?.session_id === sess.session_id;
                const st = sess.stats || {};
                return (
                  <div
                    key={sess.session_id}
                    onClick={() => inspectSession(sess.session_id)}
                    className={`p-3 rounded-xl border transition-colors cursor-pointer ${
                      active
                        ? 'bg-indigo-500/15 border-indigo-500/50'
                        : 'bg-slate-950/60 hover:bg-slate-950 border-slate-800/80'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-bold text-white truncate">{sess.game_name}</span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] uppercase px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-semibold">
                          {sess.status}
                        </span>
                        <button
                          onClick={(e) => handleDelete(sess.session_id, e)}
                          className="text-slate-500 hover:text-rose-400 p-1 cursor-pointer"
                          title="Delete session"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </div>

                    <div className="text-[11px] text-slate-400 font-mono truncate mt-1">
                      {sess.provider} / {sess.model}
                    </div>

                    <div className="flex items-center gap-3 mt-2 text-[11px] text-slate-400">
                      <span className="inline-flex items-center gap-1">
                        <Zap size={11} className="text-sky-400" /> {st.total_actions || 0} acts
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <Clock size={11} className="text-indigo-400" />{' '}
                        {formatDuration(st.elapsed_seconds)}
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <Coins size={11} className="text-amber-400" /> $
                        {Number(st.estimated_cost_usd || 0).toFixed(4)}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Right detail pane */}
          <div className="md:col-span-7 overflow-y-auto p-4 max-h-[72vh]">
            {!selectedDetail ? (
              <div className="h-full flex items-center justify-center text-xs text-slate-500">
                Select a session on the left to inspect its strategy, lessons, and action log.
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-start justify-between gap-2 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-bold text-white">{selectedDetail.game_name}</h3>
                    <p className="text-xs font-mono text-sky-400">{selectedDetail.url}</p>
                    {selectedDetail.status_reason && (
                      <p className="text-xs text-slate-400 mt-1">{selectedDetail.status_reason}</p>
                    )}
                  </div>
                  <button
                    onClick={() => {
                      onLoadConfig(selectedDetail.config);
                      onClose();
                    }}
                    className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold text-white flex items-center gap-1.5 cursor-pointer"
                  >
                    <span>Load Game Config</span>
                    <ArrowUpRight size={14} />
                  </button>
                </div>

                {selectedDetail.strategy && (
                  <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs">
                    <div className="font-semibold text-violet-400 mb-1">Final Strategy</div>
                    <div className="text-slate-300">{selectedDetail.strategy}</div>
                  </div>
                )}

                <div>
                  <div className="text-xs font-semibold text-slate-300 mb-2">
                    Recorded Action Steps ({(selectedDetail.steps || []).length})
                  </div>
                  <div className="space-y-2">
                    {(selectedDetail.steps || []).map((st, idx) => (
                      <div
                        key={idx}
                        className="bg-slate-950 border border-slate-800/80 rounded-xl p-2.5 text-xs space-y-1"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-slate-200">Step #{st.iteration}</span>
                          <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 font-mono">
                            {formatActionBadge(st.action)}
                          </span>
                        </div>
                        <div className="text-slate-300">
                          <strong className="text-sky-400">Obs:</strong> {st.observation}
                        </div>
                        <div className="text-slate-400">
                          <strong className="text-violet-400">Why:</strong> {st.reasoning}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
