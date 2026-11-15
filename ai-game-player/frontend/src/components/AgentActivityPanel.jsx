import React, { useEffect, useRef, useState } from 'react';
import {
  Brain,
  Terminal,
  Lightbulb,
  CheckCircle2,
  AlertTriangle,
  Code2,
  Compass,
  ArrowDownToLine,
} from 'lucide-react';
import { formatActionBadge } from '../constants/defaultConfig';

export default function AgentActivityPanel({ sessionState, logs, sideBySide = false }) {
  const [expandedIdx, setExpandedIdx] = useState(null);
  const [activeSubTab, setActiveSubTab] = useState('feed'); // 'feed' | 'strategy' (used in sideBySide mode)
  const [autoFollow, setAutoFollow] = useState(true);
  const logContainerRef = useRef(null);
  const s = sessionState || {};

  // Scroll ONLY inside the internal log container — NEVER scroll the browser window!
  useEffect(() => {
    const el = logContainerRef.current;
    if (!el || !autoFollow) return;
    el.scrollTop = el.scrollHeight;
  }, [logs.length, autoFollow, activeSubTab]);

  const handleLogScroll = () => {
    const el = logContainerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setAutoFollow(distanceFromBottom < 48);
  };

  const renderStrategyContent = () => (
    <div className="flex flex-col gap-3.5 flex-1 overflow-y-auto pr-1">
      {/* Active Strategy */}
      <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-violet-400 mb-1 flex items-center gap-1.5">
          <Compass size={12} />
          Current Strategy
        </div>
        <p className="text-xs text-slate-200 leading-relaxed">
          {s.current_strategy || 'Start a session to synthesize game strategy and sub-goals.'}
        </p>
      </div>

      {/* Sub-Goals Breakdown */}
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
          Active Sub-Goals
        </div>
        {s.sub_goals && s.sub_goals.length > 0 ? (
          <ul className="space-y-1.5">
            {s.sub_goals.map((goal, idx) => {
              const isCurrent = s.current_subgoal && goal === s.current_subgoal;
              return (
                <li
                  key={idx}
                  className={`text-xs px-2.5 py-1.5 rounded-lg border flex items-start gap-2 ${
                    isCurrent
                      ? 'bg-sky-500/10 border-sky-500/40 text-sky-200'
                      : 'bg-slate-950/50 border-slate-800/70 text-slate-300'
                  }`}
                >
                  <CheckCircle2
                    size={13}
                    className={`shrink-0 mt-0.5 ${isCurrent ? 'text-sky-400' : 'text-slate-600'}`}
                  />
                  <span>{goal}</span>
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="text-xs text-slate-500 italic">No sub-goals generated yet.</div>
        )}
      </div>

      {/* In-Session Lessons Learned */}
      <div className="mt-auto pt-2 border-t border-slate-800/80">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-amber-400 mb-1.5 flex items-center gap-1.5">
          <Lightbulb size={12} />
          In-Session Lessons &amp; Mechanics ({(s.lessons_learned || []).length})
        </div>
        {s.lessons_learned && s.lessons_learned.length > 0 ? (
          <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
            {s.lessons_learned.map((lesson, idx) => (
              <div
                key={idx}
                className="text-[11px] text-amber-200/90 bg-amber-500/10 border border-amber-500/20 rounded-md px-2.5 py-1"
              >
                • {lesson}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500">
            Contextual patterns, ineffective inputs, and discovered mechanics appear here automatically.
          </p>
        )}
      </div>
    </div>
  );

  const renderTacticalLogList = () => (
    <div
      ref={logContainerRef}
      onScroll={handleLogScroll}
      className="flex-1 overflow-y-auto space-y-2.5 pr-1.5"
    >
      {logs.length === 0 ? (
        <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 text-xs px-4 py-8">
          <Terminal size={28} className="mb-2 opacity-25" />
          <p>Real-time visual observations, AI reasoning, and structured actions will stream here.</p>
        </div>
      ) : (
        logs.map((entry, idx) => {
          if (entry.kind === 'error') {
            return (
              <div
                key={idx}
                className="bg-rose-950/40 border border-rose-800/70 rounded-xl p-3 text-xs text-rose-200 flex items-start gap-2"
              >
                <AlertTriangle size={15} className="text-rose-400 shrink-0 mt-0.5" />
                <div>{entry.message}</div>
              </div>
            );
          }

          if (entry.kind === 'note' || entry.kind === 'plan') {
            return (
              <div
                key={idx}
                className="bg-violet-950/30 border border-violet-800/50 rounded-xl px-3 py-2 text-xs text-violet-200"
              >
                <span className="font-semibold text-violet-300">
                  {entry.kind === 'plan' ? 'Strategic Plan Updated: ' : 'System Event: '}
                </span>
                {entry.message}
              </div>
            );
          }

          const step = entry.data;
          const isExpanded = expandedIdx === idx;
          const cardBorderClass =
            step.actor_role === 'human'
              ? 'border-emerald-500/40 hover:border-emerald-400/60'
              : step.actor_role === 'api'
              ? 'border-sky-500/35 hover:border-sky-400/55'
              : step.actor_role === 'ppo'
              ? 'border-amber-500/35 hover:border-amber-400/55'
              : 'border-slate-800/90 hover:border-slate-700';

          return (
            <div
              key={idx}
              className={`bg-slate-950/75 border rounded-xl p-3 text-xs space-y-1.5 transition-colors ${cardBorderClass}`}
            >
              {/* Step Header */}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="px-2 py-0.5 rounded-md bg-slate-800 text-slate-200 font-bold tabular-nums">
                    #{step.iteration || idx + 1}
                  </span>
                  {step.actor_label && (
                    <span
                      className={`px-2 py-0.5 rounded-md font-semibold border ${
                        step.actor_role === 'human'
                          ? 'bg-emerald-500/20 border-emerald-400/50 text-emerald-200'
                          : step.actor_role === 'api'
                          ? 'bg-sky-500/15 border-sky-500/40 text-sky-200'
                          : step.actor_role === 'ppo'
                          ? 'bg-amber-500/15 border-amber-500/40 text-amber-200'
                          : 'bg-violet-500/15 border-violet-500/40 text-violet-200'
                      }`}
                    >
                      {step.actor_label}
                    </span>
                  )}
                  <span className="px-2 py-0.5 rounded-md bg-emerald-500/15 border border-emerald-500/35 text-emerald-300 font-mono font-semibold">
                    {formatActionBadge(step.action)}
                  </span>
                  {step.visual_delta_pct !== undefined && (
                    <span
                      className={`text-[11px] px-1.5 py-0.5 rounded ${
                        step.visual_delta_pct < 0.35
                          ? 'bg-amber-500/15 text-amber-300'
                          : 'bg-slate-900 text-slate-400'
                      }`}
                    >
                      Δ {step.visual_delta_pct}%
                    </span>
                  )}
                  {step.zoom_note && (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-500/15 border border-indigo-500/35 text-indigo-300 font-mono">
                      {step.zoom_note}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 text-[11px] text-slate-500">
                  {step.step_tokens > 0 && (
                    <span className="tabular-nums">
                      {step.step_tokens} tok · ${Number(step.step_cost_usd || 0).toFixed(4)}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                    className="text-slate-400 hover:text-indigo-300 p-0.5 cursor-pointer"
                    title="Toggle structured JSON"
                  >
                    <Code2 size={13} />
                  </button>
                </div>
              </div>

              <div className="text-slate-300 leading-relaxed">
                <span className="font-semibold text-sky-400">Observed: </span>
                {step.observation}
              </div>

              {step.previous_action_evaluation && (
                <div className="text-slate-400">
                  <span className="font-semibold text-amber-300/90">Feedback: </span>
                  {step.previous_action_evaluation}
                </div>
              )}

              <div className="text-slate-300 leading-relaxed">
                <span className="font-semibold text-violet-400">Decision: </span>
                {step.reasoning_summary}
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-[11px] text-slate-400 border-t border-slate-900">
                <span>
                  <strong className="text-slate-300">Expected:</strong> {step.expected_result}
                </span>
                {step.execution_status && (
                  <span className="text-emerald-400/90 font-mono">{step.execution_status}</span>
                )}
              </div>

              {isExpanded && (
                <div className="mt-2 space-y-2">
                  {step.zoom_image && (
                    <div className="p-2 rounded-lg bg-slate-900 border border-indigo-500/30">
                      <div className="text-[10px] font-semibold text-indigo-300 mb-1">
                        4× Magnifier Zoom-Snap Crop (A1–L9 · 20×20 game px per cell)
                      </div>
                      <img
                        src={`data:image/jpeg;base64,${step.zoom_image}`}
                        alt="4x Zoom-Snap Scope"
                        className="w-full max-w-md rounded border border-slate-800"
                      />
                    </div>
                  )}
                  <pre className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto">
                    {JSON.stringify(
                      { ...step, zoom_image: step.zoom_image ? '[base64_zoom_crop]' : undefined },
                      null,
                      2
                    )}
                  </pre>
                </div>
              )}
            </div>
          );
        })
      )}
    </div>
  );

  // Side-by-Side Split Monitor Mode: fits right next to LiveViewport on one screen without page scrolling
  if (sideBySide) {
    return (
      <aside className="bg-slate-900/80 border border-slate-800/90 rounded-2xl p-3.5 flex flex-col shadow-xl h-[calc(100vh-148px)] min-h-[500px]">
        <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-2.5 mb-2.5">
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setActiveSubTab('feed')}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border transition-colors cursor-pointer ${
                activeSubTab === 'feed'
                  ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <Terminal size={13} />
              <span>Action Log ({logs.length})</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveSubTab('strategy')}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border transition-colors cursor-pointer ${
                activeSubTab === 'strategy'
                  ? 'bg-violet-500/15 border-violet-500/40 text-violet-300'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <Brain size={13} />
              <span>Strategy</span>
            </button>
          </div>

          {activeSubTab === 'feed' && !autoFollow && (
            <button
              type="button"
              onClick={() => setAutoFollow(true)}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-indigo-500/20 border border-indigo-500/40 text-[11px] font-medium text-indigo-200 cursor-pointer"
              title="Jump to latest log entry"
            >
              <ArrowDownToLine size={11} />
              <span>Follow Live</span>
            </button>
          )}
        </div>

        {activeSubTab === 'feed' ? renderTacticalLogList() : renderStrategyContent()}
      </aside>
    );
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
      {/* Strategic Brain & In-Session Adaptation Card */}
      <div className="xl:col-span-5 bg-slate-900/80 border border-slate-800/90 rounded-2xl p-4 flex flex-col gap-3.5 shadow-xl h-[430px]">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
          <div className="flex items-center gap-2">
            <Brain size={17} className="text-violet-400" />
            <h2 className="text-sm font-semibold text-white">Strategic Planner &amp; Memory</h2>
          </div>
          <span className="text-[11px] px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-300 border border-violet-500/30">
            Two-Level AI
          </span>
        </div>
        {renderStrategyContent()}
      </div>

      {/* Live Tactical Decision Feed */}
      <div className="xl:col-span-7 bg-slate-900/80 border border-slate-800/90 rounded-2xl p-4 flex flex-col shadow-xl h-[430px]">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2.5 mb-3">
          <div className="flex items-center gap-2">
            <Terminal size={17} className="text-emerald-400" />
            <h2 className="text-sm font-semibold text-white">Tactical Observation &amp; Action Log</h2>
          </div>
          <div className="flex items-center gap-2">
            {!autoFollow && (
              <button
                type="button"
                onClick={() => setAutoFollow(true)}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-indigo-500/20 border border-indigo-500/40 text-[11px] font-medium text-indigo-200 cursor-pointer"
              >
                <ArrowDownToLine size={11} />
                <span>Follow Live</span>
              </button>
            )}
            <span className="text-xs text-slate-400 tabular-nums">{logs.length} events</span>
          </div>
        </div>

        {renderTacticalLogList()}
      </div>
    </div>
  );
}
