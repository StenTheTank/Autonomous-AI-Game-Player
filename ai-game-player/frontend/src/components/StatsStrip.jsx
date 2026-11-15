import React from 'react';
import {
  Zap,
  Clock,
  Coins,
  Layers,
  Target,
  Activity,
} from 'lucide-react';
import { formatDuration } from '../constants/defaultConfig';

export default function StatsStrip({ stats, displayElapsed, status }) {
  const s = stats || {};
  const iterPct = Math.min(100, Math.round(((s.current_iteration || 0) / Math.max(1, s.max_iterations || 100)) * 100));
  const budgetPct = Math.min(100, Math.round(((s.estimated_cost_usd || 0) / Math.max(0.01, s.max_budget_usd || 2)) * 100));
  const progressPct = Math.min(100, Math.max(0, s.progress_percent || 0));

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-2.5">
      {/* Actions & Iteration */}
      <div className="bg-slate-900/75 border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium">Actions &amp; Step</span>
          <Zap size={14} className="text-sky-400" />
        </div>
        <div className="mt-1.5 flex items-baseline justify-between">
          <span className="text-lg font-bold text-white tabular-nums">{s.total_actions || 0}</span>
          <span className="text-xs text-slate-400 tabular-nums">
            Iter {s.current_iteration || 0}/{s.max_iterations || 100}
          </span>
        </div>
        <div className="mt-2 h-1 w-full bg-slate-800 rounded-full overflow-hidden">
          <div className="h-full bg-sky-500 transition-all duration-300" style={{ width: `${iterPct}%` }} />
        </div>
      </div>

      {/* Elapsed Time */}
      <div className="bg-slate-900/75 border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium">Elapsed Time</span>
          <Clock size={14} className="text-indigo-400" />
        </div>
        <div className="mt-1.5 flex items-baseline justify-between">
          <span className="text-lg font-bold text-white tabular-nums">
            {formatDuration(displayElapsed)}
          </span>
          <span className="text-[11px] uppercase tracking-wide text-slate-400">{status}</span>
        </div>
        <div className="mt-2 text-[11px] text-slate-500 truncate">
          Visual Δ: <span className="text-slate-300 font-medium">{s.last_visual_delta_pct ?? 0}%</span>
        </div>
      </div>

      {/* API Calls */}
      <div className="bg-slate-900/75 border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium">LLM API Calls</span>
          <Activity size={14} className="text-violet-400" />
        </div>
        <div className="mt-1.5 flex items-baseline justify-between">
          <span className="text-lg font-bold text-white tabular-nums">{s.api_calls || 0}</span>
          <span
            className={`text-xs font-medium ${
              (s.consecutive_failures || 0) > 0 ? 'text-amber-400' : 'text-emerald-400'
            }`}
          >
            {(s.consecutive_failures || 0) > 0 ? `${s.consecutive_failures} retries` : 'Healthy'}
          </span>
        </div>
        <div className="mt-2 text-[11px] text-slate-500">Strategic + Tactical</div>
      </div>

      {/* Token Usage */}
      <div className="bg-slate-900/75 border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium">Token Usage</span>
          <Layers size={14} className="text-cyan-400" />
        </div>
        <div className="mt-1.5 flex items-baseline justify-between">
          <span className="text-lg font-bold text-white tabular-nums">
            {(s.total_tokens || 0).toLocaleString()}
          </span>
        </div>
        <div className="mt-2 text-[11px] text-slate-500 truncate tabular-nums">
          In {(s.prompt_tokens || 0).toLocaleString()} · Out {(s.completion_tokens || 0).toLocaleString()}
        </div>
      </div>

      {/* Estimated API Cost */}
      <div className="bg-slate-900/75 border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium">Est. API Cost</span>
          <Coins size={14} className="text-amber-400" />
        </div>
        <div className="mt-1.5 flex items-baseline justify-between">
          <span className="text-lg font-bold text-white tabular-nums">
            ${Number(s.estimated_cost_usd || 0).toFixed(4)}
          </span>
          <span className="text-xs text-slate-400 tabular-nums">
            / ${Number(s.max_budget_usd || 2).toFixed(2)}
          </span>
        </div>
        <div className="mt-2 h-1 w-full bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              budgetPct > 85 ? 'bg-rose-500' : budgetPct > 60 ? 'bg-amber-400' : 'bg-emerald-500'
            }`}
            style={{ width: `${budgetPct}%` }}
          />
        </div>
      </div>

      {/* Objective Progress */}
      <div className="bg-slate-900/75 border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium">Goal Progress</span>
          <Target size={14} className="text-emerald-400" />
        </div>
        <div className="mt-1.5 flex items-baseline justify-between">
          <span className="text-lg font-bold text-white tabular-nums">{progressPct}%</span>
          <span className="text-xs text-emerald-400 font-medium">
            {status === 'completed' ? 'Achieved' : status === 'failed' ? 'Failed' : 'Active'}
          </span>
        </div>
        <div className="mt-2 h-1 w-full bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-300"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
