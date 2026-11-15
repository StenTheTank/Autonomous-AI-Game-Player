import React from 'react';
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  History,
  Cpu,
  BrainCircuit,
  Wifi,
  WifiOff,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';
import { STATUS_STYLES } from '../constants/defaultConfig';

export default function HeaderBar({
  status,
  wsConnected,
  sidebarOpen,
  onToggleSidebar,
  onStart,
  onPause,
  onResume,
  onStop,
  onRestart,
  onOpenHistory,
  onOpenRLTrainer,
  rlTrainingRunning = false,
}) {
  const style = STATUS_STYLES[status] || STATUS_STYLES.idle;
  const isActive = ['starting', 'planning', 'running', 'paused'].includes(status);
  const isPaused = status === 'paused';

  return (
    <header className="sticky top-0 z-30 bg-slate-950/90 backdrop-blur-md border-b border-slate-800/90 px-4 lg:px-6 py-3">
      <div className="max-w-[1800px] mx-auto flex flex-wrap items-center justify-between gap-3">
        {/* Brand & Sidebar Toggle */}
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleSidebar}
            title={sidebarOpen ? 'Collapse Configuration Panel' : 'Expand Configuration Panel'}
            className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition-colors cursor-pointer"
          >
            {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
          </button>

          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 via-sky-500 to-emerald-400 p-[1px] shadow-lg shadow-indigo-500/20">
              <div className="w-full h-full bg-slate-950 rounded-[11px] flex items-center justify-center">
                <Cpu size={19} className="text-sky-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-bold tracking-tight text-white">
                  AI Game Player
                </h1>
                <span className="hidden sm:inline-block text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                  Vision v2
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden md:block">
                Autonomous Browser Observation, Planning &amp; Control
              </p>
            </div>
          </div>
        </div>

        {/* Status Badges & Session Controls */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {/* WebSocket Link Indicator */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
              wsConnected
                ? 'bg-emerald-950/50 text-emerald-300 border-emerald-800/60'
                : 'bg-rose-950/60 text-rose-300 border-rose-800/60'
            }`}
          >
            {wsConnected ? <Wifi size={13} /> : <WifiOff size={13} />}
            <span className="hidden sm:inline">{wsConnected ? 'Engine Online' : 'Disconnected'}</span>
          </div>

          {/* Session Status Pill */}
          <div
            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold border ${style.badge}`}
          >
            <span className="relative flex h-2 w-2">
              <span className={`h-2 w-2 rounded-full ${style.dot}`} />
            </span>
            <span>{style.label}</span>
          </div>

          <div className="h-5 w-[1px] bg-slate-800 hidden sm:block" />

          {/* Primary Session Controls */}
          <div className="flex items-center gap-1.5">
            {!isActive ? (
              <button
                onClick={onStart}
                disabled={!wsConnected}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white shadow-lg shadow-emerald-600/20 transition-all cursor-pointer disabled:cursor-not-allowed"
              >
                <Play size={15} fill="currentColor" />
                <span>Start</span>
              </button>
            ) : isPaused ? (
              <button
                onClick={onResume}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-semibold bg-amber-500 hover:bg-amber-400 text-slate-950 shadow-lg shadow-amber-500/20 transition-all cursor-pointer"
              >
                <Play size={15} fill="currentColor" />
                <span>Resume</span>
              </button>
            ) : (
              <button
                onClick={onPause}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-semibold bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/35 transition-all cursor-pointer"
              >
                <Pause size={15} />
                <span>Pause</span>
              </button>
            )}

            <button
              onClick={onStop}
              disabled={!isActive}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-semibold bg-rose-600/20 hover:bg-rose-600 hover:text-white disabled:opacity-35 text-rose-300 border border-rose-500/35 transition-all cursor-pointer disabled:cursor-not-allowed"
            >
              <Square size={14} fill="currentColor" />
              <span>Stop</span>
            </button>

            <button
              onClick={onRestart}
              disabled={!wsConnected}
              title="Restart Session"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium bg-slate-900 hover:bg-slate-800 disabled:opacity-40 text-slate-300 border border-slate-800 transition-all cursor-pointer disabled:cursor-not-allowed"
            >
              <RotateCcw size={14} />
              <span className="hidden lg:inline">Restart</span>
            </button>
          </div>

          <button
            onClick={onOpenRLTrainer}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-semibold border transition-colors cursor-pointer ${
              rlTrainingRunning
                ? 'bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border-emerald-500/40'
                : 'bg-indigo-500/15 hover:bg-indigo-500/25 text-indigo-200 border-indigo-500/35'
            }`}
            title="Open Brass: Birmingham Genetic Algorithm (Neuroevolution) Trainer"
          >
            <BrainCircuit size={15} className={rlTrainingRunning ? 'text-emerald-400 animate-pulse' : 'text-indigo-400'} />
            <span className="hidden sm:inline">{rlTrainingRunning ? 'GA Evolving...' : 'GA Trainer'}</span>
          </button>

          <button
            onClick={onOpenHistory}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition-colors cursor-pointer"
          >
            <History size={15} className="text-indigo-400" />
            <span className="hidden sm:inline">History</span>
          </button>
        </div>
      </div>
    </header>
  );
}
