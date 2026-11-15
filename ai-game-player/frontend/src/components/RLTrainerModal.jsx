import React, { useState, useEffect } from 'react';
import {
  X,
  Play,
  Square,
  BrainCircuit,
  Trophy,
  Gauge,
  TrendingUp,
  Cpu,
  Sparkles,
  RefreshCw,
  Trash2,
  Dna,
} from 'lucide-react';
import { API_BASE } from '../constants/defaultConfig';

function LearningCurveChart({ episodes = [], showOnlyGA = true, onToggleFilter, hasLegacyEpisodes = false }) {
  const gaEpisodes = episodes.filter((e) => String(e.run_id || '').startsWith('ga_'));
  const displayed = showOnlyGA && gaEpisodes.length > 0 ? gaEpisodes : episodes;

  if (!displayed || displayed.length === 0) {
    return (
      <div className="h-56 rounded-xl bg-slate-950/90 border border-slate-800/80 flex flex-col items-center justify-center text-slate-400 text-xs gap-2 p-4">
        <Dna size={26} className="text-indigo-400/70" />
        <span>
          No Genetic Algorithm generations recorded yet. Click &ldquo;Start Evolution&rdquo; above to evolve weights &amp; biases!
        </span>
      </div>
    );
  }

  const width = 760;
  const height = 220;
  const padL = 42;
  const padR = 18;
  const padT = 18;
  const padB = 28;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const allVps = displayed.flatMap((e) => [e.your_vp || 0, e.automa_vp || 0]);
  const minY = Math.max(0, Math.min(...allVps, 40) - 10);
  const maxY = Math.max(...allVps, 140) + 12;
  const ySpan = Math.max(20, maxY - minY);

  const xAt = (i) =>
    displayed.length === 1
      ? padL + plotW / 2
      : padL + (i / (displayed.length - 1)) * plotW;
  const yAt = (vp) => padT + plotH - ((vp - minY) / ySpan) * plotH;

  const yourPts = displayed.map((e, i) => `${xAt(i).toFixed(1)},${yAt(e.your_vp).toFixed(1)}`).join(' ');
  const automaPts = displayed.map((e, i) => `${xAt(i).toFixed(1)},${yAt(e.automa_vp).toFixed(1)}`).join(' ');

  // Rolling 10-match smoothed champion curve
  const rollingPts = displayed
    .map((_, i) => {
      const slice = displayed.slice(Math.max(0, i - 9), i + 1);
      const avg = slice.reduce((acc, item) => acc + (item.your_vp || 0), 0) / slice.length;
      return `${xAt(i).toFixed(1)},${yAt(avg).toFixed(1)}`;
    })
    .join(' ');

  // Population Mean VP curve (stored in policy_loss field on ga_* rows)
  const hasPopMean = displayed.every((e) => String(e.run_id || '').startsWith('ga_') && Number(e.policy_loss || 0) > 0);
  const popMeanPts = hasPopMean
    ? displayed.map((e, i) => `${xAt(i).toFixed(1)},${yAt(Number(e.policy_loss || 0)).toFixed(1)}`).join(' ')
    : '';

  const gridTicks = [minY, Math.round(minY + ySpan * 0.33), Math.round(minY + ySpan * 0.66), maxY];

  return (
    <div className="rounded-xl bg-slate-950/90 border border-slate-800/90 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-slate-300">
            Victory Points Evolution Curve ({displayed.length} champion games)
          </span>
          {hasLegacyEpisodes && gaEpisodes.length > 0 && (
            <button
              type="button"
              onClick={onToggleFilter}
              className="px-2 py-0.5 rounded-md bg-slate-900 hover:bg-slate-800 border border-slate-700 text-[10px] text-indigo-300 cursor-pointer"
            >
              {showOnlyGA ? 'Showing: GA Only' : 'Showing: GA + Legacy PPO'}
            </button>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3.5 text-[11px]">
          <span className="inline-flex items-center gap-1.5 text-emerald-300">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> Champion VP
          </span>
          <span className="inline-flex items-center gap-1.5 text-indigo-300">
            <span className="w-3 h-0.5 bg-indigo-400" /> Rolling Mean
          </span>
          {hasPopMean && (
            <span className="inline-flex items-center gap-1.5 text-amber-300">
              <span className="w-3 h-0.5 bg-amber-400" /> Pop Mean VP
            </span>
          )}
          <span className="inline-flex items-center gap-1.5 text-rose-300">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-400/80" /> Opponent VP
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-52 overflow-visible">
        {gridTicks.map((tick, idx) => {
          const y = yAt(tick);
          return (
            <g key={idx}>
              <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="#1e293b" strokeDasharray="4 4" />
              <text x={padL - 8} y={y + 4} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
                {tick}
              </text>
            </g>
          );
        })}

        {displayed.length > 1 && (
          <>
            <polyline
              fill="none"
              stroke="#fb7185"
              strokeOpacity="0.45"
              strokeWidth="1.5"
              strokeDasharray="3 3"
              points={automaPts}
            />
            {hasPopMean && (
              <polyline
                fill="none"
                stroke="#fbbf24"
                strokeOpacity="0.6"
                strokeWidth="1.5"
                strokeDasharray="4 3"
                points={popMeanPts}
              />
            )}
            <polyline
              fill="none"
              stroke="#34d399"
              strokeOpacity="0.45"
              strokeWidth="1.6"
              points={yourPts}
            />
            <polyline
              fill="none"
              stroke="#818cf8"
              strokeWidth="2.6"
              points={rollingPts}
            />
          </>
        )}

        {displayed.map((e, i) => {
          const isGa = String(e.run_id || '').startsWith('ga_');
          const genNum = isGa && e.value_loss ? `Gen ${Math.round(e.value_loss)}` : `Ep #${e.episode_num}`;
          return (
            <g key={e.id || i}>
              <circle
                cx={xAt(i)}
                cy={yAt(e.automa_vp)}
                r={displayed.length > 60 ? 1.8 : 2.6}
                fill="#fb7185"
                fillOpacity="0.65"
              />
              <circle
                cx={xAt(i)}
                cy={yAt(e.your_vp)}
                r={displayed.length > 60 ? 2.3 : 3.3}
                fill={e.won ? '#10b981' : '#38bdf8'}
              >
                <title>
                  {genNum}: Champion {e.your_vp} VP vs Automa {e.automa_vp} VP ({e.won ? 'WIN' : 'LOSS'})
                  {e.checkpoint_tag ? ` · ${e.checkpoint_tag}` : ''}
                </title>
              </circle>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function RLTrainerModal({
  isOpen,
  onClose,
  rlStatus,
  onRefreshStatus,
  onPlayBestInArena,
  agentActive = false,
}) {
  const [workers, setWorkers] = useState(6);
  const [trainingMode, setTrainingMode] = useState('self_play');
  const [generations, setGenerations] = useState(40);
  const [populationSize, setPopulationSize] = useState(12);
  const [gamesPerIndividual, setGamesPerIndividual] = useState(2);
  const [mutationSigma, setMutationSigma] = useState(0.05);
  const [mutationRate, setMutationRate] = useState(0.20);
  const [eliteCount, setEliteCount] = useState(2);
  const [resumeCheckpoint, setResumeCheckpoint] = useState(true);
  const [showOnlyGA, setShowOnlyGA] = useState(true);
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState(null);

  useEffect(() => {
    if (isOpen && onRefreshStatus) {
      onRefreshStatus();
    }
  }, [isOpen, onRefreshStatus]);

  if (!isOpen) return null;

  const st = rlStatus || {};
  const isTraining = !!st.running;
  const activeMode = isTraining ? (st.training_mode || trainingMode) : trainingMode;
  const recent = st.recent_episodes || [];
  const hasLegacyEpisodes = recent.some((e) => !String(e.run_id || '').startsWith('ga_'));
  const plannedGames =
    trainingMode === 'self_play'
      ? Number(generations) * Math.ceil((Number(populationSize) * Number(gamesPerIndividual)) / 2)
      : Number(generations) * Number(populationSize) * Number(gamesPerIndividual);

  const handleStartTraining = async () => {
    setLocalError(null);
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/rl/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workers: Number(workers),
          training_mode: trainingMode,
          generations: Number(generations),
          population_size: Number(populationSize),
          games_per_individual: Number(gamesPerIndividual),
          mutation_sigma: Number(mutationSigma),
          mutation_rate: Number(mutationRate),
          elite_count: Math.min(Number(eliteCount), Math.max(1, Number(populationSize) - 1)),
          resume_checkpoint: Boolean(resumeCheckpoint),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setLocalError(data.detail || 'Could not start Genetic Algorithm evolution');
      } else if (onRefreshStatus) {
        onRefreshStatus(data);
      }
    } catch (err) {
      setLocalError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleStopTraining = async () => {
    setLocalError(null);
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/rl/train/stop`, { method: 'POST' });
      const data = await res.json();
      if (onRefreshStatus) onRefreshStatus(data);
    } catch (err) {
      setLocalError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleClearHistory = async () => {
    setLocalError(null);
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/rl/episodes/clear`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        setLocalError(data.detail || 'Could not clear episode history');
      } else if (onRefreshStatus) {
        onRefreshStatus(data.status || null);
      }
    } catch (err) {
      setLocalError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-5xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-5 py-4 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-500/15 border border-indigo-500/35 flex items-center justify-center">
              <Dna size={20} className="text-indigo-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base sm:text-lg font-bold text-white">
                  Brass: Birmingham Genetic Algorithm Trainer
                </h2>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-950/70 text-emerald-300 border border-emerald-700/50">
                  Neuroevolution
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-sky-300 border border-slate-700">
                  device: {st.device || 'cpu'}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Direct Weight &amp; Bias Evolution · Paired-Seed Elitism (Zero Policy Collapse) · $0 API Cost
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {recent.length > 0 && !isTraining && (
              <button
                type="button"
                disabled={busy}
                onClick={handleClearHistory}
                className="px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-rose-950/60 border border-slate-800 hover:border-rose-700/60 text-slate-400 hover:text-rose-200 text-xs flex items-center gap-1.5 cursor-pointer"
                title="Clear old episode points from chart (keeps saved checkpoints)"
              >
                <Trash2 size={13} />
                <span className="hidden sm:inline">Clear Chart</span>
              </button>
            )}
            <button
              type="button"
              onClick={() => onRefreshStatus && onRefreshStatus()}
              className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 cursor-pointer"
              title="Refresh telemetry"
            >
              <RefreshCw size={15} />
            </button>
            <button
              type="button"
              onClick={onClose}
              className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 cursor-pointer"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="p-5 overflow-y-auto space-y-4">
          {localError && (
            <div className="px-3.5 py-2.5 rounded-xl bg-rose-950/60 border border-rose-700/70 text-xs text-rose-200">
              {localError}
            </div>
          )}

          {/* Controls Bar */}
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/90 space-y-3">
            {/* Training Arena Mode Switcher */}
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-800/80">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Evolution Opponent Mode
              </span>
              <div className="inline-flex rounded-lg bg-slate-900 p-0.5 border border-slate-800 text-xs">
                <button
                  type="button"
                  disabled={isTraining}
                  onClick={() => setTrainingMode('self_play')}
                  className={`px-3 py-1 rounded-md font-semibold transition-colors cursor-pointer disabled:opacity-60 ${
                    activeMode === 'self_play'
                      ? 'bg-indigo-600 text-white shadow'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  🧬 2P Self-Play (Population vs Itself)
                </button>
                <button
                  type="button"
                  disabled={isTraining}
                  onClick={() => setTrainingMode('solo_mautoma')}
                  className={`px-3 py-1 rounded-md font-semibold transition-colors cursor-pointer disabled:opacity-60 ${
                    activeMode === 'solo_mautoma'
                      ? 'bg-sky-600 text-white shadow'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  🤖 Solo (vs Mautoma)
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
              <div>
                <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                  Parallel Browsers
                </label>
                <input
                  type="number"
                  min="1"
                  max="12"
                  disabled={isTraining}
                  value={workers}
                  onChange={(e) => setWorkers(Math.max(1, Math.min(12, parseInt(e.target.value || '6', 10))))}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                  Generations
                </label>
                <input
                  type="number"
                  min="1"
                  max="2000"
                  step="5"
                  disabled={isTraining}
                  value={generations}
                  onChange={(e) => setGenerations(Math.max(1, parseInt(e.target.value || '40', 10)))}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                  Population Size
                </label>
                <input
                  type="number"
                  min="4"
                  max="64"
                  step="2"
                  disabled={isTraining}
                  value={populationSize}
                  onChange={(e) => setPopulationSize(Math.max(4, Math.min(64, parseInt(e.target.value || '12', 10))))}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-[11px] font-semibold text-slate-400 mb-1" title="Every genome in a generation plays these exact same Mautoma seeds">
                  Paired Seeds / Genome
                </label>
                <select
                  disabled={isTraining}
                  value={gamesPerIndividual}
                  onChange={(e) => setGamesPerIndividual(parseInt(e.target.value, 10))}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none disabled:opacity-50"
                >
                  <option value={1}>1 game (Fastest)</option>
                  <option value={2}>2 games (Balanced)</option>
                  <option value={3}>3 games (Low Variance)</option>
                  <option value={4}>4 games (Tournament)</option>
                </select>
              </div>

              <div>
                <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                  Mutation Sigma (σ)
                </label>
                <select
                  disabled={isTraining}
                  value={mutationSigma}
                  onChange={(e) => setMutationSigma(parseFloat(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none disabled:opacity-50"
                >
                  <option value={0.025}>0.025 (Fine Polish)</option>
                  <option value={0.05}>0.050 (Balanced)</option>
                  <option value={0.08}>0.080 (Broad Search)</option>
                  <option value={0.12}>0.120 (High Diversity)</option>
                </select>
              </div>

              <div>
                <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                  Mutate Rate / Elites
                </label>
                <div className="grid grid-cols-2 gap-1.5">
                  <select
                    disabled={isTraining}
                    value={mutationRate}
                    onChange={(e) => setMutationRate(parseFloat(e.target.value))}
                    title="Fraction of weights & biases mutated per child"
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-1.5 py-1.5 text-xs text-white outline-none disabled:opacity-50"
                  >
                    <option value={0.1}>10%</option>
                    <option value={0.2}>20%</option>
                    <option value={0.35}>35%</option>
                  </select>
                  <select
                    disabled={isTraining}
                    value={eliteCount}
                    onChange={(e) => setEliteCount(parseInt(e.target.value, 10))}
                    title="Top elite genomes kept unchanged each generation"
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-1.5 py-1.5 text-xs text-white outline-none disabled:opacity-50"
                  >
                    <option value={1}>1 Elite</option>
                    <option value={2}>2 Elites</option>
                    <option value={3}>3 Elites</option>
                    <option value={4}>4 Elites</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  disabled={isTraining}
                  checked={resumeCheckpoint}
                  onChange={(e) => setResumeCheckpoint(e.target.checked)}
                  className="w-4 h-4 accent-indigo-500"
                />
                <span>
                  Include saved <span className="font-mono text-indigo-300">brass_ppo_best.pt</span> alongside clean heuristic anchor in Gen 1
                </span>
              </label>

              <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
                <span className="text-[11px] font-mono text-slate-400">
                  Total evals: <strong className="text-slate-200">{plannedGames}</strong> games
                </span>
                {!isTraining ? (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={handleStartTraining}
                    className="py-2 px-4 rounded-lg text-xs sm:text-sm font-bold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white flex items-center justify-center gap-1.5 shadow-lg shadow-emerald-600/20 cursor-pointer"
                  >
                    <Play size={14} fill="currentColor" />
                    <span>Start Evolution</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={handleStopTraining}
                    className="py-2 px-4 rounded-lg text-xs sm:text-sm font-bold bg-rose-600 hover:bg-rose-500 disabled:opacity-40 text-white flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    <Square size={13} fill="currentColor" />
                    <span>Stop &amp; Save Best</span>
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Status Banner */}
          <div className="px-3.5 py-2 rounded-xl bg-indigo-950/30 border border-indigo-800/40 flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2 text-indigo-200">
              <span
                className={`w-2 h-2 rounded-full ${
                  isTraining ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'
                }`}
              />
              <span className="font-medium">{st.status_message || 'Ready to evolve weights & biases'}</span>
            </div>
            <div className="font-mono text-[11px] text-slate-400">
              Gen Eval: {st.generation_progress || '0 / 0'} · Elites: {st.elite_count || eliteCount} · σ:{' '}
              {Number(st.mutation_sigma || mutationSigma).toFixed(4)}
            </div>
          </div>

          {/* KPI Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[11px] text-slate-400 flex items-center gap-1">
                <Cpu size={12} className="text-sky-400" /> Generation
              </div>
              <div className="text-lg font-bold text-white mt-0.5 font-mono">
                {st.current_generation || 0}
                {st.target_generations ? ` / ${st.target_generations}` : ''}
              </div>
              <div className="text-[10px] text-slate-500">
                {st.completed_episodes || 0}
                {st.target_episodes ? ` / ${st.target_episodes}` : ''} total games
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[11px] text-slate-400 flex items-center gap-1">
                <Dna size={12} className="text-emerald-400" /> Champion Avg VP
              </div>
              <div className="text-lg font-bold text-emerald-300 mt-0.5 font-mono">
                {Number(st.champion_avg_vp || st.rolling_avg_vp || 0).toFixed(1)} VP
              </div>
              <div className="text-[10px] text-slate-400">
                Pop Mean: {Number(st.generation_mean_vp || 0).toFixed(1)} VP
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[11px] text-slate-400 flex items-center gap-1">
                <TrendingUp size={12} className="text-indigo-400" /> Champion Win Rate
              </div>
              <div className="text-lg font-bold text-indigo-300 mt-0.5 font-mono">
                {Number(st.rolling_win_rate_pct || 0).toFixed(1)}%
              </div>
              <div className="text-[10px] text-slate-500">
                vs Opponent {Number(st.rolling_automa_vp || 0).toFixed(1)} VP
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[11px] text-slate-400 flex items-center gap-1">
                <Trophy size={12} className="text-amber-400" /> Peak Game VP
              </div>
              <div className="text-lg font-bold text-amber-300 mt-0.5 font-mono">
                {st.best_vp || 0} VP
              </div>
              <div className="text-[10px] text-slate-500">
                {st.has_best_checkpoint ? 'brass_ppo_best.pt ready' : 'No checkpoint yet'}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[11px] text-slate-400 flex items-center gap-1">
                <Gauge size={12} className="text-teal-400" /> Throughput
              </div>
              <div className="text-lg font-bold text-teal-300 mt-0.5 font-mono">
                {Number(st.games_per_min || 0).toFixed(1)}
              </div>
              <div className="text-[10px] text-slate-500">full games / min</div>
            </div>
          </div>

          {/* Chart */}
          <LearningCurveChart
            episodes={recent}
            showOnlyGA={showOnlyGA}
            hasLegacyEpisodes={hasLegacyEpisodes}
            onToggleFilter={() => setShowOnlyGA((v) => !v)}
          />

          {/* Recent Champion Matches Table */}
          {recent.length > 0 && (
            <div className="rounded-xl bg-slate-950/80 border border-slate-800 overflow-hidden">
              <div className="px-3.5 py-2 border-b border-slate-800 text-xs font-semibold text-slate-300">
                Latest Generation Champion Matches
              </div>
              <div className="max-h-40 overflow-y-auto divide-y divide-slate-800/70 text-xs font-mono">
                {[...recent]
                  .reverse()
                  .slice(0, 12)
                  .map((ep) => {
                    const isGa = String(ep.run_id || '').startsWith('ga_');
                    const genLabel = isGa && ep.value_loss ? `Gen #${Math.round(ep.value_loss)}` : `Ep #${ep.episode_num}`;
                    return (
                      <div key={ep.id} className="px-3.5 py-1.5 flex items-center justify-between gap-2">
                        <span className="text-slate-400">{genLabel}</span>
                        <span className={ep.won ? 'text-emerald-300 font-bold' : 'text-slate-300'}>
                          Champion {ep.your_vp} VP vs Opp {ep.automa_vp} VP ({ep.won ? 'WIN' : 'LOSS'})
                        </span>
                        <span className="text-slate-500">
                          {isGa ? `Fit=${Number(ep.total_reward || 0).toFixed(1)}` : `R=${Number(ep.total_reward || 0).toFixed(2)}`}{' '}
                          · {ep.steps} turns
                          {ep.checkpoint_tag ? ` · [${ep.checkpoint_tag}]` : ''}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}
        </div>

        {/* Footer CTA */}
        <div className="px-5 py-3.5 bg-slate-950/90 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <div className="text-[11px] text-slate-400 font-mono">
            CLI: <span className="text-sky-300">python train_brass_rl.py --workers 6 --generations 40 --pop-size 12</span>
          </div>
          <button
            type="button"
            disabled={isTraining || agentActive}
            onClick={() => onPlayBestInArena && onPlayBestInArena()}
            className="px-4 py-2 rounded-xl text-xs sm:text-sm font-bold bg-gradient-to-r from-indigo-600 via-sky-600 to-emerald-600 hover:from-indigo-500 hover:via-sky-500 hover:to-emerald-500 disabled:opacity-40 text-white flex items-center gap-2 shadow-lg cursor-pointer disabled:cursor-not-allowed"
          >
            <Play size={14} fill="currentColor" />
            <span>Play Live in Arena with Evolved Champion ($0 Cost)</span>
          </button>
        </div>
      </div>
    </div>
  );
}
