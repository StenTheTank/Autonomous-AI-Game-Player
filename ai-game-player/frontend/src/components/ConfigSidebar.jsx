import React, { useState } from 'react';
import {
  KeyRound,
  Gamepad2,
  Sliders,
  Sparkles,
  Eye,
  EyeOff,
  CheckCircle2,
  AlertCircle,
  Loader2,
  BookmarkPlus,
  Trash2,
  Play,
  ShieldCheck,
  RefreshCw,
  BrainCircuit,
  Swords,
  Dices,
} from 'lucide-react';
import { API_BASE, DEFAULT_PROVIDER_MODELS } from '../constants/defaultConfig';

export default function ConfigSidebar({
  config,
  setConfig,
  presets,
  providerModels,
  savedProfiles,
  onSaveProfile,
  onDeleteProfile,
  onStart,
  isRunning,
  onOpenRLTrainer,
  rlStatus,
}) {
  const [activeTab, setActiveTab] = useState('llm'); // 'llm' | 'game' | 'agent'
  const [showKey, setShowKey] = useState(false);
  const [testingKey, setTestingKey] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [fetchingModels, setFetchingModels] = useState(false);
  const [liveModels, setLiveModels] = useState({});
  const [profileName, setProfileName] = useState('');

  const modelsCatalog = providerModels || DEFAULT_PROVIDER_MODELS;
  const availableModels = liveModels[config.provider] || modelsCatalog[config.provider] || [];

  const updateField = (name, value) => {
    setConfig((prev) => ({ ...prev, [name]: value }));
  };

  const handleProviderChange = (e) => {
    const nextProvider = e.target.value;
    const defaultModel = (modelsCatalog[nextProvider] || ['gpt-4o'])[0];
    setTestResult(null);
    const brassPreset = (presets || []).find((p) => p.id === 'brass-birmingham');
    setConfig((prev) => ({
      ...prev,
      ...(nextProvider === 'brass_rl' && !prev.url.includes('brass')
        ? brassPreset || {
            game_name: 'Brass: Birmingham (Solo vs Mautoma)',
            url: 'http://localhost:8000/brass-birmingham/',
            max_iterations: 200,
          }
        : {}),
      ...(nextProvider === 'brass_rl' && prev.match_mode === 'brass_2p_duel'
        ? { match_mode: 'single' }
        : {}),
      api_key: prev.api_key,
      provider: nextProvider,
      model: defaultModel,
    }));
  };

  const applyPreset = (preset) => {
    setConfig((prev) => {
      const nextMode = preset.match_mode || 'single';
      let nextProvider = prev.provider;
      let nextModel = prev.model;
      if (nextMode === 'brass_2p_duel' && prev.provider === 'brass_rl') {
        nextProvider = 'gemini';
        nextModel = (modelsCatalog.gemini || ['gemini-2.5-flash'])[0];
      } else if (nextMode === 'brass_human_vs_ai') {
        // Default You-vs-AI to Local Trained PPO ($0 cost, ready immediately with 0 API key!)
        nextProvider = preset.provider || 'brass_rl';
        nextModel = preset.model || 'brass-ppo-best';
      }
      return {
        ...prev,
        match_mode: 'single',
        ...preset,
        api_key: prev.api_key,
        provider: nextProvider,
        model: nextModel,
      };
    });
  };

  const selectBrassMode = (targetMode) => {
    const duelPreset = (presets || []).find((p) => p.id === 'brass-2p-duel');
    const humanPreset = (presets || []).find((p) => p.id === 'brass-human-vs-ai');
    const soloPreset = (presets || []).find((p) => p.id === 'brass-birmingham');

    if (targetMode === 'brass_2p_duel') {
      applyPreset(
        duelPreset || {
          game_name: '⚔️ Brass 2P Duel: API vs Trained PPO',
          url: 'http://localhost:8000/brass-birmingham/',
          match_mode: 'brass_2p_duel',
          opponent_model: 'brass-ppo-best',
          api_player_seat: 0,
          game_seed: 202611,
          enable_rl_advisor: false,
          default_action_duration_ms: 260,
          observation_interval_ms: 1100,
          max_iterations: 220,
          max_budget_usd: 5.0,
        }
      );
    } else if (targetMode === 'brass_human_vs_ai') {
      applyPreset(
        humanPreset || {
          game_name: '🎮 Brass 2P: You vs AI Model',
          url: 'http://localhost:8000/brass-birmingham/',
          match_mode: 'brass_human_vs_ai',
          provider: 'brass_rl',
          model: 'brass-ppo-best',
          human_player_seat: 0,
          game_seed: 202611,
          enable_rl_advisor: false,
          default_action_duration_ms: 240,
          observation_interval_ms: 900,
          max_iterations: 300,
          max_session_duration_sec: 7200,
          max_budget_usd: 5.0,
        }
      );
    } else {
      setConfig((prev) => ({
        ...prev,
        ...(soloPreset || {}),
        match_mode: 'single',
        api_key: prev.api_key,
        provider: prev.provider,
        model: prev.model,
      }));
    }
  };

  const randomizeGameSeed = () => {
    updateField('game_seed', Math.floor(100000 + Math.random() * 900000));
  };

  const handleTestKey = async () => {
    if (!config.api_key.trim()) {
      setTestResult({ ok: false, error: 'Enter your API key first.' });
      return;
    }
    setTestingKey(true);
    setTestResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/test-llm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: config.provider,
          api_key: config.api_key,
          model: config.model,
          base_url: config.base_url || null,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setTestResult({ ok: false, error: data.detail || 'Verification failed' });
      } else {
        setTestResult(data);
      }
    } catch (err) {
      setTestResult({ ok: false, error: err.message });
    } finally {
      setTestingKey(false);
    }
  };

  const handleFetchModels = async () => {
    if (!config.api_key.trim()) {
      setTestResult({ ok: false, error: 'Enter your API key first to fetch available models.' });
      return;
    }
    setFetchingModels(true);
    try {
      const res = await fetch(`${API_BASE}/api/list-models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: config.provider,
          api_key: config.api_key,
          model: config.model,
          base_url: config.base_url || null,
        }),
      });
      const data = await res.json();
      if (data.ok && data.models?.length > 0) {
        setLiveModels((prev) => ({ ...prev, [config.provider]: data.models }));
        if (!data.models.includes(config.model)) {
          updateField('model', data.models[0]);
        }
      } else {
        setTestResult({ ok: false, error: data.error || data.detail || 'Could not list models' });
      }
    } catch (err) {
      setTestResult({ ok: false, error: err.message });
    } finally {
      setFetchingModels(false);
    }
  };

  const handleCreateProfile = () => {
    const name = profileName.trim() || config.game_name;
    if (!name) return;
    onSaveProfile(name);
    setProfileName('');
  };

  return (
    <aside className="w-full lg:w-[390px] xl:w-[420px] shrink-0 bg-slate-900/80 border border-slate-800/90 rounded-2xl flex flex-col overflow-hidden shadow-2xl">
      {/* Preset Quick-Load Strip */}
      <div className="p-3.5 bg-slate-950/60 border-b border-slate-800/80">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Sparkles size={13} className="text-indigo-400" />
            Quick-Load Game Preset
          </span>
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          {(presets || []).map((p) => {
            const selected =
              config.url === p.url &&
              (p.match_mode || 'single') === (config.match_mode || 'single');
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => applyPreset(p)}
                className={`text-left px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all border truncate cursor-pointer ${
                  selected
                    ? 'bg-indigo-500/20 text-indigo-200 border-indigo-500/50'
                    : 'bg-slate-900 hover:bg-slate-800 text-slate-300 border-slate-800'
                }`}
                title={p.game_description}
              >
                {p.game_name}
              </button>
            );
          })}
        </div>

        {savedProfiles && savedProfiles.length > 0 && (
          <div className="mt-2.5 pt-2 border-t border-slate-800/70">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
              Saved Custom Profiles
            </div>
            <div className="flex flex-wrap gap-1.5">
              {savedProfiles.map((prof) => (
                <div
                  key={prof.id}
                  className="inline-flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-md pl-2 pr-1 py-0.5 text-xs text-slate-300"
                >
                  <button
                    type="button"
                    onClick={() => applyPreset(prof.config)}
                    className="hover:text-white truncate max-w-[140px] cursor-pointer"
                  >
                    {prof.name}
                  </button>
                  <button
                    type="button"
                    onClick={() => onDeleteProfile(prof.id)}
                    className="text-slate-500 hover:text-rose-400 p-0.5 cursor-pointer"
                    title="Delete profile"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Section Tabs */}
      <div className="grid grid-cols-3 border-b border-slate-800 bg-slate-900/90 text-xs font-semibold">
        <button
          type="button"
          onClick={() => setActiveTab('llm')}
          className={`py-2.5 flex items-center justify-center gap-1.5 border-b-2 transition-colors cursor-pointer ${
            activeTab === 'llm'
              ? 'border-indigo-500 text-indigo-300 bg-indigo-500/5'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <KeyRound size={14} />
          <span>1. LLM API</span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('game')}
          className={`py-2.5 flex items-center justify-center gap-1.5 border-b-2 transition-colors cursor-pointer ${
            activeTab === 'game'
              ? 'border-sky-500 text-sky-300 bg-sky-500/5'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Gamepad2 size={14} />
          <span>2. Game Rules</span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('agent')}
          className={`py-2.5 flex items-center justify-center gap-1.5 border-b-2 transition-colors cursor-pointer ${
            activeTab === 'agent'
              ? 'border-emerald-500 text-emerald-300 bg-emerald-500/5'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Sliders size={14} />
          <span>3. Agent</span>
        </button>
      </div>

      {/* Scrollable Form Body */}
      <div className="p-4 overflow-y-auto flex-1 space-y-3.5 max-h-[calc(100vh-310px)] min-h-[380px]">
        {activeTab === 'llm' && (
          <>
            {((config.url || '').includes('brass') ||
              config.match_mode === 'brass_2p_duel' ||
              config.match_mode === 'brass_human_vs_ai') && (
              <div className="p-3 rounded-xl bg-gradient-to-br from-indigo-950/55 via-slate-900/90 to-amber-950/25 border border-indigo-500/45 space-y-2.5">
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs font-bold text-indigo-200 flex items-center gap-1.5">
                    <Swords size={14} className="text-amber-400" />
                    Brass: Birmingham Match Mode
                  </span>
                  <div className="grid grid-cols-3 rounded-lg bg-slate-950 p-0.5 border border-slate-800 text-[11px] gap-0.5">
                    <button
                      type="button"
                      onClick={() => selectBrassMode('single')}
                      className={`px-1.5 py-1 rounded-md font-semibold cursor-pointer transition-colors ${
                        (config.match_mode || 'single') === 'single'
                          ? 'bg-slate-800 text-white'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Solo Automa
                    </button>
                    <button
                      type="button"
                      onClick={() => selectBrassMode('brass_2p_duel')}
                      className={`px-1.5 py-1 rounded-md font-semibold cursor-pointer transition-colors ${
                        config.match_mode === 'brass_2p_duel'
                          ? 'bg-indigo-600 text-white'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      ⚔️ API vs GA
                    </button>
                    <button
                      type="button"
                      onClick={() => selectBrassMode('brass_human_vs_ai')}
                      className={`px-1.5 py-1 rounded-md font-semibold cursor-pointer transition-colors ${
                        config.match_mode === 'brass_human_vs_ai'
                          ? 'bg-emerald-600 text-white'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      🎮 You vs AI
                    </button>
                  </div>
                </div>

                {(config.match_mode === 'brass_2p_duel' ||
                  config.match_mode === 'brass_human_vs_ai') && (
                  <div className="space-y-2.5 pt-1 border-t border-indigo-500/25">
                    <p className="text-[11px] text-slate-300 leading-relaxed">
                      {config.match_mode === 'brass_human_vs_ai' ? (
                        <>
                          Play official 2P <span className="text-white font-semibold">Brass: Birmingham</span> yourself! Click directly on the Live Arena board on your turn — the AI&apos;s hand is masked (<span className="font-mono text-amber-300">🂠 Hidden</span>).
                        </>
                      ) : (
                        <>
                          Spectate official 2P <span className="text-white font-semibold">Brass: Birmingham</span> (40-card 2P deck). Each AI sees only its own hand and plays visually step-by-step.
                        </>
                      )}
                    </p>

                    {config.match_mode === 'brass_human_vs_ai' ? (
                      <>
                        <div>
                          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                            Your AI Opponent
                          </label>
                          <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                            <button
                              type="button"
                              onClick={() => {
                                updateField('provider', 'brass_rl');
                                updateField('model', 'brass-ppo-best');
                              }}
                              className={`py-1.5 px-2 rounded-lg border font-semibold cursor-pointer transition-colors ${
                                config.provider === 'brass_rl'
                                  ? 'bg-indigo-500/25 border-indigo-400/70 text-indigo-100'
                                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              🧬 Evolved AI ($0)
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                if (config.provider === 'brass_rl') {
                                  const defGem = (modelsCatalog.gemini || ['gemini-2.5-flash'])[0];
                                  updateField('provider', 'gemini');
                                  updateField('model', defGem);
                                }
                              }}
                              className={`py-1.5 px-2 rounded-lg border font-semibold cursor-pointer transition-colors ${
                                config.provider !== 'brass_rl'
                                  ? 'bg-sky-500/25 border-sky-400/70 text-sky-100'
                                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              ☁️ Cloud API Model
                            </button>
                          </div>
                        </div>

                        <div>
                          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                            Your Seat Color
                          </label>
                          <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                            <button
                              type="button"
                              onClick={() => updateField('human_player_seat', 0)}
                              className={`py-1.5 px-2 rounded-lg border font-medium cursor-pointer transition-colors ${
                                Number(config.human_player_seat || 0) === 0
                                  ? 'bg-emerald-500/20 border-emerald-400/60 text-emerald-200'
                                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              P1: You · P2: AI
                            </button>
                            <button
                              type="button"
                              onClick={() => updateField('human_player_seat', 1)}
                              className={`py-1.5 px-2 rounded-lg border font-medium cursor-pointer transition-colors ${
                                Number(config.human_player_seat || 0) === 1
                                  ? 'bg-emerald-500/20 border-emerald-400/60 text-emerald-200'
                                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              P1: AI · P2: You
                            </button>
                          </div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                            Local Trained PPO Opponent
                          </label>
                          <div className="flex gap-1.5">
                            {['brass-ppo-best', 'brass-ppo-latest'].map((ckpt) => (
                              <button
                                key={ckpt}
                                type="button"
                                onClick={() => updateField('opponent_model', ckpt)}
                                className={`flex-1 py-1 px-2 rounded-md text-[11px] font-mono border cursor-pointer transition-colors ${
                                  (config.opponent_model || 'brass-ppo-best') === ckpt
                                    ? 'bg-amber-500/20 border-amber-400/60 text-amber-200'
                                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                                }`}
                              >
                                {ckpt}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div>
                          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                            Player Seat Assignment
                          </label>
                          <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                            <button
                              type="button"
                              onClick={() => updateField('api_player_seat', 0)}
                              className={`py-1.5 px-2 rounded-lg border font-medium cursor-pointer transition-colors ${
                                Number(config.api_player_seat || 0) === 0
                                  ? 'bg-sky-500/20 border-sky-400/60 text-sky-200'
                                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              P1: API · P2: PPO
                            </button>
                            <button
                              type="button"
                              onClick={() => updateField('api_player_seat', 1)}
                              className={`py-1.5 px-2 rounded-lg border font-medium cursor-pointer transition-colors ${
                                Number(config.api_player_seat || 0) === 1
                                  ? 'bg-sky-500/20 border-sky-400/60 text-sky-200'
                                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              P1: PPO · P2: API
                            </button>
                          </div>
                        </div>
                      </>
                    )}

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                          2P Deck Seed
                        </label>
                        <div className="flex gap-1">
                          <input
                            type="number"
                            min="1"
                            max="999999999"
                            value={config.game_seed ?? 202611}
                            onChange={(e) =>
                              updateField('game_seed', Math.max(0, parseInt(e.target.value || '1', 10)))
                            }
                            className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-lg px-2 py-1 text-xs font-mono text-white outline-none"
                          />
                          <button
                            type="button"
                            onClick={randomizeGameSeed}
                            className="px-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-indigo-300 cursor-pointer"
                            title="Generate random seed"
                          >
                            <Dices size={13} />
                          </button>
                        </div>
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                          AI Pace
                        </label>
                        <div className="grid grid-cols-3 gap-1 text-[10px]">
                          {[
                            { lbl: 'Slow', obs: 2000, act: 380 },
                            { lbl: 'Live', obs: 1100, act: 260 },
                            { lbl: 'Fast', obs: 350, act: 120 },
                          ].map((sp) => {
                            const activeSp = Math.abs((config.observation_interval_ms || 1100) - sp.obs) < 280;
                            return (
                              <button
                                key={sp.lbl}
                                type="button"
                                onClick={() => {
                                  updateField('observation_interval_ms', sp.obs);
                                  updateField('default_action_duration_ms', sp.act);
                                }}
                                className={`py-1 rounded border font-semibold cursor-pointer ${
                                  activeSp
                                    ? 'bg-emerald-500/20 border-emerald-400/60 text-emerald-200'
                                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                                }`}
                              >
                                {sp.lbl}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                {config.match_mode === 'brass_2p_duel'
                  ? 'API Challenger Provider'
                  : config.match_mode === 'brass_human_vs_ai'
                  ? 'AI Opponent Provider'
                  : 'AI Model Provider'}
              </label>
              <select
                value={config.provider}
                onChange={handleProviderChange}
                className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-lg px-3 py-2 text-sm text-white outline-none transition-colors"
              >
                <option value="openai">OpenAI (Vision)</option>
                <option value="anthropic">Anthropic Claude (Vision)</option>
                <option value="gemini">Google Gemini (OpenAI-Compatible)</option>
                {config.match_mode !== 'brass_2p_duel' && (
                  <option value="brass_rl">Brass Evolved Policy (Local PyTorch GA · $0 Cost)</option>
                )}
              </select>
            </div>

            {config.provider === 'brass_rl' ? (
              <div className="p-3.5 rounded-xl bg-indigo-950/35 border border-indigo-500/40 space-y-3">
                <div className="flex items-start gap-2.5">
                  <BrainCircuit size={18} className="text-indigo-400 shrink-0 mt-0.5" />
                  <div>
                    <div className="text-xs font-bold text-indigo-200">
                      Local PyTorch Genetic Algorithm Policy · $0.00 API Cost
                    </div>
                    <p className="text-[11px] text-slate-300 mt-0.5 leading-relaxed">
                      Plays <span className="font-mono text-sky-300">Brass: Birmingham</span> visually in the Arena with no API key required, using neuroevolved weights &amp; biases.
                    </p>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">
                    Checkpoint Selection
                  </label>
                  <div className="flex gap-2 flex-wrap">
                    {['brass-ppo-best', 'brass-ppo-latest'].map((ckpt) => (
                      <button
                        key={ckpt}
                        type="button"
                        onClick={() => updateField('model', ckpt)}
                        className={`px-2.5 py-1 rounded-md text-xs font-mono border transition-colors cursor-pointer ${
                          config.model === ckpt
                            ? 'bg-indigo-500/25 border-indigo-400 text-indigo-100'
                            : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {ckpt}
                      </button>
                    ))}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-1.5">
                    {rlStatus?.has_best_checkpoint
                      ? `Evolved checkpoint ready (Peak: ${rlStatus.best_vp || 0} VP · Champion Avg: ${Number(rlStatus.champion_avg_vp || rlStatus.rolling_avg_vp || 0).toFixed(1)} VP)`
                      : 'No saved checkpoint yet — open GA Trainer below to evolve model weights & biases first!'}
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">
                    Move Selection Temperature ({config.temperature} · {config.temperature <= 0.15 ? 'Greedy Best Move' : 'Stochastic'})
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={config.temperature}
                    onChange={(e) => updateField('temperature', parseFloat(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                </div>

                <button
                  type="button"
                  onClick={() => onOpenRLTrainer && onOpenRLTrainer()}
                  className="w-full py-2 px-3 rounded-lg bg-indigo-600/25 hover:bg-indigo-600/40 border border-indigo-500/45 text-xs font-semibold text-indigo-200 flex items-center justify-center gap-2 transition-colors cursor-pointer"
                >
                  <BrainCircuit size={14} />
                  <span>Open Genetic Algorithm Trainer</span>
                </button>
              </div>
            ) : (
              <>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-slate-300">API Key</label>
                <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400">
                  <ShieldCheck size={11} /> Memory-only (never logged)
                </span>
              </div>
              <div className="relative flex items-center">
                <input
                  type={showKey ? 'text' : 'password'}
                  value={config.api_key}
                  onChange={(e) => updateField('api_key', e.target.value)}
                  placeholder={
                    config.provider === 'anthropic'
                      ? 'sk-ant-api03-...'
                      : config.provider === 'gemini'
                      ? 'AIzaSy...'
                      : 'sk-proj-...'
                  }
                  autoComplete="off"
                  className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-lg pl-3 pr-9 py-2 text-sm text-white font-mono outline-none transition-colors"
                />
                <button
                  type="button"
                  onClick={() => setShowKey((s) => !s)}
                  className="absolute right-2.5 text-slate-400 hover:text-slate-200 cursor-pointer"
                >
                  {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-slate-300">Vision Model</label>
                <button
                  type="button"
                  onClick={handleFetchModels}
                  disabled={fetchingModels}
                  className="inline-flex items-center gap-1 text-[11px] text-indigo-400 hover:text-indigo-300 cursor-pointer"
                >
                  <RefreshCw size={11} className={fetchingModels ? 'animate-spin' : ''} />
                  <span>{fetchingModels ? 'Loading...' : 'Fetch Live Models'}</span>
                </button>
              </div>
              <div className="flex gap-2 mb-1.5 flex-wrap max-h-24 overflow-y-auto">
                {availableModels.map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => updateField('model', m)}
                    className={`px-2 py-1 rounded-md text-[11px] font-mono border transition-colors cursor-pointer ${
                      config.model === m
                        ? 'bg-indigo-500/20 border-indigo-500 text-indigo-200'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
              <input
                type="text"
                value={config.model}
                onChange={(e) => updateField('model', e.target.value)}
                placeholder="Or enter any vision-capable model ID..."
                className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-200 outline-none"
              />
            </div>

            {/* Verify Connection Button */}
            <div>
              <button
                type="button"
                onClick={handleTestKey}
                disabled={testingKey}
                className="w-full py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-semibold text-slate-200 flex items-center justify-center gap-2 transition-colors cursor-pointer"
              >
                {testingKey ? (
                  <>
                    <Loader2 size={14} className="animate-spin text-indigo-400" />
                    <span>Verifying Model Connection...</span>
                  </>
                ) : (
                  <span>Verify API Key &amp; Model</span>
                )}
              </button>

              {testResult && (
                <div
                  className={`mt-2 p-2.5 rounded-lg text-xs border flex items-start gap-2 ${
                    testResult.ok
                      ? 'bg-emerald-950/40 border-emerald-800/70 text-emerald-300'
                      : 'bg-rose-950/40 border-rose-800/70 text-rose-300'
                  }`}
                >
                  {testResult.ok ? (
                    <CheckCircle2 size={15} className="shrink-0 mt-0.5 text-emerald-400" />
                  ) : (
                    <AlertCircle size={15} className="shrink-0 mt-0.5 text-rose-400" />
                  )}
                  <div className="break-words">
                    {testResult.ok
                      ? `Connected to ${testResult.model} (${testResult.tokens} tokens)`
                      : testResult.error}
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3 pt-1">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Temperature ({config.temperature})
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={config.temperature}
                  onChange={(e) => updateField('temperature', parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 mt-1.5"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Max Output Tokens
                </label>
                <input
                  type="number"
                  min="200"
                  max="16384"
                  step="256"
                  value={config.max_tokens}
                  onChange={(e) => updateField('max_tokens', parseInt(e.target.value || '2048', 10))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1">
                Custom API Base URL (Optional)
              </label>
              <input
                type="text"
                value={config.base_url}
                onChange={(e) => updateField('base_url', e.target.value)}
                placeholder="Leave blank for official provider endpoint"
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300 outline-none"
              />
            </div>
              </>
            )}
          </>
        )}

        {activeTab === 'game' && (
          <>
            <div className="grid grid-cols-1 gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Browser Game URL
                </label>
                <input
                  type="url"
                  value={config.url}
                  onChange={(e) => updateField('url', e.target.value)}
                  placeholder="https://..."
                  className="w-full bg-slate-950 border border-slate-800 focus:border-sky-500 rounded-lg px-3 py-2 text-xs font-mono text-sky-300 outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Game Name</label>
                <input
                  type="text"
                  value={config.game_name}
                  onChange={(e) => updateField('game_name', e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-sm text-white outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Main Objective
                </label>
                <textarea
                  rows={2}
                  value={config.objective}
                  onChange={(e) => updateField('objective', e.target.value)}
                  placeholder="What is the primary goal?"
                  className="w-full bg-slate-950 border border-slate-800 focus:border-sky-500 rounded-lg p-2.5 text-xs text-slate-200 outline-none resize-y"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Rules &amp; Game Mechanics
                </label>
                <textarea
                  rows={4}
                  value={config.rules}
                  onChange={(e) => updateField('rules', e.target.value)}
                  placeholder="Describe how the game works, obstacles, scoring..."
                  className="w-full bg-slate-950 border border-slate-800 focus:border-sky-500 rounded-lg p-2.5 text-xs text-slate-200 outline-none resize-y"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">
                    Keyboard Controls
                  </label>
                  <textarea
                    rows={2}
                    value={config.keyboard_controls}
                    onChange={(e) => updateField('keyboard_controls', e.target.value)}
                    placeholder="Arrow keys, WASD, Space..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">
                    Mouse Controls
                  </label>
                  <textarea
                    rows={2}
                    value={config.mouse_controls}
                    onChange={(e) => updateField('mouse_controls', e.target.value)}
                    placeholder="Left-click buttons, drag items..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 outline-none"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Winning Conditions
                  </label>
                  <input
                    type="text"
                    value={config.winning_conditions}
                    onChange={(e) => updateField('winning_conditions', e.target.value)}
                    placeholder="Victory screen / target score"
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Losing Conditions
                  </label>
                  <input
                    type="text"
                    value={config.losing_conditions}
                    onChange={(e) => updateField('losing_conditions', e.target.value)}
                    placeholder="Game Over / 0 HP"
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1">
                  Secondary Objectives &amp; Strategy Hints (Optional)
                </label>
                <textarea
                  rows={2}
                  value={config.additional_instructions}
                  onChange={(e) => updateField('additional_instructions', e.target.value)}
                  placeholder="Optional custom tactics or tips..."
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 outline-none"
                />
              </div>

              {/* Save Game Profile */}
              <div className="pt-2 border-t border-slate-800 flex gap-2">
                <input
                  type="text"
                  value={profileName}
                  onChange={(e) => setProfileName(e.target.value)}
                  placeholder="Save rules as profile..."
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none"
                />
                <button
                  type="button"
                  onClick={handleCreateProfile}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 flex items-center gap-1.5 border border-slate-700 cursor-pointer"
                >
                  <BookmarkPlus size={13} />
                  <span>Save</span>
                </button>
              </div>
            </div>
          </>
        )}

        {activeTab === 'agent' && (
          <>
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Playstyle / Strategy Preference
              </label>
              <select
                value={config.strategy_preference}
                onChange={(e) => updateField('strategy_preference', e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white outline-none"
              >
                <option value="balanced">Balanced (Default)</option>
                <option value="cautious">Cautious &amp; Deliberate</option>
                <option value="aggressive">Aggressive / High Tempo</option>
                <option value="exploratory">Exploratory (Learn Mechanics)</option>
                <option value="speedrun">Speedrun (Fastest Path to Goal)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Action Duration (ms)
                </label>
                <input
                  type="number"
                  min="20"
                  max="5000"
                  step="20"
                  value={config.default_action_duration_ms}
                  onChange={(e) =>
                    updateField('default_action_duration_ms', parseInt(e.target.value || '150', 10))
                  }
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Loop Delay (ms)
                </label>
                <input
                  type="number"
                  min="100"
                  max="10000"
                  step="100"
                  value={config.observation_interval_ms}
                  onChange={(e) =>
                    updateField('observation_interval_ms', parseInt(e.target.value || '600', 10))
                  }
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Max Iterations
                </label>
                <input
                  type="number"
                  min="1"
                  max="2000"
                  value={config.max_iterations}
                  onChange={(e) =>
                    updateField('max_iterations', parseInt(e.target.value || '100', 10))
                  }
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Max Budget ($ USD)
                </label>
                <input
                  type="number"
                  min="0.05"
                  max="100"
                  step="0.25"
                  value={config.max_budget_usd}
                  onChange={(e) =>
                    updateField('max_budget_usd', parseFloat(e.target.value || '2.0'))
                  }
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Max Duration (sec)
                </label>
                <input
                  type="number"
                  min="30"
                  max="7200"
                  step="30"
                  value={config.max_session_duration_sec}
                  onChange={(e) =>
                    updateField('max_session_duration_sec', parseInt(e.target.value || '900', 10))
                  }
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Re-plan Every N Steps
                </label>
                <input
                  type="number"
                  min="2"
                  max="50"
                  value={config.strategic_planning_interval}
                  onChange={(e) =>
                    updateField('strategic_planning_interval', parseInt(e.target.value || '8', 10))
                  }
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-sm text-white outline-none"
                />
              </div>
            </div>

            <div className="pt-1 space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Hybrid Visual Grounding Pipeline
              </div>

              <label className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800 cursor-pointer">
                <div>
                  <div className="text-xs font-semibold text-slate-200">
                    Vision LLM Canvas Grounding
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Detects Canvas/WebGL board tiles, rings &amp; clickable game objects
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.enable_vision_grounding ?? true}
                  onChange={(e) => updateField('enable_vision_grounding', e.target.checked)}
                  className="w-4 h-4 accent-violet-500"
                />
              </label>

              <label className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800 cursor-pointer">
                <div>
                  <div className="text-xs font-semibold text-slate-200">
                    Computer Vision Hotspot Detector
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Extracts pixel-exact contours &amp; glowing markers via CV salience
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.enable_cv_detection ?? true}
                  onChange={(e) => updateField('enable_cv_detection', e.target.checked)}
                  className="w-4 h-4 accent-amber-500"
                />
              </label>

              <label className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800 cursor-pointer">
                <div>
                  <div className="text-xs font-semibold text-slate-200">
                    4× Magnifier Zoom-Snap Verification
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Auto-verifies &amp; snaps small Canvas clicks within ±3px
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.precision_zoom_click ?? true}
                  onChange={(e) => updateField('precision_zoom_click', e.target.checked)}
                  className="w-4 h-4 accent-indigo-500"
                />
              </label>

              <label className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800 cursor-pointer">
                <div>
                  <div className="text-xs font-semibold text-slate-200">
                    16×9 Reference Grid &amp; Target Overlays
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Annotates AI screenshots with target IDs and A1–P9 cell codes
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.show_coordinate_grid_to_llm}
                  onChange={(e) => updateField('show_coordinate_grid_to_llm', e.target.checked)}
                  className="w-4 h-4 accent-emerald-500"
                />
              </label>

              <label className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800 cursor-pointer">
                <div>
                  <div className="text-xs font-semibold text-slate-200">
                    Brass RL Critic Advisor Hints
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Injects trained PyTorch top-3 legal move probabilities into LLM prompts
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.enable_rl_advisor ?? true}
                  onChange={(e) => updateField('enable_rl_advisor', e.target.checked)}
                  className="w-4 h-4 accent-sky-500"
                />
              </label>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-300 mb-1">
                  <span>Min Target Confidence Threshold</span>
                  <span className="font-mono text-emerald-300">
                    {Math.round((config.min_target_confidence ?? 0.45) * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="0.9"
                  step="0.05"
                  value={config.min_target_confidence ?? 0.45}
                  onChange={(e) => updateField('min_target_confidence', parseFloat(e.target.value))}
                  className="w-full accent-emerald-500"
                />
              </div>
            </div>
          </>
        )}
      </div>

      {/* Bottom Launch Bar */}
      <div className="p-3.5 bg-slate-950/80 border-t border-slate-800">
        <button
          type="button"
          onClick={onStart}
          disabled={isRunning}
          className="w-full py-2.5 px-4 rounded-xl font-bold text-sm bg-gradient-to-r from-indigo-600 via-sky-600 to-emerald-600 hover:from-indigo-500 hover:via-sky-500 hover:to-emerald-500 disabled:opacity-40 text-white shadow-lg shadow-indigo-600/20 flex items-center justify-center gap-2 transition-all cursor-pointer disabled:cursor-not-allowed"
        >
          <Play size={16} fill="currentColor" />
          <span>{isRunning ? 'Autonomous Agent Active' : 'Launch Autonomous Player'}</span>
        </button>
      </div>
    </aside>
  );
}
