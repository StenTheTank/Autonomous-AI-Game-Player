import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AlertCircle, X } from 'lucide-react';
import {
  API_BASE,
  WS_URL,
  DEFAULT_CONFIG,
  DEFAULT_PROVIDER_MODELS,
} from './constants/defaultConfig';
import HeaderBar from './components/HeaderBar';
import StatsStrip from './components/StatsStrip';
import ConfigSidebar from './components/ConfigSidebar';
import LiveViewport from './components/LiveViewport';
import AgentActivityPanel from './components/AgentActivityPanel';
import DuelScoreboardBanner from './components/DuelScoreboardBanner';
import HistoryModal from './components/HistoryModal';
import RLTrainerModal from './components/RLTrainerModal';

export default function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [presets, setPresets] = useState([]);
  const [providerModels, setProviderModels] = useState(DEFAULT_PROVIDER_MODELS);
  const [savedProfiles, setSavedProfiles] = useState([]);

  const [wsConnected, setWsConnected] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [splitView, setSplitView] = useState(true);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [rlTrainerOpen, setRlTrainerOpen] = useState(false);
  const [rlStatus, setRlStatus] = useState(null);

  const [sessionState, setSessionState] = useState({
    status: 'idle',
    browser_status: 'idle',
    stats: {
      current_iteration: 0,
      max_iterations: DEFAULT_CONFIG.max_iterations,
      total_actions: 0,
      elapsed_seconds: 0,
      api_calls: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      estimated_cost_usd: 0,
      max_budget_usd: DEFAULT_CONFIG.max_budget_usd,
      consecutive_failures: 0,
      last_visual_delta_pct: 0,
      progress_percent: 0,
    },
  });

  const [displayElapsed, setDisplayElapsed] = useState(0);
  const [image, setImage] = useState(null);
  const [logs, setLogs] = useState([]);
  const [bannerError, setBannerError] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const status = sessionState.status || 'idle';
  const isRunning = ['starting', 'planning', 'running', 'paused'].includes(status);

  // Fetch presets, saved profiles & RL status on mount
  const refreshRLStatus = useCallback(async (directData = null) => {
    if (directData && typeof directData === 'object' && 'running' in directData) {
      setRlStatus(directData);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/api/rl/status`);
      if (res.ok) {
        setRlStatus(await res.json());
      }
    } catch {
      // Backend may still be starting
    }
  }, []);

  const loadMetadata = useCallback(async () => {
    try {
      const [presRes, profRes, rlRes] = await Promise.all([
        fetch(`${API_BASE}/api/presets`),
        fetch(`${API_BASE}/api/profiles`),
        fetch(`${API_BASE}/api/rl/status`),
      ]);
      if (presRes.ok) {
        const d = await presRes.json();
        if (d.presets) setPresets(d.presets);
        if (d.providers) setProviderModels(d.providers);
      }
      if (profRes.ok) {
        const p = await profRes.json();
        setSavedProfiles(p.profiles || []);
      }
      if (rlRes.ok) {
        setRlStatus(await rlRes.json());
      }
    } catch {
      // Backend may still be starting up; WebSocket reconnect will pick up
    }
  }, []);

  useEffect(() => {
    loadMetadata();
  }, [loadMetadata]);

  // Resilient WebSocket connection with automatic reconnect
  useEffect(() => {
    let unmounted = false;

    function connect() {
      if (unmounted) return;
      const socket = new WebSocket(WS_URL);
      wsRef.current = socket;

      socket.onopen = () => {
        if (unmounted) return;
        setWsConnected(true);
        loadMetadata();
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'state_update') {
            if (data.state) {
              setSessionState(data.state);
              if (data.state.stats?.elapsed_seconds !== undefined) {
                setDisplayElapsed(data.state.stats.elapsed_seconds);
              }
            }
            if (data.image) {
              setImage(data.image);
            }
            if (data.strategic_plan) {
              setLogs((prev) => [
                ...prev,
                {
                  kind: 'plan',
                  message: data.strategic_plan.current_strategy,
                },
              ]);
            }
            if (data.llm_response) {
              setLogs((prev) => [...prev, { kind: 'step', data: data.llm_response }]);
            }
            if (data.event_note) {
              setLogs((prev) => [...prev, { kind: 'note', message: data.event_note }]);
            }
            if (data.error) {
              setLogs((prev) => [...prev, { kind: 'error', message: data.error }]);
            }
          } else if (data.type === 'rl_training_update') {
            if (data.status) {
              setRlStatus(data.status);
            }
          } else if (data.type === 'command_error' || data.error) {
            setBannerError(data.error);
          }
        } catch (err) {
          console.error('WS parse error:', err);
        }
      };

      socket.onclose = () => {
        if (unmounted) return;
        setWsConnected(false);
        reconnectTimerRef.current = setTimeout(connect, 2000);
      };

      socket.onerror = () => {
        socket.close();
      };
    }

    connect();
    return () => {
      unmounted = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [loadMetadata]);

  // Smooth local elapsed ticker when actively running
  useEffect(() => {
    if (!['starting', 'planning', 'running'].includes(status)) return;
    const id = setInterval(() => {
      setDisplayElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(id);
  }, [status]);

  const sendWsCommand = (payload) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    } else {
      setBannerError('Backend WebSocket is not connected. Ensure the Python backend is running on port 8000.');
    }
  };

  const buildPayloadConfig = () => ({
    ...config,
    base_url: config.base_url?.trim() || null,
  });

  const handleStart = (overrideCfg = null) => {
    const activeCfg = overrideCfg && !overrideCfg.nativeEvent ? overrideCfg : config;
    setBannerError(null);
    if (activeCfg.provider !== 'brass_rl' && (!activeCfg.api_key || !activeCfg.api_key.trim())) {
      setBannerError('Please enter your LLM API key in the "1. LLM API" configuration tab before starting.');
      setSidebarOpen(true);
      return;
    }
    if (!activeCfg.url || !activeCfg.url.trim()) {
      setBannerError('Please enter a valid http:// or https:// Game URL.');
      return;
    }
    setLogs([]);
    setImage(null);
    setDisplayElapsed(0);
    if (splitView) setSidebarOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    sendWsCommand({
      command: 'start',
      config: {
        ...activeCfg,
        base_url: activeCfg.base_url?.trim() || null,
      },
    });
  };

  const handlePause = () => sendWsCommand({ command: 'pause' });
  const handleResume = () => sendWsCommand({ command: 'resume' });
  const handleStop = () => sendWsCommand({ command: 'stop' });

  const handleRestart = () => {
    setBannerError(null);
    if (config.provider !== 'brass_rl' && (!config.api_key || !config.api_key.trim())) {
      setBannerError('Please enter your LLM API key before restarting.');
      return;
    }
    setLogs([]);
    setImage(null);
    setDisplayElapsed(0);
    if (splitView) setSidebarOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    sendWsCommand({ command: 'restart', config: buildPayloadConfig() });
  };

  const handlePlayBestRLInArena = () => {
    const brassPreset = presets.find((p) => p.id === 'brass-birmingham') || {
      game_name: 'Brass: Birmingham (Solo vs Mautoma)',
      url: 'http://localhost:8000/brass-birmingham/',
      objective: 'Defeat Mautoma by scoring the highest total Victory Points across Canal and Rail Eras.',
      max_iterations: 200,
    };
    const nextCfg = {
      ...config,
      ...brassPreset,
      match_mode: 'single',
      api_key: config.api_key,
      provider: 'brass_rl',
      model: 'brass-ppo-best',
      temperature: 0.0,
    };
    setConfig(nextCfg);
    setRlTrainerOpen(false);
    handleStart(nextCfg);
  };

  const handleSendHint = (hintText) => {
    sendWsCommand({ command: 'hint', hint: hintText });
  };

  const handleUserClick = (x, y) => {
    sendWsCommand({ command: 'user_click', x, y });
  };

  const handleUserScroll = (x, y, deltaY) => {
    sendWsCommand({ command: 'user_scroll', x, y, delta_y: deltaY });
  };

  const handleSaveProfile = async (name) => {
    try {
      await fetch(`${API_BASE}/api/profiles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, config: buildPayloadConfig() }),
      });
      loadMetadata();
    } catch (err) {
      setBannerError(`Failed to save profile: ${err.message}`);
    }
  };

  const handleDeleteProfile = async (id) => {
    try {
      await fetch(`${API_BASE}/api/profiles/${id}`, { method: 'DELETE' });
      loadMetadata();
    } catch (err) {
      setBannerError(`Failed to delete profile: ${err.message}`);
    }
  };

  const handleLoadHistoryConfig = (loadedCfg) => {
    setConfig((prev) => ({
      ...prev,
      ...loadedCfg,
      api_key: prev.api_key,
    }));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <HeaderBar
        status={status}
        wsConnected={wsConnected}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
        onStart={handleStart}
        onPause={handlePause}
        onResume={handleResume}
        onStop={handleStop}
        onRestart={handleRestart}
        onOpenHistory={() => setHistoryOpen(true)}
        onOpenRLTrainer={() => setRlTrainerOpen(true)}
        rlTrainingRunning={!!rlStatus?.running}
      />

      {bannerError && (
        <div className="max-w-[1800px] w-full mx-auto px-4 lg:px-6 pt-3">
          <div className="bg-rose-950/70 border border-rose-700/80 text-rose-200 px-4 py-2.5 rounded-xl text-xs sm:text-sm flex items-center justify-between gap-3 shadow-lg">
            <div className="flex items-center gap-2">
              <AlertCircle size={16} className="text-rose-400 shrink-0" />
              <span>{bannerError}</span>
            </div>
            <button
              onClick={() => setBannerError(null)}
              className="text-rose-300 hover:text-white p-1 cursor-pointer"
            >
              <X size={15} />
            </button>
          </div>
        </div>
      )}

      <main className="max-w-[1800px] w-full mx-auto px-4 lg:px-6 py-4 flex-1 flex flex-col lg:flex-row gap-4">
        {sidebarOpen && (
          <ConfigSidebar
            config={config}
            setConfig={setConfig}
            presets={presets}
            providerModels={providerModels}
            savedProfiles={savedProfiles}
            onSaveProfile={handleSaveProfile}
            onDeleteProfile={handleDeleteProfile}
            onStart={handleStart}
            isRunning={isRunning}
            onOpenRLTrainer={() => setRlTrainerOpen(true)}
            rlStatus={rlStatus}
          />
        )}

        <div className="flex-1 min-w-0 flex flex-col gap-4">
          <StatsStrip
            stats={sessionState.stats}
            displayElapsed={displayElapsed}
            status={status}
          />

          <DuelScoreboardBanner duelState={sessionState.duel_state} />

          {splitView ? (
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-start">
              <div className="xl:col-span-8 xl:sticky xl:top-[72px]">
                <LiveViewport
                  image={image}
                  sessionState={sessionState}
                  onSendHint={handleSendHint}
                  onUserClick={handleUserClick}
                  onUserScroll={handleUserScroll}
                  splitView={splitView}
                  onToggleSplitView={() => setSplitView((v) => !v)}
                />
              </div>
              <div className="xl:col-span-4">
                <AgentActivityPanel
                  sessionState={sessionState}
                  logs={logs}
                  sideBySide={true}
                />
              </div>
            </div>
          ) : (
            <>
              <LiveViewport
                image={image}
                sessionState={sessionState}
                onSendHint={handleSendHint}
                onUserClick={handleUserClick}
                onUserScroll={handleUserScroll}
                splitView={splitView}
                onToggleSplitView={() => setSplitView((v) => !v)}
              />
              <AgentActivityPanel
                sessionState={sessionState}
                logs={logs}
                sideBySide={false}
              />
            </>
          )}
        </div>
      </main>

      <HistoryModal
        isOpen={historyOpen}
        onClose={() => setHistoryOpen(false)}
        onLoadConfig={handleLoadHistoryConfig}
      />

      <RLTrainerModal
        isOpen={rlTrainerOpen}
        onClose={() => setRlTrainerOpen(false)}
        rlStatus={rlStatus}
        onRefreshStatus={refreshRLStatus}
        onPlayBestInArena={handlePlayBestRLInArena}
        agentActive={isRunning}
      />
    </div>
  );
}
