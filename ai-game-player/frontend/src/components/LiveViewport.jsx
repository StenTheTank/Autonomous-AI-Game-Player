import React, { useState, useRef } from 'react';
import {
  MonitorPlay,
  Crosshair,
  Grid,
  Send,
  Globe,
  Keyboard,
  MousePointerClick,
  Navigation,
  ScanEye,
  Layers,
  History,
  FileJson,
  Columns2,
  Rows2,
} from 'lucide-react';
import { formatActionBadge } from '../constants/defaultConfig';

const SOURCE_STYLES = {
  dom: {
    stroke: '#10b981',
    fill: 'rgba(16,185,129,0.14)',
    tagBg: '#064e3b',
    badge: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/35',
    label: 'DOM',
  },
  vision: {
    stroke: '#a855f7',
    fill: 'rgba(168,85,247,0.14)',
    tagBg: '#3b0764',
    badge: 'bg-purple-500/15 text-purple-300 border-purple-500/35',
    label: 'VISION',
  },
  cv: {
    stroke: '#f59e0b',
    fill: 'rgba(245,158,11,0.14)',
    tagBg: '#451a03',
    badge: 'bg-amber-500/15 text-amber-300 border-amber-500/35',
    label: 'CV',
  },
};

export default function LiveViewport({
  image,
  sessionState,
  onSendHint,
  onUserClick,
  onUserScroll,
  splitView = true,
  onToggleSplitView,
}) {
  const [showHud, setShowHud] = useState(true);
  const [showGrid, setShowGrid] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [debugFilter, setDebugFilter] = useState('all'); // 'all' | 'dom' | 'vision' | 'cv' | 'clicks' | 'json'
  const [highlightedTargetId, setHighlightedTargetId] = useState(null);
  const [hoverCoords, setHoverCoords] = useState(null);
  const [hintInput, setHintInput] = useState('');
  const stageRef = useRef(null);

  const s = sessionState || {};
  const lastAction = s.last_action;
  const isActive = ['starting', 'planning', 'running', 'paused'].includes(s.status);
  const awaitingHumanTurn = Boolean(s.duel_state?.awaiting_human && isActive);

  const interactionMap = s.interaction_map || { targets: [], screenshot_version: 0 };
  const allTargets = interactionMap.targets || [];
  const clickHistory = s.click_history || [];

  const filteredTargets =
    debugFilter === 'all' || debugFilter === 'clicks' || debugFilter === 'json'
      ? allTargets
      : allTargets.filter((t) => t.source === debugFilter);

  const domCount = allTargets.filter((t) => t.source === 'dom').length;
  const visionCount = allTargets.filter((t) => t.source === 'vision').length;
  const cvCount = allTargets.filter((t) => t.source === 'cv').length;

  // Persistent current AI cursor coordinates (always available, defaults to 640, 360)
  const mouseX = s.mouse_x ?? 640;
  const mouseY = s.mouse_y ?? 360;
  const cursorPctX = Math.min(100, Math.max(0, (mouseX / 1280) * 100));
  const cursorPctY = Math.min(100, Math.max(0, (mouseY / 720) * 100));

  const isClickAction =
    lastAction &&
    ['click_target', 'click_coordinates', 'mouse_click', 'mouse_down', 'mouse_drag'].includes(
      lastAction.type
    );

  const computeGameCoords = (e) => {
    const el = stageRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = Math.max(0, Math.min(1279, Math.round(((e.clientX - rect.left) / rect.width) * 1280)));
    const y = Math.max(0, Math.min(719, Math.round(((e.clientY - rect.top) / rect.height) * 720)));
    return { x, y };
  };

  const handleMouseMove = (e) => {
    const coords = computeGameCoords(e);
    if (coords) setHoverCoords(coords);
  };

  const handleStageClick = (e) => {
    const coords = computeGameCoords(e);
    if (!coords) return;
    if (awaitingHumanTurn && onUserClick) {
      onUserClick(coords.x, coords.y);
      return;
    }
    setHintInput((prev) =>
      prev.trim()
        ? `${prev.trim()} (x=${coords.x}, y=${coords.y})`
        : `Click at x=${coords.x}, y=${coords.y} — `
    );
  };

  const handleStageWheel = (e) => {
    if (!awaitingHumanTurn || !onUserScroll) return;
    const coords = computeGameCoords(e);
    if (!coords) return;
    onUserScroll(coords.x, coords.y, Math.round(e.deltaY));
  };

  const handleSelectTargetForHint = (t) => {
    setHighlightedTargetId(t.id);
    setHintInput(`Use click_target on ${t.id} ("${t.label}" at ${t.center[0]}, ${t.center[1]})`);
  };

  const handleHintSubmit = (e) => {
    e.preventDefault();
    if (!hintInput.trim()) return;
    onSendHint(hintInput.trim());
    setHintInput('');
  };

  return (
    <section className="bg-slate-900/80 border border-slate-800/90 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
      {/* Viewport Header */}
      <div className="px-4 py-2.5 bg-slate-950/70 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5 min-w-0">
          <MonitorPlay size={17} className="text-sky-400 shrink-0" />
          <h2 className="text-sm font-semibold text-white shrink-0">Live Browser Arena</h2>
          {s.current_url && (
            <span className="hidden xl:inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-400 truncate max-w-[210px]">
              <Globe size={11} className="text-slate-500 shrink-0" />
              {s.current_url}
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Always-visible AI Mouse Position Readout */}
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-mono font-semibold bg-rose-500/15 border border-rose-500/35 text-rose-200">
            <Navigation size={12} className="text-rose-400" />
            AI Cursor: ({mouseX}, {mouseY})
          </span>

          {/* User Hover Coordinate Inspector */}
          {hoverCoords && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-mono bg-amber-500/15 border border-amber-500/35 text-amber-200">
              Inspect: ({hoverCoords.x}, {hoverCoords.y})
            </span>
          )}

          {/* Visual Grounding Debug Mode Toggle */}
          <button
            type="button"
            onClick={() => setDebugMode((v) => !v)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold border transition-colors cursor-pointer ${
              debugMode
                ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-200 shadow-sm'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700'
            }`}
            title="Toggle Visual Grounding Debug Overlay & Interaction Map Inspector"
          >
            <ScanEye size={13} className={debugMode ? 'text-emerald-400' : 'text-slate-400'} />
            <span>Visual Debug</span>
            <span className="px-1.5 py-0.2 rounded bg-slate-950/80 text-[10px] font-mono text-emerald-300">
              {allTargets.length}
            </span>
          </button>

          <button
            type="button"
            onClick={() => setShowGrid((v) => !v)}
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
              showGrid
                ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
                : 'bg-slate-900 border-slate-800 text-slate-400'
            }`}
            title="Toggle 16x9 A1..P9 macro cell grid"
          >
            <Grid size={13} />
            <span className="hidden sm:inline">Grid</span>
          </button>

          <button
            type="button"
            onClick={() => setShowHud((v) => !v)}
            className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
              showHud
                ? 'bg-indigo-500/15 border-indigo-500/40 text-indigo-300'
                : 'bg-slate-900 border-slate-800 text-slate-400'
            }`}
          >
            <Crosshair size={13} />
            <span>Cursor HUD</span>
          </button>

          {onToggleSplitView && (
            <button
              type="button"
              onClick={onToggleSplitView}
              className={`hidden xl:inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
                splitView
                  ? 'bg-sky-500/15 border-sky-500/40 text-sky-300'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
              title={
                splitView
                  ? 'Switch to Wide Stacked View'
                  : 'Switch to Side-by-Side Split Monitor View'
              }
            >
              {splitView ? <Columns2 size={13} /> : <Rows2 size={13} />}
              <span>{splitView ? 'Split View' : 'Wide View'}</span>
            </button>
          )}
        </div>
      </div>

      {awaitingHumanTurn && (
        <div className="px-4 py-2 bg-emerald-950/75 border-b border-emerald-500/45 text-xs font-semibold text-emerald-200 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <MousePointerClick size={15} className="text-emerald-300 animate-bounce shrink-0" />
            <span>
              YOUR TURN — Click directly on Action buttons, Hand cards, Cities/Links, and Options on the board below (or in the Chromium window)!
            </span>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-400/40 text-emerald-100 shrink-0">
            Interactive Mode
          </span>
        </div>
      )}

      {/* 16:9 Screen Stage (exact 1280x720 aspect ratio) */}
      <div
        ref={stageRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverCoords(null)}
        onClick={image ? handleStageClick : undefined}
        onWheel={image && awaitingHumanTurn ? handleStageWheel : undefined}
        title={
          awaitingHumanTurn
            ? 'Your turn! Click directly on Brass UI controls or board cities/links'
            : image
            ? 'Click anywhere on preview to insert exact (x, y) coordinates into Guide AI'
            : ''
        }
        className={`relative w-full aspect-video bg-slate-950 flex items-center justify-center overflow-hidden ${
          awaitingHumanTurn ? 'cursor-pointer ring-2 ring-inset ring-emerald-400/60' : image ? 'cursor-crosshair' : ''
        }`}
      >
        {image ? (
          <>
            <img
              src={`data:image/jpeg;base64,${image}`}
              alt="Live Browser Game Observation"
              className="w-full h-full object-fill select-none"
            />

            {/* Optional 16x9 Named Macro Cell Grid Preview (A1..P9, 80x80px) */}
            {showGrid && (
              <svg
                viewBox="0 0 1280 720"
                className="pointer-events-none absolute inset-0 w-full h-full"
              >
                {Array.from({ length: 15 }, (_, i) => (i + 1) * 80).map((gx) => (
                  <line
                    key={`x-${gx}`}
                    x1={gx}
                    y1={0}
                    x2={gx}
                    y2={720}
                    stroke="rgba(56,189,248,0.32)"
                    strokeWidth="1"
                  />
                ))}
                {Array.from({ length: 8 }, (_, i) => (i + 1) * 80).map((gy) => (
                  <line
                    key={`y-${gy}`}
                    x1={0}
                    y1={gy}
                    x2={1280}
                    y2={gy}
                    stroke="rgba(56,189,248,0.32)"
                    strokeWidth="1"
                  />
                ))}
                {Array.from({ length: 9 }, (_, r) =>
                  Array.from({ length: 16 }, (_, c) => {
                    const code = `${String.fromCharCode(65 + c)}${r + 1}`;
                    return (
                      <text
                        key={code}
                        x={c * 80 + 4}
                        y={r * 80 + 12}
                        fill="rgba(125,249,255,0.85)"
                        fontSize="10"
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        {code}
                      </text>
                    );
                  })
                )}
              </svg>
            )}

            {/* Visual Grounding Debug Overlay: Bounding Boxes, Target IDs, Confidence & Click History */}
            {debugMode && (
              <svg
                viewBox="0 0 1280 720"
                className="pointer-events-none absolute inset-0 w-full h-full"
              >
                {/* 1. Detected Targets Bounding Boxes & IDs */}
                {filteredTargets.map((t) => {
                  const [bx, by, bw, bh] = t.bbox || [0, 0, 24, 24];
                  const [cx, cy] = t.center || [bx + bw / 2, by + bh / 2];
                  const style = SOURCE_STYLES[t.source] || SOURCE_STYLES.dom;
                  const isHighlighted = highlightedTargetId === t.id;
                  const confPct = Math.round((t.confidence ?? 1) * 100);
                  const badgeText = `${t.id.replace('target_', '#')} · ${confPct}%`;
                  const labelW = Math.max(66, badgeText.length * 6.5 + 8);
                  const labelX = Math.max(2, Math.min(1278 - labelW, bx));
                  const labelY = by >= 18 ? by - 16 : Math.min(702, by + bh + 2);

                  return (
                    <g key={t.id}>
                      <rect
                        x={bx}
                        y={by}
                        width={bw}
                        height={bh}
                        fill={isHighlighted ? 'rgba(56,189,248,0.28)' : style.fill}
                        stroke={isHighlighted ? '#38bdf8' : style.stroke}
                        strokeWidth={isHighlighted ? '3' : '1.75'}
                        rx="3"
                      />
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isHighlighted ? '4.5' : '2.8'}
                        fill={isHighlighted ? '#38bdf8' : style.stroke}
                      />
                      <rect
                        x={labelX}
                        y={labelY}
                        width={labelW}
                        height="15"
                        rx="3"
                        fill={style.tagBg}
                        stroke={isHighlighted ? '#38bdf8' : style.stroke}
                        strokeWidth="1"
                      />
                      <text
                        x={labelX + 4}
                        y={labelY + 11}
                        fill="#ffffff"
                        fontSize="10"
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        {badgeText}
                      </text>
                    </g>
                  );
                })}

                {/* 2. Previous Clicks History Trail */}
                {clickHistory.map((clk, i) => {
                  const prev = i > 0 ? clickHistory[i - 1] : null;
                  const effective = clk.visual_delta_pct === null || clk.visual_delta_pct >= 0.35;
                  const ringColor = effective ? '#34d399' : '#fbbf24';
                  return (
                    <g key={`clk-${clk.iteration}-${i}`}>
                      {prev && (
                        <line
                          x1={prev.x}
                          y1={prev.y}
                          x2={clk.x}
                          y2={clk.y}
                          stroke="rgba(244,63,94,0.45)"
                          strokeWidth="1.5"
                          strokeDasharray="4 4"
                        />
                      )}
                      <circle
                        cx={clk.x}
                        cy={clk.y}
                        r="9"
                        fill="rgba(2,6,23,0.78)"
                        stroke={ringColor}
                        strokeWidth="2"
                      />
                      <text
                        x={clk.x}
                        y={clk.y + 3.5}
                        textAnchor="middle"
                        fill={ringColor}
                        fontSize="9.5"
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        {clk.iteration}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}

            {/* Persistent Live AI Cursor Crosshair (clean & unobstructed — no text banners over the game!) */}
            {showHud && (
              <div className="pointer-events-none absolute inset-0">
                <div
                  style={{ left: `${cursorPctX}%` }}
                  className="absolute top-0 bottom-0 w-[1px] bg-rose-500/30"
                />
                <div
                  style={{ top: `${cursorPctY}%` }}
                  className="absolute left-0 right-0 h-[1px] bg-rose-500/30"
                />

                <div
                  style={{ left: `${cursorPctX}%`, top: `${cursorPctY}%` }}
                  className="absolute"
                >
                  {isClickAction && (
                    <div className="w-7 h-7 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-rose-400 animate-target-ping" />
                  )}
                  <div className="w-3.5 h-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-rose-500/25 border-2 border-rose-400 flex items-center justify-center shadow">
                    <div className="w-1 h-1 rounded-full bg-white" />
                  </div>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center px-6 py-12 max-w-md">
            <div className="w-14 h-14 mx-auto mb-3 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center">
              <MonitorPlay size={28} className="text-slate-600" />
            </div>
            <h3 className="text-sm font-semibold text-slate-300">
              Dedicated Chromium Window Standby
            </h3>
            <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
              Enter your API key, pick a game preset or paste any browser game URL, and click{' '}
              <span className="text-emerald-400 font-semibold">Start</span>. Enable{' '}
              <span className="text-emerald-300 font-semibold">Visual Debug</span> at any time to
              inspect the live Interaction Map, bounding boxes, confidence scores, and click trail.
            </p>
          </div>
        )}
      </div>

      {/* Non-blocking Telemetry Strip BELOW the 16:9 game screen (never covers gameplay!) */}
      {(lastAction || s.status_reason) && (
        <div className="px-3.5 py-1.5 bg-slate-950/90 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-[11px]">
          <div className="flex items-center gap-2 min-w-0">
            {lastAction && (
              <>
                {isClickAction ? (
                  <MousePointerClick size={13} className="text-rose-400 shrink-0" />
                ) : (
                  <Keyboard size={13} className="text-emerald-400 shrink-0" />
                )}
                <span className="font-mono font-semibold text-slate-200 shrink-0">
                  {formatActionBadge(lastAction)}
                </span>
                {s.last_action_status && (
                  <span className="text-slate-400 truncate max-w-[420px]">
                    · {s.last_action_status}
                  </span>
                )}
              </>
            )}
          </div>
          {s.status_reason && (
            <span className="text-slate-400 truncate max-w-[360px] ml-auto">
              {s.status_reason}
            </span>
          )}
        </div>
      )}

      {/* Collapsible Visual Debugging & Interaction Map Inspector */}
      {debugMode && (
        <div className="bg-slate-950/90 border-t border-slate-800 p-3 space-y-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="inline-flex items-center gap-1 font-semibold text-emerald-300">
                <Layers size={13} />
                Interaction Map v{interactionMap.screenshot_version || 0}
              </span>
              {interactionMap.regeneration_reason && (
                <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 font-mono text-slate-400">
                  {interactionMap.regeneration_reason}
                </span>
              )}
              <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 font-mono text-slate-400">
                DPR {interactionMap.device_pixel_ratio || 1} · Canvas:{' '}
                {interactionMap.has_canvas_or_webgl ? 'Yes' : 'No'}
              </span>
            </div>

            {/* Filter & View Switcher */}
            <div className="flex flex-wrap items-center gap-1 text-[11px]">
              {[
                { id: 'all', label: `All (${allTargets.length})` },
                { id: 'dom', label: `DOM (${domCount})` },
                { id: 'vision', label: `Vision (${visionCount})` },
                { id: 'cv', label: `CV (${cvCount})` },
                { id: 'clicks', label: `Clicks (${clickHistory.length})`, icon: History },
                { id: 'json', label: 'Map JSON', icon: FileJson },
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setDebugFilter(tab.id)}
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border font-medium cursor-pointer transition-colors ${
                      debugFilter === tab.id
                        ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-200'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {Icon && <Icon size={11} />}
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {debugFilter === 'json' ? (
            <pre className="max-h-48 overflow-auto p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300">
              {JSON.stringify(interactionMap, null, 2)}
            </pre>
          ) : debugFilter === 'clicks' ? (
            <div className="max-h-44 overflow-y-auto space-y-1">
              {clickHistory.length === 0 ? (
                <div className="text-xs text-slate-500 py-3 text-center">
                  No clicks executed in this session yet.
                </div>
              ) : (
                [...clickHistory].reverse().map((clk, idx) => (
                  <div
                    key={idx}
                    className="flex flex-wrap items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs"
                  >
                    <div className="flex items-center gap-2">
                      <span className="px-1.5 py-0.5 rounded bg-slate-800 font-mono font-bold text-slate-200">
                        #{clk.iteration}
                      </span>
                      <span className="font-mono text-rose-300">
                        ({clk.x}, {clk.y})
                      </span>
                      {clk.target_id && (
                        <span className="px-1.5 py-0.5 rounded bg-indigo-500/15 border border-indigo-500/30 font-mono text-[11px] text-indigo-300">
                          {clk.target_id}
                        </span>
                      )}
                      <span className="text-slate-300 truncate max-w-[240px]">{clk.label}</span>
                    </div>
                    <div className="flex items-center gap-2 text-[11px] font-mono">
                      <span className="text-slate-400">{clk.source}</span>
                      {clk.visual_delta_pct !== null && clk.visual_delta_pct !== undefined && (
                        <span
                          className={
                            clk.visual_delta_pct >= 0.35 ? 'text-emerald-400' : 'text-amber-400'
                          }
                        >
                          Δ {clk.visual_delta_pct}%
                        </span>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          ) : (
            <div className="max-h-48 overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-1.5 pr-1">
              {filteredTargets.length === 0 ? (
                <div className="col-span-2 text-xs text-slate-500 py-3 text-center">
                  No detected targets for this filter on the current frame.
                </div>
              ) : (
                filteredTargets.map((t) => {
                  const st = SOURCE_STYLES[t.source] || SOURCE_STYLES.dom;
                  const confPct = Math.round((t.confidence ?? 1) * 100);
                  return (
                    <button
                      key={t.id}
                      type="button"
                      onMouseEnter={() => setHighlightedTargetId(t.id)}
                      onMouseLeave={() => setHighlightedTargetId(null)}
                      onClick={() => handleSelectTargetForHint(t)}
                      className={`text-left px-2.5 py-1.5 rounded-lg border transition-colors flex items-center justify-between gap-2 cursor-pointer ${
                        highlightedTargetId === t.id
                          ? 'bg-sky-500/15 border-sky-500/50'
                          : 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                      }`}
                      title="Hover to highlight bounding box; click to populate Guide AI with this target_id"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className="font-mono font-bold text-xs text-white">{t.id}</span>
                          <span
                            className={`px-1.5 py-0.2 rounded text-[10px] font-semibold border ${st.badge}`}
                          >
                            {st.label}
                          </span>
                          <span className="text-[11px] text-slate-300 truncate">{t.label}</span>
                        </div>
                        <div className="text-[10px] font-mono text-slate-500 mt-0.5">
                          center: [{t.center.join(', ')}] · bbox: [{t.bbox.join(', ')}]
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <div className="text-[11px] font-mono font-semibold text-emerald-300">
                          {confPct}%
                        </div>
                        <div className="w-12 h-1.5 bg-slate-800 rounded-full overflow-hidden mt-0.5">
                          <div
                            style={{ width: `${confPct}%` }}
                            className="h-full bg-emerald-400 rounded-full"
                          />
                        </div>
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          )}
        </div>
      )}

      {/* Live Operator Guidance Input */}
      <form
        onSubmit={handleHintSubmit}
        className="px-3.5 py-2.5 bg-slate-950/75 border-t border-slate-800/80 flex items-center gap-2"
      >
        <input
          type="text"
          value={hintInput}
          onChange={(e) => setHintInput(e.target.value)}
          disabled={!isActive}
          placeholder={
            isActive
              ? 'Tip: Click Visual Debug to inspect targets by ID, click the screen for (x, y), or type guidance...'
              : 'Start a session to inspect the Interaction Map or send live tactical guidance...'
          }
          className="flex-1 bg-slate-900 border border-slate-800 focus:border-indigo-500 disabled:opacity-50 rounded-lg px-3 py-1.5 text-xs text-slate-200 outline-none"
        />
        <button
          type="submit"
          disabled={!isActive || !hintInput.trim()}
          className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-xs font-semibold text-white flex items-center gap-1.5 transition-colors cursor-pointer disabled:cursor-not-allowed"
        >
          <Send size={13} />
          <span>Guide AI</span>
        </button>
      </form>
    </section>
  );
}
