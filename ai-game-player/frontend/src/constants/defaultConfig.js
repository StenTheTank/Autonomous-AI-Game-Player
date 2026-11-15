export const API_BASE = 'http://localhost:8000';
export const WS_URL = 'ws://localhost:8000/ws';

export const DEFAULT_PROVIDER_MODELS = {
  openai: ['gpt-4o', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4o-mini', 'o4-mini'],
  anthropic: [
    'claude-sonnet-4-20250514',
    'claude-3-7-sonnet-latest',
    'claude-3-5-sonnet-latest',
    'claude-3-5-haiku-latest',
  ],
  gemini: ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3.8-flash', 'gemini-3.8-pro'],
  brass_rl: ['brass-ppo-best', 'brass-ppo-latest'],
};

export const DEFAULT_CONFIG = {
  // LLM settings
  provider: 'openai',
  api_key: '',
  model: 'gpt-4o',
  temperature: 0.2,
  max_tokens: 2048,
  base_url: '',

  // 2P Duel & Human-vs-AI settings
  match_mode: 'single',
  opponent_model: 'brass-ppo-best',
  api_player_seat: 0,
  human_player_seat: 0,
  game_seed: 202611,

  // Game settings (defaults to built-in local Crystal Quest demo)
  game_name: 'Crystal Quest (Built-in Demo)',
  url: 'http://localhost:8000/demo/crystal-quest',
  game_description: 'Turn-based tactical grid arena built for testing vision + keyboard/mouse navigation.',
  rules:
    "You control the cyan circle labeled PLAYER on an 8x5 grid. First click 'START MISSION' or press Space to dismiss the title overlay. Gray squares labeled WALL block movement. Red circles labeled MINE deal 1 shield damage. Gold diamonds labeled GEM are crystals. Once all 3 crystals are collected, the PORTAL on the right turns bright green.",
  keyboard_controls:
    'Space or Enter to start; ArrowUp, ArrowDown, ArrowLeft, ArrowRight (or W, A, S, D) to move one grid cell.',
  mouse_controls:
    "Click 'START MISSION' button to begin; click directional D-Pad buttons on the right or click an adjacent grid cell.",
  objective: 'Collect all 3 gold GEM crystals and then step onto the green PORTAL tile at (7,2).',
  secondary_objectives: 'Avoid stepping on red MINE circles to preserve all 3 Shield HP.',
  winning_conditions: "The 'VICTORY! MISSION COMPLETE' banner is displayed on screen.",
  losing_conditions: "Shield HP reaches 0 and 'MISSION FAILED' appears.",
  additional_instructions: '',

  // Agent & budget settings
  strategy_preference: 'cautious',
  default_action_duration_ms: 120,
  max_action_duration_ms: 4000,
  observation_interval_ms: 600,
  max_iterations: 80,
  max_session_duration_sec: 900,
  max_budget_usd: 2.0,
  strategic_planning_interval: 8,
  max_consecutive_failures: 5,
  show_coordinate_grid_to_llm: true,
  precision_zoom_click: true,
  enable_vision_grounding: true,
  enable_cv_detection: true,
  enable_rl_advisor: true,
  min_target_confidence: 0.45,
};

export const STATUS_STYLES = {
  idle: {
    badge: 'bg-slate-800 text-slate-300 border-slate-700',
    dot: 'bg-slate-400',
    label: 'Idle',
  },
  starting: {
    badge: 'bg-sky-500/15 text-sky-300 border-sky-500/35',
    dot: 'bg-sky-400 animate-pulse',
    label: 'Launching',
  },
  planning: {
    badge: 'bg-violet-500/15 text-violet-300 border-violet-500/35',
    dot: 'bg-violet-400 animate-pulse',
    label: 'Strategic Planning',
  },
  running: {
    badge: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/35',
    dot: 'bg-emerald-400 animate-ping',
    label: 'Playing Live',
  },
  paused: {
    badge: 'bg-amber-500/15 text-amber-300 border-amber-500/35',
    dot: 'bg-amber-400',
    label: 'Paused',
  },
  completed: {
    badge: 'bg-teal-500/15 text-teal-300 border-teal-500/35',
    dot: 'bg-teal-400',
    label: 'Victory / Completed',
  },
  failed: {
    badge: 'bg-rose-500/15 text-rose-300 border-rose-500/35',
    dot: 'bg-rose-400',
    label: 'Objective Failed',
  },
  stopped: {
    badge: 'bg-slate-700/60 text-slate-300 border-slate-600',
    dot: 'bg-slate-400',
    label: 'Stopped',
  },
  error: {
    badge: 'bg-red-500/15 text-red-300 border-red-500/40',
    dot: 'bg-red-500',
    label: 'Error',
  },
};

export function formatDuration(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  const hrs = Math.floor(mins / 60);
  if (hrs > 0) {
    return `${hrs}:${String(mins % 60).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

export function formatActionBadge(action) {
  if (!action) return 'none';
  const t = action.type;
  if (t === 'key_press' || t === 'key_hold' || t === 'key_release') {
    const dur = action.duration_ms ? ` · ${action.duration_ms}ms` : '';
    return `${t} [${action.key || '?'}]${dur}`;
  }
  if (t === 'key_combo') {
    return `combo [${(action.keys || []).join(' + ')}]`;
  }
  if (t === 'click_target' || t === 'hover_target') {
    const coords =
      action.x !== null && action.x !== undefined ? ` → (${action.x}, ${action.y})` : '';
    return `${t} [${action.target_id || '?'}]${coords}`;
  }
  if (t === 'click_coordinates') {
    return `click_coordinates (${action.x ?? '?'}, ${action.y ?? '?'})`;
  }
  if (t === 'mouse_click' || t === 'mouse_move' || t === 'mouse_down' || t === 'mouse_up') {
    if (action.target_id) {
      return `${t} [${action.target_id}] (${action.x ?? '?'}, ${action.y ?? '?'})`;
    }
    if (action.element_id !== null && action.element_id !== undefined) {
      return `${t} #${action.element_id} (${action.x ?? '?'}, ${action.y ?? '?'})`;
    }
    if (action.grid_cell) {
      return `${t} [${action.grid_cell}] → (${action.x ?? '?'}, ${action.y ?? '?'})`;
    }
    return `${t} (${action.x}, ${action.y})`;
  }
  if (t === 'mouse_drag') {
    return `drag (${action.from_x ?? '?'}, ${action.from_y ?? '?'}) → (${action.x}, ${action.y})`;
  }
  if (t === 'scroll') {
    return `scroll (${action.scroll_x || 0}, ${action.scroll_y || 0})`;
  }
  if (t === 'wait') {
    return `wait ${action.duration_ms || 1000}ms`;
  }
  return t;
}
