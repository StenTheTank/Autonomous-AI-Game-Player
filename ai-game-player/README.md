# AI Game Player (v2.0)

An autonomous, general-purpose **Vision AI Game Player** that observes any browser-based game via live screenshots, formulates high-level strategies, and plays autonomously using structured keyboard and mouse actions in a dedicated, visible Chromium window.

---

## Key Capabilities

- **Modern Mission-Control Dashboard:** Sleek dark-theme React + Tailwind CSS v4 interface with live 16:9 game observation, interactive **Action HUD** overlay (showing click coordinates and key presses), real-time token/cost telemetry, and live tactical operator hints.
- **Provider-Independent Vision Layer:** Supports **OpenAI** (`gpt-4o`, `gpt-4.1`, `gpt-4o-mini`, `o4-mini`), **Anthropic** (`claude-sonnet-4-20250514`, `claude-3-7-sonnet-latest`, `claude-3-5-sonnet-latest`), and **Google Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`) with one-click API connection verification.
- **Two-Level Autonomous Intelligence:**
  1. **Strategic Planner:** Synthesizes high-level strategy, ordered sub-goals, confirmed mechanics, and lessons learned periodically or whenever gameplay stagnates.
  2. **Tactical Executor:** Analyzes `1280x720` screenshots (annotated with an optional pixel coordinate ruler grid for pinpoint mouse clicks), evaluates whether the previous action succeeded via objective frame-to-frame visual delta `%`, and selects the next structured action.
- **Rich Browser Input Protocol:** Supports `key_press`, `key_hold`, `key_release`, `key_combo` (simultaneous multi-key chords), `mouse_click` (single/double, left/right/middle), `mouse_move`, `mouse_down`, `mouse_up`, `mouse_drag` (interpolated canvas drags), `scroll`, `wait`, `request_observation`, `declare_success`, and `declare_failure`.
- **Full Session Lifecycle & Guardrails:** Start, **Pause**, **Resume**, **Emergency Stop**, and **Restart**, backed by configurable budget (`$ USD`), iteration, duration, action-length, and consecutive API failure limits, automatic JSON schema repair retries, CAPTCHA detection (auto-pauses for human action without bypassing), and **SQLite** session/profile persistence (API keys are never stored in logs or SQLite).
- **2 Built-In Playable HTML5 Demo Games:** Ready to test immediately offline/locally with zero ads or external dependencies:
  - **Crystal Quest Arena** (`http://localhost:8000/demo/crystal-quest`) — Keyboard + mouse grid navigation, gems, hazards, and extraction portal.
  - **Sequence Strike** (`http://localhost:8000/demo/sequence-strike`) — Mouse coordinate precision test with numbered targets and bombs.

---

## Project Structure

```text
ai-game-player/
├── start.ps1                       # One-command launcher for Windows PowerShell
├── .env.example                    # Example environment variable reference
├── README.md
├── backend/
│   ├── main.py                     # FastAPI REST + WebSocket server & demo game routes
│   ├── requirements.txt            # Python dependencies
│   ├── pytest.ini                  # Pytest configuration
│   ├── agent/
│   │   ├── prompts.py              # Strategic & Tactical system prompts + JSON schemas
│   │   ├── memory.py               # Bounded context compression, stagnation detection & learning
│   │   ├── llm.py                  # OpenAI / Anthropic / Gemini vision client, cost tracker & retry
│   │   └── loop.py                 # Observation-Action-Feedback loop with Pause/Resume/Stop
│   ├── browser/
│   │   └── playwright_manager.py   # Visible Chromium control, coordinate overlay & safety checks
│   ├── models/
│   │   └── schemas.py              # Strict Pydantic v2 validation schemas
│   ├── storage/
│   │   └── database.py             # SQLite session history & game profile storage
│   ├── demo_games/
│   │   ├── grid_quest.html         # Built-in Crystal Quest HTML5 game
│   │   └── target_matrix.html      # Built-in Sequence Strike HTML5 game
│   └── tests/
│       └── test_basic.py           # Automated test suite covering 15 core scenarios
└── frontend/
    ├── package.json
    ├── postcss.config.js
    └── src/
        ├── App.jsx
        ├── index.css
        ├── constants/defaultConfig.js
        └── components/
            ├── HeaderBar.jsx       # Status indicators & Start/Pause/Resume/Stop/Restart controls
            ├── StatsStrip.jsx      # Real-time actions, timer, tokens, USD cost & goal progress
            ├── ConfigSidebar.jsx   # Presets, LLM setup, Game rules & Agent safety limits
            ├── LiveViewport.jsx    # Live game view, coordinate Action HUD & operator hint bar
            ├── AgentActivityPanel.jsx # Strategic planner memory & tactical decision stream
            └── HistoryModal.jsx    # SQLite session history & step-by-step inspector
```

---

## Quick Start

### Option A: One-Command Launch (Windows PowerShell)

From inside `ai-game-player/`:

```powershell
# First-time setup (installs Python/Node packages + Playwright Chromium) and starts servers:
./start.ps1 -Install

# Subsequent launches (starts Backend + Frontend immediately):
./start.ps1
```

### Option B: Manual Two-Terminal Launch

**Terminal 1 — Backend (`http://localhost:8000`):**
```powershell
cd backend
pip install -r requirements.txt
playwright install chromium
python main.py
```

**Terminal 2 — Frontend (`http://localhost:5173`):**
```powershell
cd frontend
npm install
npm run dev
```

---

## How to Play Any Browser Game with AI

1. Open **`http://localhost:5173`** in your browser.
2. In **1. LLM API**, select your provider (`OpenAI`, `Anthropic`, or `Google Gemini`), paste your API key, choose a vision model (e.g., `gpt-4o`), and optionally click **Verify API Key & Model**.
3. Pick one of the **Quick-Load Game Presets** at the top of the left sidebar (e.g., *Crystal Quest (Built-in Demo)* or *2048 Puzzle*), or switch to **2. Game Rules** and paste any browser game URL along with its rules, controls, and objective.
4. Click **Start** (or **Launch Autonomous Player**).
5. Watch the dedicated Chromium window open and observe the AI's live strategic plan, visual observations, action coordinates, and cost telemetry in real time.
6. Use **Pause**, **Resume**, **Stop**, **Restart**, or the **Guide AI** live hint bar below the viewport at any time.

---

## Running Automated Tests

```powershell
cd backend
python -m pytest tests/test_basic.py -v
```
