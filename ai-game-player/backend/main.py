import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure backend directory is on sys.path regardless of working directory
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models.schemas import GameConfig, RLTrainRequest, SessionState
from agent.loop import AgentLoop
from agent.llm import LLMManager
from storage.database import SessionDatabase
from rl.ga_trainer import BrassGATrainer

app = FastAPI(title="AI Game Player API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BRASS_WEBAPP_DIR = BACKEND_DIR / "webapp"
if BRASS_WEBAPP_DIR.exists():
    app.mount(
        "/brass-birmingham",
        StaticFiles(directory=str(BRASS_WEBAPP_DIR), html=True),
        name="brass-birmingham",
    )

db = SessionDatabase()

active_agent_task: Optional[asyncio.Task] = None
active_agent_loop: Optional[AgentLoop] = None
last_known_config: Optional[GameConfig] = None
latest_frame_b64: Optional[str] = None
connected_clients: set[WebSocket] = set()


BUILT_IN_PRESETS = [
    {
        "id": "crystal-quest",
        "game_name": "Crystal Quest (Built-in Demo)",
        "url": "http://localhost:8000/demo/crystal-quest",
        "game_description": "Turn-based tactical grid arena built for testing vision + keyboard/mouse navigation.",
        "rules": (
            "You control the cyan circle labeled PLAYER on an 8x5 grid. "
            "First click 'START MISSION' or press Space to dismiss the title overlay. "
            "Gray squares labeled WALL block movement. Red circles labeled MINE deal 1 shield damage. "
            "Gold diamonds labeled GEM are crystals. Once all 3 crystals are collected, thePORTAL on the right turns bright green."
        ),
        "keyboard_controls": "Space or Enter to start; ArrowUp, ArrowDown, ArrowLeft, ArrowRight (or W, A, S, D) to move one grid cell.",
        "mouse_controls": "Click 'START MISSION' button to begin; click directional D-Pad buttons on the right or click an adjacent grid cell.",
        "objective": "Collect all 3 gold GEM crystals and then step onto the green PORTAL tile at (7,2).",
        "secondary_objectives": "Avoid stepping on red MINE circles to preserve all 3 Shield HP.",
        "winning_conditions": "The 'VICTORY! MISSION COMPLETE' banner is displayed on screen.",
        "losing_conditions": "Shield HP reaches 0 and 'MISSION FAILED' appears.",
        "strategy_preference": "cautious",
        "default_action_duration_ms": 120,
        "observation_interval_ms": 600,
    },
    {
        "id": "sequence-strike",
        "game_name": "Sequence Strike (Mouse Precision Demo)",
        "url": "http://localhost:8000/demo/sequence-strike",
        "game_description": "Coordinate-precision mouse game with numbered green targets and red bombs.",
        "rules": (
            "Five green circular targets numbered #1 through #5 are placed on the arena alongside red BOMB circles. "
            "Targets only activate when clicked in exact ascending numerical order (#1, then #2, #3, #4, #5)."
        ),
        "keyboard_controls": "None required.",
        "mouse_controls": "Left-click the center of each green circle (#1 -> #2 -> #3 -> #4 -> #5) using x, y coordinates.",
        "objective": "Click green targets #1, #2, #3, #4, and #5 in order without clicking any red BOMB.",
        "secondary_objectives": "Complete in minimal clicks with 100% accuracy.",
        "winning_conditions": "'VICTORY! SEQUENCE COMPLETE' banner appears.",
        "losing_conditions": "Clicking a red BOMB triggers 'GAME OVER'.",
        "strategy_preference": "cautious",
        "default_action_duration_ms": 120,
        "observation_interval_ms": 500,
    },
    {
        "id": "2048-web",
        "game_name": "2048 Puzzle",
        "url": "https://play2048.co/",
        "game_description": "Classic sliding number-tile puzzle on a 4x4 grid.",
        "rules": (
            "Slide all tiles on the 4x4 board in one of four directions. "
            "Matching adjacent numbers merge into their sum (2+2=4, 4+4=8, etc.). "
            "After every valid move, a new 2 or 4 tile spawns."
        ),
        "keyboard_controls": "ArrowUp, ArrowDown, ArrowLeft, ArrowRight to slide all tiles.",
        "mouse_controls": "Click 'New Game' or dismiss any cookie banner if visible.",
        "objective": "Merge tiles strategically to build the 2048 tile (keep largest tile anchored in a corner).",
        "secondary_objectives": "Keep board open and avoid filling all 16 cells without merges.",
        "winning_conditions": "A 2048 tile is visible on the board.",
        "losing_conditions": "'Game over!' overlay appears with no valid moves remaining.",
        "strategy_preference": "balanced",
        "default_action_duration_ms": 120,
        "observation_interval_ms": 650,
    },
    {
        "id": "classic-snake",
        "game_name": "Classic Snake",
        "url": "https://playsnake.org/",
        "game_description": "Arcade snake game requiring quick directional turns.",
        "rules": (
            "Guide the snake toward food items to grow longer and score points. "
            "Select level or press Space/click Play on the menu screen first. "
            "Colliding with outer walls or the snake's own body ends the game."
        ),
        "keyboard_controls": "ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Space.",
        "mouse_controls": "Click menu buttons / level difficulty to start.",
        "objective": "Start the game and eat as much food as possible without crashing.",
        "secondary_objectives": "Stay near open center space as the snake grows.",
        "winning_conditions": "Reach a high score of 10+ apples.",
        "losing_conditions": "Game Over screen appears.",
        "strategy_preference": "aggressive",
        "default_action_duration_ms": 100,
        "observation_interval_ms": 400,
    },
    {
        "id": "colonist-catan",
        "game_name": "Catan (Colonist.io vs Bots)",
        "url": "https://colonist.io/",
        "game_description": "4-player Catan board game vs AI bots on Colonist.io.",
        "rules": (
            "1. LOBBY: Dismiss any cookie/consent popup first, then click 'Play vs Bots' on the home screen to enter a 4-player match.\n"
            "2. INITIAL SETUP: When prompted to place a Settlement or Road, valid intersections/edges appear as glowing white circles on the board. Click a glowing circle directly. Prioritize 6, 8, 5, 9 hexes and diverse resources.\n"
            "3. YOUR TURN: Click the Dice button in the bottom-right to roll. Check lit/colored build icons along the bottom-right bar (Road, Settlement, City, Dev Card); grayed-out buttons are unaffordable. Click a lit build button, wait one step for glowing circles on the board, then click a glowing circle. Decline bad bot trades with the red X. When done building, click the green arrow/checkmark End Turn button at bottom-right.\n"
            "4. ROBBER (7): Discard cards if >7 and click green checkmark. Move Robber onto an opponent's 6 or 8 hex (never your own hex).\n"
            "5. BOT TURNS: While bots are playing and bottom-right action icons are inactive, use `wait` (1500ms)."
        ),
        "keyboard_controls": "None needed (controlled via mouse clicks).",
        "mouse_controls": "Left-click lobby buttons, bottom-right icons (Dice, Road, Settlement, City, Dev Card, End Turn), and glowing white placement circles on the hex board.",
        "objective": "Start a 'Play vs Bots' game and reach 10 Victory Points before any opponent.",
        "secondary_objectives": "Prioritize Wheat + Ore for Cities and early Wood + Brick to secure a 3rd settlement spot.",
        "winning_conditions": "Victory scoreboard shows your player at 10+ Victory Points (1st Place).",
        "losing_conditions": "End-game scoreboard shows an opponent reached 10 Victory Points first.",
        "strategy_preference": "balanced",
        "default_action_duration_ms": 150,
        "observation_interval_ms": 1500,
        "max_iterations": 300,
        "max_session_duration_sec": 1800,
        "max_budget_usd": 3.0,
        "strategic_planning_interval": 10,
    },
    {
        "id": "brass-birmingham",
        "game_name": "Brass: Birmingham (Solo vs Mautoma)",
        "url": "http://localhost:8000/brass-birmingham/",
        "game_description": "Full Canal & Rail Era economic strategy board game vs Mautoma AI (includes Double Rail & Same-Industry 2x Develop; supports PyTorch RL Policy and Vision LLMs).",
        "rules": (
            "1. GOAL: Score more Victory Points (VP) than Mautoma across Canal Era (10 rounds) and Rail Era (10 rounds).\n"
            "2. SCORING: Unflipped consumer tiles (Cotton, Goods, Pottery) and un-depleted mines/breweries score 0 VP! Flip consumers via Sell (requires connected Merchant icon + Beer), deplete Coal/Iron/Beer cubes to flip producers, and build Links between high-dot flipped cities & Merchants.\n"
            "3. KEY ACTIONS: Build, Network (1 canal for £3; in Rail Era 1 rail for £5+coal OR 2 rails for £15+2 coal+1 beer!), Sell (flip multiple buildings in 1 action!), Develop (remove 1-2 low-level tiles for 1 iron each, including 2x on the same industry!), Loan (+£30, -3 income tiers), Scout (discard 3 cards for Wild Location + Wild Industry)."
        ),
        "keyboard_controls": "None needed (mouse click controls).",
        "mouse_controls": "Click Start game, then click Action buttons (Build, Network, Sell, Develop, Loan, Scout), Hand cards, and Option buttons.",
        "objective": "Defeat Mautoma by scoring the highest total Victory Points across Canal and Rail Eras (target 130+ VP).",
        "secondary_objectives": "Flip every built consumer tile before Round 10 of each era, secure Brewery beer early, and connect high-dot cities.",
        "winning_conditions": "Game Over scoreboard shows your final VP strictly greater than Automa VP.",
        "losing_conditions": "Game Over scoreboard shows Automa VP higher than your VP.",
        "strategy_preference": "aggressive",
        "default_action_duration_ms": 140,
        "observation_interval_ms": 500,
        "max_iterations": 250,
        "max_session_duration_sec": 1800,
        "max_budget_usd": 2.5,
        "strategic_planning_interval": 8,
    },
    {
        "id": "brass-2p-duel",
        "game_name": "⚔️ Brass 2P Duel: API vs Evolved AI",
        "url": "http://localhost:8000/brass-birmingham/",
        "match_mode": "brass_2p_duel",
        "opponent_model": "brass-ppo-best",
        "api_player_seat": 0,
        "game_seed": 202611,
        "enable_rl_advisor": False,
        "game_description": "Head-to-head official 2-player Brass: Birmingham duel! Watch Gemini / GPT / Claude play live against our locally evolved PyTorch Genetic Algorithm model on a deterministic seeded 40-card 2P deck.",
        "rules": (
            "1. OFFICIAL 2P RULESET & SEEDED DECK: Uses the exact 40-card 2-player Brass: Birmingham deck (northern 3p/4p cities & cotton/goods industry cards excluded) shuffled deterministically from Game Seed. Canal Era deals 38 cards (1 unseen discard/player; Rd 1 = 1 action each, Rds 2-10 = 2 actions each). Rail Era deals all 40 cards (Rds 1-10 = 2 actions each).\n"
            "2. TURN ORDER & SCORING: Round spending (£ Spent) determines next round's turn order (lowest spender goes 1st!). Only FLIPPED tiles score VP and contribute link dots (●); Level 2+ tiles survive Canal -> Rail transition.\n"
            "3. FAIR HEAD-TO-HEAD: Each AI sees only its own private hand plus public board state, with zero advisor hints leaked to the API challenger."
        ),
        "keyboard_controls": "None needed (spectator mode — moves are clicked visually on screen).",
        "mouse_controls": "Automatic visual step-by-step clicks for both the API Challenger and Evolved AI agent.",
        "objective": "Defeat the opponent AI by scoring the highest total Victory Points across Canal and Rail Eras (78 total moves).",
        "secondary_objectives": "Control round turn order via smart cash spending, secure Beer barrels, and flip high-VP Level 2+ buildings.",
        "winning_conditions": "Highest final VP after Rail Era Round 10 (tie-breakers: Income level, then remaining Cash).",
        "losing_conditions": "Opponent finishes Rail Era Round 10 with higher VP.",
        "strategy_preference": "aggressive",
        "default_action_duration_ms": 260,
        "observation_interval_ms": 1100,
        "max_iterations": 220,
        "max_session_duration_sec": 3600,
        "max_budget_usd": 5.0,
        "strategic_planning_interval": 10,
    },
    {
        "id": "brass-human-vs-ai",
        "game_name": "🎮 Brass 2P: You vs AI Model",
        "url": "http://localhost:8000/brass-birmingham/",
        "match_mode": "brass_human_vs_ai",
        "provider": "brass_rl",
        "model": "brass-ppo-best",
        "human_player_seat": 0,
        "game_seed": 202611,
        "enable_rl_advisor": False,
        "game_description": "Play official 2-player Brass: Birmingham yourself against our locally evolved PyTorch model ($0 cost, no API key needed) OR against Gemini / GPT / Claude!",
        "rules": (
            "1. INTERACTIVE PLAY: On your turn, click directly on the Live Browser Arena board/cards (or in the game window) to make your moves. Your AI opponent's hand is masked (🂠 Hidden) so neither side sees private cards.\n"
            "2. OFFICIAL 2P DECK & RULES: Deterministic 40-card 2-player deck from Game Seed (38 Canal cards, 40 Rail cards; Rd 1 Canal = 1 action each, all other rounds = 2 actions each; lowest round spender acts 1st next round).\n"
            "3. AI OPPONENT: Choose Local Evolved Model (brass-ppo-best, $0 cost) or any Cloud Vision API model."
        ),
        "keyboard_controls": "Mouse scroll wheel scrolls option lists.",
        "mouse_controls": "Click Action buttons, Hand cards, Cities/Links, and Confirm buttons directly inside the Live Browser Arena!",
        "objective": "Outscore the AI model across Canal Era (10 rounds) and Rail Era (10 rounds) under official 2-player Brass: Birmingham rules.",
        "secondary_objectives": "Watch round spending to steal turn order, secure Beer barrels early, and flip Level 2+ buildings.",
        "winning_conditions": "Finish Rail Era Round 10 with more Victory Points than the AI opponent.",
        "losing_conditions": "AI opponent finishes Rail Era Round 10 with more Victory Points.",
        "strategy_preference": "aggressive",
        "default_action_duration_ms": 240,
        "observation_interval_ms": 900,
        "max_iterations": 300,
        "max_session_duration_sec": 7200,
        "max_budget_usd": 5.0,
        "strategic_planning_interval": 10,
    },
]

PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini", "o4-mini"],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.8-flash", "gemini-3.8-pro"],
    "brass_rl": ["brass-ppo-best", "brass-ppo-latest"],
}


async def broadcast_update(message: dict) -> None:
    global latest_frame_b64
    if message.get("image"):
        latest_frame_b64 = message["image"]
    msg_str = json.dumps(message)
    for client in list(connected_clients):
        try:
            await client.send_text(msg_str)
        except Exception:
            connected_clients.discard(client)


rl_trainer = BrassGATrainer(db=db, on_update=broadcast_update)


@app.get("/demo/crystal-quest", response_class=HTMLResponse)
async def serve_crystal_quest():
    html_path = BACKEND_DIR / "demo_games" / "grid_quest.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/demo/sequence-strike", response_class=HTMLResponse)
async def serve_sequence_strike():
    html_path = BACKEND_DIR / "demo_games" / "target_matrix.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    is_running = bool(active_agent_task and not active_agent_task.done())
    return {
        "status": "ok",
        "agent_active": is_running,
        "session_state": active_agent_loop.state.model_dump() if active_agent_loop else SessionState().model_dump(),
    }


@app.get("/api/presets")
async def get_presets():
    return {
        "presets": BUILT_IN_PRESETS,
        "providers": PROVIDER_MODELS,
    }


class TestLLMRequest(BaseModel):
    provider: str = "openai"
    api_key: str
    model: str = "gpt-4o"
    base_url: Optional[str] = None


@app.post("/api/test-llm")
async def test_llm_connection(req: TestLLMRequest):
    if not req.api_key or not req.api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required.")
    cfg = GameConfig(
        provider=req.provider,
        api_key=req.api_key.strip(),
        model=req.model.strip() or "gpt-4o",
        base_url=req.base_url,
        url="http://localhost:8000/demo/crystal-quest",
    )
    mgr = LLMManager(cfg)
    return await mgr.test_connection()


@app.post("/api/list-models")
async def list_provider_models(req: TestLLMRequest):
    if not req.api_key or not req.api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required.")
    cfg = GameConfig(
        provider=req.provider,
        api_key=req.api_key.strip(),
        model=req.model.strip() or "gemini-3.8-flash",
        base_url=req.base_url,
        url="http://localhost:8000/demo/crystal-quest",
    )
    mgr = LLMManager(cfg)
    return await mgr.list_available_models()


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": db.list_sessions(limit=40)}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Session not found.")
    return detail


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = db.delete_session(session_id)
    return {"deleted": deleted}


class SaveProfileRequest(BaseModel):
    name: str
    config: Dict[str, Any]


@app.get("/api/profiles")
async def list_profiles():
    return {"profiles": db.list_profiles()}


@app.post("/api/profiles")
async def save_profile(req: SaveProfileRequest):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Profile name is required.")
    saved = db.save_profile(req.name, req.config)
    return saved


@app.delete("/api/profiles/{profile_id}")
async def delete_profile(profile_id: int):
    return {"deleted": db.delete_profile(profile_id)}


@app.get("/api/rl/status")
async def get_rl_status():
    return rl_trainer.get_status().model_dump()


@app.post("/api/rl/train/start")
async def start_rl_training(req: RLTrainRequest):
    if rl_trainer.running:
        raise HTTPException(status_code=400, detail="Genetic Algorithm training is already running.")
    rl_trainer.start_background(
        workers=req.workers,
        training_mode=req.training_mode,
        generations=req.generations,
        population_size=req.population_size,
        games_per_individual=req.games_per_individual,
        mutation_sigma=req.mutation_sigma,
        mutation_rate=req.mutation_rate,
        elite_count=req.elite_count,
        resume_checkpoint=req.resume_checkpoint,
    )
    return rl_trainer.get_status().model_dump()


@app.post("/api/rl/train/stop")
async def stop_rl_training():
    await rl_trainer.stop()
    return rl_trainer.get_status().model_dump()


@app.post("/api/rl/episodes/clear")
async def clear_rl_training_history():
    if rl_trainer.running:
        raise HTTPException(status_code=400, detail="Stop training before clearing episode chart history.")
    deleted = db.clear_rl_episodes()
    return {"deleted": deleted, "status": rl_trainer.get_status().model_dump()}


async def _stop_current_agent() -> None:
    global active_agent_task, active_agent_loop
    if active_agent_loop:
        await active_agent_loop.stop()
    if active_agent_task and not active_agent_task.done():
        try:
            await asyncio.wait_for(active_agent_task, timeout=4.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            active_agent_task.cancel()
    active_agent_loop = None
    active_agent_task = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_agent_task, active_agent_loop, last_known_config, latest_frame_b64
    await websocket.accept()
    connected_clients.add(websocket)

    # Send current state snapshot on initial connection
    if active_agent_loop:
        snap = {
            "type": "state_update",
            "state": active_agent_loop.state.model_dump(),
        }
        if latest_frame_b64:
            snap["image"] = latest_frame_b64
        await websocket.send_text(json.dumps(snap))

    await websocket.send_text(
        json.dumps(
            {
                "type": "rl_training_update",
                "status": rl_trainer.get_status().model_dump(),
            }
        )
    )

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            cmd = payload.get("command")

            if cmd == "start":
                if active_agent_task and not active_agent_task.done():
                    await websocket.send_text(
                        json.dumps({"type": "command_error", "error": "Agent session is already active. Stop or restart instead."})
                    )
                    continue

                config_data = payload.get("config") or {}
                try:
                    config = GameConfig(**config_data)
                except Exception as e:
                    await websocket.send_text(
                        json.dumps({"type": "command_error", "error": f"Invalid configuration: {e}"})
                    )
                    continue

                last_known_config = config
                latest_frame_b64 = None
                active_agent_loop = AgentLoop(config, broadcast_update, db=db)
                active_agent_task = asyncio.create_task(active_agent_loop.start())

            elif cmd == "pause":
                if active_agent_loop:
                    await active_agent_loop.pause()

            elif cmd == "resume":
                if active_agent_loop:
                    await active_agent_loop.resume()

            elif cmd == "stop":
                await _stop_current_agent()

            elif cmd == "restart":
                await _stop_current_agent()
                config_data = payload.get("config")
                try:
                    config = GameConfig(**config_data) if config_data else last_known_config
                except Exception as e:
                    await websocket.send_text(
                        json.dumps({"type": "command_error", "error": f"Invalid configuration: {e}"})
                    )
                    continue
                if not config:
                    await websocket.send_text(
                        json.dumps({"type": "command_error", "error": "No configuration available to restart."})
                    )
                    continue
                last_known_config = config
                latest_frame_b64 = None
                active_agent_loop = AgentLoop(config, broadcast_update, db=db)
                active_agent_task = asyncio.create_task(active_agent_loop.start())

            elif cmd == "hint":
                hint_text = payload.get("hint", "")
                if active_agent_loop and hint_text:
                    active_agent_loop.add_user_hint(hint_text)
                    await broadcast_update(
                        {
                            "type": "state_update",
                            "state": active_agent_loop.state.model_dump(),
                            "event_note": f"Operator guidance queued: \"{hint_text}\"",
                        }
                    )

            elif cmd == "user_click":
                if active_agent_loop:
                    asyncio.create_task(
                        active_agent_loop.handle_human_click(
                            int(payload.get("x", 640)),
                            int(payload.get("y", 360)),
                        )
                    )

            elif cmd == "user_scroll":
                if active_agent_loop:
                    asyncio.create_task(
                        active_agent_loop.handle_human_scroll(
                            int(payload.get("x", 640)),
                            int(payload.get("y", 360)),
                            int(payload.get("delta_y", 0)),
                        )
                    )

    except WebSocketDisconnect:
        connected_clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
