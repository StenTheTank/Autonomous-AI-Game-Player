import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from pydantic import ValidationError

# Ensure backend root is importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models.schemas import (
    GameConfig,
    Action,
    LLMResponse,
    StrategicPlan,
    VisionDetectedObject,
    VisionGroundingResponse,
    coords_to_macro_cell,
    macro_cell_to_coords,
)
from vision.grounding import VisualGroundingEngine
from agent.llm import LLMManager, LLMCallTelemetry, extract_json_object
from agent.memory import SessionMemory
from agent.loop import AgentLoop
from browser.playwright_manager import BrowserManager
from storage.database import SessionDatabase


def make_dummy_jpeg_bytes(color=(30, 60, 120)) -> bytes:
    img = Image.new("RGB", (1280, 720), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_sessions.db"
    return SessionDatabase(db_path=str(db_file))


@pytest.fixture
def sample_config():
    return GameConfig(
        provider="openai",
        api_key="sk-secret-do-not-leak-12345",
        model="gpt-4o",
        url="http://localhost:8000/demo/crystal-quest",
        game_name="Crystal Quest",
        rules="Collect 3 gems and enter portal.",
        objective="Win the level.",
        keyboard_controls="Arrow keys, Space",
        mouse_controls="Click start button",
        max_iterations=5,
        max_budget_usd=0.50,
        observation_interval_ms=100,
        max_consecutive_failures=2,
        headless=True,
    )


# 1 & 2. Browser launching & Game loading
@pytest.mark.asyncio
async def test_browser_launch_and_navigation():
    bm = BrowserManager(headless=True)
    mock_pw = AsyncMock()
    mock_browser = AsyncMock()
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)

    mock_pw.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    with patch("browser.playwright_manager.async_playwright") as mock_apw:
        mock_apw.return_value.start = AsyncMock(return_value=mock_pw)
        await bm.start()
        assert bm.browser is mock_browser
        assert bm.page is mock_page

        await bm.navigate("http://localhost:8000/demo/crystal-quest")
        mock_page.goto.assert_called_once()

        # Verify unsafe file:// URLs are blocked
        with pytest.raises(ValueError):
            await bm.navigate("file:///etc/passwd")

        await bm.stop()


# 3. Screenshot capture, coordinate overlay & visual delta
@pytest.mark.asyncio
async def test_screenshot_capture_and_visual_delta():
    bm = BrowserManager(headless=True)
    bm.page = AsyncMock()
    bm.page.is_closed = MagicMock(return_value=False)

    frame1 = make_dummy_jpeg_bytes((20, 30, 40))
    frame2 = make_dummy_jpeg_bytes((220, 200, 180))

    bm.page.screenshot.return_value = frame1
    ui_b64_1, llm_b64_1, delta_1 = await bm.capture_observation(add_grid_for_llm=True)
    assert len(ui_b64_1) > 100
    assert len(llm_b64_1) > 100
    assert ui_b64_1 != llm_b64_1  # LLM copy has grid overlay
    assert delta_1 == 100.0

    # Identical frame should yield 0% delta
    _, _, delta_same = await bm.capture_observation(add_grid_for_llm=False)
    assert delta_same == 0.0

    # Distinct frame should yield high visual delta
    bm.page.screenshot.return_value = frame2
    _, _, delta_changed = await bm.capture_observation(add_grid_for_llm=False)
    assert delta_changed > 90.0


# 4 & 5. LLM API connection & Vision Image Submission
@pytest.mark.asyncio
async def test_llm_connection_and_image_submission(sample_config):
    mgr = LLMManager(sample_config)
    mock_client = AsyncMock()
    mgr.openai_client = mock_client

    # Mock connection test response
    ping_resp = MagicMock()
    ping_resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
    ping_resp.usage = MagicMock(prompt_tokens=20, completion_tokens=5)
    mock_client.chat.completions.create.return_value = ping_resp

    conn_res = await mgr.test_connection()
    assert conn_res["ok"] is True
    assert conn_res["tokens"] == 25

    # Mock tactical vision call and verify base64 image payload was sent
    tac_json = {
        "observation": "Start screen with START MISSION button.",
        "reasoning_summary": "Need to click START MISSION to begin.",
        "action": {"type": "mouse_click", "x": 450, "y": 380, "button": "left"},
        "expected_result": "Game grid becomes active.",
        "goal_status": "in_progress",
    }
    tac_resp = MagicMock()
    tac_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(tac_json)))]
    tac_resp.usage = MagicMock(prompt_tokens=600, completion_tokens=85)
    mock_client.chat.completions.create.return_value = tac_resp

    parsed, tel = await mgr.get_next_action("FAKE_BASE64_IMG", "Context string")
    assert parsed.action.type == "mouse_click"
    assert parsed.action.x == 450
    assert tel.total_tokens == 685
    assert tel.cost_usd > 0

    sent_kwargs = mock_client.chat.completions.create.call_args.kwargs
    user_msg = sent_kwargs["messages"][1]["content"]
    assert any(
        block.get("type") == "image_url"
        and "data:image/jpeg;base64,FAKE_BASE64_IMG" in block["image_url"]["url"]
        for block in user_msg
    )


# 6. Action parsing, normalization & strict schema validation
def test_action_parsing_and_validation():
    # Key normalization ('SPACE' -> 'Space', 'UP' -> 'ArrowUp')
    act = Action(type="key_press", key="SPACE", duration_ms=200)
    assert act.key == "Space"

    act_up = Action(type="key_press", key="UP")
    assert act_up.key == "ArrowUp"

    # Simultaneous multi-key combo
    combo = Action(type="key_combo", keys=["RIGHT", "SPACE"], duration_ms=150)
    assert combo.keys == ["ArrowRight", "Space"]

    # 3-Way Precision Mouse Targeting: #ID element, 16x9 macro grid cell, and 4x Zoom-Snap math
    assert macro_cell_to_coords("E7") == (360, 520)
    cell_click = Action(type="mouse_click", grid_cell="e7")
    assert cell_click.grid_cell == "E7"
    assert (cell_click.x, cell_click.y) == (360, 520)

    elem_click = Action(type="mouse_click", element_id=3)
    assert elem_click.element_id == 3

    bm = BrowserManager(headless=True)
    bm._last_raw_image = Image.new("RGB", (1280, 720), color=(40, 80, 120))
    zoom_b64, cx0, cy0 = bm.build_zoom_crop_b64(360, 520)
    assert len(zoom_b64) > 200
    assert (cx0, cy0) == (240, 430)
    # Cell G5 in 240x180 crop starting at (240, 430) -> col 6 (x=240+120), row 4 (y=430+80)
    assert bm.resolve_zoom_cell(cx0, cy0, "G5", "center") == (370, 520)
    assert bm.resolve_zoom_cell(cx0, cy0, "F5", "bottom") == (350, 526)

    # Missing required coordinates/element_id/grid_cell for mouse_click must raise ValidationError
    with pytest.raises(ValidationError):
        Action(type="mouse_click")

    # Unsupported action type must raise ValidationError
    with pytest.raises(ValidationError):
        Action(type="execute_shell_command")


# 7. Keyboard input execution (press, hold, release, multi-key combo)
@pytest.mark.asyncio
async def test_keyboard_inputs():
    bm = BrowserManager(headless=True, max_action_duration_ms=500)
    bm.page = AsyncMock()
    bm.page.is_closed = MagicMock(return_value=False)

    await bm.execute_action({"type": "key_press", "key": "Space"})
    bm.page.keyboard.press.assert_called_with("Space")

    await bm.execute_action({"type": "key_hold", "key": "ArrowRight", "duration_ms": 60})
    bm.page.keyboard.down.assert_called_with("ArrowRight")
    bm.page.keyboard.up.assert_called_with("ArrowRight")

    bm.page.keyboard.down.reset_mock()
    bm.page.keyboard.up.reset_mock()
    await bm.execute_action({"type": "key_combo", "keys": ["Control", "z"], "duration_ms": 40})
    assert bm.page.keyboard.down.call_count == 2
    assert bm.page.keyboard.up.call_count == 2


# 8. Mouse input execution (click, move, drag, scroll)
@pytest.mark.asyncio
async def test_mouse_inputs():
    bm = BrowserManager(headless=True)
    bm.page = AsyncMock()
    bm.page.is_closed = MagicMock(return_value=False)

    await bm.execute_action({"type": "mouse_click", "x": 640, "y": 360, "button": "left", "clicks": 1})
    bm.page.mouse.move.assert_called_with(640, 360, steps=6)
    bm.page.mouse.down.assert_called_with(button="left")
    bm.page.mouse.up.assert_called_with(button="left")
    assert bm.mouse_pos == (640, 360)

    await bm.execute_action({"type": "mouse_move", "x": 200, "y": 150})
    bm.page.mouse.move.assert_called_with(200, 150, steps=6)
    assert bm.mouse_pos == (200, 150)

    await bm.execute_action({"type": "mouse_drag", "from_x": 100, "from_y": 100, "x": 300, "y": 300, "button": "left"})
    bm.page.mouse.down.assert_called_with(button="left")
    bm.page.mouse.up.assert_called_with(button="left")

    await bm.execute_action({"type": "scroll", "scroll_x": 0, "scroll_y": 240})
    bm.page.mouse.wheel.assert_called_with(0, 240)


# 9 & 10. Pause, Resume, and Emergency Stop functionality
@pytest.mark.asyncio
async def test_pause_resume_and_stop(sample_config, temp_db):
    updates = []
    loop = AgentLoop(sample_config, lambda msg: updates.append(msg), db=temp_db)
    loop.browser = AsyncMock()
    loop.browser.mouse_pos = (640, 360)
    loop.browser.page = MagicMock()
    loop.browser.page.is_closed.return_value = False
    loop.browser.page.url = sample_config.url
    loop.browser.ensure_alive.return_value = True
    loop.browser.detect_access_gate.return_value = None
    loop.browser.scan_interactive_elements.return_value = []
    loop.browser.should_zoom_refine.return_value = False
    loop.browser.capture_observation.return_value = ("IMG_UI", "IMG_LLM", 12.5)
    loop.browser.execute_action.return_value = "Tapped ArrowRight"

    plan = StrategicPlan(
        game_state_summary="Active",
        current_strategy="Collect gems",
        sub_goals=["Get gem 1"],
    )
    step_resp = LLMResponse(
        observation="Robot on grid",
        reasoning_summary="Move right",
        action=Action(type="key_press", key="ArrowRight"),
        expected_result="Moves to (1,2)",
        goal_status="in_progress",
    )
    tel = LLMCallTelemetry(prompt_tokens=100, completion_tokens=30, total_tokens=130, cost_usd=0.001)

    loop.llm.get_strategic_plan = AsyncMock(return_value=(plan, tel))
    loop.llm.get_next_action = AsyncMock(return_value=(step_resp, tel))

    task = asyncio.create_task(loop.start())
    await asyncio.sleep(0.15)

    # Pause session
    await loop.pause()
    assert loop.state.status == "paused"
    loop.browser.release_all_inputs.assert_called()

    # Resume session
    await loop.resume()
    assert loop.state.status == "running"

    # Emergency Stop
    await loop.stop()
    await asyncio.wait_for(task, timeout=3.0)
    assert loop.state.status == "stopped"


# 11. Invalid LLM JSON response & automatic correction retry
@pytest.mark.asyncio
async def test_invalid_llm_response_correction_retry(sample_config):
    mgr = LLMManager(sample_config)
    mock_client = AsyncMock()
    mgr.openai_client = mock_client

    bad_resp = MagicMock()
    bad_resp.choices = [MagicMock(message=MagicMock(content="NOT VALID JSON AT ALL"))]
    bad_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=10)

    valid_json = {
        "observation": "Recovered valid JSON",
        "reasoning_summary": "Corrected format",
        "action": {"type": "wait", "duration_ms": 500},
        "expected_result": "Wait briefly",
        "goal_status": "in_progress",
    }
    good_resp = MagicMock()
    good_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(valid_json)))]
    good_resp.usage = MagicMock(prompt_tokens=70, completion_tokens=30)

    mock_client.chat.completions.create.side_effect = [bad_resp, good_resp]

    result, tel = await mgr.get_next_action("IMG", "CTX")
    assert result.action.type == "wait"
    assert tel.api_calls == 2
    assert tel.total_tokens == 160


# 12. Consecutive API failures & API key redaction safety
@pytest.mark.asyncio
async def test_api_failure_limit_and_key_redaction(sample_config, temp_db):
    loop = AgentLoop(sample_config, lambda m: None, db=temp_db)
    loop.browser = AsyncMock()
    loop.browser.mouse_pos = (640, 360)
    loop.browser.page = MagicMock()
    loop.browser.page.is_closed.return_value = False
    loop.browser.page.url = sample_config.url
    loop.browser.ensure_alive.return_value = True
    loop.browser.detect_access_gate.return_value = None
    loop.browser.scan_interactive_elements.return_value = []
    loop.browser.capture_observation.return_value = ("IMG", "IMG", 10.0)

    # Simulate failure containing the raw API key to verify redaction
    redacted_msg = loop.llm._redact_error(
        RuntimeError(f"Auth failed for key {sample_config.api_key}")
    )
    assert sample_config.api_key not in redacted_msg
    assert "[REDACTED_API_KEY]" in redacted_msg

    loop.llm.get_strategic_plan = AsyncMock(side_effect=RuntimeError("API Down"))
    loop.llm.get_next_action = AsyncMock(side_effect=RuntimeError("API Down"))

    await loop.start()
    assert loop.state.status == "error"
    assert loop.state.stats.consecutive_failures >= sample_config.max_consecutive_failures


# 13. SQLite Session & Profile Persistence (Ensuring API Key is NEVER stored)
def test_sqlite_session_and_profile_persistence(sample_config, temp_db):
    raw_cfg = sample_config.model_dump()
    assert "api_key" in raw_cfg

    temp_db.create_session("sess_001", raw_cfg, {"total_actions": 0})
    temp_db.record_step(
        session_id="sess_001",
        iteration=1,
        observation="Saw start button",
        reasoning="Click start",
        action={"type": "mouse_click", "x": 640, "y": 400},
        expected_result="Game starts",
        visual_delta_pct=25.4,
        goal_status="in_progress",
        tokens_used=450,
        cost_usd=0.0021,
    )
    temp_db.update_session(
        "sess_001",
        status="completed",
        stats={"total_actions": 1, "estimated_cost_usd": 0.0021},
        status_reason="Won!",
        strategy="Direct gem path",
        lessons=["Click button first"],
    )

    detail = temp_db.get_session_detail("sess_001")
    assert detail is not None
    assert detail["status"] == "completed"
    assert len(detail["steps"]) == 1
    assert "api_key" not in detail["config"]
    assert sample_config.api_key not in json.dumps(detail)

    # Profile persistence check
    temp_db.save_profile("My Custom Game", raw_cfg)
    profiles = temp_db.list_profiles()
    assert len(profiles) == 1
    assert "api_key" not in profiles[0]["config"]


# 14. Budget and Iteration Guardrail Limits
@pytest.mark.asyncio
async def test_budget_and_iteration_limits(sample_config, temp_db):
    # Test budget limit triggers clean halt
    budget_cfg = sample_config.model_copy(update={"max_budget_usd": 0.01, "max_iterations": 50})
    loop = AgentLoop(budget_cfg, lambda m: None, db=temp_db)
    loop.browser = AsyncMock()
    loop.browser.mouse_pos = (640, 360)
    loop.browser.page = MagicMock()
    loop.browser.page.is_closed.return_value = False
    loop.browser.page.url = budget_cfg.url
    loop.browser.ensure_alive.return_value = True
    loop.browser.detect_access_gate.return_value = None
    loop.browser.scan_interactive_elements.return_value = []
    loop.browser.should_zoom_refine.return_value = False
    loop.browser.capture_observation.return_value = ("IMG", "IMG", 15.0)
    loop.browser.execute_action.return_value = "OK"

    plan = StrategicPlan(game_state_summary="S", current_strategy="Strat", sub_goals=["G1"])
    step_resp = LLMResponse(
        observation="Obs",
        reasoning_summary="Reason",
        action=Action(type="key_press", key="Space"),
        expected_result="Exp",
        goal_status="in_progress",
    )
    expensive_tel = LLMCallTelemetry(
        prompt_tokens=2000, completion_tokens=500, total_tokens=2500, cost_usd=0.015
    )
    loop.llm.get_strategic_plan = AsyncMock(return_value=(plan, expensive_tel))
    loop.llm.get_next_action = AsyncMock(return_value=(step_resp, expensive_tel))

    await loop.start()
    assert loop.state.status == "stopped"
    assert "budget" in loop.state.status_reason.lower()
    assert loop.state.stats.current_iteration == 1


# 15. In-session learning, stagnation detection & context compression
def test_session_memory_learning_and_compression(sample_config):
    mem = SessionMemory(sample_config)
    mem.add_user_hint("Click START MISSION first")

    for i in range(1, 9):
        resp = LLMResponse(
            observation=f"Frame {i} observation",
            reasoning_summary="Testing step",
            action=Action(type="key_press", key="ArrowRight"),
            expected_result="Move right",
            goal_status="in_progress",
            learned_lesson="Walls block movement" if i == 2 else None,
            progress_percent=i * 10,
        )
        mem.record_tactical_step(i, resp, "Tapped ArrowRight")
        # Simulate 0% screen change to trigger stagnation detection
        mem.record_previous_action_effect(0.0)

    # Verify context window stays bounded to 6 recent steps + compressed summary
    assert len(mem.recent_steps) == 6
    assert len(mem.compressed_milestones) == 2
    assert mem.consecutive_stagnant_steps >= 4
    assert mem.needs_strategic_replan(9) is True
    assert "Walls block movement" in mem.lessons_learned
    # Low-level dead-input warnings stay in tactical_ui_notes and NEVER pollute strategic lessons_learned!
    assert any("no visible change" in n for n in mem.tactical_ui_notes)
    assert all("no visible change" not in l for l in mem.lessons_learned)

    ctx = mem.build_tactical_context(9, 0.0)
    assert "STAGNATION DETECTED" in ctx
    assert "Click START MISSION first" in ctx

    # Verify markdown-fenced JSON extraction
    fenced = '```json\n{"ok": true, "count": 3}\n```'
    assert extract_json_object(fenced) == {"ok": True, "count": 3}


def test_strategy_ignores_ui_reliability_and_button_locations(sample_config):
    mem = SessionMemory(sample_config)

    # Simulate tactical step that succeeded on Develop button and tried to record UI reliability / button location
    resp_ui_biased = LLMResponse(
        observation="Canal Era Round 2, £30 cash, hand has Worcester & brewery near target_008 (954, 339).",
        reasoning_summary="Click Develop.",
        action=Action(type="click_target", target_id="target_008", x=954, y=339),
        expected_result="Develop iron works.",
        goal_status="in_progress",
        learned_lesson="Develop action can be done reliably and the Develop button is situated in the right sidebar.",
        current_subgoal="Build Coal Mine in Coalbrookdale",
        progress_percent=20,
    )
    mem.record_tactical_step(1, resp_ui_biased, "Clicked target_008")
    mem.record_previous_action_effect(14.0)

    # Simulate tactical step that discovered a genuine game rule
    resp_domain_rule = LLMResponse(
        observation="Board shows Coalbrookdale connected to Birmingham via canal.",
        reasoning_summary="Check coal rule.",
        action=Action(type="wait", duration_ms=500),
        expected_result="Observe board.",
        goal_status="in_progress",
        learned_lesson="Coal consumption requires a connected transport link to a Coal Mine or Market.",
    )
    mem.record_tactical_step(2, resp_domain_rule, "Waited 500ms")

    # Apply a StrategicPlan containing mixed domain + polluted UI-button/reliability entries
    plan = StrategicPlan(
        game_state_summary="Canal Era Round 2, £30 cash, 0 industries on board.",
        current_strategy="Build Coal Mine and Brewery to flip Cotton Mills for Victory Points.",
        sub_goals=["Build Coal Mine in Coalbrookdale", "Build Canal link to Birmingham"],
        known_mechanics=[
            "Iron can be consumed from any Iron Works without a network connection.",
            "The Loan button is located at target_007 (954, 339).",
        ],
        lessons_learned=[
            "Avoid taking repeated Loans when cash is already sufficient to Build industries.",
            "Loan and Develop work reliably every time.",
        ],
    )
    mem.apply_strategic_plan(plan)

    # UI reliability and button location entries must be completely filtered out
    assert mem.lessons_learned == [
        "Coal consumption requires a connected transport link to a Coal Mine or Market.",
        "Avoid taking repeated Loans when cash is already sufficient to Build industries.",
    ]
    assert mem.known_mechanics == [
        "Iron can be consumed from any Iron Works without a network connection.",
    ]

    # Strategic context must contain zero effective/ineffective click counts, target_ids, or pixel coords
    strat_ctx = mem.build_strategic_context(3)
    assert "Effective Actions" not in strat_ctx
    assert "Ineffective Actions" not in strat_ctx
    assert "target_008" not in strat_ctx
    assert "954" not in strat_ctx
    assert "Canal Era Round 2" in strat_ctx

    # Tactical context must also omit 'Actions Proven Effective So Far' reward-hacking list
    tac_ctx = mem.build_tactical_context(3, 14.0)
    assert "Actions Proven Effective" not in tac_ctx


# 16. Hybrid Visual Grounding Pipeline, Interaction Map & Staleness Invalidation
def test_hybrid_visual_grounding_and_interaction_map():
    # Option A ('click_target') and Option B ('click_coordinates') protocol + 'action' alias
    opt_a = Action(**{"action": "click_target", "target_id": "target_002"})
    assert opt_a.type == "click_target"
    assert opt_a.target_id == "target_002"

    opt_b = Action(**{"action": "click_coordinates", "x": 542, "y": 318})
    assert opt_b.type == "click_coordinates"
    assert (opt_b.x, opt_b.y) == (542, 318)

    # Verify coords_to_macro_cell and automatic 0..1000 -> 1280x720 rescaling (including mid-board E5 / Coalbrookdale case)
    assert coords_to_macro_cell(965, 500) == "M7"
    norm_fix = Action(type="click_coordinates", grid_cell="M7", x=750, y=690)
    assert (norm_fix.x, norm_fix.y) == (960, 497)

    coalbrookdale_fix = Action(type="click_coordinates", grid_cell="E5", x=285, y=465)
    assert (coalbrookdale_fix.x, coalbrookdale_fix.y) == (365, 335)

    # Top-level flat action payload normalization
    flat_resp = LLMResponse(
        **{
            "observation": "Settlement spot target_002 is open.",
            "reasoning_summary": "Place initial settlement on high-yield node.",
            "action": "click_target",
            "target_id": "settlement_03",
            "expected_result": "Settlement placed.",
            "goal_status": "in_progress",
        }
    )
    assert flat_resp.action.type == "click_target"
    assert flat_resp.action.target_id == "settlement_03"

    # Synthetic 1280x720 board frame with two glowing Canvas rings + one DOM button
    from PIL import ImageDraw

    board_img = Image.new("RGB", (1280, 720), color=(28, 68, 98))
    draw = ImageDraw.Draw(board_img)
    # Glowing ring 1 at (360, 530), radius 11
    draw.ellipse([(349, 519), (371, 541)], fill=(255, 245, 190), outline=(255, 255, 255), width=3)
    # Glowing ring 2 at (620, 310), radius 11
    draw.ellipse([(609, 299), (631, 321)], fill=(255, 245, 190), outline=(255, 255, 255), width=3)
    # Bright DOM button region at (870, 672) that must be masked out of CV because DOM already owns it
    draw.rectangle([(800, 650), (940, 695)], fill=(240, 120, 40))

    dom_items = [
        {
            "id": 1,
            "tag": "button",
            "role": "button",
            "label": "Build settlement button",
            "x": 870,
            "y": 672,
            "bx": 800,
            "by": 650,
            "w": 140,
            "h": 45,
            "bbox": [800, 650, 140, 45],
            "confidence": 1.0,
        }
    ]
    layout_meta = {
        "url": "https://example.com/game",
        "viewport_w": 1280,
        "viewport_h": 720,
        "dpr": 1.0,
        "scroll_x": 0,
        "scroll_y": 0,
        "has_canvas": True,
        "canvas_rects": [[100, 60, 1080, 640]],
        "dialog_signature": "",
    }

    engine = VisualGroundingEngine()
    engine.next_screenshot_version()
    regen_1, reason_1 = engine.check_regeneration_needed(layout_meta, dom_items, visual_delta_pct=100.0)
    assert regen_1 is True
    assert reason_1 == "initial_observation"

    cv_targets = engine.detect_cv_targets(board_img, dom_targets=dom_items, canvas_rects=layout_meta["canvas_rects"])
    assert len(cv_targets) >= 2
    # Verify (360, 530) ring was localized within +/-4px by CV and DOM button at (870, 672) was excluded
    assert any(abs(c["center"][0] - 360) <= 4 and abs(c["center"][1] - 530) <= 4 for c in cv_targets)
    assert all(abs(c["center"][0] - 870) > 30 for c in cv_targets)

    # Simulate Vision LLM grounding with sensor fusion + 0..1000 scale drift guard
    ring1_cv = next(c for c in cv_targets if abs(c["center"][0] - 360) <= 6)
    vis_resp = VisionGroundingResponse(
        scene_summary="Two valid settlement locations on board.",
        targets=[
            VisionDetectedObject(
                label="Empty settlement location (hex 10)",
                cv_ref_id=ring1_cv["cv_ref_id"],
                grid_cell="E7",
                confidence=0.91,
            ),
            # Deliberately supply 0..1000 normalized coords (484, 430) for (620, 310) in cell H4!
            VisionDetectedObject(
                label="Empty settlement location (center hex)",
                grid_cell="H4",
                center=[484, 430],
                bbox=[470, 415, 28, 30],
                confidence=0.88,
            ),
        ],
    )
    vision_resolved = engine.resolve_vision_detections(vis_resp, cv_targets)
    imap = engine.build_interaction_map(
        dom_targets=dom_items,
        cv_targets=cv_targets,
        vision_targets=vision_resolved,
        layout_meta=layout_meta,
        regenerated=regen_1,
        regeneration_reason=reason_1,
    )

    assert imap.screenshot_version == 1
    assert len(imap.targets) == 3  # 1 DOM + 2 fused Vision targets (CV duplicates suppressed!)
    assert imap.targets[0].id == "target_001"
    assert imap.targets[0].source == "dom"
    assert imap.targets[0].center == [870, 672]
    assert imap.targets[1].id == "target_002"
    assert imap.targets[1].source == "vision"
    assert abs(imap.targets[2].center[0] - 620) <= 4 and abs(imap.targets[2].center[1] - 310) <= 4

    # Verify stable frame reuses cached vision map, then dialog/layout change invalidates it
    engine.next_screenshot_version()
    regen_2, reason_2 = engine.check_regeneration_needed(layout_meta, dom_items, visual_delta_pct=0.2)
    assert regen_2 is False
    assert reason_2 == "stable_layout"

    modal_meta = {**layout_meta, "dialog_signature": "DIALOG:Trade Offer:420"}
    engine.next_screenshot_version()
    regen_3, reason_3 = engine.check_regeneration_needed(modal_meta, dom_items, visual_delta_pct=0.5)
    assert regen_3 is True
    assert reason_3 == "dialog_or_modal_changed"


def test_same_prompt_noop_quarantine(sample_config):
    from models.schemas import InteractionMap, InteractionTarget

    mem = SessionMemory(sample_config)
    resp = LLMResponse(
        observation="Build prompt open with brewery cards.",
        reasoning_summary="Select brewery card.",
        action=Action(type="click_target", target_id="target_016", x=944, y=268),
        expected_result="Brewery options appear",
        goal_status="in_progress",
    )
    mem.record_tactical_step(
        1, resp, 'Clicked target_016', target_label="brewery [hand-card-1]"
    )
    # 0.0% screen change quarantines base label 'brewery' across duplicate slots
    mem.record_previous_action_effect(0.0)
    assert "brewery" in mem.current_prompt_noop_labels

    imap = InteractionMap(
        screenshot_version=2,
        regeneration_reason="stable_layout",
        targets=[
            InteractionTarget(
                id="target_011",
                label="cancel",
                bbox=[950, 220, 40, 24],
                center=[970, 232],
                source="dom",
                confidence=1.0,
            ),
            InteractionTarget(
                id="target_017",
                label="brewery [hand-card-2]",
                bbox=[975, 255, 60, 26],
                center=[1005, 268],
                source="dom",
                confidence=1.0,
            ),
        ],
    )
    tac_ctx = mem.build_tactical_context(
        2,
        0.0,
        interaction_map=imap,
        disabled_labels=["Sell", "Burton-on-Trent", "brewery"],
    )
    assert 'label="brewery [hand-card-2]" ⚠️ [NO-OP IN THIS PROMPT' in tac_ctx
    assert "UNPLAYABLE / NO-OP OPTIONS IN CURRENT PROMPT" in tac_ctx
    assert 'DISABLED / CURRENTLY ILLEGAL CONTROLS ON SCREEN' in tac_ctx
    assert '"Burton-on-Trent"' in tac_ctx

    # Ensure strategic context remains 100% clean of UI quarantine flags
    strat_ctx = mem.build_strategic_context(2)
    assert "NO-OP" not in strat_ctx
    assert "DISABLED / CURRENTLY ILLEGAL" not in strat_ctx

    # Screen transition >= 0.35% clears the quarantine automatically
    mem.record_previous_action_effect(1.2)
    assert mem.current_prompt_noop_labels == []



def test_brass_rl_provider_config_without_api_key():
    from pydantic import ValidationError

    # Standalone brass_rl requires NO API key
    rl_cfg = GameConfig(
        provider="brass_rl",
        api_key="",
        model="brass-ppo-best",
        url="https://solo.kdc.sh/brass-birmingham/",
    )
    assert rl_cfg.provider == "brass_rl"
    assert rl_cfg.api_key == ""

    # Cloud LLM providers still strictly require a non-empty API key
    with pytest.raises(ValidationError):
        GameConfig(
            provider="openai",
            api_key="   ",
            model="gpt-4o",
            url="https://solo.kdc.sh/brass-birmingham/",
        )


def test_brass_rl_episode_sqlite_persistence(temp_db):
    temp_db.record_rl_episode(
        run_id="run01",
        episode_num=1,
        your_vp=98,
        automa_vp=112,
        won=False,
        total_reward=4.25,
        steps=39,
        policy_loss=0.12,
        value_loss=0.34,
        entropy=1.8,
        checkpoint_tag="latest",
    )
    temp_db.record_rl_episode(
        run_id="run01",
        episode_num=2,
        your_vp=136,
        automa_vp=118,
        won=True,
        total_reward=9.15,
        steps=39,
        policy_loss=0.09,
        value_loss=0.22,
        entropy=1.6,
        checkpoint_tag="best",
    )
    summary = temp_db.get_rl_training_summary(limit=20)
    assert summary["total_episodes_recorded"] == 2
    assert summary["best_vp"] == 136
    assert summary["rolling_avg_vp"] == 117.0
    assert summary["rolling_win_rate_pct"] == 50.0
    assert len(summary["episodes"]) == 2
    assert summary["episodes"][-1]["won"] is True


def test_brass_rl_features_and_actor_critic_ppo_step(tmp_path):
    import torch
    from rl.brass_features import encode_action_list, encode_state
    from rl.brass_env import compute_shaped_step_reward
    from rl.model import (
        ACTION_DIM,
        STATE_DIM,
        BrassActorCritic,
        load_brass_checkpoint,
        save_brass_checkpoint,
    )
    from rl.ppo_trainer import compute_gae, ppo_update_step

    mock_state = {
        "era": "canal",
        "turn": 3,
        "actions_left": 2,
        "game_over": False,
        "money": 18,
        "income_space": 12,
        "your_vp": 0,
        "automa_vp": 0,
        "draw_pile": 9,
        "coal_cubes": 12,
        "iron_cubes": 7,
        "hand": [
            {"kind": "location", "city": "birmingham"},
            {"kind": "industry", "industries": ["cotton"]},
            {"kind": "wildLocation"},
        ],
        "mat": {
            "cotton": [2, 2, 3, 3],
            "goods": [1, 2, 1, 1, 2, 1, 1, 2],
            "pottery": [1, 1, 1, 1, 1],
            "coal": [1, 2, 2, 2],
            "iron": [0, 1, 1, 1],
            "brewery": [1, 2, 2, 1],
        },
        "board": {
            "birmingham": [
                {"owner": 0, "industry": "cotton", "level": 1, "flipped": False, "resources": 0},
                None,
                None,
                None,
            ],
            "farmSouth": [
                {"owner": 0, "industry": "brewery", "level": 1, "flipped": False, "resources": 1}
            ],
        },
        "links": {"birmingham-worcester": 0, "gloucester-worcester": 0},
        "merchants": [
            {"id": "gloucester", "tiles": ["cotton", "any"], "beer": [True, False]},
        ],
        "availability": {"build": True, "network": True, "sell": True, "develop": True, "loan": True, "scout": True},
    }

    candidates = [
        {
            "action_type": "sell",
            "card_slot": 0,
            "sales": [{"city": "birmingham", "industry": "cotton", "level": 1, "beer_needed": 1}],
            "cost": 0,
        },
        {
            "action_type": "develop",
            "card_slot": 0,
            "industries": ["cotton", "coal"],
            "cost": 4,
        },
        {
            "action_type": "network",
            "card_slot": 0,
            "link_id": "birmingham-coventry",
            "cost": 3,
        },
    ]

    s_vec = encode_state(mock_state)
    a_vecs = encode_action_list(mock_state, candidates)
    assert s_vec.shape == (STATE_DIM,)
    assert a_vecs.shape == (3, ACTION_DIM)

    # Flipping Cotton L1 (+5 VP, +5 income spaces, +1 link dot on birmingham-worcester) yields positive shaped reward
    next_state = {
        **mock_state,
        "income_space": 17,
        "board": {
            **mock_state["board"],
            "birmingham": [
                {"owner": 0, "industry": "cotton", "level": 1, "flipped": True, "resources": 0},
                None,
                None,
                None,
            ],
        },
    }
    rew = compute_shaped_step_reward(mock_state, next_state, candidates[0])
    assert rew > 0.3

    dev = torch.device("cpu")
    model = BrassActorCritic(STATE_DIM, ACTION_DIM, hidden_dim=128).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    logits, val = model.forward_single(torch.from_numpy(s_vec), torch.from_numpy(a_vecs))
    assert logits.shape == (3,)
    assert val.dim() == 0

    advs, rets = compute_gae([rew, 0.5], [float(val.item()), 0.2], [False, True])
    transitions = [
        {
            "state": s_vec,
            "actions": a_vecs,
            "action_idx": 0,
            "log_prob": float(torch.log_softmax(logits, dim=-1)[0].item()),
            "value": float(val.item()),
            "advantage": float(advs[0]),
            "return_val": float(rets[0]),
        },
        {
            "state": encode_state(next_state),
            "actions": a_vecs[:2],
            "action_idx": 1,
            "log_prob": -0.7,
            "value": 0.2,
            "advantage": float(advs[1]),
            "return_val": float(rets[1]),
        },
    ]
    losses = ppo_update_step(model, opt, transitions, dev, ppo_epochs=2, minibatch_size=2)
    assert "policy_loss" in losses and "value_loss" in losses

    ckpt_file = tmp_path / "test_brass_ppo.pt"
    save_brass_checkpoint(model, opt, ckpt_file, metadata={"rolling_avg_vp": 124.5})
    loaded_model, meta = load_brass_checkpoint(ckpt_file, device=dev)
    assert meta["rolling_avg_vp"] == 124.5
    logits2, _ = loaded_model.forward_single(torch.from_numpy(s_vec), torch.from_numpy(a_vecs))
    assert logits2.shape == (3,)
