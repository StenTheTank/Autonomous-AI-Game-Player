import asyncio
import time
import traceback
import uuid
from typing import Callable, Any, Optional
from PIL import Image

from models.schemas import Action, GameConfig, LLMResponse, SessionState, SessionStats
from browser.playwright_manager import BrowserManager
from agent.llm import LLMManager, LLMCallTelemetry
from agent.memory import SessionMemory
from storage.database import SessionDatabase
from vision.grounding import VisualGroundingEngine
from rl.rl_player import BrassRLAgentPlayer
from rl.brass_bridge import (
    enumerate_legal_actions,
    execute_macro_action,
    extract_brass_state,
    start_new_brass_game,
    _resolve_element_center,
)
from rl.brass_features import (
    coal_market_price,
    compute_board_summary,
    income_space_to_pounds,
    iron_market_price,
)


class AgentLoop:
    def __init__(
        self,
        config: GameConfig,
        on_update: Callable[[dict], Any],
        db: Optional[SessionDatabase] = None,
    ):
        self.config = config
        self.on_update = on_update
        self.db = db or SessionDatabase()
        self.session_id = uuid.uuid4().hex[:12]

        self.browser = BrowserManager(
            headless=config.headless,
            max_action_duration_ms=config.max_action_duration_ms,
        )
        self.is_brass_human_vs_ai = config.match_mode == "brass_human_vs_ai"
        self.is_brass_duel = config.match_mode in ("brass_2p_duel", "brass_human_vs_ai")
        self.is_brass_rl = config.provider == "brass_rl" and not self.is_brass_duel

        if self.is_brass_human_vs_ai:
            self.llm = None if config.provider == "brass_rl" else LLMManager(config)
            self.rl_player: Optional[BrassRLAgentPlayer] = (
                BrassRLAgentPlayer(checkpoint_preference=config.model or "brass-ppo-best")
                if config.provider == "brass_rl"
                else None
            )
        elif config.match_mode == "brass_2p_duel":
            self.llm = LLMManager(config)
            self.rl_player = BrassRLAgentPlayer(
                checkpoint_preference=config.opponent_model or "brass-ppo-best"
            )
        elif self.is_brass_rl:
            self.llm = None
            self.rl_player = BrassRLAgentPlayer(checkpoint_preference=config.model)
        elif config.enable_rl_advisor and "brass" in (config.url or "").lower():
            self.llm = LLMManager(config)
            self.rl_player = BrassRLAgentPlayer(checkpoint_preference=config.model)
        else:
            self.llm = LLMManager(config)
            self.rl_player = None

        self._duel_initialized: bool = False
        self._page_lock = asyncio.Lock()
        self.memory = SessionMemory(config)
        self.grounding = VisualGroundingEngine()

        self.state = SessionState(
            session_id=self.session_id,
            status="starting",
            browser_status="launching",
            current_url=config.url,
            game_name=config.game_name,
            model=config.model,
            provider=config.provider,
            current_objective=config.objective,
            current_subgoal=self.memory.current_subgoal,
            current_strategy=self.memory.current_strategy,
            sub_goals=list(self.memory.sub_goals),
            stats=SessionStats(
                max_iterations=config.max_iterations,
                max_budget_usd=config.max_budget_usd,
            ),
        )

        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Set = unpaused / running

        self._active_started_at: Optional[float] = None
        self._accumulated_elapsed: float = 0.0

    def _update_elapsed(self) -> float:
        total = self._accumulated_elapsed
        if self._active_started_at is not None:
            total += time.monotonic() - self._active_started_at
        self.state.stats.elapsed_seconds = round(total, 1)
        return self.state.stats.elapsed_seconds

    def _record_telemetry(self, tel: LLMCallTelemetry) -> None:
        self.state.stats.api_calls += tel.api_calls
        self.state.stats.prompt_tokens += tel.prompt_tokens
        self.state.stats.completion_tokens += tel.completion_tokens
        self.state.stats.total_tokens += tel.total_tokens
        self.state.stats.estimated_cost_usd = round(
            self.state.stats.estimated_cost_usd + tel.cost_usd, 6
        )

    async def _interruptible_sleep(self, duration_sec: float) -> None:
        if duration_sec <= 0:
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=duration_sec)
        except asyncio.TimeoutError:
            pass

    async def _wait_if_paused(self) -> bool:
        """Wait while paused. Returns False if stop was requested."""
        if not self._pause_event.is_set():
            await self._pause_event.wait()
        return not self._stop_event.is_set()

    async def start(self) -> None:
        self.db.create_session(
            self.session_id,
            self.config.sanitized_dict(),
            self.state.stats.model_dump(),
        )
        self._active_started_at = time.monotonic()

        try:
            self.state.status = "starting"
            self.state.status_reason = "Launching Chromium and loading game..."
            await self._notify_state()

            await self.browser.start()
            self.state.browser_status = "navigating"
            await self._notify_state()

            nav_url = self.config.url
            if self.is_brass_human_vs_ai and "hideSeat=" not in nav_url:
                ai_seat = 1 - int(self.config.human_player_seat)
                sep = "&" if "?" in nav_url else "?"
                nav_url = f"{nav_url}{sep}hideSeat={ai_seat}"
            await self.browser.navigate(nav_url)
            self.state.browser_status = "connected"
            self.state.status = "running"
            self.state.status_reason = "Game loaded. Starting observation loop."
            await self._notify_state()

            # Allow initial canvas/assets to settle
            await self._interruptible_sleep(1.2)

            while not self._stop_event.is_set():
                if not await self._wait_if_paused():
                    break

                # Guardrail 1: Max iterations
                if self.state.stats.current_iteration >= self.config.max_iterations:
                    self.state.status = "completed"
                    self.state.status_reason = f"Reached maximum iterations ({self.config.max_iterations})."
                    break

                # Guardrail 2: Max session duration
                if self._update_elapsed() >= self.config.max_session_duration_sec:
                    self.state.status = "completed"
                    self.state.status_reason = f"Reached maximum session duration ({self.config.max_session_duration_sec}s)."
                    break

                # Guardrail 3: Max API budget
                if self.state.stats.estimated_cost_usd >= self.config.max_budget_usd:
                    self.state.status = "stopped"
                    self.state.status_reason = f"Reached API cost budget limit (${self.config.max_budget_usd:.2f})."
                    break

                # Ensure browser hasn't been closed or crashed
                alive = await self.browser.ensure_alive()
                if not alive:
                    self.state.browser_status = "recovered"
                    await self._notify_state(
                        event_note="Browser window was closed or crashed; recovered session."
                    )

                # Safety check: CAPTCHA / anti-bot gate detection (never bypass!)
                gate = await self.browser.detect_access_gate()
                if gate:
                    await self.pause(
                        reason=f"{gate}. Paused automatically — please complete verification manually in the browser window, then click Resume."
                    )
                    if not await self._wait_if_paused():
                        break

                self.state.stats.current_iteration += 1
                iteration = self.state.stats.current_iteration

                # 1. Hybrid Observation & Interaction Map Pipeline
                #    (a) Inspect DOM & Accessibility Tree + viewport/scroll/dialog metadata
                dom_elements = []
                layout_meta = {
                    "url": self.state.current_url,
                    "viewport_w": 1280,
                    "viewport_h": 720,
                    "dpr": 1.0,
                    "scroll_x": 0,
                    "scroll_y": 0,
                    "has_canvas": False,
                    "canvas_rects": [],
                    "dialog_signature": "",
                }
                inspected = await self.browser.inspect_page_and_dom()
                if isinstance(inspected, tuple) and len(inspected) == 2 and isinstance(inspected[0], list):
                    dom_elements, layout_meta = inspected
                else:
                    dom_elements = await self.browser.scan_interactive_elements()

                #    (b) Capture screenshot & compute visual delta %
                ui_b64, llm_b64, delta_pct = await self.browser.capture_observation(
                    add_grid_for_llm=self.config.show_coordinate_grid_to_llm,
                    interactive_elements=dom_elements,
                )
                if not ui_b64:
                    await self._interruptible_sleep(0.5)
                    continue

                mx, my = self.browser.mouse_pos
                self.state.mouse_x = mx
                self.state.mouse_y = my
                self.state.stats.last_visual_delta_pct = delta_pct
                self.memory.record_previous_action_effect(delta_pct)
                if self.state.click_history and self.state.click_history[-1].get("visual_delta_pct") is None:
                    self.state.click_history[-1]["visual_delta_pct"] = delta_pct

                if self.browser.page and not self.browser.page.is_closed():
                    self.state.current_url = self.browser.page.url
                    layout_meta["url"] = self.state.current_url

                #    (c) Check layout/visual staleness & run CV + Vision Grounding when needed
                self.grounding.next_screenshot_version()
                needs_regen, regen_reason = self.grounding.check_regeneration_needed(
                    layout_meta, dom_elements, delta_pct
                )

                raw_img = getattr(self.browser, "raw_image", None)
                has_pil_frame = isinstance(raw_img, Image.Image)

                cv_targets = []
                # Run Computer Vision hotspot detector ONLY on true bitmap <canvas> games (or pure canvas pages),
                # never on <svg> boards where DOM/SVG inspection already extracts exact interactive nodes without
                # false-positives on static decorative circles (e.g. number tokens or port badges).
                if (
                    has_pil_frame
                    and self.config.enable_cv_detection
                    and layout_meta.get("has_bitmap_canvas", False)
                ):
                    cv_targets = self.grounding.detect_cv_targets(
                        raw_img,
                        dom_targets=dom_elements,
                        canvas_rects=layout_meta.get("canvas_rects"),
                    )

                vision_targets = None
                # Only invoke the separate LLM visual-grounding pass on true bitmap <canvas> games
                # when local DOM/SVG/pointer-cursor + CV detectors found fewer than 4 targets.
                if (
                    not self.is_brass_rl
                    and not self.is_brass_duel
                    and has_pil_frame
                    and self.config.enable_vision_grounding
                    and layout_meta.get("has_bitmap_canvas", layout_meta.get("has_canvas"))
                    and not layout_meta.get("has_svg")
                    and (len(dom_elements) + len(cv_targets) < 4)
                    and needs_regen
                    and self.state.stats.estimated_cost_usd < self.config.max_budget_usd
                ):
                    self.state.status_reason = (
                        f"Building Visual Interaction Map (v{self.grounding.screenshot_version} · {regen_reason})..."
                    )
                    await self._notify_state(image=ui_b64)
                    try:
                        grounding_img_b64 = self.grounding.render_vision_grounding_image(raw_img, cv_targets)
                        grounding_hint = (
                            f"Active Sub-Goal: {self.memory.current_subgoal}\n"
                            f"Overall Strategy: {self.memory.current_strategy}\n"
                            f"Computer-Vision Candidate Hotspots: {len(cv_targets)}"
                        )
                        vis_resp, vis_tel = await self.llm.get_visual_grounding(grounding_img_b64, grounding_hint)
                        self._record_telemetry(vis_tel)
                        vision_targets = self.grounding.resolve_vision_detections(vis_resp, cv_targets)
                    except Exception:
                        # Fall back cleanly to DOM + CV targets if optional vision grounding pass errors
                        vision_targets = None

                #    (d) Fuse DOM + Vision + CV into unified InteractionMap
                interaction_map = self.grounding.build_interaction_map(
                    dom_targets=dom_elements,
                    cv_targets=cv_targets,
                    vision_targets=vision_targets,
                    layout_meta=layout_meta,
                    regenerated=needs_regen,
                    regeneration_reason=regen_reason,
                )
                self.state.interaction_map = interaction_map.model_dump()
                self.state.detected_elements_count = len(interaction_map.targets)

                if has_pil_frame:
                    llm_b64 = self.grounding.annotate_screenshot_for_llm(
                        raw_img,
                        interaction_map=interaction_map,
                        mouse_pos=self.browser.mouse_pos,
                        show_grid=self.config.show_coordinate_grid_to_llm,
                    )

                # Send live frame & updated Interaction Map immediately
                await self._notify_state(image=ui_b64)

                # Head-to-Head 2-Player Brass Duel Mode (API Challenger vs Local Trained PPO)
                if self.is_brass_duel:
                    should_stop = await self._run_brass_duel_turn(iteration, ui_b64, delta_pct)
                    if should_stop:
                        break
                    await self._interruptible_sleep(self.config.observation_interval_ms / 1000.0)
                    continue

                # Standalone Local PyTorch Brass RL Policy Mode ($0 API cost)
                if self.is_brass_rl:
                    should_stop = await self._run_brass_rl_turn(iteration, delta_pct)
                    if should_stop:
                        break
                    await self._interruptible_sleep(self.config.observation_interval_ms / 1000.0)
                    continue

                rl_hint: Optional[str] = None
                if self.rl_player is not None and self.browser.page and not self.browser.page.is_closed():
                    rl_hint = await self.rl_player.get_advisor_hint(self.browser.page)

                # 2. Strategic Planning Phase (periodic or when stagnant)
                if self.memory.needs_strategic_replan(iteration) and not self._stop_event.is_set():
                    self.state.status = "planning"
                    self.state.status_reason = f"Updating high-level strategy (Step {iteration})..."
                    await self._notify_state()

                    try:
                        # Pass clean unannotated `ui_b64` (NO #001 badges, cursor HUD, or A1..P9 grid)
                        # so Strategic Planning evaluates pure board/game state without UI-target bias!
                        plan, plan_tel = await self.llm.get_strategic_plan(
                            ui_b64, self.memory.build_strategic_context(iteration, rl_advisor_hint=rl_hint)
                        )
                        self._record_telemetry(plan_tel)
                        self.memory.apply_strategic_plan(plan)
                        self._sync_memory_to_state()
                        self.state.stats.consecutive_failures = 0
                        await self._notify_state(
                            strategic_plan=plan.model_dump(),
                        )
                    except Exception as plan_err:
                        # Non-fatal: log strategic error and proceed to tactical execution
                        self.state.stats.consecutive_failures += 1
                        await self._notify_state(error=f"Strategic planning warning: {plan_err}")
                        if self.state.stats.consecutive_failures >= self.config.max_consecutive_failures:
                            self.state.status = "error"
                            self.state.status_reason = (
                                f"Halted after {self.state.stats.consecutive_failures} consecutive API failures."
                            )
                            break

                if not await self._wait_if_paused():
                    break

                # Budget re-check before tactical call
                if self.state.stats.estimated_cost_usd >= self.config.max_budget_usd:
                    self.state.status = "stopped"
                    self.state.status_reason = f"Reached API cost budget limit (${self.config.max_budget_usd:.2f})."
                    break

                # 3. Tactical Decision Phase
                self.state.status = "running"
                self.state.status_reason = f"Analyzing frame #{iteration} ({len(interaction_map.targets)} targets mapped)..."
                tactical_ctx = self.memory.build_tactical_context(
                    iteration,
                    delta_pct,
                    mouse_pos=self.browser.mouse_pos,
                    interactive_elements=dom_elements,
                    interaction_map=interaction_map,
                    disabled_labels=layout_meta.get("disabled_labels"),
                    rl_advisor_hint=rl_hint,
                )

                try:
                    llm_resp, tac_tel = await self.llm.get_next_action(llm_b64, tactical_ctx)
                    self._record_telemetry(tac_tel)
                    self.state.stats.consecutive_failures = 0
                except Exception as tac_err:
                    self.state.stats.consecutive_failures += 1
                    await self._notify_state(error=str(tac_err))
                    if self.state.stats.consecutive_failures >= self.config.max_consecutive_failures:
                        self.state.status = "error"
                        self.state.status_reason = (
                            f"Halted after {self.state.stats.consecutive_failures} consecutive API failures: {tac_err}"
                        )
                        break
                    await self._interruptible_sleep(1.5)
                    continue

                if not await self._wait_if_paused():
                    break

                # 4. Resolve Target IDs / Coordinates, Enforce Freshness & Confidence, and Execute
                action_dict = llm_resp.action.model_dump()
                zoom_note: Optional[str] = None
                zoom_image_b64: Optional[str] = None
                skip_execution_reason: Optional[str] = None
                resolved_target_meta: Optional[dict] = None
                step_tokens = tac_tel.total_tokens
                step_cost_usd = tac_tel.cost_usd

                # Resolve optional drag start target ID
                if action_dict.get("from_target_id"):
                    start_t = self.grounding.lookup_target(action_dict["from_target_id"])
                    if start_t is not None:
                        action_dict["from_x"], action_dict["from_y"] = start_t.center
                        llm_resp.action.from_x, llm_resp.action.from_y = start_t.center

                act_type = action_dict.get("type")
                req_target_id = action_dict.get("target_id")

                if req_target_id and act_type in (
                    "click_target",
                    "hover_target",
                    "mouse_click",
                    "mouse_move",
                    "mouse_drag",
                ):
                    matched_target = self.grounding.lookup_target(req_target_id)
                    if matched_target is None or matched_target.screenshot_version != interaction_map.screenshot_version:
                        if action_dict.get("x") is None or action_dict.get("y") is None:
                            skip_execution_reason = (
                                f"Target '{req_target_id}' not found in current Interaction Map "
                                f"(v{interaction_map.screenshot_version}); refreshing observation"
                            )
                            self.grounding.invalidate("missing_target_id")
                    else:
                        resolved_target_meta = matched_target.model_dump()
                        tx, ty = matched_target.center[0], matched_target.center[1]

                        if matched_target.source == "dom":
                            reverified = await self.browser.reverify_dom_target(
                                matched_target.label, matched_target.center
                            )
                            if isinstance(reverified, tuple) and len(reverified) == 2:
                                live_xy, dom_status = reverified
                                if live_xy is None:
                                    skip_execution_reason = (
                                        f"DOM target {matched_target.id} ('{matched_target.label[:28]}') "
                                        f"became stale or occluded ({dom_status}); regenerating Interaction Map"
                                    )
                                    self.grounding.invalidate("stale_dom_target")
                                else:
                                    tx, ty = live_xy
                            if not skip_execution_reason:
                                action_dict["x"], action_dict["y"] = tx, ty
                                llm_resp.action.x, llm_resp.action.y = tx, ty
                                zoom_note = (
                                    f"Target {matched_target.id} [dom · {int(matched_target.confidence * 100)}%] "
                                    f"\"{matched_target.label[:26]}\" -> ({tx},{ty})"
                                )
                        else:
                            # Vision or CV target on Canvas/WebGL:
                            # CV contours already have pixel-exact centroids; only zoom-verify low-conf or pure vision estimates
                            needs_zoom_verify = (
                                act_type in ("click_target", "mouse_click")
                                and self.config.precision_zoom_click
                                and has_pil_frame
                                and layout_meta.get("has_bitmap_canvas", False)
                                and (
                                    matched_target.confidence < 0.62
                                    or (
                                        matched_target.source == "vision"
                                        and (matched_target.bbox[2] <= 42 or matched_target.bbox[3] <= 42)
                                    )
                                )
                            )
                            if needs_zoom_verify:
                                self.state.status_reason = (
                                    f"4x Zoom-Verifying {matched_target.id} near ({tx}, {ty})..."
                                )
                                await self._notify_state()
                                await self.browser.refresh_raw_frame()
                                zoom_b64, crop_x0, crop_y0 = self.browser.build_zoom_crop_b64(tx, ty)
                                if zoom_b64:
                                    zoom_image_b64 = zoom_b64
                                    target_desc = (
                                        f"Target {matched_target.id}: {matched_target.label}\n"
                                        f"Reasoning: {llm_resp.reasoning_summary}\n"
                                        f"Expected: {llm_resp.expected_result}"
                                    )
                                    try:
                                        zoom_ref, zoom_tel = await self.llm.get_zoom_refinement(
                                            zoom_b64, target_desc
                                        )
                                        self._record_telemetry(zoom_tel)
                                        step_tokens += zoom_tel.total_tokens
                                        step_cost_usd = round(step_cost_usd + zoom_tel.cost_usd, 6)
                                        if zoom_ref.target_visible:
                                            tx, ty = self.browser.resolve_zoom_cell(
                                                crop_x0, crop_y0, zoom_ref.cell, zoom_ref.sub_pos
                                            )
                                            zoom_note = (
                                                f"Target {matched_target.id} [{matched_target.source} · "
                                                f"{int(matched_target.confidence * 100)}%] + 4x Snap -> ({tx},{ty})"
                                            )
                                        else:
                                            skip_execution_reason = (
                                                f"Target {matched_target.id} not confirmed inside 4x crop; "
                                                "requesting fresh observation instead of blind click"
                                            )
                                            self.grounding.invalidate("zoom_verification_rejected")
                                    except Exception:
                                        pass

                            if not skip_execution_reason:
                                if matched_target.confidence < self.config.min_target_confidence and not zoom_note:
                                    skip_execution_reason = (
                                        f"Target {matched_target.id} confidence ({matched_target.confidence:.2f}) "
                                        f"below threshold ({self.config.min_target_confidence:.2f}); refreshing observation"
                                    )
                                    self.grounding.invalidate("low_target_confidence")
                                else:
                                    action_dict["x"], action_dict["y"] = tx, ty
                                    llm_resp.action.x, llm_resp.action.y = tx, ty
                                    if not zoom_note:
                                        zoom_note = (
                                            f"Target {matched_target.id} [{matched_target.source} · "
                                            f"{int(matched_target.confidence * 100)}%] -> ({tx},{ty})"
                                        )

                elif (
                    act_type in ("click_coordinates", "mouse_click")
                    and action_dict.get("x") is not None
                    and action_dict.get("y") is not None
                ):
                    # Step A: Smart DOM/SVG & 0..1000 coordinate verification before clicking!
                    is_gemini_model = (
                        self.config.provider == "gemini"
                        or "gemini" in (self.config.model or "").lower()
                    )
                    hint_for_snap = f"{llm_resp.expected_result} | {llm_resp.reasoning_summary} | {llm_resp.current_subgoal or ''}"
                    smart_res = await self.browser.resolve_smart_click_coordinates(
                        int(action_dict["x"]),
                        int(action_dict["y"]),
                        hint_text=hint_for_snap,
                        is_gemini=is_gemini_model,
                    )
                    if isinstance(smart_res, tuple) and len(smart_res) == 3:
                        sx, sy, smart_note = smart_res
                        action_dict["x"], action_dict["y"] = int(sx), int(sy)
                        llm_resp.action.x, llm_resp.action.y = int(sx), int(sy)
                        if smart_note:
                            zoom_note = smart_note

                    # Step B: 4x Magnifier Zoom-Snap if point still lands on <canvas> or bare <svg> background
                    if (
                        self.config.precision_zoom_click
                        and has_pil_frame
                        and await self.browser.should_zoom_refine(
                            action_dict["x"], action_dict["y"], element_id=None
                        )
                    ):
                        coarse_x, coarse_y = int(action_dict["x"]), int(action_dict["y"])
                        self.state.status_reason = f"4x Zoom-Snapping coordinates ({coarse_x}, {coarse_y})..."
                        await self._notify_state()

                        await self.browser.refresh_raw_frame()
                        zoom_b64, crop_x0, crop_y0 = self.browser.build_zoom_crop_b64(coarse_x, coarse_y)
                        if zoom_b64:
                            zoom_image_b64 = zoom_b64
                            target_desc = (
                                f"Sub-goal: {llm_resp.current_subgoal or 'N/A'}\n"
                                f"Reasoning: {llm_resp.reasoning_summary}\n"
                                f"Expected target: {llm_resp.expected_result}"
                            )
                            try:
                                zoom_ref, zoom_tel = await self.llm.get_zoom_refinement(zoom_b64, target_desc)
                                self._record_telemetry(zoom_tel)
                                step_tokens += zoom_tel.total_tokens
                                step_cost_usd = round(step_cost_usd + zoom_tel.cost_usd, 6)

                                if zoom_ref.target_visible:
                                    rx, ry = self.browser.resolve_zoom_cell(
                                        crop_x0, crop_y0, zoom_ref.cell, zoom_ref.sub_pos
                                    )
                                    snap_msg = (
                                        f"4x Zoom-Snap ({coarse_x},{coarse_y}) -> "
                                        f"{zoom_ref.cell}:{zoom_ref.sub_pos} ({rx},{ry})"
                                    )
                                    zoom_note = f"{zoom_note} + {snap_msg}" if zoom_note else snap_msg
                                    action_dict["x"] = rx
                                    action_dict["y"] = ry
                                    llm_resp.action.x = rx
                                    llm_resp.action.y = ry
                                else:
                                    action_dict["type"] = "mouse_move"
                                    llm_resp.action.type = "mouse_move"
                                    zoom_note = (
                                        f"Target not inside 4x crop ({zoom_ref.observation[:75]}); "
                                        f"moved cursor to ({coarse_x},{coarse_y}) without blind click"
                                    )
                            except Exception as zoom_err:
                                zoom_note = f"Zoom-Snap fallback ({str(zoom_err)[:55]}); clicked ({coarse_x},{coarse_y})"

                if not await self._wait_if_paused():
                    break

                if skip_execution_reason:
                    exec_status = skip_execution_reason
                else:
                    exec_status = await self.browser.execute_action(
                        action_dict,
                        default_duration_ms=self.memory.adjusted_action_duration_ms,
                    )
                    if zoom_note:
                        exec_status = f"{exec_status} [{zoom_note}]"

                    # Record executed click into click_history for Visual Debugging UI
                    if action_dict.get("type") in ("click_target", "click_coordinates", "mouse_click") and (
                        action_dict.get("x") is not None and action_dict.get("y") is not None
                    ):
                        self.state.click_history.append(
                            {
                                "iteration": iteration,
                                "x": int(action_dict["x"]),
                                "y": int(action_dict["y"]),
                                "target_id": action_dict.get("target_id"),
                                "label": (resolved_target_meta or {}).get("label") or llm_resp.expected_result[:40],
                                "source": (resolved_target_meta or {}).get("source", "coordinates"),
                                "confidence": (resolved_target_meta or {}).get("confidence"),
                                "visual_delta_pct": None,
                                "timestamp": round(time.time(), 3),
                            }
                        )
                        if len(self.state.click_history) > 10:
                            self.state.click_history = self.state.click_history[-10:]

                self.state.mouse_x, self.state.mouse_y = self.browser.mouse_pos
                self.memory.record_tactical_step(
                    iteration,
                    llm_resp,
                    exec_status,
                    target_label=(resolved_target_meta or {}).get("label"),
                )
                self._sync_memory_to_state()

                self.state.last_observation = llm_resp.observation
                self.state.last_reasoning = llm_resp.reasoning_summary
                self.state.last_expected_result = llm_resp.expected_result
                self.state.last_action = action_dict
                self.state.last_action_status = exec_status
                self.state.stats.total_actions += 1
                self._update_elapsed()

                # Persist step & session update to SQLite
                self.db.record_step(
                    session_id=self.session_id,
                    iteration=iteration,
                    observation=llm_resp.observation,
                    reasoning=llm_resp.reasoning_summary,
                    action=action_dict,
                    expected_result=llm_resp.expected_result,
                    evaluation=llm_resp.previous_action_evaluation or "",
                    visual_delta_pct=delta_pct,
                    goal_status=llm_resp.goal_status,
                    tokens_used=step_tokens,
                    cost_usd=step_cost_usd,
                )
                self.db.update_session(
                    session_id=self.session_id,
                    status=self.state.status,
                    stats=self.state.stats.model_dump(),
                    status_reason=self.state.status_reason,
                    strategy=self.state.current_strategy,
                    lessons=self.state.lessons_learned,
                )

                step_log_payload = {
                    **llm_resp.model_dump(),
                    "action": action_dict,
                    "iteration": iteration,
                    "timestamp": time.time(),
                    "execution_status": exec_status,
                    "visual_delta_pct": delta_pct,
                    "step_tokens": step_tokens,
                    "step_cost_usd": step_cost_usd,
                }
                if zoom_note:
                    step_log_payload["zoom_note"] = zoom_note
                if zoom_image_b64:
                    step_log_payload["zoom_image"] = zoom_image_b64
                await self._notify_state(llm_response=step_log_payload)

                # 5. Check Objective Completion
                if llm_resp.goal_status == "success" or llm_resp.action.type == "declare_success":
                    self.state.status = "completed"
                    self.state.status_reason = "Agent achieved the game objective!"
                    self.state.stats.progress_percent = 100
                    break

                if llm_resp.goal_status == "failure" or llm_resp.action.type == "declare_failure":
                    self.state.status = "failed"
                    self.state.status_reason = f"Agent declared failure: {llm_resp.reasoning_summary[:140]}"
                    break

                # 6. Observation cadence wait before next screenshot
                await self._interruptible_sleep(self.config.observation_interval_ms / 1000.0)

            # Capture one final frame before closing
            if self.browser.page and not self.browser.page.is_closed():
                try:
                    final_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)
                    if final_ui:
                        await self._notify_state(image=final_ui)
                except Exception:
                    pass

            if self._stop_event.is_set() and self.state.status in ("running", "planning", "paused", "starting"):
                self.state.status = "stopped"
                self.state.status_reason = "Stopped by user."

        except asyncio.CancelledError:
            self.state.status = "stopped"
            self.state.status_reason = "Session cancelled."
        except Exception as e:
            traceback.print_exc()
            self.state.status = "error"
            self.state.status_reason = str(e)
            await self._notify_state(error=str(e))
        finally:
            self._update_elapsed()
            self._active_started_at = None
            self.state.browser_status = "closed"
            await self.browser.stop()
            self.db.update_session(
                session_id=self.session_id,
                status=self.state.status,
                stats=self.state.stats.model_dump(),
                status_reason=self.state.status_reason,
                strategy=self.state.current_strategy,
                lessons=self.state.lessons_learned,
            )
            await self._notify_state()

    def _sync_memory_to_state(self) -> None:
        self.state.current_strategy = self.memory.current_strategy
        self.state.current_subgoal = self.memory.current_subgoal
        self.state.sub_goals = list(self.memory.sub_goals)
        self.state.known_mechanics = list(self.memory.known_mechanics)
        self.state.lessons_learned = list(self.memory.lessons_learned)
        self.state.stats.progress_percent = self.memory.progress_percent

    async def _run_brass_rl_turn(self, iteration: int, delta_pct: float) -> bool:
        """
        Execute one decision of the local PyTorch BrassActorCritic policy in the Live Arena.
        Returns True if the session has reached terminal completion.
        """
        assert self.rl_player is not None
        page = self.browser.page
        if not page or page.is_closed():
            return False

        eval_res = await self.rl_player.evaluate_board(
            page,
            greedy=(self.config.temperature <= 0.15),
            temperature=self.config.temperature,
        )
        mode = eval_res.get("mode")
        b_state = eval_res.get("state") or {}

        if mode == "setup_modal":
            xy = await _resolve_element_center(page, '[data-testid="start-game"]')
            tx, ty = xy if xy else (640, 325)
            exec_status = await self.browser.execute_action(
                {"type": "click_coordinates", "x": tx, "y": ty, "button": "left", "clicks": 1},
                default_duration_ms=100,
            )
            action_dict = {"type": "click_coordinates", "x": tx, "y": ty}
            resp = LLMResponse(
                observation="New solo game setup screen detected on Brass: Birmingham.",
                reasoning_summary=f"Launching match using local PyTorch policy ({self.rl_player.checkpoint_source}).",
                action=Action(type="click_coordinates", x=tx, y=ty),
                expected_result="Start Canal Era Turn 1.",
                goal_status="in_progress",
                current_subgoal="Initialize Canal Era Round 1",
                progress_percent=2,
            )
        elif mode == "game_over":
            your_vp = int(b_state.get("your_vp", 0))
            automa_vp = int(b_state.get("automa_vp", 0))
            won = your_vp > automa_vp
            verdict = "VICTORY" if won else ("TIE" if your_vp == automa_vp else "COMPLETED")
            obs = f"Game Over! Final Score — You: {your_vp} VP vs Mautoma: {automa_vp} VP ({verdict})."
            reasoning = (
                f"20-round Brass: Birmingham match finished ({your_vp - automa_vp:+d} VP margin) "
                f"using {self.rl_player.checkpoint_source} ($0.00 API cost)."
            )
            action_dict = {"type": "declare_success"}
            exec_status = f"{verdict}: You {your_vp} VP vs Automa {automa_vp} VP"
            resp = LLMResponse(
                observation=obs,
                reasoning_summary=reasoning,
                action=Action(type="declare_success"),
                expected_result="Record final Brass: Birmingham scoreboard.",
                goal_status="success",
                current_subgoal=f"Final Score: {your_vp} VP vs {automa_vp} VP",
                progress_percent=100,
            )
            self.state.status = "completed"
            self.state.status_reason = obs
            self.state.stats.progress_percent = 100
        elif mode == "play" and eval_res.get("chosen_action"):
            chosen = eval_res["chosen_action"]
            summary = eval_res.get("summary") or {}
            era_lbl = str(b_state.get("era", "canal")).capitalize()
            turn_num = int(b_state.get("turn", 1))
            acts_left = int(b_state.get("actions_left", 1))
            prog = min(98, max(5, int(((0 if era_lbl == "Canal" else 10) + turn_num) * 4.8)))

            top_strs = [
                f"#{r['rank']} {r['action']['description']} ({r['prob_pct']:.1f}%)"
                for r in (eval_res.get("ranked_actions") or [])[:3]
            ]
            obs = (
                f"{era_lbl} Era · Turn {turn_num}/10 ({acts_left} act left) · "
                f"£{b_state.get('money', 0)} · Income {summary.get('income_pounds', 0):+d} · "
                f"You {b_state.get('your_vp', 0)} VP (Flipped +{summary.get('flipped_tile_vp', 0)} / Link +{summary.get('connected_link_vp', 0)}) "
                f"vs Automa {b_state.get('automa_vp', 0)} VP"
            )
            reasoning = (
                f"PyTorch PPO Critic V(s)={eval_res.get('value', 0.0):+.2f} [{eval_res.get('checkpoint_source')}] · "
                + " | ".join(top_strs)
            )

            async def _on_substep(xy: tuple[int, int], lbl: str) -> None:
                self.state.mouse_x, self.state.mouse_y = xy
                sub_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)
                if sub_ui:
                    await self._notify_state(image=sub_ui)

            step_delay = max(90, min(400, int(self.config.default_action_duration_ms)))
            exec_res = await execute_macro_action(
                page,
                chosen,
                visual_delay_ms=step_delay,
                browser_mgr=self.browser,
                on_substep=_on_substep,
            )
            lx, ly = exec_res.get("last_xy", (640, 360))
            action_dict = {"type": "click_coordinates", "x": lx, "y": ly}
            exec_status = (
                f"Executed RL macro-move ({exec_res.get('clicks', 1)} UI clicks): {chosen['description']}"
                if exec_res.get("ok")
                else f"RL macro-move fallback on: {chosen['description']}"
            )
            resp = LLMResponse(
                observation=obs,
                reasoning_summary=reasoning,
                action=Action(type="click_coordinates", x=lx, y=ly),
                expected_result=chosen["description"],
                goal_status="in_progress",
                current_subgoal=f"{era_lbl} Era Turn {turn_num}: {chosen['action_type'].capitalize()}",
                progress_percent=prog,
            )
            self.state.current_strategy = (
                f"PyTorch Actor-Critic PPO ({eval_res.get('checkpoint_source')}) · "
                f"State Value V(s) = {eval_res.get('value', 0.0):+.2f}"
            )
            self.memory.current_strategy = self.state.current_strategy
            self.state.click_history.append(
                {
                    "iteration": iteration,
                    "x": lx,
                    "y": ly,
                    "target_id": None,
                    "label": chosen["description"][:42],
                    "source": "dom",
                    "confidence": round(float(eval_res.get("chosen_prob_pct", 100.0)) / 100.0, 2),
                    "visual_delta_pct": None,
                    "timestamp": round(time.time(), 3),
                }
            )
            if len(self.state.click_history) > 10:
                self.state.click_history = self.state.click_history[-10:]
        else:
            await self._interruptible_sleep(0.5)
            return False

        self.state.mouse_x, self.state.mouse_y = self.browser.mouse_pos
        self.memory.record_tactical_step(iteration, resp, exec_status)
        self._sync_memory_to_state()

        self.state.last_observation = resp.observation
        self.state.last_reasoning = resp.reasoning_summary
        self.state.last_expected_result = resp.expected_result
        self.state.last_action = action_dict
        self.state.last_action_status = exec_status
        self.state.stats.total_actions += 1
        self._update_elapsed()

        self.db.record_step(
            session_id=self.session_id,
            iteration=iteration,
            observation=resp.observation,
            reasoning=resp.reasoning_summary,
            action=action_dict,
            expected_result=resp.expected_result,
            evaluation="",
            visual_delta_pct=delta_pct,
            goal_status=resp.goal_status,
            tokens_used=0,
            cost_usd=0.0,
        )
        self.db.update_session(
            session_id=self.session_id,
            status=self.state.status,
            stats=self.state.stats.model_dump(),
            status_reason=self.state.status_reason,
            strategy=self.state.current_strategy,
            lessons=self.state.lessons_learned,
        )

        step_log_payload = {
            **resp.model_dump(),
            "action": action_dict,
            "iteration": iteration,
            "timestamp": time.time(),
            "execution_status": exec_status,
            "visual_delta_pct": delta_pct,
            "step_tokens": 0,
            "step_cost_usd": 0.0,
        }
        await self._notify_state(llm_response=step_log_payload)
        return mode == "game_over"

    def _duel_seat_meta(self) -> list[dict]:
        """Return [{'role': 'human'|'api'|'ppo', 'label': ..., 'model_name': ...}] for Seats 0 and 1."""
        prov_tag = self.config.provider.capitalize()
        short_model = (self.config.model or "").replace("models/", "")[:20]
        ckpt_src = (
            self.rl_player.checkpoint_source
            if self.rl_player is not None
            else (self.config.opponent_model or self.config.model or "brass-ppo-best")
        )
        ckpt_short = ckpt_src.replace(".pt", "")

        if self.is_brass_human_vs_ai:
            h_seat = int(self.config.human_player_seat)
            human_meta = {
                "role": "human",
                "label": "You (Human)",
                "model_name": "human:interactive",
            }
            if self.config.provider == "brass_rl":
                ai_meta = {
                    "role": "ppo",
                    "label": f"Trained PPO ({ckpt_short})",
                    "model_name": f"pytorch:{ckpt_src}",
                }
            else:
                ai_meta = {
                    "role": "api",
                    "label": f"{prov_tag} ({short_model})",
                    "model_name": f"{self.config.provider}:{self.config.model}",
                }
            return [human_meta, ai_meta] if h_seat == 0 else [ai_meta, human_meta]

        api_seat = int(self.config.api_player_seat)
        api_meta = {
            "role": "api",
            "label": f"{prov_tag} ({short_model})",
            "model_name": f"{self.config.provider}:{self.config.model}",
        }
        ppo_meta = {
            "role": "ppo",
            "label": f"Trained PPO ({ckpt_short})",
            "model_name": f"pytorch:{ckpt_src}",
        }
        return [api_meta, ppo_meta] if api_seat == 0 else [ppo_meta, api_meta]

    def _duel_seat_labels(self) -> tuple[str, str]:
        """Return compact in-game display names for (Seat 0 / Player 1, Seat 1 / Player 2)."""
        metas = self._duel_seat_meta()
        return (metas[0]["label"], metas[1]["label"])

    def _build_duel_hud_state(self, b_state: dict, awaiting_human: bool = False) -> dict:
        """Build structured live scoreboard payload for the Frontend 2P Spectator / Play Banner."""
        summary = compute_board_summary(b_state)
        self_id = int(b_state.get("self_player_id", 0))
        opp_id = 1 - self_id
        seat_metas = self._duel_seat_meta()

        flipped_by_seat = {
            self_id: int(summary.get("flipped_tile_vp", 0)),
            opp_id: int(summary.get("opp_flipped_tile_vp", 0)),
        }
        link_by_seat = {
            self_id: int(summary.get("connected_link_vp", 0)),
            opp_id: int(summary.get("opp_connected_link_vp", 0)),
        }
        risk_by_seat = {
            self_id: int(summary.get("unflipped_tile_vp_at_risk", 0)),
            opp_id: int(summary.get("opp_unflipped_tile_vp_at_risk", 0)),
        }

        players_info = b_state.get("players_info") or []
        game_over = bool(b_state.get("game_over", False))
        in_setup = bool(b_state.get("in_setup_modal", False))
        active_seat = int(b_state.get("active_player", 0))

        players_hud = []
        for seat_i in (0, 1):
            p_raw = players_info[seat_i] if seat_i < len(players_info) else {}
            vp = int(p_raw.get("vp", 0))
            proj_era = flipped_by_seat[seat_i] + link_by_seat[seat_i]
            inc_space = int(p_raw.get("income_space", 10))
            meta = seat_metas[seat_i]
            players_hud.append(
                {
                    "seat": seat_i,
                    "label": p_raw.get("name") or meta["label"],
                    "role": meta["role"],
                    "model_name": meta["model_name"],
                    "vp": vp,
                    "projected_era_vp": 0 if game_over else proj_era,
                    "total_est_vp": vp if game_over else (vp + proj_era),
                    "flipped_vp": flipped_by_seat[seat_i],
                    "link_vp": link_by_seat[seat_i],
                    "unflipped_risk_vp": risk_by_seat[seat_i],
                    "money": int(p_raw.get("money", 17)),
                    "spent": int(p_raw.get("spent", 0)),
                    "income_space": inc_space,
                    "income_pounds": income_space_to_pounds(inc_space),
                    "hand_count": int(p_raw.get("hand_count", 8)),
                    "is_active": (not game_over and not in_setup and active_seat == seat_i),
                }
            )

        winner_seat: Optional[int] = None
        winner_summary = ""
        if game_over and len(players_hud) == 2:
            p0, p1 = players_hud[0], players_hud[1]
            key0 = (p0["vp"], p0["income_space"], p0["money"])
            key1 = (p1["vp"], p1["income_space"], p1["money"])
            if key0 > key1:
                winner_seat = 0
                reason = "VP" if p0["vp"] != p1["vp"] else ("Income tie-breaker" if p0["income_space"] != p1["income_space"] else "Cash tie-breaker")
                winner_summary = f"{p0['label']} wins ({p0['vp']} VP vs {p1['vp']} VP · {reason})!"
            elif key1 > key0:
                winner_seat = 1
                reason = "VP" if p1["vp"] != p0["vp"] else ("Income tie-breaker" if p1["income_space"] != p0["income_space"] else "Cash tie-breaker")
                winner_summary = f"{p1['label']} wins ({p1['vp']} VP vs {p0['vp']} VP · {reason})!"
            else:
                winner_seat = -1
                winner_summary = f"Exact Tie ({p0['vp']} VP each)!"

        return {
            "active": True,
            "match_mode": self.config.match_mode,
            "awaiting_human": bool(awaiting_human and not game_over),
            "seed": int(b_state.get("seed") if b_state.get("seed") is not None else self.config.game_seed),
            "era": b_state.get("era", "canal"),
            "turn": int(b_state.get("turn", 1)),
            "actions_left": int(b_state.get("actions_left", 1)),
            "draw_pile": int(b_state.get("draw_pile", 0)),
            "coal_cubes": int(b_state.get("coal_cubes", 13)),
            "iron_cubes": int(b_state.get("iron_cubes", 8)),
            "active_player": active_seat,
            "turn_order": b_state.get("turn_order", [0, 1]),
            "game_over": game_over,
            "winner_seat": winner_seat,
            "winner_summary": winner_summary,
            "players": players_hud,
        }

    def _format_api_duel_prompt_state(self, b_state: dict, duel_hud: dict) -> str:
        """
        Format complete public 2P Brass: Birmingham state + ONLY the API player's private hand
        (zero opponent hand leakage and zero PPO advisor hints).
        """
        my_seat = int(b_state.get("self_player_id", 0))
        opp_seat = 1 - my_seat
        p_me = duel_hud["players"][my_seat]
        p_opp = duel_hud["players"][opp_seat]
        era_str = str(b_state.get("era", "canal")).upper()
        turn_num = int(b_state.get("turn", 1))
        acts_left = int(b_state.get("actions_left", 1))
        turn_order = b_state.get("turn_order") or [0, 1]
        order_str = " -> ".join(
            f"P{s + 1} ({duel_hud['players'][s]['label']})" for s in turn_order if s in (0, 1)
        )

        def _mat_summary(mat_dict: dict) -> str:
            parts = []
            for ind in ("cotton", "goods", "coal", "iron", "brewery", "pottery"):
                lvls = mat_dict.get(ind) or []
                rem = sum(int(x) for x in lvls)
                lowest = next((i + 1 for i, c in enumerate(lvls) if int(c) > 0), None)
                if lowest is not None:
                    parts.append(f"{ind}:L{lowest}({rem} left)")
                else:
                    parts.append(f"{ind}:empty")
            return ", ".join(parts)

        coal_c = int(b_state.get("coal_cubes", 13))
        iron_c = int(b_state.get("iron_cubes", 8))

        merch_lines = []
        for m in b_state.get("merchants") or []:
            if not isinstance(m, dict):
                continue
            tiles = [t for t in (m.get("tiles") or []) if t and t != "blank"]
            if not tiles:
                continue
            beers = sum(1 for b in (m.get("beer") or []) if b)
            merch_lines.append(f"  - {m.get('id')}: buys {tiles} (merchant beer barrels available: {beers})")

        board_lines = []
        board = b_state.get("board") or {}
        for loc, slots in sorted(board.items()):
            if not isinstance(slots, list):
                continue
            occ = []
            for idx_s, t in enumerate(slots):
                if isinstance(t, dict):
                    tag = "YOU" if t.get("owner") == my_seat else "OPP"
                    st = "FLIPPED" if t.get("flipped") else f"unflipped(res={t.get('resources', 0)})"
                    occ.append(f"slot{idx_s}:[{tag}] {t.get('industry')} L{t.get('level')} {st}")
            if occ:
                board_lines.append(f"  - {loc}: " + "; ".join(occ))

        links = b_state.get("links") or {}
        my_links = sorted(lid for lid, ow in links.items() if ow == my_seat)
        opp_links = sorted(lid for lid, ow in links.items() if ow == opp_seat)

        hand_labels = b_state.get("hand_labels") or []
        hand_str = "\n".join(f"  Slot #{i}: {lbl}" for i, lbl in enumerate(hand_labels)) or "  (empty)"

        recent_logs = b_state.get("recent_log") or []
        log_str = "\n".join(f"  • {line}" for line in recent_logs[-8:]) or "  (start of match)"

        return (
            f"=== LIVE 2-PLAYER MATCH STATUS (Seed #{duel_hud['seed']}) ===\n"
            f"Era: {era_str} ERA | Round: {turn_num}/10 | Your Actions Left This Turn: {acts_left} | Draw Pile: {duel_hud['draw_pile']} cards\n"
            f"Current Round Turn Order: {order_str}\n\n"
            f"=== PLAYERS ===\n"
            f"• YOU (Seat P{my_seat + 1} · {p_me['label']}): {p_me['vp']} VP (+{p_me['projected_era_vp']} projected era VP: "
            f"{p_me['flipped_vp']} flipped + {p_me['link_vp']} link; {p_me['unflipped_risk_vp']} unflipped VP at risk) | "
            f"Cash: £{p_me['money']} (Spent this round: £{p_me['spent']}) | Income: {p_me['income_pounds']:+d}/rd (space {p_me['income_space']})\n"
            f"  Your Mat: {_mat_summary(b_state.get('mat') or {})}\n"
            f"• OPPONENT (Seat P{opp_seat + 1} · {p_opp['label']}): {p_opp['vp']} VP (+{p_opp['projected_era_vp']} projected era VP: "
            f"{p_opp['flipped_vp']} flipped + {p_opp['link_vp']} link) | "
            f"Cash: £{p_opp['money']} (Spent this round: £{p_opp['spent']}) | Income: {p_opp['income_pounds']:+d}/rd | Hand size: {p_opp['hand_count']} cards (hidden)\n"
            f"  Opponent Mat: {_mat_summary(b_state.get('automa_mat') or {})}\n\n"
            f"=== MARKETS & MERCHANTS ===\n"
            f"Coal Market: {coal_c}/14 cubes (£{coal_market_price(coal_c)} next) | Iron Market: {iron_c}/10 cubes (£{iron_market_price(iron_c)} next)\n"
            + ("\n".join(merch_lines) if merch_lines else "  (no active merchant tiles)")
            + "\n\n=== BUILT BOARD TILES & LINKS ===\n"
            + ("\n".join(board_lines) if board_lines else "  (board is currently empty)")
            + f"\n  Your Links ({len(my_links)}): {', '.join(my_links) if my_links else 'none'}\n"
            + f"  Opponent Links ({len(opp_links)}): {', '.join(opp_links) if opp_links else 'none'}\n\n"
            f"=== YOUR PRIVATE HAND ({len(hand_labels)} cards) ===\n{hand_str}\n\n"
            f"=== RECENT MATCH LOG ===\n{log_str}"
        )

    async def handle_human_click(self, x: int, y: int) -> None:
        """Forward an interactive click from LiveViewport directly to Playwright during Human vs AI."""
        if not self.is_brass_human_vs_ai:
            return
        page = self.browser.page
        if not page or page.is_closed():
            return
        cx = max(0, min(1279, int(x)))
        cy = max(0, min(719, int(y)))
        fresh_ui: Optional[str] = None
        async with self._page_lock:
            pre_state = await extract_brass_state(page, reset_open_flow=False)
            seat_metas = self._duel_seat_meta()
            if pre_state.get("game_over") or seat_metas[int(pre_state.get("active_player", 0))]["role"] != "human":
                return
            await page.mouse.click(cx, cy)
            self.browser.mouse_pos = (cx, cy)
            self.state.mouse_x, self.state.mouse_y = cx, cy
            await asyncio.sleep(0.05)
            cur_b_state = await extract_brass_state(page, reset_open_flow=False)
            cur_role = seat_metas[int(cur_b_state.get("active_player", 0))]["role"]
            still_human = (not cur_b_state.get("game_over")) and (cur_role == "human")
            self.state.duel_state = self._build_duel_hud_state(cur_b_state, awaiting_human=still_human)
            fresh_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)
        if fresh_ui:
            await self._notify_state(image=fresh_ui)

    async def handle_human_scroll(self, x: int, y: int, delta_y: int) -> None:
        """Forward mouse-wheel scroll from LiveViewport to Playwright (for scrolling Brass option lists)."""
        if not self.is_brass_human_vs_ai:
            return
        page = self.browser.page
        if not page or page.is_closed():
            return
        cx = max(0, min(1279, int(x)))
        cy = max(0, min(719, int(y)))
        dy = max(-600, min(600, int(delta_y)))
        fresh_ui: Optional[str] = None
        async with self._page_lock:
            pre_state = await extract_brass_state(page, reset_open_flow=False)
            seat_metas = self._duel_seat_meta()
            if pre_state.get("game_over") or seat_metas[int(pre_state.get("active_player", 0))]["role"] != "human":
                return
            await page.mouse.move(cx, cy)
            await page.mouse.wheel(0, dy)
            await asyncio.sleep(0.04)
            fresh_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)
        if fresh_ui:
            await self._notify_state(image=fresh_ui)

    async def _run_brass_duel_turn(
        self, iteration: int, ui_b64: Optional[str], delta_pct: float
    ) -> bool:
        """
        Execute one turn in an Official 2-Player Brass: Birmingham Match:
        - brass_2p_duel: API Challenger (Gemini/OpenAI/Anthropic) vs Local Trained PyTorch PPO
        - brass_human_vs_ai: Interactive Human Player vs Local Trained PPO OR Cloud API Model
        """
        page = self.browser.page
        if not page or page.is_closed():
            return False

        seat_metas = self._duel_seat_meta()
        seat0_lbl, seat1_lbl = seat_metas[0]["label"], seat_metas[1]["label"]

        # Read non-destructively first so we never cancel an active human UI flow
        async with self._page_lock:
            b_state = await extract_brass_state(page, reset_open_flow=False)

        # Initialize fresh seeded official 2-player match at session start (or if setup modal is open)
        if not self._duel_initialized or b_state.get("in_setup_modal") or b_state.get("mode") != "2p":
            async with self._page_lock:
                await start_new_brass_game(
                    page,
                    seed=int(self.config.game_seed),
                    mode="2p",
                    player_names=[seat0_lbl, seat1_lbl],
                )
                self._duel_initialized = True
                b_state = await extract_brass_state(page, reset_open_flow=True)
                first_seat = int(b_state.get("active_player", 0))
                first_is_human = seat_metas[first_seat]["role"] == "human"
                self.state.duel_state = self._build_duel_hud_state(b_state, awaiting_human=first_is_human)
                fresh_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)

            order = b_state.get("turn_order") or [0, 1]
            first_lbl = self.state.duel_state["players"][order[0]]["label"]
            mode_title = "Human vs AI Match" if self.is_brass_human_vs_ai else "AI vs AI Duel"
            obs = (
                f"Started Official 2-Player Brass: Birmingham {mode_title} (Seed #{self.config.game_seed} · "
                f"40-card 2P deck). {first_lbl} acts first in Round 1!"
            )
            reasoning = (
                f"Seat P1: {seat0_lbl} vs Seat P2: {seat1_lbl}. "
                f"Cards & merchant tiles deterministically dealt from seed #{self.config.game_seed}."
            )
            action_dict = {"type": "click_coordinates", "x": 640, "y": 360}
            exec_status = f"Initialized 2P Match (Seed #{self.config.game_seed})"
            resp = LLMResponse(
                observation=obs,
                reasoning_summary=reasoning,
                action=Action(type="click_coordinates", x=640, y=360),
                expected_result="Begin Canal Era Round 1.",
                goal_status="in_progress",
                current_subgoal="Canal Era Round 1 Kickoff",
                progress_percent=2,
            )
            actor_role = "system"
            actor_label = f"Match Referee · Seed #{self.config.game_seed}"
            actor_seat = -1
            step_tokens = 0
            step_cost = 0.0
            if fresh_ui:
                ui_b64 = fresh_ui

        elif b_state.get("game_over"):
            self.state.duel_state = self._build_duel_hud_state(b_state, awaiting_human=False)
            p0, p1 = self.state.duel_state["players"][0], self.state.duel_state["players"][1]
            win_sum = self.state.duel_state.get("winner_summary") or "Match Complete!"
            obs = f"2P Match Finished (Seed #{self.state.duel_state['seed']}) — {win_sum}"
            reasoning = (
                f"Final Official 2P Scoreboard: P1 {p0['label']} = {p0['vp']} VP (Income {p0['income_pounds']:+d}, £{p0['money']}) "
                f"vs P2 {p1['label']} = {p1['vp']} VP (Income {p1['income_pounds']:+d}, £{p1['money']})."
            )
            action_dict = {"type": "declare_success"}
            exec_status = win_sum
            resp = LLMResponse(
                observation=obs,
                reasoning_summary=reasoning,
                action=Action(type="declare_success"),
                expected_result="Display final 2P victory scoreboard.",
                goal_status="success",
                current_subgoal=win_sum,
                progress_percent=100,
            )
            self.state.status = "completed"
            self.state.status_reason = obs
            self.state.stats.progress_percent = 100
            actor_role = "system"
            actor_label = "Final Scoreboard"
            actor_seat = self.state.duel_state.get("winner_seat", -1)
            step_tokens = 0
            step_cost = 0.0

        else:
            active_seat = int(b_state.get("active_player", 0))
            seat_role = seat_metas[active_seat]["role"]
            duel_hud = self._build_duel_hud_state(b_state, awaiting_human=(seat_role == "human"))
            self.state.duel_state = duel_hud
            active_p = duel_hud["players"][active_seat]
            opp_p = duel_hud["players"][1 - active_seat]
            era_lbl = str(b_state.get("era", "canal")).capitalize()
            turn_num = int(b_state.get("turn", 1))
            acts_left = int(b_state.get("actions_left", 1))
            prog = min(98, max(4, int(((0 if era_lbl == "Canal" else 10) + turn_num) * 4.8)))

            actor_seat = active_seat
            step_tokens = 0
            step_cost = 0.0

            async def _on_substep(xy: tuple[int, int], lbl: str) -> None:
                self.state.mouse_x, self.state.mouse_y = xy
                sub_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)
                if sub_ui:
                    await self._notify_state(image=sub_ui)

            step_delay = max(90, min(600, int(self.config.default_action_duration_ms)))

            if seat_role == "human":
                # --- INTERACTIVE HUMAN PLAYER'S TURN ---
                actor_role = "human"
                actor_label = f"P{active_seat + 1} · {active_p['label']}"
                self.state.status = "running"
                self.state.status_reason = (
                    f"🎮 YOUR TURN (P{active_seat + 1} · {era_lbl} Rd {turn_num}/10, {acts_left} action{'s' if acts_left != 1 else ''} left) — "
                    f"Click directly on the Live Arena board below (or in the Chromium window) to play!"
                )
                async with self._page_lock:
                    live_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)
                if live_ui:
                    ui_b64 = live_ui
                await self._notify_state(image=ui_b64)

                initial_sig = (
                    b_state.get("era"),
                    int(b_state.get("turn", 1)),
                    int(b_state.get("active_player", 0)),
                    int(b_state.get("actions_left", 1)),
                    bool(b_state.get("game_over", False)),
                )
                cur_state = b_state
                poll_ticks = 0
                while not self._stop_event.is_set():
                    if not await self._wait_if_paused():
                        return True
                    await self._interruptible_sleep(0.35)
                    if self._stop_event.is_set():
                        return True
                    poll_ticks += 1
                    async with self._page_lock:
                        cur_state = await extract_brass_state(page, reset_open_flow=False)
                        cur_sig = (
                            cur_state.get("era"),
                            int(cur_state.get("turn", 1)),
                            int(cur_state.get("active_player", 0)),
                            int(cur_state.get("actions_left", 1)),
                            bool(cur_state.get("game_over", False)),
                        )
                        if cur_sig != initial_sig or cur_state.get("in_setup_modal"):
                            break
                        # Periodically refresh live preview in case user is clicking directly inside the Chromium window
                        if poll_ticks % 2 == 0:
                            win_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)
                        else:
                            win_ui = None
                    if win_ui:
                        await self._notify_state(image=win_ui)

                if cur_state.get("in_setup_modal"):
                    self._duel_initialized = False
                    return False

                human_move_desc = cur_state.get("last_log") or "Completed turn action."
                lx, ly = self.browser.mouse_pos
                action_dict = {"type": "click_coordinates", "x": lx, "y": ly}
                exec_status = f"[P{active_seat + 1} You] {human_move_desc}"
                obs = (
                    f"[P{active_seat + 1} · You] {era_lbl} Rd {turn_num}/10 ({acts_left} act left) — "
                    f"Committed move on board."
                )
                reasoning = f"Human player executed: {human_move_desc}"
                subgoal = f"P{active_seat + 1} Human move committed"
                conf_val = 1.0
                chosen_desc = human_move_desc

            elif seat_role == "api":
                # --- API CHALLENGER'S TURN ---
                assert self.llm is not None
                actor_role = "api"
                actor_label = f"P{active_seat + 1} · {active_p['label']}"

                if self.state.stats.estimated_cost_usd >= self.config.max_budget_usd:
                    self.state.status = "stopped"
                    self.state.status_reason = (
                        f"API budget cap (${self.config.max_budget_usd:.2f}) reached during 2P Match."
                    )
                    await self._notify_state(
                        event_note=self.state.status_reason,
                        image=ui_b64,
                    )
                    return True

                async with self._page_lock:
                    b_state = await extract_brass_state(page, reset_open_flow=True)
                    duel_hud = self._build_duel_hud_state(b_state, awaiting_human=False)
                    self.state.duel_state = duel_hud
                    legal = await enumerate_legal_actions(page, b_state)
                    clean_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)

                if not legal:
                    await self._interruptible_sleep(0.4)
                    return False

                self.state.status = "running"
                self.state.status_reason = (
                    f"[P{active_seat + 1} API] {active_p['label']} thinking ({era_lbl} Rd {turn_num}/10, {len(legal)} legal moves)..."
                )
                await self._notify_state(image=clean_ui or ui_b64)

                prompt_state_txt = self._format_api_duel_prompt_state(b_state, duel_hud)
                decision, tel = await self.llm.choose_brass_macro_action(
                    image_base64=clean_ui or ui_b64,
                    state_summary_text=prompt_state_txt,
                    legal_actions=legal,
                    player_name=active_p["label"],
                    opponent_name=opp_p["label"],
                )
                self._record_telemetry(tel)
                step_tokens = tel.total_tokens
                step_cost = tel.cost_usd

                chosen = legal[decision.chosen_index]
                async with self._page_lock:
                    exec_res = await execute_macro_action(
                        page,
                        chosen,
                        visual_delay_ms=step_delay,
                        browser_mgr=self.browser,
                        on_substep=_on_substep,
                    )
                lx, ly = exec_res.get("last_xy", (640, 360))
                action_dict = {"type": "click_coordinates", "x": lx, "y": ly}
                exec_status = (
                    f"[P{active_seat + 1} API] {chosen['description']}"
                    if exec_res.get("ok")
                    else f"[P{active_seat + 1} API fallback] {chosen['description']}"
                )
                obs = (
                    f"[P{active_seat + 1} · {active_p['label']}] {era_lbl} Rd {turn_num}/10 ({acts_left} act left) · "
                    f"{decision.observation}"
                )
                reasoning = decision.reasoning_summary
                subgoal = decision.current_subgoal or f"P{active_seat + 1} API: {chosen['action_type'].capitalize()}"
                conf_val = 1.0
                chosen_desc = chosen["description"]

            else:
                # --- TRAINED PYTORCH PPO MODEL'S TURN ---
                assert self.rl_player is not None
                actor_role = "ppo"
                actor_label = f"P{active_seat + 1} · {active_p['label']}"
                self.state.status = "running"
                self.state.status_reason = (
                    f"[P{active_seat + 1} PPO] {active_p['label']} evaluating board ({era_lbl} Rd {turn_num}/10)..."
                )
                await self._notify_state(image=ui_b64)

                async with self._page_lock:
                    eval_res = await self.rl_player.evaluate_board(page, greedy=True, temperature=0.0)
                    chosen = eval_res.get("chosen_action")
                    if not chosen:
                        exec_res = None
                    else:
                        exec_res = await execute_macro_action(
                            page,
                            chosen,
                            visual_delay_ms=step_delay,
                            browser_mgr=self.browser,
                            on_substep=_on_substep,
                        )
                if not chosen or exec_res is None:
                    await self._interruptible_sleep(0.4)
                    return False

                top_strs = [
                    f"#{r['rank']} {r['action']['description']} ({r['prob_pct']:.1f}%)"
                    for r in (eval_res.get("ranked_actions") or [])[:3]
                ]
                lx, ly = exec_res.get("last_xy", (640, 360))
                action_dict = {"type": "click_coordinates", "x": lx, "y": ly}
                exec_status = (
                    f"[P{active_seat + 1} PPO] {chosen['description']}"
                    if exec_res.get("ok")
                    else f"[P{active_seat + 1} PPO fallback] {chosen['description']}"
                )
                obs = (
                    f"[P{active_seat + 1} · {active_p['label']}] {era_lbl} Rd {turn_num}/10 ({acts_left} act left) · "
                    f"£{active_p['money']} · {active_p['vp']} VP (+{active_p['projected_era_vp']} proj) vs "
                    f"{opp_p['label']} {opp_p['vp']} VP (+{opp_p['projected_era_vp']} proj)"
                )
                reasoning = (
                    f"PyTorch PPO Critic V(s)={eval_res.get('value', 0.0):+.2f} [{eval_res.get('checkpoint_source')}] · "
                    + " | ".join(top_strs)
                )
                subgoal = f"P{active_seat + 1} PPO: {chosen['action_type'].capitalize()}"
                conf_val = round(float(eval_res.get("chosen_prob_pct", 100.0)) / 100.0, 2)
                chosen_desc = chosen["description"]

            resp = LLMResponse(
                observation=obs,
                reasoning_summary=reasoning,
                action=Action(type="click_coordinates", x=lx, y=ly),
                expected_result=chosen_desc,
                goal_status="in_progress",
                current_subgoal=subgoal,
                progress_percent=prog,
            )
            self.state.click_history.append(
                {
                    "iteration": iteration,
                    "x": lx,
                    "y": ly,
                    "target_id": None,
                    "label": f"P{active_seat + 1}: {chosen_desc[:36]}",
                    "source": "dom",
                    "confidence": conf_val,
                    "visual_delta_pct": None,
                    "timestamp": round(time.time(), 3),
                }
            )
            if len(self.state.click_history) > 10:
                self.state.click_history = self.state.click_history[-10:]

            # Refresh post-move board state & frame immediately so Spectator/Play HUD stays 100% live!
            async with self._page_lock:
                post_b_state = await extract_brass_state(page, reset_open_flow=False)
                nxt_seat_chk = int(post_b_state.get("active_player", 0))
                nxt_is_human = (not post_b_state.get("game_over")) and (seat_metas[nxt_seat_chk]["role"] == "human")
                self.state.duel_state = self._build_duel_hud_state(post_b_state, awaiting_human=nxt_is_human)
                post_ui, _, _ = await self.browser.capture_observation(add_grid_for_llm=False)
            if post_ui:
                ui_b64 = post_ui

            p0_now, p1_now = self.state.duel_state["players"][0], self.state.duel_state["players"][1]
            self.state.current_strategy = (
                f"Seed #{self.state.duel_state['seed']} · "
                f"P1 {p0_now['label']}: {p0_now['vp']} VP (+{p0_now['projected_era_vp']} proj) vs "
                f"P2 {p1_now['label']}: {p1_now['vp']} VP (+{p1_now['projected_era_vp']} proj)"
            )
            self.memory.current_strategy = self.state.current_strategy
            if not post_b_state.get("game_over"):
                nxt_seat = int(post_b_state.get("active_player", 0))
                nxt_p = self.state.duel_state["players"][nxt_seat]
                self.state.status_reason = (
                    f"Up next: P{nxt_seat + 1} {nxt_p['label']} · "
                    f"{str(post_b_state.get('era', 'canal')).capitalize()} Rd {post_b_state.get('turn', 1)}/10"
                )

        self.state.mouse_x, self.state.mouse_y = self.browser.mouse_pos
        self.memory.record_tactical_step(iteration, resp, exec_status)
        self._sync_memory_to_state()

        self.state.last_observation = resp.observation
        self.state.last_reasoning = resp.reasoning_summary
        self.state.last_expected_result = resp.expected_result
        self.state.last_action = action_dict
        self.state.last_action_status = exec_status
        self.state.stats.total_actions += 1
        self._update_elapsed()

        self.db.record_step(
            session_id=self.session_id,
            iteration=iteration,
            observation=resp.observation,
            reasoning=resp.reasoning_summary,
            action=action_dict,
            expected_result=resp.expected_result,
            evaluation="",
            visual_delta_pct=delta_pct,
            goal_status=resp.goal_status,
            tokens_used=step_tokens,
            cost_usd=step_cost,
        )
        self.db.update_session(
            session_id=self.session_id,
            status=self.state.status,
            stats=self.state.stats.model_dump(),
            status_reason=self.state.status_reason,
            strategy=self.state.current_strategy,
            lessons=self.state.lessons_learned,
        )

        step_log_payload = {
            **resp.model_dump(),
            "action": action_dict,
            "iteration": iteration,
            "timestamp": time.time(),
            "execution_status": exec_status,
            "visual_delta_pct": delta_pct,
            "step_tokens": step_tokens,
            "step_cost_usd": step_cost,
            "actor_role": actor_role,
            "actor_label": actor_label,
            "actor_seat": actor_seat,
        }
        await self._notify_state(llm_response=step_log_payload, image=ui_b64)
        return self.state.status == "completed"

    async def pause(self, reason: str = "Paused by user.") -> None:
        if self._pause_event.is_set() and not self._stop_event.is_set():
            self._pause_event.clear()
            if self._active_started_at is not None:
                self._accumulated_elapsed += time.monotonic() - self._active_started_at
                self._active_started_at = None
            await self.browser.release_all_inputs()
            self.state.status = "paused"
            self.state.status_reason = reason
            self._update_elapsed()
            await self._notify_state()

    async def resume(self) -> None:
        if not self._pause_event.is_set() and not self._stop_event.is_set():
            self._active_started_at = time.monotonic()
            self.state.status = "running"
            self.state.status_reason = "Resumed gameplay."
            self._pause_event.set()
            await self._notify_state()

    async def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()  # Unblock loop if paused
        await self.browser.release_all_inputs()
        self.state.status = "stopped"
        self.state.status_reason = "Emergency stop triggered."
        self._update_elapsed()
        await self._notify_state()

    def add_user_hint(self, hint: str) -> None:
        self.memory.add_user_hint(hint)

    async def _notify_state(
        self,
        llm_response: Optional[dict] = None,
        strategic_plan: Optional[dict] = None,
        image: Optional[str] = None,
        error: Optional[str] = None,
        event_note: Optional[str] = None,
    ) -> None:
        self._update_elapsed()
        payload: dict = {
            "type": "state_update",
            "state": self.state.model_dump(),
        }
        if llm_response is not None:
            payload["llm_response"] = llm_response
        if strategic_plan is not None:
            payload["strategic_plan"] = strategic_plan
        if image is not None:
            payload["image"] = image
        if error is not None:
            payload["error"] = error
        if event_note is not None:
            payload["event_note"] = event_note

        if asyncio.iscoroutinefunction(self.on_update):
            await self.on_update(payload)
        else:
            self.on_update(payload)
