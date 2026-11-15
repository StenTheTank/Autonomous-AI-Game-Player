import re
from typing import Dict, Any, List, Optional
from models.schemas import GameConfig, InteractionMap, LLMResponse, StrategicPlan, coords_to_macro_cell


_UI_OR_RELIABILITY_RE = re.compile(
    r"("
    r"target_\d+|"
    r"#\d{2,3}\b|"
    r"\b(?:click_target|click_coordinates|mouse_click|mouse_move|mouse_drag|key_press|key_hold|key_combo|hover_target)\b|"
    r"\(\s*\d{2,4}\s*,\s*\d{2,4}\s*\)|"
    r"\b[xy]\s*=\s*\d+\b|"
    r"\bcell\s*=\s*[A-P][1-9]\b|"
    r"\bbutton\b|"
    r"\b(?:situated|located\s+at|top-left|top-right|bottom-left|bottom-right|sidebar|toolbar|dropdown|canvas|cursor|pixel|bbox|viewport)\b|"
    r"\b(?:reliably|reliable|unreliable|easier\s+to\s+click|easy\s+to\s+execute|works?\s+every\s+time|works?\s+consistently|no\s+visible\s+change|screen\s+change|focus\s+canvas)\b"
    r")",
    re.IGNORECASE,
)

_OBS_UI_NOISE_RE = re.compile(
    r"(\btarget_\d+\b|#\d{2,3}\b|\(\s*\d{2,4}\s*,\s*\d{2,4}\s*\)|\bcell\s*=\s*[A-P][1-9]\b|\b[xy]\s*=\s*\d+\b)",
    re.IGNORECASE,
)


def is_ui_or_reliability_artifact(text: str) -> bool:
    """Return True if a lesson/mechanic talks about UI button locations, coordinates, or click reliability instead of pure game rules."""
    if not text or not text.strip():
        return True
    return bool(_UI_OR_RELIABILITY_RE.search(text))


def sanitize_domain_observation(text: str) -> str:
    """Strip target_ids, coordinates, and cell codes so Strategic Planning sees only pure game state."""
    if not text:
        return ""
    cleaned = _OBS_UI_NOISE_RE.sub("", text)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;-")
    return cleaned


def normalize_target_base_label(label: str) -> str:
    """Normalize a target label (stripping slot suffixes like '[hand-card-2]') so identical dead options share quarantine."""
    if not label:
        return ""
    base = re.sub(r"\s*\[[^\]]*\]\s*$", "", label).strip().lower()
    return base


class SessionMemory:
    """
    Manages bounded persistent context with strict separation between:
    1. Pure Game-Domain Strategy (`current_strategy`, `sub_goals`, `known_mechanics`, `lessons_learned`)
       — NEVER polluted by UI click reliability, button positions, or pixel coordinates.
    2. Tactical UI Execution Guardrails (`ineffective_actions`, `tactical_ui_notes`, `consecutive_stagnant_steps`)
       — used strictly by the tactical executor to avoid repeating dead clicks.
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.current_strategy: str = (
            f"Pursue a {config.strategy_preference} game strategy focused on achieving '{config.objective}' "
            "according to the game rules and current board state."
        )
        self.sub_goals: List[str] = [
            "Enter active gameplay if on a title, menu, or setup screen",
            "Assess initial board state, resources, and available legal moves",
            config.objective,
        ]
        self.known_mechanics: List[str] = []
        self.lessons_learned: List[str] = []
        self.tactical_ui_notes: List[str] = []
        self.current_prompt_noop_labels: List[str] = []
        self.recent_steps: List[Dict[str, Any]] = []
        self.compressed_milestones: List[str] = []
        self.effective_actions: Dict[str, int] = {}
        self.ineffective_actions: Dict[str, int] = {}
        self.consecutive_stagnant_steps: int = 0
        self.progress_percent: int = 0
        self.current_subgoal: str = self.sub_goals[0]
        self.pending_user_hints: List[str] = []
        self.adjusted_action_duration_ms: int = config.default_action_duration_ms

    @staticmethod
    def format_action_signature(action: Dict[str, Any]) -> str:
        t = action.get("type", "unknown")
        if t in ("key_press", "key_hold", "key_release"):
            dur = f"({action.get('duration_ms')}ms)" if action.get("duration_ms") else ""
            return f"{t}:{action.get('key')}{dur}"
        if t == "key_combo":
            return f"key_combo:{'+'.join(action.get('keys') or [])}"
        if t in ("click_target", "hover_target"):
            return f"{t}:{action.get('target_id')}({action.get('x')},{action.get('y')})"
        if t in ("click_coordinates", "mouse_click", "mouse_move", "mouse_down", "mouse_up"):
            if action.get("target_id"):
                return f"{t}:{action.get('target_id')}({action.get('x')},{action.get('y')})"
            if action.get("element_id") is not None:
                return f"{t}:#{action.get('element_id')}({action.get('x')},{action.get('y')})"
            if action.get("grid_cell"):
                return f"{t}:{action.get('grid_cell')}->({action.get('x')},{action.get('y')})"
            return f"{t}:({action.get('x')},{action.get('y')})"
        if t == "mouse_drag":
            return f"mouse_drag:({action.get('from_x')},{action.get('from_y')})->({action.get('x')},{action.get('y')})"
        if t == "scroll":
            return f"scroll:({action.get('scroll_x')},{action.get('scroll_y')})"
        if t == "wait":
            return f"wait:{action.get('duration_ms', 1000)}ms"
        return t

    def add_user_hint(self, hint: str) -> None:
        cleaned = hint.strip()
        if cleaned:
            self.pending_user_hints.append(cleaned)

    def apply_strategic_plan(self, plan: StrategicPlan) -> None:
        if plan.current_strategy:
            self.current_strategy = plan.current_strategy
        if plan.sub_goals:
            self.sub_goals = plan.sub_goals[:6]
            self.current_subgoal = self.sub_goals[0]

        for mech in plan.known_mechanics:
            cleaned_mech = (mech or "").strip()
            if cleaned_mech and not is_ui_or_reliability_artifact(cleaned_mech) and cleaned_mech not in self.known_mechanics:
                self.known_mechanics.append(cleaned_mech)
        self.known_mechanics = [
            m for m in self.known_mechanics if not is_ui_or_reliability_artifact(m)
        ][-10:]

        for lesson in plan.lessons_learned:
            cleaned_lesson = (lesson or "").strip()
            if cleaned_lesson and not is_ui_or_reliability_artifact(cleaned_lesson) and cleaned_lesson not in self.lessons_learned:
                self.lessons_learned.append(cleaned_lesson)
        self.lessons_learned = [
            l for l in self.lessons_learned if not is_ui_or_reliability_artifact(l)
        ][-12:]

        if plan.recommended_action_cadence_ms:
            self.adjusted_action_duration_ms = max(
                40, min(self.config.max_action_duration_ms, int(plan.recommended_action_cadence_ms))
            )

    def record_previous_action_effect(self, visual_delta_pct: float) -> None:
        """Update tactical UI outcome of the most recently executed step without polluting game strategy."""
        if not self.recent_steps:
            return
        last = self.recent_steps[-1]
        last["post_visual_delta_pct"] = visual_delta_pct
        action_type = last["action"].get("type", "")
        sig = last["signature"]

        if action_type in ("wait", "request_observation"):
            return

        if visual_delta_pct < 0.35:
            self.consecutive_stagnant_steps += 1
            self.ineffective_actions[sig] = self.ineffective_actions.get(sig, 0) + 1
            if visual_delta_pct < 0.20 and action_type == "click_target":
                base_lbl = normalize_target_base_label(last.get("target_label") or "")
                if base_lbl and base_lbl not in ("cancel", "back", "close", "pass", "confirm"):
                    if base_lbl not in self.current_prompt_noop_labels:
                        self.current_prompt_noop_labels.append(base_lbl)
                        self.current_prompt_noop_labels = self.current_prompt_noop_labels[-8:]
            if self.ineffective_actions[sig] == 2:
                ui_note = f"Input '{sig}' produced no visible change twice; focus canvas or select a different target."
                if ui_note not in self.tactical_ui_notes:
                    self.tactical_ui_notes.append(ui_note)
                    self.tactical_ui_notes = self.tactical_ui_notes[-6:]
        else:
            self.consecutive_stagnant_steps = 0
            self.effective_actions[sig] = self.effective_actions.get(sig, 0) + 1
            # Prompt or board state changed visibly -> reset same-prompt dead-option quarantine
            self.current_prompt_noop_labels.clear()
            if visual_delta_pct >= 2.5:
                # Screen state advanced meaningfully; clear transient UI-click warnings
                self.tactical_ui_notes.clear()

    def record_tactical_step(
        self,
        iteration: int,
        resp: LLMResponse,
        execution_status: str,
        target_label: Optional[str] = None,
    ) -> None:
        action_dict = resp.action.model_dump()
        sig = self.format_action_signature(action_dict)

        if resp.learned_lesson and resp.learned_lesson.strip():
            lesson = resp.learned_lesson.strip()
            # Strictly block UI button locations, target_ids, coordinates, and action-reliability remarks from strategy!
            if not is_ui_or_reliability_artifact(lesson) and lesson not in self.lessons_learned:
                self.lessons_learned.append(lesson)
                self.lessons_learned = self.lessons_learned[-12:]

        if resp.current_subgoal:
            self.current_subgoal = resp.current_subgoal
        if resp.progress_percent is not None:
            self.progress_percent = resp.progress_percent

        step_entry = {
            "iteration": iteration,
            "observation": resp.observation,
            "evaluation": resp.previous_action_evaluation or "",
            "reasoning": resp.reasoning_summary,
            "subgoal": self.current_subgoal,
            "action": action_dict,
            "target_label": target_label or "",
            "signature": sig,
            "expected_result": resp.expected_result,
            "execution_status": execution_status,
            "post_visual_delta_pct": None,
        }
        self.recent_steps.append(step_entry)

        # Compress older steps beyond window of 6 to keep API prompt concise
        while len(self.recent_steps) > 6:
            oldest = self.recent_steps.pop(0)
            clean_obs = sanitize_domain_observation(oldest["observation"])[:100]
            if clean_obs:
                summary_line = f"Turn/Step {oldest['iteration']}: {clean_obs}"
                self.compressed_milestones.append(summary_line)
                if len(self.compressed_milestones) > 10:
                    self.compressed_milestones = self.compressed_milestones[-10:]

    def needs_strategic_replan(self, iteration: int) -> bool:
        if iteration > 1 and iteration % self.config.strategic_planning_interval == 0:
            return True
        if self.consecutive_stagnant_steps >= 4:
            return True
        return False

    def build_tactical_context(
        self,
        iteration: int,
        latest_delta_pct: float,
        mouse_pos: tuple[int, int] = (640, 360),
        interactive_elements: Optional[List[Dict[str, Any]]] = None,
        interaction_map: Optional[InteractionMap] = None,
        disabled_labels: Optional[List[str]] = None,
        rl_advisor_hint: Optional[str] = None,
    ) -> str:
        mx, my = mouse_pos
        cursor_cell = coords_to_macro_cell(mx, my)
        sections: List[str] = [
            f"=== SESSION STATUS (Iteration {iteration} / {self.config.max_iterations}) ===",
            f"CURRENT MOUSE CURSOR: cell={cursor_cell} (x={mx}, y={my} in 1280x720 viewport)",
            f"Overall Strategy: {self.current_strategy}",
            f"Active Sub-Goal: {self.current_subgoal}",
            f"Estimated Objective Progress: {self.progress_percent}%",
            f"Recommended Key/Action Duration: {self.adjusted_action_duration_ms}ms",
        ]

        if interaction_map is not None and interaction_map.targets:
            noop_set = set(self.current_prompt_noop_labels)
            target_lines = []
            for t in interaction_map.targets[:95]:
                noop_flag = (
                    " ⚠️ [NO-OP IN THIS PROMPT: 0% effect — DO NOT CLICK]"
                    if normalize_target_base_label(t.label) in noop_set
                    else ""
                )
                target_lines.append(
                    f"  - id=\"{t.id}\" | badge=#{t.id.split('_')[-1]} | cell={coords_to_macro_cell(t.center[0], t.center[1])} | "
                    f"source={t.source} | conf={t.confidence:.2f} | label=\"{t.label}\"{noop_flag}"
                )
            sections.append(
                f"CURRENT INTERACTION MAP (screenshot_version={interaction_map.screenshot_version}, "
                f"status={interaction_map.regeneration_reason} — ALWAYS prefer `{{\"type\": \"click_target\", \"target_id\": \"...\"}}`!):\n"
                + "\n".join(target_lines)
            )
            if self.current_prompt_noop_labels:
                quoted_noops = ", ".join(f'"{lbl}"' for lbl in self.current_prompt_noop_labels)
                cancel_hint = (
                    " If NO remaining card/option has a legal move in this mode right now, click `cancel` / exit this prompt immediately and choose a different action!"
                    if len(self.current_prompt_noop_labels) >= 2
                    else " Do NOT click another card/option with the same label — pick a DIFFERENT legal option or click `cancel`!"
                )
                sections.append(
                    f"UNPLAYABLE / NO-OP OPTIONS IN CURRENT PROMPT: {quoted_noops} produced 0% screen change (no legal placements/moves right now).{cancel_hint}"
                )
        elif interactive_elements:
            elem_lines = [
                f"  - id=\"target_{el['id']:03d}\" | cell={coords_to_macro_cell(el['x'], el['y'])} | source=dom | conf=1.00 | label=\"{el['label']}\""
                for el in interactive_elements[:50]
            ]
            sections.append(
                "CURRENT INTERACTION MAP (Prefer `{\"type\": \"click_target\", \"target_id\": \"...\"}`):\n"
                + "\n".join(elem_lines)
            )
        else:
            sections.append(
                "CURRENT INTERACTION MAP: 0 pre-detected targets on this frame. "
                "Use `{\"type\": \"click_coordinates\", \"grid_cell\": \"A1\"..\"P9\"}` (or integer x, y) or keyboard controls."
            )

        if disabled_labels:
            quoted_dis = ", ".join(f'"{d}"' for d in disabled_labels[:14])
            sections.append(
                f"DISABLED / CURRENTLY ILLEGAL CONTROLS ON SCREEN (intentionally omitted from Interaction Map because game rules disable them right now — NEVER use `click_coordinates` on these!): {quoted_dis}"
            )

        if rl_advisor_hint and rl_advisor_hint.strip():
            sections.append(rl_advisor_hint.strip())

        if self.sub_goals:
            sections.append("Sub-Goals Queue:\n" + "\n".join(f"  {idx + 1}. {g}" for idx, g in enumerate(self.sub_goals)))

        if self.known_mechanics:
            sections.append("Confirmed Game Mechanics:\n" + "\n".join(f"  - {m}" for m in self.known_mechanics[-6:]))

        if self.lessons_learned:
            sections.append("Strategic Domain Lessons:\n" + "\n".join(f"  - {l}" for l in self.lessons_learned[-8:]))

        if self.tactical_ui_notes:
            sections.append(
                "Recent Dead-Click / Input Warnings (do NOT change game strategy; pick the right UI target for your goal):\n"
                + "\n".join(f"  - {n}" for n in self.tactical_ui_notes[-4:])
            )

        repeated_failures = [k for k, v in self.ineffective_actions.items() if v >= 2]
        if repeated_failures:
            sections.append(
                "Inputs That Previously Had NO Effect (avoid repeating the exact same dead click): "
                + ", ".join(repeated_failures[-6:])
            )

        # NOTE: We intentionally NEVER inject 'Actions Proven Effective So Far' here or in strategic context,
        # because listing frequently-clicked buttons biases the agent into repeating easy auxiliary actions
        # (e.g. Loan/Develop) instead of executing high-value multi-step game strategy!

        if self.compressed_milestones:
            sections.append("Earlier Game Progression:\n" + "\n".join(f"  * {m}" for m in self.compressed_milestones[-6:]))

        if self.recent_steps:
            recent_lines = []
            for s in self.recent_steps:
                delta_str = (
                    f"post-action visual change={s['post_visual_delta_pct']}%"
                    if s["post_visual_delta_pct"] is not None
                    else "pending"
                )
                recent_lines.append(
                    f"  [Iter {s['iteration']}] Observed: {s['observation'][:140]} | "
                    f"Action: {s['signature']} ({s['execution_status']}, {delta_str}) | "
                    f"Expected: {s['expected_result'][:100]}"
                )
            sections.append("Recent Tactical Steps:\n" + "\n".join(recent_lines))
        else:
            sections.append("Recent Tactical Steps: None yet (first action of session).")

        if self.consecutive_stagnant_steps >= 2:
            sections.append(
                f"WARNING - STAGNATION DETECTED: The last {self.consecutive_stagnant_steps} inputs caused almost zero "
                f"visual change ({latest_delta_pct}%). Check if an in-game prompt requires selecting a specific card/slot first, "
                "a modal/confirm dialog is open, or the game canvas needs focus—complete the current strategic goal using the appropriate target!"
            )

        if self.pending_user_hints:
            hints = "\n".join(f"  >> {h}" for h in self.pending_user_hints)
            sections.append(f"LIVE OPERATOR GUIDANCE (Prioritize this!):\n{hints}")
            self.pending_user_hints.clear()

        return "\n\n".join(sections)

    def build_strategic_context(
        self,
        iteration: int,
        rl_advisor_hint: Optional[str] = None,
    ) -> str:
        """
        Build a 100% UI-agnostic, pure game-domain context for the Strategic Planner.
        Deliberately excludes action success/failure counts, target_ids, pixel coordinates,
        and UI stagnation metrics so strategy is driven strictly by game rules and winning conditions.
        """
        parts = [
            f"Iteration: {iteration} / {self.config.max_iterations}",
            f"Estimated Objective Progress: {self.progress_percent}%",
            f"Previous Strategy: {self.current_strategy}",
        ]
        if self.sub_goals:
            parts.append("Previous Sub-Goals:\n" + "\n".join(f"- {g}" for g in self.sub_goals))
        if self.known_mechanics:
            parts.append("Confirmed Game Rules & Mechanics:\n" + "\n".join(f"- {m}" for m in self.known_mechanics))
        if self.lessons_learned:
            parts.append("Strategic Domain Insights:\n" + "\n".join(f"- {l}" for l in self.lessons_learned))
        if self.compressed_milestones:
            parts.append("Earlier Game State Progression:\n" + "\n".join(f"- {m}" for m in self.compressed_milestones[-6:]))
        if self.recent_steps:
            state_lines = []
            for s in self.recent_steps:
                clean_obs = sanitize_domain_observation(s["observation"])[:140]
                sub = sanitize_domain_observation(s.get("subgoal") or "")[:80]
                if clean_obs:
                    line = f"- Step {s['iteration']}: {clean_obs}"
                    if sub:
                        line += f" (Sub-goal pursued: {sub})"
                    state_lines.append(line)
            if state_lines:
                parts.append(
                    "Recent Game State Progression (Evaluate board/resource progress and ensure balanced, high-scoring play rather than repeating auxiliary moves):\n"
                    + "\n".join(state_lines)
                )
        if rl_advisor_hint and rl_advisor_hint.strip():
            parts.append(rl_advisor_hint.strip())
        if self.pending_user_hints:
            parts.append("Live Operator Guidance:\n" + "\n".join(f"- {h}" for h in self.pending_user_hints))
        return "\n\n".join(parts)
